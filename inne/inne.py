# Copyright 2022 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Reference:
#     T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
#     Isolation-based anomaly detection using nearest-neighbor ensembles.
#     In Computational Intelligence, vol. 34, 2018, pp. 968-998.

from __future__ import division

import random

import numpy as np
from scipy.spatial.distance import cdist


# Calculate Minkowski distance
def pdist2(X, Y, k):
    """
        Parameters
        ----------
        X : np.array
        Y : np.array
        Returns
        -------
        D,I
        D corresponds to the pairwise distance between samples, 
        and I is the corresponding index
    """

    if k == 1:
        ed = cdist(X, Y, 'euclidean').T
        D = [row.min() for row in ed]
        I = [row.argmin() for row in ed]
    if k == 2:
        D = np.zeros((2, len(X)))
        I = np.zeros((2, len(X)))

        ed = cdist(X, Y, 'euclidean').T
        D[0, :Y.shape[0]] = [row.min() for row in ed]
        I[0, :Y.shape[0]] = [row.argmin() for row in ed]
        D[1, :Y.shape[0]] = [sorted(list(row))[1] for row in ed]
        I[1, :Y.shape[0]] = [list(row).index(
            sorted(list(row))[1]) for row in ed]

    return D, I

# Get index


def get_index(Ndata, NCurtIndex, D, I):
    """
        Parameters
        ----------
        Ndata : np.array,
        Selected data
        NCurtIndex : list, 
        Index of the selected data
        D:Distance between samples
        I:Index of samples
        Returns
        -------
        index
    """

    NCurtIndex_I = np.zeros((2, len(Ndata)))
    NCurtIndex_I = [[NCurtIndex[int(value)] for value in i] for i in I]
    D_I = [D[1, int(i)] for i in I[1]]
    index = np.vstack((NCurtIndex_I, D[1], np.array(D_I), 1-D_I/D[1]))

    return index


class iNNE():
    """
    Parameters
    ----------
    t:int, default=100
    The number of base estimators in the ensemble.
    psi:int,pow(2,n),n in range(1,11)
    Random selection from training data psi sample points as subsample.
    contamination:float, default=0.5
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
        - If float, the contamination should be in the range (0, contamination].
    Returns
    -------
    Abnormal scores

    """

    def __init__(self, t=100, psi=16, contamination=0.5, seed=None):
        self.t = t
        self.psi = psi
        self.pdata = {}
        self.index = {}
        self.seed = seed
        self.offset_ = 0.5
        self.contamination = contamination
    # calculate abnormal score

    def cigrid(self, data, psi):
        """
            Parameters
            ----------
            data : np.array,
            Input data
            psi : int 
            Number of randomly selected samples
            Returns
            -------
            index
        """
        if self.seed is not None:
            self.seed = self.seed + 5
            random.seed(self.seed)
        Ndata = np.array([])
        while Ndata.shape[0] < 2:
            # sampling
            CurtIndex = [random.randint(0, data.shape[0]-1)
                         for _ in range(int(psi))]
            #CurtIndex = list(range(round(data.shape[0]*0.2)))
            Ndata = data[CurtIndex]
            # filter out repeat
            _, IA = np.unique(Ndata, axis=0, return_index=True)
            NCurtIndex = [CurtIndex[i] for i in IA]
            Ndata = data[NCurtIndex]
        D, I = pdist2(Ndata, Ndata, 2)
        index = get_index(Ndata, NCurtIndex, D, I)
        return index

    def fit(self, train_data):
        """
        Fit estimator.

        Parameters
        ----------
        traindata : {array}
            The input samples.  
        """
        self.train_data = train_data
        for i in range(self.t):
            self.index[i] = self.cigrid(train_data, self.psi)
            pindex = self.index[i][0, :]
            self.pdata[i] = train_data[pindex.astype('int64')]
        return self

    def predict(self, test_data):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        testdata : {array} 
        The input samples.
        Returns
        -------
        Abnormal scores
        """
        self.offset_ = np.percentile(self.decision_function(
            self.train_data), 100.0 * (1-self.contamination))
        is_inlier = np.ones(test_data.shape[0], dtype=int)
        is_inlier[self.decision_function(test_data) > self.offset_] = -1
        return is_inlier

    def decision_function(self, test_data):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        testdata : {array} 
        The input samples.
        Returns
        -------
        Abnormal scores
        """
        Iso = np.zeros((test_data.shape[0], self.t))

        for i in range(self.t):
            dist_index = self.index[i][2, :]
            ratio_index = self.index[i][-1, :]
            D, I = pdist2(self.pdata[i], test_data, 1)
            Iso[:, i] = list(np.array(ratio_index).reshape(
                len(ratio_index), 1)[I].T[0])
            temp_D = [dist_index[j] for j in I]
            distIndex_D = list(
                map(lambda x: 0 if x[0]-x[1] > 0 else 1, zip(temp_D, D)))
            Iso[:, i] = list(map(lambda x: 1 if distIndex_D[x]
                             == 1 else Iso[x, i], range(len(distIndex_D))))
        Iscore = np.sum(Iso, axis=1)/Iso.shape[1]

        return Iscore


if __name__ == '__main__':
    pass