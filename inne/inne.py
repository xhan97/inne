# Copyright 2022. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Reference:
#     T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
#     Isolation-based anomaly detection using nearest-neighbor ensembles.
#     In Computational Intelligence, vol. 34, 2018, pp. 968-998.

# from __future__ import division

import random
from matplotlib.pyplot import axis

import numpy as np
from scipy.spatial.distance import cdist
from warnings import warn


class INNE():
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
        self._psi = psi
        self.pdata = {}
        self.index = {}
        self.seed = seed
        self.offset_ = 0.5
        self.contamination = contamination
    # calculate abnormal score

    def _cigrid(self, X):
        n = X.shape[0]
        self._psi = min(self._psi, n)
        center_index = np.random.choice(n, self._psi, replace=False)
        center_data = X[center_index]
        center_dist = cdist(center_data, center_data, 'euclidean')
        np.fill_diagonal(center_dist, np.inf)
        center_redius = np.amin(center_dist, axis=1)
        conn_index = np.argmin(center_dist, axis=1)
        conn_redius = center_redius[conn_index]
        ratio = 1 - conn_redius / center_redius
        return center_data, center_redius, conn_redius, ratio

    def fit(self, X):
        self.train_data = X
        for i in range(self.t):
            center_data, center_redius, conn_redius, ratio = self._cigrid(X)
            if i == 0:
                self._center_data_set = np.array([center_data])
                self._center_redius_set = np.array([center_redius])
                self._conn_redius_set = np.array([conn_redius])
                self._ratio_set = np.array([ratio])
            else:
                self._center_data_set = np.append(
                    self._center_data_set, np.array([center_data]), axis=0)
                self._center_redius_set = np.append(
                    self._center_redius_set, np.array([center_redius]), axis=0)
                self._conn_redius_set = np.append(
                    self._conn_redius_set, np.array([conn_redius]), axis=0)
                self._ratio_set = np.append(
                    self._ratio_set, np.array([ratio]), axis=0)
        return self

    def decision_function(self, test_data):
        for i in range(self.t):
            # TODO: check dimension of test_data and train_data
            x_dists = cdist(self._center_data_set[i], test_data)
            nn_center_dist = np.amin(x_dists, axis=0)
            nn_center_index = np.argmin(x_dists, axis=0)
            Iso = self._ratio_set[i][nn_center_index]
            Iso = np.where(nn_center_dist <
                           self._center_redius_set[i][nn_center_index], Iso, 1)
            if i == 0:
                Iso_set = np.array([Iso])
            else:
                Iso_set = np.append(
                    Iso_set, np.array([Iso]), axis=0)
        Iscore = np.mean(Iso_set, axis=0)
        return Iscore

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


if __name__ == '__main__':
    from sklearn.datasets import make_moons, make_blobs
    import time

    def generate_outlier_data(n_samples, outliers_fraction):
        n_samples = n_samples
        outliers_fraction = outliers_fraction
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        datasets = 4.0 * (
            make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0]
            - np.array([0.5, 0.25])
        )

        rng = np.random.RandomState(42)

        X = np.concatenate([datasets, rng.uniform(
            low=-6, high=6, size=(n_outliers, 2))], axis=0)
        return X

    X = generate_outlier_data(300, 0.15)

    inne_model = INNE(t=200, psi=20)
    st = time.time()
    inne_model.fit(X).predict(X)
    et = time.time()
    print(et-st)
