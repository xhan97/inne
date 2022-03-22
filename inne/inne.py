# Copyright 2022. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Reference:
#     T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
#     Isolation-based anomaly detection using nearest-neighbor ensembles.
#     In Computational Intelligence, vol. 34, 2018, pp. 968-998.

# from __future__ import division

from warnings import warn

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

class IsolationNNE(OutlierMixin, BaseEstimator):
    """
    Parameters
    ----------
    t:int, default=100
    The number of base estimators in the ensemble.
    psi: int, pow(2,n), n in range(1,11)
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

    def __init__(self, n_estimators=100, psi=16, contamination="auto", random_state=None):
        self.n_estimators = n_estimators
        self.psi = psi
        self.random_state = random_state
        self.contamination = contamination

    def fit(self, X, y=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Check data
        X = check_array(X, accept_sparse=False)
        
        for i in range(self.n_estimators):
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
        self.is_fitted_ = True
        
        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(X), 100.0 * self.contamination)
        
        return self

    def _cigrid(self, X):

        n = X.shape[0]
        self.psi = min(self.psi, n)

        if self.random_state is not None:
            self.random_state = self.random_state + 5
            np.random.seed(self.random_state)

        center_index = np.random.choice(n, self.psi, replace=False)
        center_data = X[center_index]
        center_dist = cdist(center_data, center_data, 'euclidean')
        np.fill_diagonal(center_dist, np.inf)
        center_redius = np.amin(center_dist, axis=1)
        conn_index = np.argmin(center_dist, axis=1)
        conn_redius = center_redius[conn_index]
        ratio = 1 - conn_redius / center_redius
        return center_data, center_redius, conn_redius, ratio

    def predict(self, X):
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
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        # TODO:check the condition.
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the .

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier.

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):

        check_is_fitted(self, 'is_fitted_')

        # check data
        X = check_array(X, accept_sparse=False)

        for i in range(self.n_estimators):
            x_dists = cdist(self._center_data_set[i], X, 'euclidean')
            nn_center_dist = np.amin(x_dists, axis=0)
            nn_center_index = np.argmin(x_dists, axis=0)
            score = self._ratio_set[i][nn_center_index]
            score = np.where(nn_center_dist <
                             self._center_redius_set[i][nn_center_index], score, 1)
            if i == 0:
                score_set = np.array([score])
            else:
                score_set = np.append(
                    score_set, np.array([score]), axis=0)
        scores = np.mean(score_set, axis=0)

        return -scores


if __name__ == '__main__':
    import time
    from sklearn.datasets import make_moons

    def generate_outlier_data(n_samples, outliers_fraction):
        n_samples = n_samples
        outliers_fraction = outliers_fraction
        n_outliers = int(outliers_fraction * n_samples)
        datasets = 4.0 * (
            make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0]
            - np.array([0.5, 0.25])
        )

        rng = np.random.RandomState(42)

        X = np.concatenate([datasets, rng.uniform(
            low=-6, high=6, size=(n_outliers, 2))], axis=0)
        return X

    X = generate_outlier_data(300, 0.15)

    inne_model = IsolationNNE(n_estimators=200, psi=20)
    st = time.time()
    pred = inne_model.fit(X).predict(X)
    et = time.time()
    print(et-st)
