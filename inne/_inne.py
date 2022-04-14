# Authors: Xin Han <xinhan197@gmail.com>
#          Xichen Tang <xichentang2021@163.com>
#          Supported by Ye Zhu <ye.zhu@deakin.edu.au>
# License: BSD 3 clause


import numbers
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_array, check_is_fitted


class IsolationNNE(OutlierMixin, BaseEstimator):
    """ Isolation-based anomaly detection using nearest-neighbor ensembles.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(16, n_samples)`.

    contamination : "auto" or float, default="auto"
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If "auto", the threshold is determined as in the original paper.
            - If float, the contamination should be in the range (0, 0.5].

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells. 
           "Isolation-based anomaly detection using nearest-neighbor ensembles." In Computational 
           Intelligence, vol. 34, 2018, pp. 968-998.

    Examples
    --------
    >>> from inne import IsolationNNE
    >>> import numpy as np
    >>> X =  [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationNNE().fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])
    """

    def __init__(self, n_estimators=200, max_samples="auto", contamination="auto", random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
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

        n_samples = X.shape[0]
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(16, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported."
                    'Valid choices are: "auto", int or'
                    "float"
                    % self.max_samples
                )

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples = max_samples

        self._fit(X)
        self.is_fitted_ = True

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
                )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(X), 100.0 * self.contamination)

        return self

    def _fit(self, X):
        n, m = X.shape
        self._centroids = np.empty(
            [self.n_estimators, self.max_samples, m])
        self._ratio = np.empty([self.n_estimators, self.max_samples])
        self._centroids_radius = np.empty(
            [self.n_estimators, self.max_samples])
        for i in range(self.n_estimators):
            if isinstance(self.random_state, numbers.Integral):
                if i == 0:
                    rn_seed = self.random_state
                else:
                    rn_seed += 5
                np.random.seed(rn_seed)
            center_index = np.random.choice(n, self.max_samples, replace=False)
            self._centroids[i] = X[center_index]
            center_dist = euclidean_distances(
                self._centroids[i], self._centroids[i], squared=True)
            np.fill_diagonal(center_dist, np.inf)

            self._centroids_radius[i] = np.amin(center_dist, axis=1)
            conn_index = np.argmin(center_dist, axis=1)
            conn_radius = self._centroids_radius[i][conn_index]
            self._ratio[i] = 1 - conn_radius / self._centroids_radius[i]
        return self

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """

        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the .

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

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
        """
        Opposite of the anomaly score defined in the original paper.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """

        check_is_fitted(self, 'is_fitted_')

        # check data
        X = check_array(X, accept_sparse=False)

        score_set = np.empty([self.n_estimators, X.shape[0]])
        for i in range(self.n_estimators):
            x_dists = euclidean_distances(
                self._centroids[i], X, squared=True)
            nn_center_dist = np.amin(x_dists, axis=0)
            nn_center_index = np.argmin(x_dists, axis=0)
            score = self._ratio[i][nn_center_index]
            score_set[i] = np.where(nn_center_dist <
                                    self._centroids_radius[i][nn_center_index], score, 1)
        scores = np.mean(score_set, axis=0)

        return -scores
