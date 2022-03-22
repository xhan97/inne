
"""Tests for `inne` package."""

import time

import numpy as np
import pytest
from inne.inne import IsolationNNE
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris, make_blobs, make_moons

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import assert_allclose

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_iris
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score

from scipy.sparse import csc_matrix, csr_matrix
from unittest.mock import Mock, patch

rng = check_random_state(0)

# load the iris dataset
# and randomly permute it
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the diabetes dataset
# and randomly permute it
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]


def test_inne():
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {"n_estimators": [100, 200], "psi": [10, 20, 30]}
    )

    with ignore_warnings():
        for params in grid:
            IsolationNNE(random_state=0, **
                         params).fit(X_train).predict(X_test)


# def test_inne_sparse():
#     """Check IForest for various parameter settings on sparse input."""
#     rng = check_random_state(0)
#     X_train, X_test, y_train, y_test = train_test_split(
#         diabetes.data[:50], diabetes.target[:50], random_state=rng
#     )

#     grid = ParameterGrid(
#         {"n_estimators": [100, 200], "psi": [10, 20, 30]}
#     )

#     for sparse_format in [csc_matrix, csr_matrix]:
#         X_train_sparse = sparse_format(X_train)
#         X_test_sparse = sparse_format(X_test)

#         for params in grid:
#             # Trained on sparse format
#             sparse_classifier = IsolationNNE(
#                 random_state=1, **params
#             ).fit(X_train_sparse)
#             sparse_results = sparse_classifier.predict(X_test_sparse)

#             # Trained on dense format
#             dense_classifier = IsolationNNE(
#                 random_state=1, **params
#             ).fit(X_train)
#             dense_results = dense_classifier.predict(X_test)

#             assert_array_equal(sparse_results, dense_results)


def test_iforest_performance():
    """Test Isolation Forest performs well"""

    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)
    X_train = np.r_[X + 2, X - 2]
    X_train = X[:100]

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)

    # fit the model
    clf = IsolationNNE(n_estimators=100, psi=16).fit(X_train)

    # predict scores (the lower, the more normal)
    y_pred = -clf.decision_function(X_test)

    # check that there is at most 6 errors (false positive or false negative)
    assert roc_auc_score(y_test, y_pred) > 0.98


@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_inne_works(contamination):
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, 7]]

    # Test IsolationForest
    clf = IsolationNNE(random_state=0, contamination=contamination)
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    pred = clf.predict(X)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert_array_equal(pred, 6 * [1] + 2 * [-1])


def test_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf1 = IsolationNNE(contamination=0.1)
    clf1.fit(X_train)
    clf2 = IsolationNNE()
    clf2.fit(X_train)
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )

# def test_inne_with_uniform_data():
#     """Test whether inne predicts inliers when using uniform data"""

#     # 2-d array of all 1s
#     X = np.ones((100, 10))
#     inne = IsolationNNE()
#     inne.fit(X)

#     rng = np.random.RandomState(0)

#     assert all(inne.predict(X) == 1)
#     assert all(inne.predict(rng.randn(100, 10)) == 1)
#     assert all(inne.predict(X + 1) == 1)
#     assert all(inne.predict(X - 1) == 1)

#     # 2-d array where columns contain the same value across rows
#     X = np.repeat(rng.randn(1, 10), 100, 0)
#     inne = IsolationNNE()
#     inne.fit(X)

#     assert all(inne.predict(X) == 1)
#     assert all(inne.predict(rng.randn(100, 10)) == 1)
#     assert all(inne.predict(np.ones((100, 10))) == 1)

#     # Single row
#     X = rng.randn(1, 10)
#     inne = IsolationNNE()
#     inne.fit(X)

#     assert all(inne.predict(X) == 1)
#     assert all(inne.predict(rng.randn(100, 10)) == 1)
#     assert all(inne.predict(np.ones((100, 10))) == 1)
