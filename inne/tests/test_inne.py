
"""Tests for `inne` package."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from inne import IsolationNNE
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.datasets import (load_diabetes, load_digits, load_iris,
                              make_blobs, make_moons)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (assert_allclose, assert_array_almost_equal,
                                    assert_array_equal, ignore_warnings)

from sklearn.ensemble import IsolationForest

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


# also load the digits dataset
# and randomly permute it
digit = load_diabetes()
perm = rng.permutation(digit.target.size)
digit.data = digit.data[perm]
digit.target = digit.target[perm]


def test_inne():
    """Check Isolation NNE for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {"n_estimators": [100, 200], "max_samples": [10, 20, 30]}
    )

    with ignore_warnings():
        for params in grid:
            IsolationNNE(random_state=0, **
                         params).fit(X_train).predict(X_test)


def test_inne_performance():
    """Test Isolation NNE performs well"""

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
    clf = IsolationNNE(n_estimators=100, max_samples=16).fit(X_train)

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


def test_fit_time():
    data = digit.data
    print(data.shape)
    clf = IsolationNNE(n_estimators=200, max_samples=100)
    t1 = time.time()
    clf.fit(data)
    t2 = time.time()
    anomaly_labels = clf.predict(data)
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)

    clf2 = IsolationForest(n_estimators=200, max_samples=256)
    t1 = time.time()
    clf2.fit(data)
    t2 = time.time()
    anomaly_labels = clf2.predict(data)
    t3 = time.time()
    print(t2-t1)
    print(t3-t2)
