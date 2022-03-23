import pytest

from sklearn.utils.estimator_checks import check_estimator

from inne import IsolationNNE


@pytest.mark.parametrize(
    "estimator",
    [IsolationNNE(n_estimators=200, psi=20, random_state=5)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
