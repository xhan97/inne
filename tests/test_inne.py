
"""Tests for `inne` package."""

import pytest

from inne.inne import INNE
from sklearn.datasets import make_moons, make_blobs
import numpy as np
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


def test_inne_fit():
    """placeholder test"""
    inne_model = INNE(t=200, psi=20)
    st = time.time()
    inne_model.fit(X).predict(X)
    et = time.time()
    print(et-st)
    assert True