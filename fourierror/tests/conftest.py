"""Shared fixtures for the fourierror tests."""

import numpy as np
import pytest
import scipp as sc

from fourierror.data_array import CovDataArray
from fourierror.variable import CovVariable

from .helpers import make_coord


def _plain(n):
    """A DataArray with independent (diagonal) variances."""
    rng = np.random.default_rng(1234)
    data = sc.array(
        dims=["x"],
        values=rng.normal(size=n),
        variances=rng.uniform(0.05, 0.5, size=n),
    )
    return sc.DataArray(data=data, coords={"x": make_coord(n)})


def _correlated(n):
    """A CovDataArray with a dense, positive-definite covariance."""
    rng = np.random.default_rng(4321)
    m = rng.normal(size=(n, n))
    cov = 0.01 * (m @ m.T) + 0.05 * np.eye(n)
    cv = CovVariable(dims=["x"], values=rng.normal(size=n), covariance=cov)
    return CovDataArray(data=cv, coords={"x": make_coord(n)})


@pytest.fixture(params=[8, 9], ids=["even", "odd"])
def n(request):
    return request.param


@pytest.fixture
def plain_data(n):
    return _plain(n)


@pytest.fixture
def correlated_data(n):
    return _correlated(n)


@pytest.fixture(params=["plain", "correlated"])
def data(request, n):
    return _plain(n) if request.param == "plain" else _correlated(n)
