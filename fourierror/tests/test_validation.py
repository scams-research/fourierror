"""Tests for input validation of the transform functions."""

import numpy as np
import pytest
import scipp as sc

from fourierror import analytical, numerical
from fourierror.utility import validate_input

from .helpers import make_coord

METHODS = [analytical.dft, analytical.fft, numerical.dft, numerical.fft]
IDS = ["analytical.dft", "analytical.fft", "numerical.dft", "numerical.fft"]
N = 8


@pytest.fixture(params=METHODS, ids=IDS)
def method(request):
    return request.param


def test_valid_input_passes(plain_data, correlated_data):
    validate_input(plain_data, "x")
    validate_input(correlated_data, "x")


def test_no_variances(method):
    da = sc.DataArray(
        data=sc.array(dims=["x"], values=np.ones(N)), coords={"x": make_coord(N)}
    )
    with pytest.raises(ValueError, match="variances or be a CovDataArray"):
        method(da, "x")


def test_two_dimensional(method):
    da = sc.DataArray(
        data=sc.array(
            dims=["x", "y"], values=np.ones((N, 2)), variances=np.ones((N, 2))
        ),
        coords={"x": make_coord(N)},
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        method(da, "x")


def test_missing_coord(method, plain_data):
    with pytest.raises(ValueError, match="'t' not found"):
        method(plain_data, "t")


def test_scalar_coord(method, plain_data):
    da = plain_data.copy()
    da.coords["t"] = sc.scalar(1.0, unit="s")
    with pytest.raises(ValueError, match="must be along the data dimension"):
        method(da, "t")
