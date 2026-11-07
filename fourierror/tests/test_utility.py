"""Tests for the output construction helpers."""

import numpy as np
import pytest
import scipp as sc

from fourierror.data_array import CovDataArray
from fourierror.utility import construct_output, is_diagonal
from fourierror.variable import CovVariable

from .helpers import cov_matrix

FREQ = sc.array(dims=["omega"], values=[0.0, 1.0, -1.0], unit="Hz")
VALUES = np.array([1.0, 2.0, 3.0])
DIAG = np.diag([0.1, 0.2, 0.3])
DENSE = np.array([[0.1, 0.05, 0.0], [0.05, 0.2, 0.01], [0.0, 0.01, 0.3]])


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (np.eye(3), True),
        (np.diag([1.0, 2.0]), True),
        (np.zeros((3, 3)), True),
        (np.array([[1.0]]), True),
        (DENSE, False),
        (np.array([[1.0, 1e-3], [1e-3, 1.0]]), False),
    ],
)
def test_is_diagonal(matrix, expected):
    assert is_diagonal(matrix) is expected


def test_is_diagonal_tolerance():
    m = np.array([[1.0, 1e-8], [1e-8, 1.0]])
    assert not is_diagonal(m)
    assert is_diagonal(m, tol=1e-6)


def test_is_diagonal_accepts_lists():
    assert is_diagonal([[1.0, 0.0], [0.0, 2.0]])


def test_returns_data_group_with_real_and_imag():
    out = construct_output(VALUES, -VALUES, DIAG, DIAG, FREQ)
    assert isinstance(out, sc.DataGroup)
    assert list(out.keys()) == ["real", "imag"]
    for key in out:
        assert out[key].dims == ("omega",)
        assert sc.identical(out[key].coords["omega"], FREQ)


def test_diagonal_gives_plain_data_array():
    out = construct_output(VALUES, -VALUES, DIAG, 2 * DIAG, FREQ)
    assert type(out["real"]) is sc.DataArray
    assert type(out["imag"]) is sc.DataArray
    np.testing.assert_allclose(out["real"].values, VALUES)
    np.testing.assert_allclose(out["imag"].values, -VALUES)
    np.testing.assert_allclose(out["real"].variances, np.diag(DIAG))
    np.testing.assert_allclose(out["imag"].variances, 2 * np.diag(DIAG))


def test_dense_gives_cov_data_array():
    out = construct_output(VALUES, -VALUES, DENSE, DENSE, FREQ)
    for key in ("real", "imag"):
        assert isinstance(out[key], CovDataArray)
        assert isinstance(out[key].data, CovVariable)
        np.testing.assert_allclose(cov_matrix(out[key]), DENSE)
        np.testing.assert_allclose(out[key].variances, np.diag(DENSE))


def test_real_and_imag_are_classified_independently():
    out = construct_output(VALUES, -VALUES, DIAG, DENSE, FREQ)
    assert type(out["real"]) is sc.DataArray
    assert isinstance(out["imag"], CovDataArray)
    out = construct_output(VALUES, -VALUES, DENSE, DIAG, FREQ)
    assert isinstance(out["real"], CovDataArray)
    assert type(out["imag"]) is sc.DataArray


def test_diag_tol_is_forwarded():
    out = construct_output(VALUES, -VALUES, DENSE, DENSE, FREQ, diag_tol=1.0)
    assert type(out["real"]) is sc.DataArray
    np.testing.assert_allclose(out["real"].variances, np.diag(DENSE))
