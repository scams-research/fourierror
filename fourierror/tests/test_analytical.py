"""Tests for the analytical module.

Values and covariances are checked against an explicit ``J C J^T`` with the
DFT Jacobian written out independently of the implementation.
"""

import numpy as np
import pytest
import scipp as sc

from fourierror import analytical
from fourierror.data_array import CovDataArray

from .helpers import cov_matrix, covariance_of_input, dft_jacobians, make_coord

METHODS = [analytical.dft, analytical.fft]


def full_cov(out):
    if isinstance(out, CovDataArray):
        return cov_matrix(out)
    return np.diag(out.variances)


@pytest.mark.parametrize("method", METHODS)
def test_output_structure(method, data):
    out = method(data, "x")
    assert isinstance(out, sc.DataGroup)
    assert list(out.keys()) == ["real", "imag"]
    for key in ("real", "imag"):
        assert out[key].dims == ("omega",)
        assert out[key].sizes == {"omega": data.sizes["x"]}
        assert list(out[key].coords.keys()) == ["omega"]
        assert out[key].coords["omega"].unit == sc.Unit("Hz")


@pytest.mark.parametrize("method", METHODS)
def test_values_are_scaled_fft(method, data):
    n = data.sizes["x"]
    out = method(data, "x")
    expected = (2 / n) * np.fft.fft(data.values)
    np.testing.assert_allclose(out["real"].values, expected.real, atol=1e-12)
    np.testing.assert_allclose(out["imag"].values, expected.imag, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_frequency_axis(method, data):
    n = data.sizes["x"]
    out = method(data, "x")
    expected = 2 * np.pi * np.fft.fftfreq(n, 0.5)
    np.testing.assert_allclose(out["real"].coords["omega"].values, expected)
    np.testing.assert_allclose(out["imag"].coords["omega"].values, expected)


@pytest.mark.parametrize("method", METHODS)
def test_covariance_matches_reference(method, data):
    n = data.sizes["x"]
    cov = covariance_of_input(data)
    j_re, j_im = dft_jacobians(n)
    out = method(data, "x")
    np.testing.assert_allclose(full_cov(out["real"]), j_re @ cov @ j_re.T, atol=1e-12)
    np.testing.assert_allclose(full_cov(out["imag"]), j_im @ cov @ j_im.T, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_variances_are_covariance_diagonal(method, data):
    out = method(data, "x")
    for key in ("real", "imag"):
        np.testing.assert_allclose(out[key].variances, np.diag(full_cov(out[key])))


@pytest.mark.parametrize("method", METHODS)
def test_dft_correlates_frequencies(method, plain_data):
    """Even independent inputs give correlated real parts at +/- frequencies."""
    out = method(plain_data, "x")
    assert isinstance(out["real"], CovDataArray)
    assert isinstance(out["imag"], CovDataArray)


def test_dft_and_fft_agree(data):
    d = analytical.dft(data, "x")
    f = analytical.fft(data, "x")
    for key in ("real", "imag"):
        np.testing.assert_allclose(d[key].values, f[key].values, atol=1e-12)
        np.testing.assert_allclose(full_cov(d[key]), full_cov(f[key]), atol=1e-12)
        assert sc.identical(d[key].coords["omega"], f[key].coords["omega"])


@pytest.mark.parametrize("method", METHODS)
def test_parseval_total_variance(method, plain_data):
    """Sum of Re+Im variances equals (4 / n) * sum of input variances."""
    n = plain_data.sizes["x"]
    out = method(plain_data, "x")
    total = out["real"].variances.sum() + out["imag"].variances.sum()
    assert total == pytest.approx(4 / n * plain_data.variances.sum())


@pytest.mark.parametrize("method", METHODS)
def test_zero_variance_gives_zero_variance(method):
    n = 6
    da = sc.DataArray(
        data=sc.array(
            dims=["x"], values=np.arange(n, dtype=float), variances=np.zeros(n)
        ),
        coords={"x": make_coord(n)},
    )
    out = method(da, "x")
    assert type(out["real"]) is sc.DataArray
    np.testing.assert_allclose(out["real"].variances, 0.0)
    np.testing.assert_allclose(out["imag"].variances, 0.0)


@pytest.mark.parametrize("method", METHODS)
def test_large_diag_tol_gives_plain_data_array(method, plain_data):
    out = method(plain_data, "x", diag_tol=1e6)
    assert type(out["real"]) is sc.DataArray
    assert type(out["imag"]) is sc.DataArray


@pytest.mark.parametrize("method", METHODS)
def test_single_cosine(method):
    """A pure cosine of amplitude A has Re F = A at +/- its frequency."""
    n, k, amp = 16, 3, 2.5
    t = np.arange(n)
    da = sc.DataArray(
        data=sc.array(
            dims=["x"], values=amp * np.cos(2 * np.pi * k * t / n), variances=np.ones(n)
        ),
        coords={"x": sc.array(dims=["x"], values=t.astype(float), unit="s")},
    )
    out = method(da, "x")
    expected = np.zeros(n)
    expected[[k, n - k]] = amp
    np.testing.assert_allclose(out["real"].values, expected, atol=1e-12)
    np.testing.assert_allclose(out["imag"].values, 0.0, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_uneven_spacing_raises(method):
    da = sc.DataArray(
        data=sc.array(dims=["x"], values=np.ones(4), variances=np.ones(4)),
        coords={"x": sc.array(dims=["x"], values=[0.0, 1.0, 3.0, 4.0], unit="s")},
    )
    with pytest.raises(TypeError):
        method(da, "x")


@pytest.mark.parametrize("method", METHODS)
def test_input_is_not_modified(method, data):
    before = data.copy()
    method(data, "x")
    assert sc.identical(
        data.to_data_array() if isinstance(data, CovDataArray) else data,
        before.to_data_array() if isinstance(before, CovDataArray) else before,
    )
    if isinstance(data, CovDataArray):
        np.testing.assert_array_equal(cov_matrix(data), cov_matrix(before))
