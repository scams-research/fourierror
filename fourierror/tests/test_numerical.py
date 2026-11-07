"""Tests for the numerical (Monte Carlo) module."""

import numpy as np
import pytest
import scipp as sc

from fourierror import numerical
from fourierror.data_array import CovDataArray

from .helpers import cov_matrix, covariance_of_input, dft_jacobians, make_coord

METHODS = [numerical.dft, numerical.fft]


def full_cov(out):
    if isinstance(out, CovDataArray):
        return cov_matrix(out)
    return np.diag(out.variances)


@pytest.mark.parametrize("method", METHODS)
def test_output_structure(method, data):
    out = method(data, "x", n_samples=500, random_state=0)
    assert isinstance(out, sc.DataGroup)
    assert list(out.keys()) == ["real", "imag"]
    for key in ("real", "imag"):
        assert out[key].dims == ("omega",)
        assert out[key].sizes == {"omega": data.sizes["x"]}
        assert out[key].coords["omega"].unit == sc.Unit("Hz")


@pytest.mark.parametrize("method", METHODS)
def test_frequency_axis(method, data):
    n = data.sizes["x"]
    out = method(data, "x", n_samples=100, random_state=0)
    np.testing.assert_allclose(
        out["real"].coords["omega"].values, 2 * np.pi * np.fft.fftfreq(n, 0.5)
    )


@pytest.mark.parametrize("method", METHODS)
def test_reproducible_with_seed(method, data):
    a = method(data, "x", n_samples=200, random_state=7)
    b = method(data, "x", n_samples=200, random_state=7)
    for key in ("real", "imag"):
        np.testing.assert_array_equal(a[key].values, b[key].values)
        np.testing.assert_array_equal(full_cov(a[key]), full_cov(b[key]))


@pytest.mark.parametrize("method", METHODS)
def test_accepts_generator(method, plain_data):
    a = method(plain_data, "x", n_samples=200, random_state=np.random.default_rng(3))
    b = method(plain_data, "x", n_samples=200, random_state=np.random.default_rng(3))
    np.testing.assert_array_equal(a["real"].values, b["real"].values)


@pytest.mark.parametrize("method", METHODS)
def test_different_seeds_differ(method, plain_data):
    a = method(plain_data, "x", n_samples=200, random_state=1)
    b = method(plain_data, "x", n_samples=200, random_state=2)
    assert not np.allclose(a["real"].values, b["real"].values)


@pytest.mark.parametrize("method", METHODS)
def test_converges_to_analytical(method, data):
    n = data.sizes["x"]
    out = method(data, "x", n_samples=50_000, random_state=11)
    expected = (2 / n) * np.fft.fft(data.values)
    cov = covariance_of_input(data)
    j_re, j_im = dft_jacobians(n)
    scale = np.sqrt(np.max(np.diag(cov))) * 2 / np.sqrt(n)
    np.testing.assert_allclose(out["real"].values, expected.real, atol=0.05 * scale)
    np.testing.assert_allclose(out["imag"].values, expected.imag, atol=0.05 * scale)
    ref_re = j_re @ cov @ j_re.T
    ref_im = j_im @ cov @ j_im.T
    np.testing.assert_allclose(full_cov(out["real"]), ref_re, atol=0.05 * ref_re.max())
    np.testing.assert_allclose(full_cov(out["imag"]), ref_im, atol=0.05 * ref_re.max())


def test_dft_and_fft_agree_with_same_seed(data):
    d = numerical.dft(data, "x", n_samples=1000, random_state=5)
    f = numerical.fft(data, "x", n_samples=1000, random_state=5)
    for key in ("real", "imag"):
        np.testing.assert_allclose(d[key].values, f[key].values, atol=1e-10)
        np.testing.assert_allclose(full_cov(d[key]), full_cov(f[key]), atol=1e-10)
        np.testing.assert_allclose(
            d[key].coords["omega"].values, f[key].coords["omega"].values
        )


@pytest.mark.parametrize("method", METHODS)
def test_zero_variance_returns_exact_values(method):
    n = 6
    values = np.arange(n, dtype=float)
    da = sc.DataArray(
        data=sc.array(dims=["x"], values=values, variances=np.zeros(n)),
        coords={"x": make_coord(n)},
    )
    out = method(da, "x", n_samples=50, random_state=0)
    expected = (2 / n) * np.fft.fft(values)
    np.testing.assert_allclose(out["real"].values, expected.real, atol=1e-12)
    np.testing.assert_allclose(out["imag"].values, expected.imag, atol=1e-12)
    np.testing.assert_allclose(out["real"].variances, 0.0, atol=1e-20)
    assert type(out["real"]) is sc.DataArray


@pytest.mark.parametrize("method", METHODS)
def test_uneven_spacing_raises(method):
    da = sc.DataArray(
        data=sc.array(dims=["x"], values=np.ones(4), variances=np.ones(4)),
        coords={"x": sc.array(dims=["x"], values=[0.0, 1.0, 3.0, 4.0], unit="s")},
    )
    with pytest.raises(TypeError, match="evenly spaced"):
        method(da, "x")


def test_generate_mv_norm_uses_covariance(correlated_data):
    dist = numerical._generate_mv_norm(correlated_data)
    np.testing.assert_allclose(dist.mean, correlated_data.values)
    np.testing.assert_allclose(dist.cov, cov_matrix(correlated_data))


def test_generate_mv_norm_from_variances(plain_data):
    dist = numerical._generate_mv_norm(plain_data)
    np.testing.assert_allclose(dist.cov, np.diag(plain_data.variances))


def test_construct_real_imag_arrays():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(3, 200)) + 1j * rng.normal(size=(3, 200))
    real, imag = numerical._construct_real_imag_arrays(samples)
    np.testing.assert_allclose(real.values, samples.real.mean(1))
    np.testing.assert_allclose(imag.values, samples.imag.mean(1))
    np.testing.assert_allclose(cov_matrix(real), np.cov(samples.real))
    np.testing.assert_allclose(cov_matrix(imag), np.cov(samples.imag))
