"""Cross-checks between the analytical and numerical routes on a realistic signal."""

import numpy as np
import pytest
import scipp as sc

from fourierror import analytical, numerical

X = sc.linspace(dim="x", start=0, stop=2 * np.pi, num=64, unit="rad", endpoint=False)
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y.variances = 0.05 + 0.1 * np.abs(Y.values)
DATA = sc.DataArray(data=Y, coords={"x": X})


@pytest.fixture(scope="module")
def ana():
    return analytical.dft(DATA, "x")


@pytest.fixture(scope="module")
def num():
    return numerical.fft(DATA, "x", n_samples=40_000, random_state=42)


def test_frequency_unit(ana, num):
    assert ana["real"].coords["omega"].unit == sc.Unit("1/rad")
    assert sc.allclose(ana["real"].coords["omega"], num["real"].coords["omega"])


def test_peaks_at_signal_frequencies(ana):
    """omega = 2pi * k / (n dx) = k for this coordinate, so peaks sit at k."""
    omega = ana["real"].coords["omega"].values
    np.testing.assert_allclose(
        ana["real"].values[np.isclose(np.abs(omega), 10)], 2.0, atol=1e-10
    )
    np.testing.assert_allclose(
        ana["imag"].values[np.isclose(omega, 2)], -1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        ana["imag"].values[np.isclose(omega, -2)], 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        ana["imag"].values[np.isclose(omega, 20)], -0.6, atol=1e-10
    )


@pytest.mark.parametrize("key", ["real", "imag"])
def test_values_agree(ana, num, key):
    sigma = np.sqrt(ana[key].variances)
    # Monte-Carlo mean error is sigma / sqrt(n_samples); allow 5 standard errors.
    assert np.all(
        np.abs(num[key].values - ana[key].values) < 5 * sigma / np.sqrt(40_000) + 1e-12
    )


@pytest.mark.parametrize("key", ["real", "imag"])
def test_variances_agree(ana, num, key):
    np.testing.assert_allclose(
        num[key].variances, ana[key].variances, rtol=0.05, atol=1e-20
    )


@pytest.mark.parametrize("key", ["real", "imag"])
def test_covariances_agree(ana, num, key):
    a = ana[key].covariance.values
    b = num[key].covariance.values
    np.testing.assert_allclose(b, a, atol=0.05 * np.abs(a).max())
