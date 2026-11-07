"""Tests for the frequency axis."""

import numpy as np
import pytest
import scipp as sc

from fourierror.freq import frequencies

from .helpers import make_coord


@pytest.mark.parametrize("n", [2, 7, 8, 100, 101])
@pytest.mark.parametrize("step", [0.1, 0.5, 2.0])
def test_matches_numpy_angular_fftfreq(n, step):
    da = sc.DataArray(
        data=sc.zeros(dims=["x"], shape=[n]), coords={"x": make_coord(n, step=step)}
    )
    omega = frequencies(da, "x")
    assert omega.dims == ("omega",)
    np.testing.assert_allclose(omega.values, 2 * np.pi * np.fft.fftfreq(n, step))


@pytest.mark.parametrize(
    ("unit", "expected"), [("s", "Hz"), ("rad", "1/rad"), ("m", "1/m")]
)
def test_unit_is_reciprocal_of_coord(unit, expected):
    da = sc.DataArray(
        data=sc.zeros(dims=["x"], shape=[4]), coords={"x": make_coord(4, unit=unit)}
    )
    assert frequencies(da, "x").unit == sc.Unit(expected)


def test_uses_named_coord():
    n = 6
    da = sc.DataArray(
        data=sc.zeros(dims=["x"], shape=[n]),
        coords={"x": make_coord(n, step=1.0), "t": make_coord(n, step=0.25)},
    )
    np.testing.assert_allclose(
        frequencies(da, "t").values, 2 * np.pi * np.fft.fftfreq(n, 0.25)
    )


def test_zero_frequency_first():
    da = sc.DataArray(data=sc.zeros(dims=["x"], shape=[5]), coords={"x": make_coord(5)})
    assert frequencies(da, "x").values[0] == 0.0


def test_uneven_spacing_raises():
    x = sc.array(dims=["x"], values=[0.0, 1.0, 2.0, 4.0], unit="s")
    da = sc.DataArray(data=sc.zeros(dims=["x"], shape=[4]), coords={"x": x})
    with pytest.raises(TypeError, match="evenly spaced"):
        frequencies(da, "x")


def test_missing_coord_raises():
    da = sc.DataArray(data=sc.zeros(dims=["x"], shape=[4]), coords={"x": make_coord(4)})
    with pytest.raises(KeyError):
        frequencies(da, "y")
