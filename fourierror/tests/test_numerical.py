"""
Tests for the numerical module
"""

# Copyright (c) fourierror developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import unittest

import numpy as np
import scipp as sc

from fourierror import numerical


X = sc.linspace(dim="x", start=0, stop=2 * np.pi, num=100, unit="rad")
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y.variances = sc.sqrt(sc.abs(Y * 0.1))
DATA = sc.DataArray(data=Y, coords={"x": X})


class TestNumerical(unittest.TestCase):
    """
    Tests for the numerical modules
    """

    def test_without_errors(self):
        f = numerical.dft(data=DATA, coord="x")
        assert list(f.keys()) == ["real", "imag"]
        assert list(f.coords.keys()) == ["omega"]
        assert f.coords["omega"].unit == sc.Unit("1/rad")
        np_result = np.fft.fft(Y.values)
        np.allclose(np_result.real, f["real"].values)
        np.allclose(np_result.imag, f["imag"].values)
        npf_result = np.fft.fftfreq(Y.values.size, 1 / (X[1].value - X[0].value))
        np.allclose(npf_result, f.coords['omega'].values)