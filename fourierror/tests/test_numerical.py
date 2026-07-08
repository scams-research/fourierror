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
DATA = sc.DataArray(data=Y, coords={"x": X})


class TestNumerical(unittest.TestCase):
    """
    Tests for the numerical modules
    """

    def test_without_errors(self):
        omega = sc.arange(dim="omega", start=0, stop=25)
        _ = numerical.dft(DATA, omega)
        pass
        # assert list(f.keys()) == ["real", "imag"]
        # assert list(f.coords.keys()) == ["omega"]
        # assert all(np.argsort(np.abs(f["real"].values))[-2:] == [20, 10])
        # assert all(np.argsort(np.abs(f["imag"].values))[-3:] == [20, 10, 2])
