"""
Tests for the numerical module
"""

# Copyright (c) fourierror developers.
# Distributed under the terms of the MIT License.
# author: Andrew R. McCluskey (arm61)

import unittest

import numpy as np
import scipp as sc
from scipp import testing

from fourierror import numerical, analytical


X = sc.linspace(dim="x", start=0, stop=2 * np.pi, num=100, unit="rad")
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y.variances = sc.sqrt(sc.abs(Y * 0.1))
DATA = sc.DataArray(data=Y, coords={"x": X})

class TestComparison(unittest.TestCase):
    """
    Tests to compare the results from the different numerical methods.
    """

    def test_compare(self):
        num = numerical.dft(data=DATA, coord="x", n_samples=20_000, random_state=42)
        ana = analytical.dft(data=DATA, coord="x")
        np.allclose(num['real'].values, ana['real'].values)
        np.allclose(num['imag'].values, ana['imag'].values)
        np.allclose(num['real'].variances, ana['real'].variances)
        np.allclose(num['imag'].variances, ana['imag'].variances)
        testing.assert_allclose(num['real'], ana['real'], atol=sc.scalar(5e-3))
        testing.assert_allclose(num['imag'], ana['imag'], atol=sc.scalar(5e-3))