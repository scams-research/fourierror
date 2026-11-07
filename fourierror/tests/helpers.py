"""Reference helpers shared by the tests."""

import numpy as np
import scipp as sc


def dft_jacobians(n):
    """Reference Jacobians of Re and Im of ``(2 / n) * fft(x)`` w.r.t. ``x``."""
    jk = np.outer(np.arange(n), np.arange(n))
    theta = 2 * np.pi * jk / n
    return (2 / n) * np.cos(theta), -(2 / n) * np.sin(theta)


def cov_matrix(obj):
    """The covariance of a CovVariable/CovDataArray as an ``(n, n)`` array."""
    n = int(np.prod(obj.shape, dtype=int))
    return np.asarray(obj.covariance.values).reshape(n, n)


def make_coord(n, unit="s", step=0.5):
    return sc.arange("x", 0.0, n * step, step, unit=unit)[:n]


def covariance_of_input(da):
    if hasattr(da.data, "covariance"):
        return cov_matrix(da.data)
    return np.diag(da.variances)
