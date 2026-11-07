"""Analytical propagation of uncertainty through the discrete Fourier transform.

The covariance of the real and imaginary parts of the transform is computed
exactly by linear propagation, ``J C J^T``, where ``J`` is the (linear) DFT
matrix and ``C`` the covariance of the input.
"""

import numpy as np
import scipp as sc

from .freq import frequencies
from .utility import construct_output


def dft(data: sc.DataArray, coord: str, diag_tol: float = 1e-12) -> sc.DataGroup:
    """
    Perform a discrete Fourier transform with analytical uncertainty propagation.

    :param data: A scipp DataArray with the data to be Fourier transformed. Its
        variances are taken as uncorrelated, unless it is a
        :class:`~fourierror.data_array.CovDataArray`, whose full covariance is
        used.
    :param coord: The coordinate to compute the Fourier transform over.
        It must be evenly spaced; a ``TypeError`` is raised otherwise.
    :param diag_tol: Tolerance used to decide if the output covariance
        matrices are diagonal. Optional, defaults to 1e-12.
    :returns: A scipp DataGroup with ``real`` and ``imag`` entries for the
        Fourier transformed result, on an ``omega`` (angular frequency) axis.
        Each entry is a :class:`~fourierror.data_array.CovDataArray` if its
        covariance is not diagonal, otherwise a plain ``scipp.DataArray``.
    """
    freq = frequencies(data, coord)

    if hasattr(data.data, "covariance"):
        cov = data.data.covariance.values
    else:
        cov = np.diag(data.variances)

    theta = (
        -2
        * np.pi
        * (
            np.arange(data.values.size)[:, np.newaxis]
            * np.arange(data.values.size)[:, np.newaxis].T
        )
        / data.values.size
    )

    cos = (2 / data.values.size) * np.cos(theta)
    sin = (2 / data.values.size) * np.sin(theta)

    dft_matrix = cos + 1j * sin

    f = dft_matrix @ data.values

    var_real = cos @ cov @ cos.T
    var_imag = sin @ cov @ sin.T

    return construct_output(f.real, f.imag, var_real, var_imag, freq, diag_tol=diag_tol)


def fft(data: sc.DataArray, coord: str, diag_tol: float = 1e-12) -> sc.DataGroup:
    """
    Perform a fast Fourier transform with analytical uncertainty propagation.

    Equivalent to :func:`dft`, but uses FFTs for both the values and the
    covariance, so scales better with the number of points.

    :param data: A scipp DataArray with the data to be Fourier transformed. Its
        variances are taken as uncorrelated, unless it is a
        :class:`~fourierror.data_array.CovDataArray`, whose full covariance is
        used.
    :param coord: The coordinate to compute the Fourier transform over.
        It must be evenly spaced; a ``TypeError`` is raised otherwise.
    :param diag_tol: Tolerance used to decide if the output covariance
        matrices are diagonal. Optional, defaults to 1e-12.
    :returns: A scipp DataGroup with ``real`` and ``imag`` entries for the
        Fourier transformed result, on an ``omega`` (angular frequency) axis.
        Each entry is a :class:`~fourierror.data_array.CovDataArray` if its
        covariance is not diagonal, otherwise a plain ``scipp.DataArray``.
    """
    freq = frequencies(data, coord)

    if hasattr(data.data, "covariance"):
        cov = data.data.covariance.values
    else:
        cov = np.diag(data.variances)

    n = data.values.size

    f = np.fft.fft(data.values) * (2 / n)

    p = np.fft.fft2(cov)
    q = np.fft.fft(np.fft.ifft(cov, axis=1) * n, axis=0)

    scale = (2 / n) ** 2

    var_real = scale * (p.real + q.real) / 2
    var_imag = scale * (q.real - p.real) / 2

    return construct_output(f.real, f.imag, var_real, var_imag, freq, diag_tol=diag_tol)
