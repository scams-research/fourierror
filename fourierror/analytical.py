import numpy as np
import scipp as sc

from .freq import frequencies
from .utility import construct_output, validate_input


def dft(data: sc.DataArray, coord: str, diag_tol: float = 1e-12) -> sc.Dataset:
    """
    Perform an analytical Fourier transform

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :raises ValueError: If ``data`` is not one-dimensional, has no variances and
        is not a ``CovDataArray``, or ``coord`` is not a coordinate along the
        dimension of ``data``.
    :returns: A scipp Dataset with real and imaginary values for the Fourier
        transformed result, with some "frequency" axis.
    """
    validate_input(data, coord)
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
    Performs an analytical Fast Fourier transform.

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :raises ValueError: If ``data`` is not one-dimensional, has no variances and
        is not a ``CovDataArray``, or ``coord`` is not a coordinate along the
        dimension of ``data``.
    :returns: A scipp Dataset with real and imaginary values for the Fourier
        transformed result, with some "frequency" axis.
    """
    validate_input(data, coord)
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
