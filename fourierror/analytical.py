import numpy as np
import scipp as sc

from .freq import frequencies
from .variable import CovVariable
from .data_array import CovDataArray


def dft(data: sc.DataArray, coord: str) -> sc.Dataset:
    """
    Perform an analytical Fourier transform

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns: A scipp Dataset with real and imaginary values for the Fourier
        transformed result, with some "frequency" axis.
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

    f_real = f.real
    real = CovVariable(dims=["omega"], values=f_real, covariance=var_real)

    f_imag = f.imag
    imag = CovVariable(dims=["omega"], values=f_imag, covariance=var_imag)
    # same as f_real but extracts the imaginary part

    return sc.DataGroup(
        {
            "real": CovDataArray(data=real, coords={"omega": freq}),
            "imag": CovDataArray(data=imag, coords={"omega": freq}),
        }
    )


def fft(data: sc.DataArray, coord: str) -> sc.DataGroup: 
    """
    Performs an analytical Fast Fourier transform.

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns: A scipp Dataset with real and imaginary values for the Fourier
        transformed result, with some "frequency" axis.
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
 
    real = CovVariable(dims=["omega"], values=f.real, covariance=var_real)
    imag = CovVariable(dims=["omega"], values=f.imag, covariance=var_imag)
 
    return sc.DataGroup(
        {
            "real": CovDataArray(data=real, coords={"omega": freq}),
            "imag": CovDataArray(data=imag, coords={"omega": freq}),
        }
    )