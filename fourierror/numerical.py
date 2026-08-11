import numpy as np
import scipp as sc
from scipy.stats import multivariate_normal

from .freq import frequencies


def _generate_mv_norm(data: sc.DataArray) -> multivariate_normal:
    """
    Generates the multivariate normal distribution to be sampled.

    :param data: A scipp DataArray to be used to construct the distribution.

    :returns: A scipy.stats.rv_continuous object that can be sampled.
    """
    cov = np.diag(data.variances)
    return multivariate_normal(mean=data.values, cov=cov)


def _construct_real_imag_arrays(f_samples: np.ndarray) -> tuple[sc.DataArray]:
    """
    Construct the scipp arrays from the samples.

    :f_samples: Samples to be turned into scipp arrays.

    :returns: A tuple of scipp DataArrays, the first is real values and the second
        is imaginary.
    """
    f_real = f_samples.real
    real = sc.array(
        dims=["omega"], values=f_real.mean(1), variances=np.cov(f_real).diagonal()
    )
    f_imag = f_samples.imag
    imag = sc.array(
        dims=["omega"], values=f_imag.mean(1), variances=np.cov(f_imag).diagonal()
    )
    return real, imag


def dft(data: sc.DataArray, coord: str, n_samples: int = 5_000) -> sc.Dataset:
    """
    Perform a numerical discrete Fourier transform.

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :param n_samples: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns: A scipp Dataset with real and imaginary values for the Fourier transformed result with some "frequency" axis.
        an omega axis.
    """
    freq = frequencies(data, coord)

    mv_norm = _generate_mv_norm(data)

    dft_matrix = np.fft.fft(np.eye(data.values.size))

    f_samples = (2 / data.values.size) * (dft_matrix @ mv_norm.rvs(n_samples).T)

    real, imag = _construct_real_imag_arrays(f_samples)

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": freq})


def fft(data: sc.DataArray, coord: str, n_samples: int = 5_000) -> sc.Dataset:
    """
    Perform a numerical fast Fourier transform.

    :param data: A scipp DataArray with the data to be fast Fourier transformed.
    :param coord: The coordinate to compute the fast Fourier transform over.
    :param n_samples: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns: A scipp Dataset with real and imaginary values for the fast Fourier transformed result with some "frequency" axis.
    """
    d = data.coords[coord][1:] - data.coords[coord][:-1]
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")

    mv_norm = _generate_mv_norm(data)

    f_samples = (2 / data.values.size) * np.fft.fft(mv_norm.rvs(n_samples)).T

    real, imag = _construct_real_imag_arrays(f_samples)

    freq_array = np.fft.fftfreq(data.values.size, d[0].value) * 2 * np.pi
    freq = sc.array(
        dims=["omega"], values=freq_array, unit=(1 / data.coords[coord]).unit
    )

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": freq})
