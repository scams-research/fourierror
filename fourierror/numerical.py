"""Monte Carlo propagation of uncertainty through the discrete Fourier transform.

Samples are drawn from a multivariate normal distribution described by the
input values and covariance, each sample is transformed, and the mean and
covariance of the transformed samples are returned.
"""

import numpy as np
import scipp as sc
from scipy.stats import multivariate_normal

from .freq import frequencies
from .variable import CovVariable
from .utility import construct_output


def _generate_mv_norm(data: sc.DataArray) -> multivariate_normal:
    """
    Generates the multivariate normal distribution to be sampled.

    :param data: A scipp DataArray to be used to construct the distribution.

    :returns: A scipy.stats.rv_continuous object that can be sampled.
    """
    if hasattr(data.data, "covariance"):
        cov = data.data.covariance.values
    else:
        cov = np.diag(data.variances)
    return multivariate_normal(mean=data.values, cov=cov, allow_singular=True)


def _construct_real_imag_arrays(
    f_samples: np.ndarray,
) -> tuple[CovVariable, CovVariable]:
    """
    Construct the scipp arrays from the samples.

    :param f_samples: Complex samples, with shape ``(n_frequencies, n_samples)``.

    :returns: A tuple of :class:`~fourierror.variable.CovVariable` objects: the
        sample mean and covariance of the real part, then of the imaginary part.
    """
    f_real = f_samples.real
    real = CovVariable(dims=["omega"], values=f_real.mean(1), covariance=np.cov(f_real))
    f_imag = f_samples.imag
    imag = CovVariable(dims=["omega"], values=f_imag.mean(1), covariance=np.cov(f_imag))
    return real, imag


def dft(
    data: sc.DataArray,
    coord: str,
    n_samples: int = 5_000,
    random_state: None | int | np.random.Generator = None,
    diag_tol: float = 1e-12,
) -> sc.DataGroup:
    """
    Perform a numerical discrete Fourier transform.

    :param data: A one-dimensional scipp DataArray with variances, holding the
        data to be Fourier transformed. Its variances are taken as
        uncorrelated, unless it is a
        :class:`~fourierror.data_array.CovDataArray`, whose full covariance is
        used.
    :param coord: The coordinate to compute the Fourier transform over.
        It must be evenly spaced; a ``TypeError`` is raised otherwise.
    :param n_samples: Number of samples to use to obtain numerical solution.
        Optional, defaults to 5000.
    :param random_state: Seed or generator passed to the random sampler, to
        enable reproducibility. Optional, defaults to ``None``.
    :param diag_tol: Tolerance used to decide if the output covariance
        matrices are diagonal. Optional, defaults to 1e-12.

    :returns: A scipp DataGroup with ``real`` and ``imag`` entries holding the
        sample mean and covariance of ``(2 / n) * numpy.fft.fft`` of the
        samples, on an ``omega`` axis of angular frequencies (see
        :func:`~fourierror.freq.frequencies`).
        Each entry is a :class:`~fourierror.data_array.CovDataArray` if its
        covariance is not diagonal, otherwise a plain ``scipp.DataArray``.
    """
    freq = frequencies(data, coord)

    mv_norm = _generate_mv_norm(data)

    dft_matrix = np.fft.fft(np.eye(data.values.size))

    f_samples = (2 / data.values.size) * (
        dft_matrix @ mv_norm.rvs(n_samples, random_state=random_state).T
    )

    return construct_output(
        f_samples.real.mean(1),
        f_samples.imag.mean(1),
        np.cov(f_samples.real),
        np.cov(f_samples.imag),
        freq,
        diag_tol=diag_tol,
    )


def fft(
    data: sc.DataArray,
    coord: str,
    n_samples: int = 5_000,
    random_state: None | int | np.random.Generator = None,
    diag_tol: float = 1e-12,
) -> sc.DataGroup:
    """
    Perform a numerical fast Fourier transform.

    :param data: A one-dimensional scipp DataArray with variances, holding the
        data to be Fourier transformed. Its variances are taken as
        uncorrelated, unless it is a
        :class:`~fourierror.data_array.CovDataArray`, whose full covariance is
        used.
    :param coord: The coordinate to compute the fast Fourier transform over.
        It must be evenly spaced; a ``TypeError`` is raised otherwise.
    :param n_samples: Number of samples to use to obtain numerical solution.
        Optional, defaults to 5000.
    :param random_state: Seed or generator passed to the random sampler, to
        enable reproducibility. Optional, defaults to ``None``.
    :param diag_tol: Tolerance used to decide if the output covariance
        matrices are diagonal. Optional, defaults to 1e-12.

    :returns: A scipp DataGroup with ``real`` and ``imag`` entries holding the
        sample mean and covariance of ``(2 / n) * numpy.fft.fft`` of the
        samples, on an ``omega`` axis of angular frequencies (see
        :func:`~fourierror.freq.frequencies`).
        Each entry is a :class:`~fourierror.data_array.CovDataArray` if its
        covariance is not diagonal, otherwise a plain ``scipp.DataArray``.
    """
    d = data.coords[coord][1:] - data.coords[coord][:-1]
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")

    mv_norm = _generate_mv_norm(data)

    f_samples = (2 / data.values.size) * np.fft.fft(
        mv_norm.rvs(n_samples, random_state=random_state)
    ).T

    freq_array = np.fft.fftfreq(data.values.size, d[0].value) * 2 * np.pi
    freq = sc.array(
        dims=["omega"], values=freq_array, unit=(1 / data.coords[coord]).unit
    )

    return construct_output(
        f_samples.real.mean(1),
        f_samples.imag.mean(1),
        np.cov(f_samples.real),
        np.cov(f_samples.imag),
        freq,
        diag_tol=diag_tol,
    )
