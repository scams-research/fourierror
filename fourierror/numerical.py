import numpy as np
import scipp as sc
from scipy.stats import multivariate_normal


def dft(data: sc.DataArray, coord: str, n_samples: int = 5_000) -> sc.Dataset:
    """
    Perform a numerical discrete Fourier transform.

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :param n_samples: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns:
    """
    cov = np.diag(data.variances)
    mv_norm = multivariate_normal(mean=data.values, cov=cov)

    dft_matrix = np.fft.fft(np.eye(data.values.size))

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")

    f_samples = (
        (2 / data.values.size) * (dft_matrix @ mv_norm.rvs(n_samples).T)
    )

    f_real = f_samples.real
    real = sc.array(
        dims=["omega"], values=f_real.mean(1), variances=np.cov(f_real).diagonal()
    )
    f_imag = f_samples.imag
    imag = sc.array(
        dims=["omega"], values=f_imag.mean(1), variances=np.cov(f_imag).diagonal()
    )

    val = 1.0 / (data.values.size * d[0])
    N = (data.values.size - 1) // 2 + 1
    p1 = sc.arange(dim="omega", start=0, stop=N, step=1)
    p2 = sc.arange(dim="omega", start=-(data.values.size // 2), stop=0, step=1)
    results = sc.concat([p1, p2], dim="omega")
    freq = results * val * 2 * np.pi

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": freq})
