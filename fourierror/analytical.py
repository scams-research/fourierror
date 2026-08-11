import numpy as np
import scipp as sc

from .freq import frequencies


def dft(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform an analytical Fourier transform

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns: A scipp Dataset with real and imaginary values for the Fourier
        transformed result, with some "frequency" axis.
    """
    freq = frequencies(data, coord)

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
    real = sc.array(dims=["omega"], values=f_real, variances=np.diag(var_real))

    f_imag = f.imag
    imag = sc.array(dims=["omega"], values=f_imag, variances=np.diag(var_imag))
    # same as f_real but extracts the imaginary part

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": freq})
