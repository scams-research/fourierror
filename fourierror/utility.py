import numpy as np
import scipp as sc

from .data_array import CovDataArray
from .variable import CovVariable


def validate_input(data: sc.DataArray, coord: str) -> None:
    """
    Check that ``data`` and ``coord`` are valid input for a Fourier transform.

    :param data: The data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :raises ValueError: If ``data`` is not one-dimensional, if it has neither
        variances nor a covariance (i.e. it is not a ``CovDataArray``), or if
        ``coord`` is not a coordinate along the dimension of ``data``.
    """
    if data.ndim != 1:
        raise ValueError(
            f"Data must be one-dimensional, got {data.ndim} dimensions {data.dims}."
        )
    if not isinstance(data, CovDataArray) and data.variances is None:
        raise ValueError(
            "Data must have variances or be a CovDataArray to propagate uncertainties."
        )
    if coord not in data.coords:
        raise ValueError(
            f"Coordinate '{coord}' not found in data, available coordinates "
            f"are {list(data.coords.keys())}."
        )
    if data.coords[coord].dims != data.dims:
        raise ValueError(
            f"Coordinate '{coord}' must be along the data dimension "
            f"{data.dims}, got {data.coords[coord].dims}."
        )


def is_diagonal(A, tol=1e-12) -> bool:
    A = np.asarray(A)
    return np.allclose(A, np.diag(np.diag(A)), atol=tol)


def construct_output(
    x_real, x_imag, var_real, var_imag, freq, diag_tol=1e-12
) -> sc.DataGroup:
    if is_diagonal(var_real, tol=diag_tol):
        real = sc.array(dims=["omega"], values=x_real, variances=np.diag(var_real))
        real_array = sc.DataArray(data=real, coords={"omega": freq})
    else:
        real = CovVariable(dims=["omega"], values=x_real, covariance=var_real)
        real_array = CovDataArray(data=real, coords={"omega": freq})

    if is_diagonal(var_imag, tol=diag_tol):
        imag = sc.array(dims=["omega"], values=x_imag, variances=np.diag(var_imag))
        imag_array = sc.DataArray(data=imag, coords={"omega": freq})
    else:
        imag = CovVariable(dims=["omega"], values=x_imag, covariance=var_imag)
        imag_array = CovDataArray(data=imag, coords={"omega": freq})

    return sc.DataGroup({"real": real_array, "imag": imag_array})
