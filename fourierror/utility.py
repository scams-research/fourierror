"""Helpers shared by the analytical and numerical transforms."""

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
    """Check whether a matrix is diagonal to within a tolerance.

    :param A: The matrix to check, as anything ``numpy.asarray`` accepts.
    :param tol: Absolute tolerance on the off-diagonal elements. Optional,
        defaults to 1e-12.

    :returns: ``True`` if every off-diagonal element is within ``tol`` of zero.
    """
    A = np.asarray(A)
    return np.allclose(A, np.diag(np.diag(A)), atol=tol)


def construct_output(
    x_real, x_imag, var_real, var_imag, freq, diag_tol=1e-12
) -> sc.DataGroup:
    """Assemble the real and imaginary parts of a transform into a DataGroup.

    Each part is returned as a plain ``scipp.DataArray`` with variances if its
    covariance matrix is diagonal (within ``diag_tol``), and as a
    :class:`~fourierror.data_array.CovDataArray` carrying the full covariance
    otherwise.

    :param x_real: Values of the real part.
    :param x_imag: Values of the imaginary part.
    :param var_real: Covariance matrix of the real part.
    :param var_imag: Covariance matrix of the imaginary part.
    :param freq: Frequency coordinate, with dimension ``omega``.
    :param diag_tol: Tolerance used to decide if a covariance matrix is
        diagonal. Optional, defaults to 1e-12.

    :returns: A scipp DataGroup with ``real`` and ``imag`` entries.
    """
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
