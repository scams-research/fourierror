import numpy as np
import scipp as sc

from .data_array import CovDataArray
from .variable import CovVariable

def is_diagonal(A, tol=1e-12) -> bool:
    A = np.asarray(A)
    return np.allclose(A, np.diag(np.diag(A)), atol=tol)


def construct_output(x_real, x_imag, var_real, var_imag, freq, diag_tol=1e-12) -> sc.DataGroup:
    if is_diagonal(var_real, tol=diag_tol):
        real = sc.array(dims=['omega'], values=x_real, variances=np.diag(var_real))
        real_array = sc.DataArray(data=real, coords={"omega": freq})
    else:
        real = CovVariable(dims=["omega"], values=x_real, covariance=var_real)
        real_array = CovDataArray(data=real, coords={"omega": freq})

    if is_diagonal(var_imag, tol=diag_tol):
        imag = sc.array(dims=['omega'], values=x_imag, variances=np.diag(var_imag))
        imag_array = sc.DataArray(data=imag, coords={"omega": freq})
    else:
        imag = CovVariable(dims=["omega"], values=x_imag, covariance=var_imag)
        imag_array = CovDataArray(data=imag, coords={"omega": freq})

    return sc.DataGroup({"real": real_array, "imag": imag_array})
