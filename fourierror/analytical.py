import numpy as np
import scipp as sc
import matplotlib.pyplot as plt
from functions import *
from scipy.stats import norm
import scipy.stats as stats

# (1) Functions defined

def dft_anal(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Performs an analytical Fourier transform outputting the real and imaginary parts separately

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.

    :returns: DataSet with the real, imaginary fourier transforms over the angular frequency coordinate
    """
    dft_matrix_manual = np.exp ( -2j * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size)
    # dft matrix of size = N x N

    f_w = (2 / data.values.size) * dft_matrix_manual @ data.values
    # computes the matrix formulated fourier transform of form : f = A @ x
    # size = N 

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    cov = np.diag(data.variances)

    cov_f = (2 / data.values.size)**2 * dft_matrix_manual @ cov @ dft_matrix_manual.conj().T

    f_real = f_w.real
    real = sc.array(
        dims=["omega"], values=f_real, variances=np.diag(cov_f.real)
    )
    # "dims = ["omega"] means to attribute a coordinate that this data lies on here F(ω) lies on the ω-coordinate
    # values are the mean values of the real part of the f_real matrix along "axis=1" which just means average across all the columns 
    # covariances are computed between rows (most usefully between the same row : diagonal elements) and the ".diagonal()" takes the diagonal of the covariance matrix

    f_imag = f_w.imag
    imag = sc.array(
        dims=["omega"], values=f_imag, variances=np.diag(cov_f.imag)
    )
    # same as f_real but extracts the imaginary part

    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k) to frequency of signal f(ν))

    P = (data.values.size - 1) // 2 + 1
    # take the size of data (N) and 
        # if even : half it ( N/2 )
        # if odd : half it ( N/2 ) and add 1 ( N/2 + 1 )

    p1 = sc.arange(dim="omega", start=0, stop=P, step=1)
    # create the freq-domain from 0 to (M-1)
    # i.e. if N = 6 , P = 3 then p1 = [0,1,2]

    p2 = sc.arange(dim="omega", start=-(data.values.size // 2), stop=0, step=1)
    # create freq-domain from 
        # if even : -(N/2) to 0 
        # if odd : [-(N/2) + 1] to 0

    results = sc.concat([p1, p2], dim="omega")
    # creates a zero-centered discrete freq-domain (if even then assymetry is pushed in negative direction)
        # if N = 6 then results = [-3 , -2 , -1 , 0, 1 , 2 ]
        # if N = 5 then results = [ -2 , -1 , 0 , 1 , 2 ]

    ang_freq = results * val * 2 * np.pi
    # multiply the index by 2π / (NΔt) to generate the angular frequency of the data

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": ang_freq})

dft_analdataset = dft_anal(data=data, coord="x")

def dft_anal_mag(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
   Performs an analytical Fourier transform outputting the magnitude of a signal (signal can be anti/symmetrised)

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.

    :returns: A scipp DataSet with the magnitude of the fourier transform and the angular frequency coordinate
    """
    dft_matrix_manual = np.exp ( -2j * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size)
    # dft matrix of size = N x N

    f_w = (2 / data.values.size) * dft_matrix_manual @ data.values
    # computes the matrix formulated fourier transform of form : f = A @ x
    # size = N 

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    cov = np.diag(data.variances)

    cov_f = (2 / data.values.size)**2 * dft_matrix_manual @ cov @ dft_matrix_manual.conj().T

    f_w = abs(f_w)
    magnitude = sc.array(
        dims=["omega"], values=f_w, variances=np.diag(abs(cov_f))
    )


    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k) to frequency of signal f(ν))

    P = (data.values.size - 1) // 2 + 1
    # take the size of data (N) and 
        # if even : half it ( N/2 )
        # if odd : half it ( N/2 ) and add 1 ( N/2 + 1 )

    p1 = sc.arange(dim="omega", start=0, stop=P, step=1)
    # create the freq-domain from 0 to (M-1)
    # i.e. if N = 6 , P = 3 then p1 = [0,1,2]

    p2 = sc.arange(dim="omega", start=-(data.values.size // 2), stop=0, step=1)
    # create freq-domain from 
        # if even : -(N/2) to 0 
        # if odd : [-(N/2) + 1] to 0

    results = sc.concat([p1, p2], dim="omega")
    # creates a zero-centered discrete freq-domain (if even then assymetry is pushed in negative direction)
        # if N = 6 then results = [-3 , -2 , -1 , 0, 1 , 2 ]
        # if N = 5 then results = [ -2 , -1 , 0 , 1 , 2 ]

    ang_freq = results * val * 2 * np.pi
    # multiply the index by 2π / (NΔt) to generate the angular frequency of the data

    return sc.Dataset({"magnitude": magnitude}, coords={"omega": ang_freq})

dft_analdataset_mag = dft_anal_mag(data=data, coord="x")

def plot_fourier(dft_dataset: sc.Dataset):
    """
    Plots the real/imaginary components (LHS) and the magnitude (RHS) of the fourier transform

    Args:
        dft_dataset : ScDataSet encoded with the {data: real(var), imag(var)} and {coords: "omega"}

    Returns:
        Two plots
    """
    omega = dft_dataset.coords["omega"].values
    magnitude = ((dft_dataset["real"].values)**2 + (dft_dataset["imag"].values)**2)**0.5
    real = dft_dataset["real"].values
    imag = dft_dataset["imag"].values

    fig, ax = plt.subplots(1,2,figsize=(12,4))  
    ax[0].plot(omega, magnitude, color ="purple", label="Magnitude")
    ax[0].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
    ax[0].set_ylabel(r"Magnitude of Fourier Transform, |F($\omega$)|", fontsize=8)
    ax[0].set_title(r"Magnitude of Fourier transform as a function of angular frequency", fontsize=8)
    
    ax[1].plot(omega, imag, color = "blue", label="Imaginary", alpha=0.5 )
    ax[1].plot(omega , real,  color = "red", label="Real", alpha=0.5)
    ax[1].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
    ax[1].set_ylabel(r"Real/Imaginary component of Fourier Transform, F($\omega$)", fontsize=8)
    ax[1].set_title(r"Real and Imaginary components of Fourier transform as a function of angular frequency", fontsize=8)
    ax[1].legend()
    plt.show()
    return fig, ax

def plot_fourier_mag(dft_dataset: sc.Dataset):
    """
    Plots the real/imaginary components (LHS) and the magnitude (RHS) of the fourier transform

    Args:
        dft_dataset : ScDataSet encoded with the {data: real(var), imag(var)} and {coords: "omega"}

    Returns:
        Two plots
    """
    omega = dft_dataset.coords["omega"].values
    magnitude = (dft_dataset["magnitude"].values)

    plt.plot(omega, magnitude)
    plt.xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
    plt.ylabel(r"Magnitude of Fourier Transform, |F($\omega$)|", fontsize=8)
    plt.title(r"Magnitude of Fourier transform as a function of angular frequency", fontsize=8)
    plt.show()

# (2) Data inputted and anti/symmetrised

X = sc.linspace(dim='x', start=0, stop=2*np.pi, num=100, unit='rad')
# creates a ScArray for time-domain

Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
# creates a ScArray corresponding to y(x) = sin(2x) + 2 cos(10x) + 0.6 sin(20x)

Y_minus = sc.sin(-2 * X) + 2 * sc.cos(-10 * X) + 0.6 * sc.sin(-20 * X)

Y_even = ( Y + Y_minus ) / 2
Y_odd = ( Y - Y_minus ) / 2

Y.variances = sc.abs(Y * 0.05)
Y_even.variances = (np.sqrt(Y.variances) * np.sqrt(2) / 2)**2
Y_odd.variances = (np.sqrt(Y.variances) * np.sqrt(2) / 2)**2

Y_even.unit = sc.Unit('s')
Y_odd.unit = sc.Unit('s')
Y.unit = sc.Unit('s')
# attributes units of "seconds" to y(x) 

data = sc.DataArray(data=Y, coords={'x': X})
# creates a DataArray for y(x) 
# note (1) : coords parameter requires a dictionary as DataArray has atleast 0 coordinates that need to be specified

data_symm = sc.Dataset(data = {"signal_even": Y_even, "signal_odd": Y_odd}, coords = {"x" : X})

dft_numdataset = dft_num_uncomplex(data=data, coord="x", M=50000)

dft_analdataset_mag_even = dft_anal_mag(data=data_symm["signal_even"], coord="x")
plot_fourier_mag(dft_analdataset_mag_even)

dft_analdataset_mag_odd = dft_anal_mag(data=data_symm["signal_odd"], coord="x")
plot_fourier_mag(dft_analdataset_mag_odd)

# (3) Plotting the comparison between the numerical and analytical error in the real and imaginary components

omega = dft_analdataset_mag_even.coords["omega"].values
error_num_real = np.sqrt(dft_numdataset["real"].variances)
error_num_imag = np.sqrt(dft_numdataset["imag"].variances)
error_anal_real = np.sqrt(dft_analdataset_mag_even["magnitude"].variances)
error_anal_imag = np.sqrt(dft_analdataset_mag_odd["magnitude"].variances)

fig, ax = plt.subplots(1,2,figsize=(12,4))  
ax[0].plot(omega, error_num_real, color = "red", label="Numerical", alpha=0.5)
ax[0].plot(omega , error_anal_real, color = "blue", label="Analytical", alpha=0.5)
ax[0].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
ax[0].set_ylabel(r"Error in real component of Fourier Transform", fontsize=8)
ax[0].set_title(r"Error in real component compared between numerical and analytical", fontsize=8)
ax[0].legend()

ax[1].plot(omega, error_num_imag, color = "red", label="Numerical", alpha=0.5 )
ax[1].plot(omega , error_anal_imag, color = "blue", label="Analytical", alpha=0.5)
ax[1].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
ax[1].set_ylabel(r"Error in imaginary component of Fourier Transform", fontsize=8)
ax[1].set_title(r"Error in imaginary component compared between numerical and analytical", fontsize=8)
ax[1].legend()
plt.show()
fig, ax;
