import numpy as np
import scipp as sc
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

X = sc.linspace(dim='x', start=0, stop=2*np.pi, num=100, unit='rad')
# creates a ScArray for time-domain

Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
# creates a ScArray corresponding to y(x) = sin(2x) + 2 cos(10x) + 0.6 sin(20x)

Y.variances = sc.abs(Y * 0.05)
# attributes a Var(y) = |0.05y| to each variance value

Y.unit = sc.Unit('s')
# attributes units of "seconds" to y(x) 

data = sc.DataArray(data=Y, coords={'x': X})
# creates a DataArray for y(x) 
# note (1) : coords parameter requires a dictionary as DataArray has atleast 0 coordinates that need to be specified

def dft_num(data: sc.DataArray, coord: str, M: int = 5_000) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform a numerical discrete Fourier transform.

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :param M: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns:
    """
    cov = np.diag(data.variances)
    # creates covariance matrix from variances in y(x)

    mv_norm = multivariate_normal(mean=data.values, cov=cov)
    # creates a distribution from y(x) and error(y)

    dft_matrix = np.fft.fft(np.eye(data.values.size))
    # creates the matrix of A(kn) = e ^ ( -2πikn / N )

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    f_samples = (
        (2 / data.values.size) * (dft_matrix @ mv_norm.rvs(M).T)
    )
    # creates a matrix of fourier transforms with the rows being the frequency-domain and the columns being the samples

    f_real = f_samples.real
    real = sc.array(
        dims=["omega"], values=f_real.mean(1), variances=np.cov(f_real).diagonal()
    )
    # "dims = ["omega"] means to attribute a coordinate that this data lies on here F(ω) lies on the ω-coordinate
    # values are the mean values of the real part of the f_real matrix along "axis=1" which just means average across all the columns 
    # covariances are computed between rows (most usefully between the same row : diagonal elements) and the ".diagonal()" takes the diagonal of the covariance matrix

    f_imag = f_samples.imag
    imag = sc.array(
        dims=["omega"], values=f_imag.mean(1), variances=np.cov(f_imag).diagonal()
    )
    # same as f_real but extracts the imaginary part

    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k) to frequency of signal f(ν))

    M = (data.values.size - 1) // 2 + 1
    # take the size of data (N) and 
        # if even : half it ( N/2 )
        # if odd : half it ( N/2 ) and add 1 ( N/2 + 1 )

    p1 = sc.arange(dim="omega", start=0, stop=M, step=1)
    # create the freq-domain from 0 to (M-1)
    # i.e. if N = 6 , M = 3 then p1 = [0,1,2]

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

dft_numdataset = dft_num(data=data, coord="x", M=5000)

def dft_num_uncomplex(data: sc.DataArray, coord: str, M: int = 5_000) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform a numerical discrete Fourier transform, uncomplexed meaning the covariance matrix is not built and only the traces are considered

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :param M: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns:
    """
    # Fix (1) : never build a covariance matrix, purely work with the single 1D (N-sized) array
    error = np.sqrt(data.variances)
    # creates error array from the data variance

    # Fix (2) : samples generated from broadcasted adapted : sample = mean + error*z-value
    samples = data.values[:, np.newaxis] + error[:, np.newaxis] * np.random.randn(data.values.size, M)
    # converts data.values and error into column vectors so that when operated with np.random.randn() of size = (N, M), 
    # it correctly does element-wise multiplication.
    # Note (1) : broadcasting only works when 1 dimension is the same length for the input arrays (y) and the other contains
    #            a 1 and a number, call it x, "broadcasting" the operation to create an output array of size = (y, x)
    # data.values(N,1) + error(N,1) * z-values(N, M) = samples(N, M)

    # Fix (3) : manually builds the matrix from an outer product (column * row)
    dft_matrix_manual = np.exp ( -2j * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size)

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    f_samples = (
        (2 / data.values.size) * (dft_matrix_manual @ samples)
    )
    # creates a matrix of fourier transforms with the rows being the frequency-domain and the columns being the samples

    f_real = f_samples.real
    real = sc.array(
        dims=["omega"], values=f_real.mean(1), variances=np.cov(f_real).diagonal()
    )
    # "dims = ["omega"] means to attribute a coordinate that this data lies on here F(ω) lies on the ω-coordinate
    # values are the mean values of the real part of the f_real matrix along "axis=1" which just means average across all the columns 
    # covariances are computed between rows (most usefully between the same row : diagonal elements) and the ".diagonal()" takes the diagonal of the covariance matrix

    f_imag = f_samples.imag
    imag = sc.array(
        dims=["omega"], values=f_imag.mean(1), variances=np.cov(f_imag).diagonal()
    )
    # same as f_real but extracts the imaginary part

    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k)) to frequency of signal (f(ν))

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

dft_numdataset = dft_num_uncomplex(data=data, coord="x", M=5000)

def dft_num_uncomplex_mag(data: sc.DataArray, coord: str, M: int = 5_000) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform a numerical discrete Fourier transform and outputs the magnitude

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :param M: Number of samples to use to obtain numerical solution. Optional, defaults to 5000.

    :returns:
    """
    # Fix (1) : never build a covariance matrix, purely work with the single 1D (N-sized) array
    error = np.sqrt(data.variances)
    # creates error array from the data variance

    # Fix (2) : samples generated from broadcasted adapted : sample = mean + error*z-value
    samples = data.values[:, np.newaxis] + error[:, np.newaxis] * np.random.randn(data.values.size, M)
    # converts data.values and error into column vectors so that when operated with np.random.randn() of size = (N, M), 
    # it correctly does element-wise multiplication.
    # Note (1) : broadcasting only works when 1 dimension is the same length for the input arrays (y) and the other contains
    #            a 1 and a number, call it x, "broadcasting" the operation to create an output array of size = (y, x)
    # data.values(N,1) + error(N,1) * z-values(N, M) = samples(N, M)

    # Fix (3) : manually builds the matrix from an outer product (column * row)
    dft_matrix_manual = np.exp ( -2j * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size)

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    f_samples = (
        (2 / data.values.size) * (dft_matrix_manual @ samples)
    )
    # creates a matrix of fourier transforms with the rows being the frequency-domain and the columns being the samples

    f_w = abs(f_samples)
    magnitude = sc.array(
        dims=["omega"], values=f_w.mean(1), variances=f_w.var(axis=1, ddof=1) 
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

    return sc.Dataset({"magnitude" : magnitude}, coords={"omega": ang_freq})

dft_numdataset_mag = dft_num_uncomplex_mag(data=data, coord="x", M=5000)

def dft_analytical(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform an analytical Fourier transform

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns:
    """
    var_x = np.diag(data.variances)

    theta = -2 * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size
    # θ = -2πkn/N

    cos = (2 / data.values.size) * np.cos(theta)
    sin = (2 / data.values.size) * np.sin(theta)

    dft_matrix = cos + 1j * sin

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    f = dft_matrix @ data.values
    # f = A @ x

    var_real = cos @ var_x @ cos.T
    var_imag = sin @ var_x @ sin.T

    f_real = f.real
    real = sc.array(dims=["omega"], values=f_real, variances=np.diag(var_real))
    # "dims = ["omega"] means to attribute a coordinate that this data lies on here F(ω) lies on the ω-coordinate
    # values are the mean values of the real part of the f_real matrix along "axis=1" which just means average across all the columns 
    # covariances are computed between rows (most usefully between the same row : diagonal elements) and the ".diagonal()" takes the diagonal of the covariance matrix

    f_imag = f.imag
    imag = sc.array(dims=["omega"], values=f_imag, variances=np.diag(var_imag))
    # same as f_real but extracts the imaginary part

    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k)) to frequency of signal (f(ν))

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

dft_analdataset = dft_analytical(data=data, coord="x")

def dft_analytical_mag(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform an analytical Fourier transform and returns the magnitude only

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns:
    """
    var_x = np.diag(data.variances)

    theta = -2 * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size
    # θ = -2πkn/N

    cos = (2 / data.values.size) * np.cos(theta)
    sin = (2 / data.values.size) * np.sin(theta)

    dft_matrix = cos + 1j * sin

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    f = dft_matrix @ data.values
    # f = A @ x

    var_f = dft_matrix @ var_x @ dft_matrix.conj().T

    magnitude = sc.array(dims=["omega"], values=abs(f), variances=np.diag(abs(var_f)))

    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k)) to frequency of signal (f(ν))

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

dft_analdataset_mag = dft_analytical_mag(data=data, coord="x")

def dft_analytical_mag_cov(data: sc.DataArray, coord: str) -> sc.Dataset:
    # "parameter: type" and "...type:" only to display expected parameter type for editor
    # input requires "coord" but can be more specific i.e. {coord : "coord"}

    """
    Perform an analytical Fourier transform and returns the magnitude only

    :param data: A scipp DataArray with the data to be Fourier transformed.
    :param coord: The coordinate to compute the Fourier transform over.
    :returns:
    """
    cov_x = np.diag(data.variances)

    theta = -2 * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size
    # θ = -2πkn/N

    cos = (2 / data.values.size) * np.cos(theta)
    sin = (2 / data.values.size) * np.sin(theta)

    dft_matrix = cos + 1j * sin

    cov_real = cos @ cov_x @ cos.T
    cov_imag = sin @ cov_x @ sin.T
    cov_real_imag = cos @ cov_x @ sin.T

    f_real = cos @ data.values
    f_imag = sin @ data.values
    f = dft_matrix @ data.values

    cov_f = ( ( (np.diag(cov_real) * ((f_real)**2)) + (np.diag(cov_imag) * ((f_imag)**2)) + (2 * f_real * f_imag * np.diag(cov_real_imag)) ) / abs(f)**2 )

    magnitude = sc.array(dims=["omega"], values=abs(f), variances=cov_f)

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d
    val = 1.0 / (data.values.size * d[0])
    # value is just the conversion from k (index of f(k)) to frequency of signal (f(ν))

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

dft_analdataset_mag_cov = dft_analytical_mag_cov(data=data, coord="x")

omega = dft_analdataset.coords["omega"].values
error_num_real = np.sqrt(dft_numdataset["real"].variances)
error_num_imag = np.sqrt(dft_numdataset["imag"].variances)
error_num_mag = np.sqrt(dft_numdataset_mag["magnitude"].variances)
error_anal_real = np.sqrt(dft_analdataset["real"].variances)
error_anal_imag = np.sqrt(dft_analdataset["imag"].variances)
error_anal_mag = np.sqrt(dft_analdataset_mag_cov["magnitude"].variances)

# Compare real and imaginary side by side

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

# Compare numerical and analytical side by side (turning off lines using #)

fig, ax = plt.subplots(1,2,figsize=(12,4))  
#ax[0].plot(omega, error_num_real, color = "red", label="Numerical Real", alpha=0.5)
#ax[0].plot(omega , error_num_imag, color = "blue", label="Numerical Imaginary", alpha=0.5)
ax[0].plot(omega, error_num_mag, color = "green", label="Numerical Magnitude", alpha=0.5)
ax[0].set_ylim(0,0.06)
ax[0].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
ax[0].set_ylabel(r"Numerical error in component of Fourier Transform", fontsize=8)
ax[0].set_title(r"Numerical error in component compared between Re, Im and Magnitude", fontsize=8)
ax[0].legend()

ax[1].plot(omega, error_anal_real, color = "red", label="Analytical Real", alpha=0.5)
#ax[1].plot(omega , error_anal_imag, color = "blue", label="Analytical Imaginary", alpha=0.5)
ax[1].plot(omega, error_anal_mag, color = "green", label="Analytical Magnitude", alpha=0.5)
ax[1].set_ylim(0,0.06)
ax[1].set_xlabel(r"Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$", fontsize=8)
ax[1].set_ylabel(r"Analytical error in component of Fourier Transform", fontsize=8)
ax[1].set_title(r"Analytical error in component compared between Re, Im and Magnitude", fontsize=8)
ax[1].legend()
plt.show()
fig, ax;

# Create plot comparing numerical (histogram) and analytical (probability density function)

from scipy.stats import norm
import scipy.stats as stats

# Data for numerical histogram
error = np.sqrt(data.variances)
samples = data.values[:, np.newaxis] + error[:, np.newaxis] * np.random.randn(data.values.size, 50000)
dft_matrix_manual = np.exp ( -2j * np.pi * ( np.arange(data.values.size)[:,np.newaxis] * np.arange(data.values.size)[:,np.newaxis].T ) / data.values.size)
f_samples = ((2 / data.values.size) * (dft_matrix_manual @ samples))
f_samples_real = f_samples.real

# Data for analytical normal distribution
f_reals = dft_analdataset["real"].values
deviations = error_anal_real
# i is the range of the angular frequency domain from - N/2 to N/2
i = 0
print(omega[i])
f_real_domain = np.linspace(f_reals[i]- 4*deviations[i], f_reals[i] + 4*deviations[i], 100)
# Plotting the histogram and normal distribution
fig, axis = plt.subplots(1,1)
axis.hist(f_samples_real[i], 50, density=True)
axis.plot(f_real_domain , stats.norm.pdf(f_real_domain, f_reals[i], deviations[i]))
