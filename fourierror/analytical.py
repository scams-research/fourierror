import numpy as np
import scipp as sc

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

