def dft_fft(data: sc.DataArray) -> sc.Dataset:
    """
    Recursion algorithm for computing the standard FFT
    :param data: A scipp DataArray with the data to be Fourier transformed.
    :returns:
    """
    if data.values.size == 1:
        val = data.values[0]
        # the fourier transform of just x0 is f0 (if unnormalised)
        real = sc.array(dims=["omega"], values=np.array([val]))
        imag = sc.array(dims=["omega"], values=np.array([0.0]))
        return sc.Dataset({"real": real, "imag": imag})
        # an input size of N=1 yields the first index to be n=0 thus the exponential descends to unity

    index = sc.arange(dim="x", start=0, stop=data.values.size, step=1)
    # creates an index array [0, 1, 2... N-1]

    mask_even = index%2==0
    # takes every even values [0, 2, 4... N-2]

    mask_odd = index%2==1
    # takes every even values [1, 3, 5... N-1]

    base = np.exp( -2j * np.pi / data.values.size)
    # every element in the dft matrix can be expressed as a natural numbered exponent of this "base" value

    x_even = data[mask_even]
    # [x0, x2... xN-2]
    x_odd = data[mask_odd]
    # [x1, x3... xN-1]

    f_even = dft_fft(x_even)
    f_odd = dft_fft(x_odd)
    # recursion until N=1 is met

    f = np.zeros(data.values.size, dtype=complex)
    # creates a zero vector with definite dimensionality N

    for k in np.arange(data.values.size // 2):
        even = f_even["real"].values[k] + 1j * f_even["imag"].values[k]
        odd = f_odd["real"].values[k] + 1j * f_odd["imag"].values[k]
        f[k] = (even + odd * (base ** k))
        f[k + data.values.size // 2] = (even - odd * (base ** k))

    real = sc.array(dims=["omega"], values=f.real, variances=None)

    imag = sc.array(dims=["omega"], values=f.imag, variances=None)

    return sc.Dataset({"real": real, "imag": imag})
    =
def dft_fft_ONlogN(data: sc.DataArray, coord: str) -> sc.Dataset:

    normalisation = 2 / data.values.size

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    # creates an array of length N-1 corresponding to the difference in x(n+1) and x(n)
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
        # checks whether d[0] is equal to d for all d

    P = (data.values.size - 1) // 2 + 1

    p1 = sc.arange(dim="omega", start=0, stop=P, step=1)

    p2 = sc.arange(dim="omega", start=-(data.values.size // 2), stop=0, step=1)

    results = sc.concat([p1, p2], dim="omega")

    ang_freq = results * 2 * np.pi / (data.values.size * d[0])
    # multiply the index by 2π / (NΔt) to generate the angular frequency of the data

    var_x = sc.array(dims="x", values=data.variances)
    data_var = sc.DataArray(data=var_x, coords={"x": data.coords["x"]})
    var_fft = dft_fft(data_var)
    # FFT over j (=2k) = [0, 1, 2... N-1]

    extended_var_fft = np.concatenate([var_fft["real"].values, var_fft["real"].values])
    # let Re(FFT(k)) = Re_k
    # this is [Re_0, Re_1, Re_2... Re_N-1, Re_0, Re_1... Re_N-1]

    index = np.arange(extended_var_fft.size)
    # [0, 1, 2... 2(N-1)]

    cos_2k = extended_var_fft[index%2==0]
    # [Re_0, Re_2, Re_4... Re_N-2, Re_0, Re_2... Re_N-2]

    var_x_sum = np.sum(data.variances)
    # sum of all var_x

    real_variances = ( (2 / data.values.size) ** 2 ) * (1/2) * (var_x_sum + cos_2k)

    imag_variances = ( (2 / data.values.size) ** 2 ) * (1/2) * (var_x_sum - cos_2k)

    unnormalised_result = dft_fft(data=data)
    real = sc.array(dims=["omega"], values=unnormalised_result["real"].values * normalisation, variances=real_variances)
    imag = sc.array(dims=["omega"], values=unnormalised_result["imag"].values * normalisation, variances=imag_variances)

    return sc.Dataset({"real": real, "imag": imag}, coords={"omega": ang_freq})

output_fft = dft_fft_ONlogN(data=data, coord = "x")