import scipp as sc
from numpy import pi


def frequencies(data: sc.DataArray, coord: str) -> sc.DataArray:
    """
    Generate the DFT frequencies.

    :param data: A scipp DataArray for which frequencies should be found.
    :param coord: The coordinate over which the frequencies should be found.

    :returns: DFT frequencies as a scipp DataArray.
    """

    d = data.coords[coord][1:] - data.coords[coord][:-1]
    if not sc.allclose(d, d[0]):
        raise TypeError("Values must be evenly spaced to compute a DFT.")
    val = 1.0 / (data.values.size * d[0])
    N = (data.values.size - 1) // 2 + 1
    p1 = sc.arange(dim="omega", start=0, stop=N, step=1)
    p2 = sc.arange(dim="omega", start=-(data.values.size // 2), stop=0, step=1)
    results = sc.concat([p1, p2], dim="omega")
    return results * val * 2 * pi
