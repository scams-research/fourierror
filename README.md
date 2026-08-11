# fourierror: Uncertainty Propagation Through Fourier Transforms

[![PyPI version](https://badge.fury.io/py/fourierror.svg)](https://badge.fury.io/py/fourierror)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scams-research/fourierror)

*Real measurements come with uncertainties. Fourier transforms of real measurements should too.*

`fourierror` propagates uncertainty through the discrete Fourier transform, giving you variances on the real and imaginary components of the result rather than a bare complex spectrum. 
Two routes are available: an **analytical** propagation and a **numerical** sampling approach.

The package is built on [`scipp`](https://scipp.github.io), so units and coordinates are carried through the transform — the correct "frequency" axis is derived for you from the coordinate you transform over, rather than left as bare array indices.

```python
import fourierror

result = fourierror.analytical.dft(data, coord="time")
result["real"], result["imaginary"]   # values and variances
```

`data` is a `scipp.DataArray` with variances; the result is a `scipp.Dataset` on a frequency axis. 
If you're new to scipp, the [getting started guide](https://scipp.github.io/getting-started/index.html) is a good first stop.

A publication describing the propagation scheme — and guidance on when to prefer the numerical or analytical method — is in preparation.