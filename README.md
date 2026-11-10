# fourierror: Uncertainty Propagation Through Fourier Transforms

[![PyPI version](https://badge.fury.io/py/fourierror.svg)](https://badge.fury.io/py/fourierror)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scams-research/fourierror)

*Real measurements come with uncertainties. Fourier transforms of real measurements should too.*

`fourierror` propagates uncertainty through the discrete Fourier transform, giving you the full covariance of the real and imaginary components of the result rather than a bare complex spectrum.
Two routes are available: an **analytical** propagation and a **numerical** (Monte Carlo) sampling approach, each with a DFT and an FFT implementation.

| Function | Method |
|---|---|
| `fourierror.analytical.dft` | exact linear propagation through the DFT matrix |
| `fourierror.analytical.fft` | the same result, computed with FFTs |
| `fourierror.numerical.dft` | Monte Carlo sampling of the input distribution |
| `fourierror.numerical.fft` | Monte Carlo sampling, using the FFT for each sample |

The package is built on [`scipp`](https://scipp.github.io), so units and coordinates are carried through the transform — the correct "frequency" axis is derived for you from the coordinate you transform over, rather than left as bare array indices.

## Installation

```console
pip install fourierror
```

## Usage

```python
import numpy as np
import scipp as sc
import fourierror

t = sc.arange("time", 0.0, 6.4, 0.05, unit="s")
y = sc.sin(sc.to_unit(2 * np.pi * 1.25 * t * sc.scalar(1.0, unit="rad/s"), "rad"))
y.variances = np.full(t.shape, 0.01)
data = sc.DataArray(data=y, coords={"time": t})

result = fourierror.analytical.dft(data, coord="time")
result["real"], result["imag"]   # values with (co)variances on an "omega" axis
```

**Input.** A one-dimensional `scipp.DataArray` with variances, and the name of an evenly spaced coordinate to transform over (unevenly spaced coordinates raise a `TypeError`).
The variances are treated as uncorrelated; to supply correlated uncertainties, pass a `fourierror.data_array.CovDataArray`, whose full covariance matrix is propagated.

**Output.** A `scipp.DataGroup` with `"real"` and `"imag"` entries:

- Values are `(2 / n) * numpy.fft.fft(values)` (for the numerical methods, the sample mean of this), so a cosine of amplitude `A` whose frequency lies on the grid appears in `"real"` with height `A` at its positive and negative frequencies (a sine appears in `"imag"` as `-A` and `+A`).
- The coordinate `omega` is the angular frequency, `2π * numpy.fft.fftfreq(n, d)`, in the reciprocal unit of the input coordinate, in NumPy's order (zero first, negative frequencies in the second half).
- Transforming correlates the output points, so each entry is usually a `CovDataArray` carrying the full covariance matrix (via `.covariance`); `.variances` gives its diagonal. If a covariance is diagonal to within `diag_tol`, a plain `scipp.DataArray` with variances is returned instead.

If you're new to scipp, the [getting started guide](https://scipp.github.io/getting-started/index.html) is a good first stop.
Full documentation, including a worked example and the API reference, is at [scams-research.github.io/fourierror](https://scams-research.github.io/fourierror).

## Citing

The propagation scheme, and guidance on when to prefer the numerical or analytical method, is described in the accompanying paper.
If you use `fourierror` in your work, please cite:

> T. Zolzaya and A. R. McCluskey, "Numerical and analytical strategies to propagate correlated uncertainties through the discrete Fourier transform", [arXiv:2611.12359](https://arxiv.org/abs/2611.12359) (2026).

```bibtex
@misc{zolzaya2026numerical,
  title         = {Numerical and analytical strategies to propagate correlated uncertainties through the discrete {F}ourier transform},
  author        = {Zolzaya, Tuguldur and McCluskey, Andrew R.},
  year          = {2026},
  eprint        = {2611.12359},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.12359}
}
```
