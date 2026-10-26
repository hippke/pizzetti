![Logo](https://raw.githubusercontent.com/hippke/pizzetti/main/docs/logo.png)

# pizzetti

[![Python](https://github.com/hippke/pizzetti/actions/workflows/python.yml/badge.svg)](https://github.com/hippke/pizzetti/actions/workflows/python.yml)
[![Julia](https://github.com/hippke/pizzetti/actions/workflows/julia.yml/badge.svg)](https://github.com/hippke/pizzetti/actions/workflows/julia.yml)
[![Examples](https://github.com/hippke/pizzetti/actions/workflows/examples.yml/badge.svg)](https://github.com/hippke/pizzetti/actions/workflows/examples.yml)
[![Docs](https://img.shields.io/badge/docs-docs%2F-blue.svg)](docs/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Fast limb-darkened transit light curves, with an elementary series and a proven error bound.**

`pizzetti` has the interface of [batman](https://github.com/lkreidberg/batman), so existing code
runs unchanged after one line:

```python
import pizzetti as batman
```

## Install

```bash
pip install pizzetti            # pip install "pizzetti[fit]" adds emcee and scipy for MarginalTransitModel
```

Julia: `] add https://github.com/hippke/pizzetti:julia` (package `Pizzetti`, see [`julia/`](julia)).

## A light curve in ten lines

```python
import numpy as np, pizzetti

p = pizzetti.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.0, 0.1, 15.0, 88.0
p.ecc, p.w = 0.0, 90.0
p.u, p.limb_dark = [0.4, 0.25], "quadratic"

t = np.linspace(-0.1, 0.1, 10_000)
flux = pizzetti.TransitModel(p, t).light_curve(p)
```

## Why it is fast and exact

- **Inside the stellar disk** (between second and third contact) the flux is `sqrt(1 - z²)` times a
  polynomial in `1/(1 - z²)` whose coefficients depend only on the radius ratio: one square root,
  one division and a dozen multiply–adds per point, with a **proven** bound on the truncation error
  (default 10⁻⁹ of the stellar flux). This covers 96–98 % of the transit of an Earth-sized planet.
- **At the limb, quadratic family**: the exact Mandel & Agol solution, vectorised.
- **At the limb, other laws** (square-root, nonlinear, power-2, any sum of powers of μ): the limb
  integral is a complete integral that a fixed-node trapezoid rule evaluates to 10⁻¹⁵ without
  set-up; long light curves use Chebyshev interpolants built from it in ~40 µs (radius ratios
  below 0.3; beyond: contact expansions or quadrature).
- **Certified mode**: `limb="proven"` extends the proven bound to the limb; `return_certified=True`
  reports which samples carry it.
- **`MarginalTransitModel`** integrates the limb darkening and the baseline out analytically, so a
  fit samples only the geometry; `fit()` returns posterior draws in about a second.

Single thread, 10⁵ points, 35 % in transit, every call recomputing the orbit as in a fit
(`benchmarks/benchmark.py`, Intel Xeon, numba 0.68, batman 2.5.3):

| law | rp = 0.0092 | rp = 0.1 |
|---|---|---|
| quadratic | 0.31 ms (**13×** faster than batman) | 0.42 ms (**10×**) |
| nonlinear | 0.49 ms (**9×**) | 0.91 ms (**44×**) |
| power-2 | 0.47 ms (**7×**) | 0.76 ms (**34×**) |

batman's numerical laws run at its default 1 ppm tolerance; pizzetti stays below 10⁻⁹. In complete
emcee fits the likelihood is 3.4–4.4× faster than batman's, and 2–3.5× for grazing transits
([examples/inference](examples/inference)).

## What is included

- **batman drop-in**: `TransitParams`, `TransitModel` (`light_curve`, `get_t_secondary`,
  `get_t_periastron`, `get_t_conjunction`, `get_true_anomaly`, `calc_err`), circular and eccentric
  orbits, supersampling, secondary eclipses, inverse transits.
- **Laws**: uniform, linear, quadratic, square-root, nonlinear (4-parameter), power-2; in
  `MarginalTransitModel` also logarithmic and power-2 with free α. (Exponential and custom laws are
  not sums of powers of μ and are not supported.)
- **Low level**: `occult(z, rp, u, law, limb=...)`, `series_cut`, `occult_terms` (basis light curves,
  linear in the limb-darkening coefficients).
- **Julia (`Pizzetti.jl`)**: the same algorithms, plus exact gradients with ForwardDiff, BigFloat
  light curves, rigorous interval enclosures, photodynamics with NbodyGradient, and a
  Limbdark.jl-compatible `transit_poly`.

## Learn more

| | |
|---|---|
| [Tutorials](examples/) | ten Python notebooks and six Julia scripts, from a first light curve to certified, differentiable models |
| [Documentation](docs/) | method, accuracy, performance and API |
| [Paper reproduction](examples/paper) | one command per table and figure of the paper |
| [Proofs](docs/proofs) | self-contained proofs of the theorems, and [validation notebooks](validation) |
| [Changelog](CHANGELOG.md) | |

## The name

The interior series is the two-dimensional form of the mean-value expansion of **Paolo Pizzetti**
(1860–1918), Italian geodesist and mathematician (*Rend. Lincei* 18, 1909).

## Contributing

All my repositories embrace collaboration between humans and AI. Contributions are welcome, whether they are written by you, by an AI assistant, or by both together, and they will be handled promptly. Please open an [issue](https://github.com/hippke/pizzetti/issues) or a pull request.

## Citing

Please cite the accompanying paper (Hippke et al., in prep.) and Mandel & Agol (2002); see
[`CITATION.cff`](CITATION.cff). License: GPL-3.0-or-later.
