![Logo](https://raw.githubusercontent.com/hippke/pizzetti/main/docs/logo.png)


# pizzetti

**Fast limb-darkened transit light curves, with an elementary series and a proven error bound.**

`pizzetti` computes exoplanet (and exomoon) transit light curves through the same interface as
[batman](https://github.com/lkreidberg/batman). Code written for batman runs unchanged after a
one-line change:

```python
import pizzetti as batman
```

## Why it is fast

Since Mandel & Agol (2002), transit codes have evaluated complete elliptic integrals of all three
kinds, or integrated the stellar disk numerically, at every point of the light curve. `pizzetti`
does not need any of that for the main part of a transit.

While the planet is entirely inside the stellar disk, between second and third contact, the flux
has an **elementary** form: `sqrt(1 - z²)` times a polynomial in `1/(1 - z²)`. Its coefficients
depend only on the radius ratio, so they are computed once per light curve. Each in-transit point
then costs **one square root, one division and a dozen multiply–adds**. There are no branches, no
iterations and no special cases.

| | elliptic-integral solution | pizzetti interior series |
|---|---|---|
| square roots + divisions per point | ~14 | **2** |
| time per point (vectorised, numba) | 10.8 ns | **0.97 ns** |

For an Earth-sized planet crossing a Sun-like star, the series covers **96–98 %** of the
in-transit points (central transit: 98 %; averaged over impact parameters: 96 %). The few points that touch the stellar limb use an exact, vectorised Mandel & Agol solver
(uniform, linear and quadratic laws) or a high-order Green's-theorem quadrature (other laws).

## Proven correctness

- **The error bound is proven, not estimated.** All terms of the series have the same sign, which
  gives a closed-form bound on the truncation error. The bound increases monotonically along the
  transit chord, so a single cut per light curve guarantees the requested accuracy. The default
  is 10⁻⁹ of the stellar flux, i.e. 0.001 ppm.
- **It is verified against a 40-digit reference.** Every law agrees with an independent mpmath
  quadrature to **2 × 10⁻⁹**, including points within 10⁻¹⁵ of the contact points, at z = 0 and
  at z = rp.
- **It is more accurate than batman.** For the power-2 and nonlinear laws batman's adaptive
  integration leaves errors of ~10⁻⁷ even at its tightest setting; `pizzetti` stays below 10⁻⁹.
  Kepler's equation is solved to 10⁻¹⁵ instead of 10⁻⁷.

## Speed against batman

Single thread, 10⁵ points, about 30 % of them in transit (`benchmarks/benchmark.py`, Intel Core
Ultra 5 226V). "Full" means that every call also recomputes the orbit, as in a fit.

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 1.79 ms | 0.16 ms | **11.0×** | 3.49 ms | 0.40 ms | **8.7×** |
| quadratic | 0.1 | 1.91 ms | 0.25 ms | **7.6×** | 3.78 ms | 0.52 ms | **7.2×** |
| nonlinear | 0.0092 | 1.88 ms | 0.58 ms | 3.2× | 3.90 ms | 0.85 ms | 4.6× |
| nonlinear | 0.1 | 30.9 ms | 5.7 ms | 5.4× | 33.2 ms | 6.2 ms | 5.4× |
| power2 | 0.0092 | 1.06 ms | 1.05 ms | 1.0× | 2.97 ms | 1.35 ms | 2.2× |
| power2 | 0.1 | 19.2 ms | 6.7 ms | 2.9× | 21.4 ms | 7.0 ms | 3.1× |

## Installation

```bash
pip install pizzetti  # recommended
pip install git+https://github.com/hippke/pizzetti  # latest dev build
```

Requirements: Python ≥ 3.9, NumPy, numba. The first call compiles the kernels (a few seconds);
they are cached on disk afterwards.

## Usage

```python
import pizzetti
import numpy as np
import matplotlib.pyplot as plt

params = pizzetti.TransitParams()
params.t0 = 0.0          # time of inferior conjunction
params.per = 3.0         # orbital period
params.rp = 0.1          # planet radius (stellar radii)
params.a = 15.0          # semi-major axis (stellar radii)
params.inc = 88.0        # inclination (degrees)
params.ecc = 0.0         # eccentricity
params.w = 90.0          # argument of periapse (degrees)
params.u = [0.4, 0.25]   # limb-darkening coefficients
params.limb_dark = "quadratic"

t = np.linspace(-0.1, 0.1, 10_000)
m = pizzetti.TransitModel(params, t)
flux = m.light_curve(params)

plt.plot(t, flux)
plt.xlabel("Time from mid-transit (days)")
plt.ylabel("Relative flux")
plt.show()
```

The low-level function works on separations directly:

```python
z = np.linspace(0, 1.2, 1000)                     # projected separation (stellar radii)
f = pizzetti.occult(z, 0.1, [0.4, 0.25], "quadratic")
z_c = pizzetti.series_cut(0.1, "quadratic", [0.4, 0.25])   # series used for z <= z_c
```

### batman features supported

- `TransitParams` and `TransitModel` with the same attributes and methods: `light_curve`,
  `get_t_periastron`, `get_t_secondary`, `get_t_conjunction`, `get_true_anomaly` and `calc_err`.
- Limb-darkening laws: `uniform`, `linear`, `quadratic`, `squareroot`, `nonlinear` (four-parameter)
  and `power2`.
- Circular and eccentric orbits, supersampling (`supersample_factor`, `exp_time`), secondary
  eclipses (`transittype="secondary"`, `fp`, `t_secondary`) and inverse transits (`rp < 0`).
- Not supported: the `logarithmic`, `exponential` and `custom` laws. They are not sums of powers
  of μ, so `pizzetti` raises `NotImplementedError` for them.

## Documentation

The full documentation, covering the method, the accuracy proofs and checks, the benchmarks and
the API, is on [Read the Docs](https://pizzetti.readthedocs.io) and in `docs/`.

## The name *pizzetti*

The series is the two-dimensional form of the mean-value expansion of **Paolo Pizzetti**
(1860–1918). He gave the mean of a function over a sphere as a series of its iterated Laplacians,
with an exact remainder (*Rend. Lincei* 18, 1909). Pizzetti was an Italian geodesist and
mathematician; geodesists still use the Somigliana–Pizzetti formula for normal gravity on the
reference ellipsoid. A century later, his formula turns the transit of a planet into a polynomial.

## Citing

If you use `pizzetti`, please cite the accompanying paper (in preparation) and Mandel & Agol
(2002). See `CITATION.cff`.
