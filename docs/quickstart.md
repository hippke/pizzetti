# Quickstart

## A transit light curve

```python
import numpy as np
import pizzetti

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
```

Later calls with changed parameters reuse the model. As in batman, the orbit is recomputed only
when an orbital parameter changes:

```python
params.rp = 0.11
params.u = [0.35, 0.3]
flux = m.light_curve(params)
```

## Limb-darkening laws

| `limb_dark` | `u` | intensity $I(\mu)$ |
|---|---|---|
| `"uniform"` | `[]` | $1$ |
| `"linear"` | `[u1]` | $1 - u_1(1-\mu)$ |
| `"quadratic"` | `[u1, u2]` | $1 - u_1(1-\mu) - u_2(1-\mu)^2$ |
| `"squareroot"` | `[u1, u2]` | $1 - u_1(1-\mu) - u_2(1-\sqrt\mu)$ |
| `"nonlinear"` | `[c1, c2, c3, c4]` | $1 - \sum_{n=1}^4 c_n (1-\mu^{n/2})$ |
| `"power2"` | `[c, alpha]` | $1 - c(1-\mu^\alpha)$ |

The conventions are identical to batman's.

## Supersampling, eccentric orbits, secondary eclipses

```python
m = pizzetti.TransitModel(params, t, supersample_factor=7, exp_time=0.0204)  # Kepler long cadence

params.ecc = 0.3
params.w = 60.0
flux = m.light_curve(params)

params.fp = 0.001
params.t_secondary = 1.5
ms = pizzetti.TransitModel(params, t + 1.5, transittype="secondary")
eclipse = ms.light_curve(params)
```

## Low-level interface

`pizzetti.occult` evaluates the flux for given projected separations `z` (in stellar radii):

```python
z = np.linspace(0, 1.2, 1000)
f = pizzetti.occult(z, 0.1, [0.4, 0.25], "quadratic")
```

`pizzetti.series_cut` returns the separation $z_c$ up to which the interior series is used. Its
flux error is rigorously below `series_tol` there:

```python
z_c = pizzetti.series_cut(0.1, "quadratic", [0.4, 0.25], series_tol=1e-9)
```

Pass `series_tol=0` to `occult` or `TransitModel` to switch the series off. This gives the
reference solution (exact for the quadratic family, quadrature for the other laws).
