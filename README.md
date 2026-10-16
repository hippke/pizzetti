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
(uniform, linear and quadratic laws) or, for the other laws, power series about first and second contact with
coefficients computed once per light curve (accurate to ~1e-15, 15-22x faster per sample than quadrature, radius
ratios up to ~0.2). Building these expansions costs about 10 µs per exponent and series term,
so they are used only when enough samples lie near the limb (e.g. ≳ 430 for rp = 0.1). Light
curves with fewer limb samples, and larger occulters, use a Green's-theorem quadrature with a
Gauss–Jacobi rule for the endpoint singularity at the limb (~1e-16, ~1 µs per limb sample and non-integer exponent).

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
- **The limb can be proven too.** With `limb="proven"` the contact expansions are split into
  pieces whose degrees are chosen so that their flux error is also proven to be below `series_tol`
  (radius ratios up to ~0.2 at 10⁻⁹, ~0.15 at 10⁻¹²). The proof costs 5–60 ms of set-up per light
  curve, so it is opt-in; `occult(..., return_certified=True)` reports which samples it covers.
  The default (`limb="fast"`) uses the same expansions at fixed order with an estimated error of ~10⁻¹⁵.

## Inference

Same posterior, less time. Two synthetic data sets were fitted with emcee, once with batman and once
with pizzetti, with identical likelihoods, starting positions and random numbers
([examples/inference](examples/inference), paper Section 6.4):

| | hot Jupiter, TESS 2-min, quadratic law | super-Earth, Kepler 1-min, power-2 law |
|---|---|---|
| largest difference of the posterior medians | 0.08 σ | 0.04 σ |
| model error over the posterior, batman / pizzetti | 9 × 10⁻⁹ / 7 × 10⁻¹⁰ | 1.9 × 10⁻⁶ / 5 × 10⁻¹⁰ |
| one likelihood evaluation, batman / pizzetti | 0.130 / 0.029 ms (**4.5×**) | 1.10 / 0.57 ms (**1.9×**) |
| 2000-step emcee run, batman / pizzetti | 9.8 / 3.0 s | 67 / 34 s |

The posteriors agree within their Monte Carlo noise. To come close to pizzetti's accuracy for the
power-2 law, batman needs `max_err=0.01`, at 6.1 ms per evaluation (still off by 4 × 10⁻⁸). The
Julia version computes the gradient of either likelihood with respect to all seven parameters in
about ten likelihood evaluations ([julia/](julia)).

Notebooks: [drop-in replacement in an emcee fit](examples/01_batman_dropin_emcee.ipynb),
[power-2 super-Earth](examples/02_power2_superearth.ipynb).

![corner plots](examples/inference/results/corner_kepler_paper.png)

## Limb darkening integrated out: `MarginalTransitModel`

`MarginalTransitModel` fits a transit or secondary eclipse with the limb darkening and the
out-of-transit flux **integrated out analytically**. The sampler only explores the geometry
(t0, per, rp, a, b, optionally ecc, w), and the limb darkening is drawn afterwards for every
posterior sample.

**The Bernstein transform.** Write the unnormalised quadratic intensity profile in the degree-2
Bernstein basis of μ:

    I(μ) = c0 (1−μ)² + c1 · 2μ(1−μ) + c2 μ²

Three facts follow:

- **The physical region is a cone.** Kipping's (2013) conditions for a positive profile that
  darkens towards the limb become `0 ≤ c0 ≤ c1 ≤ c2`. His triangle sampling coordinates (q1, q2)
  turn out to be Bernstein coefficients, so a flat prior on c over the cone is Kipping's prior
  up to a known weight.
- **The flux is linear in c**, baseline included: `m = A(θ) c`, where A holds three basis light
  curves for the geometry θ. pizzetti computes them in one pass (`occult_terms`).
- **The Gaussian integral over the cone is closed-form** except for the probability that a
  trivariate normal lies in an orthant. That term is evaluated deterministically (Genz), so the
  likelihood is smooth in θ:

      log L(θ) = −½ χ²_min(θ) − ½ log det F(θ) + log P(c ∈ cone) + const,   F = AᵀA/σ²

**The sampler.** Without the limb darkening and baseline, the posterior has 4–5 dimensions and is
close to Gaussian. `model.fit()` uses that with **profile importance sampling**:
- it finds the maximum and the profile shape;
- it draws from a heavy-tailed fit to them;
- it corrects the draws with exact importance weights (Pareto-smoothed, with k̂ and ESS
  diagnostics);
- it falls back to emcee automatically when the diagnostics fail.

There is no burn-in and no autocorrelation.

```python
model = pizzetti.MarginalTransitModel(params, t, y, yerr, free=["t0", "per", "rp", "a", "b"])
result = model.fit()                      # profile IS, emcee fallback
print(result.summary())                   # rp, a, b, ... and the limb-darkening coefficients
result.corner()                           # weighted draws

sampler = model.EnsembleSampler(32)       # or emcee directly, on the 5-D geometry only
```

Example 01's data, each method run to convergence (50 τ), on a loaded machine
([notebook 03](examples/03_marginal_emcee_vs_is.ipynb)). The posteriors agree within 0.11 σ.

| | likelihood evaluations | wall time | effective samples / s |
|---|---|---|---|
| `TransitModel` + emcee (7-D) | 752 k | 39 s | 37 |
| `MarginalTransitModel` + emcee (5-D) | 336 k | 10.6 s | 248 (6.7×) |
| `MarginalTransitModel.fit()` | 17.8 k | 1.0 s | 5830 (157×) |

**Other laws.** `params.limb_dark` selects the law, each with its own physical region:

| law | physical region |
|---|---|
| uniform, linear | 1- and 2-D cones |
| quadratic, square-root | the cone above (square-root: quadratic in √μ) |
| logarithmic | a cone; the μ ln μ basis is computed here (pizzetti's `occult` has no logarithmic law) |
| power-2 | a cone at fixed α; α may also be free |
| nonlinear | a curved region; its probability is computed by line integration |

**Secondary eclipses:** use `transittype="secondary"`. They are marginalised over the planet flux
fp ≥ 0 and the baseline.

**Priors:** `ld_prior="physical"` (default) is flat over the law's physical region. `"flat"`
drops the constraint. A Gaussian prior on (u1, u2) is exact for the quadratic law, and any prior
can be passed as a callable.

**Extras:** supersampling and per-point errors work as in `TransitModel`. emcee and scipy are
optional: `pip install pizzetti[fit]`.

**Cost of one likelihood evaluation** (n = 1316, single call):

| law | per evaluation |
|---|---|
| uniform, linear, secondary eclipse | ~20 µs |
| quadratic | ~19 µs |
| square-root, power-2 | ~35 µs |
| nonlinear | ~53 µs |
| power-2 with free α | ~54 µs |
| logarithmic | ~68 µs |

Every law runs in a fused numba kernel. For the non-integer powers of μ, the contact expansions
at the limb are precomputed once per fit on small panels of radius ratio (and α), interpolated
in between, and checked against quadrature before use. A one-call fit takes 0.4–1.7 s for every
law (7 transits). See [performance](docs/performance.md).

Not supported yet: inverse transits (rp < 0) and the exponential law.

## Speed against batman

Single thread, 10⁵ points, about 30 % of them in transit (`benchmarks/benchmark.py`, Intel Xeon
2.8 GHz, numba 0.68). "Full" means that every call also recomputes the orbit, as in a fit.

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 2.42 ms | 0.15 ms | **17×** | 5.96 ms | 0.40 ms | **15×** |
| quadratic | 0.1 | 3.01 ms | 0.32 ms | **9.5×** | 6.57 ms | 0.61 ms | **11×** |
| nonlinear | 0.0092 | 3.13 ms | 1.07 ms | 2.9× | 6.69 ms | 1.41 ms | 4.8× |
| nonlinear | 0.1 | 57.1 ms | 2.73 ms | **21×** | 67.4 ms | 3.87 ms | **17×** |
| power2 | 0.0092 | 2.00 ms | 0.83 ms | 2.4× | 5.25 ms | 1.16 ms | 4.5× |
| power2 | 0.1 | 35.9 ms | 1.77 ms | **20×** | 39.7 ms | 2.13 ms | **19×** |

For the quadratic law, pizzetti skips blocks of samples that are entirely off the star (separation
≥ 1 + rp, flux exactly 1) without evaluating the series. This matters most for long light curves
with a small transit duty cycle: with 3 % of the points in transit
(`python benchmarks/benchmark.py --half-width 2.5`) the quadratic flux takes 0.09 ms (rp = 0.0092)
and 0.11 ms (rp = 0.1) for 10⁵ points, 7× and 6.5× faster than batman, and 2.7× and 2.4× faster than
before the block skipping.

batman's numerical laws (nonlinear, power2) run at its default tolerance of 1 ppm; pizzetti's
errors stay below 10⁻⁹. In a complete emcee fit the likelihood is 1.9–4.5 times faster
([examples/inference](examples/inference)).

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
  of μ, so `pizzetti` raises `NotImplementedError` for them. (`MarginalTransitModel` does fit the
  logarithmic law; it builds the μ ln μ term from power-law fluxes.)

## Examples and inference

[`examples/`](examples/) has notebooks that fit synthetic TESS- and Kepler-like light curves with emcee,
once with batman and once with pizzetti (same likelihood, starting points and random numbers). The posteriors
agree within their Monte Carlo noise; the likelihood evaluations are 4.5x (quadratic law, hot Jupiter) and
1.9x (power-2 law, super-Earth) faster, and pizzetti's model error over the posterior is below 1e-9, against
9e-9 and 1.9e-6 for batman. See [`examples/inference`](examples/inference) for the full runs.

## Julia

[`julia/`](julia/) contains **Pizzetti.jl**, a Julia implementation of the same methods with
ForwardDiff support (exact derivatives of the flux from the paper's Section 5.4), BigFloat evaluation,
rigorous interval enclosures, and Limbdark.jl's `transit_poly` interface.

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
