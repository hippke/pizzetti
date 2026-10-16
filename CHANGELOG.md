# Changelog

## Unreleased

- `MarginalTransitModel`: transit and secondary-eclipse likelihood with the limb darkening and
  the baseline flux integrated out analytically. In the Bernstein basis the physical
  limb-darkening region is a cone (quadratic law: 0 <= c0 <= c1 <= c2, Kipping 2013), and the
  Gaussian integral over it is closed-form up to a deterministic orthant probability.
  - Laws: uniform, linear, quadratic, square-root, logarithmic (μ ln μ basis from power-law
    fluxes), power-2 (α fixed or free), nonlinear (Bernstein basis in √μ, exact physical prior
    by importance weights).
  - `ld_prior`: "physical" (default), "flat", Gaussian (quadratic law), or a callable.
  - `fit()`: profile importance sampling with PSIS diagnostics, falling back to emcee; it
    returns emcee-like flat samples plus the weighted draws.
  - `EnsembleSampler`: an emcee subclass that samples only the geometry.
  - On example 01: 42× fewer likelihood evaluations than `TransitModel` + emcee, with the same
    posterior.
  - New optional extra: `pip install pizzetti[fit]` (emcee, scipy).
- Faster general path of `MarginalTransitModel`: 5–15× for the square-root, power-2,
  nonlinear and logarithmic laws, 2× for uniform/linear.
  - One fused numba kernel per batch replaces the Python per-row loop.
  - Equilibrated, refined normal equations replace QR (QR remains the fallback).
  - New module `_powbasis`: the blocked fluxes of non-integer powers of μ come from contact
    expansions precomputed per panel of radius ratio, interpolated, and validated against
    quadrature. Free α uses 2-D panels, with a regularised form at α = 1, 3.
  - Accuracy: ≤ 1e-12 against quadrature. The log-likelihood is unchanged within 3e-8 nats;
    at α = 1 exactly, the new column is more accurate than the old `occult_terms`-based one.
  - One-call fits on 7 transits now take 0.4–1.7 s for every law.
- Nonlinear law, physical prior: the probability of the curved physical region is now part of
  the likelihood. It is computed by line integration with fixed quasi-random points
  (`_physnl`, ~0.003 nats, smooth in the geometry), and the coefficients are drawn by
  rejection. Profile importance sampling now works at low S/N (300 ppm, 3 transits), where the
  previous scheme (unconstrained likelihood, physicality applied afterwards) left a degenerate
  geometry posterior.
- Profile importance sampling: adaptive refinement of the b-grid (`refine_tol`,
  `max_refine`). On 8 Kepler Q3 light curves × 3 seeds: 24/24 successful (k̂ ≤ 0.46), against
  2 failures before (Kepler-2 LC, k̂ 0.70 / 1.03).
- `occult_terms` / `quadratic_from_terms`: the three quadratic-family basis fluxes in one pass.
- Eccentric orbits: faster `rsky`.
- Example 03: marginal model with emcee and with `fit()`, compared with `TransitModel` and batman.
- Example 04: tutorial that fits one light curve with `fit()` for every limb-darkening law and compares the geometry, the
  intensity profiles and the residuals. Example 03's scope note now lists all supported laws.

## 0.2.1

- Pizzetti.jl: `evaluate!` skips runs of 64-sample blocks that lie entirely off the star
  (|z| >= 1 + p), for all laws. Fluxes are unchanged; with 3 % of 10^5 samples in transit the
  quadratic flux is up to 5x faster, the power-2 flux up to 4x. An array with every sample on the
  star costs 0.1-0.3 ns per sample more.
- Quadratic family: blocks of samples that lie entirely off the star (separation >= 1 + rp)
  are filled with 1 without evaluating the series, and are skipped when the samples for the exact
  solver are collected. The fluxes are unchanged; with 3 % of 10^5 points in transit the flux is
  2.2-2.8x faster (0.24 -> 0.09 ms at rp = 0.0092), with 30 % in transit 1.1-1.7x.
- Examples (notebooks, inference timings) re-run with this version.

## 0.2.0 and earlier

- Interior series with a rigorous error bound for all power-law intensity laws.
- Exact vectorised Mandel & Agol solver near the limb (uniform, linear, quadratic laws).
- Contact expansions near the limb for the square-root, nonlinear and power-2 laws: power series
  about first and second contact (with the explicit non-analytic term at second contact), built
  per light curve from a single-variable hypergeometric form of the Mandel & Agol (2002) solution.
- Optional proven limb mode (`limb="proven"`): the contact expansions split into pieces whose
  degrees are chosen so that the flux error is proven to be below `series_tol` (radius ratios up to
  ~0.2); `return_certified=True` reports which samples the proof covers. Costs 10-100 ms of set-up
  per light curve, so the fixed-order expansions remain the default.
- Green's-theorem quadrature as fallback (short light curves, radius ratios above ~0.25, z <= p).
- batman-compatible `TransitParams` / `TransitModel`: eccentric orbits, supersampling, secondary
  eclipses, inverse transits.
