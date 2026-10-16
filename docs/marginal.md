# Fitting with the limb darkening integrated out

`MarginalTransitModel` turns a light curve into a likelihood in the **geometry only**:
- the parameters are t0 (or t_secondary), per, rp, a, b, optionally ecc and w, and α for the
  power-2 law;
- the limb-darkening coefficients and the out-of-transit flux are integrated out
  analytically over their physical region;
- they are drawn afterwards, once per posterior sample.

## The Bernstein cone

For a fixed geometry θ, every supported law is linear in its coefficients. Together with the
baseline flux, the model is

$$ m = A(\theta)\, c , $$

where the columns of $A$ are basis light curves. For the quadratic law, write the unnormalised
intensity in the degree-2 Bernstein basis,

$$ I(\mu) = c_0 (1-\mu)^2 + 2 c_1 \mu (1-\mu) + c_2 \mu^2 . $$

Kipping's (2013) conditions for a physical profile (positive everywhere and darkening towards
the limb) are then exactly

$$ 0 \le c_0 \le c_1 \le c_2 , $$

which is a polyhedral **cone**. Kipping's sampling coordinates $(q_1, q_2)$ are Bernstein
coefficients in disguise. A flat prior on $c$ over the cone, with a flat baseline, equals his
uniform prior on $(q_1, q_2)$ up to the importance weight $f_0 / c_2^3$, which is applied when
the coefficients are drawn.

For Gaussian noise the integral over the cone has a closed form, apart from one orthant
probability:

$$ \log L(\theta) = -\tfrac12 \chi^2_{\min}(\theta) - \tfrac12 \log\det F(\theta)
   + \log P\big(c \in \text{cone} \mid \hat c, F^{-1}\big) + \text{const},
   \qquad F = A^\top \Sigma^{-1} A . $$

The orthant probability is computed deterministically (Genz separation of variables on a
fixed Gauss–Legendre grid, error ≲ 10⁻⁴ nats). The likelihood is therefore smooth in θ and
free of Monte Carlo noise. The quadratic law has a fused numba path: orbit, basis light curves
(`occult_terms`, one pass) and the 3×3 algebra together cost about 15–20 µs for 1300 points.

## Laws and physical regions

| law | linear coefficients | physical region (`ld_prior="physical"`) |
|---|---|---|
| uniform | f0 | f0 ≥ 0 |
| linear | 2 | 2-D cone |
| quadratic | 3 | Bernstein cone (Kipping 2013) |
| square-root | 3 | the same cone in √μ |
| logarithmic | 3 | polyhedral cone |
| power-2 | 2 (α fixed or free) | 0 ≤ c ≤ 1 cone at fixed α |
| nonlinear | 5 | curved convex region: its probability is computed by line integration |
| secondary eclipse | 2 | fp ≥ 0 |

Notes:
- **Logarithmic law:** pizzetti's `occult` does not support it. The μ ln μ basis is the
  α-derivative of μ^α, taken by a 5-point difference of power-law fluxes. Its accuracy is
  8.7 × 10⁻¹² against a 30-digit reference.
- **Nonlinear law:** it uses a Bernstein basis in √μ. The power basis is ill-conditioned:
  cond(F) ~ 10¹³, compared with ~10⁹ for the Bernstein basis.
  - The physical region (I(0) ≥ 0, and I′(s) ≥ 0 on [0, 1] with s = √μ) is a curved convex
    cone, not a polyhedral one.
  - Its probability under the Gaussian of the coefficients takes the place of the orthant
    term: log L = (unconstrained marginal) + log P(physical).
  - P is computed by line integration. Along the largest-variance direction of the coefficients,
    a line meets the region in an interval, found exactly from the cubic I′. The line contributes
    Φ(hi) − Φ(lo), and the other four directions use 256 fixed quasi-random points. The estimate is
    deterministic and smooth in the geometry, accurate to ~0.003 nats, and costs ~45 µs.
  - Afterwards the coefficients are drawn from the Gaussian truncated to the region, by
    rejection with the exact test.
  - With the constraint in the likelihood, the geometry posterior stays well defined at low S/N:
    `fit()` succeeds by importance sampling (k̂ 0.03–0.33) down to 300 ppm with 3 simulated
    transits. Before, the unconstrained geometry posterior was degenerate there and importance
    sampling failed.
- **`ld_prior="flat"`** removes the physical constraint and is flat in the absolute
  linear coefficients. This choice is always proper; a flat prior in the normalised
  coefficients is not, for example for the nonlinear law.
- **Other `ld_prior` forms:** a Gaussian prior on (u1, u2), given as `{"mean": ..., "cov": ...}`,
  is exact for the quadratic law. Any other prior can be passed as a callable and is applied by
  reweighting.

## Sampling

```python
model = pizzetti.MarginalTransitModel(params, t, y, yerr,
                                      free=["t0", "per", "rp", "a", "b"])
result = model.fit()            # method="auto": profile importance sampling, emcee fallback
print(result.summary())
result.samples, result.labels   # emcee-like flat array (geometry + LD + baseline)
result.weighted_samples, result.weights
result.diagnostics              # method, k-hat, ESS, physical_fraction, ...
```

The geometry posterior has 4–5 dimensions and is close to Gaussian, especially after the
internal reparametrisation from a/R⋆ to the total duration T14 (`reparam="auto"`, which keeps
the prior uniform in a). `fit()` proceeds in four steps:
1. It finds the maximum and the profile of the posterior.
2. It draws from a heavy-tailed proposal fitted to that profile.
3. It corrects the draws with exact importance weights, Pareto-smoothed and checked with k̂
   and the effective sample size. The grid is refined where interpolating the conditional
   mean between grid points would be off by more than 0.25 conditional sd (near the b = 0 wall
   it moves non-linearly). If k̂ is still too high, a second stage adds a multivariate t fitted
   to the weighted draws.
4. It falls back to emcee, run to 50 τ with Δτ < 1 %, if the diagnostics fail.

To run emcee directly, use `model.EnsembleSampler(nwalkers)`. This is an
`emcee.EnsembleSampler` that samples only the geometry, and its `get_chain` returns the limb
darkening and baseline as extra columns.

Example 01's data with each method run to convergence ([notebook 03](https://github.com/hippke/pizzetti/blob/main/examples/03_marginal_emcee_vs_is.ipynb)):

| | likelihood evaluations | wall time | effective samples / s |
|---|---|---|---|
| `TransitModel` + emcee (7-D) | 752 k | 39 s | 37 |
| `MarginalTransitModel` + emcee (5-D) | 336 k | 10.6 s | 248 |
| `MarginalTransitModel.fit()` | 17.8 k | 1.0 s | 5830 |

The posteriors agree within 0.11 σ. The timings were taken on a loaded machine, so the
evaluation counts are the more reliable measure.

## Cost per likelihood evaluation

n = 1316 points, rp = 0.1, single thread:

| law | single call | per row in batches of 64 |
|---|---|---|
| uniform, linear, secondary eclipse | ~20 µs | ~12 µs |
| quadratic (fast path) | ~19 µs | ~15 µs |
| square-root, power-2 (fixed α) | ~35 µs | ~25 µs |
| nonlinear | ~53 µs | ~39 µs |
| power-2 with free α | ~54 µs | ~44 µs |
| logarithmic | ~68 µs | ~60 µs |

All laws run in one numba kernel per batch: orbit, `occult_terms`, the power-law columns,
supersampling and the normal equations. The normal equations are equilibrated, refined once,
and take χ² from the explicit residual; they fall back to QR when ill-conditioned.

**Power-law columns.** These are the blocked fluxes of the non-integer powers μ^½, μ^{3/2},
μ^α. Interior samples use pizzetti's interior series (bound 10⁻¹²). Samples near the limb use
its contact expansions, but those normally have to be rebuilt for every new radius ratio. Here
they are computed once per panel instead:
- Panels are geometric in rp (width ×1.15); with α free they are 2-D, with α steps of 0.25.
- The coefficients are interpolated in rp (and α) with Chebyshev polynomials.
- Each panel is validated against Green's-theorem quadrature when it is first used: errors
  ≤ 10⁻¹² (typically 10⁻¹⁵). A panel that fails validation uses quadrature.
- The series are extended to degree 64, so that they reach the interior cut.
- Near α = 1 and 3 the expansions have cancelling poles; they are stored in a regularised form.

A new panel costs 10–100 ms, and panels are shared by all models in a session. The expansions
are validated for 0.001 ≤ rp ≲ 0.2 (and 0 < α < 3); outside that range the limb samples use the
quadrature (~1 µs each).

One-call `fit()` on 7 simulated transits, before and after these optimisations:

| law | before | after |
|---|---|---|
| uniform | 0.64 s | 0.40 s |
| linear | 0.92 s | 0.46 s |
| square-root | 6.7 s | 0.84 s |
| power-2 | 5.3 s | 0.75 s |
| power-2, free α | 7.8 s | 1.7 s |
| nonlinear | 11.3 s | 1.25 s |
| logarithmic | 18.9 s | 1.29 s |

## Limitations

- Not supported (they raise `NotImplementedError`): inverse transits (rp < 0), the
  exponential law, and an exact Gaussian `ld_prior` for laws other than quadratic.
- Limb brightening is excluded by the physical prior.
- emcee and scipy are optional dependencies: `pip install pizzetti[fit]`.
