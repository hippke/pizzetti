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
| nonlinear | 5 | curved convex region: exact physicality test + importance weights |
| secondary eclipse | 2 | fp ≥ 0 |

Notes:
- **Logarithmic law:** pizzetti's `occult` does not support it. The μ ln μ basis is the
  α-derivative of μ^α, taken by a 5-point difference of power-law fluxes. Its accuracy is
  8.7 × 10⁻¹² against a 30-digit reference.
- **Nonlinear law:** it uses a Bernstein basis in √μ. The power basis is ill-conditioned:
  cond(F) ~ 10¹³, compared with ~10⁹ for the Bernstein basis, and the solve uses QR. The
  physical region is not polyhedral. The likelihood therefore integrates over all coefficients,
  and the physical prior is applied when the coefficients are drawn: 64 draws per sample, kept
  with weight equal to their physical fraction. This is exact, but it needs enough
  signal-to-noise: at 100 ppm and 3 transits only ~12 % of the draws are physical, and the
  geometry posterior without the constraint becomes broad. At modest S/N, use a
  Gaussian or fixed limb darkening for this law.
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
   and the effective sample size.
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

n = 1316, rp = 0.1, 196 samples near the limb:

| law | per evaluation | dominated by |
|---|---|---|
| quadratic | 15–20 µs | fused path |
| uniform, linear | 40–60 µs | Python overhead of the general path |
| square-root, power-2 | ~0.3 ms | limb quadrature for μ^½ / μ^α |
| nonlinear | ~0.6 ms | two non-integer exponents |
| logarithmic | ~1.3 ms | four power-law calls for μ ln μ |

## Limitations

- Not supported (they raise `NotImplementedError`): inverse transits (rp < 0), the
  exponential law, and an exact Gaussian `ld_prior` for laws other than quadratic.
- Limb brightening is excluded by the physical prior.
- emcee and scipy are optional dependencies: `pip install pizzetti[fit]`.
