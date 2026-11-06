# Fitting with the limb darkening integrated out

`MarginalTransitModel` turns a light curve into a likelihood in the **geometry only**:
- the parameters are t0 (or t_secondary), per, rp, a, b, optionally ecc and w, and α for the
  power-2 law;
- the limb-darkening coefficients and the out-of-transit flux are integrated out
  analytically over their physical region;
- they are drawn afterwards, once per posterior sample;
- optionally, correlated noise is fitted together with the transit ([below](#correlated-noise-a-gaussian-process)).

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

The orthant probability is computed deterministically on a fixed Gauss–Legendre grid (a ray
integral, or Genz separation of variables far inside the cone; error typically 10⁻⁵ nats,
≲ 10⁻³ down to ln P ≈ −30). The likelihood is therefore smooth in θ and free of Monte Carlo noise. The quadratic law has a fused numba path: orbit, basis light curves
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
result = model.fit()            # method="auto": profile importance sampling (no MCMC)
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
   and the effective sample size (accepted for k̂ < 0.5 and an ESS of at least 25% of the draws). The grid is refined where interpolating the conditional
   mean between grid points would be off by more than 0.25 conditional sd (near the b = 0 wall
   it moves non-linearly). If k̂ is still too high, a second stage adds a multivariate t fitted
   to the weighted draws.
4. If the diagnostics fail and the bounds allow grazing geometries, it splits the posterior at the
   grazing boundary b = 1 − Rp/R⋆ and integrates each side on its own, in coordinates that suit that side, with
   adaptive importance sampling per side where needed. `strata` gives each side's share of the posterior.
5. If that does not pass either, it repeats the construction in unbounded coordinates (logit/log of
   the bounded parameters; β = b/(1 + Rp/R⋆) for grazing posteriors) and, if needed, refines the
   pooled draws of all attempts by adaptive multiple importance sampling, with a component over the
   whole prior box, until k̂ < 0.7, the ESS is at least 25% of the draws and the quantiles are
   stable. It does not fall back to MCMC; `diagnostics["is_ok"]` says whether the draws pass, and
   `fit()` warns if they do not. Tutorials 12–16 explain each step.

The raw weights also give the Bayesian evidence, without nested sampling: `profile_is()` returns `lnz` and
`lnz_err`, and `fit()` puts them in `diagnostics`. This needs finite `bounds` on every free parameter, since `per`
and `a` have no upper bound by default, and the evidence depends on them. The prior is uniform over the bounds and
over (q₁, q₂). The flat prior on the baseline has unit density, so subtract ln(width) for a flat f₀ of that width.
Where the prior has no known normalisation (`log_prior`, a Gaussian or callable `ld_prior`, `gp`, free `ecc` or `w`,
other laws), `lnz` is nan and `lnz_note` says why; `log_int` is still the integral of the log posterior. Trust it only
if `is_ok`: `lnz_err` is a Monte Carlo error and cannot see a mode that no draw reached. On a grazing branch that reaches the
`rp` bound, the evidence depends on that bound.

**Eccentricity from a stellar density.** A transit hardly depends on e and ω once its duration and centre are fixed, so a
fit of a circular orbit serves any orbit: the density it implies is then wrong by a factor g³, g = (1 + e sin ω)/√(1 − e²),
which is the photoeccentric effect (Dawson & Johnson 2012). `fit().reweight_density(rho=(1.41, 0.14))` turns that around, with no new
likelihood evaluation. It repeats each draw with (e, ω) from their prior, with a/R⋆ and t₀ such that the duration
and the centre of the transit on the Keplerian orbit stay what they were, and weights it with N(ρ) |∂ρ/∂T14| / |∂a_circ/∂T14|, because the fit
is flat in the circular a/R⋆. The result has `ecc`, `w`, `b_conj` and `rho` as extra columns, and `a`, `b`, `t0` of the eccentric orbit. Fit with `ecc=0`
and bounds on `a` wide enough not to cut its posterior. The prior is flat in e, ω (or `log_prior`), rp and b_conj.
Orbits with periastron a(1 − e) ≤ 1 + rp are excluded. Prior-drawn light curves give calibrated ranks for all parameters up to an S/N of about 1000. Above that,
the intervals of `b_conj` and `rp` are too narrow, and a free fit of `ecc` and `w` with dynesty is the check ([tutorial 20](https://github.com/hippke/pizzetti/blob/main/examples/20_eccentricity_from_a_stellar_density.ipynb)).

To run emcee directly, use `model.EnsembleSampler(nwalkers)`. This is an
`emcee.EnsembleSampler` that samples only the geometry, and its `get_chain` returns the limb
darkening and baseline as extra columns.

A TESS-like hot Jupiter with each method run to convergence ([tutorial 06](https://github.com/hippke/pizzetti/blob/main/examples/06_marginal_transit_model.ipynb)):

| | likelihood evaluations | wall time | effective samples / s |
|---|---|---|---|
| `TransitModel` + emcee (7-D) | 752 k | 39 s | 37 |
| `MarginalTransitModel` + emcee (5-D) | 336 k | 10.6 s | 248 |
| `MarginalTransitModel.fit()` | 17.8 k | 1.0 s | 5830 |

The posteriors agree within 0.11 σ. The timings were taken on a loaded machine, so the
evaluation counts are the more reliable measure.

## Correlated noise: a Gaussian process

`gp=pizzetti.GP()` fits correlated noise together with the transit, instead of detrending first. The
light curve stays in time order and is not filtered:

```python
model = pizzetti.MarginalTransitModel(params, t, y, yerr, free=["t0", "rp", "a", "b"],
                                      gp=pizzetti.GP())
result = model.fit()
result.labels                    # t0, rp, a, b, ln_gp_amp, ln_gp_tau, ln_err_scale, q1, q2, f0
trend = model.gp_predict(np.median(result.samples, axis=0))
detrended = y - trend            # for one row of samples; gp_predict(row, t=times) between the data
```

[Tutorial 18](https://github.com/hippke/pizzetti/blob/main/examples/18_fitting_with_a_gaussian_process.ipynb)
goes through a real light curve.

**The model.** The noise covariance is

$$ \Sigma = K(\text{amp}, \tau) + \mathrm{diag}\big((s\, \sigma_i)^2\big), \qquad
   k(\Delta t) = \text{amp}^2\, e^{-x} (\cos x + \sin x), \quad x = \frac{|\Delta t|}{\sqrt{2}\, \tau} , $$

with σᵢ = `yerr`. The kernel is a stochastically driven, damped harmonic oscillator with
Q = 1/√2 (SHO; Foreman-Mackey et al. 2017): smooth, not periodic. The GP is additive and has zero mean, so its
constant part is degenerate with f0, and the sum of the two is what the data constrain.

**The marginalisation is unchanged.** With Σ = LLᵀ, the limb darkening and the baseline integrate
out as in the white-noise case, using L⁻¹A and L⁻¹y:

$$ \log L(\theta, \eta) = -\tfrac12 \chi^2_{\min} - \tfrac12 \log\det F
   + \log P(c \in \text{cone}) - \sum_i \log L_{ii} + \text{const}, \qquad F = A^\top \Sigma^{-1} A . $$

**Linear cost.** The SHO is a two-state linear stochastic differential equation. The innovations of its
Kalman filter, divided by their standard deviation, are exactly L⁻¹y. So whitening costs O(N) per column for any spacing of the times, with no matrix stored. It agrees with a dense
Cholesky factorisation to ~10⁻¹⁰ in log L. The transition over a time step is computed once per distinct step
(steps within 10⁻⁶ of each other share one, which changes log L by ~10⁻⁶): a few dozen per set of GP parameters on real light curves.

**Parameters.** Three more parameters are sampled, with log-uniform priors. Pass a tuple for other bounds, or a number to hold
a parameter fixed:

| argument | label | meaning | default bounds |
|---|---|---|---|
| `amp` | `ln_gp_amp` | rms of the GP, flux units | 0.02 × median `yerr` to 0.2 × median flux |
| `tau` | `ln_gp_tau` | time scale 1/ω₀, days | median cadence to 180 T14 |
| `err_scale` | `ln_err_scale` | factor s on `yerr` | 1/3 to 3 |

`log_prior` also receives these three, by label. `fit()`, `profile_is`, `FitResult`, `log_prob`, `expand` and
`EnsembleSampler` work as without a GP, with the extra columns.

**Sampling.** The GP adds up to three dimensions to the geometry. Where the data constrain them, the
profile along b works as usual. Where they do not, its first attempt fails: a weak GP gives only an upper limit on `amp`, with a flat `tau`, and a trend much slower than the windows fixes only about amp/τ. Then
step 4 above takes over. The bounded retry samples the GP parameters in logit coordinates, where a uniform plateau is
well behaved, and the rescue adds draws over their whole box. The priors matter on such a plateau, so the bounds are part of the model.

**Data.**
- `t` must be strictly ascending, so a folded light curve raises an error. Keep the epochs apart, with a fixed or free period.
- Keep enough out-of-transit data around each transit for the trend, for example ±3 T14.
- Clip outliers within each window, not on the fold: differences in level between windows look like outliers.
- Flares are not modelled.

**Fixed GP parameters.** With `amp`, `tau` and `err_scale` all numbers, on a circular orbit with a fixed period, the filter runs once. Each geometry then
only filters the points within 0.75 T14 + one exposure of a transit, and the stretches between them are one transition in
closed form (a row whose transit reaches beyond them takes the full filter). This is exact and costs 1.2–6 × less per evaluation on ±3 T14 windows, growing with the number of points, but it ignores the uncertainty of the GP parameters. For example, take their medians from a fit with free parameters: on 7 light curves with a circular orbit, plus AU Mic b on the full filter, the transit parameters then stayed within 0.05 σ of the free fit and their widths within 5%, and the fits took 1–6 s.

**Threads.** `threads=` spreads the rows of a batch over threads, as without a GP. The optimiser's single evaluations stay serial.

**Scope.** Quadratic law, primary transit, `ld_prior` "physical" or a callable. Anything else raises
`NotImplementedError`.

**Cost.** Single thread, on 14 raw light curves from Kepler, K2 and TESS (windows of ±3 T14, default priors):

| points × sub-exposures | per evaluation | evaluations | whole fit |
|---|---|---|---|
| 170 × 15 to 1300 × 15 | 0.1–0.4 ms | 13 000–52 000 | 1–22 s |
| 2000 × 3 to 8100 × 1 | 0.2–0.5 ms | 14 000–110 000 | 3–34 s |
| 9000 × 5 | 0.75 ms | 18 000 | 14 s |
| 28 000 × 1 | 1.0 ms | 13 000 | 13 s |
| 47 000 × 3 | 3.3 ms | 15 000 | 49 s |

The whole fit took a median of about 1.6 × as long as biweight detrending + `fit()` on half the points, without counting the filter. It took about 19 × as long
where the retries ran (GJ 436 b, LP 791-18 c, about 105 000 evaluations). The timings come from a loaded machine.

**How far it has been tested.** Its likelihood agrees with dense algebra. On simulated light curves the posterior of the transit parameters agrees with long `emcee` runs to about the noise of those runs, and for a trend near the transit's time scale the error bars of Rp/R⋆ stayed honest where
two-step fits did not (rms z of 0.55, against 1.6–1.8; 8 noise realisations). On the 14 real light curves all fits passed the diagnostics (k̂ < 0.45), and 10 of them agree with the biweight fits within 0.4 σ. GJ 436 b is at 0.9 σ in a and b and depends on two clipped in-transit points. WASP-62 b is at 1.1 σ with 40% narrower posteriors, because the biweight leaves red noise that its white-noise fit then counts as larger errors.

**When to be careful.** On very active stars (AU Mic b, DS Tuc A b), the amplitude is 6–8 × the depth and τ is close to T14. Rp/R⋆ then differs from biweight fits by 2–3 σ, and it is not yet established which is right: a single-time-scale GP may absorb part of the transit. Compare with a filter, and test with injected transits. A sampler can also miss a tail in the GP parameters without k̂ noticing (seen on a simulated trend near T14, with the transit parameters unaffected).


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
