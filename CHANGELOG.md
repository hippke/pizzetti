# Changelog

## Unreleased

- `fit()`'s ESS floor is now `min_ess=4000` (was 2000): at 2000 a corner plot has about 2000 distinct points and looks
  visibly noisy (tutorial 11, WASP-18 b). With the two speed-ups below, a fit still needs fewer likelihood evaluations
  than with 0.5.0: 94 benchmark light curves x 2 seeds, 1.24x fewer summed (1.16x median); ESS median 4100.
- The conditional fits of the profile grid use BFGS with a finite-difference step and tolerance of 1e-3 conditional sd
  (scipy's default step, 1.5e-8, is below the numerical noise of log p, so BFGS searched on near the optimum) and a loose
  Powell pre-step (xtol 1e-2, ftol 1e-3; was 1e-3, 1e-8). At the 2000 floor this needs 1.63x fewer evaluations summed
  on the benchmark, at the same accuracy.
- The limb-darkening step of `fit()` reuses the normal equations the likelihood computed at each draw instead of a
  second pass over the light curve (output identical; 1.23x faster summed on 12 benchmark light curves).

## 0.5.0

- `fit()` has a new default sampler: randomised quasi-Monte Carlo draws (`qmc=True`; scrambled Sobol points through the
  inverse CDFs of the proposals) and a stopping rule. It starts with `n=2048` draws per stage and adds draws from the
  final proposal until the Monte Carlo error of the posterior quantiles, measured from 8 independently scrambled
  replicates, is at most `mc_tol=0.03` posterior sd (`diagnostics["mc_err"]`) and the output's effective sample size is
  at least `min_ess=2000` (corner plots). On the 100 benchmark light curves of the paper (3 seeds) it needs 1.10 x fewer
  likelihood evaluations in total than 0.4.1 (1.14 x median), every output has ESS >= 2000, and the largest quantile
  deviation from the emcee references fell from 0.45 to 0.27 sd. Seeded results differ from 0.4.1;
  `fit(qmc=False, mc_tol=None, min_ess=None, n=4000)` reproduces them exactly.
- `pizzetti._sampling.mc_options` sets these options for `profile_is` and `pmc_refine` when called directly.

## 0.4.1

- Fixed: `fit()` with `gp=` could raise `ValueError: zero-size array to reduction operation maximum` (or `LinAlgError`)
  in the rescue when its GP seed started next to a bound, e.g. a/R* close to its upper limit. The Hessian's steps
  crossed the bound, the direction got no curvature and an unbounded proposal width, and no draw was inside the
  support. The steps now shrink until they stay inside, the widths are at most 50 x the default scales, and a seed
  without finite draws returns zero weights (the rescue skips it). Seen in 40 of the TLS-detrend candidate fits.

## 0.4.0

- Tutorial 21, the Grand Tour with simultaneous GP: from the raw K2 light curve of K2-33 to the planet's posterior
  without detrending, with the GP search of TLS 2.0 and `gp=pizzetti.GP()`. Tutorial 17 is now the Grand Tour with
  wotan; both are linked from the README. The examples workflow installs TLS from GitHub (the GP search is not on PyPI
  yet).
- Tutorial 18 now fits K2-33 b, a young star whose transits sit on steep, curved trends (K2-24 b was nearly flat), and
  compares the GP with a biweight filter and a robust Huber spline. Linked from the README.
- Seeded runs no longer depend on whether the kernels were just compiled or came from the cache (or on which kernel was
  loaded first in a process). numba keeps a copy of each kernel in every caller, and with `fastmath=True` the copies rounded
  differently (a few 1e-14 on the limb, 1e-12 in the log likelihood, which an optimiser amplifies to a different path:
  tutorial 19 took 11336 evaluations after compiling and 12128 from the cache). The kernels no longer allow reassociation
  (`_cachecheck.FASTMATH`). Seeded numbers change once. Time per evaluation: quadratic +2 %, power-2 +7 %, linear +12 %;
  light curves +5 %.
- `FitResult.reweight_density(rho=(mean, sd))`: the posterior of e and omega from a stellar density, after a fit of a
  circular orbit, with no new likelihood evaluation. Each draw is repeated with (e, omega) from their prior, keeping the
  duration and the centre of the transit (exact contact times on the Keplerian orbit), and weighted with the exact Jacobian
  of the density. One fit serves any density and any prior on (e, omega) (`log_prior`). Calibrated on 1100 prior-drawn light
  curves, with reference posteriors from dynesty and a grid in (e, omega); above an S/N of about 1000 the intervals of
  b_conj and rp are too narrow (a 95 % interval covers 86 % and 92 %). Tutorial 20.
- `profile_is`: `strata[...]["mass_err"]` of the two sides now has the same value, from both integrals; each side's used
  only its own integral, which understated the error.
- `profile_is` / `fit()`: where the first profile is not usable and the bounds allow grazing geometries, the posterior is
  split at the grazing boundary b_conj = 1 - rp and each side is integrated on its own, with no overlap: on the grazing
  side a profile in (ln lambda, v), lambda = rp^2 (1 + gamma) about twice the depth, v the position along the ridge between
  fixed walls. Adaptive importance sampling runs per side only if its diagnostics fail. About two times fewer evaluations
  than the bounded retry and rescue where it applies (1.9 on 20 of the paper's light curves, 1.3 at the median of 120
  grazing injections, where a third still falls back), the same ln Z within 0.05 nats, but it stops at an ESS of 1000 to 3000,
  where the retry reached about 12 000, so seeds scatter more. New: the share of the
  posterior on each side with its error (`strata`), down to masses no draw of a joint sampler would reach; `stratified`,
  `strata_note`, `diagnostics["grazing_mass"]`. The old attempts remain the fallback; `stratify=False` skips the strata.
  Seeded runs with a usable first profile do not change. The split follows the windows of Gilbert (2022).
- `profile_is`: where no attempt has any draws (a nearly flat posterior at S/N of about 3 with wide bounds: no conditional
  Hessian anywhere), the rescue now starts from draws over the prior box instead of returning nothing.
- `fit` no longer stops with `LinAlgError: Matrix is not positive definite`. `expand` outside the quadratic fast path:
  a geometry whose coefficient covariance is numerically singular gets weight 0, as in the fast path, and counts in
  `diagnostics["ld_dropped_weight"]`; a `RuntimeError` that says why if no geometry is left. `profile_is`, `laplace_is`,
  `pmc_refine`: the proposal covariance from weighted draws is floored relative to its diagonal; the absolute floor
  could be lost to rounding; `pmc_refine` stops if one draw holds all the weight. `ld_prior="flat"` is not proper where
  the planet's area vanishes, contrary to the docstring: the marginal likelihood grows as rp^(-2 (K - 1)) for rp -> 0,
  with coefficients ~ 1 / rp^2 that can still give the observed dip. `fit` warns if the posterior has collapsed there
  (`diagnostics["collapsed_fraction"]`).
- Bayesian evidence from profile importance sampling: `profile_is` returns `lnz`, `lnz_err` (and `fit` puts them in
  `diagnostics`), from the raw weights of the returned draws, with the Gaussian constants, the prior volume and the
  limb-darkening weight. For the quadratic law, `ld_prior="physical"`, finite `bounds` and no `log_prior`, `gp`, free
  `ecc` or `w`; otherwise nan and `lnz_note` says why. The flat prior on f0 has unit density: subtract ln(width).
  `log_int`, `log_int_err` are the bare integral of `log_prob_internal`, for any law and prior. Tested against
  quadrature and dynesty (`tests/test_evidence.py`, grazing cases included). `profile_is(evidence=False)` skips the 1000
  limb-darkening draws; seeded draws and evaluation counts do not change either way.
- `MarginalTransitModel(..., gp=pizzetti.GP())`: an additive Gaussian process in the noise model, fitted together
  with the transit instead of detrending first (quadratic law, primary transit). SHO kernel with Q = 1/sqrt(2),
  whitened by a Kalman filter at O(N) per column; the limb darkening and f0 are integrated out as before. New
  sampled columns `ln_gp_amp`, `ln_gp_tau`, `ln_err_scale` (log-uniform priors, bounds or fixed values in `GP`;
  also passed to `log_prior`), and `gp_predict(row, t=None)` for the trend, at the data or at any times. `t` must be ascending. Models without `gp` are
  unchanged. New tutorial 18.
- GP speed (ideas from the GP search in the TLS package): transitions per distinct time step (1.1-1.6 x per evaluation);
  `threads=` over the rows of a batch; with all GP parameters fixed, on a circular orbit with a fixed period, one filter per
  light curve and only the points around the transits per geometry (1.2-6 x per evaluation, exact). Log L changes by ~1e-6.
- Cone probability of the quadratic marginal likelihood (`marg_theta_parts`) where the least-squares limb
  darkening lies outside the cone: the ray integral's outer quadrature was centred on the Gaussian's mean, while
  the mass sits at the nearest point of the cone's cross-section, between the nodes. `ln P` was off by up to 0.08
  nats at 3-4 sd outside the cone and several nats beyond 6 sd (worst beyond c0 = 0), at any posterior width. The
  rule is now centred on that nearest point: better than 1e-3 nats down to `ln P` = -30 in random tests, 3e-5 at the
  posterior draws of 100 light curves (4e-3 before). The blend no longer depends on the width of the Gaussian: the ray
  integral is used everywhere except where all constraints hold by more than 4.5 to 6 sd, since the 12-point Genz
  rule is off by 2e-3 inside the cone at strong correlation and by far more outside it; its rare -inf (covariance
  rounded to singular, grazing small planets) can no longer reach the likelihood. Posteriors do not change measurably
  (<= 1e-4 sd over those 100 light curves); seeded runs differ slightly. About 8 us more per evaluation. Where the
  Gaussian is narrow and far outside the cone, the Genz rule was the more accurate one when it worked (1e-9 nats against
  1e-5, 2e-4 at `ln P` = -1000).
- `fit(max_evals=, max_time=)` and `profile_is(max_evals=, max_time=)`: an evaluation and wall-time
  budget for profile importance sampling (default: none). An expensive likelihood (a weak, many-epoch
  Kepler long-cadence signal: ~3600 binned points x 15 sub-exposures, 2e5 evaluations with the retry
  and 22 rescue iterations, 7 min; hours at 65 000 points) otherwise has no practical bound. When the
  budget is spent, the best attempt so far is returned with `diagnostics["budget_exceeded"]` and
  `is_ok` False and a warning; the rescue's iterations are capped to fit an evaluation budget; a budget
  too small for the first proposal raises RuntimeError. The conditional fits themselves were not the
  problem (Powell <= 700, BFGS <= 527 evaluations per call there): no caps added, results unchanged.
- `fit()` / `expand()`: where no geometry gives a valid limb-darkening draw (no visible transit), the
  coefficients come from the prior; this fallback read the c-space moments, which the default prior no
  longer computes (theta-space draws), and raised UnboundLocalError. f0 now comes from the theta
  moments (the data mean where they are not finite).
- `fit()` / `profile_is()` from a non-transiting start (b_conj >= 1 + rp, e.g. inc 89.5 deg with a/R* 123):
  a/R* is undefined there in the T14 parametrisation and the profile failed at once, returning the start
  repeated (k-hat inf). The start's b is now moved to half the transiting range; on flat data the fit
  returns a broad posterior, on data with a transit it finds the transit.
- Bounded retry (`_BoundedCoords.to_x`): `_conj_factor` and `_t14_range` converted the internal rows
  to canonical ones, including T14 -> a/R*, before the T14 column was set; a T14 of 0 raised
  ZeroDivisionError in the numba kernel (`fit()` crashed for a K2-131-like ultra-short-period planet at
  30-min cadence). They now read only the parameters they need (e, w; per, rp, b), and
  `a_logjac_from_T14` rejects a period of 0 and a T14 so small that sin^2 underflows (the retry's
  optimiser can step to per -> 0 and T14 -> 0 in log / logit coordinates). Results are unchanged
  wherever it did not crash.
- Quadratic marginal likelihood: well conditioned for any transit depth. The normal equations were
  formed in the Bernstein coefficients c, whose three columns are all constant out of transit
  (condition ~ 1 / rp^4): log L carried numerical noise (0.07 nats rms for Kepler-78 b, rp 0.014,
  80 000 points; 4e-4 nats for rp 0.012 on 6 900 points), systematic errors of up to 0.2 nats at
  S/N ~5 (rp 0.03) and of 5 to 13 000 nats at rp 0.01 to 0.001, and -inf below rp ~ 3e-4. They are now
  solved in theta = (f0, c1, c2) (f0 the out-of-transit flux; the two other columns are deficits,
  computed from the occultation terms without cancellation, exactly 0 out of transit), with scaled
  columns (`design_theta`, `marg_theta_parts`, `theta_moments`). The cone term is evaluated as
  int_cone exp(-q / 2) dtheta along rays c = f0 v (f0 analytic, v over the fixed cross-section of the
  cone by Gauss-Legendre in the principal axes of its Gaussian approximation) where the
  limb-darkening posterior is broad on the cone, and by the Genz orthant transform (moments from the
  theta solution) where it is narrow, blended smoothly in between. Against 50-digit references
  (mpmath; `tests/_mpref.py`): within 4e-5 nats at all tested geometries within ~500 nats of the
  maximum (posterior draws of 9 light curves of the extended tests, a S/N ~5 simulation from rp 0.03
  to 1e-6, the fast-vs-general test points), 1e-10 at high S/N; the noise on Kepler-78 b is 1e-8
  nats. As rp -> 0 log L tends to its finite no-transit limit. A geometry with (nearly) collinear
  deficit columns (extreme grazing) gets an extremely weak ridge (1e-9 of the scaled diagonal)
  instead of -inf. The limb-darkening-prior path (`ld_prior`) and the general laws are unchanged.
- No fallback to MCMC: `fit(method="auto")` (and "is") always returns importance draws. Where the
  profile is not usable (incomplete, k-hat >= 0.7 or ESS < 0.1 n), `profile_is` pools the weighted
  draws of all its attempts and runs adaptive importance sampling (AMIS, `_sampling.pmc_refine`)
  from them in unbounded coordinates (logit / log of the bounded parameters; logit of
  beta = b_conj / (1 + rp) whenever the bounds allow the whole support 0 <= beta < 1), with a
  defensive component over the whole prior box, for 3-30 iterations, until k-hat < 0.7, ESS >= 0.1 n
  and the 2.5-97.5 % quantiles are stable to 0.1 sd (`diagnostics["rescue"]`, `["rescue_iters"]`,
  `["is_ok"]`; a warning if they still fail). "auto" runs emcee only if b is not free. The five light
  curves of the extended tests that fell back to emcee (K2-100 b, K2-141 b, Kepler-447 b, WASP-47 e,
  Kepler-78 b; 0.39-3.3 M evaluations) pass in all 15 runs (3 seeds). With the well-conditioned
  likelihood (below) the profile itself succeeds for K2-100 b, K2-141 b and Kepler-447 b (53-85 k
  evaluations, within 0.03-0.17 sigma of the emcee references), and Kepler-10 b needs 9 k instead of
  53 k; the rescue runs for WASP-47 e (119-123 k, within 0.09-0.27 sigma) and two seeds of Kepler-22 b.
  The injection campaign (400 injections, 6 hosts, real and white noise) needs neither. A barely
  detected transit (S/N ~5, posterior over most of the prior) converges in 4 iterations, within
  0.09 sigma of emcee.
- `MarginalTransitModel.profile_is(khat_ok=0.5)`: a profile result is usable only for k-hat < 0.5
  (finite-variance weights; was 0.7), otherwise the further attempts and the rescue follow. On
  K2-131 b (a grazing branch with b up to 1.3 and rp up to 0.3, 6-8 % of the posterior) two of three
  seeds were accepted with k-hat 0.65 and 0.69 without a draw on the branch (3.8 sd off in the 97.5 %
  quantile of rp); with the rescue all three find it (P(b > 1) 6.1-6.9 %, emcee 5.9-8.0 %). 25 of 318
  runs of the extended tests had a final k-hat in [0.5, 0.7).
- `profile_is(ess_min=0.25)` (was 0.1): with an ESS of 900 of 4000 draws, Kepler-21 b's tails carried the
  noise of a few hundred draws (0.37 sd off in the 2.5 % quantile of T14; 0.06 sd with 40 000 draws).
  The threshold also sets the ESS that the rescue needs to stop.
- Profile IS draws 10 % extra points from a broad t around its weighted draws (2 x their sd beyond
  the inflation) and weights all draws against the deterministic mixture (`profile_is(defensive=0.1)`).
  A proposal that stops short of part of the posterior then gets large weights there, which k-hat
  sees: K2-141 b with the flat-limb-darkening proposal (grid ending at b = 0.8) had k-hat 0.20 and
  ESS 1070 -- both acceptable -- and rp off by 1.3 sigma. About 4 % more evaluations.
- `pmc_refine`: `extra` fixed proposal components, adaptive stopping (`min_iters`, `khat_ok`,
  `ess_min`, `dq_tol`; `converged` in the result); each proposal's density at each draw is computed
  once (the cost of the deterministic-mixture weights grew with the cube of the iterations). The
  defaults reproduce the previous results exactly.
- Profile IS on long, precise light curves: the first conditional Hessian (finite-difference steps of
  1e-3 x the default scales) is computed again with steps of 0.1 x the scales if it is not negative
  definite. The log likelihood carries numerical noise of ~5e-3 nats there (Kepler-10, short
  cadence, 79 000 points), which dominated the small steps: the profile failed at its first grid
  point and fell back to emcee (340 k evaluations); it now succeeds (k-hat 0.45, 32 k evaluations,
  within 0.12 sigma of emcee). Unchanged where the first Hessian was fine.
- Limb-darkening draws in the cone (`_gibbs_cone_one`, the fallback for rows whose truncated
  Gaussian is far in the tail): a coordinate whose conditional mean was more than ~8 sd below its
  wall was drawn below it (the inverse-CDF update saturated at u = 1 - 1e-16), i.e. outside the
  physical region (negative limb intensity, q1 > 1; 0.3 % of the draws for K2-100 b, long cadence).
  Such updates now use the upper-tail form, exact to ~37 sd. The geometry posterior is unaffected.
- emcee in `fit()` (method "emcee" and the "auto" fallback): a walker whose mean log posterior over
  the last 500 steps is more than max(25, 10 MAD) below the median walker's at two checks in a row
  (stuck at a local mode the stretch move does not leave) is moved next to another walker, and
  burn-in and the convergence test start again from there (`diagnostics["walkers_reset"]`;
  `_sampling.stuck_walkers`, `reset_walkers`). Unchanged, bit for bit, when no walker is stuck. The
  sampler runs in chunks of 500 steps (the chain no longer allocates `max_steps` up front). Seen in
  reference emcee runs started from catalogue values: 1-2 of 32 walkers at b = 0.14 for HD 189733 b,
  1400 below the posterior, widening sd(b) 20-fold.
- Profile importance sampling is usable only with an effective sample size >= 0.1 n
  (`profile_is(ess_min=0.1)`, also in `fit()`), besides a complete profile and k-hat < 0.7. A proposal
  that misses part of the posterior altogether produces no large weights there, so k-hat cannot see
  it: TrES-2 b (Kepler Q3 long cadence) with the flat-limb-darkening proposal had k-hat 0.4-0.6 but an
  ESS of 150-250 of 4000 and missed the upper tail of rp (up to 1.4 sigma); it now continues to the
  bounded retry with AMIS (ESS 15 000, within 0.06 sigma of emcee). No accepted run of the injection
  campaigns (829) had an ESS below 1000.
- Prior support for eccentric orbits: the region with a transit is b_conj = b (1 - e^2) / (1 + e sin w)
  < 1 + rp (mid-eclipse: 1 - e sin w), not b < 1 + rp (pizzetti's b is a cos i). With e sin w > 0 the
  old support cut off allowed grazing geometries (Kepler-447 b, e = 0.12, w = 98 deg: the posterior was
  pressed against b = 1 + rp), with e sin w < 0 it admitted geometries without a transit. Circular
  orbits are unchanged.
- Grazing transits: if the first profile reaches a grazing geometry (b_conj + rp > 1) and the bounds
  allow b_conj up to 1 + rp, the bounded retry profiles beta = b_conj / (1 + rp) between the fixed
  walls 0 and 1 instead of b. The edge b_conj = 1 + rp moves with rp, so a grazing posterior along
  b - rp = const (at low S/N up to the rp bound) could not be closed in b and fell back to emcee
  (0.3-1.2 M evaluations); it now takes 70-95 k. A warning is issued if rp has no upper bound (the
  posterior can then be improper).
- Profile IS at the limb-darkening cliff: if a complete profile has poor weights, the proposal is
  built on the posterior with a flat prior over the coefficients (smooth) and the draws are weighted
  against the physical posterior = flat x P(cone), P <= 1, so the weights stay bounded
  (`log_prob_internal(..., parts=True)`, `_sampling.profile_is(logp_target=...)`). A start on a
  strict bound is moved inside the prior first.
- `MarginalTransitModel(threads=1)`: `threads="all"` or n > 1 parallelises the likelihood for fitting
  (rows of a batch over threads, or for long light curves the data points of a single row);
  agreement with `threads=1` to rounding (~1e-11 nats), identical for any n > 1.
- Fixes found by injection campaigns: a b-grid step outside the support (target -inf for every value
  of the other parameters) was kept as a grid point with a NaN covariance; it closed the profile end,
  but every draw in the last interval was then dropped silently, so mass near the edge could be
  missed without k-hat noticing. Such steps are now rejected (the step is halved). A profile with a
  single grid point reports failure instead of crashing. A coefficient covariance without a Cholesky
  factor (a geometry that hardly constrains the limb darkening) gives that draw weight 0 instead of
  raising (also in the Gibbs fallback of the draw); `fit()` reports the dropped geometry weight as
  `ld_dropped_weight`. If no geometry constrains the limb darkening at all (a transit too shallow or
  too grazing to be seen), the coefficients are drawn from the prior (uniform q1, q2) instead of
  raising (`ld_from_prior`). A nearly singular BFGS seed in the conditional fits of the profile is
  dropped instead of raising.
- Profile importance sampling for posteriors that press against the prior bound of a parameter
  other than b (e.g. a/R* for an unresolved ingress, or a grazing transit): new method
  `MarginalTransitModel.profile_is()`, used by `fit()`. If the profile in the original
  coordinates is not usable (k̂ ≥ 0.7 or incomplete) and free parameters have finite bounds, the
  sampling is repeated in unbounded coordinates (logit / log of the bounded parameters, with the
  Jacobian; for a/R* under the T14 reparametrisation, logit of T14 between its values at the
  bounds of a for the same (P, rp, b)). The Laplace steps stay unreliable there (the posterior is
  flat and skewed near the bound), so the draws are refined by adaptive multiple importance
  sampling (`_sampling.pmc_refine`: six iterations of Gaussian-mixture proposals fitted to all
  weighted draws, deterministic-mixture weights, a broad defensive component). In injection tests
  in real Kepler/TESS noise the emcee fallback (0.25–2 M evaluations) is replaced by ~50–100 k
  evaluations (32 of 35 such injections), and the quantiles agree with emcee within its own noise; k̂ alone did not detect the
  tails the plain retry missed. If the result is still not usable (strongly non-Gaussian
  conditionals, e.g. near-grazing at very high S/N), the same AMIS refinement runs as a last resort
  in the original coordinates before the emcee fallback. Runs whose first profile is usable are
  unchanged.
- `fit(method="emcee", seed=...)` is reproducible: emcee's RandomState is seeded from `seed`.
- `expand()` (quadratic fast path): rows without a valid coefficient draw get weight 0 instead of
  turning all weights into NaN.
- `psis()` and `pmc_refine()` fail cleanly (k̂ = inf) instead of raising when no weight is finite.
- Robustness fixes found by extended tests on real light curves (`limb-darken/extended_tests`):
  - The orthant probability of the marginal likelihood is now finite everywhere. Where the
    quadrature underflowed (least-squares limb darkening more than ~37σ outside the physical
    cone) it returned -inf, so optimisers and the emcee fallback could not move from a poor
    start; near the edge of the limb-darkening prior even catalogue-like starting values were
    affected. An asymptotic form (sequential conditioning on the most violated constraint) takes
    over there, within ~1 nat of a tilted importance-sampling reference. Finite values are
    unchanged.
  - `fit()` raised `LinAlgError` for grazing transits: importance draws beyond b = 1 + rp (zero
    weight, degenerate design) reached the coefficient draws. They are dropped first.
  - The emcee fallback of `fit()` crashed when the autocorrelation time was NaN (walkers starting
    outside the prior); such walkers are now drawn again closer to the start, and a non-finite
    tau counts as not converged.
  - User bounds on b are walls of the profile grid (they were ignored, and a grid point placed
    on a strict bound failed, so posteriors cut off by the bounds fell back to emcee).
- `TransitModel(..., offsets="batman" | "midpoint")`: placement of the sub-exposures, as already
  in `MarginalTransitModel`. The default stays batman's (equally spaced points including the
  exposure edges, i.e. a midpoint rule over an exposure stretched by k/(k−1), error O(1/k));
  "midpoint" uses the midpoints of k equal sub-intervals (error O(1/k²); for Kepler long cadence
  with k = 15 about 3 ppm instead of 125 ppm).
- Profile importance sampling: the b-grid refinement is now a last resort (`lazy_refine=True`).
  It runs only when the final k̂ (after the defensive second stage) is still ≥ `khat_refine`
  (0.7), and the sampling is then repeated on the refined grid. Refining always cost 10 extra
  conditional fits (+24–49 % likelihood evaluations) without improving the accuracy, and most
  poor first-stage proposals come from non-Gaussian conditionals (e.g. Kepler-8 LC, where the
  Laplace estimate of the b-marginal is off by ±0.5 nats), which the second stage handles and
  refinement does not. On 12 test light curves × 3 seeds: 16 % fewer evaluations in total
  than refining always, the same accuracy. `lazy_refine=False` restores the previous behaviour;
  the result reports `n_refine`.

## 0.3.0 (Pizzetti.jl 0.2.0)

- Fix: `import pizzetti` no longer needs scipy. The quasi-random points of the nonlinear-law
  physical prior (`_physnl`) are built on first use; CI now also checks the import without the
  optional extras, and installs `[fit]` and setuptools (batman needs distutils on Python ≥ 3.12)
  for the tests.
- Documentation and examples reorganised:
  - README shortened to an overview; details moved to the documentation.
  - `examples/`: a tutorial curriculum of ten notebooks (01 first light curve … 10 linear-basis
    light curves) with level, prerequisites and run time, replacing the four former notebooks;
    all run in CI in a fast smoke mode (`PIZZETTI_SMOKE=1`, workflow `examples.yml`).
  - `examples/paper/`: one script per table and figure of the paper.
  - `julia/examples/`: five Literate-style scripts J1–J5 (quickstart, gradients, precision and
    rigour, photodynamics, interoperability with Limbdark.jl and the Python package).
  - Documentation numbers and limb methods brought up to date.
- Faster limb for the square-root, nonlinear and power-2 laws (radius ratios < 0.3): 2–8x per
  light curve against 0.2.1, same accuracy (~1e-15 against the quadrature).
  - New module `_trap`: the limb integral written as a complete integral (no split, no
    incomplete integrals), evaluated by a fixed-node trapezoid rule in s = asinh(tan th). The
    integrand is analytic in |Im s| < pi/2 for every modulus (the Landen/AGM structure), so the
    rule converges uniformly up to the contacts; the nodes do not depend on z, so the
    exponents only enter precomputed weights. The pole b = x is removed by its residue term.
    Inside the disk (z_c < z < 1 - p) the same integral is taken along a complex contour.
  - New module `_interp`: per-curve Chebyshev interpolants of these rules. The singular part at
    second contact follows from the connection formula J = A + lam^nu B with
    B = -k^-nu J(k; ...) / sin(pi nu): the same integral at the complementary modulus. Set-up
    ~40 us per light curve (the hypergeometric contact tables took 0.1–1.7 ms), ~10 ns per
    sample and exponent.
  - `limb="fast"` (default) picks the direct rules or the interpolants by the number of limb
    samples; `"trapezoid"`, `"interpolate"` force one, `"contact"` the former expansions.
  - Per-call overhead for the non-quadratic laws 40 -> 15 us (cached per-exponent plans), and
    a fused, block-skipping interior pass as for the quadratic law.
  - The inside interpolant now reaches lam = -1 (was -1/2; still 0.3 of the way to the
    singularity at the occultor rim through the stellar centre), so no band samples fall back
    to the slower direct contour rule for p <= 0.2.
- Pizzetti.jl: the same limb methods (`src/trap.jl`, `limb = :trapezoid`, `:interpolate`; `:auto`
  picks them for p < 0.3). Light curves of the square-root, nonlinear and power-2 laws 1.1–10x
  faster, the Kepler power-2 likelihood of the inference example 0.36 -> 0.22 ms; gradients
  unchanged (dual numbers keep the exact-derivative path). 5993 tests pass.

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
