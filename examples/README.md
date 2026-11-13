# Tutorials

Twenty-two notebooks, from a first light curve to certified and linear-basis models, the importance
sampler behind `fit()`, the Grand Tour from a raw light curve to a posterior (with wotan, and with a simultaneous Gaussian process), and fitting with a Gaussian process. Each one starts
with its level, prerequisites and run time, and ends with what you learned and where to go next. The
first notebook you run also compiles pizzetti's numba kernels (about a minute, cached afterwards).

```bash
pip install "pizzetti[fit]" batman-package corner matplotlib mpmath jupyter lightkurve wotan transitleastsquares
# Python >= 3.12: batman also needs `pip install setuptools`
jupyter lab examples/
```

## Getting started

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 01 | [Your first light curve](01_first_light_curve.ipynb) | `TransitParams`, `TransitModel`, units, what each parameter does, the six limb-darkening laws | ~20 s |
| 02 | [Coming from batman](02_coming_from_batman.ipynb) | `import pizzetti as batman`, supported options, accuracy and speed against batman, identical emcee chains | ~1 min |
| 03 | [Your first fit](03_first_fit.ipynb) | simulated TESS data, a likelihood, maximum likelihood with scipy, MCMC with emcee, reading the posterior | ~1 min |

## Real data, real choices

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 04 | [Observing realities](04_observing_realities.ipynb) | long exposures, eccentric orbits and timing helpers, secondary eclipses, grazing and inverse transits, `calc_err` | ~20 s |
| 05 | [Choosing a limb-darkening law](05_choosing_a_limb_darkening_law.ipynb) | the same light curve fitted with every law, comparing geometry, intensity profiles and residuals | ~1–2 min |
| 06 | [MarginalTransitModel](06_marginal_transit_model.ipynb) | limb darkening and baseline integrated out; `fit()`, `FitResult`, `EnsembleSampler`, priors, free α, secondary eclipses; against `TransitModel` + emcee and batman | ~2 min |
| 22 | [Long exposures in a fit](22_long_exposures_in_a_fit.ipynb) | a real Kepler long-cadence light curve (TrES-2 b) fitted with and without supersampling: the smeared dip, a precise posterior in the wrong place, `supersample_factor` and `exp_time` in `MarginalTransitModel`, midpoint sub-exposures, how many with a χ² check | ~1 min |

## Under the hood

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 07 | [How pizzetti computes a light curve](07_how_pizzetti_computes_a_light_curve.ipynb) | `occult`, the cut z_c, the regimes of a transit, every `limb` option with its accuracy and cost | ~30 s |
| 08 | [Error control and certification](08_error_control_and_certification.ipynb) | `series_tol`, the proven bound, `limb="proven"`, `return_certified`, an independent 40-digit check | ~1 min |
| 09 | [Performance](09_performance.ipynb) | cost per sample, duty cycle, reusing models, fixed vs free exponents, grazing transits, batman's `max_err` | ~1 min |
| 10 | [Linear basis light curves](10_linear_basis_light_curves.ipynb) | `occult_terms`, `quadratic_from_terms`, linear least squares for limb darkening, many wavelength channels | ~20 s |

## Importance sampling

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 11 | [Your first fit with Importance Sampling (IS)](11_your_first_fit_with_is.ipynb) | a real TESS light curve (WASP-18 b): download with lightkurve, detrending with wotan, `fit()` and a corner plot, step by step | ~1 min |
| 12 | [Importance sampling from scratch](12_importance_sampling_from_scratch.ipynb) | weights, effective sample size, Pareto k̂ (`psis`), the blind spot of weight diagnostics, defensive mixtures, adaptive IS (`pmc_refine`) on toy densities | ~10 s |
| 13 | [Why b is hard](13_why_b_is_hard.ipynb) | the wall at b = 0, the skewed marginal and the curved b–a/R★ ridge; a Laplace proposal (`laplace_is`) that passes its diagnostics and misses the tail; what emcee pays | ~30 s |
| 14 | [Profile importance sampling step by step](14_profile_importance_sampling.ipynb) | `profile_is` taken apart: the grid of conditional optima, ℓ(b), the proposal, the weights, stage 2, defensive draws, refinement; accuracy and cost against emcee | ~30 s |
| 15 | [When the profile is not enough](15_when_the_profile_is_not_enough.ipynb) | a grazing branch, the acceptance rule (k̂ < 0.5, ESS ≥ 25%), a transit at S/N ≈ 5; strata at the grazing boundary, bounded retry, flat proposal and adaptive rescue; reading `diagnostics` | ~1 min |
| 16 | [Trusting the result: IS against MCMC](16_is_against_mcmc.ipynb) | the Monte Carlo noise of IS and of emcee, measured alike; slow tails and short chains; more draws as a cheap check; cost | ~2 min |

## The Grand Tour

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 17 | [The Grand Tour with wotan](17_the_grand_tour.ipynb) | the whole chain for LHS 3844 b: download (lightkurve), detrending (wotan), the search with [TLS](https://github.com/hippke/tls), the fit with importance sampling, planet radius and stellar density | ~2 min |
| 21 | [The Grand Tour with simultaneous GP](21_the_grand_tour_with_simultaneous_gp.ipynb) | the whole chain without detrending, for K2-33 b on a young, strongly variable star: download (lightkurve), bad cadences, the search with [TLS](https://github.com/hippke/tls) 2.0 + GP, the fit with `gp=pizzetti.GP()` in time order, planet radius and stellar density | ~1 min |

## Correlated noise

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 18 | [Detrending and fitting together](18_fitting_with_a_gaussian_process.ipynb) | a young, strongly variable star (K2-33 b, K2) without detrending: `gp=pizzetti.GP()`, `gp_predict`, reading the GP parameters, against no detrending, a biweight filter and a robust spline, when to be careful | ~2 min |

## Model comparison

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 19 | [The Bayesian evidence](19_bayesian_evidence.ipynb) | `lnz` from `profile_is` and `fit`, piece by piece; against dynesty, on the marginal and the conventional fit, including a grazing transit; accuracy against cost; a Bayes factor for "is there a transit?"; how the prior enters; when not to trust the number | ~30 min (dynesty) |

## Eccentricity

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 20 | [Eccentricity from a stellar density](20_eccentricity_from_a_stellar_density.ipynb) | `FitResult.reweight_density`: the photoeccentric effect; one circular fit for any density and prior on (e, ω); why the Jacobian and the centre of the transit matter; checks against dynesty and a grid, where the reference itself was wrong; calibration on 1100 light curves; the limit at high S/N | ~1 min |

Paths through the tutorials: **batman users** 02 → 04 → 06; **fitting** 01 → 03 → 05 → 06;
**methods** 07 → 08 → 09; **sampling** 11 → 12 → 13 → 14 → 15 → 16; **real data, start to end** 11 → 17 → 18; **comparing models** 11 → 14 → 19.

## Feature coverage

| Feature | Notebooks |
|---|---|
| `TransitParams`, `TransitModel.light_curve` | 01, 02, 03, 04 |
| all limb-darkening laws (uniform … power-2, logarithmic) | 01, 05, 06 |
| supersampling / exposure time | 04, 22 |
| eccentric orbits, `get_t_periastron`, `get_t_secondary`, `get_t_conjunction`, `get_true_anomaly` | 04 |
| secondary eclipses (`transittype="secondary"`) | 04, 06 |
| grazing and inverse transits | 04, 09 |
| `calc_err`, `max_err` | 02, 04, 09 |
| emcee with `TransitModel` | 02, 03, 06 |
| `MarginalTransitModel`, `fit()`, `FitResult`, `EnsembleSampler`, `ld_prior` | 05, 06 |
| `occult`, `series_cut`, `limb=` options | 07, 08, 09 |
| `series_tol`, `limb="proven"`, `return_certified` | 08 |
| `occult_terms`, `quadratic_from_terms` | 10 |
| importance weights, ESS, `psis`, `pmc_refine`, `laplace_is` | 12, 13 |
| `profile_is` (sampler and `MarginalTransitModel` method), `FitResult.diagnostics` | 11, 14, 15 |
| bounded retry, flat-limb-darkening proposal, adaptive rescue | 15 |
| comparing samplers: quantile differences, Monte Carlo noise | 14, 16 |
| real data: lightkurve download, wotan detrending, TLS search, `FitResult.corner` | 11, 17 |
| correlated noise: `GP`, `gp=`, `gp_predict` | 18 |

## More

- [`paper/`](paper): one script per table and figure of the paper.
- [`inference/`](inference): the full-length emcee runs of the paper (two data sets × two codes,
  corner plots, timing; ~25 min).
- [`../julia/examples`](../julia/examples): seven Julia notebooks (installation, quickstart,
  gradients with ForwardDiff, precision and rigour, photodynamics with NbodyGradient,
  interoperability, and the limb integral in Bartky's variable step by step).
- [`../validation`](../validation): independent numerical checks of the theorems.

All notebooks are executed in CI in a fast smoke mode (`PIZZETTI_SMOKE=1` shortens MCMC chains and
benchmarks):

```bash
PIZZETTI_SMOKE=1 jupyter nbconvert --to notebook --execute --output-dir /tmp examples/*.ipynb
```
