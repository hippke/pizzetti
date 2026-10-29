# Tutorials

Ten notebooks, from a first light curve to certified and linear-basis models. Each one starts with
its level, prerequisites and run time, and ends with what you learned and where to go next. The
first notebook you run also compiles pizzetti's numba kernels (about a minute, cached afterwards).

```bash
pip install "pizzetti[fit]" batman-package corner matplotlib mpmath jupyter
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

## Under the hood

| | Notebook | What you learn | Run time |
|---|---|---|---|
| 07 | [How pizzetti computes a light curve](07_how_pizzetti_computes_a_light_curve.ipynb) | `occult`, the cut z_c, the regimes of a transit, every `limb` option with its accuracy and cost | ~30 s |
| 08 | [Error control and certification](08_error_control_and_certification.ipynb) | `series_tol`, the proven bound, `limb="proven"`, `return_certified`, an independent 40-digit check | ~1 min |
| 09 | [Performance](09_performance.ipynb) | cost per sample, duty cycle, reusing models, fixed vs free exponents, grazing transits, batman's `max_err` | ~1 min |
| 10 | [Linear basis light curves](10_linear_basis_light_curves.ipynb) | `occult_terms`, `quadratic_from_terms`, linear least squares for limb darkening, many wavelength channels | ~20 s |

Paths through the tutorials: **batman users** 02 → 04 → 06; **fitting** 01 → 03 → 05 → 06;
**methods** 07 → 08 → 09.

## Feature coverage

| Feature | Notebooks |
|---|---|
| `TransitParams`, `TransitModel.light_curve` | 01, 02, 03, 04 |
| all limb-darkening laws (uniform … power-2, logarithmic) | 01, 05, 06 |
| supersampling / exposure time | 04 |
| eccentric orbits, `get_t_periastron`, `get_t_secondary`, `get_t_conjunction`, `get_true_anomaly` | 04 |
| secondary eclipses (`transittype="secondary"`) | 04, 06 |
| grazing and inverse transits | 04, 09 |
| `calc_err`, `max_err` | 02, 04, 09 |
| emcee with `TransitModel` | 02, 03, 06 |
| `MarginalTransitModel`, `fit()`, `FitResult`, `EnsembleSampler`, `ld_prior` | 05, 06 |
| `occult`, `series_cut`, `limb=` options | 07, 08, 09 |
| `series_tol`, `limb="proven"`, `return_certified` | 08 |
| `occult_terms`, `quadratic_from_terms` | 10 |

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
