# Examples

| Notebook | What it shows | Run time |
|---|---|---|
| [`01_batman_dropin_emcee.ipynb`](01_batman_dropin_emcee.ipynb) | `import pizzetti as batman`: the same emcee fit of a TESS-like hot Jupiter (quadratic law) with both codes; posteriors, model errors, timing | ~30 s |
| [`02_power2_superearth.ipynb`](02_power2_superearth.ipynb) | Kepler-like super-Earth with the power-2 law: model error along a transit, cost per likelihood call (also batman at a tighter tolerance), maximum likelihood, a short MCMC run, and a light curve with a proven error bound for every sample | ~2 min |
| [`inference/`](inference/) | The full-length runs of the paper (Section 6.4): two data sets x two codes, 20 000 emcee steps each, corner plots, timing | ~25 min |

Requirements: `pizzetti`, `batman-package`, `emcee`, `corner`, `matplotlib`, `scipy`.

```bash
pip install pizzetti batman-package emcee corner matplotlib scipy
jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
```

For Julia examples (gradients with ForwardDiff, photodynamics, interval enclosures) see
[`../julia/examples`](../julia/examples).
