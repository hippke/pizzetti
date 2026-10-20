# Reproducing the paper

One script per table and figure of the paper (Hippke et al., in prep.). Each script writes its
LaTeX table to `tables/`, its figure to `figures/` and a plain-text log to `out/` (all created
on first run). Run them from any directory:

```bash
pip install "pizzetti[fit]" batman-package mpmath matplotlib corner
python examples/paper/validate.py
```

| Paper | Script | Output | Runtime |
|---|---|---|---|
| Fig. `power2` (power-2 light curves and errors) | [`make_figures.py`](make_figures.py) | `figures/fig_power2_lightcurves.pdf` (and `fig_polynomial_limits.pdf`) | ~2 min |
| Table `bound` (error bound against the true truncation error) | [`validate.py`](validate.py) | `tables/error_bound_binomial.tex`, `tables/error_bound.tex` | ~5 min |
| Table `powers` (exponents and series orders) | [`validate.py`](validate.py) | `tables/powers.tex` | (same run) |
| Table `zcut` (series cut-off z_c) | [`validate.py`](validate.py) | `tables/zcut.tex` | (same run) |
| Table `coverage` (fraction of a transit inside z_c) | [`validate.py`](validate.py) | `tables/coverage.tex` | (same run) |
| Table `timing` (interior cost: series, closed form, batman) | [`interior_timing.py`](interior_timing.py) | printed table | ~1 min |
| Table `contact` (contact expansions, proven pieces, quadrature) | [`contact_table.py`](contact_table.py) | `tables/contact.tex` | ~10 min |
| Table `trap` (fixed-node limb rule and interpolants) | [`limb_table.py`](limb_table.py) | `tables/trap.tex` | ~5 min |
| Section 5.1, round-off of the series in double precision | [`roundoff.py`](roundoff.py) | `out/roundoff.txt` | ~1 min |
| Table `inference`, Fig. `inference` (emcee fits, corner plots) | [`../inference`](../inference) | `../inference/results/` | ~25 min |
| Section 6.4, grazing transits | [`grazing.py`](grazing.py) (needs `../inference` results) | `grazing.json` | ~3 min |
| Theorems 1–3, Appendix B (independent checks of the proofs) | [`../../validation`](../../validation) notebooks | | ~10 min |

`series.py` is a small mpmath implementation of the interior series used by `validate.py`
and `roundoff.py`, independent of the package code.

Timings depend on the machine. The numbers in the paper were measured on a single thread
(Intel Xeon, numba 0.68, batman 2.5.3); the accuracy columns are machine independent and
should be reproduced to the printed digits.
