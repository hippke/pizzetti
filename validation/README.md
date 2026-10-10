# Validation notebooks

These notebooks check every bound and every non-trivial equation of the pizzetti paper,
and pizzetti itself, against references that are **independent of the paper's derivations
and of the pizzetti code**:

- the **definition** of the flux deficit, the area integral of the intensity over the
  occulted part of the star, evaluated by numerical quadrature (`refcheck.py`), and
- **Eric Agol's code** for the quadratic law (`exoplanet-core`; Agol, Luger &
  Foreman-Mackey 2020).

The paper's formulas are transcribed into `refcheck.py` from the text of the paper, not
taken from pizzetti. Each notebook stops with an error if a check fails.

| Notebook | What it checks | Run time |
|---|---|---|
| `01_reference` | The reference itself: double vs. arbitrary precision, exact cases, Agol's code | ~35 s |
| `02_series_theorem2` | Polynomial form, Eq. (S3), **Theorem 2** (bound, sign, monotonicity), **Corollary 1** (cut) with Agol's code as judge | ~60 s |
| `03_meanvalue_appendixB` | Eqs. (lap), (lowmu), (Tn) symbolically, Eq. (contact), **Propositions 1 and 2**, Eq. (dDdz) | ~35 s |
| `04_hypergeometric_section5` | Eqs. (P1d), (ingress), (interiorhyp), **Lemma 2** | ~55 s |
| `05_contact_theorem3` | **Theorem 3** piece by piece, and complete light curves in the fast and proven modes | ~10 s |

Run times are for two cores. Every notebook has a switch `FULL = False` at the top. Set it
to `True` for the larger grids (more exponents, radius ratios, separations and orders);
these take minutes to hours. The checks are the same; only the number of cases changes.

The full grids were run once while preparing the paper:
- Theorem 2: 2132 points over 150 configurations (6 exponents, 5 radius ratios, M = 1–21).
  No violations; error/bound up to 0.99998.
- Theorem 3: 183 certified pieces over 30 configurations (5 laws, p = 0.0092–0.2,
  tolerances 1e-9 and 1e-12). Error/bound up to 0.98.
- Complete light curves: every certified sample within the tolerance.

Requirements: `numpy`, `mpmath`, `sympy`, `jupyter`, `pizzetti`, and optionally
`exoplanet-core` (the cells that use it are skipped if it is missing).

```bash
pip install pizzetti mpmath sympy jupyter exoplanet-core
jupyter nbconvert --to notebook --execute --inplace validation/*.ipynb
```

Note: for `b` exactly one floating-point step below `r`, exoplanet-core 0.4.0 returns a
flux that is off by 1. Notebooks 1 and 2 skip `|z - p| < 1e-12` for this reason.
