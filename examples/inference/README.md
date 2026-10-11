# Parameter inference: batman vs pizzetti (paper, Section 6.4)

Two synthetic data sets are fitted with [emcee](https://emcee.readthedocs.io), once with batman and once
with pizzetti, each with its default settings. Both runs use the same likelihood, the same starting
positions and the same random-number stream, so they differ only in the light-curve call.

| case | cadence | transits | samples | law | noise |
|---|---|---|---|---|---|
| `tess`: hot Jupiter, p = 0.1, a/R = 9, b = 0.35, P = 3.5 d | 2 min | 7 | 1316 | quadratic (0.40, 0.25) | 500 ppm |
| `kepler`: super-Earth, p = 0.02, a/R = 20, b = 0.2, P = 10 d | 1 min | 30 | 10 380 | power-2 (0.6, 0.6) | 150 ppm |

The data are computed with pizzetti's Green's-theorem quadrature (series off, ~1e-16) plus white noise.
Free parameters: t0, P, p, a/R, b and two limb-darkening parameters (Kipping's q1, q2 for the quadratic
law; c and alpha for the power-2 law), uniform priors; 32 walkers, 20 000 steps, the first 4000 discarded.

```bash
./runall.sh                      # the four runs (~25 min on one core)
python analyze.py tess 4000      # corner plots, quantiles, Monte Carlo errors -> results/
python analyze.py kepler 4000
python benchmark_calls.py        # cost per likelihood evaluation at 200 posterior samples (interleaved)
python benchmark_wall.py         # wall-clock time of 2000-step runs
```

## Results (Intel Xeon 2.8 GHz, single thread)

| | hot Jupiter (quadratic) | super-Earth (power-2) |
|---|---|---|
| largest difference of the medians | 0.08 sigma (1.8 MCSE) | 0.04 sigma (0.6 MCSE) |
| largest difference of the 68 % widths | 6 % | 1 % |
| chains step-by-step identical for | 846 steps | 6 steps |
| largest model error over the posterior, batman | 9e-9 | 1.9e-6 |
| largest model error over the posterior, pizzetti | 7e-10 | 5e-10 |
| likelihood evaluation, batman | 0.133 ms | 1.11 ms |
| likelihood evaluation, pizzetti | 0.056 ms (2.4x) | 0.77 ms (1.4x) |
| likelihood evaluation, batman with `max_err=0.01` | — | 6.0 ms (error still 4e-8) |
| 2000-step emcee run, batman / pizzetti | 10.3 s / 4.5 s | 69 s / 46 s |

MCSE: Monte Carlo standard error of the difference of the medians, from the integrated autocorrelation
times. The posteriors agree within their Monte Carlo noise. The chains, which draw identical proposals, stay
identical until a model difference changes an acceptance decision: batman's power-2 integrator (1 ppm
tolerance) differs from the reference by 1.9e-6, 1.3 % of the noise, and the chains part after 6 steps.

In the super-Earth case two thirds of pizzetti's time goes to the 330 limb samples per light curve, which
use quadrature because there are fewer than 300 per exponent (below that the set-up of the contact
expansions does not pay off).

![corner plots](results/corner_kepler_paper.png)
