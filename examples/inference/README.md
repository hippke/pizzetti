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
python benchmark_calls.py        # cost per likelihood evaluation, a new posterior sample per call (interleaved)
python benchmark_wall.py         # wall-clock time of 2000-step runs
```

## Results (Intel Xeon cloud instance, single thread; Python 3.13, numba 0.68, batman 2.5.3, Julia 1.13.1)

| | hot Jupiter (quadratic) | super-Earth (power-2) |
|---|---|---|
| largest difference of the medians | 0.08 sigma (1.8 MCSE) | 0.04 sigma (0.6 MCSE) |
| largest difference of the 68 % widths | 6 % | 1 % |
| chains step-by-step identical for | 846 steps | 6 steps |
| largest model error over the posterior, batman | 9e-9 | 1.9e-6 |
| largest model error over the posterior, pizzetti | 7e-10 | 5e-10 |
| likelihood evaluation, batman | 0.079 ms | 0.67 ms |
| likelihood evaluation, pizzetti | 0.018 ms (4.4x) | 0.20 ms (3.4x) |
| likelihood evaluation, batman with `max_err=0.01` | — | 3.8 ms (error still 4e-8) |
| 2000-step emcee run, batman / pizzetti | 5.6 s / 1.6 s | 41 s / 12 s |
| likelihood / gradient (7 parameters), Pizzetti.jl | 0.026 / 0.24 ms | 0.22 / 3.2 ms |

In these fits 50-66 % of the samples are in transit. `benchmark_calls.py` uses `results/posterior_samples.json`
(jittered by 1e-4) when no stored chains are present.

MCSE: Monte Carlo standard error of the difference of the medians, from the integrated autocorrelation
times. The posteriors agree within their Monte Carlo noise. The chains, which draw identical proposals, stay
identical until a model difference changes an acceptance decision: batman's power-2 integrator (1 ppm
tolerance) differs from the reference by 1.9e-6, 1.3 % of the noise, and the chains part after 6 steps.

In the super-Earth case the power-2 exponent alpha changes at every step, so every call rebuilds the
weights of the trapezoid rule and the limb interpolants (paper Section 5.3, ~40 us); with alpha held
fixed the likelihood takes 0.14 ms (4.7x faster than batman). Before the trapezoid rule, the contact
expansions' hypergeometric tables were rebuilt instead, and the likelihood took 0.32 ms. The Julia numbers are from `julia/benchmark/inference_cases.jl`
(same light curves and posterior samples, `results/posterior_samples.json`).

![corner plots](results/corner_kepler_paper.png)
