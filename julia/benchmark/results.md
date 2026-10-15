# Pizzetti.jl benchmarks

Julia 1.13.1, Intel(R) Xeon(R) Processor @ 2.80GHz, single thread. Minimum over repeated runs.

## Cost per sample

| law | p | interior series | limb: contact | limb: quadrature | quadratic: exact |
|---|---|---|---|---|---|
| Quadratic | 0.0092 | 2.26 ns | — | — | 75 ns |
| Quadratic | 0.1 | 2.26 ns | — | — | 71 ns |
| Power2 | 0.0092 | 5.54 ns | 92 ns | 1712 ns | — |
| Power2 | 0.1 | 5.88 ns | 89 ns | 1616 ns | — |
| Nonlinear | 0.0092 | 3.33 ns | 154 ns | 2931 ns | — |
| Nonlinear | 0.1 | 3.36 ns | 154 ns | 2687 ns | — |

## Set-up per light curve

| law | p | series cut + coefficients | contact expansions | proven pieces |
|---|---|---|---|---|
| Power2 | 0.0092 | 5.0 µs | 0.12 ms | 9.6 ms |
| Power2 | 0.1 | 5.0 µs | 0.27 ms | 18.1 ms |
| Nonlinear | 0.0092 | 10.8 µs | 0.35 ms | 25.1 ms |
| Nonlinear | 0.1 | 10.8 µs | 0.82 ms | 47.0 ms |

## Light curves (orbit + flux), as in a fit

Kepler-like super-Earth: 30 transits at 1-min cadence (10 380 samples), p = 0.02, a/R = 20, b = 0.4;
TESS-like hot Jupiter: 7 transits at 2-min cadence (1316 samples), p = 0.1, a/R = 9, b = 0.35.

| case | law | light curve | gradient (ForwardDiff, 7 parameters) |
|---|---|---|---|
| Kepler super-Earth | Quadratic | 0.089 ms | 3.20 ms |
| Kepler super-Earth | Power2 | 0.259 ms | 4.12 ms |
| TESS hot Jupiter | Quadratic | 0.027 ms | 0.32 ms |
| TESS hot Jupiter | Power2 | 0.324 ms | 2.05 ms |

## Against Limbdark.jl (quadratic law, p = 0.1, 10 000 samples)

| | Pizzetti.jl | Limbdark.jl |
|---|---|---|
| light curve | 0.209 ms | 0.820 ms |
| light curve with gradient (r, b, u) | 1.572 ms | 1.462 ms |
| single call `transit_poly` | 46 ns | 2505 ns |

Limbdark.jl re-uses its work structure across samples (`transit_poly!(t)`); `transit_poly(r, b, u)`
allocates it per call. Pizzetti.jl's gradient for the quadratic family uses closed forms
(`flux_gradient_quadratic`).

Largest difference of the light curves: 7.2e-10 (the interior series, proven below 1e-9).
