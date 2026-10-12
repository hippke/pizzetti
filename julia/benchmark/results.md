# Pizzetti.jl benchmarks

Julia 1.13.1, Intel(R) Xeon(R) Processor @ 2.10GHz, single thread. Minimum over repeated runs.

## Cost per sample

| law | p | interior series | limb: contact | limb: quadrature | quadratic: exact |
|---|---|---|---|---|---|
| Quadratic | 0.0092 | 1.57 ns | — | — | 44 ns |
| Quadratic | 0.1 | 1.57 ns | — | — | 42 ns |
| Power2 | 0.0092 | 3.32 ns | 55 ns | 1223 ns | — |
| Power2 | 0.1 | 3.31 ns | 55 ns | 1128 ns | — |
| Nonlinear | 0.0092 | 2.38 ns | 92 ns | 2035 ns | — |
| Nonlinear | 0.1 | 2.46 ns | 93 ns | 1835 ns | — |

## Set-up per light curve

| law | p | series cut + coefficients | contact expansions | proven pieces |
|---|---|---|---|---|
| Power2 | 0.0092 | 3.3 µs | 0.06 ms | 8.0 ms |
| Power2 | 0.1 | 3.3 µs | 0.14 ms | 19.3 ms |
| Nonlinear | 0.0092 | 7.4 µs | 0.23 ms | 20.1 ms |
| Nonlinear | 0.1 | 7.4 µs | 0.42 ms | 36.9 ms |

## Light curves (orbit + flux), as in a fit

Kepler-like super-Earth: 30 transits at 1-min cadence (10 380 samples), p = 0.02, a/R = 20, b = 0.4;
TESS-like hot Jupiter: 7 transits at 2-min cadence (1316 samples), p = 0.1, a/R = 9, b = 0.35.

| case | law | light curve | gradient (ForwardDiff, 7 parameters) |
|---|---|---|---|
| Kepler super-Earth | Quadratic | 0.059 ms | 2.41 ms |
| Kepler super-Earth | Power2 | 0.156 ms | 2.24 ms |
| TESS hot Jupiter | Quadratic | 0.021 ms | 0.25 ms |
| TESS hot Jupiter | Power2 | 0.328 ms | 1.51 ms |

## Against Limbdark.jl (quadratic law, p = 0.1, 10 000 samples)

| | Pizzetti.jl | Limbdark.jl |
|---|---|---|
| light curve | 0.126 ms | 0.480 ms |
| light curve with gradient (r, b, u) | 1.072 ms | 1.072 ms |
| single call `transit_poly` | 29 ns | 1541 ns |

Limbdark.jl re-uses its work structure across samples (`transit_poly!(t)`); `transit_poly(r, b, u)`
allocates it per call. Pizzetti.jl's gradient for the quadratic family uses closed forms
(`flux_gradient_quadratic`).

Largest difference of the light curves: 7.2e-10 (the interior series, proven below 1e-9).
