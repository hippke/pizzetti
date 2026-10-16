# Performance

## Kernels

Interior points, quadratic law, rp = 0.1, single thread (`pytest -m perf -s`):

| kernel | time per point |
|---|---|
| interior series (this work) | 0.97 ns |
| exact Mandel & Agol, vectorised (5 Bulirsch steps) | 10.8 ns |

The series needs 2 square roots and divisions per point instead of about 14, and it is about 11×
faster.

## Light curves against batman

The numbers come from `python benchmarks/benchmark.py` (10⁵ points, about 30 % in transit, Intel
Xeon 2.8 GHz, numba 0.68, single thread):
- "flux only": consecutive calls change only `rp` and `u`;
- "full": every call also changes the orbit.

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 2.42 ms | 0.15 ms | 16.7× | 5.96 ms | 0.40 ms | 14.9× |
| quadratic | 0.1 | 3.01 ms | 0.32 ms | 9.5× | 6.57 ms | 0.61 ms | 10.8× |
| nonlinear | 0.0092 | 3.13 ms | 1.07 ms | 2.9× | 6.69 ms | 1.41 ms | 4.8× |
| nonlinear | 0.1 | 57.1 ms | 2.73 ms | 20.9× | 67.4 ms | 3.87 ms | 17.4× |
| power2 | 0.0092 | 2.00 ms | 0.83 ms | 2.4× | 5.25 ms | 1.16 ms | 4.5× |
| power2 | 0.1 | 35.9 ms | 1.77 ms | 20.2× | 39.7 ms | 2.13 ms | 18.7× |

### Low transit duty cycle

Long light curves have few points in transit. With `python benchmarks/benchmark.py --half-width 2.5`
(10⁵ points, about 3 % in transit):

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 0.63 ms | 0.09 ms | 7.1× | 4.36 ms | 0.35 ms | 12.3× |
| quadratic | 0.1 | 0.69 ms | 0.11 ms | 6.5× | 4.40 ms | 0.38 ms | 11.6× |
| nonlinear | 0.0092 | 0.39 ms | 0.54 ms | 0.7× | 4.11 ms | 0.85 ms | 4.8× |
| nonlinear | 0.1 | 5.77 ms | 1.64 ms | 3.5× | 9.45 ms | 1.91 ms | 5.0× |
| power2 | 0.0092 | 0.25 ms | 0.50 ms | 0.5× | 4.00 ms | 0.82 ms | 4.9× |
| power2 | 0.1 | 3.74 ms | 1.19 ms | 3.2× | 7.54 ms | 1.66 ms | 4.5× |

In the same setting the quadratic flux was 0.24 ms (rp = 0.0092) and 0.26 ms (rp = 0.1) before blocks
without on-star samples were skipped. The nonlinear and power-2 laws do not skip yet: they still
evaluate their series at every sample, which is why batman, which skips off-star samples, is
faster for small planets when few points are in transit.

## Notes

- Quadratic family: the series is evaluated branch-free in blocks of 64 samples. A block in which
  no sample lies on the star (all separations ≥ 1 + rp, flux exactly 1) is filled with 1 and skipped
  (about 0.4 ns per sample); only points near the limb are gathered for the reference solver.
  The flux is identical to that of the unblocked evaluation, because a sample is skipped only if it
  fails the test `z < 1 + rp` that already selected its flux of 1.
- Nonlinear, square-root and power-2 laws: the series is still evaluated for all samples (about
  1 ns each, more with `exp` / `log`) and selected.
- For the power-2 law with an arbitrary exponent, each interior point needs one `exp` and one `log`
  (about 20 ns), because numba cannot vectorise them. The quarter powers of the nonlinear and
  square-root laws are computed with square roots instead.
- Points near the limb cost about 10 ns with the exact solver (quadratic family) and 0.6–0.7 µs
  with the quadrature (other laws).
- Circular orbits use a branch-free sine polynomial in turns, which vectorises.

## Marginal likelihood (`MarginalTransitModel`)

One likelihood evaluation, n = 1316, rp = 0.1, 196 samples near the limb (single thread, loaded
machine, so the numbers are approximate):

| law | per evaluation | where the time goes |
|---|---|---|
| quadratic | 15–20 µs | fused numba path: orbit 4 µs, `occult_terms` 7 µs, 3×3 algebra |
| uniform | ~40 µs | general path: Python overhead (~20 µs) |
| linear | ~60 µs | general path: Python overhead (~30 µs) |
| square-root | ~330 µs | one μ^½ flux: ~22 µs call overhead + ~1.1 µs per limb sample (quadrature) |
| power-2 | ~370 µs | as square-root, plus `exp`/`log` per interior sample |
| nonlinear | ~610 µs | two non-integer exponents (μ^½, μ^{3/2}) |
| logarithmic | ~1270 µs | μ ln μ from four power-law fluxes (5-point difference in the exponent) |

For the non-quadratic laws the samples near the limb dominate. With fewer than about
4.5 × (number of series terms) limb samples, i.e. ≲ 430 for rp = 0.1, `occult` uses the
quadrature rather than the contact expansions, because building the expansions for a new rp
costs about as much as the quadrature (~180 µs here).
