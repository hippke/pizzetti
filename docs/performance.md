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

The numbers come from `python benchmarks/benchmark.py` (10⁵ points, about 35 % in transit, Intel
Xeon, numba 0.68, batman 2.5.3, single thread):
- "flux only": consecutive calls change only `rp` and `u`;
- "full": every call also changes the orbit, as in a fit.

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 1.74 ms | 0.11 ms | 16.3× | 4.02 ms | 0.31 ms | 13.0× |
| quadratic | 0.1 | 2.07 ms | 0.24 ms | 8.7× | 4.31 ms | 0.42 ms | 10.2× |
| nonlinear | 0.0092 | 2.26 ms | 0.29 ms | 7.8× | 4.44 ms | 0.49 ms | 9.1× |
| nonlinear | 0.1 | 38.3 ms | 0.85 ms | 44.9× | 40.4 ms | 0.91 ms | 44.3× |
| power2 | 0.0092 | 1.06 ms | 0.26 ms | 4.0× | 3.26 ms | 0.47 ms | 6.9× |
| power2 | 0.1 | 23.6 ms | 0.53 ms | 44.7× | 25.9 ms | 0.76 ms | 34.0× |

batman's numerical laws (nonlinear, power-2) run at its default `max_err` of 1 ppm; pizzetti stays
below 10⁻⁹.

### Low transit duty cycle

Long light curves have few points in transit. With `python benchmarks/benchmark.py --half-width 2.5`
(10⁵ points, about 3 % in transit):

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 0.43 ms | 0.06 ms | 6.9× | 2.72 ms | 0.26 ms | 10.6× |
| quadratic | 0.1 | 0.48 ms | 0.08 ms | 6.2× | 2.82 ms | 0.26 ms | 10.8× |
| nonlinear | 0.0092 | 0.29 ms | 0.16 ms | 1.8× | 2.58 ms | 0.38 ms | 6.8× |
| nonlinear | 0.1 | 3.90 ms | 0.22 ms | 17.5× | 6.19 ms | 0.43 ms | 14.5× |
| power2 | 0.0092 | 0.18 ms | 0.15 ms | 1.2× | 2.49 ms | 0.34 ms | 7.3× |
| power2 | 0.1 | 2.46 ms | 0.25 ms | 9.8× | 4.90 ms | 0.48 ms | 10.3× |

## In a fit

In complete emcee fits (orbit, light curve and χ² at every step, all parameters changing), one
likelihood evaluation is 4.4× faster than batman's for a TESS-like hot Jupiter (quadratic law)
and 3.4× for a Kepler-like super-Earth (power-2 law with free α); for grazing transits, which
have no interior phase, the gain is 2–3.5× (`examples/inference`, `examples/paper/grazing.py`).

## Notes

- The series is evaluated branch-free in blocks of 64 samples for every law. A block in which
  no sample lies on the star (all separations ≥ 1 + rp, flux exactly 1) is filled with 1 and
  skipped (about 0.4 ns per sample); only points near the limb are gathered for the limb solver.
  The flux is identical to that of the unblocked evaluation, because a sample is skipped only if
  it fails the test `z < 1 + rp` that already selected its flux of 1.
- For the power-2 law with an arbitrary exponent, each interior point needs one `exp` and one `log`
  (about 20 ns), because numba cannot vectorise them. The quarter powers of the nonlinear and
  square-root laws are computed with square roots instead.
- Points near the limb cost about 10 ns with the exact solver (quadratic family). For the other
  laws: 170–300 ns with the direct trapezoid rule (no set-up), 30–100 ns with its interpolants
  (about 40 µs set-up per light curve, all exponents), against 0.6–1 µs with the quadrature.
  `limb="fast"` switches to the interpolants from 150 limb samples on.
- The per-call overhead of the non-quadratic laws is about 15 µs (cached per-exponent plans).
- Circular orbits use a branch-free sine polynomial in turns, which vectorises.

## Marginal likelihood (`MarginalTransitModel`)

One likelihood evaluation, n = 1316, rp = 0.1, 280 samples near the limb, single thread:

| law | single call | per row (batch of 64) | where the time goes |
|---|---|---|---|
| uniform, linear | ~21 µs | ~12 µs | orbit 4 µs, `occult_terms` 7 µs, normal equations |
| quadratic (fast path) | ~19 µs | ~15 µs | as above, 3×3 closed form |
| square-root, power-2 | ~35 µs | ~25 µs | + one power-law column (~10 µs) |
| nonlinear | ~53 µs | ~39 µs | + two columns, 5×5 system |
| power-2, free α | ~54 µs | ~44 µs | + collapsing the α axis of the panel (~5 µs) |
| logarithmic | ~68 µs | ~60 µs | + four columns (μ ln μ by a difference in the exponent) |

One power-law column costs about 12 µs:
- ~3 µs fixed: Chebyshev interpolation of the expansion coefficients, and the interior cut;
- ~2.5 µs for 378 interior samples;
- ~5 µs for 280 limb samples (contact expansions evaluated across samples, so the loops
  vectorise; ~18 ns each).

The same 280 samples cost ~300 µs with the quadrature and ~50 µs with `occult`'s default
interpolants (set-up included); the panel scheme avoids even that per-light-curve set-up.
