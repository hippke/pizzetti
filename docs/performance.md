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
Core Ultra 5 226V, single thread):
- "flux only": consecutive calls change only `rp` and `u`;
- "full": every call also changes the orbit.

| law | rp | batman | pizzetti | speed-up | batman (full) | pizzetti (full) | speed-up |
|---|---|---|---|---|---|---|---|
| quadratic | 0.0092 | 1.79 ms | 0.16 ms | 11.0× | 3.49 ms | 0.40 ms | 8.7× |
| quadratic | 0.1 | 1.91 ms | 0.25 ms | 7.6× | 3.78 ms | 0.52 ms | 7.2× |
| nonlinear | 0.0092 | 1.88 ms | 0.58 ms | 3.2× | 3.90 ms | 0.85 ms | 4.6× |
| nonlinear | 0.1 | 30.9 ms | 5.7 ms | 5.4× | 33.2 ms | 6.2 ms | 5.4× |
| power2 | 0.0092 | 1.06 ms | 1.05 ms | 1.0× | 2.97 ms | 1.35 ms | 2.2× |
| power2 | 0.1 | 19.2 ms | 6.7 ms | 2.9× | 21.4 ms | 7.0 ms | 3.1× |

## Notes

- Out-of-transit points cost about 1 ns each. The series is evaluated branch-free for all points
  and selected, and only points near the limb are gathered for the reference solver.
- For the power-2 law with an arbitrary exponent, each interior point needs one `exp` and one `log`
  (about 20 ns), because numba cannot vectorise them. The quarter powers of the nonlinear and
  square-root laws are computed with square roots instead.
- Points near the limb cost about 10 ns with the exact solver (quadratic family) and 0.6–0.7 µs
  with the quadrature (other laws).
- Circular orbits use a branch-free sine polynomial in turns, which vectorises.
