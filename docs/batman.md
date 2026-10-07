# Moving from batman

`pizzetti` implements batman's public interface, so in most cases one line is enough:

```python
import pizzetti as batman
```

## Identical

- `TransitParams` attributes: `t0`, `per`, `rp`, `a`, `inc`, `ecc`, `w`, `u`, `limb_dark`, `fp`,
  `t_secondary`.
- `TransitModel(params, t, max_err=1.0, nthreads=1, fac=None, transittype="primary",
  supersample_factor=1, exp_time=0.0)`.
- Methods: `light_curve`, `get_t_periastron`, `get_t_secondary`, `get_t_conjunction`,
  `get_true_anomaly`, `calc_err`. The attribute `ds` holds the separations.
- Orbit conventions: batman's computation of the time of periastron, its circular-orbit branch
  for `ecc < 1e-5`, and its rule that a planet behind the star gives separation 100 for primary
  transits.
- Inverse transits for `rp < 0` (flux `2 - F`) and the uniform-disk secondary eclipse.

## Different

| | batman | pizzetti |
|---|---|---|
| quadratic / linear | Mandel & Agol with Hastings polynomials for K, E (errors ~2e-8) | interior series (proven ≤ 1e-9) + exact vectorised solver (≤ 1e-14) |
| power-2, nonlinear, square-root | numerical integration over annuli, step from `max_err` (errors ~1e-7 at the tightest setting) | interior series (proven ≤ 1e-9) + Green's-theorem quadrature (≤ 3e-11) |
| `logarithmic`, `exponential`, `custom` | supported | not supported (`NotImplementedError`): not sums of powers of μ |
| `max_err`, `fac` | control the integration step | accepted and ignored; accuracy is set by `series_tol` |
| `nthreads` | OpenMP | accepted; pizzetti runs single-threaded (a warning is issued for `nthreads > 1`) |
| Kepler's equation | Newton until residual < 1e-7 | Newton to 1e-15 |
| new keyword | – | `series_tol` (default 1e-9): rigorous bound on the series' flux error |
| new method | – | `series_cut()`: the separation up to which the series is used |

Because batman's Kepler iteration stops at a residual of 1e-7, eccentric orbits differ from batman
by up to ~1e-7·a in separation. `pizzetti` is the more accurate of the two.
