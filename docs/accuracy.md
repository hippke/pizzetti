# Accuracy

The test suite (`tests/`) checks `pizzetti` against two references.

## 40-digit reference

`tests/reference.py` computes the flux of any power-law intensity by Green's theorem with adaptive
mpmath quadrature at 40 significant digits. It uses neither the series nor elliptic integrals.

The checks cover:
- every supported law;
- radius ratios 0.0092, 0.1, 0.3, 0.7 and 1.5;
- separations at z = 0, z = p and 0.5(1 − p);
- separations within 10⁻¹⁵ … 0.2 of both contact points.

The results:

| component | max. error |
|---|---|
| full model, every law | < 2 × 10⁻⁹ (test bound) |
| exact quadratic-family solver | ≤ 7 × 10⁻¹⁵ |
| limb quadrature (power-2, nonlinear, square-root) | ≤ 3 × 10⁻¹¹ |
| interior series (default `series_tol`) | ≤ 10⁻⁹, proven |

## The series bound

- `test_series_error_below_bound` checks that the series stays within `series_tol` of the
  reference solution for all $z\le z_c$. It covers six laws and radius ratios from 0.001 to 0.4.
- `test_bound_is_rigorous_and_tight_high_precision` compares the truncated series with the
  40-digit disk mean. It checks that the error is non-negative and below the bound, and that the
  bound is less than twice the error for $z\le0.9(1-p)$.
- `test_bound_monotone_in_z` checks the monotonicity on which the cut relies.

## batman

Light curves agree with batman to:
- 5 × 10⁻⁸ for the uniform, linear and quadratic laws, a level set by batman's polynomial
  approximations of the elliptic integrals;
- 3 × 10⁻⁷ for the numerical laws, a level set by batman's integration step even at its tightest
  `max_err`.

`test_closer_to_reference_than_batman` confirms that `pizzetti` is at least ten times closer to the
40-digit reference than batman.

Eccentric orbits agree to within batman's Kepler tolerance (10⁻⁷). Secondary eclipses agree to
10⁻¹².
