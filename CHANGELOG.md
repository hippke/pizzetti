# Changelog

## 0.1.0 (unreleased)

- First release.
- Interior series with a rigorous error bound for all power-law intensity laws.
- Exact vectorised Mandel & Agol solver near the limb (uniform, linear, quadratic laws).
- Contact expansions near the limb for the square-root, nonlinear and power-2 laws: power series
  about first and second contact (with the explicit non-analytic term at second contact), built
  per light curve from a single-variable hypergeometric form of the Mandel & Agol (2002) solution.
- Optional proven limb mode (`limb="proven"`): the contact expansions split into pieces whose
  degrees are chosen so that the flux error is proven to be below `series_tol` (radius ratios up to
  ~0.2); `return_certified=True` reports which samples the proof covers. Costs 10-100 ms of set-up
  per light curve, so the fixed-order expansions remain the default.
- Green's-theorem quadrature as fallback (short light curves, radius ratios above ~0.25, z <= p).
- batman-compatible `TransitParams` / `TransitModel`: eccentric orbits, supersampling, secondary
  eclipses, inverse transits.
