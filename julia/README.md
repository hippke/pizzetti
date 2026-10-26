# Pizzetti.jl

**Limb-darkened transit light curves from an elementary series with a proven error bound, in Julia.**

Pizzetti.jl implements the methods of Hippke & Agol (in prep.) and of the Python package
[pizzetti](https://github.com/hippke/pizzetti) in Julia:

- **Interior series.** While the occultor lies inside the stellar disk, the flux of any law built
  from powers of μ is `w^γ` times a polynomial in `v = 1/(1 - z²)` (Pizzetti's mean-value
  expansion). Its truncation error has a closed-form bound (Theorem 2), so one cut `z_c` per light
  curve guarantees a prescribed tolerance (default `1e-9`). Cost: one square root, one division and
  a dozen multiply-adds per sample, in a SIMD loop.
- **Near and across the limb.** The exact Mandel & Agol (2002) solution for the quadratic family;
  for the other laws and `p < 0.3`, a fixed-node sinh-trapezoid rule for the limb integral written
  as a complete integral (no set-up), or, for light curves with at least 150 limb samples, Chebyshev
  interpolants built from it through the connection formula at second contact (~40 µs per light
  curve; both `~1e-15`; paper Section 5.3). Also available: power series about first and second
  contact built from Gauss hypergeometric functions (`~1e-15`), optionally split into pieces with a
  **proven** bound (Theorem 3), and Green's-theorem quadrature with a Gauss–Jacobi first segment (`~1e-16`).
- **Exact derivatives.** With ForwardDiff, dual numbers flow through orbits and light curves; the
  flux derivatives with respect to `z` and `p` are the closed-form hypergeometric expressions of the
  paper (Section 5.4), not derivatives of truncated series or of quadrature rules.
- **Any precision.** Everything is generic: `BigFloat` light curves (with a higher series order),
  and rigorous `IntervalArithmetic` enclosures of the exact interior flux, including rounding.
- **Drop-in for Limbdark.jl.** `transit_poly(r, b, u_n)` and `transit_poly!(r, b, u_n, dfdrbu)` with
  the same arguments and the same polynomial law; for degrees ≤ 2 the flux and its gradient come
  from closed forms (~100 ns per call with the gradient).

```julia
julia> using Pizzetti

julia> occult([0.0, 0.5, 0.95, 1.05], 0.1, Power2(0.6, 0.6))     # flux at separations z
4-element Vector{Float64}:
 0.9884033280505852
 0.9889845552077373
 0.9934649431396444
 0.9986520901005472
```

## Installation

From the repository (the package lives in the `julia/` folder):

```julia
using Pkg
Pkg.add(url = "https://github.com/hippke/pizzetti", subdir = "julia")
```

Requirements: Julia ≥ 1.10, SpecialFunctions. ForwardDiff and IntervalArithmetic are optional
(package extensions load when they are loaded).

## Usage

### Laws

| law | intensity |
|---|---|
| `Uniform()` | `1` |
| `Linear(u1)` | `1 - u1 (1 - μ)` |
| `Quadratic(u1, u2)` | `1 - u1 (1 - μ) - u2 (1 - μ)²` |
| `SquareRoot(u1, u2)` | `1 - u1 (1 - μ) - u2 (1 - √μ)` |
| `Nonlinear(c1, c2, c3, c4)` | `1 - Σ cₙ (1 - μ^(n/2))` (Claret 2000) |
| `Power2(c, α)` | `1 - c (1 - μ^α)` |
| `Polynomial(u)` | `1 - Σ uₙ (1 - μ)ⁿ` (Agol, Luger & Foreman-Mackey 2020) |
| `PowerSum(α, c)` | `Σ cᵢ μ^αᵢ` |

`limb_darkening("power2", [0.6, 0.6])` builds a law from batman's names.

### Flux

```julia
f = occult(z, p, law)                       # vector of separations
o = Occultor(p, law)                        # set-up once per (p, law) ...
f = o.(z)                                   # ... then evaluate (or evaluate!(out, o, z))
method(o, 0.95)                             # :series, :quadratic, :interpolate, :trapezoid, :contact, :proven or :quadrature
series_cut(p, law; series_tol = 1e-9)       # z_c: the series is used, with a proven bound, for z ≤ z_c
```

Keywords: `series_tol` (proven bound of the interior series; `0` switches it off), `limb`
(`:auto`, `:trapezoid`, `:interpolate`, `:fast` = contact expansions, `:quadrature`, `:proven`), `M`
(series order, odd; raise it for `BigFloat`). The trapezoid rules are used for Float64; dual numbers
and `BigFloat` keep the exact-derivative and quadrature paths.

### Light curves

```julia
prm = TransitParams(per = 3.5, rp = 0.1, a = 9.0, b = 0.35, law = Quadratic(0.40, 0.25))
flux = light_curve(t, prm)                                      # Keplerian orbit (batman conventions)
flux = light_curve(t, prm; exp_time = 30 / 1440, supersample = 7)   # exposure integration
z = rsky(t, prm)                                                # projected separations
```

`TransitParams` also takes `inc` (degrees) instead of `b`, `ecc`, `w`, and batman's `u` and
`limb_dark`.

### Derivatives

```julia
using ForwardDiff
ForwardDiff.gradient(v -> sum(light_curve(t, TransitParams(per = 3.5, rp = v[1], a = v[2], b = v[3],
                                                           law = Power2(v[4], v[5])))), [0.1, 9, 0.3, 0.6, 0.6])

F, dFdz, dFdp, dFdc = flux_gradient(z, p, law)       # exact, without automatic differentiation
F, dFdz, dFdp, dFdu = flux_gradient_law(z, p, law)   # with respect to the law's parameters
```

Under ForwardDiff the value is computed as usual (series within its proven bound, contact
expansions, closed forms), and the partial derivatives with respect to `z` and `p` come from
the exact expressions of the paper's Section 5.4 everywhere, including the interior (where the
derivative of the truncated series would only be accurate to ~1e-7 near the cut) and exactly at the
contact points (where automatic differentiation of the elliptic integrals or of a quadrature rule
fails). Derivatives with respect to exponents (the power-2 α) are Richardson-extrapolated
differences of the contact expansions or the quadrature (~1e-12). Nested duals (Hessians) use
plain automatic differentiation.

### Proven accuracy

```julia
f, certified = occult_certified(z, p, law; series_tol = 1e-9)   # proven pieces at the limb too

using IntervalArithmetic
I = occult_interval(0.3, 0.1, Power2(0.6, 0.6))   # contains the exact flux, rounding included
```

`occult_certified` uses the pieces of Theorem 3 near the limb and reports which samples carry a
proven bound (all of them for `p ≤ 0.2` at `1e-9`, `p ≤ 0.15` at `1e-12`). `occult_interval`
evaluates the interior series in interval arithmetic with the exact rational coefficient table
and adds the Theorem-2 bound: the result is an interval that contains the true flux.

### Limbdark.jl interface

```julia
transit_poly(r, b, u_n)                 # flux
transit_poly!(r, b, u_n, dfdrbu)        # flux, and dfdrbu = [∂F/∂r, ∂F/∂b, ∂F/∂u_1, ...]
```

## Examples

`examples/` (run from that folder with `julia --project=. <file>`):

| script | level | content |
|---|---|---|
| [`J1_quickstart.jl`](examples/J1_quickstart.jl) | beginner | laws as sums of powers of μ, the `Occultor`, which method computes which sample (`:interpolate`, `:trapezoid`, …), `evaluate!`, light curves with exposure time and eccentric orbits, first derivatives and BigFloat |
| [`J2_gradients.jl`](examples/J2_gradients.jl) | intermediate | Kepler-like super-Earth with the power-2 law: gradient check, Nelder–Mead vs L-BFGS with ForwardDiff, Hessian uncertainties |
| [`J3_precision_and_rigour.jl`](examples/J3_precision_and_rigour.jl) | advanced | certified light curves, interval enclosures, Theorem 2 against BigFloat, converged BigFloat references |
| [`J4_photodynamics.jl`](examples/J4_photodynamics.jl) | advanced | two planets near 2:1 with NbodyGradient.jl: transit times, impact parameters and sky velocities with their derivatives, chained with Pizzetti's exact ∂F/∂z into the derivative of every light-curve sample with respect to every initial orbital element and mass; checked against finite differences of the whole pipeline |
| [`J5_interoperability.jl`](examples/J5_interoperability.jl) | intermediate | fluxes, gradients and timing against Limbdark.jl; the same orbits and light curves as the Python package |
| [`J6_bartky.jl`](examples/J6_bartky.jl) | advanced | the limb integral in Bartky's variable (paper Section 5.3) step by step: the complete integral, the strip of analyticity, exp(-π²/h) convergence up to both contacts, the shifted contour inside the disk, the connection formula behind the interpolants, a generic version for BigFloat references, cost |

The scripts are in the [Literate.jl](https://github.com/fredrikekre/Literate.jl) format, so
`Literate.notebook("J1_quickstart.jl")` turns any of them into a Jupyter notebook.

![photodynamics](examples/figures/J4_transit_sensitivity.png)

## Accuracy and tests

`test/` (5993 tests, `] test Pizzetti`):

- against the **independent references** of [`testdata/`](../testdata): the blocked integral by
  tanh-sinh quadrature of the area integral (no series, no Green's theorem, no hypergeometric
  functions), in double precision for 11 exponents × 7 radius ratios including the contacts
  (every limb method within 3e-15 of the stellar flux; the series within its tolerance), and a
  30-digit subset (BigFloat quadrature within 1e-28);
- Theorem 2 (bound and one-sidedness) in BigFloat; the hypergeometric functions against
  HypergeometricFunctions.jl; the Python package (proven-mode certification identical sample by
  sample; light curves and orbits);
- all derivatives against finite differences and BigFloat differences, exactly at the contacts too;
- interval enclosures contain 60-digit values; Aqua.jl package checks.

## Speed

Single thread, Intel Xeon 2.8 GHz, Julia 1.13 (`benchmark/benchmarks.jl`, results in
`benchmark/results.md`). Per sample:

| | quadratic | power-2 | nonlinear |
|---|---|---|---|
| interior series (SIMD) | 2.3 ns | 5.5–5.9 ns | 3.3 ns |
| limb: trapezoid rule (direct) | — | 260–440 ns | 290–440 ns |
| limb: interpolants | — | 41 ns | 54–58 ns |
| contact expansions | — | 90 ns | 154 ns |
| quadrature (Gauss–Jacobi) | — | ~1.7 µs | ~2.8 µs |
| exact quadratic solution | 71–75 ns | — | — |

Set-up per light curve: 5–11 µs (series cut and coefficients); limb interpolants 36–55 µs;
contact expansions 0.12–0.8 ms; proven pieces 10–47 ms. (Trapezoid and interpolant rows: Intel
Xeon cloud instance, Julia 1.13.1.)

`evaluate!` skips runs of 64-sample blocks in which no sample lies on the star (`|z| ≥ 1 + p`,
flux exactly 1). This matters for long light curves with few points in transit. For 10⁵ sorted
separations with 3 % of them in transit, p = 0.0092: quadratic 0.23 → 0.048 ms, power-2
0.69 → 0.16 ms, nonlinear 0.52 → 0.23 ms; p = 0.1: 0.28 → 0.099 ms, 1.83 → 1.30 ms,
2.40 → 2.13 ms. With 30 % in transit the gain is 1.0–1.9×. A 10⁵-sample light curve (orbit and
flux) over 10 transits, p = 0.0092, takes 0.44 ms (quadratic) and 0.50 ms (power-2) instead of
0.59 ms and 1.06 ms. Short light curves that are mostly in transit (the Kepler and TESS cases
below) are unchanged. The price is the scan of the separations, 0.1–0.3 ns per sample: an array
in which every sample is on the star costs 2.0 → 2.3 ns (quadratic), 5.4 → 5.8 ns (power-2) and
3.2 → 3.3 ns (nonlinear) per sample.

The likelihoods of the paper's inference example (`benchmark/inference_cases.jl`, a new posterior
sample at every call):

| | hot Jupiter, quadratic, 1316 samples | super-Earth, power-2, 10 380 samples |
|---|---|---|
| likelihood (orbit, light curve, χ²) | 0.039 ms | 0.52 ms |
| gradient, 7 parameters (ForwardDiff) | 0.39 ms | 5.2 ms |
| for comparison: Python pizzetti / batman (other machine) | 0.024 / 0.107 ms | 0.48 / 0.83 ms |

Against Limbdark.jl (quadratic law, p = 0.1, 10 000 samples): a light curve costs 0.21 ms
(Limbdark.jl 0.82 ms with its work structure re-used); a light curve with the gradient
`[∂F/∂r, ∂F/∂b, ∂F/∂u₁, ∂F/∂u₂]` at every sample 1.6 ms (Limbdark.jl 1.5 ms); a single
`transit_poly` call 46 ns (2.5 µs). Fluxes agree to 5e-15 and gradients to 1e-14, except within 1e-16 of a contact,
where the derivative has a √δ term and both codes are limited by the rounding of their input.

## Citing

Please cite the accompanying paper (Hippke & Agol, in prep.), Mandel & Agol (2002) and, for the
polynomial interface, Agol, Luger & Foreman-Mackey (2020).
