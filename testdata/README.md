# Test vectors

Language-neutral reference values, shared by the Python package (`tests/test_reference_vectors.py`)
and the Julia package (`julia/test/test_reference.jl`), and available to further implementations.

| file | content | generator |
|---|---|---|
| `blocked_reference.json` | The blocked integral `P_γ(z, p) = ∫ (1 - r²)^γ dA` over the occulted part of the unit stellar disk, for 11 exponents (γ = 0, ¼, ½, ¾, 1, 3/2, 2, 0.05, 0.3, 0.6, 1.37) × 7 radius ratios (0.0092 … 1.3) at the contacts (±1e-12, ±1e-7, ±1e-3), `z = p`, the interior and the limb, in double precision; and a 30-digit subset (strings). | `make_testdata.py` |
| `python_lightcurves.json` | Light curves and separations of the Python package (Keplerian orbits incl. eccentric, five laws, radius ratios 0.009–0.3) for cross-implementation checks. | `make_crosscheck.py` |

The references in `blocked_reference.json` are **independent** of the paper's derivations and of the
code: tanh-sinh quadrature of the area integral in polar coordinates about the occultor centre
(`validation/refcheck.py`), with no series, no Green's theorem and no hypergeometric functions.
The flux of a law `I = Σ cᵢ μ^αᵢ` follows as `F = 1 - Σ cᵢ P_{αᵢ/2} / (2π Σ cᵢ/(αᵢ + 2))`.

`python_lightcurves.json` is a consistency check, not a reference.
