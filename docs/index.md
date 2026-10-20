# pizzetti

**Fast limb-darkened transit light curves, with an elementary series and a proven error bound.**

`pizzetti` is a drop-in replacement for [batman](https://github.com/lkreidberg/batman). While the
planet lies entirely inside the stellar disk, it evaluates the transit flux with an elementary
series: `sqrt(1 - z²)` times a polynomial in `1/(1 - z²)`, whose coefficients depend only on the
radius ratio. A closed-form bound on the truncation error is proven, and it guarantees the
requested accuracy (default: 10⁻⁹ of the stellar flux). Points near the stellar limb use the exact
Mandel & Agol solution (quadratic family) or a fixed-node trapezoid rule for the limb integral
that is accurate to 10⁻¹⁵ (other laws).

- **Fast:** one square root, one division and a dozen multiply–adds per in-transit point; light
  curves 7–44× faster than batman, likelihoods in complete MCMC fits 3.4–4.4× faster.
- **Accurate:** verified against a 40-digit reference to 2 × 10⁻⁹ for every supported law,
  including the contact points.
- **Certified on request:** `limb="proven"` extends the proven bound to the limb.
- **Compatible:** the same `TransitParams` / `TransitModel` interface as batman.
- **Limb darkening integrated out:** `MarginalTransitModel` marginalises the limb darkening and
  the baseline analytically over their physical region (a cone in the Bernstein basis) and
  samples the geometry by profile importance sampling, typically 40× fewer likelihood
  evaluations than emcee on the full model ([details](marginal.md)).

```python
import pizzetti as batman   # existing batman code runs unchanged
```

Tutorials (ten notebooks from a first light curve to certified models, and five Julia scripts)
are in the repository's [`examples/`](https://github.com/hippke/pizzetti/tree/main/examples) folder.

```{toctree}
:maxdepth: 2

installation
quickstart
batman
method
marginal
accuracy
performance
api
citing
changelog
```
