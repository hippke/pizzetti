# pizzetti

**Fast limb-darkened transit light curves, with an elementary series and a proven error bound.**

`pizzetti` is a drop-in replacement for [batman](https://github.com/lkreidberg/batman). While the
planet lies entirely inside the stellar disk, it evaluates the transit flux with an elementary
series: `sqrt(1 - z²)` times a polynomial in `1/(1 - z²)`, whose coefficients depend only on the
radius ratio. A closed-form bound on the truncation error is proven, and it guarantees the
requested accuracy (default: 10⁻⁹ of the stellar flux). Points near the stellar limb use an exact
or high-order quadrature solution.

- **Fast:** one square root, one division and a dozen multiply–adds per in-transit point; light
  curves 3–11× faster than batman.
- **Accurate:** verified against a 40-digit reference to 2 × 10⁻⁹ for every supported law,
  including the contact points.
- **Compatible:** the same `TransitParams` / `TransitModel` interface as batman.

```python
import pizzetti as batman   # existing batman code runs unchanged
```

```{toctree}
:maxdepth: 2

installation
quickstart
batman
method
accuracy
performance
api
citing
changelog
```
