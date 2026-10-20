"""Table (timing), upper block: cost per interior sample of the quadratic-law series
against the vectorised elliptic-integral closed form (Bulirsch cel at a fixed five steps;
pizzetti's _exact, adapted from Pandora) and batman. Single thread.
Run: python interior_timing.py   (the paper's numbers: Intel Core Ultra 5 226V)"""
import time
import numpy as np
import batman
from pizzetti import occult as _o  # noqa: F401  (compiles the kernels)
from pizzetti import _exact, _series
import importlib

O = importlib.import_module("pizzetti.occult")


def best(f, r=15):
    f(); ts = []
    for _ in range(r):
        s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
    return min(ts)


u1, u2 = 0.4, 0.25
print(f"{'p':>7s} {'series [ns]':>12s} {'closed form [ns]':>17s} {'batman [ns]':>12s} {'series speed-up':>16s}")
for k in (0.0092, 0.1):
    zc = O.series_cut(k, "quadratic", [u1, u2])
    z = np.random.default_rng(0).uniform(0, zc, 10**6)
    om = 1 / (1 - u1 / 3 - u2 / 6); c1 = 1 - u1 - 2 * u2; c2 = u1 + 2 * u2
    a = _series.coeffs(O._A_HALF, k, k * k * om * c2)
    out = np.empty_like(z); dev = np.empty_like(z)
    ts = best(lambda: _series.series_quadratic(z, a, 1.0, 0.0, out)) / z.size * 1e9
    te = best(lambda: _exact.exact_batch(z, z.size, k, c1, c2, u2, om, out, dev)) / z.size * 1e9
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.limb_dark, p.u = 0.0, 1.0, k, 10.0, 90.0, 0.0, 90.0, "quadratic", [u1, u2]
    zb = z[:10**5].copy()
    from batman import _quadratic_ld
    tb = best(lambda: _quadratic_ld._quadratic_ld(zb, k, u1, u2, 1)) / zb.size * 1e9
    print(f"{k:7.4f} {ts:12.2f} {te:17.2f} {tb:12.2f} {te / ts:15.1f}x")
