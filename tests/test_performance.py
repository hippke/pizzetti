"""Performance against batman (run with ``pytest -m perf -s`` to see timings).

Timings depend on the machine; the assertions only require pizzetti to be
faster than batman by a modest margin for the quadratic law, and the interior
series to be faster than the exact solution."""
import time

import numpy as np
import pytest

import pizzetti
from pizzetti import _exact
from pizzetti.occult import _A_HALF
from pizzetti import _series

pytestmark = pytest.mark.perf


def best_time(f, repeat=7):
    f()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def params(mod, rp, law, u):
    p = mod.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 10.0, rp, 20.0, 89.5, 0.0, 90.0
    p.u, p.limb_dark = list(u), law
    return p


@pytest.mark.parametrize("rp", [0.0092, 0.1])
@pytest.mark.parametrize("law,u", [("quadratic", [0.4, 0.25]), ("nonlinear", [0.5, 0.1, 0.1, -0.1]), ("power2", [0.7, 0.6])])
def test_light_curve_speed_vs_batman(rp, law, u):
    batman = pytest.importorskip("batman")
    t = np.linspace(-0.25, 0.25, 100_000)
    out = {}
    for name, mod in (("batman", batman), ("pizzetti", pizzetti)):
        p = params(mod, rp, law, u)
        m = mod.TransitModel(p, t)
        out[name] = best_time(lambda: m.light_curve(p))
    speedup = out["batman"] / out["pizzetti"]
    print(f"\n{law:10s} rp={rp:<7} batman {out['batman'] * 1e3:8.3f} ms  pizzetti {out['pizzetti'] * 1e3:8.3f} ms  "
          f"speed-up {speedup:5.1f}")
    if law == "quadratic":
        assert speedup > 1.5


def test_interior_series_vs_exact_kernel():
    k, u1, u2 = 0.1, 0.4, 0.25
    n = 1_000_000
    z = np.random.default_rng(0).uniform(0, 0.9 * (1 - k), n)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1, c2 = 1 - u1 - 2 * u2, u1 + 2 * u2
    kk = k * k * omega
    a = _series.coeffs(_A_HALF, k, kk * c2)
    A0 = 1 - kk * ((1 - u1 - u2) - u2 * (1 - k * k / 2))
    out = np.empty(n)
    fb, dev = np.empty(n), np.empty(n)
    ts = best_time(lambda: _series.series_quadratic(z, a, A0, -kk * u2, out))
    te = best_time(lambda: _exact.exact_batch(z, n, k, c1, c2, u2, omega, fb, dev))
    print(f"\ninterior series {ts / n * 1e9:.2f} ns/sample, exact {te / n * 1e9:.2f} ns/sample, ratio {te / ts:.1f}")
    assert np.max(np.abs(out - fb)) < 1e-9
    assert te / ts > 3
