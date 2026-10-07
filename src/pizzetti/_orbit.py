"""Sky-projected separation and true anomaly (same conventions as batman's _rsky)."""
from math import pi

import numpy as np
from numba import njit

#: separation returned when the planet is behind the star (primary) or in
#: front of it (secondary), as in batman
BIGD = 100.0


# fdlibm __kernel_sin / __kernel_cos coefficients on [-pi/4, pi/4]
_S1 = -1.66666666666666324348e-01
_S2 = 8.33333333332248946124e-03
_S3 = -1.98412698298579493134e-04
_S4 = 2.75573137070700676789e-06
_S5 = -2.50507602534068634195e-08
_S6 = 1.58969099521155010221e-10
_C1 = 4.16666666666666019037e-02
_C2 = -1.38888888888741095749e-03
_C3 = 2.48015872894767294178e-05
_C4 = -2.75573143513906633035e-07
_C5 = 2.08757232129817482790e-09
_C6 = -1.13596475577881948265e-11


@njit(cache=True, fastmath=True, inline="always")
def sin_turns(u):
    """sin(2 pi u) for a phase u in turns, branch-free polynomial (vectorises);
    error ~1e-16 independent of |u| (the phase is reduced in turns)."""
    f = u - np.floor(u + 0.5)
    q = np.floor(4.0 * f + 0.5)
    r = 2.0 * pi * (f - 0.25 * q)
    z = r * r
    s = r + r * z * (_S1 + z * (_S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))))
    c = 1.0 - 0.5 * z + z * z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    k = np.int64(q) & 3
    v = c if (k & 1) == 1 else s
    return -v if k >= 2 else v


@njit(cache=True, fastmath=True)
def _rsky_circular(t, tp, per, a, si, w_turns, transittype, d):
    """Circular orbit: d = a sqrt(1 - sin^2(f + w) sin^2 i), f = 2 pi (t - tp) / per."""
    inv_per = 1.0 / per
    for i in range(t.shape[0]):
        s = sin_turns((t[i] - tp) * inv_per + w_turns)
        ss = s * si
        dd = a * np.sqrt(max(1.0 - ss * ss, 0.0))
        bad = (ss <= 0.0) if transittype == 1 else (ss >= 0.0)
        d[i] = BIGD if bad else dd


@njit(cache=True)
def eccentric_anomaly(M, e):
    """Solve Kepler's equation E - e sin E = M (Newton, robust start)."""
    Mr = M - 2.0 * pi * np.floor(M / (2.0 * pi))
    E = Mr + e * np.sin(Mr) if e < 0.8 else pi
    for _ in range(100):
        f = E - e * np.sin(E) - Mr
        dE = f / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-15 * max(1.0, abs(E)):
            break
    return E + (M - Mr)


@njit(cache=True)
def _true_anomaly(t, tc, per, ecc, omega):
    f0 = pi / 2.0 - omega
    if ecc < 1.0e-5:
        E = f0
        M = f0
    else:
        E = 2.0 * np.arctan(np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(f0 / 2.0))
        M = E - ecc * np.sin(E)
    tp = tc - per * M / 2.0 / pi
    n = t.shape[0]
    f = np.empty(n)
    if ecc < 1.0e-5:
        for i in range(n):
            x = (t[i] - tp) / per
            f[i] = (x - np.trunc(x)) * 2.0 * pi
    else:
        sq = np.sqrt((1.0 + ecc) / (1.0 - ecc))
        nm = 2.0 * pi / per
        for i in range(n):
            E = eccentric_anomaly(nm * (t[i] - tp), ecc)
            f[i] = 2.0 * np.arctan(sq * np.tan(E / 2.0))
    return f


@njit(cache=True)
def rsky(t, tc, per, a, inc, ecc, omega, transittype):
    """Separation of centres in stellar radii; inc and omega in radians;
    transittype 1 (primary) or 2 (secondary)."""
    n = t.shape[0]
    d = np.empty(n)
    si = np.sin(inc)
    if ecc < 1.0e-5:
        # time of periastron as in batman (true = mean anomaly for e = 0)
        tp = tc - per * (pi / 2.0 - omega) / 2.0 / pi
        _rsky_circular(t, tp, per, a, si, omega / (2.0 * pi), transittype, d)
        return d
    f = _true_anomaly(t, tc, per, ecc, omega)
    for i in range(n):
        s = np.sin(f[i] + omega)
        if transittype == 1 and s * si <= 0.0:
            d[i] = BIGD
        elif transittype == 2 and s * si >= 0.0:
            d[i] = BIGD
        else:
            d[i] = a * (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(f[i])) * np.sqrt(1.0 - s * s * si * si)
    return d


@njit(cache=True)
def true_anomaly(t, tc, per, ecc, omega):
    return _true_anomaly(t, tc, per, ecc, omega)


@njit(cache=True)
def eclipse(d, p, fp):
    """Secondary eclipse with a uniform planet disk (batman's _eclipse)."""
    if abs(p - 0.5) < 1.0e-3:
        p = 0.5
    n = d.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = d[i]
        if x >= 1.0 + p:
            out[i] = 1.0 + fp
        elif x < 1.0 - p:
            out[i] = 1.0
        else:
            kap1 = np.arccos(min((1.0 - p * p + x * x) / 2.0 / x, 1.0))
            kap0 = np.arccos(min((p * p + x * x - 1.0) / 2.0 / p / x, 1.0))
            at = (p * p * kap0 + kap1 - 0.5 * np.sqrt(max(4.0 * x * x - (1.0 + x * x - p * p) ** 2, 0.0))) / pi
            out[i] = 1.0 + fp * (1.0 - at / p / p)
    return out
