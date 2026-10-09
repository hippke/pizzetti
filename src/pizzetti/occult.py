"""Occultation flux for arrays of projected separations.

* body entirely inside the stellar disk and z <= z_c: the interior series
  (rigorous bound <= ``series_tol`` on the flux error);
* all other on-star samples: exact Mandel & Agol (2002) solution for the
  uniform, linear and quadratic laws; for the other power laws, the
  contact expansions (_contact; power series about first and second
  contact with coefficients computed once per light curve), or Green's-
  theorem quadrature for short light curves and where the expansions do
  not apply (radius ratios above about 0.25, or z <= p).
"""
from math import pi

import numpy as np
from numba import njit

from . import _contact, _exact, _general, _lens, _series
from .laws import QUADRATIC_FAMILY, power_terms, quadratic_coefficients

#: Default bound on the flux error of the interior series (absolute flux).
SERIES_TOL = 1e-9
#: Relative truncation tolerance of the hypergeometric limb solution.
LIMB_TOL = 1e-16
#: The contact expansions (set-up 0.1-1 ms per exponent and light curve) are
#: used when there are at least this many limb samples per exponent.
FIT_MIN_SAMPLES = 300

_A_HALF = _series.table(0.5, _series.M)


@njit(cache=True, fastmath=True, error_model="numpy")
def _series_select(z, a, A0, A1, z_c, one_k, out):
    """Branch-free, vectorised: out[i] = series flux if z <= z_c, else 1.
    Returns the number of samples with z_c < z < 1 + k (need the exact flux)."""
    a0, a1, a2, a3, a4, a5 = a[0], a[1], a[2], a[3], a[4], a[5]
    a6, a7, a8, a9, a10, a11 = a[6], a[7], a[8], a[9], a[10], a[11]
    cnt = 0
    for i in range(z.shape[0]):
        zi = abs(z[i])
        z2 = zi * zi
        w = max(1.0 - z2, 1e-12)
        v = 1.0 / w
        P = ((((((((((a11 * v + a10) * v + a9) * v + a8) * v + a7) * v + a6) * v + a5) * v
                 + a4) * v + a3) * v + a2) * v + a1) * v + a0
        f = A0 + A1 * z2 - np.sqrt(w) * P
        out[i] = f if zi <= z_c else 1.0
        cnt += 1 if (zi > z_c) & (zi < one_k) else 0
    return cnt


@njit(cache=True, fastmath=True)
def _count_rest(z, z_c, one_k):
    cnt = 0
    for i in range(z.shape[0]):
        zi = abs(z[i])
        cnt += 1 if (zi > z_c) & (zi < one_k) else 0
    return cnt


@njit(cache=True)
def _quadratic_flux(z, k, u1, u2, z_c, A_half):
    n = z.shape[0]
    out = np.empty(n)
    if k <= 0.0:
        out[:] = 1.0
        return out
    k2 = k * k
    omega = 1.0 / (1.0 - u1 / 3.0 - u2 / 6.0)
    c1 = 1.0 - u1 - 2.0 * u2
    c2 = u1 + 2.0 * u2
    kk = k2 * omega
    one_k = 1.0 + k
    # pass 1: interior series where z <= z_c, 1 elsewhere (vectorised)
    if z_c >= 0.0:
        a = _series.coeffs(A_half, k, kk * c2)
        A0 = 1.0 - kk * ((1.0 - u1 - u2) - u2 * (1.0 - k2 / 2.0))
        m = _series_select(z, a, A0, -kk * u2, z_c, one_k, out)
    else:
        out[:] = 1.0
        m = _count_rest(z, z_c, one_k)
    if m == 0:
        return out
    # pass 2: collect the samples that need the exact solution
    zb = np.empty(m)
    idx = np.empty(m, dtype=np.int64)
    j = 0
    for i in range(n):
        zi = abs(z[i])
        if (zi > z_c) & (zi < one_k):
            if k >= 1.0 and zi <= k - 1.0:
                out[i] = 0.0
                continue
            if abs(zi - k) < _exact.Z_K_MIN:
                zi = k + max(k * 2.220446049250313e-16, _exact.Z_K_MIN)
            zb[j] = zi
            idx[j] = i
            j += 1
    m = j
    # pass 3: exact solution (vectorised), scalar fallback, scatter
    if m > 0:
        fb = np.empty(m)
        dev = np.empty(m)
        _exact.exact_batch(zb, m, k, c1, c2, u2, omega, fb, dev)
        for j in range(m):
            f = fb[j]
            if not dev[j] <= 0.0:
                f = _exact.exact_point(zb[j], k, c1, c2, u2, omega)
            out[idx[j]] = f
    return out


@njit(cache=True)
def _general_flux(z, k, gams, cs, z_c, tables, gu, gw, ltabs, llens, use_hyp):
    """Power-law intensities: interior series for z <= z_c, Green's-theorem
    quadrature (limb) otherwise. Integer exponents 0 and 1 are folded into
    A0 + A1 z^2; the exponents {1/4, 1/2, 3/4} and a single arbitrary
    exponent have unrolled vectorised kernels."""
    n = z.shape[0]
    out = np.ones(n)
    if k <= 0.0:
        return out
    nt = gams.shape[0]
    norm = 0.0
    for i in range(nt):
        norm += cs[i] / (2.0 * gams[i] + 2.0)
    scale = k * k / (2.0 * norm)
    total = 2.0 * pi * norm
    # interior: F = A0 + A1 z^2 - sum_{non-integer g} w^g P_g(v) (+ generic rest)
    A0 = 1.0
    A1 = 0.0
    aq = np.zeros(_series.M + 1)
    ah = np.zeros(_series.M + 1)
    a3q = np.zeros(_series.M + 1)
    ag = np.zeros(_series.M + 1)
    g_other = -1.0
    n_other = 0
    for i in range(nt):
        g = gams[i]
        if g == 0.0:
            A0 -= cs[i] * scale
        elif g == 1.0:
            A0 -= cs[i] * scale * (1.0 - k * k / 2.0)
            A1 += cs[i] * scale
        else:
            a = _series.coeffs(tables[i], k, cs[i] * scale)
            if g == 0.25:
                aq += a
            elif g == 0.5:
                ah += a
            elif g == 0.75:
                a3q += a
            elif n_other == 0 or g == g_other:
                ag += a
                g_other = g
                n_other = 1
            else:
                n_other = 2
    one_k = 1.0 + k
    ms = 0
    ml = 0
    for i in range(n):
        zi = abs(z[i])
        if zi <= z_c:
            ms += 1
        elif zi < one_k:
            ml += 1
    zb = np.empty(ms)
    idx = np.empty(ms, dtype=np.int64)
    zl = np.empty(ml)
    il = np.empty(ml, dtype=np.int64)
    j = 0
    jl = 0
    for i in range(n):
        zi = abs(z[i])
        if zi <= z_c:
            zb[j] = zi
            idx[j] = i
            j += 1
        elif zi < one_k:
            if k >= 1.0 and zi <= k - 1.0:
                out[i] = 0.0
            else:
                zl[jl] = zi
                il[jl] = i
                jl += 1
    ml = jl
    # limb samples: hypergeometric solution, compressed per light curve, where
    # it applies and pays off; Green's-theorem quadrature otherwise
    bl = np.full(ml, np.nan)
    if use_hyp and ml >= FIT_MIN_SAMPLES * nt and k < 0.5:
        bounds, REG, SA, SB, SCc, mexp, est = _contact.build(k, gams, cs, ltabs, llens, z_c, _contact.NS)
        if est <= _contact.TOL:
            _contact.evaluate(zl[:ml], k, bounds, REG, SA, SB, SCc, mexp, bl)
    for j in range(ml):
        b = bl[j]
        if np.isnan(b):
            b = _general.blocked_limb(zl[j], k, gams, cs, gu, gw)
        out[il[j]] = 1.0 - b / total
    if ms > 0:
        fs = np.empty(ms)
        if n_other <= 1:
            _series.series_quarter(zb, ms, aq, ah, a3q, A0, A1, fs)
            if n_other == 1:
                fo = np.empty(ms)
                _series.series_pow(zb, ms, ag, g_other, 0.0, 0.0, fo)
                for j in range(ms):
                    fs[j] += fo[j]
        else:
            T = np.empty((nt, _series.M + 1))
            for i in range(nt):
                T[i, :] = _series.coeffs(tables[i], k, cs[i] * scale)
            _series.series_general(zb, ms, gams, T, fs)
        for j in range(ms):
            out[idx[j]] = fs[j]
    return out


def series_cut(rp, limb_dark, u, series_tol=SERIES_TOL):
    """Cut z_c: the interior series is used for z <= z_c, where its flux error
    is rigorously bounded by ``series_tol``. Returns -1 if the series is not
    used (e.g. large radius ratios or ``series_tol <= 0``)."""
    k = abs(float(rp))
    al, c = power_terms(limb_dark, u)
    gams = al / 2.0
    norm = np.sum(c / (al + 2.0))
    if k <= 0 or k >= 1 or series_tol <= 0 or norm == 0:
        return -1.0
    return _series.zcut(k, gams, np.abs(c), k * k / (2.0 * abs(norm)), float(series_tol))


def limb_tables(gams):
    """Hypergeometric coefficient tables for the limb solution (cached per exponent)."""
    ts, ls = [], []
    for g in gams:
        key = float(g)
        if key not in _LIMB_CACHE:
            if len(_LIMB_CACHE) > 256:
                _LIMB_CACHE.clear()
            _LIMB_CACHE[key] = _lens.table(key, _contact.JC)
        t, l = _LIMB_CACHE[key]
        ts.append(t)
        ls.append(l)
    return np.ascontiguousarray(ts), np.ascontiguousarray(ls)


_LIMB_CACHE = {}


def occult(z, rp, u, limb_dark="quadratic", series_tol=SERIES_TOL, limb="hypergeometric"):
    """Relative flux of a star occulted by an opaque disk.

    :param z: projected centre separations (stellar radii), array-like
    :param rp: radius ratio (planet / star)
    :param u: limb-darkening coefficients (batman conventions)
    :param limb_dark: "uniform", "linear", "quadratic", "squareroot", "nonlinear" or "power2"
    :param series_tol: bound on the flux error of the interior series; 0 disables it
    :param limb: "hypergeometric" (default) or "quadrature": method for the
        samples beyond z_c with non-quadratic laws. "hypergeometric" uses the
        contact expansions and falls back to quadrature where they do not
        reach the tolerance (large radius ratios) or do not apply (z <= p).
    :return: relative flux (ndarray)
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    shape = z.shape
    z = z.ravel()
    k = abs(float(rp))
    z_c = series_cut(k, limb_dark, u, series_tol)
    if limb_dark in QUADRATIC_FAMILY:
        u1, u2 = quadratic_coefficients(limb_dark, u)
        f = _quadratic_flux(z, k, u1, u2, z_c, _A_HALF)
    else:
        al, c = power_terms(limb_dark, u)
        gams = al / 2.0
        tables = np.empty((gams.size, _series.M + 1, _series.M + 1))
        for i, g in enumerate(gams):
            tables[i] = _series.table(float(g), _series.M)
        if limb not in ("hypergeometric", "quadrature"):
            raise ValueError('limb must be "hypergeometric" or "quadrature"')
        ltabs, llens = limb_tables(gams)
        f = _general_flux(z, k, gams, c, z_c, tables, _general.GL_U, _general.GL_W,
                          ltabs, llens, limb == "hypergeometric")
    return f.reshape(shape)
