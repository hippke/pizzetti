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


from . import _contact, _exact, _general, _lens, _proven, _series
from .laws import QUADRATIC_FAMILY, power_terms, quadratic_coefficients

#: Default bound on the flux error of the interior series (absolute flux).
SERIES_TOL = 1e-9
#: Relative truncation tolerance of the hypergeometric limb solution.
LIMB_TOL = 1e-16
#: The contact expansions are used (limb="fast") when there are at least this many
#: limb samples per j-term of their set-up (tables and coefficients, ~10 us per
#: non-integer exponent and j-term, against ~1 us per exponent and sample for the
#: quadrature; measured break-even 150 samples for p = 0.02, 280 for p = 0.1).
CONTACT_BREAK_EVEN = 4.5

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
def _general_flux(z, k, gams, cs, z_c, tables, gu, gw, ju, jw, has, ltabs, llens, cgams, ccs, c01, mode, ltol):
    """Power-law intensities: interior series for z <= z_c, Green's-theorem
    quadrature (limb) otherwise. Integer exponents 0 and 1 are folded into
    A0 + A1 z^2; the exponents {1/4, 1/2, 3/4} and a single arbitrary
    exponent have unrolled vectorised kernels."""
    n = z.shape[0]
    out = np.ones(n)
    cert = np.ones(n, dtype=np.bool_)
    if k <= 0.0:
        return out, cert
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
    if mode == 2 and ml > 0:
        # proven: pieces with a proven flux bound <= ltol
        if k < 0.5:
            zb_, kinds_, s0s_, store_, degs_, okp, bnds_ = _proven.build(k, gams, cs, ltabs, llens, z_c, total, ltol)
            # pieces whose bound could not be met are left to quadrature
            for q in range(bnds_.shape[0]):
                if not bnds_[q] < np.inf:
                    zb_[q, 0] = np.nan
                    zb_[q, 1] = np.nan
            _proven.evaluate(zl[:ml], k, gams, zb_, kinds_, s0s_, store_, degs_, bl)
    elif mode == 1 and k < 0.5 and ltabs.shape[2] > 1:
        # exponents 0 and 1 in closed form (c01), the others (cgams, ccs) by the expansions
        bounds, REG, SA, SB, SCc, mexp, est = _contact.build(k, cgams, ccs, ltabs, llens, z_c, _contact.NS)
        if est <= _contact.TOL:
            _contact.evaluate(zl[:ml], k, bounds, REG, SA, SB, SCc, mexp, bl, c01[0], c01[1])
    for j in range(ml):
        b = bl[j]
        if np.isnan(b):
            b = _general.blocked_limb(zl[j], k, gams, cs, gu, gw, ju, jw, has)
        if mode != 2 or np.isnan(bl[j]):
            cert[il[j]] = False
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
    return out, cert


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


def limb_jmax(k, z_c):
    """Number of j-terms the limb tables need for radius ratio k when the limb
    solution starts at z_c: the j-series converges like x^j with
    x <= 1 - (z_c - k)^2; the margin covers the prefactors. Fewer terms than
    needed are safe (the contact build then reports non-convergence and the
    samples go to quadrature), so this only controls cost."""
    zl = max(z_c, 0.0) - k
    if zl <= 0.0:
        return _contact.JC
    xm = 1.0 - zl * zl
    J = int(np.ceil(1.3 * np.log(2.0 ** -60 * (1.0 - xm)) / np.log(xm))) + 8
    return int(min(_contact.JC, max(_contact.NS, 16 * ((J + 15) // 16))))


def limb_tables(gams, jmax=None):
    """Hypergeometric coefficient tables for the limb solution, j = 0..jmax
    (cached per exponent; a cached table with more terms is reused)."""
    if jmax is None:
        jmax = _contact.JC
    ts, ls = [], []
    for g in gams:
        key = float(g)
        hit = _LIMB_CACHE.get(key)
        if hit is None or hit[0].shape[1] - 1 < jmax:
            if len(_LIMB_CACHE) > 256:
                _LIMB_CACHE.clear()
            hit = _lens.table(key, jmax)
            _LIMB_CACHE[key] = hit
        t, l = hit
        ts.append(t[:, :jmax + 1])
        ls.append(l[:, :jmax + 1])
    return np.ascontiguousarray(ts), np.ascontiguousarray(ls)


_LIMB_CACHE = {}


def occult(z, rp, u, limb_dark="quadratic", series_tol=SERIES_TOL, limb="fast", return_certified=False):
    """Relative flux of a star occulted by an opaque disk.

    :param z: projected centre separations (stellar radii), array-like
    :param rp: radius ratio (planet / star)
    :param u: limb-darkening coefficients (batman conventions)
    :param limb_dark: "uniform", "linear", "quadratic", "squareroot", "nonlinear" or "power2"
    :param series_tol: bound on the flux error of the interior series; 0 disables it
    :param limb: method for the samples beyond z_c with non-quadratic laws:
        "fast" (default): power series about first and second contact, of
            fixed order, with coefficients computed once per light curve
            (error ~1e-15 by an a posteriori estimate, not a proof). Used
            for radius ratios below 0.5 when the light curve has enough
            limb samples to amortise the set-up (about 4.5 per j-term of the
            hypergeometric tables: ~150 for p = 0.02, ~430 for p = 0.1);
            quadrature otherwise. The exponents 0 and 1 are added in closed form.
        "proven": the same expansions, split into pieces whose degrees are
            chosen so that the flux error is proven to be at most
            ``series_tol`` (the guarantee of the interior series). Costs
            10-100 ms of set-up per light curve. Pieces whose bound cannot be
            met (radius ratios above ~0.2, or ~0.15 for series_tol=1e-12)
            fall back to quadrature (see ``return_certified``).
        "quadrature": Green's-theorem quadrature with a Gauss-Jacobi first
            segment (~1e-16 for every exponent and radius ratio).
    :param return_certified: also return a boolean array, True where the
        flux error is proven to be at most ``series_tol`` (interior series,
        proven limb pieces, closed forms and out-of-transit samples)
    :return: relative flux (ndarray), or (flux, certified) if requested
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    shape = z.shape
    z = z.ravel()
    k = abs(float(rp))
    z_c = series_cut(k, limb_dark, u, series_tol)
    if limb_dark in QUADRATIC_FAMILY:
        u1, u2 = quadratic_coefficients(limb_dark, u)
        f = _quadratic_flux(z, k, u1, u2, z_c, _A_HALF)
        cert = np.ones(z.shape, dtype=bool)
    else:
        al, c = power_terms(limb_dark, u)
        gams = al / 2.0
        tables = np.empty((gams.size, _series.M + 1, _series.M + 1))
        for i, g in enumerate(gams):
            tables[i] = _series.table(float(g), _series.M)
        modes = {"quadrature": 0, "fast": 1, "hypergeometric": 1, "proven": 2}
        if limb not in modes:
            raise ValueError('limb must be "proven", "fast" or "quadrature"')
        mode = modes[limb]
        ltol = float(series_tol) if series_tol > 0 else SERIES_TOL
        if mode == 2 and z_c < 0:
            mode = 1          # interior series disabled: nothing to match
        # limb tables only when the limb solution will be used: always for the proven
        # pieces; for the contact expansions when there are enough limb samples
        az = np.abs(z)
        n_limb = int(np.count_nonzero((az > z_c) & (az < 1.0 + k)))
        jl = limb_jmax(k, z_c) if k < 0.5 else _contact.JC
        integer = (gams == 0.0) | (gams == 1.0)
        c01 = np.array([np.sum(c[gams == 0.0]), np.sum(c[gams == 1.0])])
        cgams, ccs = gams, c
        if mode == 2 and k < 0.5:
            ltabs, llens = limb_tables(gams, jl)
        elif mode == 1 and k < 0.5 and n_limb >= CONTACT_BREAK_EVEN * jl:
            cgams, ccs = np.ascontiguousarray(gams[~integer]), np.ascontiguousarray(c[~integer])
            ltabs, llens = limb_tables(cgams, jl)
            if cgams.size == 0:
                ltabs = np.zeros((0, 2, jl + 1, 4, _lens.NC))
                llens = np.zeros((0, 2, jl + 1, 4), dtype=np.int64)
        else:
            ltabs = np.zeros((gams.size, 2, 1, 4, _lens.NC))
            llens = np.zeros((gams.size, 2, 1, 4), dtype=np.int64)
        ju, jw, has = _general.quad_rules(gams)
        f, cert = _general_flux(z, k, gams, c, z_c, tables, _general.GL_U, _general.GL_W, ju, jw, has,
                                ltabs, llens, cgams, ccs, c01, mode, ltol)
    if return_certified:
        return f.reshape(shape), cert.reshape(shape)
    return f.reshape(shape)
