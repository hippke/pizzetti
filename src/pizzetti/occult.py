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


from . import _contact, _exact, _general, _interp, _lens, _proven, _series, _trap
from .laws import QUADRATIC_FAMILY, check_law, power_terms, quadratic_coefficients

#: Default bound on the flux error of the interior series (absolute flux).
SERIES_TOL = 1e-9
#: Relative truncation tolerance of the hypergeometric limb solution.
LIMB_TOL = 1e-16
#: The contact expansions are used (limb="fast") when there are at least this many
#: limb samples per j-term of their set-up (tables and coefficients, ~10 us per
#: non-integer exponent and j-term, against ~1 us per exponent and sample for the
#: quadrature; measured break-even 150 samples for p = 0.02, 280 for p = 0.1).
CONTACT_BREAK_EVEN = 4.5

#: limb samples from which the Chebyshev interpolants (_interp) beat the direct rules
INTERP_BREAK_EVEN = 150

_A_HALF = _series.table(0.5, _series.M)


#: Block length of the series pass: blocks without any on-star sample are skipped.
BLK = 64


@njit(cache=True, fastmath=True, error_model="numpy")
def _series_select(z, a, A0, A1, z_c, one_k, out, bcnt):
    """Branch-free, vectorised series pass, in blocks of BLK samples:
    out[i] = series flux if z <= z_c, else 1. A block in which no sample lies on
    the star (all |z| >= 1 + k, flux exactly 1) is filled with 1 without
    evaluating the series. bcnt[b] = number of samples of block b with
    z_c < z < 1 + k (they need the exact flux); returns their total.
    The blocks are sliced (zr, fr) so that LLVM vectorises the inner loops."""
    a0, a1, a2, a3, a4, a5 = a[0], a[1], a[2], a[3], a[4], a[5]
    a6, a7, a8, a9, a10, a11 = a[6], a[7], a[8], a[9], a[10], a[11]
    n = z.shape[0]
    tot = 0
    for b in range((n + BLK - 1) // BLK):
        lo = b * BLK
        hi = min(lo + BLK, n)
        zr = z[lo:hi]
        fr = out[lo:hi]
        on = 0
        for q in range(hi - lo):
            on += 1 if (abs(zr[q]) < one_k) else 0
        if on == 0:
            for q in range(hi - lo):
                fr[q] = 1.0
            bcnt[b] = 0
            continue
        cnt = 0
        for q in range(hi - lo):
            zi = abs(zr[q])
            z2 = zi * zi
            w = max(1.0 - z2, 1e-12)
            v = 1.0 / w
            P = ((((((((((a11 * v + a10) * v + a9) * v + a8) * v + a7) * v + a6) * v + a5) * v
                     + a4) * v + a3) * v + a2) * v + a1) * v + a0
            f = A0 + A1 * z2 - np.sqrt(w) * P
            fr[q] = f if zi <= z_c else 1.0
            cnt += 1 if (zi > z_c) & (zi < one_k) else 0
        bcnt[b] = cnt
        tot += cnt
    return tot


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
        bcnt = np.empty((n + BLK - 1) // BLK, dtype=np.int64)
        m = _series_select(z, a, A0, -kk * u2, z_c, one_k, out, bcnt)
    else:
        out[:] = 1.0
        m = _count_rest(z, z_c, one_k)
        bcnt = np.ones((n + BLK - 1) // BLK, dtype=np.int64)
    if m == 0:
        return out
    # pass 2: collect the samples that need the exact solution
    zb = np.empty(m)
    idx = np.empty(m, dtype=np.int64)
    j = 0
    for b in range(bcnt.shape[0]):
        if bcnt[b] == 0:
            continue
        for i in range(b * BLK, min(b * BLK + BLK, n)):
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
def _quadratic_flux_cut(z, k, u1, u2, tol, A_half, Gt, qg):
    """_quadratic_flux with the cut computed in the same compiled call."""
    return _quadratic_flux(z, k, u1, u2, _quadratic_cut(k, u1, u2, tol, Gt, qg), A_half)


@njit(cache=True)
def _general_flux(z, k, gams, cs, z_c, tables, gu, gw, ju, jw, has, ltabs, llens, cgams, ccs, c01, mode, ltol,
                  TW, Ttt, Tb, BWr, BWi, Bttr, Btti, Bbr, Bbi, band, gam_in, CW, Ctt, Cb):
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
    uq = np.any(aq != 0.0) or np.any(ah != 0.0) or np.any(a3q != 0.0)
    fused = (n_other == 0) or (n_other == 1 and not uq)
    if fused:
        # interior series in one fused pass (vectorised, blocks off the star skipped)
        bcnt = np.empty((n + _series.BLK - 1) // _series.BLK, dtype=np.int64)
        zcut = z_c if z_c >= 0.0 else -1.0
        if n_other == 1:
            ml = _series.select_pow(z, aq, ah, a3q, ag, g_other, A0, A1, zcut, one_k, out, bcnt)
        else:
            ml = _series.select_quarter(z, aq, ah, a3q, ag, g_other, A0, A1, zcut, one_k, out, bcnt)
        ms = 0
        zb = np.empty(0)
        idx = np.empty(0, dtype=np.int64)
        zl = np.empty(ml)
        il = np.empty(ml, dtype=np.int64)
        jl = 0
        for bb in range(bcnt.shape[0]):
            if bcnt[bb] == 0:
                continue
            for i in range(bb * _series.BLK, min(bb * _series.BLK + _series.BLK, n)):
                zi = abs(z[i])
                if (zi > z_c) & (zi < one_k):
                    if k >= 1.0 and zi <= k - 1.0:
                        out[i] = 0.0
                    else:
                        zl[jl] = zi
                        il[jl] = i
                        jl += 1
    else:
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
    elif mode == 3 and k < _trap.P_MAX and ml > 0:
        # sinh-trapezoid rules (_trap): crossing samples by the real rule, the band
        # z_c < z < 1 - k by the complex-contour rule; exponents 0 and 1 in closed form
        _trap_limb(zl[:ml], k, cgams, ccs, c01, TW, Ttt, Tb, BWr, BWi, Bttr, Btti, Bbr, Bbi, band, gam_in, bl)
    elif mode == 4 and k < _interp.P_MAX and ml > 0:
        # Chebyshev pieces built from the same rules (connection formula), then the
        # direct rules for whatever they do not cover
        lam_lo = 0.0
        if band and z_c > k:
            lam_lo = ((z_c - 1.0) + k) * (z_c + k + 1.0) / (4.0 * z_c * k)
            # keep the inside piece well clear of the singularity of A, B at z = p
            # (lam = 1 - 1/(4 k^2)); samples below it use the direct rules
            lam_lo = min(max(lam_lo, _interp.LAM_MIN, 0.3 * (1.0 - 0.25 / (k * k))), 0.0)
        EO, AC, BC, AN, BN = _interp.build(k, cgams, lam_lo, TW, Ttt, Tb, BWr, BWi, Bttr, Btti, Bbr, Bbi,
                                           gam_in, Bbr[Bttr.shape[0]], CW, Ctt, Cb)
        _interp.evaluate(zl[:ml], ml, k, cgams, ccs, lam_lo, EO, AC, BC, AN, BN, bl)
        nrest = 0
        for j in range(ml):
            if np.isnan(bl[j]):
                nrest += 1
            elif c01[0] != 0.0 or c01[1] != 0.0:
                P0, P1 = _contact.p01(zl[j], k)
                bl[j] += c01[0] * P0 + c01[1] * P1
        if nrest > 0:
            zr = np.empty(nrest)
            ir = np.empty(nrest, dtype=np.int64)
            q = 0
            for j in range(ml):
                if np.isnan(bl[j]):
                    zr[q] = zl[j]
                    ir[q] = j
                    q += 1
            br_ = np.full(nrest, np.nan)
            _trap_limb(zr, k, cgams, ccs, c01, TW, Ttt, Tb, BWr, BWi, Bttr, Btti, Bbr, Bbi, band, gam_in, br_)
            for q in range(nrest):
                bl[ir[q]] = br_[q]
    for j in range(ml):
        b = bl[j]
        if np.isnan(b):
            b = _general.blocked_limb(zl[j], k, gams, cs, gu, gw, ju, jw, has)
        if mode != 2 or np.isnan(bl[j]):
            cert[il[j]] = False
        out[il[j]] = 1.0 - b / total
    if ms > 0 and not fused:
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


@njit(cache=True)
def _trap_limb(zl, k, cgams, ccs, c01, TW, Ttt, Tb, BWr, BWi, Bttr, Btti, Bbr, Bbi, band, gam_in, out):
    """Blocked flux on the limb by the _trap rules; leaves NaN where they do not apply."""
    m = zl.shape[0]
    one_m = 1.0 - k
    nc = 0
    nb = 0
    for j in range(m):
        if zl[j] >= one_m and zl[j] > k:
            nc += 1
        elif band and zl[j] > k:
            nb += 1
    zc_ = np.empty(nc)
    ic = np.empty(nc, dtype=np.int64)
    zb_ = np.empty(nb)
    ib = np.empty(nb, dtype=np.int64)
    a = 0
    b = 0
    for j in range(m):
        zj = zl[j]
        if zj >= one_m and zj > k:
            zc_[a] = zj
            ic[a] = j
            a += 1
        elif band and zj > k:
            zb_[b] = zj
            ib[b] = j
            b += 1
    nt = cgams.shape[0]
    if nc > 0:
        oc = np.zeros(nc)
        if nt > 0:
            _trap.blocked_cross(zc_, nc, k, cgams, ccs, TW, Ttt, Tb, oc)
        for q in range(nc):
            r = oc[q]
            if c01[0] != 0.0 or c01[1] != 0.0:
                P0, P1 = _contact.p01(zc_[q], k)
                r += c01[0] * P0 + c01[1] * P1
            out[ic[q]] = r
    if nb > 0:
        ob = np.zeros(nb)
        if nt > 0:
            _trap.blocked_band(zb_, nb, k, cgams, ccs, BWr, BWi, Bttr, Btti, Bbr, Bbi, gam_in, ob)
        for q in range(nb):
            r = ob[q]
            if c01[0] != 0.0 or c01[1] != 0.0:
                P0, P1 = _contact.p01(zb_[q], k)
                r += c01[0] * P0 + c01[1] * P1
            out[ib[q]] = r


@njit(cache=True)
def _cut_terms(k, gams, cs, tol, Gt):
    """z_c for I = sum c_i mu^(2 g_i) (one compiled call: normalisation and bound search)."""
    norm = 0.0
    for i in range(gams.shape[0]):
        norm += cs[i] / (2.0 * gams[i] + 2.0)
    if not (k > 0.0 and k < 1.0 and tol > 0.0 and norm != 0.0):
        return -1.0
    return _series.zcut(k, gams, np.abs(cs), k * k / (2.0 * abs(norm)), tol, Gt)


_QGAMS = np.array([0.0, 0.5, 1.0])
_SERIES_CACHE = {}


def series_tables(gams):
    """Coefficient tables A[j, q] of the interior series, one per exponent (cached)."""
    out = np.empty((len(gams), _series.M + 1, _series.M + 1))
    for i, g in enumerate(gams):
        key = float(g)
        t = _SERIES_CACHE.get(key)
        if t is None:
            if len(_SERIES_CACHE) > 1024:
                _SERIES_CACHE.clear()
            t = _series.table(key, _series.M)
            _SERIES_CACHE[key] = t
        out[i] = t
    return out


@njit(cache=True)
def _quadratic_cut(k, u1, u2, tol, Gt, qg):
    """z_c of the quadratic law I = 1 - u1 (1 - mu) - u2 (1 - mu)^2."""
    cs = np.empty(3)
    cs[0] = 1.0 - u1 - u2
    cs[1] = u1 + 2.0 * u2
    cs[2] = -u2
    return _cut_terms(k, qg, cs, tol, Gt)


def series_cut(rp, limb_dark, u, series_tol=SERIES_TOL):
    """Cut z_c: the interior series is used for z <= z_c, where its flux error
    is rigorously bounded by ``series_tol``. Returns -1 if the series is not
    used (e.g. large radius ratios or ``series_tol <= 0``)."""
    k = abs(float(rp))
    if limb_dark in QUADRATIC_FAMILY:
        u1, u2 = quadratic_coefficients(limb_dark, check_law(limb_dark, u))
        return _quadratic_cut(k, u1, u2, float(series_tol), _series._GTAB, _QGAMS)
    al, c = power_terms(limb_dark, u)
    return _cut_terms(k, al / 2.0, c, float(series_tol), _series._GTAB)


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


def limb_tables(gams, jmax=None, nc=_lens.NC):
    """Hypergeometric coefficient tables for the limb solution, j = 0..jmax, nc
    coefficients per series (cached per exponent; a cached table with at least as
    many terms is reused without copying when it has exactly the requested size)."""
    if jmax is None:
        jmax = _contact.JC
    ts, ls = [], []
    for g in gams:
        key = float(g)
        hit = _LIMB_CACHE.get(key)
        if hit is None or hit[0].shape[1] - 1 < jmax or hit[0].shape[3] < nc:
            if len(_LIMB_CACHE) > 256:
                _LIMB_CACHE.clear()
            hit = _lens.table(key, jmax, nc)
            _LIMB_CACHE[key] = hit
        t, l = hit
        if t.shape[1] - 1 != jmax or t.shape[3] != nc:
            t, l = t[:, :jmax + 1, :, :nc], l[:, :jmax + 1]
        ts.append(t)
        ls.append(l)
    if len(ts) == 1:
        return np.ascontiguousarray(ts[0][np.newaxis]), np.ascontiguousarray(ls[0][np.newaxis])
    return np.ascontiguousarray(ts), np.ascontiguousarray(ls)


_LIMB_CACHE = {}
_MODES = {"quadrature": 0, "contact": 1, "hypergeometric": 1, "proven": 2, "trapezoid": 3,
          "interpolate": 4, "fast": 5}
_PLANS = {}


@njit(cache=True)
def _count_limb(z, z_c, one_k):
    n = 0
    for i in range(z.shape[0]):
        a = abs(z[i])
        n += 1 if (a > z_c) & (a < one_k) else 0
    return n


def _plan(gams):
    """Everything that depends only on the exponents (cached): series tables,
    quadrature rules, trapezoid rules, masks (plain Python where the arrays are tiny)."""
    key = tuple(gams.tolist())
    pl = _PLANS.get(key)
    if pl is not None:
        return pl
    nt = len(key)
    intl = [g == 0.0 or g == 1.0 for g in key]
    integer = np.array(intl)
    nonint = [i for i in range(nt) if not intl[i]]
    ngams = np.array([key[i] for i in nonint])
    nq = _general.N_NODES
    pl = {"tables": series_tables(gams), "quad": None, "integer": integer, "nonint": np.array(nonint, dtype=np.int64),
          "qdummy": (np.zeros((nt, nq)), np.zeros((nt, nq)), np.zeros(nt, dtype=np.int64)),
          "i0": [i for i in range(nt) if key[i] == 0.0], "i1": [i for i in range(nt) if key[i] == 1.0],
          "ngams": ngams, "interp_ok": _interp.supported(key),
          "ltabs0": None, "llens0": None,
          "amax_all": 2.0 * max(key), "amax_n": 2.0 * max(ngams.tolist()) if nonint else 0.0,
          "rules_all": None, "rules_n": None, "rules_c": None, "band": {}}
    if len(_PLANS) > 512:
        _PLANS.clear()
    _PLANS[key] = pl
    return pl


def _c01(c, pl):
    cl = c.tolist()
    return np.array([sum(cl[i] for i in pl["i0"]), sum(cl[i] for i in pl["i1"])])


def _ltabs0(pl, nt):
    if pl["ltabs0"] is None:
        pl["ltabs0"] = np.zeros((nt, 2, 1, 4, _lens.NC))
        pl["llens0"] = np.zeros((nt, 2, 1, 4), dtype=np.int64)
    return pl["ltabs0"], pl["llens0"]


def _get(pl, name, gams):
    """Lazily computed parts of a plan."""
    r = pl[name]
    if r is None:
        if name == "quad":
            r = _general.quad_rules(gams)
        elif name == "rules_all":
            r = _trap.rules(gams)
        elif name == "rules_n":
            r = _trap.rules(pl["ngams"]) if pl["ngams"].size else _NO_CROSS
        else:
            r = _trap.rules(gams, _interp.H_CB)
        pl[name] = r
    return r


def _band_rules(pl, gams_sel, which, gam):
    key = (which, gam)
    r = pl["band"].get(key)
    if r is None:
        r = _trap.rules_in(gams_sel, gam)
        pl["band"][key] = r
    return r


def _limb_fast(z, k, gams, c, z_c, pl, mode, ltol):
    """Limb by the trapezoid rules (mode 3) or their Chebyshev interpolants (mode 4)."""
    if mode == 4:
        cgams, ccs, c01, TWr, which, amax = gams, c, _ZERO2, _get(pl, "rules_all", gams), "all", pl["amax_all"]
    else:
        cgams = pl["ngams"]
        ccs = c[pl["nonint"]]
        c01 = _c01(c, pl)
        TWr, which, amax = _get(pl, "rules_n", gams), "n", pl["amax_n"]
    BW, CW = _NO_BAND, _NO_CROSS
    band = False
    gam_in = _trap.GAMMA_IN
    if k < _trap.P_MAX:
        g = _trap.band_gamma(k, z_c if z_c > 0 else k, amax)
        if g > 0.0:
            band, gam_in = True, g
            if cgams.size > 0:
                BW = _band_rules(pl, cgams, which, g)
        if mode == 4:
            CW = _get(pl, "rules_c", gams)
    if cgams.size == 0:
        TWr = _NO_CROSS
    # the quadrature is needed where the rules do not apply: a band without a
    # contour, large occultors, z <= p
    if band and k < _trap.P_MAX:
        ju, jw, has = pl["qdummy"]
    else:
        ju, jw, has = _get(pl, "quad", gams)
    lt, ll = _ltabs0(pl, gams.size)
    return _general_flux(z, k, gams, c, z_c, pl["tables"], _general.GL_U, _general.GL_W, ju, jw, has,
                         lt, ll, cgams, ccs, c01, mode, ltol, *TWr, *BW, band, gam_in, *CW)


def _limb_tables(z, k, gams, c, z_c, pl, mode, ltol, n_limb):
    """Limb by quadrature (mode 0), contact expansions (1) or proven pieces (2)."""
    jl = limb_jmax(k, z_c) if k < 0.5 else _contact.JC
    integer = pl["integer"]
    c01 = _c01(c, pl)
    cgams, ccs = gams, c
    ltabs, llens = _ltabs0(pl, gams.size)
    if mode == 2 and k < 0.5:
        ltabs, llens = limb_tables(gams, jl)
    elif mode == 1 and k < 0.5 and n_limb >= CONTACT_BREAK_EVEN * jl:
        cgams, ccs = np.ascontiguousarray(gams[~integer]), np.ascontiguousarray(c[~integer])
        # the fixed-order expansions use the first NS + 1 coefficients of each series
        ltabs, llens = limb_tables(cgams, jl, _contact.NS + 1)
        if cgams.size == 0:
            ltabs = np.zeros((0, 2, jl + 1, 4, _contact.NS + 1))
            llens = np.zeros((0, 2, jl + 1, 4), dtype=np.int64)
    ju, jw, has = _get(pl, "quad", gams)
    return _general_flux(z, k, gams, c, z_c, pl["tables"], _general.GL_U, _general.GL_W, ju, jw, has,
                         ltabs, llens, cgams, ccs, c01, mode, ltol, *_NO_CROSS, *_NO_BAND, False,
                         _trap.GAMMA_IN, *_NO_CROSS)


_ZERO2 = np.zeros(2)
_NO_CROSS = (np.zeros((0, 1)), np.zeros(1), np.ones(2))
_NO_BAND = (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(1), np.zeros(1), np.ones(2), np.zeros(1))


def occult(z, rp, u, limb_dark="quadratic", series_tol=SERIES_TOL, limb="fast", return_certified=False):
    """Relative flux of a star occulted by an opaque disk.

    :param z: projected centre separations (stellar radii), array-like
    :param rp: radius ratio (planet / star)
    :param u: limb-darkening coefficients (batman conventions)
    :param limb_dark: "uniform", "linear", "quadratic", "squareroot", "nonlinear" or "power2"
    :param series_tol: bound on the flux error of the interior series; 0 disables it
    :param limb: method for the samples beyond z_c with non-quadratic laws:
        "fast" (default): for radius ratios below 0.3, the sinh-trapezoid
            rules of _trap (a fixed-node rule for the complete-integral form
            of the limb integral; no set-up), or, from INTERP_BREAK_EVEN limb
            samples on, their Chebyshev interpolants (_interp, built from ~70
            rule evaluations through the connection formula at second contact;
            ~40 us per light curve). Both agree with the quadrature to ~1e-15.
            Larger occultors use the contact expansions or quadrature as before.
        "trapezoid", "interpolate": force one of the two.
        "contact": power series about first and second contact, of
            fixed order, with coefficients computed once per light curve
            (error ~1e-15 by an a posteriori estimate, not a proof), when
            the light curve has enough limb samples to amortise the set-up
            (about 4.5 per j-term of the hypergeometric tables); quadrature
            otherwise. The exponents 0 and 1 are added in closed form.
        "proven": the same expansions, split into pieces whose degrees are
            chosen so that the flux error is proven to be at most
            ``series_tol`` (the guarantee of the interior series). Costs
            5-60 ms of set-up per light curve. Pieces whose bound cannot be
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
    if limb_dark in QUADRATIC_FAMILY:
        u1, u2 = quadratic_coefficients(limb_dark, check_law(limb_dark, u))
        f = _quadratic_flux_cut(z, k, u1, u2, float(series_tol), _A_HALF, _series._GTAB, _QGAMS)
        cert = np.ones(z.shape, dtype=bool) if return_certified else None
    else:
        al, c = power_terms(limb_dark, u)
        gams = al / 2.0
        z_c = _cut_terms(k, gams, c, float(series_tol), _series._GTAB)
        mode = _MODES.get(limb)
        if mode is None:
            raise ValueError('limb must be "fast", "proven", "interpolate", "trapezoid", "contact" or "quadrature"')
        pl = _plan(gams)
        ltol = float(series_tol) if series_tol > 0 else SERIES_TOL
        if mode == 2 and z_c < 0:
            mode = 1          # interior series disabled: nothing to match
        n_limb = _count_limb(z, z_c, 1.0 + k)
        if mode == 5 and (k >= _trap.P_MAX or
                          (z_c > k and _trap.band_gamma(k, z_c, pl["amax_all"]) < 0.0)):
            mode = 1          # large occultors, or no contour for the band: contact expansions or quadrature
        if mode == 5:
            # interpolants when enough limb samples amortise their set-up, direct rules otherwise
            mode = 4 if (k < _interp.P_MAX and n_limb >= INTERP_BREAK_EVEN and pl["interp_ok"]) else 3
        if mode == 4 and not (k < _interp.P_MAX and pl["interp_ok"]):
            mode = 3
        if mode >= 3:
            f, cert = _limb_fast(z, k, gams, c, z_c, pl, mode, ltol)
        else:
            f, cert = _limb_tables(z, k, gams, c, z_c, pl, mode, ltol, n_limb)
    if return_certified:
        return f.reshape(shape), cert.reshape(shape)
    return f.reshape(shape)
