"""The three limb-darkening-independent occultation terms of the quadratic family.

For the quadratic law I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2 the relative flux is

    F = 1 - (c1 lex + c2 ldx + u2 edx) omega,
    c1 = 1 - u1 - 2 u2,  c2 = u1 + 2 u2,  omega = 1 / (1 - u1/3 - u2/6),

where lex (blocked area / pi), ldx (lambda_d + 2/3 Theta(k - z)) and edx (eta_d)
depend only on the geometry (z, k). ``occult_terms`` returns all three in one
pass, sharing the work that ``occult`` would repeat for every coefficient set:
the separation scan, block skipping, the interior series (only ldx needs it; lex
and edx are elementary there) and, near the limb, the elliptic integral (only
ldx needs it; lex and edx are closed forms).

Equivalently, for an intensity in the power basis I = a0 + a1 mu + a2 mu^2 the
blocked flux (in units of pi I-weighted area) is

    a0 lex + a1 ldx + a2 (lex - edx),

so any linear model in the intensity coefficients follows from these terms.

Accuracy: the exact branch is the same Mandel & Agol evaluation as ``occult``.
The interior series is used for z <= z_c with a rigorous bound on the ldx error
of ``series_tol / 4``; for physically plausible quadratic coefficients
(0 <= omega c2 <= 2.4) the flux error is then below ``series_tol``.
"""
from math import pi

import numpy as np
from numba import njit

from . import _exact, _series
from ._exact import ELL_BATCH_ITERATIONS, KC_MIN, Z_K_MIN, ell_combo_kc, limb_terms

BLK = 64
_A_HALF = _series.table(0.5, _series.M)
_GH = np.array([0.5])
_ONE = np.array([1.0])


@njit(cache=True)
def terms_cut(k, tol, Gt):
    """z_c such that the series error of ldx is <= tol / 4 (-1: no series)."""
    if not (k > 0.0 and k < 1.0 and tol > 0.0):
        return -1.0
    return _series.zcut(k, _GH, _ONE, k * k, 0.25 * tol, Gt)


@njit(cache=True, fastmath=True)
def exact_point_terms(z, k):
    """(lex, ldx, edx) at one on-star z (z != k, not fully occulted), scalar
    early-exit version (fallback for lanes the batch does not converge)."""
    INV_PI = 1 / pi
    k2 = k * k
    z2 = z * z
    s = z + k
    d = z - k
    x1 = d * d
    x2 = s * s
    x3 = k2 - z2
    sg = 1.0 if d > 0 else -1.0
    if z < 1 - k:
        lex = k2
        edx = k2 / 2 * (k2 + 2 * z2)
        num = ((1.0 - z) - k) * (1 + s)
        den = 1 - x1
        A = 1 - 5 * z2 + k2 + x3 * x3
        p = s / abs(d)
        Cp = 3 * sg
    else:
        lex, edx = limb_terms(z, k)
        num = ((z - 1.0) + k) * (s + 1)
        den = 4 * z * k
        A = (1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)
        p = 1 / abs(d)
        Cp = 3 * s * sg
    sden = np.sqrt(den)
    kc = max(np.sqrt(max(num, 0.0)) / sden, KC_MIN)
    ldx = 2 / 9 * INV_PI / sden * ell_combo_kc(kc, p, A, den * (z2 + 7 * k2 - 4), 3 * s / d, Cp)
    if z < k:
        ldx = ldx + 2 / 3
    return lex, ldx, edx


@njit(cache=True, fastmath=True, error_model="numpy")
def exact_batch_terms(zb, m, k, lex, ldx, edx, dev):
    """Vectorised (lex, ldx, edx) for zb[:m]; dev[j] > 0 marks lanes for the
    scalar fallback. Same arithmetic as _exact.exact_batch."""
    INV_PI = 1 / pi
    k2 = k * k
    for j in range(m):
        z = zb[j]
        z2 = z * z
        s = z + k
        d = z - k
        x1 = d * d
        x2 = s * s
        x3 = k2 - z2
        inner = z < 1 - k
        num = ((1.0 - z) - k) * (1 + s) if inner else ((z - 1.0) + k) * (s + 1)
        den = 1 - x1 if inner else 4 * z * k
        inv_sden = 1 / np.sqrt(den)
        kc = max(np.sqrt(max(num, 0.0)) * inv_sden, KC_MIN)
        sg = 1.0 if d > 0 else -1.0
        inv_ad = 1 / abs(d)
        p = s * inv_ad if inner else inv_ad
        a2 = 3 * s * sg * inv_ad
        b2 = 3 * sg if inner else 3 * s * sg
        A = (1 - 5 * z2 + k2 + x3 * x3) if inner else ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2))
        B = den * (z2 + 7 * k2 - 4)
        a1 = A + B
        b1 = A + B * kc * kc
        e = kc
        m0 = 1.0
        dd = 0.0
        for _ in range(ELL_BATCH_ITERATIONS):
            r = 1 / (m0 * p)
            f1 = a1
            a1 = a1 + b1 * (p * r)
            b1 = 2 * (b1 + f1 * kc)
            inv_p = m0 * r
            f2 = a2
            a2 = a2 + b2 * inv_p
            g = e * inv_p
            b2 = 2 * (b2 + f2 * g)
            p = g + p
            g = m0
            m0 = kc + m0
            dd = abs(g - kc) - 1e-8 * g
            kc = 2 * np.sqrt(e)
            e = kc * m0
        mp = m0 + p
        ell = 0.25 * pi * ((b1 + a1 * m0) * mp + 2 * m0 * (b2 + a2 * m0)) / (m0 * m0 * mp)
        ldx[j] = (2 / 9 * INV_PI) * inv_sden * ell + (2 / 3 if z < k else 0.0)
        lex_l, edx_l = limb_terms(z, k)
        lex[j] = k2 if inner else lex_l
        edx[j] = k2 / 2 * (k2 + 2 * z2) if inner else edx_l
        dev[j] = dd


@njit(cache=True, fastmath=True, error_model="numpy")
def _series_terms(z, a, k, z_c, one_k, T, bcnt):
    """Interior series pass in blocks of BLK (cf. occult._series_select):
    T[:, i] = (k^2, sqrt(w) P(v), k^2 (k^2 + 2 z^2) / 2) where z <= z_c, 0 elsewhere.
    bcnt[b] = number of samples of block b needing the exact solution."""
    a0, a1, a2, a3, a4, a5 = a[0], a[1], a[2], a[3], a[4], a[5]
    a6, a7, a8, a9, a10, a11 = a[6], a[7], a[8], a[9], a[10], a[11]
    k2 = k * k
    h = 0.5 * k2
    n = z.shape[0]
    tot = 0
    lexr, ldxr, edxr = T[0], T[1], T[2]
    for b in range((n + BLK - 1) // BLK):
        lo = b * BLK
        hi = min(lo + BLK, n)
        zr = z[lo:hi]
        on = 0
        for q in range(hi - lo):
            on += 1 if (abs(zr[q]) < one_k) else 0
        if on == 0:
            for q in range(lo, hi):
                lexr[q] = 0.0
                ldxr[q] = 0.0
                edxr[q] = 0.0
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
            ins = zi <= z_c
            lexr[lo + q] = k2 if ins else 0.0
            ldxr[lo + q] = np.sqrt(w) * P if ins else 0.0
            edxr[lo + q] = h * (k2 + 2.0 * z2) if ins else 0.0
            cnt += 1 if (zi > z_c) & (zi < one_k) else 0
        bcnt[b] = cnt
        tot += cnt
    return tot


@njit(cache=True)
def _occult_terms(z, k, z_c, A_half):
    n = z.shape[0]
    T = np.zeros((3, n))
    if k <= 0.0:
        return T
    one_k = 1.0 + k
    nb = (n + BLK - 1) // BLK
    bcnt = np.ones(nb, dtype=np.int64)
    if z_c >= 0.0:
        a = _series.coeffs(A_half, k, k * k)
        m = _series_terms(z, a, k, z_c, one_k, T, bcnt)
    else:
        m = 0
        for i in range(n):
            zi = abs(z[i])
            m += 1 if (zi > z_c) & (zi < one_k) else 0
    if m == 0:
        return T
    zb = np.empty(m)
    idx = np.empty(m, dtype=np.int64)
    j = 0
    for b in range(nb):
        if bcnt[b] == 0:
            continue
        for i in range(b * BLK, min(b * BLK + BLK, n)):
            zi = abs(z[i])
            if (zi > z_c) & (zi < one_k):
                if k >= 1.0 and zi <= k - 1.0:          # fully occulted: F = 0 for any u
                    T[0, i] = 1.0
                    T[1, i] = 2.0 / 3.0
                    T[2, i] = 0.5
                    continue
                if abs(zi - k) < Z_K_MIN:
                    zi = k + max(k * 2.220446049250313e-16, Z_K_MIN)
                zb[j] = zi
                idx[j] = i
                j += 1
    m = j
    if m > 0:
        lex = np.empty(m)
        ldx = np.empty(m)
        edx = np.empty(m)
        dev = np.empty(m)
        exact_batch_terms(zb, m, k, lex, ldx, edx, dev)
        for j in range(m):
            if not dev[j] <= 0.0:
                lex[j], ldx[j], edx[j] = exact_point_terms(zb[j], k)
            T[0, idx[j]] = lex[j]
            T[1, idx[j]] = ldx[j]
            T[2, idx[j]] = edx[j]
    return T


@njit(cache=True)
def _occult_terms_cut(z, k, tol, A_half, Gt):
    return _occult_terms(z, k, terms_cut(k, tol, Gt), A_half)


def occult_terms(z, rp, series_tol=1e-9):
    """The three LD-independent terms (lex, ldx, edx) in one pass.

    :param z: projected centre separations (stellar radii), array-like
    :param rp: radius ratio (planet / star)
    :param series_tol: flux-error bound of the interior series for physically
        plausible quadratic coefficients (ldx itself to series_tol / 4); 0 disables it
    :return: array of shape (3,) + z.shape: lex, ldx, edx. For quadratic
        coefficients u: F = 1 - (c1 lex + c2 ldx + u2 edx) omega
        (see ``quadratic_from_terms``).
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    shape = z.shape
    T = _occult_terms_cut(z.ravel(), abs(float(rp)), float(series_tol), _A_HALF, _series._GTAB)
    return T.reshape((3,) + shape)


def quadratic_from_terms(T, u):
    """Relative flux of the quadratic law from ``occult_terms`` output."""
    u1, u2 = float(u[0]), float(u[1])
    omega = 1.0 / (1.0 - u1 / 3.0 - u2 / 6.0)
    return 1.0 - ((1.0 - u1 - 2.0 * u2) * T[0] + (u1 + 2.0 * u2) * T[1] + u2 * T[2]) * omega
