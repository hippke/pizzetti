"""Interior series (body entirely inside the stellar disk, z + p <= 1).

For a term (1 - r^2)^g of the intensity, the mean over the occulting disk is
    D_g(z, p) = w^g * sum_{m>=0} beta_m <eps^m>,     w = 1 - z^2,
    beta_m = (-1)^m binom(g, m),  eps = (2 z rho cos(phi) + rho^2) / w,
    <eps^m> = v^m sum_{l even <= m} G(m, l) z^l p^(2m-l),   v = 1/w,
    G(m, l) = binom(m, l) binom(l, l/2) * 2 / (2m - l + 2)  >= 0.
Truncated after m = M it is w^g * sum_j a_j(p) v^j with
    a_j(p) = sum_q A[j, q] p^(2q).
All beta_m with m > g share one sign and |beta_m| does not increase, which
gives the rigorous truncation bound (M odd, M + 1 > g)
    |S_M - D_g| <= B_M = w^g |beta_{M+1}| <eps^(M+1)> / (1 - e),
    e = (2 z p + p^2) / w < 1,
increasing in z. See the accompanying paper for the proofs.
"""
from math import pi

import numpy as np
from numba import njit

#: Truncation order of the series (odd): polynomial of degree M in 1/(1 - z^2).
M = 11


@njit(cache=True)
def _comb(n, k):
    r = 1.0
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


@njit(cache=True)
def _G(m, l):
    return _comb(m, l) * _comb(l, l // 2) * 2.0 / (2 * m - l + 2)


@njit(cache=True)
def beta(g, m):
    """(-1)^m binom(g, m)."""
    b = 1.0
    for i in range(1, m + 1):
        b = b * (i - 1 - g) / i
    return b


@njit(cache=True)
def table(g, MM):
    """A[j, q] such that the truncation after m = MM is w^g sum_j v^j sum_q A[j, q] p^(2q)."""
    A = np.zeros((MM + 1, MM + 1))
    for m in range(MM + 1):
        bm = beta(g, m)
        if bm == 0.0:
            continue
        for l in range(0, m + 1, 2):
            c = bm * _G(m, l)
            q = m - l // 2
            h = l // 2
            for i in range(h + 1):
                A[m - i, q] += c * _comb(h, i) * (-1.0) ** i
    return A


@njit(cache=True)
def coeffs(A, p, scale):
    """scale * a_j(p), j = 0..M (Horner in p^2)."""
    n = A.shape[0]
    p2 = p * p
    a = np.empty(n)
    for j in range(n):
        s = 0.0
        for q in range(n - 1, -1, -1):
            s = s * p2 + A[j, q]
        a[j] = s * scale
    return a


def _gtable(MM):
    """G(MM + 1, l) for even l (index l // 2)."""
    m1 = MM + 1
    return np.array([_G(m1, l) for l in range(0, m1 + 1, 2)])


#: G(M + 1, l) of the default order, precomputed (the bound is evaluated many times
#: per light curve while the cut is searched)
_GTAB = _gtable(M)


@njit(cache=True)
def _bound_g(z, p, g, MM, Gt):
    """B_M with the coefficients G(M + 1, l) given (Gt[l // 2])."""
    m1 = MM + 1
    bm = abs(beta(g, m1))
    if bm == 0.0:
        return 0.0
    w = 1.0 - z * z
    e = (2.0 * z * p + p * p) / w
    if not (e < 1.0):
        return 1e300
    lam = 2.0 * z / (2.0 * z + p)
    h = 0.5 * lam
    q = 1.0 - lam
    # sum_l G (lam/2)^l (1-lam)^(M+1-l) by Horner in (h/q)^2, scaled by q^(M+1);
    # direct powers where q is tiny (z -> 0 is lam = 0, q = 1: no issue; q -> 0 only for p -> 0)
    s = 0.0
    if q > 1e-3:
        r = (h / q) ** 2
        for i in range(Gt.shape[0] - 1, -1, -1):
            s = s * r + Gt[i]
        s *= q ** m1
    else:
        for i in range(Gt.shape[0]):
            l = 2 * i
            s += Gt[i] * h ** l * q ** (m1 - l)
    return w ** g * bm * e ** m1 * s / (1.0 - e)


@njit(cache=True)
def bound(z, p, g, MM):
    """Rigorous bound B_M(z, p; g) on |S_M - D_g| (MM odd, MM + 1 > g).
    Written as e^(M+1) sum_l G (lam/2)^l (1-lam)^(M+1-l), lam = 2z/(2z+p), so
    that no power over- or underflows. Returns 0 for integer g <= MM."""
    m1 = MM + 1
    Gt = np.empty(m1 // 2 + 1)
    for l in range(0, m1 + 1, 2):
        Gt[l // 2] = _G(m1, l)
    return _bound_g(z, p, g, MM, Gt)


@njit(cache=True)
def flux_bound(z, p, gams, cabs, scale, Gt):
    """Bound on the flux error of the truncated series: scale * sum_i |c_i| B_M(g_i)."""
    t = 0.0
    for i in range(gams.shape[0]):
        if cabs[i] != 0.0:
            t += cabs[i] * _bound_g(z, p, gams[i], M, Gt)
    return scale * t


@njit(cache=True)
def zcut(p, gams, cabs, scale, tol, Gt):
    """Largest z in [0, 1 - p) with flux_bound(z) <= tol, to ~1e-13 in z. The bound
    increases with z and log(bound) is smooth, so the root of log(bound/tol) is
    bracketed and found by regula falsi (Illinois); the lower end of the bracket
    always satisfies the bound, and it is returned once the bound there is within
    1e-9 (relative) of the tolerance. About 10 evaluations of the bound instead of
    60 bisection steps. Returns -1 if not even z = 0 qualifies."""
    if not (p > 0.0 and p < 1.0 and tol > 0.0):
        return -1.0
    f0 = flux_bound(0.0, p, gams, cabs, scale, Gt)
    if f0 > tol:
        return -1.0
    lo = 0.0
    hi = (1.0 - p) * (1.0 - 1e-15)
    fh = flux_bound(hi, p, gams, cabs, scale, Gt)
    if fh <= tol:
        return hi
    ltol = np.log(tol)
    if f0 <= 0.0:
        # only integer exponents (exact series): the bound is 0 everywhere
        return hi
    flo = np.log(f0) - ltol            # < 0 (or 0)
    fhi = np.log(fh) - ltol if fh < 1e300 else 700.0
    side = 0
    for _ in range(200):
        # done when the bracket is tight, or when the bound at lo is within 1e-9
        # (relative) of the tolerance: lo is then the cut for all practical purposes
        if hi - lo <= 1e-13 * hi or flo >= -1e-9:
            break
        z = (lo * fhi - hi * flo) / (fhi - flo)
        d = hi - lo
        if not (lo + 1e-12 * d < z < hi - 1e-12 * d):
            z = 0.5 * (lo + hi)
        fb = flux_bound(z, p, gams, cabs, scale, Gt)
        fz = np.log(fb) - ltol if (fb > 0.0 and fb < 1e300) else (700.0 if fb >= 1e300 else -700.0)
        if fb <= tol:                      # the decision on the bound itself, not on its log
            lo, flo = z, min(fz, 0.0)
            if side == -1:
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = z, max(fz, 1e-300)
            if side == 1:
                flo *= 0.5
            side = 1
    return lo


@njit(cache=True, fastmath=True, error_model="numpy")
def series_quadratic(z, a, A0, A1, out):
    """out[i] = A0 + A1 z^2 - sqrt(w) * sum_j a_j v^j for all i (branch-free,
    vectorised; meaningful only where the caller selects z <= z_c).
    Unrolled Horner for M = 11."""
    a0, a1, a2, a3, a4, a5 = a[0], a[1], a[2], a[3], a[4], a[5]
    a6, a7, a8, a9, a10, a11 = a[6], a[7], a[8], a[9], a[10], a[11]
    for i in range(z.shape[0]):
        z2 = z[i] * z[i]
        w = max(1.0 - z2, 1e-12)
        v = 1.0 / w
        P = ((((((((((a11 * v + a10) * v + a9) * v + a8) * v + a7) * v + a6) * v + a5) * v
                 + a4) * v + a3) * v + a2) * v + a1) * v + a0
        out[i] = A0 + A1 * z2 - np.sqrt(w) * P


@njit(cache=True, fastmath=True, error_model="numpy", inline="always")
def _h12(a, v):
    """Horner of degree 11 (unrolled)."""
    return (((((((((((a[11] * v + a[10]) * v + a[9]) * v + a[8]) * v + a[7]) * v + a[6]) * v
                  + a[5]) * v + a[4]) * v + a[3]) * v + a[2]) * v + a[1]) * v + a[0])


@njit(cache=True, fastmath=True, error_model="numpy")
def series_quarter(zb, m, aq, ah, a3q, A0, A1, out):
    """out[j] = A0 + A1 z^2 - w^(1/4) P_q(v) - w^(1/2) P_h(v) - w^(3/4) P_3q(v)
    for zb[:m] (exponents 1/4, 1/2, 3/4: square-root and nonlinear laws)."""
    for j in range(m):
        z2 = zb[j] * zb[j]
        w = 1.0 - z2
        v = 1.0 / w
        s = np.sqrt(w)
        q = np.sqrt(s)
        out[j] = A0 + A1 * z2 - q * _h12(aq, v) - s * _h12(ah, v) - s * q * _h12(a3q, v)


_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.4426950408889634
_SHIFT = 6.755399441055744e15            # 1.5 * 2^52
_TWO52 = 4.503599627370496e15            # 2^52


@njit(cache=True, fastmath=False, inline="always")
def _vlog(x):
    """log(x) for normal x > 0, branch-free (vectorises); a few ulp."""
    b = np.float64(x).view(np.uint64)
    ebits = (b >> np.uint64(52)) & np.uint64(0x7ff)
    m = np.uint64((b & np.uint64(0x000fffffffffffff)) | np.uint64(0x3ff0000000000000)).view(np.float64)
    big = m > 1.4142135623730951
    m = 0.5 * m if big else m
    e = np.uint64(ebits | np.uint64(0x4330000000000000)).view(np.float64) - (_TWO52 + 1023.0)
    e = e + 1.0 if big else e
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    s = 2.0 + t2 * (2 / 3 + t2 * (2 / 5 + t2 * (2 / 7 + t2 * (2 / 9 + t2 * (2 / 11 + t2 * (
        2 / 13 + t2 * (2 / 15 + t2 * (2 / 17 + t2 * (2 / 19 + t2 * (2 / 21 + t2 * (2 / 23)))))))))))
    return e * _LN2_HI + (e * _LN2_LO + t * s)


@njit(cache=True, fastmath=False, inline="always")
def _vexp(y):
    """exp(y) for -700 < y <= 1, branch-free (vectorises); ~1 ulp. Safe under
    fastmath (inlined into fastmath loops): the integer part is taken by rint,
    not by the add-and-subtract trick (x + 1.5*2^52) - 1.5*2^52, which fastmath
    may simplify to x."""
    y = y if y > -700.0 else -700.0
    k = np.rint(y * _INV_LN2)
    r = (y - k * _LN2_HI) - k * _LN2_LO
    p = 1.0 + r * (1.0 + r * (1 / 2 + r * (1 / 6 + r * (1 / 24 + r * (1 / 120 + r * (1 / 720 + r * (
        1 / 5040 + r * (1 / 40320 + r * (1 / 362880 + r * (1 / 3628800 + r * (1 / 39916800 + r * (
            1 / 479001600 + r * (1 / 6227020800 + r * (1 / 87178291200))))))))))))))
    sc = np.int64((np.int64(k) + np.int64(1023)) << np.int64(52)).view(np.float64)
    return p * sc


@njit(cache=True, fastmath=True, error_model="numpy")
def series_pow(zb, m, a, g, A0, A1, out):
    """out[j] = A0 + A1 z^2 - w^g P(v) for zb[:m] (one arbitrary exponent:
    power-2 law). w^g by the branch-free log and exp above."""
    for j in range(m):
        z2 = zb[j] * zb[j]
        w = 1.0 - z2
        v = 1.0 / w
        out[j] = A0 + A1 * z2 - _vexp(g * _vlog(w)) * _h12(a, v)


@njit(cache=True, fastmath=True)
def _wpow(w, g):
    if g == 0.0:
        return 1.0
    if g == 0.5:
        return np.sqrt(w)
    if g == 1.0:
        return w
    if g == 0.25:
        return np.sqrt(np.sqrt(w))
    if g == 0.75:
        s = np.sqrt(w)
        return s * np.sqrt(s)
    return w ** g


@njit(cache=True, fastmath=True)
def series_general(zb, m, gams, T, out):
    """out[j] = 1 - sum_i w^{g_i} sum_l T[i, l] v^l for zb[:m] (T already
    scaled by c_i p^2 / (2 N))."""
    nt = gams.shape[0]
    nl = T.shape[1]
    for j in range(m):
        z2 = zb[j] * zb[j]
        w = 1.0 - z2
        v = 1.0 / w
        s = 0.0
        for i in range(nt):
            P = 0.0
            for l in range(nl - 1, -1, -1):
                P = P * v + T[i, l]
            s += _wpow(w, gams[i]) * P
        out[j] = 1.0 - s


#: block length of the fused pass
BLK = 64


@njit(cache=True, fastmath=True, error_model="numpy")
def select_quarter(z, aq, ah, a3q, ag, g, A0, A1, z_c, one_k, out, bcnt):
    """Fused, branch-free pass for the exponents {1/4, 1/2, 3/4} (as _series_select of
    occult for the quadratic law): out[i] = series flux where |z| <= z_c, 1 elsewhere;
    blocks without on-star samples are filled with 1. bcnt[b] = samples of block b with
    z_c < |z| < 1 + k; returns their total."""
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
            s = np.sqrt(w)
            qq = np.sqrt(s)
            f = A0 + A1 * z2 - qq * _h12(aq, v) - s * _h12(ah, v) - s * qq * _h12(a3q, v)
            fr[q] = f if zi <= z_c else 1.0
            cnt += 1 if (zi > z_c) & (zi < one_k) else 0
        bcnt[b] = cnt
        tot += cnt
    return tot


@njit(cache=True, fastmath=True, error_model="numpy")
def select_pow(z, aq, ah, a3q, ag, g, A0, A1, z_c, one_k, out, bcnt):
    """As select_quarter, for one arbitrary exponent g (power-2 law)."""
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
            f = A0 + A1 * z2 - _vexp(g * _vlog(w)) * _h12(ag, v)
            fr[q] = f if zi <= z_c else 1.0
            cnt += 1 if (zi > z_c) & (zi < one_k) else 0
        bcnt[b] = cnt
        tot += cnt
    return tot
