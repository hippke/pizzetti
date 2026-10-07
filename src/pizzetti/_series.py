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


@njit(cache=True)
def bound(z, p, g, MM):
    """Rigorous bound B_M(z, p; g) on |S_M - D_g| (MM odd, MM + 1 > g).
    Written as e^(M+1) sum_l G (lam/2)^l (1-lam)^(M+1-l), lam = 2z/(2z+p), so
    that no power over- or underflows. Returns 0 for integer g <= MM."""
    m1 = MM + 1
    bm = abs(beta(g, m1))
    if bm == 0.0:
        return 0.0
    w = 1.0 - z * z
    e = (2.0 * z * p + p * p) / w
    if not (e < 1.0):
        return 1e300
    lam = 2.0 * z / (2.0 * z + p)
    s = 0.0
    for l in range(0, m1 + 1, 2):
        s += _G(m1, l) * (0.5 * lam) ** l * (1.0 - lam) ** (m1 - l)
    return w ** g * bm * e ** m1 * s / (1.0 - e)


@njit(cache=True)
def flux_bound(z, p, gams, cabs, scale):
    """Bound on the flux error of the truncated series: scale * sum_i |c_i| B_M(g_i)."""
    t = 0.0
    for i in range(gams.shape[0]):
        if cabs[i] != 0.0:
            t += cabs[i] * bound(z, p, gams[i], M)
    return scale * t


@njit(cache=True)
def zcut(p, gams, cabs, scale, tol):
    """Largest z in [0, 1 - p) with flux_bound(z) <= tol (bisection; the bound
    increases with z). Returns -1 if not even z = 0 qualifies."""
    if not (p > 0.0 and p < 1.0 and tol > 0.0):
        return -1.0
    if flux_bound(0.0, p, gams, cabs, scale) > tol:
        return -1.0
    lo = 0.0
    hi = 1.0 - p
    if flux_bound(hi * (1.0 - 1e-15), p, gams, cabs, scale) <= tol:
        return hi * (1.0 - 1e-15)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if flux_bound(mid, p, gams, cabs, scale) <= tol:
            lo = mid
        else:
            hi = mid
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


@njit(cache=True, fastmath=True, error_model="numpy")
def series_pow(zb, m, a, g, A0, A1, out):
    """out[j] = A0 + A1 z^2 - w^g P(v) for zb[:m] (one arbitrary exponent:
    power-2 law)."""
    for j in range(m):
        z2 = zb[j] * zb[j]
        w = 1.0 - z2
        v = 1.0 / w
        out[j] = A0 + A1 * z2 - np.exp(g * np.log(w)) * _h12(a, v)


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
