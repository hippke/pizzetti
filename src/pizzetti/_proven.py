"""Contact expansions with a proven flux-error bound (the 'proven' limb mode).

The limb region of a non-quadratic power law is split into pieces in the
hypergeometric argument (k for the limb crossing, kappa = 1/k inside the disk):

  A   first contact     k in [0, 1/4]:        P = k^m RA(k)
  MO  outer middle      k in [1/4, 3/4]:      P = sum_n a_n (k - k0)^n
  B   second contact    y = 1-k in [0, 1/4]:  P = R(y) + y^m [ln y SL(y) + SC(y)]
  C   second contact    y' = 1-kappa <= 1/4:  same form in y'
  MI  inner middle      kappa in [kappa_c, 3/4]: P = sum_n a_n (kappa - kappa0)^n

(m = g + 3/2; MI only if the interior series stops below kappa = 3/4.)
For each piece the error of the evaluated polynomial is bounded by

  (1) the j-remainder of the hypergeometric representation (_lens), which is
      sign-definite with non-increasing functions:
        |P - P_J| <= pre_max |p^2 - z^2|_max x_max^(J+1) F_(J+1),max / (1 - x_max);
  (2) the n-truncation of P_J: exact coefficients up to NB, and beyond NB
      Cauchy's estimate |f_n| <= M(rho) rho^-n on a circle |u - u0| = rho,
        tail <= sum_{N<n<=NB} |f_n| w^n + M(rho) (w/rho)^(NB+1) / (1 - w/rho),
      w the half-width (or umax) of the piece. M(rho) is bounded by the
      product of the maxima of the elementary factors on the circle (dense
      sampling plus a derivative bound) and majorants of the Gauss functions:
      H_j has positive Taylor coefficients, so |H_j(k)| <= H_j(|k|); Euler's
      integral gives |G_j(kappa)| <= 1 when |kappa| < 1 and |1 - kappa| <= 1;
      the connection-formula parts at second contact are summed with
      absolute values and a closed-form geometric tail.

The polynomial degree of each piece is the smallest that meets the flux
tolerance. All bounds hold in exact arithmetic; rounding is assessed
separately (it is at the level of 1e-16 of the flux).
"""
from math import lgamma, log, pi, sqrt

import numpy as np
from numba import njit

from . import _contact
from ._lens import _digamma

NB = 64          # exact coefficients up to this order
KS = 1024        # sampling points on a circle
KA = 0.25        # first-contact piece: k <= KA
YB = 0.25        # second-contact pieces: y, y' <= YB
NMO = 2          # outer middle pieces in k in [KA, 1 - YB]
WMI = 0.07       # largest half-width of inner middle pieces
WPAD = 1.0 + 1e-9   # piece half-widths are widened by this factor in the bounds
SAFETY = 1.0 + 1e-6  # margin for rounding in the evaluation of the bounds


# ======================================================== coefficient sequences
@njit(cache=True)
def _params(g, j, fam):
    if fam == 0:
        return 0.5, 0.5, g + 2.5 + j
    return -(g + 1.0 + j), 0.5, 1.0


@njit(cache=True)
def coef_seq(g, j, fam, slot, T, nmax):
    """Exact coefficients (to nmax) of the tabulated series (fam, j, slot),
    continued beyond the stored ones by their recurrences."""
    a, b, c = _params(g, j, fam)
    m = c - a - b
    mi = np.floor(m + 0.5)
    out = np.zeros(nmax + 1)
    t = T[fam, j, slot, 0]
    if t == 0.0:
        return out
    if slot == 0:
        for n in range(nmax + 1):
            out[n] = t
            t *= (n + a) * (n + b) / ((n + c) * (n + 1.0))
        return out
    if abs(m - mi) > 1e-9:
        if slot == 1:
            aa, bb, cc = a, b, 1.0 - m
        else:
            aa, bb, cc = c - a, c - b, 1.0 + m
        for n in range(nmax + 1):
            out[n] = t
            t *= (n + aa) * (n + bb) / ((n + cc) * (n + 1.0))
        return out
    mm = int(mi)
    if slot == 1:
        for n in range(min(mm, nmax + 1)):
            out[n] = t
            if n + 1 < mm:
                t *= (n + a) * (n + b) / ((n + 1.0) * (n + 1.0 - mi))
        return out
    # slots 2, 3 (log case): Bl_n = t_n, Bc_n = t_n psi_n
    tl = T[fam, j, 2, 0]
    for n in range(nmax + 1):
        if slot == 2:
            out[n] = tl
        else:
            psi = (-_digamma(n + 1.0) - _digamma(n + mi + 1.0)
                   + _digamma(a + n + mi) + _digamma(b + n + mi))
            out[n] = tl * psi
        tl *= (a + mi + n) * (b + mi + n) / ((n + 1.0) * (n + mi + 1.0))
    return out


@njit(cache=True)
def _shift(seq, s0, nb):
    """Taylor coefficients at s0 of sum_n seq_n s^n (s = s0 + t), orders <= nb."""
    out = np.zeros(nb + 1)
    for n in range(seq.shape[0]):
        cn = seq[n]
        if cn == 0.0:
            continue
        term = cn * s0 ** n                    # m = 0: C(n,0) s0^n
        for mo in range(min(n, nb) + 1):
            out[mo] += term
            if mo < n:
                term *= (n - mo) / ((mo + 1.0) * s0)
    return out


# ======================================================== majorant sums
@njit(cache=True)
def _rbar(n, a, b, c):
    """Upper bound of |t_{n'+1}/t_{n'}| for all n' >= n (Lemma B2); valid when
    n + a, n + b, n + c > 0."""
    return 1.0 + max(0.0, a + b - c - 1.0) / (n + c) + abs(a * b - c) / ((n + c) * (n + 1.0))


@njit(cache=True)
def _gauss_maj(t0, a, b, c, rho, n0, nfrom):
    """sum_{n >= nfrom} |t_n| rho^n, t_{n+1} = t_n (n+a)(n+b)/((n+c)(n+1)):
    exact terms until the ratio bound rbar(n) gives rbar rho < 1, then a
    geometric tail."""
    if t0 == 0.0:
        return 0.0
    s = 0.0
    t = abs(t0)
    w = 1.0
    n = 0
    nmin = max(n0, nfrom, int(abs(a) + abs(b) + abs(c)) + 4)
    while n < 400000:
        if n >= nfrom:
            s += t * w
        if n >= nmin and n + a > 0 and n + b > 0 and n + c > 0:
            sr = _rbar(n, a, b, c) * rho
            if sr < 0.999:
                return s + t * w * sr / (1.0 - sr)
        t *= abs((n + a) * (n + b) / ((n + c) * (n + 1.0)))
        w *= rho
        n += 1
        if t * w > 1e250:
            return np.inf
    return np.inf


@njit(cache=True)
def _phi(n, mi, a, b):
    """Non-increasing bound of |psi_n| (Lemma B3)."""
    t1 = abs(mi + a - 1.0) / (n + min(1.0, mi + a))
    t2 = abs(b - 1.0) / (n + mi + min(1.0, b))
    t3 = max(1.0 / (n + 1.0) + 1.0 / (n + mi + 1.0), 1.0 / (n + mi + a) + 1.0 / (n + mi + b))
    return t1 + t2 + t3


@njit(cache=True)
def table_maj(g, j, fam, slot, T, rho, nfrom):
    """sum_{n >= nfrom} |coef_n| rho^n of the series (fam, j, slot)."""
    a, b, c = _params(g, j, fam)
    m = c - a - b
    mi = np.floor(m + 0.5)
    t0 = T[fam, j, slot, 0]
    if slot == 0:
        return _gauss_maj(t0, a, b, c, rho, 0, nfrom)
    if abs(m - mi) > 1e-9:
        if slot == 1:
            return _gauss_maj(t0, a, b, 1.0 - m, rho, int(m) + 2, nfrom)
        if slot == 2:
            return 0.0
        return _gauss_maj(t0, c - a, c - b, 1.0 + m, rho, 0, nfrom)
    mm = int(mi)
    if slot == 1:
        s = 0.0
        t = t0
        w = 1.0
        for i in range(mm):
            if i >= nfrom:
                s += abs(t) * w
            if i + 1 < mm:
                t *= (i + a) * (i + b) / ((i + 1.0) * (i + 1.0 - mi))
                w *= rho
        return s
    tl0 = T[fam, j, 2, 0]
    aa, bb, cc = a + mi, b + mi, mi + 1.0
    if slot == 2:
        return _gauss_maj(tl0, aa, bb, cc, rho, 0, nfrom)
    # slot 3: |Bl_n psi_n|; explicit terms with the bound _psi_abs, tail with
    # the non-increasing bound _phi
    s = 0.0
    t = abs(tl0)
    w = 1.0
    n = 0
    nmin = max(nfrom, int(abs(aa) + abs(bb) + abs(cc)) + 4)
    while n < 400000:
        if n >= nfrom:
            s += t * w * _psi_abs(n, mi, a, b)
        if n >= nmin:
            sr = _rbar(n, aa, bb, cc) * rho
            if sr < 0.999:
                return s + t * w * _phi(n + 1.0, mi, a, b) * sr / (1.0 - sr)
        t *= abs((n + aa) * (n + bb) / ((n + cc) * (n + 1.0)))
        w *= rho
        n += 1
        if t * w > 1e250:
            return np.inf
    return np.inf


@njit(cache=True)
def _psi_abs(n, mi, a, b):
    x1 = n + 1.0
    x2 = n + mi + 1.0
    x3 = n + mi + a
    x4 = n + mi + b
    if x3 <= 0.0 or x4 <= 0.0:
        return 1e300
    up = -(log(x1) - 1.0 / x1) - (log(x2) - 1.0 / x2) + log(x3) + log(x4)
    lo = -log(x1) - log(x2) + (log(x3) - 1.0 / x3) + (log(x4) - 1.0 / x4)
    return max(abs(up), abs(lo))


# ======================================================== geometry
@njit(cache=True)
def z_of_k(p, k):
    a1 = 1.0 - 2.0 * k
    return p * a1 + sqrt(p * p * a1 * a1 + 1.0 - p * p)


@njit(cache=True)
def z_of_kappa(p, kap):
    a1 = 2.0 - kap
    return (sqrt(p * p * a1 * a1 + kap * kap * (1.0 - p * p)) - p * a1) / kap


@njit(cache=True)
def _elem(kind, p, u, s0, prev):
    """(z, x, sqrt-branch) at complex point; kind 0 A (u=k), 1 B (u=y),
    2 C (u=y'), 3 MO (k = s0 + u), 4 MI (kappa = s0 + u)."""
    if kind == 0 or kind == 1 or kind == 3:
        if kind == 0:
            k = u
        elif kind == 1:
            k = 1.0 - u
        else:
            k = s0 + u
        W = p * p * (1.0 - 2.0 * k) ** 2 + 1.0 - p * p
        sw = np.sqrt(W)
        z = p * (1.0 - 2.0 * k) + sw
        return z, 4.0 * p * z * k, sw
    if kind == 2:
        kap = 1.0 - u
    else:
        kap = s0 + u
    V = p * p * (2.0 - kap) ** 2 + kap * kap * (1.0 - p * p)
    sv = np.sqrt(V)          # principal branch: Re V > 0 on the disk (_crude)
    z = (sv - p * (2.0 - kap)) / kap
    return z, 4.0 * p * z / kap, sv


@njit(cache=True)
def _crude(kind, p, s0, r):
    """Crude upper bounds (|z|, |x|) on the closed disk |u| <= r, or -1 if the
    analyticity conditions (no zero of z, W or V, kappa or k) may fail."""
    if kind == 0 or kind == 1 or kind == 3:
        if kind == 0:
            kmax = r
            km1 = 1.0 + r
        elif kind == 1:
            kmax = 1.0 + r
            km1 = r
        else:
            if r >= s0:
                return -1.0, -1.0
            kmax = s0 + r
            km1 = 1.0 - s0 + r
        dW = 4.0 * p * p * kmax * km1                  # |W - 1| = 4 p^2 |k| |k - 1|
        if dW >= 1.0:
            return -1.0, -1.0
        a1 = 1.0 + 2.0 * kmax
        zlo = sqrt(1.0 - dW) - p * a1
        if zlo <= 0.0:
            return -1.0, -1.0
        zc = sqrt(1.0 + dW) + p * a1
        return zc, 4.0 * p * zc * kmax
    if kind == 2:
        kmin = 1.0 - r
        kmax = 1.0 + r
    else:
        kmin = s0 - r
        kmax = s0 + r
    if kmin <= 0.0:
        return -1.0, -1.0
    # V = kappa^2 - 4 p^2 kappa + 4 p^2 = (kappa - r1)(kappa - r2),
    # r1,2 = 2p^2 +- 2ip sqrt(1-p^2): |V| between (d - r)^2 and (d + r)^2,
    # d the distance of the centre from the roots
    cen = 1.0 if kind == 2 else s0
    d = sqrt((cen - 2.0 * p * p) ** 2 + 4.0 * p * p * (1.0 - p * p))
    if r >= d:
        return -1.0, -1.0
    # V = (kappa - 2p^2)^2 + h^2 with h^2 = 4p^2(1-p^2). Re V is harmonic, so its
    # minimum over the disk is on the circle kappa = cen + r e^{it}:
    # Re V = c^2 + h^2 - r^2 + 2 c r u + 2 r^2 u^2, u = cos t, c = cen - 2p^2.
    # Re V > 0 makes the principal square root the analytic branch.
    cc = cen - 2.0 * p * p
    u = min(1.0, max(-1.0, -cc / (2.0 * r)))
    remin = cc * cc + 4.0 * p * p * (1.0 - p * p) - r * r + 2.0 * cc * r * u + 2.0 * r * r * u * u
    if remin <= 0.0:
        return -1.0, -1.0
    vhi = (d + r) ** 2
    zc = (sqrt(vhi) + p * (2.0 + kmax)) / kmin
    return zc, 4.0 * p * zc / kmin


@njit(cache=True)
def elem_max(kind, p, s0, rho):
    """Rigorous (|z|, |x|) maxima on |u| = rho: K samples plus derivative bound
    from Cauchy's estimate on |u| = rho2 with the crude bounds."""
    rho2 = rho * 1.25
    if kind == 3 and rho2 >= s0:
        rho2 = 0.5 * (rho + s0)
    if kind == 4 and rho2 >= s0:
        rho2 = 0.5 * (rho + s0)
    zc, xc = _crude(kind, p, s0, rho2)
    if zc < 0.0:
        return -1.0, -1.0
    zm = 0.0
    xm = 0.0
    prev = 1.0 + 0.0j
    for i in range(KS + 1):
        th = 2.0 * pi * i / KS
        u = rho * (np.cos(th) + 1j * np.sin(th))
        z, x, sv = _elem(kind, p, u, s0, prev)
        prev = sv
        zm = max(zm, abs(z))
        xm = max(xm, abs(x))
    h = pi * rho / KS
    d = rho2 - rho
    return zm + zc / d * h, xm + xc / d * h


# ======================================================== bounds per piece
@njit(cache=True)
def _hval(g, j, T, k):
    """B_j H_j(k), 0 <= k <= 1 (k = 1: Gauss's sum)."""
    if k >= 1.0:
        c = g + 2.5 + j
        return np.exp(lgamma(0.5) + lgamma(g + 2.0 + j) - lgamma(c) + lgamma(c) + lgamma(c - 1.0)
                      - 2.0 * lgamma(c - 0.5))
    return table_maj(g, j, 0, 0, T, k, 0)


@njit(cache=True)
def jbound(kind, g, p, J, slo, shi, T):
    """|P - P_J| over the real range [slo, shi] of the piece argument
    (k for kinds 0, 1, 3; kappa for 2, 4)."""
    if kind == 0 or kind == 1 or kind == 3:
        zl = z_of_k(p, shi)                   # z decreases with k; x increases
        zh_ = z_of_k(p, slo)
        xmax = 1.0 - (zl - p) ** 2
        Q = zh_ * zh_ - p * p
        pre = xmax ** (g + 1.0) * sqrt(min(shi, 1.0)) / (2.0 * (g + 1.0))
        F = _hval(g, J + 1, T, shi)
    else:
        zl = z_of_kappa(p, slo)               # z increases with kappa
        zh_ = z_of_kappa(p, shi) if shi < 1.0 else 1.0 - p
        xmax = 1.0 - (zl - p) ** 2
        Q = zh_ * zh_ - p * p
        pre = xmax ** (g + 1.0) / (2.0 * (g + 1.0))
        F = pi
    if xmax >= 1.0:
        return np.inf
    return pre * abs(Q) * xmax ** (J + 1) * F / (1.0 - xmax)


@njit(cache=True)
def _nmax_mid(kind, s0, nb):
    """Number of terms of the series about 0 (kind 3: in k; kind 4: in y = 1 -
    kappa) that are re-centred at s0 for the middle pieces."""
    if kind == 3:
        return int((nb + 60) / (1.0 - s0)) + 40
    return int((nb + 60) / s0) + 40


@njit(cache=True)
def far_M(kind, g, p, s0, rho, J, T):
    """Upper bounds of |part| on |u| = rho. Kinds 0, 3, 4: one part (index 0);
    kinds 1, 2: (R, SL, SC). Kinds 3, 4: index 1 bounds |f - f~| on the circle,
    f~ the function whose Taylor coefficients are computed from the series
    about 0 truncated after _nmax_mid terms (Lemma B5)."""
    out = np.full(3, np.inf)
    Zm, Xm = elem_max(kind, p, s0, rho)
    if Zm < 0.0:
        return out
    Q = p * p + Zm * Zm
    if kind == 0:
        P = (4.0 * p * Zm) ** (g + 1.0) / (2.0 * (g + 1.0))
    elif kind == 1:
        P = Xm ** (g + 1.0) * sqrt(1.0 + rho) / (2.0 * (g + 1.0))
    elif kind == 3:
        P = Xm ** (g + 1.0) * sqrt(s0 + rho) / (2.0 * (g + 1.0))
    else:
        P = Xm ** (g + 1.0) / (2.0 * (g + 1.0))
    if kind == 4 and (s0 + rho >= 1.0 or rho > s0):
        return out
    if kind == 3 and s0 + rho >= 1.0:
        return out
    s = np.zeros(3)
    xj = 1.0
    rj = 1.0
    nt = _nmax_mid(kind, s0, NB) + 1 if (kind == 3 or kind == 4) else 0
    if kind == 4:
        ry = 1.0 - s0 + rho                       # |y| <= ry < 1 on the circle
        lnmax = max(abs(log(1.0 - s0 - rho)), abs(log(ry))) + 0.5 * pi
    for j in range(J + 1):
        cj = 1.0 + Q if j == 0 else Q * xj
        if kind == 0:
            s[0] += cj * table_maj(g, j, 0, 0, T, rho, 0)
        elif kind == 3:
            s[0] += cj * table_maj(g, j, 0, 0, T, s0 + rho, 0)
            s[1] += cj * table_maj(g, j, 0, 0, T, s0 + rho, nt)
        elif kind == 4:
            s[0] += cj * pi
            mj = g + 1.5 + j
            s[1] += cj * (table_maj(g, j, 1, 1, T, ry, nt) + ry ** mj
                          * (lnmax * table_maj(g, j, 1, 2, T, ry, nt) + table_maj(g, j, 1, 3, T, ry, nt)))
        else:
            fam = 0 if kind == 1 else 1
            s[0] += cj * table_maj(g, j, fam, 1, T, rho, 0)
            s[1] += cj * rj * table_maj(g, j, fam, 2, T, rho, 0)
            s[2] += cj * rj * table_maj(g, j, fam, 3, T, rho, 0)
        xj *= Xm
        rj *= rho
    for f in range(3):
        out[f] = P * s[f]
    return out


@njit(cache=True)
def _tail(coef, w, N, Mfar, rho):
    nb = coef.shape[0] - 1
    s = 0.0
    t = w ** (N + 1)
    for n in range(N + 1, nb + 1):
        s += abs(coef[n]) * t
        t *= w
    r = w / rho
    return s + Mfar * r ** (nb + 1) / (1.0 - r)


@njit(cache=True)
def _lstar(m, umax):
    ue = np.exp(-1.0 / m)
    if umax <= ue:
        return umax ** m * abs(log(umax))
    return 1.0 / (np.e * m)


# ======================================================== middle-piece series
@njit(cache=True)
def mid_coeffs(kind, g, p, s0, T, J, nb):
    """Taylor coefficients in t of P_J at the centre s0 (kind 3: k = s0 + t,
    kind 4: kappa = s0 + t)."""
    n = nb + 1
    one = np.zeros(n)
    one[0] = 1.0
    t = np.zeros(n)
    t[1] = 1.0
    if kind == 3:
        kk = s0 * one + t
        a1 = one - 2.0 * kk
        z = p * a1 + _contact._pow(p * p * _contact._mulr(a1, a1) + (1.0 - p * p) * one, 0.5)
        x = 4.0 * p * _contact._mulr(z, kk)
        pre = -_contact._mulr(_contact._pow(x, g + 1.0), _contact._pow(kk, 0.5)) / (2.0 * (g + 1.0))
    else:
        kap = s0 * one + t
        a1 = 2.0 * one - kap
        V = p * p * _contact._mulr(a1, a1) + (1.0 - p * p) * _contact._mulr(kap, kap)
        z = _contact._mulr(_contact._pow(V, 0.5) - p * a1, _contact._inv(kap))
        x = 4.0 * p * _contact._mulr(z, _contact._inv(kap))
        pre = -_contact._pow(x, g + 1.0) / (2.0 * (g + 1.0))
    q = p * p * one - _contact._mulr(z, z)
    tot = np.zeros(n)
    xj = one.copy()
    for j in range(J + 1):
        if j >= 1:
            xj = _contact._mulr(xj, x)
            cj = _contact._mulr(q, xj)
        else:
            cj = one + q
        if kind == 3:
            Fj = _shift(coef_seq(g, j, 0, 0, T, _nmax_mid(3, s0, nb)), s0, nb)
        else:
            Fj = _g_shift(g, j, T, 1.0 - s0, nb)
        tot += _contact._mulr(cj, Fj)
    return _contact._mulr(pre, tot)


@njit(cache=True)
def _g_shift(g, j, T, y0, nb):
    """Taylor coefficients in t of pi G_j(kappa0 + t), y = y0 - t, from the
    connection form A(y) + y^m [ln y Bl(y) + Bc(y)] about y = 0."""
    a, b, c = _params(g, j, 1)
    m = c - a - b
    n = nb + 1
    nmax = _nmax_mid(4, 1.0 - y0, nb)
    A = _shift(coef_seq(g, j, 1, 1, T, nmax), y0, nb)
    Bl = _shift(coef_seq(g, j, 1, 2, T, nmax), y0, nb)
    Bc = _shift(coef_seq(g, j, 1, 3, T, nmax), y0, nb)
    # in s = y - y0: y^m = y0^m (1 + s/y0)^m, ln y = ln y0 + ln(1 + s/y0)
    ym = np.zeros(n)
    lg = np.zeros(n)
    ym[0] = y0 ** m
    lg[0] = log(y0)
    for k in range(1, n):
        ym[k] = ym[k - 1] * (m - k + 1.0) / (k * y0)
        lg[k] = ((-1.0) ** (k + 1)) / (k * y0 ** k)
    S = A + _contact._mulr(ym, _contact._mulr(lg, Bl) + Bc)
    # s = -t
    for k in range(1, n, 2):
        S[k] = -S[k]
    return S


# ======================================================== per light curve
@njit(cache=True)
def _best_far(kind, g, p, s0, w, J, T, rmax):
    """Minimise the far-tail term over rho in (w, rmax)."""
    best = np.inf
    bestM = np.full(3, np.inf)
    bestrho = 0.0
    for i in range(1, 9):
        rho = w + (rmax - w) * i / 9.0
        M = far_M(kind, g, p, s0, rho, J, T)
        r = w / rho
        if kind == 1 or kind == 2:
            val = (M[0] + M[1] + M[2]) * r ** (NB + 1) / (1.0 - r)
        elif kind == 3 or kind == 4:
            val = (M[0] * r ** (NB + 1) + M[1]) / (1.0 - r)
        else:
            val = M[0] * r ** (NB + 1) / (1.0 - r)
        if val < best:
            best = val
            bestM = M
            bestrho = rho
    return bestM, bestrho


@njit(cache=True)
def _do_piece(kd, s0, lo_, hi_, p, gams, cs, tabs, lens, tau_abs, store_pc):
    """Coefficients and proven bound for one piece. Fills store_pc and returns
    (degree, flux bound in absolute units; inf if the bound is not met)."""
    nt = gams.shape[0]
    jerr = 0.0
    rhos = np.zeros(nt)
    Ms = np.zeros((nt, 3))
    store_pc[:] = 0.0
    for i in range(nt):
        g = gams[i]
        T = tabs[i]
        J = 1
        jmax = T.shape[1] - 2
        while J < jmax and abs(cs[i]) * jbound(kd, g, p, J, lo_, hi_, T) > 0.25 * tau_abs / nt:
            J += 1
        jb = abs(cs[i]) * jbound(kd, g, p, J, lo_, hi_, T)
        if not jb <= 0.25 * tau_abs:
            return NB, np.inf
        jerr += jb
        if kd == 3 or kd == 4:
            co = mid_coeffs(kd, g, p, s0, T, J, NB)
            store_pc[0, 0] += cs[i] * co
            w = hi_ - s0
            rmax = 0.95 * min(s0, 1.0 - s0)
            M, rho = _best_far(kd, g, p, s0, w, J, T, rmax)
        else:
            Jf = np.array([J, J, J])
            RA, Bp, Cp, e = _contact.pieces(g, p, T, lens[i], NB, 0.5, Jf)
            if kd == 0:
                store_pc[1 + i, 0] = cs[i] * RA
                w = hi_
            elif kd == 1:
                store_pc[0, 0] += cs[i] * Bp[0]
                store_pc[1 + i, 0] = cs[i] * Bp[1]
                store_pc[1 + i, 1] = cs[i] * Bp[2]
                w = 1.0 - lo_
            else:
                store_pc[0, 0] += cs[i] * Cp[0]
                store_pc[1 + i, 0] = cs[i] * Cp[1]
                store_pc[1 + i, 1] = cs[i] * Cp[2]
                w = 1.0 - lo_
            M, rho = _best_far(kd, g, p, 0.0, w, J, T, 0.95)
        rhos[i] = rho
        Ms[i] = M
    if kd == 3 or kd == 4:
        w = hi_ - s0
    elif kd == 0:
        w = hi_
    else:
        w = 1.0 - lo_
    w *= WPAD          # cover samples assigned to the piece by rounded z-limits
    for N in range(2, NB + 1):
        err = jerr
        if kd == 3 or kd == 4:
            rmin = 1.0
            Fs = 0.0
            for i in range(nt):
                rmin = min(rmin, rhos[i])
                Fs += abs(cs[i]) * Ms[i, 0]
            if rmin <= w:
                return NB, np.inf
            err += _tail(store_pc[0, 0], w, N, Fs, rmin)
            for i in range(nt):
                err += abs(cs[i]) * Ms[i, 1] / (1.0 - w / rhos[i])
        elif kd == 0:
            for i in range(nt):
                if rhos[i] <= w:
                    return NB, np.inf
                err += w ** (gams[i] + 1.5) * _tail(store_pc[1 + i, 0], w, N, abs(cs[i]) * Ms[i, 0], rhos[i])
        else:
            rmin = 1.0
            Fs = 0.0
            for i in range(nt):
                if rhos[i] <= w:
                    return NB, np.inf
                rmin = min(rmin, rhos[i])
                Fs += abs(cs[i]) * Ms[i, 0]
                mm = gams[i] + 1.5
                err += _lstar(mm, w) * _tail(store_pc[1 + i, 0], w, N, abs(cs[i]) * Ms[i, 1], rhos[i])
                err += w ** mm * _tail(store_pc[1 + i, 1], w, N, abs(cs[i]) * Ms[i, 2], rhos[i])
            err += _tail(store_pc[0, 0], w, N, Fs, rmin)
        err *= SAFETY
        if err <= tau_abs:
            return N, err
    return NB, np.inf


@njit(cache=True)
def build(p, gams, cs, tabs, lens, z_lo, total, tau):
    """Pieces with a proven flux bound <= tau each (flux units).
    Returns zb (npc, 2) z-ranges, kinds, centres s0, polynomial store
    (npc, nt+1, 3, NB+1), degrees, ok, bounds (flux units)."""
    nt = gams.shape[0]
    tau_abs = tau * total
    zl_series = max(z_lo, p + 1e-12)
    kap_c = 4.0 * zl_series * p / ((1.0 - (zl_series - p)) * (1.0 + (zl_series - p)))
    kap_c = min(kap_c, 1.0)
    MAXP = 24
    kinds = np.zeros(MAXP, dtype=np.int64)
    s0s = np.zeros(MAXP)
    lo = np.zeros(MAXP)
    hi = np.zeros(MAXP)
    degs = np.zeros(MAXP, dtype=np.int64)
    bnds = np.zeros(MAXP)
    store = np.zeros((MAXP, nt + 1, 3, NB + 1))
    ok = True
    npc = 0
    # A
    kinds[npc] = 0; lo[npc] = 0.0; hi[npc] = KA
    degs[npc], bnds[npc] = _do_piece(0, 0.0, 0.0, KA, p, gams, cs, tabs, lens, tau_abs, store[npc])
    npc += 1
    # outer middle
    wmo = (1.0 - YB - KA) / (2.0 * NMO)
    for i in range(NMO):
        kinds[npc] = 3
        s0s[npc] = KA + wmo * (2 * i + 1)
        lo[npc] = s0s[npc] - wmo
        hi[npc] = s0s[npc] + wmo
        degs[npc], bnds[npc] = _do_piece(3, s0s[npc], lo[npc], hi[npc], p, gams, cs, tabs, lens, tau_abs, store[npc])
        npc += 1
    # B
    kinds[npc] = 1; lo[npc] = 1.0 - YB; hi[npc] = 1.0
    degs[npc], bnds[npc] = _do_piece(1, 0.0, 1.0 - YB, 1.0, p, gams, cs, tabs, lens, tau_abs, store[npc])
    npc += 1
    # C: largest width (<= YB, <= 1 - kappa_c) whose bound holds
    yc = min(YB, 1.0 - kap_c)
    while True:
        dg, bd = _do_piece(2, 0.0, 1.0 - yc, 1.0, p, gams, cs, tabs, lens, tau_abs, store[npc])
        if bd < np.inf or yc < 0.01:
            break
        yc *= 0.6
    kinds[npc] = 2; lo[npc] = 1.0 - yc; hi[npc] = 1.0
    degs[npc], bnds[npc] = dg, bd
    npc += 1
    # inner middle pieces down to kappa_c, half-width <= WMI and <= 0.45 (1 - kappa0)
    top = 1.0 - yc
    while top > kap_c + 1e-12 and npc < MAXP:
        w = min(WMI, 0.5 * (top - kap_c))
        # keep the circle away from kappa = 1: w <= 0.45 (1 - s0) with s0 = top - w
        w = min(w, 0.45 * (1.0 - top) / 1.45)
        if w < 1e-4:
            w = min(0.5 * (top - kap_c), 1e-4)
        s0 = top - w
        kinds[npc] = 4; s0s[npc] = s0; lo[npc] = s0 - w; hi[npc] = top
        degs[npc], bnds[npc] = _do_piece(4, s0, s0 - w, top, p, gams, cs, tabs, lens, tau_abs, store[npc])
        top = s0 - w
        npc += 1
    for i in range(npc):
        if not bnds[i] < np.inf:
            ok = False
    zb = np.zeros((npc, 2))
    for i in range(npc):
        if kinds[i] == 0 or kinds[i] == 1 or kinds[i] == 3:
            zb[i, 0] = z_of_k(p, hi[i])
            zb[i, 1] = z_of_k(p, lo[i])
        else:
            zb[i, 0] = z_of_kappa(p, lo[i])
            zb[i, 1] = z_of_kappa(p, hi[i]) if hi[i] < 1.0 else 1.0 - p
    zb[0, 1] = 1.0 + p
    return zb, kinds[:npc].copy(), s0s[:npc].copy(), store[:npc].copy(), degs[:npc].copy(), ok, bnds[:npc] / total


@njit(cache=True, fastmath=True, error_model="numpy")
def _horner(c, N, x):
    r = 0.0
    for i in range(N, -1, -1):
        r = r * x + c[i]
    return r


@njit(cache=True, fastmath=True, error_model="numpy")
def evaluate(z, p, gams, zb, kinds, s0s, store, degs, out):
    """Blocked flux from the proven pieces; NaN outside them."""
    nt = gams.shape[0]
    npc = kinds.shape[0]
    for jz in range(z.shape[0]):
        zj = z[jz]
        r = np.nan
        for pc in range(npc):
            if zb[pc, 0] <= zj <= zb[pc, 1]:
                kd = kinds[pc]
                N = degs[pc]
                zmp = zj - p
                zpp = zj + p
                if kd == 0:
                    k = (1.0 - zmp) * (1.0 + zmp) / (4.0 * zj * p)
                    r = 0.0
                    if k > 0.0:
                        lk = np.log(k)
                        for i in range(nt):
                            r += np.exp((gams[i] + 1.5) * lk) * _horner(store[pc, 1 + i, 0], N, k)
                elif kd == 3:
                    k = (1.0 - zmp) * (1.0 + zmp) / (4.0 * zj * p)
                    r = _horner(store[pc, 0, 0], N, k - s0s[pc])
                elif kd == 4:
                    kap = 4.0 * zj * p / ((1.0 - zmp) * (1.0 + zmp))
                    r = _horner(store[pc, 0, 0], N, kap - s0s[pc])
                else:
                    if kd == 1:
                        y = (zpp - 1.0) * (zpp + 1.0) / (4.0 * zj * p)
                    else:
                        y = (1.0 - zpp) * (1.0 + zpp) / ((1.0 - zmp) * (1.0 + zmp))
                    r = _horner(store[pc, 0, 0], N, y)
                    if y > 0.0:
                        ly = np.log(y)
                        for i in range(nt):
                            r += np.exp((gams[i] + 1.5) * ly) * (ly * _horner(store[pc, 1 + i, 0], N, y)
                                                                 + _horner(store[pc, 1 + i, 1], N, y))
                break
        out[jz] = r
