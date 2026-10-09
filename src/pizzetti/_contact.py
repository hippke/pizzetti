"""Contact expansions: the flux near and across the stellar limb as power series
in the hypergeometric argument, with coefficients that depend only on the
radius ratio p (computed once per light curve).

For a term (1 - r^2)^g of the intensity let P_g(z) be the integral of
(1 - r^2)^g over the occulted part of the stellar disk, m = g + 3/2, and
    k = [1 - (z - p)^2] / (4 z p),  kappa = 1 / k.
k runs from 0 (first contact) to 1 (second contact); kappa from 1 (second
contact) towards 0 inside the disk. With y = 1 - k and y' = 1 - kappa,

  A  first contact,           0 <= k <= 1/2:   P_g = k^m RA(k)
  B  second contact, outside, 0 <= y <= 1/2:   P_g = RB(y) + y^m [ln(y) LB(y) + CB(y)]
  C  second contact, inside,  0 <= y' <= y'_c: P_g = RC(y') + y'^m [ln(y') LC(y') + CC(y')]

RA, RB, LB, CB, RC, LC, CC are power series; LB = LC = 0 unless g is a
half-integer. On each piece z is an algebraic function of the argument,
    z = p (1 - 2k) + sqrt(p^2 (1 - 2k)^2 + 1 - p^2),
    z = [sqrt(p^2 (2 - kappa)^2 + kappa^2 (1 - p^2)) - p (2 - kappa)] / kappa,
so the elementary factors of the hypergeometric representation (_lens) are
power series in the argument, and the universal functions enter through their
tabulated series unchanged. All coefficients follow by exact power-series
arithmetic (products, reciprocals, real powers).

Convergence: the nearest other singularity is the opposite contact (k = 1 for
A, y = 1 for B) or the occulter covering the centre (y' = 1 - 4p^2 for C). The
truncation error is estimated from the last terms and from the convergence of
the j-series; light curves whose estimate exceeds the tolerance fall back to
quadrature.
"""
import numpy as np
from numba import njit

from . import _lens

#: order of the contact series (polynomial degree)
NS = 32
#: largest j of the hypergeometric expansion used to build the series
JC = 200
#: accepted estimate of the relative truncation error (else: quadrature)
TOL = 1e-11


@njit(cache=True)
def _mul(a, b, out):
    n = a.shape[0]
    for i in range(n):
        s = 0.0
        for k in range(i + 1):
            s += a[k] * b[i - k]
        out[i] = s


@njit(cache=True)
def _mulr(a, b):
    out = np.empty(a.shape[0])
    _mul(a, b, out)
    return out


@njit(cache=True)
def _inv(a):
    n = a.shape[0]
    b = np.zeros(n)
    b[0] = 1.0 / a[0]
    for i in range(1, n):
        s = 0.0
        for k in range(1, i + 1):
            s += a[k] * b[i - k]
        b[i] = -s / a[0]
    return b


@njit(cache=True)
def _pow(a, al):
    """a^al for a[0] > 0 (J. C. P. Miller's recurrence)."""
    n = a.shape[0]
    f = np.zeros(n)
    f[0] = a[0] ** al
    for i in range(1, n):
        s = 0.0
        for k in range(1, i + 1):
            s += ((al + 1.0) * k - i) * a[k] * f[i - k]
        f[i] = s / (i * a[0])
    return f


@njit(cache=True)
def _log(a):
    n = a.shape[0]
    l = np.zeros(n)
    l[0] = np.log(a[0])
    for i in range(1, n):
        s = 0.0
        for k in range(1, i):
            s += k * l[k] * a[i - k]
        l[i] = (a[i] - s / i) / a[0]
    return l


@njit(cache=True)
def _powers(y, deg):
    """Y[d] = y^d as series, y[0] = 0."""
    n = y.shape[0]
    Y = np.zeros((deg + 1, n))
    Y[0, 0] = 1.0
    for d in range(1, deg + 1):
        _mul(Y[d - 1], y, Y[d])
    return Y


@njit(cache=True)
def _compose(co, L, Y):
    n = Y.shape[1]
    m = min(L, Y.shape[0])
    out = np.zeros(n)
    for d in range(m):
        c = co[d]
        if c != 0.0:
            for i in range(d, n):
                out[i] += c * Y[d, i]
    return out


@njit(cache=True)
def _size(c, umax):
    s = 0.0
    w = 1.0
    for i in range(c.shape[0]):
        s += abs(c[i]) * w
        w *= umax
    return s


@njit(cache=True)
def _shift(a, j, out):
    """out = u^j a (truncated)."""
    n = a.shape[0]
    for i in range(n):
        out[i] = a[i - j] if i >= j else 0.0


@njit(cache=True)
def _tail(c, umax):
    """Relative size of the last three terms at umax."""
    N = c.shape[0] - 1
    tot = _size(c, umax)
    if tot == 0.0:
        return 0.0
    tail = 0.0
    w = umax ** (N - 2)
    for i in range(N - 2, N + 1):
        tail += abs(c[i]) * w
        w *= umax
    return tail / tot


@njit(cache=True)
def pieces(g, p, T, L, N, ymax_c):
    """Series for one exponent g: RA (N+1,), B (3, N+1), C (3, N+1) (regular,
    log and power parts) and an estimate of the relative truncation error."""
    n = N + 1
    jm = min(T.shape[1] - 1, JC)
    one = np.zeros(n)
    one[0] = 1.0
    u = np.zeros(n)
    u[1] = 1.0
    B = np.zeros((3, n))
    C = np.zeros((3, n))
    est = 0.0
    m0 = g + 1.5
    t1 = np.empty(n)
    t2 = np.empty(n)
    for piece in range(2):
        if piece == 0:
            # u = y = 1 - k;  1 - 2k = 2u - 1
            a1 = 2.0 * u - one
            z = p * a1 + _pow(p * p * _mulr(a1, a1) + (1.0 - p * p) * one, 0.5)
            k = one - u
            x = 4.0 * p * _mulr(z, k)
            pre = -_mulr(_pow(x, g + 1.0), _pow(k, 0.5)) / (2.0 * (g + 1.0))
            fam = 0
            umax = 0.5
        else:
            # u = y' = 1 - kappa;  2 - kappa = 1 + u
            kap = one - u
            a1 = one + u
            z = _mulr(_pow(p * p * _mulr(a1, a1) + (1.0 - p * p) * _mulr(kap, kap), 0.5) - p * a1,
                      _inv(kap))
            x = 4.0 * p * _mulr(z, _inv(kap))
            pre = -_pow(x, g + 1.0) / (2.0 * (g + 1.0))
            fam = 1
            umax = ymax_c
        q = p * p * one - _mulr(z, z)
        R = np.zeros(n)
        SL = np.zeros(n)
        SC = np.zeros(n)
        xj = one.copy()
        cj = one + q
        small = 0
        contrib = 0.0
        scale = 1.0
        conv = False
        for j in range(jm + 1):
            if j >= 1:
                _mul(xj, x, t1)
                xj[:] = t1
                _mul(q, xj, cj)
            dR = _mulr(cj, T[fam, j, 1, :n])
            _shift(cj, j, t2)                      # u^j c_j
            dL = _mulr(t2, T[fam, j, 2, :n])
            dC = _mulr(t2, T[fam, j, 3, :n])
            R += dR
            SL += dL
            SC += dC
            if j >= 2:
                contrib = _size(dR, umax) + _size(dL, umax) + _size(dC, umax)
                scale = _size(R, umax) + _size(SL, umax) + _size(SC, umax)
                if contrib <= 1e-18 * scale:
                    small += 1
                    if small >= 3:
                        conv = True
                        break
                else:
                    small = 0
        if not conv:
            x0 = x[0]
            est = max(est, contrib / max(scale, 1e-300) * x0 / max(1.0 - x0, 1e-3))
        out = B if piece == 0 else C
        out[0] = _mulr(pre, R)
        out[1] = _mulr(pre, SL)
        out[2] = _mulr(pre, SC)
        for f in range(3):
            est = max(est, _tail(out[f], umax))
    # piece A, u = k: P = k^m RA(k), x = 4 z p k
    a1 = one - 2.0 * u
    z = p * a1 + _pow(p * p * _mulr(a1, a1) + (1.0 - p * p) * one, 0.5)
    q = p * p * one - _mulr(z, z)
    zg = _pow(z, g + 1.0)
    PA = -((4.0 * p) ** (g + 1.0)) * zg / (2.0 * (g + 1.0))
    xs = 4.0 * p * _mulr(z, u)                         # x = 4 p z k
    R = np.zeros(n)
    xj = one.copy()
    cj = np.empty(n)
    for j in range(min(jm, N) + 1):
        if j == 0:
            cj[:] = one + q
        else:
            _mul(q, xj, cj)
        R += _mulr(cj, T[0, j, 0, :n])
        _mul(xj, xs, t1)
        xj[:] = t1
    RA = _mulr(PA, R)
    est = max(est, _tail(RA, 0.5))
    return RA, B, C, est


@njit(cache=True)
def build(p, gams, cs, tabs, lens, z_lo, N):
    """Per-light-curve expansions for I = sum_i c_i (1 - r^2)^{g_i}.
    The regular parts of pieces B and C are summed over the exponents; the
    singular parts and piece A are kept per exponent.
    Returns bounds (3, 2), REG (2, N+1), SA (nt, N+1), SB (nt, 2, N+1),
    SC (nt, 2, N+1), m (nt), and the largest error estimate."""
    nt = gams.shape[0]
    n = N + 1
    bounds = np.empty((3, 2))
    zh = np.sqrt((1.0 - p) * (1.0 + p))   # k = 1/2
    bounds[0, 0] = zh
    bounds[0, 1] = 1.0 + p
    bounds[1, 0] = 1.0 - p
    bounds[1, 1] = zh
    # inner piece down to z_lo, but at most to y' = 1/2 (z = sqrt(1 + 8 p^2) - 3 p)
    zl = max(z_lo, np.sqrt(1.0 + 8.0 * p * p) - 3.0 * p, p + 1e-12)
    bounds[2, 0] = zl
    bounds[2, 1] = 1.0 - p
    xl = (1.0 - (zl - p)) * (1.0 + (zl - p))
    umax_c = max((1.0 - (zl + p)) * (1.0 + (zl + p)) / xl, 1e-3)   # y' at z_lo
    REG = np.zeros((2, n))
    SA = np.zeros((nt, n))
    SB = np.zeros((nt, 2, n))
    SCc = np.zeros((nt, 2, n))
    ms = np.empty(nt)
    est = 0.0
    for i in range(nt):
        g = gams[i]
        RA, Bp, Cp, e = pieces(g, p, tabs[i], lens[i], N, umax_c)
        if np.isnan(e):
            est = np.nan
        elif not np.isnan(est):
            est = max(est, e)
        ci = cs[i]
        SA[i] = ci * RA
        REG[0] += ci * Bp[0]
        REG[1] += ci * Cp[0]
        SB[i, 0] = ci * Bp[1]
        SB[i, 1] = ci * Bp[2]
        SCc[i, 0] = ci * Cp[1]
        SCc[i, 1] = ci * Cp[2]
        ms[i] = g + 1.5
    return bounds, REG, SA, SB, SCc, ms, est


@njit(cache=True, fastmath=True, error_model="numpy")
def _horner(c, x):
    r = 0.0
    for i in range(c.shape[0] - 1, -1, -1):
        r = r * x + c[i]
    return r


@njit(cache=True, fastmath=True, error_model="numpy")
def _pw(x, sx, qx, lx, e):
    """x^e from sqrt(x), x^(1/4) and ln x: quarter-integer e without exp."""
    e4 = 4.0 * e
    if e4 == np.floor(e4) and e4 < 24.0:
        n4 = int(e4)
        r = 1.0
        for _ in range(n4 // 4):
            r *= x
        rem = n4 % 4
        if rem == 1:
            r *= qx
        elif rem == 2:
            r *= sx
        elif rem == 3:
            r *= sx * qx
        return r
    return np.exp(e * lx)


@njit(cache=True, fastmath=True, error_model="numpy")
def evaluate(z, p, bounds, REG, SA, SB, SCc, ms, out):
    """out[j] = blocked flux sum_i c_i P_{g_i}(z[j]) for samples inside the
    three pieces; NaN elsewhere."""
    nt = ms.shape[0]
    for j in range(z.shape[0]):
        zj = z[j]
        r = np.nan
        if bounds[0, 0] <= zj < bounds[0, 1]:
            zmp = zj - p
            k = (1.0 - zmp) * (1.0 + zmp) / (4.0 * zj * p)
            r = 0.0
            if k > 0.0:
                lk = np.log(k)
                sk = np.sqrt(k)
                qk = np.sqrt(sk)
                for i in range(nt):
                    r += _pw(k, sk, qk, lk, ms[i]) * _horner(SA[i], k)
        elif bounds[1, 0] <= zj < bounds[1, 1]:
            zpp = zj + p
            y = (zpp - 1.0) * (zpp + 1.0) / (4.0 * zj * p)
            r = _horner(REG[0], y)
            if y > 0.0:
                ly = np.log(y)
                sy = np.sqrt(y)
                qy = np.sqrt(sy)
                for i in range(nt):
                    r += _pw(y, sy, qy, ly, ms[i]) * (ly * _horner(SB[i, 0], y) + _horner(SB[i, 1], y))
        elif bounds[2, 0] <= zj < bounds[2, 1] and bounds[2, 1] > bounds[2, 0]:
            zmp = zj - p
            zpp = zj + p
            y = (1.0 - zpp) * (1.0 + zpp) / ((1.0 - zmp) * (1.0 + zmp))
            r = _horner(REG[1], y)
            if y > 0.0:
                ly = np.log(y)
                sy = np.sqrt(y)
                qy = np.sqrt(sy)
                for i in range(nt):
                    r += _pw(y, sy, qy, ly, ms[i]) * (ly * _horner(SCc[i, 0], y) + _horner(SCc[i, 1], y))
        out[j] = r
