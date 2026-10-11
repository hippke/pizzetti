"""Flux for intensity laws I(mu) = sum_i c_i mu^(alpha_i) where the occultor
touches or crosses the stellar limb, by Green's theorem.

With the radial primitive F_i(r) = int_0^r (1 - s^2)^g s ds
= (1 - (1 - r^2)^(g+1)) / (2 (g + 1)), g = alpha_i / 2, the field
(F(r)/r^2) (x, y) has divergence (1 - r^2)^g and is smooth at r = 0. The
blocked flux is the outward flux through the boundary of the occulted part
of the stellar disk:
  * stellar-limb arc inside the occultor (r = 1): F(1) * 2 theta_s,
    cos theta_s = (1 + z^2 - p^2) / (2 z);
  * occultor arc inside the star, points (z + p cos t, p sin t),
    cos t < c = (1 - z^2 - p^2) / (2 z p):
    2 p int_{t0}^{pi} (F(r)/r^2) (p + z cos t) dt,  t0 = arccos(c).
The arc integral uses composite Gauss rules on segments that grow
geometrically away from the nearest singularity of (1 - r^2)^(g+1) (at
t = -t0, or at +-i sqrt(y_min/(z p)) for an interior disk). On the limb, the
first segment starts at the branch point t0, where the integrand is a regular
function plus (t - t0)^(g+1) times another regular function: the first part
is integrated by Gauss-Legendre, the second by Gauss-Jacobi with the weight
(t - t0)^(g+1), so every segment converges geometrically, for every exponent
(~1e-16 against the independent references in testdata/).
"""
from math import lgamma, pi

import numpy as np
from numba import njit

#: Gauss-Legendre (and Gauss-Jacobi) nodes per segment of the limb arc integral
#: (geometric segments towards the nearest singularity).
N_NODES = 12
_X, _W = np.polynomial.legendre.leggauss(N_NODES)
GL_U = 0.5 * (_X + 1.0)
GL_W = 0.5 * _W


@njit(cache=True)
def _jacobi_p(n, a, b, x):
    """P_n^(a,b)(x) and P_(n-1)^(a,b)(x) by the three-term recurrence."""
    if n == 0:
        return 1.0, 0.0
    p0 = 1.0
    p1 = 0.5 * ((a + b + 2.0) * x + (a - b))
    for k in range(1, n):
        c1 = 2.0 * (k + 1) * (k + a + b + 1) * (2 * k + a + b)
        c2 = (2 * k + a + b + 1) * ((2 * k + a + b + 2) * (2 * k + a + b) * x + a * a - b * b)
        c3 = 2.0 * (k + a) * (k + b) * (2 * k + a + b + 2)
        p0, p1 = p1, (c2 * p1 - c3 * p0) / c1
    return p1, p0


@njit(cache=True)
def gauss_jacobi01(n, beta):
    """Gauss-Jacobi rule on [0, 1] for the weight u^beta: int_0^1 u^beta f(u) du
    = sum w_i f(u_i) for polynomials f of degree < 2n. Golub-Welsch for the
    nodes, refined by Newton's method; weights from the Christoffel formula."""
    a = 0.0
    b = beta
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] = (b - a) / (a + b + 2.0) if k == 0 else (b * b - a * a) / ((2 * k + a + b) * (2 * k + a + b + 2))
    for k in range(1, n):
        e = np.sqrt(4.0 * k * (k + a) * (k + b) * (k + a + b)
                    / ((2 * k + a + b) ** 2 * (2 * k + a + b + 1) * (2 * k + a + b - 1)))
        A[k - 1, k] = e
        A[k, k - 1] = e
    x = np.linalg.eigvalsh(A)
    u = np.empty(n)
    w = np.empty(n)
    lc = lgamma(n + a + 1.0) + lgamma(n + b + 1.0) - lgamma(n + a + b + 1.0) - lgamma(n + 1.0) + (a + b + 1.0) * np.log(2.0)
    for i in range(n):
        xi = x[i]
        dP = 1.0
        for it in range(6):
            P, Pm = _jacobi_p(n, a, b, xi)
            dP = (n * ((a - b) - (2 * n + a + b) * xi) * P + 2 * (n + a) * (n + b) * Pm) / ((2 * n + a + b) * (1 - xi * xi))
            xi -= P / dP
        P, Pm = _jacobi_p(n, a, b, xi)
        dP = (n * ((a - b) - (2 * n + a + b) * xi) * P + 2 * (n + a) * (n + b) * Pm) / ((2 * n + a + b) * (1 - xi * xi))
        u[i] = 0.5 * (xi + 1.0)
        w[i] = np.exp(lc) / ((1 - xi * xi) * dP * dP) * 2.0 ** (-(b + 1.0))
    return u, w


_GJ_CACHE = {}


def quad_rules(gams):
    """Gauss-Jacobi rules (nt, N_NODES) for the first segment, one per exponent
    (weight u^(g+1)); rows of integer exponents are unused (flag 0)."""
    nt = len(gams)
    ju = np.zeros((nt, N_NODES))
    jw = np.zeros((nt, N_NODES))
    has = np.zeros(nt, dtype=np.int64)
    for i, g in enumerate(gams):
        g = float(g)
        if g == np.floor(g):
            continue
        if g not in _GJ_CACHE:
            if len(_GJ_CACHE) > 1024:
                _GJ_CACHE.clear()
            _GJ_CACHE[g] = gauss_jacobi01(N_NODES, g + 1.0)
        ju[i], jw[i] = _GJ_CACHE[g]
        has[i] = 1
    return ju, jw, has


@njit(cache=True, fastmath=False)
def _h(x, y, g):
    """(1 - y^(g+1)) / (2 (g+1) x) with x = r^2 and y = 1 - r^2 (both given
    accurately), stable for x -> 0 and y -> 0."""
    if x < 1e-8:
        return 0.5 * (1.0 - 0.5 * g * x)
    if y <= 0.0:
        return 0.5 / ((g + 1.0) * x)
    if x < 0.5:
        return -np.expm1((g + 1.0) * np.log1p(-x)) / (2.0 * (g + 1.0) * x)
    return (1.0 - np.exp((g + 1.0) * np.log(y))) / (2.0 * (g + 1.0) * x)


@njit(cache=True, fastmath=False)
def _acos_acc(x, om, op):
    """arccos(x) from accurate 1 - x (om) and 1 + x (op)."""
    if x > 0.5:
        return 2.0 * np.arcsin(np.sqrt(max(om, 0.0) / 2.0))
    if x < -0.5:
        return pi - 2.0 * np.arcsin(np.sqrt(max(op, 0.0) / 2.0))
    return np.arccos(x)


@njit(cache=True, fastmath=False)
def _hv(gi, y, r2, ly, l1x, sy, qy):
    if gi == 0.0:
        return 0.5
    if gi == 1.0:
        return 0.25 * (1.0 + y)
    if r2 < 1e-8:
        return 0.5 * (1.0 - 0.5 * gi * r2)
    if r2 < 0.05:
        return -np.expm1((gi + 1.0) * l1x) / (2.0 * (gi + 1.0) * r2)
    if gi == 0.5:
        return (1.0 - y * sy) / (3.0 * r2)
    if gi == 0.25:
        return (1.0 - y * qy) / (2.5 * r2)
    if gi == 0.75:
        return (1.0 - y * sy * qy) / (3.5 * r2)
    return (1.0 - np.exp((gi + 1.0) * ly)) / (2.0 * (gi + 1.0) * r2)


@njit(cache=True, fastmath=False)
def blocked_limb(z, p, gams, cs, gu, gw, ju, jw, has):
    """sum_i c_i * (blocked flux of (1 - r^2)^{g_i}) for an occultor that
    touches or crosses the limb (or covers the star). ju, jw, has: the
    Gauss-Jacobi rules of quad_rules(gams)."""
    nt = gams.shape[0]
    tot = 0.0
    # stellar-limb arc inside the occultor. cos(theta_s) = (1 + z^2 - p^2)/(2z),
    # 1 - cos = (z-1+p)(1-z+p)/(2z), 1 + cos = (1-p+z)(1+z+p)/(2z); the
    # cancelling sums are ordered so that they are exact (Sterbenz), and the
    # case decisions use them (not the rounded cosine).
    if z > 0.0:
        om = ((z - 1.0) + p) * ((1.0 - z) + p) / (2.0 * z)
        op = ((1.0 - p) + z) * (1.0 + z + p) / (2.0 * z)
        if om <= 0.0:
            th = 0.0
        elif op <= 0.0:
            th = pi
        else:
            th = _acos_acc(1.0 - om if om < op else op - 1.0, om, op)
    else:
        th = pi if p >= 1.0 else 0.0
    if th > 0.0:
        for i in range(nt):
            tot += cs[i] * th / (gams[i] + 1.0)
    # occultor arc inside the star: cos t < c = (1 - z^2 - p^2) / (2 z p),
    # 1 - c = (z-1+p)(z+p+1)/(2zp), 1 + c = (1-z+p)(1-p+z)/(2zp)
    if z > 0.0 and p > 0.0:
        omc = ((z - 1.0) + p) * (z + p + 1.0) / (2.0 * z * p)
        opc = ((1.0 - z) + p) * ((1.0 - p) + z) / (2.0 * z * p)
        arc = opc > 0.0
    else:
        omc = 0.0 if p < 1.0 else 2.0
        opc = 2.0 if p < 1.0 else 0.0
        arc = p < 1.0
    if not arc:
        return tot
    if omc <= 0.0:
        t0 = 0.0
    else:
        t0 = _acos_acc(1.0 - omc if omc < opc else opc - 1.0, omc, opc)
    zp = z * p
    omz2p2 = ((1.0 - z) - p) * (1.0 + z + p) + 2.0 * zp   # 1 - z^2 - p^2
    limb = t0 > 0.0
    if limb:
        # first segment up to the mirror branch point -t0, away from t = pi (r -> 0 if z = p)
        scale = min(t0, 0.5 * (pi - t0))
    else:
        ymin = ((1.0 - z) - p) * (1.0 + z + p)
        scale = np.sqrt(max(ymin, 0.0) / max(zp, 1e-300)) if zp > 0.0 else pi
    scale = min(max(scale, 1e-12), pi - t0)
    st0 = np.sin(t0)
    ct0 = np.cos(t0)
    a = t0
    b = t0 + scale
    first = limb
    for _ in range(200):
        L = b - a
        for n in range(gu.shape[0]):
            t = a + L * gu[n]
            jac = L * gw[n]
            if limb:
                h = 0.5 * (t - t0)
                sh = np.sin(h)
                ch = np.cos(h)
                y = 4.0 * zp * (sh * ct0 + ch * st0) * sh          # 1 - r^2, accurate near t0
                ct = (1.0 - 2.0 * sh * sh) * ct0 - 2.0 * sh * ch * st0
            else:
                ct = np.cos(t)
                y = omz2p2 - 2.0 * zp * ct
            y = min(max(y, 0.0), 1.0)
            r2 = max(1.0 - y, 0.0) if y > 0.5 else min(max(z * z + p * p + 2.0 * zp * ct, 0.0), 1.0)
            geo = 2.0 * p * (p + z * ct) * jac
            sy = np.sqrt(y)
            qy = np.sqrt(sy)
            ly = np.log(y) if y > 0.0 else -745.0
            l1x = np.log1p(-r2) if r2 < 0.05 else 0.0
            for i in range(nt):
                if first and has[i] == 1:
                    tot += cs[i] * geo / (2.0 * (gams[i] + 1.0) * r2)            # regular part
                else:
                    tot += cs[i] * _hv(gams[i], y, r2, ly, l1x, sy, qy) * geo
        if first:
            # singular parts: -(t - t0)^(g+1) phi(t), phi = (y/(t-t0))^(g+1) (p + z cos t) 2p / (2(g+1) r^2)
            for i in range(nt):
                if has[i] == 0:
                    continue
                g = gams[i]
                s = 0.0
                for n in range(ju.shape[1]):
                    d = L * ju[i, n]
                    sh = np.sin(0.5 * d)
                    ch = np.cos(0.5 * d)
                    ct = (1.0 - 2.0 * sh * sh) * ct0 - 2.0 * sh * ch * st0
                    sinc = 1.0 if d == 0.0 else 2.0 * sh / d
                    yd = 2.0 * zp * (sh * ct0 + ch * st0) * sinc
                    y = yd * d
                    r2 = max(1.0 - y, 0.0) if y > 0.5 else min(max(z * z + p * p + 2.0 * zp * ct, 0.0), 1.0)
                    s += jw[i, n] * np.exp((g + 1.0) * np.log(yd)) * 2.0 * p * (p + z * ct) / (2.0 * (g + 1.0) * r2)
                tot -= cs[i] * s * np.exp((g + 2.0) * np.log(L))
        if b >= pi:
            break
        first = False
        a = b
        b = min(t0 + 3.0 * (b - t0), pi)
        if pi - b < 0.25 * (b - a):
            b = pi
    return tot
