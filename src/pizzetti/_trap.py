"""Limb flux by a fixed-node sinh-trapezoid rule (no per-light-curve set-up).

For an occultor that crosses the stellar limb (1 - p < z < 1 + p, z > p), the
blocked flux of mu^alpha (Green's theorem, s = r^2) is the complete integral

    P_alpha = -2 sqrt(k) x^(alpha/2+1) / (alpha+2)
              * int_0^{pi/2} cos^(alpha+3)(th) [1 + c0 / (1 - x cos^2 th)] / sqrt(1 - k sin^2 th) dth,

    x = 1 - (z-p)^2,  k = x / (4 z p),  c0 = p^2 - z^2

(substitute sin^2 phi = k sin^2 th in the arc integral; no incomplete
integral is needed). With tan th = sinh s this becomes

    int_0^inf cosh(s)^-(alpha+3) (1 + kc2 sinh^2 s)^(-1/2) [1 + c0 b/(b - x)] ds,
    b = cosh^2 s,  kc2 = 1 - k,

whose integrand is even and analytic in the strip |Im s| < pi/2 for every z
(the elliptic branch point sits at Im s = pi/2 whatever the modulus, which is
the Landen/AGM structure seen in a logarithmic variable), apart from the
simple poles b = x at s = +-i acos(sqrt x). The trapezoid rule on s = n h then
converges like exp(-pi^2/h), uniformly up to the contacts; the pole's error
term is known in closed form (residue) and subtracted.

The nodes do not depend on z, so the exponent-dependent factor
cosh(s_n)^-(alpha+3) is folded into the weights once per exponent: a sample
costs one square root and one division per node, shared by all exponents,
plus two dot products per exponent.
"""
import math
from math import pi

import numpy as np
from numba import njit

from ._exact import acos_poly
from ._series import _vexp, _vlog

#: trapezoid step in s; the discretisation error is ~exp(-pi^2 / H) times a factor
#: that grows with the exponent (the singularity cosh^-(alpha+3) at Im s = pi/2)
H = 0.2


def step(amax):
    """Trapezoid step for the largest exponent alpha (flux error ~1e-15)."""
    return H if amax <= 3.0 else (0.17 if amax <= 6.0 else 0.14)
#: relative size of the neglected tail
TAIL = 1e-16
#: largest radius ratio for the rules (beyond, the pole b = x nears the real axis
#: at second contact and the quadrature of _general is used)
P_MAX = 0.3

_RULES = {}
_NODES = {}


def _nodes(h, n):
    """Node data of the real rule (cached per step and length): log cosh(s_n) * w-free,
    weights w_n, t_n^2, cosh^2(s_n) followed by h."""
    key = (h, n)
    hit = _NODES.get(key)
    if hit is None:
        s = h * np.arange(n)
        w = np.full(n, h)
        w[0] = 0.5 * h
        ch = np.cosh(s)
        hit = (np.log(ch), w, np.sinh(s) ** 2, np.append(ch * ch, h))
        _NODES[key] = hit
    return hit


@njit(cache=True)
def _weights_real(lc, w, gams):
    W = np.empty((gams.shape[0], lc.shape[0]))
    for i in range(gams.shape[0]):
        e = -(2.0 * gams[i] + 3.0)
        for n in range(lc.shape[0]):
            W[i, n] = w[n] * np.exp(e * lc[n])
    return W


def _length(gams, h):
    amin = max(2.0 * min(float(g) for g in gams), 0.0)
    # cosh^-(a+3) < TAIL beyond S
    S = (math.log(1.0 / TAIL) + (amin + 3.0) * math.log(2.0)) / (amin + 3.0)
    return int(math.ceil(S / h)) + 1


def rules(gams, h=None):
    """Weights W[i, n] = w_n cosh(s_n)^-(2 g_i + 3) and node data t_n^2, cosh^2(s_n)
    (followed by the step h) for the exponents alpha_i = 2 g_i (cached)."""
    key = (tuple(float(g) for g in gams), h)
    hit = _RULES.get(key)
    if hit is not None:
        return hit
    hh = step(2.0 * max(key[0])) if h is None else h
    lc, w, tt, b = _nodes(hh, _length(gams, hh))
    out = (_weights_real(lc, w, np.asarray(gams, dtype=np.float64)), tt, b)
    if len(_RULES) > 1024:
        _RULES.clear()
    _RULES[key] = out
    return out


@njit(cache=True, fastmath=True, error_model="numpy")
def blocked_cross(zl, m, p, gams, cs, W, tt, b, out):
    """out[j] = sum_i cs[i] P_{2 gams[i]}(zl[j], p) for occultors crossing the limb
    (1 - p < z < 1 + p, z > p; requires p < 0.5)."""
    nt = gams.shape[0]
    nn = tt.shape[0]
    h = b[nn]                          # the step is stored after the nodes
    r = np.empty(nn)
    q = np.empty(nn)
    cpre = np.empty(nt)
    bet = np.empty(nt)
    cpole = 0.0
    for i in range(nt):
        cpre[i] = -2.0 * cs[i] / (2.0 * gams[i] + 2.0)
        bet[i] = gams[i] + 1.0
        cpole += cpre[i]
    for j in range(m):
        z = zl[j]
        # accurate factors: x = (1 - z + p)(1 + z - p), 1 - k = (z + p - 1)(z + p + 1)/(4 z p)
        x = ((1.0 - z) + p) * ((1.0 + z) - p)
        q4 = 4.0 * z * p
        kk = x / q4
        kc2 = ((z - 1.0) + p) * (z + p + 1.0) / q4
        kc2 = max(kc2, 0.0)
        c0 = (p - z) * (p + z)
        for n in range(nn):
            rn = 1.0 / np.sqrt(1.0 + kc2 * tt[n])
            r[n] = rn
            q[n] = rn * b[n] / (b[n] - x)
        sk = np.sqrt(kk)
        lx = _vlog(x)
        tot = 0.0
        for i in range(nt):
            s1 = 0.0
            s2 = 0.0
            for n in range(nn):
                s1 += W[i, n] * r[n]
                s2 += W[i, n] * q[n]
            tot += cpre[i] * _vexp(bet[i] * lx) * (s1 + c0 * s2)
        # trapezoid error of the poles b = x (s = +-i acos sqrt x), subtracted:
        # dI = pi c0 sqrt(x) g(s0) / sqrt(1 - x) / (exp(2 pi a / H) - 1), x^(g+1) g(s0) sqrt x = 1/sqrt(1 - kc2 (1 - x))
        a = acos_poly(np.sqrt(x))
        e = _vexp(-2.0 * pi * a / h)
        om = (z - p) * (z - p)              # 1 - x
        corr = pi * c0 / np.sqrt(om * (1.0 - kc2 * om)) * e / (1.0 - e)
        out[j] = sk * (tot - cpole * corr)


# ---------------------------------------------------------------------------
# occultor inside the disk, close to the limb (z_c < z < 1 - p)
#
# There k > 1 and the same complete integral, continued analytically, has a
# branch point on the integration path (t = tan th = T = 1/sqrt(k - 1)); the
# blocked flux is its real part. With t = sinh(s + i gamma), s real, the path
# passes above T, and the integrand f(s) satisfies f(-s) = conj f(s), so
#     P = Re int_0^inf f(s) ds  (half of the full-line integral, which is real).
# Strip of analyticity: gamma below (branch point T), pi/2 - gamma above
# (t = i), and the poles b = x at s = i (a - gamma), -i (a + gamma),
# a = acos sqrt(x), whose trapezoid error is subtracted as above.
# ---------------------------------------------------------------------------

#: contour angle and step of the interior-band rule
GAMMA_IN = 0.5
H_IN = 0.1


def step_in(gamma, amax):
    """Step of the band rule: the upper strip pi/2 - gamma narrows for large gamma,
    which matters for the stronger singularity of large exponents."""
    return H_IN if (gamma <= 0.7 or amax <= 1.6) else 0.07

_RULES_IN = {}


_NODES_IN = {}


@njit(cache=True)
def _weights_cplx(lbr, lbi, wdr, wdi, gams):
    """w_n b_n^-(g+2) cosh(s_n) from Log b_n and w_n cosh(s_n) (real and imaginary parts)."""
    nt = gams.shape[0]
    nn = lbr.shape[0]
    Wr = np.empty((nt, nn))
    Wi = np.empty((nt, nn))
    for i in range(nt):
        e = -(gams[i] + 2.0)
        for n in range(nn):
            m = np.exp(e * lbr[n])
            ph = e * lbi[n]
            cr = m * np.cos(ph)
            ci = m * np.sin(ph)
            Wr[i, n] = cr * wdr[n] - ci * wdi[n]
            Wi[i, n] = cr * wdi[n] + ci * wdr[n]
    return Wr, Wi


def rules_in(gams, gamma=GAMMA_IN):
    """Complex weights and node data of the band rule for the contour angle gamma (cached)."""
    key = (tuple(float(g) for g in gams), float(gamma))
    hit = _RULES_IN.get(key)
    if hit is not None:
        return hit
    hin = step_in(gamma, 2.0 * max(key[0]))
    n = _length(gams, hin)
    nk = (float(gamma), n, hin)
    nd = _NODES_IN.get(nk)
    if nd is None:
        s = hin * np.arange(n) + 1j * gamma
        w = np.full(n, hin)
        w[0] = 0.5 * hin
        t = np.sinh(s)
        b = np.cosh(s) ** 2
        lb = np.log(b)
        wd = w * np.cosh(s)
        tt = t * t
        nd = (np.ascontiguousarray(lb.real), np.ascontiguousarray(lb.imag), np.ascontiguousarray(wd.real),
              np.ascontiguousarray(wd.imag), np.ascontiguousarray(tt.real), np.ascontiguousarray(tt.imag),
              np.ascontiguousarray(np.append(b.real, hin)), np.ascontiguousarray(b.imag))
        _NODES_IN[nk] = nd
    Wr, Wi = _weights_cplx(nd[0], nd[1], nd[2], nd[3], np.asarray(gams, dtype=np.float64))
    out = (Wr, Wi, nd[4], nd[5], nd[6], nd[7])
    if len(_RULES_IN) > 1024:
        _RULES_IN.clear()
    _RULES_IN[key] = out
    return out


#: contour angles tried for the band, and the smallest distance of the pole
#: angle a = acos sqrt(x) from the contour (the residue term covers either side)
GAMMAS_IN = (0.5, 0.65, 0.8)
POLE_GAP = 0.1


def band_gamma(p, z_lo, amax=0.0):
    """Contour angle for the band z_lo < z < 1 - p, or -1 if none applies: over the
    band, a = acos sqrt(x) runs from a(z_lo) to a(1 - p), and the contour must keep
    POLE_GAP from it (the pole may lie on either side; its residue term is continuous)."""
    if not (0.0 < p < 0.5) or z_lo <= p:
        return -1.0
    a_lo = math.acos(math.sqrt(1.0 - (z_lo - p) ** 2))
    a_hi = math.acos(math.sqrt(4.0 * p * (1.0 - p)))
    for g in GAMMAS_IN:
        if g < a_lo - POLE_GAP or g > a_hi + POLE_GAP:
            return g
    return -1.0


def band_ok(p, z_lo):
    return band_gamma(p, z_lo) > 0.0


@njit(cache=True, fastmath=True, error_model="numpy")
def blocked_band(zl, m, p, gams, cs, Wr, Wi, ttr, tti, br, bi, gamma, out):
    """out[j] = sum_i cs[i] P_{2 gams[i]}(zl[j], p) for occultors inside the disk
    near the limb (p < z < 1 - p, band_ok)."""
    nt = gams.shape[0]
    nn = ttr.shape[0]
    hin = br[nn]                       # the step is stored after the nodes
    rr = np.empty(nn)
    ri = np.empty(nn)
    qr = np.empty(nn)
    qi = np.empty(nn)
    cpre = np.empty(nt)
    bet = np.empty(nt)
    cpole = 0.0
    for i in range(nt):
        cpre[i] = -2.0 * cs[i] / (2.0 * gams[i] + 2.0)
        bet[i] = gams[i] + 1.0
        cpole += cpre[i]
    for j in range(m):
        z = zl[j]
        x = ((1.0 - z) + p) * ((1.0 + z) - p)
        q4 = 4.0 * z * p
        kk = x / q4
        lam = ((z - 1.0) + p) * (z + p + 1.0) / q4        # 1 - k < 0
        c0 = (p - z) * (p + z)
        for n in range(nn):
            # r = (1 + lam t^2)^(-1/2), principal branch (Im(1 + lam t^2) <= 0 on the path)
            u = 1.0 + lam * ttr[n]
            v = lam * tti[n]
            mod = np.sqrt(u * u + v * v)
            a = np.sqrt(0.5 * (mod + abs(u)))
            d = 0.5 * v / a
            sre = a if u >= 0.0 else abs(d)
            sim = d if u >= 0.0 else (a if v >= 0.0 else -a)
            im = 1.0 / mod
            r_re = sre * im
            r_im = -sim * im
            rr[n] = r_re
            ri[n] = r_im
            # q = r b / (b - x)
            dr = br[n] - x
            di = bi[n]
            idn = 1.0 / (dr * dr + di * di)
            gr = (br[n] * dr + bi[n] * di) * idn
            gi = (bi[n] * dr - br[n] * di) * idn
            qr[n] = r_re * gr - r_im * gi
            qi[n] = r_re * gi + r_im * gr
        sk = np.sqrt(kk)
        lx = _vlog(x)
        tot = 0.0
        for i in range(nt):
            s = 0.0
            for n in range(nn):
                s += Wr[i, n] * (rr[n] + c0 * qr[n]) - Wi[i, n] * (ri[n] + c0 * qi[n])
            tot += cpre[i] * _vexp(bet[i] * lx) * s
        # pole errors (residue terms), poles at s = i(a - gamma) and -i(a + gamma)
        a = acos_poly(np.sqrt(x))
        om = (z - p) * (z - p)
        e1 = np.exp(-2.0 * pi * (a - gamma) / hin)
        e2 = _vexp(-2.0 * pi * (a + gamma) / hin)
        A = pi * c0 / np.sqrt(om * (1.0 - lam * om))
        corr = 0.5 * A * (e1 / (1.0 - e1) + e2 / (1.0 - e2))
        out[j] = sk * (tot - cpole * corr)
