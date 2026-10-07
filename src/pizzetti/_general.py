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
The arc integral uses composite Gauss-Legendre quadrature on segments that
grow geometrically away from the nearest singularity of (1 - r^2)^(g+1)
(at t = -t0, or at +-i sqrt(y_min/(z p)) for an interior disk); the first
segment uses t = t0 + L u^2 to remove the (t - t0)^(g+1) endpoint behaviour.
"""
from math import pi

import numpy as np
from numba import njit

#: Gauss-Legendre nodes per segment of the limb arc integral (geometric
#: segments towards the nearest singularity).
N_NODES = 12
_X, _W = np.polynomial.legendre.leggauss(N_NODES)
GL_U = 0.5 * (_X + 1.0)
GL_W = 0.5 * _W


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
def blocked_limb(z, p, gams, cs, gu, gw):
    """sum_i c_i * (blocked flux of (1 - r^2)^{g_i}) for an occultor that
    touches or crosses the limb (or covers the star)."""
    nt = gams.shape[0]
    tot = 0.0
    need_log = False
    for i in range(nt):
        gi = gams[i]
        if not (gi == 0.0 or gi == 1.0 or gi == 0.5 or gi == 0.25 or gi == 0.75):
            need_log = True
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
            tot += cs[i] * 2.0 * th * 0.5 / (gams[i] + 1.0)
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
    if arc:
        if omc <= 0.0:
            t0 = 0.0
        else:
            t0 = _acos_acc(1.0 - omc if omc < opc else opc - 1.0, omc, opc)
        zp = z * p
        omz2p2 = ((1.0 - z) - p) * (1.0 + z + p) + 2.0 * zp   # 1 - z^2 - p^2
        # length scale of the nearest singularity of (1 - r^2)^(g+1) in t:
        # the second zero at -t0 (limb), or +-i sqrt(y_min / (z p)) (interior)
        if t0 > 0.0:
            scale = t0
        else:
            ymin = ((1.0 - z) - p) * (1.0 + z + p)
            scale = np.sqrt(max(ymin, 0.0) / max(zp, 1e-300)) if zp > 0.0 else pi
        scale = min(max(scale, 1e-12), pi - t0)
        a = t0
        b = t0 + scale
        first = True
        for _ in range(64):
            L = b - a
            for n in range(gu.shape[0]):
                u = gu[n]
                if first:      # t = t0 + L u^2: removes the (t - t0)^(g+1) endpoint
                    t = a + L * u * u
                    jac = 2.0 * L * u * gw[n]
                else:
                    t = a + L * u
                    jac = L * gw[n]
                ct = np.cos(t)
                # 1 - r^2 = 2 z p (cos t0 - cos t); cos t0 = 1 - omc from its accurate form
                if t0 > 0.0:
                    y = 2.0 * zp * ((1.0 - ct) - omc)
                else:
                    y = omz2p2 - 2.0 * zp * ct
                y = min(max(y, 0.0), 1.0)
                r2 = max(1.0 - y, 0.0) if y > 0.5 else min(max(z * z + p * p + 2.0 * zp * ct, 0.0), 1.0)
                geo = 2.0 * p * (p + z * ct) * jac
                sy = np.sqrt(y)
                qy = np.sqrt(sy)
                ly = np.log(y) if (y > 0.0 and need_log) else -745.0
                l1x = np.log1p(-r2) if r2 < 0.05 else 0.0
                for i in range(nt):
                    gi = gams[i]
                    if gi == 0.0:
                        hv = 0.5
                    elif gi == 1.0:
                        hv = 0.25 * (1.0 + y)
                    elif r2 < 1e-8:
                        hv = 0.5 * (1.0 - 0.5 * gi * r2)
                    elif r2 < 0.05:
                        hv = -np.expm1((gi + 1.0) * l1x) / (2.0 * (gi + 1.0) * r2)
                    elif gi == 0.5:
                        hv = (1.0 - y * sy) / (3.0 * r2)
                    elif gi == 0.25:
                        hv = (1.0 - y * qy) / (2.5 * r2)
                    elif gi == 0.75:
                        hv = (1.0 - y * sy * qy) / (3.5 * r2)
                    else:
                        hv = (1.0 - np.exp((gi + 1.0) * ly)) / (2.0 * (gi + 1.0) * r2)
                    tot += cs[i] * hv * geo
            if b >= pi:
                break
            first = False
            a = b
            b = min(t0 + 3.0 * (b - t0), pi)
            if pi - b < 0.25 * (b - a):
                b = pi
    return tot
