"""High-precision reference for the quadratic marginal likelihood (tests only; needs mpmath).

The flux model is A c with the Bernstein columns A_k = kappa_k - D_k (kappa = 1/6, 1/3, 1/2: the
disk-integrated basis, D_k the deficits in transit, exactly 0 out of transit). With
theta = (f0, c1, c2), f0 = kappa . c (the out-of-transit flux), c0 = 6 f0 - 2 c1 - 3 c2 and

    A c = B theta,  B = [1 - 6 D0, 2 D0 - D1, 3 D0 - D2]   (rows times the noise weights),

so the marginal likelihood (flat prior on c over the cone 0 <= c0 <= c1 <= c2) is

    log L = -chi2_min / 2 + log I + log 6 - 3/2 log 2 pi,
    I = integral over the cone of exp(-(theta - theta_hat)^T G (theta - theta_hat) / 2) d theta,

G = B^T B. In theta the cone is f0 > 0, 0 <= c1 <= 6 f0 / 5, max(c1, 2 f0 - c1) <= c2 <= 2 f0 - 2 c1 / 3.
The deficits are computed in float64 (as by the kernels); everything after them in mpmath.
"""
import math

import mpmath as mp
import numpy as np

from pizzetti import _marginal as K
from pizzetti import _orbit, _series
from pizzetti.marginal import _A_HALF


def deficits(m, prow):
    """Noise-weighted, exposure-averaged deficits D (n x 3) and weights for canonical row prow
    (t0, per, rp, a, b, ecc, w) of a quadratic MarginalTransitModel m (fast path)."""
    t0, per, rp, a, b, ecc, w = (float(v) for v in prow)
    z = np.empty(m.tt.shape[0])
    inc = math.acos(min(abs(b) / a, 1.0))
    if ecc < 1e-5:
        _orbit._rsky_circular(m.tt, t0, per, a, math.sin(inc), 0.25, 1, z)
    else:
        _orbit._rsky_eccentric(m.tt, t0, per, a, inc, ecc, w, 1, z)
    T = K._occult_terms_cut(z, abs(rp), m.tol, _A_HALF, _series._GTAB)
    lex, ldx, edx = T
    Df = np.column_stack([2 * lex - 2 * ldx - edx, 2 * (ldx - lex + edx), lex - edx])
    D = Df.reshape(-1, m.k, 3).mean(axis=1) * m.wts[:, None]
    return D


def marg_loglike_ref(D, wts, yw, dps=40):
    """Reference log L (as marg_loglike: v + log P) from the weighted deficits D, the weights
    (the weighted constant column) and the weighted data."""
    mp.mp.dps = dps
    n = len(yw)
    B = np.column_stack([wts - 6 * D[:, 0], 2 * D[:, 0] - D[:, 1], 3 * D[:, 0] - D[:, 2]])
    G = mp.matrix(3, 3)
    r = mp.matrix(3, 1)
    for i in range(3):
        r[i] = mp.mpf(math.fsum(B[:, i] * yw))
        for j in range(3):
            G[i, j] = mp.mpf(math.fsum(B[:, i] * B[:, j]))
    th = mp.lu_solve(G, r)
    thf = np.array([float(th[i]) for i in range(3)])
    res = yw - B @ thf
    # chi2 about the float theta, corrected to the exact minimum (the quadratic form's offset)
    d = mp.matrix([[mp.mpf(thf[i]) - th[i]] for i in range(3)])
    chi2 = mp.mpf(math.fsum(res * res)) - (d.T * G * d)[0]
    # sequential factorisation of the Gaussian: c2 | (f0, c1), c1 | f0, f0
    g22 = G[2, 2]
    H = mp.matrix(2, 2)
    for i in range(2):
        for j in range(2):
            H[i, j] = G[i, j] - G[i, 2] * G[2, j] / g22
    h = H[0, 0] - H[0, 1] ** 2 / H[1, 1]

    def mu2(f0, c1):
        return th[2] - (G[2, 0] * (f0 - th[0]) + G[2, 1] * (c1 - th[1])) / g22

    def mu1(f0):
        return th[1] - H[1, 0] * (f0 - th[0]) / H[1, 1]

    def j2(f0, c1):
        lo, hi = max(c1, 2 * f0 - c1), 2 * f0 - 2 * c1 / 3
        if hi <= lo:
            return mp.mpf(0)
        s = mp.sqrt(g22 / 2)
        m2 = mu2(f0, c1)
        return mp.sqrt(mp.pi / (2 * g22)) * (mp.erf(s * (hi - m2)) - mp.erf(s * (lo - m2)))

    def j1(f0):
        m1, s1 = mu1(f0), 1 / mp.sqrt(H[1, 1])
        tot = mp.mpf(0)
        for a_, b_ in ((mp.mpf(0), f0), (f0, 6 * f0 / 5)):
            pts = sorted({a_, b_} | {min(max(x, a_), b_) for x in (m1 - 8 * s1, m1, m1 + 8 * s1)})
            tot += mp.quad(lambda c1: mp.exp(-H[1, 1] * (c1 - m1) ** 2 / 2) * j2(f0, c1), pts)
        return tot

    s0 = 1 / mp.sqrt(h)
    lo0 = max(mp.mpf(0), th[0] - 12 * s0)
    I = mp.quad(lambda f0: mp.exp(-h * (f0 - th[0]) ** 2 / 2) * j1(f0), [lo0, th[0], th[0] + 12 * s0])
    out = -chi2 / 2 + mp.log(I) + mp.log(6) - mp.mpf(1.5) * mp.log(2 * mp.pi)
    return float(out), dict(theta=thf, chi2=float(chi2), logI=float(mp.log(I)))


def marg_loglike_ref_ray(D, wts, yw, dps=40):
    """As marg_loglike_ref, with the cone integral in the ray form (f0 analytic, then the triangle
    T1 by 2-d tanh-sinh, whose nodes cluster at the edges: accurate also where the Gaussian's centre
    is far outside the cone and the mass sits at the cone's boundary)."""
    mp.mp.dps = dps
    B = np.column_stack([wts - 6 * D[:, 0], 2 * D[:, 0] - D[:, 1], 3 * D[:, 0] - D[:, 2]])
    G = mp.matrix(3, 3)
    r = mp.matrix(3, 1)
    for i in range(3):
        r[i] = mp.mpf(math.fsum(B[:, i] * yw))
        for j in range(3):
            G[i, j] = mp.mpf(math.fsum(B[:, i] * B[:, j]))
    th = mp.lu_solve(G, r)
    thf = np.array([float(th[i]) for i in range(3)])
    res = yw - B @ thf
    d = mp.matrix([[mp.mpf(thf[i]) - th[i]] for i in range(3)])
    chi2 = mp.mpf(math.fsum(res * res)) - (d.T * G * d)[0]
    tGt = (th.T * G * th)[0]

    def g(v1, v2):
        e = mp.matrix([[1], [v1], [v2]])
        Ge = G * e
        a = (e.T * Ge)[0]
        b = (th.T * Ge)[0]
        minq = (tGt * a - b * b) / a
        mu, sd = b / a, 1 / mp.sqrt(a)
        z = mu / sd
        m2 = mp.sqrt(2 * mp.pi) * sd ** 3 * ((z * z + 1) * mp.ncdf(z) + z * mp.npdf(z))
        return mp.exp(-minq / 2) * m2

    I = mp.mpf(0)
    for a_, b_ in ((mp.mpf(0), mp.mpf(1)), (mp.mpf(1), mp.mpf(6) / 5)):
        I += mp.quad(lambda v1: mp.quad(lambda v2: g(v1, v2), [max(v1, 2 - v1), 2 - 2 * v1 / 3]), [a_, b_])
    return float(-chi2 / 2 + mp.log(I) + mp.log(6) - mp.mpf(1.5) * mp.log(2 * mp.pi))
