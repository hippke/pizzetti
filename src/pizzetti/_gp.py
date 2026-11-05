"""Gaussian-process noise for ``MarginalTransitModel``.

With Sigma = K(eta) + diag(s^2 yerr^2) the marginal likelihood is the white-noise one on whitened data:
with Sigma = L L^T, the kernels of ``_marginal`` take L^-1 A and L^-1 y, and -sum ln diag(L) is added. The SHO
kernel with Q = 1 / sqrt(2),

    k(dt) = amp^2 exp(-x) (cos x + sin x),    x = |dt| / (sqrt(2) tau),

is a two-state linear stochastic differential equation: the innovations of its Kalman filter divided by their
standard deviation are L^-1 y, at O(N) per column, for any spacing of the times, and without storing a matrix.
"""
import math

import numpy as np
from numba import njit

from . import _marginal as K

_SQRT2 = math.sqrt(2.0)
_NAMES = ("amp", "tau", "err_scale")
_LABELS = ("ln_gp_amp", "ln_gp_tau", "ln_err_scale")


class GP:
    """Correlated noise for ``MarginalTransitModel(..., gp=GP())``: an additive Gaussian process, fitted together
    with the transit, instead of detrending the light curve first.

    :param kernel: "sho" (a stochastically driven, damped harmonic oscillator with Q = 1 / sqrt(2): smooth,
        no periodicity, power falling as frequency^-4 above 1 / (2 pi tau))
    :param amp: rms of the GP, in flux units
    :param tau: time scale 1 / omega_0 in days (the correlation falls to 1/2 at about 1.4 tau, and to 0 at 3.1 tau)
    :param err_scale: factor on ``yerr``, the white noise of the fit is ``err_scale * yerr``

    Each of ``amp``, ``tau`` and ``err_scale`` is a tuple (lo, hi), the bounds of a log-uniform prior (the parameter
    is sampled), or a number (held fixed). Defaults: ``amp`` from 0.02 x the median ``yerr`` to 0.2 x the median
    flux; ``tau`` from the median cadence to 180 T14 (T14 of the starting ``TransitParams``); ``err_scale`` from
    1/3 to 3. The bounds matter when the data do not constrain a parameter: a trend much slower than the
    time covered by each window constrains only ``amp / tau``, a weak GP gives only an upper limit on ``amp``.
    """

    def __init__(self, kernel="sho", amp=None, tau=None, err_scale=(1.0 / 3.0, 3.0)):
        if kernel != "sho":
            raise NotImplementedError(f'kernel {kernel!r} is not implemented; supported: "sho"')
        self.kernel = kernel
        self.amp, self.tau, self.err_scale = amp, tau, err_scale
        for name, v in zip(_NAMES, (amp, tau, err_scale)):
            if v is None:
                continue
            lo, hi = (v, v) if np.ndim(v) == 0 else v
            if not 0.0 < lo <= hi or (np.ndim(v) != 0 and lo == hi):
                raise ValueError(f"{name} must be a positive number or a tuple 0 < lo < hi, got {v!r}")

    def __repr__(self):
        return f"GP(kernel={self.kernel!r}, amp={self.amp!r}, tau={self.tau!r}, err_scale={self.err_scale!r})"

    def _resolve(self, t, y, yerr, t14):
        """(free columns of the three, ln fixed value or nan, ln lo, ln hi, labels of the free ones) for this data."""
        cad = float(np.median(np.diff(t)))
        ye = float(np.median(yerr))
        dflt = dict(amp=(0.02 * ye, 0.2 * abs(float(np.median(y)))), tau=(cad, max(180.0 * t14, 10.0 * cad)),
                    err_scale=(1.0 / 3.0, 3.0))
        val = np.full(3, np.nan)
        lo, hi = np.full(3, np.nan), np.full(3, np.nan)
        free = []
        for j, (name, v) in enumerate(zip(_NAMES, (self.amp, self.tau, self.err_scale))):
            v = dflt[name] if v is None else v
            if np.ndim(v) == 0:
                val[j] = math.log(v)
            else:
                lo[j], hi[j] = math.log(v[0]), math.log(v[1])
                free.append(j)
        return np.array(free, dtype=np.int64), val, lo, hi, [_LABELS[j] for j in free]


@njit(cache=True)
def _step(x, s2):
    """Transition matrix and process noise of the state (g, g' / omega_0) over a step of x = dt / (sqrt(2) tau)."""
    e = math.exp(-x)
    c, sn = math.cos(x), math.sin(x)
    f01 = 1.4142135623730951 * e * sn
    e2 = e * e
    em = -math.expm1(-2.0 * x)
    return (e * (c + sn), f01, -f01, e * (c - sn),
            s2 * (em - e2 * 2.0 * sn * (c + sn)), s2 * (2.8284271247461903 * e2 * sn * sn),
            s2 * (em - e2 * 2.0 * sn * (sn - c)))


@njit(cache=True)
def whiten(t, var, amp, tau, Y):
    """Replace the columns of Y (n x m, ascending t) by L^-1 Y, with L L^T = K + diag(var). Returns sum ln diag(L)."""
    n, mc = Y.shape
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    m0 = np.zeros(mc)
    m1 = np.zeros(mc)
    p00, p01, p11 = s2, 0.0, s2
    ld = 0.0
    for i in range(n):
        if i > 0:
            f00, f01, f10, f11, q00, q01, q11 = _step(lam * (t[i] - t[i - 1]), s2)
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + q00
            p01 = a00 * f10 + a01 * f11 + q01
            p11 = a10 * f10 + a11 * f11 + q11
            for j in range(mc):
                u = f00 * m0[j] + f01 * m1[j]
                m1[j] = f10 * m0[j] + f11 * m1[j]
                m0[j] = u
        S = p00 + var[i]
        rs = math.sqrt(S)
        ld += math.log(rs)
        k0, k1 = p00 / S, p01 / S
        for j in range(mc):
            inn = Y[i, j] - m0[j]
            Y[i, j] = inn / rs
            m0[j] += k0 * inn
            m1[j] += k1 * inn
        p11 -= k1 * p01
        p01 -= k1 * p00
        p00 -= k0 * p00
    return ld


@njit(cache=True)
def predict(t, var, amp, tau, r):
    """Posterior mean and variance of the GP at the times t, given r = y - transit model: Kalman filter + RTS smoother."""
    n = t.shape[0]
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    F = np.empty((n, 4))
    mp = np.empty((n, 2))
    Pp = np.empty((n, 3))
    mf = np.empty((n, 2))
    Pf = np.empty((n, 3))
    m0, m1, p00, p01, p11 = 0.0, 0.0, s2, 0.0, s2
    for i in range(n):
        if i > 0:
            f00, f01, f10, f11, q00, q01, q11 = _step(lam * (t[i] - t[i - 1]), s2)
            F[i, 0], F[i, 1], F[i, 2], F[i, 3] = f00, f01, f10, f11
            m0, m1 = f00 * m0 + f01 * m1, f10 * m0 + f11 * m1
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + q00
            p01 = a00 * f10 + a01 * f11 + q01
            p11 = a10 * f10 + a11 * f11 + q11
        mp[i, 0], mp[i, 1] = m0, m1
        Pp[i, 0], Pp[i, 1], Pp[i, 2] = p00, p01, p11
        S = p00 + var[i]
        k0, k1 = p00 / S, p01 / S
        inn = r[i] - m0
        m0 += k0 * inn
        m1 += k1 * inn
        p11 -= k1 * p01
        p01 -= k1 * p00
        p00 -= k0 * p00
        mf[i, 0], mf[i, 1] = m0, m1
        Pf[i, 0], Pf[i, 1], Pf[i, 2] = p00, p01, p11
    mean = np.empty(n)
    v = np.empty(n)
    mean[n - 1], v[n - 1] = m0, p00
    s0, s1, c00, c01, c11 = m0, m1, p00, p01, p11
    for i in range(n - 2, -1, -1):
        f00, f01, f10, f11 = F[i + 1, 0], F[i + 1, 1], F[i + 1, 2], F[i + 1, 3]
        a00, a01, a11 = Pp[i + 1, 0], Pp[i + 1, 1], Pp[i + 1, 2]
        det = a00 * a11 - a01 * a01
        b00, b01, b11 = Pf[i, 0], Pf[i, 1], Pf[i, 2]
        # G = Pf F^T Pp(i+1)^-1
        h00 = b00 * f00 + b01 * f01
        h01 = b00 * f10 + b01 * f11
        h10 = b01 * f00 + b11 * f01
        h11 = b01 * f10 + b11 * f11
        g00 = (h00 * a11 - h01 * a01) / det
        g01 = (h01 * a00 - h00 * a01) / det
        g10 = (h10 * a11 - h11 * a01) / det
        g11 = (h11 * a00 - h10 * a01) / det
        d0, d1 = s0 - mp[i + 1, 0], s1 - mp[i + 1, 1]
        s0, s1 = mf[i, 0] + g00 * d0 + g01 * d1, mf[i, 1] + g10 * d0 + g11 * d1
        e00, e01, e11 = c00 - a00, c01 - a01, c11 - a11
        u0 = e00 * g00 + e01 * g01
        u1 = e01 * g00 + e11 * g01
        w0 = e00 * g10 + e01 * g11
        w1 = e01 * g10 + e11 * g11
        c00 = b00 + g00 * u0 + g01 * u1
        c01 = b01 + g00 * w0 + g01 * w1
        c11 = b11 + g10 * w0 + g11 * w1
        mean[i], v[i] = s0, c00
    return mean, v


@njit(cache=True)
def _whitened(t, tt, ksup, wts, y, var0, P, amp, tau, s2, tol, A_half, Gt):
    n = t.shape[0]
    B = K.design_theta(tt, ksup, wts, P[0], P[1], P[2], P[3], P[4], P[5], P[6], tol, A_half, Gt)
    Y = np.empty((n, 4))
    for i in range(n):
        Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3] = y[i], B[i, 0], B[i, 1], B[i, 2]
    ld = whiten(t, var0 * s2, amp, tau, Y)
    yw = np.empty(n)
    Bw = np.empty((n, 3))
    for i in range(n):
        yw[i], Bw[i, 0], Bw[i, 1], Bw[i, 2] = Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3]
    return Bw, yw, ld


@njit(cache=True)
def loglike_batch(P, lj, E, ok, lo, hi, t, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out):
    """Log-likelihood + box prior + log-Jacobian for canonical rows P and E = (amp, tau, err_scale) per row; rows with
    ok[i] False are outside the box of the GP parameters. As ``_marginal.loglike_batch``, with whitening."""
    n = P.shape[0]
    out = np.empty(n)
    for i in range(n):
        good = ok[i]
        for k in range(7):
            v = P[i, k]
            if not (v == v and lo[k] < v < hi[k]):
                good = False
        rp, a, b, ecc, w = P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        if not (good and 0.0 <= ecc < 1.0 and 0.0 <= b and K.b_conj(b, ecc, w, 1) < 1.0 + rp and P[i, 1] > 0.0
                and rp > 0.0 and a > 1.0):
            out[i] = -np.inf
            if logP_out.shape[0] > 0:
                logP_out[i] = 0.0
            continue
        Bw, yw, ld = _whitened(t, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol, A_half, Gt)
        v, lP = K.marg_theta_parts(Bw, yw, K.GL_X, K.GL_W)
        out[i] = v + lP - ld + lj[i]
        if logP_out.shape[0] > 0:
            logP_out[i] = lP
    return out


@njit(cache=True)
def theta_moments_batch(P, E, t, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """Mean (n x 3) and covariance (n x 3 x 3) of the Gaussian of theta = (f0, c1, c2), as
    ``_marginal.theta_moments_batch``, with whitening."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    L = np.zeros((3, 3))
    for i in range(n):
        Bw, yw, _ = _whitened(t, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol, A_half, Gt)
        G, r = K.normal_eqs(Bw, yw)
        s = np.empty(3)
        ok = True
        for k in range(3):
            if not G[k, k] > 0.0:
                ok = False
            else:
                s[k] = math.sqrt(G[k, k])
        if not ok:
            continue
        Gs = np.empty((3, 3))
        for k in range(3):
            for j in range(3):
                Gs[k, j] = G[k, j] / (s[k] * s[j])
        ok, ld, Gsi = K.inv3_chol(Gs, L)
        if not ok or L[2, 2] * L[2, 2] < 1e-12:
            for k in range(3):
                Gs[k, k] += 1e-9
            ok, ld, Gsi = K.inv3_chol(Gs, L)
            if not ok:
                continue
        for k in range(3):
            acc = 0.0
            for j in range(3):
                S[i, k, j] = Gsi[k, j] / (s[k] * s[j])
                acc += S[i, k, j] * r[j]
            T[i, k] = acc
    return T, S
