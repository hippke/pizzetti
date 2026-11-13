"""Gaussian-process noise for ``MarginalTransitModel``.

With Sigma = K(eta) + diag(s^2 yerr^2) the marginal likelihood is the white-noise one on whitened data:
with Sigma = L L^T, the kernels of ``_marginal`` take L^-1 A and L^-1 y, and -sum ln diag(L) is added. The SHO
kernel with Q = 1 / sqrt(2),

    k(dt) = amp^2 exp(-x) (cos x + sin x),    x = |dt| / (sqrt(2) tau),

is a two-state linear stochastic differential equation: the innovations of its Kalman filter divided by their
standard deviation are L^-1 y, at O(N) per column, for any spacing of the times, and without storing a matrix.

Two savings. The transition over a time step is computed once per distinct step (``time_table``): an even cadence costs a few
exponentials and sines per set of GP parameters instead of one per point. With all GP parameters fixed (``fixed_setup``), the
gains and the whitened flux are computed once, and each geometry only filters the points around the transits; the
out-of-transit stretches between them, where the template is zero, are one transition in closed form
(``_segment``, as the tail operators of the GP search in the TLS package).
"""
import math

import numpy as np
from numba import njit, prange

from . import _marginal as K

_SQRT2 = math.sqrt(2.0)
DT_TOL = 1e-6                # relative: time steps within this share their transition (stored times are rounded, or jitter)
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
def time_table(t, tol):
    """(entry of each time step, distinct steps): steps within a relative tol of a group's smallest step share its mean.
    idx[i] is the step from t[i - 1] to t[i]; idx[0] is unused."""
    n = t.shape[0]
    dt = np.diff(t)
    o = np.argsort(dt)
    idx = np.zeros(n, dtype=np.int64)
    steps = np.empty(max(n - 1, 1))
    u = -1
    ref = acc = 0.0
    cnt = 0
    for r in range(n - 1):
        v = dt[o[r]]
        if u < 0 or v > ref * (1.0 + tol):
            if u >= 0:
                steps[u] = acc / cnt
            u += 1
            ref, acc, cnt = v, 0.0, 0
        acc += v
        cnt += 1
        idx[o[r] + 1] = u
    if u >= 0:
        steps[u] = acc / cnt
    return idx, steps[:u + 1]


@njit(cache=True)
def _transitions(steps, amp, tau):
    """The transition matrix and process noise per distinct step, n x 7 (see _step)."""
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    T = np.empty((steps.shape[0], 7))
    for u in range(steps.shape[0]):
        T[u, 0], T[u, 1], T[u, 2], T[u, 3], T[u, 4], T[u, 5], T[u, 6] = _step(lam * steps[u], s2)
    return T


@njit(cache=True)
def whiten(idx, steps, var, amp, tau, Y):
    """Replace the columns of Y (n x m, ascending times) by L^-1 Y, with L L^T = K + diag(var). Returns sum ln diag(L)."""
    return _whiten_T(idx, _transitions(steps, amp, tau), var, amp, Y)


@njit(cache=True)
def _transitions_into(steps, amp, tau, T):
    """_transitions into the caller's array T (len(steps) x 7)."""
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    for u in range(steps.shape[0]):
        T[u, 0], T[u, 1], T[u, 2], T[u, 3], T[u, 4], T[u, 5], T[u, 6] = _step(lam * steps[u], s2)


@njit(cache=True)
def _whiten_T(idx, T, var, amp, Y):
    """whiten() with the transition table T of its steps (_transitions)."""
    n, mc = Y.shape
    s2 = amp * amp
    m0 = np.zeros(mc)
    m1 = np.zeros(mc)
    p00, p01, p11 = s2, 0.0, s2
    ld = 0.0
    for i in range(n):
        if i > 0:
            u = idx[i]
            f00, f01, f10, f11 = T[u, 0], T[u, 1], T[u, 2], T[u, 3]
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + T[u, 4]
            p01 = a00 * f10 + a01 * f11 + T[u, 5]
            p11 = a10 * f10 + a11 * f11 + T[u, 6]
            for j in range(mc):
                v = f00 * m0[j] + f01 * m1[j]
                m1[j] = f10 * m0[j] + f11 * m1[j]
                m0[j] = v
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
def _whitened(idx, steps, tt, ksup, wts, y, var0, P, amp, tau, s2, tol, A_half, Gt):
    n = y.shape[0]
    B = K.design_theta(tt, ksup, wts, P[0], P[1], P[2], P[3], P[4], P[5], P[6], tol, A_half, Gt)
    Y = np.empty((n, 4))
    for i in range(n):
        Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3] = y[i], B[i, 0], B[i, 1], B[i, 2]
    ld = whiten(idx, steps, var0 * s2, amp, tau, Y)
    yw = np.empty(n)
    Bw = np.empty((n, 3))
    for i in range(n):
        yw[i], Bw[i, 0], Bw[i, 1], Bw[i, 2] = Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3]
    return Bw, yw, ld


def workspace(tt, ksup, n, nsteps):
    """Arrays for the workspace kernels (loglike_batch_ws, theta_moments_batch_ws): the design workspace of
    _marginal.workspace, the n x 4 columns to whiten, the whitened design and flux, the scaled variances and the
    transition table. Reused by every evaluation, so that a large light curve does not fault fresh pages in each time."""
    return (K.workspace(tt, ksup), np.empty((n, 4)), np.empty((n, 3)), np.empty(n), np.empty(n), np.empty((nsteps, 7)))


@njit(cache=True)
def _whitened_ws(idx, steps, tt, ksup, wts, y, var0, P, amp, tau, s2, tol, A_half, Gt, gws):
    """_whitened in the workspace gws (workspace()): identical values."""
    dws, Y, Bw, yw, var, Tt = gws
    n = y.shape[0]
    B, _ = K.design_theta_ws(tt, ksup, wts, P[0], P[1], P[2], P[3], P[4], P[5], P[6], tol, A_half, Gt, dws, True)
    for i in range(n):
        Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3] = y[i], B[i, 0], B[i, 1], B[i, 2]
        var[i] = var0[i] * s2
    _transitions_into(steps, amp, tau, Tt)
    ld = _whiten_T(idx, Tt, var, amp, Y)
    for i in range(n):
        yw[i], Bw[i, 0], Bw[i, 1], Bw[i, 2] = Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3]
    return Bw, yw, ld


@njit(cache=True)
def _inside(P, i, ok_i, lo, hi):
    """The checks of _marginal._loglike_row on canonical row i."""
    good = ok_i
    for k in range(7):
        v = P[i, k]
        if not (v == v and lo[k] < v < hi[k]):
            good = False
    rp, a, b, ecc, w = P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
    return good and 0.0 <= ecc < 1.0 and 0.0 <= b and K.b_conj(b, ecc, w, 1) < 1.0 + rp and P[i, 1] > 0.0 and rp > 0.0 and a > 1.0


@njit(cache=True)
def _row(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """(log-likelihood + box prior + log-Jacobian, log P of the cone) of canonical row i."""
    if not _inside(P, i, ok[i], lo, hi):
        return -np.inf, 0.0
    Bw, yw, ld = _whitened(idx, steps, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol, A_half, Gt)
    v, lP = K.marg_theta_parts(Bw, yw, K.GL_X, K.GL_W)
    return v + lP - ld + lj[i], lP


@njit(cache=True)
def loglike_batch(P, lj, E, ok, lo, hi, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out):
    """Log-likelihood + box prior + log-Jacobian for canonical rows P and E = (amp, tau, err_scale) per row; rows with
    ok[i] False are outside the box of the GP parameters. As ``_marginal.loglike_batch``, with whitening."""
    n = P.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        v, lP = _row(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True)
def _row_ws(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, gws):
    """_row in the workspace gws: identical values."""
    if not _inside(P, i, ok[i], lo, hi):
        return -np.inf, 0.0
    Bw, yw, ld = _whitened_ws(idx, steps, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol, A_half,
                              Gt, gws)
    v, lP = K.marg_theta_parts(Bw, yw, K.GL_X, K.GL_W)
    return v + lP - ld + lj[i], lP


@njit(cache=True)
def loglike_batch_ws(P, lj, E, ok, lo, hi, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out, gws):
    """loglike_batch reusing the workspace gws (workspace()) for every row: identical values."""
    n = P.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        v, lP = _row_ws(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, gws)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True, parallel=True)
def loglike_batch_par(P, lj, E, ok, lo, hi, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out):
    """loglike_batch with the rows spread over numba's threads (identical results)."""
    n = P.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in prange(n):
        v, lP = _row(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True)
def _moments(G, r, T, S, i):
    """Row i of the mean and covariance of theta = (f0, c1, c2) from the normal equations, as
    ``_marginal.theta_moments_batch``; left unchanged where they are singular."""
    s = np.empty(3)
    for k in range(3):
        if not G[k, k] > 0.0:
            return
        s[k] = math.sqrt(G[k, k])
    Gs = np.empty((3, 3))
    for k in range(3):
        for j in range(3):
            Gs[k, j] = G[k, j] / (s[k] * s[j])
    L = np.zeros((3, 3))
    ok, ld, Gsi = K.inv3_chol(Gs, L)
    if not ok or L[2, 2] * L[2, 2] < 1e-12:
        for k in range(3):
            Gs[k, k] += 1e-9
        ok, ld, Gsi = K.inv3_chol(Gs, L)
        if not ok:
            return
    for k in range(3):
        acc = 0.0
        for j in range(3):
            S[i, k, j] = Gsi[k, j] / (s[k] * s[j])
            acc += S[i, k, j] * r[j]
        T[i, k] = acc


@njit(cache=True)
def _moments_row(P, E, T, S, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    Bw, yw, ld = _whitened(idx, steps, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol, A_half, Gt)
    G, r = K.normal_eqs(Bw, yw)
    _moments(G, r, T, S, i)


@njit(cache=True)
def theta_moments_batch(P, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """Mean (n x 3) and covariance (n x 3 x 3) of the Gaussian of theta = (f0, c1, c2), as
    ``_marginal.theta_moments_batch``, with whitening."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    for i in range(n):
        _moments_row(P, E, T, S, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
    return T, S


@njit(cache=True)
def theta_moments_batch_ws(P, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, gws):
    """theta_moments_batch reusing the workspace gws: identical values."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    for i in range(n):
        Bw, yw, ld = _whitened_ws(idx, steps, tt, ksup, wts, y, var0, P[i], E[i, 0], E[i, 1], E[i, 2] * E[i, 2], tol,
                                  A_half, Gt, gws)
        G, r = K.normal_eqs(Bw, yw)
        _moments(G, r, T, S, i)
    return T, S


@njit(cache=True, parallel=True)
def theta_moments_batch_par(P, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """theta_moments_batch with the rows spread over numba's threads."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    for i in prange(n):
        _moments_row(P, E, T, S, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
    return T, S


# -- all GP parameters fixed: one filter per light curve, then only the points around the transits -----------------------------
@njit(cache=True)
def _segment(j0, j1, T, idx, k0, k1, rs, ev, eu):
    """Points j0 ... j1 - 1 with a zero template, from the filtered state x at j0 - 1: the state after them is Phi x, and
    their sums are x' W x (template with template), wv' x (with the flux) and wu' x (with the constant column).
    W_j = g g' + F' W_(j+1) F, w_j = e g + F' w_(j+1), with F = (I - K H) A and g = -(H A)' / sqrt(S), backwards."""
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    w00, w01, w11 = 0.0, 0.0, 0.0
    v0, v1, u0, u1 = 0.0, 0.0, 0.0, 0.0
    for j in range(j1 - 1, j0 - 1, -1):
        t = idx[j]
        a00, a01, a10, a11 = T[t, 0], T[t, 1], T[t, 2], T[t, 3]
        f00 = (1.0 - k0[j]) * a00
        f01 = (1.0 - k0[j]) * a01
        f10 = a10 - k1[j] * a00
        f11 = a11 - k1[j] * a01
        g0, g1 = -a00 / rs[j], -a01 / rs[j]
        # F' W F + g g'
        h00 = w00 * f00 + w01 * f10
        h01 = w00 * f01 + w01 * f11
        h10 = w01 * f00 + w11 * f10
        h11 = w01 * f01 + w11 * f11
        n00 = f00 * h00 + f10 * h10 + g0 * g0
        n01 = f00 * h01 + f10 * h11 + g0 * g1
        n11 = f01 * h01 + f11 * h11 + g1 * g1
        w00, w01, w11 = n00, n01, n11
        v0, v1 = f00 * v0 + f10 * v1 + ev[j] * g0, f01 * v0 + f11 * v1 + ev[j] * g1
        u0, u1 = f00 * u0 + f10 * u1 + eu[j] * g0, f01 * u0 + f11 * u1 + eu[j] * g1
        p00, p01 = p00 * f00 + p01 * f10, p00 * f01 + p01 * f11
        p10, p11 = p10 * f00 + p11 * f10, p10 * f01 + p11 * f11
    return p00, p01, p10, p11, w00, w01, w11, v0, v1, u0, u1


@njit(cache=True)
def fixed_setup(idx, steps, var, amp, tau, y, shift, inblock, tt, ksup):
    """Everything that does not depend on the geometry, for fixed GP parameters. inblock: the points that may be in transit.
    Returns the tuple that ``fixed_row`` reads. The flux minus shift is whitened (rather than the flux) to keep chi2 from cancelling."""
    n = y.shape[0]
    s2 = amp * amp
    T = _transitions(steps, amp, tau)
    k0, k1, rs = np.empty(n), np.empty(n), np.empty(n)
    ev, eu = np.empty(n), np.empty(n)
    mv0, mv1, mu0, mu1 = 0.0, 0.0, 0.0, 0.0
    p00, p01, p11 = s2, 0.0, s2
    ld = uu = uv = vv = 0.0
    for i in range(n):
        if i > 0:
            t = idx[i]
            f00, f01, f10, f11 = T[t, 0], T[t, 1], T[t, 2], T[t, 3]
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + T[t, 4]
            p01 = a00 * f10 + a01 * f11 + T[t, 5]
            p11 = a10 * f10 + a11 * f11 + T[t, 6]
            mv0, mv1 = f00 * mv0 + f01 * mv1, f10 * mv0 + f11 * mv1
            mu0, mu1 = f00 * mu0 + f01 * mu1, f10 * mu0 + f11 * mu1
        S = p00 + var[i]
        rs[i] = math.sqrt(S)
        ld += math.log(rs[i])
        k0[i], k1[i] = p00 / S, p01 / S
        inn = y[i] - shift - mv0
        ev[i] = inn / rs[i]
        mv0 += k0[i] * inn
        mv1 += k1[i] * inn
        inn = 1.0 - mu0
        eu[i] = inn / rs[i]
        mu0 += k0[i] * inn
        mu1 += k1[i] * inn
        uu += eu[i] * eu[i]
        uv += eu[i] * ev[i]
        vv += ev[i] * ev[i]
        p11 -= k1[i] * p01
        p01 -= k1[i] * p00
        p00 -= k0[i] * p00
    # blocks of consecutive points, and the zero-template segment after each
    starts = np.flatnonzero(inblock & ~np.concatenate((np.zeros(1, dtype=np.bool_), inblock[:-1])))
    ends = np.flatnonzero(inblock & ~np.concatenate((inblock[1:], np.zeros(1, dtype=np.bool_)))) + 1
    nb = starts.shape[0]
    seg = np.empty((nb, 11))
    for b in range(nb):
        j1 = starts[b + 1] if b + 1 < nb else n
        vals = _segment(ends[b], j1, T, idx, k0, k1, rs, ev, eu)
        for q in range(11):
            seg[b, q] = vals[q]
    pts = np.flatnonzero(inblock)
    ttb = np.empty(pts.shape[0] * ksup)
    for r in range(pts.shape[0]):
        for q in range(ksup):
            ttb[r * ksup + q] = tt[pts[r] * ksup + q]
    return (T[:, :4].copy(), idx[pts], k0[pts], k1[pts], rs[pts], ev[pts], eu[pts], ttb, starts, ends, seg,
            np.array([ld, uu, uv, vv, shift]), np.flatnonzero(inblock))


@njit(cache=True)
def _fixed_normal(P, fx, ksup, tol, A_half, Gt):
    """The normal equations of one geometry with the data of the blocks only: G, r and yy for the data minus shift x the first column."""
    T, tab, k0, k1, rs, ev, eu, ttb, starts, ends, seg, c, _ = fx
    uu, uv, vv, shift = c[1], c[2], c[3], c[4]
    m = tab.shape[0]
    B = K.design_theta(ttb, ksup, np.ones(m), P[0], P[1], P[2], P[3], P[4], P[5], P[6], tol, A_half, Gt)
    x = np.zeros((3, 2))
    ss = np.zeros((3, 3))
    su, sv = np.zeros(3), np.zeros(3)
    e = np.empty(3)
    r = 0
    for b in range(starts.shape[0]):
        for _i in range(ends[b] - starts[b]):
            t = tab[r]
            if starts[b] + _i > 0:
                for q in range(3):
                    v = T[t, 0] * x[q, 0] + T[t, 1] * x[q, 1]
                    x[q, 1] = T[t, 2] * x[q, 0] + T[t, 3] * x[q, 1]
                    x[q, 0] = v
            for q in range(3):
                inn = (B[r, 0] - 1.0 if q == 0 else B[r, q]) - x[q, 0]
                e[q] = inn / rs[r]
                x[q, 0] += k0[r] * inn
                x[q, 1] += k1[r] * inn
                su[q] += e[q] * eu[r]
                sv[q] += e[q] * ev[r]
            for q in range(3):
                for h in range(q, 3):
                    ss[q, h] += e[q] * e[h]
            r += 1
        s_ = seg[b]
        for q in range(3):
            a0, a1 = x[q, 0], x[q, 1]
            su[q] += s_[9] * a0 + s_[10] * a1
            sv[q] += s_[7] * a0 + s_[8] * a1
            for h in range(q, 3):
                d0, d1 = x[h, 0], x[h, 1]
                ss[q, h] += a0 * (s_[4] * d0 + s_[5] * d1) + a1 * (s_[5] * d0 + s_[6] * d1)
        for q in range(3):
            a0, a1 = x[q, 0], x[q, 1]
            x[q, 0] = s_[0] * a0 + s_[1] * a1
            x[q, 1] = s_[2] * a0 + s_[3] * a1
    G = np.empty((3, 3))
    G[0, 0] = uu + 2.0 * su[0] + ss[0, 0]
    G[0, 1] = G[1, 0] = su[1] + ss[0, 1]
    G[0, 2] = G[2, 0] = su[2] + ss[0, 2]
    G[1, 1], G[1, 2], G[2, 1], G[2, 2] = ss[1, 1], ss[1, 2], ss[1, 2], ss[2, 2]
    rv = np.array([uv + sv[0] - shift * (su[0] + ss[0, 0]), sv[1] - shift * ss[0, 1], sv[2] - shift * ss[0, 2]])
    yy = vv - 2.0 * shift * sv[0] + shift * shift * ss[0, 0]
    return G, rv, yy


@njit(cache=True)
def _fits_blocks(P, t_ref, half, half_exp):
    """All in-transit points of this circular geometry lie in the blocks (|time from t_ref, folded| < half)."""
    rp, a, b = P[2], P[3], P[4]
    q = max((1.0 + rp) * (1.0 + rp) - b * b, 0.0) / max(a * a - b * b, 1e-300)
    d = P[1] / math.pi * math.asin(min(math.sqrt(q), 1.0))
    return abs(P[0] - t_ref) + 0.5 * d + half_exp < half


@njit(cache=True)
def _fixed_row(P, lj, ok, lo, hi, i, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    if not _inside(P, i, ok[i], lo, hi):
        return -np.inf, 0.0
    if not _fits_blocks(P[i], t_ref, half, half_exp):
        return _row(P, lj, E, ok, lo, hi, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
    G, r, yy = _fixed_normal(P[i], fx, ksup, tol, A_half, Gt)
    v, lP = K.theta_parts_normal(G, r, np.empty((0, 3)), np.empty(0), yy, fx[11][4], K.GL_X, K.GL_W)
    return v + lP - fx[11][0] + lj[i], lP


@njit(cache=True)
def fixed_batch(P, lj, ok, lo, hi, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out):
    """loglike_batch for fixed GP parameters; a row whose transit reaches beyond the blocks takes the full filter."""
    n = P.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        v, lP = _fixed_row(P, lj, ok, lo, hi, i, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True, parallel=True)
def fixed_batch_par(P, lj, ok, lo, hi, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt, logP_out):
    """fixed_batch with the rows spread over numba's threads (identical results)."""
    n = P.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in prange(n):
        v, lP = _fixed_row(P, lj, ok, lo, hi, i, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True)
def _fixed_moments_row(P, fx, T, S, i, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    if _fits_blocks(P[i], t_ref, half, half_exp):
        G, r, yy = _fixed_normal(P[i], fx, ksup, tol, A_half, Gt)
        _moments(G, r, T, S, i)
        T[i, 0] += fx[11][4]
    else:
        _moments_row(P, E, T, S, i, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)


@njit(cache=True)
def fixed_moments_batch(P, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """theta_moments_batch for fixed GP parameters."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    for i in range(n):
        _fixed_moments_row(P, fx, T, S, i, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
    return T, S


@njit(cache=True, parallel=True)
def fixed_moments_batch_par(P, fx, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt):
    """fixed_moments_batch with the rows spread over numba's threads."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    for i in prange(n):
        _fixed_moments_row(P, fx, T, S, i, t_ref, half, half_exp, E, idx, steps, tt, ksup, wts, y, var0, tol, A_half, Gt)
    return T, S
