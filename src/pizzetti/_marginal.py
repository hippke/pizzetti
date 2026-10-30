"""Numba kernels for transit likelihoods with quadratic limb darkening and the
flux scale integrated out analytically (``pizzetti.MarginalTransitModel``).

Model. The unnormalised quadratic intensity in the degree-2 Bernstein basis,
    I(mu) = c0 (1-mu)^2 + c1 2 mu (1-mu) + c2 mu^2,
makes the observed flux linear in c for fixed geometry: m = A(theta) c. The
physical conditions (positive, darkening towards the limb; Kipping 2013) are
exactly the cone 0 <= c0 <= c1 <= c2. With a flat prior on c over the cone,
    log L(theta) = -chi2_min/2 - log det F / 2 + log P(c in cone)  (+ const),
F = A^T W A, where P is a trivariate normal orthant probability (Genz transform
on a fixed Gauss-Legendre grid; deterministic and smooth in theta).
"""
import math

import numpy as np
from numba import njit

from . import _orbit
from ._powbasis import alpha_column, columns_kernel
from ._physnl import log_prob_physical_line
from .terms import _occult_terms_cut

# 12-point Gauss-Legendre with the endpoint-smoothing map u = t^2 (3 - 2t)
_t, _w = np.polynomial.legendre.leggauss(12)
_t = 0.5 * (_t + 1.0)
GL_X = _t * _t * (3.0 - 2.0 * _t)
GL_W = 0.5 * _w * 6.0 * _t * (1.0 - _t)
DISK = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0])

# canonical parameter order of the kernels
CANON = ("t0", "per", "rp", "a", "b", "ecc", "w")

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


@njit(cache=True, inline="always")
def ndtr(x):
    return 0.5 * math.erfc(-x / _SQRT2)


@njit(cache=True)
def ndtri(p):
    """Inverse normal CDF: Acklam + one Halley step, lower tail (~1e-15)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    if p > 0.5:
        return -ndtri(1.0 - p)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    e = ndtr(x) - p
    u = e * _SQRT2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


@njit(cache=True)
def chol3(S, L):
    for i in range(3):
        for j in range(i + 1):
            s = S[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    return False
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return True


@njit(cache=True)
def log_ndtr(x):
    """log Phi(x), finite for all x (asymptotic series below x = -20)."""
    if x > -20.0:
        return math.log(0.5 * math.erfc(-x / _SQRT2))
    r = 1.0 / (x * x)
    return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + \
        math.log(1.0 - r + 3.0 * r * r - 15.0 * r * r * r)


@njit(cache=True)
def log_orthant_asym(m, S):
    """Asymptotic log P(w > 0), w ~ N(m, S), for P too small for the quadrature: the most
    violated constraint contributes log Phi(z), the others are conditioned on it lying on its
    boundary (where the tail mass concentrates), recursively. Finite, and decreasing with the
    violation, so that optimisers and samplers can climb out of a region where the least-squares
    coefficients are far outside the physical cone (the exact value underflows to 0 there)."""
    k = m.shape[0]
    mm = m.copy()
    SS = S.copy()
    alive = np.ones(k, np.bool_)
    out = 0.0
    for _ in range(k):
        i, zmin = -1, np.inf
        for j in range(k):
            if alive[j]:
                z = mm[j] / math.sqrt(SS[j, j])
                if z < zmin:
                    i, zmin = j, z
        out += log_ndtr(zmin)
        alive[i] = False
        sii = SS[i, i]
        for j in range(k):
            if alive[j]:
                mm[j] -= SS[j, i] / sii * mm[i]
        for j in range(k):
            for l_ in range(k):
                if alive[j] and alive[l_]:
                    SS[j, l_] -= SS[j, i] * SS[i, l_] / sii
        for j in range(k):
            if alive[j] and SS[j, j] <= 0.0:
                SS[j, j] = 1e-300
    return out


@njit(cache=True)
def log_orthant_prob(m, S, gx, gw):
    """log P(w > 0) for w ~ N(m, S), 3-d (Genz transform on a tensor GL grid).
    Constraints with marginal z > 6 are dropped while P is not tiny."""
    zs = np.empty(3)
    for i in range(3):
        zs[i] = m[i] / math.sqrt(S[i, i])
    if zs.min() > 6.0:
        return 0.0
    o = np.argsort(zs)
    mm = np.empty(3)
    SS = np.empty((3, 3))
    for i in range(3):
        mm[i] = m[o[i]]
        for j in range(3):
            SS[i, j] = S[o[i], o[j]]
    L = np.zeros((3, 3))
    if not chol3(SS, L):
        return -np.inf
    x1 = mm[0] / L[0, 0]
    if x1 < -37.0:
        return log_orthant_asym(m, S)
    e1 = ndtr(x1)
    n = gx.shape[0]
    zsorted = zs[o]
    if x1 > -3.0 and zsorted[1] > 6.0:
        return math.log(e1)
    if x1 > -3.0 and zsorted[2] > 6.0:
        acc = 0.0
        for i in range(n):
            z1 = -ndtri((1.0 - gx[i]) * e1)
            acc += gw[i] * ndtr((mm[1] + L[1, 0] * z1) / L[1, 1])
        return math.log(e1) + math.log(acc) if acc > 0.0 else log_orthant_asym(m, S)
    acc = 0.0
    for i in range(n):
        z1 = -ndtri((1.0 - gx[i]) * e1)
        e2 = ndtr((mm[1] + L[1, 0] * z1) / L[1, 1])
        if e2 <= 0.0:
            continue
        inner = 0.0
        for j in range(n):
            p = (1.0 - gx[j]) * e2
            if p < 1e-300:
                p = 1e-300
            z2 = -ndtri(p)
            inner += gw[j] * ndtr((mm[2] + L[2, 0] * z1 + L[2, 1] * z2) / L[2, 2])
        acc += gw[i] * e2 * inner
    if acc <= 0.0:
        return log_orthant_asym(m, S)
    return math.log(e1) + math.log(acc)


@njit(cache=True)
def inv3_chol(F, L):
    """Cholesky (in L) and inverse of a 3x3 SPD matrix: (ok, logdet, Finv)."""
    Finv = np.zeros((3, 3))
    if not chol3(F, L):
        return False, 0.0, Finv
    logdet = 2.0 * (math.log(L[0, 0]) + math.log(L[1, 1]) + math.log(L[2, 2]))
    Li = np.zeros((3, 3))
    for i in range(3):
        Li[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= L[i, k] * Li[k, j]
            Li[i, j] = s / L[i, i]
    for i in range(3):
        for j in range(i + 1):
            acc = 0.0
            for k in range(i, 3):
                acc += Li[k, i] * Li[k, j]
            Finv[i, j] = acc
            Finv[j, i] = acc
    return True, logdet, Finv


@njit(cache=True, inline="always")
def cone_ms(c0, c1, c2, Fi):
    """m = D c, S = D Fi D^T for the cone 0 <= c0 <= c1 <= c2."""
    m = np.empty(3)
    m[0], m[1], m[2] = c0, c1 - c0, c2 - c1
    S = np.empty((3, 3))
    S[0, 0] = Fi[0, 0]
    S[0, 1] = S[1, 0] = Fi[0, 1] - Fi[0, 0]
    S[0, 2] = S[2, 0] = Fi[0, 2] - Fi[0, 1]
    S[1, 1] = Fi[1, 1] - 2.0 * Fi[0, 1] + Fi[0, 0]
    S[1, 2] = S[2, 1] = Fi[1, 2] - Fi[1, 1] - Fi[0, 2] + Fi[0, 1]
    S[2, 2] = Fi[2, 2] - 2.0 * Fi[1, 2] + Fi[1, 1]
    return m, S


@njit(cache=True)
def normal_eqs(A, y):
    """A^T A and A^T y (A, y already noise-weighted)."""
    F = np.zeros((3, 3))
    r = np.zeros(3)
    f00 = f10 = f11 = f20 = f21 = f22 = 0.0
    r0 = r1 = r2 = 0.0
    for t in range(A.shape[0]):
        a0, a1, a2, yt = A[t, 0], A[t, 1], A[t, 2], y[t]
        f00 += a0 * a0
        f10 += a1 * a0
        f11 += a1 * a1
        f20 += a2 * a0
        f21 += a2 * a1
        f22 += a2 * a2
        r0 += a0 * yt
        r1 += a1 * yt
        r2 += a2 * yt
    F[0, 0], F[1, 1], F[2, 2] = f00, f11, f22
    F[1, 0] = F[0, 1] = f10
    F[2, 0] = F[0, 2] = f20
    F[2, 1] = F[1, 2] = f21
    r[0], r[1], r[2] = r0, r1, r2
    return F, r


@njit(cache=True)
def marg_loglike(A, y):
    """Marginal log-likelihood for weighted A, y (unit noise)."""
    F, r = normal_eqs(A, y)
    L = np.zeros((3, 3))
    ok, logdet, Fi = inv3_chol(F, L)
    if not ok:
        return -np.inf
    c0 = Fi[0, 0] * r[0] + Fi[0, 1] * r[1] + Fi[0, 2] * r[2]
    c1 = Fi[1, 0] * r[0] + Fi[1, 1] * r[1] + Fi[1, 2] * r[2]
    c2 = Fi[2, 0] * r[0] + Fi[2, 1] * r[1] + Fi[2, 2] * r[2]
    chi2 = 0.0
    for t in range(A.shape[0]):
        d = y[t] - A[t, 0] * c0 - A[t, 1] * c1 - A[t, 2] * c2
        chi2 += d * d
    m, S = cone_ms(c0, c1, c2, Fi)
    return -0.5 * chi2 - 0.5 * logdet + log_orthant_prob(m, S, GL_X, GL_W)


@njit(cache=True)
def _post_with_ldprior(F, r, mb, Pb):
    """Combine the Gaussian likelihood in c (precision F, F chat = r) with a
    Gaussian prior on beta(c) = (c0/c2, c1/c2) ~ N(mb, Pb^-1) (times 1/c2^2, i.e.
    N(beta) dbeta x flat c2). beta(c) is linearised around the posterior mode
    (Gauss-Newton, 4 iterations), so the prior becomes a rank-2 Gaussian in c;
    the residual error is second order in the relative spread of c2 (~1e-4 nats).
    Returns (ok, c*, F', r', extra) with the marginal
      -chi2_min/2 + extra - log det F'/2 + log P(cone | c*, F'^-1)."""
    L = np.zeros((3, 3))
    ok, ld, Fi = inv3_chol(F, L)
    chat = Fi @ r
    if not ok or not chat[2] > 0.0:
        return False, chat, F, r, 0.0
    # Gauss-Newton in the small offset delta = c - chat (no large cancelling terms:
    # chat^T F chat ~ 1e10 for typical light curves, cond(F) ~ 1e6)
    delta = np.zeros(3)
    c = chat.copy()
    Fp = F.copy()
    for _ in range(5):
        c2 = c[2]
        if not c2 > 0.0:
            return False, chat, F, r, 0.0
        J = np.zeros((2, 3))
        J[0, 0] = 1.0 / c2
        J[0, 2] = -c[0] / (c2 * c2)
        J[1, 1] = 1.0 / c2
        J[1, 2] = -c[1] / (c2 * c2)
        d = np.array([c[0] / c2, c[1] / c2]) - mb          # exact prior residual at c
        JtP = J.T @ Pb
        Fp = F + JtP @ J
        # minimise (delta')^T F delta' + (d + J (delta' - delta))^T Pb (...) over delta'
        g = JtP @ (d - J @ delta)
        ok2, ldp, Fpi = inv3_chol(Fp, L)
        if not ok2:
            return False, chat, F, r, 0.0
        delta = -(Fpi @ g)
        c = chat + delta
    c2 = c[2]
    if not c2 > 0.0:
        return False, chat, F, r, 0.0
    d = np.array([c[0] / c2, c[1] / c2]) - mb
    q = delta @ (F @ delta) + d @ (Pb @ d)
    rp_ = Fp @ c
    logdetPb = math.log(Pb[0, 0] * Pb[1, 1] - Pb[0, 1] * Pb[1, 0])
    return True, c, Fp, rp_, -0.5 * q + 0.5 * logdetPb - 2.0 * math.log(c2)


@njit(cache=True)
def marg_loglike_ldprior(A, y, mb, Pb):
    """marg_loglike with an (exact) Gaussian prior on (beta0, beta1)."""
    F, r = normal_eqs(A, y)
    ok, cs, Fp, rp_, extra = _post_with_ldprior(F, r, mb, Pb)
    if not ok:
        return -np.inf
    L = np.zeros((3, 3))
    ok2, logdet, Fpi = inv3_chol(Fp, L)
    Fi_ok, _, Fi = inv3_chol(F, L)
    chat = Fi @ r
    chi2 = 0.0
    for t in range(A.shape[0]):
        d = y[t] - A[t, 0] * chat[0] - A[t, 1] * chat[1] - A[t, 2] * chat[2]
        chi2 += d * d
    m, S = cone_ms(cs[0], cs[1], cs[2], Fpi)
    return -0.5 * chi2 + extra - 0.5 * logdet + log_orthant_prob(m, S, GL_X, GL_W)


@njit(cache=True)
def bernstein_cols(T):
    """n x 3 Bernstein design matrix from occult_terms output (lex, ldx, edx)."""
    n = T.shape[1]
    A = np.empty((n, 3))
    for i in range(n):
        lex, ldx, edx = T[0, i], T[1, i], T[2, i]
        A[i, 0] = 1.0 / 6.0 - (2.0 * lex - 2.0 * ldx - edx)
        A[i, 1] = 1.0 / 3.0 - 2.0 * (ldx - lex + edx)
        A[i, 2] = 0.5 - (lex - edx)
    return A


@njit(cache=True)
def design(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt):
    """Noise-weighted Bernstein design matrix (n x 3): orbit -> occult_terms ->
    columns -> exposure average over ksup sub-exposures -> rows times wts."""
    nt = tt.shape[0]
    z = np.empty(nt)
    ab = abs(b)
    inc = math.acos(min(ab / a, 1.0))
    if ecc < 1.0e-5:
        _orbit._rsky_circular(tt, t0, per, a, math.sin(inc), 0.25, 1, z)
    else:
        _orbit._rsky_eccentric(tt, t0, per, a, inc, ecc, w, 1, z)
    Af = bernstein_cols(_occult_terms_cut(z, abs(rp), tol, A_half, Gt))
    n = nt // ksup
    A = np.zeros((n, 3))
    inv = 1.0 / ksup
    for i in range(n):
        for s in range(ksup):
            r = i * ksup + s
            A[i, 0] += Af[r, 0]
            A[i, 1] += Af[r, 1]
            A[i, 2] += Af[r, 2]
        f = wts[i] * inv
        A[i, 0] *= f
        A[i, 1] *= f
        A[i, 2] *= f
    return A


@njit(cache=True, inline="always")
def a_logjac_from_T14(T14, per, rp, b):
    """a/R* of a circular orbit with total duration T14 and log|da/dT14|; (nan, 0) if invalid."""
    Q = (1.0 + rp) ** 2 - b * b
    x = math.pi * T14 / per
    if not (Q > 0.0 and 0.0 < x < 0.5 * math.pi):
        return math.nan, 0.0
    s = math.sin(x)
    a = math.sqrt(b * b + Q / (s * s))
    return a, math.log(Q * math.cos(x) * math.pi / (a * s * s * s * per))


@njit(cache=True)
def to_canonical(X, idx, fixed, t14):
    """Rows of free parameters (internal coordinates) -> canonical rows
    (t0, per, rp, a, b, ecc, w_rad) and log-Jacobian. t14: the 'a' slot holds T14."""
    n = X.shape[0]
    P = np.empty((n, 7))
    lj = np.zeros(n)
    for i in range(n):
        for k in range(7):
            P[i, k] = fixed[k]
        for j in range(idx.shape[0]):
            P[i, idx[j]] = X[i, j]
        if t14:
            a, l = a_logjac_from_T14(P[i, 3], P[i, 1], P[i, 2], abs(P[i, 4]))
            P[i, 3] = a
            lj[i] = l
    return P, lj


@njit(cache=True)
def loglike_batch(X, idx, fixed, t14, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt,
                  has_ldp=False, mb=np.zeros(2), Pb=np.eye(2)):
    """log-likelihood + uniform box prior (lo < canonical < hi; 0 <= b < 1 + rp;
    0 <= ecc < 1) + log-Jacobian of the T14 map, for every row of X."""
    P, lj = to_canonical(X, idx, fixed, t14)
    n = X.shape[0]
    out = np.empty(n)
    for i in range(n):
        ok = True
        for k in range(7):
            v = P[i, k]
            if not (v == v and lo[k] < v < hi[k]):
                ok = False
        t0, per, rp, a, b, ecc, w = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        if not (ok and 0.0 <= b < 1.0 + rp and 0.0 <= ecc < 1.0 and per > 0.0 and rp > 0.0
                and a > 1.0):
            out[i] = -np.inf
            continue
        A = design(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt)
        if has_ldp:
            out[i] = marg_loglike_ldprior(A, yw, mb, Pb) + lj[i]
        else:
            out[i] = marg_loglike(A, yw) + lj[i]
    return out


@njit(cache=True)
def c_moments_batch(P, tt, ksup, wts, yw, tol, A_half, Gt, has_ldp=False, mb=np.zeros(2),
                    Pb=np.eye(2)):
    """For canonical rows P: mean chat (n x 3) and covariance Fi (n x 3 x 3) of
    the (untruncated) Gaussian of c given the geometry."""
    n = P.shape[0]
    C = np.empty((n, 3))
    S = np.empty((n, 3, 3))
    L = np.zeros((3, 3))
    for i in range(n):
        A = design(tt, ksup, wts, P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5],
                   P[i, 6], tol, A_half, Gt)
        F, r = normal_eqs(A, yw)
        if has_ldp:
            okp, cs, Fp, rp_, _ = _post_with_ldprior(F, r, mb, Pb)
            if okp:
                F, r = Fp, rp_
        ok, ld, Fi = inv3_chol(F, L)
        for k in range(3):
            C[i, k] = Fi[k, 0] * r[0] + Fi[k, 1] * r[1] + Fi[k, 2] * r[2]
            for j in range(3):
                S[i, k, j] = Fi[k, j]
    return C, S


# ----------------------------------------------------------------------------
# General K-coefficient kernels (all laws other than the fast quadratic path)
# ----------------------------------------------------------------------------
@njit(cache=True)
def chol_gen(F):
    """Cholesky of a small SPD matrix; returns (ok, L)."""
    K = F.shape[0]
    L = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1):
            s = F[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    return False, L
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return True, L


@njit(cache=True)
def inv_from_chol(L):
    K = L.shape[0]
    Li = np.zeros((K, K))
    for i in range(K):
        Li[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= L[i, k] * Li[k, j]
            Li[i, j] = s / L[i, i]
    return Li.T @ Li


@njit(cache=True)
def log_orthant_prob2(m, S, gx, gw):
    """log P(w > 0), w ~ N(m, S), 2-d (Genz, one-dimensional quadrature)."""
    s0, s1 = math.sqrt(S[0, 0]), math.sqrt(S[1, 1])
    z0, z1 = m[0] / s0, m[1] / s1
    if min(z0, z1) > 6.0:
        return 0.0
    if z1 < z0:                                   # most restrictive first
        m = np.array([m[1], m[0]])
        S = np.array([[S[1, 1], S[1, 0]], [S[0, 1], S[0, 0]]])
        z0, z1 = z1, z0
    l00 = math.sqrt(S[0, 0])
    l10 = S[1, 0] / l00
    l11 = math.sqrt(max(S[1, 1] - l10 * l10, 1e-300))
    x1 = m[0] / l00
    if x1 < -37.0:
        return log_orthant_asym(m, S)
    e1 = ndtr(x1)
    if x1 > -3.0 and z1 > 6.0:
        return math.log(e1)
    acc = 0.0
    for i in range(gx.shape[0]):
        zz = -ndtri((1.0 - gx[i]) * e1)
        acc += gw[i] * ndtr((m[1] + l10 * zz) / l11)
    return math.log(e1) + math.log(acc) if acc > 0.0 else log_orthant_asym(m, S)


@njit(cache=True)
def _qr_solve(A, y):
    """Least squares for y = A c via column equilibration + QR (no normal equations,
    which square the condition number). Returns (ok, c, Finv, logdetF, chi2)."""
    n, K = A.shape
    sc = np.empty(K)
    As = np.empty((n, K))
    for k in range(K):
        nrm = 0.0
        for t in range(n):
            nrm += A[t, k] * A[t, k]
        nrm = math.sqrt(nrm)
        if not nrm > 0.0:
            return False, np.zeros(K), np.eye(K), 0.0, 0.0
        sc[k] = nrm
        for t in range(n):
            As[t, k] = A[t, k] / nrm
    Q, R = np.linalg.qr(As)
    logdet = 0.0
    for k in range(K):
        if not abs(R[k, k]) > 1e-14:
            return False, np.zeros(K), np.eye(K), 0.0, 0.0
        logdet += 2.0 * (math.log(abs(R[k, k])) + math.log(sc[k]))
    Ri = np.linalg.inv(R)
    cs = Ri @ (Q.T @ y)
    c = cs / sc
    Fis = Ri @ Ri.T
    Fi = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            Fi[i, j] = Fis[i, j] / (sc[i] * sc[j])
    chi2 = 0.0
    for t in range(n):
        d = y[t]
        for k in range(K):
            d -= A[t, k] * c[k]
        chi2 += d * d
    return True, c, Fi, logdet, chi2


@njit(cache=True, fastmath=True)
def _dot(a, b):
    s = 0.0
    for t in range(a.shape[0]):
        s += a[t] * b[t]
    return s


@njit(cache=True)
def _ne_solve(AT, y):
    """As _qr_solve, for the transposed design AT (K x n, rows contiguous), via
    equilibrated normal equations (Cholesky) plus one step of iterative refinement;
    chi2 from the explicit residual (no y'y - r'F^-1 r cancellation). Falls back to
    QR if F is not safely positive definite (pivot ratio < 1e-6, cond(F) > ~1e12)."""
    K, n = AT.shape
    sc = np.zeros(K)
    F = np.zeros((K, K))
    r = np.zeros(K)
    for i in range(K):
        r[i] = _dot(AT[i], y)
        for j in range(i + 1):
            F[i, j] = _dot(AT[i], AT[j])
    for i in range(K):
        if not F[i, i] > 0.0:
            return _qr_solve(np.ascontiguousarray(AT.T), y)
        sc[i] = math.sqrt(F[i, i])
    for i in range(K):
        r[i] /= sc[i]
        for j in range(i + 1):
            F[i, j] /= sc[i] * sc[j]
            F[j, i] = F[i, j]
    ok, L = chol_gen(F)
    if not ok:
        return _qr_solve(np.ascontiguousarray(AT.T), y)
    dmin = L[0, 0]
    dmax = L[0, 0]
    for i in range(K):
        dmin = min(dmin, L[i, i])
        dmax = max(dmax, L[i, i])
    if dmin < 1e-6 * dmax:
        return _qr_solve(np.ascontiguousarray(AT.T), y)
    Fis = inv_from_chol(L)
    cs = Fis @ r
    c = cs / sc
    # refinement: residual d = y - A c, correction delta = F^-1 A' d
    d = y.copy()
    for i in range(K):
        ci = c[i]
        a = AT[i]
        for t in range(n):
            d[t] -= a[t] * ci
    g = np.zeros(K)
    for i in range(K):
        g[i] = _dot(AT[i], d) / sc[i]
    delta = Fis @ g
    chi2 = _dot(d, d) - delta @ g          # |y - A (c + delta)|^2 = |d|^2 - delta' F delta
    logdet = 0.0
    for i in range(K):
        logdet += 2.0 * (math.log(L[i, i]) + math.log(sc[i]))
    Fi = np.empty((K, K))
    for i in range(K):
        c[i] = (cs[i] + delta[i]) / sc[i]
        for j in range(K):
            Fi[i, j] = Fis[i, j] / (sc[i] * sc[j])
    return True, c, Fi, logdet, chi2


@njit(cache=True)
def marg_loglike_genT(AT, y, D, cone):
    """Marginal log-likelihood for y = A c + unit noise (AT = A transposed, K x n),
    flat prior on c over the cone D c >= 0 (cone=True) or unconstrained; K <= 5."""
    K = AT.shape[0]
    ok, c, Fi, logdet, chi2 = _ne_solve(AT, y)
    if not ok:
        return -np.inf
    lp = -0.5 * chi2 - 0.5 * logdet
    if not cone:
        return lp
    m = D @ c
    S = D @ Fi @ D.T
    if K == 1:
        z = m[0] / math.sqrt(S[0, 0])
        return lp + (0.0 if z > 8.0 else math.log(max(ndtr(z), 1e-300)))
    if K == 2:
        return lp + log_orthant_prob2(m, S, GL_X, GL_W)
    return lp + log_orthant_prob(m, S, GL_X, GL_W)


@njit(cache=True)
def marg_loglike_gen(A, y, D, cone):
    """As marg_loglike_genT for A (n x K)."""
    return marg_loglike_genT(np.ascontiguousarray(A.T), y, D, cone)


@njit(cache=True)
def moments_gen(A, y):
    """(chat, F^-1) of the untruncated Gaussian of c (A, y noise-weighted)."""
    ok, c, Fi, logdet, chi2 = _ne_solve(np.ascontiguousarray(A.T), y)
    return c, Fi


@njit(cache=True)
def gibbs_cone_gen(mu, cov, D, rng_u, sweeps):
    """One draw of c ~ N(mu, cov) truncated to D c >= 0 (D invertible, K <= 3):
    Gibbs in w = D c with inverse-CDF truncated-normal updates."""
    K = mu.shape[0]
    m = D @ mu
    S = D @ cov @ D.T
    ok, L = chol_gen(S)
    P = inv_from_chol(L)
    w = np.empty(K)
    for i in range(K):
        w[i] = max(m[i], 1e-12 * math.sqrt(S[i, i]))
    k = 0
    for _ in range(sweeps):
        for i in range(K):
            sd = 1.0 / math.sqrt(P[i, i])
            mi = m[i]
            for j in range(K):
                if j != i:
                    mi -= P[i, j] * (w[j] - m[j]) / P[i, i]
            lo = ndtr((0.0 - mi) / sd)
            u = lo + rng_u[k] * (1.0 - lo)
            k += 1
            u = min(max(u, 1e-300), 1.0 - 1e-16)
            w[i] = mi + sd * ndtri(u)
    return np.linalg.solve(D, w)


# ----------------------------------------------------------------------------
# Fused general path: all laws with fixed exponents, and secondary eclipses
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def gen_design(tt, ksup, wts, t0, per, rp, a, b, ecc, w, ttype, tol, A_half, Gt,
               dvec, W, gams, sertabs, ptol, cheb, lens, valid, ycs, plo, phi, ju, jw, has,
               amode, alpha, cheb2, lens2, valid2, yc2, plo2, phi2, alo2, ahi2, nreg2):
    """Noise-weighted design matrix, transposed (K x n), of one geometry:
    primary: A = d - [lex, ldx, edx, P_g...] @ W (occult_terms + _powbasis columns);
    secondary (ttype 2): A = [1, v - 1] with v the visible planet fraction."""
    m = tt.shape[0]
    inc = math.acos(min(abs(b) / a, 1.0))
    z = _orbit.rsky(tt, t0, per, a, inc, ecc, w, ttype)
    K = dvec.shape[0]
    Af = np.empty((K, m))
    if ttype == 2:
        v = _orbit.eclipse(z, abs(rp), 1.0)
        for r in range(m):
            Af[0, r] = 1.0
            Af[1, r] = v[r] - 1.0
    else:
        T = _occult_terms_cut(z, abs(rp), tol, A_half, Gt)
        npw = 1 if amode else gams.shape[0]
        for j in range(K):
            dj, w0, w1, w2 = dvec[j], W[0, j], W[1, j], W[2, j]
            row = Af[j]
            T0 = T[0]
            T1 = T[1]
            T2 = T[2]
            for r in range(m):
                row[r] = dj - T0[r] * w0 - T1[r] * w1 - T2[r] * w2
        if npw > 0:
            Pc = np.empty((npw, m))
            if amode:
                alpha_column(z, abs(rp), alpha, ptol, Gt, cheb2, lens2, valid2, yc2, plo2, phi2,
                             alo2, ahi2, nreg2, Pc, 0)
            else:
                columns_kernel(z, abs(rp), gams, sertabs, ptol, Gt, cheb, lens, valid, ycs, plo,
                               phi, ju, jw, has, Pc, 0)
            for q in range(npw):
                pq = Pc[q]
                for j in range(K):
                    wq = W[3 + q, j]
                    if wq != 0.0:
                        row = Af[j]
                        for r in range(m):
                            row[r] -= pq[r] * wq
    if ksup == 1:
        for j in range(K):
            row = Af[j]
            for r in range(m):
                row[r] *= wts[r]
        return Af
    n = m // ksup
    A = np.zeros((K, n))
    inv = 1.0 / ksup
    for j in range(K):
        for i in range(n):
            s_ = 0.0
            for s in range(ksup):
                s_ += Af[j, i * ksup + s]
            A[j, i] = s_ * inv * wts[i]
    return A


@njit(cache=True)
def _phase_ecl(TA, e):
    E = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(TA / 2.0))
    return (E - e * math.sin(E)) / (2.0 * math.pi)


@njit(cache=True)
def t0_from_tsec(tsec, per, ecc, w):
    """Time of inferior conjunction from the time of secondary eclipse (batman)."""
    return tsec + per * (_phase_ecl(math.pi / 2.0 - w, ecc) - _phase_ecl(1.5 * math.pi - w, ecc))


@njit(cache=True)
def gen_batch(P, lj, al, check, lo, hi, has_alpha, alo, ahi, rowpan, tt, ksup, wts, yw, ttype,
              tol, A_half, Gt, dvec, W, gams, sertabs, ptol, CHEB, LENS, VALID, YCS, PLO, PHI,
              ju, jw, has, D, cone, moments, amode, rowpan2, CH2, LE2, VA2, YC2, PL2, PH2, AL2,
              AH2, NR2, curved, Z4, B3, Zbig):
    """amode: power-2 law with free alpha (al), 2-D panels (rowpan2 into CH2 ...).
    For every canonical row of P (log-Jacobian lj, exponent al, panel index rowpan
    into the stacked panels): the marginal log-likelihood + lj, -inf outside the box
    prior (check=True; as loglike_batch, plus alo < al < ahi if has_alpha), or the
    moments (chat, F^-1) of c (moments=True). Secondary eclipses (ttype 2): P[:, 0] is
    the time of secondary eclipse."""
    n = P.shape[0]
    K = dvec.shape[0]
    out = np.full(n, -np.inf)
    C = np.zeros((n, K))
    S = np.zeros((n, K, K))
    for i in range(n):
        ok = True
        for k in range(7):
            v = P[i, k]
            if not v == v:
                ok = False
            elif check and not (lo[k] < v < hi[k]):
                ok = False
        t0, per, rp, a, b, ecc, w = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        if check and not (ok and 0.0 <= b < 1.0 + rp and 0.0 <= ecc < 1.0 and per > 0.0
                          and rp > 0.0 and a > 1.0
                          and (not has_alpha or alo < al[i] < ahi)):
            ok = False
        if not ok:
            continue
        if ttype == 2:
            t0 = t0_from_tsec(t0, per, ecc, w)
        ip = rowpan[i]
        j = ip if ip >= 0 else 0
        plo_ = PLO[j] if ip >= 0 else -1.0
        q = rowpan2[i] if amode else -1
        j2 = q if q >= 0 else 0
        va2 = VA2[j2] and q >= 0
        pl2 = PL2[j2] if q >= 0 else -1.0
        al_ = al[i]
        dv = dvec
        if amode:
            dv = np.array([al_ / (al_ + 2.0), 2.0 / (al_ + 2.0)])
        A = gen_design(tt, ksup, wts, t0, per, rp, a, b, ecc, w, ttype, tol, A_half, Gt, dv,
                       W, gams, sertabs, ptol, CHEB[j], LENS[j], VALID[j], YCS[j], plo_, PHI[j],
                       ju, jw, has, amode, al_, CH2[j2], LE2[j2], va2, YC2[j2], pl2, PH2[j2],
                       AL2[j2], AH2[j2], NR2[j2])
        if moments:
            okc, c, Fi, ld_, ch_ = _ne_solve(A, yw)
            C[i] = c
            S[i] = Fi
        elif curved:
            okc, c, Fi, ld_, ch_ = _ne_solve(A, yw)
            if okc:
                out[i] = -0.5 * ch_ - 0.5 * ld_ + log_prob_physical_line(c, Fi, Z4, B3, Zbig) + lj[i]
        else:
            out[i] = marg_loglike_genT(A, yw, D, cone) + lj[i]
    return out, C, S
