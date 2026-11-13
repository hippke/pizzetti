"""Numba kernels for transit likelihoods with quadratic limb darkening and the
flux scale integrated out analytically (``pizzetti.MarginalTransitModel``).

Model. The unnormalised quadratic intensity in the degree-2 Bernstein basis,
    I(mu) = c0 (1-mu)^2 + c1 2 mu (1-mu) + c2 mu^2,
makes the observed flux linear in c for fixed geometry: m = A(theta) c. The
physical conditions (positive, darkening towards the limb; Kipping 2013) are
exactly the cone 0 <= c0 <= c1 <= c2. With a flat prior on c over the cone,
    log L(theta) = -chi2_min/2 - log det F / 2 + log P(c in cone)  (+ const),
F = A^T W A, where P is a trivariate normal orthant probability. The quadratic fast path
solves in theta = (f0, c1, c2) (design_theta, marg_theta_parts; well conditioned for any
transit depth) and evaluates the cone term by a ray integral, or by the Genz transform on a
fixed Gauss-Legendre grid where the Gaussian lies far inside the cone (P = 1), blended
smoothly; deterministic and smooth in theta.
"""
import math

import numpy as np
from numba import njit, prange

from ._cachecheck import FASTMATH
from . import _orbit
from ._powbasis import alpha_column, columns_kernel
from ._physnl import log_prob_physical_line
from .terms import _occult_terms, _occult_terms_cut, _occult_terms_into, terms_cut

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
    v, lP = marg_loglike_parts(A, y)
    return v + lP


@njit(cache=True)
def marg_loglike_parts(A, y):
    """(marginal log-likelihood without the cone, log P of the cone) for weighted A, y: the
    likelihood with a flat prior over all coefficients, and the log probability of the
    physical cone under the conditional Gaussian (log P <= 0)."""
    F, r = normal_eqs(A, y)
    L = np.zeros((3, 3))
    ok, logdet, Fi = inv3_chol(F, L)
    if not ok:
        return -np.inf, 0.0
    c0 = Fi[0, 0] * r[0] + Fi[0, 1] * r[1] + Fi[0, 2] * r[2]
    c1 = Fi[1, 0] * r[0] + Fi[1, 1] * r[1] + Fi[1, 2] * r[2]
    c2 = Fi[2, 0] * r[0] + Fi[2, 1] * r[1] + Fi[2, 2] * r[2]
    chi2 = 0.0
    for t in range(A.shape[0]):
        d = y[t] - A[t, 0] * c0 - A[t, 1] * c1 - A[t, 2] * c2
        chi2 += d * d
    m, S = cone_ms(c0, c1, c2, Fi)
    return -0.5 * chi2 - 0.5 * logdet, log_orthant_prob(m, S, GL_X, GL_W)


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


# ----------------------------------------------------------------------------
# Quadratic law in theta = (f0, c1, c2): well-conditioned normal equations and the cone integral
# ----------------------------------------------------------------------------
# With A_k = kappa_k - D_k (kappa = 1/6, 1/3, 1/2: the disk-integrated basis; D_k the deficits,
# exactly 0 out of transit) and f0 = kappa . c (the out-of-transit flux), c0 = 6 f0 - 2 c1 - 3 c2,
#     A c = B theta,  B = [1 - 6 D0, 2 D0 - D1, 3 D0 - D2],  theta = (f0, c1, c2).
# The columns 2 and 3 of B vanish out of transit and are computed from the occultation terms without
# cancellation; with the columns scaled to unit norm the normal equations are well conditioned for any
# depth (in c, cond(F) ~ 1 / rp^4: 0.07 nats of noise in log L for Kepler-78 b, -inf below rp ~ 3e-4).
# The cone integral I = int_cone exp(-(theta - th)^T G (theta - th) / 2) dtheta is evaluated along rays
# c = f0 v: f0 analytically (a truncated second moment), v over the fixed triangle T1 (the cone at
# f0 = 1: vertices (0, 2), (1, 1), (6/5, 6/5) in (v1, v2)) by Gauss-Legendre in the principal axes of
# the Gaussian approximation of the integrand, outer axis the one along which T1 is longer (in sd).
# log L = -chi2 / 2 + log I + log 6 - 3/2 log 2 pi; it tends to a finite limit as rp -> 0.
_LOG2PI = math.log(2.0 * math.pi)
_TRI = np.array([[0.0, 2.0], [1.0, 1.0], [1.2, 1.2]])


@njit(cache=True)
def theta_cols(T):
    """n x 3 design in theta = (f0, c1, c2) from occult_terms output (lex, ldx, edx)."""
    n = T.shape[1]
    B = np.empty((n, 3))
    for i in range(n):
        lex, ldx, edx = T[0, i], T[1, i], T[2, i]
        B[i, 0] = 1.0 - (12.0 * lex - 12.0 * ldx - 6.0 * edx)
        B[i, 1] = 6.0 * lex - 6.0 * ldx - 4.0 * edx
        B[i, 2] = 5.0 * lex - 6.0 * ldx - 2.0 * edx
    return B


@njit(cache=True)
def _log_m2(alpha, mu):
    """log int_0^inf x^2 exp(-alpha (x - mu)^2 / 2) dx."""
    sd = 1.0 / math.sqrt(alpha)
    z = mu / sd
    if z > 8.0:
        return 0.5 * _LOG2PI + math.log(sd) + math.log(mu * mu + sd * sd)
    if z < -8.0:                     # (z^2 + 1) Phi(z) + z phi(z) ~ 2 phi(z) / |z|^3
        return 0.5 * _LOG2PI + 3.0 * math.log(sd) - 0.5 * z * z - 0.5 * _LOG2PI + math.log(2.0) - 3.0 * math.log(-z)
    R = (z * z + 1.0) * ndtr(z) + z * math.exp(-0.5 * z * z) / _SQRT2PI
    return 0.5 * _LOG2PI + 3.0 * math.log(sd) + math.log(R)


@njit(cache=True, inline="always")
def _lt_mv(L, s, x0, x1, x2):
    """L^T (s * x) for the lower-triangular 3 x 3 L."""
    y0, y1, y2 = s[0] * x0, s[1] * x1, s[2] * x2
    return (L[0, 0] * y0 + L[1, 0] * y1 + L[2, 0] * y2, L[1, 1] * y1 + L[2, 1] * y2, L[2, 2] * y2)


@njit(cache=True)
def _ray_log_g(v1, v2, L, s, eh1, eh2, A0, A1, A2):
    """log of the f0 integral along the ray c = f0 (1, v1, v2) (see above)."""
    b0, b1, b2 = _lt_mv(L, s, 1.0, v1, v2)
    d0, d1, d2 = _lt_mv(L, s, 0.0, v1 - eh1, v2 - eh2)
    alpha = b0 * b0 + b1 * b1 + b2 * b2
    mu = (b0 * A0 + b1 * A1 + b2 * A2) / alpha
    c0 = A1 * d2 - A2 * d1
    c1 = A2 * d0 - A0 * d2
    c2 = A0 * d1 - A1 * d0
    return -0.5 * (c0 * c0 + c1 * c1 + c2 * c2) / alpha + _log_m2(alpha, mu)


@njit(cache=True)
def _nodes(lo, hi, gx, gw, x, lw):
    """Nodes x and log weights lw for int_lo^hi f(z) dz against the standard normal: Gauss-Legendre
    on [lo, hi] where the normal is broad there, else substitution by its CDF (lw includes the
    1 / phi(z)), and by the exponential tangent to it for a window beyond 8 sd."""
    n = gx.shape[0]
    if hi - lo < 2.5:
        for k in range(n):
            x[k] = lo + (hi - lo) * gx[k]
            lw[k] = math.log((hi - lo) * gw[k])
        return
    flip = lo > 0.0
    za, zb = (-hi, -lo) if flip else (lo, hi)
    if zb < -8.0:
        lam = -zb
        E = -math.expm1(-lam * (zb - za))
        for k in range(n):
            u = -math.log1p(-gx[k] * E) / lam
            z = zb - u
            x[k] = -z if flip else z
            lw[k] = math.log(gw[k] * E / lam) + lam * u
        return
    pa, pb = ndtr(za), ndtr(zb)
    la, lb = log_ndtr(za), log_ndtr(zb)
    lmass = lb + math.log1p(-math.exp(la - lb))
    for k in range(n):
        z = ndtri(pa + gx[k] * (pb - pa))
        x[k] = -z if flip else z
        lw[k] = math.log(gw[k]) + lmass + 0.5 * _LOG2PI + 0.5 * z * z


@njit(cache=True, inline="always")
def _lse_add(m, acc, v):
    """Running log-sum-exp: (max, sum of exp(. - max)) after adding v."""
    if v == -np.inf:
        return m, acc
    if v > m:
        return v, acc * math.exp(m - v) + 1.0
    return m, acc + math.exp(v - m)


@njit(cache=True)
def _outer_centre(Z, oa, ia):
    """Outer coordinate of the point of the triangle (vertices Z, whitened about the Gaussian's mean)
    nearest to the mean; 0 for a mean inside the triangle. The cone integrand concentrates there."""
    sign0, inside = 0.0, True
    for i in range(3):
        j = (i + 1) % 3
        cr = Z[i, oa] * Z[j, ia] - Z[i, ia] * Z[j, oa]
        if i == 0:
            sign0 = cr
        elif cr * sign0 < 0.0:
            inside = False
    if inside:
        return 0.0
    ts, best = 0.0, np.inf
    for i in range(3):
        j = (i + 1) % 3
        dt, du = Z[j, oa] - Z[i, oa], Z[j, ia] - Z[i, ia]
        ll = dt * dt + du * du
        if ll <= 0.0:
            continue
        al = min(max(-(Z[i, oa] * dt + Z[i, ia] * du) / ll, 0.0), 1.0)
        pt, pu = Z[i, oa] + al * dt, Z[i, ia] + al * du
        if pt * pt + pu * pu < best:
            best, ts = pt * pt + pu * pu, pt
    return ts


@njit(cache=True)
def log_cone_integral(L, s, Gs, th, gx, gw):
    """log int_cone exp(-(theta - th)^T G (theta - th) / 2) dtheta (G = s Gs s, Gs = L L^T)."""
    A0, A1, A2 = _lt_mv(L, s, th[0], th[1], th[2])
    eh1, eh2 = th[1] / th[0], th[2] / th[0]
    es0, es1, es2 = s[0], s[1] * eh1, s[2] * eh2
    g0 = Gs[0, 0] * es0 + Gs[0, 1] * es1 + Gs[0, 2] * es2
    g1 = Gs[1, 0] * es0 + Gs[1, 1] * es1 + Gs[1, 2] * es2
    g2 = Gs[2, 0] * es0 + Gs[2, 1] * es1 + Gs[2, 2] * es2
    ege = es0 * g0 + es1 * g1 + es2 * g2
    f2 = th[0] * th[0]
    pa = f2 * s[1] * s[1] * (Gs[1, 1] - g1 * g1 / ege)
    pb = f2 * s[1] * s[2] * (Gs[1, 2] - g1 * g2 / ege)
    pc = f2 * s[2] * s[2] * (Gs[2, 2] - g2 * g2 / ege)
    phi = 0.5 * math.atan2(2.0 * pb, pa - pc)
    cs, sn = math.cos(phi), math.sin(phi)
    lam0 = max(pa * cs * cs + 2.0 * pb * cs * sn + pc * sn * sn, 1e-300)
    lam1 = max(pa * sn * sn - 2.0 * pb * cs * sn + pc * cs * cs, 1e-300)
    # columns of U: (cs, sn), (-sn, cs); whitened z_k = sqrt(lam_k) U[:, k] . (v - vh)
    u00, u10, u01, u11 = cs, sn, -sn, cs
    q0, q1 = math.sqrt(lam0), math.sqrt(lam1)
    Z = np.empty((3, 2))
    for i in range(3):
        dv1, dv2 = _TRI[i, 0] - eh1, _TRI[i, 1] - eh2
        Z[i, 0] = q0 * (u00 * dv1 + u10 * dv2)
        Z[i, 1] = q1 * (u01 * dv1 + u11 * dv2)
    ldet = -0.5 * (math.log(lam0) + math.log(lam1))
    oa = 0                                   # outer axis: the longer extent of the triangle
    if max(Z[0, 1], Z[1, 1], Z[2, 1]) - min(Z[0, 1], Z[1, 1], Z[2, 1]) > \
            max(Z[0, 0], Z[1, 0], Z[2, 0]) - min(Z[0, 0], Z[1, 0], Z[2, 0]):
        oa = 1
    ia = 1 - oa
    # vertices sorted by the outer coordinate
    i0, i1, i2 = 0, 1, 2
    if Z[i0, oa] > Z[i1, oa]:
        i0, i1 = i1, i0
    if Z[i1, oa] > Z[i2, oa]:
        i1, i2 = i2, i1
    if Z[i0, oa] > Z[i1, oa]:
        i0, i1 = i1, i0
    # the outer rule is centred on the integrand's mass: for a mean outside the triangle that is the
    # nearest point of the triangle, not the mean (the integrand itself is evaluated exactly)
    ts = _outer_centre(Z, oa, ia)
    n = gx.shape[0]
    xt, lwt, xu, lwu = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    m_all, acc_all = -np.inf, 0.0
    for piece in range(2):
        ja, jb = (i0, i1) if piece == 0 else (i1, i2)
        ta, tb = Z[ja, oa], Z[jb, oa]
        if not tb > ta:
            continue
        _nodes(ta - ts, tb - ts, gx, gw, xt, lwt)
        for k in range(n):
            t = ts + xt[k]
            # cross-section: the edge (ja, jb) and the long edge (i0, i2)
            ua = Z[ja, ia] + (Z[jb, ia] - Z[ja, ia]) * (t - ta) / (tb - ta)
            ub = Z[i0, ia] + (Z[i2, ia] - Z[i0, ia]) * (t - Z[i0, oa]) / (Z[i2, oa] - Z[i0, oa])
            lo, hi = min(ua, ub), max(ua, ub)
            if not hi > lo:
                continue
            _nodes(lo, hi, gx, gw, xu, lwu)
            m_in, acc_in = -np.inf, 0.0
            for j in range(n):
                zz0 = t if oa == 0 else xu[j]
                zz1 = xu[j] if oa == 0 else t
                w0, w1 = zz0 / q0, zz1 / q1
                v1 = eh1 + u00 * w0 + u01 * w1
                v2 = eh2 + u10 * w0 + u11 * w1
                m_in, acc_in = _lse_add(m_in, acc_in, lwu[j] + _ray_log_g(v1, v2, L, s, eh1, eh2, A0, A1, A2))
            if acc_in > 0.0:
                m_all, acc_all = _lse_add(m_all, acc_all, lwt[k] + m_in + math.log(acc_in))
    if not acc_all > 0.0:
        return -np.inf
    return ldet + m_all + math.log(acc_all)


@njit(cache=True)
def marg_theta_parts(B, y, gx, gw):
    """(marginal log-likelihood without the cone, log P of the cone) for the weighted theta design
    B and data y, as marg_loglike_parts (their sum is the marginal log-likelihood)."""
    G, r = normal_eqs(B, y)
    return theta_parts_normal(G, r, B, y, 0.0, 0.0, gx, gw)


@njit(cache=True)
def theta_parts_normal(G, r, B, y, yy, shift, gx, gw):
    """marg_theta_parts from the normal equations. With rows in B, chi2 comes from the explicit residual.
    Without rows (the sums were built elsewhere): chi2 = yy - 2 th r + th G th, where r and yy belong to the data
    minus shift x the first column of the design, so that the solution for f0 is the shift + the first solved value."""
    s = np.empty(3)
    for i in range(3):
        if not G[i, i] > 0.0:
            return -np.inf, 0.0
        s[i] = math.sqrt(G[i, i])
    Gs = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            Gs[i, j] = G[i, j] / (s[i] * s[j])
    L = np.zeros((3, 3))
    ridge = 0.0
    if not chol3(Gs, L) or L[2, 2] * L[2, 2] < 1e-12:
        # (nearly) collinear deficit columns (an extreme grazing geometry: the data constrain one
        # combination of c1, c2): an extremely weak Gaussian on the scaled coefficients (precision
        # 1e-9 against the unit diagonal) keeps G invertible; over the bounded cone its effect is
        # negligible, and it is included in the quadratic form (chi2 below)
        ridge = 1e-9
        for i in range(3):
            Gs[i, i] += ridge
        L[:, :] = 0.0
        if not chol3(Gs, L):
            return -np.inf, 0.0
    # theta = s^-1 Gs^-1 s^-1 r
    b0, b1, b2 = r[0] / s[0], r[1] / s[1], r[2] / s[2]
    z0 = b0 / L[0, 0]
    z1 = (b1 - L[1, 0] * z0) / L[1, 1]
    z2 = (b2 - L[2, 0] * z0 - L[2, 1] * z1) / L[2, 2]
    x2 = z2 / L[2, 2]
    x1 = (z1 - L[2, 1] * x2) / L[1, 1]
    x0 = (z0 - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]
    th = np.array([x0 / s[0], x1 / s[1], x2 / s[2]])
    if B.shape[0] > 0:
        chi2 = 0.0
        if not th[0] > 0.0:
            return -np.inf, 0.0
        for t in range(B.shape[0]):
            d = y[t] - B[t, 0] * th[0] - B[t, 1] * th[1] - B[t, 2] * th[2]
            chi2 += d * d
    else:
        chi2 = yy
        for i in range(3):
            chi2 -= 2.0 * th[i] * r[i]
            for j in range(3):
                chi2 += th[i] * G[i, j] * th[j]
        th[0] += shift
        if not th[0] > 0.0:
            return -np.inf, 0.0
    for i in range(3):
        chi2 += ridge * (s[i] * th[i]) ** 2
    logdetG = 2.0 * (math.log(L[0, 0]) + math.log(L[1, 1]) + math.log(L[2, 2])) + \
        2.0 * (math.log(s[0]) + math.log(s[1]) + math.log(s[2]))
    v = -0.5 * chi2 - 0.5 * logdetG + math.log(6.0)
    # log P of the cone: the ray integral, except where every constraint holds by more than 6 sd (P = 1
    # to 1e-9), where the Genz orthant probability (log_orthant_prob, its moments from the theta
    # solution) is exact and far cheaper; blended smoothly from 4.5 sd
    ok, ld_, Gsi = inv3_chol(Gs, np.zeros((3, 3)))
    St = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            St[i, j] = Gsi[i, j] / (s[i] * s[j])
    c0 = 6.0 * th[0] - 2.0 * th[1] - 3.0 * th[2]
    # Fi in c = Mi St Mi^T, Mi = [[6, -2, -3], [0, 1, 0], [0, 0, 1]]
    r0 = np.array([6.0 * St[0, 0] - 2.0 * St[1, 0] - 3.0 * St[2, 0],
                   6.0 * St[0, 1] - 2.0 * St[1, 1] - 3.0 * St[2, 1],
                   6.0 * St[0, 2] - 2.0 * St[1, 2] - 3.0 * St[2, 2]])
    Fi = np.empty((3, 3))
    Fi[0, 0] = 6.0 * r0[0] - 2.0 * r0[1] - 3.0 * r0[2]
    Fi[0, 1] = Fi[1, 0] = r0[1]
    Fi[0, 2] = Fi[2, 0] = r0[2]
    Fi[1, 1], Fi[1, 2], Fi[2, 1], Fi[2, 2] = St[1, 1], St[1, 2], St[2, 1], St[2, 2]
    m, S = cone_ms(c0, th[1], th[2], Fi)
    zmin = np.inf
    for i in range(3):
        zmin = min(zmin, m[i] / math.sqrt(S[i, i])) if S[i, i] > 0.0 else -np.inf
    w = min(max((zmin - 4.5) / 1.5, 0.0), 1.0)
    w = w * w * (3.0 - 2.0 * w)
    lG = log_orthant_prob(m, S, GL_X, GL_W) if w > 0.0 else -np.inf
    if lG == -np.inf:                     # S rounded to singular: the ray integral does not need it
        w = 0.0
    lP = 0.0
    if w < 1.0:
        lP += (1.0 - w) * (log_cone_integral(L, s, Gs, th, gx, gw) - 1.5 * _LOG2PI + 0.5 * logdetG)
    if w > 0.0:
        lP += w * lG
    return v, lP


@njit(cache=True)
def theta_moments(B, y):
    """(ok, chat, Fi): mean and covariance of the (untruncated) Gaussian of c from the theta design."""
    G, r = normal_eqs(B, y)
    s = np.empty(3)
    chat = np.zeros(3)
    Fi = np.zeros((3, 3))
    for i in range(3):
        if not G[i, i] > 0.0:
            return False, chat, Fi
        s[i] = math.sqrt(G[i, i])
    Gs = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            Gs[i, j] = G[i, j] / (s[i] * s[j])
    L = np.zeros((3, 3))
    ok, ld, Gsi = inv3_chol(Gs, L)
    if not ok:
        return False, chat, Fi
    th = np.zeros(3)
    St = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            St[i, j] = Gsi[i, j] / (s[i] * s[j])
            th[i] += St[i, j] * r[j]
    Mi = np.array([[6.0, -2.0, -3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range(3):
        for j in range(3):
            chat[i] += Mi[i, j] * th[j]
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                for l_ in range(3):
                    acc += Mi[i, k] * St[k, l_] * Mi[j, l_]
            Fi[i, j] = acc
    return True, chat, Fi


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


@njit(cache=True, parallel=True)
def design_par(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, chunk):
    """design() with the data points split into chunks of ``chunk`` output rows that run on
    numba's threads. Orbit, occultation terms and columns are computed point by point, so the
    result is identical to design() for any chunking and thread count."""
    nt = tt.shape[0]
    n = nt // ksup
    A = np.zeros((n, 3))
    ab = abs(b)
    inc = math.acos(min(ab / a, 1.0))
    si = math.sin(inc)
    k = abs(rp)
    z_c = terms_cut(k, tol, Gt)
    inv = 1.0 / ksup
    nch = (n + chunk - 1) // chunk
    for c in prange(nch):
        i0 = c * chunk
        i1 = min(n, i0 + chunk)
        r0, r1 = i0 * ksup, i1 * ksup
        z = np.empty(r1 - r0)
        if ecc < 1.0e-5:
            _orbit._rsky_circular(tt[r0:r1], t0, per, a, si, 0.25, 1, z)
        else:
            _orbit._rsky_eccentric(tt[r0:r1], t0, per, a, inc, ecc, w, 1, z)
        Af = bernstein_cols(_occult_terms(z, k, z_c, A_half))
        for i in range(i0, i1):
            a0 = a1 = a2 = 0.0
            for s_ in range(ksup):
                r = (i - i0) * ksup + s_
                a0 += Af[r, 0]
                a1 += Af[r, 1]
                a2 += Af[r, 2]
            f = wts[i] * inv
            A[i, 0] = a0 * f
            A[i, 1] = a1 * f
            A[i, 2] = a2 * f
    return A


@njit(cache=True)
def design_theta(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt):
    """design() for theta = (f0, c1, c2) (theta_cols): the well-conditioned design of the
    quadratic law."""
    nt = tt.shape[0]
    z = np.empty(nt)
    ab = abs(b)
    inc = math.acos(min(ab / a, 1.0))
    if ecc < 1.0e-5:
        _orbit._rsky_circular(tt, t0, per, a, math.sin(inc), 0.25, 1, z)
    else:
        _orbit._rsky_eccentric(tt, t0, per, a, inc, ecc, w, 1, z)
    Af = theta_cols(_occult_terms_cut(z, abs(rp), tol, A_half, Gt))
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


@njit(cache=True, parallel=True)
def design_theta_par(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, chunk):
    """design_theta() on numba's threads, as design_par()."""
    nt = tt.shape[0]
    n = nt // ksup
    A = np.zeros((n, 3))
    ab = abs(b)
    inc = math.acos(min(ab / a, 1.0))
    si = math.sin(inc)
    k = abs(rp)
    z_c = terms_cut(k, tol, Gt)
    inv = 1.0 / ksup
    nch = (n + chunk - 1) // chunk
    for c in prange(nch):
        i0 = c * chunk
        i1 = min(n, i0 + chunk)
        r0, r1 = i0 * ksup, i1 * ksup
        z = np.empty(r1 - r0)
        if ecc < 1.0e-5:
            _orbit._rsky_circular(tt[r0:r1], t0, per, a, si, 0.25, 1, z)
        else:
            _orbit._rsky_eccentric(tt[r0:r1], t0, per, a, inc, ecc, w, 1, z)
        Af = theta_cols(_occult_terms(z, k, z_c, A_half))
        for i in range(i0, i1):
            a0 = a1 = a2 = 0.0
            for s_ in range(ksup):
                r = (i - i0) * ksup + s_
                a0 += Af[r, 0]
                a1 += Af[r, 1]
                a2 += Af[r, 2]
            f = wts[i] * inv
            A[i, 0] = a0 * f
            A[i, 1] = a1 * f
            A[i, 2] = a2 * f
    return A


@njit(cache=True)
def design_theta_ws(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, ws):
    """design_theta() in the arrays of the workspace ws (workspace() makes one): the same values without allocating.
    Data points whose sub-exposures are all off the star are not computed: their row is that of an unocculted point
    (wts / ksup x ksup ones, then zeros; exactly what the full calculation gives them). A point is skipped only if z at
    its mid-time, minus the largest change the orbit allows within half its exposure (the largest orbital speed,
    2 pi a / per sqrt((1 + e) / (1 - e)), x half), stays above 1 + rp + 1e-9 (a + 1), with a (1 - e) bounding a
    mid-time behind the star. The other points go through the same orbit, occultation and exposure-average code as
    before, on compact arrays. Circular orbits and eccentric ones without sub-exposures give identical values; with
    sub-exposures, the eccentric orbit's Kepler solver is warm-started from a different neighbour after a skipped
    stretch, which can change the last bit of the eccentric anomaly.
    Returns A (n x 3), a view into ws that the next call overwrites."""
    z, Tf, A, bcnt, zb, idx, lex, ldx, edx, dev, tc, tmid, half, zm, keep = ws
    ab = abs(b)
    inc = math.acos(min(ab / a, 1.0))
    k = abs(rp)
    z_c = terms_cut(k, tol, Gt)
    nt = tt.shape[0]
    n = A.shape[0]
    circ = ecc < 1.0e-5
    si = math.sin(inc)
    if circ:
        _orbit._rsky_circular(tmid, t0, per, a, si, 0.25, 1, zm)
    else:
        _orbit._rsky_eccentric(tmid, t0, per, a, inc, ecc, w, 1, zm)
    # the sky separation changes no faster than the orbital speed, at most 2 pi a / per sqrt((1 + e) / (1 - e)) (at
    # periastron); a mid-time behind the star (zm = BIGD) is bounded by the separation where the planet crosses the
    # sky plane, r >= a (1 - e)
    speed = 2.0 * math.pi * a / per * math.sqrt((1.0 + ecc) / (1.0 - ecc))
    rmin = a * (1.0 - ecc)
    lim = 1.0 + k + 1e-9 * (a + 1.0)
    m = 0
    for i in range(n):
        if min(zm[i], rmin) - speed * half[i] < lim:
            keep[m] = i
            m += 1
    mt = m * ksup
    zk = z[:mt]
    if ksup == 1:
        for j in range(m):
            zk[j] = zm[keep[j]]
    else:
        tk = tc[:mt]
        for j in range(m):
            i0 = keep[j] * ksup
            for s in range(ksup):
                tk[j * ksup + s] = tt[i0 + s]
        if circ:
            _orbit._rsky_circular(tk, t0, per, a, si, 0.25, 1, zk)
        else:
            _orbit._rsky_eccentric(tk, t0, per, a, inc, ecc, w, 1, zk)
    T = Tf[:3 * mt].reshape((3, mt))
    _occult_terms_into(zk, k, z_c, A_half, T, bcnt, zb, idx, lex, ldx, edx, dev)
    inv = 1.0 / ksup
    j = 0
    for i in range(n):
        f = wts[i] * inv
        if j < m and keep[j] == i:
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for s in range(ksup):
                r = j * ksup + s
                lx, ld, ed = T[0, r], T[1, r], T[2, r]
                s0 += 1.0 - (12.0 * lx - 12.0 * ld - 6.0 * ed)
                s1 += 6.0 * lx - 6.0 * ld - 4.0 * ed
                s2 += 5.0 * lx - 6.0 * ld - 2.0 * ed
            A[i, 0] = s0 * f
            A[i, 1] = s1 * f
            A[i, 2] = s2 * f
            j += 1
        else:
            A[i, 0] = float(ksup) * f
            A[i, 1] = 0.0
            A[i, 2] = 0.0
    return A


@njit(cache=True)
def _theta_average(T, ksup, wts, A):
    """A[i] = wts[i] / ksup x the sum over the ksup sub-exposures of theta_cols(T): theta_cols() and the exposure
    average of design_theta() in one pass, in the same order of operations (identical values)."""
    n = A.shape[0]
    inv = 1.0 / ksup
    for i in range(n):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for s in range(ksup):
            r = i * ksup + s
            lx, ld, ed = T[0, r], T[1, r], T[2, r]
            s0 += 1.0 - (12.0 * lx - 12.0 * ld - 6.0 * ed)
            s1 += 6.0 * lx - 6.0 * ld - 4.0 * ed
            s2 += 5.0 * lx - 6.0 * ld - 2.0 * ed
        f = wts[i] * inv
        A[i, 0] = s0 * f
        A[i, 1] = s1 * f
        A[i, 2] = s2 * f


def workspace(tt, ksup):
    """Arrays for design_theta_ws / loglike_batch_ws for the model times tt (len(tt) // ksup data points of ksup
    sub-exposures each), with the mid-time and half-span of each data point. About 110 bytes per model point; the
    lists are touched only up to the number of computed / limb points."""
    tt = np.ascontiguousarray(tt, dtype=float)
    nt = len(tt)
    n = nt // ksup
    S = tt[:n * ksup].reshape(n, ksup)
    lo, hi = S.min(axis=1), S.max(axis=1)
    nb = (nt + 63) // 64
    return (np.empty(nt), np.empty(3 * nt), np.empty((n, 3)), np.empty(nb, dtype=np.int64),
            np.empty(nt), np.empty(nt, dtype=np.int64), np.empty(nt), np.empty(nt), np.empty(nt), np.empty(nt),
            np.empty(nt), np.ascontiguousarray(0.5 * (lo + hi)), np.ascontiguousarray(0.5 * (hi - lo)), np.empty(n),
            np.empty(n, dtype=np.int64))


@njit(cache=True, inline="always")
def a_logjac_from_T14(T14, per, rp, b):
    """a/R* of a circular orbit with total duration T14 and log|da/dT14|; (nan, 0) if invalid."""
    Q = (1.0 + rp) ** 2 - b * b
    if not (per > 0.0):                     # an optimiser's step to per = 0 (logit / log coordinates)
        return math.nan, 0.0
    x = math.pi * T14 / per
    if not (Q > 0.0 and 0.0 < x < 0.5 * math.pi):
        return math.nan, 0.0
    s = math.sin(x)
    if s * s * s < 1e-300:                  # T14 -> 0 (an unbounded a/R*): s^2 underflows, Q / 0 raised
        return math.nan, 0.0
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
def b_conj(b, ecc, w, ttype):
    """Sky separation at conjunction in stellar radii (the transit impact parameter): b = a cos i
    times r/a = (1 - e^2) / (1 + e sin w) at mid-transit (ttype 1) or (1 - e^2) / (1 - e sin w) at
    mid-eclipse (ttype 2). The prior support 0 <= b_conj < 1 + rp is the region with an event; for
    an eccentric orbit it differs from 0 <= b < 1 + rp."""
    s = 1.0 if ttype == 1 else -1.0
    return b * (1.0 - ecc * ecc) / (1.0 + s * ecc * math.sin(w))


@njit(cache=True)
def _loglike_row(P, lj, i, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, has_ldp, mb, Pb, ne_out):
    """(log-likelihood + box prior + log-Jacobian, log P of the cone) of canonical row i. If ne_out has
    a row i, it receives the theta normal equations (_store_ne; NaN where they are not computed)."""
    ok = True
    for k in range(7):
        v = P[i, k]
        if not (v == v and lo[k] < v < hi[k]):
            ok = False
    t0, per, rp, a, b, ecc, w = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
    if not (ok and 0.0 <= ecc < 1.0 and 0.0 <= b and b_conj(b, ecc, w, 1) < 1.0 + rp and per > 0.0
            and rp > 0.0 and a > 1.0):
        return -np.inf, 0.0
    if has_ldp:
        A = design(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt)
        return marg_loglike_ldprior(A, yw, mb, Pb) + lj[i], 0.0
    B = design_theta(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt)
    G, r = normal_eqs(B, yw)                            # = marg_theta_parts, keeping G, r
    if ne_out.shape[0] > i:
        _store_ne(ne_out, i, G, r)
    v, lP = theta_parts_normal(G, r, B, yw, 0.0, 0.0, GL_X, GL_W)
    return v + lP + lj[i], lP


@njit(cache=True)
def _store_ne(ne_out, i, G, r):
    """Row i of ne_out = (G00, G01, G02, G11, G12, G22, r0, r1, r2) (normal_eqs: G symmetric)."""
    ne_out[i, 0], ne_out[i, 1], ne_out[i, 2] = G[0, 0], G[0, 1], G[0, 2]
    ne_out[i, 3], ne_out[i, 4], ne_out[i, 5] = G[1, 1], G[1, 2], G[2, 2]
    ne_out[i, 6], ne_out[i, 7], ne_out[i, 8] = r[0], r[1], r[2]


@njit(cache=True)
def loglike_batch(X, idx, fixed, t14, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt,
                  has_ldp=False, mb=np.zeros(2), Pb=np.eye(2), logP_out=np.zeros(0),
                  ne_out=np.zeros((0, 9))):
    """log-likelihood + uniform box prior (lo < canonical < hi; 0 <= b_conj < 1 + rp;
    0 <= ecc < 1) + log-Jacobian of the T14 map, for every row of X. If logP_out has one entry
    per row, it receives log P of the physical cone (0 outside the prior; not with has_ldp). If
    ne_out has one row per row of X (n x 9, NaN-filled), it receives the theta normal equations
    (_store_ne; not with has_ldp), from which theta_moments_ne gives theta_moments_batch's output."""
    P, lj = to_canonical(X, idx, fixed, t14)
    n = X.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        v, lP = _loglike_row(P, lj, i, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, has_ldp, mb, Pb, ne_out)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True)
def _loglike_row_ws(P, lj, i, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, ne_out, ws):
    """_loglike_row (no limb-darkening prior) with the design in the workspace ws: identical values."""
    ok = True
    for k in range(7):
        v = P[i, k]
        if not (v == v and lo[k] < v < hi[k]):
            ok = False
    t0, per, rp, a, b, ecc, w = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
    if not (ok and 0.0 <= ecc < 1.0 and 0.0 <= b and b_conj(b, ecc, w, 1) < 1.0 + rp and per > 0.0
            and rp > 0.0 and a > 1.0):
        return -np.inf, 0.0
    B = design_theta_ws(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, ws)
    G, r = normal_eqs(B, yw)
    if ne_out.shape[0] > i:
        _store_ne(ne_out, i, G, r)
    v, lP = theta_parts_normal(G, r, B, yw, 0.0, 0.0, GL_X, GL_W)
    return v + lP + lj[i], lP


@njit(cache=True)
def loglike_batch_ws(X, idx, fixed, t14, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, logP_out, ne_out, ws):
    """loglike_batch (no limb-darkening prior) reusing the workspace ws for every row: no per-row allocation of
    the ~56 bytes per model point that loglike_batch frees and re-allocates, nor its n x k x 3 column array. Identical values."""
    P, lj = to_canonical(X, idx, fixed, t14)
    n = X.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        v, lP = _loglike_row_ws(P, lj, i, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, ne_out, ws)
        out[i] = v
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True)
def loglike_batch_dp(X, idx, fixed, t14, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt,
                     has_ldp=False, mb=np.zeros(2), Pb=np.eye(2), logP_out=np.zeros(0), chunk=1024,
                     ne_out=np.zeros((0, 9))):
    """loglike_batch with the data points of each row spread over numba's threads (design_par);
    the rows one after the other (for the single evaluations of the optimisers on large light
    curves). Identical results."""
    P, lj = to_canonical(X, idx, fixed, t14)
    n = X.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in range(n):
        ok = True
        for k in range(7):
            v = P[i, k]
            if not (v == v and lo[k] < v < hi[k]):
                ok = False
        t0, per, rp, a, b, ecc, w = P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        if not (ok and 0.0 <= ecc < 1.0 and 0.0 <= b and b_conj(b, ecc, w, 1) < 1.0 + rp and per > 0.0
                and rp > 0.0 and a > 1.0):
            out[i] = -np.inf
            if keep:
                logP_out[i] = 0.0
            continue
        if has_ldp:
            A = design_par(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, chunk)
            out[i] = marg_loglike_ldprior(A, yw, mb, Pb) + lj[i]
            lP = 0.0
        else:
            B = design_theta_par(tt, ksup, wts, t0, per, rp, a, b, ecc, w, tol, A_half, Gt, chunk)
            G, r = normal_eqs(B, yw)
            if ne_out.shape[0] == n:
                _store_ne(ne_out, i, G, r)
            v, lP = theta_parts_normal(G, r, B, yw, 0.0, 0.0, GL_X, GL_W)
            out[i] = v + lP + lj[i]
        if keep:
            logP_out[i] = lP
    return out


@njit(cache=True, parallel=True)
def loglike_batch_par(X, idx, fixed, t14, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt,
                      has_ldp=False, mb=np.zeros(2), Pb=np.eye(2), logP_out=np.zeros(0),
                      ne_out=np.zeros((0, 9))):
    """loglike_batch with the rows spread over numba's threads (identical results)."""
    P, lj = to_canonical(X, idx, fixed, t14)
    n = X.shape[0]
    out = np.empty(n)
    keep = logP_out.shape[0] == n
    for i in prange(n):
        v, lP = _loglike_row(P, lj, i, lo, hi, tt, ksup, wts, yw, tol, A_half, Gt, has_ldp, mb, Pb, ne_out)
        out[i] = v
        if keep:
            logP_out[i] = lP
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
        if not has_ldp:
            B = design_theta(tt, ksup, wts, P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5],
                             P[i, 6], tol, A_half, Gt)
            ok, ch, Fi = theta_moments(B, yw)
            for k in range(3):
                C[i, k] = ch[k]
                for j in range(3):
                    S[i, k, j] = Fi[k, j]
            continue
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


@njit(cache=True)
def theta_moments_batch(P, tt, ksup, wts, yw, tol, A_half, Gt):
    """For canonical rows P: mean (n x 3) and covariance (n x 3 x 3) of the (untruncated) Gaussian
    of theta = (f0, c1, c2) given the geometry (c = (6 f0 - 2 c1 - 3 c2, c1, c2)); NaN where the
    normal equations are singular."""
    n = P.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    L = np.zeros((3, 3))
    for i in range(n):
        B = design_theta(tt, ksup, wts, P[i, 0], P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5],
                         P[i, 6], tol, A_half, Gt)
        G, r = normal_eqs(B, yw)
        _theta_moments_row(G, r, T, S, i, L)
    return T, S


@njit(cache=True)
def theta_moments_ne(NE):
    """theta_moments_batch from the normal equations stored by loglike_batch (ne_out, n x 9): no pass
    over the light curve. NaN rows (not computed) stay NaN."""
    n = NE.shape[0]
    T = np.full((n, 3), np.nan)
    S = np.full((n, 3, 3), np.nan)
    L = np.zeros((3, 3))
    G = np.empty((3, 3))
    r = np.empty(3)
    for i in range(n):
        G[0, 0], G[0, 1], G[0, 2] = NE[i, 0], NE[i, 1], NE[i, 2]
        G[1, 0], G[1, 1], G[1, 2] = NE[i, 1], NE[i, 3], NE[i, 4]
        G[2, 0], G[2, 1], G[2, 2] = NE[i, 2], NE[i, 4], NE[i, 5]
        r[0], r[1], r[2] = NE[i, 6], NE[i, 7], NE[i, 8]
        _theta_moments_row(G, r, T, S, i, L)
    return T, S


@njit(cache=True)
def _theta_moments_row(G, r, T, S, i, L):
    """Row i of theta_moments_batch's output from the normal equations G, r (L: 3 x 3 work array)."""
    s = np.empty(3)
    for k in range(3):
        if not G[k, k] > 0.0:                         # also NaN (no stored normal equations)
            return
        s[k] = math.sqrt(G[k, k])
    Gs = np.empty((3, 3))
    for k in range(3):
        for j in range(3):
            Gs[k, j] = G[k, j] / (s[k] * s[j])
    ok, ld, Gsi = inv3_chol(Gs, L)
    if not ok or L[2, 2] * L[2, 2] < 1e-12:          # as marg_theta_parts: a weak ridge
        for k in range(3):
            Gs[k, k] += 1e-9
        ok, ld, Gsi = inv3_chol(Gs, L)
        if not ok:
            return
    for k in range(3):
        acc = 0.0
        for j in range(3):
            S[i, k, j] = Gsi[k, j] / (s[k] * s[j])
            acc += S[i, k, j] * r[j]
        T[i, k] = acc


@njit(cache=True)
def _trunc_normal(mu, sd, lo, hi, u):
    """Inverse-CDF draw from N(mu, sd^2) truncated to [lo, hi] (tails and narrow windows exact
    enough: an exponential substitution beyond 8 sd, a linear-in-log density for windows below
    1e-6 sd)."""
    if not hi > lo:
        return lo
    za, zb = (lo - mu) / sd, (hi - mu) / sd
    flip = za > 0.0
    if flip:
        za, zb = -zb, -za
        u = 1.0 - u
    if zb - za < 1e-6:
        z = za + u * (zb - za)
    elif zb < -8.0:                                     # exponential tangent at zb
        lam = -zb
        E = -math.expm1(-lam * (zb - za))
        z = zb + math.log1p(-u * E) / lam
    else:
        pa, pb = ndtr(za), ndtr(zb)
        p = pa + u * (pb - pa)
        z = ndtri(min(max(p, 1e-300), 1.0 - 1e-16))
        z = min(max(z, za), zb)
    if flip:
        z = -z
    return mu + sd * z


@njit(cache=True)
def gibbs_theta_one(th, Sig, rng_u, sweeps):
    """One draw of theta = (f0, c1, c2) ~ N(th, Sig) truncated to the cone 0 <= c0 <= c1 <= c2,
    c0 = 6 f0 - 2 c1 - 3 c2 (Gibbs, explicit intervals: f0 in [(2 c1 + 3 c2) / 6, (c1 + c2) / 2],
    c1 in [2 f0 - c2, min(c2, 3 f0 - 3 c2 / 2)], c2 in [max(c1, 2 f0 - c1), 2 f0 - 2 c1 / 3]).
    Well conditioned for any transit depth (in c the covariance is nearly singular for shallow
    transits). Starts at the centroid of the cone's cross-section at f0 = max(th0, 0)."""
    d = np.sqrt(np.array([Sig[0, 0], Sig[1, 1], Sig[2, 2]]))
    R = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = Sig[i, j] / (d[i] * d[j])
    Ri = np.linalg.inv(R)                               # scaled: the entries of Sig span decades
    P = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            P[i, j] = Ri[i, j] / (d[i] * d[j])
    f0 = th[0] if th[0] > 0.0 else 1.0
    x = np.array([f0, f0 * 2.2 / 3.0, f0 * 4.2 / 3.0])
    k = 0
    for _ in range(sweeps):
        for i in range(3):
            mi = th[i]
            for j in range(3):
                if j != i:
                    mi -= P[i, j] * (x[j] - th[j]) / P[i, i]
            sd = 1.0 / math.sqrt(P[i, i])
            if i == 0:
                lo, hi = (2.0 * x[1] + 3.0 * x[2]) / 6.0, (x[1] + x[2]) / 2.0
            elif i == 1:
                lo, hi = 2.0 * x[0] - x[2], min(x[2], 3.0 * x[0] - 1.5 * x[2])
            else:
                lo, hi = max(x[1], 2.0 * x[0] - x[1]), 2.0 * x[0] - 2.0 * x[1] / 3.0
            x[i] = _trunc_normal(mi, sd, lo, hi, rng_u[k])
            k += 1
    return x


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


@njit(cache=True, fastmath=FASTMATH)
def _dot(a, b):
    # four running sums: vectorises without reassociation
    n = a.shape[0]
    m = n - n % 4
    s0 = s1 = s2 = s3 = 0.0
    for t in range(0, m, 4):
        s0 += a[t] * b[t]
        s1 += a[t + 1] * b[t + 1]
        s2 += a[t + 2] * b[t + 2]
        s3 += a[t + 3] * b[t + 3]
    for t in range(m, n):
        s0 += a[t] * b[t]
    return (s0 + s2) + (s1 + s3)


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
@njit(cache=True, fastmath=FASTMATH)
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
        if check and not (ok and 0.0 <= ecc < 1.0 and 0.0 <= b and b_conj(b, ecc, w, ttype) < 1.0 + rp
                          and per > 0.0 and rp > 0.0 and a > 1.0
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
