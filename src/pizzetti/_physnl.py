"""Probability that the nonlinear law's coefficients are physical, for the
marginal likelihood (``MarginalTransitModel``, limb_dark="nonlinear",
ld_prior="physical").

The nonlinear law is a quartic in s = sqrt(mu), written in the Bernstein basis
I(s) = sum_j c_j B^4_j(s). It is physical (positive, darkening towards the limb)
iff c_0 = I(0) >= 0 and I'(s) >= 0 on [0, 1] (non-decreasing from a non-negative
value is positive), with I'(s) = 4 sum_j d_j B^3_j(s), d_j = c_{j+1} - c_j. The
region is a convex, curved cone, so the marginal likelihood needs
P(c in region) for c ~ N(chat, F^-1) -- the analogue of the orthant probability
of the polyhedral laws.

Line integration: along the largest-variance eigenvector e of F^-1, the line
c = b + t e meets the convex region in an interval [lo, hi]; c_0 >= 0 is one
bound, and I'(s) = A(s) + t B(s) >= 0 for all s gives the others, from the
extremes of -A(s)/B(s) over s (grid of G_S points, refined by a golden-section
search on the exact function, which can only tighten the grid bound). The line contributes Phi(hi) - Phi(lo); the other
four eigen-directions are integrated with fixed scrambled-Sobol points (256).
The integrand is continuous, so the estimate is deterministic and smooth in the
geometry. Accuracy: 256 vs 4096 points agree to <= 4e-4 nats; vs brute-force
Monte Carlo with the exact test to ~0.003 nats (simulated light curves,
S/N from 10 to 300 ppm).

(A decomposition along d -> d + t (1, 1, 1, 1), which shifts I' uniformly and
gives a bivariate-normal integrand, was tried first: that direction is the
total limb contrast, which the data constrain well, so the integrand becomes a
step function and quasi-Monte Carlo fails away from the truth.)
"""
import math

import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# line integration along the largest-variance direction
# ----------------------------------------------------------------------------
G_S = 33
_S = np.linspace(0.0, 1.0, G_S)
B3_GRID = np.ascontiguousarray(np.column_stack([(1 - _S) ** 3, 3 * _S * (1 - _S) ** 2,
                                                3 * _S ** 2 * (1 - _S), _S ** 3]))


def qmc_normal4(n=256, seed=20261016):
    from scipy.special import ndtri
    from scipy.stats import qmc
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    return np.ascontiguousarray(ndtri(np.clip(u, 1e-12, 1 - 1e-12)))


Z4_QMC = qmc_normal4()
Z4_BIG = qmc_normal4(4096, 20261017)          # retry set when no line of Z4_QMC hits the region


@njit(cache=True, inline="always")
def _bern3(q, s):
    r = 1.0 - s
    return q[0] * r * r * r + 3.0 * q[1] * s * r * r + 3.0 * q[2] * s * s * r + q[3] * s * s * s


@njit(cache=True)
def _refine(al, be, s0, s1, sign, v0):
    """Golden-section search on [s0, s1] for the extreme of f(s) = -A(s)/B(s) over the
    points with sign * B(s) > 0 (sign +1: maximise, lower bound; -1: minimise, upper
    bound). Every evaluated value is a valid bound, so the result only tightens v0."""
    gr = 0.6180339887498949
    best = v0
    a, b = s0, s1
    for _ in range(24):
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        B1 = _bern3(be, x1)
        B2 = _bern3(be, x2)
        f1 = sign * (-_bern3(al, x1) / B1) if sign * B1 > 0.0 else -np.inf
        f2 = sign * (-_bern3(al, x2) / B2) if sign * B2 > 0.0 else -np.inf
        for f in (f1, f2):
            if f > -np.inf:
                best = max(best, f) if sign > 0 else min(best, -f)
        if f1 >= f2:
            b = x2
        else:
            a = x1
    return best


@njit(cache=True)
def line_interval(b, e, B3):
    """Feasible t-interval [lo, hi] of c = b + t e in the physical region:
    c_0 >= 0, and I'(s) = A(s) + t B(s) >= 0 for all s, from the extremes of
    -A/B over an s-grid, refined by a golden-section search on the exact
    function (never looser than the grid, never tighter than the truth)."""
    lo = -np.inf
    hi = np.inf
    if e[0] > 0.0:
        lo = -b[0] / e[0]
    elif e[0] < 0.0:
        hi = -b[0] / e[0]
    elif b[0] < 0.0:
        return 1.0, -1.0
    al = np.empty(4)
    be = np.empty(4)
    for j in range(4):
        al[j] = b[j + 1] - b[j]
        be[j] = e[j + 1] - e[j]
    G = B3.shape[0]
    klo, khi = -1, -1
    vlo, vhi = -np.inf, np.inf
    for k in range(G):
        A = al[0] * B3[k, 0] + al[1] * B3[k, 1] + al[2] * B3[k, 2] + al[3] * B3[k, 3]
        Bv = be[0] * B3[k, 0] + be[1] * B3[k, 1] + be[2] * B3[k, 2] + be[3] * B3[k, 3]
        if Bv > 0.0:
            v = -A / Bv
            if v > vlo:
                vlo, klo = v, k
        elif Bv < 0.0:
            v = -A / Bv
            if v < vhi:
                vhi, khi = v, k
        elif A < 0.0:
            return 1.0, -1.0
    h = 1.0 / (G - 1)
    if klo >= 0:
        vlo = _refine(al, be, max(0.0, (klo - 1) * h), min(1.0, (klo + 1) * h), 1.0, vlo)
    if khi >= 0:
        vhi = _refine(al, be, max(0.0, (khi - 1) * h), min(1.0, (khi + 1) * h), -1.0, vhi)
    return max(lo, vlo), min(hi, vhi)


@njit(cache=True)
def _line_prob(c, V, sd, Z4, B3):
    e = V[:, 4] * sd[4]
    n = Z4.shape[0]
    b = np.empty(5)
    acc = 0.0
    for i in range(n):
        for r in range(5):
            s_ = c[r]
            for q in range(4):
                s_ += V[r, q] * sd[q] * Z4[i, q]
            b[r] = s_
        lo, hi = line_interval(b, e, B3)
        if hi > lo:
            acc += 0.5 * (math.erfc(lo / math.sqrt(2.0)) - math.erfc(hi / math.sqrt(2.0)))
    return acc / n


@njit(cache=True)
def log_prob_physical_line(c, Fi, Z4, B3, Zbig=np.zeros((0, 4))):
    """log P(c in the physical region), c ~ N(c, Fi): line integration along the
    largest-variance eigenvector of Fi, quasi-Monte Carlo (fixed points Z4) over
    the other four eigen-directions. If no line hits the region, the estimate is
    repeated with the larger point set Zbig (if given)."""
    ev, V = np.linalg.eigh(Fi)
    sd = np.sqrt(np.maximum(ev, 0.0))
    p = _line_prob(c, V, sd, Z4, B3)
    if p == 0.0 and Zbig.shape[0] > 0:
        p = _line_prob(c, V, sd, Zbig, B3)
    return math.log(p) if p > 0.0 else -np.inf


def sample_physical(c, Fi, rng, m=512):
    """One draw of N(c, Fi) truncated to the physical region, for rows where plain
    rejection fails (tiny physical probability): m random lines along the
    largest-variance direction, one picked in proportion to its in-region
    probability, then t from the normal truncated to its exact interval
    (sampling-importance-resampling over lines; exact as m -> inf).
    Returns NaN if no line meets the region."""
    from scipy.special import ndtr, ndtri
    ev, V = np.linalg.eigh(Fi)
    sd = np.sqrt(np.maximum(ev, 0.0))
    e = V[:, 4] * sd[4]
    Z = rng.standard_normal((m, 4))
    B = c[None, :] + (Z * sd[:4]) @ V[:, :4].T
    iv = np.array([line_interval(b, e, B3_GRID) for b in B])
    pr = np.where(iv[:, 1] > iv[:, 0], ndtr(iv[:, 1]) - ndtr(iv[:, 0]), 0.0)
    if not pr.sum() > 0:
        return np.full(5, np.nan)
    i = rng.choice(m, p=pr / pr.sum())
    lo, hi = ndtr(iv[i, 0]), ndtr(iv[i, 1])
    t = ndtri(lo + rng.random() * (hi - lo))
    return B[i] + t * e
