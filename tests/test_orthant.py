"""Orthant probability of the marginal likelihood far outside the physical cone."""
import itertools
import math

import numpy as np

from pizzetti import _marginal as K

D3 = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])


def tilted_reference(m, S, n=100_000, seed=1):
    """log P(w > 0), w ~ N(m, S), by importance sampling from N(w*, S) with w* the dominating
    point (minimum Mahalanobis distance to the orthant): accurate far in the tail."""
    P = np.linalg.inv(S)
    best = (np.inf, None)
    for act in itertools.product((0, 1), repeat=len(m)):
        A = [i for i in range(len(m)) if act[i]]
        F = [i for i in range(len(m)) if not act[i]]
        w = np.zeros(len(m))
        if F:
            w[F] = m[F] + (np.linalg.solve(P[np.ix_(F, F)], P[np.ix_(F, A)] @ m[A]) if A else 0.0)
        if np.all(w >= -1e-12):
            d2 = (w - m) @ P @ (w - m)
            if d2 < best[0]:
                best = (d2, w)
    ws = best[1]
    W = ws + np.random.default_rng(seed).standard_normal((n, len(m))) @ np.linalg.cholesky(S).T
    lw = -0.5 * (np.einsum("ni,ij,nj->n", W - m, P, W - m) - np.einsum("ni,ij,nj->n", W - ws, P, W - ws))
    lw = lw[np.all(W >= 0, axis=1)]
    return float(np.log(np.sum(np.exp(lw - lw.max()))) + lw.max() - math.log(n))


def cases(k, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        A = rng.standard_normal((3, 3))
        S = (D3 @ (A @ A.T + 0.05 * np.eye(3)) @ D3.T)[:k, :k]
        m = -rng.uniform(40, 80) * np.sqrt(np.diag(S)) * rng.uniform(0.3, 1.0, k)
        out.append((m, S))
    return out


def test_log_ndtr():
    from scipy.special import log_ndtr
    for x in (-60.0, -37.0, -20.0001, -19.9, -3.0, 0.0, 2.0):
        assert abs(K.log_ndtr(x) - log_ndtr(x)) < 1e-8 * max(1.0, abs(log_ndtr(x)))


def test_finite_far_outside_the_cone():
    """Beyond 37 sigma the quadrature underflowed to -inf; now finite and accurate to
    leading order (asymptotic form, checked against tilted importance sampling)."""
    for k, f in ((3, K.log_orthant_prob), (2, K.log_orthant_prob2)):
        for m, S in cases(k, 4, seed=k):
            v = f(m, S, K.GL_X, K.GL_W)
            ref = tilted_reference(m, S)
            assert np.isfinite(v)
            assert abs(v - ref) < 0.03 * abs(ref) + 2.0


def test_increases_towards_the_cone():
    m0, S = cases(3, 1, seed=5)[0]
    vals = [K.log_orthant_prob(f * m0, S, K.GL_X, K.GL_W) for f in (1.0, 0.8, 0.6, 0.4)]
    assert all(np.isfinite(vals)) and all(np.diff(vals) > 0)


def test_continuous_at_the_switch():
    """Where the quadrature stops (Cholesky-ordered x1 = -37) the asymptotic form takes over with
    a jump of well under 1 % of log P (which is several hundred nats or more there)."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    S = D3 @ (A @ A.T + 0.05 * np.eye(3)) @ D3.T
    m0 = np.array([-1.0, 0.05, 0.05]) * np.sqrt(np.diag(S))
    f = np.linspace(20, 80, 601)
    v = np.array([K.log_orthant_prob(x * m0, S, K.GL_X, K.GL_W) for x in f])
    assert np.all(np.isfinite(v))
    jumps = np.abs(np.diff(v)) - np.abs(np.median(np.diff(v)))
    assert np.max(jumps / np.abs(v[1:])) < 0.01
