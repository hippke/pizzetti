"""MarginalTransitModel: likelihood, coordinates, scope errors, sampler API."""
import math
import warnings

import numpy as np
import pytest

import pizzetti
from pizzetti import _marginal as K

TRUTH = dict(t0=0.0, per=3.5, rp=0.10, a=9.0, b=0.35, u1=0.40, u2=0.25)


def params(**kw):
    d = dict(TRUTH, **kw)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = d["t0"], d["per"], d["rp"], d["a"]
    p.inc = math.degrees(math.acos(d["b"] / d["a"]))
    p.ecc, p.w, p.limb_dark, p.u = d.get("ecc", 0.0), d.get("w", 90.0), "quadratic", [d["u1"], d["u2"]]
    return p


def data(sigma=500e-6, seed=42, **kw):
    t = (np.arange(3)[:, None] * 3.5 + np.arange(-0.13, 0.13, 2 / 1440)[None, :]).ravel()
    p = params(**kw)
    f = pizzetti.TransitModel(p, t).light_curve(p)
    return t, f + sigma * np.random.default_rng(seed).standard_normal(t.size), p


def dense_reference(model, X):
    """Independent numpy/scipy evaluation of the marginal log-likelihood."""
    from scipy.stats import multivariate_normal
    out = []
    for x in np.atleast_2d(X):
        P = model._canonical_physical(x[None, :])[0]
        tp = pizzetti.TransitParams()
        tp.t0, tp.per, tp.rp, tp.a = P[0], P[1], P[2], P[3]
        tp.inc = math.degrees(math.acos(P[4] / P[3]))
        tp.ecc, tp.w, tp.limb_dark = P[5], P[6] / K.math.pi * 180, "quadratic"
        cols = []
        for u, sc in (((0.0, 0.0), 1.0), ((1.0, 0.0), 2 / 3), ((0.0, 1.0), 5 / 6)):
            tp.u = list(u)
            cols.append(pizzetti.TransitModel(tp, model.t).light_curve(tp) * sc)
        p0, p1, p2 = cols[0], cols[0] - cols[1], cols[0] - cols[2]
        A = np.column_stack((p2, 2 * p1 - 2 * p2, p0 - 2 * p1 + p2)) * model.wts[:, None]
        F = A.T @ A
        Fi = np.linalg.inv(F)
        c = Fi @ (A.T @ model.yw)
        r = model.yw - A @ c
        D = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1.0]])
        prob = multivariate_normal(mean=np.zeros(3), cov=D @ Fi @ D.T).cdf(D @ c)
        out.append(-0.5 * r @ r - 0.5 * np.linalg.slogdet(F)[1] + np.log(prob))
    return np.array(out)


def test_loglike_matches_dense_reference():
    t, y, p = data()
    m = pizzetti.MarginalTransitModel(p, t, y, 500e-6, reparam="none")
    X = np.array([[0.0, 3.5, 0.1, 9.0, 0.35], [1e-4, 3.5, 0.099, 9.1, 0.3], [0.0, 3.5, 0.1, 8.8, 0.45]])
    got = m.log_prob(X)
    ref = dense_reference(m, X)
    assert np.allclose(got - got[0], ref - ref[0], atol=1e-4)


def test_heteroscedastic_and_T14_consistency():
    t, y, p = data()
    X = np.array([[0.0, 3.5, 0.1, 9.0, 0.35], [2e-4, 3.5001, 0.098, 9.3, 0.2]])
    m1 = pizzetti.MarginalTransitModel(p, t, y, 500e-6, reparam="none")
    m2 = pizzetti.MarginalTransitModel(p, t, y, np.full(t.size, 500e-6), reparam="none")
    m3 = pizzetti.MarginalTransitModel(p, t, y, 500e-6)                      # T14 internally
    assert m3.t14
    assert np.allclose(m1.log_prob(X), m2.log_prob(X), rtol=1e-12)
    assert np.allclose(m1.log_prob(X), m3.log_prob(X), rtol=1e-12)
    assert np.allclose(m3.to_physical(m3.to_internal(X)), X, rtol=1e-12)
    # per-point errors really enter: doubling some errors changes the likelihood
    e = np.full(t.size, 500e-6)
    e[::2] *= 2
    assert not np.isclose(pizzetti.MarginalTransitModel(p, t, y, e).log_prob(X[0]), m1.log_prob(X[0]))


def test_eccentric_and_supersampled_free_ecc():
    t, y, p = data(ecc=0.2, w=60.0)
    m = pizzetti.MarginalTransitModel(p, t, y, 500e-6, free=["t0", "rp", "a", "b", "ecc", "w"],
                                      supersample_factor=3, exp_time=0.002)
    assert not m.t14
    v = m.log_prob([0.0, 0.1, 9.0, 0.35, 0.2, 60.0])
    assert np.isfinite(v)
    assert m.log_prob([0.0, 0.1, 9.0, 0.35, 1.2, 60.0]) == -np.inf


def test_scope_errors():
    t, y, p = data()
    for attr, val, msg in (("limb_dark", "exponential", "exponential"), ("rp", -0.1, "rp > 0")):
        q = params()
        setattr(q, attr, val)
        with pytest.raises(NotImplementedError, match=msg):
            pizzetti.MarginalTransitModel(q, t, y, 1e-3)
    with pytest.raises(ValueError, match="inc"):
        pizzetti.MarginalTransitModel(p, t, y, 1e-3, free=["inc"])
    with pytest.raises(NotImplementedError, match="quadratic"):
        q = params()
        q.limb_dark, q.u = "squareroot", [0.3, 0.3]
        pizzetti.MarginalTransitModel(q, t, y, 1e-3, ld_prior={"mean": (0.3, 0.3), "cov": np.eye(2)})


def test_emcee_sampler_api_and_ld_recovery():
    pytest.importorskip("emcee")
    t, y, p = data(sigma=200e-6)
    m = pizzetti.MarginalTransitModel(p, t, y, 200e-6)
    rng = np.random.default_rng(1)
    th0 = np.array([0.0, 3.5, 0.1, 9.0, 0.35, 0.4225, 0.3077])         # old 7-column p0
    p0 = th0 + np.array([1e-4, 1e-6, 1e-3, 0.02, 0.01, 0.01, 0.01]) * rng.standard_normal((32, 7))
    s = m.EnsembleSampler(32)
    s.run_mcmc(p0, 1500)
    ch = s.get_chain(discard=500, flat=True, seed=2)
    assert s.labels == ["t0", "per", "rp", "a", "b", "q1", "q2", "f0"]
    assert ch.shape == (1000 * 32, 8)
    assert abs(np.median(ch[:, 2]) - 0.1) < 0.003
    assert abs(np.median(ch[:, 7]) - 1.0) < 1e-4
    raw = s.get_chain(discard=500, expand=False)
    assert raw.shape == (1000, 32, 5)
    full = s.get_chain(discard=500, ld="u")
    assert full.shape == (1000, 32, 8) and s.labels[5:] == ["u1", "u2", "f0"]
    assert abs(np.median(full[..., 5]) - 0.4) < 0.15


def test_fit_importance_sampling():
    pytest.importorskip("scipy")
    t, y, p = data(sigma=200e-6)
    m = pizzetti.MarginalTransitModel(p, t, y, 200e-6)
    r = m.fit(method="is", seed=3)
    assert r.diagnostics["method"] == "profile-IS" and r.diagnostics["khat"] < 0.7
    j = r.labels.index("rp")
    assert abs(np.median(r.samples[:, j]) - 0.1) < 0.003
    assert "rp" in r.summary()


def test_log_prior_and_ld_prior():
    pytest.importorskip("scipy")
    t, y, p = data(sigma=500e-6)
    base = pizzetti.MarginalTransitModel(p, t, y, 500e-6)
    X = np.array([[0.0, 3.5, 0.1, 9.0, 0.35], [0.0, 3.5, 0.1, 9.5, 0.2]])
    lp = lambda d: -0.5 * ((d["a"] - 9.0) / 0.1) ** 2           # noqa: E731
    m = pizzetti.MarginalTransitModel(p, t, y, 500e-6, log_prior=lp)
    assert np.allclose(m.log_prob(X) - base.log_prob(X), -0.5 * ((X[:, 3] - 9.0) / 0.1) ** 2)
    # a tight Gaussian LD prior pulls the recovered LD to its mean
    mq = pizzetti.MarginalTransitModel(p, t, y, 500e-6,
                                       ld_prior={"mean": (0.3, 0.3), "cov": np.diag([1e-4, 1e-4])})
    r = mq.fit(method="is", seed=4, ld="u")
    j = r.labels.index("u1")
    assert abs(np.mean(r.samples[:, j]) - 0.3) < 0.03
    assert r.diagnostics["ld_reweight_ess"] > 50


def test_profile_is_lazy_refinement():
    """A curved conditional mean, x | b ~ N(4 b^2, 0.05^2) with b ~ N(0.3, 0.2) on b >= 0:
    the unrefined grid is good enough, so lazy refinement adds no conditional fits;
    eager refinement adds some and gives the same posterior."""
    from pizzetti._sampling import profile_is

    def logp(X):
        X = np.atleast_2d(X)
        b, x = X[:, 1], X[:, 0]
        out = -0.5 * ((b - 0.3) / 0.2) ** 2 - 0.5 * ((x - 4 * b ** 2) / 0.05) ** 2
        return np.where(b >= 0, out, -np.inf)

    th0, sc = np.array([0.4, 0.3]), np.array([0.05, 0.2])
    lazy = profile_is(logp, th0, sc, prof=1, n=4000, rng=np.random.default_rng(1))
    eager = profile_is(logp, th0, sc, prof=1, n=4000, rng=np.random.default_rng(1), lazy_refine=False)
    assert lazy["khat"] < 0.5 and lazy["n_refine"] == 0
    assert eager["n_refine"] > 0 and eager["n_eval"] > lazy["n_eval"]
    for r in (lazy, eager):
        assert r["complete"]
        mb = np.sum(r["w"] * r["theta"][:, 1])
        mx = np.sum(r["w"] * r["theta"][:, 0])
        assert abs(mb - 0.32776) < 0.01                # truncated-normal mean of b
        assert abs(mx - 4 * np.sum(r["w"] * r["theta"][:, 1] ** 2)) < 0.01


@pytest.mark.parametrize("b_true", [0.95, 0.98])
def test_fit_is_grazing(b_true):
    """Grazing transits: importance draws beyond b = 1 + rp (zero weight, degenerate design)
    must not reach the coefficient draws (LinAlgError before the fix). rp is bounded: without
    an upper bound a grazing posterior is improper (rp, b -> infinity along b - rp = const)."""
    t = np.linspace(-0.12, 0.12, 2000)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.1, 9.0
    p.inc = math.degrees(math.acos(b_true / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 3e-4 * np.random.default_rng(0).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 3e-4, free=("t0", "rp", "a", "b"), bounds={"rp": (0.0, 0.3)})
    r = m.fit(method="is", seed=0, n=4000)
    b = r.samples[:, r.labels.index("b")]
    assert np.all(np.isfinite(r.samples))
    assert np.percentile(b, 0.5) < b_true < np.percentile(b, 99.5)       # broad: b-rp degeneracy
    # the profile cannot end at b = 1 + rp in the original coordinates; the bounded retry completes it
    assert r.diagnostics["complete"] and r.diagnostics["bounded_retry"] is True


def test_bounded_coords_graze():
    """beta = b / (1 + rp) in the retry coordinates: u -> x -> u round trip, b < 1 + rp for every
    beta < 1, and the log Jacobian against finite differences."""
    from pizzetti.marginal import _BoundedCoords
    t = np.linspace(-0.12, 0.12, 200)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.95 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    m = pizzetti.MarginalTransitModel(p, t, np.ones_like(t), 1e-3, free=("t0", "rp", "a", "b"),
                                      bounds={"t0": (-0.02, 0.02), "rp": (0.03, 0.5), "a": (3.0, 20.0),
                                              "b": (0.0, 1.5)})
    T = _BoundedCoords(m, graze=True)
    rng = np.random.default_rng(0)
    U = rng.normal(0, 1.5, (50, 4))
    U[:, 3] = rng.uniform(0.0, 0.999, 50)                  # beta
    X, lj = T.to_x(U)
    P = m.to_physical(X)
    assert np.all(P[:, 3] < 1.0 + P[:, 1])
    assert np.allclose(P[:, 3] / (1.0 + P[:, 1]), U[:, 3])
    assert np.allclose(T.to_u(X), U, atol=1e-8)
    for i in range(5):
        J = np.empty((4, 4))
        for j in range(4):
            h = 1e-6
            up, um = U[i].copy(), U[i].copy()
            up[j] += h
            um[j] -= h
            J[:, j] = (T.to_x(up[None])[0][0] - T.to_x(um[None])[0][0]) / (2 * h)
        assert abs(np.log(abs(np.linalg.det(J))) - lj[i]) < 1e-5


@pytest.mark.parametrize("w_deg", [90.0, -90.0, 30.0])
def test_eccentric_support_edge(w_deg):
    """The prior support is the region with a transit: b_conj = b (1 - e^2) / (1 + e sin w) < 1 + rp,
    not b < 1 + rp (for e = 0.3, w = 90 deg the edge lies at b = 1.43 (1 + rp)). Fast and general
    paths agree; the retry coordinates map beta = b_conj / (1 + rp) and back."""
    from pizzetti.marginal import _BoundedCoords
    t = np.linspace(-0.2, 0.2, 400)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 5.0, 0.1, 12.0
    p.inc = 89.0
    p.ecc, p.w, p.limb_dark, p.u = 0.3, w_deg, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 5e-4 * np.random.default_rng(0).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 5e-4, free=("t0", "rp", "a", "b"), bounds={"rp": (0.01, 0.5)})
    g = (1 - 0.09) / (1 + 0.3 * math.sin(math.radians(w_deg)))
    edge = 1.1 / g
    X = np.array([[0.0, 0.1, 12.0, edge * (1 - 1e-4)], [0.0, 0.1, 12.0, edge * (1 + 1e-4)]])
    lp = m.log_prob(X)
    assert np.isfinite(lp[0]) and lp[1] == -np.inf
    Xt = m.to_internal(np.array([[0.0, 0.1, 12.0, 12.0 * math.cos(math.radians(89.0))]]))   # the truth
    fast = m.log_prob_internal(Xt)
    m.fast, m._fused = False, False
    assert abs(m.log_prob_internal(Xt)[0] - fast[0]) < 1e-6 * max(1.0, abs(fast[0]))
    m.fast = True
    Xi = m.to_internal(X[:1])
    T = _BoundedCoords(m, graze=True)
    U = T.to_u(Xi)
    assert abs(U[0, 3] - (1 - 1e-4)) < 1e-9
    assert np.allclose(T.to_x(U)[0], Xi, rtol=1e-9, atol=1e-12)


def test_draw_c_singular_row():
    """A coefficient covariance without a Cholesky factor (geometry that hardly constrains the
    limb darkening) gives a NaN draw for that row only (weight 0 downstream), not an error."""
    from pizzetti.marginal import _draw_c
    rng = np.random.default_rng(0)
    C = np.tile([0.2, 0.5, 0.9], (3, 1))
    S = np.tile(np.diag([1e-3, 1e-3, 1e-3]), (3, 1, 1))
    S[1] = np.outer([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) * 1e-3 - np.eye(3) * 1e-6    # not PD
    c = _draw_c(C, S, rng)
    assert np.all(np.isnan(c[1])) and np.all(np.isfinite(c[[0, 2]]))
    assert np.all(c[[0, 2], 0] >= 0) and np.all(np.diff(c[[0, 2]], axis=1) >= 0)


def test_gibbs_cone_deep_tail():
    """Gibbs draws in the cone 0 <= c0 <= c1 <= c2 for a mean far outside it (the conditional of a
    coordinate 10-40 sd below its wall): the inverse-CDF update saturated there and returned draws
    below the wall (negative limb intensity, q1 > 1)."""
    from pizzetti.marginal import _gibbs_cone_one
    rng = np.random.default_rng(5)
    cov = np.array([[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]])
    for mu in ([-0.2, 0.5, 0.9], [0.3, 0.2, 0.9], [0.1, 0.5, 0.1], [-0.4, -0.3, -0.35]):
        for _ in range(20):
            c = _gibbs_cone_one(np.array(mu), cov, rng.random(300 * 3), 300)
            assert c[0] >= 0 and c[1] >= c[0] and c[2] >= c[1]


def test_profile_is_single_grid_point():
    """Every step away from the MAP fails (finite target only at the MAP's b): profile IS reports
    failure (k-hat = inf) instead of building a proposal from one grid point."""
    from pizzetti._sampling import profile_is

    def logp(X):
        out = -0.5 * np.sum(X[:, 1:] ** 2, axis=1)
        return np.where(np.abs(X[:, 0] - 0.5) < 1e-12, out, -np.inf)
    r = profile_is(logp, np.array([0.5, 0.1, -0.1]), np.ones(3), prof=0, n=500,
                   rng=np.random.default_rng(1), lo_wall=None)
    assert not np.isfinite(r["khat"]) and not r["complete"] and r["n_grid"] == 1


def test_stuck_walkers_reset():
    """A walker at a local mode (log posterior 1400 below the others, as for a stretch-move
    ensemble started away from the posterior) is found and moved next to another walker; walkers
    spread over the typical set are not."""
    from pizzetti._sampling import reset_walkers, stuck_walkers
    rng = np.random.default_rng(3)
    lp = -0.5 * rng.chisquare(4, size=(500, 32)) - 4400.0
    assert stuck_walkers(lp).size == 0
    lp[:, 6] = -5777.0 + 0.5 * rng.standard_normal(500)
    assert list(stuck_walkers(lp)) == [6]
    X = rng.standard_normal((32, 4)) * 0.01 + [0.0, 0.155, 8.9, 0.66]
    X[6] = [0.0, 0.138, 11.0, 0.14]
    Y = reset_walkers(X, [6], rng, lambda Z: np.where(Z[:, 3] >= 0, 0.0, -np.inf))
    assert np.allclose(np.delete(Y, 6, 0), np.delete(X, 6, 0))
    G = np.delete(X, 6, 0)
    assert np.max(np.abs(Y[6] - G.mean(axis=0)) / G.std(axis=0)) < 5


def test_fit_is_with_b_bounds():
    """A posterior cut off by user bounds on b: the bounds are walls of the profile grid, so the
    grid is complete and IS is used (it fell back to emcee before)."""
    t = np.linspace(-0.12, 0.12, 2000)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.85 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 1e-3 * np.random.default_rng(1).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-3, free=("t0", "rp", "a", "b"), bounds={"b": (0.0, 0.9)})
    r = m.fit(method="auto", seed=0, n=4000)
    assert r.diagnostics["method"] == "profile-IS" and r.diagnostics["complete"]
    assert r.samples[:, r.labels.index("b")].max() < 0.9


def test_fit_is_bounded_retry():
    """Unresolved ingress (30-min cadence, 1-h transit) with a/R* pressed against its upper bound:
    the profile in the original coordinates fails (MAP on the bound), the retry in unbounded
    coordinates succeeds and agrees with emcee."""
    rng = np.random.default_rng(4)
    t = np.concatenate([e * 2.6 + np.arange(-0.07, 0.07, 1765.5 / 86400) for e in range(20)])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 2.6, 0.065, 19.5
    p.inc = math.degrees(math.acos(0.5 / 19.5))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.5, 0.2]
    y = pizzetti.TransitModel(p, t, supersample_factor=15, exp_time=1765.5 / 86400,
                              offsets="midpoint").light_curve(p)
    y = y + 2e-4 * rng.standard_normal(t.size)
    bounds = {"t0": (-0.02, 0.02), "rp": (0.03, 0.15), "a": (3.0, 20.0), "b": (0.0, 0.9)}
    m = pizzetti.MarginalTransitModel(p, t, y, 2e-4, free=("t0", "rp", "a", "b"), bounds=bounds,
                                      supersample_factor=15, exp_time=1765.5 / 86400, offsets="midpoint")
    r = m.fit(method="auto", seed=0, n=4000)
    assert r.diagnostics["method"] == "profile-IS" and r.diagnostics["bounded_retry"] is True
    e = m.fit(method="emcee", seed=1, n=4000)
    for q in ("rp", "a", "b"):
        x, y_ = r.samples[:, r.labels.index(q)], e.samples[:, e.labels.index(q)]
        assert abs(np.median(x) - np.median(y_)) < 0.2 * np.std(y_)
        assert abs(np.std(x) / np.std(y_) - 1) < 0.2


def test_pmc_refine_curved_target():
    """pmc_refine (AMIS) on a curved, skewed target (x ~ N(0, 1), y | x ~ N(x^2, 0.3^2): E y = 1,
    sd y = sqrt(2 + 0.09)), starting from draws of a poor Gaussian proposal."""
    from pizzetti._sampling import pmc_refine

    def logp(X):
        return -0.5 * X[:, 0] ** 2 - 0.5 * ((X[:, 1] - X[:, 0] ** 2) / 0.3) ** 2

    rng = np.random.default_rng(2)
    X0 = rng.standard_normal((4000, 2)) * [0.7, 0.7] + [0.0, 0.5]
    lq = -0.5 * np.sum(((X0 - [0.0, 0.5]) / 0.7) ** 2, axis=1)
    r = pmc_refine(logp, X0, logp(X0) - lq, rng, n=4000)
    w = r["w"] / r["w"].sum()
    my = np.sum(w * r["theta"][:, 1])
    sy = np.sqrt(np.sum(w * (r["theta"][:, 1] - my) ** 2))
    assert r["khat"] < 0.7 and r["ess"] > 5000
    assert abs(my - 1.0) < 0.05 and abs(sy - math.sqrt(2.09)) < 0.15        # (heavy tail in y)


def test_pmc_refine_adaptive_with_box_component():
    """Adaptive AMIS (min_iters) from start draws that miss a second mode entirely: the fixed
    component over the whole box keeps a proposal density there, the iterations find the mode and
    stop once k-hat, ESS and the quantiles are stable (converged); the mode masses are right."""
    from pizzetti._sampling import pmc_refine
    from scipy.special import logsumexp

    def logp(X):                         # 0.7 N((0, 0), 0.3^2) + 0.3 N((4, 2), 0.3^2), inside |x| < 8
        a = np.log(0.7) - 0.5 * np.sum((X / 0.3) ** 2, axis=1)
        b = np.log(0.3) - 0.5 * np.sum(((X - [4.0, 2.0]) / 0.3) ** 2, axis=1)
        return np.where(np.all(np.abs(X) < 8, axis=1), logsumexp(np.column_stack([a, b]), axis=1), -np.inf)

    rng = np.random.default_rng(5)
    X0 = 0.3 * rng.standard_normal((2000, 2))                # draws of the first mode only
    lw0 = logp(X0) + 0.5 * np.sum((X0 / 0.3) ** 2, axis=1)
    r = pmc_refine(logp, X0, lw0, rng, n=2000, iters=30, min_iters=3,
                   extra=[(0.05, np.zeros(2), np.diag([16.0, 16.0]))])
    w = r["w"] / r["w"].sum()
    frac = np.sum(w[r["theta"][:, 0] > 2.0])
    assert r["converged"] and r["khat"] < 0.7
    assert abs(frac - 0.3) < 0.03, frac
    assert len(r["history"]) < 30 and all("dq" in h for h in r["history"])


@pytest.mark.parametrize("law,u", [("quadratic", [0.05, 0.4]), ("squareroot", [0.1, 0.3]), ("linear", [0.3])])
def test_log_prob_parts_flat_factorisation(law, u):
    """log posterior (physical prior) - log P(cone) = log posterior with a flat coefficient prior
    (+ const), and log P <= 0; the physical value is unchanged by parts=True."""
    t = np.linspace(-0.12, 0.12, 1500)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.4 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, law, u
    y = pizzetti.TransitModel(p, t).light_curve(p) + 2e-4 * np.random.default_rng(0).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 2e-4, free=("t0", "rp", "a", "b"))
    mf = pizzetti.MarginalTransitModel(p, t, y, 2e-4, free=("t0", "rp", "a", "b"), ld_prior="flat")
    x0 = m.to_internal(m.fixed_physical_free())[0]
    X = x0 + np.random.default_rng(1).normal(0, 1, (100, 4)) * m._default_scales(x0) * [3, 3, 3, 4]
    lp, lP = m.log_prob_internal(X, parts=True)
    ok = np.isfinite(lp)
    assert ok.sum() > 50 and np.all(lP[ok] <= 1e-12)
    d = (lp - lP)[ok] - mf.log_prob_internal(X)[ok]
    assert np.ptp(d) < 1e-6
    assert np.array_equal(lp, m.log_prob_internal(X))


def test_profile_is_weight_target_cliff():
    """A smooth Gaussian cut off by a steep wall through its mode (x + b < 0.6): the proposal is
    built on the smooth part, the weights refer to the cut-off target (bounded by 1), and the
    moments of the target are recovered (reference: 2-d quadrature)."""
    from pizzetti._sampling import profile_is

    def smooth(X):
        X = np.atleast_2d(X)
        return -0.5 * ((X[:, 0] - 0.1) / 0.1) ** 2 - 0.5 * ((X[:, 1] - 0.5) / 0.05) ** 2

    def wall(X):
        X = np.atleast_2d(X)
        z = (0.6 - X[:, 0] - X[:, 1]) / 0.002
        return -np.logaddexp(0.0, -z)                        # log sigmoid, steep

    def target(X):
        return smooth(X) + wall(X)

    g0, g1 = np.meshgrid(np.linspace(-0.4, 0.6, 801), np.linspace(0.2, 0.8, 801), indexing="ij")
    G = np.column_stack([g0.ravel(), g1.ravel()])
    pg = np.exp(target(G) - target(G).max())
    pg /= pg.sum()
    mean_b = pg @ G[:, 1]
    sd_b = math.sqrt(pg @ (G[:, 1] - mean_b) ** 2)
    th0, sc = np.array([0.1, 0.5]), np.array([0.1, 0.05])
    plain = profile_is(target, th0, sc, prof=1, n=4000, rng=np.random.default_rng(0), lo_wall=None)
    fact = profile_is(smooth, th0, sc, prof=1, n=4000, rng=np.random.default_rng(0), lo_wall=None,
                      logp_target=target)
    assert fact["khat"] < 0.5 and fact["ess"] > 1000
    w = fact["w"] / fact["w"].sum()
    mb = np.sum(w * fact["theta"][:, 1])
    assert abs(mb - mean_b) < 0.05 * sd_b
    assert abs(math.sqrt(np.sum(w * (fact["theta"][:, 1] - mb) ** 2)) - sd_b) < 0.05 * sd_b
    assert fact["khat"] < plain["khat"] or not np.isfinite(plain["khat"])


def test_profile_is_start_on_bound():
    """A start exactly on a (strict) prior bound has log posterior -inf; it is moved inside."""
    t = np.linspace(-0.12, 0.12, 1500)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.4 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 3e-4 * np.random.default_rng(0).standard_normal(t.size)
    p.t0 = 0.01                                              # start t0 on the upper bound
    m = pizzetti.MarginalTransitModel(p, t, y, 3e-4, free=("t0", "rp", "a", "b"), bounds={"t0": (-0.01, 0.01)})
    x0 = m.to_internal(m.fixed_physical_free())[0]
    assert not np.isfinite(m.log_prob_internal(x0[None])[0])
    r = m.fit(method="auto", seed=0, n=4000)
    assert r.diagnostics["method"] == "profile-IS"
    assert abs(np.median(r.samples[:, r.labels.index("t0")])) < 0.002


def test_threads_identical(monkeypatch):
    """threads > 1 (rows over threads for batches, data points over threads for single
    evaluations): results independent of the number of threads, equal to the serial ones to
    rounding; threads must be a positive integer or "all"."""
    from pizzetti import marginal as M
    monkeypatch.setattr(M, "PAR_MIN_POINTS", 1000)        # exercise the data-point path
    monkeypatch.setattr(M, "PAR_CHUNK", 300)
    rng = np.random.default_rng(3)
    t = np.concatenate([e * 3.5 + np.arange(-0.15, 0.15, 120 / 86400) for e in range(10)])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.5, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.4 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 3e-4 * rng.standard_normal(t.size)
    kw = dict(free=("t0", "rp", "a", "b"))
    m1 = pizzetti.MarginalTransitModel(p, t, y, 3e-4, **kw)
    m2 = pizzetti.MarginalTransitModel(p, t, y, 3e-4, threads=2, **kw)
    m3 = pizzetti.MarginalTransitModel(p, t, y, 3e-4, threads=3, **kw)
    x0 = m1.to_internal(m1.fixed_physical_free())[0]
    X = x0 + rng.normal(0, 1, (40, 4)) * m1._default_scales(x0)
    for rows in (X[:1], X[:5], X):                         # single, small batch, large batch
        a, la = m1.log_prob_internal(rows, parts=True)
        b, lb = m2.log_prob_internal(rows, parts=True)
        c, lc = m3.log_prob_internal(rows, parts=True)
        assert np.array_equal(b, c) and np.array_equal(lb, lc)
        ok = np.isfinite(a)
        assert np.array_equal(ok, np.isfinite(b))
        assert np.allclose(a[ok], b[ok], rtol=1e-12, atol=1e-9) and np.allclose(la, lb, rtol=1e-12, atol=1e-9)
    r2, r3 = m2.fit(method="is", seed=0, n=2000), m3.fit(method="is", seed=0, n=2000)
    assert np.array_equal(r2.samples, r3.samples)
    for bad in (0, -1, 1.5, "many"):
        with pytest.raises(ValueError):
            pizzetti.MarginalTransitModel(p, t, y, 3e-4, threads=bad, **kw)


def _shallow_toy(sigma=1e-3, seed=1):
    """A barely detected transit (rp 0.03, S/N ~5, 6 transits at 30-min cadence)."""
    exp = 1765.5 / 86400
    rng = np.random.default_rng(seed)
    t = np.concatenate([e * 3.0 + np.arange(-0.25, 0.25, exp) for e in range(6)])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.0, 0.03, 10.0
    p.inc = math.degrees(math.acos(0.5 / 10.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    y = pizzetti.TransitModel(p, t, supersample_factor=15, exp_time=exp, offsets="midpoint").light_curve(p)
    y = y + sigma * rng.standard_normal(t.size)
    return pizzetti.MarginalTransitModel(p, t, y, sigma, free=("t0", "rp", "a", "b"),
                                         bounds={"t0": (-0.03, 0.03), "rp": (0.0, 0.5), "a": (2.0, 40.0)},
                                         supersample_factor=15, exp_time=exp, offsets="midpoint")


def _long_period_model(transit, seed=1):
    """P = 95 d, rp 0.026, a/R* 123, 400 points over 1.2 d around t0; the TransitParams start is
    non-transiting (inc 89.5 deg: b = 1.07 > 1 + rp). transit: data with a transit at b = 0.5,
    else flat noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.6, 0.6, 400)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 95.0, 0.026, 123.0
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    p.inc = math.degrees(math.acos(0.5 / 123.0))
    y = (pizzetti.TransitModel(p, t).light_curve(p) if transit else np.ones_like(t)) + 5e-4 * rng.standard_normal(t.size)
    p.inc = 89.5
    return pizzetti.MarginalTransitModel(p, t, y, 5e-4, free=["t0", "rp", "a", "b"],
                                         bounds=dict(t0=(-0.1, 0.1), rp=(0.003, 0.5), a=(1.5, 2000.0), b=(0.0, 1.2)))


def test_expand_no_valid_ld_draw_uses_prior():
    """Geometries without any valid limb-darkening draw (no transit in the data window): the
    coefficients come from the prior and f0 from the theta moments; this fallback raised
    UnboundLocalError (it read the c-space moments C, which the default prior does not compute)."""
    m = _long_period_model(transit=False)
    X = np.tile(m.fixed_physical_free(), (20, 1))           # b = 1.07: no transit at all
    S, labels, w = m.expand(X, ld="u", rng=np.random.default_rng(0), resample=False)
    assert m.last_ld_from_prior
    f0 = S[:, labels.index("f0")]
    assert np.all(np.isfinite(f0)) and np.all(np.abs(f0 - 1.0) < 1e-3)


def test_fit_from_non_transiting_start():
    """A non-transiting start (b = 1.07 > 1 + rp; a/R* undefined in the T14 parametrisation) is
    moved inside the transiting range: on flat data the fit returns a broad posterior instead of
    the start repeated (k-hat inf); on data with a transit it finds the transit."""
    r = pizzetti_fit_quiet(_long_period_model(transit=False))
    rp = r.samples[:, r.labels.index("rp")]
    assert r.diagnostics["is_ok"] and np.std(rp) > 0 and np.all(np.isfinite(r.samples))
    r = pizzetti_fit_quiet(_long_period_model(transit=True))
    rp = r.samples[:, r.labels.index("rp")]
    assert r.diagnostics["is_ok"] and abs(np.median(rp) - 0.026) < 0.005


def pizzetti_fit_quiet(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(ld="u", n=4000, seed=0)


def test_fit_budget():
    """max_evals / max_time stop profile importance sampling (a hard posterior: the S/N ~5 toy needs
    the retry and the rescue, ~7e4 evaluations) with the best attempt so far, flagged; a budget that
    does not bind changes nothing; a budget too small for the first proposal raises."""
    m = _shallow_toy()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = m.fit(seed=0, n=4000, max_evals=20_000)
    d = r.diagnostics
    assert d["budget_exceeded"] and not d["is_ok"] and d["n_eval"] <= 20_000
    assert any("budget" in str(x.message) for x in wl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = m.fit(seed=0, n=4000)
        r1 = m.fit(seed=0, n=4000, max_evals=10 ** 8, max_time=1e6)
    assert not r1.diagnostics["budget_exceeded"] and r1.diagnostics["n_eval"] == r0.diagnostics["n_eval"]
    assert np.array_equal(r1.samples, r0.samples)
    with pytest.raises(RuntimeError, match="budget"):
        m.fit(seed=0, n=4000, max_evals=10)


def test_bounded_coords_ignore_t14_column():
    """_conj_factor (e, w) and _BoundedCoords._t14_range (per, rp, b) are called by
    _BoundedCoords.to_x before the T14 column is set; a T14 of 0 raised ZeroDivisionError in the
    T14 -> a/R* conversion (bounded retry for an ultra-short-period planet at 30-min cadence)."""
    m = _shallow_toy()
    assert m.t14
    X = m.to_internal(m.fixed_physical_free())
    X[:, m.free.index("a")] = 0.0
    assert np.allclose(m._conj_factor(X), 1.0)
    from pizzetti.marginal import _BoundedCoords
    lo, hi = _BoundedCoords(m, graze=True)._t14_range(X)          # the T14 slot is ignored there
    assert np.all(np.isfinite(lo)) and np.all(hi > lo)
    X[:, m.free.index("a")] = 1e-205                                # T14 -> 0: sin^2 underflows
    assert np.all(m.log_prob_internal(X) == -np.inf)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.limb_dark, p.u = 0.0, 3.0, 0.03, 10.0, 89.0, 0.0, 90.0, "quadratic", [0.4, 0.25]
    m2 = pizzetti.MarginalTransitModel(p, m.t, m.y, 1e-3, free=("t0", "per", "rp", "a", "b"))
    X2 = m2.to_internal(m2.fixed_physical_free())
    X2[:, 1] = 0.0                                                   # per = 0 (an optimiser's step)
    assert np.all(m2.log_prob_internal(X2) == -np.inf)


def _kernel_loglike(m, rp, a, b, t0=-0.004):
    from pizzetti import _marginal as K, _series
    from pizzetti.marginal import _A_HALF
    B = K.design_theta(m.tt, m.k, m.wts, t0, 3.0, rp, a, b, 0.0, math.pi / 2, m.tol, _A_HALF, _series._GTAB)
    v, lP = K.marg_theta_parts(B, m.yw, K.GL_X, K.GL_W)
    return v + lP


@pytest.mark.parametrize("rp,a,b,ref", [(0.03, 10.0, 0.5, -68.87821618952417),
                                        (0.03, 30.0, 0.2, -72.03724950431679),
                                        (0.01, 30.0, 0.2, -79.11943978503167),
                                        (1e-3, 10.0, 0.5, -80.42118213524589)])
def test_marginal_likelihood_reference(rp, a, b, ref):
    """The quadratic marginal likelihood against 50-digit references (mpmath: the normal equations
    and the cone integral, from the same occultation terms; tests/_mpref.py). In the old
    parametrisation (normal equations in c, cond ~ 1 / rp^4) these were off by 4e-3, 0.2, 5.3 and
    13 000 nats."""
    m = _shallow_toy()
    assert abs(_kernel_loglike(m, rp, a, b) - ref) < 1e-8


def test_marginal_likelihood_small_rp_limit():
    """As rp -> 0 the marginal likelihood tends to the no-transit value: f0 ~ Gaussian with the
    limb darkening free in the cone, whose cross-section at f0 = 1 has area 1/5:
        log L -> -chi2_0 / 2 + log(int_0^inf f0^2 N(f0) df0 / 5) + log 6 - 3/2 log(2 pi)
    (finite; the old parametrisation gave -inf below rp ~ 3e-4)."""
    m = _shallow_toy()
    w, yw = m.wts, m.yw
    G00 = np.sum(w * w)
    f0 = np.sum(w * yw) / G00
    chi2 = np.sum((yw - f0 * w) ** 2)
    m2 = math.sqrt(2 * math.pi / G00) * (f0 * f0 + 1.0 / G00)
    lim = -0.5 * chi2 + math.log(m2 / 5.0) + math.log(6.0) - 1.5 * math.log(2 * math.pi)
    vals = [_kernel_loglike(m, rp, 10.0, 0.5) for rp in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    assert np.all(np.isfinite(vals))
    assert abs(vals[-1] - lim) < 1e-5
    assert abs(vals[-2] - lim) < 1e-4


def test_marginal_likelihood_smooth_shallow():
    """No numerical noise for a shallow transit on a long light curve (Kepler-78 b: 0.07 nats rms in
    the old parametrisation): log L along a short line in b is a smooth curve to 1e-7 nats."""
    rng = np.random.default_rng(3)
    t = np.concatenate([e * 0.9 + np.arange(-0.06, 0.06, 60 / 86400) for e in range(40)])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 0.9, 0.012, 3.5
    p.inc = math.degrees(math.acos(0.4 / 3.5))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.5, 0.2]
    y = pizzetti.TransitModel(p, t).light_curve(p) + 2e-4 * rng.standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 2e-4, free=("t0", "rp", "a", "b"))
    h = np.linspace(-1, 1, 41) * 1e-4
    X = np.column_stack([np.zeros(41), np.full(41, 0.012), np.full(41, 3.5), 0.4 + h])
    lp = m.log_prob(X)
    c = np.polyfit(h, lp, 3)
    assert np.std(lp - np.polyval(c, h)) < 1e-7


def test_ld_draws_theta_shallow():
    """Limb-darkening draws in theta = (f0, c1, c2) are valid (in the cone) for any transit depth; in c
    the coefficient covariance of a very shallow transit is numerically singular (the Gibbs fallback
    divided by zero at rp ~ 1e-4)."""
    from pizzetti import marginal as M, _marginal as K, _series
    from pizzetti.marginal import _A_HALF
    m = _shallow_toy()
    P = np.array([[-0.004, 3.0, rp, 10.0, 0.5, 0.0, math.pi / 2] for rp in (3e-2, 1e-3, 1e-4, 1e-6)])
    T, S = K.theta_moments_batch(P, m.tt, m.k, m.wts, m.yw, m.tol, _A_HALF, _series._GTAB)
    c = M._draw_c_theta(np.repeat(T, 50, 0), np.repeat(S, 50, 0), np.random.default_rng(0))
    assert np.all(np.isfinite(c))
    assert np.all((c[:, 0] >= 0) & (c[:, 1] >= c[:, 0]) & (c[:, 2] >= c[:, 1]))


def test_fit_barely_detected_transit():
    """S/N ~5: the posterior spreads over most of the prior (rp up to its bound, a/R* over its box);
    importance sampling converges without a warning (it did not in 30 AMIS iterations while the
    likelihood was inaccurate at small rp)."""
    m = _shallow_toy()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = m.fit(method="auto", seed=0, n=4000)
    assert r.diagnostics["is_ok"]
    assert not any("diagnostics failed" in str(x.message) for x in wl)
    rp = r.samples[:, r.labels.index("rp")]
    assert 0.02 < np.median(rp) < 0.08 and np.percentile(rp, 97.5) > 0.2


_CONE_FACES = np.array([[6.0, -2.0, -3.0], [-6.0, 3.0, 3.0], [0.0, -1.0, 1.0]])    # (c0, c1 - c0, c2 - c1) from theta
_IN_CONE = np.array([1.0, 0.97, 1.21])                                             # theta, c = (0.43, 0.97, 1.21)


def _cone_design(rp, b, sigma):
    """Whitened theta design of a white-noise light curve."""
    from pizzetti import _series
    from pizzetti.marginal import _A_HALF
    t = np.linspace(-0.25, 0.25, 400)
    return K.design_theta(t, 1, np.full(t.size, 1.0 / sigma), 0.0, 4.0, rp, 12.9, b, 0.0, math.pi / 2, 1e-9,
                          _A_HALF, _series._GTAB)


def _beyond_face(B, face, nsd):
    """Mean nsd marginal sd beyond one face of the cone (0: c0 = 0, 1: c0 = c1, 2: c1 = c2)."""
    Gi = np.linalg.inv(B.T @ B)
    a = _CONE_FACES[face]
    sd = math.sqrt(a @ Gi @ a)
    return _IN_CONE - (a @ _IN_CONE / sd + nsd) * Gi @ a / sd


def _orthant_reference(B, th, n=300):
    """ln P of the cone by the Genz transform in log space on an n x n grid, with none of the kernel's
    shortcuts (independent of log_orthant_prob and of the ray integral)."""
    from scipy.special import log_ndtr, logsumexp, ndtri_exp
    Gi = np.linalg.inv(B.T @ B)
    m, S = _CONE_FACES @ th, _CONE_FACES @ Gi @ _CONE_FACES.T
    o = np.argsort(m / np.sqrt(np.diag(S)))
    m, S = m[o], S[np.ix_(o, o)]
    L = np.linalg.cholesky(S)
    t, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t + 1.0)
    x, w = t * t * (3.0 - 2.0 * t), 0.5 * w * 6.0 * t * (1.0 - t)
    le1 = log_ndtr(m[0] / L[0, 0])
    z1 = -ndtri_exp(np.log1p(-x) + le1)
    le2 = log_ndtr((m[1] + L[1, 0] * z1) / L[1, 1])
    z2 = -ndtri_exp(np.log1p(-x)[None, :] + le2[:, None])
    le3 = log_ndtr((m[2] + L[2, 0] * z1[:, None] + L[2, 1] * z2) / L[2, 2])
    return le1 + logsumexp(le2 + logsumexp(le3, b=w[None, :], axis=1), b=w)


@pytest.mark.parametrize("face", [0, 1, 2])
@pytest.mark.parametrize("nsd", [3.0, 4.0, 6.0, 8.0])
@pytest.mark.parametrize("rp,b,sigma", [(0.1, 0.3, 1.2e-3), (0.1, 0.3, 7e-4), (0.05, 0.46, 1.1e-4), (0.012, 0.73, 1.2e-5),
                                        (0.012, 0.73, 4e-6)])
def test_cone_probability_outside_cone(rp, b, sigma, face, nsd):
    """The Gaussian's mean outside the cone (limb-darkening widths of 1/12 to 1/60 of the cone's cross-section): the
    ray integral's outer rule follows the mass to the nearest point of the triangle. Centred on the mean it was
    off by 0.08 nats at 3-4 sd and by nats beyond 6 sd, worst beyond c0 = 0, whatever the width."""
    pytest.importorskip("scipy")
    B = _cone_design(rp, b, sigma)
    th = _beyond_face(B, face, nsd)
    ref = _orthant_reference(B, th)
    assert abs(ref - _orthant_reference(B, th, 150)) < 2e-5
    assert abs(K.marg_theta_parts(B, B @ th, K.GL_X, K.GL_W)[1] - ref) < 2e-4


def test_cone_probability_smooth_leaving_cone():
    """ln P along a line of the mean out through c0 = 0 (0.04 sd steps): the error stays small, without jumps."""
    pytest.importorskip("scipy")
    B = _cone_design(0.012, 0.73, 1.2e-5)
    th1 = _beyond_face(B, 0, 8.0)
    err = []
    for s in np.linspace(0.0, 1.0, 301):
        th = _IN_CONE + s * (th1 - _IN_CONE)
        err.append(K.marg_theta_parts(B, B @ th, K.GL_X, K.GL_W)[1] - _orthant_reference(B, th))
    err = np.array(err)
    assert np.abs(err).max() < 2e-4
    assert np.abs(np.diff(err)).max() < 5e-5


def test_cone_probability_blend_inside_cone():
    """A mean inside the cone, through the blend to the Genz rule (nearest constraint 4.5 to 6 sd), to P = 1."""
    pytest.importorskip("scipy")
    B1 = _cone_design(0.1, 0.3, 1.0)
    Gi = np.linalg.inv(B1.T @ B1)
    z1 = (_CONE_FACES @ _IN_CONE / np.sqrt(np.diag(_CONE_FACES @ Gi @ _CONE_FACES.T))).min()
    for z in (3.0, 4.5, 5.0, 5.25, 5.5, 6.0, 7.0):
        B = B1 * z / z1
        lP = K.marg_theta_parts(B, B @ _IN_CONE, K.GL_X, K.GL_W)[1]
        assert abs(lP - _orthant_reference(B, _IN_CONE)) < 2e-5
