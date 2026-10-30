"""MarginalTransitModel: likelihood, coordinates, scope errors, sampler API."""
import math

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
