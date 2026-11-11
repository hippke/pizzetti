"""MarginalTransitModel(gp=GP()): whitening, likelihood, prediction, interface, sampling."""
import math

import numpy as np
import pytest
from scipy.linalg import solve_triangular

import pizzetti
from pizzetti import _gp, _marginal as K

P, A, B, RP = 3.5, 9.0, 0.35, 0.10


def params():
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, P, RP, A
    p.inc = math.degrees(math.acos(B / A))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.40, 0.25]
    return p


def sho(dt, amp, tau):
    x = np.abs(dt) / (tau * math.sqrt(2.0))
    return amp ** 2 * np.exp(-x) * (np.cos(x) + np.sin(x))


def data(amp=1e-3, tau=0.15, sigma=5e-4, seed=3, n=160):
    """One transit, with a GP draw and white noise, on a short regular grid."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) - (n - 1) / 2) * 5.0 / 1440.0
    p = params()
    gp = np.linalg.cholesky(sho(t[:, None] - t[None, :], amp, tau) + 1e-13 * np.eye(n)) @ rng.standard_normal(n)
    return t, pizzetti.TransitModel(p, t).light_curve(p) + gp + sigma * rng.standard_normal(n)


def box(amp=(1e-5, 0.1), tau=(3e-3, 20.0), err_scale=(0.2, 5.0)):
    return pizzetti.GP(amp=amp, tau=tau, err_scale=err_scale)


def x_true(m, ln_amp=math.log(1e-3), ln_tau=math.log(0.15), ln_s=0.0):
    X = m.fixed_physical_free()
    X[0, m.gp_cols] = [ln_amp, ln_tau, ln_s]
    return X


# -- the Kalman kernel against dense algebra ------------------------------------------------------
@pytest.mark.parametrize("amp,tau,s", [(1e-3, 0.01, 5e-4), (1e-3, 0.1, 5e-4), (3e-3, 2.0, 5e-4), (1e-5, 0.1, 5e-3),
                                       (0.1, 100.0, 5e-4)])
def test_whiten_is_cholesky(amp, tau, s):
    rng = np.random.default_rng(1)
    t = np.sort(np.concatenate([rng.uniform(0, 1, 120), rng.uniform(5, 6, 120)]))       # irregular, with a gap
    Y0 = rng.standard_normal((240, 4))
    var = s ** 2 * rng.uniform(0.5, 2.0, 240)                                            # per-point errors
    L = np.linalg.cholesky(sho(t[:, None] - t[None, :], amp, tau) + np.diag(var))
    Y = Y0.copy()
    ld = _gp.whiten(*_gp.time_table(t, 0.0), var, amp, tau, Y)
    D = solve_triangular(L, Y0, lower=True)
    assert np.max(np.abs(Y - D)) < 1e-6 * np.max(np.abs(D))
    assert abs(ld - np.log(np.diag(L)).sum()) < 1e-6


def test_predict_is_conditional_gaussian():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 3, 150))
    var = (5e-4) ** 2 * rng.uniform(0.5, 2.0, 150)
    r = 1e-3 * rng.standard_normal(150)
    amp, tau = 1e-3, 0.2
    Kt = sho(t[:, None] - t[None, :], amp, tau)
    Si = np.linalg.inv(Kt + np.diag(var))
    mean, v = _gp.predict(t, var, amp, tau, r)
    assert np.max(np.abs(mean - Kt @ Si @ r)) < 1e-10
    assert np.max(np.abs(v - np.diag(Kt - Kt @ Si @ Kt))) < 1e-12


# -- the likelihood -------------------------------------------------------------------------------
def test_likelihood_against_dense():
    """Whole model with supersampling: differences of ln L against dense algebra in the coefficients."""
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"),
                                      supersample_factor=3, exp_time=5.0 / 1440.0)
    from pizzetti._series import _GTAB
    from pizzetti.terms import _A_HALF
    rng = np.random.default_rng(4)
    Xi = m.to_internal(x_true(m))[0] + rng.normal(size=(12, 7)) * np.array([2e-3, 4e-3, 3e-3, 0.05, 0.3, 0.3, 0.05])
    Xi[:, 3] = np.abs(Xi[:, 3])
    out = m.log_prob_internal(Xi)
    Pc, lj = m._canonical_internal(Xi)
    ref = np.empty(12)
    for i in range(12):
        Sg = sho(t[:, None] - t[None, :], math.exp(Xi[i, 4]), math.exp(Xi[i, 5])) + (5e-4 * math.exp(Xi[i, 6])) ** 2 * np.eye(len(t))
        L = np.linalg.cholesky(Sg)
        Bm = K.design_theta(m.tt, m.k, np.ones(len(t)), *Pc[i], m.tol, _A_HALF, _GTAB)
        v, lP = K.marg_theta_parts(np.ascontiguousarray(solve_triangular(L, Bm, lower=True)),
                                   np.ascontiguousarray(solve_triangular(L, y, lower=True)), K.GL_X, K.GL_W)
        ref[i] = v + lP - np.log(np.diag(L)).sum() + lj[i]
    assert np.max(np.abs(out - ref)) < 1e-6


def test_negligible_gp_is_white_noise():
    """amp -> 0 with the errors held fixed: the white-noise model, up to the constant sum ln yerr."""
    t, y = data()
    white = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, free=("t0", "rp", "a", "b"))
    g = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, free=("t0", "rp", "a", "b"),
                                      gp=pizzetti.GP(amp=1e-12, tau=0.1, err_scale=1.0))
    assert g.ndim == 4
    X = white.to_internal(white.fixed_physical_free())[0] + np.random.default_rng(5).normal(size=(8, 4)) * np.array([1e-3, 2e-3, 2e-3, 0.03])
    X[:, 3] = np.abs(X[:, 3])
    d = g.log_prob_internal(X) - white.log_prob_internal(X)
    assert np.max(np.abs(d + len(t) * math.log(5e-4))) < 1e-5


def multi_epoch(jitter=0.0, seed=7):
    """Three transits in windows with a gap in one of them, time stamps with a little jitter, a GP and white noise."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([e * P + (np.arange(100) - 49.5) * 5.0 / 1440.0 for e in range(3)])
    t = np.delete(t, np.arange(120, 135))
    t = t + jitter * 5.0 / 1440.0 * rng.standard_normal(t.size)
    pr = params()
    gp = np.linalg.cholesky(sho(t[:, None] - t[None, :], 1e-3, 0.4) + 1e-13 * np.eye(t.size)) @ rng.standard_normal(t.size)
    return t, pizzetti.TransitModel(pr, t).light_curve(pr) + gp + 5e-4 * rng.standard_normal(t.size)


@pytest.mark.parametrize("tol", [0.0, 1e-6])
def test_time_table(tol):
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.choice([0.001, 0.002], 500) * (1.0 + 1e-7 * rng.standard_normal(500)))
    idx, steps = _gp.time_table(t, tol)
    assert len(steps) == (499 if tol == 0.0 else 2)
    assert np.max(np.abs(steps[idx[1:]] / np.diff(t) - 1.0)) <= 2 * tol + 1e-12


def test_table_tolerance_is_small():
    """Jittered time stamps: the shared transitions change ln L by far less than the sampler's noise."""
    t, y = multi_epoch(jitter=1e-7)
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    assert len(m.gp_steps) < 20
    X = m.to_internal(x_true(m))
    a = m.log_prob_internal(X)[0]
    m.gp_idx, m.gp_steps = _gp.time_table(t, 0.0)
    assert abs(a - m.log_prob_internal(X)[0]) < 1e-5


def test_threads_same_result():
    t, y = data()
    a = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    b = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"), threads=2)
    X = a.to_internal(x_true(a))[0] + np.random.default_rng(1).normal(size=(40, 7)) * np.array([1e-3, 2e-3, 2e-3, 0.03, 0.1, 0.1, 0.02])
    X[:, 3] = np.abs(X[:, 3])
    assert np.array_equal(a.log_prob_internal(X), b.log_prob_internal(X))
    assert np.array_equal(a.expand(a.to_physical(X[:30]), rng=np.random.default_rng(2), resample=False)[0],
                          b.expand(b.to_physical(X[:30]), rng=np.random.default_rng(2), resample=False)[0])


@pytest.mark.parametrize("k", [1, 3])
def test_fixed_gp_blocks_against_full_filter(k):
    """Fixed GP parameters: the block kernels against the full filter (and so dense algebra), inside and beyond the blocks."""
    t, y = multi_epoch()
    kw = dict(free=("t0", "rp", "a", "b"), supersample_factor=k, exp_time=5.0 / 1440.0 if k > 1 else 0.0,
              gp=pizzetti.GP(amp=1.2e-3, tau=0.4, err_scale=1.1))
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, **kw)
    assert m.gp_fx is not None and m.ndim == 4 and m.gp_fx[1].size < t.size
    f = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, **kw)
    f.gp_fx = None
    X = m.to_internal(m.fixed_physical_free())[0] + np.random.default_rng(1).normal(size=(30, 4)) * np.array([2e-3, 3e-3, 2e-3, 0.1])
    X[:, 3] = np.abs(X[:, 3])
    X[-1, 2] *= 3.0                                    # long transit: beyond the blocks
    lP, lPf = m.log_prob_internal(X, parts=True), f.log_prob_internal(X, parts=True)
    assert np.max(np.abs(lP[0] - lPf[0])) < 1e-8 and np.max(np.abs(lP[1] - lPf[1])) < 1e-8
    Sm = m.expand(m.to_physical(X[:10]), rng=np.random.default_rng(3), resample=False)[0]
    Sf = f.expand(f.to_physical(X[:10]), rng=np.random.default_rng(3), resample=False)[0]
    assert np.max(np.abs(Sm - Sf)) < 1e-6


def test_fixed_gp_full_filter_when_needed():
    t, y = multi_epoch()
    gp = pizzetti.GP(amp=1e-3, tau=0.4, err_scale=1.0)
    assert pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=gp, free=("t0", "per", "rp", "a", "b")).gp_fx is None
    assert pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=pizzetti.GP(amp=1e-3)).gp_fx is None


def test_fit_fixed_gp():
    t, y = multi_epoch()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=pizzetti.GP(amp=1e-3, tau=0.4, err_scale=1.0), free=("t0", "rp", "a", "b"))
    r = m.fit(seed=1)
    assert r.diagnostics["is_ok"] and r.labels == ["t0", "rp", "a", "b", "q1", "q2", "f0"]
    assert abs(np.median(r.samples[:, 1]) - RP) < 3 * np.std(r.samples[:, 1])


# -- the interface --------------------------------------------------------------------------------
def test_labels_and_fixed_parameters():
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=pizzetti.GP(), free=("t0", "rp", "a", "b"))
    assert m.labels == ["t0", "rp", "a", "b", "ln_gp_amp", "ln_gp_tau", "ln_err_scale"] and m.ndim == 7
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=pizzetti.GP(err_scale=1.0, tau=0.2), free=("t0", "rp", "a", "b"))
    assert m.labels == ["t0", "rp", "a", "b", "ln_gp_amp"] and m.ndim == 5
    assert np.isfinite(m.log_prob(m.fixed_physical_free()))


@pytest.mark.parametrize("kw", [dict(amp=(0.1, 0.01)), dict(tau=-1.0), dict(err_scale=(1.0, 1.0))])
def test_bad_bounds(kw):
    with pytest.raises(ValueError):
        pizzetti.GP(**kw)


def test_scope():
    t, y = data()
    with pytest.raises(NotImplementedError):
        pizzetti.GP(kernel="matern32")
    p = params()
    p.limb_dark, p.u = "linear", [0.4]
    with pytest.raises(NotImplementedError):
        pizzetti.MarginalTransitModel(p, t, y, 5e-4, gp=pizzetti.GP())
    with pytest.raises(NotImplementedError):
        pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=pizzetti.GP(), ld_prior={"mean": (0.4, 0.2), "cov": np.eye(2) * 0.01})
    with pytest.raises(ValueError, match="folded"):
        pizzetti.MarginalTransitModel(params(), t[::-1], y, 5e-4, gp=pizzetti.GP())


def test_prior_box_and_log_prior():
    t, y = data()
    seen = {}

    def lp(d):
        seen.update(d)
        return -0.5 * (d["ln_gp_tau"] - math.log(0.15)) ** 2
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(tau=(0.05, 1.0)), free=("t0", "rp", "a", "b"), log_prior=lp)
    X = m.to_internal(x_true(m))
    assert np.isfinite(m.log_prob_internal(X)[0]) and "ln_err_scale" in seen
    X[0, 5] = math.log(5.0)
    assert m.log_prob_internal(X)[0] == -np.inf


def test_gp_predict():
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    row = np.concatenate([x_true(m)[0], [0.4225, 0.31, 1.0]])
    tr, v = m.gp_predict(row, return_var=True)
    p = params()
    sq = math.sqrt(row[-3])
    p.u = [2 * sq * row[-2], sq * (1 - 2 * row[-2])]
    r = y - pizzetti.TransitModel(p, t).light_curve(p)
    Kt = sho(t[:, None] - t[None, :], 1e-3, 0.15)
    ref = Kt @ np.linalg.solve(Kt + (5e-4) ** 2 * np.eye(len(t)), r)
    assert np.max(np.abs(tr - ref)) < 1e-6 and np.all(v > 0)
    with pytest.raises(ValueError):
        m.gp_predict(row[:-1])
    # at other times: unsorted, between the data, outside them, and on a data time
    tn = np.array([0.31, t[40], -0.5 + 1e-3, t[0] - 0.2, (t[10] + t[11]) / 2, t[-1] + 0.1])
    mn, vn = m.gp_predict(row, t=tn, return_var=True)
    Ks = sho(tn[:, None] - t[None, :], 1e-3, 0.15)
    Si = np.linalg.inv(Kt + (5e-4) ** 2 * np.eye(len(t)))
    assert np.max(np.abs(mn - Ks @ Si @ r)) < 1e-6
    assert np.max(np.abs(vn - (1e-6 - np.sum((Ks @ Si) * Ks, axis=1)))) < 1e-10
    assert np.allclose(m.gp_predict(row, t=t), tr, atol=1e-9)


# -- sampling -------------------------------------------------------------------------------------
def test_fit_recovers_transit():
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    r = m.fit(seed=1)
    assert r.diagnostics["is_ok"]
    assert r.labels[-3:] == ["q1", "q2", "f0"] and r.samples.shape[1] == 10
    med, sd = np.median(r.samples, axis=0), np.std(r.samples, axis=0)
    assert abs(med[1] - RP) < 3 * sd[1] and abs(med[6]) < 0.3
    assert "ln_gp_amp" in r.summary()
    assert np.all(np.isfinite(m.gp_predict(med)))


def test_fit_with_weak_gp():
    """A GP far below the noise: only an upper limit on the amplitude and a flat time scale. The profile in the
    original coordinates has no curvature there."""
    t, y = data(amp=2e-5)
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    r = m.fit(seed=1)
    assert r.diagnostics["is_ok"]
    med, sd = np.median(r.samples, axis=0), np.std(r.samples, axis=0)
    assert abs(med[1] - RP) < 3 * sd[1]
    assert sd[5] > 1.0                              # the time scale stays unconstrained


def test_rescue_seed_without_profile_draws():
    """_gp_seed gives the rescue weighted draws when no profile produced any."""
    t, y = data(amp=2e-5)
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    nev = []
    s = m._gp_seed(m._start_internal(), 500, np.random.default_rng(1), nev)
    assert s["theta"].shape == (1000, 7) and np.sum(np.isfinite(s["logw"])) > 100 and nev[0] > 1000


@pytest.mark.parametrize("f", [1.001, 1.005, 1.02])
def test_rescue_seed_next_to_a_bound(f):
    """a/R* just below its upper bound: the Hessian's steps cross it (-inf), which left T14 without curvature and an
    unbounded proposal width; every draw was outside the support and _gp_seed raised (0.4.0)."""
    t, y = data(amp=2e-5)
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"),
                                      bounds={"a": (2.0, f * A)})
    s = m._gp_seed(m._start_internal(), 500, np.random.default_rng(1), [])
    assert np.sum(np.isfinite(s["logw"])) > 200


def test_rescue_seed_without_finite_draws():
    """No draw inside the support (a start far outside the t0 box): zero weights, which the rescue skips."""
    t, y = data(amp=2e-5)
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"),
                                      bounds={"t0": (-0.05, 0.05)})
    x = m._start_internal()
    x[0] = 2.0
    s = m._gp_seed(x, 500, np.random.default_rng(1), [])
    assert not np.isfinite(s["logw"]).any() and np.all(s["w"] == 0)


def test_emcee_sampler():
    pytest.importorskip("emcee")
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    p0 = x_true(m) + 1e-4 * np.random.default_rng(1).standard_normal((20, 7))
    p0[:, 3] = np.abs(p0[:, 3])
    s = m.EnsembleSampler(20)
    s.run_mcmc(p0, 20)
    assert s.get_chain(flat=True, expand=True, seed=1).shape[1] == 10
