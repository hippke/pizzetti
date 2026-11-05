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
    ld = _gp.whiten(t, var, amp, tau, Y)
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


def test_emcee_sampler():
    pytest.importorskip("emcee")
    t, y = data()
    m = pizzetti.MarginalTransitModel(params(), t, y, 5e-4, gp=box(), free=("t0", "rp", "a", "b"))
    p0 = x_true(m) + 1e-4 * np.random.default_rng(1).standard_normal((20, 7))
    p0[:, 3] = np.abs(p0[:, 3])
    s = m.EnsembleSampler(20)
    s.run_mcmc(p0, 20)
    assert s.get_chain(flat=True, expand=True, seed=1).shape[1] == 10
