"""Nonlinear law with the physical prior: the curved-region probability (_physnl)
and its use in MarginalTransitModel."""
import numpy as np
import pytest

import pizzetti
from pizzetti import _laws, _physnl

from test_marginal_laws import setup

LAW = _laws.Nonlinear()


def _moments(sigma, ntr, shift, seed=0):
    t, y, p = setup("nonlinear", [0.5, 0.1, 0.1, -0.1], sigma=sigma, ntr=ntr)
    m = pizzetti.MarginalTransitModel(p, t, y, sigma)
    x = m.to_internal(m.fixed_physical_free())
    x = x + np.array([2e-4, 2e-6, 1e-3, 2e-4, 0.05]) * shift
    P, _ = m._canonical_internal(x)
    _, C, S = m._gen(P, np.zeros(1), m._alpha(x), False, True)
    return m, x, C[0], S[0]


def test_line_interval_is_exact():
    """The interval of a line inside the region agrees with the exact test."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = LAW.from_u(rng.uniform(-0.3, 0.6, 4)) + 0.05 * rng.standard_normal(5)
        e = rng.standard_normal(5)
        lo, hi = _physnl.line_interval(b, e, _physnl.B3_GRID)
        ts = np.linspace(-3, 3, 601)
        inside = LAW.is_physical(b[None, :] + ts[:, None] * e[None, :])
        pred = (ts >= lo) & (ts <= hi)
        mism = np.nonzero(inside != pred)[0]
        # disagreement only at the interval ends (resolution of the t grid)
        for k in mism:
            assert min(abs(ts[k] - lo), abs(ts[k] - hi)) < 0.011, (ts[k], lo, hi)


@pytest.mark.parametrize("sigma,ntr", [(1e-4, 3), (3e-4, 3), (3e-5, 7)])
def test_log_prob_physical_vs_monte_carlo(sigma, ntr):
    m, x, c, Fi = _moments(sigma, ntr, 0.0)
    est = _physnl.log_prob_physical_line(c, Fi, _physnl.Z4_QMC, _physnl.B3_GRID)
    fine = _physnl.log_prob_physical_line(c, Fi, _physnl.qmc_normal4(4096, 7), _physnl.B3_GRID)
    assert abs(est - fine) < 3e-3
    d = np.random.default_rng(2).multivariate_normal(c, Fi, size=400_000)
    frac = LAW.is_physical(d).mean()
    se = np.sqrt((1 - frac) / frac / len(d))
    assert abs(est - np.log(frac)) < 4 * se + 0.005


def test_likelihood_includes_physical_probability():
    """physical = flat-prior marginal + log P(physical); smooth in the geometry."""
    t, y, p = setup("nonlinear", [0.5, 0.1, 0.1, -0.1], sigma=1e-4, ntr=3)
    mp = pizzetti.MarginalTransitModel(p, t, y, 1e-4)
    mf = pizzetti.MarginalTransitModel(p, t, y, 1e-4, ld_prior="flat")
    x = mp.to_internal(mp.fixed_physical_free())
    X = x + np.outer(np.linspace(-1, 1, 41), [0, 0, 0, 0, 0.05])     # line in b
    lp, lf = mp.log_prob_internal(X), mf.log_prob_internal(X)
    P, _ = mp._canonical_internal(X)
    _, C, S = mp._gen(P, np.zeros(len(X)), mp._alpha(X), False, True)
    logP = np.array([_physnl.log_prob_physical_line(C[i], S[i], _physnl.Z4_QMC, _physnl.B3_GRID, _physnl.Z4_BIG)
                     for i in range(len(X))])
    np.testing.assert_allclose(lp - lf, logP, atol=1e-9)
    s = np.linspace(-1, 1, len(X))                 # smooth estimator: no jumps
    resid = (lp - lf) - np.polyval(np.polyfit(s, lp - lf, 8), s)
    assert np.std(resid) < 2e-3


def test_fit_low_snr_uses_importance_sampling():
    """At low S/N the geometry posterior without the constraint is degenerate;
    with the region probability in the likelihood profile IS succeeds."""
    t, y, p = setup("nonlinear", [0.5, 0.1, 0.1, -0.1], sigma=3e-4, ntr=3)
    m = pizzetti.MarginalTransitModel(p, t, y, 3e-4)
    r = m.fit(seed=1)
    d = r.diagnostics
    assert d["method"] == "profile-IS" and d["khat"] < 0.7
    u = r.samples[:, [r.labels.index(k) for k in ("u1", "u2", "u3", "u4")]]
    assert np.all(LAW.is_physical(np.array([LAW.from_u(v) for v in u])))
