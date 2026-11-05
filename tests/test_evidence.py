"""ln Z of profile importance sampling (raw weights) against brute-force integration and nested sampling."""
import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp

import pizzetti
from pizzetti import _marginal as K
from pizzetti import _series
from pizzetti.marginal import _A_HALF
from _evidence_refs import ALL, F_HI, F_LO, PATH, build, ln_volume, params


def small_model(free=("rp", "b"), bounds=None, seed=11, rp=0.1, a=9.0, b=0.8, sig=3000e-6, **kw):
    t = np.concatenate([np.arange(-0.2, 0.2, 5.0 / 1440.0) + i * 3.5 for i in range(3)])
    p = params(rp, a, b, 3.5)
    y = pizzetti.TransitModel(p, t).light_curve(p) + sig * np.random.default_rng(seed).standard_normal(t.size)
    return pizzetti.MarginalTransitModel(p, t, y, sig, free=free, bounds=bounds or {"rp": (0.02, 0.3)}, **kw)


def direct_log_like(m, rp, b, x, w):
    """ln of the integral of L over (beta0, beta1, f0) with density 2 on the triangle and 1 in f0, written out:
    f0 in closed form (a Gaussian in f0 far from f0 = 0), the triangle by Gauss-Legendre (x, w on 0..1)."""
    A = K.design(m.tt, m.k, m.wts, 0.0, 3.5, rp, 9.0, b, 0.0, 90.0, m.tol, _A_HALF, _series._GTAB)
    B1, f = np.meshgrid(x, x, indexing="ij")
    B0 = f * B1
    g = (A[:, 0, None, None] * B0 + A[:, 1, None, None] * B1 + A[:, 2, None, None]) / (B0 / 6 + B1 / 3 + 0.5)
    aa, bb = np.sum(g * g, axis=0), np.sum(g * m.yw[:, None, None], axis=0)
    val = -0.5 * (m.yw @ m.yw) + 0.5 * bb * bb / aa + 0.5 * np.log(2 * np.pi / aa)
    return math.log(2.0) + logsumexp(val + np.log(np.outer(w, w) * B1))


@pytest.mark.parametrize("seed, kw", [(11, dict()), (12, dict(b=1.0, sig=2000e-6))])
def test_lnz_brute_force(seed, kw):
    """Nothing of the marginal kernel: geometry, limb darkening and baseline integrated by quadrature."""
    m = small_model(seed=seed, **kw)
    res = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert res["is_ok"]
    x, w = np.polynomial.legendre.leggauss(10)
    xt, wt = 0.5 * (x + 1), 0.5 * w
    x, w = np.polynomial.legendre.leggauss(80)
    x, w = 0.5 * (x + 1), 0.5 * w
    r_lo, r_hi = 0.02, 0.3
    lv = []
    for rj, wj in zip(r_lo + (r_hi - r_lo) * x, (r_hi - r_lo) * w):
        for bk, wk in zip((1 + rj) * x, (1 + rj) * w):
            lv.append(math.log(wj * wk) + direct_log_like(m, rj, bk, xt, wt))
    n = len(m.y)
    area = (r_hi - r_lo) + 0.5 * (r_hi ** 2 - r_lo ** 2)
    ref = logsumexp(lv) - 0.5 * n * math.log(2 * math.pi) + np.sum(np.log(m.wts)) - math.log(area)
    assert abs(res["lnz"] - ref) < 0.02 + 4 * res["lnz_err"]


with open(PATH) as _f:
    REFS = json.load(_f)["runs"]


@pytest.mark.parametrize("case", ALL)
def test_lnz_against_nested_sampling(case):
    """The stored dynesty runs (tests/_evidence_refs.py); "ns7" where the case has it: no marginalisation at all."""
    m, _ = build(case)
    res = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert res["is_ok"]
    lnz = res["lnz"] - math.log(F_HI - F_LO)
    runs = [r for r in REFS if r["case"] == case]
    assert len(runs) >= 3
    for mode in ("ns7", "ns_marg"):
        rr = [r for r in runs if r["mode"] == mode]
        if rr:
            assert abs(lnz - np.mean([r["lnz"] for r in rr])) < max(0.2, 3 * np.mean([r["se"] for r in rr]))
    X, w = m.to_physical(res["theta"]), res["w"] / res["w"].sum()
    jb, jr = m.free.index("b"), m.free.index("rp")
    assert abs(np.sum(w[X[:, jb] + X[:, jr] > 1]) - np.mean([r["grazing"] for r in runs])) < 0.06
    assert res["lnz_err"] < 0.05


def test_log_prior_volume():
    m, box = build("t15")
    assert m._log_prior_volume() == pytest.approx(ln_volume(box), abs=1e-4)
    m, box = build("b035")
    assert m._log_prior_volume() == pytest.approx(ln_volume(box), abs=1e-4)
    # a window of b, an eccentric orbit, a fixed rp: Monte Carlo over the box
    rng = np.random.default_rng(0)
    t = np.linspace(-0.1, 0.1, 50)
    y = np.ones_like(t)
    p = params(0.1, 9.0, 0.3, 3.5)
    p.ecc, p.w = 0.3, 40.0
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-3, free=("t0", "rp", "a", "b"), reparam="none",
                                      bounds={"t0": (-1, 1), "rp": (0.0, 0.6), "a": (1.05, 4.0), "b": (0.2, 1.4)})
    g = (1 - 0.3 ** 2) / (1 + 0.3 * math.sin(math.radians(40.0)))
    r, a, b = rng.uniform(0, 0.6, 2000000), rng.uniform(1.05, 4.0, 2000000), rng.uniform(0.2, 1.4, 2000000)
    frac = np.mean((b * g < 1 + r) & (a > b))
    assert m._log_prior_volume() == pytest.approx(math.log(2 * 0.6 * 2.95 * 1.2 * frac), abs=2e-3)


def test_seeded_draws_unchanged_by_evidence():
    m = small_model()
    a = m.profile_is(n=2000, rng=np.random.default_rng(5))
    b = m.profile_is(n=2000, rng=np.random.default_rng(5), evidence=False)
    assert np.array_equal(a["theta"], b["theta"]) and np.array_equal(a["w"], b["w"])
    assert a["log_int"] == b["log_int"] and "lnz" not in b


def test_fit_diagnostics():
    m = small_model()
    d = m.fit(seed=2, n=4000).diagnostics
    res = m.profile_is(n=4000, rng=np.random.default_rng(3))
    assert d["lnz_note"] is None
    assert abs(d["lnz"] - res["lnz"]) < 0.1


@pytest.mark.parametrize("kw, word", [
    (dict(free=("t0", "rp", "a", "b"), bounds={"rp": (0.02, 0.3), "t0": (-0.1, 0.1)}), "finite"),
    (dict(log_prior=lambda d: np.zeros(len(d["rp"]))), "log_prior"),
    (dict(ld_prior=lambda u1, u2: np.zeros(len(u1))), "quadratic"),
])
def test_lnz_unavailable(kw, word):
    m = small_model(**kw)
    res = m.profile_is(n=2000, rng=np.random.default_rng(1))
    assert math.isnan(res["lnz"]) and word in res["lnz_note"]
    assert np.isfinite(res["log_int"]) and np.isfinite(res["log_int_err"])


def test_log_int_of_a_known_integrand():
    """``log_int`` of the sampling routines: the normalisation of a Gaussian in the profile's coordinates."""
    from pizzetti._sampling import profile_is, laplace_is
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.25]])
    Pi = np.linalg.inv(cov)
    mu = np.array([0.3, -0.2, 2.0])
    ref = 1.7 + 1.5 * math.log(2 * math.pi) + 0.5 * math.log(np.linalg.det(cov))

    def lp(X):
        d = np.atleast_2d(X) - mu
        return 1.7 - 0.5 * np.einsum("ni,ij,nj->n", d, Pi, d)
    sc = np.sqrt(np.diag(cov))
    for f, kw in ((profile_is, dict(prof=2, lo_wall=None)), (laplace_is, dict())):
        r = f(lp, mu + 0.1 * sc, sc, n=4000, rng=np.random.default_rng(0), **kw)
        assert abs(r["log_int"] - ref) < 5 * r["log_int_err"] + 1e-3
