"""Stratified profile importance sampling: the posterior split at the grazing boundary."""
import json
import math

import numpy as np
import pytest

import pizzetti
from pizzetti import _sampling
from pizzetti.marginal import _StratumCoords
from _evidence_refs import F_HI, F_LO, PATH, build, params

with open(PATH) as _f:
    REFS = json.load(_f)["runs"]


def grazing_masses(m, r):
    X, w = m.to_physical(r["theta"]), r["w"] / r["w"].sum()
    g = m._conj_factor(r["theta"])
    return float(np.sum(w[X[:, m.free.index("b")] * g + X[:, m.free.index("rp")] > 1]))


@pytest.mark.parametrize("kind", ["beta", "graze"])
@pytest.mark.parametrize("ecc, w", [(0.0, 90.0), (0.4, 250.0)])
def test_coordinates_round_trip_and_jacobian(kind, ecc, w):
    """to_u inverts to_x, and the log Jacobian is that of (rp, b) by finite differences."""
    t = np.linspace(-0.2, 0.2, 40)
    p = params(0.1, 9.0, 0.5, 3.5)
    p.ecc, p.w = ecc, w
    m = pizzetti.MarginalTransitModel(p, t, np.ones_like(t), 1e-3, free=("t0", "rp", "a", "b"), reparam="none",
                                      bounds={"rp": (1e-3, 0.5), "a": (2.0, 30.0), "t0": (-0.1, 0.1)})
    T = _StratumCoords(m, kind)
    g = float(m._conj_factor(np.array([[0.0, 0.1, 9.0, 0.5]]))[0])
    bc = np.array([0.3, 0.7]) if kind == "beta" else np.array([0.95, 1.2])
    X = np.column_stack([np.zeros(2), [0.1, 0.25], [9.0, 9.0], bc / g])
    U = T.to_u(X)
    X2, lj = T.to_x(U)
    assert np.allclose(X2, X, atol=1e-10)
    for i in range(2):
        J = np.empty((2, 2))
        for k, col in enumerate((1, 3)):
            d = np.zeros(4)
            d[col] = 1e-6
            J[:, k] = (T.to_x(U[i] + d)[0][0, [1, 3]] - T.to_x(U[i] - d)[0][0, [1, 3]]) / 2e-6
        assert lj[i] == pytest.approx(math.log(abs(np.linalg.det(J))), abs=1e-5)


@pytest.mark.parametrize("case", ["b100", "b100lo"])
def test_both_sides_found_and_match_the_references(case):
    m, _ = build(case)
    r = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert r["is_ok"] and r["stratified"] and set(r["strata"]) == {"nongrazing", "grazing"}
    runs = [q for q in REFS if q["case"] == case and q["mode"] == "ns_marg"]
    assert abs(r["strata"]["grazing"]["mass"] - np.mean([q["grazing"] for q in runs])) < 0.03
    assert sum(s["mass"] for s in r["strata"].values()) == pytest.approx(1.0)
    assert abs(grazing_masses(m, r) - r["strata"]["grazing"]["mass"]) < 1e-9
    lnz = r["lnz"] - math.log(F_HI - F_LO)
    assert abs(lnz - np.mean([q["lnz"] for q in runs])) < 0.35
    assert r["n_eval"] < 50_000          # the retry and the rescue took 55 000 to 90 000


def test_shares_of_the_posterior_have_one_error():
    """Two shares that sum to one have the same error, from both integrals."""
    m, _ = build("b100lo")
    r = m.profile_is(n=2000, rng=np.random.default_rng(1), evidence=False)
    assert r["stratified"]
    a, b = r["strata"]["nongrazing"], r["strata"]["grazing"]
    assert a["mass"] + b["mass"] == pytest.approx(1.0)
    assert a["mass_err"] == pytest.approx(b["mass_err"]) and a["mass_err"] > 0


def test_side_without_proposal_is_not_dropped(monkeypatch):
    """No proposal on the grazing side, which holds most of the mass: the old attempts run, nothing is lost."""
    m, _ = build("b100lo")
    orig = _sampling.profile_is

    def no_grazing(lp, x0, scale, prof=4, n=4000, **kw):
        if getattr(getattr(lp, "__self__", None), "kind", None) == "graze":
            return dict(theta=np.atleast_2d(x0), w=np.ones(1), logw=np.zeros(1), log_int=np.nan, log_int_err=np.nan,
                        khat=np.inf, ess=0.0, n_eval=0, theta_map=np.asarray(x0, float), n_grid=0, complete=False)
        return orig(lp, x0, scale, prof=prof, n=n, **kw)
    monkeypatch.setattr(_sampling, "profile_is", no_grazing)
    r = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert not r["stratified"] and "no proposal" in r["strata_note"]
    assert r["is_ok"] and grazing_masses(m, r) > 0.85


def test_empty_side_has_a_tiny_measured_mass():
    m, _ = build("small")
    r = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert r["stratified"] and r["is_ok"]
    assert r["strata"]["grazing"]["mass"] < 1e-5


def test_stratify_false_keeps_the_old_path():
    m, _ = build("b100lo")
    r = m.profile_is(n=4000, rng=np.random.default_rng(1), stratify=False)
    assert r["stratified"] is None and "strata" not in r and r["is_ok"]


def test_usable_first_profile_is_untouched():
    m, _ = build("b035")
    a = m.profile_is(n=2000, rng=np.random.default_rng(1), evidence=False)
    b = m.profile_is(n=2000, rng=np.random.default_rng(1), evidence=False, stratify=False)
    assert a["stratified"] is None and np.array_equal(a["theta"], b["theta"]) and a["n_eval"] == b["n_eval"]


def test_eccentric_orbit_agrees_with_the_old_path():
    t = np.concatenate([np.arange(-0.2, 0.2, 3 / 1440) + i * 3.5 for i in range(5)])
    p = params(0.1, 9.0, 0.5, 3.5)
    p.ecc, p.w = 0.3, 40.0
    g = (1 - 0.09) / (1 + 0.3 * math.sin(math.radians(40.0)))
    p.inc = math.degrees(math.acos(1.0 / g / 9.0))
    y = pizzetti.TransitModel(p, t).light_curve(p) + 800e-6 * np.random.default_rng(3).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 800e-6, free=("t0", "rp", "a", "b"),
                                      bounds={"t0": (-0.05, 0.05), "rp": (1e-3, 0.5), "a": (1.5, 40.0)})
    a = m.profile_is(n=4000, rng=np.random.default_rng(1))
    b = m.profile_is(n=4000, rng=np.random.default_rng(1), stratify=False)
    assert a["stratified"] and a["is_ok"] and b["is_ok"]
    assert abs(a["log_int"] - b["log_int"]) < 0.05
    assert abs(grazing_masses(m, a) - grazing_masses(m, b)) < 0.03


def test_fit_reports_the_grazing_mass_and_budget_stops():
    m, _ = build("b100lo")
    d = m.fit(seed=1, n=4000).diagnostics
    assert d["stratified"] and d["is_ok"] and 0.85 < d["grazing_mass"] < 1.0
    r = m.profile_is(n=4000, rng=np.random.default_rng(1), max_evals=18000)
    assert r["budget_exceeded"] and not r["is_ok"]


def test_no_profile_at_all_gets_a_prior_seed():
    """S/N 2.6: no conditional Hessian anywhere, so no attempt has draws; the rescue starts from the prior box. Reference:
    dynesty on the same integrand, ln I = -617.09 +- 0.05 (1500 live points), grazing mass 0.29."""
    t = np.concatenate([np.arange(-0.13, 0.13, 2 / 1440) + i * 3.5 for i in range(7)])
    p = params(0.01, 9.0, 0.3, 3.5)
    y = pizzetti.TransitModel(p, t).light_curve(p) + 1e-3 * np.random.default_rng(7).standard_normal(t.size)
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-3, free=("t0", "rp", "a", "b"),
                                      bounds={"t0": (-0.05, 0.05), "rp": (1e-4, 0.5), "a": (1.5, 30.0)})
    r = m.profile_is(n=4000, rng=np.random.default_rng(1))
    assert r["is_ok"]
    assert abs(r["log_int"] + 617.09) < 0.25
    assert abs(grazing_masses(m, r) - 0.29) < 0.04
