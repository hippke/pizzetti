"""``FitResult.reweight_density``: e and omega from a stellar density after a circular fit."""
import json
import math
import warnings

import numpy as np
import pytest

import pizzetti
from pizzetti.marginal import FitResult, _contact_times
from _density_refs import CASES, ECC_MAX, PATH, PER, RHO, RHO_SD, circular_model, wq

G_CGS = 6.6743e-8


def params(t0, per, rp, a, b, e, w):
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w, p.limb_dark, p.u = t0, per, rp, a, e, w, "quadratic", [0.4, 0.25]
    p.inc = math.degrees(math.acos(b / a))
    return p


def flat_fit(n=20000, per=PER, rp=0.05, seed=1, a_lo=1.5, a_hi=400.0, **kw):
    """A fit that has seen no data: draws flat in a/R*, b and t0."""
    rng = np.random.default_rng(seed)
    S = np.column_stack([rng.uniform(-0.01, 0.01, n), np.full(n, rp), rng.uniform(a_lo, a_hi, n), rng.uniform(0, 1 + rp, n)])
    orbit = dict(per=per, rp=rp, a=20.0, b=0.3, ecc=0.0, a_bounds=(1.0, 1000.0), secondary=False, free=("t0", "rp", "a", "b"))
    orbit.update(kw)
    return FitResult(S, ["t0", "rp", "a", "b"], weighted=(S, np.full(n, 1.0 / n)), diagnostics={}, orbit=orbit)


@pytest.fixture(scope="module")
def easy():
    return circular_model("e40w090").fit(seed=1, n=2000)


@pytest.mark.parametrize("per,rp,a,b_conj,e,w", [(10.0, 0.1, 19.5, 0.3, 0.7, 0.0), (10.0, 0.12, 19.5, 0.66, 0.83, 267.0),
                                                 (3.0, 0.1, 8.7, 0.3, 0.5, 30.0), (28.0, 0.145, 39.0, 0.51, 0.8, 114.0),
                                                 (10.0, 0.1, 19.5, 0.95, 0.4, 90.0), (10.0, 0.05, 19.5, 0.3, 0.0, 90.0)])
def test_contact_times_against_the_light_curve(per, rp, a, b_conj, e, w):
    bo = b_conj * (1 + e * math.sin(math.radians(w))) / (1 - e * e)
    t1, t4 = _contact_times(np.array([per]), np.array([rp]), np.array([a]), np.array([bo]), np.array([e]), np.array([math.radians(w)]))
    span = float(t4[0] - t1[0])
    t = np.linspace(t1[0] - 0.05 * span, t4[0] + 0.05 * span, 100001)
    p = params(0.0, per, rp, a, bo, e, w)
    inside = t[pizzetti.TransitModel(p, t).light_curve(p) < 1 - 1e-12]
    assert inside[0] == pytest.approx(t1[0], abs=3e-3 * span / 100)
    assert inside[-1] == pytest.approx(t4[0], abs=3e-3 * span / 100)


def test_prior_recovered_without_data():
    """With no data the fit is flat in the circular a/R*, so the result must be the prior itself: N(rho), flat e and
    omega, whatever they do to the duration. A missing or wrong Jacobian shows up here."""
    r = flat_fit().reweight_density((RHO, RHO_SD), repeats=200, seed=3, min_ess=20000)
    S, L, w = r.weighted_samples, r.labels, r.weights
    rho, e, om = S[:, L.index("rho")], S[:, L.index("ecc")], S[:, L.index("w")]
    assert r.diagnostics["density"]["ess"] > 15000
    assert np.sum(w * rho) == pytest.approx(RHO, abs=0.005)
    assert math.sqrt(np.sum(w * (rho - np.sum(w * rho)) ** 2)) == pytest.approx(RHO_SD, abs=0.005)
    # flat: no trend of the density with e or omega, in five bins each
    for x, lo, hi in ((e, 0.0, ECC_MAX), (om, -90.0, 270.0)):
        share, _ = np.histogram(x, bins=5, range=(lo, hi), weights=w)
        assert np.max(np.abs(share - 0.2)) < 0.012


def test_duration_centre_and_light_curve_kept():
    """The eccentric draw has the duration and the centre of its circular parent; at high S/N the two light curves agree."""
    rp, per, a_c, b_c = 0.1, 10.0, 25.0, 0.3
    n = 2000
    S = np.column_stack([np.zeros(n), np.full(n, rp), np.full(n, a_c), np.full(n, b_c)])
    orbit = dict(per=per, rp=rp, a=a_c, b=b_c, ecc=0.0, a_bounds=(1.0, 1000.0), secondary=False, free=("t0", "rp", "a", "b"))
    fit = FitResult(S, ["t0", "rp", "a", "b"], weighted=(S, np.full(n, 1.0 / n)), diagnostics={}, orbit=orbit)
    r = fit.reweight_density((RHO, 10 * RHO_SD), seed=5, ecc_max=0.85)
    T, L = r.weighted_samples, r.labels
    i = np.random.default_rng(1).choice(len(T), 300, p=r.weights)
    t0, a, b, e, w = (T[i, L.index(q)] for q in ("t0", "a", "b", "ecc", "w"))
    t1, t4 = _contact_times(per, rp, a, b, e, np.radians(w))
    s_c = math.asin(math.sqrt(((1 + rp) ** 2 - b_c ** 2) / (a_c ** 2 - b_c ** 2)))
    assert np.max(np.abs(t4 - t1 - per * s_c / math.pi)) < 1e-6
    assert np.max(np.abs(t0 + 0.5 * (t1 + t4))) < 1e-12
    assert e.max() > 0.5
    t = np.linspace(-0.4, 0.4, 801)
    pc = params(0.0, per, rp, a_c, b_c, 0.0, 90.0)
    circ = pizzetti.TransitModel(pc, t).light_curve(pc)
    rms = []
    for k in range(len(i)):
        p = params(t0[k], per, rp, a[k], b[k], e[k], w[k])
        rms.append(np.std(pizzetti.TransitModel(p, t).light_curve(p) - circ))
    rms = np.array(rms)
    # against a depth of 0.01; the shape does change on an orbit that passes the star at less than 3 stellar radii
    assert np.median(rms) < 3e-6
    assert np.max(rms[a * (1 - e) > 3.0]) < 3e-5
    assert np.max(rms) < 2e-4


@pytest.mark.parametrize("case", list(CASES))
def test_against_nested_sampling(case):
    """Dynesty on the eccentric forward model, with no assumption about the transit (tests/_density_refs.py)."""
    with open(PATH) as f:
        stored = json.load(f)
    refs = [q for q in stored["runs"] if q["case"] == case]
    grid = {q["case"]: q["ecc"] for q in stored["grid"]}
    assert len(refs) == 2
    r = circular_model(case).fit(seed=1, n=4000).reweight_density((RHO, RHO_SD), ecc_max=ECC_MAX, seed=2)
    S, L, w = r.weighted_samples, r.labels, r.weights
    d = r.diagnostics["density"]
    assert d["ess"] >= 2000 and d["ess_parents"] > 200
    for q in ("rp", "a", "b_conj", "ecc", "w"):
        ours = np.array(wq(S[:, L.index(q)], w))
        theirs = np.mean([x[q] for x in refs], axis=0)
        sd = np.mean([x["sd"][q] for x in refs])
        scatter = np.abs(np.array(refs[0][q]) - np.array(refs[1][q])) / sd
        if q == "ecc" and case == "e70w180":                 # dynesty overstates the tail here: see tests/_density_refs.py
            theirs, scatter = np.array(grid[case]), 0.0 * scatter
        # the two dynesty runs differ among themselves: 0.05 sd at the median, more in the tails
        assert np.max(np.abs(ours - theirs)[1:4] / sd) < 0.2 + np.max(scatter[1:4]), q
def test_result_columns_and_diagnostics(easy):
    r = easy.reweight_density((RHO, RHO_SD), seed=2)
    L = r.labels
    assert L[:len(easy.labels)] == easy.labels and L[len(easy.labels):] == ["ecc", "w", "b_conj", "rho"]
    assert r.samples.shape[1] == len(L) and r.weighted_samples.shape == (len(r.weights), len(L))
    assert r.weights.sum() == pytest.approx(1.0)
    S = r.weighted_samples
    a, e, w, b, bc, rho = (S[:, L.index(q)] for q in ("a", "ecc", "w", "b", "b_conj", "rho"))
    assert np.allclose(rho, 3 * np.pi * a ** 3 / (G_CGS * (PER * 86400.0) ** 2))
    assert np.allclose(bc, b * (1 - e * e) / (1 + e * np.sin(np.radians(w))))
    assert np.all((e >= 0) & (e < 0.92) & (w >= -90) & (w < 270) & (a * (1 - e) > 1))
    assert set(("ess", "ess_parents", "a_wall_mass", "passes")) <= set(r.diagnostics["density"])
    assert math.isnan(r.diagnostics["lnz"]) and "evidence" in r.diagnostics["lnz_note"]
    assert r.diagnostics["n_eval"] == easy.diagnostics["n_eval"]
    assert "rp" in r.summary()
    a2 = easy.reweight_density((RHO, RHO_SD), seed=2).weighted_samples
    assert np.array_equal(S, a2)                             # seeded


def test_log_prior_and_ecc_max(easy):
    base = easy.reweight_density((RHO, RHO_SD), seed=2)
    low = easy.reweight_density((RHO, RHO_SD), seed=2, ecc_max=0.3)
    lim = easy.reweight_density((RHO, RHO_SD), seed=2, log_prior=lambda e, w: -0.5 * (e / 0.1) ** 2)
    prob = easy.reweight_density((RHO, RHO_SD), seed=2, log_prior=lambda e, w: np.log1p(e * np.sin(np.radians(w))) - np.log(1 - e ** 2))
    q = lambda r: np.sum(r.weights * r.weighted_samples[:, r.labels.index("ecc")])
    assert np.max(low.weighted_samples[:, low.labels.index("ecc")]) < 0.3
    assert q(lim) < 0.15 < q(base)
    assert q(prob) != q(base)


def test_narrow_density_adds_passes(easy):
    r = easy.reweight_density((RHO, 0.01 * RHO), seed=2, min_ess=1000)
    assert r.diagnostics["density"]["passes"] > 1
    assert r.diagnostics["density"]["ess"] >= 1000
    rho = r.weighted_samples[:, r.labels.index("rho")]
    assert abs(np.sum(r.weights * rho) - RHO) < 0.003 * RHO


def test_free_period_and_fixed_b():
    m = circular_model("e40w090", n_free=("t0", "per", "rp", "a", "b"))
    r = m.fit(seed=1, n=2000).reweight_density((RHO, RHO_SD), seed=2)
    L = r.labels
    a, per, rho = (r.weighted_samples[:, L.index(q)] for q in ("a", "per", "rho"))
    assert np.ptp(per) > 0
    assert np.allclose(rho, 3 * np.pi * a ** 3 / (G_CGS * (per * 86400.0) ** 2))
    f = circular_model("e40w090", n_free=("t0", "rp", "a")).fit(seed=1, method="emcee", n=500, max_steps=4000)
    r = f.reweight_density((RHO, RHO_SD), seed=2)
    assert "b" in r.labels and "b_conj" in r.labels
    bc = r.weighted_samples[:, r.labels.index("b_conj")]
    assert np.ptp(bc) < 1e-9


def test_no_likelihood_evaluation(easy):
    r = easy.reweight_density((RHO, RHO_SD), seed=2)
    assert r.diagnostics["n_eval"] == easy.diagnostics["n_eval"]


def test_refused():
    good = flat_fit(n=100)
    for rho in ((RHO, 0.0), (-1.0, 0.1)):
        with pytest.raises(ValueError):
            good.reweight_density(rho)
    with pytest.raises(ValueError):
        good.reweight_density((RHO, RHO_SD), ecc_max=1.0)
    with pytest.raises(ValueError, match="ecc = 0"):
        flat_fit(n=100, ecc=0.3).reweight_density((RHO, RHO_SD))
    with pytest.raises(ValueError, match="ecc = 0"):
        flat_fit(n=100, free=("t0", "rp", "a", "b", "ecc")).reweight_density((RHO, RHO_SD))
    with pytest.raises(ValueError, match="eclipse"):
        flat_fit(n=100, secondary=True).reweight_density((RHO, RHO_SD))
    with pytest.raises(ValueError, match="free"):
        flat_fit(n=100, free=("t0", "rp", "b")).reweight_density((RHO, RHO_SD))
    with pytest.raises(ValueError, match="once"):            # the result is no longer a circular fit
        flat_fit(n=2000).reweight_density((RHO, RHO_SD), seed=1).reweight_density((RHO, RHO_SD))
    with pytest.raises(ValueError, match="no draw"):         # e < 0.01 cannot reach it from a short duration
        flat_fit(n=200, a_lo=100.0, a_hi=101.0).reweight_density((RHO, RHO_SD), ecc_max=0.01, seed=1)


def test_warns_at_the_bound_and_with_few_draws():
    with pytest.warns(RuntimeWarning, match="bounds on a"):
        flat_fit(a_bounds=(1.0, 25.0), a_lo=24.8, a_hi=25.0, n=3000).reweight_density((RHO, 3 * RHO_SD), seed=1, min_ess=100)
    one = flat_fit(n=5, a_lo=19.0, a_hi=20.0)
    with pytest.warns(RuntimeWarning, match="few effective"):
        one.reweight_density((RHO, RHO_SD), seed=1)
