"""Exposure integration: the batman (default) and midpoint placements of sub-exposures."""
import warnings

import numpy as np
import pytest

import pizzetti

EXP = 1765.5 / 86400.0                 # Kepler long cadence (days)


def params():
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.5, 0.1, 9.0, 88.5
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
    return p


def binned(t, k, offsets="batman"):
    p = params()
    return pizzetti.TransitModel(p, t, supersample_factor=k, exp_time=EXP, offsets=offsets).light_curve(p)


def test_default_is_batman():
    t = np.linspace(-0.15, 0.15, 301)
    p = params()
    m = pizzetti.TransitModel(p, t, supersample_factor=5, exp_time=EXP)
    assert m.offsets == "batman"
    np.testing.assert_array_equal(m.light_curve(p), binned(t, 5, "batman"))
    off = m.t_supersample[:5] - t[0]
    np.testing.assert_allclose(off, np.linspace(-EXP / 2, EXP / 2, 5), atol=1e-15)


def test_midpoint_offsets():
    t = np.zeros(1)
    p = params()
    m = pizzetti.TransitModel(p, t, supersample_factor=4, exp_time=EXP, offsets="midpoint")
    np.testing.assert_allclose(m.t_supersample, EXP * np.array([-3, -1, 1, 3]) / 8, atol=1e-15)


def test_midpoint_converges_faster():
    """Against the exposure integral (midpoint rule with many points): the midpoint error
    falls as 1/k^2, the batman error only as 1/k (it integrates over an exposure
    stretched by k/(k-1))."""
    t = np.linspace(-0.15, 0.15, 601)
    ref = binned(t, 2001, "midpoint")
    eb = {k: np.max(np.abs(binned(t, k, "batman") - ref)) for k in (15, 30)}
    em = {k: np.max(np.abs(binned(t, k, "midpoint") - ref)) for k in (15, 30)}
    assert em[15] < eb[15] / 10
    assert 3.0 < em[15] / em[30] < 5.0          # second order
    assert 1.6 < eb[15] / eb[30] < 2.4          # first order


def test_marginal_model_uses_the_same_offsets():
    t = np.linspace(-0.15, 0.15, 301)
    p = params()
    y = binned(t, 7, "midpoint")
    lp = {}
    for off in ("batman", "midpoint"):
        m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, supersample_factor=7, exp_time=EXP, offsets=off)
        lp[off] = m.log_prob(m.fixed_physical_free())
    assert lp["midpoint"] > lp["batman"]        # the data were made with midpoints


def test_invalid_offsets():
    p = params()
    with pytest.raises(ValueError):
        pizzetti.TransitModel(p, np.zeros(3), offsets="gauss")


def kepler_like(sigma=1e-4, ntr=8):
    """Time-ordered 30-min exposures around ntr transits of a hot Jupiter, midpoint-averaged truth plus noise."""
    p = params()
    t = np.concatenate([np.arange(-0.25, 0.25, EXP) + i * p.per for i in range(ntr)])
    f = pizzetti.TransitModel(p, t, supersample_factor=101, exp_time=EXP, offsets="midpoint").light_curve(p)
    return p, t, f + sigma * np.random.default_rng(3).standard_normal(t.size)


def test_exposure_chi2_falls_steeply():
    """Midpoint error ~ 1/k^2 per point or faster, so chi^2 falls by >= 8 per doubling of k; zero without an exposure."""
    p, t, _ = kepler_like()
    c = [pizzetti.exposure_chi2(p, t, EXP, 1e-4, k) for k in (5, 10, 20)]
    assert c[0] > c[1] > c[2] > 0
    assert c[0] / c[1] > 8 and c[1] / c[2] > 8
    assert pizzetti.exposure_chi2(p, t, 0.0, 1e-4, 1) == 0.0


def test_choose_supersample_smallest_k_below_tol():
    p, t, _ = kepler_like()
    k, tab = pizzetti.choose_supersample(p, t, EXP, 1e-4, return_table=True)
    assert tab[k] <= 0.01 and all(v > 0.01 for kk, v in tab.items() if kk < k)
    # noisier data need fewer sub-exposures; no exposure needs none
    assert pizzetti.choose_supersample(p, t, EXP, 1e-3) < k
    assert pizzetti.choose_supersample(p, t, 0.0, 1e-4) == 1
    assert pizzetti.choose_supersample(p, t, EXP, 1e-6, k_max=8) == 8          # cannot reach tol: k_max


def test_auto_supersample_and_the_check_after_fit():
    p, t, y = kepler_like()
    k = pizzetti.choose_supersample(p, t, EXP, 1e-4)
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, free=["t0", "rp", "a", "b"], supersample_factor="auto", exp_time=EXP)
    assert m.k == k and m.supersample_table[k] <= 0.01
    with pytest.raises(ValueError):
        pizzetti.MarginalTransitModel(p, t, y, 1e-4, supersample_factor="auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = m.fit(seed=1).diagnostics
    assert d["supersample_chi2"] < 0.1 and not any("exposure average" in str(c.message) for c in caught)
    m1 = pizzetti.MarginalTransitModel(p, t, y, 1e-4, free=["t0", "rp", "a", "b"], supersample_factor=1, exp_time=EXP)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d1 = m1.fit(seed=1).diagnostics
    assert d1["supersample_chi2"] > 100 and any("exposure average" in str(c.message) for c in caught)


def test_fit_without_exp_time_checks_the_cadence():
    """No supersampling and no exp_time: fit() assumes exposures as long as the cadence and warns if they need an
    average (30-min data of a hot Jupiter at 100 ppm), not if they do not (the same at 2 min and 3000 ppm)."""
    p, t, y = kepler_like()
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, free=["t0", "rp", "a", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = m.fit(seed=1).diagnostics
    assert abs(d["supersample_exp_assumed"] - EXP) < 1e-6 and d["supersample_chi2"] > 100
    assert any("no exp_time given" in str(c.message) for c in caught)
    t2 = np.concatenate([np.arange(-0.25, 0.25, 2.0 / 1440) + i * p.per for i in range(2)])
    y2 = pizzetti.TransitModel(p, t2).light_curve(p) + 3e-3 * np.random.default_rng(4).standard_normal(t2.size)
    m2 = pizzetti.MarginalTransitModel(p, t2, y2, 3e-3, free=["t0", "rp", "a", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d2 = m2.fit(seed=1).diagnostics
    assert d2["supersample_chi2"] < 0.1 and not any("no exp_time given" in str(c.message) for c in caught)
