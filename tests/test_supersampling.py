"""Exposure integration: the batman (default) and midpoint placements of sub-exposures."""
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
