"""The workspace likelihood (loglike_batch_ws, the default single-thread path of MarginalTransitModel) gives exactly
the values of the allocating kernel (loglike_batch), and reuses its memory."""
import copy
import pickle
import resource

import numpy as np
import pytest

import pizzetti
from pizzetti import _marginal as K, _series
from pizzetti.terms import _A_HALF


def model(ecc=0.0, series_tol=1e-9, k=7, n=2500):
    rng = np.random.default_rng(4)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.0, 0.09, 8.0, 87.5
    p.ecc, p.w, p.limb_dark, p.u = ecc, 70.0, "quadratic", [0.4, 0.2]
    t = np.sort(rng.uniform(-0.2, 0.2, n))
    y = pizzetti.TransitModel(p, t, supersample_factor=k, exp_time=0.02, offsets="midpoint").light_curve(p)
    y = y + rng.normal(0, 3e-4, n)
    free = ["t0", "rp", "a", "b"] + (["ecc", "w"] if ecc > 0 else [])
    return pizzetti.MarginalTransitModel(p, t, y, 3e-4, free=free, supersample_factor=k, exp_time=0.02,
                                         series_tol=series_tol)


def rows(m, n=200):
    x0 = np.atleast_2d(m._start_internal())
    rng = np.random.default_rng(2)
    return x0 + rng.normal(size=(n, x0.shape[1])) * 0.05 * np.abs(x0 + 1e-2)


def both(m, X):
    a = (m._geo(X), m.idx, m.fixed, m.t14, m.lo, m.hi, m.tt, m.k, m.wts, m.yw, m.tol, _A_HALF, _series._GTAB)
    lp_o, ne_o = np.zeros(len(X)), np.full((len(X), 9), np.nan)
    lp_n, ne_n = np.zeros(len(X)), np.full((len(X), 9), np.nan)
    old = K.loglike_batch(*a, False, np.zeros(2), np.eye(2), lp_o, ne_o)
    new = K.loglike_batch_ws(*a, lp_n, ne_n, K.workspace(m.tt, m.k))
    return (old, lp_o, ne_o), (new, lp_n, ne_n)


@pytest.mark.parametrize("kw", [dict(), dict(ecc=0.3), dict(series_tol=0.0), dict(k=1)],
                         ids=["circular", "eccentric", "no-series", "no-supersampling"])
def test_workspace_kernel_is_identical(kw):
    m = model(**kw)
    old, new = both(m, rows(m))
    assert np.isfinite(old[0]).sum() > 50
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_model_uses_the_workspace_and_stays_identical():
    m = model()
    X = rows(m)
    old = both(m, X)[0][0]
    np.testing.assert_array_equal(m.log_prob_internal(X), old)
    ws = m._workspace()
    assert m._workspace() is ws                          # made once
    m.tt = m.tt.copy()                                   # replaced times: a new workspace (the old one has their mid-times)
    assert m._workspace() is not ws
    np.testing.assert_array_equal(m.log_prob_internal(X), old)
    assert "_ws" not in pickle.loads(pickle.dumps(m)).__dict__
    np.testing.assert_array_equal(copy.copy(m).log_prob_internal(X), old)


_FAULTS = """
import resource, sys
sys.path.insert(0, {here!r})
import test_workspace as T
m = T.model(n=20000, k=15)                              # 300 000 model points: ~17 MB of arrays per evaluation
X = T.rows(m, 40)
m.log_prob_internal(X[:1])
f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
for x in X:
    m.log_prob_internal(x[None])
print((resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0) / len(X))
"""


@pytest.mark.skipif(not hasattr(resource, "getrusage"), reason="needs getrusage")
def test_no_page_faults_per_evaluation():
    """In a fresh process (as in a benchmark), the allocating kernel faulted ~1900-3700 fresh pages in per evaluation
    on glibc (the freed arrays went back to the OS each time), which cost more than the arithmetic."""
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "-c", _FAULTS.format(here=here)], capture_output=True, text=True,
                         env=os.environ.copy(), check=True).stdout
    assert float(out.split()[-1]) < 50


def gp_model(k=5):
    rng = np.random.default_rng(8)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.0, 0.09, 8.0, 87.5
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.2]
    t = np.sort(rng.uniform(-0.25, 0.25, 1500))
    y = pizzetti.TransitModel(p, t, supersample_factor=k, exp_time=0.02, offsets="midpoint").light_curve(p)
    y = y + 4e-4 * np.sin(t / 0.07) + rng.normal(0, 3e-4, len(t))
    return pizzetti.MarginalTransitModel(p, t, y, 3e-4, free=["t0", "rp", "a", "b"], supersample_factor=k,
                                         exp_time=0.02, gp=pizzetti.GP())


@pytest.mark.parametrize("k", [1, 5])
def test_gp_workspace_kernels_are_identical(k):
    """_gp.loglike_batch_ws and theta_moments_batch_ws (the default single-thread GP path) against the allocating
    kernels, with the out-of-transit points skipped in the design."""
    from pizzetti import _gp
    m = gp_model(k)
    X = rows(m)
    P, lj = m._canonical_internal(X)
    E, ok = m._gp_rows(X)
    base = (m.gp_idx, m.gp_steps, m.tt, m.k, m.wts, m.y, m.gp_var0, m.tol, _A_HALF, _series._GTAB)
    lp_o, lp_n = np.zeros(len(X)), np.zeros(len(X))
    old = _gp.loglike_batch(P, lj, E, ok, m.lo, m.hi, *base, lp_o)
    new = _gp.loglike_batch_ws(P, lj, E, ok, m.lo, m.hi, *base, lp_n, m._gp_workspace())
    assert np.isfinite(old).sum() > 50
    np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(lp_o, lp_n)
    np.testing.assert_array_equal(m.log_prob_internal(X), old)
    Pc = np.ascontiguousarray(P)
    for a, b in zip(_gp.theta_moments_batch(Pc, E, *base), _gp.theta_moments_batch_ws(Pc, E, *base, m._gp_workspace())):
        np.testing.assert_array_equal(a, b)
    assert "_gws" not in pickle.loads(pickle.dumps(m)).__dict__


def test_gp_workspace_follows_a_replaced_step_table():
    """A model whose GP step table is replaced after its first evaluation gets a new workspace (the kernels do not
    check sizes: the old transition table would be written past its end)."""
    from pizzetti import _gp
    m = gp_model(1)
    X = rows(m, 20)
    m.log_prob_internal(X)
    m.gp_idx, m.gp_steps = _gp.time_table(m.t, 0.0)
    P, lj = m._canonical_internal(X)
    E, ok = m._gp_rows(X)
    base = (m.gp_idx, m.gp_steps, m.tt, m.k, m.wts, m.y, m.gp_var0, m.tol, _A_HALF, _series._GTAB)
    np.testing.assert_array_equal(m.log_prob_internal(X), _gp.loglike_batch(P, lj, E, ok, m.lo, m.hi, *base, np.zeros(0)))


@pytest.mark.parametrize("ecc", [0.5, 0.7, 0.9])
def test_skip_on_eccentric_orbits(ecc):
    """High eccentricity with sub-exposures: the Kepler solver is warm-started from a different neighbour after a
    skipped stretch, which may change the last bits (identical without sub-exposures, and in practice up to e ~ 0.3)."""
    m = model(ecc=ecc, k=9)
    old, new = both(m, rows(m))
    np.testing.assert_allclose(new[0], old[0], rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(np.isfinite(new[0]), np.isfinite(old[0]))
    m1 = model(ecc=ecc, k=1)
    for a, b in zip(*both(m1, rows(m1))):
        np.testing.assert_array_equal(a, b)
