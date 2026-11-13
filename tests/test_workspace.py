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
    new = K.loglike_batch_ws(*a, lp_n, ne_n, K.workspace(len(m.tt), m.k))
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
