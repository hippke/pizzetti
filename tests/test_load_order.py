"""Seeded results do not depend on which copy of a kernel a process runs (numba: the one loaded first)."""
import glob
import os
import subprocess
import sys

import pizzetti

SCRIPT = """
import hashlib
import sys

import numpy as np

import pizzetti
from pizzetti import _exact, terms

if sys.argv[1] == "callees_first":      # as in a run that compiles: each kernel is loaded before its callers
    z = np.linspace(0.92, 1.08, 8)
    terms.exact_batch_terms(z, 8, 0.1, *[np.empty(8) for _ in range(4)])
    terms.exact_point_terms(1.0, 0.1)
    _exact.exact_batch(z, 8, 0.1, 0.1, 0.5, 0.25, 1.0, np.empty(8), np.empty(8))


def h(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


t = (np.arange(7)[:, None] * 3.5 + np.arange(-0.13, 0.13, 2.0 / 1440)[None, :]).ravel()
p = pizzetti.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.5, 0.1, 9.0, np.degrees(np.arccos(0.35 / 9.0)), 0.0, 90.0
p.limb_dark, p.u = "quadratic", [0.4, 0.25]
lc = pizzetti.TransitModel(p, t).light_curve(p)
y = lc + 5e-4 * np.random.default_rng(42).standard_normal(t.size)
m = pizzetti.MarginalTransitModel(p, t, y, 5e-4, free=("t0", "rp", "a", "b"),
                                  bounds=dict(t0=(-0.05, 0.05), rp=(1e-4, 0.5), a=(1.5, 30.0)))
r = m.profile_is(n=1000, rng=np.random.default_rng(1))
print(h(lc), h(m.log_prob_internal(r["theta"])), r["n_eval"], h(r["w"]))
"""


def run(order):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(pizzetti.__file__)))
    return subprocess.run([sys.executable, "-c", SCRIPT, order], env=env, capture_output=True, text=True, check=True).stdout


def test_same_bits_for_any_load_order():
    assert run("callers_first") == run("callees_first")


def test_no_kernel_allows_reassociation():
    # the copies of a kernel with reassociation round differently; the test above cannot visit every kernel
    for f in glob.glob(os.path.join(os.path.dirname(pizzetti.__file__), "*.py")):
        with open(f) as fh:
            assert "fastmath=True" not in fh.read(), f
