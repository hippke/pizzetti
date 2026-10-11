"""The language-neutral reference vectors of testdata/ (independent tanh-sinh quadrature
of the area integral): every limb method of the Python package, and the interior series
within its tolerance. The Julia package runs the same checks."""
import json
import os

import numpy as np
import pytest

from pizzetti import _general, occult

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "testdata", "blocked_reference.json")


def _cases():
    with open(DATA) as f:
        return json.load(f)["double"]


def _blocked(z, p, g, **kw):
    """P_g from occult with the single-term law I = mu^(2g) (normalisation pi/(g+1))."""
    f = occult(np.array([z]), p, [1.0, 2.0 * g], "power2", **kw)[0]   # c = 1: I = mu^alpha
    return (1.0 - f) * np.pi / (g + 1.0)


@pytest.mark.parametrize("case", range(77))
def test_quadrature(case):
    cs = _cases()[case]
    p, g = cs["p"], cs["gamma"]
    gams = np.array([g])
    ju, jw, has = _general.quad_rules(gams)
    for z, P in zip(cs["z"], cs["P"]):
        v = _general.blocked_limb(z, p, gams, np.array([1.0]), _general.GL_U, _general.GL_W, ju, jw, has)
        assert abs(v - P) <= 3e-15 * np.pi / (g + 1)


@pytest.mark.parametrize("case", range(0, 77, 3))
def test_occult_modes(case):
    cs = _cases()[case]
    p, g = cs["p"], cs["gamma"]
    if g == 0:
        pytest.skip("power2 needs alpha > 0")
    unit = np.pi / (g + 1)
    for z, P in zip(cs["z"], cs["P"]):
        for limb in ("quadrature", "fast", "proven"):
            assert abs(_blocked(z, p, g, series_tol=0.0, limb=limb) - P) <= 3e-15 * unit
        assert abs(_blocked(z, p, g) - P) <= 1.0001e-9 * unit
