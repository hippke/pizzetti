"""Sinh-trapezoid limb rules (_trap) and their Chebyshev interpolants (_interp)
against the Green's-theorem quadrature, and the connection formula they rest on."""
import numpy as np
import pytest

from pizzetti import _general, _interp, _trap
from pizzetti.occult import occult, series_cut

LAWS = [("nonlinear", [0.5, 0.1, 0.1, -0.1]), ("squareroot", [0.3, 0.2]), ("power2", [0.7, 0.6]),
        ("power2", [0.6, 0.05]), ("power2", [0.6, 1.0]), ("power2", [0.6, 3.7]), ("power2", [0.6, 8.0])]


def _z(p):
    d = np.geomspace(1e-14, 1e-1, 40) * p
    return np.concatenate([np.linspace(0.0, 1.0 + 1.2 * p, 3001), 1 - p + d, 1 - p - d, 1 + p - d])


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("p", [0.001, 0.0092, 0.05, 0.1, 0.15, 0.2, 0.29, 0.45])
@pytest.mark.parametrize("limb", ["trapezoid", "interpolate", "fast"])
def test_limb_rules_vs_quadrature(law, u, p, limb):
    z = _z(p)
    for tol in (1e-9, 0.0):
        ref = occult(z, p, u, law, series_tol=tol, limb="quadrature")
        assert np.max(np.abs(occult(z, p, u, law, series_tol=tol, limb=limb) - ref)) < 3e-15


@pytest.mark.parametrize("g", [0.05, 0.25, 0.5, 0.75, 1.25, 2.0])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.25])
def test_cross_kernel(g, p):
    gams, cs = np.array([g]), np.array([1.0])
    W, tt, b = _trap.rules(gams)
    z = np.concatenate([1 - p + np.geomspace(1e-15, 2 * p, 60)[:-1], 1 + p - np.geomspace(1e-15, 2 * p, 60)[1:]])
    out = np.empty(z.size)
    _trap.blocked_cross(z, z.size, p, gams, cs, W, tt, b, out)
    ju, jw, has = _general.quad_rules(gams)
    ref = np.array([_general.blocked_limb(x, p, gams, cs, _general.GL_U, _general.GL_W, ju, jw, has) for x in z])
    assert np.max(np.abs(out - ref)) < 3e-15 * np.pi / (g + 1)


def test_connection_formula_continuity():
    """A and B of J = A + lam^nu B (lam > 0) and Re J = A + cos(pi nu) |lam|^nu B (lam < 0)
    join smoothly at second contact: the interpolants of both sides agree there."""
    p, gams = 0.1, np.array([0.3])
    zc = series_cut(p, "power2", [0.7, 0.6])
    lam_lo = 1 - (1 - (zc - p) ** 2) / (4 * zc * p)
    g = _trap.band_gamma(p, zc)
    W, tt, b = _trap.rules(gams)
    BW = _trap.rules_in(gams, g)
    CW = _trap.rules(gams, _interp.H_CB)
    EO, AC, BC, AN, BN = _interp.build(p, gams, lam_lo, W, tt, b, *BW, g, BW[4][BW[2].shape[0]], *CW)
    a_c = np.polynomial.chebyshev.chebval(-1.0, AC[0])        # lam = 0 from outside
    a_n = np.polynomial.chebyshev.chebval(1.0, AN[0])         # lam = 0 from inside
    b_c = np.polynomial.chebyshev.chebval(-1.0, BC[0])
    b_n = np.polynomial.chebyshev.chebval(1.0, BN[0])
    assert abs(a_c - a_n) < 1e-13 and abs(b_c - b_n) < 1e-12 * abs(b_c)
