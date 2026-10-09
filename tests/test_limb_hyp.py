"""Hypergeometric limb solution (ingress/egress and the interior band beyond
z_c) against the 40-digit reference and against the quadrature."""
import numpy as np
import pytest

from pizzetti import occult, series_cut
from pizzetti import _lens
from pizzetti.laws import power_terms

mpmath = pytest.importorskip("mpmath")
from reference import flux_ref  # noqa: E402

LAWS = [
    ("squareroot", [0.3, 0.4]),
    ("nonlinear", [0.5, 0.1, 0.1, -0.1]),
    ("power2", [0.7, 0.6]),
    ("power2", [0.7, 1.0]),          # exponent 1/2: logarithmic connection formula
    ("power2", [0.7, 0.99998]),      # near-integer m: cancellation-prone
]


def limb_grid(k, law, u, n=6000):
    zc = series_cut(k, law, u)
    lo = zc if zc > 0 else 1 - k
    d = np.geomspace(1e-13, 1e-3, 8)
    special = np.concatenate([1 - k - d, 1 - k + d, 1 + k - d])
    z = np.concatenate([np.linspace(lo, 1 + k, n)[1:-1], special])
    return np.sort(z[(z > lo) & (z < 1 + k)])


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("k", [0.0092, 0.05, 0.1, 0.2])
def test_hyp_against_mpmath(law, u, k):
    z = limb_grid(k, law, u)
    f = occult(z, k, u, law, limb="fast")
    al, c = power_terms(law, u)
    rng = np.random.default_rng(0)
    pick = np.unique(np.concatenate([rng.choice(z.size, 12, replace=False),
                                     np.arange(z.size - 8, z.size), np.searchsorted(z, 1 - k) + np.arange(-4, 4)]))
    pick = pick[(pick >= 0) & (pick < z.size)]
    ref = np.array([flux_ref(z[i], k, al, c) for i in pick])
    # k = 0.2: samples near second contact fall back to quadrature (~1e-12)
    tol = 2e-14 if k < 0.2 else 3e-12
    if u[-1] == 0.99998:
        tol = max(tol, 1e-12)
    assert np.max(np.abs(f[pick] - ref)) < tol


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("k", [0.0092, 0.05, 0.15, 0.2, 0.3])
@pytest.mark.parametrize("tol", [1e-9, 1e-12])
def test_proven_against_mpmath(law, u, k, tol):
    """Proven mode: the flux error is at most series_tol where certified."""
    z = limb_grid(k, law, u)
    f, cert = occult(z, k, u, law, series_tol=tol, limb="proven", return_certified=True)
    if k <= 0.15 or (k <= 0.2 and tol >= 1e-9):
        assert cert.all()
    al, c = power_terms(law, u)
    rng = np.random.default_rng(1)
    pick = np.unique(np.concatenate([rng.choice(z.size, 16, replace=False), np.arange(z.size - 6, z.size),
                                     np.searchsorted(z, 1 - k) + np.arange(-4, 4)]))
    pick = pick[(pick >= 0) & (pick < z.size)]
    ref = np.array([flux_ref(z[i], k, al, c) for i in pick])
    err = np.abs(f[pick] - ref)
    assert np.max(err[cert[pick]], initial=0.0) <= tol
    assert np.max(err, initial=0.0) <= max(tol, 3e-12)


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("k", [0.0092, 0.1, 0.3])
def test_hyp_matches_quadrature(law, u, k):
    z = limb_grid(k, law, u)
    f = occult(z, k, u, law, limb="fast")
    g = occult(z, k, u, law, limb="quadrature")
    assert np.max(np.abs(f - g)) < 1e-11


@pytest.mark.parametrize("g", [0.0, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5])
def test_universal_functions(g):
    T, L = _lens.table(g)
    for j in (0, 1, 5, 20):
        for x in (0.0, 0.2, 0.5, 0.6, 0.9, 0.999, 1 - 1e-9):
            for fam in (0, 1):
                if fam == 0:
                    a, b, c = 0.5, 0.5, g + 2.5 + j
                    s = mpmath.beta(0.5, g + 2 + j)
                else:
                    a, b, c = -(g + 1 + j), 0.5, 1.0
                    s = mpmath.pi
                ref = float(s * mpmath.hyp2f1(a, b, c, x))
                y = 1 - x
                ly = np.log(y)
                ym = np.exp((g + 1.5 + j) * ly) if x > 0.5 else 0.0
                v = _lens._family(T, L, fam, j, x, ym, ly)
                # j = 20 interior functions: mild cancellation (they enter the flux times x^20)
                assert abs(v - ref) <= 5e-13 * abs(ref) + 1e-300


def test_transitmodel_limb_option():
    import pizzetti
    prm = pizzetti.TransitParams()
    prm.t0, prm.per, prm.rp, prm.a, prm.inc, prm.ecc, prm.w = 0.0, 3.0, 0.1, 10.0, 89.0, 0.0, 90.0
    prm.limb_dark, prm.u = "power2", [0.6, 0.6]
    t = np.linspace(-0.08, 0.08, 4001)
    fa = pizzetti.TransitModel(prm, t).light_curve(prm)
    fb = pizzetti.TransitModel(prm, t, limb="proven").light_curve(prm)
    fc = pizzetti.TransitModel(prm, t, limb="quadrature").light_curve(prm)
    assert np.max(np.abs(fa - fc)) < 1e-11
    assert np.max(np.abs(fb - fc)) < 1e-9
