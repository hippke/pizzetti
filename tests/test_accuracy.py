"""Accuracy against an independent high-precision reference (mpmath) and the
rigorous bound of the interior series."""
import numpy as np
import pytest

from pizzetti import occult, series_cut
from pizzetti import _series
from pizzetti.laws import power_terms

mpmath = pytest.importorskip("mpmath")
from reference import disk_mean_ref, flux_ref  # noqa: E402

LAWS = [
    ("uniform", []),
    ("linear", [0.6]),
    ("quadratic", [0.4, 0.25]),
    ("squareroot", [0.3, 0.4]),
    ("nonlinear", [0.5, 0.1, 0.1, -0.1]),
    ("power2", [0.7, 0.6]),
]


def contact_points(k):
    d = np.geomspace(1e-15, 0.2, 14)
    z = np.concatenate([1 - k - d, 1 - k + d, 1 + k - d, [0.0, k, k * (1 + 1e-12), 0.5 * (1 - k)]])
    z = z[(z >= 0) & (z < 1 + k)]
    if k >= 1:
        z = z[z > k - 1]
    return z


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("k", [0.0092, 0.1, 0.3, 0.7, 1.5])
def test_against_mpmath_reference(law, u, k):
    """Full model (series + limb method) against 40-digit quadrature,
    including points within 1e-15 of the contacts, z = 0 and z = k."""
    z = contact_points(k)
    al, c = power_terms(law, u)
    f = occult(z, k, u, law)
    ref = np.array([flux_ref(x, k, al, c) for x in z])
    assert np.max(np.abs(f - ref)) < 2e-9


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("k", [0.001, 0.0092, 0.03, 0.1, 0.2, 0.4])
def test_series_error_below_bound(law, u, k):
    """The interior series stays within series_tol of the limb method, which
    is exact or 1e-11 accurate, for all z <= z_c."""
    tol = 1e-9
    zc = series_cut(k, law, u, tol)
    if zc < 0:
        pytest.skip("series not used")
    z = np.concatenate([np.linspace(0, zc, 4001), zc * (1 - np.geomspace(1e-15, 1e-3, 30))])
    f = occult(z, k, u, law, series_tol=tol)
    f0 = occult(z, k, u, law, series_tol=0)
    assert np.max(np.abs(f - f0)) <= tol * 1.05 + 1e-13


def test_bound_is_rigorous_and_tight_high_precision():
    """Binomial truncation vs 40-digit reference of the disk mean: error >= 0,
    error <= bound, bound within a factor 2 for z <= 0.9 (1 - k)."""
    M = _series.M
    A = _series.table(0.5, M)
    for k in (0.01, 0.1, 0.2):
        for zf in (0.0, 0.3, 0.6, 0.9, 0.97):
            z = zf * (1 - k)
            D = float(disk_mean_ref(z, k, 0.5))
            a = _series.coeffs(A, k, 1.0)
            w = 1 - z * z
            S = np.sqrt(w) * sum(a[j] / w ** j for j in range(M + 1))
            err = S - D
            b = _series.bound(z, k, 0.5, M)
            assert err >= -1e-14
            assert err <= b * (1 + 1e-9) + 1e-14
            if zf <= 0.9 and err > 1e-12:
                assert b / err < 2


@pytest.mark.parametrize("g", [0.25, 0.3, 0.5, 0.75, 1.25])
def test_bound_monotone_in_z(g):
    for k in (0.01, 0.1, 0.3):
        z = np.linspace(0, (1 - k) * 0.999, 500)
        b = np.array([_series.bound(x, k, g, _series.M) for x in z])
        assert np.all(np.diff(b) >= -1e-300)


def test_table_matches_exact_rationals():
    from fractions import Fraction as Fr
    from math import comb

    def beta(g, m):
        r = Fr(1)
        for i in range(1, m + 1):
            r *= Fr(i - 1) - g
            r /= i
        return r

    M = _series.M
    g = Fr(1, 2)
    A = [[Fr(0)] * (M + 1) for _ in range(M + 1)]
    for m in range(M + 1):
        for l in range(0, m + 1, 2):
            cc = beta(g, m) * Fr(comb(m, l) * comb(l, l // 2) * 2, 2 * m - l + 2)
            q = m - l // 2
            for i in range(l // 2 + 1):
                A[m - i][q] += cc * comb(l // 2, i) * (-1) ** i
    exact = np.array([[float(x) for x in r] for r in A])
    assert np.allclose(_series.table(0.5, M), exact, rtol=1e-13, atol=1e-15)


def test_full_occultation_and_no_transit():
    assert occult(np.array([0.1]), 1.5, [0.4, 0.25])[0] == 0.0
    assert occult(np.array([2.0]), 0.1, [0.4, 0.25])[0] == 1.0
    assert occult(np.array([0.3]), 0.0, [0.4, 0.25])[0] == 1.0
    assert occult(np.array([0.1]), 1.5, [0.7, 0.6], "power2")[0] == 0.0
