"""occult_terms: the three LD-independent terms in one pass."""
import mpmath as mp
import numpy as np
import pytest

import pizzetti


def _physical_u(rng):
    a = rng.uniform(0, 2)
    return a, rng.uniform(-a / 2, 1 - a)


@pytest.mark.parametrize("k", [0.005, 0.02, 0.1, 0.2, 0.35, 0.6, 0.95, 1.3])
def test_matches_occult(k):
    rng = np.random.default_rng(0)
    z = np.concatenate([np.linspace(0, 1 + k + 0.05, 3001),
                        [k, abs(1 - k), 1 + k, 1 + k - 1e-12, k + 1e-9, k - 1e-9]])
    T = pizzetti.occult_terms(z, k)
    assert T.shape == (3, z.size)
    for _ in range(6):
        u = _physical_u(rng)
        # both evaluations are within series_tol = 1e-9 of the exact flux
        assert np.max(np.abs(pizzetti.quadratic_from_terms(T, u) - pizzetti.occult(z, k, u))) < 2e-9


def test_exact_branch_identical_and_series_bound():
    rng = np.random.default_rng(1)
    for k in (0.02, 0.1, 0.2, 0.35):
        z = np.linspace(0, 1 + k + 0.02, 4001)
        T, Tx = pizzetti.occult_terms(z, k), pizzetti.occult_terms(z, k, series_tol=0)
        assert np.max(np.abs(T[1] - Tx[1])) <= 2.5e-10          # ldx: series_tol / 4
        assert np.array_equal(T[0], Tx[0]) or np.max(np.abs(T[0] - Tx[0])) < 1e-15
        for _ in range(5):
            u = _physical_u(rng)
            assert np.max(np.abs(pizzetti.quadratic_from_terms(Tx, u)
                                 - pizzetti.occult(z, k, u, series_tol=0))) < 1e-15
            assert np.max(np.abs(pizzetti.quadratic_from_terms(T, u)
                                 - pizzetti.quadratic_from_terms(Tx, u))) < 1e-9


def _blocked(z, k, n):
    """(1/pi) * integral of mu^n over the part of the planet disk on the star (30 digits)."""
    mp.mp.dps = 30
    z, k = mp.mpf(z), mp.mpf(k)

    def alpha(r):
        if z == 0:
            return mp.pi if r < k else mp.mpf(0)
        c = (r * r + z * z - k * k) / (2 * r * z)
        return mp.pi if c <= -1 else (mp.mpf(0) if c >= 1 else mp.acos(c))

    f = lambda r: (1 - r * r) ** (mp.mpf(n) / 2) * 2 * r * alpha(r)  # noqa: E731
    lo, hi = abs(z - k), min(z + k, mp.mpf(1))
    pts = [mp.mpf(0), k] if z == 0 else ([mp.mpf(0), lo, hi] if z < k else [lo, hi])
    return float(mp.quad(f, pts) / mp.pi)


@pytest.mark.parametrize("z", [0.0, 0.05, 0.1, 0.3, 0.85, 0.9, 0.95, 1.0, 1.05, 1.09])
def test_against_quadrature(z):
    lex, ldx, edx = pizzetti.occult_terms(np.array([z]), 0.1)[:, 0]
    for got, n in ((lex, 0), (ldx, 1), (lex - edx, 2)):
        assert abs(got - _blocked(z, 0.1, n)) < 3e-10


def test_out_of_transit_and_full_occultation():
    T = pizzetti.occult_terms(np.array([1.2, 5.0]), 0.1)
    assert np.all(T == 0.0)
    T = pizzetti.occult_terms(np.array([0.0, 0.2]), 1.5)            # fully occulted
    for u in ((0.4, 0.25), (0.0, 1.0), (2.0, -1.0)):
        assert np.allclose(pizzetti.quadratic_from_terms(T, u), 0.0, atol=1e-15)
