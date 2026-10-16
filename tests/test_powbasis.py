"""Fast power-law basis columns (_powbasis) and the fused marginal likelihood."""
import numpy as np
import pytest

import pizzetti
from pizzetti import _general, _laws, _marginal, _powbasis

from test_marginal_laws import LAW_CASES, setup


def _z(p):
    d = np.geomspace(1e-13, 1e-2, 30) * p
    return np.concatenate([np.linspace(0.0, 1.0 + 1.2 * p, 2501), 1 - p + d, 1 - p - d, 1 + p - d])


def _quad(z, p, g):
    ju, jw, has = _general.quad_rules(np.array([g]))
    return np.array([_general.blocked_limb(abs(x), p, np.array([g]), np.ones(1), _general.GL_U,
                                           _general.GL_W, ju, jw, has) if abs(x) < 1 + p else 0.0
                     for x in z]) / np.pi


@pytest.mark.parametrize("gams", [[0.25], [0.25, 0.75], [0.03, 0.4, 1.4]])
@pytest.mark.parametrize("p", [0.004, 0.05, 0.1, 0.17, 0.26])
def test_power_columns_match_occult(gams, p):
    """Columns (series + interpolated contact expansions) vs pizzetti.occult at
    series_tol 1e-12 (quadrature at the limb); p = 0.26 is outside the validated
    expansions and must fall back to quadrature."""
    pc = _powbasis.PowerColumns(gams)
    z = _z(p)
    out = pc(z, p)
    ref = np.column_stack([_laws.blocked_power(z, p, 2 * g) for g in gams])
    assert np.max(np.abs(out - ref)) < 2e-12
    if p > 0.25:
        assert not pc.panel(p)[2].any()          # not validated: quadrature


def test_power_columns_vs_quadrature_near_cut():
    """Between the interior cut and piece C (the region the degree-64 expansions
    were extended to) against Green's-theorem quadrature."""
    p, g = 0.1, 0.25
    pc = _powbasis.PowerColumns([g])
    zc = _powbasis._cut_terms(p, np.array([g]), np.ones(1), 1e-12, _powbasis._series._GTAB)
    z = np.linspace(zc, 1 + p, 400)[:-1]
    assert np.max(np.abs(pc(z, p)[:, 0] - _quad(z, p, g))) < 1e-12


@pytest.mark.parametrize("p", [0.03, 0.1, 0.15])
def test_alpha_columns_free_exponent(p):
    """Free exponent: 2-D (p, alpha) panels, including alpha near and at 1 (g = 1/2,
    where the contact series have cancelling poles: regularised form)."""
    ac = _powbasis.AlphaColumns()
    z = _z(p)
    for al in (0.31, 0.6, 0.97, 0.9999, 1.0001, 1.05, 1.5, 2.71):
        assert np.max(np.abs(ac(z, p, al) - _laws.blocked_power(z, p, al))) < 1e-12, al
    zl = np.linspace(0.6, 1 + p, 300)[:-1]
    assert np.max(np.abs(ac(zl, p, 1.0) - _quad(zl, p, 0.5))) < 1e-12


def test_ne_solve_matches_qr():
    rng = np.random.default_rng(0)
    for K in (1, 3, 5):
        A = rng.standard_normal((500, K)) @ np.triu(rng.standard_normal((K, K)) + 3 * np.eye(K))
        y = A @ rng.standard_normal(K) + rng.standard_normal(500)
        r1 = _marginal._qr_solve(A, y)
        r2 = _marginal._ne_solve(np.ascontiguousarray(A.T), y)
        assert r1[0] and r2[0]
        np.testing.assert_allclose(r2[1], r1[1], rtol=1e-9, atol=1e-12)
        assert abs(r1[3] - r2[3]) < 1e-9 and abs(r1[4] - r2[4]) < 1e-8 * r1[4]


def _perturb(m, x0, n, seed):
    """Rows around x0 on the scale of typical posterior widths."""
    sc = {"t0": 2e-4, "t_secondary": 2e-4, "per": 2e-6, "rp": 1e-3, "a": 2e-4, "b": 0.02,
          "alpha": 0.1}
    s = np.array([sc[q] for q in m.free])
    return x0 + s * np.random.default_rng(seed).standard_normal((n, len(x0)))


def _reference_loglike(m, X):
    """Python reference: per-row design from the laws' original basis functions
    (pizzetti.occult at series_tol 1e-12) and the QR solver."""
    P, lj = m._canonical_internal(X)
    al = m._alpha(X)
    out = np.full(len(X), -np.inf)
    for i in range(len(X)):
        if not (np.all(np.isfinite(P[i])) and m._prior_ok(P[i], al[i])):
            continue
        A = m._design_row(P[i], al[i])
        ok, c, Fi, logdet, chi2 = _marginal._qr_solve(A, m.yw)
        lp = -0.5 * chi2 - 0.5 * logdet
        if m.cone:
            lp = _marginal.marg_loglike_gen(A, m.yw, m.Dmat, True)
        out[i] = lp + lj[i]
    return out


@pytest.mark.parametrize("ksup,prior", [(1, "physical"), (3, "flat")])
@pytest.mark.parametrize("law,u", [c for c in LAW_CASES if c[0] != "quadratic"])
def test_fused_path_matches_reference(law, u, ksup, prior):
    t, y, p = setup(law, u, sigma=1e-4, ntr=3)
    free = ["t0", "per", "rp", "a", "b"] + (["alpha"] if law == "power2" else [])
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, free=free, supersample_factor=ksup,
                                      exp_time=0.02 if ksup > 1 else 0.0, ld_prior=prior)
    assert m._fused
    x0 = m.to_internal(m.fixed_physical_free())[0]
    X = _perturb(m, x0, 12, 1)
    new = m.log_prob_internal(X)
    ref = _reference_loglike(m, X)
    ok = np.isfinite(ref)
    assert ok.sum() >= 6 and np.array_equal(np.isfinite(new), ok)
    assert np.max(np.abs(new[ok] - ref[ok])) < 1e-4


def test_fused_secondary_matches_reference():
    t, y, p = setup("quadratic", [0.4, 0.25], sigma=1e-4, ntr=3, transittype="secondary", fp=1e-3)
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, transittype="secondary",
                                      free=["t_secondary", "per", "rp", "a", "b"])
    x0 = m.to_internal(m.fixed_physical_free())[0]
    X = _perturb(m, x0, 8, 2)
    new = m.log_prob_internal(X)
    ref = _reference_loglike(m, X)
    ok = np.isfinite(ref)
    assert ok.sum() >= 6 and np.max(np.abs(new[ok] - ref[ok])) < 1e-6
