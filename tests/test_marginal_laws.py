"""MarginalTransitModel for all limb-darkening laws and secondary eclipses."""
import math

import numpy as np
import pytest

import pizzetti
from pizzetti import _laws, _marginal as K

T_ONE = np.arange(-0.13, 0.13, 2 / 1440)


def setup(law, u, sigma=3e-4, seed=1, ntr=3, transittype="primary", fp=None):
    t = (np.arange(ntr)[:, None] * 3.5 + T_ONE[None, :]).ravel()
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, 3.5, 0.1, 9.0
    p.inc = math.degrees(math.acos(0.35 / 9.0))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, law, list(u)
    if transittype == "secondary":
        p.fp, p.t_secondary = fp, 1.75
        t = t + 1.75
    if law == "logarithmic":            # not in TransitModel: our basis (validated vs mpmath)
        lw = _laws.Logarithmic()
        p2 = pizzetti.TransitParams()
        p2.t0, p2.per, p2.rp, p2.a, p2.inc, p2.ecc, p2.w = p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w
        p2.limb_dark, p2.u = "uniform", []
        d = pizzetti.TransitModel(p2, t).ds
        c = lw.from_u(u)
        f = lw.basis(d, p.rp) @ c / (lw.disk() @ c)
    else:
        f = pizzetti.TransitModel(p, t, transittype=transittype).light_curve(p)
    y = f + sigma * np.random.default_rng(seed).standard_normal(t.size)
    return t, y, p


def test_generic_path_equals_fast_quadratic():
    t, y, p = setup("quadratic", [0.4, 0.25])
    fast = pizzetti.MarginalTransitModel(p, t, y, 3e-4, supersample_factor=3, exp_time=0.003)
    gen = pizzetti.MarginalTransitModel(p, t, y, 3e-4, supersample_factor=3, exp_time=0.003)
    gen.fast = False
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(0, 3e-4, 20), np.full(20, 3.5), rng.uniform(0.09, 0.11, 20),
                         rng.uniform(8.5, 9.5, 20), rng.uniform(0.0, 0.6, 20)])
    a, b = fast.log_prob(X), gen.log_prob(X)
    fin = np.isfinite(a)
    assert np.array_equal(fin, np.isfinite(b))
    # same model; the two paths use slightly different series bounds (both < 1e-9 in flux)
    assert np.max(np.abs(a[fin] - b[fin])) < 1e-4


def _brute(model, geo, alpha):
    """Integral over the law's normalised coefficients g (grid with physical indicator;
    a wide box for the flat prior) and the flux scale s (Gauss-Hermite) of
    N(y | s A g) |s|^(K-1), i.e. the flat-in-c prior used in the likelihood."""
    law = model.law
    P = model._canonical_physical(np.atleast_2d(geo))[0]
    A = model._design_row(P, alpha)
    lo, hi = (-0.2, 1.2) if model.physical else (-8.0, 8.0)
    n = 4000 if law.K == 2 else (220 if model.physical else 900)
    if law.K == 2:
        g = np.linspace(lo, hi, n)
        G = np.column_stack([g, np.ones_like(g)])
        dA = g[1] - g[0]
    else:
        gg = np.linspace(lo, hi, n)
        g0, g1 = np.meshgrid(gg, gg, indexing="ij")
        G = np.column_stack([g0.ravel(), g1.ravel(), np.ones(g0.size)])
        dA = (gg[1] - gg[0]) ** 2
    if model.physical:
        G = G[np.all(G @ law.D.T >= 0, axis=1)]
    m = G @ A.T
    yy = model.yw @ model.yw
    mm, my = np.sum(m * m, 1), m @ model.yw
    shat, sd = my / mm, 1 / np.sqrt(mm)
    x, wq = np.polynomial.hermite_e.hermegauss(40)
    s_ = shat[:, None] + sd[:, None] * x[None, :]
    vals = (np.log(np.sum(wq[None, :] * np.abs(s_) ** (law.K - 1), axis=1)) + np.log(sd)
            - 0.5 * (yy - my ** 2 / mm))
    return vals.max() + np.log(np.sum(np.exp(vals - vals.max())) * dA)


@pytest.mark.parametrize("law,u", [("linear", [0.5]), ("quadratic", [0.4, 0.25]),
                                   ("squareroot", [0.3, 0.4]), ("logarithmic", [0.6, 0.3]),
                                   ("power2", [0.6, 0.8])])
@pytest.mark.parametrize("prior", ["physical", "flat"])
def test_marginal_likelihood_vs_brute_force(law, u, prior):
    t, y, p = setup(law, u, sigma=2e-3, ntr=1)
    m = pizzetti.MarginalTransitModel(p, t, y, 2e-3, reparam="none", ld_prior=prior)
    m.fast = False                                        # test the generic machinery
    ga, gb = np.array([0.0, 3.5, 0.1, 9.0, 0.35]), np.array([5e-4, 3.5, 0.097, 9.3, 0.25])
    al = 0.8 if law == "power2" else None
    exact = m.log_prob(ga) - m.log_prob(gb)
    brute = _brute(m, ga, al) - _brute(m, gb, al)
    assert abs(exact - brute) < 0.01, (exact, brute)


def test_secondary_basis_matches_transitmodel():
    t, y, p = setup("quadratic", [0.4, 0.25], transittype="secondary", fp=2e-3)
    m = pizzetti.MarginalTransitModel(p, t, y, 3e-4, transittype="secondary",
                                      free=["t_secondary", "per", "rp", "a", "b"])
    P = m._canonical_physical(m.fixed_physical_free())[0]
    A = m._design_row(P, None) / m.wts[:, None]
    ref = pizzetti.TransitModel(p, t, transittype="secondary").light_curve(p)
    assert np.max(np.abs(A @ np.array([1.0, 2e-3]) - ref)) < 1e-12


LAW_CASES = [("uniform", []), ("linear", [0.5]), ("quadratic", [0.4, 0.25]),
             ("squareroot", [0.3, 0.4]), ("logarithmic", [0.6, 0.3]), ("power2", [0.6, 0.8]),
             ("nonlinear", [0.5, 0.1, 0.1, -0.1])]


@pytest.mark.parametrize("law,u", LAW_CASES)
def test_fit_recovers_truth(law, u):
    pytest.importorskip("scipy")
    # the nonlinear law needs high S/N for its four coefficients (see guide)
    sigma, ntr = (1e-5, 7) if law == "nonlinear" else (1e-4, 3)
    t, y, p = setup(law, u, sigma=sigma, ntr=ntr)
    m = pizzetti.MarginalTransitModel(p, t, y, sigma)
    r = m.fit(method="auto", seed=3, n=2000, ld="u", max_steps=20000)
    S, lab = r.samples, r.labels
    for k, v in (("rp", 0.1), ("a", 9.0), ("b", 0.35)):
        x = S[:, lab.index(k)]
        assert abs(np.median(x) - v) < 4 * x.std() + 1e-6, (law, k, np.median(x), x.std())
    if law == "nonlinear":
        assert r.diagnostics["physical_fraction"] > 0.5
        uu = [np.median(S[:, lab.index(f"u{k}")]) for k in range(1, 5)]
        sd = [S[:, lab.index(f"u{k}")].std() for k in range(1, 5)]
        assert all(abs(a_ - b_) < 4 * s_ for a_, b_, s_ in zip(uu, u, sd)), (uu, sd)
    f0 = S[:, lab.index("f0")]
    assert abs(np.median(f0) - 1.0) < 5 * max(f0.std(), 1e-6)


def test_power2_free_alpha_and_secondary_fit():
    pytest.importorskip("scipy")
    t, y, p = setup("power2", [0.6, 0.8], sigma=5e-5)
    m = pizzetti.MarginalTransitModel(p, t, y, 5e-5, free=["t0", "per", "rp", "a", "b", "alpha"],
                                      bounds={"alpha": (0.05, 5.0)})
    r = m.fit(method="is", seed=4, n=2000, ld="u")
    al = r.samples[:, r.labels.index("alpha")]
    assert abs(np.median(al) - 0.8) < 4 * al.std() + 0.02
    # secondary eclipse: fp recovered
    t, y, p = setup("quadratic", [0.4, 0.25], sigma=1e-4, transittype="secondary", fp=2e-3)
    # (rp fixed from the transit, as usual: an eclipse alone barely constrains it)
    m = pizzetti.MarginalTransitModel(p, t, y, 1e-4, transittype="secondary",
                                      free=["t_secondary", "a", "b"])
    r = m.fit(method="auto", seed=5, n=2000)
    fp = r.samples[:, r.labels.index("fp")]
    assert r.labels[-2:] == ["fp", "f0"]
    assert abs(np.median(fp) - 2e-3) < 4 * fp.std()


def test_nonlinear_physicality_test_is_exact():
    law = _laws.Nonlinear()
    rng = np.random.default_rng(6)
    A = rng.uniform(-3, 3, (20000, 4))
    c = np.column_stack([1 - A.sum(1), A])                 # power coefficients
    b = np.linalg.solve(_laws._M4, c.T).T                  # law coefficients (Bernstein)
    s = np.linspace(0, 1, 4001)
    I = c @ np.vstack([s ** k for k in range(5)])
    dI = c[:, 1:] @ np.vstack([(k + 1) * s ** k for k in range(4)])
    grid = (I.min(1) >= -1e-12) & (dI.min(1) >= -1e-9)
    exact = law.is_physical(b)
    assert np.mean(grid == exact) > 0.999
