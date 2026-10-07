"""Comparison with batman (Kreidberg 2015) through the identical public API."""
import numpy as np
import pytest

batman = pytest.importorskip("batman")
import pizzetti  # noqa: E402

# Differences to batman are dominated by batman's own errors: its quadratic
# law uses Hastings polynomials for K and E (~2e-8), its numerical laws use a
# step size tuned to `max_err` (here the smallest allowed value, 0.001 ppm),
# which leaves errors up to ~1.2e-7 against an exact reference (see
# test_closer_to_reference_than_batman). pizzetti itself is verified to 2e-9
# against a 40-digit reference in test_accuracy.py.
TOL_QUAD = 5e-8
TOL_NUM = 3e-7
NUMERICAL = ("nonlinear", "power2", "squareroot")

LAWS = [
    ("uniform", []),
    ("linear", [0.6]),
    ("quadratic", [0.4, 0.25]),
    ("quadratic", [0.1, 0.6]),
    ("squareroot", [0.3, 0.4]),
    ("nonlinear", [0.5, 0.1, 0.1, -0.1]),
    ("nonlinear", [0.8, -0.6, 0.9, -0.3]),
    ("power2", [0.7, 0.6]),
    ("power2", [0.5, 1.3]),
]


def make_params(mod, rp=0.1, a=15.0, inc=88.5, ecc=0.0, w=90.0, law="quadratic", u=(0.4, 0.25)):
    p = mod.TransitParams()
    p.t0 = 0.0
    p.per = 3.0
    p.rp = rp
    p.a = a
    p.inc = inc
    p.ecc = ecc
    p.w = w
    p.u = list(u)
    p.limb_dark = law
    p.fp = 0.001
    p.t_secondary = 1.5
    return p


def both(t, law, u, max_err=0.001, **kw):
    pb = make_params(batman, law=law, u=u, **kw)
    pp = make_params(pizzetti, law=law, u=u, **kw)
    mb = batman.TransitModel(pb, t, max_err=max_err if law in NUMERICAL else 1.0)
    mp = pizzetti.TransitModel(pp, t)
    return mb.light_curve(pb), mp.light_curve(pp), mb, mp


@pytest.mark.parametrize("law,u", LAWS)
@pytest.mark.parametrize("rp", [0.0092, 0.05, 0.1, 0.2])
def test_light_curve_matches_batman(law, u, rp):
    t = np.linspace(-0.12, 0.12, 3001)
    fb, fp, _, _ = both(t, law, u, rp=rp, inc=89.0)
    tol = TOL_NUM if law in NUMERICAL else TOL_QUAD
    assert np.max(np.abs(fb - fp)) < tol


@pytest.mark.parametrize("law,u", [("quadratic", [0.4, 0.25]), ("nonlinear", [0.5, 0.1, 0.1, -0.1])])
def test_grazing_and_large_planets(law, u):
    t = np.linspace(-0.12, 0.12, 2001)
    for rp, inc in ((0.1, 86.5), (0.3, 87.0), (0.5, 89.0)):
        fb, fp, _, _ = both(t, law, u, rp=rp, inc=inc)
        tol = TOL_NUM if law in NUMERICAL else TOL_QUAD
        assert np.max(np.abs(fb - fp)) < tol, (rp, inc)


def test_random_parameters_quadratic():
    rng = np.random.default_rng(42)
    t = np.linspace(-0.2, 0.2, 2001)
    for _ in range(40):
        rp = rng.uniform(0.003, 0.3)
        a = rng.uniform(5, 50)
        b = rng.uniform(0, 1 + rp)
        inc = np.degrees(np.arccos(b / a))
        u1 = rng.uniform(0, 1)
        u2 = rng.uniform(-0.2, 1 - u1)
        fb, fp, _, _ = both(t, "quadratic", [u1, u2], rp=rp, a=a, inc=inc)
        assert np.max(np.abs(fb - fp)) < TOL_QUAD


def test_eccentric_orbit_separations():
    # batman stops its Kepler iteration once the residual is below 1e-7 rad,
    # so its true anomaly is uncertain by ~1e-7 (separations by ~1e-7 a);
    # pizzetti solves Kepler's equation to 1e-15 (test_kepler_solver_precision)
    t = np.linspace(-1.0, 4.0, 5001)
    a = 15.0
    for ecc, w in ((0.1, 30.0), (0.5, 250.0), (0.9, 100.0)):
        _, _, mb, mp = both(t, "quadratic", [0.4, 0.25], ecc=ecc, w=w)
        ok = mb.ds < 99
        assert np.array_equal(ok, mp.ds < 99)
        assert np.max(np.abs(mb.ds[ok] - mp.ds[ok])) < 3e-7 * a * (1 + ecc) / (1 - ecc)
        assert np.allclose(mb.get_true_anomaly() % (2 * np.pi), mp.get_true_anomaly() % (2 * np.pi), atol=1e-6)


def test_eccentric_light_curve():
    t = np.linspace(-0.2, 0.2, 2001)
    fb, fp, _, _ = both(t, "quadratic", [0.3, 0.2], ecc=0.3, w=60.0, inc=89.0)
    assert np.max(np.abs(fb - fp)) < TOL_QUAD


def test_supersampling():
    t = np.linspace(-0.1, 0.1, 501)
    pb = make_params(batman)
    pp = make_params(pizzetti)
    mb = batman.TransitModel(pb, t, supersample_factor=7, exp_time=0.02)
    mp = pizzetti.TransitModel(pp, t, supersample_factor=7, exp_time=0.02)
    assert np.max(np.abs(mb.light_curve(pb) - mp.light_curve(pp))) < TOL_QUAD


def test_secondary_eclipse():
    t = np.linspace(1.3, 1.7, 2001)
    pb = make_params(batman)
    pp = make_params(pizzetti)
    mb = batman.TransitModel(pb, t, transittype="secondary")
    mp = pizzetti.TransitModel(pp, t, transittype="secondary")
    assert np.max(np.abs(mb.light_curve(pb) - mp.light_curve(pp))) < 1e-12


def test_inverse_transit():
    t = np.linspace(-0.1, 0.1, 501)
    fb, fp, _, _ = both(t, "quadratic", [0.4, 0.25], rp=-0.1)
    assert np.max(np.abs(fb - fp)) < TOL_QUAD
    assert fp.max() > 1.0


def test_parameter_updates_recompute_orbit():
    t = np.linspace(-0.1, 0.1, 501)
    pb = make_params(batman)
    pp = make_params(pizzetti)
    mb = batman.TransitModel(pb, t)
    mp = pizzetti.TransitModel(pp, t)
    for p in (pb, pp):
        p.t0 = 0.01
        p.inc = 89.0
        p.rp = 0.12
        p.u = [0.2, 0.3]
    assert np.max(np.abs(mb.light_curve(pb) - mp.light_curve(pp))) < TOL_QUAD


def test_timing_helpers():
    t = np.linspace(-0.1, 0.1, 11)
    pb = make_params(batman, ecc=0.3, w=40.0)
    pp = make_params(pizzetti, ecc=0.3, w=40.0)
    mb = batman.TransitModel(pb, t)
    mp = pizzetti.TransitModel(pp, t)
    for name in ("get_t_periastron", "get_t_secondary", "get_t_conjunction"):
        assert getattr(mb, name)(pb) == pytest.approx(getattr(mp, name)(pp), abs=1e-12)


def test_input_validation():
    t = np.linspace(-0.1, 0.1, 11)
    p = make_params(pizzetti)
    with pytest.raises(TypeError):
        pizzetti.TransitModel(p, list(t))
    p.u = [0.1]
    with pytest.raises(ValueError):
        pizzetti.TransitModel(p, t)
    p.limb_dark = "logarithmic"
    p.u = [0.1, 0.2]
    with pytest.raises(NotImplementedError):
        pizzetti.TransitModel(p, t)
    p = make_params(pizzetti)
    with pytest.raises(ValueError):
        pizzetti.TransitModel(p, t, supersample_factor=3)


def test_kepler_solver_precision():
    from pizzetti._orbit import eccentric_anomaly
    rng = np.random.default_rng(3)
    for _ in range(2000):
        e = rng.uniform(0, 0.99)
        M = rng.uniform(-20, 20)
        E = eccentric_anomaly(M, e)
        assert abs(E - e * np.sin(E) - M) < 1e-12


def test_closer_to_reference_than_batman():
    mpmath = pytest.importorskip("mpmath")  # noqa: F841
    from reference import flux_ref
    from pizzetti.laws import power_terms
    k = 0.2
    z = np.linspace(0, 1 + k, 120)
    for law, u in (("nonlinear", [0.8, -0.6, 0.9, -0.3]), ("power2", [0.7, 0.6])):
        pb = make_params(batman, rp=k, law=law, u=u)
        mb = batman.TransitModel(pb, np.array([0.0]), max_err=0.001)
        al, c = power_terms(law, u)
        ref = np.array([flux_ref(x, k, al, c) for x in z])
        import batman._nonlinear_ld as nl
        import batman._power2_ld as p2
        if law == "nonlinear":
            fb = nl._nonlinear_ld(z, k, *u, mb.fac, 1)
        else:
            fb = p2._power2_ld(z, k, u[0], u[1], mb.fac, 1)
        fp = pizzetti.occult(z, k, u, law)
        assert np.max(np.abs(fp - ref)) < 2e-9
        assert np.max(np.abs(fp - ref)) < 0.1 * np.max(np.abs(fb - ref))
