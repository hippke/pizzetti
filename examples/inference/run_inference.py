"""Parameter inference with emcee: batman vs pizzetti, same likelihood, same
starting positions, same random seed. Writes chains and timings to results/.

Usage: python run_inference.py <case> <code> [nsteps]
  case: tess   hot Jupiter, TESS 2-min cadence, quadratic law
        kepler super-Earth, Kepler 1-min cadence, power-2 law
  code: batman | pizzetti
"""
import json
import os
import sys
import time

import numpy as np
import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

CASES = {
    "tess": dict(
        label="Hot Jupiter, TESS 2-min, quadratic law",
        law="quadratic", per=3.5, t0=0.0, rp=0.10, a=9.0, b=0.35, ld=[0.40, 0.25],
        ntransits=7, cadence=2.0 / 1440, halfwindow=0.13, sigma=500e-6, seed=42),
    "kepler": dict(
        label="Super-Earth, Kepler 1-min, power-2 law",
        law="power2", per=10.0, t0=0.0, rp=0.02, a=20.0, b=0.20, ld=[0.60, 0.60],
        ntransits=30, cadence=1.0 / 1440, halfwindow=0.12, sigma=150e-6, seed=7),
}
NAMES = ["t0", "per", "rp", "a", "b", "ld1", "ld2"]


def make_time(c):
    epochs = np.arange(c["ntransits"])
    t_one = np.arange(-c["halfwindow"], c["halfwindow"], c["cadence"])
    return (c["t0"] + epochs[:, None] * c["per"] + t_one[None, :]).ravel()


def ld_from_theta(c, l1, l2):
    """Sampling parameters -> limb-darkening coefficients.
    quadratic: Kipping (2013) q1, q2 in (0, 1); power-2: c in (0, 1), alpha in (0, 2)."""
    if c["law"] == "quadratic":
        s = np.sqrt(l1)
        return [2 * s * l2, s * (1 - 2 * l2)]
    return [l1, l2]


def theta_true(c):
    if c["law"] == "quadratic":
        u1, u2 = c["ld"]
        q1 = (u1 + u2) ** 2
        q2 = u1 / (2 * (u1 + u2))
        return np.array([c["t0"], c["per"], c["rp"], c["a"], c["b"], q1, q2])
    return np.array([c["t0"], c["per"], c["rp"], c["a"], c["b"], *c["ld"]])


def in_prior(c, th):
    t0, per, rp, a, b, l1, l2 = th
    if not (0 < rp < 0.5 and 1.5 < a < 100 and 0 <= b < 1 + rp and per > 0):
        return False
    if c["law"] == "quadratic":
        return 0 < l1 < 1 and 0 < l2 < 1
    return 0 < l1 < 1 and 0 < l2 < 2


def build(code, c, t):
    mod = __import__(code)
    prm = mod.TransitParams()
    prm.t0, prm.per, prm.rp, prm.a = c["t0"], c["per"], c["rp"], c["a"]
    prm.inc = np.degrees(np.arccos(c["b"] / c["a"]))
    prm.ecc, prm.w, prm.limb_dark, prm.u = 0.0, 90.0, c["law"], list(c["ld"])
    model = mod.TransitModel(prm, t)              # default settings of each package
    return prm, model


def set_params(c, prm, th):
    t0, per, rp, a, b, l1, l2 = th
    prm.t0, prm.per, prm.rp, prm.a = t0, per, rp, a
    prm.inc = np.degrees(np.arccos(b / a))
    prm.u = ld_from_theta(c, l1, l2)


def simulate(c):
    t = make_time(c)
    # noise-free reference: pizzetti's Green's-theorem quadrature, series switched off
    import pizzetti
    prm = pizzetti.TransitParams()
    prm.t0, prm.per, prm.rp, prm.a = c["t0"], c["per"], c["rp"], c["a"]
    prm.inc = np.degrees(np.arccos(c["b"] / c["a"]))
    prm.ecc, prm.w, prm.limb_dark, prm.u = 0.0, 90.0, c["law"], list(c["ld"])
    f = pizzetti.TransitModel(prm, t, series_tol=0.0, limb="quadrature").light_curve(prm)
    rng = np.random.default_rng(c["seed"])
    y = f + c["sigma"] * rng.standard_normal(t.size)
    return t, y


def main(case, code, nsteps=6000, nwalkers=32):
    c = CASES[case]
    t, y = simulate(c)
    prm, model = build(code, c, t)
    ivar = 1.0 / c["sigma"] ** 2

    def log_prob(th):
        if not in_prior(c, th):
            return -np.inf
        set_params(c, prm, th)
        r = y - model.light_curve(prm)
        return -0.5 * ivar * np.dot(r, r)

    th0 = theta_true(c)
    scale = np.array([2e-4, 2e-6, 1e-3, 0.05, 0.02, 0.02, 0.02])
    rng = np.random.default_rng(1234)
    p0 = th0 + scale * rng.standard_normal((nwalkers, th0.size))
    log_prob(p0[0])                                 # compile / warm up
    t_call = []
    for k in range(200):
        s = time.perf_counter()
        log_prob(p0[k % nwalkers])
        t_call.append(time.perf_counter() - s)
    sampler = emcee.EnsembleSampler(nwalkers, th0.size, log_prob)
    sampler.random_state = np.random.RandomState(2024).get_state()      # identical proposals for both codes
    s = time.perf_counter()
    sampler.run_mcmc(p0, nsteps, progress=False)
    wall = time.perf_counter() - s
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"chain_{case}_{code}.npy"), sampler.get_chain())
    np.save(os.path.join(OUT, f"lnp_{case}_{code}.npy"), sampler.get_log_prob())
    try:
        tau = sampler.get_autocorr_time(tol=0).tolist()
    except Exception:
        tau = None
    info = dict(case=case, code=code, label=c["label"], n_points=int(t.size), nwalkers=nwalkers, nsteps=nsteps,
                wall_s=wall, call_ms_median=1e3 * float(np.median(t_call)), autocorr=tau,
                acceptance=float(np.mean(sampler.acceptance_fraction)), theta_true=th0.tolist(), names=NAMES)
    json.dump(info, open(os.path.join(OUT, f"info_{case}_{code}.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
