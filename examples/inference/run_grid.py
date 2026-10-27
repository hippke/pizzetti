"""Inference grid of the paper (Section on parameter inference): hot Jupiter and super-Earth,
power-2 limb darkening with the exponent alpha fixed or free, plus the quadratic-law hot Jupiter,
each fitted with batman and with pizzetti under identical likelihoods, walkers and random
streams. Every run records the wall time of the whole emcee run, the time spent inside the
light-curve call (orbit + flux), and the remainder (emcee, Python, chi^2).

Usage: python run_grid.py <case> <code> [nsteps]
  case: jup_quad | jup_fixed | jup_free | se_fixed | se_free
  code: batman | pizzetti
Writes results/grid_<case>_<code>.json and results/gridchain_<case>_<code>.npy.
"""
import json
import os
import sys
import time

import numpy as np
import emcee

import run_inference as R

HOT = dict(R.CASES["tess"])
SE = dict(R.CASES["kepler"])
CASES = {
    "jup_quad": dict(HOT, key="tess", fixed_alpha=False,
                     label="Hot Jupiter, quadratic law"),
    "jup_fixed": dict(HOT, key="jup", law="power2", ld=[0.60, 0.60], fixed_alpha=True,
                      label=r"Hot Jupiter, power-2, $\alpha$ fixed"),
    "jup_free": dict(HOT, key="jup", law="power2", ld=[0.60, 0.60], fixed_alpha=False,
                     label=r"Hot Jupiter, power-2, $\alpha$ free"),
    "se_fixed": dict(SE, key="se", fixed_alpha=True,
                     label=r"Super-Earth, power-2, $\alpha$ fixed"),
    "se_free": dict(SE, key="se", fixed_alpha=False,
                    label=r"Super-Earth, power-2, $\alpha$ free"),
}


def expand(c, th):
    """Walker vector -> full 7-parameter vector (alpha appended when it is held fixed)."""
    if c["fixed_alpha"]:
        return np.append(th, c["ld"][1])
    return th


def main(case, code, nsteps=20000, nwalkers=32):
    c = CASES[case]
    t, y = R.simulate(c)
    prm, model = R.build(code, c, t)
    import pizzetti
    pp, pm = R.build("pizzetti", c, t)
    pm.light_curve(pp)
    n_in = int(np.sum(np.abs(pm.ds) < 1 + c["rp"]))
    ivar = 1.0 / c["sigma"] ** 2
    acc = {"lc": 0.0, "lp": 0.0, "n": 0}

    def log_prob(th):
        s0 = time.perf_counter()
        thf = expand(c, th)
        if not R.in_prior(c, thf):
            acc["lp"] += time.perf_counter() - s0
            return -np.inf
        R.set_params(c, prm, thf)
        s1 = time.perf_counter()
        f = model.light_curve(prm)
        s2 = time.perf_counter()
        r = y - f
        lp = -0.5 * ivar * np.dot(r, r)
        acc["lc"] += s2 - s1
        acc["lp"] += time.perf_counter() - s0
        acc["n"] += 1
        return lp

    th0 = R.theta_true(c)
    ndim = 6 if c["fixed_alpha"] else 7
    th0w = th0[:ndim]
    scale = np.array([2e-4, 2e-6, 1e-3, 0.05, 0.02, 0.02, 0.02])[:ndim]
    rng = np.random.default_rng(1234)
    p0 = th0w + scale * rng.standard_normal((nwalkers, ndim))
    log_prob(p0[0])                                     # compile / warm up
    acc.update(lc=0.0, lp=0.0, n=0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.random_state = np.random.RandomState(2024).get_state()
    s = time.perf_counter()
    sampler.run_mcmc(p0, nsteps, progress=False)
    wall = time.perf_counter() - s
    os.makedirs(R.OUT, exist_ok=True)
    np.save(os.path.join(R.OUT, f"gridchain_{case}_{code}.npy"), sampler.get_chain())
    try:
        tau = sampler.get_autocorr_time(tol=0).tolist()
    except Exception:
        tau = None
    ncalls = nwalkers * (nsteps + 1)
    info = dict(case=case, code=code, label=c["label"], law=c["law"], fixed_alpha=c["fixed_alpha"],
                n_points=int(t.size), n_in_transit=n_in,
                nwalkers=nwalkers, nsteps=nsteps, ndim=ndim, ncalls=ncalls, ncalls_model=acc["n"],
                wall_s=wall, lightcurve_s=acc["lc"], logprob_s=acc["lp"],
                overhead_s=wall - acc["lc"], emcee_s=wall - acc["lp"],
                per_call_total_us=1e6 * wall / ncalls, per_call_lc_us=1e6 * acc["lc"] / max(acc["n"], 1),
                autocorr=tau, acceptance=float(np.mean(sampler.acceptance_fraction)),
                theta_true=th0.tolist(), names=R.NAMES[:ndim])
    json.dump(info, open(os.path.join(R.OUT, f"grid_{case}_{code}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "autocorr"}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
