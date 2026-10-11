"""Wall-clock time of a short emcee run (2000 steps x 32 walkers) for both codes,
run one after the other on an otherwise idle machine. Writes results/wall.json."""
import json
import os
import time

import numpy as np
import emcee

import run_inference as R


def main(nsteps=2000, nwalkers=32):
    out = {}
    for case in ("tess", "kepler"):
        c = R.CASES[case]
        t, y = R.simulate(c)
        ivar = 1.0 / c["sigma"] ** 2
        for code in ("batman", "pizzetti"):
            prm, model = R.build(code, c, t)

            def log_prob(th):
                if not R.in_prior(c, th):
                    return -np.inf
                R.set_params(c, prm, th)
                r = y - model.light_curve(prm)
                return -0.5 * ivar * np.dot(r, r)
            th0 = R.theta_true(c)
            scale = np.array([2e-4, 2e-6, 1e-3, 0.05, 0.02, 0.02, 0.02])
            p0 = th0 + scale * np.random.default_rng(1234).standard_normal((nwalkers, th0.size))
            log_prob(p0[0])
            s = emcee.EnsembleSampler(nwalkers, th0.size, log_prob)
            s.random_state = np.random.RandomState(2024).get_state()
            t1 = time.perf_counter()
            s.run_mcmc(p0, nsteps, progress=False)
            out[f"{case}_{code}"] = time.perf_counter() - t1
            print(case, code, round(out[f"{case}_{code}"], 2), "s")
    json.dump(out, open(os.path.join(R.OUT, "wall.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
