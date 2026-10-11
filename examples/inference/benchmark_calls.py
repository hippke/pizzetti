"""Cost of one likelihood evaluation (orbit + light curve + chi^2) at posterior samples,
batman vs pizzetti, measured interleaved on an otherwise idle machine, and the model
error of each code against the reference (pizzetti's quadrature with the series off).

Usage: python benchmark_calls.py   (writes results/timing.json)
"""
import json
import os
import time

import numpy as np

import run_inference as R


def timed(fn, reps):
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return 1e3 * float(np.median(t))


def main(nsel=200, reps=20):
    out = {}
    for case in ("tess", "kepler"):
        c = R.CASES[case]
        t, y = R.simulate(c)
        ivar = 1.0 / c["sigma"] ** 2
        chain = np.load(os.path.join(R.OUT, f"chain_{case}_pizzetti.npy"))
        samp = chain[4000:].reshape(-1, chain.shape[-1])
        sel = samp[np.random.default_rng(1).choice(len(samp), nsel, replace=False)]
        configs = {"batman": dict(code="batman", kw={}),
                   "pizzetti": dict(code="pizzetti", kw={})}
        if c["law"] == "power2":
            configs["batman, max_err=0.01 ppm"] = dict(code="batman", kw=dict(max_err=0.01))
        import pizzetti
        ref_p, _ = R.build("pizzetti", c, t)
        ref_m = pizzetti.TransitModel(ref_p, t, series_tol=0.0, limb="quadrature")
        res = {}
        models = {}
        for name, cf in configs.items():
            mod = __import__(cf["code"])
            prm, _ = R.build(cf["code"], c, t)
            m = mod.TransitModel(prm, t, **cf["kw"])
            models[name] = (prm, m)
        for name in configs:                      # warm up
            prm, m = models[name]
            R.set_params(c, prm, sel[0])
            m.light_curve(prm)
        # per-call time: loop over all samples (every call changes the parameters, so the
        # orbit is recomputed as in a fit), repeated, codes interleaved
        err = {name: 0.0 for name in configs}
        for th in sel:
            R.set_params(c, ref_p, th)
            fr = ref_m.light_curve(ref_p)
            for name in configs:
                prm, m = models[name]
                R.set_params(c, prm, th)
                err[name] = max(err[name], float(np.max(np.abs(m.light_curve(prm) - fr))))
        times = {name: [] for name in configs}
        for _ in range(reps):
            for name in configs:
                prm, m = models[name]
                s = time.perf_counter()
                for th in sel:
                    R.set_params(c, prm, th)
                    r = y - m.light_curve(prm)
                    _ = -0.5 * ivar * np.dot(r, r)
                times[name].append(1e3 * (time.perf_counter() - s) / len(sel))
        for name in configs:
            res[name] = dict(call_ms=float(np.median(times[name])), max_model_error=err[name])
        res["n_points"] = int(t.size)
        res["n_in_transit"] = int(np.sum(R.build("pizzetti", c, t)[1].ds < 1 + c["rp"]))
        out[case] = res
        print(case, json.dumps(res, indent=1))
    json.dump(out, open(os.path.join(R.OUT, "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
