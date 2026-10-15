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
        fn = os.path.join(R.OUT, f"chain_{case}_pizzetti.npy")
        if os.path.exists(fn):
            chain = np.load(fn)
            samp = chain[4000:].reshape(-1, chain.shape[-1])
        else:  # chains are not stored: use the stored posterior samples, jittered so that
            # every call still sees a new parameter vector (relative 1e-4, far below the width)
            base = np.asarray(json.load(open(os.path.join(R.OUT, "posterior_samples.json")))[case]["samples"])
            rng = np.random.default_rng(0)
            samp = np.repeat(base, -(-nsel * (reps + 1) // len(base)), axis=0)
            samp = samp * (1 + 1e-4 * rng.standard_normal(samp.shape))
        # distinct posterior samples for every repetition, so that no code profits from
        # caches keyed by parameter values (pizzetti caches the hypergeometric tables
        # per limb-darkening exponent; in a fit every call has a new exponent)
        allsel = samp[np.random.default_rng(1).choice(len(samp), nsel * (reps + 1), replace=False)]
        sel = allsel[:nsel]
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
        # per-call time: loop over posterior samples (every call changes the parameters, so the
        # orbit is recomputed as in a fit), repeated with new samples, codes interleaved
        err = {name: 0.0 for name in configs}
        for th in sel:
            R.set_params(c, ref_p, th)
            fr = ref_m.light_curve(ref_p)
            for name in configs:
                prm, m = models[name]
                R.set_params(c, prm, th)
                err[name] = max(err[name], float(np.max(np.abs(m.light_curve(prm) - fr))))
        times = {name: [] for name in configs}
        for rep in range(reps):
            chunk = allsel[(rep + 1) * nsel:(rep + 2) * nsel]
            for name in configs:
                prm, m = models[name]
                s = time.perf_counter()
                for th in chunk:
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
