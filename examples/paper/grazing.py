"""Section 6.4, grazing transits: likelihood cost (orbit + light curve + chi^2) at posterior
samples, as ../inference/benchmark_calls.py, for the two paper cases at their own impact
parameter and at near-grazing ones (b = 1 - p: no interior phase; b = 1), batman vs pizzetti.
Run: python grazing.py [out.json] [path/to/other/pizzetti/src]   (~3 min)"""
import sys, json, os, time
if len(sys.argv) > 2: sys.path.insert(0, sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "inference"))
import numpy as np
import run_inference as R
import pizzetti, batman
print("pizzetti", pizzetti.__file__, file=sys.stderr)
VAR = {"tess": [None, 0.85, 0.9, 1.0], "kepler": [None, 0.97, 0.98, 1.0]}   # b; None = paper value
base_all = json.load(open(os.path.join(R.OUT, "posterior_samples.json")))
out = {}
nsel, reps = 200, 15
for case in ("tess", "kepler"):
    for bnew in VAR[case]:
        c = dict(R.CASES[case]); 
        if bnew is not None: c["b"] = bnew
        t, y = R.simulate(c)
        ivar = 1 / c["sigma"] ** 2
        base = np.asarray(base_all[case]["samples"]).copy()
        if bnew is not None:
            base[:, 4] = bnew        # every call at this impact parameter (other parameters from the posterior)
        rng = np.random.default_rng(0)
        samp = np.repeat(base, -(-nsel * (reps + 1) // len(base)), axis=0)
        samp = samp * (1 + 1e-4 * rng.standard_normal(samp.shape))
        allsel = samp[np.random.default_rng(1).choice(len(samp), nsel * (reps + 1), replace=False)]
        allsel[:, 4] = np.abs(allsel[:, 4])
        if bnew is not None:
            allsel[:, 4] = bnew
        codes = ["pizzetti", "batman"] if len(sys.argv) <= 2 else ["pizzetti"]   # a second pizzetti: compare versions
        res = {}
        models = {}
        for code in codes:
            prm, m = R.build(code, c, t)
            models[code] = (prm, m)
            R.set_params(c, prm, allsel[0]); m.light_curve(prm)
        times = {k: [] for k in codes}
        for rep in range(reps):
            chunk = allsel[(rep + 1) * nsel:(rep + 2) * nsel]
            for code in codes:
                prm, m = models[code]
                s = time.perf_counter()
                for th in chunk:
                    R.set_params(c, prm, th)
                    r = y - m.light_curve(prm)
                    _ = -0.5 * ivar * np.dot(r, r)
                times[code].append(1e3 * (time.perf_counter() - s) / nsel)
        prm, m = models["pizzetti"]
        R.set_params(c, prm, allsel[0])
        z = m.ds; k = c["rp"]
        from pizzetti.occult import series_cut
        zc = series_cut(k, c["law"], prm.u)
        nin = int(np.sum(np.abs(z) < 1 + k)); nser = int(np.sum(np.abs(z) <= zc))
        key = f"{case} b={c['b']}"
        out[key] = {k_: float(np.median(v)) for k_, v in times.items()}
        out[key].update(n=int(t.size), n_in=nin, n_series=nser)
        print(key, out[key], file=sys.stderr)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "grazing.json"), "w"), indent=1)
