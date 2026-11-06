"""Nested-sampling references for ``FitResult.reweight_density`` (needs dynesty; tests read the stored result,
testdata/density_refs.json).

    python tests/_density_refs.py [number of processes]        # about 30 min with 5
    python tests/_density_refs.py grid [number of processes]   # about 40 min with 5; keeps the dynesty runs

Dynesty over (t0, rp, a, b_conj, ecc, w) on the marginal likelihood of the eccentric forward model, which has no
assumption about the symmetry of the transit. Prior as in ``reweight_density``: flat in t0, rp, b_conj < 1 + rp, ecc < ECC_MAX
and w; N(RHO, RHO_SD) on the density.

"grid": a second reference, which shares neither sampler nor mapping with the others: the 4-D integral over (t0, rp, a, b) by
``profile_is`` at fixed (e, omega) on a grid gives p(e, omega | data), hence the quantiles of e. On e70w180 dynesty, with
2000 live points and two seeds that agree, puts the 84 % quantile of e at 0.78; the grid and ``reweight_density`` at 0.70.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import pizzetti

PATH = os.path.join(os.path.dirname(__file__), "..", "testdata", "density_refs.json")
G = 6.6743e-8
RHO, RHO_SD, PER, SIGMA, ECC_MAX = 1.41, 0.141, 10.0, 250e-6, 0.92
KAPPA = 3.0 * math.pi / (G * (PER * 86400.0) ** 2)                  # rho = KAPPA a^3
A_TRUE = (RHO / KAPPA) ** (1.0 / 3.0)
BOX = dict(t0=(-0.08, 0.08), rp=(1e-3, 0.3))
CASES = {"e40w090": (0.03, 0.40, 90.0, 0.40), "e40w270": (0.03, 0.40, 270.0, 0.40), "e05": (0.03, 0.05, 40.0, 0.40),
         "e70w180": (0.03, 0.70, 178.0, 0.30), "e40graz": (0.03, 0.40, 90.0, 0.985)}


def simulate(case):
    """Times and flux of a case: (rp, e, omega, b_conj) on an orbit of the density RHO, eight transits at 5 min."""
    rp, e, w, b_conj = CASES[case]
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w, p.limb_dark, p.u = 0.0, PER, rp, A_TRUE, e, w, "quadratic", [0.4, 0.25]
    p.inc = math.degrees(math.acos(b_conj * (1 + e * math.sin(math.radians(w))) / (1 - e * e) / A_TRUE))
    t = np.concatenate([np.arange(-0.35, 0.35, 5.0 / 1440.0) + i * PER for i in range(8)])
    rng = np.random.default_rng(sum(map(ord, case)))
    return t, pizzetti.TransitModel(p, t).light_curve(p) + SIGMA * rng.standard_normal(t.size)


def circular_model(case, n_free=("t0", "rp", "a", "b")):
    """The circular fit that ``reweight_density`` starts from."""
    t, y = simulate(case)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w, p.limb_dark, p.u = 0.0, PER, CASES[case][0], A_TRUE, 0.0, 90.0, "quadratic", [0.4, 0.25]
    p.inc = math.degrees(math.acos(min(CASES[case][3], 0.9) / A_TRUE))
    return pizzetti.MarginalTransitModel(p, t, y, SIGMA, free=n_free, bounds=dict(BOX, a=(1.5, 300.0)))


def wq(x, w, qs=(0.025, 0.16, 0.5, 0.84, 0.975)):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return [float(v) for v in np.interp(qs, c / c[-1], x[i])]


def run_nested(case, seed, nlive):
    import dynesty
    from scipy.special import logsumexp
    t, y = simulate(case)
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w, p.limb_dark, p.u = 0.0, PER, CASES[case][0], A_TRUE, 0.4, 90.0, "quadratic", [0.4, 0.25]
    p.inc = 89.0
    m = pizzetti.MarginalTransitModel(p, t, y, SIGMA, free=("t0", "rp", "a", "b", "ecc", "w"),
                                      bounds=dict(BOX, a=(0.5 * A_TRUE, 1.5 * A_TRUE), ecc=(0.0, ECC_MAX), w=(-90.0, 270.0)))
    lo = np.array([BOX["t0"][0], BOX["rp"][0], 0.5 * A_TRUE, 0.0, 0.0, -90.0])
    hi = np.array([BOX["t0"][1], BOX["rp"][1], 1.5 * A_TRUE, 1.0 + BOX["rp"][1], ECC_MAX, 270.0])

    def transform(u):                                               # the cube is flat in b_conj
        return lo + u * (hi - lo)

    def loglike(x):
        fac = (1 - x[4] ** 2) / (1 + x[4] * math.sin(math.radians(x[5])))
        if x[3] >= 1 + x[1] or x[2] * (1 - x[4]) <= 1 + x[1]:
            return -1e300
        r = KAPPA * x[2] ** 3
        z = x.copy()
        z[3] = x[3] / fac
        if z[3] >= x[2]:
            return -1e300
        v = float(m.log_prob_internal(z[None])[0])
        if not v > -1e299:
            return -1e300
        return v - 0.5 * ((r - RHO) / RHO_SD) ** 2 + math.log(3.0 * r / x[2])

    s = dynesty.NestedSampler(loglike, transform, 6, nlive=nlive, sample="rslice", slices=12, bound="multi",
                              rstate=np.random.default_rng(seed))
    s.run_nested(dlogz=0.05, print_progress=False)
    r = s.results
    w = np.exp(r.logwt - logsumexp(r.logwt))
    X = r.samples
    return dict(case=case, seed=seed, nlive=nlive, ncall=int(np.sum(r.ncall)), ess=float(1 / np.sum(w * w)), rp=wq(X[:, 1], w),
                a=wq(X[:, 2], w), b_conj=wq(X[:, 3], w), ecc=wq(X[:, 4], w), w=wq(X[:, 5], w),
                sd=dict(rp=float(np.sqrt(np.sum(w * (X[:, 1] - np.sum(w * X[:, 1])) ** 2))),
                        a=float(np.sqrt(np.sum(w * (X[:, 2] - np.sum(w * X[:, 2])) ** 2))),
                        b_conj=float(np.sqrt(np.sum(w * (X[:, 3] - np.sum(w * X[:, 3])) ** 2))),
                        ecc=float(np.sqrt(np.sum(w * (X[:, 4] - np.sum(w * X[:, 4])) ** 2))),
                        w=float(np.sqrt(np.sum(w * (X[:, 5] - np.sum(w * X[:, 5])) ** 2)))))


GRIDS = {"e70w180": (0.02, 0.04, 15.0), "e40graz": (0.04, 0.08, 30.0)}     # first e, step of e, step of omega


def grid_points(case):
    e0, de, dw = GRIDS[case]
    return [(float(e), float(w)) for e in np.arange(e0, ECC_MAX, de) for w in np.arange(-90.0 + 0.5 * dw, 270.0, dw)]


def grid_cell(case, e, w, idx):
    """ln of the integral at fixed (e, omega), flat in b_conj."""
    t, y = simulate(case)
    fac = (1 - e * e) / (1 + e * math.sin(math.radians(w)))

    def log_prior(d):
        r = KAPPA * d["a"] ** 3
        return -0.5 * ((r - RHO) / RHO_SD) ** 2 + np.log(3.0 * r / d["a"])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w, p.limb_dark, p.u = 0.0, PER, CASES[case][0], A_TRUE, e, w, "quadratic", [0.4, 0.25]
    p.inc = math.degrees(math.acos(min(CASES[case][3], 0.9) / fac / A_TRUE))
    m = pizzetti.MarginalTransitModel(p, t, y, SIGMA, free=("t0", "rp", "a", "b"), log_prior=log_prior,
                                      bounds=dict(BOX, a=(0.5 * A_TRUE, 1.5 * A_TRUE)))
    r = m.profile_is(n=2000, rng=np.random.default_rng(idx), evidence=False)
    return dict(e=e, w=w, li=float(r["log_int"]) + math.log(fac), ok=bool(r["is_ok"]))


def grid_summary(case, rows):
    """Quantiles of e from the cells of a grid (midpoint rule; each cell reaches halfway to its neighbours)."""
    es = sorted({q["e"] for q in rows})
    li = np.array([q["li"] for q in rows])
    pe = np.array([np.sum(np.exp(li[[q["e"] == x for q in rows]] - np.max(li))) for x in es])
    pe /= pe.sum()
    ee = np.array(es)
    edges = np.concatenate([[0.0], 0.5 * (ee[1:] + ee[:-1]), [ECC_MAX]])
    qs = np.interp((0.025, 0.16, 0.5, 0.84, 0.975), np.concatenate([[0.0], np.cumsum(pe)]), edges)
    return dict(case=case, ecc=[float(v) for v in qs], cells=len(rows), not_ok=sum(not q["ok"] for q in rows))


def main_grid(nproc):
    import concurrent.futures as cf
    out = []
    with cf.ProcessPoolExecutor(nproc) as ex:
        for case in GRIDS:
            cells = grid_points(case)
            rows = list(ex.map(grid_cell, [case] * len(cells), *zip(*cells), range(len(cells))))
            out.append(grid_summary(case, rows))
            print(out[-1], flush=True)
    with open(PATH) as f:
        d = json.load(f)
    d["grid"] = out
    with open(PATH, "w") as f:
        json.dump(d, f, indent=1)


def main(nproc):
    import dynesty
    jobs = [(c, s, 2000) for c in CASES for s in (1, 2)]
    running, out = [], []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    while jobs or running:
        while jobs and len(running) < nproc:
            running.append(subprocess.Popen([sys.executable, __file__, "one", *map(str, jobs.pop(0))], env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
        for pr in [pr for pr in running if pr.poll() is not None]:
            o, _ = pr.communicate()
            out.append(json.loads(o.decode().strip().splitlines()[-1]))
            print({k: out[-1][k] for k in ("case", "seed", "ncall", "ecc")}, flush=True)
            running.remove(pr)
        time.sleep(1)
    with open(PATH, "w") as f:
        json.dump(dict(rho=RHO, rho_sd=RHO_SD, ecc_max=ECC_MAX, dynesty=dynesty.__version__, runs=out), f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(json.dumps(run_nested(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
    elif len(sys.argv) > 1 and sys.argv[1] == "grid":
        main_grid(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
