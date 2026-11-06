"""Simulation-based calibration of ``FitResult.reweight_density`` (tests do not use it; tutorial 20 reads the stored
result, testdata/density_calibration.json).

    python tests/_density_calibration.py [number of processes]        # about 40 min with 5

For each injection the truth comes from the prior of the reweighted posterior, so that the rank of the truth in its
posterior is uniform if that posterior is right. The light curve comes from the eccentric forward model, the fit from the
circular one. Prior: a star of mean density log-uniform in (0.2, 5) g/cm3, known to a relative sd; density from that
Gaussian; P log-uniform in (2, 30) d (known); e, omega flat; (rp, b_conj) flat with rp in (0.02, 0.15), b_conj < 1 + rp;
t0 flat; q1, q2 flat. Excluded, as in the method: periastron a (1 - e) <= 1 + rp, and pi T14 g / P >= pi / 2, circular
a/R* outside the bounds of the fit. Four transits, white noise log-uniform in (100, 1000) ppm.
"""
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np

import pizzetti
from pizzetti._sampling import psis
from pizzetti.marginal import systematic_resample

PATH = os.path.join(os.path.dirname(__file__), "..", "testdata", "density_calibration.json")
G = 6.6743e-8
ECC_MAX = 0.92
BOX = dict(t0=(-0.02, 0.02), rp=(0.02, 0.15), a=(1.5, 300.0))
GROUPS = {0.10: 500, 0.03: 300, 0.30: 300}          # relative sd of the density: injections
RANKED = ("t0", "rp", "a", "b_conj", "ecc", "w", "rho", "q1", "q2")


def draw(rng, rel_sd):
    while True:
        rho0 = math.exp(rng.uniform(math.log(0.2), math.log(5.0)))
        rho = rng.normal(rho0, rel_sd * rho0)
        per = math.exp(rng.uniform(math.log(2.0), math.log(30.0)))
        e, w = rng.uniform(0.0, ECC_MAX), rng.uniform(-90.0, 270.0)
        rp, b_conj = rng.uniform(*BOX["rp"]), rng.uniform(0.0, 1.0 + BOX["rp"][1])
        if rho <= 0.0 or b_conj >= 1.0 + rp:
            continue
        a = (rho * G * (per * 86400.0) ** 2 / (3.0 * math.pi)) ** (1.0 / 3.0)
        esw = e * math.sin(math.radians(w))
        g = (1.0 + esw) / math.sqrt(1.0 - e * e)
        bo = b_conj * (1.0 + esw) / (1.0 - e * e)
        Q = (1.0 + rp) ** 2 - b_conj ** 2
        if a * (1.0 - e) <= 1.0 + rp or bo >= a or Q >= a * a - bo * bo:
            continue
        s = math.asin(math.sqrt(Q / (a * a - bo * bo)))
        s_c = s / g
        if s >= 0.5 * math.pi or s_c >= 0.5 * math.pi:
            continue
        a_c = math.sqrt(Q / math.sin(s_c) ** 2 + b_conj ** 2)
        if not BOX["a"][0] < a_c < BOX["a"][1]:
            continue
        q1, q2 = rng.uniform(), rng.uniform()
        return dict(rho0=rho0, rho=rho, per=per, ecc=e, w=w, rp=rp, b_conj=b_conj, a=a, b=bo, a_c=a_c, g=g,
                    t14=per * s / math.pi, t0=rng.uniform(*BOX["t0"]), q1=q1, q2=q2,
                    sigma=math.exp(rng.uniform(math.log(100e-6), math.log(1000e-6))))


def rank(x, w, truth):
    return float(np.sum(w[x < truth]))


def injection(rel_sd, i):
    rng = np.random.default_rng([int(rel_sd * 100), i])
    tr = draw(rng, rel_sd)
    sq = math.sqrt(tr["q1"])
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w = tr["t0"], tr["per"], tr["rp"], tr["a"], tr["ecc"], tr["w"]
    p.inc, p.limb_dark, p.u = math.degrees(math.acos(tr["b"] / tr["a"])), "quadratic", [2 * sq * tr["q2"], sq * (1 - 2 * tr["q2"])]
    hw = 1.5 * tr["t14"] + 0.02
    dt = min(5.0 / 1440.0, tr["t14"] / 15.0)
    t = np.concatenate([np.arange(-hw, hw, dt) + k * tr["per"] for k in range(4)])
    y = pizzetti.TransitModel(p, t).light_curve(p) + tr["sigma"] * rng.standard_normal(t.size)
    # a catalogue-like start of the circular fit
    pc = pizzetti.TransitParams()
    pc.t0, pc.per, pc.ecc, pc.w, pc.limb_dark, pc.u = 0.0, tr["per"], 0.0, 90.0, "quadratic", [0.4, 0.25]
    pc.rp = np.clip(tr["rp"] * (1 + 0.03 * rng.standard_normal()), 1.01 * BOX["rp"][0], 0.99 * BOX["rp"][1])
    pc.a = np.clip(tr["a_c"] * (1 + 0.03 * rng.standard_normal()), 1.01 * BOX["a"][0], 0.99 * BOX["a"][1])
    pc.inc = math.degrees(math.acos(min(0.95, abs(tr["b_conj"] + 0.05 * rng.standard_normal())) / pc.a))
    t_start = time.perf_counter()
    m = pizzetti.MarginalTransitModel(pc, t, y, tr["sigma"], free=("t0", "rp", "a", "b"), bounds=BOX)
    fit = m.fit(seed=i, n=4000)
    t_fit = time.perf_counter() - t_start
    out = dict(rel_sd=rel_sd, i=i, n_eval=fit.diagnostics["n_eval"], is_ok=bool(fit.diagnostics.get("is_ok")), t_fit=t_fit,
               stratified=bool(fit.diagnostics.get("stratified")), n_pts=len(t), snr=tr["rp"] ** 2 / tr["sigma"] * math.sqrt(4 * tr["t14"] / dt),
               **{k: tr[k] for k in ("rho", "per", "ecc", "w", "rp", "b_conj", "a", "g", "t14")})
    r = fit.reweight_density((tr["rho0"], rel_sd * tr["rho0"]), ecc_max=ECC_MAX, seed=i)
    S, L, W = r.weighted_samples, r.labels, r.weights
    d = r.diagnostics["density"]
    out.update(ess=d["ess"], ess_parents=d["ess_parents"], wall=d["a_wall_mass"], passes=d["passes"])
    out["ranks"] = {q: rank(S[:, L.index(q)], W, tr[q]) for q in RANKED}
    # would the ratio of the likelihoods of the eccentric and the circular orbit improve on this? It does not.
    me = pizzetti.MarginalTransitModel(p, t, y, tr["sigma"], free=("t0", "rp", "a", "b", "ecc", "w"),
                                       bounds=dict(a=(1.0, 1e4), ecc=(0.0, 0.999), w=(-90.0, 270.0)))
    rng_x = np.random.default_rng([i, 1])
    u, count = np.unique(systematic_resample(W, 4000, rng_x), return_counts=True)
    le = me.log_prob(np.column_stack([S[u, L.index(q)] for q in ("t0", "rp", "a", "b", "ecc", "w")]))
    Lc, Sc = fit.labels, fit.weighted_samples[r._parents[u]]
    lc = m.log_prob(np.column_stack([Sc[:, Lc.index(q)] for q in ("t0", "rp", "a", "b")]))
    dl = le - lc
    ok = np.isfinite(dl)
    lw, khat = psis(dl[ok])
    wx = count[ok] * np.exp(lw - np.max(lw))
    wx /= wx.sum()
    out.update(khat=khat, ess_ratio=1.0 / np.sum(wx * wx), dlnl_abs95=np.percentile(np.abs(dl[ok]), 95))
    out["ranks_ratio"] = {q: rank(S[u[ok], L.index(q)], wx, tr[q]) for q in RANKED}
    for key in ("ess", "ess_parents", "wall", "t_fit", "rho", "per", "ecc", "w", "rp", "b_conj", "a", "g", "t14", "snr", "khat", "ess_ratio", "dlnl_abs95"):
        out[key] = round(float(out[key]), 4)
    for key in ("ranks", "ranks_ratio"):
        out[key] = {q: round(v, 4) for q, v in out[key].items()}
    return out


def main(nproc):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = [(sd, k, min(k + 25, n)) for sd, n in GROUPS.items() for k in range(0, n, 25)]
    out, running, tries = {}, [], {}
    while jobs or running:
        while jobs and len(running) < nproc:
            job = jobs.pop(0)
            tries[job] = tries.get(job, 0) + 1
            running.append((job, subprocess.Popen([sys.executable, __file__, "one", *map(str, job)], env=env,
                                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)))
        for job, pr in [q for q in running if q[1].poll() is not None]:
            o, _ = pr.communicate()
            for line in o.decode().splitlines():
                if line.startswith("{"):
                    d = json.loads(line)
                    out[d["rel_sd"], d["i"]] = d
            miss = [i for i in range(job[1], job[2]) if (job[0], i) not in out]
            if miss and tries[job] < 3:                    # numba's cache can race on a fresh start
                jobs.append(job)
            running.remove((job, pr))
            print(len(out), "done", flush=True)
        time.sleep(1)
    rows = [out[k] for k in sorted(out)]
    with open(PATH, "w") as f:
        json.dump(dict(ecc_max=ECC_MAX, box=BOX, ranked=RANKED, runs=rows), f, separators=(",", ":"),
                  default=float)
    print(sum(GROUPS.values()) - len(rows), "missing")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        for i in range(int(sys.argv[3]), int(sys.argv[4])):
            try:
                print(json.dumps(injection(float(sys.argv[2]), i), default=float), flush=True)
            except Exception as err:
                print("failed", sys.argv[2], i, repr(err)[:200], file=sys.stderr, flush=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
