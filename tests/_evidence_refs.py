"""Nested-sampling references for the ln evidence of profile importance sampling (needs dynesty; tests read
the stored result, testdata/evidence_refs.json).

    python tests/_evidence_refs.py [number of processes]        # about 30 min with 6

Prior: uniform in the free parameters, over the boxes below and 0 <= b < 1 + rp; uniform (q1, q2); uniform f0 in
(F_LO, F_HI). Two estimates per case:

* "ns7": conventional, no marginalisation: dynesty over t0, rp, a, b, q1, q2, f0 on the Gaussian likelihood of the
  forward model. Shares only the light-curve kernel with the code under test.
* "ns_marg": dynesty over the free parameters on ``log_prob_internal`` (without the T14 Jacobian), times the
  constants and <f0 / c2^3> as written out in ``ln_z_from_integral``. Same integrand as importance sampling, a
  different way to integrate it. The only one for "t15", whose 15 sub-exposures make 8-D sampling too slow.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import quad

import pizzetti
from pizzetti import _marginal as K
from pizzetti import _series
from pizzetti.marginal import _A_HALF, _draw_c_theta

PATH = os.path.join(os.path.dirname(__file__), "..", "testdata", "evidence_refs.json")
F_LO, F_HI = 0.99, 1.01
PER, A_TRUE = 3.5, 9.0
BOX = dict(t0=(-0.05, 0.05), rp=(1e-4, 0.5), a=(1.5, 30.0))
BOX_T15 = dict(t0=(-0.02, 0.02), per=(0.3675, 0.3705), rp=(1e-4, 0.5), a=(1.01, 30.0))
CASES = {"b035": (0.10, 0.35, 500e-6), "b085lo": (0.103, 0.85, 1500e-6), "b100": (0.103, 1.00, 500e-6),
         "b100lo": (0.103, 1.00, 1500e-6), "small": (0.03, 0.50, 500e-6)}
ALL = list(CASES) + ["t15"]


def params(rp, a, b, per):
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, per, rp, a
    p.inc = math.degrees(math.acos(b / a))
    p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.40, 0.25]
    return p


def build(name):
    """Simulated light curve of a case -> (model, boxes). "t15": section 1 of the tutorial
    examples/15_when_the_profile_is_not_enough.ipynb (grazing branch) with finite boxes for per and a."""
    if name == "t15":
        exp = 1765.5 / 86400
        t = (np.arange(147)[:, None] * 0.369 + np.arange(-0.0625, 0.0625, exp)[None, :]).ravel()
        p = params(0.0172, 3.0, 0.4, 0.369)
        kw = dict(supersample_factor=15, exp_time=exp, offsets="midpoint")
        y = pizzetti.TransitModel(p, t, **kw).light_curve(p) + 415e-6 * np.random.default_rng(3).standard_normal(t.size)
        return pizzetti.MarginalTransitModel(p, t, y, 415e-6, bounds=BOX_T15, **kw), BOX_T15
    rp, b, sig = CASES[name]
    rng = np.random.default_rng(sum((i + 1) * ord(c) for i, c in enumerate(name)))
    t = np.concatenate([np.arange(-0.25, 0.25, 2.0 / 1440.0) + i * PER for i in range(7)])
    p = params(rp, A_TRUE, b, PER)
    y = pizzetti.TransitModel(p, t).light_curve(p) + rng.normal(0, sig, t.size)
    return pizzetti.MarginalTransitModel(p, t, y, sig, free=("t0", "rp", "a", "b"), bounds=BOX), BOX


def ln_volume(box):
    """ln of the prior volume, written out: boxes x integral over rp of (1 + rp) (a_hi - max(a_lo, 1 + rp))."""
    r_lo, r_hi = box["rp"]
    a_lo, a_hi = box["a"]
    kink = [a_lo - 1.0] if r_lo < a_lo - 1.0 < r_hi else None
    v = quad(lambda r: (1 + r) * (a_hi - max(a_lo, 1 + r)), r_lo, r_hi, points=kink)[0]
    return math.log(v) + sum(math.log(hi - lo) for q, (lo, hi) in box.items() if q not in ("rp", "a"))


def ln_z_from_integral(ln_int, ln_mean_w, m, box):
    """ln Z of the prior above from ln integral of exp(log_prob) and ln <f0 / c2^3>."""
    n = len(m.y)
    return (ln_int - 0.5 * (n - 3) * math.log(2 * math.pi) + np.sum(np.log(m.wts)) - ln_volume(box) + ln_mean_w
            + math.log(2.0) - math.log(F_HI - F_LO))


def run_nested(name, mode, seed, nlive):
    import dynesty
    from scipy.special import logsumexp
    m, box = build(name)
    free = list(m.free) if mode == "ns_marg" else list(m.free) + ["q1", "q2", "f0"]
    rhi = box["rp"][1]
    bounds = dict(box, b=(0.0, 1.0 + rhi), q1=(0.0, 1.0), q2=(0.0, 1.0), f0=(F_LO, F_HI))
    lo, hi = np.array([bounds[q][0] for q in free]), np.array([bounds[q][1] for q in free])
    jr, jb = free.index("rp"), free.index("b")
    ln_cube = float(np.sum(np.log(hi - lo)))
    cst = -0.5 * len(m.y) * math.log(2 * math.pi) + np.sum(np.log(m.wts))

    def ll_marg(x):
        if x[jb] >= 1 + x[jr]:
            return -1e300
        Xi = m.to_internal(x[None])
        v = float(m.log_prob_internal(Xi)[0]) - float(m._canonical_internal(Xi)[1][0])   # integrand in a, not T14
        return v if v > -1e299 else -1e300

    def ll_conv(x):
        if x[jb] >= 1 + x[jr]:
            return -1e300
        sq = math.sqrt(x[4])
        u1, u2 = 2 * sq * x[5], sq * (1 - 2 * x[5])
        b0, b1 = 1 - u1 - u2, 1 - 0.5 * u1
        c2 = x[6] / (b0 / 6 + b1 / 3 + 0.5)
        A = K.design(m.tt, m.k, m.wts, x[0], PER, x[1], x[2], x[3], 0.0, 90.0, m.tol, _A_HALF, _series._GTAB)
        d = m.yw - c2 * (A[:, 0] * b0 + A[:, 1] * b1 + A[:, 2])
        return cst - 0.5 * float(d @ d)

    s = dynesty.NestedSampler(ll_marg if mode == "ns_marg" else ll_conv, lambda u: lo + u * (hi - lo), len(free),
                              nlive=nlive, sample="rslice", slices=10 if mode == "ns_marg" else 14, bound="multi",
                              rstate=np.random.default_rng(seed))
    s.run_nested(dlogz=0.01, print_progress=False)
    r = s.results
    w = np.exp(r.logwt - logsumexp(r.logwt))
    X = r.samples
    if mode == "ns_marg":
        g = X[np.random.default_rng(seed).choice(len(X), 4000, p=w)]
        P = m._canonical_physical(g)
        T, S = K.theta_moments_batch(np.ascontiguousarray(P), m.tt, m.k, m.wts, m.yw, m.tol, _A_HALF, _series._GTAB)
        c = _draw_c_theta(T, S, np.random.default_rng(seed + 1))
        lw = np.log(c @ K.DISK) - 3.0 * np.log(c[:, 2])
        lmw = float(logsumexp(lw[np.isfinite(lw)]) - math.log(np.sum(np.isfinite(lw))))
        lnz = ln_z_from_integral(float(r.logz[-1]) + ln_cube, lmw, m, box)
    else:
        # the cube's geometry has L = 0 at b >= 1 + rp, so only its volume changes; q1, q2, f0 are already the prior
        lnz = float(r.logz[-1]) + ln_cube - math.log(F_HI - F_LO) - ln_volume(box)
    return dict(case=name, mode=mode, seed=seed, nlive=nlive, lnz=float(lnz), se=float(r.logzerr[-1]),
                ncall=int(np.sum(r.ncall)), grazing=float(np.sum(w[X[:, jb] + X[:, jr] > 1])))


def main(nproc):
    import dynesty
    jobs = []
    for c in CASES:
        jobs += [(c, md, s, 1000) for md in ("ns7", "ns_marg") for s in (1, 2)]
    jobs += [("t15", "ns_marg", s, 3000) for s in (1, 2, 3)]
    running, out = [], []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    while jobs or running:
        while jobs and len(running) < nproc:
            running.append(subprocess.Popen([sys.executable, __file__, "one", *map(str, jobs.pop(0))], env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
        for pr in [pr for pr in running if pr.poll() is not None]:
            o, _ = pr.communicate()
            out.append(json.loads(o.decode().strip().splitlines()[-1]))
            print(out[-1], flush=True)
            running.remove(pr)
        time.sleep(1)
    with open(PATH, "w") as f:
        json.dump(dict(F_LO=F_LO, F_HI=F_HI, dynesty=dynesty.__version__, runs=out), f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(json.dumps(run_nested(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
