"""Table (trap): accuracy and cost of the fixed-node limb rule and its interpolants against a
40-digit reference, and the Green's-theorem quadrature. Writes tables/trap.tex.
Run: python limb_table.py   (~5 min)"""
import os, sys, time, json, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import numpy as np
from reference import flux_ref
O = importlib.import_module("pizzetti.occult"); T = importlib.import_module("pizzetti._trap"); I = importlib.import_module("pizzetti._interp")
from pizzetti.laws import power_terms
LAWS = [("square-root", "squareroot", [0.3, 0.2]), ("four-parameter", "nonlinear", [0.5, 0.1, 0.1, -0.1]),
        (r"power-2, $\alpha=0.6$", "power2", [0.6, 0.6]), (r"power-2, $\alpha=1$", "power2", [0.6, 1.0])]
def best(f, r=9):
    f(); ts = []
    for _ in range(r):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
rows = []
for name, law, u in LAWS:
    al, c = power_terms(law, u); gams = al / 2
    for p in (0.0092, 0.1, 0.2):
        zc = O.series_cut(p, law, u)
        d = np.geomspace(1e-13, 1e-1, 8) * p
        zs = np.unique(np.concatenate([np.linspace(zc, 1 + p, 22)[1:-1], 1 - p + d, 1 - p - d, 1 + p - d]))
        zs = zs[(zs > zc) & (zs < 1 + p)]
        ref = np.array([flux_ref(z, p, al, c) for z in zs])
        err = {m: float(np.max(np.abs(O.occult(zs, p, u, law, limb=m) - ref))) for m in ("trapezoid", "interpolate")}
        zl = np.linspace(zc + 1e-9, 1 + p - 1e-9, 20000)
        z0 = np.linspace(zc + 1e-9, 1 + p - 1e-9, 2)
        per = {}
        for m in ("trapezoid", "interpolate", "quadrature"):
            n = 20000 if m != "quadrature" else 2000
            zz = np.linspace(zc + 1e-9, 1 + p - 1e-9, n)
            per[m] = (best(lambda: O.occult(zz, p, u, law, limb=m)) - best(lambda: O.occult(z0, p, u, law, limb="trapezoid"))) / n * 1e9
        # interpolant set-up
        x = 1 - (zc - p) ** 2; lam_lo = max(1 - x / (4 * zc * p), I.LAM_MIN, 0.3 * (1 - 0.25 / p**2))
        g = T.band_gamma(p, zc, 2 * max(gams)); W, tt, b = T.rules(gams); BW = T.rules_in(gams, g); CW = T.rules(gams, I.H_CB)
        setup = best(lambda: I.build(p, gams, lam_lo, W, tt, b, *BW, g, BW[4][BW[2].shape[0]], *CW)) * 1e6
        r = dict(law=name, p=p, err_trap=err["trapezoid"], err_interp=err["interpolate"], ns_trap=per["trapezoid"],
                 setup_us=setup, ns_interp=per["interpolate"], ns_quad=per["quadrature"], npts=len(zs))
        rows.append(r); print(r, flush=True)
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)


def e(v):
    if v < 1e-16:
        return r"$<10^{-16}$"
    m, ex = f"{v:.1e}".split("e")
    return rf"${m}\times10^{{{int(ex)}}}$"


with open(os.path.join(HERE, "tables", "trap.tex"), "w") as fh:
    for r in rows:
        fh.write(f"{r['law']} & {r['p']} & {e(r['err_trap'])} & {r['ns_trap']:.0f} & {e(r['err_interp'])} & "
                 f"{r['setup_us']:.0f} & {r['ns_interp']:.0f} & {r['ns_quad']:.0f} \\\\\n")
