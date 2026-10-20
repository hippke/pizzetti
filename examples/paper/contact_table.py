"""Table (contact): accuracy and cost of the fixed-order contact expansions and of the proven
pieces (tau = 1e-9, 1e-12) against a 40-digit reference, and the Green's-theorem quadrature.
Set-up and per-sample cost from two light-curve lengths. Writes tables/contact.tex.
Run: python contact_table.py   (~10 min)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import numpy as np
from reference import flux_ref
from pizzetti import occult, series_cut
from pizzetti.laws import power_terms

LAWS = [("square-root", "squareroot", [0.3, 0.2]), ("four-parameter", "nonlinear", [0.5, 0.1, 0.1, -0.1]),
        (r"power-2, $\alpha=0.6$", "power2", [0.6, 0.6]), (r"power-2, $\alpha=1$", "power2", [0.6, 1.0])]


def best(f, r=5):
    f(); ts = []
    for _ in range(r):
        s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
    return min(ts)


def cost(p, law, u, limb, tol):
    """(set-up ms, ns per sample) from light curves of 20 000 and 60 000 limb samples."""
    zc = series_cut(p, law, u, tol)
    t = []
    for n in (20000, 60000):
        z = np.linspace(zc + 1e-9, 1 + p - 1e-9, n)
        t.append(best(lambda: occult(z, p, u, law, series_tol=tol, limb=limb)))
    per = (t[1] - t[0]) / 40000
    return 1e3 * (t[0] - 20000 * per), 1e9 * per


def e(v):
    if v < 1e-16:
        return r"$<10^{-16}$"
    m, ex = f"{v:.1e}".split("e")
    return rf"${m}\times10^{{{int(ex)}}}$"


rows = []
for name, law, u in LAWS:
    al, c = power_terms(law, u)
    for p in (0.0092, 0.1, 0.2):
        cells = [name, str(p)]
        for limb, tol in (("contact", 1e-9), ("proven", 1e-9), ("proven", 1e-12)):
            zc = series_cut(p, law, u, tol)
            d = np.geomspace(1e-13, 1e-1, 8) * p
            zs = np.unique(np.concatenate([np.linspace(zc, 1 + p, 22)[1:-1], 1 - p + d, 1 - p - d, 1 + p - d]))
            zs = zs[(zs > zc) & (zs < 1 + p)]
            ref = np.array([flux_ref(x, p, al, c) for x in zs])
            # many samples so that the expansions are used (not the short-light-curve fallback)
            big = np.concatenate([zs, np.linspace(zc + 1e-9, 1 + p - 1e-9, 20000)])
            f = occult(big, p, u, law, series_tol=tol, limb=limb)[:zs.size]
            su, per = cost(p, law, u, limb, tol)
            cells += [e(np.max(np.abs(f - ref))), f"{su:.2f}" if limb == "contact" else f"{su:.0f}", f"{per:.0f}"]
        cells.append(f"{cost(p, law, u, 'quadrature', 1e-9)[1]:.0f}")
        # the paper's table: quadrature per sample only (no set-up column)
        rows.append(" & ".join(cells[:10] + cells[11:12]) + " \\\\")
        print(rows[-1], flush=True)
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
open(os.path.join(HERE, "tables", "contact.tex"), "w").write("\n".join(rows) + "\n")
