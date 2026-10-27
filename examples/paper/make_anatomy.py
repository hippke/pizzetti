"""Figure (anatomy): the components of a light curve and the evaluation strategy in the
(z, p) plane, for the power-2 law with c = 0.6 and alpha = 0.5 / 1.5 (default settings:
tau = 1e-9, p < 0.3 for the fixed-node rule). Writes figures/fig_anatomy.pdf."""
import importlib
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pizzetti import occult
from pizzetti.laws import power_terms
O = importlib.import_module("pizzetti.occult")
T = importlib.import_module("pizzetti._trap")
S = importlib.import_module("pizzetti._series")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
TAU = 1e-9
C = 0.6
# method codes and colours
OUTSIDE, SERIES, BARTKY, QUAD = 0, 1, 2, 3
COL = {OUTSIDE: "#f2f2f2", SERIES: "#4477aa", BARTKY: "#ee8866", QUAD: "#999933"}
LAB = {SERIES: r"interior series $S_M$, error $\leq\tau$ proven", BARTKY: "fixed-node limb rule in Bartky's variable, or its interpolants",
       QUAD: "Green's-theorem quadrature", OUTSIDE: "no overlap, $F=1$"}


def zcut(p, alpha):
    al, c = power_terms("power2", [C, alpha])
    return O._cut_terms(p, al / 2, c, TAU, S._GTAB)


def method_map(alpha, zs, ps):
    m = np.full((ps.size, zs.size), OUTSIDE, dtype=int)
    for i, p in enumerate(ps):
        zc = zcut(p, alpha)
        limb = (zs > zc) & (zs < 1 + p)
        m[i, zs <= zc] = SERIES
        if p < T.P_MAX:
            m[i, limb & (zs > p)] = BARTKY
            m[i, limb & (zs <= p)] = QUAD
        else:
            m[i, limb] = QUAD
    return m


def main():
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), gridspec_kw=dict(height_ratios=[1, 1.25]))
    p = 0.1
    z = np.linspace(0, 1.25, 2501)
    for j, alpha in enumerate((0.5, 1.5)):
        ax = axes[0, j]
        f = occult(z, p, [C, alpha], "power2")
        zc = zcut(p, alpha)
        segs = [(0, zc, SERIES), (zc, 1 + p, BARTKY), (1 + p, z[-1], OUTSIDE)]
        for lo, hi, code in segs:
            sel = (z >= lo) & (z <= hi)
            ax.plot(z[sel], 1e6 * (f[sel] - 1), color=COL[code] if code != OUTSIDE else "0.55", lw=2.2)
            ax.axvspan(lo, hi, color=COL[code], alpha=0.25 if code != OUTSIDE else 0.6, lw=0)
        ymin = 1e6 * (f.min() - 1)
        for xv, lab in ((zc, r"$z_c$"), (1 - p, r"$1-p$"), (1 + p, r"$1+p$")):
            ax.axvline(xv, color="0.3", lw=0.7, ls=":")
            ax.text(xv + 0.008, ymin * 1.04, lab, ha="left", va="bottom", fontsize=7.5, rotation=90)
        ax.set(xlim=(0, z[-1]), xlabel=r"separation $z$ (stellar radii)",
               title=rf"$p={p}$, $c={C}$, $\alpha={alpha}$", ylim=(ymin * 1.08, 900))
        if j == 0:
            ax.set_ylabel(r"$F-1$ (ppm)")
        # annotate regimes
        ax.text(zc / 2, ymin * 0.45, "interior", ha="center", va="center", fontsize=8)
        ax.text((zc + 1 - p) / 2, ymin * 0.3, "band", ha="center", fontsize=7.5, rotation=90, va="center")
        ax.text(0.985, ymin * 0.45, "limb\ncrossing", ha="center", va="center", fontsize=7.5)
        ax.text(1.185, ymin * 0.45, "no\noverlap", ha="center", va="center", fontsize=7.5)

    zs = np.linspace(0, 1.5, 751)
    ps = np.linspace(0.005, 0.5, 200)
    cmap = ListedColormap([COL[k] for k in (OUTSIDE, SERIES, BARTKY, QUAD)])
    for j, alpha in enumerate((0.5, 1.5)):
        ax = axes[1, j]
        m = method_map(alpha, zs, ps)
        ax.pcolormesh(zs, ps, m, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True)
        ax.plot(1 - ps, ps, "k", lw=0.8)
        ax.plot(1 + ps, ps, "k", lw=0.8)
        ax.plot(ps, ps, "k", lw=0.8, ls="--")
        zcs = np.array([zcut(pp, alpha) for pp in ps])
        ax.plot(zcs, ps, "k", lw=0.8, ls=":")
        ax.axhline(T.P_MAX, color="k", lw=0.6, ls="-.")
        ax.text(0.86, 0.165, r"$z=1-p$", fontsize=7.5, rotation=-60, ha="center", va="center")
        ax.text(1.22, 0.195, r"$z=1+p$", fontsize=7.5, rotation=61, ha="center", va="center")
        ax.text(0.665, 0.13, r"$z_c(p)$", fontsize=7.5, rotation=-50, ha="center", va="center")
        ax.text(0.445, 0.405, r"$z=p$", fontsize=7.5, rotation=61, ha="center", va="center")
        ax.text(1.49, 0.31, r"$p=0.3$", fontsize=7.5, ha="right", va="bottom")
        ax.set(xlim=(0, 1.5), ylim=(0, 0.5), xlabel=r"separation $z$ (stellar radii)", title=rf"$\alpha={alpha}$")
        if j == 0:
            ax.set_ylabel(r"radius ratio $p$")
    handles = [Patch(facecolor=COL[k], label=LAB[k]) for k in (SERIES, BARTKY, QUAD, OUTSIDE)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_anatomy.pdf"))
    fig.savefig(os.path.join(OUT, "fig_anatomy.png"), dpi=130)
    for alpha in (0.5, 1.5):
        print(f"alpha={alpha}: z_c(p=0.1) = {zcut(0.1, alpha):.4f}")


if __name__ == "__main__":
    main()
