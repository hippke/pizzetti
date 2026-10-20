"""Regenerate Figures 1 and 2 of the paper with the current pizzetti
(interior series up to z_c, contact expansions beyond). Reference: pizzetti's
Green's-theorem quadrature with the interior series switched off, checked
against 30-digit mpmath at selected points."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import batman

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
from reference import flux_ref  # noqa: E402
from pizzetti import occult, series_cut  # noqa: E402
from pizzetti.laws import power_terms  # noqa: E402

assert "batman_patched" not in batman.__file__
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.unicode_minus": False})
CB, CP = "#d55e00", "#0072b2"
P_EARTH, A_EARTH, PER = 0.00916, 215.0, 365.25
OUT = os.path.join(HERE, "figures")
TOL = 1e-9
FLOOR = 1e-16


def lg(e):
    return np.maximum(np.abs(e), FLOOR)


def ref_flux(z, p, u, law):
    return occult(z, p, u, law, series_tol=0.0, limb="quadrature")


def check_reference(z, p, u, law, n=8):
    al, c = power_terms(law, u)
    idx = np.linspace(0, z.size - 1, n).astype(int)
    r = ref_flux(z, p, u, law)
    return max(abs(r[i] - flux_ref(z[i], p, al, c, dps=30)) for i in idx)


def default_fac(law, u):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, PER, P_EARTH, A_EARTH, 90.0, 0.0, 90.0
    p.limb_dark, p.u = law, list(u)
    return batman.TransitModel(p, np.array([0.0])).fac


def fig1():
    import sys as _s
    _s.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
    import refcheck as rc
    law, u = "nonlinear", [0.60, -0.20, 0.30, -0.10]
    p = P_EARTH
    fac = default_fac(law, u)
    from batman import _nonlinear_ld
    zc = series_cut(p, law, u, TOL)
    grids = [np.linspace(0.0, 0.02, 2001), np.linspace(0.0, 2 * p, 2001), np.linspace(1 - 2 * p, 1.0, 4001)]
    titles = [r"$z\to0$", r"$z\to p$", r"$z\to1-p$"]
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.3), sharex="col",
                            gridspec_kw={"height_ratios": [1, 1.6]})
    stats = {}
    gam_cols = {0.25: "#009e73", 0.5: "#cc79a7", 0.75: "#e69f00", 1.0: "#56b4e9"}
    for col, (z, ti) in enumerate(zip(grids, titles)):
        ax = axs[0, col]
        zs = np.linspace(z[0], z[-1], 241)
        for g, colr in gam_cols.items():
            d = np.array([rc.P_fast(x, p, g) for x in zs]) / (np.pi * p * p)
            ax.plot(zs, d, color=colr, lw=1.0, label=rf"$\alpha={2 * g:g}$")
        ax.set_title(ti)
        ax.grid(alpha=0.3, lw=0.5)
        ax = axs[1, col]
        fb = _nonlinear_ld._nonlinear_ld(z, p, *u, fac, 1)
        fp = occult(z, p, u, law, series_tol=TOL)
        fr = ref_flux(z, p, u, law)
        ax.semilogy(z, lg(fb - fr), color=CB, lw=1.0)
        ax.semilogy(z, lg(fp - fr), color=CP, lw=1.0)
        ax.axhline(TOL, color="k", ls="--", lw=0.7)
        ax.set_xlabel(r"$z$ (stellar radii)")
        ax.set_xlim(z[0], z[-1])
        ax.set_ylim(5e-17, 3e-6)
        ax.grid(alpha=0.3, lw=0.5)
        if col > 0:
            ax.set_yticklabels([])
        stats[ti] = (np.max(np.abs(fb - fr)), np.max(np.abs(fp - fr)))
    for r in range(2):
        axs[r, 1].axvline(p, color="k", ls=":", lw=0.7)
        axs[r, 2].axvline(1 - p, color="k", ls=":", lw=0.7)
        axs[r, 2].axvline(zc, color="k", ls="-.", lw=0.7)
    axs[0, 0].set_ylabel(r"$P_\alpha/(\pi p^2)$")
    axs[0, 0].legend(fontsize=6.5, loc="lower left", frameon=False)
    axs[1, 0].set_ylabel(r"$|$error$|$ (flux units)")
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    fig.savefig(f"{OUT}/fig_polynomial_limits.pdf")
    z3 = grids[2]
    fb = _nonlinear_ld._nonlinear_ld(z3, p, *u, fac, 1)
    fp = occult(z3, p, u, law, series_tol=TOL)
    fr = ref_flux(z3, p, u, law)
    beyond = z3 > zc
    print("Fig 1: default fac", fac, "z_c", zc)
    for k, v in stats.items():
        print(f"  {k}: max batman err {v[0]:.2e}, max pizzetti err {v[1]:.2e}")
    print(f"  right panel: pizzetti err <= z_c {np.max(np.abs(fp - fr)[~beyond]):.2e}, beyond z_c {np.max(np.abs(fp - fr)[beyond]):.2e}")


def fig2():
    p = P_EARTH
    T14 = PER / np.pi * np.arcsin((1 + p) / A_EARTH)
    t = np.linspace(-0.6 * T14, 0.6 * T14, 6001)
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    for col, alpha in enumerate([0.5, 1.5]):
        u = [0.60, alpha]
        prm = batman.TransitParams()
        prm.t0, prm.per, prm.rp, prm.a, prm.inc, prm.ecc, prm.w = 0.0, PER, p, A_EARTH, 90.0, 0.0, 90.0
        prm.limb_dark, prm.u = "power2", u
        m = batman.TransitModel(prm, t) if alpha == 0.5 else batman.TransitModel(prm, t, fac=1e-3)
        fb = m.light_curve(prm)
        z = np.asarray(m.ds)
        fp = occult(z, p, u, "power2", series_tol=TOL)
        fr = ref_flux(z, p, u, "power2")
        th = t * 24
        ax = axs[0, col]
        ax.plot(th, (fb - 1) * 1e6, color=CB, lw=1.4)
        ax.plot(th, (fp - 1) * 1e6, color=CP, lw=1.2, ls="--")
        ax.set_ylim(-120, 10)
        ax.set_yticks([0, -50, -100])
        ax.set_title(rf"Power-2, $\alpha={alpha}$, $c=0.60$")
        ax = axs[1, col]
        ax.semilogy(th, lg(fb - fr), color=CB, lw=0.9)
        ax.semilogy(th, lg(fp - fr), color=CP, lw=0.9)
        ax.axhline(TOL, color="k", ls="--", lw=0.7)
        ax.set_ylim(5e-17, 3e-6)
        ax.set_xlabel("time from mid-transit (h)")
        for a in axs[:, col]:
            a.axvline(-T14 / 2 * 24, color="k", ls=":", lw=0.7)
            a.axvline(T14 / 2 * 24, color="k", ls=":", lw=0.7)
            a.grid(alpha=0.3, lw=0.5)
            a.set_xlim(th[0], th[-1])
            a.set_xticks([-6, 0, 6])
        print(f"Fig 2 alpha={alpha}: batman max err {np.max(np.abs(fb - fr)):.2e}, pizzetti max err {np.max(np.abs(fp - fr)):.2e}"
              f", fac={m.fac}; reference vs mpmath {check_reference(z[np.abs(z) < 1 + p], p, u, 'power2'):.1e}")
    axs[0, 0].set_ylabel(r"$F-1$ (ppm)")
    axs[1, 0].set_ylabel(r"$|$error$|$ (flux units)")
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    fig.savefig(f"{OUT}/fig_power2_lightcurves.pdf")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    import sys as _sys
    if "fig1" in _sys.argv or len(_sys.argv) == 1:
        fig1()
    if "fig2" in _sys.argv or len(_sys.argv) == 1:
        fig2()
