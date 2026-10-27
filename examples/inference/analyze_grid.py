"""Summarise the inference grid of run_grid.py: posterior agreement, model errors and the
time budget (total, light curve, overhead) of every run; corner plots for two cases.

Usage: python analyze_grid.py [burn]
Writes results/grid_summary.json, results/grid_table.tex, results/corner_<case>_paper.pdf.
"""
import json
import os
import sys

import numpy as np

import run_inference as R
import run_grid as G

LABELS = {
    "quadratic": [r"$t_0$", r"$P$", r"$p$", r"$a/R_\star$", r"$b$", r"$q_1$", r"$q_2$"],
    "power2": [r"$t_0$", r"$P$", r"$p$", r"$a/R_\star$", r"$b$", r"$c$", r"$\alpha$"],
}


def load(case, code):
    ch = np.load(os.path.join(R.OUT, f"gridchain_{case}_{code}.npy"))
    info = json.load(open(os.path.join(R.OUT, f"grid_{case}_{code}.json")))
    return ch, info


def model_difference(case, samples, nsel=200, seed=0):
    c = G.CASES[case]
    t = R.make_time(c)
    import pizzetti
    pr = {code: R.build(code, c, t) for code in ("batman", "pizzetti")}
    ref_p, _ = R.build("pizzetti", c, t)
    ref_m = pizzetti.TransitModel(ref_p, t, series_tol=0.0, limb="quadrature")
    rng = np.random.default_rng(seed)
    sel = samples[rng.choice(len(samples), nsel, replace=False)]
    d = {"batman": 0.0, "pizzetti": 0.0}
    for th in sel:
        thf = G.expand(c, th)
        R.set_params(c, ref_p, thf)
        fr = ref_m.light_curve(ref_p)
        for code, (prm, mod) in pr.items():
            R.set_params(c, prm, thf)
            d[code] = max(d[code], float(np.max(np.abs(mod.light_curve(prm) - fr))))
    return d


def summarise(case, burn):
    chb, ib = load(case, "batman")
    chp, ip = load(case, "pizzetti")
    sb = chb[burn:].reshape(-1, chb.shape[-1])
    sp = chp[burn:].reshape(-1, chp.shape[-1])
    taub, taup = np.array(ib["autocorr"]), np.array(ip["autocorr"])
    nkeep = (chb.shape[0] - burn) * chb.shape[1]
    dmed, dmcse, dwidth = [], [], []
    for i in range(sb.shape[1]):
        qb = np.percentile(sb[:, i], [16, 50, 84])
        qp = np.percentile(sp[:, i], [16, 50, 84])
        sig = 0.5 * (qb[2] - qb[0])
        se = 1.2533 * sig * np.sqrt(taub[i] / nkeep + taup[i] / nkeep)
        dmed.append(abs(qp[1] - qb[1]) / sig)
        dmcse.append(abs(qp[1] - qb[1]) / se)
        dwidth.append(abs((qp[2] - qp[0]) / (qb[2] - qb[0]) - 1.0))
    same = np.all(np.abs(chb - chp) <= 1e-12 * (1 + np.abs(chb)), axis=(1, 2))
    first_diff = int(np.argmin(same)) if not np.all(same) else int(chb.shape[0])
    md = model_difference(case, sb)
    return dict(case=case, label=G.CASES[case]["label"], law=ib["law"], fixed_alpha=ib["fixed_alpha"],
                n_points=ib["n_points"], n_in_transit=ib["n_in_transit"], ndim=ib["ndim"], ncalls=ib["ncalls"],
                dmed_sigma_max=max(dmed), dmed_mcse_max=max(dmcse), dwidth_max=max(dwidth),
                neff_min=float(nkeep / max(taub.max(), taup.max())), steps_identical=first_diff,
                model_err=md,
                batman=dict(wall=ib["wall_s"], lc=ib["lightcurve_s"], overhead=ib["overhead_s"], emcee=ib["emcee_s"],
                            per_call=ib["per_call_total_us"], per_call_lc=ib["per_call_lc_us"]),
                pizzetti=dict(wall=ip["wall_s"], lc=ip["lightcurve_s"], overhead=ip["overhead_s"], emcee=ip["emcee_s"],
                              per_call=ip["per_call_total_us"], per_call_lc=ip["per_call_lc_us"]))


def corner_plot(case, burn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    c = G.CASES[case]
    chb, _ = load(case, "batman")
    chp, _ = load(case, "pizzetti")
    sb = chb[burn:].reshape(-1, chb.shape[-1])
    sp = chp[burn:].reshape(-1, chp.shape[-1])
    th0 = R.theta_true(c)[:sb.shape[1]]
    lab = LABELS[c["law"]][:sb.shape[1]]
    off = np.zeros_like(th0)
    off[:2] = th0[:2]
    sel = list(range(2, sb.shape[1]))
    rng_ = [(np.percentile(np.r_[sb[:, i], sp[:, i]] - off[i], 0.3), np.percentile(np.r_[sb[:, i], sp[:, i]] - off[i], 99.7))
            for i in sel]
    kw = dict(labels=[lab[i] for i in sel], bins=40, smooth=1.0, levels=(0.393, 0.865), plot_datapoints=False,
              range=rng_, plot_density=False, fill_contours=False, truths=(th0 - off)[sel], truth_color="0.6",
              max_n_ticks=3, label_kwargs=dict(fontsize=16))
    fig = corner.corner(sb[:, sel], color="C0", hist_kwargs=dict(lw=2.5, color="C0"),
                        contour_kwargs=dict(linewidths=2.5), **kw)
    corner.corner(sp[:, sel], fig=fig, color="C3", hist_kwargs=dict(lw=1.2, ls="--", color="C3"),
                  contour_kwargs=dict(linewidths=1.2, linestyles="--"), **kw)
    title = c["label"].replace(", ", "\n", 1)
    fig.legend(handles=[plt.Line2D([], [], color="C0", lw=2.5, label="batman"),
                        plt.Line2D([], [], color="C3", lw=1.2, ls="--", label="pizzetti")],
               loc="upper right", fontsize=17, frameon=False, title=title, title_fontsize=15, alignment="right")
    for ax in fig.get_axes():
        ax.tick_params(labelsize=11)
    fig.savefig(os.path.join(R.OUT, f"corner_{case}_paper.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(R.OUT, f"corner_{case}_paper.png"), dpi=90, bbox_inches="tight")


def fmt_e(v):
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def main(burn=4000):
    rows = []
    for case in G.CASES:
        if not os.path.exists(os.path.join(R.OUT, f"grid_{case}_pizzetti.json")):
            continue
        rows.append(summarise(case, burn))
        print(json.dumps(rows[-1]))
    json.dump(rows, open(os.path.join(R.OUT, "grid_summary.json"), "w"), indent=1)
    # LaTeX table body: one line per case
    lines = []
    for r in rows:
        b, p = r["batman"], r["pizzetti"]
        name = {"jup_quad": "Hot Jupiter, quadratic", "jup_fixed": r"Hot Jupiter, power-2, $\alpha$ fixed",
                "jup_free": r"Hot Jupiter, power-2, $\alpha$ free", "se_fixed": r"Super-Earth, power-2, $\alpha$ fixed",
                "se_free": r"Super-Earth, power-2, $\alpha$ free"}[r["case"]]
        lines.append(" & ".join([
            name, str(r["ndim"]),
            f"{b['wall']:.0f}", f"{b['lc']:.0f}", f"{b['overhead']:.0f}",
            f"{p['wall']:.1f}", f"{p['lc']:.1f}", f"{p['overhead']:.1f}",
            f"{b['wall'] / p['wall']:.1f}", f"{b['lc'] / p['lc']:.1f}",
            f"{r['dmed_sigma_max']:.2f}", f"{100 * r['dwidth_max']:.0f}\\,\\%",
            fmt_e(r["model_err"]["batman"]), fmt_e(r["model_err"]["pizzetti"])]) + r" \\")
    open(os.path.join(R.OUT, "grid_table.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for case in ("jup_quad", "se_free"):
        if os.path.exists(os.path.join(R.OUT, f"grid_{case}_pizzetti.json")):
            corner_plot(case, burn)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
