"""Compare the batman and pizzetti posteriors written by run_inference.py.

Writes results/corner_<case>.pdf, results/summary_<case>.json and prints a table.
Usage: python analyze.py <case> [burn]
"""
import json
import os
import sys

import numpy as np

import run_inference as R

LABELS = {
    "quadratic": [r"$t_0$", r"$P$", r"$p$", r"$a/R_\star$", r"$b$", r"$q_1$", r"$q_2$"],
    "power2": [r"$t_0$", r"$P$", r"$p$", r"$a/R_\star$", r"$b$", r"$c$", r"$\alpha$"],
}


def load(case, code):
    ch = np.load(os.path.join(R.OUT, f"chain_{case}_{code}.npy"))
    info = json.load(open(os.path.join(R.OUT, f"info_{case}_{code}.json")))
    return ch, info


def flat(ch, burn):
    return ch[burn:].reshape(-1, ch.shape[-1])


def model_difference(case, samples, nsel=200, seed=0):
    """Largest |batman - pizzetti| and |code - reference| over posterior samples."""
    c = R.CASES[case]
    t = R.make_time(c)
    out = {}
    pr = {code: R.build(code, c, t) for code in ("batman", "pizzetti")}
    import pizzetti
    ref_p, ref_m = R.build("pizzetti", c, t)
    ref_m = pizzetti.TransitModel(ref_p, t, series_tol=0.0, limb="quadrature")
    rng = np.random.default_rng(seed)
    sel = samples[rng.choice(len(samples), nsel, replace=False)]
    d = {"batman": 0.0, "pizzetti": 0.0, "between": 0.0}
    for th in sel:
        f = {}
        for code, (prm, mod) in pr.items():
            R.set_params(c, prm, th)
            f[code] = mod.light_curve(prm)
        R.set_params(c, ref_p, th)
        fr = ref_m.light_curve(ref_p)
        d["batman"] = max(d["batman"], float(np.max(np.abs(f["batman"] - fr))))
        d["pizzetti"] = max(d["pizzetti"], float(np.max(np.abs(f["pizzetti"] - fr))))
        d["between"] = max(d["between"], float(np.max(np.abs(f["batman"] - f["pizzetti"]))))
    out.update(d)
    return out


def main(case, burn=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    c = R.CASES[case]
    chb, ib = load(case, "batman")
    chp, ip = load(case, "pizzetti")
    sb, sp = flat(chb, burn), flat(chp, burn)
    names = R.NAMES
    rows = []
    # Monte Carlo error of a median: 1.2533 sigma / sqrt(N_eff), N_eff = walkers x steps / tau
    taub = np.array(ib["autocorr"])
    taup = np.array(ip["autocorr"])
    nkeep = (chb.shape[0] - burn) * chb.shape[1]
    for i, n in enumerate(names):
        qb = np.percentile(sb[:, i], [16, 50, 84])
        qp = np.percentile(sp[:, i], [16, 50, 84])
        sig = 0.5 * (qb[2] - qb[0])
        se = 1.2533 * sig * np.sqrt(taub[i] / nkeep + taup[i] / nkeep)     # of the difference
        rows.append(dict(name=n, batman=qb.tolist(), pizzetti=qp.tolist(),
                         dmed_sigma=float((qp[1] - qb[1]) / sig),
                         dmed_mcse=float((qp[1] - qb[1]) / se),
                         neff_batman=float(nkeep / taub[i]), neff_pizzetti=float(nkeep / taup[i]),
                         dwidth=float((qp[2] - qp[0]) / (qb[2] - qb[0]) - 1.0)))
    # identical proposals: how long do the two chains stay identical?
    same = np.all(np.abs(chb - chp) <= 1e-12 * (1 + np.abs(chb)), axis=(1, 2))
    first_diff = int(np.argmin(same)) if not np.all(same) else int(chb.shape[0])
    md = model_difference(case, sb)
    summ = dict(case=case, label=c["label"], burn=burn, rows=rows, steps_identical=first_diff,
                model_diff=md,
                timing=dict(batman_call_ms=ib["call_ms_median"], pizzetti_call_ms=ip["call_ms_median"],
                            batman_wall_s=ib["wall_s"], pizzetti_wall_s=ip["wall_s"],
                            n_points=ib["n_points"], nsteps=ib["nsteps"], nwalkers=ib["nwalkers"]),
                autocorr=dict(batman=ib["autocorr"], pizzetti=ip["autocorr"]))
    json.dump(summ, open(os.path.join(R.OUT, f"summary_{case}.json"), "w"), indent=1)

    th0 = R.theta_true(c)
    # plot offsets from the truth so that the tick labels stay readable
    lab = LABELS[c["law"]]
    lab = [l if i >= 2 else r"$\Delta$" + l for i, l in enumerate(lab)]
    off = np.zeros_like(th0)
    off[:2] = th0[:2]
    rng_ = [(np.percentile(np.r_[sb[:, i], sp[:, i]] - off[i], 0.3), np.percentile(np.r_[sb[:, i], sp[:, i]] - off[i], 99.7))
            for i in range(len(th0))]
    kw = dict(labels=lab, bins=40, smooth=1.0, levels=(0.393, 0.865, 0.989), plot_datapoints=False, range=rng_,
              plot_density=False, fill_contours=False, truths=th0 - off, truth_color="0.6",
              max_n_ticks=3, label_kwargs=dict(fontsize=13))
    fig = corner.corner(sb - off, color="C0", hist_kwargs=dict(lw=2.5, color="C0"),
                        contour_kwargs=dict(linewidths=2.5), **kw)
    corner.corner(sp - off, fig=fig, color="C3", hist_kwargs=dict(lw=1.2, ls="--", color="C3"),
                  contour_kwargs=dict(linewidths=1.2, linestyles="--"), **kw)
    fig.legend(handles=[plt.Line2D([], [], color="C0", lw=2.5, label="batman + emcee"),
                        plt.Line2D([], [], color="C3", lw=1.2, ls="--", label="pizzetti + emcee")],
               loc="upper right", fontsize=14, frameon=False)
    fig.suptitle(c["label"], x=0.62, y=0.86, fontsize=14, ha="left")
    fig.savefig(os.path.join(R.OUT, f"corner_{case}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(R.OUT, f"corner_{case}.png"), dpi=110, bbox_inches="tight")

    # compact version for the paper: radius ratio, a, b and the two limb-darkening parameters
    sel = [2, 3, 4, 5, 6]
    kw2 = dict(kw)
    kw2["labels"] = [lab[i] for i in sel]
    kw2["truths"] = (th0 - off)[sel]
    kw2["range"] = [rng_[i] for i in sel]
    kw2["label_kwargs"] = dict(fontsize=16)
    kw2["levels"] = (0.393, 0.865)
    fig = corner.corner(sb[:, sel], color="C0", hist_kwargs=dict(lw=2.5, color="C0"),
                        contour_kwargs=dict(linewidths=2.5), **kw2)
    corner.corner(sp[:, sel], fig=fig, color="C3", hist_kwargs=dict(lw=1.2, ls="--", color="C3"),
                  contour_kwargs=dict(linewidths=1.2, linestyles="--"), **kw2)
    fig.legend(handles=[plt.Line2D([], [], color="C0", lw=2.5, label="batman"),
                        plt.Line2D([], [], color="C3", lw=1.2, ls="--", label="pizzetti")],
               loc="upper right", fontsize=17, frameon=False, title=c["label"].replace(", ", "\n", 1),
               title_fontsize=15, alignment="right")
    for ax in fig.get_axes():
        ax.tick_params(labelsize=11)
    fig.savefig(os.path.join(R.OUT, f"corner_{case}_paper.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(R.OUT, f"corner_{case}_paper.png"), dpi=90, bbox_inches="tight")

    print(c["label"])
    print(f"{'param':6s} {'batman median':>16s} {'pizzetti median':>16s} {'dmed/sigma':>11s} {'dmed/MCSE':>10s} {'dwidth':>8s}")
    for r in rows:
        print(f"{r['name']:6s} {r['batman'][1]:16.8g} {r['pizzetti'][1]:16.8g} {r['dmed_sigma']:11.3f} "
              f"{r['dmed_mcse']:10.2f} {r['dwidth']:8.3f}")
    print("chains identical for", first_diff, "of", chb.shape[0], "steps")
    print("max |model - reference| over posterior samples:", md)
    print("timing:", summ["timing"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
