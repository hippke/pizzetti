"""Numerical validation for the paper. Writes text outputs to out/ and LaTeX
table bodies to tables/.  Run: python validate.py  (a few minutes)."""
import os
from fractions import Fraction as Fr

import mpmath as mp

from series import (alpha_table, bound_abel, bound_binomial, bound_pizzetti_remainder,
                    contact_mean_mu, disk_mean_ref, series_binomial, series_pizzetti)

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TAB = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def e(x, d=1):
    """LaTeX scientific notation"""
    x = float(x)
    if x == 0:
        return "0"
    s = f"{x:.{d}e}"
    m, ex = s.split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


# ------------------------------------------------------------ Table: error vs bound
def table_error_bound():
    lines, tex = [], []
    for k in (0.0092, 0.1, 0.2):
        for zf in ("0.5", "0.8", "0.9", "0.95", "0.99"):
            z = mp.mpf(zf) * (1 - k)
            ref = disk_mean_ref(z, k)
            err = series_binomial(z, k, 11) - ref
            b = bound_binomial(z, k, 11)
            errp = series_pizzetti(z, k, 5) - ref
            ba = bound_abel(z, k, 5)
            lines.append(f"k={k} z/(1-k)={zf} err_M11={mp.nstr(err, 4)} bound={mp.nstr(b, 4)} ratio={mp.nstr(b / err, 3)}"
                         f" | err_N5={mp.nstr(errp, 4)} abel={mp.nstr(ba, 4)} ratio={mp.nstr(ba / errp, 3)}")
            tex.append(f"{k} & {zf} & {e(k*k*err)} & {e(k*k*b)} & {float(b/err):.2f} & {e(k*k*errp)} & {e(k*k*ba)} & {float(ba/errp):.2f} \\\\")
    open(os.path.join(OUT, "error_bound.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(TAB, "error_bound.tex"), "w").write("\n".join(tex) + "\n")
    # the paper's Table (bound) shows the binomial columns only
    open(os.path.join(TAB, "error_bound_binomial.tex"), "w").write(
        "\n".join(" & ".join(r.split(" & ")[:5]) + " \\\\" for r in tex) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------ z_c and coverage
def zcut(k, M, tol_ldx):
    """largest z with k^2 * bound_binomial(z) <= tol (bound increases with z)"""
    f = lambda z: k * k * bound_binomial(z, k, M) - tol_ldx
    lo, hi = mp.mpf(0), mp.mpf(1 - k) * (1 - mp.mpf(10) ** -12)
    if f(lo) > 0:
        return mp.mpf(0)
    if f(hi) <= 0:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return lo


def coverage(k, zc, nb=400):
    """fraction of in-transit time (z < 1 + k) with z < zc, for b = 0 and averaged
    over impact parameters b uniform in [0, 1 + k) (straight chord)."""
    k = mp.mpf(k)
    f0 = zc / (1 + k)
    tot_t, tot_c = mp.mpf(0), mp.mpf(0)
    for i in range(nb):
        b = (i + mp.mpf(1) / 2) / nb * (1 + k)
        t = mp.sqrt((1 + k) ** 2 - b * b)
        c = mp.sqrt(zc * zc - b * b) if zc > b else 0
        tot_t += t
        tot_c += c
    return f0, tot_c / tot_t


def table_zcut():
    lines, tex = [], []
    ks = (0.005, 0.0092, 0.02, 0.05, 0.1, 0.15, 0.2)
    for tol in (1e-8, 1e-10):
        for M in (5, 11, 21):
            row = []
            for k in ks:
                zc = zcut(k, M, tol)
                row.append(zc / (1 - k))
            lines.append(f"tol={tol} M={M}: " + "  ".join(f"k={k}:{mp.nstr(r, 6)}" for k, r in zip(ks, row)))
            tex.append(f"$10^{{{round(float(mp.log10(tol)))}}}$ & {M} & " + " & ".join(f"{float(r):.5f}" for r in row) + " \\\\")
    open(os.path.join(OUT, "zcut.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(TAB, "zcut.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(lines))
    # coverage for M = 11, tol = 1e-8 (ldx)
    lines, tex = [], []
    for k in ks:
        zc = zcut(k, 11, 1e-8)
        f0, fb = coverage(k, zc)
        i0 = (1 - k) / (1 + k)           # interior fraction of a central transit
        _, ib = coverage(k, mp.mpf(1 - k))
        lines.append(f"k={k}: zc/(1-k)={mp.nstr(zc/(1-k), 6)} cov(b=0)={mp.nstr(f0, 4)} cov(b avg)={mp.nstr(fb, 4)}"
                     f" interior(b=0)={mp.nstr(i0, 4)} interior(b avg)={mp.nstr(ib, 4)}")
        tex.append(f"{k} & {float(zc/(1-k)):.4f} & {float(i0):.3f} & {float(f0):.3f} & {float(ib):.3f} & {float(fb):.3f} \\\\")
    open(os.path.join(OUT, "coverage.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(TAB, "coverage.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------ bound holds on dense grids
def check_dense():
    lines = []
    for k in (0.0092, 0.1, 0.2):
        for M in (5, 11):
            worst = mp.mpf(0)
            n = 60
            for i in range(n + 1):
                z = mp.mpf(i) / n * (1 - k) * (1 - mp.mpf(10) ** -3)
                with mp.workdps(80):
                    ref = disk_mean_ref(z, k, dps=80)
                    err = series_binomial(z, k, M) - ref
                    b = bound_binomial(z, k, M)
                if b < mp.mpf(10) ** -70:      # below the reference precision
                    continue
                assert err >= 0, (k, M, z, err)
                worst = max(worst, err / b)
            lines.append(f"k={k} M={M}: max err/bound over {n+1} z in [0, 0.999(1-k)] = {mp.nstr(worst, 10)} (all err >= 0; points with bound < 1e-70 skipped)")
    open(os.path.join(OUT, "dense.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------ other exponents
def table_powers():
    lines, tex = [], []
    for g in (Fr(1, 4), Fr(3, 10), Fr(3, 4), Fr(5, 4)):
        for k in (0.0092, 0.1):
            for zf in ("0.5", "0.9", "0.95"):
                z = mp.mpf(zf) * (1 - k)
                ref = disk_mean_ref(z, k, g)
                err = series_binomial(z, k, 11, g, alpha_table(11, g)) - ref
                b = bound_binomial(z, k, 11, g)
                ok = abs(err) <= b
                lines.append(f"g={g} k={k} z/(1-k)={zf}: err={mp.nstr(err, 4)} bound={mp.nstr(b, 4)} ok={ok}")
                tex.append(f"{float(g):.2f} & {k} & {zf} & {e(abs(err))} & {e(b)} & {float(b/abs(err)):.2f} \\\\")
    open(os.path.join(OUT, "powers.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(TAB, "powers.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(lines))


def check_lagrange():
    lines = []
    for k in (0.0092, 0.1):
        for N in (2, 5):
            for zf in ("0.3", "0.6", "0.85"):
                z = mp.mpf(zf) * (1 - k)
                rem = disk_mean_ref(z, k) - series_pizzetti(z, k, N)
                lo, hi = bound_pizzetti_remainder(z, k, N)
                lines.append(f"k={k} N={N} z/(1-k)={zf}: remainder={mp.nstr(rem, 4)} in [{mp.nstr(lo, 4)}, {mp.nstr(hi, 4)}]: {lo <= rem <= hi}")
    open(os.path.join(OUT, "lagrange.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check_contact():
    lines = [f"k={k}: closed form - quadrature = {mp.nstr(contact_mean_mu(k) - disk_mean_ref(1 - mp.mpf(k), k), 3)}"
             for k in (0.001, 0.0092, 0.1, 0.3, 0.5, 0.8)]
    open(os.path.join(OUT, "contact.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    check_contact()
    table_error_bound()
    check_lagrange()
    table_powers()
    check_dense()
    table_zcut()
