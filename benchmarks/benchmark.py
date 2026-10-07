"""Benchmark pizzetti against batman with the identical API.

Usage: python benchmarks/benchmark.py [--n 100000]

Two scenarios per law and radius ratio:
  * "flux only": the orbit is unchanged between calls (only rp / u change)
  * "full": every call changes the orbit (inc), as in a fit
"""
import argparse
import time

import numpy as np

import pizzetti

try:
    import batman
except ImportError:  # pragma: no cover
    batman = None


def best(f, repeat=7):
    f()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def make(mod, rp, law, u):
    p = mod.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 10.0, rp, 20.0, 89.5, 0.0, 90.0
    p.u, p.limb_dark = list(u), law
    return p


def run(n):
    t = np.linspace(-0.25, 0.25, n)
    rows = []
    for law, u in (("quadratic", [0.4, 0.25]), ("nonlinear", [0.5, 0.1, 0.1, -0.1]), ("power2", [0.7, 0.6])):
        for rp in (0.0092, 0.1):
            res = {}
            for name, mod in (("batman", batman), ("pizzetti", pizzetti)):
                if mod is None:
                    continue
                p = make(mod, rp, law, u)
                m = mod.TransitModel(p, t)
                flux_only = best(lambda: m.light_curve(p))
                state = {"i": 0}

                def full():
                    state["i"] += 1
                    p.inc = 89.5 + 1e-7 * (state["i"] % 2)
                    return m.light_curve(p)

                res[name] = (flux_only, best(full))
            rows.append((law, rp, res))
    print(f"\n{n} samples, transit duty cycle ~30 %\n")
    print("| law | rp | batman flux only [ms] | pizzetti flux only [ms] | speed-up | batman full [ms] | pizzetti full [ms] | speed-up |")
    print("|---|---|---|---|---|---|---|---|")
    for law, rp, res in rows:
        pz = res["pizzetti"]
        if "batman" in res:
            bt = res["batman"]
            print(f"| {law} | {rp} | {bt[0]*1e3:.3f} | {pz[0]*1e3:.3f} | {bt[0]/pz[0]:.1f} | "
                  f"{bt[1]*1e3:.3f} | {pz[1]*1e3:.3f} | {bt[1]/pz[1]:.1f} |")
        else:
            print(f"| {law} | {rp} | – | {pz[0]*1e3:.3f} | – | – | {pz[1]*1e3:.3f} | – |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    run(ap.parse_args().n)
