"""Cross-implementation vectors: light curves of the Python package (orbit, laws, limb
modes), to check that other implementations (Julia, JAX) reproduce them. These are
consistency checks, not independent references (see make_testdata.py for those).

    python testdata/make_crosscheck.py      # writes testdata/python_lightcurves.json
"""
import json
import os

import numpy as np

import pizzetti

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    dict(t0=0.0, per=3.5, rp=0.1, a=9.0, inc=87.8, ecc=0.0, w=90.0, limb_dark="quadratic", u=[0.4, 0.25]),
    dict(t0=0.1, per=10.0, rp=0.02, a=20.0, inc=89.4, ecc=0.0, w=90.0, limb_dark="power2", u=[0.6, 0.6]),
    dict(t0=0.0, per=5.0, rp=0.08, a=12.0, inc=88.5, ecc=0.3, w=40.0, limb_dark="nonlinear", u=[0.5, 0.1, 0.2, -0.1]),
    dict(t0=0.0, per=2.0, rp=0.15, a=6.0, inc=85.0, ecc=0.1, w=200.0, limb_dark="squareroot", u=[0.3, 0.4]),
    dict(t0=0.0, per=1.2, rp=0.3, a=4.0, inc=86.0, ecc=0.0, w=90.0, limb_dark="power2", u=[0.7, 0.4]),
    dict(t0=0.0, per=365.25, rp=0.00916, a=215.0, inc=90.0, ecc=0.0167, w=102.9, limb_dark="nonlinear",
         u=[0.6, -0.2, 0.3, -0.1]),
]


def main():
    out = []
    for c in CASES:
        prm = pizzetti.TransitParams()
        for k, v in c.items():
            setattr(prm, k, v)
        # times across the transit
        dur = c["per"] / np.pi / c["a"] * 1.3
        t = np.linspace(c["t0"] - dur, c["t0"] + dur, 801)
        m = pizzetti.TransitModel(prm, t, series_tol=0.0, limb="quadrature")
        z = m.ds
        f = m.light_curve(prm)
        out.append(dict(params=c, t=t.tolist(), z=z.tolist(), flux=f.tolist()))
    json.dump(dict(description="Python pizzetti light curves (series off, quadrature limb) with rsky separations",
                   cases=out), open(os.path.join(HERE, "python_lightcurves.json"), "w"))
    print("wrote", len(out), "cases")


if __name__ == "__main__":
    main()
