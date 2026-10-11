"""Language-neutral test vectors for all pizzetti implementations (Python, Julia, JAX).

The references are independent of the paper's derivations and of the pizzetti code:
the blocked integral P_gamma(z, p) = int (1 - r^2)^gamma dA over the occulted part
of the stellar disk, by tanh-sinh quadrature in polar coordinates about the occultor
centre (validation/refcheck.py): P_fast in double precision (~1e-16 relative) and,
for a subset, P_mp in 30-digit arithmetic (stored as strings).

    python testdata/make_testdata.py          # writes testdata/blocked_reference.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import refcheck  # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 0.05, 0.3, 0.6, 1.37]
PS = [0.0092, 0.02, 0.1, 0.2, 0.45, 0.9, 1.3]


def zgrid(p, rng):
    z = []
    z += list(np.linspace(0, max(1 - p, 0) * 0.999, 8)) if p < 1 else [0.0, 0.1, (p - 1) * 0.5]
    for c in (abs(1 - p), 1 + p, p):            # contacts and z = p
        for d in (-1e-3, -1e-7, -1e-12, 0.0, 1e-12, 1e-7, 1e-3):
            zz = c + d
            if 0 <= zz < 1 + p:
                z.append(zz)
    z += list(np.sort(rng.uniform(abs(1 - p), 1 + p, 8)))
    return sorted(set(float(x) for x in z if x >= 0 and x < 1 + p))


def main():
    rng = np.random.default_rng(20261011)
    cases = []
    t0 = time.time()
    for p in PS:
        zs = zgrid(p, rng)
        for g in GAMMAS:
            P = [refcheck.P_fast(z, p, g) for z in zs]
            cases.append(dict(p=p, gamma=g, z=zs, P=P))
    # 30-digit subset
    mp_cases = []
    for p in (0.02, 0.1, 0.2):
        zs = [0.0, 0.3, 1 - p - 1e-7, 1 - p + 1e-7, 1 - p / 3, 1.0, 1 + p - 1e-7, p, p + 1e-9]
        for g in (0.25, 0.5, 0.3, 1.37):
            P = [refcheck.mp.nstr(refcheck.P_mp(z, p, g, dps=30), 30) for z in zs]
            mp_cases.append(dict(p=p, gamma=g, z=zs, P=P))
    out = dict(
        description=("Blocked integral P_gamma(z, p) = integral of (1 - r^2)^gamma over the occulted part "
                     "of the unit stellar disk (no normalisation; P = pi p^2 D). Independent reference: "
                     "tanh-sinh quadrature of the area integral (pizzetti/validation/refcheck.py). "
                     "'double': P_fast, ~1e-16 relative. 'mp30': P_mp with 30 digits, as strings."),
        double=cases, mp30=mp_cases)
    json.dump(out, open(os.path.join(HERE, "blocked_reference.json"), "w"), indent=0)
    print("done in %.0f s, %d double cases, %d mp cases" % (time.time() - t0, len(cases), len(mp_cases)))


if __name__ == "__main__":
    main()
