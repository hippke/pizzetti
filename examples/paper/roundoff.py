"""Round-off of the float64 Horner evaluation of the M = 11 truncation
(coefficients a_j(k) in float64) against the same truncation in mpmath."""
import numpy as np, mpmath as mp
from series import alpha_table, series_binomial
mp.mp.dps = 40
M = 11
A = alpha_table(M)
AL = np.array([[float(x) for x in r] for r in A])
out = []
for k in (0.001, 0.0092, 0.05, 0.1, 0.2, 0.5):
    a = np.array([np.polyval(AL[j][::-1], k * k) for j in range(M + 1)])
    worst = 0.0
    for z in np.linspace(0, (1 - k) * (1 - 1e-9), 400):
        w = 1 - z * z; v = 1 / w
        P = 0.0
        for j in range(M, -1, -1):
            P = P * v + a[j]
        f = np.sqrt(w) * P
        ex = series_binomial(z, k, M, A=A)
        worst = max(worst, abs(float(k * k * (f - ex))))
    out.append(f"k={k}: max |k^2 (float64 - exact truncation)| over [0, 1-k) = {worst:.2e}")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
open(os.path.join(HERE, "out", "roundoff.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
