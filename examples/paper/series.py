"""Interior transit series: exact coefficient tables, evaluation, error bounds
and an independent high-precision reference.

Notation: occultor radius k, centre distance z, interior z + k <= 1;
w = 1 - z^2, v = 1/w. Intensity term mu^(2g) = (1 - r^2)^g (g = 1/2: mu).
D(z, k; g) = <(1 - r^2)^g> = disk mean over the occultor.

Binomial expansion (Section 3 of the paper):
  (1 - r^2)^g = w^g (1 - eps)^g,  eps = (2 z rho cos(phi) + rho^2)/w
  <eps^m> = v^m sum_{l even <= m} G(m,l) z^l k^(2m-l),
  G(m,l) = binom(m,l) binom(l,l/2) 2/(2m-l+2) >= 0
  D = w^g sum_m beta_m <eps^m>,  beta_m = (-1)^m binom(g, m).
"""
from fractions import Fraction as Fr
from math import comb, factorial

import mpmath as mp


# ---------------------------------------------------------------- exact tables
def beta(g, m):
    """(-1)^m binom(g, m), exact (g a Fraction)."""
    r = Fr(1)
    for i in range(m):
        r *= (g - i)
    return (-1) ** m * r / factorial(m)


def G(m, l):
    return Fr(comb(m, l) * comb(l, l // 2) * 2, 2 * m - l + 2)


def alpha_table(M, g=Fr(1, 2)):
    """alpha[j][p]: the binomial truncation after m = M equals
    w^g * sum_j v^j sum_p alpha[j][p] k^(2p)   (j, p = 0..M)."""
    A = [[Fr(0)] * (M + 1) for _ in range(M + 1)]
    for m in range(0, M + 1):
        bm = beta(g, m)
        for l in range(0, m + 1, 2):
            c = bm * G(m, l)
            p = m - l // 2                       # k^(2m-l) = k^(2p)
            for i in range(l // 2 + 1):          # z^l = (1 - 1/v)^(l/2)
                A[m - i][p] += c * comb(l // 2, i) * (-1) ** i
    return A


def pizzetti_table(N, g=Fr(1, 2)):
    """c[n][j]: Pizzetti term n = (k^2/4)^n/(n!(n+1)!) Lap^n (1-r^2)^g at r = z
    equals w^g k^(2n) sum_j c[n][j] v^j  (j = n..2n).
    Uses Lap w^s = 4 s (s-1) w^(s-2) - 4 s^2 w^(s-1) (2-D, w = 1 - r^2)."""
    out = []
    lap = {g: Fr(1)}                             # Lap^0 w^g, as {exponent: coeff}
    for n in range(N + 1):
        pref = Fr(1, 4 ** n * factorial(n) * factorial(n + 1))
        row = {}
        for s, c in lap.items():                 # w^s = w^g v^(g - s)
            row[g - s] = row.get(g - s, 0) + pref * c
        out.append({int(j): c for j, c in row.items()})
        new = {}
        for s, c in lap.items():
            new[s - 2] = new.get(s - 2, 0) + c * 4 * s * (s - 1)
            new[s - 1] = new.get(s - 1, 0) - c * 4 * s * s
        lap = {s: c for s, c in new.items() if c != 0}
    return out


# ---------------------------------------------------------------- evaluation
def series_binomial(z, k, M, g=Fr(1, 2), A=None):
    """Truncated binomial series (after m = M), mpmath."""
    A = alpha_table(M, g) if A is None else A
    z = mp.mpf(z); k = mp.mpf(k)
    w = 1 - z * z; v = 1 / w; k2 = k * k
    P = mp.mpf(0)
    for j in range(M, -1, -1):
        a = mp.mpf(0)
        for p in range(M, -1, -1):
            a = a * k2 + mp.mpf(A[j][p].numerator) / A[j][p].denominator
        P = P * v + a
    return w ** mp.mpf(g) * P


def series_pizzetti(z, k, N, g=Fr(1, 2), C=None):
    """Pizzetti partial sum n = 0..N, mpmath."""
    C = pizzetti_table(N, g) if C is None else C
    z = mp.mpf(z); k = mp.mpf(k)
    w = 1 - z * z; v = 1 / w
    tot = mp.mpf(0)
    for n in range(N + 1):
        tot += k ** (2 * n) * sum(mp.mpf(c.numerator) / c.denominator * v ** j for j, c in C[n].items())
    return w ** mp.mpf(g) * tot


# ---------------------------------------------------------------- bounds
def eps_moment(m, z, k):
    z = mp.mpf(z); k = mp.mpf(k)
    w = 1 - z * z
    s = sum(mp.mpf(G(m, l).numerator) / G(m, l).denominator * z ** l * k ** (2 * m - l)
            for l in range(0, m + 1, 2))
    return s / w ** m


def bound_binomial(z, k, M, g=Fr(1, 2)):
    """|S_M - D| <= w^g |beta_{M+1}| <eps^(M+1)> / (1 - e), M + 1 even, M + 1 > g."""
    assert (M + 1) % 2 == 0 and M + 1 > g
    z = mp.mpf(z); k = mp.mpf(k)
    w = 1 - z * z
    e = (2 * z * k + k * k) / w
    b = abs(beta(g, M + 1))
    return w ** mp.mpf(g) * mp.mpf(b.numerator) / b.denominator * eps_moment(M + 1, z, k) / (1 - e)


def lap_n_at(r, n, g=Fr(1, 2)):
    """Lap^n (1 - r^2)^g at radius r (2-D)."""
    lap = {g: Fr(1)}
    for _ in range(n):
        new = {}
        for s, c in lap.items():
            new[s - 2] = new.get(s - 2, 0) + c * 4 * s * (s - 1)
            new[s - 1] = new.get(s - 1, 0) - c * 4 * s * s
        lap = {s: c for s, c in new.items() if c != 0}
    w = 1 - mp.mpf(r) ** 2
    return sum(mp.mpf(c.numerator) / c.denominator * w ** (mp.mpf(s.numerator) / s.denominator)
               for s, c in lap.items())


def bound_pizzetti_remainder(z, k, N, g=Fr(1, 2)):
    """Two-sided Lagrange remainder of the Pizzetti partial sum n = 0..N:
    D - S_N = c_{N+1} (Lap^{N+1} f)_xi, xi in the closed disk.
    Returns (lo, hi) of D - S_N (for 0 < g < 1: both <= 0)."""
    n = N + 1
    c = (mp.mpf(k) ** 2 / 4) ** n / (mp.factorial(n) * mp.factorial(n + 1))
    a = c * lap_n_at(mp.mpf(z) + mp.mpf(k), n, g)
    b = c * lap_n_at(abs(mp.mpf(z) - mp.mpf(k)), n, g)
    return (min(a, b), max(a, b))


def contact_mean_mu(k):
    """<mu> for a disk of radius k internally tangent to the limb (z = 1 - k), closed form."""
    k = mp.mpf(k)
    return (2 * mp.acos(1 - 2 * k) - mp.mpf(4) / 3 * (3 + 2 * k - 8 * k * k) * mp.sqrt(k * (1 - k))) / (3 * mp.pi * k * k)


def bound_abel(z, k, N):
    """Pizzetti partial sum S_N (g = 1/2): 0 <= S_N - <mu> <= (k/(1-z))^(2N+2) [S_N(z, 1-z) - <mu>(z, 1-z)]."""
    z = mp.mpf(z); k = mp.mpf(k)
    kc = 1 - z
    return (k / kc) ** (2 * N + 2) * (series_pizzetti(z, kc, N) - contact_mean_mu(kc))


# ---------------------------------------------------------------- reference
def disk_mean_ref(z, k, g=Fr(1, 2), dps=30):
    """D(z,k;g) by Green's theorem: (1/(pi k^2)) int_0^{2pi} F(r)/r^2 k (k + z cos t) dt,
    F(r) = int_0^r (1-s^2)^g s ds = (1 - (1-r^2)^(g+1)) / (2 (g+1)); adaptive mpmath."""
    with mp.workdps(dps):
        z = mp.mpf(z); k = mp.mpf(k); gg = mp.mpf(g.numerator) / g.denominator

        def f(t):
            r2 = z * z + k * k + 2 * z * k * mp.cos(t)
            if r2 == 0:
                h = mp.mpf(1) / 2
            else:
                h = (1 - max(1 - r2, 0) ** (gg + 1)) / (2 * (gg + 1) * r2)
            return h * (k + z * mp.cos(t))

        pts = [0] + [mp.pi * x for x in (mp.mpf(10) ** -6, mp.mpf(10) ** -4, mp.mpf(10) ** -2, mp.mpf(1) / 10, mp.mpf(1) / 2)] + [mp.pi]
        val = 2 * mp.quad(f, pts) / (mp.pi * k)
        return +val
