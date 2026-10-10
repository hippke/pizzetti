"""Independent references and the paper's formulas, for the validation notebooks.

Two references, both straight from the definition of the blocked integral

    P_gamma(z, p) = integral over {rho <= p, r < 1} of (1 - r^2)^gamma dA,

in polar coordinates (rho, phi) about the occulter centre, with
r^2 = z^2 + rho^2 + 2 z rho cos(phi):

  P_fast(z, p, g)   NumPy, double precision, tanh-sinh (double-exponential)
                    quadrature in both variables; ~1e-16 relative, ~10 ms.
  P_mp(z, p, g)     mpmath, arbitrary precision, adaptive tanh-sinh; slow.

Neither uses the paper's derivations or pizzetti: no series, no Green's
theorem, no hypergeometric functions. The rho-range is split where the
circle of radius rho about the occulter centre touches the limb
(rho = |1 - z| and 1 + z), so that every integrand is smooth inside each
interval and has at most an algebraic endpoint singularity, which
tanh-sinh quadrature handles.

The second half of the module transcribes the paper's equations (labels as
in paper.tex) into mpmath.
"""
import mpmath as mp
import numpy as np

# ------------------------------------------------------------ fast reference
_H = 1.0 / 40.0
with mp.workdps(40):                               # nodes and weights in high precision, then rounded once
    _t = [mp.mpf(k) * mp.mpf(1) / 40 for k in range(-160, 161)]
    _u = [mp.pi / 2 * mp.sinh(t) for t in _t]
    _W = np.array([float(mp.pi / 2 * mp.cosh(t) / mp.cosh(u) ** 2 / 40) for t, u in zip(_t, _u)])
    _DL = np.array([float(1 / (mp.exp(-2 * u) + 1)) for u in _u])     # (1 + x)/2, distance from the left end
    _DR = np.array([float(1 / (mp.exp(2 * u) + 1)) for u in _u])      # (1 - x)/2, distance from the right end
_keep = (_DL > 0) & (_DR > 0) & (_W > 1e-300)
_DL, _DR, _W = _DL[_keep], _DR[_keep], _W[_keep]


def _ts(a, b):
    """tanh-sinh nodes on [a, b]: (distance from a, distance from b, weights)."""
    L = b - a
    return L * _DL, L * _DR, 0.5 * L * _W


def P_fast(z, p, g):
    z = float(z)
    p = float(p)
    g = float(g)
    pts = [0.0] + [s for s in (abs(1 - z), 1 + z) if 0 < s < p] + [p]
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        da, db, wr = _ts(a, b)
        rho = a + da
        if z == 0.0:
            q = (1 - rho) * (1 + rho)
            vals = np.where(q > 0, 2 * np.pi * rho * np.abs(q) ** g, 0.0)
            total += np.sum(wr * vals)
            continue
        c = (1 - z * z - rho * rho) / (2 * z * rho)
        inner = np.zeros_like(rho)
        full = c >= 1                                # circle entirely on the star
        part = (c > -1) & ~full
        dphi_a, dphi_b, wphi = _ts(0.0, np.pi)
        if full.any():
            r = rho[full][:, None]
            phi = dphi_a[None, :]
            q = (1 - z - r) * (1 + z + r) + 4 * z * r * np.sin(phi / 2) ** 2
            inner[full] = 2 * np.sum(wphi[None, :] * np.maximum(q, 0.0) ** g, axis=1)
        if part.any():
            r = rho[part]
            phi0 = np.arccos(c[part])
            span = np.pi - phi0
            d = span[:, None] * _DL[None, :]         # phi - phi0, without cancellation
            w = 0.5 * span[:, None] * _W[None, :]
            q = 4 * z * r[:, None] * np.sin(phi0[:, None] + d / 2) * np.sin(d / 2)
            inner[part] = 2 * np.sum(w * np.where(q > 0, q, 0.0) ** g, axis=1)
        total += np.sum(wr * rho * inner)
    return total


def D_fast(z, p, g):
    return P_fast(z, p, g) / (np.pi * p * p)


# ------------------------------------------------------------ mpmath reference
def P_mp(z, p, g, dps=30):
    with mp.workdps(dps + 10):
        z = mp.mpf(z)
        p = mp.mpf(p)
        g = mp.mpf(g)

        def inner(rho):
            if rho == 0:
                return mp.mpf(0)
            if z == 0:
                return 2 * mp.pi * rho * (1 - rho * rho) ** g if rho < 1 else mp.mpf(0)
            c = (1 - z * z - rho * rho) / (2 * z * rho)
            if c <= -1:
                return mp.mpf(0)
            phi0 = mp.mpf(0) if c >= 1 else mp.acos(c)

            def f(phi):
                q = 1 - z * z - rho * rho - 2 * z * rho * mp.cos(phi)
                return q ** g if q > 0 else mp.mpf(0)
            return 2 * rho * mp.quad(f, [phi0, mp.pi])

        pts = [mp.mpf(0)] + [s for s in (abs(1 - z), 1 + z) if 0 < s < p] + [p]
        return +mp.quad(inner, sorted(set(pts)))


def D_mp(z, p, g, dps=30):
    with mp.workdps(dps + 10):
        return P_mp(z, p, g, dps) / (mp.pi * mp.mpf(p) ** 2)


# ------------------------------------------------------------ the paper's equations
def beta(g, m):                                   # eq:beta
    return (-1) ** m * mp.binomial(g, m)


def G(m, l):                                      # eq:moments
    return mp.binomial(m, l) * mp.binomial(l, l // 2) * mp.mpf(2) / (2 * m - l + 2)


def moment(m, z, p):                              # <eps^m>, eq:moments
    w = 1 - z * z
    return sum(G(m, l) * z ** l * p ** (2 * m - l) for l in range(0, m + 1, 2)) / w ** m


def eps_max(z, p):                                # eq:e
    return (2 * z * p + p * p) / (1 - z * z)


def S(M, z, p, g):                                # eq:SM
    return (1 - z * z) ** g * sum(beta(g, m) * moment(m, z, p) for m in range(M + 1))


def B(M, z, p, g):                                # eq:bound (Theorem 2)
    return (1 - z * z) ** g * abs(beta(g, M + 1)) * moment(M + 1, z, p) / (1 - eps_max(z, p))


def A_coef(M, g, j, q):                           # eq:A
    s = mp.mpf(0)
    for m in range(max(j, q), min(M, 2 * q) + 1):
        s += beta(g, m) * G(m, 2 * (m - q)) * mp.binomial(m - q, m - j) * (-1) ** (m - j)
    return s


def S_poly(M, z, p, g):                           # eq:poly with eq:A
    v = 1 / (1 - z * z)
    return (1 - z * z) ** g * sum(sum(A_coef(M, g, j, q) * p ** (2 * q) for q in range(M + 1)) * v ** j
                                  for j in range(M + 1))


def T(n, z, p, g):                                # eq:Tn
    v = 1 / (1 - z * z)
    return (1 - z * z) ** g * p ** (2 * n) * sum(beta(g, m) * G(m, 2 * (m - n)) * z ** (2 * (m - n)) * v ** m
                                                 for m in range(n, 2 * n + 1))


def SP(N, z, p, g):                               # S^P_N = sum_{n<=N} T_n
    return sum(T(n, z, p, g) for n in range(N + 1))


def lap_n(n, r, g, terms=4000):
    """Delta^n (1-r^2)^g = sum_{m>=n} beta_m 4^n (m!/(m-n)!)^2 r^(2(m-n)) (Appendix B.1)."""
    s = mp.mpf(0)
    r2 = r * r
    for m in range(n, n + terms):
        t = beta(g, m) * 4 ** n * (mp.factorial(m) / mp.factorial(m - n)) ** 2 * r2 ** (m - n)
        s += t
        if m > n + 50 and abs(t) < mp.eps * abs(s) * 1e-5:
            break
    return s


def lagrange_bounds(N, z, p, g, rmin=None):       # prop:lagrange
    """Remainder bounds from Delta^{N+1} f at the smallest and largest r on the
    disk. The smallest r is max(0, z - p); pass rmin to test other choices."""
    k = (p * p / 4) ** (N + 1) / (mp.factorial(N + 1) * mp.factorial(N + 2))
    r0 = max(mp.mpf(0), z - p) if rmin is None else rmin
    a = k * lap_n(N + 1, r0, g)
    b = k * lap_n(N + 1, z + p, g)
    return min(a, b), max(a, b)


def D_tangent(q):                                 # eq:contact, D_{1/2}(1-q, q)
    return (2 * mp.acos(1 - 2 * q) - mp.mpf(4) / 3 * (3 + 2 * q - 8 * q * q) * mp.sqrt(q * (1 - q))) / (3 * mp.pi * q * q)


def dDdz(z, p, g):                                # eq:dDdz
    x = 1 - (z - p) ** 2
    kap = 4 * z * p / x
    return 2 * x ** g / p * (mp.hyp2f1(-g, mp.mpf(3) / 2, 2, kap) - mp.hyp2f1(-g, mp.mpf(1) / 2, 1, kap))


def P_1d(z, p, g):                                # eq:P1d
    """With s = a + (b - a)(1 - cos t)/2, ds / sqrt((s-a)(b-s)) = dt."""
    a = (z - p) ** 2
    b = (z + p) ** 2
    t1 = mp.pi if b <= 1 else mp.acos(1 - 2 * (1 - a) / (b - a))

    def f(t):
        s = a + (b - a) * (1 - mp.cos(t)) / 2
        return (1 - s) ** (g + 1) * (1 + (p * p - z * z) / s) if s < 1 else mp.mpf(0)
    return -mp.quad(f, [0, t1]) / (2 * (g + 1))


def P_ingress(z, p, g, J):                        # eq:ingress (limb, z > p)
    x = 1 - (z - p) ** 2
    k = x / (4 * z * p)
    s = mp.mpf(0)
    for j in range(J + 1):
        cj = 1 + p * p - z * z if j == 0 else (p * p - z * z) * x ** j
        s += cj * mp.beta(mp.mpf(1) / 2, g + 2 + j) * mp.hyp2f1(0.5, 0.5, g + mp.mpf(5) / 2 + j, k)
    return -x ** (g + 1) * mp.sqrt(k) / (2 * (g + 1)) * s


def P_interiorhyp(z, p, g, J):                    # eq:interiorhyp (inside, z > p)
    x = 1 - (z - p) ** 2
    kap = 4 * z * p / x
    s = mp.mpf(0)
    for j in range(J + 1):
        cj = 1 + p * p - z * z if j == 0 else (p * p - z * z) * x ** j
        s += cj * mp.hyp2f1(-g - 1 - j, 0.5, 1, kap)
    return -mp.pi * x ** (g + 1) / (2 * (g + 1)) * s


def lemma2_bound(z, p, g, J):                     # eq:Ej at a single point
    x = 1 - (z - p) ** 2
    Q = z * z - p * p
    if z > 1 - p:
        k = x / (4 * z * p)
        F = mp.beta(0.5, g + 2 + J + 1) * mp.hyp2f1(0.5, 0.5, g + mp.mpf(5) / 2 + J + 1, k)
        return x ** (g + 1) * mp.sqrt(k) / (2 * (g + 1)) * Q * x ** (J + 1) * F / (1 - x)
    return mp.pi * x ** (g + 1) / (2 * (g + 1)) * Q * x ** (J + 1) / (1 - x)
