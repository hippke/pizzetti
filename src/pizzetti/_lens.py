"""Hypergeometric solution near and across the stellar limb (no quadrature).

For a term (1 - r^2)^g of the intensity, Green's theorem with the radial field
x (1 - (1 - r^2)^(g+1)) / (2 (g+1) r^2) reduces the blocked flux to a line
integral over the part of the occultor's rim that lies on the star. With
s = r^2, a = (z - p)^2, b = (z + p)^2 and z > p (origin outside the occultor),

    P_g = -1/(2(g+1)) int (1-s)^(g+1) [1 + (p^2 - z^2)/s] ds / sqrt((s-a)(b-s))

over s in [a, 1] (limb crossing, 1-p < z < 1+p) or s in [a, b] (interior).
For the limb crossing this is Case II of Mandel & Agol (2002): a Gauss
function plus an Appell F1 from the factor 1/s. Writing s = 1 - x (1 - u)
with x = 1 - a (< 1 for z > p) and expanding 1/s = sum_j x^j (1 - u)^j
leaves Euler integrals, i.e. Gauss hypergeometric functions of ONE variable
whose parameters depend only on g and j:

  limb crossing, k = (1 - a) / (b - a) in [0, 1]:
    P_g = -x^(g+1) sqrt(k) / (2(g+1)) sum_j c_j B(1/2, g+2+j) H_j(k),
    H_j(k) = 2F1(1/2, 1/2; g+5/2+j; k)
  interior, kappa = (b - a) / (1 - a) in [0, 1):
    P_g = -pi x^(g+1) / (2(g+1)) sum_j c_j G_j(kappa),
    G_j(kappa) = 2F1(-(g+1+j), 1/2; 1; kappa)

  c_0 = 1 + p^2 - z^2,  c_j = (p^2 - z^2) x^j (j >= 1).

k = 0 is first contact, k = kappa = 1 is second contact. All terms with j >= 1
have one sign and B_j H_j, G_j do not increase with j, so the remainder after
j = J is bounded by |p^2 - z^2| x^(J+1) / (1 - x) times the (J+1)-th function.

Both families have c - a - b = m_j = g + 3/2 + j. Their universal functions
are evaluated without quadrature or special-function libraries:
  argument <= 1/2: Taylor series;
  argument  > 1/2: connection formula in y = 1 - argument,
                   F = A(y) + y^m (ln(y) Bl(y) + Bc(y)),
  with Bl = 0 for non-integer m and the A&S 15.3.11 form for integer m
  (g half-integer, e.g. the quadratic law). The y^m (y^m ln y) term is the
  singular behaviour of the flux at second contact.
The coefficient tables depend only on g (not on p or z). The contact
expansions of _contact are built from them.
"""
from math import gamma, lgamma, log, pi

import numpy as np
from numba import njit

#: series switches from the Taylor form to the connection form at this argument
XS = 0.5
#: number of stored coefficients per power series
NC = 96
#: largest j stored
JMAX = 60
_TINY = 2.0 ** -62


@njit(cache=True)
def _rgamma(x):
    if x <= 0.0 and x == np.floor(x):
        return 0.0
    if x > 170.0:
        return 0.0
    return 1.0 / gamma(x)


@njit(cache=True)
def _lgs(x):
    """(log|Gamma(x)|, sign Gamma(x)); sign 0 at the poles."""
    if x <= 0.0 and x == np.floor(x):
        return 0.0, 0.0
    sg = 1.0
    if x < 0.0 and int(np.floor(-x)) % 2 == 0:
        sg = -1.0
    return lgamma(x), sg


@njit(cache=True)
def _gratio(a1, a2, b1, b2):
    """Gamma(a1) Gamma(a2) / (Gamma(b1) Gamma(b2)) without overflow."""
    l1, s1 = _lgs(a1)
    l2, s2 = _lgs(a2)
    l3, s3 = _lgs(b1)
    l4, s4 = _lgs(b2)
    if s3 == 0.0 or s4 == 0.0:
        return 0.0
    return s1 * s2 * s3 * s4 * np.exp(l1 + l2 - l3 - l4)


@njit(cache=True)
def _digamma(x):
    r = 0.0
    if x <= 0.0:                      # reflection (x not an integer)
        return _digamma(1.0 - x) - pi / np.tan(pi * x)
    while x < 8.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))))


@njit(cache=True)
def _taylor(a, b, c, out):
    t = 1.0
    for i in range(out.shape[0]):
        out[i] = t
        t *= (i + a) * (i + b) / ((i + c) * (i + 1.0))


@njit(cache=True)
def _connection(a, b, c, A, Bl, Bc):
    """A, Bl, Bc with F = A(y) + y^m (ln y Bl(y) + Bc(y)), y = 1 - x."""
    n = A.shape[0]
    m = c - a - b
    mi = np.floor(m + 0.5)
    A[:] = 0.0
    Bl[:] = 0.0
    Bc[:] = 0.0
    if abs(m - mi) > 1e-9:
        CA = _gratio(c, m, c - a, c - b)
        CB = _gratio(c, -m, a, b)
        _taylor(a, b, 1.0 - m, A)
        _taylor(c - a, c - b, 1.0 + m, Bc)
        A *= CA
        Bc *= CB
        return
    mm = int(mi)
    t = _gratio(mi, c, a + mi, b + mi)
    for i in range(min(mm, n)):
        A[i] = t
        if i + 1 < mm:
            t *= (i + a) * (i + b) / ((i + 1.0) * (i + 1.0 - mi))
    l1, s1 = _lgs(c)
    l3, s3 = _lgs(a)
    l4, s4 = _lgs(b)
    pre = 0.0 if s3 == 0.0 or s4 == 0.0 else \
        -((-1.0) ** mm) * s1 * s3 * s4 * np.exp(l1 - l3 - l4 - lgamma(mi + 1.0))
    t = 1.0
    for i in range(n):
        psi = (-_digamma(i + 1.0) - _digamma(i + mi + 1.0)
               + _digamma(a + i + mi) + _digamma(b + i + mi))
        Bl[i] = pre * t
        Bc[i] = pre * t * psi
        t *= (a + mi + i) * (b + mi + i) / ((i + 1.0) * (i + mi + 1.0))


@njit(cache=True)
def _length(co, xmax):
    """Number of coefficients needed for |argument| <= xmax (2^-60 relative)."""
    big = 0.0
    w = 1.0
    for i in range(co.shape[0]):
        big = max(big, abs(co[i]) * w)
        w *= xmax
    if big == 0.0:
        return 0
    L = 0
    w = 1.0
    for i in range(co.shape[0]):
        if abs(co[i]) * w > 2.0 ** -60 * big:
            L = i + 1
        w *= xmax
    return L


@njit(cache=True)
def table(g, jmax=JMAX):
    """Coefficient table for one exponent g: T of shape (2 families, jmax+1, 4, NC)
    and the used lengths L (same leading shape). Slot 0: Taylor in the
    argument; slots 1-3: A, Bl, Bc of the connection form. The limb-crossing
    family is scaled by B(1/2, g+2+j), the interior one by pi."""
    T = np.zeros((2, jmax + 1, 4, NC))
    L = np.zeros((2, jmax + 1, 4), dtype=np.int64)
    for j in range(jmax + 1):
        for fam in range(2):
            if fam == 0:
                a, b, c = 0.5, 0.5, g + 2.5 + j
                s = np.exp(lgamma(0.5) + lgamma(g + 2.0 + j) - lgamma(g + 2.5 + j))
            else:
                a, b, c = -(g + 1.0 + j), 0.5, 1.0
                s = pi
            _taylor(a, b, c, T[fam, j, 0])
            _connection(a, b, c, T[fam, j, 1], T[fam, j, 2], T[fam, j, 3])
            for sl in range(4):
                T[fam, j, sl] *= s
                L[fam, j, sl] = _length(T[fam, j, sl], XS)
    return T, L


@njit(cache=True, fastmath=False)
def _horner(co, n, x):
    s = 0.0
    for i in range(n - 1, -1, -1):
        s = s * x + co[i]
    return s


@njit(cache=True, fastmath=False)
def _family(T, L, fam, j, arg, ym, ly):
    """B_j H_j(arg) (fam 0) or pi G_j(arg) (fam 1); ym = y^(m_j), ly = ln y."""
    if arg <= 0.5:
        return _horner(T[fam, j, 0], L[fam, j, 0], arg)
    y = 1.0 - arg
    r = _horner(T[fam, j, 1], L[fam, j, 1], y)
    if ym != 0.0:
        r += ym * (ly * _horner(T[fam, j, 2], L[fam, j, 2], y) + _horner(T[fam, j, 3], L[fam, j, 3], y))
    return r


@njit(cache=True, fastmath=False)
def blocked_term(z, p, g, T, L, tol):
    """int (1 - r^2)^g dA over the occulted part of the stellar disk, for
    z > p and z < 1 + p (limb crossing or interior). Returns NaN if the
    expansion cannot reach ``tol`` (relative) with the stored j <= JMAX."""
    if not (z > p):
        return np.nan
    zmp = z - p
    a = zmp * zmp
    x = (1.0 - zmp) * (1.0 + zmp)                 # 1 - a, accurate
    if x <= 0.0:
        return 0.0
    q = (p - z) * (p + z)                          # p^2 - z^2
    c0 = 1.0 + q
    lim = (1.0 - z) - p                            # > 0 inside, < 0 across the limb
    if lim < 0.0:
        fam = 0
        arg = x / (4.0 * z * p)                   # k
        pre = -np.exp((g + 1.0) * np.log(x)) * np.sqrt(arg) / (2.0 * (g + 1.0))
    else:
        fam = 1
        arg = 4.0 * z * p / x                     # kappa
        pre = -np.exp((g + 1.0) * np.log(x)) / (2.0 * (g + 1.0))
    if arg > 1.0:
        arg = 1.0
    m0 = g + 1.5
    y = 1.0 - arg
    if arg > 0.5 and y > 0.0:
        ly = np.log(y)
        ym = np.exp(m0 * ly)
    else:
        ly = 0.0
        ym = 0.0
    s = c0 * _family(T, L, fam, 0, arg, ym, ly)
    xj = 1.0
    jm = T.shape[1] - 1
    for j in range(1, jm + 1):
        xj *= x
        ym *= y
        t = q * xj * _family(T, L, fam, j, arg, ym, ly)
        s += t
        # remainder <= |t| x / (1 - x) (one sign, non-increasing functions)
        if abs(t) * x <= tol * (1.0 - x) * abs(s):
            return pre * s
    return np.nan


@njit(cache=True, fastmath=False)
def blocked_hyp(z, p, gams, cs, tabs, lens, tol):
    """sum_i c_i * blocked flux of (1 - r^2)^{g_i}; NaN if not applicable."""
    tot = 0.0
    for i in range(gams.shape[0]):
        v = blocked_term(z, p, gams[i], tabs[i], lens[i], tol)
        if np.isnan(v):
            return np.nan
        tot += cs[i] * v
    return tot
