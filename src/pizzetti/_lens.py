"""Hypergeometric solution near and across the stellar limb (no quadrature).

For a term (1 - r^2)^g of the intensity, Green's theorem with the radial field
x (1 - (1 - r^2)^(g+1)) / (2 (g+1) r^2) reduces the blocked flux to a line
integral over the part of the occultor's rim that lies on the star. With
s = r^2, a = (z - p)^2, b = (z + p)^2 and z > p (origin outside the occultor),

    P_g = -1/(2(g+1)) int (1-s)^(g+1) [1 + (p^2 - z^2)/s] ds / sqrt((s-a)(b-s))

over s in [a, 1] (limb crossing, 1-p < z < 1+p) or s in [a, b] (interior).
Expanding 1/s = sum_j x^j (1-s)^j/(1-a)^j-style in x = 1 - a (always < 1 for
z > p) leaves Euler integrals, i.e. Gauss hypergeometric functions of ONE
variable whose parameters depend only on g and j:

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
The coefficient tables depend only on g (not on p or z).
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
        CA = gamma(c) * gamma(m) * _rgamma(c - a) * _rgamma(c - b)
        CB = gamma(c) * gamma(-m) * _rgamma(a) * _rgamma(b)
        _taylor(a, b, 1.0 - m, A)
        _taylor(c - a, c - b, 1.0 + m, Bc)
        A *= CA
        Bc *= CB
        return
    mm = int(mi)
    t = gamma(mi) * gamma(c) * _rgamma(a + mi) * _rgamma(b + mi)
    for i in range(min(mm, n)):
        A[i] = t
        if i + 1 < mm:
            t *= (i + a) * (i + b) / ((i + 1.0) * (i + 1.0 - mi))
    pre = -((-1.0) ** mm) * gamma(c) * _rgamma(a) * _rgamma(b)
    t = 1.0 / gamma(mi + 1.0)
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


# ---------------------------------------------------------------------------
# Per-light-curve compression. For fixed p and g every quantity above is a
# function of z alone. On each of four pieces of the z axis,
#   limb crossing   I1: k <= 1/2  (sqrt(1-p^2) <= z < 1+p)
#                   I2: k >= 1/2  (1-p < z <= sqrt(1-p^2))
#   interior band   N1: kappa <= 1/2 (z_lo <= z <= sqrt(1+8p^2) - 3p)
#                   N2: kappa >= 1/2 (sqrt(1+8p^2) - 3p <= z < 1-p)
# the sum S = P / pre is
#   S = R(z)                                   (I1, N1)
#   S = R(z) + y^m0 [ln(y) Q1(z) + Q2(z)]      (I2, N2), y = 1 - k or 1 - kappa
# with R, Q1, Q2 analytic on the piece. They are replaced by Chebyshev
# series in z (NFIT nodes), so each sample costs a few Clenshaw steps and
# three or four elementary functions per exponent.
# ---------------------------------------------------------------------------
NFIT = 32


@njit(cache=True, fastmath=False)
def _parts(z, p, g, T, L, fam):
    """R, Q1, Q2 at z for family fam (no prefactor); NaN if j <= JMAX does
    not reach 2^-60 relative."""
    zmp = z - p
    x = (1.0 - zmp) * (1.0 + zmp)
    q = (p - z) * (p + z)
    if fam == 0:
        arg = x / (4.0 * z * p)
    else:
        arg = 4.0 * z * p / x
    arg = min(max(arg, 0.0), 1.0)
    y = 1.0 - arg
    R = 0.0
    Q1 = 0.0
    Q2 = 0.0
    cj = 1.0 + q
    yj = 1.0
    jm = T.shape[1] - 1
    for j in range(jm + 1):
        if j == 1:
            cj = q * x
        elif j > 1:
            cj *= x
        if arg <= 0.5:
            tr = cj * _horner(T[fam, j, 0], L[fam, j, 0], arg)
            R += tr
            t1 = 0.0
            t2 = 0.0
        else:
            tr = cj * _horner(T[fam, j, 1], L[fam, j, 1], y)
            t1 = cj * yj * _horner(T[fam, j, 2], L[fam, j, 2], y)
            t2 = cj * yj * _horner(T[fam, j, 3], L[fam, j, 3], y)
            R += tr
            Q1 += t1
            Q2 += t2
            yj *= y
        if j >= 1 and x <= 0.5:
            sc = abs(R) + abs(Q1) + abs(Q2)
            if (abs(tr) + abs(t1) + abs(t2)) * x <= 2.0 ** -60 * (1.0 - x) * sc:
                return R, Q1, Q2
    return np.nan, np.nan, np.nan


@njit(cache=True, fastmath=False)
def fit(p, gams, tabs, lens, z_lo, nfit=NFIT):
    """Chebyshev compression for one light curve.
    Returns bounds (4, 2), coefficients (nt, 4, 3, nfit), ok flags (nt, 4).
    A piece is accepted if its last three coefficients are below 1e-13 of the
    largest (the coefficients reach rounding level, ~1e-15, well before)."""
    nt = gams.shape[0]
    zh = np.sqrt(1.0 - p * p)
    zk = np.sqrt(1.0 + 8.0 * p * p) - 3.0 * p
    bounds = np.empty((4, 2))
    bounds[0, 0] = zh
    bounds[0, 1] = 1.0 + p
    bounds[1, 0] = 1.0 - p
    bounds[1, 1] = zh
    bounds[2, 0] = min(z_lo, zk)
    bounds[2, 1] = zk
    bounds[3, 0] = max(z_lo, zk)
    bounds[3, 1] = 1.0 - p
    C = np.zeros((nt, 4, 3, nfit))
    ok = np.zeros((nt, 4), dtype=np.bool_)
    th = np.empty(nfit)
    for k in range(nfit):
        th[k] = pi * (k + 0.5) / nfit
    V = np.empty((nfit, 3))
    for pc in range(4):
        a = bounds[pc, 0]
        b = bounds[pc, 1]
        if not (b > a) or a <= p or z_lo < 0.0 and pc >= 2:
            continue
        fam = 0 if pc < 2 else 1
        for i in range(nt):
            good = True
            for k in range(nfit):
                zz = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(th[k])
                r, q1, q2 = _parts(zz, p, gams[i], tabs[i], lens[i], fam)
                if np.isnan(r):
                    good = False
                    break
                V[k, 0] = r
                V[k, 1] = q1
                V[k, 2] = q2
            if not good:
                continue
            for f in range(3):
                big = 0.0
                for m in range(nfit):
                    acc = 0.0
                    for k in range(nfit):
                        acc += V[k, f] * np.cos(m * th[k])
                    acc *= 2.0 / nfit
                    if m == 0:
                        acc *= 0.5
                    C[i, pc, f, m] = acc
                    big = max(big, abs(acc))
                tail = max(abs(C[i, pc, f, nfit - 1]), abs(C[i, pc, f, nfit - 2]), abs(C[i, pc, f, nfit - 3]))
                if tail > 1e-13 * big:
                    good = False
            ok[i, pc] = good
    return bounds, C, ok


@njit(cache=True, fastmath=False)
def _clenshaw(c, t):
    b1 = 0.0
    b2 = 0.0
    t2 = 2.0 * t
    for i in range(c.shape[0] - 1, 0, -1):
        b1, b2 = t2 * b1 - b2 + c[i], b1
    return t * b1 - b2 + c[0]


@njit(cache=True, fastmath=False)
def blocked_fit(z, p, gams, cs, bounds, C, ok):
    """As blocked_hyp, from the per-light-curve compression; NaN if z is in
    no fitted piece."""
    pc = -1
    for i in range(4):
        if bounds[i, 0] <= z <= bounds[i, 1] and bounds[i, 1] > bounds[i, 0]:
            pc = i
            break
    if pc < 0 or not (z > p):
        return np.nan
    a = bounds[pc, 0]
    b = bounds[pc, 1]
    t = (2.0 * z - a - b) / (b - a)
    zmp = z - p
    x = (1.0 - zmp) * (1.0 + zmp)
    if x <= 0.0:
        return 0.0
    lx = np.log(x)
    if pc < 2:
        arg = x / (4.0 * z * p)
        sk = np.sqrt(min(arg, 1.0))
    else:
        arg = 4.0 * z * p / x
        sk = 1.0
    sing = pc == 1 or pc == 3
    y = 1.0 - min(arg, 1.0)
    ly = np.log(y) if (sing and y > 0.0) else 0.0
    tot = 0.0
    for i in range(gams.shape[0]):
        if not ok[i, pc]:
            return np.nan
        g = gams[i]
        S = _clenshaw(C[i, pc, 0], t)
        if sing and y > 0.0:
            S += np.exp((g + 1.5) * ly) * (ly * _clenshaw(C[i, pc, 1], t) + _clenshaw(C[i, pc, 2], t))
        tot += cs[i] * (-np.exp((g + 1.0) * lx) * sk / (2.0 * (g + 1.0))) * S
    return tot


@njit(cache=True, fastmath=False)
def fit_lengths(C, ok):
    """Effective Chebyshev lengths: coefficients below 2e-15 of the largest
    coefficient of the same function are dropped from the tail."""
    nt, npc, nf, n = C.shape
    N = np.zeros((nt, npc, nf), dtype=np.int64)
    for i in range(nt):
        for pc in range(npc):
            if not ok[i, pc]:
                continue
            for f in range(nf):
                big = 0.0
                for m in range(n):
                    big = max(big, abs(C[i, pc, f, m]))
                L = 0
                for m in range(n):
                    if abs(C[i, pc, f, m]) > 2e-15 * big:
                        L = m + 1
                N[i, pc, f] = L
    return N


@njit(cache=True, fastmath=True, error_model="numpy")
def _clenshaw_vec(c, L, t, out, acc):
    """acc[j] += sum_m c[m] T_m(t[j]) (vectorised over samples)."""
    n = t.shape[0]
    if L == 0:
        return
    b1 = np.zeros(n)
    b2 = np.zeros(n)
    for m in range(L - 1, 0, -1):
        cm = c[m]
        for j in range(n):
            b0 = 2.0 * t[j] * b1[j] - b2[j] + cm
            b2[j] = b1[j]
            b1[j] = b0
    c0 = c[0]
    for j in range(n):
        out[j] = t[j] * b1[j] - b2[j] + c0
        acc[j] = out[j]


@njit(cache=True, fastmath=True, error_model="numpy")
def _pw(x, lx, e):
    """x^e; quarter-integer e by square roots, otherwise exp(e ln x)."""
    e4 = 4.0 * e
    if e4 == np.floor(e4) and e4 < 16.0:
        n4 = int(e4)
        r = 1.0
        for _ in range(n4 // 4):
            r *= x
        rem = n4 % 4
        if rem:
            sx = np.sqrt(x)
            if rem == 2:
                r *= sx
            else:
                qx = np.sqrt(sx)
                r *= qx if rem == 1 else sx * qx
        return r
    return np.exp(e * lx)


@njit(cache=True, fastmath=True, error_model="numpy")
def blocked_fit_vec(z, p, gams, cs, bounds, C, ok, NL, out):
    """out[j] = blocked flux at z[j] from the compression, NaN where no piece
    applies (z outside the fitted pieces, z <= p, or a piece not accepted)."""
    n = z.shape[0]
    nt = gams.shape[0]
    gen = False
    for i in range(nt):
        if (4.0 * gams[i]) != np.floor(4.0 * gams[i]):
            gen = True
    piece = np.full(n, -1, dtype=np.int64)
    cnt = np.zeros(4, dtype=np.int64)
    for j in range(n):
        zj = z[j]
        if zj > p:
            for pc in range(4):
                if bounds[pc, 1] > bounds[pc, 0] and bounds[pc, 0] <= zj <= bounds[pc, 1]:
                    good = True
                    for i in range(nt):
                        good = good and ok[i, pc]
                    if good:
                        piece[j] = pc
                        cnt[pc] += 1
                    break
        out[j] = np.nan
    for pc in range(4):
        m = cnt[pc]
        if m == 0:
            continue
        idx = np.empty(m, dtype=np.int64)
        k = 0
        for j in range(n):
            if piece[j] == pc:
                idx[k] = j
                k += 1
        a = bounds[pc, 0]
        b = bounds[pc, 1]
        t = np.empty(m)
        lx = np.empty(m)
        sk = np.empty(m)
        ly = np.empty(m)
        xs = np.empty(m)
        ys = np.empty(m)
        tot = np.zeros(m)
        sing = pc == 1 or pc == 3
        for k in range(m):
            zz = z[idx[k]]
            t[k] = (2.0 * zz - a - b) / (b - a)
            zmp = zz - p
            x = max((1.0 - zmp) * (1.0 + zmp), 1e-300)
            xs[k] = x
            lx[k] = np.log(x) if gen else 0.0
            if pc < 2:
                arg = min(x / (4.0 * zz * p), 1.0)
                sk[k] = np.sqrt(arg)
            else:
                arg = min(4.0 * zz * p / x, 1.0)
                sk[k] = 1.0
            ys[k] = max(1.0 - arg, 1e-300)
            ly[k] = np.log(ys[k]) if sing else 0.0
        R = np.empty(m)
        Q1 = np.empty(m)
        Q2 = np.empty(m)
        tmp = np.empty(m)
        for i in range(nt):
            g = gams[i]
            _clenshaw_vec(C[i, pc, 0], NL[i, pc, 0], t, tmp, R)
            if sing:
                Q1[:] = 0.0
                Q2[:] = 0.0
                _clenshaw_vec(C[i, pc, 1], NL[i, pc, 1], t, tmp, Q1)
                _clenshaw_vec(C[i, pc, 2], NL[i, pc, 2], t, tmp, Q2)
                for k in range(m):
                    R[k] += _pw(ys[k], ly[k], g + 1.5) * (ly[k] * Q1[k] + Q2[k])
            f = -cs[i] / (2.0 * (g + 1.0))
            for k in range(m):
                tot[k] += f * _pw(xs[k], lx[k], g + 1.0) * sk[k] * R[k]
        for k in range(m):
            out[idx[k]] = tot[k]
