"""Limb flux from Chebyshev interpolants built per light curve from the
sinh-trapezoid rules of _trap (no hypergeometric tables).

Write the blocked flux of mu^alpha (z > p, occultor touching the limb) as
P = Phi * J with Phi = -2 sqrt(k) x^(alpha/2+1) / (alpha+2) and

    J(lam; x, c0) = int_0^inf cosh(s)^-(alpha+3) (1 + lam sinh^2 s)^(-1/2) [1 + c0 cosh^2 s / (cosh^2 s - x)] ds,

lam = 1 - k. As a function of lam, J has a branch point at lam = 0 (second
contact) and the connection formula (the integral form of DLMF 15.8.4)

    J = A(lam) + lam^nu B(lam),       nu = (alpha + 3) / 2,
    B(lam) = -k^-nu J(k; -x lam / k, c0) / sin(pi nu)

with A and B analytic around lam = 0; i.e. the singular part is the same
integral at the complementary modulus, and the rule of _trap evaluates both.
For an odd alpha (integer nu) lam^nu B becomes lam^nu ln(lam) B with
B = -(-1)^nu k^-nu J(k; ...) / pi. Inside the disk (lam < 0) the flux is the
real part of J(lam - i0) = A + cos(pi nu) |lam|^nu B (integer nu:
A + (-1)^nu |lam|^nu ln|lam| B).

Three Chebyshev interpolants per exponent cover the limb:
  O: k in [0, 1/2]        P = k^nu E(k)                 (first contact)
  C: lam in [0, 1/2]      P = Phi A + lam^nu Phi B      (outside second contact)
  N: lam in [lam_lo, 0]   P = Phi A + cos(pi nu) |lam|^nu Phi B   (inside)
Their node values come from ~70 evaluations of the trapezoid rules, a few
tens of microseconds per light curve; a sample then costs one or two powers
and two Clenshaw sums per exponent.
"""
import math
from math import pi

import numpy as np
from numba import njit

from ._exact import acos_poly
from ._series import _vexp, _vlog

#: Chebyshev degrees (number of nodes) of the three pieces
NO = 16
NCH = 18
NN = 14
#: step of the rule for the complementary integral inside the disk (modulus k > 1:
#: branch point at Im s = asin(1 / sqrt(k)))
H_CB = 0.17
#: smallest |sin(pi nu)| handled by the connection formula (otherwise: direct rules)
SIN_MIN = 0.02
#: largest radius ratio
P_MAX = 0.3
#: lower end of the inside piece (lam = 1 - k); A and B are singular at
#: lam = 1 - 1/(4 p^2) (the occultor rim through the stellar centre)
LAM_MIN = -0.5


@njit(cache=True, fastmath=True, error_model="numpy")
def core_real(lam, x, c0, gams, W, tt, b, out):
    """out[i] = J(lam; x, c0) for the exponents 2 gams[i] by the real rule
    (W, tt, b) of _trap.rules (b holds the step after the nodes)."""
    nt = gams.shape[0]
    nn = tt.shape[0]
    h = b[nn]
    for i in range(nt):
        out[i] = 0.0
    for n in range(nn):
        rn = 1.0 / np.sqrt(1.0 + lam * tt[n])
        g = rn * (1.0 + c0 * b[n] / (b[n] - x))
        for i in range(nt):
            out[i] += W[i, n] * g
    # pole errors where they are not swamped by the merger with the cosh singularity
    d = 1.0 - lam * (1.0 - x)
    if x > 0.02 and x < 1.0 and d > 0.0:
        a = acos_poly(np.sqrt(x))
        e = _vexp(-2.0 * pi * a / h)
        base = pi * c0 * np.sqrt(x) / np.sqrt((1.0 - x) * d) * e / (1.0 - e)
        lx = _vlog(x)
        for i in range(nt):
            out[i] -= base * _vexp(-(gams[i] + 1.5) * lx)


@njit(cache=True, fastmath=True, error_model="numpy")
def core_band(lam, x, c0, gams, Wr, Wi, ttr, tti, br, bi, gamma, hin, out):
    """out[i] = Re J(lam - i0; x, c0), lam < 0, by the complex-contour rule of _trap.rules_in."""
    nt = gams.shape[0]
    nn = ttr.shape[0]
    for i in range(nt):
        out[i] = 0.0
    for n in range(nn):
        u = 1.0 + lam * ttr[n]
        v = lam * tti[n]
        mod = np.sqrt(u * u + v * v)
        a = np.sqrt(0.5 * (mod + abs(u)))
        d = 0.5 * v / a
        sre = a if u >= 0.0 else abs(d)
        sim = d if u >= 0.0 else (a if v >= 0.0 else -a)
        r_re = sre / mod
        r_im = -sim / mod
        dr = br[n] - x
        di = bi[n]
        idn = 1.0 / (dr * dr + di * di)
        gr = 1.0 + c0 * (br[n] * dr + bi[n] * di) * idn
        gi = c0 * (bi[n] * dr - br[n] * di) * idn
        fr = r_re * gr - r_im * gi
        fi = r_re * gi + r_im * gr
        for i in range(nt):
            out[i] += Wr[i, n] * fr - Wi[i, n] * fi
    a = acos_poly(np.sqrt(x))
    om = 1.0 - x
    e1 = np.exp(-2.0 * pi * (a - gamma) / hin)
    e2 = _vexp(-2.0 * pi * (a + gamma) / hin)
    base = 0.5 * pi * c0 * np.sqrt(x) / np.sqrt(om * (1.0 - lam * om)) * (e1 / (1.0 - e1) + e2 / (1.0 - e2))
    lx = _vlog(x)
    for i in range(nt):
        out[i] -= base * _vexp(-(gams[i] + 1.5) * lx)


def _dct_matrix(n):
    m = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    D = (2.0 / n) * np.cos(np.pi * m * (j + 0.5) / n)
    D[0] *= 0.5
    return D


_DO = _dct_matrix(NO)
_DC = _dct_matrix(NCH)
_DN = _dct_matrix(NN)


@njit(cache=True)
def _cheb_coef(vals, D, out):
    """Chebyshev coefficients from values at the first-kind nodes cos(pi (j + 1/2) / n)."""
    n = vals.shape[0]
    for m in range(n):
        s = 0.0
        for j in range(n):
            s += D[m, j] * vals[j]
        out[m] = s


@njit(cache=True, fastmath=True, inline="always")
def _clenshaw(c, t):
    b1 = 0.0
    b2 = 0.0
    t2 = 2.0 * t
    for m in range(c.shape[0] - 1, 0, -1):
        b1, b2 = t2 * b1 - b2 + c[m], b1
    return t * b1 - b2 + c[0]


@njit(cache=True)
def _z_of_k(k, p):
    q = p * (1.0 - 2.0 * k)
    return q + np.sqrt(q * q + 1.0 - p * p)


@njit(cache=True)
def build(p, gams, lam_lo, W, tt, b, Wr, Wi, ttr, tti, br, bi, gamma, hin, Wc, ttc, bc):
    """Chebyshev coefficients (per exponent) of the three pieces. lam_lo < 0 is
    the lower end of the inside piece (0: no inside piece).
    Returns EO, AC, BC, AN, BN (shape (nt, n))."""
    nt = gams.shape[0]
    EO = np.zeros((nt, NO))
    AC = np.zeros((nt, NCH))
    BC = np.zeros((nt, NCH))
    AN = np.zeros((nt, NN))
    BN = np.zeros((nt, NN))
    nu = np.empty(nt)
    sg = np.empty(nt)          # 1 / sin(pi nu), or (-1)^nu / pi for an integer nu
    isint = np.zeros(nt, dtype=np.bool_)
    for i in range(nt):
        nu[i] = gams[i] + 1.5
        r = np.round(nu[i])
        if nu[i] == r:
            isint[i] = True
            sg[i] = (1.0 if int(r) % 2 == 0 else -1.0) / pi
        else:
            sg[i] = 1.0 / np.sin(pi * nu[i])
    I = np.empty(nt)
    Jc = np.empty(nt)
    v = np.empty((nt, max(NO, NCH, NN)))
    w = np.empty((nt, max(NO, NCH, NN)))
    # O: E(k) = -2 (4 z p)^(g+1) / (2g+2) J(1 - k)
    for j in range(NO):
        k = 0.25 * (1.0 + np.cos(pi * (j + 0.5) / NO))          # [0, 1/2]
        z = _z_of_k(k, p)
        x = 4.0 * z * p * k
        core_real(1.0 - k, x, (p - z) * (p + z), gams, W, tt, b, I)
        l4 = np.log(4.0 * z * p)
        for i in range(nt):
            v[i, j] = -2.0 * np.exp((gams[i] + 1.0) * l4) / (2.0 * gams[i] + 2.0) * I[i]
    for i in range(nt):
        _cheb_coef(v[i, :NO], _DO, EO[i])
    # C: lam in [0, 1/2]
    for j in range(NCH):
        lam = 0.25 * (1.0 + np.cos(pi * (j + 0.5) / NCH))
        k = 1.0 - lam
        z = _z_of_k(k, p)
        x = 4.0 * z * p * k
        c0 = (p - z) * (p + z)
        core_real(lam, x, c0, gams, W, tt, b, I)
        core_real(k, -x * lam / k, c0, gams, W, tt, b, Jc)
        ll = np.log(lam)
        lk = np.log(k)
        lx = np.log(x)
        for i in range(nt):
            phi = -2.0 * np.sqrt(k) * np.exp((gams[i] + 1.0) * lx) / (2.0 * gams[i] + 2.0)
            B = -np.exp(-nu[i] * lk) * Jc[i] * sg[i]
            S = np.exp(nu[i] * ll) * (ll if isint[i] else 1.0)
            v[i, j] = phi * (I[i] - S * B)
            w[i, j] = phi * B
    for i in range(nt):
        _cheb_coef(v[i, :NCH], _DC, AC[i])
        _cheb_coef(w[i, :NCH], _DC, BC[i])
    # N: lam in [lam_lo, 0]
    if lam_lo < 0.0:
        for j in range(NN):
            lam = 0.5 * lam_lo * (1.0 - np.cos(pi * (j + 0.5) / NN))
            k = 1.0 - lam
            z = _z_of_k(k, p)
            x = 4.0 * z * p * k
            c0 = (p - z) * (p + z)
            core_band(lam, x, c0, gams, Wr, Wi, ttr, tti, br, bi, gamma, hin, I)
            core_real(k, -x * lam / k, c0, gams, Wc, ttc, bc, Jc)
            ll = np.log(-lam)
            lk = np.log(k)
            lx = np.log(x)
            for i in range(nt):
                phi = -2.0 * np.sqrt(k) * np.exp((gams[i] + 1.0) * lx) / (2.0 * gams[i] + 2.0)
                B = -np.exp(-nu[i] * lk) * Jc[i] * sg[i]
                if isint[i]:
                    S = (1.0 if int(np.round(nu[i])) % 2 == 0 else -1.0) * np.exp(nu[i] * ll) * ll
                else:
                    S = np.cos(pi * nu[i]) * np.exp(nu[i] * ll)
                v[i, j] = phi * (I[i] - S * B)
                w[i, j] = phi * B
        for i in range(nt):
            _cheb_coef(v[i, :NN], _DN, AN[i])
            _cheb_coef(w[i, :NN], _DN, BN[i])
    return EO, AC, BC, AN, BN


@njit(cache=True, fastmath=True, error_model="numpy")
def _clen_block(C, t, m, b1, b2):
    """b1 <- sum_k C[k] T_k(t[j]) for j < m, vectorised over the samples."""
    for j in range(m):
        b1[j] = 0.0
        b2[j] = 0.0
    for q in range(C.shape[0] - 1, 0, -1):
        c = C[q]
        for j in range(m):
            tmp = 2.0 * t[j] * b1[j] - b2[j] + c
            b2[j] = b1[j]
            b1[j] = tmp
    c = C[0]
    for j in range(m):
        b1[j] = t[j] * b1[j] - b2[j] + c


@njit(cache=True, fastmath=True, error_model="numpy")
def evaluate(zl, m, p, gams, cs, lam_lo, EO, AC, BC, AN, BN, out):
    """out[j] = sum_i cs[i] P_i(zl[j]) where a piece covers zl[j] (z > p,
    lam >= lam_lo); NaN elsewhere. Samples are grouped by piece and the
    Clenshaw sums run over the samples (vectorised)."""
    nt = gams.shape[0]
    zO = np.sqrt((1.0 - p) * (1.0 + p))
    reg = np.empty(m, dtype=np.int64)
    cnt = np.zeros(3, dtype=np.int64)
    for j in range(m):
        z = zl[j]
        q4 = 4.0 * z * p
        r = -1
        if z >= zO and z < 1.0 + p:
            r = 0
        elif z > p and z < zO:
            lam = ((z - 1.0) + p) * (z + p + 1.0) / q4
            if lam >= 0.0:
                r = 1
            elif lam >= lam_lo:
                r = 2
        reg[j] = r
        if r >= 0:
            cnt[r] += 1
        else:
            out[j] = np.nan
    for R in range(3):
        n = cnt[R]
        if n == 0:
            continue
        ix = np.empty(n, dtype=np.int64)
        q = 0
        for j in range(m):
            if reg[j] == R:
                ix[q] = j
                q += 1
        t = np.empty(n)
        v = np.empty(n)          # log of k (O) or |lam| (C, N)
        pos = np.empty(n)        # 1 where the singular factor is nonzero
        for q in range(n):
            z = zl[ix[q]]
            q4 = 4.0 * z * p
            if R == 0:
                y = ((1.0 - z) + p) * ((1.0 + z) - p) / q4
                t[q] = 4.0 * y - 1.0
            else:
                y = ((z - 1.0) + p) * (z + p + 1.0) / q4
                t[q] = 4.0 * y - 1.0 if R == 1 else 1.0 - 2.0 * y / lam_lo
                y = abs(y)
            pos[q] = 1.0 if y > 0.0 else 0.0
            v[q] = _vlog(max(y, 1e-300))
        acc = np.zeros(n)
        b1 = np.empty(n)
        b2 = np.empty(n)
        b3 = np.empty(n)
        if R > 0:
            # the regular parts of all exponents share one Clenshaw sum
            A = AC if R == 1 else AN
            Asum = np.zeros(A.shape[1])
            for i in range(nt):
                for c in range(A.shape[1]):
                    Asum[c] += cs[i] * A[i, c]
            _clen_block(Asum, t, n, b1, b2)
            for q in range(n):
                acc[q] = b1[q]
        for i in range(nt):
            nu = gams[i] + 1.5
            rn = np.round(nu)
            isint = nu == rn
            ci = cs[i]
            if R == 0:
                _clen_block(EO[i], t, n, b1, b2)
                for q in range(n):
                    acc[q] += ci * pos[q] * _vexp(nu * v[q]) * b1[q]
            else:
                if R == 1:
                    _clen_block(BC[i], t, n, b3, b2)
                    fac = 1.0
                else:
                    _clen_block(BN[i], t, n, b3, b2)
                    if isint:
                        fac = 1.0 if int(rn) % 2 == 0 else -1.0
                    else:
                        fac = np.cos(pi * nu)
                fac *= ci
                if isint:
                    for q in range(n):
                        acc[q] += fac * pos[q] * _vexp(nu * v[q]) * v[q] * b3[q]
                else:
                    for q in range(n):
                        acc[q] += fac * pos[q] * _vexp(nu * v[q]) * b3[q]
        for q in range(n):
            out[ix[q]] = acc[q]

def supported(gams):
    """True if no exponent is close to (but not at) an odd integer alpha, where
    1 / sin(pi nu) would amplify rounding in the connection formula."""
    for g in gams:
        g = float(g)
        if g == math.floor(g):
            continue          # alpha = 0, 2, ...: closed form, not interpolated
        nu = g + 1.5
        if nu != round(nu) and abs(math.sin(math.pi * nu)) < SIN_MIN:
            return False
    return True
