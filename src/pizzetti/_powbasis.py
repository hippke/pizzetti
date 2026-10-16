"""Blocked fluxes of fixed power-law intensity terms, fast for many radius ratios.

For a term (1 - r^2)^g (= mu^(2g)) of the intensity, ``occult`` rebuilds the
contact expansions of ``_contact`` for every new radius ratio p (~150 us per
exponent) and therefore uses them only for long light curves; inside a fit it
falls back to quadrature at the limb (~1 us per sample and exponent).

Here the expansion coefficients are built once per *panel* of radius ratios
(geometric panels p in [P_MIN * RATIO^i, P_MIN * RATIO^(i+1)]) at DEG + 1
Chebyshev nodes, and interpolated in p for every light curve: the coefficients
are analytic in p, and the interpolation error is ~1e-15. The series have
degree NB (= 64, twice ``_contact.NS``), so that piece C reaches down to the
cut of the interior series and no limb sample is left to quadrature.

Every panel is validated when it is built: the interpolated expansions are
compared with the Green's-theorem quadrature (``_general.blocked_limb``) on a
grid of separations (dense near the contacts) at Chebyshev nodes and at
off-node radius ratios. A panel whose error exceeds VALID_TOL tries a shorter
reach of piece C (y' <= 1/2, as ``_contact``) and then the degree of
``_contact``; if that fails too, the exponent uses quadrature in that panel.
So the expansions are only used where they are verified; outside
[P_MIN, P_MAX] (and for unvalidated panels) the limb samples use quadrature.

``AlphaColumns`` does the same for mu^alpha with a free exponent (power-2 law
with alpha sampled): 2-D panels in (p, alpha). Near alpha = 1, 3 the contact
series have poles that cancel between their regular and singular parts; they
are stored in a regularised form that is analytic in alpha.

Units: ``columns`` returns blocked flux / pi (as ``_laws.blocked_power``); the
interior samples use the interior series with the rigorous bound ``tol``.
"""
import math

import numpy as np
from numba import njit

from . import _contact, _general, _series
from .occult import _cut_terms, limb_tables, series_tables

NB = 64                 # degree of the contact series
DEG = 10                # Chebyshev degree per panel
P_MIN, P_MAX = 1.0e-3, 0.3
RATIO = 1.15            # panel width (geometric)
VALID_TOL = 2.0e-11     # accepted |blocked flux / pi| error of a validated panel
_LOG_RATIO = math.log(RATIO)
#: validated panels, shared by all instances: (g, tol, panel) -> per-exponent arrays,
#: (tol, panel, alpha panel) -> 2-D arrays (bounded: ~40 kB / ~300 kB per entry)
_CACHE_1D = {}
_CACHE_2D = {}
_MAX_1D, _MAX_2D = 1000, 100
_NS7 = 7                # RA, B (R, L, C), C (R, L, C)


def panel_index(p):
    """Panel number of radius ratio p, or -1 outside [P_MIN, P_MAX)."""
    if not (P_MIN <= p < P_MAX):
        return -1
    return int(math.floor(math.log(p / P_MIN) / _LOG_RATIO))


def panel_range(i):
    return P_MIN * RATIO ** i, P_MIN * RATIO ** (i + 1)


def _cheb_nodes(lo, hi, deg):
    x = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x, x


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _clenshaw_coefs(cc, x, out, L):
    """out[s, :L[s]] = sum_j cc[j, s, :L[s]] T_j(x) (Clenshaw, vectorised over the
    coefficients); cc (DEG+1, 7, NB+1)."""
    nd = cc.shape[0]
    nc = cc.shape[2]
    b1 = np.zeros(nc)
    b2 = np.zeros(nc)
    x2 = 2.0 * x
    for s in range(cc.shape[1]):
        Ls = L[s]
        for i in range(Ls):
            b1[i] = 0.0
            b2[i] = 0.0
        for j in range(nd - 1, 0, -1):
            c = cc[j, s]
            for i in range(Ls):
                t = x2 * b1[i] - b2[i] + c[i]
                b2[i] = b1[i]
                b1[i] = t
        c = cc[0, s]
        o = out[s]
        for i in range(Ls):
            o[i] = x * b1[i] - b2[i] + c[i]


@njit(cache=True, fastmath=True, error_model="numpy")
def _horner_n(c, n, x):
    r = 0.0
    for i in range(n - 1, -1, -1):
        r = r * x + c[i]
    return r


@njit(cache=True, fastmath=True, error_model="numpy")
def _contact_point(zj, p, m, S, L, yc):
    """Blocked flux (absolute) of one term from the expansions S (7, NB+1) with
    lengths L (7,); NaN if zj is outside the validated pieces."""
    zh = math.sqrt((1.0 - p) * (1.0 + p))
    if zh <= zj < 1.0 + p:
        zmp = zj - p
        k = (1.0 - zmp) * (1.0 + zmp) / (4.0 * zj * p)
        if k <= 0.0:
            return 0.0
        lk = math.log(k)
        return _contact._pw(k, math.sqrt(k), math.sqrt(math.sqrt(k)), lk, m) * _horner_n(S[0], L[0], k)
    if 1.0 - p <= zj < zh:
        zpp = zj + p
        y = (zpp - 1.0) * (zpp + 1.0) / (4.0 * zj * p)
        r = _horner_n(S[1], L[1], y)
        if y > 0.0:
            ly = math.log(y)
            sy = math.sqrt(y)
            r += _contact._pw(y, sy, math.sqrt(sy), ly, m) * (ly * _horner_n(S[2], L[2], y)
                                                              + _horner_n(S[3], L[3], y))
        return r
    if zj < 1.0 - p and zj > p:
        zmp = zj - p
        zpp = zj + p
        y = (1.0 - zpp) * (1.0 + zpp) / ((1.0 - zmp) * (1.0 + zmp))
        if y > yc:
            return np.nan
        r = _horner_n(S[4], L[4], y)
        if y > 0.0:
            ly = math.log(y)
            sy = math.sqrt(y)
            r += _contact._pw(y, sy, math.sqrt(sy), ly, m) * (ly * _horner_n(S[5], L[5], y)
                                                              + _horner_n(S[6], L[6], y))
        return r
    return np.nan


@njit(cache=True, fastmath=True, error_model="numpy")
def _horner_vec(c, L, u, n, out):
    """out[:n] = sum_i c[i] u^i (Horner across samples: vectorises)."""
    v = c[L - 1]
    for t in range(n):
        out[t] = v
    for i in range(L - 2, -1, -1):
        ci = c[i]
        for t in range(n):
            out[t] = out[t] * u[t] + ci


@njit(cache=True, fastmath=True, error_model="numpy")
def _logpow(u, n, m, lu, um):
    """lu = ln u, um = u^m for u[:n] > 0 (branch-free, vectorises)."""
    for t in range(n):
        l_ = _series._vlog(u[t])
        lu[t] = l_
        um[t] = _series._vexp(m * l_)


@njit(cache=True, fastmath=True, error_model="numpy")
def _pieces_vec(kind, u, n, m, S, L, out, nreg=-1):
    """Blocked flux (absolute) of one term at arguments u[:n] of piece kind
    (0: A, k; 1: B, y; 2: C, y'), from the expansions S (7, NB+1), lengths L.
    nreg >= 0: pieces B and C are in the regularised form (free exponent, see
    ``AlphaColumns``): P = Rt(u) + u^n E_d(ln u) Ct(u), d = m - n,
    E_d(l) = (e^{d l} - 1) / d (-> l for d -> 0), with Rt, Ct in the slots of R, L."""
    lu = np.empty(n)
    um = np.empty(n)
    h = np.empty(n)
    for t in range(n):
        if not u[t] > 0.0:
            u[t] = 1e-300
    if kind > 0 and nreg >= 0:
        o = 1 if kind == 1 else 4
        d = m - nreg
        ds = d if d != 0.0 else 1.0
        _logpow(u, n, float(nreg), lu, um)          # um = u^n
        _horner_vec(S[o + 1], L[o + 1], u, n, h)    # Ct
        _horner_vec(S[o], L[o], u, n, out)          # Rt
        for t in range(n):
            x_ = d * lu[t]
            ser = lu[t] * (1.0 + x_ * (0.5 + x_ * (1 / 6 + x_ * (1 / 24 + x_ * (1 / 120 + x_ * (
                1 / 720 + x_ * (1 / 5040 + x_ * (1 / 40320 + x_ / 362880))))))))
            ex = (_series._vexp(x_) - 1.0) / ds
            e_ = ser if abs(x_) < 0.05 else ex
            out[t] += um[t] * e_ * h[t]
        return
    _logpow(u, n, m, lu, um)
    if kind == 0:
        _horner_vec(S[0], L[0], u, n, h)
        for t in range(n):
            out[t] = um[t] * h[t]
        return
    o = 1 if kind == 1 else 4
    _horner_vec(S[o + 2], L[o + 2], u, n, h)               # C part
    has_log = L[o + 1] > 1 or S[o + 1, 0] != 0.0
    if has_log:
        hl = np.empty(n)
        _horner_vec(S[o + 1], L[o + 1], u, n, hl)
        for t in range(n):
            h[t] += lu[t] * hl[t]
    _horner_vec(S[o], L[o], u, n, out)                      # regular part
    for t in range(n):
        out[t] += um[t] * h[t]


@njit(cache=True)
def columns_kernel(z, p, gams, sertabs, tol, Gt, cheb, lens, valid, ycs, plo, phi,
                   ju, jw, has, out, col0, nreg=-1):
    """out[col0 + i, :] = blocked flux / pi of (1 - r^2)^gams[i] at separations z.

    cheb (nt, DEG+1, 7, NB+1): Chebyshev coefficients of the expansions in p on
    [plo, phi] (ignored if plo < 0); lens (nt, 7), valid (nt,), ycs (nt,): their
    lengths, validity and reach of piece C. Interior samples: interior series;
    limb samples: the expansions (vectorised per piece) or quadrature."""
    n = z.shape[0]
    nt = gams.shape[0]
    one_k = 1.0 + p
    zh = math.sqrt((1.0 - p) * (1.0 + p))
    use_panel = plo >= 0.0
    x = (2.0 * p - plo - phi) / (phi - plo) if use_panel else 0.0
    S = np.empty((_NS7, NB + 1))
    one = np.ones(1)
    ipi = 1.0 / math.pi
    # piece arguments are exponent-independent; the interior cut and the reach of
    # piece C are not, so the samples are classified per exponent
    for i in range(nt):
        g = gams[i]
        gi = gams[i:i + 1]
        zc = _cut_terms(p, gi, one, tol, Gt)
        a = _series.coeffs(sertabs[i], p, p * p)
        has_c = use_panel and valid[i]
        if has_c:
            _clenshaw_coefs(cheb[i], x, S, lens[i])
        yc = ycs[i]
        m = g + 1.5
        cnt = np.zeros(5, dtype=np.int64)        # interior, A, B, C, quadrature
        kind = np.empty(n, dtype=np.int8)
        arg = np.empty(n)
        for j in range(n):
            zj = abs(z[j])
            if zj <= zc:
                kind[j] = 0
            elif zj >= one_k:
                kind[j] = 5
                out[col0 + i, j] = 0.0
                continue
            elif not has_c:
                kind[j] = 4
            elif zj >= zh:
                zmp = zj - p
                kind[j] = 1
                arg[j] = (1.0 - zmp) * (1.0 + zmp) / (4.0 * zj * p)
            elif zj >= 1.0 - p:
                zpp = zj + p
                kind[j] = 2
                arg[j] = (zpp - 1.0) * (zpp + 1.0) / (4.0 * zj * p)
            elif zj > p:
                zmp = zj - p
                zpp = zj + p
                yy = (1.0 - zpp) * (1.0 + zpp) / ((1.0 - zmp) * (1.0 + zmp))
                if yy <= yc:
                    kind[j] = 3
                    arg[j] = yy
                else:
                    kind[j] = 4
            else:
                kind[j] = 4
            cnt[kind[j]] += 1
        # interior series
        ms = cnt[0]
        if ms > 0:
            zb = np.empty(ms)
            idx = np.empty(ms, dtype=np.int64)
            q = 0
            for j in range(n):
                if kind[j] == 0:
                    zb[q] = abs(z[j])
                    idx[q] = j
                    q += 1
            fs = np.empty(ms)
            _series.series_pow(zb, ms, a, g, 0.0, 0.0, fs)
            for q in range(ms):
                out[col0 + i, idx[q]] = -fs[q]
        # contact pieces, vectorised per piece
        for kd in range(1, 4):
            nk = cnt[kd]
            if nk == 0:
                continue
            u = np.empty(nk)
            idx = np.empty(nk, dtype=np.int64)
            q = 0
            for j in range(n):
                if kind[j] == kd:
                    u[q] = arg[j]
                    idx[q] = j
                    q += 1
            vals = np.empty(nk)
            _pieces_vec(kd - 1, u, nk, m, S, lens[i], vals, nreg)
            for q in range(nk):
                out[col0 + i, idx[q]] = vals[q] * ipi
        # quadrature for the rest (has[i] == -1: Gauss-Jacobi rule not built yet)
        if cnt[4] > 0:
            if has[i] == -1:
                u_, w_ = _general.gauss_jacobi01(_general.N_NODES, g + 1.0)
                ju[i] = u_
                jw[i] = w_
                has[i] = 1
            for j in range(n):
                if kind[j] == 4:
                    b = _general.blocked_limb(abs(z[j]), p, gi, one, _general.GL_U, _general.GL_W,
                                              ju[i:i + 1], jw[i:i + 1], has[i:i + 1])
                    out[col0 + i, j] = b * ipi


# ----------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------
def _validation_z(p, yc):
    """Separations for validation: the whole limb range used by the expansions
    (piece C down to y' = yc), dense near both contacts and the piece borders."""
    zlo = _z_of_yprime(p, yc)
    zh = math.sqrt((1 - p) * (1 + p))
    d = np.geomspace(1e-13, 1e-2, 25) * p
    z = np.concatenate([np.linspace(zlo, 1 + p, 160)[1:-1], 1 + p - d, 1 - p + d, 1 - p - d,
                        zh + d, zh - d, [zlo, zlo * (1 + 1e-9)]])
    return np.unique(z[(z >= zlo) & (z < 1 + p)])


def _z_of_yprime(p, yc):
    """Separation z < 1 - p at which y' = (1 - (z+p)^2) / (1 - (z-p)^2) equals yc."""
    if yc >= 1.0:
        return p * (1.0 + 1e-9)
    lo, hi = p, 1.0 - p
    for _ in range(80):
        zm = 0.5 * (lo + hi)
        y = (1 - (zm + p) ** 2) / (1 - (zm - p) ** 2)
        lo, hi = (zm, hi) if y > yc else (lo, zm)
    return hi


def _yprime(z, p):
    return (1 - (z + p) ** 2) / (1 - (z - p) ** 2)


class PowerColumns:
    """Blocked fluxes / pi of the terms (1 - r^2)^g, g in ``gams`` (fixed, non-integer),
    with panels of contact-expansion coefficients built lazily per radius ratio."""

    def __init__(self, gams, tol=1e-12):
        self.gams = np.ascontiguousarray(gams, dtype=float)
        self.nt = len(self.gams)
        self.tol = float(tol)
        self.sertabs = series_tables(self.gams)
        self.ju, self.jw, self.has = _general.quad_rules(self.gams)
        self._ltabs = {}
        self.panels = {}
        self._empty = (np.zeros((self.nt, DEG + 1, _NS7, NB + 1)), np.zeros((self.nt, _NS7), np.int64),
                       np.zeros(self.nt, np.bool_), np.zeros(self.nt), -1.0, -1.0)

    # -- panel construction --------------------------------------------------
    def _tables(self, g, N):
        key = (g, N)
        if key not in self._ltabs:
            self._ltabs[key] = limb_tables(np.array([g]), _contact.JC, N + 1)
        return self._ltabs[key]

    def _zc(self, p, g):
        return _cut_terms(p, np.array([g]), np.ones(1), self.tol, _series._GTAB)

    def _build_one(self, g, lo, hi, N, ycap):
        """Chebyshev coefficients (DEG+1, 7, NB+1) for exponent g on [lo, hi] with
        series of degree N and piece C reaching y' <= yc; lengths; validation error."""
        ps, x = _cheb_nodes(lo, hi, DEG)
        tabs, lens = self._tables(g, N)
        yc = ycap
        if ycap >= 1.0:          # down to the interior cut over the whole panel
            yc = max(_yprime(max(self._zc(p, g), p * 1.001), p) for p in (lo, hi))
            yc = min(yc, 0.999)
        V = np.zeros((DEG + 1, _NS7, NB + 1))
        for j, p in enumerate(ps):
            RA, B, C, est = _contact.pieces(g, p, tabs[0], lens[0], N, yc)
            if not np.all(np.isfinite(RA)) or not (np.all(np.isfinite(B)) and np.all(np.isfinite(C))):
                return None
            V[j, 0, :N + 1] = RA
            V[j, 1:4, :N + 1] = B
            V[j, 4:7, :N + 1] = C
        Tm = np.cos(np.outer(np.arange(DEG + 1), np.arccos(x)))
        cheb = np.linalg.solve(Tm.T, V.reshape(DEG + 1, -1)).reshape(DEG + 1, _NS7, NB + 1)
        # effective lengths: drop trailing terms negligible at the largest argument
        umax = np.array([0.5, 0.5, 0.5, 0.5, yc, yc, yc])
        L = np.zeros(_NS7, np.int64)
        for s in range(_NS7):
            w = np.abs(V[:, s, :]) * umax[s] ** np.arange(NB + 1)
            tot = w.sum(axis=1).max()
            keep = np.nonzero(w.max(axis=0) > 1e-18 * max(tot, 1e-300))[0]
            L[s] = keep[-1] + 1 if keep.size else 1
        return cheb, L, yc

    def _validate(self, g, lo, hi, cheb, L, yc):
        ps, _ = _cheb_nodes(lo, hi, DEG)
        test = np.concatenate([ps[::4], lo + (hi - lo) * np.array([0.0, 0.13, 0.5, 0.77, 0.999])])
        gi = np.array([g])
        one = np.ones(1)
        i = int(np.nonzero(self.gams == g)[0][0])
        ju, jw, has = self.ju[i:i + 1], self.jw[i:i + 1], self.has[i:i + 1]
        S = np.empty((_NS7, NB + 1))
        err = 0.0
        for p in test:
            _clenshaw_coefs(cheb, (2 * p - lo - hi) / (hi - lo), S, L)
            for zj in _validation_z(p, yc):
                b = _contact_point(zj, p, g + 1.5, S, L, yc)
                if b != b:          # not covered (y' > yc by rounding): quadrature at run time
                    continue
                q = _general.blocked_limb(zj, p, gi, one, _general.GL_U, _general.GL_W, ju, jw, has)
                err = max(err, abs(b - q) / math.pi)
                if err > VALID_TOL:
                    return err
        return err

    def _build_panel(self, ip):
        lo, hi = panel_range(ip)
        cheb = np.zeros((self.nt, DEG + 1, _NS7, NB + 1))
        lens = np.zeros((self.nt, _NS7), np.int64)
        valid = np.zeros(self.nt, np.bool_)
        ycs = np.zeros(self.nt)
        info = []
        for i, g in enumerate(self.gams):
            key = (float(g), self.tol, ip)
            hit = _CACHE_1D.get(key)
            if hit is not None:
                cheb[i], lens[i], valid[i], ycs[i] = hit
                info.append((g, "cached"))
                continue
            for N, ycap in ((NB, 1.0), (NB, 0.5), (_contact.NS, 0.5)):
                r = self._build_one(g, lo, hi, N, ycap)
                if r is None:
                    continue
                c, L, yc = r
                err = self._validate(g, lo, hi, c, L, yc)
                if err <= VALID_TOL:
                    cheb[i], lens[i], valid[i], ycs[i] = c, L, True, yc
                    info.append((g, N, yc, err))
                    break
            else:
                info.append((g, None, None, None))
            if len(_CACHE_1D) >= _MAX_1D:
                _CACHE_1D.clear()
            _CACHE_1D[key] = (cheb[i].copy(), lens[i].copy(), bool(valid[i]), float(ycs[i]))
        self.panels[ip] = (cheb, lens, valid, ycs, lo, hi)
        self.last_info = info
        return self.panels[ip]

    def panel(self, p):
        """(cheb, lens, valid, ycs, plo, phi) for radius ratio p (built on first use)."""
        ip = panel_index(p)
        if ip < 0:
            return self._empty
        P = self.panels.get(ip)
        if P is None:
            P = self._build_panel(ip)
        return P

    def __call__(self, z, p):
        """(n, nt) blocked flux / pi."""
        z = np.ascontiguousarray(z, dtype=float)
        out = np.empty((self.nt, len(z)))
        cheb, lens, valid, ycs, lo, hi = self.panel(float(p))
        columns_kernel(z, float(p), self.gams, self.sertabs, self.tol, _series._GTAB, cheb, lens,
                       valid, ycs, lo, hi, self.ju, self.jw, self.has, out, 0)
        return out.T


# ----------------------------------------------------------------------------
# free exponent (power-2 law with alpha sampled): 2-D panels in (p, alpha)
# ----------------------------------------------------------------------------
DEG_A = 8               # Chebyshev degree in alpha per panel
WIDTH_A = 0.25          # alpha-panel width; boundaries at the odd integers (g = 1/2, 3/2)
A_MAX = 3.0             # panels for 0 < alpha < A_MAX (quadrature beyond)


def alpha_index(alpha):
    if not (0.0 < alpha < A_MAX):
        return -1
    return int(math.floor(alpha / WIDTH_A))


def alpha_range(ia):
    return ia * WIDTH_A, (ia + 1) * WIDTH_A


@njit(cache=True, fastmath=True)
def _clenshaw_outer(cc2, x, out, L):
    """out[j, s, :L[s]] = sum_k cc2[k, j, s, :L[s]] T_k(x): collapse the alpha axis."""
    nk = cc2.shape[0]
    nj = cc2.shape[1]
    nc = cc2.shape[3]
    b1 = np.zeros(nc)
    b2 = np.zeros(nc)
    x2 = 2.0 * x
    for j in range(nj):
        for s in range(cc2.shape[2]):
            Ls = L[s]
            for i in range(Ls):
                b1[i] = 0.0
                b2[i] = 0.0
            for k in range(nk - 1, 0, -1):
                c = cc2[k, j, s]
                for i in range(Ls):
                    t = x2 * b1[i] - b2[i] + c[i]
                    b2[i] = b1[i]
                    b1[i] = t
            c = cc2[0, j, s]
            o = out[j, s]
            for i in range(Ls):
                o[i] = x * b1[i] - b2[i] + c[i]


@njit(cache=True)
def alpha_column(z, p, alpha, tol, Gt, cheb2, lens, valid, yc, plo, phi, alo, ahi, nreg, out, row):
    """out[row, :] = blocked flux / pi of mu^alpha (alpha free): the alpha axis of
    the 2-D panel cheb2 (DEG_A+1, DEG+1, 7, NB+1) on [alo, ahi] x [plo, phi] is
    collapsed at alpha, then as columns_kernel (regularised pieces, nreg)."""
    g = 0.5 * alpha
    gams = np.array([g])
    sert = np.empty((1, _series.M + 1, _series.M + 1))
    sert[0] = _series.table(g, _series.M)
    ju = np.zeros((1, _general.N_NODES))
    jw = np.zeros((1, _general.N_NODES))
    has = np.zeros(1, dtype=np.int64)
    if g != math.floor(g):
        has[0] = -1                     # rule built on demand by columns_kernel
    c1 = np.zeros((1, DEG + 1, _NS7, NB + 1))
    v = np.zeros(1, dtype=np.bool_)
    ycs = np.full(1, yc)
    L1 = np.zeros((1, _NS7), dtype=np.int64)
    if valid and plo >= 0.0:
        _clenshaw_outer(cheb2, (2.0 * alpha - alo - ahi) / (ahi - alo), c1[0], lens)
        v[0] = True
        L1[0] = lens
    columns_kernel(z, p, gams, sert, tol, Gt, c1, L1, v, ycs, plo, phi, ju, jw, has, out, row,
                   nreg)


class AlphaColumns:
    """Blocked flux / pi of mu^alpha with alpha free: panels of contact-expansion
    coefficients on (p, alpha) rectangles, built lazily and validated against
    quadrature like ``PowerColumns``. Near alpha = 1, 3 (g half-integer) the pieces
    B and C have poles ~ 1/(g - n + 3/2) that cancel in their sum; they are stored in
    the regularised form Rt = R + y^n C, Ct = d C (d = g + 3/2 - n, n fixed per
    panel), which is analytic in alpha (see ``_pieces_vec``)."""

    def __init__(self, tol=1e-12):
        self.tol = float(tol)
        self.panels = {}
        self._ltabs = {}
        self._empty = (np.zeros((DEG_A + 1, DEG + 1, _NS7, NB + 1)), np.zeros(_NS7, np.int64),
                       False, 0.0, -1.0, -1.0, -1.0, -1.0, -1)

    def _tables(self, g, N):
        key = (g, N)
        if key not in self._ltabs:
            if len(self._ltabs) > 4096:
                self._ltabs.clear()
            self._ltabs[key] = limb_tables(np.array([g]), _contact.JC, N + 1)
        return self._ltabs[key]

    def _zc(self, p, g):
        return _cut_terms(p, np.array([g]), np.ones(1), self.tol, _series._GTAB)

    def _node_pieces(self, g, p, n, N, yc):
        """Regularised series (7, NB+1) at (g, p): RA, Rt_B, Ct_B, 0, Rt_C, Ct_C, 0."""
        tabs, lens = self._tables(g, N + n)
        RA, B, C, est = _contact.pieces(g, p, tabs[0], lens[0], N + n, yc)
        d = g + 1.5 - n
        V = np.zeros((_NS7, NB + 1))
        V[0, :N + 1] = RA[:N + 1]
        for o, X in ((1, B), (4, C)):
            R, Lg, Cp = X
            Rt = R.copy()
            Rt[n:] += Cp[:N + n + 1 - n]
            Ct = Lg if d == 0.0 else d * Cp
            V[o, :N + n + 1] = Rt
            V[o + 1, :N + 1] = Ct[:N + 1]
        return V

    def _build(self, ip, ia, N, ycap):
        plo, phi = panel_range(ip)
        alo, ahi = alpha_range(ia)
        n = int(round(0.5 * (alo + ahi) / 2.0 + 1.5))
        ps, xp = _cheb_nodes(plo, phi, DEG)
        als, xa = _cheb_nodes(alo, ahi, DEG_A)
        yc = ycap
        if ycap >= 1.0:
            yc = max(_yprime(max(self._zc(p_, a_ / 2.0), p_ * 1.001), p_)
                     for p_ in (plo, phi) for a_ in (alo, ahi))
            yc = min(yc, 0.999)
        V = np.zeros((DEG_A + 1, DEG + 1, _NS7, NB + 1))
        for k, a_ in enumerate(als):
            for j, p_ in enumerate(ps):
                V[k, j] = self._node_pieces(a_ / 2.0, p_, n, N, yc)
        if not np.all(np.isfinite(V)):
            return None
        Tp = np.cos(np.outer(np.arange(DEG + 1), np.arccos(xp)))
        Ta = np.cos(np.outer(np.arange(DEG_A + 1), np.arccos(xa)))
        cheb = np.einsum("ka,jb,abi->kji", np.linalg.inv(Ta.T), np.linalg.inv(Tp.T),
                         V.reshape(DEG_A + 1, DEG + 1, -1)).reshape(V.shape)
        umax = np.array([0.5, 0.5, 0.5, 0.5, yc, yc, yc])
        L = np.zeros(_NS7, np.int64)
        for s_ in range(_NS7):
            w = np.abs(V[:, :, s_, :]).reshape(-1, NB + 1) * umax[s_] ** np.arange(NB + 1)
            tot = w.sum(axis=1).max()
            keep = np.nonzero(w.max(axis=0) > 1e-18 * max(tot, 1e-300))[0]
            L[s_] = keep[-1] + 1 if keep.size else 1
        return np.ascontiguousarray(cheb), L, yc, n

    def _validate(self, ip, ia, cheb, L, yc, n):
        plo, phi = panel_range(ip)
        alo, ahi = alpha_range(ia)
        rng = np.random.default_rng(ip * 1000 + ia)
        pts = [(plo, alo), (phi * (1 - 1e-12), ahi * (1 - 1e-12)), (plo, ahi * (1 - 1e-12)),
               (phi * (1 - 1e-12), alo)] + [(plo + (phi - plo) * a, alo + (ahi - alo) * b)
                                            for a, b in rng.random((5, 2))]
        out = np.empty((1, 0))
        err = 0.0
        for p_, a_ in pts:
            z = _validation_z(p_, yc)
            out = np.empty((1, len(z)))
            alpha_column(z, p_, a_, self.tol, _series._GTAB, cheb, L, True, yc, plo, phi, alo, ahi,
                         n, out, 0)
            ju, jw, has = _general.quad_rules(np.array([a_ / 2.0]))
            q = np.array([_general.blocked_limb(zj, p_, np.array([a_ / 2.0]), np.ones(1),
                                                _general.GL_U, _general.GL_W, ju, jw, has)
                          for zj in z]) / math.pi
            err = max(err, float(np.max(np.abs(out[0] - q))))
            if not err <= VALID_TOL:
                return np.inf if err != err else err
        return err

    def _build_panel(self, key):
        ip, ia = key
        hit = _CACHE_2D.get((self.tol,) + key)
        if hit is not None:
            self.panels[key] = hit
            return hit
        res = None
        for N, ycap in ((NB - 3, 1.0), (NB - 3, 0.5), (_contact.NS, 0.5)):
            r = self._build(ip, ia, N, ycap)
            if r is None:
                continue
            cheb, L, yc, n = r
            err = self._validate(ip, ia, cheb, L, yc, n)
            if err <= VALID_TOL:
                plo, phi = panel_range(ip)
                alo, ahi = alpha_range(ia)
                res = (cheb, L, True, yc, plo, phi, alo, ahi, n)
                self.last_info = (key, N, yc, err)
                break
        if res is None:
            self.last_info = (key, None)
            res = self._empty
        self.panels[key] = res
        if len(_CACHE_2D) >= _MAX_2D:
            _CACHE_2D.clear()
        _CACHE_2D[(self.tol,) + key] = res
        return res

    def panel(self, p, alpha):
        ip, ia = panel_index(p), alpha_index(alpha)
        if ip < 0 or ia < 0:
            return self._empty
        key = (ip, ia)
        P = self.panels.get(key)
        return P if P is not None else self._build_panel(key)

    def __call__(self, z, p, alpha):
        z = np.ascontiguousarray(z, dtype=float)
        out = np.empty((1, len(z)))
        cheb, L, valid, yc, plo, phi, alo, ahi, n = self.panel(float(p), float(alpha))
        alpha_column(z, float(p), float(alpha), self.tol, _series._GTAB, cheb, L, valid, yc,
                     plo, phi, alo, ahi, n, out, 0)
        return out[0]
