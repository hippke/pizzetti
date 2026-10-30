"""Transit fits with the limb darkening and the flux baseline integrated out.

``MarginalTransitModel`` wraps the analytic marginalisation (see ``_marginal``):
the sampler only explores the geometry (t0, per, rp, a, b [, ecc, w]); the limb
darkening and the out-of-transit flux are integrated out exactly over their
physical region (Kipping 2013 conditions) and drawn afterwards for every
posterior sample. Two ways of use:

Layer 1, emcee-native (a subclass of ``emcee.EnsembleSampler``)::

    model = pizzetti.MarginalTransitModel(params, t, y, yerr, free=["t0", "per", "rp", "a", "b"])
    sampler = model.EnsembleSampler(32)
    sampler.run_mcmc(p0, 3000)                       # p0 may still carry q1, q2 columns
    chain = sampler.get_chain(discard=1000, flat=True)   # t0, per, rp, a, b, q1, q2, f0

Layer 2, one call (profile importance sampling with emcee fallback)::

    result = model.fit()
    result.samples, result.labels, result.summary(), result.diagnostics

Laws: uniform, linear, quadratic, square-root, logarithmic, power-2 (alpha fixed
or free) and nonlinear, plus secondary eclipses (``transittype="secondary"``).
The coefficients are linear in the flux; in a Bernstein basis the physical region
(positive, limb-darkened profile) is a cone, and the Gaussian integral over it is
closed-form up to an orthant probability (quadratic: 0 <= c0 <= c1 <= c2,
Kipping 2013). The nonlinear law's region is curved; its probability is computed
by line integration (``_physnl``) and enters the likelihood in the same way.

Priors: uniform within ``bounds`` on the free parameters (plus 0 <= b < 1 + rp,
0 <= ecc < 1), an optional vectorised ``log_prior`` callback on the physical
parameters, and ``ld_prior``: "physical" (default; flat over the law's physical
region x flat baseline f0 -- Kipping's (q1, q2) prior for the quadratic law),
"flat" (no constraint), a Gaussian dict (quadratic law, exact) or a callable
(importance reweighting).

emcee and scipy are only needed for sampling (``pip install pizzetti[fit]``).
"""
import math

import numpy as np
from numba import njit

from . import _laws, _orbit, _physnl, _powbasis
from . import _marginal as K
from . import _series
from .terms import _A_HALF
from .transitmodel import _exposure_offsets

CANON = K.CANON
_NO_ROWPAN2 = np.full(1, -1, dtype=np.int64)
_NO_ASTACK = (np.zeros((1, _powbasis.DEG_A + 1, _powbasis.DEG + 1, 7, _powbasis.NB + 1)),
              np.zeros((1, 7), np.int64), np.zeros(1, np.bool_), np.zeros(1), np.full(1, -1.0),
              np.full(1, -1.0), np.full(1, -1.0), np.full(1, -1.0), np.full(1, -1, np.int64))
_DEG = math.pi / 180.0


@njit(cache=True)
def _gibbs_cone_one(mu, cov, rng_u, sweeps):
    """One draw of c ~ N(mu, cov) truncated to 0 <= c0 <= c1 <= c2 (Gibbs in
    w = D c with inverse-CDF truncated-normal updates). rng_u: uniforms (sweeps*3)."""
    D = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    m = D @ mu
    S = D @ cov @ D.T
    P = np.linalg.inv(S)
    w = np.maximum(m, 1e-12 * math.sqrt(S[2, 2]))
    k = 0
    for _ in range(sweeps):
        for i in range(3):
            sd = 1.0 / math.sqrt(P[i, i])
            mi = m[i]
            for j in range(3):
                if j != i:
                    mi -= P[i, j] * (w[j] - m[j]) / P[i, i]
            lo = K.ndtr((0.0 - mi) / sd)
            u = lo + rng_u[k] * (1.0 - lo)
            k += 1
            u = min(max(u, 1e-300), 1.0 - 1e-16)
            w[i] = mi + sd * K.ndtri(u)
    c = np.empty(3)
    c[0] = w[0]
    c[1] = w[0] + w[1]
    c[2] = w[0] + w[1] + w[2]
    return c


def systematic_resample(w, n, rng):
    """Indices of n equal-weight draws from weights w (systematic resampling:
    each index appears floor(n w_i) or ceil(n w_i) times -- no extra Monte Carlo
    duplication, unlike multinomial resampling with replacement)."""
    cw = np.cumsum(w)
    cw /= cw[-1]
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(cw, u), len(w) - 1)


def _draw_c(C, S, rng):
    """One truncated-Gaussian draw of c per row (vectorised rejection, Gibbs fallback)."""
    n = C.shape[0]
    out = np.empty((n, 3))
    todo = np.arange(n)
    L = np.linalg.cholesky(S)
    for _ in range(60):
        if todo.size == 0:
            break
        z = rng.standard_normal((todo.size, 8, 3))
        cand = C[todo, None, :] + np.einsum("nij,nkj->nki", L[todo], z)
        ok = (cand[:, :, 0] >= 0) & (cand[:, :, 1] >= cand[:, :, 0]) & (cand[:, :, 2] >= cand[:, :, 1])
        has = ok.any(axis=1)
        first = np.argmax(ok, axis=1)
        out[todo[has]] = cand[has, first[has]]
        todo = todo[~has]
    for i in todo:                                         # rare: cone far in the tail
        out[i] = _gibbs_cone_one(C[i], S[i], rng.random(300 * 3), 300)
    return out


class _BoundedCoords:
    """Unbounded coordinates u for the internal coordinates x of a model, for the retry of
    profile importance sampling when the posterior presses against a prior bound of a
    parameter other than b (the profiled b has walls of its own): x = lo + (hi - lo) sigmoid(u)
    for a box, x = lo + exp(u) or hi - exp(u) for a one-sided bound, x = u otherwise. With the T14
    reparametrisation the bounds on a/R* (and a > b) become bounds on T14 that depend on
    (per, rp, b) of the same row; T14 is mapped last, after those. log|dx/du| is the Jacobian."""

    def __init__(self, model):
        self.m = model
        self.cols = []                          # (column, kind, lo, hi); kind: box/lo/hi/a14
        for j, q in enumerate(model.free):
            if q == "b":
                continue
            if q == "alpha":
                lo, hi = model.alpha_lo, model.alpha_hi
            else:
                k = CANON.index("t0" if q == "t_secondary" else q)
                lo, hi = model.lo[k], model.hi[k]
                if q == "w":
                    lo, hi = lo / _DEG, hi / _DEG
                if q in ("per", "rp"):
                    lo = max(lo, 0.0)
                if q == "ecc":
                    lo, hi = max(lo, 0.0), min(hi, 1.0)
            if q == "a" and model.t14:
                self.cols.append((j, "a14", max(model.lo[3], 1.0), model.hi[3]))
            elif np.isfinite(lo) and np.isfinite(hi):
                self.cols.append((j, "box", lo, hi))
            elif np.isfinite(lo):
                self.cols.append((j, "lo", lo, hi))
            elif np.isfinite(hi):
                self.cols.append((j, "hi", lo, hi))
        self.active = bool(self.cols)

    def _t14_range(self, X):
        """Rows of internal x (T14 slot ignored) -> (T14 at a_hi, T14 at a_lo)."""
        m = self.m
        P = m._canonical_physical(m.to_physical(X))         # per, rp, b of the rows
        per, rp, b = P[:, 1], P[:, 2], P[:, 4]
        Q = np.clip((1.0 + rp) ** 2 - b * b, 0.0, None)
        _, _, a_lo, a_hi = next(c for c in self.cols if c[1] == "a14")

        def t14(a):
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.sqrt(Q / np.clip(a * a - b * b, 1e-300, None))
            return per / math.pi * np.arcsin(np.clip(r, 0.0, 1.0))
        lo = t14(np.full(len(X), a_hi)) if np.isfinite(a_hi) else np.zeros(len(X))
        hi = t14(np.full(len(X), a_lo))
        return lo, hi

    def to_x(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        X = U.copy()
        lj = np.zeros(len(U))
        a14 = None
        for j, kind, lo, hi in self.cols:
            u = U[:, j]
            if kind == "box":
                s = 0.5 * (1.0 + np.tanh(0.5 * u))
                X[:, j] = lo + (hi - lo) * s
                lj += np.log((hi - lo) * np.clip(s * (1.0 - s), 1e-300, None))
            elif kind == "lo":
                X[:, j] = lo + np.exp(u)
                lj += u
            elif kind == "hi":
                X[:, j] = hi - np.exp(u)
                lj += u
            else:
                a14 = j
        if a14 is not None:
            tlo, thi = self._t14_range(X)
            s = 0.5 * (1.0 + np.tanh(0.5 * U[:, a14]))
            X[:, a14] = tlo + (thi - tlo) * s
            lj += np.log(np.clip((thi - tlo) * s * (1.0 - s), 1e-300, None))
        return X, lj

    def to_u(self, X, margin=1e-6):
        """Internal x -> u; points on (or beyond) a bound are moved inside by a small margin."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U = X.copy()
        for j, kind, lo, hi in self.cols:
            x = X[:, j]
            if kind == "box":
                s = np.clip((x - lo) / (hi - lo), margin, 1.0 - margin)
                U[:, j] = np.log(s / (1.0 - s))
            elif kind == "lo":
                U[:, j] = np.log(np.clip(x - lo, margin * max(abs(lo), 1.0), None))
            elif kind == "hi":
                U[:, j] = np.log(np.clip(hi - x, margin * max(abs(hi), 1.0), None))
            else:
                tlo, thi = self._t14_range(X)
                s = np.clip((x - tlo) / (thi - tlo), margin, 1.0 - margin)
                U[:, j] = np.log(s / (1.0 - s))
        return U

    def log_prob(self, U):
        X, lj = self.to_x(U)
        out = self.m.log_prob_internal(X)
        return np.where(np.isfinite(out), out + lj, -np.inf)


class MarginalTransitModel:
    """Transit (or secondary-eclipse) likelihood with the limb darkening and the
    baseline flux integrated out.

    :param params: ``pizzetti.TransitParams`` (initial values / fixed values).
        ``limb_dark`` selects the law: "uniform", "linear", "quadratic",
        "squareroot", "logarithmic", "power2" or "nonlinear" (``u`` is ignored --
        the coefficients are free and integrated out; for "power2" ``u[1]`` is the
        exponent alpha, fixed unless "alpha" is in ``free``). For secondary eclipses
        the limb darkening is irrelevant and ``params.t_secondary`` is used.
    :param t, y: times and fluxes
    :param yerr: flux uncertainties (scalar or array)
    :param free: names of the sampled parameters, a subset of
        ("t0", "per", "rp", "a", "b", "ecc", "w") (b = a cos(inc), w in degrees),
        plus "alpha" for the power-2 law; for transittype="secondary" use
        "t_secondary" instead of "t0"
    :param bounds: dict name -> (lo, hi) of uniform priors (defaults are wide)
    :param log_prior: optional vectorised callback ``f(dict of arrays) -> array``
        on the physical parameters
    :param ld_prior: prior on the linear (limb-darkening / eclipse) coefficients:
        "physical" (default): flat in the law's coefficients over its physical region
        (positive intensity, darkening towards the limb; for the quadratic and
        square-root laws this is Kipping 2013's prior) x flat baseline; for eclipses
        flat in fp >= 0 x flat baseline.
        "flat": no physical constraint; flat in the absolute linear coefficients
        (always proper -- a flat prior in the normalised coefficients u can be improper,
        e.g. for the nonlinear law, when the data allow I(1) -> 0).
        A dict {"mean": (u1, u2), "cov": 2x2} (quadratic law): physical region x
        Gaussian, included exactly. A callable f(u1, u2, ...) -> log prior: physical
        region x f, applied by importance reweighting (fine for priors broader than
        the likelihood).
    :param supersample_factor, exp_time: as in ``TransitModel``
    :param offsets: sub-exposure placement, "batman" (default) or "midpoint"
    :param reparam: "auto" samples the total duration T14 instead of a/R*
        internally (circular orbits; prior kept uniform in a); "none" disables it
    :param transittype: "primary" or "secondary"
    """

    def __init__(self, params, t, y, yerr, free=("t0", "per", "rp", "a", "b"), bounds=None,
                 log_prior=None, ld_prior="physical", supersample_factor=1, exp_time=0.0,
                 offsets="batman", reparam="auto", series_tol=1e-9, transittype="primary"):
        if transittype not in ("primary", "secondary"):
            raise ValueError('transittype must be "primary" or "secondary"')
        self.secondary = transittype == "secondary"
        self.ttype = 2 if self.secondary else 1
        law = "secondary" if self.secondary else getattr(params, "limb_dark", "quadratic")
        self._check_supported(law, params)
        self.law = _laws.Secondary() if self.secondary else _laws.LAWS[law]()
        # -- prior on the linear coefficients
        if ld_prior is None:
            ld_prior = "physical"
        if isinstance(ld_prior, str) and ld_prior not in ("physical", "flat"):
            raise ValueError('ld_prior must be "physical", "flat", a dict or a callable')
        if isinstance(ld_prior, dict) and law != "quadratic":
            raise NotImplementedError("Gaussian ld_prior (dict) is implemented for the quadratic "
                                      "law only; use a callable for other laws")
        self.ld_prior = ld_prior
        self.physical = not (isinstance(ld_prior, str) and ld_prior == "flat")
        # -- free parameters
        self.free = list(free)
        timing = "t_secondary" if self.secondary else "t0"
        allowed = [timing] + list(CANON[1:]) + (["alpha"] if self.law.has_alpha else [])
        bad = [q for q in self.free if q not in allowed]
        if bad:
            raise ValueError(f"unknown free parameter(s) {bad}; allowed: {allowed} "
                             "(use b = a cos(inc) instead of inc; w in degrees)")
        if len(set(self.free)) != len(self.free) or not self.free:
            raise ValueError("free must be a non-empty list of distinct names")
        self.geo_cols = [j for j, q in enumerate(self.free) if q != "alpha"]
        self.geo_names = ["t0" if q == "t_secondary" else q for q in self.free if q != "alpha"]
        self.alpha_col = self.free.index("alpha") if "alpha" in self.free else -1
        self.alpha_fixed = float(params.u[1]) if (self.law.has_alpha and len(params.u) > 1) else 1.0
        # -- data
        self.t = np.ascontiguousarray(t, dtype=float)
        self.y = np.ascontiguousarray(y, dtype=float)
        yerr = np.broadcast_to(np.asarray(yerr, dtype=float), self.t.shape)
        if np.any(~(yerr > 0)):
            raise ValueError("yerr must be positive")
        self.wts = np.ascontiguousarray(1.0 / yerr)
        self.yw = self.y * self.wts
        self.k = int(supersample_factor) if supersample_factor > 1 else 1
        if self.k > 1 and not exp_time > 0:
            raise ValueError("supersample_factor > 1 needs exp_time > 0")
        off = _exposure_offsets(exp_time, self.k, offsets)
        self.tt = np.ascontiguousarray((self.t[:, None] + off[None, :]).ravel())
        self.tol = float(series_tol)
        a, inc = float(params.a), float(params.inc)
        t_ref = float(params.t_secondary) if self.secondary else float(params.t0)
        self.fixed = np.array([t_ref, params.per, params.rp, a, a * math.cos(inc * _DEG),
                               params.ecc, params.w * _DEG], dtype=float)
        self.idx = np.array([CANON.index(q) for q in self.geo_names], dtype=np.int64)
        circular = params.ecc < 1e-5 and "ecc" not in self.free
        if reparam not in ("auto", "none"):
            raise ValueError('reparam must be "auto" or "none"')
        self.t14 = bool(reparam == "auto" and "a" in self.free and circular)
        lo = np.full(7, -np.inf)
        hi = np.full(7, np.inf)
        lo[1], lo[2], lo[3] = 0.0, 0.0, 1.0
        hi[5] = 1.0
        self.alpha_lo, self.alpha_hi = 0.0, 50.0
        for name, (l_, h_) in (bounds or {}).items():
            if name == "alpha":
                self.alpha_lo, self.alpha_hi = float(l_), float(h_)
                continue
            key = "t0" if name == "t_secondary" else name
            if key not in CANON:
                raise ValueError(f"bounds for unknown parameter {name!r}")
            j = CANON.index(key)
            lo[j], hi[j] = float(l_), float(h_)
            if name == "w":
                lo[j], hi[j] = lo[j] * _DEG, hi[j] * _DEG
        self.lo, self.hi = lo, hi
        self.log_prior = log_prior
        # Gaussian LD prior (quadratic): exact inside the likelihood (prior on beta = M u + v)
        self.has_ldp = isinstance(ld_prior, dict)
        self.mb, self.Pb = np.zeros(2), np.eye(2)
        if self.has_ldp:
            M = np.array([[-1.0, -1.0], [-0.5, 0.0]])
            mu_u = np.asarray(ld_prior["mean"], float)
            cov_u = np.asarray(ld_prior["cov"], float)
            self.mb = M @ mu_u + 1.0
            self.Pb = np.linalg.inv(M @ cov_u @ M.T)
        # fast fused path: quadratic primary transit with a physical (or Gaussian) prior
        self.fast = (law == "quadratic" and not self.secondary and self.physical)
        self.cone = self.physical and self.law.D is not None      # orthant in the likelihood
        self.curved = self.physical and self.law.D is None and not self.secondary  # nonlinear
        self.Dmat = np.ascontiguousarray(self.law.D if self.law.D is not None else np.eye(self.law.K))
        # fused general path (numba batch): every law but the fast quadratic one
        self._fused = not self.fast
        self._amode = self.law.has_alpha and self.alpha_col >= 0
        if self._fused:
            self._setup_general()
        self.labels = list(self.free)
        self.ndim = len(self.free)
        self.n_eval = 0
        self._ja = self.free.index("a") if "a" in self.free else -1

    # -- scope checks (remove when the features are implemented) -------------
    @staticmethod
    def _check_supported(law, params):
        if law in ("exponential", "custom"):
            raise NotImplementedError(
                f'limb_dark="{law}" is not supported by MarginalTransitModel (the exponential law '
                "diverges at the limb); supported: " + ", ".join(_laws.LAWS))
        if law != "secondary" and law not in _laws.LAWS:
            raise ValueError(f"unknown limb_dark {law!r}; supported: " + ", ".join(_laws.LAWS))
        if float(params.rp) < 0:
            raise NotImplementedError("MarginalTransitModel needs rp > 0; inverse transits "
                                      "(rp < 0) are not implemented yet.")

    # -- coordinates -----------------------------------------------------------
    def _alpha(self, X):
        return X[:, self.alpha_col] if self.alpha_col >= 0 else np.full(len(X), self.alpha_fixed)

    def _geo(self, X, to_rad=True):
        G = np.ascontiguousarray(X[:, self.geo_cols], dtype=float)
        if to_rad and "w" in self.geo_names:
            G = G.copy()
            G[:, self.geo_names.index("w")] *= _DEG
        return G

    def to_internal(self, X):
        """Physical free-parameter rows -> internal sampling coordinates (T14 for a)."""
        X = np.array(np.atleast_2d(X), dtype=float)
        if self.t14:
            P = self._canonical_physical(X)
            Q = (1.0 + P[:, 2]) ** 2 - P[:, 4] ** 2
            with np.errstate(invalid="ignore"):
                X[:, self._ja] = P[:, 1] / math.pi * np.arcsin(
                    np.sqrt(Q) / np.sqrt(P[:, 3] ** 2 - P[:, 4] ** 2))
        return X

    def _canonical_physical(self, X):
        P = np.repeat(self.fixed[None, :], len(X), axis=0)
        P[:, self.idx] = self._geo(X)
        P[:, 4] = np.abs(P[:, 4])
        return P

    def _canonical_internal(self, Xi):
        return K.to_canonical(self._geo(np.atleast_2d(Xi)), self.idx, self.fixed, self.t14)

    def to_physical(self, Xi):
        """Internal coordinates -> physical free-parameter rows."""
        Xi = np.atleast_2d(np.asarray(Xi, dtype=float))
        P, _ = self._canonical_internal(Xi)
        X = Xi.copy()
        X[:, self.geo_cols] = P[:, self.idx]
        if "w" in self.geo_names:
            X[:, self.geo_cols[self.geo_names.index("w")]] /= _DEG
        return X

    @staticmethod
    def _t0_from_tsec(tsec, per, ecc, w):
        """Time of inferior conjunction from the time of secondary eclipse (batman)."""
        def phase(TA):
            E = 2.0 * np.arctan(np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(TA / 2.0))
            return (E - ecc * np.sin(E)) / (2.0 * math.pi)
        return tsec + per * (phase(math.pi / 2.0 - w) - phase(3.0 * math.pi / 2.0 - w))

    # -- fused general path ------------------------------------------------------
    def _setup_general(self):
        if self.secondary:
            d, W, gams = np.ones(2), np.zeros((3, 2)), np.zeros(0)
        elif self._amode:
            # power-2, free alpha: A = d(alpha) - [lex, ldx, edx, P_alpha] @ W
            d, W, gams = np.ones(2), np.array([[1.0, 0.0], [0, 0], [0, 0], [-1.0, 1.0]]), np.zeros(0)
        else:
            d, W, gams = self.law.linmap(self.alpha_fixed)
        self.acols = _powbasis.AlphaColumns(_laws._TOL) if self._amode else None
        self._apos = {}
        self._astack = None
        self._dvec = np.ascontiguousarray(d, dtype=float)
        self._W = np.ascontiguousarray(W, dtype=float)
        self._gams = np.ascontiguousarray(gams, dtype=float)
        self.pcols = _powbasis.PowerColumns(self._gams, _laws._TOL) if self._gams.size else None
        if self.pcols is not None:
            self._qr = (self.pcols.sertabs, self.pcols.ju, self.pcols.jw, self.pcols.has)
        else:
            from . import _general
            self._qr = (np.zeros((0, _series.M + 1, _series.M + 1)),
                        np.zeros((0, _general.N_NODES)), np.zeros((0, _general.N_NODES)),
                        _general.quad_rules(np.array([0.5]))[2][:0])
        self._pan_pos = {}
        self._stack = None
        self._restack()

    def _restack(self):
        """Stack the built panels (in order of first use) for the numba kernel."""
        nt = self._gams.size
        sh = (_powbasis.DEG + 1, 7, _powbasis.NB + 1)
        self._lut = np.full(max(self._pan_pos, default=-1) + 1, -1, dtype=np.int64)
        for ip, pos in self._pan_pos.items():
            self._lut[ip] = pos
        if not self._pan_pos:
            self._stack = (np.zeros((1, nt) + sh), np.zeros((1, nt, 7), np.int64),
                           np.zeros((1, nt), np.bool_), np.zeros((1, nt)), np.full(1, -1.0),
                           np.full(1, -1.0))
            return
        Ps = [self.pcols.panels[ip] for ip in sorted(self._pan_pos, key=self._pan_pos.get)]
        self._stack = tuple(np.ascontiguousarray(np.stack([P_[k] for P_ in Ps])) for k in range(4)) + (
            np.array([P_[4] for P_ in Ps]), np.array([P_[5] for P_ in Ps]))

    def _panel_pos(self, ip):
        pos = self._pan_pos.get(ip)
        if pos is None:
            lo_, hi_ = _powbasis.panel_range(ip)
            self.pcols.panel(0.5 * (lo_ + hi_))          # build (validated) on first use
            pos = self._pan_pos[ip] = len(self._pan_pos)
            self._restack()
        return pos

    def _rowpanels(self, rp):
        """Index of every row's panel in the stack (-1: quadrature); builds new panels."""
        if len(rp) == 0:
            return np.zeros(0, dtype=np.int64)
        if len(rp) == 1:
            r = float(rp[0])
            ip = _powbasis.panel_index(r) if r == r else -1
            return np.array([self._panel_pos(ip) if ip >= 0 else -1], dtype=np.int64)
        with np.errstate(all="ignore"):
            ids = np.floor(np.log(rp / _powbasis.P_MIN) / _powbasis._LOG_RATIO)
        good = (rp >= _powbasis.P_MIN) & (rp < _powbasis.P_MAX)
        ids = np.where(good, ids, -1).astype(np.int64)
        lut = self._lut
        if ids.max() >= len(lut) or np.any(lut[ids[ids >= 0]] < 0):
            for ip in np.unique(ids[ids >= 0]):
                self._panel_pos(int(ip))
            lut = self._lut
        return np.where(ids >= 0, lut[np.clip(ids, 0, len(lut) - 1)], -1)

    def _alpha_rowpanels(self, rp, al):
        """Free alpha: index of every row's (p, alpha) panel in the 2-D stack (-1: none)."""
        out = np.full(len(rp), -1, dtype=np.int64)
        new = False
        for i in range(len(rp)):
            r, a_ = float(rp[i]), float(al[i])
            if not (r == r and a_ == a_):
                continue
            ip, ia = _powbasis.panel_index(r), _powbasis.alpha_index(a_)
            if ip < 0 or ia < 0:
                continue
            key = (ip, ia)
            pos = self._apos.get(key)
            if pos is None:
                self.acols.panel(r, a_)
                pos = self._apos[key] = len(self._apos)
                new = True
            out[i] = pos
        if new or self._astack is None:
            Ps = [self.acols.panels[k] for k in sorted(self._apos, key=self._apos.get)]
            if not Ps:
                Ps = [self.acols._empty]
            self._astack = (np.ascontiguousarray(np.stack([P_[0] for P_ in Ps])),
                            np.ascontiguousarray(np.stack([P_[1] for P_ in Ps])),
                            np.array([P_[2] for P_ in Ps], dtype=np.bool_),
                            *(np.array([P_[k] for P_ in Ps], dtype=float) for k in range(3, 8)),
                            np.array([P_[8] for P_ in Ps], dtype=np.int64))
        return out

    def _gen(self, P, lj, al, check, moments):
        P = np.ascontiguousarray(P, dtype=float)
        rp = P[:, 2]
        al = np.ascontiguousarray(al, dtype=float)
        if self.pcols is None:
            rowpan = np.full(len(P), -1, dtype=np.int64)
        else:
            rowpan = self._rowpanels(rp)
        sertabs, ju, jw, has = self._qr
        CHEB, LENS, VALID, YCS, PLO, PHI = self._stack
        args = (P, np.ascontiguousarray(lj, dtype=float), al, check, self.lo, self.hi,
                self.law.has_alpha, self.alpha_lo, self.alpha_hi, rowpan, self.tt,
                self.k, self.wts, self.yw, self.ttype, self.tol, _A_HALF, _series._GTAB,
                self._dvec, self._W, self._gams, sertabs, _laws._TOL, CHEB, LENS, VALID,
                YCS, PLO, PHI, ju, jw, has, self.Dmat, self.cone, moments)
        tail = (self.curved, _physnl.Z4_QMC, _physnl.B3_GRID, _physnl.Z4_BIG)
        if not self._amode:
            return K.gen_batch(*args, False, _NO_ROWPAN2, *_NO_ASTACK, *tail)
        rowpan2 = self._alpha_rowpanels(rp, al)
        return K.gen_batch(*args, True, rowpan2, *self._astack, *tail)

    # -- generic (power-2 with free alpha) design ---------------------------------
    def _design_row(self, prow, alpha):
        t0, per, rp, a, b, ecc, w = prow
        if self.secondary:
            t0 = float(self._t0_from_tsec(t0, per, ecc, w))
        inc = math.acos(min(abs(b) / a, 1.0))
        z = _orbit.rsky(self.tt, t0, per, a, inc, ecc, w, self.ttype)
        A = (self.law.basis if self.secondary else self.law.basis_ref)(z, rp, alpha)
        if self.k > 1:
            A = A.reshape(len(self.t), self.k, -1).mean(axis=1)
        return np.ascontiguousarray(A * self.wts[:, None])

    def _prior_ok(self, prow, alpha):
        t0, per, rp, a, b, ecc, w = prow
        return (bool(np.all((prow > self.lo) & (prow < self.hi))) and 0.0 <= b < 1.0 + rp
                and 0.0 <= ecc < 1.0 and per > 0 and rp > 0 and a > 1.0
                and (not self.law.has_alpha or self.alpha_lo < alpha < self.alpha_hi))

    # -- log-probability -------------------------------------------------------
    def log_prob_internal(self, Xi):
        """Vectorised log posterior (up to a constant) in internal coordinates."""
        Xi = np.ascontiguousarray(np.atleast_2d(Xi), dtype=float)
        if self.fast:
            out = K.loglike_batch(self._geo(Xi), self.idx, self.fixed, self.t14, self.lo, self.hi,
                                  self.tt, self.k, self.wts, self.yw, self.tol, _A_HALF,
                                  _series._GTAB, self.has_ldp, self.mb, self.Pb)
        elif self._fused:
            P, lj = self._canonical_internal(Xi)
            out = self._gen(P, lj, self._alpha(Xi), True, False)[0]
        else:
            P, lj = self._canonical_internal(Xi)
            al = self._alpha(Xi)
            out = np.full(len(Xi), -np.inf)
            for i in range(len(Xi)):
                if not (np.all(np.isfinite(P[i])) and self._prior_ok(P[i], al[i])):
                    continue
                A = self._design_row(P[i], al[i])
                out[i] = K.marg_loglike_gen(A, self.yw, self.Dmat, self.cone) + lj[i]
        self.n_eval += len(Xi)
        if self.log_prior is not None:
            ok = np.isfinite(out)
            if ok.any():
                P, _ = self._canonical_internal(Xi[ok])
                d = self._as_dict(P)
                if self.law.has_alpha:
                    d["alpha"] = self._alpha(Xi[ok])
                out[ok] += np.asarray(self.log_prior(d), dtype=float)
        return out

    def log_prob(self, theta):
        """Log posterior in physical coordinates (theta: one row or an array of rows).
        Includes the box prior and log_prior; no T14 Jacobian (physical density)."""
        X = np.atleast_2d(np.asarray(theta, dtype=float))
        if self.t14:
            Xi = self.to_internal(X)
            out = self.log_prob_internal(Xi)
            _, lj = self._canonical_internal(Xi)
            out = out - lj
        else:
            out = self.log_prob_internal(X)
        return out if np.ndim(theta) > 1 else float(out[0])

    def _as_dict(self, P):
        a, b = P[:, 3], P[:, 4]
        d = dict(per=P[:, 1], rp=P[:, 2], a=a, b=b,
                 inc=np.degrees(np.arccos(np.clip(b / a, -1, 1))), ecc=P[:, 5], w=P[:, 6] / _DEG)
        d["t_secondary" if self.secondary else "t0"] = P[:, 0]
        return d

    # -- LD (and baseline) recovery ---------------------------------------------
    def expand(self, X, ld="q", rng=None, resample=True):
        """Physical rows -> rows with the law's coefficients and the baseline appended.

        Draws the linear coefficients c for every row from their conditional
        (Gaussian, truncated to the physical cone), converts them to the law's usual
        coefficients (q1, q2 or u1, u2 for the quadratic and square-root laws; u1..
        otherwise; fp for eclipses) and f0, and applies the importance weights that
        turn the flat-in-c prior into the documented prior (flat in the law's
        coefficients x flat baseline; times ld_prior; times the exact physicality
        indicator for the nonlinear law). With ``resample`` the weighted rows are
        resampled to equal weight. Returns (samples, labels, weights).
        """
        if self.fast:
            return self._expand_fast(X, ld, rng, resample)
        rng = np.random.default_rng() if rng is None else rng
        X = np.atleast_2d(np.asarray(X, dtype=float))
        P = self._canonical_physical(X)
        al = self._alpha(X)
        Kc = self.law.K
        C = np.empty((len(X), Kc))
        S = np.empty((len(X), Kc, Kc))
        if self._fused:
            _, C, S = self._gen(P, np.zeros(len(P)), al, False, True)
        else:
            for i in range(len(X)):
                C[i], S[i] = K.moments_gen(self._design_row(P[i], al[i]), self.yw)
        disk = np.array([self.law.disk(a_) for a_ in al])
        if self.cone:
            c = self._draw_cone(C, S, rng)
        else:
            L = np.linalg.cholesky(S)
            if self.curved:
                # nonlinear, physical: the region probability is in the likelihood, so c is
                # drawn from its Gaussian truncated to the (curved) physical region, by
                # rejection with the exact test
                c, acc = self._draw_curved(C, L, al, rng)
                self.last_physical_frac = acc
            else:
                c = C + np.einsum("nij,nj->ni", L, rng.standard_normal(C.shape))
        f0 = np.einsum("nk,nk->n", disk, c)
        I1 = self.law.central(c)
        if self.physical:
            # flat in c over the cone -> flat in the law's coefficients x flat baseline
            logw = np.log(f0) - Kc * np.log(I1)
        else:
            # "flat": flat in the absolute coefficients c (always proper; a flat prior in
            # the normalised coefficients can be improper when I(1) -> 0 is allowed)
            logw = np.zeros(len(c))
        out_c, out_l = self.law.outputs(c, al, ld)
        if callable(self.ld_prior):
            uu, _ = self.law.outputs(c, al, "u")
            logw = logw + np.asarray(self.ld_prior(*uu.T), dtype=float)
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        if not np.isfinite(logw).any():
            raise RuntimeError("no physical limb-darkening draws: the data prefer an unphysical "
                               'profile (try ld_prior="flat" to inspect)')
        w = np.exp(logw - logw.max())
        w /= w.sum()
        out = np.column_stack([X, out_c, f0])
        labels = self.labels + out_l + ["f0"]
        self.last_ld_ess = float(1.0 / np.sum(w ** 2))
        if resample:
            j = systematic_resample(w, len(w), rng)
            return out[j], labels, np.full(len(w), 1.0 / len(w))
        return out, labels, w

    def _draw_curved(self, C, L, al, rng, chunk=64, max_draws=65536):
        """One draw per row from N(C, L L^T) truncated to the nonlinear law's physical
        region (exact test), by rejection; rows without an accepted draw within
        max_draws candidates are drawn by line sampling (``_physnl.sample_physical``).
        Returns (draws, acceptance rate of
        the first chunk, i.e. the mean physical probability over the rows)."""
        n, Kc = C.shape
        out = np.full((n, Kc), np.nan)
        todo = np.arange(n)
        tried = 0
        rate = np.nan
        while todo.size and tried < max_draws:
            cand = C[todo, None, :] + np.einsum("nij,nmj->nmi", L[todo],
                                                rng.standard_normal((todo.size, chunk, Kc)))
            ok = self.law.is_physical(cand.reshape(-1, Kc), None).reshape(todo.size, chunk)
            if tried == 0:
                rate = float(ok.mean())
            tried += chunk
            has = ok.any(axis=1)
            first = np.argmax(ok, axis=1)
            out[todo[has]] = cand[has, first[has]]
            todo = todo[~has]
            chunk = min(4 * chunk, 8192)
        for i in todo:                       # tiny physical probability: line sampling
            out[i] = _physnl.sample_physical(C[i], L[i] @ L[i].T, rng)
        return out, rate

    def _draw_cone(self, C, S, rng):
        D = self.law.D
        n, Kc = C.shape
        out = np.empty((n, Kc))
        todo = np.arange(n)
        L = np.linalg.cholesky(S)
        for _ in range(60):
            if todo.size == 0:
                break
            z = rng.standard_normal((todo.size, 8, Kc))
            cand = C[todo, None, :] + np.einsum("nij,nkj->nki", L[todo], z)
            ok = np.all(np.einsum("ij,nkj->nki", D, cand) >= 0.0, axis=2)
            has = ok.any(axis=1)
            first = np.argmax(ok, axis=1)
            out[todo[has]] = cand[has, first[has]]
            todo = todo[~has]
        for i in todo:
            out[i] = K.gibbs_cone_gen(C[i], S[i], np.ascontiguousarray(D), rng.random(300 * Kc), 300)
        return out

    def _expand_fast(self, X, ld="q", rng=None, resample=True):
        rng = np.random.default_rng() if rng is None else rng
        X = np.atleast_2d(np.asarray(X, dtype=float))
        P = self._canonical_physical(X)
        C, S = K.c_moments_batch(np.ascontiguousarray(P), self.tt, self.k, self.wts, self.yw,
                                 self.tol, _A_HALF, _series._GTAB, self.has_ldp, self.mb, self.Pb)
        c = _draw_c(C, S, rng)
        f0 = c @ K.DISK
        b0, b1 = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
        u1 = 2.0 * (1.0 - b1)
        u2 = 1.0 - u1 - b0
        if self.has_ldp:
            # likelihood used N(beta)/c2^2 x flat c2 (exact Gaussian prior) -> x flat f0
            logw = np.log(f0) - np.log(c[:, 2])
        else:
            logw = np.log(f0) - 3.0 * np.log(c[:, 2])    # flat c -> uniform q x flat f0
            if callable(self.ld_prior):                  # post-hoc reweighting
                logw = logw + np.asarray(self.ld_prior(u1, u2), dtype=float)
        # rows without a valid coefficient draw (extreme geometries) get weight 0, as in expand()
        with np.errstate(invalid="ignore"):
            logw = np.where(np.isfinite(logw), logw, -np.inf)
        if not np.isfinite(logw).any():
            raise RuntimeError("no valid limb-darkening draws for these geometries")
        w = np.exp(logw - logw.max())
        w /= w.sum()
        if ld == "q":
            s = u1 + u2
            with np.errstate(invalid="ignore", divide="ignore"):
                ldc = np.column_stack([s ** 2, np.where(s > 0, u1 / (2 * s), 0.5)])
            ldl = ["q1", "q2"]
        elif ld == "u":
            ldc, ldl = np.column_stack([u1, u2]), ["u1", "u2"]
        else:
            raise ValueError('ld must be "q" or "u"')
        out = np.column_stack([X, ldc, f0])
        labels = self.labels + ldl + ["f0"]
        self.last_ld_ess = float(1.0 / np.sum(w ** 2))
        if resample:
            i = systematic_resample(w, len(w), rng)
            return out[i], labels, np.full(len(w), 1.0 / len(w))
        return out, labels, w

    # -- Layer 1: emcee ---------------------------------------------------------
    def EnsembleSampler(self, nwalkers, **kwargs):
        """An ``emcee.EnsembleSampler`` subclass sampling this model (vectorised).
        kwargs are passed to emcee (moves, backend, ...)."""
        cls = _sampler_class()
        return cls(self, nwalkers, **kwargs)

    def _initial(self, p0):
        """Accept p0 with the free columns first (extra columns, e.g. q1, q2, are dropped)."""
        p0 = np.atleast_2d(np.asarray(p0, dtype=float))
        if p0.shape[1] < self.ndim:
            raise ValueError(f"p0 has {p0.shape[1]} columns, need {self.ndim} ({self.labels})")
        return self.to_internal(p0[:, :self.ndim])

    # -- Layer 2: one call ------------------------------------------------------
    def profile_is(self, n=4000, rng=None, x0=None, retry=True, last_resort=True):
        """Profile importance sampling along b (``_sampling.profile_is``) in internal coordinates.

        The walls of the b profile are b = 0 (the likelihood depends on b^2) or the bounds on b,
        just inside them (the bounds are strict). If the result is not usable (k-hat >= 0.7 or an
        incomplete profile) and other free parameters have finite bounds, the sampling is
        repeated in unbounded coordinates (logit / log of the bounded parameters, with the
        Jacobian): a posterior pressed against such a bound (e.g. a/R* for an unresolved ingress)
        breaks the Laplace steps in the original coordinates. Such posteriors are skewed in the new
        coordinates too, so the retry's draws are refined by population Monte Carlo
        (``_sampling.pmc_refine``: mixture proposals fitted to the weighted draws), which also
        covers tails that the Laplace proposal misses without k-hat noticing. The better of the two results is
        returned; ``theta`` and ``theta_map`` are internal coordinates in either case,
        ``bounded_retry`` says whether the retry was run (None), used (True) or not (False).
        If the result is still not usable (e.g. strongly non-Gaussian conditionals at very high
        S/N), ``last_resort`` refines its weighted draws by the same adaptive importance sampling
        in the original coordinates (``last_resort_pmc``: True if that result is kept); this is
        far cheaper than the emcee fallback of ``fit()`` and is kept only if it improves k-hat.
        """
        from ._sampling import pmc_refine, profile_is
        rng = np.random.default_rng() if rng is None else rng
        x0 = self.to_internal(self.fixed_physical_free())[0] if x0 is None else np.asarray(x0, float)
        jb = self.free.index("b")
        lo_b, hi_b = self.lo[4], self.hi[4]
        walls = dict(lo_wall=max(0.0, lo_b) + (1e-9 if lo_b >= 0 else 0.0),
                     hi_wall=hi_b - 1e-9 if np.isfinite(hi_b) else None)
        res = profile_is(self.log_prob_internal, x0, self._default_scales(x0), prof=jb, n=n, rng=rng,
                         **walls)
        res["bounded_retry"] = None

        def usable(r):
            return bool(r["complete"] and r["khat"] < 0.7)

        def finish(r):
            r["last_resort_pmc"] = None
            if usable(r) or not last_resort or not np.isfinite(r["khat"]) or np.sum(r["w"] > 0) <= 50:
                return r
            pm = pmc_refine(self.log_prob_internal, r["theta"], r["logw"], rng, n=n)
            r["n_eval"] += pm["n_eval"]
            r["last_resort_pmc"] = bool(pm["khat"] < r["khat"])
            if r["last_resort_pmc"]:
                r.update(theta=pm["theta"], w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"],
                         stage="pmc", pmc=pm["history"])
            return r

        T = _BoundedCoords(self) if retry else None
        if usable(res) or T is None or not T.active:
            return finish(res)
        start = res["theta_map"] if np.all(np.isfinite(res["theta_map"])) else x0
        u0 = T.to_u(start)[0]
        # scales in u: the x scales divided by |dx/du| at the start, at most 2 (logit units)
        eps = 1e-6
        X0, _ = T.to_x(u0[None])
        su = np.empty_like(u0)
        sx = self._default_scales(X0[0])
        for j in range(len(u0)):
            up = u0.copy()
            up[j] += eps
            dx = abs(T.to_x(up[None])[0][0, j] - X0[0, j]) / eps
            su[j] = min(sx[j] / dx, 2.0) if dx > 0 else sx[j]
        r2 = profile_is(T.log_prob, u0, su, prof=jb, n=n, rng=rng, **walls)
        if np.isfinite(r2["khat"]) and np.sum(r2["w"] > 0) > 50:
            # the Laplace steps are unreliable here (posterior against a bound: flat and skewed
            # in the transformed coordinates), so the proposal is adapted to the weighted draws
            pm = pmc_refine(T.log_prob, r2["theta"], r2["logw"], rng, n=n)
            r2.update(theta=pm["theta"], w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"],
                      stage="pmc", pmc=pm["history"])
            r2["n_eval"] += pm["n_eval"]
        r2["n_eval"] += res["n_eval"]
        better = usable(r2) or (not usable(res) and r2["complete"] and r2["khat"] < res["khat"])
        if not better:
            res["n_eval"] = r2["n_eval"]
            res["bounded_retry"] = False
            return finish(res)
        r2["theta"] = T.to_x(r2["theta"])[0]
        r2["theta_map"] = T.to_x(r2["theta_map"][None])[0][0]
        r2["bounded_retry"] = True
        return finish(r2)

    def fit(self, method="auto", n=4000, nwalkers=32, max_steps=100_000, ntau=50, seed=None,
            ld="q", progress=False):
        """Posterior samples in one call.

        method="is": profile importance sampling along b (needs "b" in free);
        "emcee": vectorised emcee until convergence (N > ntau * tau, tau stable to 1 %);
        "auto": importance sampling, emcee fallback if k-hat >= 0.7 or the profile is
        incomplete. Returns a ``FitResult``.
        """
        from ._sampling import hessian
        rng = np.random.default_rng(seed)
        self.n_eval = 0
        x0 = self.to_internal(self.fixed_physical_free())[0]
        scale = self._default_scales(x0)
        diag = dict(method=None)
        Xw = w = None
        if method in ("is", "auto") and "b" in self.free:
            res = self.profile_is(n=n, rng=rng, x0=x0)
            ok = bool(res["complete"] and res["khat"] < 0.7)
            diag.update(method="profile-IS", khat=res["khat"], ess=res["ess"],
                        complete=res["complete"], n_grid=res["n_grid"], is_ok=ok,
                        bounded_retry=res["bounded_retry"], last_resort_pmc=res["last_resort_pmc"])
            if ok or method == "is":
                Xw, w = self.to_physical(res["theta"]), res["w"]
            x0 = res["theta_map"]
        elif method == "is":
            raise ValueError('method="is" needs "b" in free')
        if Xw is None:
            if method not in ("emcee", "auto"):
                raise ValueError('method must be "auto", "is" or "emcee"')
            diag["fallback_reason"] = ("" if method == "emcee" else
                                       "importance sampling diagnostics failed")
            H, _ = hessian(self.log_prob_internal, x0, 1e-2 * scale)
            try:
                sd = np.sqrt(np.diag(np.linalg.inv(-H)))
            except np.linalg.LinAlgError:
                sd = scale.copy()
            bad = ~np.isfinite(sd) | (sd <= 0)          # degenerate Hessian: default scales
            sd = np.where(bad, scale, sd)
            P0 = x0 + 0.1 * sd * rng.standard_normal((nwalkers, self.ndim))
            if "b" in self.free:
                P0[:, self.free.index("b")] = np.abs(P0[:, self.free.index("b")])
            # walkers outside the prior (e.g. x0 on a bound) are drawn again, closer to x0
            for shrink in (0.5, 0.25, 0.1, 0.03, 0.01):
                bad = ~np.isfinite(self.log_prob_internal(P0))
                if not bad.any():
                    break
                P0[bad] = x0 + shrink * 0.1 * sd * rng.standard_normal((int(bad.sum()), self.ndim))
                if "b" in self.free:
                    P0[bad, self.free.index("b")] = np.abs(P0[bad, self.free.index("b")])
            sampler = _sampler_class()(self, nwalkers, internal=True)
            # emcee draws its moves from a numpy RandomState: seed it from rng (reproducible fits)
            sampler.random_state = np.random.RandomState(int(rng.integers(2 ** 32))).get_state()
            old, conv = np.inf, False
            for _ in sampler.sample(P0, iterations=max_steps, progress=progress):
                if sampler.iteration % 500:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                if (np.all(np.isfinite(tau)) and np.all(ntau * tau < sampler.iteration)
                        and np.all(np.abs(old - tau) / tau < 0.01)):
                    conv = True
                    break
                old = tau
            tau = sampler.get_autocorr_time(tol=0)
            # tau is NaN for a parameter that has not moved (e.g. walkers stuck at a bound)
            burn = int(5 * np.max(tau)) if np.all(np.isfinite(tau)) else sampler.iteration // 2
            ch = sampler.get_chain(discard=min(burn, sampler.iteration // 2), flat=True, expand=False)
            ch = ch[:: max(1, len(ch) // n)]
            Xw, w = ch, np.full(len(ch), 1.0 / len(ch))
            diag.update(method="emcee" if method == "emcee" else "emcee (fallback)",
                        converged=conv, steps=int(sampler.iteration), tau=tau.tolist())
        diag["n_eval"] = int(self.n_eval)
        # coefficients + baseline for every weighted geometry draw; one combined weight,
        # then a single systematic resampling to equal weight. Draws with zero weight (outside
        # the prior, e.g. beyond b = 1 + rp for a grazing transit, where the design is
        # degenerate) are dropped first.
        pos = w > 0
        Xw, w = Xw[pos], w[pos]
        S_, labels, wl = self.expand(Xw, ld=ld, rng=rng, resample=False)
        diag["ld_reweight_ess"] = float(1.0 / np.sum(wl ** 2))
        if self.physical and self.law.D is None:
            diag["physical_fraction"] = self.last_physical_frac
        wc = w * wl
        wc /= wc.sum()
        diag["ess_total"] = float(1.0 / np.sum(wc ** 2))
        j = systematic_resample(wc, len(wc), rng)
        return FitResult(S_[j], labels, weighted=(S_, wc), diagnostics=diag)

    def fixed_physical_free(self):
        X = np.empty((1, self.ndim))
        X[0, self.geo_cols] = self.fixed[self.idx]
        if "w" in self.geo_names:
            X[0, self.geo_cols[self.geo_names.index("w")]] /= _DEG
        if self.alpha_col >= 0:
            X[0, self.alpha_col] = self.alpha_fixed
        return X

    def _default_scales(self, x0):
        P = self.fixed
        T14 = P[1] / math.pi * math.asin(min(1.0, (1 + P[2]) / max(P[3], 1.0 + P[2] + 1e-9)))
        sc = {"t0": 1e-2 * T14, "t_secondary": 1e-2 * T14, "per": 1e-6 * P[1], "rp": 1e-2 * P[2],
              "a": (1e-2 * T14 if self.t14 else 1e-2 * P[3]), "b": 0.05, "ecc": 0.01, "w": 1.0,
              "alpha": 0.05}
        return np.array([sc[q] for q in self.free])


class FitResult:
    """Result of ``MarginalTransitModel.fit``: equal-weight ``samples`` (emcee-like
    flat array, systematic resampling) with ``labels``; ``weighted_samples`` and
    ``weights`` are the exact weighted draws (preferred for histograms / corner plots:
    resampled draws repeat values in proportion to their weights)."""

    def __init__(self, samples, labels, weighted, diagnostics):
        self.samples = samples
        self.labels = labels
        self.weighted_samples, self.weights = weighted
        self.diagnostics = diagnostics

    def summary(self):
        """Table of median and 16th/84th-percentile offsets for every column."""
        q = np.percentile(self.samples, [16, 50, 84], axis=0)
        lines = [f"{'':5s} {'median':>14s} {'-1sigma':>11s} {'+1sigma':>11s}"]
        for j, name in enumerate(self.labels):
            lines.append(f"{name:5s} {q[1, j]:14.7g} {q[1, j] - q[0, j]:11.3g} {q[2, j] - q[1, j]:11.3g}")
        return "\n".join(lines)

    def corner(self, **kwargs):
        """Corner plot of the weighted draws (no resampling duplicates)."""
        import corner
        return corner.corner(self.weighted_samples, weights=self.weights, labels=self.labels, **kwargs)


_SAMPLER_CLASS = None


def _sampler_class():
    """Build the emcee subclass lazily (emcee is an optional dependency)."""
    global _SAMPLER_CLASS
    if _SAMPLER_CLASS is not None:
        return _SAMPLER_CLASS
    try:
        import emcee
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("sampling needs emcee: pip install pizzetti[fit]") from e

    class MarginalEnsembleSampler(emcee.EnsembleSampler):
        """``emcee.EnsembleSampler`` for a MarginalTransitModel.

        Walkers live in the model's internal coordinates (T14 instead of a/R*
        when ``reparam="auto"``); ``run_mcmc``/``sample`` take physical starting
        points (extra columns such as q1, q2 are ignored) and ``get_chain``
        returns physical parameters plus LD columns (q1, q2 or u1, u2) and f0.
        """

        def __init__(self, model, nwalkers, internal=False, **kwargs):
            kwargs.pop("vectorize", None)
            super().__init__(nwalkers, model.ndim, model.log_prob_internal, vectorize=True,
                             **kwargs)
            self.model = model
            self.labels = list(model.labels)
            self._p0_internal = internal

        def _convert(self, state):
            if state is None or isinstance(state, emcee.State) or self._p0_internal:
                return state
            return self.model._initial(state)

        def sample(self, initial_state, *args, **kwargs):
            # (emcee's run_mcmc calls sample, so the conversion happens once, here)
            return super().sample(self._convert(initial_state), *args, **kwargs)

        def get_chain(self, expand=True, ld="q", seed=None, **kwargs):
            """As emcee's get_chain. expand=True (default): physical parameters +
            limb-darkening (or eclipse) coefficients + f0; the column names are
            in ``self.labels`` after the call. With flat=True the LD prior weights are
            applied by resampling (equal-weight rows); with flat=False the rows keep
            the chain structure and are unweighted (weights: get_ld_weights)."""
            ch = super().get_chain(**kwargs)
            shape = ch.shape
            X = self.model.to_physical(ch.reshape(-1, shape[-1]))
            if not expand:
                return X.reshape(shape)
            flat = kwargs.get("flat", False)
            out, labels, w = self.model.expand(X, ld=ld, rng=np.random.default_rng(seed),
                                               resample=flat)
            self.labels = labels
            self.ld_weights = w
            return out if flat else out.reshape(shape[:-1] + (out.shape[-1],))

        def get_ld_weights(self):
            """Importance weights of the last get_chain(flat=False, expand=True) call."""
            return self.ld_weights

    _SAMPLER_CLASS = MarginalEnsembleSampler
    return _SAMPLER_CLASS
