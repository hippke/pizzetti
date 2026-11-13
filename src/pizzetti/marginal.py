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

Priors: uniform within ``bounds`` on the free parameters (plus 0 <= b_conj < 1 + rp, where
b_conj = b (1 - e^2) / (1 +- e sin w) is the impact parameter at mid-transit / mid-eclipse, = b for
a circular orbit, and 0 <= ecc < 1), an optional vectorised ``log_prior`` callback on the physical
parameters, and ``ld_prior``: "physical" (default; flat over the law's physical
region x flat baseline f0 -- Kipping's (q1, q2) prior for the quadratic law),
"flat" (no constraint), a Gaussian dict (quadratic law, exact) or a callable
(importance reweighting).

Evidence: profile importance sampling also returns the ln evidence, as ``lnz`` of ``profile_is`` and in
``diagnostics`` of ``fit`` (needs finite ``bounds``).

Correlated noise: ``gp=pizzetti.GP()`` fits an additive Gaussian process together with the transit, instead of
detrending first (``_gp``; quadratic law, primary transit).

emcee and scipy are only needed for sampling (``pip install pizzetti[fit]``).
"""
import copy
import math
import warnings

import numpy as np
import numba
from numba import njit

from . import _gp, _laws, _orbit, _physnl, _powbasis
from . import _marginal as K
from . import _series
from .terms import _A_HALF
from .transitmodel import _exposure_offsets

CANON = K.CANON
#: with threads > 1: batches of at least this many rows are spread over the threads (rows);
#: smaller calls on light curves with at least PAR_MIN_POINTS (sub-exposure) points spread the
#: points of each evaluation over the threads in fixed chunks of PAR_CHUNK data points, so that
#: the results do not depend on the number of threads
PAR_MIN_ROWS = 16
PAR_MIN_POINTS = 20000
PAR_CHUNK = 2048
_NO_ROWPAN2 = np.full(1, -1, dtype=np.int64)
_NO_ASTACK = (np.zeros((1, _powbasis.DEG_A + 1, _powbasis.DEG + 1, 7, _powbasis.NB + 1)),
              np.zeros((1, 7), np.int64), np.zeros(1, np.bool_), np.zeros(1), np.full(1, -1.0),
              np.full(1, -1.0), np.full(1, -1.0), np.full(1, -1.0), np.full(1, -1, np.int64))
_DEG = math.pi / 180.0
_G_CGS = 6.6743e-8          # the constant of gravitation, cm3 / g / s2 (CODATA 2018)


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
            a = (0.0 - mi) / sd
            if a > 5.0:
                # deep tail: u = 1 - (1 - r) Phi(-a) rounds to 1 for a > ~8 and the draw fell below
                # the wall (outside the cone); the upper-tail form is exact to a ~ 37
                w[i] = mi - sd * K.ndtri(max((1.0 - rng_u[k]) * K.ndtr(-a), 1e-300))
                w[i] = max(w[i], 0.0)
                k += 1
                continue
            lo = K.ndtr(a)
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


def _chol_rows(S, jitter=(0.0, 1e-12, 1e-10)):
    """Cholesky factors of the rows of S; a row that is not positive definite (a geometry whose
    light curve hardly constrains the coefficients, e.g. a barely grazing transit) is retried
    with a small diagonal jitter (relative to the mean variance) and otherwise left NaN."""
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        pass
    L = np.full_like(S, np.nan)
    for i in range(len(S)):
        Si = 0.5 * (S[i] + S[i].T)
        if not np.all(np.isfinite(Si)):
            continue
        for jit in jitter:
            try:
                L[i] = np.linalg.cholesky(Si + jit * np.trace(Si) / len(Si) * np.eye(len(Si)))
                break
            except np.linalg.LinAlgError:
                continue
    return L


def _draw_c(C, S, rng):
    """One truncated-Gaussian draw of c per row (vectorised rejection, Gibbs fallback); NaN for
    rows whose covariance has no Cholesky factor (see _chol_rows)."""
    n = C.shape[0]
    out = np.full((n, 3), np.nan)
    L = _chol_rows(S)
    todo = np.flatnonzero(np.all(np.isfinite(L.reshape(n, -1)), axis=1) & np.all(np.isfinite(C), axis=1))
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
        try:
            out[i] = _gibbs_cone_one(C[i], L[i] @ L[i].T, rng.random(300 * 3), 300)   # (jittered) S
        except np.linalg.LinAlgError:                      # numerically singular: weight 0 downstream
            out[i] = np.nan
    return out


def _draw_c_theta(T, S, rng):
    """As _draw_c, from the Gaussian of theta = (f0, c1, c2) (theta_moments_batch): rejection, then
    Gibbs in theta (gibbs_theta_one); well conditioned for any transit depth. Returns c."""
    n = T.shape[0]
    th = np.full((n, 3), np.nan)
    good = np.all(np.isfinite(T), axis=1) & np.all(np.isfinite(S.reshape(n, -1)), axis=1)
    L = np.full((n, 3, 3), np.nan)
    if good.any():
        sd = np.sqrt(np.einsum("nii->ni", S[good]))
        R = S[good] / (sd[:, :, None] * sd[:, None, :])
        try:
            L[good] = np.linalg.cholesky(R) * sd[:, :, None]
        except np.linalg.LinAlgError:
            Lg = np.full((good.sum(), 3, 3), np.nan)
            for j, Rj in enumerate(R):
                try:
                    Lg[j] = np.linalg.cholesky(Rj) * sd[j][:, None]
                except np.linalg.LinAlgError:
                    pass
            L[good] = Lg
    todo = np.flatnonzero(good & np.all(np.isfinite(L.reshape(n, -1)), axis=1))
    for _ in range(60):
        if todo.size == 0:
            break
        z = rng.standard_normal((todo.size, 8, 3))
        cand = T[todo, None, :] + np.einsum("nij,nkj->nki", L[todo], z)
        c0 = 6.0 * cand[:, :, 0] - 2.0 * cand[:, :, 1] - 3.0 * cand[:, :, 2]
        ok = (c0 >= 0) & (cand[:, :, 1] >= c0) & (cand[:, :, 2] >= cand[:, :, 1])
        has = ok.any(axis=1)
        first = np.argmax(ok, axis=1)
        th[todo[has]] = cand[has, first[has]]
        todo = todo[~has]
    for i in todo:                                         # cone far in the tail, or a broad posterior
        th[i] = K.gibbs_theta_one(T[i], S[i], rng.random(300 * 3), 300)
    rest = np.flatnonzero(good & ~np.all(np.isfinite(L.reshape(n, -1)), axis=1))
    for i in rest:
        th[i] = K.gibbs_theta_one(T[i], S[i], rng.random(300 * 3), 300)
    return np.column_stack([6.0 * th[:, 0] - 2.0 * th[:, 1] - 3.0 * th[:, 2], th[:, 1], th[:, 2]])


class _BudgetExceeded(Exception):
    """Raised by MarginalTransitModel.log_prob_internal when the budget of profile_is is spent."""


class _Budget:
    """Evaluation and wall-time budget of one profile_is call (None: no limit). Charged in
    log_prob_internal, through which every evaluation passes."""

    def __init__(self, max_evals=None, max_time=None):
        import time
        self.max_evals, self.max_time, self.used = max_evals, max_time, 0
        self._clock = time.perf_counter
        self.t0 = self._clock()

    def charge(self, n):
        self.used += n
        if ((self.max_evals is not None and self.used > self.max_evals)
                or (self.max_time is not None and self._clock() - self.t0 > self.max_time)):
            raise _BudgetExceeded

    def remaining_evals(self):
        return None if self.max_evals is None else max(self.max_evals - self.used, 0)


class _BoundedCoords:
    """Unbounded coordinates u for the internal coordinates x of a model, for the retry of
    profile importance sampling when the posterior presses against a prior bound of a
    parameter other than b (the profiled b has walls of its own): x = lo + (hi - lo) sigmoid(u)
    for a box, x = lo + exp(u) or hi - exp(u) for a one-sided bound, x = u otherwise. With the T14
    reparametrisation the bounds on a/R* (and a > b) become bounds on T14 that depend on
    (per, rp, b) of the same row; T14 is mapped last, after those. log|dx/du| is the Jacobian.
    ``graze``: the b column of u holds beta = b_conj / (1 + rp) instead of b (b_conj = g b, g the
    conjunction factor of an eccentric orbit, 1 if circular). The support b_conj < 1 + rp moves with
    rp, beta < 1 does not: a grazing posterior along b - rp = const, which at low S/N reaches the rp
    bound, is compact in beta, and at fixed beta rp takes its whole box."""

    def __init__(self, model, graze=False, logit_beta=False):
        self.m = model
        self.graze = bool(graze)
        self.logit_beta = bool(graze and logit_beta)     # u_b = logit(beta): the support 0 <= beta < 1
        if self.graze:
            self.jb, self.jr = model.free.index("b"), model.free.index("rp")
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
        if model.gp is not None:                # log-uniform priors: boxes in the sampled logarithms
            self.cols += [(j, "box", lo, hi) for j, lo, hi in zip(model.gp_cols, model.gp_lo, model.gp_hi)]
        self.active = bool(self.cols) or self.graze

    def _t14_range(self, X):
        """Rows of internal x (T14 slot ignored) -> (T14 at a_hi, T14 at a_lo)."""
        m = self.m
        # per, rp, b of the rows, without the T14 -> a/R* conversion: the T14 slot is not set yet
        # (a T14 of 0 divided by zero in the numba kernel)
        P, _ = K.to_canonical(m._geo(np.nan_to_num(np.atleast_2d(X))), m.idx, m.fixed, False)
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
        if self.graze:                          # b = beta (1 + rp) / g, after rp (ecc, w), before T14
            g = self.m._conj_factor(X)
            beta = U[:, self.jb]
            if self.logit_beta:
                beta = 0.5 * (1.0 + np.tanh(0.5 * beta))
                lj += np.log(np.clip(beta * (1.0 - beta), 1e-300, None))
            X[:, self.jb] = beta * (1.0 + X[:, self.jr]) / g
            lj += np.log1p(X[:, self.jr]) - np.log(g)
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
        if self.graze:
            beta = X[:, self.jb] * self.m._conj_factor(X) / (1.0 + X[:, self.jr])
            if self.logit_beta:
                beta = np.clip(beta, margin, 1.0 - margin)
                beta = np.log(beta / (1.0 - beta))
            U[:, self.jb] = beta
        return U

    def log_prob(self, U):
        X, lj = self.to_x(U)
        out = self.m.log_prob_internal(X)
        return np.where(np.isfinite(out), out + lj, -np.inf)


class _StratumCoords:
    """Coordinates of one stratum of the posterior of ``profile_is``, split at the grazing boundary b_conj = 1 - rp,
    i.e. gamma = (1 - b_conj) / rp = 1 (Gilbert 2022), where the shape of the posterior changes. Two columns replace
    (rp, b); log_prob is the integrand of ``log_prob_internal`` with the Jacobian, so integrals are unchanged.
    ``beta``: non-grazing, (rp, beta = b_conj / (1 - rp)); the walls 0 and 1 of beta are fixed.
    ``graze``: grazing, (ln lambda, v), with lambda = rp^2 (1 + gamma), about twice the transit depth, which is what the
    data fix along the ridge, and v = (ln rp - ln r0) / (ln rp_hi - ln r0), r0 = max(sqrt(lambda / 2), rp_lo): the
    position along the ridge. The walls v = 0 (gamma = 1, or the lower bound on rp) and v = 1 (the upper bound on rp)
    are fixed, so that the conditional in ln lambda has no edge; in ln rp it would be cut off at gamma = 1, in
    ln(1 + gamma) at the bound on rp."""

    def __init__(self, model, kind):
        self.m, self.kind = model, kind
        self.jr, self.jb = model.free.index("rp"), model.free.index("b")
        self.r_lo, self.r_hi = max(model.lo[2], 1e-300), model.hi[2]

    def _r0(self, ln_lam):
        return np.maximum(0.5 * (ln_lam - math.log(2.0)), math.log(self.r_lo))

    def to_x(self, U):
        """Internal rows and log|d(rp, b) / d(columns)|."""
        U = np.atleast_2d(np.asarray(U, float))
        p, q = U[:, self.jr], U[:, self.jb]
        with np.errstate(all="ignore"):
            if self.kind == "beta":
                r, bc, lj = p, q * (1.0 - p), np.log(1.0 - p)
            else:
                ln_r0 = self._r0(p)
                span = math.log(self.r_hi) - ln_r0     # > 0: lambda < 2 rp_hi^2, else not finite below
                r = np.exp(ln_r0 + q * span)
                bc, lj = 1.0 - np.exp(p) / r + r, p + np.log(span)
            g = self.m._conj_factor(U)
            X = U.copy()
            X[:, self.jr], X[:, self.jb] = r, bc / g
            return X, lj - np.log(g)

    def to_u(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        U = X.copy()
        r = X[:, self.jr]
        with np.errstate(all="ignore"):
            bc = X[:, self.jb] * self.m._conj_factor(X)
            lam1 = (1.0 - bc) / r + 1.0                # 1 + gamma
            if self.kind == "beta":
                U[:, self.jb] = bc / (1.0 - r)
            else:
                ln_lam = np.log(r * r * lam1)
                ln_r0 = self._r0(ln_lam)
                U[:, self.jr], U[:, self.jb] = ln_lam, (np.log(r) - ln_r0) / (math.log(self.r_hi) - ln_r0)
        return U

    def log_prob(self, U):
        U = np.atleast_2d(np.asarray(U, float))
        X, lj = self.to_x(U)
        r, b = X[:, self.jr], X[:, self.jb]
        with np.errstate(all="ignore"):
            bc = b * self.m._conj_factor(U)
            gam = (1.0 - bc) / r
            ok = gam > 1.0 if self.kind == "beta" else (gam > -1.0) & (gam < 1.0)
            ok &= np.isfinite(r) & np.isfinite(b) & (b >= 0.0) & (r > 0.0) & np.isfinite(lj)
        out = np.full(len(U), -np.inf)
        if ok.any():
            out[ok] = self.m.log_prob_internal(X[ok]) + lj[ok]
        return out


_LNZ_LD_DRAWS = 1000       # conditional limb-darkening draws of profile_is for ln <f0 / c2^3> of its lnz
_STRATUM_MIN_MASS = 1e-3   # a stratum below this share of the posterior needs no refinement
_COLLAPSED_RP = 1e-3      # with ld_prior="flat", fit warns if most of the posterior has an effective rp below this (or 2 x the bound)
_NE_MIN_ROWS = 16         # fit(): log_prob_internal keeps the normal equations of calls with at least this many rows
_NO_NE = np.zeros((0, 9))  # ne_out of the likelihood kernels: none kept


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
        on the physical parameters (with ``gp``, also on its sampled logarithms, by their labels)
    :param ld_prior: prior on the linear (limb-darkening / eclipse) coefficients:
        "physical" (default): flat in the law's coefficients over its physical region
        (positive intensity, darkening towards the limb; for the quadratic and
        square-root laws this is Kipping 2013's prior) x flat baseline; for eclipses
        flat in fp >= 0 x flat baseline.
        "flat": no physical constraint; flat in the absolute linear coefficients
        (a flat prior in the normalised coefficients u can be improper, e.g. for the
        nonlinear law, when the data allow I(1) -> 0). Proper at fixed geometry, but not
        where the planet's area vanishes: the shape coefficients scale as 1 / rp^2, so the
        marginal likelihood grows as rp^(-2 (K - 1)) for rp -> 0 (K coefficients). Such
        coefficients can still give the observed dip, even a deep one, so a strong transit
        is no protection: the posterior can pile up there, on the bounds, also towards
        b = 1 + rp (``fit`` warns). For a parameter fit with free rp use "physical".
        A dict {"mean": (u1, u2), "cov": 2x2} (quadratic law): physical region x
        Gaussian, included exactly. A callable f(u1, u2, ...) -> log prior: physical
        region x f, applied by importance reweighting (fine for priors broader than
        the likelihood).
    :param supersample_factor, exp_time: as in ``TransitModel`` (exp_time in days); ``supersample_factor="auto"``
        picks the smallest number of sub-exposures whose model error, summed over the data, is chi^2 <= 0.01 at
        ``params`` (``pizzetti.choose_supersample``; the table is in ``supersample_table``). With ``exp_time`` > 0,
        ``fit()`` checks the error again at the posterior median (``diagnostics["supersample_chi2"]``) and warns if it
        could move the posterior by more than ~0.3 sigma. Without supersampling and without ``exp_time``, ``fit()``
        assumes exposures as long as the cadence (the typical spacing of the time stamps, as for Kepler and TESS) and
        warns in the same way if they would need an exposure average (``diagnostics["supersample_exp_assumed"]``;
        tutorial 22).
    :param offsets: sub-exposure placement: "midpoint" (default; the midpoints of k equal sub-intervals, error of the
        exposure average ~ 1/k^2) or "batman" (k points from edge to edge, as batman and ``TransitModel``; error ~ 1/k:
        a hot Jupiter in 30-min exposures at k = 15 is off by 147 ppm against 3.8 ppm with midpoints, enough to move b by
        several sigma). Up to version 0.5.0 the default was "batman".
    :param reparam: "auto" samples the total duration T14 instead of a/R*
        internally (circular orbits; prior kept uniform in a); "none" disables it
    :param transittype: "primary" or "secondary"
    :param threads: threads for fitting (quadratic law): 1 (default) runs serially; an integer
        or "all" (numba's thread count, all cores unless NUMBA_NUM_THREADS is set) spreads
        batches of parameter sets over threads (one light curve each) and, for the single
        evaluations of the optimisers, the data points of a large light curve. Results with
        threads > 1 do not depend on the number of threads; they agree with threads=1 to rounding
        (~1e-13 relative: the occultation terms of the limb points and Kepler's equation are
        evaluated per chunk of data points), so a fit can take a slightly different but
        equivalent path. threads=1 reproduces earlier results exactly. Leave at 1 when running
        many fits in parallel processes.
    :param gp: a ``pizzetti.GP``: an additive Gaussian process in the noise model (``Sigma = K + diag((s yerr)^2)``),
        its parameters sampled with the free ones (labels ``ln_gp_amp``, ``ln_gp_tau``, ``ln_err_scale``, after the free
        ones; log-uniform priors). For light curves that are not detrended. ``t`` must be in ascending time order, not
        folded. Quadratic law, primary transit, ``ld_prior`` "physical" or a callable. ``threads`` spreads the rows of a batch
        over threads (not the points of one evaluation). With all three GP parameters fixed on a circular orbit with a fixed
        period, each geometry only filters the points around the transits.
    """

    _ne_cache = None    # fit(): list of (internal rows, their theta normal equations) from log_prob_internal

    def __init__(self, params, t, y, yerr, free=("t0", "per", "rp", "a", "b"), bounds=None,
                 log_prior=None, ld_prior="physical", supersample_factor=1, exp_time=0.0,
                 offsets="midpoint", reparam="auto", series_tol=1e-9, transittype="primary", threads=1, gp=None):
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
        self.gp = gp
        if gp is None:
            self.wts = np.ascontiguousarray(1.0 / yerr)
        else:                                # the weights are in the whitening
            self.gp_var0 = np.ascontiguousarray(yerr ** 2)
            self.wts = np.ones(self.t.shape)
        self.yw = self.y * self.wts
        self.exp_time, self.offsets, self._params0 = float(exp_time), offsets, copy.copy(params)
        self._transittype = transittype
        self.supersample_table = None
        if isinstance(supersample_factor, str):
            if supersample_factor != "auto":
                raise ValueError('supersample_factor must be an integer or "auto"')
            if not exp_time > 0:
                raise ValueError('supersample_factor="auto" needs exp_time > 0 (the exposure, in days)')
            from .supersample import choose_supersample
            supersample_factor, self.supersample_table = choose_supersample(
                params, self.t, exp_time, yerr, offsets=offsets, transittype=transittype, return_table=True)
        self.k = int(supersample_factor) if supersample_factor > 1 else 1
        if self.k > 1 and not exp_time > 0:
            raise ValueError("supersample_factor > 1 needs exp_time > 0")

        off = _exposure_offsets(exp_time, self.k, offsets)
        self.max_off = float(np.max(np.abs(off)))
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
        if gp is not None:
            self._setup_gp(yerr)
        if not (threads == "all" or (isinstance(threads, (int, np.integer)) and threads >= 1)):
            raise ValueError('threads must be a positive integer or "all"')
        self.threads = threads
        self.n_eval = 0
        self._budget = None                  # set by profile_is(max_evals=, max_time=) while it runs
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

    def _setup_gp(self, yerr):
        if not self.fast or self.has_ldp:
            raise NotImplementedError('gp= is implemented for limb_dark="quadratic", transittype="primary" and '
                                      'ld_prior="physical" or a callable')
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("gp= needs t in strictly ascending order: the Gaussian process correlates neighbours in "
                             "time, so the light curve must not be folded or unsorted")
        T14 = self._start_t14()
        self.gp_idx, self.gp_steps = _gp.time_table(self.t, _gp.DT_TOL)
        self.gp_free, self.gp_val, lo, hi, labels = self.gp._resolve(self.t, self.y, yerr, T14)
        self.gp_lo, self.gp_hi = lo[self.gp_free], hi[self.gp_free]
        self.gp_cols = np.arange(len(self.free), len(self.free) + len(labels))
        self.labels += labels
        self.ndim += len(labels)
        # start: an amplitude from the scatter of the flux, the transit's time scale, the given errors
        amp = math.log(max(np.std(self.y), np.median(yerr)))
        self.gp_x0 = np.clip(np.array([amp, math.log(T14), 0.0])[self.gp_free], self.gp_lo + 1e-3, self.gp_hi - 1e-3)
        self.gp_fx = None
        if len(self.gp_free) == 0 and self.fixed[5] < 1e-5 and not ({"per", "ecc", "w"} & set(self.free)):
            self._setup_gp_fixed(T14)

    def _setup_gp_fixed(self, T14):
        """All GP parameters fixed, circular orbit, fixed period: filter once, and for each geometry only the blocks around the
        transits (0.75 T14 of the start + one exposure from the centre). Rows that reach beyond them take the full filter."""
        self.gp_half_exp = self.max_off
        self.gp_half = 0.75 * T14 + 2.0 * self.max_off
        t_ref, per = self.fixed[0], self.fixed[1]
        phase = (self.t - t_ref + 0.5 * per) % per - 0.5 * per
        inblock = np.abs(phase) < self.gp_half
        if inblock.all() or not inblock.any():          # nothing to gain
            return
        amp, tau, s_ = np.exp(self.gp_val)
        self.gp_fx = _gp.fixed_setup(self.gp_idx, self.gp_steps, self.gp_var0 * s_ * s_, amp, tau, self.y,
                                     float(np.median(self.y)), inblock, self.tt, self.k)

    def _start_t14(self):
        """T14 of the starting parameters (for the defaults of the GP only)."""
        t0, per, rp, a, b, ecc, w = self.fixed
        arg = max((1.0 + rp) ** 2 - b * b, 1e-4) / max(a * a - b * b, 1.0)
        return per / math.pi * math.asin(min(math.sqrt(arg), 1.0))

    @staticmethod
    def _gp_threads(kern, arg, nthr):
        if nthr == 1:
            return kern(*arg)
        old = numba.get_num_threads()
        numba.set_num_threads(nthr)
        try:
            return kern(*arg)
        finally:
            numba.set_num_threads(old)

    def _gp_rows(self, Xi):
        """(amp, tau, err_scale) per row of internal coordinates, and whether the row is inside their prior."""
        L = np.tile(self.gp_val, (len(Xi), 1))
        G = Xi[:, self.gp_cols]
        L[:, self.gp_free] = G
        ok = np.all((G > self.gp_lo) & (G < self.gp_hi), axis=1)
        return np.exp(np.clip(L, -700.0, 700.0)), ok

    def gp_predict(self, row, t=None, return_var=False):
        """The Gaussian process for one row of ``FitResult.samples`` (ld="q"): its posterior mean given that row's
        transit, at the data times, so that ``y - gp_predict(row)`` is the detrended light curve, or at any times
        ``t``, in any order (for plotting between the data); with ``return_var``, also its variance given that row."""
        if self.gp is None:
            raise ValueError("gp_predict needs a model with gp=")
        row = np.asarray(row, dtype=float)
        if row.shape != (self.ndim + 3,):
            raise ValueError(f"row needs {self.ndim + 3} values: {self.labels} + q1, q2, f0")
        Xi = self.to_internal(row[:self.ndim][None])
        P, _ = self._canonical_internal(Xi)
        E, ok = self._gp_rows(Xi)
        if not ok[0]:
            raise ValueError("row is outside the prior of the GP parameters")
        q1, q2, f0 = row[self.ndim:]
        sq = math.sqrt(q1)
        c = np.asarray(self.law.from_u((2.0 * sq * q2, sq * (1.0 - 2.0 * q2))), dtype=float)
        c = c * f0 / (K.DISK @ c)
        B = K.design_theta(self.tt, self.k, self.wts, *P[0], self.tol, _A_HALF, _series._GTAB)
        r = self.y - B @ np.array([f0, c[1], c[2]])
        var0 = self.gp_var0 * E[0, 2] ** 2
        if t is None:
            mean, var = _gp.predict(self.t, var0, E[0, 0], E[0, 1], r)
        else:                                # new times: points without a measurement (infinite variance)
            t = np.atleast_1d(np.asarray(t, dtype=float))
            ta = np.concatenate([self.t, t])
            o = np.argsort(ta, kind="stable")
            pad = np.full(t.shape, np.inf)
            mo, vo = _gp.predict(ta[o], np.concatenate([var0, pad])[o], E[0, 0], E[0, 1],
                                 np.concatenate([r, np.zeros(t.shape)])[o])
            mean, var = np.empty(ta.shape), np.empty(ta.shape)
            mean[o], var[o] = mo, vo
            mean, var = mean[len(self.t):], var[len(self.t):]
        return (mean, var) if return_var else mean

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

    def _conj_factor(self, Xi):
        """g = (1 - e^2) / (1 +- e sin w) of internal rows (b_conj = g b; 1 for a circular orbit).
        Only e and w are needed: no T14 -> a/R* conversion (in _BoundedCoords.to_x the T14 column is
        not set yet when g is needed, and a T14 of 0 divided by zero in the numba kernel)."""
        X = np.nan_to_num(np.atleast_2d(np.asarray(Xi, dtype=float)))
        P, _ = K.to_canonical(self._geo(X), self.idx, self.fixed, False)
        e, w = P[:, 5], P[:, 6]
        return (1.0 - e * e) / (1.0 + (-1.0 if self.secondary else 1.0) * e * np.sin(w))

    def _prior_ok(self, prow, alpha):
        t0, per, rp, a, b, ecc, w = prow
        g = (1.0 - ecc * ecc) / (1.0 + (-1.0 if self.secondary else 1.0) * ecc * math.sin(w))
        return (bool(np.all((prow > self.lo) & (prow < self.hi))) and 0.0 <= b and b * g < 1.0 + rp
                and 0.0 <= ecc < 1.0 and per > 0 and rp > 0 and a > 1.0
                and (not self.law.has_alpha or self.alpha_lo < alpha < self.alpha_hi))

    # -- log-probability -------------------------------------------------------
    def log_prob_internal(self, Xi, parts=False):
        """Vectorised log posterior (up to a constant) in internal coordinates. With ``parts``,
        (log posterior, log P): log P <= 0 is the log probability of the physical cone of the
        coefficients (0 without a cone), so that log posterior - log P is the posterior with a
        flat prior over all coefficients (smooth; the cone can cut the posterior off sharply)."""
        Xi = np.ascontiguousarray(np.atleast_2d(Xi), dtype=float)
        if self._budget is not None:
            self._budget.charge(len(Xi))
        logP = np.zeros(len(Xi))
        if self.gp is not None:
            P, lj = self._canonical_internal(Xi)
            E, ok = self._gp_rows(Xi)
            lPo = logP if parts else np.zeros(0)
            nthr = self._nthreads() if len(Xi) >= PAR_MIN_ROWS else 1
            arg = (self.gp_idx, self.gp_steps, self.tt, self.k, self.wts, self.y, self.gp_var0, self.tol, _A_HALF, _series._GTAB, lPo)
            if self.gp_fx is not None:
                kern = _gp.fixed_batch_par if nthr > 1 else _gp.fixed_batch
                arg = (P, lj, ok, self.lo, self.hi, self.gp_fx, self.fixed[0], self.gp_half, self.gp_half_exp, E) + arg
            elif nthr == 1:                     # one workspace for every evaluation (see _workspace)
                kern = _gp.loglike_batch_ws
                arg = (P, lj, E, ok, self.lo, self.hi) + arg + (self._gp_workspace(),)
            else:
                kern = _gp.loglike_batch_par
                arg = (P, lj, E, ok, self.lo, self.hi) + arg
            out = self._gp_threads(kern, arg, nthr)
        elif self.fast:
            args = (self._geo(Xi), self.idx, self.fixed, self.t14, self.lo, self.hi, self.tt, self.k,
                    self.wts, self.yw, self.tol, _A_HALF, _series._GTAB, self.has_ldp, self.mb, self.Pb,
                    logP if (parts and not self.has_ldp) else np.zeros(0))
            # inside fit(): keep each row's normal equations for the limb-darkening step (_expand_fast)
            # (batches only: the draws that reach it come in batches; single rows of the optimisers would only add
            # bookkeeping)
            keep_ne = self._ne_cache is not None and not self.has_ldp and len(Xi) >= _NE_MIN_ROWS
            ne = np.full((len(Xi), 9), np.nan) if keep_ne else _NO_NE
            nthr = self._nthreads()
            if nthr == 1 and not self.has_ldp:
                # one workspace for every evaluation: allocating the per-row arrays afresh made glibc return them
                # to the OS after each row and fault them in again (Kepler-78 b: 4x the cost of the arithmetic)
                out = K.loglike_batch_ws(*args[:13], args[16], ne, self._workspace())
            elif nthr == 1:
                out = K.loglike_batch(*args, ne)
            else:
                old = numba.get_num_threads()
                numba.set_num_threads(nthr)
                try:
                    if len(Xi) >= PAR_MIN_ROWS:
                        out = K.loglike_batch_par(*args, ne)           # rows over threads
                    elif len(self.tt) >= PAR_MIN_POINTS:
                        out = K.loglike_batch_dp(*args, PAR_CHUNK, ne)   # data points over threads
                    else:
                        out = K.loglike_batch(*args, ne)
                finally:
                    numba.set_num_threads(old)
            if keep_ne:
                self._ne_cache.append((Xi.copy(), ne))
        elif self._fused:
            P, lj = self._canonical_internal(Xi)
            out = self._gen(P, lj, self._alpha(Xi), True, False)[0]
            if parts and self.cone:
                self.cone = False
                try:
                    flat = self._gen(P, lj, self._alpha(Xi), True, False)[0]
                finally:
                    self.cone = True
                ok = np.isfinite(out) & np.isfinite(flat)
                logP[ok] = out[ok] - flat[ok]
        else:
            P, lj = self._canonical_internal(Xi)
            al = self._alpha(Xi)
            out = np.full(len(Xi), -np.inf)
            for i in range(len(Xi)):
                if not (np.all(np.isfinite(P[i])) and self._prior_ok(P[i], al[i])):
                    continue
                A = self._design_row(P[i], al[i])
                out[i] = K.marg_loglike_gen(A, self.yw, self.Dmat, self.cone) + lj[i]
                if parts and self.cone:
                    logP[i] = out[i] - lj[i] - K.marg_loglike_gen(A, self.yw, self.Dmat, False)
        self.n_eval += len(Xi)
        if self.log_prior is not None:
            ok = np.isfinite(out)
            if ok.any():
                P, _ = self._canonical_internal(Xi[ok])
                d = self._as_dict(P)
                if self.law.has_alpha:
                    d["alpha"] = self._alpha(Xi[ok])
                if self.gp is not None:
                    d.update(zip(self.labels[len(self.free):], Xi[ok][:, self.gp_cols].T))
                out[ok] += np.asarray(self.log_prior(d), dtype=float)
        return (out, logP) if parts else out

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
    def expand(self, X, ld="q", rng=None, resample=True, _xi=None):
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
            return self._expand_fast(X, ld, rng, resample, _xi)
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
            # no jitter: the coefficients have variances many orders of magnitude apart (f0 is known, the shape not),
            # jitter on the scale of the largest would swamp f0. Rows without a factor have weight 0
            L = _chol_rows(S, jitter=(0.0,))
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
            # "flat": flat in the absolute coefficients c (see the note on ld_prior: improper in rp)
            logw = np.where(np.all(np.isfinite(c), axis=1), 0.0, -np.inf)
        out_c, out_l = self.law.outputs(c, al, ld)
        if callable(self.ld_prior):
            uu, _ = self.law.outputs(c, al, "u")
            logw = logw + np.asarray(self.ld_prior(*uu.T), dtype=float)
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        if not np.isfinite(logw).any():
            if not self.physical:
                raise RuntimeError(f"no limb-darkening draws: the coefficient covariance is numerically singular at "
                                   f"all {len(X)} geometries (median rp {np.median(P[:, 2]):.3g}, b "
                                   f'{np.median(P[:, 4]):.3g}). ld_prior="flat" is improper where the planet\'s area '
                                   'vanishes (rp -> 0 or b -> 1 + rp), so the fit can end there; use ld_prior="physical"')
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
        todo = np.flatnonzero(np.all(np.isfinite(L.reshape(n, -1)), axis=1))
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
        out = np.full((n, Kc), np.nan)
        L = _chol_rows(S, jitter=(0.0,))
        todo = np.flatnonzero(np.all(np.isfinite(L.reshape(n, -1)), axis=1))
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

    def _ne_lookup(self, Xi):
        """Normal equations that log_prob_internal kept (fit()) for internal rows Xi, matched by their exact bytes;
        NaN rows where there are none."""
        out = np.full((len(Xi), 9), np.nan)
        if not self._ne_cache:
            return out
        if len(self._ne_cache) > 1:                      # one sorted block (kept for a second lookup)
            Xc = np.concatenate([x for x, _ in self._ne_cache])
            Nc = np.concatenate([e for _, e in self._ne_cache])
            kc = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
            o = np.argsort(kc, kind="stable")
            self._ne_cache[:] = [(Xc[o], Nc[o])]
        Xc, Nc = self._ne_cache[0]
        void = np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))
        kc = Xc.view(void).ravel()
        kq = np.ascontiguousarray(Xi, dtype=float).view(void).ravel()
        j = np.minimum(np.searchsorted(kc, kq), len(kc) - 1)
        hit = kc[j] == kq
        out[hit] = Nc[j[hit]]
        return out

    def _expand_fast(self, X, ld="q", rng=None, resample=True, xi=None):
        """expand() for the fast path. xi: the internal rows of X (fit()): their moments come from the normal
        equations of the likelihood evaluations (_ne_lookup) where those were kept, identical to theta_moments_batch."""
        rng = np.random.default_rng() if rng is None else rng
        X = np.atleast_2d(np.asarray(X, dtype=float))
        P = self._canonical_physical(X)
        if self.has_ldp:
            C, S = K.c_moments_batch(np.ascontiguousarray(P), self.tt, self.k, self.wts, self.yw,
                                     self.tol, _A_HALF, _series._GTAB, self.has_ldp, self.mb, self.Pb)
            c = _draw_c(C, S, rng)
        elif self.gp is not None:
            E, _ = self._gp_rows(X)
            Pc = np.ascontiguousarray(P)
            nthr = self._nthreads() if len(X) >= PAR_MIN_ROWS else 1
            arg = (self.gp_idx, self.gp_steps, self.tt, self.k, self.wts, self.y, self.gp_var0, self.tol, _A_HALF, _series._GTAB)
            if self.gp_fx is not None:
                kern = _gp.fixed_moments_batch_par if nthr > 1 else _gp.fixed_moments_batch
                arg = (Pc, self.gp_fx, self.fixed[0], self.gp_half, self.gp_half_exp, E) + arg
            elif nthr == 1:
                kern = _gp.theta_moments_batch_ws
                arg = (Pc, E) + arg + (self._gp_workspace(),)
            else:
                kern = _gp.theta_moments_batch_par
                arg = (Pc, E) + arg
            T, S = self._gp_threads(kern, arg, nthr)
            c = _draw_c_theta(T, S, rng)
        else:
            NE = self._ne_lookup(xi) if xi is not None else np.full((len(X), 9), np.nan)
            T, S = K.theta_moments_ne(NE)
            miss = np.flatnonzero(~np.all(np.isfinite(NE), axis=1))
            self.last_ld_cache_hits = len(X) - len(miss)
            if len(miss):
                T[miss], S[miss] = K.theta_moments_batch(np.ascontiguousarray(P[miss]), self.tt, self.k, self.wts,
                                                         self.yw, self.tol, _A_HALF, _series._GTAB)
            c = _draw_c_theta(T, S, rng)
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
        self.last_ld_from_prior = False
        self._ld_logw = logw                             # ln(f0 / c2^3) (for lnz); -inf: no valid draw
        if not np.isfinite(logw).any():
            if self.has_ldp or callable(self.ld_prior):
                raise RuntimeError("no valid limb-darkening draws for these geometries")
            # no geometry constrains the limb darkening (e.g. a transit too shallow or too grazing to
            # be seen): its conditional posterior is the prior, uniform in (q1, q2); f0 from the data
            q1, q2 = rng.random(len(X)), rng.random(len(X))
            sq = np.sqrt(q1)
            u1, u2 = 2.0 * sq * q2, sq * (1.0 - 2.0 * q2)
            # T[:, 0]: the conditional mean of f0 (theta = (f0, c1, c2)); the data mean where it is not finite
            f0 = np.where(np.isfinite(T[:, 0]), T[:, 0], float(self.yw @ self.wts / (self.wts @ self.wts)))
            logw = np.zeros(len(X))
            self.last_ld_from_prior = True
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
    def _workspace(self):
        """The arrays K.loglike_batch_ws reuses for every evaluation (made at the first call; not pickled). Made again
        if the model times or the sub-exposures were replaced since (the workspace holds the mid-time and half-span of
        each data point, and the kernels do not check sizes)."""
        key = (self.tt, self.k)
        c = self.__dict__.get("_ws")
        if c is None or c[0][0] is not key[0] or c[0][1] != key[1]:
            c = self._ws = (key, K.workspace(self.tt, self.k))
        return c[1]

    def _gp_workspace(self):
        """The arrays _gp.loglike_batch_ws / theta_moments_batch_ws reuse (made at the first call; not pickled; made
        again if the times, the flux or the GP's step table were replaced since)."""
        key = (self.tt, self.k, self.y, self.gp_steps)
        c = self.__dict__.get("_gws")
        if c is None or any(a is not b for a, b in zip(c[0][:1] + c[0][2:], key[:1] + key[2:])) or c[0][1] != key[1]:
            c = self._gws = (key, _gp.workspace(self.tt, self.k, len(self.y), len(self.gp_steps)))
        return c[1]

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_ws", None)
        d.pop("_gws", None)
        return d

    def _nthreads(self):
        """Threads used for fitting: 1, or self.threads capped at numba's thread count."""
        if self.threads == 1 or not self.fast:
            return 1
        cap = numba.config.NUMBA_NUM_THREADS
        return cap if self.threads == "all" else max(1, min(int(self.threads), cap))

    def _valid_start(self, x0, rng):
        """x0 (internal) moved inside the prior if its log posterior is not finite (e.g. a start
        on a strict bound): bounded parameters are clipped 1e-4 of their range inside, then up to
        20 jittered points around it are tried. Returns x0 unchanged if it is valid."""
        x0 = np.asarray(x0, dtype=float)
        if np.isfinite(self.log_prob_internal(x0[None])[0]):
            return x0
        X = self.to_physical(x0[None])[0]
        for j, q in enumerate(self.free):
            if q == "alpha":
                lo, hi = self.alpha_lo, self.alpha_hi
            else:
                k = CANON.index("t0" if q == "t_secondary" else q)
                lo, hi = self.lo[k], self.hi[k]
                if q == "w":
                    lo, hi = lo / _DEG, hi / _DEG
            if np.isfinite(lo) and np.isfinite(hi):
                d = 1e-4 * (hi - lo)
                X[j] = min(max(X[j], lo + d), hi - d)
            elif np.isfinite(lo):
                X[j] = max(X[j], lo + 1e-4 * max(abs(lo), 1.0))
            elif np.isfinite(hi):
                X[j] = min(X[j], hi - 1e-4 * max(abs(hi), 1.0))
        x1 = self.to_internal(X[None])[0]
        if np.isfinite(self.log_prob_internal(x1[None])[0]):
            return x1
        sc = self._default_scales(x1)
        for _ in range(20):
            x2 = x1 + 3.0 * sc * rng.standard_normal(len(x1))
            if np.isfinite(self.log_prob_internal(x2[None])[0]):
                return x2
        return x0

    def profile_is(self, n=4000, rng=None, x0=None, retry=True, flat_proposal=True, last_resort=True,
                   ess_min=0.25, rescue=True, rescue_iters=30, khat_ok=0.5, max_evals=None, max_time=None,
                   evidence=True, stratify=True):
        """Profile importance sampling along b (``_sampling.profile_is``) in internal coordinates.

        The walls of the b profile are b = 0 (the likelihood depends on b^2) or the bounds on b,
        just inside them (the bounds are strict). A start with a non-finite log posterior (e.g. on
        a bound) is first moved inside the prior. If the result is not usable (k-hat >= ``khat_ok``
        or an incomplete profile), up to two further attempts follow, and the best result is kept:

        * ``flat_proposal`` (laws with a physical cone): the proposal is built on the posterior
          with a flat prior over the coefficients, which is smooth, and the draws are weighted
          against the physical posterior = flat posterior x P(cone), P <= 1, so the weights stay
          bounded. At high S/N the cone can cut the physical posterior off sharply (a cliff where
          the best-fit limb darkening reaches the boundary of the physical region), which a
          Laplace-based proposal cannot follow.
        * ``retry``, if free parameters have finite bounds: the sampling in unbounded coordinates
          (logit / log of the bounded parameters, with the Jacobian), refined by adaptive multiple
          importance sampling (``_sampling.pmc_refine``): a posterior pressed against such a bound
          (e.g. a/R* for an unresolved ingress) breaks the Laplace steps in the original
          coordinates. If the first profile reached a grazing geometry (b_conj + rp > 1) and the bounds
          allow b_conj up to 1 + rp, the retry profiles beta = b_conj / (1 + rp) between the fixed
          walls 0 and 1 instead of b: the edge b_conj = 1 + rp moves with rp, so a grazing posterior along
          b - rp = const (at low S/N up to the rp bound) cannot be closed in b. Without an upper
          bound on rp such a posterior is improper (a warning is issued).

        The cone attempt comes first if the first profile was complete (a complete profile with
        bad weights points to the shape of the posterior), the bounded retry first otherwise.
        Usable: a complete profile, k-hat < ``khat_ok`` and an effective sample size >= ``ess_min`` x n.
        ``khat_ok`` = 0.5 (finite-variance weights), not the usual 0.7: a profile can miss a long
        narrow ridge that the Laplace volume along the grid underrates (K2-131 b: a grazing branch, b up
        to 1.3 and rp up to 0.3, with a few per cent of the mass; two of three seeds had k-hat 0.65 and
        0.69 and no draws on it), and the rescue below finds it. The
        ESS bound catches proposals that miss part of the posterior altogether (no draws there, so no
        large weights for k-hat to see): TrES-2 b with the flat-limb-darkening proposal had k-hat 0.4-0.6
        but an ESS of 150-250 of 4000 and missed the upper tail of rp; with an ESS of 900 of 4000
        (Kepler-21 b) the tails carry the Monte Carlo noise of a few hundred draws (0.37 sd off in the
        2.5 % quantile of T14; 0.06 sd with 40 000 draws), hence ``ess_min`` = 0.25. Accepted runs of
        the injection campaigns all had ESS > 1000.
        If the result is still not usable, ``rescue`` (default) pools the weighted draws of all
        attempts and runs adaptive importance sampling (``_sampling.pmc_refine``) from them in the
        unbounded coordinates of the retry, with defensive components over the whole prior box, for
        at least 3 and at most ``rescue_iters`` iterations of n draws, until k-hat < 0.7, ESS >=
        ``ess_min`` x n and the quantiles are stable (``is_ok``). An incomplete profile is never
        accepted on its weights alone: a proposal that stops short of part of the posterior can have
        a low k-hat and miss that part (K2-141 b: the flat-limb-darkening proposal ended at b = 0.8,
        k-hat 0.09, rp off by 1.2 sd). Without ``rescue``, ``last_resort`` refines the best attempt's
        weighted draws by adaptive importance sampling in the original coordinates (kept only if it
        improves k-hat).
        ``theta`` and ``theta_map`` are internal coordinates (``grid_b`` of a retry with beta:
        beta); ``grazing_beta``: whether a retry would profile beta; ``bounded_retry``,
        ``flat_proposal`` and ``last_resort_pmc`` say whether each step was not run (None), run
        but not used (False) or used (True); ``rescue``: whether the rescue ran; ``is_ok``: whether
        the returned draws pass the diagnostics.
        ``max_evals``, ``max_time`` (seconds): a budget for the whole call (None: none). An expensive
        likelihood (long light curves with many sub-exposures) times the evaluations of a hard
        posterior (a few 1e5 with the retries and the rescue) can take hours. When the budget is
        spent, the best attempt so far is returned with ``budget_exceeded`` True and ``is_ok`` False
        (the rescue's iterations are capped to fit an evaluation budget); if the first proposal
        could not be completed, RuntimeError.

        Strata (``stratify``, default): if the first profile is not usable and the bounds leave the whole support of b
        with a finite bound on rp, the posterior is split at the grazing boundary b_conj = 1 - rp and each side is
        integrated on its own, with no overlap (``_StratumCoords``): a profile of n / 2 draws in coordinates that suit
        that side, then adaptive importance sampling (``_sampling.pmc_refine``) in the same coordinates if its diagnostics
        fail. Draws combine with the two integrals as weights. ``strata``: per side (``nongrazing``, ``grazing``) its share of
        the posterior ``mass``, ``mass_err``, ``khat``, ``ess``, ``n_grid``, ``amis_iters``, ``coord`` (flat in the
        coefficients c: ``expand`` gives the limb-darkening weight). If it does not pass, the attempts below follow as
        before, and its draws seed the rescue; ``stratified`` says whether it ran and was used, as for the others, and
        ``strata_note`` why it did not pass: a side without a proposal, a side of the maximum so far that does not reach
        it, or a mass far below the draws of the first profile. A posterior with a complete first profile that does not reach a grazing geometry tries
        ``flat_proposal`` first (a cut-off by the cone, not by the geometry). ``stratify=False`` skips all this.

        Evidence: ``log_int``, ``log_int_err``: ln of the integral of exp(``log_prob_internal``) over the
        free parameters, from the raw importance weights of the returned draws, and its sd (any law or
        prior). ``lnz``, ``lnz_err`` (``evidence``, default True): the ln evidence with every constant, for
        the uniform prior on the free parameters (normalised over ``bounds``) and the physical limb darkening
        (uniform (q1, q2)). Its flat prior on f0 has unit density: subtract ln(width) for a flat f0 of that
        width. It costs ``_LNZ_LD_DRAWS`` conditional limb-darkening draws; seeded draws do not change.
        ``lnz`` is nan where the prior has no known normalisation (``lnz_note`` says why): an infinite bound
        (per and a/R* have none by default), ecc or w free, ``log_prior``, a Gaussian or callable ``ld_prior``,
        ``gp`` or another law. Only trust it if ``is_ok``: a mode that no draw reached is missing from it
        without a sign in ``lnz_err``.
        """
        budget = _Budget(max_evals, max_time) if (max_evals is not None or max_time is not None) else None
        self._budget = budget
        try:
            from ._sampling import mc_options
            with mc_options(defer=True):
                res = self._profile_is(n, rng, x0, retry, flat_proposal, last_resort, ess_min, rescue,
                                       rescue_iters, khat_ok, budget, stratify)
            self._last_topup = res.pop("topup_to", None)      # fit(): the ESS floor after the LD step
            if evidence:
                res.update(self._lnz_from_draws(res))
            return res
        finally:
            self._budget = None

    def _gp_seed(self, xm, n, rng, nev):
        """Weighted draws for the rescue that need no curvature in the GP parameters: a flat amplitude or time scale (a
        weak GP, a trend slower than the windows) makes the profile fail before it has draws. Geometry: a heavy-tailed
        Laplace proposal at xm, whose GP parameters are held fixed. GP parameters: half uniform over their box, half a
        t around xm."""
        from ._sampling import hessian, _mvt_logpdf, _mvt_sample
        ng = len(self.free)
        sc = self._default_scales(xm)
        xm = np.array(xm)
        xm[self.gp_cols] = np.clip(xm[self.gp_cols], self.gp_lo + 1e-3, self.gp_hi - 1e-3)
        jb = self.free.index("b")
        xm[jb] = abs(xm[jb])

        def fg(G):                               # mirrored in b: the maximum is often at b = 0
            G = np.array(G)
            G[:, jb] = np.abs(G[:, jb])
            return self.log_prob_internal(np.column_stack([G, np.tile(xm[ng:], (len(G), 1))]))

        def curvature(h):
            """Hessian with stencil points inside the support: near a bound (xm at a/R* close to its limit, say) a
            step that crosses it gives -inf; shorter steps along those coordinates. Entries that stay non-finite
            (xm on the bound) carry no curvature: 0."""
            k = 0
            for _ in range(6):
                with np.errstate(invalid="ignore"):
                    H, kh = hessian(fg, xm[:ng], h)
                k += kh
                bad = ~np.all(np.isfinite(H), axis=0)
                if not bad.any():
                    break
                h = np.where(bad, 0.25 * h, h)
            return np.where(np.isfinite(H), H, 0.0), k
        H, k1 = curvature(0.5 * sc[:ng])
        with np.errstate(invalid="ignore"):
            sd = np.sqrt(np.abs(np.diag(np.linalg.pinv(-H))))
        H, k2 = curvature(0.3 * np.minimum(np.where(sd > 0, sd, sc[:ng]), 5.0 * sc[:ng]))
        # widths at most 50 x the default scales (half a T14 in t0 and T14, half of rp, 2.5 in b): a direction without
        # curvature (flat, or cut off by a bound) would otherwise get an unbounded width, and no draw would be inside
        wmax = 50.0 * sc[:ng]
        d = np.sqrt(np.maximum(np.abs(np.diag(H)), wmax ** -2.0))
        ev, V = np.linalg.eigh(-0.5 * (H + H.T) / np.outer(d, d))
        C = ((V / np.clip(np.abs(ev), 1e-6, None)) @ V.T) / np.outer(d, d) * 4.0
        f = np.minimum(1.0, wmax / np.sqrt(np.diag(C)))
        Lg = np.linalg.cholesky(C * np.outer(f, f))
        nu = 5.0
        m = 2 * n
        Xg = _mvt_sample(rng, xm[:ng], Lg, nu, m)
        lq = _mvt_logpdf(Xg, xm[:ng], Lg, nu)
        ngp = len(self.gp_cols)
        Lh = np.diag(np.array([0.4, 0.4, 0.05])[self.gp_free])
        Xb = rng.uniform(self.gp_lo, self.gp_hi, size=(m, ngp))
        Xt = _mvt_sample(rng, xm[ng:], Lh, nu, m)
        Xe = np.where((rng.random(m) < 0.5)[:, None], Xb, Xt)
        inb = np.all((Xe > self.gp_lo) & (Xe < self.gp_hi), axis=1)
        lq = lq + np.logaddexp(math.log(0.5) - np.log(self.gp_hi - self.gp_lo).sum() + np.where(inb, 0.0, -np.inf),
                               math.log(0.5) + _mvt_logpdf(Xe, xm[ng:], Lh, nu))
        X = np.column_stack([Xg, Xe])
        logw = self.log_prob_internal(X) - lq
        nev.append(k1 + k2 + m)
        ok = np.isfinite(logw)
        w = np.zeros(m)
        if ok.any():                             # none: the rescue skips this seed
            w[ok] = np.exp(logw[ok] - np.max(logw[ok]))
        return dict(theta=X, logw=logw, w=w)

    def _grazing_start(self, T, x_ref):
        """Best point of a grid of the grazing stratum ``T``: ln lambda by two refinements at each of 12 values of v;
        the other parameters from x_ref. Returns (U, evaluations)."""
        prof = np.linspace(0.02, 0.98, 12)
        centre, width = 0.5 * (math.log(1e-6) + math.log(1.0)), 0.5 * (math.log(1.0) - math.log(1e-6))
        centres = np.full(len(prof), centre)
        U = np.tile(np.asarray(T.to_u(x_ref))[0], (21, 1))
        best, best_v, nev = None, -np.inf, 0
        for it in range(3):
            w_ = width / 8.0 ** it
            for k, pv in enumerate(prof):
                U[:, T.jr] = centres[k] + np.linspace(-w_, w_, 21)
                U[:, T.jb] = pv
                v = T.log_prob(U)
                nev += len(U)
                if np.isfinite(v).any():
                    j = int(np.nanargmax(np.where(np.isfinite(v), v, np.nan)))
                    centres[k] = U[j, T.jr]
                    if v[j] > best_v:
                        best, best_v = U[j].copy(), float(v[j])
        if best is None:
            best = np.asarray(T.to_u(x_ref))[0]
            best[T.jr], best[T.jb] = math.log(2.0 * x_ref[T.jr] ** 2), 0.5 * (prof[0] + prof[-1])
        return best, nev

    def _log_prior_volume(self):
        """ln volume of the region of the free parameters with a prior density (box, 0 <= b_conj < 1 + rp,
        a > b and a > 1; with T14 sampled a >= 1 + rp, which T14 cannot leave). inf for an infinite box."""
        out = 0.0
        for q in self.free:
            if q not in ("rp", "a", "b"):
                k = CANON.index(q)
                lo, hi = max(self.lo[k], 0.0) if q == "per" else self.lo[k], self.hi[k]
                if not np.isfinite(hi - lo):
                    return np.inf
                out += math.log(hi - lo)
        x, wx = np.polynomial.legendre.leggauss(200)
        x, wx = 0.5 * (x + 1.0), 0.5 * wx
        if "rp" in self.free:
            r_lo, r_hi = max(self.lo[2], 0.0), self.hi[2]
            if not np.isfinite(r_hi):
                return np.inf
            rp, wr = r_lo + (r_hi - r_lo) * x, (r_hi - r_lo) * wx
        else:
            rp, wr = np.array([self.fixed[2]]), np.ones(1)
        g = float(self._conj_factor(self.fixed_physical_free())[0])
        b_lo = max(self.lo[4], 0.0)
        b_hi = np.minimum(self.hi[4], (1.0 + rp) / g)
        if "a" in self.free:
            a_lo, a_hi = max(self.lo[3], 1.0), self.hi[3]
            if not np.isfinite(a_hi):
                return np.inf
        else:
            b_hi = np.minimum(b_hi, self.fixed[3])
        wb = np.clip(b_hi - b_lo, 0.0, None)                # b: rows of rp, columns of the b nodes
        b = b_lo + wb[:, None] * x
        if "a" in self.free:
            lo_ab = np.maximum(a_lo, b)
            if self.t14:
                lo_ab = np.maximum(lo_ab, 1.0 + rp[:, None])
            area = np.clip(a_hi - lo_ab, 0.0, None)
        else:
            area = np.ones_like(b)
        vol = np.sum(wr * wb * (area @ wx))
        return out + math.log(vol) if vol > 0 else np.inf

    def _lnz_note(self):
        """Why ``lnz`` is not available, or None."""
        if not self.fast or self.gp is not None or self.has_ldp or callable(self.ld_prior):
            return 'lnz needs limb_dark="quadratic", a primary transit, ld_prior="physical" and no gp'
        if self.log_prior is not None:
            return "lnz needs a normalised prior: log_prior is not known to be"
        if "ecc" in self.free or "w" in self.free:
            return "lnz: the prior volume with a free ecc or w is not implemented"
        if not np.isfinite(self._log_prior_volume()):
            return "lnz needs finite bounds on every free parameter (and an upper bound on rp and a)"
        return None

    def _lnz_from_ld(self, log_int, log_int_err, logw_ld, weights):
        """ln Z = ln of the integral of exp(log_prob_internal) + the Gaussian constants of the kernel
        (the integral of L over the cone is (2 pi)^(3/2) exp(log_prob) / prod(sqrt(2 pi) yerr)) - ln(prior
        volume) + ln <f0 / c2^3> over the joint posterior (flat in c -> flat in f0, beta0, beta1) + ln 2 (uniform
        density on the triangle of beta0, beta1, of area 1/2). logw_ld: ln(f0 / c2^3) of draws with these
        weights. See the paper, Sect. 3."""
        note = self._lnz_note()
        ok = np.isfinite(logw_ld) & (weights > 0)
        if note is None and not ok.any():
            note = "lnz: no valid limb-darkening draws"
        if note is not None:
            return dict(lnz=np.nan, lnz_err=np.nan, lnz_note=note)
        p = weights[ok] / weights[ok].sum()
        top = np.max(logw_ld[ok])
        v = np.exp(logw_ld[ok] - top)
        mean = p @ v
        err = math.sqrt(np.sum(p * p * (v - mean) ** 2)) / mean
        const = -0.5 * (len(self.y) - 3) * math.log(2.0 * math.pi) + np.sum(np.log(self.wts))
        lnz = log_int + const - self._log_prior_volume() + top + math.log(mean) + math.log(2.0)
        return dict(lnz=float(lnz), lnz_err=float(math.hypot(log_int_err, err)), lnz_note=None)

    def _lnz_from_draws(self, res):
        """``_lnz_from_ld`` for the draws of ``profile_is``, with ``_LNZ_LD_DRAWS`` conditional limb-darkening
        draws of its own (a generator of its own: the caller's stream, and so its seeded draws, do not change)."""
        if self._lnz_note() is not None:
            return self._lnz_from_ld(res["log_int"], res["log_int_err"], np.zeros(0), np.zeros(0))
        rng = np.random.default_rng(0)
        w = np.asarray(res["w"], float)
        if not w.sum() > 0:
            return self._lnz_from_ld(res["log_int"], res["log_int_err"], np.zeros(0), np.zeros(0))
        j = systematic_resample(w / w.sum(), _LNZ_LD_DRAWS, rng)
        P = self._canonical_physical(self.to_physical(res["theta"][j]))
        T, S = K.theta_moments_batch(np.ascontiguousarray(P), self.tt, self.k, self.wts, self.yw,
                                     self.tol, _A_HALF, _series._GTAB)
        c = _draw_c_theta(T, S, rng)
        with np.errstate(invalid="ignore", divide="ignore"):
            logw = np.log(c @ K.DISK) - 3.0 * np.log(c[:, 2])
        return self._lnz_from_ld(res["log_int"], res["log_int_err"], logw, np.ones(len(logw)))

    def _box_seed(self, T, xm, n, rng, nev):
        """Weighted draws for the rescue where no profile gave any: 2 n over the prior in the unbounded coordinates of
        ``T`` (uniform over a box is a standard logistic in its logit; elsewhere a wide normal around the best point).
        ``theta`` is in internal coordinates, as for the other attempts. They only seed the first proposal."""
        um = T.to_u(xm)[0]
        um = np.where(np.isfinite(um), um, 0.0)
        kinds = {j: kind for j, kind, _, _ in T.cols}
        if T.logit_beta:
            kinds[T.jb] = "box"
        sc = self._default_scales(xm)
        m = 2 * n
        U = np.empty((m, len(um)))
        lq = np.zeros(m)
        for j in range(len(um)):
            if kinds.get(j) in ("box", "a14"):
                U[:, j] = rng.logistic(size=m)
                lq += -U[:, j] - 2.0 * np.log1p(np.exp(-U[:, j]))
            else:
                sd = 2.0 if kinds.get(j) in ("lo", "hi") else 5.0 * sc[j]
                U[:, j] = um[j] + sd * rng.standard_normal(m)
                lq += -0.5 * ((U[:, j] - um[j]) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2.0 * math.pi)
        logw = T.log_prob(U) - lq
        nev.append(m)
        ok = np.isfinite(logw)
        w = np.zeros(m)
        if ok.any():
            w[ok] = np.exp(logw[ok] - np.max(logw[ok]))
        return dict(theta=T.to_x(U)[0], logw=logw, w=w)

    def _profile_is(self, n, rng, x0, retry, flat_proposal, last_resort, ess_min, rescue, rescue_iters,
                    khat_ok, budget, stratify):
        from ._sampling import _MC, _cat_labels, mc_error, pmc_refine, profile_is
        rng = np.random.default_rng() if rng is None else rng
        x0 = self._start_internal() if x0 is None else np.asarray(x0, float)
        x0 = self._valid_start(x0, rng)
        _TOP = ("theta", "w", "logw", "khat", "ess", "rep", "mc_err", "log_int", "log_int_err")

        def in_x(tu, T):
            """A top-up of draws in T's coordinates that returns internal coordinates."""
            if tu is None:
                return None

            def f(tol, max_add, min_ess=None):
                o = dict(tu(tol, max_add, min_ess))
                o["theta"] = T.to_x(o["theta"])[0]
                return o
            return f

        def topped(r, tol, max_add):
            """r with draws added from its final proposal until the replicate MC error is <= tol (if it can)."""
            if (tol is None and _MC["min_ess"] is None) or r.get("topup_to") is None:
                return r
            f = r["topup_to"](tol, max_add, _MC["min_ess"])
            if not f["n_added"]:
                return r
            nev.append(f["n_added"])
            r = dict(r)
            r.update({k: f[k] for k in _TOP})
            r["n_topup"] = r.get("n_topup", 0) + f["n_added"]
            return r
        jb = self.free.index("b")
        lo_b, hi_b = self.lo[4], self.hi[4]
        walls = dict(lo_wall=max(0.0, lo_b) + (1e-9 if lo_b >= 0 else 0.0),
                     hi_wall=hi_b - 1e-9 if np.isfinite(hi_b) else None)
        try:
            res = profile_is(self.log_prob_internal, x0, self._default_scales(x0), prof=jb, n=n, rng=rng,
                             **walls)
        except _BudgetExceeded:
            raise RuntimeError(f"profile_is: budget spent ({budget.used} evaluations, max_evals={budget.max_evals}, "
                               f"max_time={budget.max_time} s) before the first proposal was complete") from None
        flags = dict(bounded_retry=None, flat_proposal=None, last_resort_pmc=None, stratified=None)
        nev = [res["n_eval"]]

        def usable(r):
            return bool(r["complete"] and r["khat"] < khat_ok and r["ess"] >= ess_min * n)

        def better(r, best):
            if r["complete"] != best["complete"]:
                return bool(r["complete"])
            return r["khat"] < best["khat"]

        tried = [res]

        def finish(r):
            if usable(r):
                r = topped(r, _MC["tol"], int(_MC["max_factor"] * n))
            r.update(flags)
            r["grazing_beta"] = graze
            r["rescue"] = None
            r["budget_exceeded"] = False
            r["is_ok"] = usable(r)
            if not r["is_ok"] and rescue:
                r = run_rescue(r)
                if r.get("rescue") and r.get("is_ok"):
                    r = topped(r, _MC["tol"], int(_MC["max_factor"] * n))
                r["n_eval"] = int(sum(nev))
                return r
            if not usable(r) and last_resort and np.isfinite(r["khat"]) and np.sum(r["w"] > 0) > 50:
                pm = pmc_refine(self.log_prob_internal, r["theta"], r["logw"], rng, n=n)
                nev.append(pm["n_eval"])
                r["last_resort_pmc"] = bool(pm["khat"] < r["khat"])
                if r["last_resort_pmc"]:
                    r.update(theta=pm["theta"], w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"],
                             stage="pmc", pmc=pm["history"], log_int=pm["log_int"], log_int_err=pm["log_int_err"],
                             rep=pm.get("rep"), mc_err=pm.get("mc_err"), topup_to=pm.get("topup_to"))
            r["n_eval"] = int(sum(nev))
            return r

        def run_strata(start):
            """Both sides of the grazing boundary; None if a side with mass does not pass."""
            from scipy.special import logsumexp
            nk = max(n // 2, 500)
            x_ref = np.array(start, float)
            lp_ref = float(self.log_prob_internal(x_ref[None])[0])
            nev.append(1)

            def refine(T, r):
                U = r["theta"]
                pw = np.exp(r["logw"] - np.max(r["logw"][np.isfinite(r["logw"])]))
                pw = np.where(np.isfinite(pw), pw, 0.0)
                pw /= pw.sum()
                mu = pw @ U
                sd = np.sqrt(np.clip(np.diag(np.atleast_2d(np.cov(U.T, aweights=pw))), 1e-300, None))
                pm = pmc_refine(T.log_prob, U, r["logw"], rng, n=nk, iters=15, min_iters=3, khat_ok=khat_ok,
                                ess_min=ess_min, extra=[(0.03, mu, np.diag((5.0 * sd) ** 2))])
                nev.append(pm["n_eval"])
                if not pm["converged"]:
                    flags["strata_note"] = f"{T.kind}: adaptive importance sampling did not converge: " + str(pm["history"][-1:])
                    return None
                r = dict(r)
                r.update(theta=pm["theta"], w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"],
                         log_int=pm["log_int"], log_int_err=pm["log_int_err"], amis_iters=len(pm["history"]),
                         rep=pm.get("rep"), mc_err=pm.get("mc_err"), topup_to=pm.get("topup_to"))
                return r

            def side_ok(r):
                return bool(r["complete"] and r["khat"] < khat_ok and r["ess"] >= ess_min * nk)

            # non-grazing side
            T_n = _StratumCoords(self, "beta")
            lp_n, r_n = -np.inf, None
            sc = self._default_scales(x_ref)
            for x_s in (x_ref, x0):                # from the maximum so far, or from the start if it fails there
                x_n = np.array(x_s, float)
                g_s = float(self._conj_factor(x_n)[0])
                if x_n[jb] * g_s >= 1.0 - x_n[jr]:
                    # at the same a/R*, not the same T14, which at this b can lie outside the bounds on a/R*
                    X_p = self.to_physical(x_n)
                    X_p[0, jb] = 0.9 * max(1.0 - x_n[jr], 0.01) / g_s
                    x_n = self.to_internal(X_p)[0]
                s_n = self._default_scales(x_n)
                s_n[jb] = 0.1
                r_n = profile_is(T_n.log_prob, T_n.to_u(x_n)[0], s_n, prof=jb, n=nk, rng=rng, lo_wall=1e-6, hi_wall=1.0 - 1e-6,
                                 clip_map=True)
                nev.append(r_n["n_eval"])
                if np.all(np.isfinite(r_n["theta_map"])):
                    lp_n = max(lp_n, float(T_n.log_prob(r_n["theta_map"])[0] - T_n.to_x(r_n["theta_map"])[1][0]))
                if np.isfinite(r_n["log_int"]):
                    break
            # grazing side
            T_g = _StratumCoords(self, "graze")
            u_g, k_g = self._grazing_start(T_g, x_ref)     # the duration of the maximum so far, or of the start
            u_0, k_0 = self._grazing_start(T_g, x0)
            nev.append(k_g + k_0)
            if T_g.log_prob(u_0)[0] > T_g.log_prob(u_g)[0]:
                u_g = u_0
            s_g = sc.copy()
            s_g[jr], s_g[jb] = 0.1, 0.1
            r_g = profile_is(T_g.log_prob, u_g, s_g, prof=jb, n=nk, rng=rng, lo_wall=1e-6, hi_wall=1.0 - 1e-6, clip_map=True)
            nev.append(r_g["n_eval"])
            lp_g = -np.inf
            if np.all(np.isfinite(r_g["theta_map"])):
                lp_g = float(T_g.log_prob(r_g["theta_map"])[0] - T_g.to_x(r_g["theta_map"])[1][0])
            if not np.isfinite(r_n["log_int"]) and np.isfinite(r_g["log_int"]):
                # the density is continuous across the boundary: from where the grazing side's depth meets it
                X_g = T_g.to_x(r_g["theta_map"])[0][0]
                ln_lam = float(r_g["theta_map"][jr])
                x_n = np.array(X_g)
                x_n[jr] = max(math.exp(0.5 * (ln_lam - math.log(2.0))), self.lo[2] * 1.001)
                g_s = float(self._conj_factor(x_n)[0])
                x_n[jb] = 0.98 * (1.0 - x_n[jr]) / g_s
                s_n = self._default_scales(x_n)
                s_n[jb] = 0.1
                r_n = profile_is(T_n.log_prob, T_n.to_u(x_n)[0], s_n, prof=jb, n=nk, rng=rng, lo_wall=1e-6, hi_wall=1.0 - 1e-6,
                                 clip_map=True)
                nev.append(r_n["n_eval"])
                if np.all(np.isfinite(r_n["theta_map"])):
                    lp_n = max(lp_n, float(T_n.log_prob(r_n["theta_map"])[0] - T_n.to_x(r_n["theta_map"])[1][0]))
            # both sides need a proposal: a side counts as empty only by its measured mass, never because it has no
            # proposal or its optimiser ended low (the optimiser can fail on a side that holds most of the posterior)
            n_ok, g_ok = np.isfinite(r_n["log_int"]), np.isfinite(r_g["log_int"])
            if not (n_ok and g_ok):
                flags["strata_note"] = "no proposal on " + ("the grazing side" if n_ok else "the non-grazing side" if g_ok else "either side")
                return None
            # the side of the maximum so far must reach it (else its optimiser ended in a local maximum)
            ref_bc = x_ref[jb] * float(self._conj_factor(x_ref)[0])
            lp_side = lp_n if ref_bc < 1.0 - x_ref[jr] else lp_g
            if lp_side < lp_ref - 2.0:
                flags["strata_note"] = f"the side of the maximum ended at {lp_side:.1f}, the maximum is {lp_ref:.1f}"
                return None
            sides = [("nongrazing", T_n, r_n), ("grazing", T_g, r_g)]
            for lab, T, r in sides:
                r.setdefault("amis_iters", 0)
            mass = np.exp(np.array([r["log_int"] for _, _, r in sides])
                          - logsumexp([r["log_int"] for _, _, r in sides]))
            out = []
            for (lab, T, r), mk in zip(sides, mass):
                if mk >= _STRATUM_MIN_MASS and not side_ok(r):
                    r = refine(T, r)
                    if r is None:
                        return None
                out.append((lab, T, r))

            def merge():
                li = np.array([r["log_int"] for _, _, r in out])
                total = logsumexp(li)
                mk = np.exp(li - total)
                th = np.vstack([T.to_x(r["theta"])[0] for _, T, r in out])
                w = np.concatenate([r["w"] / r["w"].sum() * m_ for (_, _, r), m_ in zip(out, mk)])
                return th, w, total, mk
            th, w, total, mk = merge()
            ess = float(1.0 / np.sum(w ** 2))
            # the draws of the first profile are a coarse independent estimate of the masses
            pw = np.asarray(res["w"], float)
            if pw.sum() > 0 and len(pw) > 1:
                Xf = np.atleast_2d(res["theta"])
                ng = Xf[:, jb] * self._conj_factor(Xf) >= 1.0 - Xf[:, jr]
                m_first = dict(grazing=float(pw[ng].sum() / pw.sum()), nongrazing=float(pw[~ng].sum() / pw.sum()))
                for (lab, _, _), m_ in zip(out, mk):
                    if m_first[lab] > 0.05 and m_ < 0.1 * m_first[lab]:
                        flags["strata_note"] = f"{lab}: mass {m_:.3g}, against {m_first[lab]:.3g} from the first profile"
                        return None
            if ess < ess_min * n:                # the draws of the side with the mass are too few
                j = int(np.argmax(mk))
                r = refine(out[j][1], out[j][2])
                if r is None:
                    return None
                out[j] = (out[j][0], out[j][1], r)
                th, w, total, mk = merge()
                ess = float(1.0 / np.sum(w ** 2))
            err = np.array([r["log_int_err"] for _, _, r in out])
            relevant = [r["khat"] for (_, _, r), m_ in zip(out, mk) if m_ >= _STRATUM_MIN_MASS]
            strata = {}
            for (lab, T, r), m_ in zip(out, mk):                               # a share depends on both integrals
                strata[lab] = dict(mass=float(m_), mass_err=float(m_ * (1.0 - m_) * math.sqrt(np.sum(err ** 2))), khat=float(r["khat"]),
                                   ess=float(r["ess"]), n_grid=int(r["n_grid"]), amis_iters=int(r["amis_iters"]),
                                   coord=T.kind)
            with np.errstate(divide="ignore"):
                logw = np.log(w)
            def labels_of():
                return _cat_labels(*[r.get("rep") if r.get("rep") is not None and len(r["rep"]) == len(r["theta"]) else None
                                     for _, _, r in out])
            rep_ = labels_of()

            def strata_topup(tol, max_add, min_ess=None):
                """Top up each side with mass, then merge again. A side's error enters the combined posterior times its
                mass, err^2 ~ sum (m_k e_k)^2, so each side gets tol / |m| (|m| = sqrt(sum m_k^2); about tol for one
                dominant side), at most max_add / 2 draws each. ESS floor: min_ess x mass per side (then the combined
                ESS 1 / sum(m_k^2 / ESS_k) >= min_ess)."""
                li0 = np.array([r["log_int"] for _, _, r in out])
                m0 = np.exp(li0 - logsumexp(li0))
                added = 0
                for i, (lab, T, r) in enumerate(out):
                    if m0[i] >= _STRATUM_MIN_MASS and r.get("topup_to") is not None:
                        f = r["topup_to"](None if tol is None else tol / float(np.sqrt(np.sum(m0 ** 2))), max_add // len(out),
                                          None if min_ess is None else min_ess * float(m0[i]))
                        if f["n_added"]:
                            r = dict(r)
                            r.update({k: f[k] for k in _TOP})
                            out[i] = (lab, T, r)
                            added += f["n_added"]
                th2, w2, total2, mk2 = merge()
                rep2 = labels_of()
                with np.errstate(divide="ignore"):
                    lw2 = np.log(w2)
                err2 = np.array([r["log_int_err"] for _, _, r in out])
                rel2 = [r["khat"] for (_, _, r), m_ in zip(out, mk2) if m_ >= _STRATUM_MIN_MASS]
                return dict(theta=th2, w=w2, logw=lw2, khat=float(max(rel2)), ess=float(1.0 / np.sum(w2 ** 2)), rep=rep2,
                            mc_err=mc_error(th2, w2, rep2), log_int=float(total2),
                            log_int_err=float(math.sqrt(np.sum((mk2 * err2) ** 2))), n_added=added)
            return dict(theta=th, w=w, logw=logw, khat=float(max(relevant)), ess=ess, n_eval=0,
                        rep=rep_, mc_err=mc_error(th, w, rep_), topup_to=strata_topup,
                        theta_map=res["theta_map"], grid_b=res["grid_b"], grid_logdens=res["grid_logdens"], stage="strata",
                        n_grid=int(sum(r["n_grid"] for _, _, r in out)), complete=True, n_eval_map=res["n_eval_map"],
                        n_refine=0, grid_lpmax=res["grid_lpmax"], grid_logdet=res["grid_logdet"], log_int=float(total),
                        log_int_err=float(math.sqrt(np.sum((mk * err) ** 2))), strata=strata)

        def run_flat():
            if not (flat_proposal and self.cone and not self.has_ldp):
                return None

            def lp_flat(X):
                lp, lP = self.log_prob_internal(X, parts=True)
                return lp - lP
            r = profile_is(lp_flat, x0, self._default_scales(x0), prof=jb, n=n, rng=rng,
                           logp_target=self.log_prob_internal, **walls)
            nev.append(r["n_eval"])
            return r

        jr = self.free.index("rp") if "rp" in self.free else None
        hi_rp = self.hi[2]
        g_map = float(self._conj_factor(res["theta_map"])[0]) if np.all(np.isfinite(res["theta_map"])) else 1.0
        can_graze = bool(jr is not None and lo_b <= 0.0 and hi_b * g_map >= 1.0 + hi_rp)
        graze = bool(can_graze and np.max(res["grid_b"]) * g_map + res["theta_map"][jr] > 1.0)

        def run_retry(start):
            T = _BoundedCoords(self, graze=graze) if retry else None
            if T is None or not T.active:
                return None
            if graze and not np.isfinite(hi_rp):
                warnings.warn("grazing geometry without an upper bound on rp: the posterior along "
                              "b - rp = const can be improper; set bounds['rp']", stacklevel=3)
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
            wl = dict(lo_wall=1e-9, hi_wall=1.0 - 1e-9) if graze else walls
            r = profile_is(T.log_prob, u0, su, prof=jb, n=n, rng=rng, **wl)
            nev.append(r["n_eval"])
            if np.isfinite(r["khat"]) and np.sum(r["w"] > 0) > 50:
                # the Laplace steps are unreliable here (posterior against a bound: flat and skewed
                # in the transformed coordinates), so the proposal is adapted to the weighted draws
                pm = pmc_refine(T.log_prob, r["theta"], r["logw"], rng, n=n)
                nev.append(pm["n_eval"])
                r.update(theta=pm["theta"], w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"],
                         stage="pmc", pmc=pm["history"], log_int=pm["log_int"], log_int_err=pm["log_int_err"],
                         rep=pm.get("rep"), mc_err=pm.get("mc_err"), topup_to=pm.get("topup_to"))
            r["topup_to"] = in_x(r.get("topup_to"), T)        # its proposal is in the retry's coordinates
            r["theta"] = T.to_x(r["theta"])[0]
            r["theta_map"] = T.to_x(np.atleast_2d(r["theta_map"]))[0][0]
            return r

        def run_rescue(best):
            """AMIS from the pooled weighted draws of all attempts, in unbounded coordinates (the
            retry's: logit / log of the bounded parameters, beta for a grazing posterior), with the
            defensive components of pmc_refine plus one over the whole prior box."""
            from scipy.special import logsumexp
            # beta = b_conj / (1 + rp) whenever the bounds leave the whole support 0 <= beta < 1
            # (not only if the first profile reached a grazing geometry), as logit(beta)
            T = _BoundedCoords(self, graze=can_graze, logit_beta=True)
            to_u = T.to_u if T.active else (lambda X: np.atleast_2d(np.asarray(X, float)).copy())
            lp_u = T.log_prob if T.active else self.log_prob_internal
            Us, lws = [], []
            seeds = [r for r in tried if r is not None and np.sum(np.asarray(r["w"]) > 0) > 50]
            if self.gp is not None:
                seeds.append(self._gp_seed(res["theta_map"] if np.all(np.isfinite(res["theta_map"])) else x0, n, rng, nev))
            elif not seeds and T.active:         # no attempt has draws (a flat posterior: no conditional Hessian)
                seeds.append(self._box_seed(T, res["theta_map"] if np.all(np.isfinite(res["theta_map"])) else x0, n, rng, nev))
            for r in seeds:
                lw = np.asarray(r["logw"], float)
                U = to_u(r["theta"])
                ok = np.isfinite(lw) & np.all(np.isfinite(U), axis=1)
                if ok.sum() < 10:
                    continue
                Us.append(U[ok])
                lws.append(lw[ok] - logsumexp(lw[ok]))          # each attempt: total weight 1
            out = dict(best)
            out["rescue"] = False
            if not Us:
                return out
            U0, lw0 = np.vstack(Us), np.concatenate(lws)
            w0 = np.exp(lw0 - logsumexp(lw0))
            mu0 = w0 @ U0
            sd0 = np.sqrt(np.clip(np.diag(np.atleast_2d(np.cov(U0.T, aweights=w0))), 1e-300, None))
            # box component: logistic-like width in the logit coordinates (a uniform x over the box
            # is a standard logistic in u, sd pi / sqrt 3), 5 x the draws' sd elsewhere
            mb, sb = mu0.copy(), 5.0 * sd0
            if T.active:
                logit_cols = [j for j, kind, _, _ in T.cols if kind in ("box", "a14")]
                if T.logit_beta:
                    logit_cols.append(T.jb)
                for j in logit_cols:
                    mb[j], sb[j] = 0.0, max(np.pi / np.sqrt(3.0), 5.0 * sd0[j])
            iters, capped = rescue_iters, False
            if budget is not None and budget.max_evals is not None:      # whole iterations of n draws
                iters = min(rescue_iters, budget.remaining_evals() // n)
                capped = iters < rescue_iters
                if iters < 1:
                    out["budget_exceeded"] = True
                    return out
            pm = pmc_refine(lp_u, U0, lw0, rng, n=n, iters=iters, min_iters=3,
                            extra=[(0.03, mb, np.diag(sb ** 2))], ess_min=ess_min)
            nev.append(pm["n_eval"])
            if not np.isfinite(pm["khat"]):
                return out
            X = T.to_x(pm["theta"])[0] if T.active else pm["theta"]
            out.update(rep=pm.get("rep"), mc_err=pm.get("mc_err"), topup_to=in_x(pm.get("topup_to"), T) if T.active
                       else pm.get("topup_to"))
            out.update(theta=X, w=pm["w"], logw=pm["logw"], khat=pm["khat"], ess=pm["ess"], stage="amis",
                       pmc=pm["history"], rescue=True, is_ok=bool(pm["converged"]),
                       log_int=pm["log_int"], log_int_err=pm["log_int_err"],
                       budget_exceeded=bool(capped and not pm["converged"]))
            return out

        def stopped(r):
            """The budget ran out: the best attempt so far, flagged."""
            r = dict(r)
            r.update(flags)
            r.update(grazing_beta=graze, rescue=r.get("rescue"), is_ok=False, budget_exceeded=True,
                     n_eval=int(budget.used))
            return r

        best = res
        try:
            if usable(res):
                return finish(res)
            start = res["theta_map"] if np.all(np.isfinite(res["theta_map"])) else x0
            order = ("flat", "retry") if res["complete"] else ("retry", "flat")
            can_strat = bool(stratify and can_graze and np.isfinite(hi_rp) and self.gp is None and not self.secondary)
            if can_strat:
                # first, unless a closed profile short of the boundary points to a cut-off by the cone
                order = (("flat", "strata", "retry") if res["complete"] and not graze else ("strata", "flat", "retry"))
            for step in order:
                r = dict(flat=run_flat, retry=lambda: run_retry(start), strata=lambda: run_strata(start))[step]()
                if r is None:
                    continue
                tried.append(r)
                key = dict(flat="flat_proposal", retry="bounded_retry", strata="stratified")[step]
                flags[key] = False
                if better(r, best):
                    if best is not res:
                        for other in ("flat_proposal", "bounded_retry", "stratified"):
                            if other != key and flags[other] is not None:
                                flags[other] = False
                    best = r
                    flags[key] = True
                if usable(r):
                    break
            return finish(best)
        except _BudgetExceeded:
            return stopped(best)

    def _supersample_check(self, Xw, wc, exp_time):
        """(chi^2 of the exposure average with self.k sub-exposures, the k that reaches 0.01) at the weighted median
        of the physical draws Xw (pizzetti.supersample); None if it cannot be evaluated."""
        from .supersample import choose_supersample, exposure_chi2
        try:
            med = np.empty(Xw.shape[1])
            for j in range(Xw.shape[1]):
                o = np.argsort(Xw[:, j])
                c = np.cumsum(wc[o])
                med[j] = Xw[o, j][min(np.searchsorted(c, 0.5 * c[-1]), len(o) - 1)]
            P = self._canonical_physical(med[None, :])[0]
            p = copy.copy(self._params0)
            if self.secondary:
                p.t_secondary = P[0]
            else:
                p.t0 = P[0]
            p.per, p.rp, p.a = P[1], P[2], P[3]
            p.inc = math.degrees(math.acos(min(abs(P[4]) / P[3], 1.0)))
            p.ecc, p.w = P[5], math.degrees(P[6])
            sig = np.sqrt(self.gp_var0) if self.gp is not None else 1.0 / self.wts
            chi2 = exposure_chi2(p, self.t, exp_time, sig, self.k, offsets=self.offsets, transittype=self._transittype)
            k_need = self.k if chi2 <= 0.01 else choose_supersample(p, self.t, exp_time, sig, offsets=self.offsets,
                                                                     transittype=self._transittype)
            return float(chi2), int(k_need)
        except Exception:                    # a diagnostic only: never let it break a fit
            return None

    def fit(self, method="auto", n=2048, nwalkers=32, max_steps=100_000, ntau=50, seed=None,
            ld="q", progress=False, max_evals=None, max_time=None, stratify=True, qmc=True, mc_reps=None,
            mc_tol=0.03, mc_max_factor=4.0, min_ess=4000):
        """Posterior samples in one call.

        Importance sampling (method "is", or "auto" with b free) draws ``n`` points per stage, then adds draws from its
        final proposal until two targets are met (at most ``mc_max_factor`` x n more):

        * ``mc_tol``: the Monte Carlo error of the posterior quantiles (2.5, 16, 50, 84, 97.5 % of every free parameter;
          rms, in posterior sd), measured from the spread of ``mc_reps`` (default 8) independent replicates of the draws;
          in ``diagnostics["mc_err"]`` / ``"mc_err_max"``. None: no error target.
        * ``min_ess``: a floor on the effective sample size of the output (Kish, after the limb-darkening step), so
          that a corner plot of the equal-weight draws has about that many distinct points (4000: twice the minimum
          effective sample size of Agarwal, Vats & Elvira 2021, arXiv:2108.13289, for four parameters at 10 % relative
          precision; 2000 gave visibly noisy corner plots). None: no floor.

        ``qmc`` (default True): randomised quasi-Monte Carlo draws (scrambled Sobol points through the inverse CDFs of
        the proposals, one independent scramble per replicate). Each draw has the proposal's distribution, so the
        estimates stay consistent; the points are spread more evenly, which lowers the variance of the quantiles about
        5-fold at equal draws on 79 benchmark light curves (Kepler, K2, TESS, CHEOPS). The ESS does not see this gain (the
        weights look as for iid draws), hence the replicate error. ``qmc=False, mc_tol=None, min_ess=None, n=4000``
        gives the iid sampler of pizzetti <= 0.4.1 (identical output for the same seed).

        method="is": profile importance sampling along b (needs "b" in free), with the adaptive
        rescue of ``profile_is`` where the profile is not usable; no fallback to MCMC: a warning if
        the final diagnostics fail (``diagnostics["is_ok"]``); ``max_evals`` / ``max_time`` (seconds)
        limit its cost (``profile_is``; ``diagnostics["budget_exceeded"]``), and ``stratify=False`` skips the
        split at the grazing boundary; the evidence is in
        ``diagnostics["lnz"]``, ``"lnz_err"``, ``"lnz_note"``, ``"log_int"``, ``"log_int_err"`` (``profile_is``);
        "emcee": vectorised emcee until convergence (N > ntau * tau, tau stable to 1 %);
        "auto": "is" if b is free, else "emcee". Returns a ``FitResult``.
        """
        self._ne_cache = [] if (self.fast and self.gp is None and not self.has_ldp) else None
        try:
            return self._fit(method, n, nwalkers, max_steps, ntau, seed, ld, progress, max_evals, max_time, stratify,
                             qmc, mc_reps, mc_tol, mc_max_factor, min_ess)
        finally:
            self._ne_cache = None

    def _fit(self, method, n, nwalkers, max_steps, ntau, seed, ld, progress, max_evals, max_time, stratify, qmc,
             mc_reps, mc_tol, mc_max_factor, min_ess):
        from ._sampling import hessian, reset_walkers, stuck_walkers
        from ._sampling import mc_error, mc_options
        rng = np.random.default_rng(seed)
        if mc_reps is None:
            mc_reps = 8 if (qmc or mc_tol is not None or min_ess is not None) else 0
        mc = dict(qmc=bool(qmc), reps=int(mc_reps), tol=mc_tol, max_factor=float(mc_max_factor), min_ess=min_ess)
        rep = None
        self._last_topup = None
        import time as _time
        t_fit = _time.perf_counter()
        self.n_eval = 0
        x0 = self._start_internal()
        scale = self._default_scales(x0)
        diag = dict(method=None)
        Xw = w = res = None
        if method in ("is", "auto") and "b" in self.free:
            with mc_options(**mc):
                res = self.profile_is(n=n, rng=rng, x0=x0, max_evals=max_evals, max_time=max_time, evidence=False,
                                      stratify=stratify)
            rep = res.get("rep")
            if rep is not None and len(rep) != len(res["theta"]):
                rep = None
            ok = bool(res["is_ok"])
            diag.update(method="profile-IS", khat=res["khat"], ess=res["ess"],
                        complete=res["complete"], n_grid=res["n_grid"], is_ok=ok,
                        bounded_retry=res["bounded_retry"], flat_proposal=res["flat_proposal"],
                        last_resort_pmc=res["last_resort_pmc"], grazing_beta=res["grazing_beta"],
                        stratified=res.get("stratified"), strata=res.get("strata"), strata_note=res.get("strata_note"),
                        grazing_mass=(res["strata"]["grazing"]["mass"] if "grazing" in res.get("strata", {}) else None),
                        rescue=res["rescue"], rescue_iters=len(res.get("pmc") or []) if res["rescue"] else 0,
                        budget_exceeded=bool(res.get("budget_exceeded")))
            # no fallback to MCMC: the importance draws are returned in any case, with a warning if
            # their diagnostics fail (method="emcee" runs the sampler)
            Xw, w = self.to_physical(res["theta"]), res["w"]
            if res.get("budget_exceeded"):
                warnings.warn(f"importance sampling stopped at its budget (max_evals={max_evals}, max_time="
                              f"{max_time} s) with k-hat {res['khat']:.2f}, ESS {res['ess']:.0f}: the draws are "
                              "the best attempt so far, not a converged posterior", stacklevel=3)
            elif not ok:
                warnings.warn(f"importance sampling diagnostics failed (k-hat {res['khat']:.2f}, ESS "
                              f"{res['ess']:.0f}); check the posterior, e.g. with method='emcee'",
                              stacklevel=3)
            x0 = res["theta_map"]
        elif method == "is":
            raise ValueError('method="is" needs "b" in free')
        if Xw is None:
            if method not in ("emcee", "auto"):
                raise ValueError('method must be "auto", "is" or "emcee"')
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
            state, start, flagged, n_reset = P0, 0, np.zeros(0, dtype=int), 0
            # chunks of 500 steps (the backend grows with the chain instead of allocating max_steps)
            while sampler.iteration < max_steps:
                state = sampler.run_mcmc(state, min(500, max_steps - sampler.iteration),
                                         progress=progress)
                # a walker stuck far below the others at two checks in a row (a local mode, a
                # plateau) is moved next to another walker; burn-in and convergence start again
                lp = sampler.get_log_prob(discard=max(start, sampler.iteration - 500))
                st = stuck_walkers(lp)
                again = np.intersect1d(st, flagged)
                flagged = st
                if again.size:
                    state = reset_walkers(state.coords, again, rng, self.log_prob_internal)
                    start, old, flagged = sampler.iteration, np.inf, np.zeros(0, dtype=int)
                    n_reset += int(again.size)
                    continue
                tau = sampler.get_autocorr_time(discard=start, tol=0)
                if (np.all(np.isfinite(tau)) and np.all(ntau * tau < sampler.iteration - start)
                        and np.all(np.abs(old - tau) / tau < 0.01)):
                    conv = True
                    break
                old = tau
            tau = sampler.get_autocorr_time(discard=start, tol=0)
            # tau is NaN for a parameter that has not moved (e.g. walkers stuck at a bound)
            n_run = sampler.iteration - start
            burn = int(5 * np.max(tau)) if np.all(np.isfinite(tau)) else n_run // 2
            ch = sampler.get_chain(discard=start + min(burn, n_run // 2), flat=True, expand=False)
            ch = ch[:: max(1, len(ch) // n)]
            Xw, w = ch, np.full(len(ch), 1.0 / len(ch))
            diag.update(method="emcee",
                        converged=conv, steps=int(sampler.iteration), tau=tau.tolist(),
                        walkers_reset=n_reset)
        diag["n_eval"] = int(self.n_eval)
        # coefficients + baseline for every weighted geometry draw; one combined weight,
        # then a single systematic resampling to equal weight. Draws with zero weight (outside
        # the prior, e.g. beyond b = 1 + rp for a grazing transit, where the design is
        # degenerate) are dropped first.
        pos = w > 0
        Xw, w = Xw[pos], w[pos]
        xi = res["theta"][pos] if res is not None else None
        if rep is not None:
            rep = np.asarray(rep)[pos]
        S_, labels, wl = self.expand(Xw, ld=ld, rng=rng, resample=False, _xi=xi)
        room = int(mc_max_factor * n)                   # the budget of fit() binds this top-up too
        if max_evals is not None:
            room = min(room, int(max_evals) - int(self.n_eval))
        late = max_time is not None and _time.perf_counter() - t_fit > max_time
        if (res is not None and min_ess is not None and self._last_topup is not None and room > 0 and not late
                and not res.get("budget_exceeded")):
            # the floor refers to the output, after the limb-darkening weights: top up the geometry draws by the
            # ESS the LD step loses, then expand again
            wc0 = w * wl
            ess_ld = float(1.0 / np.sum((wc0 / wc0.sum()) ** 2))
            if ess_ld < min_ess:
                ess_g = float(1.0 / np.sum((w / w.sum()) ** 2))
                with mc_options(**mc):
                    f = self._last_topup(mc_tol, room, 1.05 * min_ess * ess_g / max(ess_ld, 1.0))
                if f["n_added"]:
                    res.update({k_: f[k_] for k_ in ("theta", "w", "logw", "khat", "ess", "rep", "mc_err")})
                    res["n_topup"] = int(res.get("n_topup", 0)) + int(f["n_added"])
                    Xw, w = self.to_physical(res["theta"]), res["w"]
                    rep = res.get("rep")
                    pos = w > 0
                    Xw, w = Xw[pos], w[pos]
                    rep = None if rep is None or len(rep) != len(pos) else np.asarray(rep)[pos]
                    S_, labels, wl = self.expand(Xw, ld=ld, rng=rng, resample=False, _xi=res["theta"][pos])
                    diag.update(n_eval=int(self.n_eval), khat=float(res["khat"]), ess=float(res["ess"]))
        self._last_topup = None
        diag["ld_reweight_ess"] = float(1.0 / np.sum(wl ** 2))
        # geometry weight of draws without a valid coefficient draw (dropped; see _chol_rows)
        diag["ld_dropped_weight"] = float(np.sum(w[wl == 0]) / np.sum(w))
        diag["ld_from_prior"] = bool(getattr(self, "last_ld_from_prior", False))
        if self.physical and self.law.D is None:
            diag["physical_fraction"] = self.last_physical_frac
        if res is not None:
            diag.update(log_int=res["log_int"], log_int_err=res["log_int_err"])
            diag.update(self._lnz_from_ld(res["log_int"], res["log_int_err"], self._ld_logw, w) if self.fast
                        else dict(lnz=np.nan, lnz_err=np.nan, lnz_note=self._lnz_note()))
        wc = w * wl
        wc /= wc.sum()
        diag["ess_total"] = float(1.0 / np.sum(wc ** 2))
        if mc["reps"] >= 2:
            # replicate Monte Carlo error of fit()'s output (free parameters, after the limb-darkening step)
            diag["mc_err"], diag["mc_err_max"] = mc_error(Xw, wc, rep)
            diag.update(mc_qmc=mc["qmc"], mc_reps=mc["reps"], mc_tol=mc_tol,
                        n_topup=int(res.get("n_topup", 0)) if res is not None else 0)
        if not self.physical and "rp" in self.free:
            diag["collapsed_fraction"] = self._collapsed_fraction(Xw, wc)
            if diag["collapsed_fraction"] > 0.5:
                warnings.warn(f'ld_prior="flat": {diag["collapsed_fraction"]:.0%} of the posterior is where the planet\'s '
                              "area vanishes (rp -> 0 or b -> 1 + rp), up to the bounds, and coefficients ~ 1 / rp^2 can "
                              "still give the observed dip. This prior is improper there (at fixed b, marginal likelihood "
                              f"~ rp^-{2 * (self.law.K - 1)}), so this is no measurement of rp or the limb darkening; "
                              'use ld_prior="physical"', stacklevel=3)
        # the exposure average at the posterior median (no draws: the rng stream is unchanged); without an exposure
        # time, exposures as long as the cadence are assumed
        exp_chk, assumed = self.exp_time, False
        if not exp_chk > 0 and self.k == 1:
            steps = np.diff(self.t)              # in the given order: phase-folded, time-ordered data keep their cadence
            steps = steps[steps > 0]
            exp_chk, assumed = (float(np.median(steps)) if len(steps) else 0.0), True
        if exp_chk > 0:
            chk = self._supersample_check(Xw, wc, exp_chk)
            if chk is not None:
                diag["supersample_chi2"], k_need = chk
                if assumed:
                    diag["supersample_exp_assumed"] = exp_chk
                if diag["supersample_chi2"] > 0.1:
                    shift = math.sqrt(diag["supersample_chi2"])
                    shift = f"{shift:.0f}" if shift >= 10 else f"{shift:.1f}"
                    if assumed:
                        warnings.warn(f"no exp_time given: if each point is an exposure as long as the cadence "
                                      f"({exp_chk * 1440:.1f} min), the model without exposure average is off by chi^2 = "
                                      f"{diag['supersample_chi2']:.3g} at the posterior median, which can move the posterior "
                                      f"by up to ~{shift} sigma; pass exp_time={exp_chk:.6g} with supersample_factor="
                                      f'{k_need} (or "auto"); if the exposures are shorter, pass their exp_time (tutorial 22)',
                                      stacklevel=3)
                    else:
                        warnings.warn(f"supersample_factor={self.k}: the model's exposure average is off by chi^2 = "
                                      f"{diag['supersample_chi2']:.3g} at the posterior median, which can move the posterior "
                                      f"by up to ~{shift} sigma; use more sub-exposures (supersample_factor={k_need} "
                                      "reaches chi^2 <= 0.01 there)", stacklevel=3)
        j = systematic_resample(wc, len(wc), rng)
        orbit = dict(per=self.fixed[1], rp=self.fixed[2], a=self.fixed[3], b=self.fixed[4], ecc=self.fixed[5],
                     a_bounds=(self.lo[3], self.hi[3]), secondary=self.secondary, free=tuple(self.free))
        return FitResult(S_[j], labels, weighted=(S_, wc), diagnostics=diag, orbit=orbit)

    def _collapsed_fraction(self, X, w):
        """Share of the weights w of physical rows X with hardly any planet area: effective radius below _COLLAPSED_RP
        or 2 x the lower bound on rp, with the squared radius rp^2, or about rp (1 + rp - b_conj) for a grazing transit."""
        P = self._canonical_physical(X)
        rp = P[:, 2]
        g = (1.0 - P[:, 5] ** 2) / (1.0 + (-1.0 if self.secondary else 1.0) * P[:, 5] * np.sin(P[:, 6]))
        r2 = rp * np.clip(1.0 + rp - P[:, 4] * g, 0.0, rp)
        return float(np.sum(w[r2 < max(2.0 * self.lo[2], _COLLAPSED_RP) ** 2]) / np.sum(w))

    def fixed_physical_free(self):
        X = np.empty((1, self.ndim))
        X[0, self.geo_cols] = self.fixed[self.idx]
        if "w" in self.geo_names:
            X[0, self.geo_cols[self.geo_names.index("w")]] /= _DEG
        if self.alpha_col >= 0:
            X[0, self.alpha_col] = self.alpha_fixed
        if self.gp is not None:
            X[0, self.gp_cols] = self.gp_x0
        return X

    def _start_internal(self):
        """Internal start from the TransitParams values. A non-transiting start (b_conj >= 1 + rp:
        the transit duration, and with it a/R* in the T14 parametrisation, is undefined there and
        the log posterior is -inf) gets b at half the transiting range, b_conj = (1 + rp) / 2."""
        X = self.fixed_physical_free()
        if "b" in self.free:
            P = self._canonical_physical(X)
            rp, b, e, w = P[0, 2], P[0, 4], P[0, 5], P[0, 6]
            g = (1.0 - e * e) / (1.0 + (-1.0 if self.secondary else 1.0) * e * math.sin(w))
            if not (abs(b) * g < 1.0 + abs(rp)):
                X[0, self.free.index("b")] = 0.5 * (1.0 + abs(rp)) / g
        return self.to_internal(X)[0]

    def _default_scales(self, x0):
        P = self.fixed
        T14 = P[1] / math.pi * math.asin(min(1.0, (1 + P[2]) / max(P[3], 1.0 + P[2] + 1e-9)))
        sc = {"t0": 1e-2 * T14, "t_secondary": 1e-2 * T14, "per": 1e-6 * P[1], "rp": 1e-2 * P[2],
              "a": (1e-2 * T14 if self.t14 else 1e-2 * P[3]), "b": 0.05, "ecc": 0.01, "w": 1.0,
              "alpha": 0.05}
        out = np.array([sc[q] for q in self.free])
        if self.gp is not None:
            gs = np.array([0.2, 0.2, 0.05])[self.gp_free]
            out = np.concatenate([out, gs])
        return out


def _contact_times(per, rp, a, bo, e, om):
    """Times of first and last contact of a transit, relative to conjunction, on a Keplerian orbit (arrays): solves
    z(f) = 1 + rp for the true anomaly on both sides of conjunction, f = pi / 2 - om + u, z^2 = r(f)^2 (cos^2 i + sin^2 i sin^2 u)."""
    cos2 = (bo / a) ** 2
    fc = 0.5 * np.pi - om
    p_ = a * (1.0 - e * e)
    M = np.empty((2,) + np.shape(e))

    def mean_anomaly(f):
        E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * f), np.sqrt(1.0 + e) * np.cos(0.5 * f))
        return E - e * np.sin(E)

    m0 = mean_anomaly(fc)
    for k, sign in enumerate((-1.0, 1.0)):
        u = np.zeros_like(e)
        for _ in range(8):                                     # r varies slowly over a transit: a contraction
            r = p_ / (1.0 + e * np.cos(fc + sign * u))
            u = np.arcsin(np.sqrt(np.maximum(((1.0 + rp) / r) ** 2 - cos2, 0.0) / (1.0 - cos2)))
        M[k] = (mean_anomaly(fc + sign * u) - m0 + np.pi) % (2.0 * np.pi) - np.pi
    return per / (2.0 * np.pi) * M[0], per / (2.0 * np.pi) * M[1]


class FitResult:
    """Result of ``MarginalTransitModel.fit``: equal-weight ``samples`` (emcee-like
    flat array, systematic resampling) with ``labels``; ``weighted_samples`` and
    ``weights`` are the exact weighted draws (preferred for histograms / corner plots:
    resampled draws repeat values in proportion to their weights)."""

    def __init__(self, samples, labels, weighted, diagnostics, orbit=None):
        self.samples = samples
        self.labels = labels
        self.weighted_samples, self.weights = weighted
        self.diagnostics = diagnostics
        self._orbit = orbit
        self._parents = None                 # reweight_density: row of the fit that each weighted draw comes from

    def summary(self):
        """Table of median and 16th/84th-percentile offsets for every column."""
        q = np.percentile(self.samples, [16, 50, 84], axis=0)
        wl = max(5, *(len(name) for name in self.labels))
        lines = [f"{'':{wl}s} {'median':>14s} {'-1sigma':>11s} {'+1sigma':>11s}"]
        for j, name in enumerate(self.labels):
            lines.append(f"{name:{wl}s} {q[1, j]:14.7g} {q[1, j] - q[0, j]:11.3g} {q[2, j] - q[1, j]:11.3g}")
        return "\n".join(lines)

    def corner(self, **kwargs):
        """Corner plot of the weighted draws (no resampling duplicates)."""
        import corner
        return corner.corner(self.weighted_samples, weights=self.weights, labels=self.labels, **kwargs)

    def reweight_density(self, rho, ecc_max=0.92, log_prior=None, repeats=20, min_ess=2000, seed=None):
        """Eccentricity from a stellar density: the posterior of (e, omega) and the rest under a Gaussian prior on the
        density, from this fit of a circular orbit, with no new likelihood evaluation.

        A transit hardly differs from that of a circular orbit with the same duration, centre and depth: only the
        density implied by a/R* changes (Dawson & Johnson 2012; MacDougall et al. 2023). Each draw of the fit is
        repeated with (e, omega) from their prior, with a/R* such that the duration T14 of the Keplerian orbit is
        unchanged and t0, the time of conjunction, such that the middle of the transit is, and weighted by
        N(rho) |d rho / d T14| / |d a_circular / d T14|, since the fit is flat in the circular a/R*.

        Calibrated on 1100 light curves drawn from this prior (tutorial 20). Above an S/N of about 1000 the transit shape
        differs measurably: e, omega and a stay calibrated, but the intervals of b_conj and rp are too narrow, with a 95 %
        interval covering 86 % and 92 %.

        The prior is N(rho) on the density, flat in e and omega (or ``log_prior``), flat in rp and b_conj with
        b_conj < 1 + rp, as in the fit. Orbits with periastron inside a(1 - e) <= 1 + rp are excluded. A ``log_prior``
        of the model stays in the weights, so it must not involve a. Fit with finite bounds on a wide enough not to cut
        the circular a/R* (``density["a_wall_mass"]``): orbits that need a value outside them are not reached.

        :param rho: (mean, sd) of the stellar density in g/cm3
        :param ecc_max: upper limit of the prior on e
        :param log_prior: optional vectorised ``f(ecc, w) -> array``, w in degrees, on top of the flat prior; for
            example the factor of transit probability, ``lambda e, w: np.log1p(e * np.sin(np.radians(w))) -
            np.log(1 - e**2)``
        :param repeats: draws of (e, omega) per draw of the fit in each pass
        :param min_ess: passes are added (at most 25) until the weights reach this ESS
        :param seed: seed of the draws of (e, omega)
        :return: a ``FitResult`` with the columns of this one, t0, a = a/R* and b = a cos(inc) of the eccentric orbit, then
            ``ecc``, ``w`` (degrees), ``b_conj`` (the transit impact parameter) and ``rho``.
            ``diagnostics["density"]`` has ``ess``, ``ess_parents`` (the ESS counted in draws of the fit, which limits
            what extra repeats can add), ``a_wall_mass``, ``passes``. The evidence of the fit does not apply: ``lnz`` is nan.
        """
        o = self._orbit
        if o is None:
            raise ValueError("reweight_density needs the result of fit() on a circular orbit, once")
        if o["secondary"]:
            raise ValueError("reweight_density needs a transit, not an eclipse")
        if o["ecc"] >= 1e-5 or "ecc" in o["free"] or "w" in o["free"]:
            raise ValueError("reweight_density needs a fit with ecc = 0 and neither ecc nor w free")
        if "a" not in o["free"]:
            raise ValueError('reweight_density needs "a" in free')
        mean, sd = float(rho[0]), float(rho[1])
        if not (mean > 0.0 and sd > 0.0 and 0.0 < ecc_max < 1.0):
            raise ValueError("reweight_density needs rho = (mean > 0, sd > 0) and 0 < ecc_max < 1")

        def col(name):
            return S[:, L.index(name)] if name in L else np.full(len(S), float(o[name]))

        rng = np.random.default_rng(seed)
        S, L = self.weighted_samples, self.labels
        per, rp, b_c, a_c = col("per"), col("rp"), np.abs(col("b")), col("a")
        Q = (1.0 + rp) ** 2 - b_c ** 2
        with np.errstate(all="ignore"):
            s_c = np.arcsin(np.sqrt(Q / (a_c ** 2 - b_c ** 2)))                # pi T14 / P of the circular orbit
            da_c = Q * np.cos(s_c) / (a_c * np.sin(s_c) ** 3)                  # |d a_c / d s_c|
            lw0 = np.log(self.weights) - np.log(da_c)
        rows, best = [], -np.inf
        ess = 0.0
        for passes in range(1, 26):
            j = np.repeat(np.arange(len(S)), repeats)
            e, om = rng.uniform(0.0, ecc_max, len(j)), rng.uniform(-0.5 * np.pi, 1.5 * np.pi, len(j))
            esw = e * np.sin(om)
            g = (1.0 + esw) / np.sqrt(1.0 - e * e)                             # T14 = g^-1 x P / pi x arcsin(...)
            bo = b_c[j] * (1.0 + esw) / (1.0 - e * e)                          # a cos(inc)
            with np.errstate(all="ignore"):
                # first the constant-speed duration, to set aside the draws far from the density
                s = s_c[j] * g
                a = np.sqrt(Q[j] / np.sin(s) ** 2 + bo * bo)
                r = 3.0 * np.pi * a ** 3 / (_G_CGS * (per[j] * 86400.0) ** 2)
                keep = (s < 0.5 * np.pi) & (a * (1.0 - e) > 1.0 + rp[j]) & (np.abs(r - mean) < 12.0 * sd)
            j, e, om, g, bo, esw = j[keep], e[keep], om[keep], g[keep], bo[keep], esw[keep]
            a, s = a[keep], s[keep]
            with np.errstate(all="ignore"):
                # a such that the duration of the Keplerian orbit is that of the fit: kappa = T14 / constant-speed T14
                # varies slowly with a, so that da / dT14 = (da / dT14 at constant speed) / kappa
                kappa = np.ones_like(a)
                T = s_c[j] * per[j] / np.pi
                for _ in range(4):
                    t1, t4 = _contact_times(per[j], rp[j], a, bo, e, om)
                    kappa = kappa * (t4 - t1) / T
                    s = s_c[j] * g / kappa
                    a = np.sqrt(Q[j] / np.sin(s) ** 2 + bo * bo)
                t1, t4 = _contact_times(per[j], rp[j], a, bo, e, om)
                r = 3.0 * np.pi * a ** 3 / (_G_CGS * (per[j] * 86400.0) ** 2)
                da = Q[j] * np.cos(s) * g / kappa / (a * np.sin(s) ** 3)       # |d a / d s_c| at this e, omega
                lw = lw0[j] - 0.5 * ((r - mean) / sd) ** 2 + np.log(3.0 * r / a * da)
                if log_prior is not None:
                    lw = lw + np.asarray(log_prior(e, np.degrees(om)), dtype=float)
                keep = np.isfinite(lw) & (s < 0.5 * np.pi) & (a * (1.0 - e) > 1.0 + rp[j])
            best = max(best, np.max(lw[keep], initial=-np.inf))
            keep &= lw > best - 25.0                                           # the rest would not change any weight
            rows.append((j[keep], e[keep], np.degrees(om[keep]), a[keep], bo[keep], b_c[j][keep], r[keep],
                         0.5 * (t1 + t4)[keep], lw[keep]))
            lw_all = np.concatenate([q[-1] for q in rows])
            if len(lw_all) == 0:
                continue
            w = np.exp(lw_all - best)
            ess = float(np.sum(w) ** 2 / np.sum(w * w))
            if ess >= min_ess:
                break
        if ess == 0.0:
            raise ValueError("reweight_density: no draw reaches this density with e < ecc_max")
        j, e, om, a, bo, bc, r, mid, lw = (np.concatenate(q) for q in zip(*rows))
        keep = lw > best - 25.0
        j, e, om, a, bo, bc, r, mid, lw = (q[keep] for q in (j, e, om, a, bo, bc, r, mid, lw))
        w = np.exp(lw - best)
        w /= w.sum()
        T = S[j].copy()
        T[:, L.index("a")] = a
        if "t0" in L:
            T[:, L.index("t0")] -= mid                                         # the middle of the transit stays
        labels = list(L)
        extra = [e, om, bc, r]
        names = ["ecc", "w", "b_conj", "rho"]
        if "b" in L:
            T[:, L.index("b")] = bo
        else:
            extra.insert(0, bo)
            names.insert(0, "b")
        T = np.column_stack([T] + extra)
        labels += names
        ess_parents = float(1.0 / np.sum(np.bincount(j, w, minlength=len(S)) ** 2))
        lo, hi = o["a_bounds"]
        wall = float(np.sum(self.weights[(a_c < 1.01 * lo) | (a_c > 0.99 * hi)]) / np.sum(self.weights))
        if ess < 0.25 * min_ess or ess_parents < 100.0:
            warnings.warn(f"reweight_density: few effective draws (ESS {ess:.0f}, {ess_parents:.0f} counted in draws of "
                          "the fit): a density far from what the fit allows, or a very narrow one; use a larger n in "
                          "fit() or raise repeats", RuntimeWarning, stacklevel=2)
        if wall > 0.01:
            warnings.warn(f"reweight_density: {wall:.1%} of the fit lies at the bounds on a; wider bounds may change "
                          "the result", RuntimeWarning, stacklevel=2)
        diag = dict(self.diagnostics, ess_total=ess, lnz=np.nan, lnz_err=np.nan,
                    lnz_note="reweight_density: the evidence of the circular fit does not apply",
                    density=dict(rho=mean, sd=sd, ecc_max=float(ecc_max), ess=ess, ess_parents=ess_parents,
                                 a_wall_mass=wall, passes=passes))
        jr = systematic_resample(w, max(len(S), min(len(w), 100_000)), rng)
        out = FitResult(T[jr], labels, weighted=(T, w), diagnostics=diag)
        out._parents = j
        return out


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
