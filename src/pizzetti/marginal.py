"""Transit fits with quadratic limb darkening and the flux baseline integrated out.

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

Priors: uniform within ``bounds`` on the free parameters (plus 0 <= b < 1 + rp,
0 <= ecc < 1), an optional vectorised ``log_prior`` callback on the physical
parameters, and Kipping's uniform prior on (q1, q2) with a flat prior on the
baseline flux f0. ``ld_prior`` adds a prior on (u1, u2): Gaussian priors
exactly (inside the likelihood), arbitrary callables by importance reweighting.

emcee and scipy are only needed for sampling (``pip install pizzetti[fit]``).
"""
import math

import numpy as np
from numba import njit

from . import _marginal as K
from . import _series
from .terms import _A_HALF

CANON = K.CANON
_DEG = math.pi / 180.0


def _exposure_offsets(exp_time, k, mode):
    if k <= 1 or exp_time <= 0:
        return np.zeros(1)
    if mode == "batman":
        return np.linspace(-exp_time / 2.0, exp_time / 2.0, k)
    if mode == "midpoint":
        return ((np.arange(k) + 0.5) / k - 0.5) * exp_time
    raise ValueError('offsets must be "batman" or "midpoint"')


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


class MarginalTransitModel:
    """Transit likelihood with quadratic LD and the baseline flux integrated out.

    :param params: ``pizzetti.TransitParams`` (initial values / fixed values;
        ``limb_dark`` must be "quadratic"; ``u`` is ignored -- LD is free)
    :param t, y: times and fluxes
    :param yerr: flux uncertainties (scalar or array)
    :param free: names of the sampled parameters, a subset of
        ("t0", "per", "rp", "a", "b", "ecc", "w"); b = a cos(inc), w in degrees
    :param bounds: dict name -> (lo, hi) of uniform priors (defaults are wide)
    :param log_prior: optional vectorised callback ``f(dict of arrays) -> array``
        on the physical parameters (all of t0, per, rp, a, b, inc, ecc, w)
    :param ld_prior: optional prior on the LD coefficients (on top of the
        physical region): a dict ``{"mean": (u1, u2), "cov": 2x2}`` for a Gaussian
        -- included exactly in the likelihood -- or a callback
        ``f(u1, u2) -> log prior`` (arrays), applied by importance reweighting of
        the posterior samples (fine for priors broader than the likelihood)
    :param supersample_factor, exp_time: as in ``TransitModel``
    :param offsets: sub-exposure placement, "batman" (default, as batman /
        TransitModel) or "midpoint" (more accurate for the same factor)
    :param reparam: "auto" samples the total duration T14 instead of a/R*
        internally (circular orbits; the prior stays uniform in a via the
        Jacobian; mixes ~2x faster); "none" disables it
    """

    def __init__(self, params, t, y, yerr, free=("t0", "per", "rp", "a", "b"), bounds=None,
                 log_prior=None, ld_prior=None, supersample_factor=1, exp_time=0.0,
                 offsets="batman", reparam="auto", series_tol=1e-9, transittype="primary"):
        self._check_supported(params, transittype)
        self.free = list(free)
        bad = [p for p in self.free if p not in CANON]
        if bad:
            raise ValueError(f"unknown free parameter(s) {bad}; allowed: {list(CANON)} "
                             "(use b = a cos(inc) instead of inc; w in degrees)")
        if len(set(self.free)) != len(self.free) or not self.free:
            raise ValueError("free must be a non-empty list of distinct names")
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
        self.fixed = np.array([params.t0, params.per, params.rp, a, a * math.cos(inc * _DEG),
                               params.ecc, params.w * _DEG], dtype=float)
        self.idx = np.array([CANON.index(p) for p in self.free], dtype=np.int64)
        circular = params.ecc < 1e-5 and "ecc" not in self.free
        self.t14 = bool(reparam == "auto" and "a" in self.free and circular)
        if reparam not in ("auto", "none"):
            raise ValueError('reparam must be "auto" or "none"')
        lo = np.full(7, -np.inf)
        hi = np.full(7, np.inf)
        lo[1], lo[2], lo[3], lo[4] = 0.0, 0.0, 1.0, -np.inf
        lo[5], hi[5] = -np.inf, 1.0
        for name, (l_, h_) in (bounds or {}).items():
            if name not in CANON:
                raise ValueError(f"bounds for unknown parameter {name!r}")
            j = CANON.index(name)
            lo[j], hi[j] = float(l_), float(h_)
            if name == "w":
                lo[j], hi[j] = lo[j] * _DEG, hi[j] * _DEG
        self.lo, self.hi = lo, hi
        self.log_prior = log_prior
        self.ld_prior = ld_prior
        # Gaussian LD prior: exact inside the likelihood (prior on beta = M u + v)
        self.has_ldp = isinstance(ld_prior, dict)
        self.mb, self.Pb = np.zeros(2), np.eye(2)
        if self.has_ldp:
            M = np.array([[-1.0, -1.0], [-0.5, 0.0]])
            mu_u = np.asarray(ld_prior["mean"], float)
            cov_u = np.asarray(ld_prior["cov"], float)
            self.mb = M @ mu_u + 1.0
            self.Pb = np.linalg.inv(M @ cov_u @ M.T)
        elif ld_prior is not None and not callable(ld_prior):
            raise ValueError('ld_prior must be a dict {"mean": (u1, u2), "cov": 2x2} or a callable')
        self.labels = list(self.free)
        self.ndim = len(self.free)
        self.n_eval = 0
        self._ja = self.free.index("a") if "a" in self.free else -1

    # -- scope checks (remove when the features are implemented) -------------
    @staticmethod
    def _check_supported(params, transittype):
        law = getattr(params, "limb_dark", "quadratic")
        if law != "quadratic":
            raise NotImplementedError(
                f'MarginalTransitModel supports limb_dark="quadratic" only (got {law!r}); '
                "the LD coefficients are free and integrated out. Square-root, nonlinear, "
                "power-2, uniform and linear laws are not implemented yet.")
        if transittype != "primary":
            raise NotImplementedError("MarginalTransitModel supports primary transits only; "
                                      "secondary eclipses are not implemented yet.")
        if float(params.rp) < 0:
            raise NotImplementedError("MarginalTransitModel needs rp > 0; inverse transits "
                                      "(rp < 0) are not implemented yet.")

    # -- coordinates -----------------------------------------------------------
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
        P[:, self.idx] = X
        if "w" in self.free:
            P[:, 6] = P[:, 6] * _DEG
        P[:, 4] = np.abs(P[:, 4])
        return P

    def _canonical_internal(self, Xi):
        Xi = np.ascontiguousarray(np.atleast_2d(Xi), dtype=float)
        if "w" in self.free:
            Xi = Xi.copy()
            Xi[:, self.free.index("w")] *= _DEG
        P, lj = K.to_canonical(Xi, self.idx, self.fixed, self.t14)
        return P, lj

    def to_physical(self, Xi):
        """Internal coordinates -> physical free-parameter rows."""
        P, _ = self._canonical_internal(Xi)
        X = P[:, self.idx].copy()
        if "w" in self.free:
            X[:, self.free.index("w")] /= _DEG
        return X

    # -- log-probability -------------------------------------------------------
    def log_prob_internal(self, Xi):
        """Vectorised log posterior (up to a constant) in internal coordinates."""
        Xi = np.ascontiguousarray(np.atleast_2d(Xi), dtype=float)
        if "w" in self.free:
            Xi = Xi.copy()
            Xi[:, self.free.index("w")] *= _DEG
        out = K.loglike_batch(Xi, self.idx, self.fixed, self.t14, self.lo, self.hi, self.tt,
                              self.k, self.wts, self.yw, self.tol, _A_HALF, _series._GTAB,
                              self.has_ldp, self.mb, self.Pb)
        self.n_eval += len(Xi)
        if self.log_prior is not None:
            ok = np.isfinite(out)
            if ok.any():
                P, _ = K.to_canonical(Xi[ok], self.idx, self.fixed, self.t14)
                out[ok] += np.asarray(self.log_prior(self._as_dict(P)), dtype=float)
        return out

    def log_prob(self, theta):
        """Log posterior in physical coordinates (theta: one row or an array of rows).
        Includes the box prior and log_prior; no T14 Jacobian (physical density)."""
        X = np.atleast_2d(np.asarray(theta, dtype=float))
        if self.t14:
            # internal density includes log|da/dT14|; remove it for the physical density
            Xi = self.to_internal(X)
            out = self.log_prob_internal(Xi)
            _, lj = self._canonical_internal(Xi)
            out = out - lj
        else:
            out = self.log_prob_internal(X)
        return out if np.ndim(theta) > 1 else float(out[0])

    @staticmethod
    def _as_dict(P):
        a, b = P[:, 3], P[:, 4]
        return dict(t0=P[:, 0], per=P[:, 1], rp=P[:, 2], a=a, b=b,
                    inc=np.degrees(np.arccos(np.clip(b / a, -1, 1))), ecc=P[:, 5],
                    w=P[:, 6] / _DEG)

    # -- LD (and baseline) recovery ---------------------------------------------
    def expand(self, X, ld="q", rng=None, resample=True):
        """Physical geometry rows -> rows with LD and baseline columns appended.

        Draws (c0, c1, c2) for every row from its conditional (truncated
        Gaussian), converts to (q1, q2) or (u1, u2) and f0, and applies the
        importance weights that turn the flat-in-c prior into Kipping's uniform
        (q1, q2) x flat f0 (times ld_prior if given). With ``resample`` the
        weighted rows are resampled to equal weight (same count).
        Returns (samples, labels, weights) -- weights are uniform if resampled.
        """
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
            if self.ld_prior is not None:                # callable: post-hoc reweighting
                logw = logw + self._ld_logprior(u1, u2)
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
            i = rng.choice(len(w), size=len(w), p=w)
            return out[i], labels, np.full(len(w), 1.0 / len(w))
        return out, labels, w

    def _ld_logprior(self, u1, u2):
        lp = self.ld_prior
        if callable(lp):
            return np.asarray(lp(u1, u2), dtype=float)
        mean = np.asarray(lp["mean"], float)
        cov = np.asarray(lp["cov"], float)
        d = np.column_stack([u1, u2]) - mean
        return -0.5 * np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)

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
    def fit(self, method="auto", n=4000, nwalkers=32, max_steps=100_000, ntau=50, seed=None,
            ld="q", progress=False):
        """Posterior samples in one call.

        method="is": profile importance sampling along b (needs "b" in free);
        "emcee": vectorised emcee until convergence (N > ntau * tau, tau stable to 1 %);
        "auto": importance sampling, emcee fallback if k-hat >= 0.7 or the profile is
        incomplete. Returns a ``FitResult``.
        """
        from ._sampling import hessian, profile_is
        rng = np.random.default_rng(seed)
        self.n_eval = 0
        x0 = self.to_internal(self.fixed_physical_free())[0]
        scale = self._default_scales(x0)
        diag = dict(method=None)
        Xw = w = None
        if method in ("is", "auto") and "b" in self.free:
            jb = self.free.index("b")
            res = profile_is(self.log_prob_internal, x0, scale, prof=jb, n=n, rng=rng, lo_wall=0.0)
            ok = bool(res["complete"] and res["khat"] < 0.7)
            diag.update(method="profile-IS", khat=res["khat"], ess=res["ess"],
                        complete=res["complete"], n_grid=res["n_grid"], is_ok=ok)
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
                sd = np.sqrt(np.clip(np.diag(np.linalg.inv(-H)), 1e-300, None))
            except np.linalg.LinAlgError:
                sd = scale
            P0 = x0 + 0.1 * sd * rng.standard_normal((nwalkers, self.ndim))
            if "b" in self.free:
                P0[:, self.free.index("b")] = np.abs(P0[:, self.free.index("b")])
            sampler = _sampler_class()(self, nwalkers, internal=True)
            old, conv = np.inf, False
            for _ in sampler.sample(P0, iterations=max_steps, progress=progress):
                if sampler.iteration % 500:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                if np.all(ntau * tau < sampler.iteration) and np.all(np.abs(old - tau) / tau < 0.01):
                    conv = True
                    break
                old = tau
            tau = sampler.get_autocorr_time(tol=0)
            ch = sampler.get_chain(discard=int(5 * np.max(tau)), flat=True, expand=False)
            ch = ch[:: max(1, len(ch) // n)]
            Xw, w = ch, np.full(len(ch), 1.0 / len(ch))
            diag.update(method="emcee" if method == "emcee" else "emcee (fallback)",
                        converged=conv, steps=int(sampler.iteration), tau=tau.tolist())
        diag["n_eval"] = int(self.n_eval)
        # LD + baseline for the weighted geometry samples; combine weights, resample
        m = min(len(w), max(n, 2000))
        idx = rng.choice(len(w), size=m, p=w)
        S_, labels, ws = self.expand(Xw[idx], ld=ld, rng=rng, resample=False)
        diag["ld_reweight_ess"] = float(1.0 / np.sum(ws ** 2))
        j = rng.choice(m, size=m, p=ws)
        return FitResult(S_[j], labels, weighted=(S_, ws), diagnostics=diag)

    def fixed_physical_free(self):
        X = self.fixed[self.idx].copy()[None, :]
        if "w" in self.free:
            X[0, self.free.index("w")] /= _DEG
        return X

    def _default_scales(self, x0):
        P = self.fixed
        T14 = P[1] / math.pi * math.asin(min(1.0, (1 + P[2]) / max(P[3], 1.0 + P[2] + 1e-9)))
        sc = {"t0": 1e-2 * T14, "per": 1e-6 * P[1], "rp": 1e-2 * P[2],
              "a": (1e-2 * T14 if self.t14 else 1e-2 * P[3]), "b": 0.05, "ecc": 0.01, "w": 1.0}
        return np.array([sc[p] for p in self.free])


class FitResult:
    """Result of ``MarginalTransitModel.fit``: equal-weight ``samples`` (emcee-like
    flat array) with ``labels``; ``weighted`` = (samples, weights) before resampling."""

    def __init__(self, samples, labels, weighted, diagnostics):
        self.samples = samples
        self.labels = labels
        self.weighted_samples, self.weights = weighted
        self.diagnostics = diagnostics

    def summary(self):
        q = np.percentile(self.samples, [16, 50, 84], axis=0)
        lines = [f"{'':5s} {'median':>14s} {'-1sigma':>11s} {'+1sigma':>11s}"]
        for j, name in enumerate(self.labels):
            lines.append(f"{name:5s} {q[1, j]:14.7g} {q[1, j] - q[0, j]:11.3g} {q[2, j] - q[1, j]:11.3g}")
        return "\n".join(lines)

    def corner(self, **kwargs):
        import corner
        return corner.corner(self.samples, labels=self.labels, **kwargs)


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
            LD + f0, shape (..., ndim + 3). With flat=True the LD prior weights are
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
