"""Profile importance sampling (and plain Laplace IS) with Pareto-smoothed weights, used by
pizzetti.MarginalTransitModel.fit().

After LD (and optionally nuisance trends) are integrated out, the posterior is
over 4-5 geometry parameters only and, in the T14 parametrisation, close to
Gaussian. Instead of MCMC:

  1. MAP by optimisation (Nelder-Mead then BFGS, in scaled coordinates);
  2. Hessian by central finite differences at the posterior scale -> Sigma = H^-1;
  3. heavy-tailed proposal q = multivariate t (nu dof, scale Sigma * s^2);
  4. importance weights w = p / q with the exact (marginalised) log-posterior,
     evaluated in one vectorised numba call per batch;
  5. Pareto-smoothed importance sampling (PSIS; Vehtari et al. 2024) for
     stable weights and the k-hat diagnostic; optionally one or two rounds of
     adaptation (proposal re-fitted to the weighted samples) if k-hat is high.

Validity: importance sampling is exact in the limit (any proposal whose support
covers the posterior); finite-sample reliability is judged by k-hat (< 0.7 is
reliable, < 0.5 good) and the weight ESS.

    res = laplace_is(logp_batch, theta0, scale, n=4000, rng=rng)
    res["theta"], res["w"]  -> weighted posterior samples (w sums to 1)

profile_is (the recommended method) builds the proposal along one parameter
(the impact parameter) from conditional optima, so that skewed posteriors and
the b = 0 wall are covered by construction.
"""
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


# ----------------------------------------------------------------------------
# PSIS (Pareto-smoothed importance sampling)
# ----------------------------------------------------------------------------
def _gpdfit(x):
    """Generalised Pareto fit (Zhang & Stephens 2009, with the weakly
    informative prior on k used by loo/arviz). x > 0, sorted ascending."""
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior_bs * x[int(n / 4 + 0.5) - 1]
    bs += 1.0 / x[-1]
    ks = np.log1p(-bs[:, None] * x).mean(axis=1)
    L = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.exp(L - L[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, bs = w[keep] / w[keep].sum(), bs[keep]
    b = np.sum(bs * w)
    k = np.log1p(-b * x).mean()
    sigma = -k / b
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    return k, sigma


def psis(logw):
    """Pareto-smooth log importance weights. Returns (normalised log weights, khat)."""
    logw = np.asarray(logw, float).copy()
    ok = np.isfinite(logw)
    lw = logw[ok]
    n = lw.size
    if n == 0:                                   # no finite weight at all
        return np.full(logw.shape, -np.inf), np.inf
    lw -= lw.max()
    M = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n))))
    khat = np.nan
    if M >= 5:
        order = np.argsort(lw)
        cutoff = lw[order[n - M - 1]]
        tail = order[n - M:]
        x = np.exp(lw[tail]) - np.exp(cutoff)
        if np.all(x > 0):
            k, sigma = _gpdfit(np.sort(x))
            khat = k
            p = (np.arange(1, M + 1) - 0.5) / M
            q = sigma * ((1 - p) ** (-k) - 1) / k if abs(k) > 1e-12 else -sigma * np.log1p(-p)
            sm = np.log(q + np.exp(cutoff))
            lw[tail] = np.minimum(sm, 0.0)               # truncate at the largest raw weight
    lw -= np.log(np.sum(np.exp(lw)))
    out = np.full(logw.shape, -np.inf)
    out[ok] = lw
    return out, khat


# ----------------------------------------------------------------------------
# Laplace approximation
# ----------------------------------------------------------------------------
def find_map(logp_batch, theta0, scale, n_restarts=1, nelder_mead=True):
    theta0 = np.asarray(theta0, float)
    scale = np.asarray(scale, float)

    def nlp(x):
        v = logp_batch((theta0 + x * scale)[None, :])[0]
        return -v if np.isfinite(v) else 1e300

    nev = 0
    best = None
    x = np.zeros_like(theta0)
    for _ in range(n_restarts):
        if nelder_mead:
            r = minimize(nlp, x, method="Nelder-Mead",
                         options=dict(xatol=1e-6, fatol=1e-9, maxiter=20000, maxfev=20000))
            nev += r.nfev
        else:
            r = minimize(nlp, x, method="Powell", options=dict(xtol=1e-3, ftol=1e-6))
            nev += r.nfev
        r2 = minimize(nlp, r.x, method="BFGS", options=dict(gtol=1e-6))
        nev += r2.nfev
        r = r2 if r2.fun <= r.fun else r
        if best is None or r.fun < best.fun:
            best = r
        x = best.x
    return theta0 + best.x * scale, -best.fun, nev


def hessian(logp_batch, theta, h):
    """Central finite-difference Hessian of log p at theta (steps h per dim);
    all stencil points in one batch call."""
    d = len(theta)
    pts = [theta.copy()]
    idx = {}
    for i in range(d):
        for j in range(i, d):
            for si in (1, -1):
                for sj in (1, -1):
                    p = theta.copy()
                    p[i] += si * h[i]
                    p[j] += sj * h[j]
                    idx[(i, j, si, sj)] = len(pts)
                    pts.append(p)
    v = logp_batch(np.array(pts))
    H = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            f = lambda si, sj: v[idx[(i, j, si, sj)]]  # noqa: E731
            H[i, j] = H[j, i] = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * h[i] * h[j])
    return H, len(pts)


def stuck_walkers(lp, min_gap=25.0):
    """Indices of ensemble walkers stuck far below the others: lp (steps, walkers) is the log
    posterior over the last stretch of the chain; a walker whose mean is more than
    max(min_gap, 10 MAD) below the median walker's sits at a local mode or on a plateau the
    stretch move does not leave (its posterior mass is negligible: e^-25 at the peak)."""
    m = np.mean(lp, axis=0)
    ok = np.isfinite(m)
    if ok.sum() < 2:
        return np.zeros(0, dtype=int)
    med = np.median(m[ok])
    mad = np.median(np.abs(m[ok] - med))
    return np.where(~ok | (m < med - max(min_gap, 10.0 * mad)))[0]


def reset_walkers(coords, bad, rng, logp_batch=None):
    """coords with the walkers ``bad`` moved next to randomly chosen other walkers (jitter of
    1 % of the ensemble spread; an exact copy where the jittered point is outside the prior)."""
    X = np.array(coords, dtype=float)
    good = np.setdiff1d(np.arange(len(X)), bad)
    src = X[rng.choice(good, len(bad))]
    new = src + 0.01 * X[good].std(axis=0) * rng.standard_normal(src.shape)
    if logp_batch is not None:
        out = ~np.isfinite(logp_batch(new))
        new[out] = src[out]
    X[bad] = new
    return X


def _mvt_logpdf(x, mu, L, nu):
    d = mu.size
    z = np.linalg.solve(L, (x - mu).T).T
    q = np.sum(z * z, axis=1)
    logdet = np.sum(np.log(np.diag(L)))
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi) - logdet
            - 0.5 * (nu + d) * np.log1p(q / nu))


def _mvt_sample(rng, mu, L, nu, n):
    d = mu.size
    z = rng.standard_normal((n, d))
    g = rng.chisquare(nu, n) / nu
    return mu + (z @ L.T) / np.sqrt(g)[:, None]



def _weighted_gmm(X, w, K, rng, iters=60):
    """Gaussian mixture fitted to weighted points (EM; w sums to 1). Returns (pi, mu, cov)."""
    from scipy.special import logsumexp
    n, d = X.shape
    C0 = np.cov(X.T, aweights=w).reshape(d, d)
    reg = 1e-6 * np.diag(np.diag(C0))
    mu = X[rng.choice(n, size=K, replace=False, p=w)].copy()
    cov = np.repeat((C0 + reg)[None], K, axis=0)
    pi = np.full(K, 1.0 / K)
    for _ in range(iters):
        lr = np.empty((n, K))
        for k in range(K):
            L = np.linalg.cholesky(cov[k])
            z = np.linalg.solve(L, (X - mu[k]).T).T
            lr[:, k] = np.log(pi[k]) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(z * z, axis=1)
        lr -= logsumexp(lr, axis=1)[:, None]
        R = np.exp(lr) * w[:, None]
        Nk = R.sum(0) + 1e-300
        pi = Nk / Nk.sum()
        for k in range(K):
            mu[k] = R[:, k] @ X / Nk[k]
            D = X - mu[k]
            cov[k] = (R[:, k][:, None] * D).T @ D / Nk[k] + reg
    keep = pi > 1e-3
    return pi[keep] / pi[keep].sum(), mu[keep], cov[keep]


def pmc_refine(logp_batch, X, logw, rng, n=4000, iters=6, K=6, nu=5.0, inflate=1.2, defensive=0.15,
               batch=2000, extra=(), min_iters=None, khat_ok=0.7, ess_min=0.1, dq_tol=0.1):
    """Adaptive multiple importance sampling (AMIS) from weighted draws (X, logw).

    Each iteration fits a Gaussian mixture to all weighted draws so far and draws n new points
    from its Student-t version (covariances x inflate^2) plus a broad defensive t (weight
    ``defensive``, 2 x the overall sd). All draws of all iterations are kept and weighted against
    the mixture of all proposals used (deterministic-mixture weights), which is unbiased and much
    more stable than the weights of a single iteration. For posteriors that a Laplace-based
    proposal covers badly (skewed or curved conditionals), where k-hat need not see the missing
    mass. The start draws (X, logw) seed the first fit only. Returns dict(theta, w, logw, khat,
    ess, n_eval, history, converged).

    ``extra``: fixed components (weight, mean, covariance) added to every iteration's proposal
    (e.g. a component over the whole prior box: then every region of the support keeps a proposal
    density, and the estimate is consistent whatever the start draws missed). ``min_iters``: run
    adaptively, at least min_iters and at most ``iters`` iterations, until k-hat < ``khat_ok``, the
    ESS of the latest n draws' share is >= ``ess_min`` n and the 2.5/16/50/84/97.5 % quantiles of
    every coordinate moved by less than ``dq_tol`` sd since the previous iteration (``converged``);
    None: exactly ``iters`` iterations."""
    from scipy.special import logsumexp
    logw = np.asarray(logw, float)
    fail = dict(theta=np.atleast_2d(X), w=np.zeros(len(logw)), logw=np.full(len(logw), -np.inf),
                khat=np.inf, ess=0.0, n_eval=0, history=[], converged=False)
    if np.sum(np.isfinite(logw)) < 10 * (np.atleast_2d(X).shape[1] + 1):
        return fail                              # too few weighted draws to fit a proposal
    w = np.exp(logw - np.max(logw[np.isfinite(logw)]))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    Xs, lps, props, counts = None, None, [], []
    nev, hist = 0, []
    extra = [(float(p), np.asarray(m, float), np.linalg.cholesky(np.asarray(c, float))) for p, m, c in extra]
    f_extra = sum(p for p, _, _ in extra)
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

    def wquant(Y, ww):
        out = np.empty((Y.shape[1], len(qs)))
        for j in range(Y.shape[1]):
            o = np.argsort(Y[:, j])
            cw = np.cumsum(ww[o])
            out[j] = np.interp(qs, (cw - 0.5 * ww[o]) / cw[-1], Y[o, j])
        return out
    q_prev, converged = None, False

    def lq_one(comps, Y):
        return logsumexp(np.column_stack([np.log(p) + _mvt_logpdf(Y, m, L, nu) for p, m, L in comps]), axis=1)

    LQ = []                                  # LQ[t]: log density of proposal t at all draws so far

    def lq_all(Xn):
        """Append the new draws Xn (of the newest proposal) and return the log density of the
        deterministic mixture of all proposals so far (weights n_t / N) at all draws. Each
        proposal's density at each draw is computed once."""
        for t in range(len(LQ)):
            LQ[t] = np.concatenate([LQ[t], lq_one(props[t], Xn)])
        LQ.append(lq_one(props[-1], Xn if Xs is None else np.vstack([Xs, Xn])))
        N = float(sum(counts))
        return logsumexp(np.column_stack([np.log(c / N) + lq for lq, c in zip(LQ, counts)]), axis=1)

    for it in range(iters):
        ess = 1.0 / np.sum(w ** 2)
        pi, mu, cov = _weighted_gmm(X, w, min(K, max(1, int(ess) // 50)), rng)
        m0 = w @ X
        C0 = np.cov(X.T, aweights=w).reshape(X.shape[1], X.shape[1])
        comps = [(p * (1 - defensive) * (1 - f_extra), m, np.linalg.cholesky(c * inflate ** 2))
                 for p, m, c in zip(pi, mu, cov)]
        comps.append((defensive * (1 - f_extra), m0, np.linalg.cholesky(C0 * 4.0)))
        comps += extra
        k = rng.multinomial(n, [c[0] for c in comps])
        Xn = np.vstack([_mvt_sample(rng, m, L, nu, c) for (p, m, L), c in zip(comps, k) if c > 0])
        lpn = np.concatenate([logp_batch(Xn[i:i + batch]) for i in range(0, len(Xn), batch)])
        nev += len(Xn)
        props.append(comps)
        counts.append(len(Xn))
        lqa = lq_all(Xn)
        Xs = Xn if Xs is None else np.vstack([Xs, Xn])
        lps = lpn if lps is None else np.concatenate([lps, lpn])
        lw, khat = psis(lps - lqa)
        if not np.isfinite(khat) or not np.any(np.isfinite(lw)):
            fail.update(n_eval=int(nev), history=hist)
            return fail                          # no usable weights (e.g. all draws outside the prior)
        w = np.exp(lw)
        X = Xs
        ess = float(1.0 / np.sum(w ** 2))
        hist.append(dict(khat=float(khat), ess=ess, n=int(len(Xs))))
        if min_iters is not None:
            q = wquant(Xs, w)
            sd = np.sqrt(np.clip(np.diag(np.atleast_2d(np.cov(Xs.T, aweights=w))), 1e-300, None))
            dq = np.inf if q_prev is None else float(np.max(np.abs(q - q_prev) / sd[:, None]))
            q_prev = q
            hist[-1]["dq"] = dq
            # the pooled draws' ESS against n (one iteration's draws), as for a single IS run
            if (it + 1 >= min_iters and khat < khat_ok and ess >= ess_min * n and dq < dq_tol):
                converged = True
                break
    return dict(theta=Xs, w=w, logw=lw, khat=float(khat), ess=float(1.0 / np.sum(w ** 2)), n_eval=int(nev),
                history=hist, converged=converged)

def laplace_is(logp_batch, theta0, scale, n=4000, nu=5.0, inflate=1.2, rng=None,
               adapt=2, khat_target=0.5, batch=2000):
    """Laplace + (adaptive) importance sampling. Returns a dict with weighted
    samples and diagnostics. ``scale``: rough posterior scales (for the
    optimiser and the first Hessian step)."""
    rng = np.random.default_rng() if rng is None else rng
    theta_map, lp_map, nev = find_map(logp_batch, theta0, scale)
    # Hessian: first with small steps to get the scales, then at the posterior scale
    H, k1 = hessian(logp_batch, theta_map, 1e-3 * np.asarray(scale, float))
    cov = np.linalg.inv(-H)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    H, k2 = hessian(logp_batch, theta_map, 0.5 * sd)
    nev += k1 + k2
    cov = np.linalg.inv(-H)
    cov = 0.5 * (cov + cov.T)
    w_ev, V = np.linalg.eigh(cov)
    cov = (V * np.clip(w_ev, 1e-30, None)) @ V.T                      # ensure PD
    mu = theta_map.copy()
    hist = []
    for it in range(adapt + 1):
        L = np.linalg.cholesky(cov * inflate ** 2)
        X = _mvt_sample(rng, mu, L, nu, n)
        lp = np.concatenate([logp_batch(X[i:i + batch]) for i in range(0, n, batch)])
        nev += n
        lq = _mvt_logpdf(X, mu, L, nu)
        lw_raw = lp - lq
        lw, khat = psis(lw_raw)
        w = np.exp(lw)
        ess = 1.0 / np.sum(w ** 2)
        hist.append(dict(iter=it, khat=float(khat), ess=float(ess), n=n,
                         frac_outside=float(np.mean(~np.isfinite(lp)))))
        if khat < khat_target or it == adapt:
            break
        # adapt: weighted mean/covariance of the current samples
        mu = np.sum(w[:, None] * X, axis=0)
        Xc = X - mu
        cov = (w[:, None] * Xc).T @ Xc
        cov = 0.5 * (cov + cov.T)
    return dict(theta=X, w=w, logw=lw, khat=float(khat), ess=float(ess), n_eval=int(nev),
                theta_map=theta_map, lp_map=float(lp_map), cov_laplace=np.linalg.inv(-H),
                history=hist)


# ----------------------------------------------------------------------------
# Profile-conditional proposal (robust to a skewed / walled parameter, e.g. b)
# ----------------------------------------------------------------------------
def profile_is(logp_batch, theta0, scale, prof=4, n=4000, nu=5.0, inflate=1.2, rng=None,
               drop=12.0, step_sd=1.0, lo_wall=0.0, hi_wall=None, max_grid=60, batch=2000,
               nelder_mead=False, b_temper=1.0, b_uniform=0.0, robust_cond=True, h_sd=0.2,
               mix_stage=True, khat_ok=0.5, h_ref="prev", refine_tol=0.25, max_refine=10,
               lazy_refine=True, khat_refine=0.7, logp_target=None, carry_cov=None, max_carry=3,
               defensive=0.1):
    """Importance sampling with a proposal built along one parameter (index ``prof``):

    * profile: for b on an adaptive grid around the MAP (step ``step_sd`` x the
      local Laplace sd, extended until the profile drops by ``drop`` nats or a
      wall is reached), maximise over the other parameters -> mu(b), Sigma(b)
      (Hessian), and the Laplace-marginal log density l(b) = lp_max(b) + 1/2 log det Sigma(b);
    * proposal: b ~ piecewise-linear density exp(l) on the grid; the others ~
      multivariate t(nu) with mean/covariance linearly interpolated in b
      (covariance x inflate^2);
    * PSIS-smoothed weights with the exact log posterior.
    Defensive options for the b proposal: ``b_temper`` < 1 uses exp(b_temper * l)
    (wider tails), ``b_uniform`` > 0 mixes in a uniform density over the grid range.
    ``refine_tol``: after the grid is built, intervals with probability mass are
    refined where the conditional mean at the midpoint differs from its linear
    interpolation by more than ``refine_tol`` conditional sd (Mahalanobis); at most
    ``max_refine`` extra conditional fits (None: no refinement). Near a wall the
    conditional optimum moves non-linearly in b (it depends on b^2) and the
    geometric grid steps are largest there (Kepler-2 LC: 0.55 sd interpolation error).
    ``lazy_refine`` (default): refine only if the final k-hat (after stage 2) is still
    >= ``khat_refine`` (0.7, where the estimate is unreliable), then sample again from the
    refined grid. The refinement costs up to ``max_refine`` conditional fits; a poor stage-1
    proposal is mostly due to non-Gaussian conditionals, which stage 2 handles and refinement
    does not. False refines always, before the first draws.
    ``logp_target``: the density the importance weights refer to, if not ``logp_batch``: the
    proposal (MAP, profile, Hessians) is built on ``logp_batch`` and the draws are weighted against
    ``logp_target``. With logp_target = logp_batch + log g, g <= 1 (e.g. a smooth posterior times
    a cut-off), the weights stay bounded.
    ``carry_cov`` (experimental, default False; environment PIZZETTI_CARRY_COV=1): a conditional
    optimum whose Hessian is not negative definite (a flat or slightly non-concave conditional, as
    near the grazing edge) takes the neighbouring grid point's covariance instead of failing (at the
    MAP: |eigenvalues| of the Hessian), for at most ``max_carry`` consecutive grid points; otherwise
    such steps fail, the step is halved and the grid can end early (incomplete profile). In 36
    grazing injections that fell back to emcee it made 14 usable instead of 10, but 4 of the 10 failed
    instead: not adopted.
    ``h_ref``: finite-difference steps of the conditional Hessians: "prev" (default) the
    neighbouring grid point's sd; "map" ``h_sd`` x the conditional sd at the MAP for
    every grid point (smoother log det Sigma(b) for strongly curved conditionals, but
    tested worse: the grid can stop early, Kepler-1 LC).
    """
    rng = np.random.default_rng() if rng is None else rng
    if carry_cov is None:                    # default off; PIZZETTI_CARRY_COV=1 for experiments
        carry_cov = os.environ.get("PIZZETTI_CARRY_COV", "0") == "1"
    scale = np.asarray(scale, float)
    d = len(theta0)
    rest = [i for i in range(d) if i != prof]
    theta_map, lp_map, nev = find_map(logp_batch, theta0, scale, nelder_mead=nelder_mead)
    nev_map = nev

    sd_ref = [None]                       # conditional sd at the MAP (h_ref="map")

    def cond_fit(bv, x0, cov_prev=None):
        """maximise over the other parameters at fixed b = bv (warm start x0).
        cov_prev (neighbouring grid point): seeds BFGS' inverse Hessian and the
        Hessian step sizes, so only one Hessian is needed (R7b)."""
        sc = scale[rest]
        if cov_prev is not None:
            sc = np.sqrt(np.clip(np.diag(cov_prev), 1e-300, None))      # posterior-scaled coords

        def nlp(x):
            th = np.empty(d)
            th[rest] = x0 + x * sc
            th[prof] = bv
            v = logp_batch(th[None, :])[0]
            return -v if np.isfinite(v) else 1e300

        opts = dict(gtol=1e-5)
        if cov_prev is not None:
            opts["hess_inv0"] = cov_prev / np.outer(sc, sc)
        x_start = np.zeros(len(rest))
        nf_extra = 0
        if robust_cond:                      # derivative-free pre-step (rough/awkward conditionals)
            rp_ = minimize(nlp, x_start, method="Powell", options=dict(xtol=1e-3, ftol=1e-8))
            x_start, nf_extra = rp_.x, rp_.nfev
        try:
            r = minimize(nlp, x_start, method="BFGS", options=opts)
        except ValueError:                   # hess_inv0 rejected (nearly singular seed): no seed
            opts.pop("hess_inv0", None)
            r = minimize(nlp, x_start, method="BFGS", options=opts)
        r.nfev += nf_extra
        if not r.fun < 1e299:                # no finite target at this b (outside the support)
            return None
        mu = x0 + r.x * sc
        th = np.empty(d)
        th[rest] = mu
        th[prof] = bv

        def lp_rest(X):
            Y = np.repeat(th[None, :], len(X), axis=0)
            Y[:, rest] = X
            return logp_batch(Y)
        k = k2 = 0
        H = None
        try:
            if cov_prev is None:
                H, k = hessian(lp_rest, mu, 1e-3 * sc)
                if not (np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(-0.5 * (H + H.T)) > 0)):
                    # steps of 1e-3 scale below the numerical noise of log p (a long, precise light
                    # curve: ~5e-3 nats for Kepler-10, 79 000 points): once more with larger steps
                    H, k1 = hessian(lp_rest, mu, 0.1 * sc)
                    k += k1
                cov = np.linalg.inv(-H)
                sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
            elif h_ref == "map" and sd_ref[0] is not None:
                sd = sd_ref[0]
            else:
                sd = sc
            H, k2 = hessian(lp_rest, mu, h_sd * sd)     # same relative step everywhere
            cov = np.linalg.inv(-H)
            cov = 0.5 * (cov + cov.T)
            ev, V = np.linalg.eigh(cov)
            if not (np.all(np.isfinite(cov)) and ev.min() > 0):     # (NaN: non-finite Hessian)
                raise np.linalg.LinAlgError("conditional Hessian not negative definite")
            ldet = np.sum(np.log(ev))
        except np.linalg.LinAlgError:
            # a finite conditional optimum whose Hessian is not negative definite (a flat or
            # slightly non-concave conditional, e.g. grazing geometries): with carry_cov, take the
            # neighbouring grid point's covariance (or at the MAP, |eigenvalues| of the Hessian);
            # the importance weights correct the approximate proposal, k-hat flags a poor one
            if not carry_cov:
                return None
            if cov_prev is not None:
                cp = 0.5 * (cov_prev + cov_prev.T)
                ev = np.linalg.eigvalsh(cp)
                if not (np.all(np.isfinite(cp)) and ev.min() > 0):
                    return None
                return mu, -r.fun, cp, float(np.sum(np.log(ev))), r.nfev + k + k2, True
            if H is None or not np.all(np.isfinite(H)):
                return None
            lam, V = np.linalg.eigh(-0.5 * (H + H.T))
            lam = np.abs(lam)
            if not lam.min() > 0:
                return None
            cov = (V / lam) @ V.T
            return mu, -r.fun, cov, float(-np.sum(np.log(lam))), r.nfev + k + k2, True
        return mu, -r.fun, cov, ldet, r.nfev + k + k2, False

    # local sd of b at the MAP (1-d second difference)
    def lp1(bv):
        th = theta_map.copy()
        th[prof] = bv
        return logp_batch(th[None, :])[0]
    hb = 1e-3 * scale[prof]
    c2 = (lp1(theta_map[prof] + hb) - 2 * lp_map + lp1(theta_map[prof] - hb)) / hb ** 2
    nev += 2
    # at a wall (e.g. b = 0, where the likelihood depends on b^2) the profile is
    # flat and the curvature useless: fall back to the user scale
    sd_b = 1.0 / np.sqrt(-c2) if (np.isfinite(c2) and c2 < 0) else scale[prof]
    sd_b = min(sd_b, 3.0 * scale[prof])

    grid = {}
    f0 = cond_fit(theta_map[prof], theta_map[rest])
    if f0 is None:                       # conditional Hessian not positive definite at the MAP
        return dict(theta=np.atleast_2d(theta_map), w=np.ones(1), logw=np.zeros(1),
                    khat=np.inf, ess=0.0, n_eval=int(nev), theta_map=theta_map,
                    grid_b=np.array([theta_map[prof]]), grid_logdens=np.zeros(1), stage=0,
                    n_grid=0, complete=False, n_eval_map=int(nev_map), n_refine=0,
                    grid_lpmax=np.zeros(1), grid_logdet=np.zeros(1))
    nev += f0[4]
    sd_ref[0] = np.sqrt(np.clip(np.diag(f0[2]), 1e-300, None))
    grid[theta_map[prof]] = f0
    lref = f0[1] + 0.5 * f0[3]
    for direction in (-1, 1):
        bv, x0 = theta_map[prof], f0[0]
        cov_prev, b_prev, x_prev = f0[2], None, None
        step = step_sd * sd_b
        tries = 0
        carried = 0                          # consecutive grid points with a borrowed covariance
        for _ in range(max_grid // 2):
            if lo_wall is not None and direction < 0 and bv <= lo_wall:
                break
            if hi_wall is not None and direction > 0 and bv >= hi_wall:
                break
            bn = bv + direction * step
            hit_wall = False
            if lo_wall is not None and bn <= lo_wall:
                bn, hit_wall = lo_wall, True
            if hi_wall is not None and bn >= hi_wall:
                bn, hit_wall = hi_wall, True
            # linear predictor of the conditional optimum from the last two grid points
            xg = x0 if b_prev is None else x0 + (x0 - x_prev) * (bn - bv) / (bv - b_prev)
            f = cond_fit(bn, xg, cov_prev)
            if f is None and b_prev is not None:
                f = cond_fit(bn, x0, cov_prev)
            if f is not None and f[5]:
                carried += 1
                if carried > max_carry:      # a long stretch without a usable curvature: stop
                    f = None
            elif f is not None:
                carried = 0
            if f is None:                    # e.g. outside the prior: halve the step, retry
                step *= 0.5
                tries += 1
                if tries > 6:
                    break
                continue
            step *= 1.3                      # geometric growth: reach long tails / walls
            nev += f[4]
            grid[bn] = f
            b_prev, x_prev = bv, x0
            bv, x0, cov_prev = bn, f[0], f[2]
            if hit_wall or (f[1] + 0.5 * f[3] - lref) < -drop:
                break
    if len(grid) < 2:                    # no step away from the MAP succeeded: no proposal in b
        return dict(theta=np.atleast_2d(theta_map), w=np.ones(1), logw=np.zeros(1),
                    khat=np.inf, ess=0.0, n_eval=int(nev), theta_map=theta_map,
                    grid_b=np.array([theta_map[prof]]), grid_logdens=np.zeros(1), stage=0,
                    n_grid=1, complete=False, n_eval_map=int(nev_map), n_refine=0,
                    grid_lpmax=np.array([f0[1]]), grid_logdet=np.array([f0[3]]))

    def refine():
        """Insert midpoints where the linear interpolation of mu(b) between grid
        points is off by more than refine_tol conditional sd; returns the number of
        extra conditional fits (at most max_refine)."""
        nonlocal nev
        todo = list(zip(sorted(grid)[:-1], sorted(grid)[1:]))
        n_ref = 0
        while todo and n_ref < max_refine:
            b0, b1 = todo.pop(0)
            g0, g1 = grid[b0], grid[b1]
            if max(g0[1] + 0.5 * g0[3], g1[1] + 0.5 * g1[3]) - lref < -drop:
                continue                     # no probability mass in this interval
            bm = 0.5 * (b0 + b1)
            pred = 0.5 * (g0[0] + g1[0])
            f = cond_fit(bm, pred, 0.5 * (g0[2] + g1[2]))
            n_ref += 1
            if f is None:
                continue
            nev += f[4]
            dlt = f[0] - pred
            try:
                err = float(np.sqrt(dlt @ np.linalg.solve(f[2], dlt)))
            except np.linalg.LinAlgError:
                continue
            grid[bm] = f
            if err > refine_tol:
                todo += [(b0, bm), (bm, b1)]
        return n_ref

    def build():
        """Proposal from the current grid; stage-1 draws and PSIS weights."""
        nonlocal nev
        bs = np.array(sorted(grid))
        lb_all = {b: grid[b][1] + 0.5 * grid[b][3] for b in bs}
        end_lo_ok = (lo_wall is not None and bs[0] <= lo_wall) or (lb_all[bs[0]] - lref < -drop)
        end_hi_ok = (hi_wall is not None and bs[-1] >= hi_wall) or (lb_all[bs[-1]] - lref < -drop)
        mus = np.array([grid[b][0] for b in bs])
        covs = np.array([grid[b][2] for b in bs])
        lb = np.array([grid[b][1] + 0.5 * grid[b][3] for b in bs])
        # piecewise-linear proposal density in b (optionally tempered / mixed with uniform)
        pb = np.exp(b_temper * (lb - lb.max()))
        seg0 = 0.5 * (pb[1:] + pb[:-1]) * np.diff(bs)
        pb = (1 - b_uniform) * pb / seg0.sum() + b_uniform / (bs[-1] - bs[0])
        seg = 0.5 * (pb[1:] + pb[:-1]) * np.diff(bs)
        Z = seg.sum()

        def q_b_logpdf(bv):
            out = np.full(bv.shape, -np.inf)
            ok = (bv >= bs[0]) & (bv <= bs[-1])
            out[ok] = np.log(np.interp(bv[ok], bs, pb) / Z)
            return out

        def q_b_sample(m):
            j = rng.choice(len(seg), size=m, p=seg / Z)
            u = rng.random(m)
            y0, y1, x0_, x1 = pb[j], pb[j + 1], bs[j], bs[j + 1]
            # inverse CDF of a linear density on [x0, x1]
            dy = y1 - y0
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(np.abs(dy) < 1e-12 * np.maximum(y0, 1e-300), u,
                             (-y0 + np.sqrt(y0 * y0 + u * (y1 * y1 - y0 * y0))) / dy)
            return x0_ + t * (x1 - x0_)

        def interp_params(bv):
            k = np.clip(np.searchsorted(bs, bv) - 1, 0, len(bs) - 2)
            f = ((bv - bs[k]) / (bs[k + 1] - bs[k]))[:, None]
            mu = mus[k] * (1 - f) + mus[k + 1] * f
            cov = covs[k] * (1 - f)[:, :, None] + covs[k + 1] * f[:, :, None]
            return mu, cov

        dr = len(rest)

        def q_prof_sample(m):
            bsamp = q_b_sample(m)
            mu, cov = interp_params(bsamp)
            Ls = np.linalg.cholesky(cov * inflate ** 2)
            z = rng.standard_normal((m, dr))
            g = rng.chisquare(nu, m) / nu
            X = np.empty((m, d))
            X[:, rest] = mu + np.einsum("nij,nj->ni", Ls, z) / np.sqrt(g)[:, None]
            X[:, prof] = bsamp
            return X

        def q_prof_logpdf(X):
            bv = X[:, prof]
            out = q_b_logpdf(bv)
            ok = np.isfinite(out)
            if ok.any():
                mu, cov = interp_params(bv[ok])
                Ls = np.linalg.cholesky(cov * inflate ** 2)
                xr = X[ok][:, rest]
                zz = np.linalg.solve(Ls, (xr - mu)[:, :, None])[:, :, 0]
                qq = np.sum(zz * zz, axis=1)
                logdet = np.sum(np.log(np.diagonal(Ls, axis1=1, axis2=2)), axis=1)
                out[ok] += (gammaln((nu + dr) / 2) - gammaln(nu / 2) - 0.5 * dr * np.log(nu * np.pi)
                            - logdet - 0.5 * (nu + dr) * np.log1p(qq / nu))
            return out

        def evaluate(X):
            f = logp_batch if logp_target is None else logp_target
            return np.concatenate([f(X[i:i + batch]) for i in range(0, len(X), batch)])

        X = q_prof_sample(n)
        lp = evaluate(X)
        nev += n
        lq = q_prof_logpdf(X)
        lw, khat = psis(lp - lq)
        return (bs, lb, end_lo_ok, end_hi_ok, q_prof_sample, q_prof_logpdf, evaluate, X, lp, lq, lw, khat)

    def stages(out):
        """Stage 1 (draws from the profile proposal) and, if k-hat >= khat_ok, stage 2."""
        nonlocal nev
        bs, lb, end_lo_ok, end_hi_ok, q_prof_sample, q_prof_logpdf, evaluate, X, lp, lq, lw, khat = out
        w = np.exp(lw)
        stage = 1
        _q_final = q_prof_logpdf                 # the density the final draws were drawn from
        if mix_stage and not (khat < khat_ok):
            # stage 2: defensive mixture 0.5 profile proposal + 0.5 multivariate t fitted to the
            # stage-1 weighted samples (covers near-Gaussian posteriors where the
            # Laplace-marginal profile l(b) is distorted by non-Gaussian conditionals)
            m1 = np.sum(w[:, None] * X, axis=0)
            C1 = (w[:, None] * (X - m1)).T @ (X - m1)
            C1 = 0.5 * (C1 + C1.T)
            ev, V = np.linalg.eigh(C1)
            C1 = (V * np.clip(ev, 1e-30, None)) @ V.T
            L1 = np.linalg.cholesky(C1 * inflate ** 2)
            n1 = n // 2
            X2 = np.vstack([q_prof_sample(n - n1), _mvt_sample(rng, m1, L1, nu, n1)])
            lp2 = evaluate(X2)
            nev += n
            # multiple-importance (balance heuristic) over all 2n samples:
            # stage-1 draws from q1, stage-2 from 0.5 q1 + 0.5 qt -> overall mixture 0.75 q1 + 0.25 qt
            Xa = np.vstack([X, X2])
            lpa = np.concatenate([lp, lp2])
            lq1 = q_prof_logpdf(Xa)
            lqt = _mvt_logpdf(Xa, m1, L1, nu)
            lqa = np.logaddexp(np.log(0.75) + lq1, np.log(0.25) + lqt)
            lw, khat = psis(lpa - lqa)
            w = np.exp(lw)
            X, lp, lq = Xa, lpa, lqa
            stage = 2

            def _q_final(Y):
                return np.logaddexp(np.log(0.75) + q_prof_logpdf(Y), np.log(0.25) + _mvt_logpdf(Y, m1, L1, nu))
        if defensive:
            # defensive draws from a broad t around the weighted draws (2 x their sd, beyond the
            # inflation), all draws weighted against the deterministic mixture: a proposal that stops
            # short of part of the posterior (an end of the grid cut off) gets large weights there,
            # which k-hat sees -- otherwise the weights of the draws it does make can look fine
            md = max(int(round(defensive * len(X))), 10 * (d + 1))
            md_mu = w @ X / w.sum()
            Cd = ((w[:, None] * (X - md_mu)).T @ (X - md_mu)) / w.sum()
            ev, V = np.linalg.eigh(0.5 * (Cd + Cd.T))
            Cd = (V * np.clip(ev, 1e-30, None)) @ V.T
            Ld = np.linalg.cholesky(Cd * (2.0 * inflate) ** 2)
            Xd = _mvt_sample(rng, md_mu, Ld, nu, md)
            lpd = evaluate(Xd)
            nev += md
            na = len(X)
            Xa = np.vstack([X, Xd])
            lpa = np.concatenate([lp, lpd])
            lq_old = np.concatenate([lq, _q_final(Xd)])
            lqa = np.logaddexp(np.log(na / (na + md)) + lq_old, np.log(md / (na + md)) + _mvt_logpdf(Xa, md_mu, Ld, nu))
            lw, khat = psis(lpa - lqa)
            w = np.exp(lw)
            X = Xa
        return X, w, lw, khat, stage, bs, lb, end_lo_ok, end_hi_ok

    can_refine = refine_tol is not None and len(grid) > 1
    n_ref = refine() if (can_refine and not lazy_refine) else 0
    X, w, lw, khat, stage, bs, lb, end_lo_ok, end_hi_ok = stages(build())
    if can_refine and lazy_refine and not (khat < khat_refine):
        # last resort before giving up: refine the grid and sample again (the draws so far are
        # discarded). Most poor stage-1 proposals come from non-Gaussian conditionals, which
        # stage 2 handles and refinement does not; refining whenever stage 1 is poor was tested
        # more expensive.
        n_ref = refine()
        if n_ref:
            X, w, lw, khat, stage, bs, lb, end_lo_ok, end_hi_ok = stages(build())
    return dict(theta=X, w=w, logw=lw, khat=float(khat), ess=float(1.0 / np.sum(w ** 2)),
                n_eval=int(nev), theta_map=theta_map, grid_b=bs, grid_logdens=lb, stage=stage,
                n_grid=len(bs), complete=bool(end_lo_ok and end_hi_ok), n_eval_map=int(nev_map),
                n_refine=int(n_ref),
                grid_lpmax=np.array([grid[b][1] for b in bs]),
                grid_logdet=np.array([grid[b][3] for b in bs]))
