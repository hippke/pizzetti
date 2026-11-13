"""How many sub-exposures does a light curve need?

A long exposure averages the light curve over its duration; a model evaluated at ``k`` sub-exposures approximates that
average. What matters for a fit is the error of that approximation summed over all data points,

    chi2(k) = sum_i (f_k(t_i) - f_exact(t_i))^2 / sigma_i^2,

with ``f_exact`` from many more sub-exposures. A model error of chi2 can move a posterior by at most about sqrt(chi2)
standard deviations (the shift along any direction of a least-squares fit is bounded by the norm of the error in the
data's metric), so chi2 <= 0.01 keeps every parameter within about 0.1 sigma of the exact exposure average. On 43
benchmark light curves with exposures of 200 s to 30 min, the medians then moved by at most 0.05 sigma.
"""
import copy
import math

import numpy as np

from .transitmodel import TransitModel

K_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 31, 37, 45, 55, 67, 81, 101, 121, 151, 201)


def _flux(params, t, k, exp_time, offsets, transittype):
    p = copy.copy(params)                      # TransitModel moves t0 of a secondary eclipse
    m = TransitModel(p, t, supersample_factor=k, exp_time=exp_time if k > 1 else 0.0, offsets=offsets,
                     transittype=transittype)
    return m.light_curve(p)


def exposure_chi2(params, t, exp_time, sigma, k, k_ref=None, offsets="midpoint", transittype="primary"):
    """chi^2 of the model error of ``k`` sub-exposures against ``k_ref`` (default max(4 k + 1, 61)) midpoint
    sub-exposures: sum((f_k - f_ref)^2 / sigma^2) over the time stamps ``t`` (days), for the transit ``params``
    (``TransitParams``, as for ``TransitModel``) and exposures of ``exp_time`` days. ``sigma``: the noise per point
    (scalar or one per point; white noise). ``offsets``: placement of the k points ("midpoint" or "batman")."""
    t = np.ascontiguousarray(t, dtype=float)
    if not exp_time > 0:
        return 0.0
    k_ref = max(4 * int(k) + 1, 61) if k_ref is None else int(k_ref)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), t.shape)
    d = _flux(params, t, int(k), exp_time, offsets, transittype) - _flux(params, t, k_ref, exp_time, "midpoint", transittype)
    return float(np.sum((d / sigma) ** 2))


def choose_supersample(params, t, exp_time, sigma, tol=0.01, k_max=201, offsets="midpoint", transittype="primary",
                       return_table=False):
    """The smallest number of sub-exposures ``k`` whose model error, ``exposure_chi2``, is at most ``tol``.

    ``params`` (``TransitParams``): the transit at which the error is evaluated, e.g. a first guess or catalogue values
    (the error depends on the geometry but little on its details; ``MarginalTransitModel.fit()`` checks again at the
    posterior median). ``t``: the time stamps (days) of the data to be fitted, ``exp_time``: the exposure (days),
    ``sigma``: the noise per point (scalar or array; for a model with a Gaussian process, its white-noise part).
    ``tol``: 0.01 (default) bounds the shift of the posterior at about 0.1 sigma. Candidates are taken from
    ``K_GRID`` up to ``k_max``; if none reaches ``tol``, ``k_max`` is returned. With ``return_table``, also the dict
    {k: chi2} of the candidates tried (in increasing order, stopping at the first that passes). Costs a few light
    curves with up to ~5 k points per data point, usually well under a second.

    >>> k = pizzetti.choose_supersample(params, t, exp_time=1765.5 / 86400, sigma=2e-4)
    >>> model = pizzetti.MarginalTransitModel(params, t, y, 2e-4, supersample_factor=k, exp_time=1765.5 / 86400)
    """
    table = {}
    k = 1
    if exp_time > 0:
        for k in [c for c in K_GRID if c <= k_max] + ([] if k_max in K_GRID else [int(k_max)]):
            table[k] = exposure_chi2(params, t, exp_time, sigma, k, offsets=offsets, transittype=transittype)
            if table[k] <= tol:
                break
    return (k, table) if return_table else k


def shift_bound(chi2):
    """Upper bound (posterior standard deviations) on the shift a model error of ``chi2`` can cause: sqrt(chi2)."""
    return math.sqrt(max(chi2, 0.0))
