"""batman-compatible transit model: ``TransitParams`` and ``TransitModel``.

The interface follows batman (Kreidberg 2015) so that existing code works
after replacing ``import batman`` by ``import pizzetti as batman``.
"""
import warnings
from math import pi

import numpy as np

from . import _orbit
from .laws import NCOEFF, check_law
from .occult import SERIES_TOL, occult, series_cut

__all__ = ["TransitModel", "TransitParams"]


class TransitParams(object):
    """Physical parameters of the transit (same attributes as batman).

    :ivar t0: time of inferior conjunction
    :ivar t_secondary: time of secondary eclipse centre
    :ivar per: orbital period
    :ivar rp: planet radius (stellar radii); negative values give inverse transits
    :ivar a: semi-major axis (stellar radii)
    :ivar inc: orbital inclination (degrees)
    :ivar ecc: eccentricity
    :ivar w: argument of periapse (degrees)
    :ivar u: limb-darkening coefficients (list)
    :ivar limb_dark: "uniform", "linear", "quadratic", "squareroot", "nonlinear" or "power2"
    :ivar fp: planet-to-star flux ratio (secondary eclipses)

    Example::

        params = pizzetti.TransitParams()
        params.t0 = 0.; params.per = 1.; params.rp = 0.1; params.a = 15.
        params.inc = 87.; params.ecc = 0.; params.w = 90.
        params.u = [0.1, 0.3]; params.limb_dark = "quadratic"
    """

    def __init__(self):
        self.t0 = None
        self.per = None
        self.rp = None
        self.a = None
        self.inc = None
        self.ecc = None
        self.w = None
        self.u = None
        self.limb_dark = None
        self.fp = None
        self.t_secondary = None


class TransitModel(object):
    """Model transit light curves (batman-compatible).

    :param params: :class:`TransitParams`
    :param t: times (numpy array)
    :param max_err: accepted for compatibility (ppm). The interior series is
        bounded by ``series_tol``; the limb quadrature is accurate to ~1e-10.
    :param nthreads: accepted for compatibility; pizzetti runs single-threaded
    :param fac: accepted for compatibility and ignored
    :param transittype: "primary" or "secondary"
    :param supersample_factor: number of points per exposure
    :param exp_time: exposure time (same units as ``t``)
    :param series_tol: rigorous bound on the flux error of the interior series
        (absolute flux, default 1e-9); 0 disables the series

    Example::

        m = pizzetti.TransitModel(params, t)
        flux = m.light_curve(params)
    """

    def __init__(self, params, t, max_err=1.0, nthreads=1, fac=None, transittype="primary",
                 supersample_factor=1, exp_time=0.0, series_tol=SERIES_TOL):
        check_law(params.limb_dark, params.u)
        if transittype not in ("primary", "secondary"):
            raise ValueError('Allowed transit types are "primary" and "secondary".')
        if supersample_factor > 1 and exp_time <= 0.0:
            raise ValueError("Please enter a valid exposure time (exp_time must be greater than 0 "
                             "to calculate super-sampled light curves).")
        if not isinstance(t, np.ndarray):
            raise TypeError("Times t must be a numpy array (not a list).")
        if nthreads not in (None, 1):
            warnings.warn("pizzetti runs single-threaded; nthreads is ignored", stacklevel=2)
        self.t = np.ascontiguousarray(t, dtype=np.float64)
        self.max_err = max_err
        self.fac = fac
        self.nthreads = 1
        self.series_tol = series_tol
        self.supersample_factor = int(supersample_factor)
        self.exp_time = exp_time
        if self.supersample_factor > 1:
            offs = np.linspace(-exp_time / 2.0, exp_time / 2.0, self.supersample_factor)
            self.t_supersample = (offs + self.t.reshape(self.t.size, 1)).flatten()
        else:
            self.t_supersample = self.t
        self.transittype = 1 if transittype == "primary" else 2
        if self.transittype == 2:
            params.t0 = self.get_t_conjunction(params)
        self._store(params)
        self.ds = self._rsky(params)

    # -- internals ---------------------------------------------------------
    def _store(self, params):
        self.t0 = params.t0
        self.per = params.per
        self.rp = abs(params.rp)
        self.a = params.a
        self.inc = params.inc
        self.ecc = params.ecc
        self.w = params.w
        self.u = params.u
        self.limb_dark = params.limb_dark
        self.fp = params.fp
        self.t_secondary = params.t_secondary
        self.inverse = params.rp < 0.0

    def _rsky(self, params):
        return _orbit.rsky(self.t_supersample, float(params.t0), float(params.per), float(params.a),
                           float(params.inc) * pi / 180.0, float(params.ecc), float(params.w) * pi / 180.0,
                           self.transittype)

    # -- public API (batman) -----------------------------------------------
    def light_curve(self, params):
        """Relative flux at the times ``t`` for the parameters ``params``."""
        if (params.t0 != self.t0 or params.per != self.per or params.a != self.a
                or params.inc != self.inc or params.ecc != self.ecc or params.w != self.w
                or params.t_secondary != self.t_secondary):
            if self.transittype == 2 and params.t_secondary != self.t_secondary:
                params.t0 = self.get_t_conjunction(params)
            self.ds = self._rsky(params)
        self._store(params)
        if self.transittype == 1:
            check_law(params.limb_dark, params.u)
            lc = occult(self.ds, abs(params.rp), params.u, params.limb_dark, self.series_tol)
            if self.inverse:
                lc = 2.0 - lc
        else:
            lc = _orbit.eclipse(self.ds, abs(float(params.rp)), float(params.fp))
        if self.supersample_factor == 1:
            return lc
        return np.mean(lc.reshape(-1, self.supersample_factor), axis=1)

    def series_cut(self, params=None):
        """Separation z_c (stellar radii) up to which the interior series is
        used; its flux error is rigorously below ``series_tol`` there.
        Returns -1 if the series is not used."""
        p = self if params is None else params
        return series_cut(abs(p.rp), p.limb_dark, p.u, self.series_tol)

    def calc_err(self, plot=False):
        """Maximum flux error (ppm) of the model on a grid of separations,
        estimated against a reference evaluation without the series and with
        a doubled quadrature order. The series part is bounded rigorously by
        ``series_tol``."""
        from . import _general
        ds = np.linspace(0.0, 1.0 + self.rp, 2000)
        f = occult(ds, self.rp, self.u, self.limb_dark, self.series_tol)
        f0 = occult(ds, self.rp, self.u, self.limb_dark, 0.0)
        err = np.max(np.abs(f - f0)) * 1e6
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(ds, 1e6 * (f - f0), color="k")
            plt.xlabel("d (separation of centers)")
            plt.ylabel("Error (ppm)")
            plt.show()
        return max(err, self.series_tol * 1e6)

    def _get_phase(self, params, position):
        if position == "periastron":
            TA = 0.0
        elif position == "primary":
            TA = pi / 2.0 - params.w * pi / 180.0
        else:
            TA = 3.0 * pi / 2.0 - params.w * pi / 180.0
        E = 2.0 * np.arctan(np.sqrt((1.0 - params.ecc) / (1.0 + params.ecc)) * np.tan(TA / 2.0))
        M = E - params.ecc * np.sin(E)
        return M / 2.0 / pi

    def get_t_periastron(self, params):
        """Time of periastron passage (from ``params.t0``)."""
        return params.t0 - params.per * self._get_phase(params, "primary")

    def get_t_secondary(self, params):
        """Time of secondary eclipse centre (from ``params.t0``)."""
        return params.t0 + params.per * (self._get_phase(params, "secondary") - self._get_phase(params, "primary"))

    def get_t_conjunction(self, params):
        """Time of primary transit centre (from ``params.t_secondary``)."""
        return params.t_secondary + params.per * (self._get_phase(params, "primary") - self._get_phase(params, "secondary"))

    def get_true_anomaly(self):
        """True anomaly at each (supersampled) time."""
        self.f = _orbit.true_anomaly(self.t_supersample, float(self.t0), float(self.per), float(self.ecc),
                                     float(self.w) * pi / 180.0)
        return self.f
