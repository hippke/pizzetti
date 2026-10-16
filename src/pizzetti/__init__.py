"""pizzetti: fast limb-darkened transit light curves.

A drop-in replacement for batman's ``TransitModel`` / ``TransitParams``. While
the planet is entirely inside the stellar disk, the flux is evaluated with an
elementary series that has a rigorous error bound; near and across the limb
an exact (quadratic family) or quadrature (other power laws) solution is used.
"""
from . import _cachecheck

_cachecheck.check()          # before any numba-compiled module is imported

from .occult import SERIES_TOL, occult, series_cut  # noqa: E402
from .transitmodel import TransitModel, TransitParams  # noqa: E402
from .terms import occult_terms, quadratic_from_terms  # noqa: E402

__version__ = "0.2.1+terms"
__all__ = ["TransitModel", "TransitParams", "occult", "occult_terms", "quadratic_from_terms",
           "series_cut", "SERIES_TOL", "__version__"]
