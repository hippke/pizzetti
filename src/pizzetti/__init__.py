"""pizzetti: fast limb-darkened transit light curves.

A drop-in replacement for batman's ``TransitModel`` / ``TransitParams``. While
the planet is entirely inside the stellar disk, the flux is evaluated with an
elementary series that has a rigorous error bound; near and across the limb
an exact (quadratic family) or quadrature (other power laws) solution is used.
"""
from .occult import SERIES_TOL, occult, series_cut
from .transitmodel import TransitModel, TransitParams

__version__ = "0.2.0"
__all__ = ["TransitModel", "TransitParams", "occult", "series_cut", "SERIES_TOL", "__version__"]
