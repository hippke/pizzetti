"""Limb-darkening laws (and the secondary eclipse) as linear models for
``MarginalTransitModel``.

Every law is written as I(mu) = sum_k c_k phi_k(mu) with K linear coefficients c
(including the flux scale). For each law this module provides

* ``basis(z, rp, alpha)``: n x K visible fluxes of the basis profiles (in units of
  pi R*^2), so that the light curve is ``basis @ c`` and the out-of-transit flux is
  ``disk(alpha) @ c``;
* the physical region: a cone ``D c >= 0`` (``D``), or ``None`` for the curved cone of
  the nonlinear law, which is enforced by an exact test (``is_physical``) instead;
* ``central(c)``: the central intensity I(1), used for the prior weight
  w = f0 / I(1)^K that turns the flat-in-c prior into "flat in the law's coefficients
  over the physical region x flat baseline";
* ``outputs(c, alpha)``: the law's usual coefficients for reporting.

Conventions follow batman / pizzetti (``laws.power_terms``):
  uniform     I = 1
  linear      I = 1 - u1 (1 - mu)
  quadratic   I = 1 - u1 (1 - mu) - u2 (1 - mu)^2
  squareroot  I = 1 - u1 (1 - mu) - u2 (1 - sqrt(mu))
  logarithmic I = 1 - u1 (1 - mu) - u2 mu ln(mu)           (as in batman)
  power2      I = 1 - c (1 - mu^alpha)
  nonlinear   I = 1 - sum_{n=1}^4 c_n (1 - mu^(n/2))
"""
import numpy as np

from .occult import occult
from .terms import occult_terms

# difference matrices of the ordered cones
_D2 = np.array([[1.0, 0.0], [-1.0, 1.0]])
_D3 = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
_TOL = 1e-12          # interior-series bound for the power-law basis curves


def blocked_power(z, rp, alpha, tol=_TOL):
    """Blocked flux (units of pi R*^2) of the profile mu^alpha."""
    if alpha == 0.0:
        return occult_terms(z, rp)[0]
    if alpha == 1.0:
        return occult_terms(z, rp)[1]
    if alpha == 2.0:
        T = occult_terms(z, rp)
        return T[0] - T[2]
    disk = 2.0 / (alpha + 2.0)
    return disk * (1.0 - occult(z, rp, [1.0, float(alpha)], "power2", series_tol=tol))


def blocked_mulogmu(z, rp, h=0.01):
    """Blocked flux of mu ln(mu) = d/dalpha mu^alpha at alpha = 1, by a fifth-order
    central difference in the exponent (truncation ~1e-11, see tests)."""
    f = [blocked_power(z, rp, 1.0 + k * h) for k in (-2, -1, 1, 2)]
    return (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h)


class Law:
    name = None
    K = None
    D = None                 # cone matrix, or None (curved: is_physical)
    has_alpha = False
    out_labels = {}

    def disk(self, alpha=None):
        raise NotImplementedError

    def basis(self, z, rp, alpha=None):
        raise NotImplementedError

    def central(self, c):
        raise NotImplementedError

    def outputs(self, c, alpha=None, ld="q"):
        raise NotImplementedError

    def from_u(self, u, alpha=None):
        """Coefficients c (with I(1) = 1) for the law's usual coefficients u."""
        raise NotImplementedError

    def is_physical(self, c, alpha=None):
        if self.D is None:
            raise NotImplementedError
        return np.all(c @ self.D.T >= 0.0, axis=1)


class Uniform(Law):
    name, K, D = "uniform", 1, np.eye(1)

    def disk(self, alpha=None):
        return np.array([1.0])

    def basis(self, z, rp, alpha=None):
        return (1.0 - occult_terms(z, rp)[0])[:, None]

    def central(self, c):
        return c[:, 0]

    def outputs(self, c, alpha=None, ld="q"):
        return np.zeros((len(c), 0)), []

    def from_u(self, u, alpha=None):
        return np.array([1.0])


class Linear(Law):
    """Bernstein degree 1: I = c0 (1 - mu) + c1 mu; physical 0 <= c0 <= c1."""
    name, K, D = "linear", 2, _D2

    def disk(self, alpha=None):
        return np.array([1.0 / 3.0, 2.0 / 3.0])

    def basis(self, z, rp, alpha=None):
        lex, ldx, _ = occult_terms(z, rp)
        return np.column_stack([1.0 / 3.0 - (lex - ldx), 2.0 / 3.0 - ldx])

    def central(self, c):
        return c[:, 1]

    def outputs(self, c, alpha=None, ld="q"):
        return (1.0 - c[:, 0] / c[:, 1])[:, None], ["u1"]

    def from_u(self, u, alpha=None):
        return np.array([1.0 - u[0], 1.0])


class Quadratic(Law):
    """Bernstein degree 2 in mu; physical 0 <= c0 <= c1 <= c2 (Kipping 2013)."""
    name, K, D = "quadratic", 3, _D3

    def disk(self, alpha=None):
        return np.array([1.0 / 6.0, 1.0 / 3.0, 0.5])

    def basis(self, z, rp, alpha=None):
        lex, ldx, edx = occult_terms(z, rp)
        return np.column_stack([1 / 6 - (2 * lex - 2 * ldx - edx), 1 / 3 - 2 * (ldx - lex + edx),
                                0.5 - (lex - edx)])

    def central(self, c):
        return c[:, 2]

    def outputs(self, c, alpha=None, ld="q"):
        b0, b1 = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
        u1 = 2.0 * (1.0 - b1)
        u2 = 1.0 - u1 - b0
        if ld == "u":
            return np.column_stack([u1, u2]), ["u1", "u2"]
        s = u1 + u2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.column_stack([s ** 2, np.where(s > 0, u1 / (2 * s), 0.5)]), ["q1", "q2"]

    def from_u(self, u, alpha=None):
        return np.array([1.0 - u[0] - u[1], 1.0 - 0.5 * u[0], 1.0])


class SquareRoot(Law):
    """A quadratic in s = sqrt(mu): Bernstein degree 2 in s, the same cone as the
    quadratic law. I = c0 (1-s)^2 + c1 2 s (1-s) + c2 s^2."""
    name, K, D = "squareroot", 3, _D3

    def disk(self, alpha=None):
        return np.array([1.0 / 15.0, 4.0 / 15.0, 2.0 / 3.0])

    def basis(self, z, rp, alpha=None):
        lex, ldx, _ = occult_terms(z, rp)
        bh = blocked_power(z, rp, 0.5)                   # blocked flux of sqrt(mu)
        # (1-s)^2 = 1 - 2 sqrt(mu) + mu ; 2 s (1-s) = 2 sqrt(mu) - 2 mu ; s^2 = mu
        return np.column_stack([1 / 15 - (lex - 2 * bh + ldx), 4 / 15 - (2 * bh - 2 * ldx),
                                2 / 3 - ldx])

    def central(self, c):
        return c[:, 2]

    def outputs(self, c, alpha=None, ld="q"):
        b0, b1 = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
        d = 2.0 * (b1 - b0)                               # batman u2 (sqrt term)
        cc = 1.0 + b0 - 2.0 * b1                          # batman u1 (linear term)
        if ld == "u":
            return np.column_stack([cc, d]), ["u1", "u2"]
        s = cc + d                                        # Kipping (2013) square-root q
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.column_stack([s ** 2, np.where(s > 0, d / (2 * s), 0.5)]), ["q1", "q2"]

    def from_u(self, u, alpha=None):
        c, d = u
        return np.array([1.0 - c - d, 1.0 - c - d / 2.0, 1.0])


class Logarithmic(Law):
    """I = A [1 - u1 (1-mu) - u2 mu ln mu], c = A (u2, u1, 1):
    I = c0 (-mu ln mu) + c1 (mu - 1) + c2 ; physical 0 <= u2 <= u1 <= 1 = the ordered cone."""
    name, K, D = "logarithmic", 3, _D3

    def disk(self, alpha=None):
        return np.array([2.0 / 9.0, -1.0 / 3.0, 1.0])

    def basis(self, z, rp, alpha=None):
        lex, ldx, _ = occult_terms(z, rp)
        bl = blocked_mulogmu(z, rp)                       # blocked flux of mu ln mu
        return np.column_stack([2 / 9 + bl, (2 / 3 - ldx) - (1 - lex), 1 - lex])

    def central(self, c):
        return c[:, 2]

    def outputs(self, c, alpha=None, ld="q"):
        return np.column_stack([c[:, 1] / c[:, 2], c[:, 0] / c[:, 2]]), ["u1", "u2"]

    def from_u(self, u, alpha=None):
        return np.array([u[1], u[0], 1.0])


class Power2(Law):
    """I = A [1 - c (1 - mu^alpha)] = c0 (1 - mu^alpha) + c1 mu^alpha, c = A (1-c, 1);
    physical 0 <= c0 <= c1 (0 <= c <= 1). alpha is a sampled (non-linear) parameter."""
    name, K, D, has_alpha = "power2", 2, _D2, True

    def disk(self, alpha=None):
        return np.array([alpha / (alpha + 2.0), 2.0 / (alpha + 2.0)])

    def basis(self, z, rp, alpha=None):
        lex = occult_terms(z, rp)[0]
        ba = blocked_power(z, rp, alpha)
        d = self.disk(alpha)
        return np.column_stack([d[0] - (lex - ba), d[1] - ba])

    def central(self, c):
        return c[:, 1]

    def outputs(self, c, alpha=None, ld="q"):
        return (1.0 - c[:, 0] / c[:, 1])[:, None], ["u1"]

    def from_u(self, u, alpha=None):
        return np.array([1.0 - u[0], 1.0])


def _bern_to_power(N):
    """M with p = M b: power coefficients (s^0..s^N) from Bernstein coefficients."""
    from math import comb
    M = np.zeros((N + 1, N + 1))
    for j in range(N + 1):                 # B_j(s) = C(N,j) s^j (1-s)^(N-j)
        for i in range(N - j + 1):
            M[j + i, j] += comb(N, j) * comb(N - j, i) * (-1) ** i
    return M


_M4 = _bern_to_power(4)


class Nonlinear(Law):
    """Degree-4 polynomial in s = sqrt(mu), written in the Bernstein basis
    B_j(s) = C(4,j) s^j (1-s)^(4-j) (well conditioned; the power basis has
    cond(F) ~ 1e13). The physical region is a curved convex cone; it is enforced by
    the exact test ``is_physical`` (importance weights), not an orthant probability."""
    name, K, D = "nonlinear", 5, None

    def disk(self, alpha=None):
        # int 2 mu B_j(sqrt mu) dmu = 4 int s^3 B_j(s) ds = 4 C(4,j) B(j+4, 5-j)
        from math import comb, factorial
        return np.array([4 * comb(4, j) * factorial(j + 3) * factorial(4 - j) / factorial(8)
                         for j in range(5)])

    def basis(self, z, rp, alpha=None):
        lex, ldx, edx = occult_terms(z, rp)
        blocked = [lex, blocked_power(z, rp, 0.5), ldx, blocked_power(z, rp, 1.5), lex - edx]
        dpow = np.array([4.0 / (k + 4.0) for k in range(5)])
        Vpow = np.column_stack([dpow[k] - blocked[k] for k in range(5)])   # power basis s^k
        return Vpow @ _M4                                                   # Bernstein columns

    def _power(self, c):
        return np.asarray(c, float) @ _M4.T

    def central(self, c):
        return np.asarray(c)[:, 4]                     # Bernstein value at s = 1

    def outputs(self, c, alpha=None, ld="q"):
        p = self._power(c)
        return p[:, 1:] / p.sum(axis=1)[:, None], ["u1", "u2", "u3", "u4"]

    def from_u(self, u, alpha=None):
        p = np.array([1.0 - sum(u)] + [float(x) for x in u])
        return np.linalg.solve(_M4, p)

    def is_physical(self, c, alpha=None):
        """Exact: I(s) = sum p_k s^k non-negative and non-decreasing on [0, 1]
        (I'(s) >= 0 at both ends and at the interior extrema of I'; I(0) >= 0)."""
        p = self._power(np.atleast_2d(c))
        ok = p[:, 0] >= 0.0                                        # I(0) >= 0
        d1 = lambda s: p[:, 1] + 2 * p[:, 2] * s + 3 * p[:, 3] * s ** 2 + 4 * p[:, 4] * s ** 3  # noqa: E731
        ok &= d1(0.0) >= 0.0
        ok &= d1(1.0) >= 0.0
        # interior extrema of I'(s): roots of I''(s) = 2 p2 + 6 p3 s + 12 p4 s^2
        a, b, cc = 12 * p[:, 4], 6 * p[:, 3], 2 * p[:, 2]
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
            for root in ((-b + sq) / (2 * a), (-b - sq) / (2 * a),
                         np.where(a == 0, -cc / b, np.nan)):
                inside = (root > 0) & (root < 1)
                val = d1(np.where(inside, root, 0.5))
                ok &= ~inside | (val >= 0.0)
        return ok


class Secondary(Law):
    """Secondary eclipse (uniform planet disk, as batman / pizzetti):
    flux = a + b v(z), v = visible fraction of the planet; a = stellar, b = planet
    flux, both >= 0. Prior: flat in fp = b / a (>= 0) x flat out-of-eclipse flux."""
    name, K, D = "secondary", 2, np.eye(2)

    def disk(self, alpha=None):
        return np.array([1.0, 1.0])                      # out of eclipse: a + b

    def basis(self, d, rp, alpha=None):
        from ._orbit import eclipse
        v = eclipse(np.ascontiguousarray(d, dtype=float), abs(float(rp)), 1.0) - 1.0
        return np.column_stack([np.ones_like(v), v])

    def central(self, c):
        return c[:, 0]

    def outputs(self, c, alpha=None, ld="q"):
        return (c[:, 1] / c[:, 0])[:, None], ["fp"]

    def from_u(self, u, alpha=None):
        return np.array([1.0, float(u)])


LAWS = {L.name: L for L in (Uniform, Linear, Quadratic, SquareRoot, Logarithmic, Power2, Nonlinear)}
