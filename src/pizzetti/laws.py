"""Limb-darkening laws in batman's conventions, written as I(mu) = sum_i c_i mu^alpha_i."""
import numpy as np

SUPPORTED = ("uniform", "linear", "quadratic", "squareroot", "nonlinear", "power2")
NCOEFF = {"uniform": 0, "linear": 1, "quadratic": 2, "squareroot": 2, "nonlinear": 4, "power2": 2}
#: laws whose limb part uses the exact elliptic-integral solution
QUADRATIC_FAMILY = ("uniform", "linear", "quadratic")


def check_law(limb_dark, u):
    if limb_dark in ("logarithmic", "exponential", "custom"):
        raise NotImplementedError(
            f'"{limb_dark}" limb darkening is not a sum of powers of mu and is not supported by pizzetti; '
            f"supported: {', '.join(SUPPORTED)}")
    if limb_dark not in SUPPORTED:
        raise ValueError(f'"{limb_dark}" limb darkening not supported; allowed options are: {", ".join(SUPPORTED)}')
    u = [] if u is None else list(u)
    if len(u) != NCOEFF[limb_dark]:
        raise ValueError(f"{limb_dark} limb darkening needs {NCOEFF[limb_dark]} coefficient(s), got {len(u)}")
    return u


def quadratic_coefficients(limb_dark, u):
    """(u1, u2) of the quadratic-family laws."""
    if limb_dark == "uniform":
        return 0.0, 0.0
    if limb_dark == "linear":
        return float(u[0]), 0.0
    return float(u[0]), float(u[1])


def power_terms(limb_dark, u):
    """(alphas, coefs) with I(mu) = sum coefs * mu**alphas (batman conventions):
      uniform    I = 1
      linear     I = 1 - u1 (1 - mu)
      quadratic  I = 1 - u1 (1 - mu) - u2 (1 - mu)^2
      squareroot I = 1 - u1 (1 - mu) - u2 (1 - sqrt(mu))
      nonlinear  I = 1 - sum_{n=1}^4 c_n (1 - mu^(n/2))
      power2     I = 1 - c (1 - mu^alpha),  u = [c, alpha]
    """
    u = check_law(limb_dark, u)
    if limb_dark == "uniform":
        al, c = [0.0], [1.0]
    elif limb_dark == "linear":
        al, c = [0.0, 1.0], [1.0 - u[0], u[0]]
    elif limb_dark == "quadratic":
        al, c = [0.0, 1.0, 2.0], [1.0 - u[0] - u[1], u[0] + 2 * u[1], -u[1]]
    elif limb_dark == "squareroot":
        al, c = [0.0, 1.0, 0.5], [1.0 - u[0] - u[1], u[0], u[1]]
    elif limb_dark == "nonlinear":
        al = [0.0, 0.5, 1.0, 1.5, 2.0]
        c = [1.0 - sum(u)] + [float(x) for x in u]
    else:  # power2
        cc, alpha = float(u[0]), float(u[1])
        if not alpha > 0:
            raise ValueError("power2 limb darkening needs alpha > 0")
        al, c = [0.0, alpha], [1.0 - cc, cc]
    return np.array(al, dtype=np.float64), np.array(c, dtype=np.float64)
