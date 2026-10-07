"""Exact Mandel & Agol (2002) flux for quadratic limb darkening, used for the
samples that the interior series does not cover (near and across the stellar
limb).

Vectorised, branch-free evaluation with Bulirsch's general complete elliptic
integral (Bulirsch 1969) at a fixed number of steps, and elliptic operands in
closed form from (z, p) as in Agol, Luger & Foreman-Mackey (2020, eq. 41).
Lanes that are not converged after the fixed steps are recomputed with a
scalar early-exit iteration. Adapted from the Pandora code (Hippke & Heller
2022, GPL-3).

Conventions: z = projected centre distance, k = radius ratio (both in stellar
radii); quadratic law I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2. The flux is
    F = 1 - (c1 lex + c2 ldx + u2 edx) omega,
    c1 = 1 - u1 - 2 u2, c2 = u1 + 2 u2, omega = 1 / (1 - u1/3 - u2/6),
with the LD-independent terms lex (blocked area / pi), ldx (lambda_d of
Mandel & Agol plus 2/3 Theta(k - z)) and edx (eta_d).
"""
from math import pi

import numpy as np
from numba import njit

# fixed Bulirsch steps of the vectorised pass; non-converged lanes fall back
ELL_BATCH_ITERATIONS = 5
# floor of the complementary modulus (kc = 0 only at internal tangency)
KC_MIN = 1e-150
# smallest |z - k| (z == k exactly is moved by one ulp)
Z_K_MIN = 1e-150

# fdlibm e_asin / e_acos rational kernel
_PS0 = 1.66666666666666657415e-01
_PS1 = -3.25565818622400915405e-01
_PS2 = 2.01212532134862925881e-01
_PS3 = -4.00555345006794114027e-02
_PS4 = 7.91534994289814532176e-04
_PS5 = 3.47933107596021167570e-05
_QS1 = -2.40339491173441421878e+00
_QS2 = 2.02094576023350569471e+00
_QS3 = -6.88283971605453293030e-01
_QS4 = 7.70381505559019352791e-02
_HALF_PI = 1.5707963267948966


@njit(cache=True, fastmath=True, error_model="numpy", inline="always")
def acos_poly(x):
    """arccos(x) for -1 <= x <= 1, branch-free (fdlibm with selects), so that
    loops calling it vectorise; <= 1 ulp from libm."""
    ax = abs(x)
    small = ax <= 0.5
    z = x * x if small else 0.5 * (1.0 - ax)
    s = np.sqrt(z)
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    r = p / q
    a_small = _HALF_PI - (x + x * r)
    t = 2.0 * (s + s * r)
    a_big = t if x > 0 else pi - t
    return a_small if small else a_big


@njit(cache=True, fastmath=True, error_model="numpy", inline="always")
def acos_acc(x, om, op):
    """arccos(x) from x and accurately computed om = 1 - x, op = 1 + x
    (both >= 0), branch-free: no loss of accuracy near x = +-1, where
    arccos(x) = 2 asin(sqrt((1 - |x|) / 2)) is evaluated from the given
    (1 - |x|). Arguments are clamped to [-1, 1]."""
    x = max(min(x, 1.0), -1.0)
    om = max(om, 0.0)
    op = max(op, 0.0)
    ax = abs(x)
    small = ax <= 0.5
    zz = x * x if small else 0.5 * min(om if x > 0 else op, 1.0)
    s = np.sqrt(zz)
    p = zz * (_PS0 + zz * (_PS1 + zz * (_PS2 + zz * (_PS3 + zz * (_PS4 + zz * _PS5)))))
    q = 1.0 + zz * (_QS1 + zz * (_QS2 + zz * (_QS3 + zz * _QS4)))
    r = p / q
    a_small = _HALF_PI - (x + x * r)
    t = 2.0 * (s + s * r)
    a_big = t if x > 0 else pi - t
    return a_small if small else a_big


@njit(cache=True, fastmath=True, error_model="numpy", inline="always")
def limb_terms(z, k):
    """lex, edx on the limb (Case III) with the arccos arguments and the
    kite area in factored form (accurate at the contact points)."""
    INV_PI = 1 / pi
    k2 = k * k
    z2 = z * z
    zs = max(z, 1e-300)
    zk = max(z * k, 1e-300)
    # factors ordered so that cancelling sums are exact (Sterbenz)
    f1 = (z - 1.0) + k  # z + k - 1 >= 0 on the limb
    f2 = (1.0 - z) + k  # 1 + k - z
    f3 = (1.0 - k) + z  # 1 + z - k
    f4 = 1.0 + z + k
    f5 = f2
    f6 = f1
    # kap1 = acos((1 - k^2 + z^2) / (2 z)), 1 - x = f1 f2 / (2z), 1 + x = f3 f4 / (2z)
    kap1 = acos_acc((1 - k2 + z2) / (2 * zs), f1 * f2 / (2 * zs), f3 * f4 / (2 * zs))
    # kap0 = acos((k^2 + z^2 - 1) / (2 k z)), 1 - x = f5 f3 / (2kz), 1 + x = f6 f4 / (2kz)
    kap0 = acos_acc((k2 + z2 - 1) / (2 * zk), f5 * f3 / (2 * zk), f6 * f4 / (2 * zk))
    kite = np.sqrt(max(f1 * f2 * f3 * f4, 0.0))
    lex = (k2 * kap0 + kap1 - 0.5 * kite) * INV_PI
    edx = 0.5 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0 - (1 + 5 * k2 + z2) / 4 * kite)
    return lex, edx


@njit(cache=True, fastmath=True)
def ell_combo_kc(kc, p, A, B, C, Cp):
    """A K + B E + C Pi from one Bulirsch iteration with early exit.
    kc: complementary modulus, p = sqrt(n + 1), Cp = C / p."""
    a1 = A + B
    b1 = A + B * kc * kc
    a2 = C
    b2 = Cp
    e = kc
    m0 = 1.0
    for _ in range(1000):
        f1 = a1
        a1 = a1 + b1 / m0
        b1 = 2 * (b1 + f1 * kc)
        inv_p = 1 / p
        f2 = a2
        a2 = a2 + b2 * inv_p
        g = e * inv_p
        b2 = 2 * (b2 + f2 * g)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(g - kc) > 1e-8 * g:
            kc = 2 * np.sqrt(e)
            e = kc * m0
        else:
            return 0.5 * pi * ((b1 + a1 * m0) / (2 * m0 * m0) + (b2 + a2 * m0) / (m0 * (m0 + p)))
    return np.nan


@njit(cache=True, fastmath=True)
def exact_point(z, k, c1, c2, u2, omega):
    """Exact flux at one on-star z (0 <= z < 1 + k, z != k, not fully
    occulted), scalar."""
    INV_PI = 1 / pi
    k2 = k * k
    z2 = z * z
    s = z + k
    d = z - k
    x1 = d * d
    x2 = s * s
    x3 = k2 - z2
    sg = 1.0 if d > 0 else -1.0
    if z < 1 - k:
        lex = k2
        edx = k2 / 2 * (k2 + 2 * z2)
        num = ((1.0 - z) - k) * (1 + s)
        den = 1 - x1
        A = 1 - 5 * z2 + k2 + x3 * x3
        p = s / abs(d)
        Cp = 3 * sg
    else:
        lex, edx = limb_terms(z, k)
        num = ((z - 1.0) + k) * (s + 1)
        den = 4 * z * k
        A = (1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)
        p = 1 / abs(d)
        Cp = 3 * s * sg
    sden = np.sqrt(den)
    kc = max(np.sqrt(max(num, 0.0)) / sden, KC_MIN)
    ldx = 2 / 9 * INV_PI / sden * ell_combo_kc(kc, p, A, den * (z2 + 7 * k2 - 4), 3 * s / d, Cp)
    if z < k:
        ldx = ldx + 2 / 3
    return 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega


@njit(cache=True, fastmath=True, error_model="numpy")
def exact_batch(zb, m, k, c1, c2, u2, omega, flux, dev):
    """Exact flux for zb[:m] (on-star, z != k, not fully occulted), branch-free
    so that LLVM vectorises the loop. dev[j] > 0 marks lanes that need the
    scalar exact_point()."""
    INV_PI = 1 / pi
    k2 = k * k
    for j in range(m):
        z = zb[j]
        z2 = z * z
        s = z + k
        d = z - k
        x1 = d * d
        x2 = s * s
        x3 = k2 - z2
        inner = z < 1 - k
        num = ((1.0 - z) - k) * (1 + s) if inner else ((z - 1.0) + k) * (s + 1)
        den = 1 - x1 if inner else 4 * z * k
        inv_sden = 1 / np.sqrt(den)
        kc = max(np.sqrt(max(num, 0.0)) * inv_sden, KC_MIN)
        sg = 1.0 if d > 0 else -1.0
        inv_ad = 1 / abs(d)
        p = s * inv_ad if inner else inv_ad
        a2 = 3 * s * sg * inv_ad
        b2 = 3 * sg if inner else 3 * s * sg
        A = (1 - 5 * z2 + k2 + x3 * x3) if inner else ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2))
        B = den * (z2 + 7 * k2 - 4)
        a1 = A + B
        b1 = A + B * kc * kc
        e = kc
        m0 = 1.0
        dd = 0.0
        for _ in range(ELL_BATCH_ITERATIONS):
            r = 1 / (m0 * p)
            f1 = a1
            a1 = a1 + b1 * (p * r)
            b1 = 2 * (b1 + f1 * kc)
            inv_p = m0 * r
            f2 = a2
            a2 = a2 + b2 * inv_p
            g = e * inv_p
            b2 = 2 * (b2 + f2 * g)
            p = g + p
            g = m0
            m0 = kc + m0
            dd = abs(g - kc) - 1e-8 * g
            kc = 2 * np.sqrt(e)
            e = kc * m0
        mp = m0 + p
        ell = 0.25 * pi * ((b1 + a1 * m0) * mp + 2 * m0 * (b2 + a2 * m0)) / (m0 * m0 * mp)
        ldx = (2 / 9 * INV_PI) * inv_sden * ell + (2 / 3 if z < k else 0.0)
        lex_l, edx_l = limb_terms(z, k)
        lex = k2 if inner else lex_l
        edx = k2 / 2 * (k2 + 2 * z2) if inner else edx_l
        flux[j] = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega
        dev[j] = dd
