# Exact Mandel & Agol (2002) flux for the quadratic family, for the samples that
# the interior series does not cover. Bulirsch's general complete elliptic
# integral, with the elliptic operands in closed form from (z, p) as in Agol,
# Luger & Foreman-Mackey (2020, eq. 41); the cancelling sums are ordered so
# that they are exact (Sterbenz), which keeps full accuracy at the contacts.
#
#     F = 1 - (c1 lex + c2 ldx + u2 edx) ω,
#     c1 = 1 - u1 - 2u2,  c2 = u1 + 2u2,  ω = 1/(1 - u1/3 - u2/6),
# with lex = blocked area/π, ldx = λ_d (+ 2/3 for z < p) and edx = η_d.
# Generic in the number type (dual numbers, BigFloat).

# arccos(x) from accurate om = 1 - x and op = 1 + x
function _acos_acc(x, om, op)
    x = clamp(x, -one(x), one(x))
    if x > 0.5
        return 2 * asin(sqrt(max(om, zero(om)) / 2))
    elseif x < -0.5
        return oftype(x, π) - 2 * asin(sqrt(max(op, zero(op)) / 2))
    end
    acos(x)
end

lex_type(z, k) = float(promote_type(typeof(z), typeof(k)))

function _limb_terms(z, k)
    k2 = k * k
    z2 = z * z
    zs = max(z, floatmin(_float(z)))
    zk = max(z * k, floatmin(_float(z)))
    f1 = (z - 1) + k            # z + k - 1 ≥ 0 on the limb
    f2 = (1 - z) + k
    f3 = (1 - k) + z
    f4 = 1 + z + k
    kap1 = _acos_acc((1 - k2 + z2) / (2zs), f1 * f2 / (2zs), f3 * f4 / (2zs))
    kap0 = _acos_acc((k2 + z2 - 1) / (2zk), f2 * f3 / (2zk), f1 * f4 / (2zk))
    kite = sqrt(max(f1 * f2 * f3 * f4, zero(f1)))
    piT = lex_type(z, k)(π)
    lex = (k2 * kap0 + kap1 - kite / 2) / piT
    edx = (kap1 + k2 * (k2 + 2z2) * kap0 - (1 + 5k2 + z2) / 4 * kite) / (2piT)
    return lex, edx
end

# A K + B E + C Π (Bulirsch 1969), kc complementary modulus, p = sqrt(n + 1), Cp = C/p
function _ell_combo(kc, p, A, B, C, Cp)
    tol = sqrt(eps(_float(kc)))
    a1 = A + B
    b1 = A + B * kc * kc
    a2 = C
    b2 = Cp
    e = kc
    m0 = one(kc)
    for _ in 1:200
        f1 = a1
        a1 = a1 + b1 / m0
        b1 = 2 * (b1 + f1 * kc)
        ip = 1 / p
        f2 = a2
        a2 = a2 + b2 * ip
        g = e * ip
        b2 = 2 * (b2 + f2 * g)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(_val(g) - _val(kc)) > tol * _val(g)
            kc = 2 * sqrt(e)
            e = kc * m0
        else
            return oftype(a1, π) / 2 * ((b1 + a1 * m0) / (2 * m0 * m0) + (b2 + a2 * m0) / (m0 * (m0 + p)))
        end
    end
    return oftype(a1, NaN)
end

"""
    quadratic_flux(z, p, u1, u2)

Exact flux of a star with quadratic limb darkening `I = 1 - u1 (1-μ) - u2 (1-μ)²`
occulted by an opaque disk of radius `p` at separation `z` (Mandel & Agol 2002),
valid everywhere (0 ≤ z, any p > 0).
"""
function quadratic_flux(z, k, u1, u2)
    T = promote_type(typeof(float(z)), typeof(float(k)), typeof(float(u1)), typeof(float(u2)))
    z = abs(T(z)); k = abs(T(k)); u1 = T(u1); u2 = T(u2)
    (k <= 0 || z >= 1 + k) && return one(T)
    k >= 1 && z <= k - 1 && return zero(T)
    ω = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2u2
    c2 = u1 + 2u2
    if abs(_val(z) - _val(k)) < 1e-150 * one(_val(k)) || z == k
        z = k + max(k * eps(_float(k)), T(1e-150))    # the formulas exclude z = p exactly
    end
    k2 = k * k
    z2 = z * z
    s = z + k
    d = z - k
    x1 = d * d
    x2 = s * s
    x3 = k2 - z2
    sg = d > 0 ? one(T) : -one(T)
    if z < 1 - k
        lex = k2
        edx = k2 / 2 * (k2 + 2z2)
        num = ((1 - z) - k) * (1 + s)
        den = 1 - x1
        A = 1 - 5z2 + k2 + x3 * x3
        pp = s / abs(d)
        Cp = 3sg
    else
        lex, edx = _limb_terms(z, k)
        num = ((z - 1) + k) * (s + 1)
        den = 4z * k
        A = (1 - x2) * (2x2 + x1 - 3) - 3x3 * (x2 - 2)
        pp = 1 / abs(d)
        Cp = 3s * sg
    end
    sden = sqrt(den)
    kc = max(sqrt(max(num, zero(T))) / sden, T(1e-150))
    ldx = 2 / (9 * T(π)) / sden * _ell_combo(kc, pp, A, den * (z2 + 7k2 - 4), 3s / d, Cp)
    z < k && (ldx += T(2) / 3)
    return 1 - (c1 * lex + c2 * ldx + u2 * edx) * ω
end
