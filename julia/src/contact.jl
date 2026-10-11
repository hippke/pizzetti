# Contact expansions (paper, Section 5.2): the flux near and across the limb as
# power series in the hypergeometric argument, with coefficients that depend only
# on p (computed once per light curve). With m = γ + 3/2, y = 1 - k, y' = 1 - κ:
#
#   A  first contact,           0 ≤ k ≤ ½:   P_γ = k^m RA(k)
#   B  second contact, outside, 0 ≤ y ≤ ½:   P_γ = RB(y) + y^m [ln(y) LB(y) + CB(y)]
#   C  second contact, inside,  0 ≤ y' ≤ y'_c: P_γ = RC(y') + y'^m [ln(y') LC(y') + CC(y')]
#
# On each piece z is an algebraic function of the argument,
#   z = p(1 - 2k) + sqrt(p²(1 - 2k)² + 1 - p²),
#   z = [sqrt(p²(2 - κ)² + κ²(1 - p²)) - p(2 - κ)] / κ,
# so the elementary factors of the hypergeometric representation are power series
# in the argument, obtained by exact power-series arithmetic. Float64.

"Order of the contact series (polynomial degree)."
const CONTACT_NS = 32
"Largest j of the hypergeometric expansion used to build the series."
const CONTACT_JC = 200
"Accepted estimate of the relative truncation error (else: quadrature)."
const CONTACT_TOL = 1e-11

# ---- truncated power-series arithmetic (length n) ----
function _smul!(out, a, b)
    n = length(out)
    @inbounds for i in 1:n
        s = 0.0
        for k in 1:i
            s += a[k] * b[i-k+1]
        end
        out[i] = s
    end
    out
end
_smul(a, b) = _smul!(similar(a), a, b)

function _sinv(a)
    n = length(a)
    b = zeros(n)
    b[1] = 1 / a[1]
    @inbounds for i in 2:n
        s = 0.0
        for k in 2:i
            s += a[k] * b[i-k+1]
        end
        b[i] = -s / a[1]
    end
    b
end

# a^al for a[1] > 0 (J. C. P. Miller's recurrence)
function _spow(a, al)
    n = length(a)
    f = zeros(n)
    f[1] = a[1]^al
    @inbounds for i in 1:n-1
        s = 0.0
        for k in 1:i
            s += ((al + 1) * k - i) * a[k+1] * f[i-k+1]
        end
        f[i+1] = s / (i * a[1])
    end
    f
end

function _ssize(c, umax)
    s = 0.0
    w = 1.0
    for x in c
        s += abs(x) * w
        w *= umax
    end
    s
end

function _sshift!(out, a, j)
    n = length(a)
    @inbounds for i in 1:n
        out[i] = i > j ? a[i-j] : 0.0
    end
    out
end

# relative size of the last three terms at umax
function _stail(c, umax)
    N = length(c) - 1
    tot = _ssize(c, umax)
    tot == 0 && return 0.0
    w = umax^(N - 2)
    t = 0.0
    for i in N-2:N
        t += abs(c[i+1]) * w
        w *= umax
    end
    t / tot
end

"""
    contact_pieces(γ, p, tab::LensTable, N, ymax_c; Jfix = (-1, -1, -1))

Series of the three contact pieces for one exponent: RA, B (3 × (N+1)), C (3 × (N+1))
(regular, log and power parts) and an estimate of the relative truncation error.
`Jfix ≥ 0` fixes the number of j-terms of a piece (used by the proven mode).
"""
function contact_pieces(g::Float64, p::Float64, tab::LensTable{Float64}, N::Int, ymax_c::Float64;
                        Jfix = (-1, -1, -1))
    T = tab.T
    n = N + 1
    jm = min(size(T, 2) - 1, CONTACT_JC)
    one_ = zeros(n); one_[1] = 1.0
    u = zeros(n); u[2] = 1.0
    B = zeros(3, n)
    C = zeros(3, n)
    est = 0.0
    t1 = zeros(n); t2 = zeros(n)
    for piece in 0:1
        if piece == 0
            a1 = 2 .* u .- one_
            z = p .* a1 .+ _spow(p^2 .* _smul(a1, a1) .+ (1 - p^2) .* one_, 0.5)
            k = one_ .- u
            x = 4p .* _smul(z, k)
            pre = -_smul(_spow(x, g + 1), _spow(k, 0.5)) ./ (2(g + 1))
            fam = 1
            umax = 0.5
        else
            kap = one_ .- u
            a1 = one_ .+ u
            z = _smul(_spow(p^2 .* _smul(a1, a1) .+ (1 - p^2) .* _smul(kap, kap), 0.5) .- p .* a1, _sinv(kap))
            x = 4p .* _smul(z, _sinv(kap))
            pre = -_spow(x, g + 1) ./ (2(g + 1))
            fam = 2
            umax = ymax_c
        end
        q = p^2 .* one_ .- _smul(z, z)
        R = zeros(n); SL = zeros(n); SC = zeros(n)
        xj = copy(one_)
        cj = one_ .+ q
        small = 0
        contrib = 1.0
        scale = 1.0
        conv = false
        jf = Jfix[2+piece]
        jlim = jf < 0 ? jm : min(jm, jf)
        dR = zeros(n); dL = zeros(n); dC = zeros(n)
        for j in 0:jlim
            if j >= 1
                _smul!(t1, xj, x); xj .= t1
                _smul!(cj, q, xj)
            end
            _smul!(dR, cj, view(T, fam, j + 1, 2, 1:n))
            _sshift!(t2, cj, j)
            _smul!(dL, t2, view(T, fam, j + 1, 3, 1:n))
            _smul!(dC, t2, view(T, fam, j + 1, 4, 1:n))
            R .+= dR; SL .+= dL; SC .+= dC
            if j >= 2 && jf < 0
                contrib = _ssize(dR, umax) + _ssize(dL, umax) + _ssize(dC, umax)
                scale = _ssize(R, umax) + _ssize(SL, umax) + _ssize(SC, umax)
                if contrib <= 1e-18 * scale
                    small += 1
                    if small >= 3
                        conv = true
                        break
                    end
                else
                    small = 0
                end
            end
        end
        if !conv && jf < 0
            x0 = x[1]
            est = max(est, contrib / max(scale, 1e-300) * x0 / max(1 - x0, 1e-3))
        end
        out = piece == 0 ? B : C
        out[1, :] .= _smul(pre, R)
        out[2, :] .= _smul(pre, SL)
        out[3, :] .= _smul(pre, SC)
        for f in 1:3
            est = max(est, _stail(view(out, f, :), umax))
        end
    end
    # piece A, u = k: P = k^m RA(k), x = 4 z p k
    a1 = one_ .- 2 .* u
    z = p .* a1 .+ _spow(p^2 .* _smul(a1, a1) .+ (1 - p^2) .* one_, 0.5)
    q = p^2 .* one_ .- _smul(z, z)
    PA = -((4p)^(g + 1)) .* _spow(z, g + 1) ./ (2(g + 1))
    xs = 4p .* _smul(z, u)
    R = zeros(n)
    xj = copy(one_)
    cj = zeros(n)
    jA = Jfix[1] < 0 ? min(jm, N) : min(jm, Jfix[1])
    tmp = zeros(n)
    for j in 0:jA
        if j == 0
            cj .= one_ .+ q
        else
            _smul!(cj, q, xj)
        end
        R .+= _smul!(tmp, cj, view(T, 1, j + 1, 1, 1:n))
        _smul!(t1, xj, xs); xj .= t1
    end
    RA = _smul(PA, R)
    est = max(est, _stail(RA, 0.5))
    return RA, B, C, est
end

"""
    ContactExpansion

Per-light-curve expansions for `I = Σ c_i (1 - r²)^γ_i`: piece bounds in z, the
regular parts of pieces B and C summed over the exponents, and the singular parts and
piece A per exponent.
"""
struct ContactExpansion
    p::Float64
    bounds::Matrix{Float64}      # 3 × 2: z ranges of pieces A, B, C
    REG::Matrix{Float64}         # 2 × (N+1)
    SA::Matrix{Float64}          # nt × (N+1)
    SB::Array{Float64,3}         # nt × 2 × (N+1)
    SC::Array{Float64,3}
    m::Vector{Float64}
    est::Float64
    c0::Float64                  # coefficients of the integer exponents 0 and 1 (closed form)
    c1::Float64
end

function contact_build(p::Float64, γ::Vector{Float64}, c::Vector{Float64}, tabs::Vector{LensTable{Float64}},
                       z_lo::Float64, N::Int = CONTACT_NS)
    nt = length(γ)
    n = N + 1
    bounds = zeros(3, 2)
    zh = sqrt((1 - p) * (1 + p))         # k = 1/2
    bounds[1, :] .= (zh, 1 + p)
    bounds[2, :] .= (1 - p, zh)
    zl = max(z_lo, sqrt(1 + 8p^2) - 3p, p + 1e-12)
    bounds[3, :] .= (zl, 1 - p)
    xl = (1 - (zl - p)) * (1 + (zl - p))
    umax_c = max((1 - (zl + p)) * (1 + (zl + p)) / xl, 1e-3)
    REG = zeros(2, n)
    SA = zeros(nt, n)
    SB = zeros(nt, 2, n)
    SC = zeros(nt, 2, n)
    ms = zeros(nt)
    est = 0.0
    for i in 1:nt
        g = γ[i]
        RA, Bp, Cp, e = contact_pieces(g, p, tabs[i], N, umax_c)
        est = isnan(e) || isnan(est) ? NaN : max(est, e)
        ci = c[i]
        SA[i, :] .= ci .* RA
        REG[1, :] .+= ci .* view(Bp, 1, :)
        REG[2, :] .+= ci .* view(Cp, 1, :)
        SB[i, 1, :] .= ci .* view(Bp, 2, :)
        SB[i, 2, :] .= ci .* view(Bp, 3, :)
        SC[i, 1, :] .= ci .* view(Cp, 2, :)
        SC[i, 2, :] .= ci .* view(Cp, 3, :)
        ms[i] = g + 1.5
    end
    ContactExpansion(p, bounds, REG, SA, SB, SC, ms, est, 0.0, 0.0)
end

# blocked integrals of the integer exponents 0 and 1 in closed form (Mandel & Agol 2002):
# P_0 = π λ_e (the blocked area), P_1 = π (λ_e - η_d)
function _P01(z, p)
    if z + p <= 1
        lex = p * p
        edx = p * p / 2 * (p * p + 2z * z)
    else
        lex, edx = _limb_terms(z, p)
    end
    return π * lex, π * (lex - edx)
end

@inline function _hornerv(c::AbstractVector, x)
    r = 0.0
    @inbounds for i in length(c):-1:1
        r = muladd(r, x, c[i])
    end
    r
end

# x^e from sqrt(x), x^(1/4) and ln x: quarter-integer e without exp
@inline function _pw(x, sx, qx, lx, e)
    e4 = 4e
    if e4 == floor(e4) && e4 < 24
        n4 = Int(e4)
        r = 1.0
        for _ in 1:(n4 >> 2)
            r *= x
        end
        rem = n4 & 3
        rem == 1 && (r *= qx)
        rem == 2 && (r *= sx)
        rem == 3 && (r *= sx * qx)
        return r
    end
    exp(e * lx)
end

"""
    (ce::ContactExpansion)(z)

Blocked flux `Σ c_i P_γi(z)` for a sample inside one of the three pieces; NaN elsewhere.
"""
function (ce::ContactExpansion)(zj::Float64)
    r = _contact_eval(ce, zj)
    if !isnan(r) && (ce.c0 != 0 || ce.c1 != 0)
        P0, P1 = _P01(zj, ce.p)
        r += ce.c0 * P0 + ce.c1 * P1
    end
    r
end

function _contact_eval(ce::ContactExpansion, zj::Float64)
    p = ce.p
    bd = ce.bounds
    nt = length(ce.m)
    if bd[1, 1] <= zj < bd[1, 2]
        zmp = zj - p
        k = (1 - zmp) * (1 + zmp) / (4zj * p)
        r = 0.0
        if k > 0
            lk = log(k); sk = sqrt(k); qk = sqrt(sk)
            for i in 1:nt
                r += _pw(k, sk, qk, lk, ce.m[i]) * _hornerv(view(ce.SA, i, :), k)
            end
        end
        return r
    elseif bd[2, 1] <= zj < bd[2, 2]
        zpp = zj + p
        y = (zpp - 1) * (zpp + 1) / (4zj * p)
        r = _hornerv(view(ce.REG, 1, :), y)
        if y > 0
            ly = log(y); sy = sqrt(y); qy = sqrt(sy)
            for i in 1:nt
                r += _pw(y, sy, qy, ly, ce.m[i]) * (ly * _hornerv(view(ce.SB, i, 1, :), y) + _hornerv(view(ce.SB, i, 2, :), y))
            end
        end
        return r
    elseif bd[3, 1] <= zj < bd[3, 2] && bd[3, 2] > bd[3, 1]
        zmp = zj - p
        zpp = zj + p
        y = (1 - zpp) * (1 + zpp) / ((1 - zmp) * (1 + zmp))
        r = _hornerv(view(ce.REG, 2, :), y)
        if y > 0
            ly = log(y); sy = sqrt(y); qy = sqrt(sy)
            for i in 1:nt
                r += _pw(y, sy, qy, ly, ce.m[i]) * (ly * _hornerv(view(ce.SC, i, 1, :), y) + _hornerv(view(ce.SC, i, 2, :), y))
            end
        end
        return r
    end
    return NaN
end
