# Flux for power-law intensities where the occultor touches or crosses the limb
# (or where the interior series is not used), by Green's theorem.
#
# With the radial primitive F_γ(r) = ∫_0^r (1 - s²)^γ s ds = (1 - (1 - r²)^(γ+1)) / (2(γ+1)),
# the field (F(r)/r²)(x, y) has divergence (1 - r²)^γ and is smooth at r = 0. The
# blocked flux is its outward flux through the boundary of the occulted region:
#   * stellar-limb arc inside the occultor (r = 1): F(1) 2θ_s,
#     cos θ_s = (1 + z² - p²)/(2z);
#   * occultor arc inside the star, points (z + p cos t, p sin t), cos t < c = (1 - z² - p²)/(2zp):
#     2p ∫_{t0}^{π} (F(r)/r²)(p + z cos t) dt,  t0 = arccos c.
# Composite Gauss-Legendre quadrature on segments that grow geometrically away from
# the nearest singularity of (1 - r²)^(γ+1); the first segment uses t = t0 + L u²
# to remove the (t - t0)^(γ+1) endpoint behaviour. Generic in the number type.

"""
    gauss_legendre(n, T = Float64) -> (u, w)

Gauss-Legendre nodes and weights on [0, 1], computed by Newton's method in type `T`
(so that BigFloat quadrature gets nodes of full precision).
"""
function gauss_legendre(n::Integer, ::Type{T} = Float64) where {T<:AbstractFloat}
    x = zeros(T, n)
    w = zeros(T, n)
    m = (n + 1) ÷ 2
    for i in 1:m
        z = T(cospi((i - big(0.25)) / (n + big(0.5))))
        pp = zero(T)
        for _ in 1:100
            p1, p2 = one(T), zero(T)
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end
            pp = n * (z * p1 - p2) / (z * z - 1)
            dz = p1 / pp
            z -= dz
            abs(dz) <= 2eps(T) * abs(z) && break
        end
        x[i] = -z
        x[n+1-i] = z
        w[i] = 2 / ((1 - z * z) * pp * pp)
        w[n+1-i] = w[i]
    end
    return (x .+ 1) ./ 2, w ./ 2
end

"""
    gauss_jacobi01(n, β, T = Float64) -> (u, w)

Gauss-Jacobi rule on [0, 1] for the weight `u^β` (β > -1):
`∫_0^1 u^β φ(u) du ≈ Σ w_i φ(u_i)`, exact for polynomials φ of degree < 2n.
Golub-Welsch in Float64 for the starting nodes, Newton's method in type `T`.
"""
function gauss_jacobi01(n::Integer, β::Real, ::Type{T} = Float64) where {T<:AbstractFloat}
    a = 0.0
    b = Float64(β)
    # Golub-Welsch (Float64) for the initial nodes on [-1, 1], weight (1 + x)^b
    d = [k == 0 ? (b - a) / (a + b + 2) : (b^2 - a^2) / ((2k + a + b) * (2k + a + b + 2)) for k in 0:n-1]
    e = [sqrt(4k * (k + a) * (k + b) * (k + a + b) / ((2k + a + b)^2 * (2k + a + b + 1) * (2k + a + b - 1))) for k in 1:n-1]
    x0 = eigvals(SymTridiagonal(d, e))
    aT = zero(T)
    bT = T(β)
    x = T.(x0)
    dP = zeros(T, n)
    for i in 1:n
        xi = x[i]
        for it in 1:20
            P, Pm = _jacobiP(n, aT, bT, xi)
            dpi = (n * ((aT - bT) - (2n + aT + bT) * xi) * P + 2 * (n + aT) * (n + bT) * Pm) /
                  ((2n + aT + bT) * (1 - xi * xi))
            dx = P / dpi
            xi -= dx
            abs(dx) <= 4eps(T) && (it > 1) && break
        end
        P, Pm = _jacobiP(n, aT, bT, xi)
        dP[i] = (n * ((aT - bT) - (2n + aT + bT) * xi) * P + 2 * (n + aT) * (n + bT) * Pm) /
                ((2n + aT + bT) * (1 - xi * xi))
        x[i] = xi
    end
    lc = loggamma(n + aT + 1) + loggamma(n + bT + 1) - loggamma(n + aT + bT + 1) - loggamma(T(n) + 1) +
         (aT + bT + 1) * log(T(2))
    w = [exp(lc) / ((1 - x[i]^2) * dP[i]^2) for i in 1:n]
    # [-1, 1], (1 + x)^β  ->  [0, 1], u^β:  u = (1 + x)/2, factor 2^(-β-1)
    return (x .+ 1) ./ 2, w .* exp(-(bT + 1) * log(T(2)))
end

# P_n^(a,b)(x) and P_{n-1}^(a,b)(x) by the three-term recurrence
function _jacobiP(n, a, b, x)
    P0 = one(x)
    n == 0 && return P0, zero(x)
    P1 = ((a + b + 2) * x + (a - b)) / 2
    for k in 1:n-1
        c1 = 2 * (k + 1) * (k + a + b + 1) * (2k + a + b)
        c2 = (2k + a + b + 1) * ((2k + a + b + 2) * (2k + a + b) * x + a^2 - b^2)
        c3 = 2 * (k + a) * (k + b) * (2k + a + b + 2)
        P0, P1 = P1, (c2 * P1 - c3 * P0) / c1
    end
    return P1, P0
end

"Gauss-Legendre (and Gauss-Jacobi) nodes per segment of the limb arc integral (Float64)."
const N_NODES = 12
const GL_U, GL_W = gauss_legendre(N_NODES)

_nnodes(::Type{Float64}) = N_NODES
_nnodes(::Type{T}) where {T<:AbstractFloat} = max(N_NODES, ceil(Int, N_NODES * precision(T) / 53 * 1.2))

"""
    QuadRules

Quadrature rules of the Green's-theorem integral for one set of exponents: Gauss-Legendre
nodes on [0, 1], and for each non-integer exponent γ a Gauss-Jacobi rule with weight
`u^(γ+1)` that integrates the endpoint branch `(t - t0)^(γ+1)` of the first segment exactly.
"""
struct QuadRules{F<:AbstractFloat}
    gu::Vector{F}
    gw::Vector{F}
    ju::Vector{Vector{F}}
    jw::Vector{Vector{F}}
end

const _GJ_CACHE = Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}()
const _GJ_LOCK = ReentrantLock()

function quad_rules(γ::AbstractVector, ::Type{F} = Float64) where {F<:AbstractFloat}
    n = _nnodes(F)
    gu, gw = F === Float64 ? (GL_U, GL_W) : gauss_legendre(n, F)
    ju = Vector{F}[]
    jw = Vector{F}[]
    for g in γ
        gv = _val(g)
        if isinteger(gv)
            push!(ju, F[]); push!(jw, F[])
        elseif F === Float64
            r = lock(_GJ_LOCK) do
                length(_GJ_CACHE) > 1024 && empty!(_GJ_CACHE)
                get!(() -> gauss_jacobi01(n, Float64(gv) + 1, Float64), _GJ_CACHE, Float64(gv))
            end
            push!(ju, r[1]); push!(jw, r[2])
        else
            r = gauss_jacobi01(n, F(gv) + 1, F)
            push!(ju, r[1]); push!(jw, r[2])
        end
    end
    QuadRules{F}(gu, gw, ju, jw)
end

"""
    blocked_quadrature(z, p, γ, c[, rules])

`Σ_i c_i ∫ (1 - r²)^γ_i dA` over the part of the stellar disk covered by an
occultor of radius `p` at separation `z` (any geometry), by Green's theorem.

The occultor arc is integrated with composite Gauss rules on segments that grow
geometrically away from the nearest singularity of `(1 - r²)^(γ+1)`. On the limb, the
first segment starts at the branch point `t0` (where the rim crosses the limb): there the
integrand is split into a regular part (Gauss-Legendre) and `(t - t0)^(γ+1)` times an
analytic function (Gauss-Jacobi with that weight), so that every segment converges
geometrically, for every exponent.
"""
function blocked_quadrature(z, p, γ::AbstractVector, c::AbstractVector,
                            rules = quad_rules(γ, _float(promote_type(typeof(float(z)), typeof(float(p)), eltype(γ), eltype(c)))))
    T = promote_type(typeof(float(z)), typeof(float(p)), eltype(γ), eltype(c))
    gu, gw = rules.gu, rules.gw
    nt = length(γ)
    tot = zero(T)
    piT = T(π)
    # stellar-limb arc inside the occultor
    if z > 0
        om = ((z - 1) + p) * ((1 - z) + p) / (2z)
        op = ((1 - p) + z) * (1 + z + p) / (2z)
        th = om <= 0 ? zero(T) : op <= 0 ? piT : T(_acos_acc(om < op ? 1 - om : op - 1, om, op))
    else
        th = p >= 1 ? piT : zero(T)
    end
    if th > 0
        for i in 1:nt
            tot += c[i] * th / (γ[i] + 1)
        end
    end
    # occultor arc inside the star
    if z > 0 && p > 0
        omc = ((z - 1) + p) * (z + p + 1) / (2z * p)
        opc = ((1 - z) + p) * ((1 - p) + z) / (2z * p)
        arc = opc > 0
    else
        omc = p < 1 ? zero(T) : 2one(T)
        opc = p < 1 ? 2one(T) : zero(T)
        arc = p < 1
    end
    arc || return tot
    t0 = omc <= 0 ? zero(T) : T(_acos_acc(omc < opc ? 1 - omc : opc - 1, omc, opc))
    zp = z * p
    omz2p2 = ((1 - z) - p) * (1 + z + p) + 2zp           # 1 - z² - p²
    limb = t0 > 0
    if limb
        # first segment up to the mirror branch point -t0, and away from t = π (where r → 0 if z = p)
        scale = min(t0, (piT - t0) / 2)
    else
        ymin = ((1 - z) - p) * (1 + z + p)
        scale = zp > 0 ? sqrt(max(ymin, zero(T)) / max(zp, floatmin(_float(zp)))) : piT
    end
    scale = min(max(scale, T(1e-12)), piT - t0)
    a = t0
    b = t0 + scale
    first = limb
    st0, ct0 = sincos(t0)
    for _ in 1:200
        L = b - a
        for n in eachindex(gu)
            t = a + L * gu[n]
            jac = L * gw[n]
            if limb
                # one sincos per node: h = (t - t0)/2, sin((t + t0)/2) = sin(h + t0), cos t = cos(2h + t0)
                sh, ch = sincos((t - t0) / 2)
                y = 4zp * (sh * ct0 + ch * st0) * sh                     # 1 - r², accurate near t0
                ct = (1 - 2sh * sh) * ct0 - 2sh * ch * st0
            else
                ct = cos(t)
                y = omz2p2 - 2zp * ct
            end
            y = clamp(y, zero(T), one(T))
            r2 = y > 0.5 ? max(1 - y, zero(T)) : clamp(z * z + p * p + 2zp * ct, zero(T), one(T))
            geo = 2p * (p + z * ct) * jac
            for i in 1:nt
                if first && !isempty(rules.ju[i])
                    tot += c[i] * geo / (2 * (γ[i] + 1) * r2)                # regular part
                else
                    tot += c[i] * _hval(γ[i], y, r2) * geo
                end
            end
        end
        if first
            # singular parts: -(t - t0)^(γ+1) φ(t), φ = (y/(t - t0))^(γ+1) (p + z cos t) 2p / (2(γ+1) r²)
            for i in 1:nt
                isempty(rules.ju[i]) && continue
                g = γ[i]
                ju = rules.ju[i]
                jw = rules.jw[i]
                s = zero(T)
                for n in eachindex(ju)
                    δ = L * ju[n]
                    sh, ch = sincos(δ / 2)
                    ct = (1 - 2sh * sh) * ct0 - 2sh * ch * st0             # cos(t0 + δ)
                    sinc = iszero(δ) ? one(δ) : 2sh / δ
                    yd = 2zp * (sh * ct0 + ch * st0) * sinc                 # y / (t - t0)
                    y = yd * δ
                    r2 = y > 0.5 ? max(1 - y, zero(T)) : clamp(z * z + p * p + 2zp * ct, zero(T), one(T))
                    v = jw[n] * _powq(yd, g + 1) * 2p * (p + z * ct) / (2 * (g + 1) * r2)
                    # the rule carries u^(γ0+1), γ0 the primal value; dual parts need u^(γ-γ0)
                    _isconst(g, _val(g)) || (v *= exp((g - _val(g)) * log(ju[n])))
                    s += v
                end
                tot -= c[i] * s * _powq(L, g + 2)
            end
        end
        b >= piT && break
        first = false
        a = b
        b = min(t0 + 3 * (b - t0), piT)
        piT - b < (b - a) / 4 && (b = piT)
    end
    return tot
end

# 2 sin(δ/2)/δ (accurate for small δ: no cancellation in sin(h)/h)
@inline function _sinc_half(δ)
    h = δ / 2
    iszero(h) ? one(h) : sin(h) / h
end

# F_γ(r)/r² = (1 - y^(γ+1)) / (2(γ+1) r²), y = 1 - r², stable for r → 0 and y → 0
@inline function _hval(γ, y, r2)
    _isconst(γ, 0) && return one(y) / 2 + zero(γ)
    _isconst(γ, 1) && return (1 + y) / 4 + zero(γ)
    r2 < sqrt(eps(_float(typeof(r2)))) / 2 && return (1 - γ * r2 / 2) / 2
    r2 < 0.05 && return -expm1((γ + 1) * log1p(-r2)) / (2 * (γ + 1) * r2)
    _isconst(γ, 0.5) && return (1 - y * sqrt(y)) / (3r2) + zero(γ)
    y <= 0 && return 1 / (2 * (γ + 1) * r2)
    return (1 - _powq(y, γ + 1)) / (2 * (γ + 1) * r2)
end
