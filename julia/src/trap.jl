# Limb flux by a fixed-node sinh-trapezoid rule, and Chebyshev interpolants built from it
# (paper Section 5.3; port of the Python modules _trap and _interp).
#
# On the limb (1 - p < z < 1 + p, z > p) the blocked flux of μ^α is the complete integral
#     P = -2 √k x^(γ+1) / (2γ+2) J(y; x),   y = 1 - k,   γ = α/2,
#     J(y; x) = ∫_0^∞ cosh(s)^-(α+3) (1 + y sinh² s)^(-1/2) [1 + c0 cosh² s / (cosh² s - x)] ds,
# x = 1 - (z-p)², k = x/(4zp), c0 = p² - z². Its integrand is analytic in |Im s| < π/2 for
# every modulus, so the trapezoid rule on s = n h converges like exp(-π²/h) uniformly up
# to the contacts; the error of the poles cosh² s = x is subtracted in closed form. The
# nodes do not depend on z, so the exponents only enter precomputed weights.
# Inside the disk near the limb (z_c < z < 1 - p, y < 0) the flux is Re J(y - i0), taken
# along the contour tan θ = sinh(s + iγc).
# Connection formula: J = A(y) + y^ν B(y), ν = γ + 3/2, B = -k^-ν J(k; -x y/k) / sin(πν)
# (odd α: y^ν ln y and (-1)^ν/π); A, B and P/k^ν are interpolated per light curve.

const TRAP_H = 0.2
const TRAP_TAIL = 1e-16
"Largest radius ratio for the trapezoid rules (beyond, the pole cosh² s = x nears the real axis)."
const TRAP_PMAX = 0.3
const BAND_H = 0.1
const BAND_GAMMAS = (0.5, 0.65, 0.8)
const BAND_POLE_GAP = 0.1
"Limb samples from which :auto uses the interpolants instead of the direct rule."
const INTERP_BREAK_EVEN = 150
const INTERP_NO = 16
const INTERP_NC = 18
const INTERP_NN = 14
const INTERP_HCB = 0.17
const INTERP_LAM_MIN = -1.0
const INTERP_SIN_MIN = 0.02

trap_step(amax) = amax <= 3 ? TRAP_H : (amax <= 6 ? 0.17 : 0.14)
band_step(γc, amax) = (γc <= 0.7 || amax <= 1.6) ? BAND_H : 0.07

function _trap_len(γ, h)
    amin = max(2 * minimum(γ), 0.0)
    S = (log(1 / TRAP_TAIL) + (amin + 3) * log(2.0)) / (amin + 3)
    return ceil(Int, S / h) + 1
end

"Real rule: weights W[n, i] = w_n cosh(s_n)^-(2γ_i + 3), t_n² and cosh²(s_n)."
struct TrapRule
    W::Matrix{Float64}
    tt::Vector{Float64}
    b::Vector{Float64}
    h::Float64
end

"Complex-contour rule (band inside the disk): weights w_n b_n^-(γ_i+2) cosh(s_n + iγc)."
struct BandRule
    Wr::Matrix{Float64}
    Wi::Matrix{Float64}
    ttr::Vector{Float64}
    tti::Vector{Float64}
    br::Vector{Float64}
    bi::Vector{Float64}
    h::Float64
    γc::Float64
end

const _NO_TRAPRULE = TrapRule(zeros(1, 0), zeros(1), ones(1), TRAP_H)
const _NO_BANDRULE = BandRule(zeros(1, 0), zeros(1, 0), zeros(1), zeros(1), ones(1), zeros(1), BAND_H, 0.5)

const _TRAP_CACHE = Dict{Tuple{Vector{Float64},Float64},TrapRule}()
const _BAND_CACHE = Dict{Tuple{Vector{Float64},Float64},BandRule}()
const _TRAP_LOCK = ReentrantLock()

function trap_rule(γ::Vector{Float64}, h::Float64 = trap_step(2 * maximum(γ)))
    lock(_TRAP_LOCK) do
        get!(_TRAP_CACHE, (copy(γ), h)) do
            length(_TRAP_CACHE) > 1024 && empty!(_TRAP_CACHE)
            n = _trap_len(γ, h)
            s = h .* (0:n-1)
            w = fill(h, n); w[1] = h / 2
            lc = log.(cosh.(s))
            W = [w[j] * exp(-(2γ[i] + 3) * lc[j]) for j in 1:n, i in eachindex(γ)]
            TrapRule(W, sinh.(s) .^ 2, cosh.(s) .^ 2, h)
        end
    end
end

function band_rule(γ::Vector{Float64}, γc::Float64)
    lock(_TRAP_LOCK) do
        get!(_BAND_CACHE, (copy(γ), γc)) do
            length(_BAND_CACHE) > 1024 && empty!(_BAND_CACHE)
            h = band_step(γc, 2 * maximum(γ))
            n = _trap_len(γ, h)
            s = h .* (0:n-1) .+ im * γc
            w = fill(h, n); w[1] = h / 2
            t = sinh.(s)
            b = cosh.(s) .^ 2
            lb = log.(b)
            wd = w .* cosh.(s)
            W = [exp(-(γ[i] + 2) * lb[j]) * wd[j] for j in 1:n, i in eachindex(γ)]
            tt = t .^ 2
            BandRule(real.(W), imag.(W), real.(tt), imag.(tt), real.(b), imag.(b), h, γc)
        end
    end
end

"""
Contour angle for the band z_lo < z < 1 - p, or -1 if none applies: over the band the pole
angle a = acos √x runs from a(z_lo) to a(1 - p), and the contour keeps BAND_POLE_GAP from it.
"""
function band_gamma(p::Float64, z_lo::Float64)
    (0 < p < 0.5 && z_lo > p) || return -1.0
    a_lo = acos(sqrt(1 - (z_lo - p)^2))
    a_hi = acos(sqrt(4p * (1 - p)))
    for g in BAND_GAMMAS
        (g < a_lo - BAND_POLE_GAP || g > a_hi + BAND_POLE_GAP) && return g
    end
    return -1.0
end

# ---------------------------------------------------------------- integral cores
# out[i] = J(λ; x, c0) for the exponents γ[i] (real rule); g: scratch of the node count
function _core_real!(out, λ, x, c0, γ, R::TrapRule, g)
    nn = length(R.tt)
    tt, b = R.tt, R.b
    @inbounds @simd for n in 1:nn
        bn = b[n]
        g[n] = @fastmath (1 + c0 * bn / (bn - x)) / sqrt(1 + λ * tt[n])
    end
    W = R.W
    @inbounds for i in eachindex(γ)
        s = 0.0
        @simd for n in 1:nn
            s = muladd(W[n, i], g[n], s)
        end
        out[i] = s
    end
    d = 1 - λ * (1 - x)
    if 0.02 < x < 1 && d > 0
        a = acos(sqrt(x))
        e = exp(-2π * a / R.h)
        base = π * c0 * sqrt(x) / sqrt((1 - x) * d) * e / (1 - e)
        lx = log(x)
        @inbounds for i in eachindex(γ)
            out[i] -= base * exp(-(γ[i] + 1.5) * lx)
        end
    end
    out
end

# out[i] = Re J(λ - i0; x, c0), λ < 0 (complex-contour rule); gr, gi: scratch
function _core_band!(out, λ, x, c0, γ, B::BandRule, gr, gi)
    nn = length(B.ttr)
    @inbounds @simd for n in 1:nn
        u = 1 + λ * B.ttr[n]
        v = λ * B.tti[n]
        md = sqrt(u * u + v * v)
        a = sqrt(0.5 * (md + abs(u)))
        d = 0.5 * v / a
        sre = ifelse(u >= 0, a, abs(d))
        sim = ifelse(u >= 0, d, ifelse(v >= 0, a, -a))
        rr = sre / md
        ri = -sim / md
        dr = B.br[n] - x
        di = B.bi[n]
        idn = 1 / (dr * dr + di * di)
        qr = 1 + c0 * (B.br[n] * dr + di * di) * idn
        qi = c0 * (di * dr - B.br[n] * di) * idn
        gr[n] = rr * qr - ri * qi
        gi[n] = rr * qi + ri * qr
    end
    @inbounds for i in eachindex(γ)
        s = 0.0
        @simd for n in 1:nn
            s = muladd(B.Wr[n, i], gr[n], muladd(-B.Wi[n, i], gi[n], s))
        end
        out[i] = s
    end
    a = acos(sqrt(x))
    om = 1 - x
    e1 = exp(-2π * (a - B.γc) / B.h)
    e2 = exp(-2π * (a + B.γc) / B.h)
    base = 0.5 * π * c0 * sqrt(x) / sqrt(om * (1 - λ * om)) * (e1 / (1 - e1) + e2 / (1 - e2))
    lx = log(x)
    @inbounds for i in eachindex(γ)
        out[i] -= base * exp(-(γ[i] + 1.5) * lx)
    end
    out
end

# ---------------------------------------------------------------- the limb solution
"""
    TrapLimb

Limb solution of an [`Occultor`](@ref) by the fixed-node rule (`interp = false`) or its
Chebyshev interpolants (`interp = true`) for `p < 0.3`.
"""
struct TrapLimb
    interp::Bool
    p::Float64
    γ::Vector{Float64}           # exponents handled here (direct: non-integer; interpolants: all)
    c::Vector{Float64}
    c0::Float64                  # closed-form exponents 0 and 1 (direct rule only)
    c1::Float64
    R::TrapRule
    band::Bool
    B::BandRule
    lamlo::Float64               # lower end of the inside interpolant (0: none)
    EO::Matrix{Float64}          # Chebyshev coefficients (coefficient × exponent)
    AC::Vector{Float64}          # regular parts summed over the exponents (Σ c_i A_i)
    BC::Matrix{Float64}
    AN::Vector{Float64}
    BN::Matrix{Float64}
    nu::Vector{Float64}
    fac::Vector{Float64}         # cos(πν) or (-1)^ν for the inside piece
    isint::Vector{Bool}
end

const _NO_TRAP = TrapLimb(false, 0.0, Float64[], Float64[], 0.0, 0.0, _NO_TRAPRULE, false, _NO_BANDRULE, 0.0,
                          zeros(0, 0), Float64[], zeros(0, 0), Float64[], zeros(0, 0), Float64[], Float64[], Bool[])

"True if no exponent is within ±SIN_MIN (in sin πν) of an odd integer α without being one."
function interp_supported(γ)
    for g in γ
        g == floor(g) && continue
        ν = g + 1.5
        ν != round(ν) && abs(sin(π * ν)) < INTERP_SIN_MIN && return false
    end
    true
end

_cheb_nodes(n) = [cos(π * (j + 0.5) / n) for j in 0:n-1]
function _dct_matrix(n)
    D = [2 / n * cos(π * m * (j + 0.5) / n) for m in 0:n-1, j in 0:n-1]
    D[1, :] ./= 2
    D
end
const _DCT = Dict(n => _dct_matrix(n) for n in (INTERP_NO, INTERP_NC, INTERP_NN))
const _CNODES = Dict(n => _cheb_nodes(n) for n in (INTERP_NO, INTERP_NC, INTERP_NN))
"Chebyshev coefficients from values at the first-kind nodes."
_cheb_coef(vals) = _DCT[length(vals)] * vals
_z_of_k(k, p) = (q = p * (1 - 2k); q + sqrt(q * q + 1 - p * p))

function TrapLimb(p::Float64, γall::Vector{Float64}, call::Vector{Float64}, zc::Float64, interp::Bool)
    if interp
        γ, c, c0, c1 = copy(γall), copy(call), 0.0, 0.0
    else
        ib = [i for i in eachindex(γall) if !(γall[i] == 0 || γall[i] == 1)]
        γ, c = γall[ib], call[ib]
        c0 = sum((call[i] for i in eachindex(γall) if γall[i] == 0); init = 0.0)
        c1 = sum((call[i] for i in eachindex(γall) if γall[i] == 1); init = 0.0)
    end
    nt = length(γ)
    gc = band_gamma(p, zc > 0 ? zc : p)
    band = gc > 0
    R = nt > 0 ? trap_rule(γ) : _NO_TRAPRULE
    B = band && nt > 0 ? band_rule(γ, gc) : _NO_BANDRULE
    nu = γ .+ 1.5
    isint = [ν == round(ν) for ν in nu]
    sg = [isint[i] ? (iseven(round(Int, nu[i])) ? 1.0 : -1.0) / π : 1 / sin(π * nu[i]) for i in 1:nt]
    fac = [isint[i] ? (iseven(round(Int, nu[i])) ? 1.0 : -1.0) : cos(π * nu[i]) for i in 1:nt]
    lamlo = 0.0
    EO = zeros(INTERP_NO, nt); AC = zeros(INTERP_NC); BC = zeros(INTERP_NC, nt); AN = zeros(INTERP_NN); BN = zeros(INTERP_NN, nt)
    if interp && nt > 0
        if band && zc > p
            lamlo = ((zc - 1) + p) * (zc + p + 1) / (4zc * p)
            lamlo = min(max(lamlo, INTERP_LAM_MIN, 0.3 * (1 - 0.25 / (p * p))), 0.0)
        end
        Rc = trap_rule(γ, INTERP_HCB)
        g = zeros(max(length(R.tt), length(Rc.tt), length(B.ttr)))
        g2 = zeros(length(B.ttr))
        I = zeros(nt); Jc = zeros(nt)
        # first contact: E(k) = P / k^ν
        V = zeros(INTERP_NO, nt)
        for (j, t) in enumerate(_CNODES[INTERP_NO])
            k = 0.25 * (1 + t)
            z = _z_of_k(k, p)
            x = 4z * p * k
            _core_real!(I, 1 - k, x, (p - z) * (p + z), γ, R, g)
            for i in 1:nt
                V[j, i] = -2 * (4z * p)^(γ[i] + 1) / (2γ[i] + 2) * I[i]
            end
        end
        for i in 1:nt
            EO[:, i] = _cheb_coef(V[:, i])
        end
        # outside second contact: λ in [0, 1/2]
        VA = zeros(INTERP_NC); VB = zeros(INTERP_NC, nt)
        for (j, t) in enumerate(_CNODES[INTERP_NC])
            λ = 0.25 * (1 + t)
            k = 1 - λ
            z = _z_of_k(k, p)
            x = 4z * p * k
            c0z = (p - z) * (p + z)
            _core_real!(I, λ, x, c0z, γ, R, g)
            _core_real!(Jc, k, -x * λ / k, c0z, γ, Rc, g)
            ll = log(λ)
            for i in 1:nt
                φ = -2sqrt(k) * x^(γ[i] + 1) / (2γ[i] + 2)
                Bv = -k^(-nu[i]) * Jc[i] * sg[i]
                S = exp(nu[i] * ll) * (isint[i] ? ll : 1.0)
                VA[j] += c[i] * φ * (I[i] - S * Bv)
                VB[j, i] = φ * Bv
            end
        end
        AC = _cheb_coef(VA)
        for i in 1:nt
            BC[:, i] = _cheb_coef(VB[:, i])
        end
        # inside second contact: λ in [lamlo, 0]
        if lamlo < 0
            VA = zeros(INTERP_NN); VB = zeros(INTERP_NN, nt)
            for (j, t) in enumerate(_CNODES[INTERP_NN])
                λ = 0.5 * lamlo * (1 - t)
                k = 1 - λ
                z = _z_of_k(k, p)
                x = 4z * p * k
                c0z = (p - z) * (p + z)
                _core_band!(I, λ, x, c0z, γ, B, g, g2)
                _core_real!(Jc, k, -x * λ / k, c0z, γ, Rc, g)
                ll = log(-λ)
                for i in 1:nt
                    φ = -2sqrt(k) * x^(γ[i] + 1) / (2γ[i] + 2)
                    Bv = -k^(-nu[i]) * Jc[i] * sg[i]
                    S = fac[i] * exp(nu[i] * ll) * (isint[i] ? ll : 1.0)
                    VA[j] += c[i] * φ * (I[i] - S * Bv)
                    VB[j, i] = φ * Bv
                end
            end
            AN = _cheb_coef(VA)
            for i in 1:nt
                BN[:, i] = _cheb_coef(VB[:, i])
            end
        end
    end
    TrapLimb(interp, p, γ, c, c0, c1, R, band, B, lamlo, EO, AC, BC, AN, BN, nu, fac, isint)
end

@inline function _clenshaw(C::AbstractVector, t)
    b1 = 0.0
    b2 = 0.0
    t2 = 2t
    @inbounds for m in length(C):-1:2
        b1, b2 = muladd(t2, b1, C[m] - b2), b1
    end
    @inbounds muladd(t, b1, C[1] - b2)
end

# blocked flux Σ c_i P_i at z by the interpolants; NaN outside their pieces
function _interp_eval(tl::TrapLimb, z::Float64)
    p = tl.p
    q4 = 4z * p
    zO = sqrt((1 - p) * (1 + p))
    nt = length(tl.γ)
    if zO <= z < 1 + p
        k = ((1 - z) + p) * ((1 + z) - p) / q4
        k > 0 || return 0.0
        t = 4k - 1
        lk = log(k)
        r = 0.0
        @inbounds for i in 1:nt
            r += tl.c[i] * exp(tl.nu[i] * lk) * _clenshaw(view(tl.EO, :, i), t)
        end
        return r
    elseif p < z < zO
        λ = ((z - 1) + p) * (z + p + 1) / q4
        if λ >= 0
            t = 4λ - 1
            r = _clenshaw(tl.AC, t)
            λ > 0 || return r
            ll = log(λ)
            @inbounds for i in 1:nt
                S = exp(tl.nu[i] * ll) * (tl.isint[i] ? ll : 1.0)
                r += tl.c[i] * S * _clenshaw(view(tl.BC, :, i), t)
            end
            return r
        elseif λ >= tl.lamlo && tl.lamlo < 0
            t = 1 - 2λ / tl.lamlo
            r = _clenshaw(tl.AN, t)
            ll = log(-λ)
            @inbounds for i in 1:nt
                S = tl.fac[i] * exp(tl.nu[i] * ll) * (tl.isint[i] ? ll : 1.0)
                r += tl.c[i] * S * _clenshaw(view(tl.BN, :, i), t)
            end
            return r
        end
    end
    return NaN
end

# blocked flux by the direct rule; NaN where it does not apply. I, g, g2: scratch
function _direct_eval(tl::TrapLimb, z::Float64, I, g, g2)
    p = tl.p
    nt = length(tl.γ)
    cross = z >= 1 - p && z > p
    inband = !cross && tl.band && z > p
    (cross || inband) || return NaN
    r = 0.0
    if nt > 0
        x = ((1 - z) + p) * ((1 + z) - p)
        q4 = 4z * p
        λ = ((z - 1) + p) * (z + p + 1) / q4
        c0 = (p - z) * (p + z)
        if cross
            _core_real!(I, max(λ, 0.0), x, c0, tl.γ, tl.R, g)
        else
            _core_band!(I, λ, x, c0, tl.γ, tl.B, g, g2)
        end
        sk = sqrt(x / q4)
        lx = log(x)
        @inbounds for i in 1:nt
            r += tl.c[i] * (-2 / (2tl.γ[i] + 2)) * exp((tl.γ[i] + 1) * lx) * I[i]
        end
        r *= sk
    end
    if tl.c0 != 0 || tl.c1 != 0
        P0, P1 = _P01(z, p)
        r += tl.c0 * P0 + tl.c1 * P1
    end
    r
end

_trap_scratch(tl::TrapLimb) = (zeros(length(tl.γ)), zeros(max(length(tl.R.tt), length(tl.B.ttr))), zeros(length(tl.B.ttr)))

"Blocked flux Σ c_i P_i(z) by the trapezoid limb solution, NaN where it does not apply."
function (tl::TrapLimb)(z::Float64)
    if tl.interp
        r = _interp_eval(tl, z)
        isnan(r) || return r
    end
    _direct_eval(tl, z, _trap_scratch(tl)...)
end

# Clenshaw sums over a batch of samples (vectorised over the samples): acc[j] += f[j] Σ C_m T_m(t[j])
function _clen_batch!(acc, C::AbstractVector, t, f, b1, b2)
    n = length(t)
    @inbounds @simd for j in 1:n
        b1[j] = 0.0
        b2[j] = 0.0
    end
    @inbounds for m in length(C):-1:2
        cm = C[m]
        @simd for j in 1:n
            tmp = muladd(2t[j], b1[j], cm - b2[j])
            b2[j] = b1[j]
            b1[j] = tmp
        end
    end
    c1 = C[1]
    @inbounds @simd for j in 1:n
        acc[j] = muladd(f[j], muladd(t[j], b1[j], c1 - b2[j]), acc[j])
    end
    acc
end

# interpolants for a batch: out[j] = blocked flux, NaN outside the pieces
function _interp_batch!(out, tl::TrapLimb, zs)
    p = tl.p
    zO = sqrt((1 - p) * (1 + p))
    nt = length(tl.γ)
    reg = Vector{Int8}(undef, length(zs))
    @inbounds for j in eachindex(zs)
        z = zs[j]
        q4 = 4z * p
        r = Int8(-1)
        if zO <= z < 1 + p
            r = Int8(0)
        elseif p < z < zO
            λ = ((z - 1) + p) * (z + p + 1) / q4
            r = λ >= 0 ? Int8(1) : (λ >= tl.lamlo && tl.lamlo < 0 ? Int8(2) : Int8(-1))
        end
        reg[j] = r
        r < 0 && (out[j] = NaN)
    end
    for R in Int8(0):Int8(2)
        ix = findall(==(R), reg)
        n = length(ix)
        n == 0 && continue
        t = Vector{Float64}(undef, n); ly = similar(t); pos = similar(t)
        @inbounds for (q, j) in enumerate(ix)
            z = zs[j]
            q4 = 4z * p
            if R == 0
                y = ((1 - z) + p) * ((1 + z) - p) / q4
                t[q] = 4y - 1
            else
                y = ((z - 1) + p) * (z + p + 1) / q4
                t[q] = R == 1 ? 4y - 1 : 1 - 2y / tl.lamlo
                y = abs(y)
            end
            pos[q] = y > 0 ? 1.0 : 0.0
            ly[q] = log(max(y, 1e-300))
        end
        acc = zeros(n); b1 = similar(t); b2 = similar(t); f = similar(t)
        if R > 0
            fill!(f, 1.0)
            _clen_batch!(acc, R == 1 ? tl.AC : tl.AN, t, f, b1, b2)
        end
        for i in 1:nt
            ν = tl.nu[i]
            fc = tl.c[i] * (R == 2 ? tl.fac[i] : 1.0)
            li = R > 0 && tl.isint[i]
            @inbounds @simd for q in 1:n
                f[q] = fc * pos[q] * _vexp(ν * ly[q]) * ifelse(li, ly[q], 1.0)
            end
            C = R == 0 ? view(tl.EO, :, i) : (R == 1 ? view(tl.BC, :, i) : view(tl.BN, :, i))
            _clen_batch!(acc, C, t, f, b1, b2)
        end
        @inbounds for (q, j) in enumerate(ix)
            out[j] = acc[q]
        end
    end
    out
end

# batch: out[j] = blocked flux at zs[j] (NaN where the rules do not apply)
function trap_blocked!(out, tl::TrapLimb, zs)
    if tl.interp
        _interp_batch!(out, tl, zs)
    else
        fill!(out, NaN)
    end
    I, g, g2 = _trap_scratch(tl)
    @inbounds for j in eachindex(zs)
        isnan(out[j]) && (out[j] = _direct_eval(tl, zs[j], I, g, g2))
    end
    out
end
