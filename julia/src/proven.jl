# Contact expansions with a proven flux-error bound (paper, Theorem 3 and Appendix C).
#
# The limb region of a non-quadratic power law is split into pieces in the
# hypergeometric argument (k for the limb crossing, κ = 1/k inside the disk):
#
#   A   first contact     k ∈ [0, 1/4]:          P = k^m RA(k)
#   MO  outer middle      k ∈ [1/4, 3/4]:        P = Σ a_n (k - k0)^n
#   B   second contact    y = 1-k ∈ [0, 1/4]:    P = R(y) + y^m [ln y SL(y) + SC(y)]
#   C   second contact    y' = 1-κ ≤ 1/4:        same form in y'
#   MI  inner middle      κ ∈ [κ_c, 3/4]:        P = Σ a_n (κ - κ0)^n
#
# For each piece the error of the evaluated polynomial is bounded by (1) the
# j-remainder of the hypergeometric representation (Lemma 2), and (2) the
# n-truncation: exact coefficients up to NB, and beyond NB Cauchy's estimate
# |f_n| ≤ M(ρ) ρ^-n on a circle (Lemmas 3-6). The polynomial degree of each piece is
# the smallest that meets the flux tolerance. This is a line-by-line port of the
# Python implementation (pizzetti/_proven.py); all bounds hold in exact arithmetic,
# with a relative safety margin of 1e-6 for the rounding of the bounds themselves.
# Kinds: 0 = A, 1 = B, 2 = C, 3 = MO, 4 = MI.

const PROVEN_NB = 64
const PROVEN_KS = 1024
const PROVEN_KA = 0.25
const PROVEN_YB = 0.25
const PROVEN_NMO = 2
const PROVEN_WMI = 0.07
const PROVEN_WPAD = 1.0 + 1e-9
const PROVEN_SAFETY = 1.0 + 1e-6

# Python-style (0-based fam, j, slot, n) access to a lens table
@inline _T0(T, fam, j, slot, n) = @inbounds T[fam+1, j+1, slot+1, n+1]

_pparams(g, j, fam) = fam == 0 ? (0.5, 0.5, g + 2.5 + j) : (-(g + 1.0 + j), 0.5, 1.0)

"Exact coefficients (to nmax) of the tabulated series (fam, j, slot), continued by their recurrences."
function _coef_seq(g, j, fam, slot, T, nmax)
    a, b, c = _pparams(g, j, fam)
    m = c - a - b
    mi = floor(m + 0.5)
    out = zeros(nmax + 1)
    t = _T0(T, fam, j, slot, 0)
    t == 0 && return out
    if slot == 0
        for n in 0:nmax
            out[n+1] = t
            t *= (n + a) * (n + b) / ((n + c) * (n + 1.0))
        end
        return out
    end
    if abs(m - mi) > 1e-9
        aa, bb, cc = slot == 1 ? (a, b, 1.0 - m) : (c - a, c - b, 1.0 + m)
        for n in 0:nmax
            out[n+1] = t
            t *= (n + aa) * (n + bb) / ((n + cc) * (n + 1.0))
        end
        return out
    end
    mm = Int(mi)
    if slot == 1
        for n in 0:min(mm, nmax + 1)-1
            out[n+1] = t
            n + 1 < mm && (t *= (n + a) * (n + b) / ((n + 1.0) * (n + 1.0 - mi)))
        end
        return out
    end
    tl = _T0(T, fam, j, 2, 0)
    for n in 0:nmax
        if slot == 2
            out[n+1] = tl
        else
            ψ = -digamma(n + 1.0) - digamma(n + mi + 1.0) + digamma(a + n + mi) + digamma(b + n + mi)
            out[n+1] = tl * ψ
        end
        tl *= (a + mi + n) * (b + mi + n) / ((n + 1.0) * (n + mi + 1.0))
    end
    out
end

"Taylor coefficients at s0 of Σ seq_n s^n (s = s0 + t), orders ≤ nb."
function _pshift(seq, s0, nb)
    out = zeros(nb + 1)
    for n in 0:length(seq)-1
        cn = seq[n+1]
        cn == 0 && continue
        term = cn * s0^n
        for mo in 0:min(n, nb)
            out[mo+1] += term
            mo < n && (term *= (n - mo) / ((mo + 1.0) * s0))
        end
    end
    out
end

# ---- majorant sums (Lemma 5) ----
_rbar(n, a, b, c) = 1.0 + max(0.0, a + b - c - 1.0) / (n + c) + abs(a * b - c) / ((n + c) * (n + 1.0))

function _gauss_maj(t0, a, b, c, ρ, n0, nfrom)
    t0 == 0 && return 0.0
    s = 0.0
    t = abs(t0)
    w = 1.0
    n = 0
    nmin = max(n0, nfrom, floor(Int, abs(a) + abs(b) + abs(c)) + 4)
    while n < 400000
        n >= nfrom && (s += t * w)
        if n >= nmin && n + a > 0 && n + b > 0 && n + c > 0
            sr = _rbar(n, a, b, c) * ρ
            sr < 0.999 && return s + t * w * sr / (1.0 - sr)
        end
        t *= abs((n + a) * (n + b) / ((n + c) * (n + 1.0)))
        w *= ρ
        n += 1
        t * w > 1e250 && return Inf
    end
    Inf
end

function _phi(n, mi, a, b)
    t1 = abs(mi + a - 1.0) / (n + min(1.0, mi + a))
    t2 = abs(b - 1.0) / (n + mi + min(1.0, b))
    t3 = max(1.0 / (n + 1.0) + 1.0 / (n + mi + 1.0), 1.0 / (n + mi + a) + 1.0 / (n + mi + b))
    t1 + t2 + t3
end

function _psi_abs(n, mi, a, b)
    x1 = n + 1.0; x2 = n + mi + 1.0; x3 = n + mi + a; x4 = n + mi + b
    (x3 <= 0 || x4 <= 0) && return 1e300
    up = -(log(x1) - 1.0 / x1) - (log(x2) - 1.0 / x2) + log(x3) + log(x4)
    lo = -log(x1) - log(x2) + (log(x3) - 1.0 / x3) + (log(x4) - 1.0 / x4)
    max(abs(up), abs(lo))
end

"Σ_{n ≥ nfrom} |coef_n| ρ^n of the series (fam, j, slot)."
function _table_maj(g, j, fam, slot, T, ρ, nfrom)
    a, b, c = _pparams(g, j, fam)
    m = c - a - b
    mi = floor(m + 0.5)
    t0 = _T0(T, fam, j, slot, 0)
    slot == 0 && return _gauss_maj(t0, a, b, c, ρ, 0, nfrom)
    if abs(m - mi) > 1e-9
        slot == 1 && return _gauss_maj(t0, a, b, 1.0 - m, ρ, floor(Int, m) + 2, nfrom)
        slot == 2 && return 0.0
        return _gauss_maj(t0, c - a, c - b, 1.0 + m, ρ, 0, nfrom)
    end
    mm = Int(mi)
    if slot == 1
        s = 0.0; t = t0; w = 1.0
        for i in 0:mm-1
            i >= nfrom && (s += abs(t) * w)
            if i + 1 < mm
                t *= (i + a) * (i + b) / ((i + 1.0) * (i + 1.0 - mi))
                w *= ρ
            end
        end
        return s
    end
    tl0 = _T0(T, fam, j, 2, 0)
    aa, bb, cc = a + mi, b + mi, mi + 1.0
    slot == 2 && return _gauss_maj(tl0, aa, bb, cc, ρ, 0, nfrom)
    s = 0.0; t = abs(tl0); w = 1.0; n = 0
    nmin = max(nfrom, floor(Int, abs(aa) + abs(bb) + abs(cc)) + 4)
    while n < 400000
        n >= nfrom && (s += t * w * _psi_abs(n, mi, a, b))
        if n >= nmin
            sr = _rbar(n, aa, bb, cc) * ρ
            sr < 0.999 && return s + t * w * _phi(n + 1.0, mi, a, b) * sr / (1.0 - sr)
        end
        t *= abs((n + aa) * (n + bb) / ((n + cc) * (n + 1.0)))
        w *= ρ
        n += 1
        t * w > 1e250 && return Inf
    end
    Inf
end

# ---- geometry (Lemma 3) ----
function z_of_k(p, k)
    a1 = 1.0 - 2.0 * k
    p * a1 + sqrt(p * p * a1 * a1 + 1.0 - p * p)
end

function z_of_kappa(p, κ)
    a1 = 2.0 - κ
    (sqrt(p * p * a1 * a1 + κ * κ * (1.0 - p * p)) - p * a1) / κ
end

function _elem(kind, p, u::Complex, s0)
    if kind == 0 || kind == 1 || kind == 3
        k = kind == 0 ? u : kind == 1 ? 1.0 - u : s0 + u
        W = p * p * (1.0 - 2.0 * k)^2 + 1.0 - p * p
        sw = sqrt(W)
        z = p * (1.0 - 2.0 * k) + sw
        return z, 4.0 * p * z * k
    end
    κ = kind == 2 ? 1.0 - u : s0 + u
    V = p * p * (2.0 - κ)^2 + κ * κ * (1.0 - p * p)
    sv = sqrt(V)             # principal branch: Re V > 0 on the disk (_crude)
    z = (sv - p * (2.0 - κ)) / κ
    return z, 4.0 * p * z / κ
end

"Crude upper bounds (|z|, |x|) on the closed disk |u| ≤ r, or (-1, -1) if analyticity may fail."
function _crude(kind, p, s0, r)
    if kind == 0 || kind == 1 || kind == 3
        if kind == 0
            kmax = r; km1 = 1.0 + r
        elseif kind == 1
            kmax = 1.0 + r; km1 = r
        else
            r >= s0 && return -1.0, -1.0
            kmax = s0 + r; km1 = 1.0 - s0 + r
        end
        dW = 4.0 * p * p * kmax * km1
        dW >= 1 && return -1.0, -1.0
        a1 = 1.0 + 2.0 * kmax
        zlo = sqrt(1.0 - dW) - p * a1
        zlo <= 0 && return -1.0, -1.0
        zc = sqrt(1.0 + dW) + p * a1
        return zc, 4.0 * p * zc * kmax
    end
    kmin, kmax = kind == 2 ? (1.0 - r, 1.0 + r) : (s0 - r, s0 + r)
    kmin <= 0 && return -1.0, -1.0
    cen = kind == 2 ? 1.0 : s0
    d = sqrt((cen - 2.0 * p * p)^2 + 4.0 * p * p * (1.0 - p * p))
    r >= d && return -1.0, -1.0
    cc = cen - 2.0 * p * p
    u = min(1.0, max(-1.0, -cc / (2.0 * r)))
    remin = cc * cc + 4.0 * p * p * (1.0 - p * p) - r * r + 2.0 * cc * r * u + 2.0 * r * r * u * u
    remin <= 0 && return -1.0, -1.0
    vhi = (d + r)^2
    zc = (sqrt(vhi) + p * (2.0 + kmax)) / kmin
    return zc, 4.0 * p * zc / kmin
end

"Rigorous (|z|, |x|) maxima on |u| = ρ: KS samples plus a derivative bound (Lemma 4)."
function _elem_max(kind, p, s0, ρ)
    ρ2 = ρ * 1.25
    (kind == 3 || kind == 4) && ρ2 >= s0 && (ρ2 = 0.5 * (ρ + s0))
    zc, xc = _crude(kind, p, s0, ρ2)
    zc < 0 && return -1.0, -1.0
    zm = 0.0
    xm = 0.0
    for i in 0:PROVEN_KS
        th = 2π * i / PROVEN_KS
        u = ρ * complex(cos(th), sin(th))
        z, x = _elem(kind, p, u, s0)
        zm = max(zm, abs(z))
        xm = max(xm, abs(x))
    end
    h = π * ρ / PROVEN_KS
    d = ρ2 - ρ
    return zm + zc / d * h, xm + xc / d * h
end

# ---- bounds per piece ----
"B_j H_j(k), 0 ≤ k ≤ 1 (k = 1: Gauss's sum)."
function _hval_j(g, j, T, k)
    if k >= 1
        c = g + 2.5 + j
        return exp(loggamma(0.5) + loggamma(g + 2.0 + j) - loggamma(c) + loggamma(c) + loggamma(c - 1.0) -
                   2.0 * loggamma(c - 0.5))
    end
    _table_maj(g, j, 0, 0, T, k, 0)
end

"Lemma 2: |P - P_J| over the real range [slo, shi] of the piece argument."
function _jbound(kind, g, p, J, slo, shi, T)
    if kind == 0 || kind == 1 || kind == 3
        zl = z_of_k(p, shi)
        zh = z_of_k(p, slo)
        xmax = 1.0 - (zl - p)^2
        Q = zh * zh - p * p
        pre = xmax^(g + 1.0) * sqrt(min(shi, 1.0)) / (2.0 * (g + 1.0))
        F = _hval_j(g, J + 1, T, shi)
    else
        zl = z_of_kappa(p, slo)
        zh = shi < 1 ? z_of_kappa(p, shi) : 1.0 - p
        xmax = 1.0 - (zl - p)^2
        Q = zh * zh - p * p
        pre = xmax^(g + 1.0) / (2.0 * (g + 1.0))
        F = Float64(π)
    end
    xmax >= 1 && return Inf
    pre * abs(Q) * xmax^(J + 1) * F / (1.0 - xmax)
end

_nmax_mid(kind, s0, nb) = kind == 3 ? floor(Int, (nb + 60) / (1.0 - s0)) + 40 : floor(Int, (nb + 60) / s0) + 40

"Upper bounds of |part| on |u| = ρ (Lemma 5; kinds 3, 4: index 2 bounds the re-centring perturbation, Lemma 6)."
function _far_M(kind, g, p, s0, ρ, J, T)
    out = fill(Inf, 3)
    Zm, Xm = _elem_max(kind, p, s0, ρ)
    Zm < 0 && return out
    Q = p * p + Zm * Zm
    P = kind == 0 ? (4.0 * p * Zm)^(g + 1.0) / (2.0 * (g + 1.0)) :
        kind == 1 ? Xm^(g + 1.0) * sqrt(1.0 + ρ) / (2.0 * (g + 1.0)) :
        kind == 3 ? Xm^(g + 1.0) * sqrt(s0 + ρ) / (2.0 * (g + 1.0)) :
                    Xm^(g + 1.0) / (2.0 * (g + 1.0))
    kind == 4 && (s0 + ρ >= 1 || ρ > s0) && return out
    kind == 3 && s0 + ρ >= 1 && return out
    s = zeros(3)
    xj = 1.0
    rj = 1.0
    nt = (kind == 3 || kind == 4) ? _nmax_mid(kind, s0, PROVEN_NB) + 1 : 0
    ry = 0.0; lnmax = 0.0
    if kind == 4
        ry = 1.0 - s0 + ρ
        lnmax = max(abs(log(1.0 - s0 - ρ)), abs(log(ry))) + 0.5π
    end
    for j in 0:J
        cj = j == 0 ? 1.0 + Q : Q * xj
        if kind == 0
            s[1] += cj * _table_maj(g, j, 0, 0, T, ρ, 0)
        elseif kind == 3
            s[1] += cj * _table_maj(g, j, 0, 0, T, s0 + ρ, 0)
            s[2] += cj * _table_maj(g, j, 0, 0, T, s0 + ρ, nt)
        elseif kind == 4
            s[1] += cj * π
            mj = g + 1.5 + j
            s[2] += cj * (_table_maj(g, j, 1, 1, T, ry, nt) +
                          ry^mj * (lnmax * _table_maj(g, j, 1, 2, T, ry, nt) + _table_maj(g, j, 1, 3, T, ry, nt)))
        else
            fam = kind == 1 ? 0 : 1
            s[1] += cj * _table_maj(g, j, fam, 1, T, ρ, 0)
            s[2] += cj * rj * _table_maj(g, j, fam, 2, T, ρ, 0)
            s[3] += cj * rj * _table_maj(g, j, fam, 3, T, ρ, 0)
        end
        xj *= Xm
        rj *= ρ
    end
    out .= P .* s
    out
end

function _ptail(coef, w, N, Mfar, ρ)
    nb = length(coef) - 1
    s = 0.0
    t = w^(N + 1)
    for n in N+1:nb
        s += abs(coef[n+1]) * t
        t *= w
    end
    r = w / ρ
    s + Mfar * r^(nb + 1) / (1.0 - r)
end

function _lstar(m, umax)
    ue = exp(-1.0 / m)
    umax <= ue && return umax^m * abs(log(umax))
    1.0 / (ℯ * m)
end

# ---- middle-piece series ----
function _mid_coeffs(kind, g, p, s0, T, J, nb)
    n = nb + 1
    one_ = zeros(n); one_[1] = 1.0
    t = zeros(n); t[2] = 1.0
    if kind == 3
        kk = s0 .* one_ .+ t
        a1 = one_ .- 2 .* kk
        z = p .* a1 .+ _spow(p^2 .* _smul(a1, a1) .+ (1 - p^2) .* one_, 0.5)
        x = 4p .* _smul(z, kk)
        pre = -_smul(_spow(x, g + 1), _spow(kk, 0.5)) ./ (2(g + 1))
    else
        kap = s0 .* one_ .+ t
        a1 = 2 .* one_ .- kap
        V = p^2 .* _smul(a1, a1) .+ (1 - p^2) .* _smul(kap, kap)
        z = _smul(_spow(V, 0.5) .- p .* a1, _sinv(kap))
        x = 4p .* _smul(z, _sinv(kap))
        pre = -_spow(x, g + 1) ./ (2(g + 1))
    end
    q = p^2 .* one_ .- _smul(z, z)
    tot = zeros(n)
    xj = copy(one_)
    for j in 0:J
        if j >= 1
            xj = _smul(xj, x)
            cj = _smul(q, xj)
        else
            cj = one_ .+ q
        end
        Fj = kind == 3 ? _pshift(_coef_seq(g, j, 0, 0, T, _nmax_mid(3, s0, nb)), s0, nb) :
                         _g_shift(g, j, T, 1.0 - s0, nb)
        tot .+= _smul(cj, Fj)
    end
    _smul(pre, tot)
end

function _g_shift(g, j, T, y0, nb)
    a, b, c = _pparams(g, j, 1)
    m = c - a - b
    n = nb + 1
    nmax = _nmax_mid(4, 1.0 - y0, nb)
    A = _pshift(_coef_seq(g, j, 1, 1, T, nmax), y0, nb)
    Bl = _pshift(_coef_seq(g, j, 1, 2, T, nmax), y0, nb)
    Bc = _pshift(_coef_seq(g, j, 1, 3, T, nmax), y0, nb)
    ym = zeros(n); lg = zeros(n)
    ym[1] = y0^m
    lg[1] = log(y0)
    for k in 1:n-1
        ym[k+1] = ym[k] * (m - k + 1.0) / (k * y0)
        lg[k+1] = (-1.0)^(k + 1) / (k * y0^k)
    end
    S = A .+ _smul(ym, _smul(lg, Bl) .+ Bc)
    for k in 1:2:n-1
        S[k+1] = -S[k+1]
    end
    S
end

# ---- per light curve ----
function _best_far(kind, g, p, s0, w, J, T, rmax)
    best = Inf
    bestM = fill(Inf, 3)
    bestρ = 0.0
    for i in 1:8
        ρ = w + (rmax - w) * i / 9.0
        M = _far_M(kind, g, p, s0, ρ, J, T)
        r = w / ρ
        val = (kind == 1 || kind == 2) ? (M[1] + M[2] + M[3]) * r^(PROVEN_NB + 1) / (1.0 - r) :
              (kind == 3 || kind == 4) ? (M[1] * r^(PROVEN_NB + 1) + M[2]) / (1.0 - r) :
                                         M[1] * r^(PROVEN_NB + 1) / (1.0 - r)
        if val < best
            best = val; bestM = M; bestρ = ρ
        end
    end
    bestM, bestρ
end

# store_pc: (nt+1) × 3 × (NB+1)
function _do_piece!(store_pc, kd, s0, lo_, hi_, p, γ, c, tabs, τabs)
    NB = PROVEN_NB
    nt = length(γ)
    jerr = 0.0
    ρs = zeros(nt)
    Ms = zeros(nt, 3)
    fill!(store_pc, 0.0)
    for i in 1:nt
        g = γ[i]
        T = tabs[i].T
        J = 1
        jm = size(T, 2) - 2
        while J < jm && abs(c[i]) * _jbound(kd, g, p, J, lo_, hi_, T) > 0.25 * τabs / nt
            J += 1
        end
        jb = abs(c[i]) * _jbound(kd, g, p, J, lo_, hi_, T)
        jb <= 0.25 * τabs || return NB, Inf
        jerr += jb
        if kd == 3 || kd == 4
            co = _mid_coeffs(kd, g, p, s0, T, J, NB)
            @views store_pc[1, 1, :] .+= c[i] .* co
            w = hi_ - s0
            rmax = 0.95 * min(s0, 1.0 - s0)
            M, ρ = _best_far(kd, g, p, s0, w, J, T, rmax)
        else
            RA, Bp, Cp, _ = contact_pieces(g, p, tabs[i], NB, 0.5; Jfix = (J, J, J))
            if kd == 0
                @views store_pc[1+i, 1, :] .= c[i] .* RA
                w = hi_
            elseif kd == 1
                @views store_pc[1, 1, :] .+= c[i] .* Bp[1, :]
                @views store_pc[1+i, 1, :] .= c[i] .* Bp[2, :]
                @views store_pc[1+i, 2, :] .= c[i] .* Bp[3, :]
                w = 1.0 - lo_
            else
                @views store_pc[1, 1, :] .+= c[i] .* Cp[1, :]
                @views store_pc[1+i, 1, :] .= c[i] .* Cp[2, :]
                @views store_pc[1+i, 2, :] .= c[i] .* Cp[3, :]
                w = 1.0 - lo_
            end
            M, ρ = _best_far(kd, g, p, 0.0, w, J, T, 0.95)
        end
        ρs[i] = ρ
        Ms[i, :] .= M
    end
    w = (kd == 3 || kd == 4) ? hi_ - s0 : kd == 0 ? hi_ : 1.0 - lo_
    w *= PROVEN_WPAD
    for N in 2:NB
        err = jerr
        if kd == 3 || kd == 4
            rmin = 1.0; Fs = 0.0
            for i in 1:nt
                rmin = min(rmin, ρs[i])
                Fs += abs(c[i]) * Ms[i, 1]
            end
            rmin <= w && return NB, Inf
            err += _ptail(view(store_pc, 1, 1, :), w, N, Fs, rmin)
            for i in 1:nt
                err += abs(c[i]) * Ms[i, 2] / (1.0 - w / ρs[i])
            end
        elseif kd == 0
            for i in 1:nt
                ρs[i] <= w && return NB, Inf
                err += w^(γ[i] + 1.5) * _ptail(view(store_pc, 1 + i, 1, :), w, N, abs(c[i]) * Ms[i, 1], ρs[i])
            end
        else
            rmin = 1.0; Fs = 0.0
            for i in 1:nt
                ρs[i] <= w && return NB, Inf
                rmin = min(rmin, ρs[i])
                Fs += abs(c[i]) * Ms[i, 1]
                mm = γ[i] + 1.5
                err += _lstar(mm, w) * _ptail(view(store_pc, 1 + i, 1, :), w, N, abs(c[i]) * Ms[i, 2], ρs[i])
                err += w^mm * _ptail(view(store_pc, 1 + i, 2, :), w, N, abs(c[i]) * Ms[i, 3], ρs[i])
            end
            err += _ptail(view(store_pc, 1, 1, :), w, N, Fs, rmin)
        end
        err *= PROVEN_SAFETY
        err <= τabs && return N, err
    end
    NB, Inf
end

"""
    ProvenLimb

Pieces of the limb solution with a proven flux bound (Theorem 3): z-ranges, kinds,
centres, polynomial coefficients, degrees and the bound of each piece (flux units).
Pieces whose bound could not be met have `bound = Inf` and are not used.
"""
struct ProvenLimb
    p::Float64
    γ::Vector{Float64}
    zb::Matrix{Float64}
    kinds::Vector{Int}
    s0::Vector{Float64}
    store::Array{Float64,4}      # piece × (nt+1) × 3 × (NB+1)
    deg::Vector{Int}
    bound::Vector{Float64}
end

function proven_build(p::Float64, γ::Vector{Float64}, c::Vector{Float64}, tabs::Vector{LensTable{Float64}},
                      z_lo::Float64, total::Float64, τ::Float64)
    NB = PROVEN_NB
    nt = length(γ)
    τabs = τ * total
    zls = max(z_lo, p + 1e-12)
    kap_c = min(4.0 * zls * p / ((1.0 - (zls - p)) * (1.0 + (zls - p))), 1.0)
    MAXP = 24
    kinds = zeros(Int, MAXP); s0s = zeros(MAXP); lo = zeros(MAXP); hi = zeros(MAXP)
    degs = zeros(Int, MAXP); bnds = zeros(MAXP)
    store = zeros(MAXP, nt + 1, 3, NB + 1)
    npc = 0
    piece!(kd, s0, l, h) = begin
        npc += 1
        kinds[npc] = kd; s0s[npc] = s0; lo[npc] = l; hi[npc] = h
        degs[npc], bnds[npc] = _do_piece!(view(store, npc, :, :, :), kd, s0, l, h, p, γ, c, tabs, τabs)
    end
    piece!(0, 0.0, 0.0, PROVEN_KA)
    wmo = (1.0 - PROVEN_YB - PROVEN_KA) / (2.0 * PROVEN_NMO)
    for i in 0:PROVEN_NMO-1
        s0 = PROVEN_KA + wmo * (2i + 1)
        piece!(3, s0, s0 - wmo, s0 + wmo)
    end
    piece!(1, 0.0, 1.0 - PROVEN_YB, 1.0)
    yc = min(PROVEN_YB, 1.0 - kap_c)
    tmp = zeros(nt + 1, 3, NB + 1)
    while true
        dg, bd = _do_piece!(tmp, 2, 0.0, 1.0 - yc, 1.0, p, γ, c, tabs, τabs)
        (bd < Inf || yc < 0.01) && break
        yc *= 0.6
    end
    piece!(2, 0.0, 1.0 - yc, 1.0)
    top = 1.0 - yc
    while top > kap_c + 1e-12 && npc < MAXP
        w = min(PROVEN_WMI, 0.5 * (top - kap_c))
        w = min(w, 0.45 * (1.0 - top) / 1.45)
        w < 1e-4 && (w = min(0.5 * (top - kap_c), 1e-4))
        s0 = top - w
        piece!(4, s0, s0 - w, top)
        top = s0 - w
    end
    zb = zeros(npc, 2)
    for i in 1:npc
        if kinds[i] in (0, 1, 3)
            zb[i, 1] = z_of_k(p, hi[i]); zb[i, 2] = z_of_k(p, lo[i])
        else
            zb[i, 1] = z_of_kappa(p, lo[i]); zb[i, 2] = hi[i] < 1 ? z_of_kappa(p, hi[i]) : 1.0 - p
        end
    end
    zb[1, 2] = 1.0 + p
    for i in 1:npc
        if !(bnds[i] < Inf)
            zb[i, 1] = NaN; zb[i, 2] = NaN
        end
    end
    ProvenLimb(p, copy(γ), zb, kinds[1:npc], s0s[1:npc], store[1:npc, :, :, :], degs[1:npc], bnds[1:npc] ./ total)
end

@inline function _hornerN(st, pc, i, f, N, x)
    r = 0.0
    @inbounds for n in N:-1:0
        r = muladd(r, x, st[pc, i, f, n+1])
    end
    r
end

"Blocked flux from the proven pieces; NaN outside them."
function (pl::ProvenLimb)(zj::Float64)
    p = pl.p
    nt = length(pl.γ)
    st = pl.store
    for pc in eachindex(pl.kinds)
        if pl.zb[pc, 1] <= zj <= pl.zb[pc, 2]
            kd = pl.kinds[pc]
            N = pl.deg[pc]
            zmp = zj - p
            zpp = zj + p
            if kd == 0
                k = (1 - zmp) * (1 + zmp) / (4zj * p)
                r = 0.0
                if k > 0
                    lk = log(k)
                    for i in 1:nt
                        r += exp((pl.γ[i] + 1.5) * lk) * _hornerN(st, pc, 1 + i, 1, N, k)
                    end
                end
                return r
            elseif kd == 3
                k = (1 - zmp) * (1 + zmp) / (4zj * p)
                return _hornerN(st, pc, 1, 1, N, k - pl.s0[pc])
            elseif kd == 4
                κ = 4zj * p / ((1 - zmp) * (1 + zmp))
                return _hornerN(st, pc, 1, 1, N, κ - pl.s0[pc])
            else
                y = kd == 1 ? (zpp - 1) * (zpp + 1) / (4zj * p) : (1 - zpp) * (1 + zpp) / ((1 - zmp) * (1 + zmp))
                r = _hornerN(st, pc, 1, 1, N, y)
                if y > 0
                    ly = log(y)
                    for i in 1:nt
                        r += exp((pl.γ[i] + 1.5) * ly) * (ly * _hornerN(st, pc, 1 + i, 1, N, y) + _hornerN(st, pc, 1 + i, 2, N, y))
                    end
                end
                return r
            end
        end
    end
    NaN
end

"""
    occult_certified(z, p, law; series_tol = 1e-9) -> (flux, certified)

Flux as [`occult`](@ref) with `limb = :proven`, and a Boolean array that is `true`
where the flux error is proven to be at most `series_tol`: the interior series
(Theorem 2), the proven limb pieces (Theorem 3), the exact quadratic-family solution
and out-of-transit samples. Uncertified samples (radius ratios above ~0.2, or
~0.15 for `series_tol = 1e-12`) use quadrature.
"""
function occult_certified(z::AbstractArray, p::Real, law::LimbDarkening; series_tol = SERIES_TOL)
    o = Occultor(Float64(p), law; series_tol = series_tol, limb = :proven)
    f = similar(z, Float64)
    cert = similar(z, Bool)
    for i in eachindex(z)
        f[i], cert[i] = _eval_certified(o, Float64(z[i]))
    end
    return f, cert
end
