# Branch-free log and exp for the vectorised interior kernel of laws with an arbitrary
# exponent (power-2): w^γ = exp(γ log w) for 1e-12 ≤ w ≤ 1. Base's log and exp use
# table lookups and branches and do not vectorise; these use only arithmetic, integer
# shifts and selects, so that the loop over samples compiles to SIMD instructions.
# Accuracy: a few ulp (tested against Base in the test suite).

# Horner's scheme expanded into a muladd chain at parse time (always inlined)
macro _horner(x, cs...)
    ex = esc(cs[end])
    for c in reverse(cs[1:end-1])
        ex = :(muladd(t, $ex, $(esc(c))))
    end
    return :(let t = $(esc(x)); $ex; end)
end

const _LN2_HI = 6.93147180369123816490e-01
const _LN2_LO = 1.90821492927058770002e-10
const _INV_LN2 = 1.4426950408889634
const _SHIFT = 6.755399441055744e15            # 1.5 * 2^52
const _TWO52 = 4.503599627370496e15            # 2^52

# log(x) for normal x > 0
@inline function _vlog(x::Float64)
    b = reinterpret(UInt64, x)
    ebits = (b >> 52) & 0x7ff
    m = reinterpret(Float64, (b & 0x000fffffffffffff) | 0x3ff0000000000000)       # [1, 2)
    big = m > 1.4142135623730951
    m = ifelse(big, 0.5 * m, m)
    # exponent as a double without int -> float conversion: (2^52 + ebits) - (2^52 + 1023)
    e = reinterpret(Float64, ebits | 0x4330000000000000) - (_TWO52 + 1023.0)
    e = ifelse(big, e + 1.0, e)
    t = (m - 1.0) / (m + 1.0)                   # |t| ≤ 0.1716
    t2 = t * t
    s = @_horner(t2, 2.0, 2 / 3, 2 / 5, 2 / 7, 2 / 9, 2 / 11, 2 / 13, 2 / 15, 2 / 17, 2 / 19, 2 / 21, 2 / 23)
    return muladd(e, _LN2_HI, muladd(e, _LN2_LO, t * s))
end

# exp(y) for -700 < y ≤ 1
@inline function _vexp(y::Float64)
    y = ifelse(y > -700.0, y, -700.0)
    kd = muladd(y, _INV_LN2, _SHIFT)
    k = kd - _SHIFT
    r = muladd(-k, _LN2_LO, muladd(-k, _LN2_HI, y))           # |r| ≤ 0.347
    p = @_horner(r, 1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320, 1 / 362880,
                          1 / 3628800, 1 / 39916800, 1 / 479001600, 1 / 6227020800, 1 / 87178291200)
    # 2^k from the low bits of kd
    sc = reinterpret(Float64, (reinterpret(UInt64, kd) + UInt64(1023)) << 52)
    return p * sc
end

@inline _vpow(w::Float64, g::Float64) = _vexp(g * _vlog(w))

# x^g for x ≥ 0 in the quadrature: the branch-free version for Float64 (x below 1e-300
# gives 0, i.e. x^g underflows for the g ≥ 1 used there), exp(g log x) otherwise
@inline _powq(x::Float64, g::Float64) = x < 1e-300 ? 0.0 : _vpow(x, g)
@inline _powq(x, g) = exp(g * log(x))
