"""High-precision reference flux (mpmath) for power-law intensities, by
Green's theorem; independent of the package's numerics (no series, no
elliptic integrals, adaptive quadrature at 40 digits).

The values are cached in testdata/mpref_cache.json (exact inputs -> value), so the tests do not repeat the
quadrature on every run (it was ~45 % of the suite's time). The cache records a hash of the source of
_flux_ref and _disk_mean_ref and is ignored when they change; values it lacks are computed and merged into the
file at the end of the session (atomic replace, safe with parallel workers). Delete the file to recompute all."""
import atexit
import hashlib
import inspect
import json
import os
import tempfile

import mpmath as mp


def _flux_ref(z, k, alphas, coefs, dps=40):
    with mp.workdps(dps):
        z = mp.mpf(float(z)); k = mp.mpf(float(k))
        gs = [mp.mpf(float(a)) / 2 for a in alphas]
        cs = [mp.mpf(float(c)) for c in coefs]
        total = 2 * mp.pi * sum(c / (2 * g + 2) for c, g in zip(cs, gs))
        if z >= 1 + k:
            return 1.0
        if k >= 1 and z <= k - 1:
            return 0.0

        def h(x, g):
            if x == 0:
                return mp.mpf(1) / 2
            x = min(x, mp.mpf(1))
            return (1 - (1 - x) ** (g + 1)) / (2 * (g + 1) * x)

        blocked = mp.mpf(0)
        if z > 0:
            cs_ = (1 + z * z - k * k) / (2 * z)
            th = mp.acos(max(min(cs_, 1), -1))
        else:
            th = mp.pi if k >= 1 else mp.mpf(0)
        blocked += sum(c * 2 * th / (2 * (g + 1)) for c, g in zip(cs, gs))
        c0 = (1 - z * z - k * k) / (2 * z * k) if z > 0 else (mp.mpf(1) if k < 1 else mp.mpf(-1))
        if c0 > -1:
            t0 = mp.acos(min(c0, 1))

            def f(t):
                r2 = z * z + k * k + 2 * z * k * mp.cos(t)
                return 2 * k * (k + z * mp.cos(t)) * sum(c * h(r2, g) for c, g in zip(cs, gs))

            pts = [t0 + (mp.pi - t0) * x for x in (0, mp.mpf(10) ** -8, mp.mpf(10) ** -4, mp.mpf(10) ** -2, mp.mpf(1) / 10, mp.mpf(1) / 2, 1)]
            blocked += mp.quad(f, pts)
        return float(1 - blocked / total)


def _disk_mean_ref(z, k, g, dps=40):
    """Mean of (1 - r^2)^g over the occulting disk (interior, z + k < 1),
    Green's-theorem line integral at ``dps`` digits; returns an mpf."""
    with mp.workdps(dps):
        z = mp.mpf(float(z)); k = mp.mpf(float(k)); g = mp.mpf(float(g))

        def f(t):
            r2 = z * z + k * k + 2 * z * k * mp.cos(t)
            hh = mp.mpf(1) / 2 if r2 == 0 else (1 - (1 - r2) ** (g + 1)) / (2 * (g + 1) * r2)
            return hh * (k + z * mp.cos(t))

        return 2 * mp.quad(f, [0, mp.pi / 8, mp.pi / 2, mp.pi]) / (mp.pi * k)


_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "testdata", "mpref_cache.json")
_HASH = hashlib.sha256((inspect.getsource(_flux_ref) + inspect.getsource(_disk_mean_ref)).encode()).hexdigest()[:16]
_new = {}


def _load():
    try:
        with open(_FILE) as fh:
            d = json.load(fh)
        return d["values"] if d.get("hash") == _HASH else {}
    except (OSError, ValueError, KeyError):
        return {}


_cache = _load()


def _key(name, *args):
    def enc(a):
        if isinstance(a, (list, tuple)) or getattr(a, "ndim", 0):
            return "[" + ",".join(enc(x) for x in a) + "]"
        return float(a).hex()
    return name + ":" + "|".join(enc(a) for a in args)


def flux_ref(z, k, alphas, coefs, dps=40):
    key = _key("flux", z, k, alphas, coefs, dps)
    v = _cache.get(key)
    if v is None:
        v = _cache[key] = _new[key] = float(_flux_ref(z, k, alphas, coefs, dps)).hex()
    return float.fromhex(v)


def disk_mean_ref(z, k, g, dps=40):
    """Mean of (1 - r^2)^g over the occulting disk (interior, z + k < 1),
    Green's-theorem line integral at ``dps`` digits; returns an mpf (at ``dps`` digits, also from the cache)."""
    key = _key("disk", z, k, g, dps)
    v = _cache.get(key)
    if v is None:
        with mp.workdps(dps):
            v = _cache[key] = _new[key] = mp.nstr(_disk_mean_ref(z, k, g, dps), dps + 5)
    with mp.workdps(dps):
        return mp.mpf(v)


@atexit.register
def _save():
    if not _new:
        return
    try:
        values = _load()
        values.update(_new)
        os.makedirs(os.path.dirname(_FILE), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_FILE), suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(dict(hash=_HASH, values=values), fh, separators=(",", ":"), sort_keys=True)
        os.chmod(tmp, 0o644)
        os.replace(tmp, _FILE)
    except OSError:
        pass
