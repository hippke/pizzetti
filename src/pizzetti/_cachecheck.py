"""Invalidate numba's on-disk cache when any source file of the package changed.

numba keys a cached function on the file that defines it. Functions it inlines or
calls from other modules, and module-level constants, are frozen into the cached
machine code, so editing one module (or updating a git checkout in which only that
module changed) can leave stale code in the caches of the others. This check hashes
all modules of the package and clears the package's cache files when the hash
changes. pip installs rewrite every file, so released versions are not affected;
the check matters for development checkouts.
"""
import glob
import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "__pycache__")
_STAMP = os.path.join(_CACHE, "numba_sources.sha256")


def check():
    try:
        h = hashlib.sha256()
        for f in sorted(glob.glob(os.path.join(_HERE, "*.py"))):
            with open(f, "rb") as fh:
                h.update(os.path.basename(f).encode() + b"\0" + fh.read())
        digest = h.hexdigest()
        old = None
        if os.path.exists(_STAMP):
            with open(_STAMP) as fh:
                old = fh.read().strip()
        if old != digest:
            for f in glob.glob(os.path.join(_CACHE, "*.nbi")) + glob.glob(os.path.join(_CACHE, "*.nbc")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            os.makedirs(_CACHE, exist_ok=True)
            with open(_STAMP, "w") as fh:
                fh.write(digest)
    except OSError:          # read-only installation: nothing to do
        pass
