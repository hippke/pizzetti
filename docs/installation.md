# Installation

```bash
pip install pizzetti                      # once released on PyPI
pip install git+https://github.com/hippke/pizzetti
pip install "pizzetti[fit] @ git+https://github.com/hippke/pizzetti"   # with emcee and scipy for MarginalTransitModel
```

From a clone, including the test and documentation extras:

```bash
git clone https://github.com/hippke/pizzetti
cd pizzetti
pip install -e ".[test,docs]"
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.22, numba ≥ 0.57. The tests additionally use
`batman-package` and `mpmath`; the documentation uses `sphinx`, `furo` and `myst-parser`.

The kernels are compiled by numba on first use, which takes a few seconds, and are then cached on
disk. Later imports start immediately.

## Running the tests

```bash
pytest -m "not perf"     # correctness: comparison with batman and with a 40-digit reference
pytest -m perf -s        # timings against batman (printed)
```

## Building the documentation

```bash
sphinx-build -b html docs docs/_build/html
```

## Julia

```julia
] add https://github.com/hippke/pizzetti:julia
```

The package `Pizzetti` and its examples are described in
[`julia/README.md`](https://github.com/hippke/pizzetti/tree/main/julia).

