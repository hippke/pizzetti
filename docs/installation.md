# Installation

```bash
pip install pizzetti                      # once released on PyPI
pip install git+https://github.com/user/repo
```

From a clone, including the test and documentation extras:

```bash
git clone https://github.com/user/repo pizzetti
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
