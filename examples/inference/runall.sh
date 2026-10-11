#!/bin/sh
# Full runs used in the paper (about 25 minutes on one core).
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for c in tess kepler; do for code in batman pizzetti; do
  python3 run_inference.py $c $code 20000
done; done
