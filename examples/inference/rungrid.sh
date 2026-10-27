#!/bin/sh
# Inference grid of the paper (about 30 minutes on one core).
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for c in jup_quad jup_fixed jup_free se_fixed se_free; do for code in batman pizzetti; do
  python3 run_grid.py $c $code 20000
done; done
