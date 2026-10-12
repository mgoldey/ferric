#!/usr/bin/env bash
# Confirming runs at the interpolated eps* (linked vs NCP), dense path. Repo root.
set -u
B=$(ls -t target/release/deps/terfc_locality_map-* | grep -v '\.d$' | head -1)
OUT=benchmarks/terfc_sparsity_ncp/logs
run() { # N op eps
  FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=3 \
  MOL=alkane_$1.xyz FC=$1 OPS="$2" EPS=$3 ANCHOR=0 PATHM=dense MAPS=trivial REPS=1 \
  scripts/ferric-limited --max=8G --high=7G -- $B --ignored --nocapture >> $OUT/confirm.log 2>&1
}
# eps* values from stageA_table.txt (tol 1e-3, 1e-2)
run 8  "terfc:1.0:0.70710678" 5.59e-05,3.51e-04
run 8  "terfc:0.991:1.354"    7.59e-05,4.37e-04
run 10 "terfc:1.0:0.70710678" 5.19e-05,3.29e-04
run 10 "terfc:0.991:1.354"    7.01e-05,4.15e-04
run 12 "terfc:1.0:0.70710678" 4.95e-05,3.16e-04
run 12 "terfc:0.991:1.354"    6.65e-05,4.00e-04
