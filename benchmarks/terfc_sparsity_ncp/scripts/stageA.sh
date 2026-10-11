#!/usr/bin/env bash
# Stage A: C8 (with anchor), C10, C12 dense-path sweeps. Run from repo root.
set -u
B=$(ls -t target/release/deps/terfc_locality_map-* | grep -v '\.d$' | head -1)
OUT=benchmarks/terfc_sparsity_ncp/logs
OPS="coul;erfc:0.42;terfc:1.0:0.70710678;terfc:0.991:1.354;terfc:1.408:1.164"
EPS=1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5
for N in 8 10 12; do
  A=0; [ $N = 8 ] && A=1
  FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=3 \
  MOL=alkane_$N.xyz FC=$N OPS="$OPS" EPS=$EPS ANCHOR=$A PATHM=dense MAPS=trivial REPS=3 \
  scripts/ferric-limited --max=8G --high=7G -- $B --ignored --nocapture > $OUT/stageA_c$N.log 2>&1
done
