#!/usr/bin/env bash
# Stage C: range-matched erfc controls, dense path, trivial maps. Repo root.
# erfc w (1/A) = 1.16308715/R: R(0.1) NCP 1.6542 A -> 0.7031; R(0.1) linked 2.2845 A -> 0.5091;
# R(0.5) NCP 0.9911 A -> 0.4812 (erfc(x)=0.5 at x = 0.476936).
B=$(ls -t target/release/deps/terfc_locality_map-* | grep -v '\.d$' | head -1)
for N in 8 10 12; do
  FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 \
  MOL=alkane_$N.xyz FC=$N OPS="erfc:0.7031;erfc:0.5091;erfc:0.4812" \
  EPS=1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5 ANCHOR=$([ $N = 8 ] && echo 1 || echo 0) PATHM=dense MAPS=trivial REPS=1 \
  scripts/ferric-limited --max=8G --high=7G -- $B --ignored --nocapture > benchmarks/terfc_sparsity_ncp/logs/stageC_c$N.log 2>&1
done
