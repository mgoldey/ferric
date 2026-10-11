#!/usr/bin/env bash
# Stage B: direct path, production maps. usage: stageB.sh N "OPS" EPS LOGNAME [threads]
# Run from repo root.
set -u
B=$(ls -t target/release/deps/terfc_locality_map-* | grep -v '\.d$' | head -1)
N=$1; OPS=$2; EPS=$3; LOG=$4; TH=${5:-3}
FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=$TH \
MOL=alkane_$N.xyz FC=$N OPS="$OPS" EPS=$EPS ANCHOR=0 PATHM=direct MAPS=prod REPS=1 \
scripts/ferric-limited --max=10G --high=9G -- $B --ignored --nocapture >> benchmarks/terfc_sparsity_ncp/logs/$LOG 2>&1
