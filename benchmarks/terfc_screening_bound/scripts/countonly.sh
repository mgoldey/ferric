#!/usr/bin/env bash
# Count-only map-overlay at C20 / C32 (NOEVAL: every triple skipped after counting; energy is garbage).
set -u
export MODE=lmp2 NOEVAL=1 SKIPBASE=1 DELTA=1e-4 TAUS=1e-6,1e-8 APPLY=0 B2_STRIDE=3000 EPS=5e-5
export OPS="terfc:1.0:0.70710678;terfc:0.991:1.354"
MOL=alkane_20.xyz FC=20 bash benchmarks/terfc_screening_bound/scripts/run.sh lmp2_countonly_c20
MOL=alkane_32.xyz FC=32 bash benchmarks/terfc_screening_bound/scripts/run.sh lmp2_countonly_c32
