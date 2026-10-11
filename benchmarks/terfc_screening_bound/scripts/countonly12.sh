#!/usr/bin/env bash
set -u
export MODE=lmp2 NOEVAL=1 SKIPBASE=1 DELTA=1e-4 TAUS=1e-6 APPLY=0 B2_STRIDE=3000 EPS=5e-5
export OPS="terfc:1.0:0.70710678"
MOL=alkane_12.xyz FC=12 bash benchmarks/terfc_screening_bound/scripts/run.sh lmp2_countonly_c12
