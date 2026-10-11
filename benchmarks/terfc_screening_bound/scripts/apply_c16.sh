#!/usr/bin/env bash
# Energy-effect of APPLYING the S|B1 skip (rule 2) at C16; includes deliberately-loose tau (mutation: the check must be able to fail).
set -u
export MODE=lmp2 MOL=alkane_16.xyz FC=16 EPS=5e-5 DELTA=1e-4 APPLY=1 RULE=2 B2_STRIDE=0 TAUS=1e-8,1e-6,1e-4,1e-3
OPS="terfc:1.0:0.70710678" bash benchmarks/terfc_screening_bound/scripts/run.sh lmp2_apply_c16_linked
OPS="terfc:0.991:1.354" bash benchmarks/terfc_screening_bound/scripts/run.sh lmp2_apply_c16_ncp
