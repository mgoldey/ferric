#!/usr/bin/env bash
# Instrument check: does the UNIQ (unique-triple) counter fall below 100% when pair partners/maps shrink?
set -u
export MODE=lmp2 NOEVAL=1 SKIPBASE=1 DELTA=1e-4 TAUS=1e-6 APPLY=0 B2_STRIDE=0
export OPS="terfc:1.0:0.70710678"
export MOL=alkane_12.xyz FC=12
EPS=1e-2 AUXR=3 VIRTR=4 AOTAIL=1e-2 bash benchmarks/terfc_screening_bound/scripts/run.sh uniq_mut_c12_eps1e-2_tight
EPS=1e-2 AUXR=10 VIRTR=12 AOTAIL=1e-3 bash benchmarks/terfc_screening_bound/scripts/run.sh uniq_mut_c12_eps1e-2_prod
EPS=1e-3 AUXR=10 VIRTR=12 AOTAIL=1e-3 bash benchmarks/terfc_screening_bound/scripts/run.sh uniq_mut_c12_eps1e-3_prod
