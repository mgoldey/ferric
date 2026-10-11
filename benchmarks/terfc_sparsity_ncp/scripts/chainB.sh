#!/usr/bin/env bash
# Stage B sizes, ascending. usage: chainB.sh N [N ...]; run from repo root.
S=benchmarks/terfc_sparsity_ncp/scripts
for N in "$@"; do
  bash $S/stageB.sh $N "coul;erfc:0.42;terfc:1.0:0.70710678;terfc:0.991:1.354" 6e-4,2.5e-4,1e-4,4e-5,1.6e-5 stageB_c$N.log
done
