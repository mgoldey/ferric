#!/usr/bin/env bash
# Sizes ascending, 4 eps for terfc, 5 for refs. Repo root.
S=benchmarks/terfc_sparsity_ncp/scripts
for N in "$@"; do
  bash $S/par.sh $N 2.5e-4,1e-4,4e-5,1.6e-5 1 6e-4,2.5e-4,1e-4,4e-5,1.6e-5
done
