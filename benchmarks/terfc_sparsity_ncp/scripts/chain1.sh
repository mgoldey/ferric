#!/usr/bin/env bash
S=benchmarks/terfc_sparsity_ncp/scripts
bash $S/confirm.sh
bash $S/stageB.sh 12 "coul;terfc:1.0:0.70710678;terfc:0.991:1.354" 1e-4 probe_c12.log
