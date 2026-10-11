#!/usr/bin/env bash
# Stage B for one size: linked and NCP in parallel processes (3 threads each), plus
# optional cheap refs. usage: par.sh N EPS_TERFC [refs=0|1 EPS_REF]. Repo root.
S=benchmarks/terfc_sparsity_ncp/scripts
N=$1; EPS=$2; REFS=${3:-0}; EPSREF=${4:-6e-4,2.5e-4,1e-4,4e-5,1.6e-5}
bash $S/stageB.sh $N "terfc:1.0:0.70710678" $EPS stageB_c${N}_linked.log 3 &
bash $S/stageB.sh $N "terfc:0.991:1.354" $EPS stageB_c${N}_ncp.log 3 &
if [ "$REFS" = 1 ]; then
  bash $S/stageB.sh $N "coul;erfc:0.42" $EPSREF stageB_c${N}_refs.log 2 &
fi
wait
