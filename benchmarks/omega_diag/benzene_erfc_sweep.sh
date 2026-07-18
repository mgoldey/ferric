#!/bin/bash
# Isolated ω-sweep of bare SR-MP2(erfc) on benzene dimer (S22 #11), aDZ.
# Reads the E(SR-MP2, erfc) line from rs-mp2-rpa output for dimer + 2 CP monomers,
# computes SR-MP2 correlation binding, finds where it crosses the ref.
# Single-thread; isolated from the production grid.
set -e
cd /home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
BIN=target/release/ferric-cli
OUT=benchmarks/omega_diag
GEOM=benchmarks/grid/geoms
mkdir -p $OUT/toml $OUT/out

mk_toml () {  # frag omega
  local frag=$1 w=$2 key="bz_${frag}_w${w}"
  cat > $OUT/toml/$key.toml <<TOML
[molecule]
xyz = "$PWD/$GEOM/s22-11_${frag}.xyz"
[basis]
name = "aug-cc-pvdz"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "aug-cc-pvdz-rifit"
omega = $w
formulation = "delta-lr"
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
TOML
  echo $key
}

for w in 0.30 0.42 0.55 0.673 0.80 1.00; do
  for frag in dimer mA_cp mB_cp; do
    key=$(mk_toml $frag $w)
    echo "[$(date +%H:%M:%S)] running $key"
    $BIN $OUT/toml/$key.toml > $OUT/out/$key.out 2> $OUT/out/$key.err || echo "  FAILED $key"
  done
done
echo "SWEEP DONE"
