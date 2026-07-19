#!/usr/bin/env bash
# BSE-TDA oscillator-strength pilot: water/ethylene/formaldehyde, cc-pVDZ,
# BSE-TDA[G0W0@HF]. See README.md in this directory for what this is (Phase 1
# pipeline-proof pilot, now with the water basis-mismatch fix from the
# 2026-07-19 apples-to-apples follow-up) and is NOT (the full Thiel-set
# benchmark, docs/bse-tda-benchmark-plan.md Phase 2).
#
# Water is run TWICE: once at cc-pVDZ (original pilot basis, kept for
# continuity/regression) and once at aug-cc-pVDZ (matches the literature TBE
# basis family for its Rydberg lowest state -- see README.md "Water basis fix"
# section). The MAE table uses the aug-cc-pVDZ number for water.
#
# Run from the repo root:
#   scripts/... or: (cd <repo root> && bash benchmarks/bse-tda-pilot/run_pilot.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ ! -x target/release/ferric ]; then
    echo "building release ferric-cli binary..."
    OPENBLAS_NUM_THREADS=1 cargo build --release -p ferric-cli
fi

export OPENBLAS_NUM_THREADS=1

OUT=benchmarks/bse-tda-pilot/results.txt
: > "$OUT"

for cfg in water water-augccpvdz c2h4 h2co; do
    echo "=== $cfg ===" | tee -a "$OUT"
    ./target/release/ferric "examples/${cfg}-bse-tda.toml" 2>&1 | tee -a "$OUT" | sed -n '1,17p'
    echo | tee -a "$OUT"
done

echo "results written to $OUT"
