#!/usr/bin/env bash
# Run the previously-dropped dosd3 cases (GeH4, CH3Br, Br2) now that the
# SAD-default + g-function-skip + plateau SCF fixes let them converge.
#
# Honors the one-molecule-on-all-cores rule: waits for the GW100 aTZ sweep
# (gw100_finish) to finish before taking the cores, then runs each case
# serially at RAYON=12. Each case is its own memory-scoped transient run via
# the ferric CLI on the pre-staged TOMLs under scripts/dosd3/runs/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG="$ROOT/scripts/queue/out/dosd_dropped.log"
FERRIC="$ROOT/target/release/ferric"

echo "[dosd] waiting for gw100_finish to free the cores..." | tee -a "$LOG"
while systemctl --user is-active gw100_finish >/dev/null 2>&1; do
  sleep 60
done
echo "[dosd] cores free — starting dropped-case runs $(date -Is)" | tee -a "$LOG"

# The 12 dropped (mol, method, basis) combos — TOMLs already staged.
CASES=(
  augccpvdz/geh4_rpa_pbe  augccpvdz/geh4_ts
  augccpvtz/geh4_rpa_pbe  augccpvtz/geh4_ts
  augccpvdz/br2_rpa_pbe   augccpvdz/br2_ts
  augccpvtz/br2_rpa_pbe   augccpvtz/br2_ts
  augccpvdz/ch3br_rpa_pbe augccpvdz/ch3br_ts
  augccpvtz/ch3br_rpa_pbe augccpvtz/ch3br_ts
)

for c in "${CASES[@]}"; do
  toml="scripts/dosd3/runs/${c}.toml"
  npz="scripts/dosd3/runs/${c}.npz"
  if [ -f "$npz" ]; then
    echo "[dosd] SKIP $c (npz exists)" | tee -a "$LOG"
    continue
  fi
  echo "[dosd] === RUN $c $(date -Is) ===" | tee -a "$LOG"
  # Per-case stdout capture so the `molecular C6 = X a.u.` line is parseable.
  caseout="scripts/dosd3/runs/${c}.stdout"
  OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12 \
    timeout 5400 "$FERRIC" "$toml" > "$caseout" 2>&1
  rc=$?
  cat "$caseout" >> "$LOG"
  grep -E 'molecular C6' "$caseout" | tee -a "$LOG" || true
  if [ -f "$npz" ]; then
    echo "[dosd] OK   $c (rc=$rc, npz written)" | tee -a "$LOG"
  else
    echo "[dosd] FAIL $c (rc=$rc, no npz)" | tee -a "$LOG"
  fi
done
echo "[dosd] all done $(date -Is)" | tee -a "$LOG"
