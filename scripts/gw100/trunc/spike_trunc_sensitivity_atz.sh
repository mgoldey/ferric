#!/usr/bin/env bash
# Spike: GW100 G0W0 IP sensitivity + wall-time vs PDEP trunc_thresh, on a
# size ladder (H2O 3 / C2H4 6 / C6H6 12 / C7H8 15 atoms), aug-cc-pVDZ.
# Measures the size-dependence of the truncation speedup that
# TRUNCATION_VERIFIED.md hypothesized but only measured on water.
#
# For each (mol, thresh): run gw100_full on that ONE molecule (GW100_DONE=rest),
# capture G0W0 IP (eV) from the result row and total wall (s). Full-rank
# (thresh=0) is the baseline for IP shift + speedup.
#
# Usage: run under systemd-run with OPENBLAS=1, a memory cap, and RAYON set.
set -u
BIN=target/release/examples/gw100_full
SRC=benchmarks/harness/examples/gw100_full.rs
BASIS=aug-cc-pvtz
MOLS=(H2O C6H6 C7H8)
THRESHOLDS=(0 1e-4 1e-3 1e-2)
OUT=scripts/gw100/trunc/spike_sensitivity_atz_results.tsv

ALL=$(grep -oE 'name:[[:space:]]*"[A-Za-z0-9]+"' "$SRC" | sed -E 's/.*"([A-Za-z0-9]+)".*/\1/')

echo -e "mol\tatoms\tthresh\tg0w0_ev\twall_s" > "$OUT"
for mol in "${MOLS[@]}"; do
  atoms=$(grep -A1 "name: \"$mol\"" "$SRC" | grep -oE 'xyz:[[:space:]]*"[0-9]+' | grep -oE '[0-9]+' | head -1)
  skip=$(echo "$ALL" | grep -v "^${mol}$" | paste -sd,)
  for th in "${THRESHOLDS[@]}"; do
    t0=$(date +%s.%N)
    # The RESULT row is the one whose 2nd field is the numeric experimental IP;
    # the "$mol UHF(neutral-seed)" diagnostic row also starts with $mol, so filter
    # to the row where $2 is a number. G0W0 is column 6 (mol exp Koop dSCF dRPA G0W0 ...).
    g0w0=$(GW100_TRUNC="$th" GW100_DONE="$skip" GW100_FULL_MAX_ATOMS=10 GW100_PBE_ALL=0 \
          "$BIN" "$BASIS" 2>/dev/null \
          | awk -v m="$mol" '$1==m && $2+0==$2 {print $6; exit}')
    t1=$(date +%s.%N)
    wall=$(echo "$t1 - $t0" | bc)
    printf "%s\t%s\t%s\t%s\t%.1f\n" "$mol" "$atoms" "$th" "${g0w0:-NaN}" "$wall" | tee -a "$OUT"
  done
done
echo "=== DONE — results in $OUT ==="
