#!/bin/bash
cd /home/matt/qc/ferric
OUT=scripts/queue/out/amplitude_lmp2_att.txt
LOG=scripts/queue/out/amplitude_lmp2_att.log
echo "# C20 arms started $(date), MemAvailable=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)GB" >> "$LOG"
for warg in "" "--omega 1.0"; do
  scripts/ferric-limited --max=13G --high=12500M -- \
    nice -n 5 env OPENBLAS_NUM_THREADS=8 \
    uv run --no-sync python scripts/amplitude_lmp2_proto.py \
      --xyz testdata/molecules/alkane_20.xyz --basis 6-31g \
      --eps 1e-3,1e-4 $warg --out "$OUT" >> "$LOG" 2>&1 \
    || echo "FAILED alkane_20 [$warg]" >> "$LOG"
done
echo "# C20 arms done $(date)" >> "$LOG"
