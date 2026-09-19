#!/bin/bash
# Benchmark pass A (deterministic parts; runs beside C20 under its own cap):
# 1. fit-radius sweep on C8 (error vs r, both operators, eps 1e-4)
# 2. composed pipeline scaling C8/C12/C16 (gate + ri-domain + ragged FLOPs)
cd /home/matt/qc/ferric
OUT=scripts/queue/out/amplitude_lmp2_bench.txt
LOG=scripts/queue/out/amplitude_lmp2_bench.log
RUN="scripts/ferric-limited --max=4G --high=3800M -- nice -n 10 env OPENBLAS_NUM_THREADS=2 uv run --no-sync python scripts/amplitude_lmp2_proto.py --basis 6-31g"
echo "# bench A started $(date)" >> "$LOG"
for r in 6 8 10 12; do
  for warg in "" "--omega 1.0"; do
    $RUN --xyz testdata/molecules/alkane_8.xyz --eps 1e-4 \
      --integrals ri-domain --fit-radius $r $warg --out "$OUT" \
      >> "$LOG" 2>&1 || echo "FAILED radius r=$r [$warg]" >> "$LOG"
  done
done
for m in alkane_8 alkane_12 alkane_16; do
  for spec in "|0.7" "--omega 1.0|0.02"; do
    warg="${spec%|*}"; cal="${spec#*|}"
    $RUN --xyz testdata/molecules/$m.xyz --eps 1e-3,1e-4 \
      --integrals ri-domain --fit-radius 10 --gate-cal $cal \
      --solver ragged $warg --out "$OUT" >> "$LOG" 2>&1 \
      || echo "FAILED pipeline $m [$warg]" >> "$LOG"
  done
done
echo "# bench A done $(date)" >> "$LOG"
