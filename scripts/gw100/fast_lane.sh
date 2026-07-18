#!/bin/bash
# Fast-lane parallel GW100 completion. The structural bottleneck: each "small"
# molecule still takes ~10 min (BLAS-serial SCF + rayon RPA + cation UHF + 4 GW
# methods), and the two main sweeps do them serially behind slow big molecules.
#
# This launches N disjoint fast-lane workers, each running gw100_full directly
# (NOT run_sweep.py) on its OWN partition of fast molecules, writing to its OWN
# log — no shared results.json write race. Merge with merge_driver_log.py after.
#
# Each worker runs at RAYON=2 + OPENBLAS=1. Slow/big molecules are handled by
# the main sweeps separately.
#
# OPENBLAS_NUM_THREADS was 2 here from this script's original commit
# (dcc70b4, 2026-06-23) reasoning only about CPU oversubscription ("SCF phase
# is ~1-thread anyway, so low real oversubscription"). That predates the
# 2026-07-10 finding (openblas-rayon-dgetrf-crash memory / df_j.rs) that
# DF-JK SCF's `v.inv()` (LU/dgetrf-based) is deliberately left OUTSIDE
# with_blas_threads and so inherits the process-level OPENBLAS_NUM_THREADS
# directly -- dgetrf_parallel at threads>1 stack-overflows regardless of
# rayon. gw100_full runs DF-JK SCF, so OPENBLAS=2 here was a live crash risk,
# not a validated-safe throughput choice. Repo convention
# (OPENBLAS_NUM_THREADS=1 always) applies; RAYON=2 is unaffected.
#
# Usage: fast_lane.sh <basis> <nworkers>
set -u
cd /home/matt/qc/ferric
BASIS="${1:-aug-cc-pvdz}"
NW="${2:-3}"
BIN=./target/release/examples/gw100_full
TS=$(date +%H%M%S)

# Compute the fast molecules still to do for this basis (small, no heavy-diffuse,
# not already converged/failed), then partition into NW round-robin groups.
mapfile -t PARTS < <(python3 - "$BASIS" "$NW" <<'PY'
import json, re, sys
basis, nw = sys.argv[1], int(sys.argv[2])
txt=open('benchmarks/harness/examples/gw100_full.rs').read()
def info(n):
    m=re.search(rf'name:\s*"{n}".*?xyz:\s*"(.*?)"',txt,re.S)
    body=m.group(1).encode().decode('unicode_escape')
    return int(body.split('\n')[0]), set(re.findall(r'^\s*([A-Z][a-z]?)\s',body,re.M))
allc=re.findall(r'name:\s*"(\w+)"', txt)
d=json.load(open(f'scripts/gw100/results_{basis}.json'))
done=set(d['molecules']) | set(d.get('failed',[]))
fast=[c for c in allc if c not in done and info(c)[0]<=10 and not (info(c)[1] & {'Br','Si','I','Ge','Se'})]
groups=[[] for _ in range(nw)]
for i,c in enumerate(fast): groups[i%nw].append(c)
for g in groups: print(','.join(g))
PY
)

ALL=$(python3 -c "import re; print(','.join(re.findall(r'name:\s*\"(\w+)\"', open('benchmarks/harness/examples/gw100_full.rs').read())))")

for i in "${!PARTS[@]}"; do
  MINE="${PARTS[$i]}"
  [ -z "$MINE" ] && continue
  # GW100_DONE = everything EXCEPT my partition
  SKIP=$(python3 -c "
mine=set('$MINE'.split(','))
allm='$ALL'.split(',')
print(','.join(c for c in allm if c not in mine))
")
  short=$(echo $BASIS|sed 's/aug-cc-pv//')
  LOG=scripts/queue/out/fastlane_${short}_w${i}_${TS}.txt
  echo "worker $i ($short): $MINE -> $LOG"
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 -u fastlane-${short}-w${i}-${TS} \
    nice -n 12 env OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 GW100_TRUNC=1e-4 GW100_FULL_MAX_ATOMS=10 \
      GW100_DONE="$SKIP" \
      timeout 190000 "$BIN" "$BASIS" \
    > "$LOG" 2>&1 &
  sleep 1
done
echo "launched $NW fast-lane workers for $BASIS"
