#!/usr/bin/env bash
# Interleaved A/B of bench_direct_alkane_series: main (A-D, relaxed cg_rtol) vs
# the commit before #601. Same instrument as wiki/amplitude-threshold-lmp2.md
# sec. 33/34. Seconds from sec. 34 predate the fan fix, so only same-box ratios
# from this run are comparable. Measurement only; no interpretation here.
#
# Plan: base/main at eps 1e-3,1e-4 in ABBA order at C20 and C32, AB at C48, then
# main-only eps 1e-5, per size and operator (separately, so a kill loses one cell).
# Each run gets its own process and a frozen binary. Slow rows are never discarded.
#
# To skip the repeated SCF, export FERRIC_LMP2_BENCH_RHF_CACHE=<existing dir> before
# running; the env reaches every run, so nothing below changes. Both binaries must
# carry the cache patch (crates/ferric-mp2/tests/common/bench_rhf_cache.rs), and each
# must pass bench_rhf_cache_matches_fresh_scf against that dir first. Stage timers
# are unaffected, but wall_s and maxrss_kb no longer include the SCF.
set -uo pipefail
O="${O:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/out/lmp2_ab_20261030}"
RSS_KILL_KB=$((14 * 1024 * 1024))
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6
mkdir -p "$O/runs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$O/harness.log"; }

gate() { # wait for a quiet box; record what we waited for
  local t0=$SECONDS
  while :; do
    read -r l1 _ < /proc/loadavg
    psi=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
    if awk -v l="$l1" -v p="$psi" 'BEGIN{exit !(l<1.0 && p==0)}'; then break; fi
    (( SECONDS - t0 > 3600 )) && { log "GATE TIMEOUT load=$l1 psi=$psi, running flagged"; GATE_TO=1; return; }
    sleep 15
  done
  GATE_TO=0
}

run() { # name build min_max eps ops
  local name=$1 build=$2 size=$3 eps=$4 ops=$5
  local out="$O/runs/$name"
  [[ -s "$out.done" ]] && { log "skip $name (done)"; return; }
  gate
  read -r l_before _ < /proc/loadavg
  psi_before=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  log "start $name build=$build C$size eps=$eps ops=$ops gate_timeout=$GATE_TO load=$l_before"
  local envs=(FERRIC_LMP2_BENCH_MIN_C="$size" FERRIC_LMP2_BENCH_MAX_C="$size")
  # The baseline predates the knobs and always runs eps 1e-3,1e-4 and both ops.
  [[ "$build" == main ]] && envs+=(FERRIC_LMP2_BENCH_EPS="$eps" FERRIC_LMP2_BENCH_OPS="$ops")
  local t0=$EPOCHREALTIME
  env "${envs[@]}" /usr/bin/time -v -o "$out.time" "$O/bin/lmp2_direct_$build" \
    bench_direct_alkane_series --ignored --exact --nocapture --test-threads=1 > "$out.out" 2> "$out.err" &
  local pid=$! maxrss=0 killed=0 kid rss
  while kill -0 "$pid" 2>/dev/null; do
    kid=$(pgrep -P "$pid" | head -1)
    if [[ -n "$kid" ]]; then
      rss=$(awk '/^VmRSS/{print $2}' "/proc/$kid/status" 2>/dev/null || echo 0)
      (( ${rss:-0} > maxrss )) && maxrss=${rss:-0}
      if (( ${rss:-0} > RSS_KILL_KB )); then log "WATCHDOG $name RSS ${rss} kB"; kill "$kid"; killed=1; fi
    fi
    sleep 1
  done
  wait "$pid"; local rc=$?
  local wall; wall=$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN{print b-a}')
  read -r l_after _ < /proc/loadavg
  psi_after=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  printf '{"name":"%s","build":"%s","c":%s,"eps":"%s","ops":"%s","rc":%s,"killed":%s,"wall_s":%s,"maxrss_kb":%s,"load_before":%s,"load_after":%s,"psi_before":%s,"psi_after":%s,"gate_timeout":%s}\n' \
    "$name" "$build" "$size" "$eps" "$ops" "$rc" "$killed" "$wall" "$maxrss" "$l_before" "$l_after" "$psi_before" "$psi_after" "$GATE_TO" >> "$O/rows.jsonl"
  log "end $name rc=$rc killed=$killed wall=${wall}s maxrss=${maxrss}kB load_after=$l_after psi_after=$psi_after"
  echo "$rc" > "$out.done"
}

cat "$O/bin/main.sha" "$O/bin/base.sha" | tr '\n' ' ' | tee "$O/version.txt"; echo
log "start; plan ABBA C20,C32; AB C48; main eps 1e-5 per size/op"
E="1e-3,1e-4"; OPS="coul,erfc1"
for c in 20 32; do
  run "c${c}_r1_base" base "$c" "$E" "$OPS"
  run "c${c}_r1_main" main "$c" "$E" "$OPS"
  run "c${c}_r2_main" main "$c" "$E" "$OPS"
  run "c${c}_r2_base" base "$c" "$E" "$OPS"
done
run c48_r1_base base 48 "$E" "$OPS"
run c48_r1_main main 48 "$E" "$OPS"
for c in 20 32 48; do
  for op in erfc1 coul; do
    run "c${c}_e5_${op}" main "$c" "1e-5" "$op"
  done
done
log "bench done"
