#!/usr/bin/env bash
# Quiet-box runner for the lmp2 t_eri3 profile and A/B runs. Measurement only.
#
#   lmp2_eri3_bench.sh OUTDIR name:binary:C:split[:pass] ...
#
# Runs each spec in order: bench_direct_alkane_series on C at eps 1e-3,1e-4 with
# both operators. `split` is 0, 1, or `-` for a binary without the knob; `pass`
# is batches or aux-major, omitted for a binary without that knob.
#
# GATE_MAX_WAIT_S (default 900) bounds each gate wait: under a lease cargo and
# exec work queue behind this job, so a foreign wait must not hold the lease
# forever. On timeout the run goes ahead and is flagged in rows.jsonl.
#
# Gate before each run: load1 < 1, memory PSI 0, no compiler, test binary or pytest
# anywhere (an idle cargo is not counted: under box-lease it is waiting for us),
# and no single process above 0.5 core over a 5 s window.
# It re-polls every 15 s. Slow rows are never discarded: after each run it records
# foreign CPU (whole-box busy jiffies minus this run's own), never raw load,
# which would count the run itself.
set -uo pipefail
O="$1"; shift
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6
mkdir -p "$O/runs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$O/harness.log"; }
psi() { awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory; }
compilers() { ps -eo comm= | grep -cE '^(rustc|cc1plus|ld|lld|rust-lld|clippy-driver|rustdoc)$'; }
# Real test processes only, judged by comm and argv[0]: a compiled test binary, or
# python/uv running pytest. Shell wrappers, box-lease, sleep and pollers that merely
# mention those words do not count.
testers() {
  ps -eo comm=,args= | awk '
    $1 ~ /^(bash|sh|dash|box-lease|pgrep|sleep|flock|timeout|grep|awk|ps)$/ { next }
    $2 ~ /(^|\/)target\/[a-z]+\/deps\// { n++; next }
    $1 ~ /^(python[0-9.]*|uv|pytest)$/ && /pytest/ { n++ }
    END { print n + 0 }'
}
busy_jiffies() { awk '/^cpu /{print $2+$3+$4+$7+$8+$9}' /proc/stat; }

# Busiest single process over a 5 s window, as "cores:comm:pid". Measured from
# /proc deltas, since ps %CPU is a lifetime average.
busiest() {
  python3 -I -c '
import glob, time
def snap():
    d = {}
    for p in glob.glob("/proc/[0-9]*/stat"):
        try:
            s = open(p).read()
        except OSError:
            continue
        r = s.rfind(")")
        f = s[r + 2:].split()
        d[p.split("/")[2]] = (int(f[11]) + int(f[12]), s[s.find("(") + 1:r])
    return d
a = snap(); time.sleep(5); b = snap()
c, pid = max((v[0] - a[k][0], k) for k, v in b.items() if k in a)
print(f"{c / 500:.2f}:{b[pid][1]}:{pid}")'
}

gate() {
  local waited=0 t0=$SECONDS
  GATE_FLAGGED=0
  while :; do
    read -r l1 _ < /proc/loadavg
    local p c b; p=$(psi); c=$(( $(compilers) + $(testers) )); b=$(busiest)
    if awk -v l="$l1" -v p="$p" -v c="$c" -v b="${b%%:*}" 'BEGIN{exit !(l<1.0 && p==0 && c==0 && b<0.5)}'; then
      GATE_BUSIEST=$b; break
    fi
    (( waited == 0 )) && log "gate waiting: load=$l1 psi=$p compilers=$c busiest=$b"
    waited=1
    if (( SECONDS - t0 >= ${GATE_MAX_WAIT_S:-900} )); then
      log "GATE TIMEOUT after $((SECONDS - t0))s: load=$l1 psi=$p testers_compilers=$c busiest=$b; running flagged"
      GATE_BUSIEST=$b; GATE_FLAGGED=1; break
    fi
    sleep 15
  done
}

run() {
  local name=$1 bin=$2 size=$3 split=$4 pass=${5:-}
  local out="$O/runs/$name"
  [[ -s "$out.done" ]] && { log "skip $name"; return; }
  gate
  read -r l_before _ < /proc/loadavg
  local p_before; p_before=$(psi)
  log "start $name bin=$bin C$size split=$split load=$l_before psi=$p_before busiest=$GATE_BUSIEST flagged=$GATE_FLAGGED"
  local envs=(FERRIC_LMP2_BENCH_MIN_C="$size" FERRIC_LMP2_BENCH_MAX_C="$size"
              FERRIC_LMP2_BENCH_EPS=1e-3,1e-4 FERRIC_LMP2_BENCH_OPS=coul,erfc1)
  [[ "$split" != "-" ]] && envs+=(FERRIC_LMP2_BENCH_SPLIT_ERI3="$split")
  [[ -n "$pass" ]] && envs+=(FERRIC_LMP2_BENCH_ERI3_PASS="$pass")
  local j0 t0; j0=$(busy_jiffies); t0=$EPOCHREALTIME
  env "${envs[@]}" /usr/bin/time -f '%U %S %M' -o "$out.time" "$O/bin/$bin" \
    bench_direct_alkane_series --ignored --exact --nocapture --test-threads=1 > "$out.out" 2> "$out.err"
  local rc=$? j1 wall own foreign l_after p_after
  j1=$(busy_jiffies)
  wall=$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN{print b-a}')
  read -r l_after _ < /proc/loadavg
  p_after=$(psi)
  own=$(awk '{print $1+$2}' "$out.time" | tail -1)
  foreign=$(awk -v a="$j0" -v b="$j1" -v o="$own" 'BEGIN{printf "%.1f", (b-a)/100-o}')
  printf '{"name":"%s","bin":"%s","c":%s,"split":"%s","rc":%s,"wall_s":%s,"own_cpu_s":%s,"foreign_cpu_s":%s,"load_before":%s,"load_after":%s,"psi_before":%s,"psi_after":%s,"gate_flagged":%s}\n' \
    "$name" "$bin" "$size" "$split" "$rc" "$wall" "$own" "$foreign" "$l_before" "$l_after" "$p_before" "$p_after" "$GATE_FLAGGED" >> "$O/rows.jsonl"
  log "end $name rc=$rc wall=${wall}s foreign_cpu=${foreign}s load_after=$l_after psi_after=$p_after"
  echo "$rc" > "$out.done"
}

for spec in "$@"; do
  IFS=: read -r name bin size split pass <<< "$spec"
  run "$name" "$bin" "$size" "$split" "$pass"
done
log "series done"
