#!/usr/bin/env bash
# ferric side of the real-size PBC benchmark (prototypes/pbc/FINDINGS.md, "Real-size benchmark
# plan — 2026-09-25"). Runs the ferric CLI on toml/<run>.toml under a cgroup memory cap
# (scripts/ferric-limited), with /usr/bin/time -v, OPENBLAS_NUM_THREADS=1 and
# RAYON_NUM_THREADS=$THREADS, and writes per run into out/:
#   ferric_<run>_t<N>.log          stdout+stderr (energy, iterations, grid points, ...)
#   ferric_<run>_t<N>.time         /usr/bin/time -v (max RSS, elapsed, user/sys CPU)
#   ferric_<run>_t<N>.ferric.jsonl ferric run log (per-iteration timestamps, run_end wall_s,
#                                  cpu_s, peak_rss_bytes = VmHWM)
# and appends one JSON line to out/ferric_summary.jsonl.
#
# Stage split: ferric prints no per-stage timers. The run log's first `scf_iter` record is
# written after hcore + RS-GDF (+ XC grid) + one SCF iteration, so
#   build_s ~= t(iter 1) - (t(iter 2) - t(iter 1)),   scf_s = t(run_end) - build_s.
# Everything before the first iteration is one lump; the plan's predictions say which part of
# it should dominate (hcore SR vs RS-GDF SR3 vs LR pair FT); neither the CLI nor the Python
# bindings (run_ferric_bench.py, which adds the aux drop count) can split hcore from RS-GDF.
#
# Usage (from the worktree root; NEVER two at once on the shared box):
#   reference/pbc/bench/run_ferric_bench.sh [--threads N] [--max 12G] [--high 10G] run [run ...]
#   reference/pbc/bench/run_ferric_bench.sh --threads 1 pre_diamond_prim_sto3g_ri_rhf
# Runs are the basenames of toml/*.toml (make_geometries.py writes them).
#
# Env: FERRIC_BIN (default target/release/ferric). The binary must be built from THIS worktree's
# HEAD (it was stale at 07:53 while the [cell] CLI landed later that day); the script refuses a
# binary older than the newest commit touching crates/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

THREADS=1
MAX=12G
HIGH=10G
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    --high) HIGH="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -ge 1 ]] || { sed -n 2,30p "$0"; exit 2; }

BIN="${FERRIC_BIN:-$ROOT/target/release/ferric}"
[[ -x "$BIN" ]] || { echo "no ferric binary at $BIN (cargo build --release -p ferric-cli)" >&2; exit 1; }
bin_t=$(stat -c %Y "$BIN")
src_t=$(git log -1 --format=%ct -- crates/ 2>/dev/null || echo 0)
if (( bin_t < src_t )); then
  echo "refusing: $BIN is older than the last commit under crates/ ($(date -d @"$src_t")); rebuild" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain -- crates/ 2>/dev/null)" ]]; then
  echo "warning: uncommitted changes under crates/; the binary may not match the source" >&2
fi

OUT="$HERE/out"
mkdir -p "$OUT"
for run in "$@"; do
  toml="$HERE/toml/$run.toml"
  [[ -f "$toml" ]] || { echo "no $toml" >&2; exit 1; }
  tag="ferric_${run}_t${THREADS}"
  read -r l1 l5 l15 _ < /proc/loadavg
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  echo "[bench] $run threads=$THREADS load=$l1/$l5/$l15 MemAvailable=$((avail_kb / 1024)) MiB cap=$MAX" >&2
  rm -f "$OUT/$tag.ferric.jsonl"
  set +e
  "$ROOT/scripts/ferric-limited" --max="$MAX" --high="$HIGH" -- \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS="$THREADS" \
    /usr/bin/time -v -o "$OUT/$tag.time" \
    "$BIN" --json "$OUT/$tag.ferric.jsonl" "$toml" > "$OUT/$tag.log" 2>&1
  status=$?
  set -e
  read -r e1 e5 e15 _ < /proc/loadavg
  python3 - "$run" "$THREADS" "$status" "$OUT/$tag" "$l1" "$e1" <<'PY' | tee -a "$OUT/ferric_summary.jsonl"
import json, re, sys
run, threads, status, base, load0, load1 = sys.argv[1:]
rec = dict(code="ferric", run=run, threads=int(threads), exit=int(status), loadavg_start=float(load0), loadavg_end=float(load1))
try:
    t = open(base + ".time").read()
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    rec["time_max_rss_bytes"] = int(m.group(1)) * 1024 if m else None
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    if m:
        parts = [float(x) for x in m.group(1).split(":")]
        rec["time_wall_s"] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    for k, key in (("User time (seconds)", "time_user_s"), ("System time (seconds)", "time_sys_s")):
        m = re.search(re.escape(k) + r": ([\d.]+)", t)
        rec[key] = float(m.group(1)) if m else None
except OSError:
    pass
iters, end = [], None
try:
    for line in open(base + ".ferric.jsonl"):
        r = json.loads(line)
        if r.get("record") == "scf_iter":
            iters.append(r)
        elif r.get("record") == "run_end":
            end = r
except OSError:
    pass
if end:
    rec.update(energy=end.get("energy"), converged=end.get("converged"), wall_s=end.get("wall_s"),
               cpu_s=end.get("cpu_s"), peak_rss_bytes=end.get("peak_rss_bytes"))
rec["scf_iters"] = len(iters)
if len(iters) >= 2:
    dt = iters[1]["t"] - iters[0]["t"]
    rec["build_s_est"] = iters[0]["t"] - dt
    rec["scf_s_per_iter"] = (iters[-1]["t"] - iters[0]["t"]) / (len(iters) - 1)
elif iters:
    rec["build_s_est"] = iters[0]["t"]
log = open(base + ".log").read()
for key, pat in (("nbasis", r"nbasis\s+=\s+(\d+)"), ("grid_points", r"grid\s+=\s+(\d+) points"),
                 ("e_nuc", r"e_nuc\s+=\s+(-?[\d.]+)"), ("madelung", r"madelung\s+=\s+(-?[\d.]+)")):
    m = re.search(pat, log)
    if m:
        rec[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
m = re.search(r"energy\s+=\s+(-?[\d.]+) Hartree/cell", log)
rec["energy_printed"] = float(m.group(1)) if m else None
print(json.dumps(rec))
PY
  if (( status != 0 )); then
    echo "[bench] $run FAILED (exit $status); see $OUT/$tag.log — stopping (later runs are larger)" >&2
    exit "$status"
  fi
done
