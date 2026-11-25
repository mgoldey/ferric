#!/usr/bin/env bash
# Quiet-box sweep of the integral-direct amplitude-threshold local MP2 through
# the CLI (kind = "rimp2" / "att-rimp2" + [local] integral_direct = true).
# Replicates and extends wiki/amplitude-threshold-lmp2.md §33/§34:
#   alkane C20/C32/C48 x eps 1e-3/1e-4/1e-5 x {Coulomb, erfc(omega=1.0 Bohr^-1)}
#   6-31G / cc-pvdz-ri, frozen_core = nc, maps at the §29 production values,
#   pair gate ON (gate_cal 0.7 / 0.02) as in the bench,
#   plus the EXACT canonical reference at C20/C32 for both operators.
# eps = 1e-5 has never been measured: its C20 cells run first as a memory
# probe and larger N run only if the extrapolated peak RSS fits (see
# lmp2_sweep_gen.py predict). Every skip / kill / flag is a ROW, never silent.
#
# The §33/§34 numbers came from the cargo bench bench_direct_alkane_series
# (scripts/queue/run_tail_piece.sh), not the CLI; this harness is new.
#
# Usage (after building origin/main's CLI in a tree nothing else rebuilds):
#   scripts/queue/lmp2_sweep.sh                # full sweep
#   DRY_RUN=1 scripts/queue/lmp2_sweep.sh      # write TOMLs + plan only
#   RESUME_DIR=<out dir> scripts/queue/lmp2_sweep.sh   # skip run_idx already in rows.jsonl
#
# Knobs (env): FERRIC_SRC_BIN, FERRIC_TREE, OUT, REQUIRED_COMMIT, ALLOW_DIRTY,
#   SIZES, EPS_LIST, OPS, REF_SIZES, REPS, SEED, RSS_KILL_GB, SKIP_GB,
#   SKIP_SAFETY, RUN_TIMEOUT_S, LOAD_MAX, GATE_MAX_WAIT_S, GATE_POLL_S
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GEN="$HERE/lmp2_sweep_gen.py"

FERRIC_TREE="${FERRIC_TREE:-$HOME/qc/ferric-mainline}"
FERRIC_SRC_BIN="${FERRIC_SRC_BIN:-$FERRIC_TREE/target/release/ferric}"
REQUIRED_COMMIT="${REQUIRED_COMMIT:-6385c0c97}"   # PR #601 (stage timers + certificate)
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
SIZES="${SIZES:-20,32,48}"
EPS_LIST="${EPS_LIST:-1e-3,1e-4,1e-5}"
OPS="${OPS:-coul,erfc}"
REF_SIZES="${REF_SIZES:-20,32}"
REPS="${REPS:-2}"
SEED="${SEED:-601}"
RSS_KILL_GB="${RSS_KILL_GB:-14}"
SKIP_GB="${SKIP_GB:-12}"
SKIP_SAFETY="${SKIP_SAFETY:-1.3}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"
LOAD_MAX="${LOAD_MAX:-1.0}"
GATE_MAX_WAIT_S="${GATE_MAX_WAIT_S:-3600}"
GATE_POLL_S="${GATE_POLL_S:-15}"
DRY_RUN="${DRY_RUN:-0}"

# Measurement environment. NO cgroup cap (§34: a cap biases fitted exponents);
# the RSS watchdog below is the only memory guard.
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=6          # 6 physical cores (nproc=12 is SMT)
export FERRIC_MEM_BUDGET_GB=2   # match the §34 bench (eri3_budget_bytes 2 GiB); 6 GiB OOMs C48: DF dressing peaks at 2x budget by construction
THREADS=6
TIGHT_EPS="$(printf '%s\n' "${EPS_LIST//,/ }" | tr ' ' '\n' | sort -g | head -1)"
MIN_SIZE="$(printf '%s\n' "${SIZES//,/ }" | tr ' ' '\n' | sort -n | head -1)"

if [[ -n "${RESUME_DIR:-}" ]]; then
  OUT="$RESUME_DIR"
else
  OUT="${OUT:-$REPO/scripts/queue/out/lmp2_sweep_$(date +%Y%m%d_%H%M%S)}"
fi
mkdir -p "$OUT/runs"
ROWS="$OUT/rows.jsonl"
LOG="$OUT/harness.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- plan
if [[ ! -f "$OUT/plan.tsv" ]]; then
  python3 "$GEN" plan --out "$OUT" --molecules "$FERRIC_TREE/testdata/molecules" \
    --sizes "$SIZES" --eps "$EPS_LIST" --ops "$OPS" --ref-sizes "$REF_SIZES" \
    --reps "$REPS" --seed "$SEED" | tee -a "$LOG" || exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN: plan + TOMLs in $OUT; nothing launched"
  exit 0
fi

# ------------------------------------------------- frozen private binary
BIN="$(ls "$OUT"/ferric-frozen-* 2>/dev/null | head -1)"
if [[ -z "$BIN" ]]; then
  [[ -x "$FERRIC_SRC_BIN" ]] || { log "no binary at $FERRIC_SRC_BIN"; exit 1; }
  BIN="$OUT/ferric-frozen-$(date +%Y%m%d_%H%M%S)"
  cp "$FERRIC_SRC_BIN" "$BIN" && chmod 0555 "$BIN"
  log "froze $FERRIC_SRC_BIN -> $BIN (sha256 $(sha256sum "$BIN" | cut -c1-16))"
fi
"$BIN" --version > "$OUT/version.txt" 2>&1 || { log "binary --version failed"; exit 1; }
COMMIT="$(awk -F': ' '/^commit:/{print $2}' "$OUT/version.txt")"
DIRTY="$(awk -F': ' '/^dirty:/{print $2}' "$OUT/version.txt")"
log "binary: commit=$COMMIT dirty=$DIRTY"
if [[ "$DIRTY" != "false" && "$ALLOW_DIRTY" != "1" ]]; then
  log "REFUSING: binary is dirty=$DIRTY (set ALLOW_DIRTY=1 to override; it is recorded per row)"
  exit 1
fi
if ! git -C "$FERRIC_TREE" merge-base --is-ancestor "$REQUIRED_COMMIT" "$COMMIT" 2>/dev/null; then
  log "REFUSING: build $COMMIT does not contain $REQUIRED_COMMIT (PR #601) -- stale build?"
  exit 1
fi

# ---------------------------------------------------------------- gate
# Pre-run gate: load1 < LOAD_MAX and memory PSI some avg10 == 0. A gate that
# times out still runs the cell and FLAGS it; nothing is discarded.
GATE_WAITED=0; GATE_TIMEOUT=0
gate_wait() {
  local t0; t0=$(date +%s)
  GATE_TIMEOUT=0
  until python3 "$GEN" gate-ok --load-max "$LOAD_MAX"; do
    if (( $(date +%s) - t0 >= GATE_MAX_WAIT_S )); then GATE_TIMEOUT=1; break; fi
    sleep "$GATE_POLL_S"
  done
  GATE_WAITED=$(( $(date +%s) - t0 ))
}

# ------------------------------------------------- run with RSS watchdog
# Sets: EXIT_CODE STATUS WALL WD_MAX_RSS_KB
run_one() {
  local toml="$1" stem="$2"
  local so="$stem.stdout" se="$stem.stderr" jl="$stem.jsonl" tf="$stem.time"
  rm -f "$so" "$se" "$jl" "$tf"
  local kill_kb=$(( RSS_KILL_GB * 1024 * 1024 ))
  local t0 t1 tpid cpid="" rss killed="" start
  t0=$(date +%s.%N); start=$(date +%s)
  /usr/bin/time -v -o "$tf" "$BIN" --json "$jl" "$toml" >"$so" 2>"$se" &
  tpid=$!
  WD_MAX_RSS_KB=0
  while kill -0 "$tpid" 2>/dev/null; do
    [[ -z "$cpid" ]] && cpid="$(pgrep -P "$tpid" | head -1)"
    if [[ -n "$cpid" && -r "/proc/$cpid/status" ]]; then
      rss="$(awk '/^VmRSS:/{print $2}' "/proc/$cpid/status" 2>/dev/null)"
      rss="${rss:-0}"
      (( rss > WD_MAX_RSS_KB )) && WD_MAX_RSS_KB=$rss
      if (( rss > kill_kb )); then
        killed="killed_rss"; kill -KILL "$cpid" 2>/dev/null
        log "WATCHDOG: RSS ${rss} kB > ${RSS_KILL_GB} GB, killed pid $cpid"
      fi
    fi
    if (( $(date +%s) - start > RUN_TIMEOUT_S )) && [[ -z "$killed" ]]; then
      killed="killed_timeout"; [[ -n "$cpid" ]] && kill -KILL "$cpid" 2>/dev/null
      log "WATCHDOG: wall > ${RUN_TIMEOUT_S} s, killed pid $cpid"
    fi
    sleep 1
  done
  wait "$tpid"; EXIT_CODE=$?
  t1=$(date +%s.%N)
  WALL="$(python3 -c "print(f'{$t1-$t0:.3f}')")"
  if [[ -n "$killed" ]]; then STATUS="$killed"
  elif (( EXIT_CODE != 0 )); then STATUS="failed"
  else STATUS="ok"; fi
}

emit_row() {  # args: extra flags for `gen row`
  python3 "$GEN" row --run-idx "$IDX" --rep "$REP" --cell-id "$CID" --kind "$KIND" --n "$N" \
    --op "$OP" --eps "$EPS" --binary "$BIN" --version-file "$OUT/version.txt" \
    --threads "$THREADS" --rss-kill-gb "$RSS_KILL_GB" --started-at "$STARTED_AT" "$@" >> "$ROWS"
}

# ---------------------------------------------------------------- loop
log "sweep start: out=$OUT tight_eps=$TIGHT_EPS probe_N=$MIN_SIZE rss_kill=${RSS_KILL_GB}GB skip>${SKIP_GB}GB"
while IFS=$'\t' read -r IDX REP CID KIND N OP EPS TOML; do
  if [[ -f "$ROWS" ]] && grep -q "\"run_idx\": $IDX," "$ROWS"; then
    continue
  fi
  STARTED_AT="$(date -Is)"
  stem="$OUT/runs/$(printf '%03d' "$IDX")_${CID}_r${REP}"

  # eps = TIGHT_EPS above the probe size: measured-memory skip rule
  if [[ "$KIND" == "local" && "$EPS" != "-" && "$N" != "$MIN_SIZE" ]] \
     && python3 -c "import sys; sys.exit(0 if abs(float('$EPS')-float('$TIGHT_EPS'))<1e-15 else 1)"; then
    pred_line="$(python3 "$GEN" predict --rows "$ROWS" --op "$OP" --n "$N" --eps "$EPS" \
                 --skip-gb "$SKIP_GB" --safety "$SKIP_SAFETY")"
    decision="$(cut -f1 <<<"$pred_line")"; pred_gb="$(cut -f2 <<<"$pred_line")"; basis="$(cut -f3 <<<"$pred_line")"
    log "[$IDX] $CID r$REP predict: $pred_line"
    if [[ "$decision" == "SKIP" ]]; then
      pg="$pred_gb"; [[ "$pg" == "inf" || "$pg" == "nan" ]] && pg=""
      emit_row --status skipped_mem --skip-reason "$basis" ${pg:+--predicted-gb "$pg"}
      continue
    fi
  fi

  gate_wait
  before="$(python3 "$GEN" sample)"
  log "[$IDX] $CID r$REP start (gate waited ${GATE_WAITED}s timeout=$GATE_TIMEOUT)"
  run_one "$TOML" "$stem"
  after="$(python3 "$GEN" sample)"
  emit_row --status "$STATUS" --exit-code "$EXIT_CODE" --wall "$WALL" \
    --stdout "$stem.stdout" --stderr "$stem.stderr" --jsonl "$stem.jsonl" --timefile "$stem.time" \
    --before "$before" --after "$after" --gate-waited "$GATE_WAITED" --gate-timeout "$GATE_TIMEOUT" \
    --watchdog-max-rss-kb "$WD_MAX_RSS_KB"
  log "[$IDX] $CID r$REP $STATUS wall=${WALL}s maxrss=${WD_MAX_RSS_KB}kB"
done < "$OUT/plan.tsv"

log "sweep done; analysis:"
python3 "$HERE/lmp2_sweep_analyze.py" "$ROWS" | tee "$OUT/analysis.txt"
