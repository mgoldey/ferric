#!/usr/bin/env bash
# Launch the unattended issue-#228 run (overnight_228.py) in the background; see OVERNIGHT.md.
#
#   reference/pbc/bench/overnight_228.sh [--build] RUN_DIR [overnight_228.py args...]
#
# --build: first refresh ~/qc/ferric-mainline to origin/main and build ferric-cli + ferric-python
#          there (release, through the cargo shim).  The driver then COPIES both into RUN_DIR/bin
#          on its first launch, so later rebuilds cannot touch the run.  Without --build the
#          driver copies whatever ~/qc/ferric-mainline/target/release holds (it refuses a dirty,
#          mismatched or pre-#508 build).
# RESUME = the same command WITHOUT --build.  --build is refused once RUN_DIR has a manifest (the
# run already copied its binary) or while a driver holds RUN_DIR/lock: rebuilding the shared
# ~/qc/ferric-mainline (which the main venv's ferric .so points at) is never part of a resume.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="${FERRIC_MAINLINE:-$HOME/qc/ferric-mainline}"
PY="${PYSCF_PY:-$HOME/qc/ferric/.venv/bin/python}"

BUILD=0
if [[ "${1:-}" == "--build" ]]; then
  BUILD=1
  shift
fi
[[ $# -ge 1 ]] || { sed -n 2,13p "$0"; exit 2; }
RUN_DIR="$1"
shift

if ((BUILD)); then
  if [[ -e "$RUN_DIR/manifest.json" ]]; then
    echo "refused: $RUN_DIR/manifest.json exists; resume WITHOUT --build" >&2
    exit 2
  fi
  if [[ -e "$RUN_DIR/lock" ]] && ! flock -n "$RUN_DIR/lock" true; then
    echo "refused: a driver holds $RUN_DIR/lock" >&2
    exit 2
  fi
  git -C "$MAINLINE" fetch -q origin
  git -C "$MAINLINE" checkout -q --detach origin/main
  (cd "$MAINLINE" && PATH=$HOME/.local/share/ferric-cargo-shim:$PATH \
    cargo build --release -p ferric-cli -p ferric-python)
fi

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/driver.log"
nohup setsid "$PY" "$HERE/overnight_228.py" --run-dir "$RUN_DIR" "$@" >>"$LOG" 2>&1 </dev/null &
echo "overnight_228.py pid $! -> $LOG (resume: $0 $RUN_DIR, no --build; analyze: $PY $HERE/analyze_228.py $RUN_DIR/results.jsonl)"
