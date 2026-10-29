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
# Re-running the same command on the same RUN_DIR resumes (the binary is NOT rebuilt or recopied
# into an existing run dir; --build only refreshes the mainline worktree).
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
  git -C "$MAINLINE" fetch -q origin
  git -C "$MAINLINE" checkout -q --detach origin/main
  (cd "$MAINLINE" && PATH=/tmp/ferric-cargo-shim:$PATH \
    cargo build --release -p ferric-cli -p ferric-python)
fi

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/driver.log"
nohup setsid "$PY" "$HERE/overnight_228.py" --run-dir "$RUN_DIR" "$@" >>"$LOG" 2>&1 </dev/null &
echo "overnight_228.py pid $! -> $LOG (re-run this command to resume; analyze: $PY $HERE/analyze_228.py $RUN_DIR/results.jsonl)"
