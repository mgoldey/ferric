#!/usr/bin/env bash
# The validation campaign's CPU slot (design §2.4).
#
# Usage:
#   scripts/validation/run_slot.sh [--light] [--max=12G] [--high=10G] -- <cmd> [args...]
#
# DEFAULT (medium/heavy step): take the box-wide whole-CPU slot, a flock on
# $FERRIC_SLOT_LOCK (default ~/.cache/ferric/validation-slot.lock, shared by
# every worktree on purpose), then run <cmd> under scripts/ferric-limited with
# OPENBLAS_NUM_THREADS=1 and RAYON/OMP at the physical core count (6 on this
# box: nproc=12 is SMT, see the box-has-6-physical-cores memory). Only one
# slot holder runs at a time; others block until it is released.
#
# --light (seconds-to-minutes step): does NOT take the slot. Runs under
# ferric-limited with OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
# OMP_NUM_THREADS=2, so two light steps can share the box beside a slot job.
#
# Every run ends with ONE sentinel line on stdout:
#     SLOT_DONE rc=<exit code>
# Waiters gate on that line in the output file, NEVER on pgrep: a pgrep -f
# waiter matches its own command line (process-inspection memory). The line is
# printed from an EXIT trap, so it appears even if the command is killed by a
# signal this script can trap.
#
# Why `flock -o`: without -o the lock fd is inherited by <cmd> and by anything
# it daemonizes (sccache!), which then holds the slot long after <cmd> exits —
# the same failure as the sccache-daemon-holds-cargo-lock memory.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LIMITED="$REPO/scripts/ferric-limited"

LIGHT=0
LIMIT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --light) LIGHT=1; shift ;;
    --max=*|--high=*) LIMIT_ARGS+=("$1"); shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--light] [--max=12G] [--high=10G] -- <cmd> [args...]" >&2
  exit 2
fi

rc=255
trap 'echo "SLOT_DONE rc=${rc}"' EXIT

PHYS_CORES="${FERRIC_SLOT_CORES:-6}"
if [[ $LIGHT -eq 1 ]]; then
  echo "SLOT_LIGHT start $(date -u +%FT%TZ) cmd: $*"
  OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
    "$LIMITED" "${LIMIT_ARGS[@]}" -- "$@"
  rc=$?
else
  LOCK="${FERRIC_SLOT_LOCK:-$HOME/.cache/ferric/validation-slot.lock}"
  mkdir -p "$(dirname "$LOCK")"
  echo "SLOT_WAIT $(date -u +%FT%TZ) lock: $LOCK"
  OPENBLAS_NUM_THREADS=1 \
  RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$PHYS_CORES}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PHYS_CORES}" \
    flock -o "$LOCK" bash -c 'echo "SLOT_START $(date -u +%FT%TZ) cmd: $*"; exec "$0" "$@"' \
      "$LIMITED" "${LIMIT_ARGS[@]}" -- "$@"
  rc=$?
fi
exit "$rc"
