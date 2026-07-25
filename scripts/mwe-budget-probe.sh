#!/usr/bin/env bash
# MWE harness: prove whether a pinned [memory] budget is actually respected.
#
# For each probe we pin a SMALL explicit budget via FERRIC_MEM_BUDGET_GB and run
# under a HARD cgroup ceiling set a little above it. The contract under test:
#
#   peak RSS <= budget * TOLERANCE
#
# Three distinguishable outcomes:
#   PASS   - ran to completion, peak RSS within tolerance of the budget
#   VIOLATE- ran to completion, but peak RSS blew past the budget (the bug:
#            the budget was accepted and then ignored)
#   KILLED - cgroup OOM-killed it (also a violation, just a louder one)
#
# The cgroup cap is the "don't clobber the box" guarantee: a runaway probe dies
# in its own scope instead of triggering the systemwide OOM killer. Defaults are
# deliberately tiny because this box is shared and frequently near-full.
set -uo pipefail

BUDGET_GB="${BUDGET_GB:-1}"
CAP="${CAP:-2G}"
# Peak RSS may legitimately exceed the budget somewhat: the budget bounds the
# large tensors, not the binary's own text/stack/allocator slack. 1.5x is loose
# enough to avoid false alarms on small systems (where fixed overhead is a large
# fraction) and tight enough that a real violation is unmissable.
TOLERANCE="${TOLERANCE:-1.5}"

if [[ $# -lt 2 ]]; then
  echo "usage: BUDGET_GB=1 CAP=2G $0 <label> <command> [args...]" >&2
  exit 2
fi

LABEL="$1"; shift

# /usr/bin/time -v reports "Maximum resident set size (kbytes)" = peak RSS of the
# whole process tree, which is what we actually care about; polling /proc would
# miss a fast spike.
TIMEFILE="$(mktemp)"
trap 'rm -f "$TIMEFILE"' EXIT

set +e
systemd-run --user --scope --quiet \
  -p MemoryHigh="${CAP}" \
  -p MemoryMax="${CAP}" \
  -p MemorySwapMax=0 \
  -- env \
     FERRIC_MEM_BUDGET_GB="${BUDGET_GB}" \
     OPENBLAS_NUM_THREADS=1 \
     RAYON_NUM_THREADS=2 \
     /usr/bin/time -v -o "$TIMEFILE" "$@" >/dev/null 2>&1
RC=$?
set -e

PEAK_KB="$(awk '/Maximum resident set size/ {print $NF}' "$TIMEFILE" 2>/dev/null)"
if [[ -z "${PEAK_KB:-}" ]]; then
  # No timing record => the process died before /usr/bin/time could write it,
  # which on a MemoryMax scope means the cgroup killed it.
  echo "${LABEL}|KILLED|budget=${BUDGET_GB}GB|peak=unknown|rc=${RC}"
  exit 0
fi

PEAK_GB="$(awk -v kb="$PEAK_KB" 'BEGIN {printf "%.2f", kb/1024/1024}')"
LIMIT_GB="$(awk -v b="$BUDGET_GB" -v t="$TOLERANCE" 'BEGIN {printf "%.2f", b*t}')"
VERDICT="$(awk -v p="$PEAK_GB" -v l="$LIMIT_GB" 'BEGIN {print (p>l) ? "VIOLATE" : "PASS"}')"

if [[ $RC -ne 0 && "$VERDICT" == "PASS" ]]; then
  VERDICT="ERROR"   # non-zero exit that is NOT a memory violation
fi

echo "${LABEL}|${VERDICT}|budget=${BUDGET_GB}GB|peak=${PEAK_GB}GB|limit=${LIMIT_GB}GB|rc=${RC}"
