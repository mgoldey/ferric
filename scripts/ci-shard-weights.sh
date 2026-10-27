#!/usr/bin/env bash
# Regenerate scripts/ci-shard-weights.tsv from the test-shard logs of one
# GREEN ci.yml run (#525).
#
#   scripts/ci-shard-weights.sh <run-id> [out.tsv]
#
# The weights are the per-binary sums of nextest's per-test durations, read
# from lines such as
#
#   PASS [  12.345s] (  17/1160) ferric-scf::dft_pbe pbe_water_energy
#
# (every status word is counted -- a retried test's attempts all cost shard
# time). Output: one `binary_id<TAB>seconds` line per binary, sorted by
# binary id, seconds to 0.001 s.
#
# scripts/ci-shard-plan.sh `assign` uses these to balance the per-shard
# archives. Stale weights only worsen the balance; they never lose a test,
# because a binary missing from the file is still assigned (at a default
# weight) and build-tests' exactly-once guard checks the union.
#
# Needs: gh (authenticated, actions:read on the repo), python3.
set -euo pipefail

RUN=${1:?usage: $0 <run-id> [out.tsv]}
OUT=${2:-$(dirname "$0")/ci-shard-weights.tsv}
REPO=${GITHUB_REPOSITORY:-mgoldey/ferric}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Assigned on its own line so a failing gh (unknown run, no auth) trips set -e.
JOB_LINES=$(gh api --paginate "repos/$REPO/actions/runs/$RUN/jobs?per_page=100" \
  --jq '.jobs[] | select(.name | startswith("test-shard (")) | "\(.id) \(.conclusion)"')
mapfile -t JOBS < <(printf '%s\n' "$JOB_LINES" | sed '/^$/d')
if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "run $RUN has no test-shard jobs" >&2
  exit 1
fi
for j in "${JOBS[@]}"; do
  read -r id conclusion <<<"$j"
  if [ "$conclusion" != success ]; then
    echo "test-shard job $id of run $RUN concluded '$conclusion': use a green run" >&2
    exit 1
  fi
  gh api --allow-escape-sequences "repos/$REPO/actions/jobs/$id/logs" >"$TMP/$id.log"
done
echo "read ${#JOBS[@]} test-shard logs from run $RUN" >&2

python3 - "$OUT" "$TMP"/*.log <<'PY'
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# `<STATUS> [ 1.234s] (  n/N) <binary_id> <test name>`; the (n/N) counter is
# absent in some nextest output modes.
LINE = re.compile(
    r"\b[A-Z][A-Z0-9 /]*\[\s*([0-9]+(?:\.[0-9]+)?)s\]\s+(?:\(\s*\d+/\d+\)\s+)?(\S+)\s+\S"
)

out, logs = sys.argv[1], sys.argv[2:]
weights = defaultdict(float)
lines = 0
for path in logs:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = LINE.search(ANSI.sub("", raw))
            if m:
                weights[m.group(2)] += float(m.group(1))
                lines += 1
if not weights:
    sys.exit("no nextest result lines found in the logs")
with open(out, "w", encoding="utf-8") as fh:
    for binary_id in sorted(weights):
        fh.write(f"{binary_id}\t{weights[binary_id]:.3f}\n")
total = sum(weights.values())
print(
    f"{lines} result lines, {len(weights)} binaries, {total:.0f} s total -> {out}",
    file=sys.stderr,
)
PY
