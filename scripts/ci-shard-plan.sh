#!/usr/bin/env bash
# The per-shard archive plan for ci.yml's build-tests and test-shard jobs
# (#525). build-tests builds the test binaries ONCE and packs one archive per
# shard; each test-shard job downloads only its own (~1/4 of the bytes the old
# single archive had), and runs ALL of it -- no `--partition`.
#
#   scripts/ci-shard-plan.sh slow-names
#       The binaries deferred from the fast tier to the nightly slow-tests
#       job, one per line. The ONE copy of the list.
#   scripts/ci-shard-plan.sh slow-filter
#       The same list as a nextest filterset: `binary(a) + binary(b) + ...`.
#       Callers run `-E "not ($SLOW)"`.
#   scripts/ci-shard-plan.sh assign <list.json> <weights.tsv> <N> <outdir>
#       <list.json> is `cargo nextest list --message-format json -E "not
#       ($SLOW)"` over the whole workspace. Keeps the binaries with at least
#       one test whose filter-match status is "matches" (the fast tier),
#       weights each by <weights.tsv> (binary_id TAB seconds, from
#       scripts/ci-shard-weights.sh), else 9.6 s per matching test, and
#       assigns them largest first to the least-loaded shard (LPT). Writes
#       <outdir>/<k>.filter (`binary_id(=a) | binary_id(=b) | ...`, the
#       archive filter for shard k = 1..N) and <outdir>/<k>.ids, and prints
#       each shard's predicted load. Fails if any matching binary is not
#       assigned exactly once.
#   scripts/ci-shard-plan.sh verify <list.json> <shard1.json> ... <shardN.json>
#       The EXACTLY-ONCE GUARD: the union of the per-shard archives' matching
#       tests (`cargo nextest list --archive-file ... --message-format json`,
#       same filter) must equal the full list's matching tests, with no test
#       in two shards. Fails closed.
#
# EXECUTABLES. Integration tests that run a package's own executable via
# CARGO_BIN_EXE_<name> need it in the same archive. nextest already packs a
# package's non-test executables into any archive that selects one of that
# package's test binaries, so nothing is added here; ci.yml checks the
# ferric-cli executable is present in every shard archive that needs it.
# Never add a bin target's own binary_id to a shard it was not assigned to:
# if that bin ever gains tests they would then run in two shards.
#
# Needs: python3 (ubuntu runners ship it).
set -euo pipefail

# Deferred from the fast tier: run nightly by the slow-tests job, never by
# test-shard. Adding a name here REQUIRES a matching `--test` line in
# slow-tests, or the binary stops running anywhere (deferral becomes
# deletion -- 2026-09-16, grid_prune_live_scf).
SLOW_BINARIES=(
  dft_wb97xv mpi_dfjk_banding dft_pbe pair_screen_criteria attenuation_plus_dlpno lmp2_amplitude
  screening_exactness csam_screening link_scf_anchor link_screen_reachability
  grid_prune_live_scf hirshfeld_symmetry
  cdft_coupling_hene cdft_state_selection rimp2_qqr3_screening
  cosx_group_screen_anchors cosx_sparse_anchors cosx_mixed_error_law
)


usage() {
  sed -n '2,/^# Needs:/p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

case "${1:-}" in
  slow-names)
    printf '%s\n' "${SLOW_BINARIES[@]}"
    ;;
  slow-filter)
    FILTER=""
    for b in "${SLOW_BINARIES[@]}"; do
      FILTER="${FILTER:+$FILTER + }binary($b)"
    done
    echo "$FILTER"
    ;;
  assign)
    [ "$#" -eq 5 ] || usage
    python3 - "$2" "$3" "$4" "$5" <<'PY'
import json
import os
import sys

list_json, weights_tsv, n_str, outdir = sys.argv[1:5]
n = int(n_str)
if n < 1:
    sys.exit(f"assign: N must be >= 1, got {n}")
DEFAULT_PER_TEST = 9.6  # s; the mean test time of run 40346982823

with open(list_json, encoding="utf-8") as fh:
    suites = json.load(fh)["rust-suites"]

matching = {}
for binary_id, suite in suites.items():
    count = sum(
        1
        for case in suite.get("testcases", {}).values()
        if case.get("filter-match", {}).get("status") == "matches"
    )
    if count:
        matching[binary_id] = count
if not matching:
    sys.exit(f"assign: {list_json} has no binary with a matching test")

weights = {}
with open(weights_tsv, encoding="utf-8") as fh:
    for lineno, line in enumerate(fh, 1):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            sys.exit(
                f"assign: {weights_tsv}:{lineno}: expected `binary_id<TAB>seconds`"
            )
        weights[parts[0]] = float(parts[1])


def weight(binary_id):
    return weights.get(binary_id, DEFAULT_PER_TEST * matching[binary_id])


# LPT: largest first, each to the currently least-loaded shard (lowest index
# on ties, so the plan is deterministic for a given input).
loads = [0.0] * n
shards = [[] for _ in range(n)]
for binary_id in sorted(matching, key=lambda b: (-weight(b), b)):
    k = min(range(n), key=lambda i: (loads[i], i))
    shards[k].append(binary_id)
    loads[k] += weight(binary_id)

# Exactly once, checked on the output rather than trusted from the loop.
seen = {}
for k, ids in enumerate(shards):
    for b in ids:
        if b in seen:
            sys.exit(f"assign: {b} assigned to shards {seen[b] + 1} and {k + 1}")
        seen[b] = k
missing = sorted(set(matching) - set(seen))
if missing:
    sys.exit(f"assign: {len(missing)} matching binaries unassigned: {missing[:10]}")

os.makedirs(outdir, exist_ok=True)
for k, ids in enumerate(shards, 1):
    ids_sorted = sorted(ids)
    with open(os.path.join(outdir, f"{k}.ids"), "w", encoding="utf-8") as fh:
        fh.write("".join(f"{b}\n" for b in ids_sorted))
    with open(os.path.join(outdir, f"{k}.filter"), "w", encoding="utf-8") as fh:
        fh.write(" | ".join(f"binary_id(={b})" for b in ids_sorted) + "\n")
    tests = sum(matching[b] for b in ids)
    print(
        f"shard {k}: {len(ids)} binaries, {tests} tests, predicted {loads[k - 1]:.0f} s"
    )
unweighted = sorted(b for b in matching if b not in weights)
total = sum(loads)
print(
    f"total {total:.0f} s over {n} shards, max/mean {max(loads) / (total / n):.3f}; "
    f"{len(unweighted)} binaries at the default {DEFAULT_PER_TEST} s/test"
)
PY
    ;;
  verify)
    [ "$#" -ge 3 ] || usage
    python3 - "${@:2}" <<'PY'
import json
import sys


def matching_tests(path):
    with open(path, encoding="utf-8") as fh:
        suites = json.load(fh)["rust-suites"]
    return {
        (binary_id, name)
        for binary_id, suite in suites.items()
        for name, case in suite.get("testcases", {}).items()
        if case.get("filter-match", {}).get("status") == "matches"
    }


full_path, shard_paths = sys.argv[1], sys.argv[2:]
full = matching_tests(full_path)
owner = {}
dupes = []
for k, path in enumerate(shard_paths, 1):
    tests = matching_tests(path)
    if not tests:
        sys.exit(f"verify: shard {k} ({path}) holds 0 matching tests")
    print(f"shard {k}: {len(tests)} tests")
    for t in tests:
        if t in owner:
            dupes.append((t, owner[t], k))
        else:
            owner[t] = k
union = set(owner)
problems = []
if dupes:
    problems.append(f"{len(dupes)} tests in more than one shard, e.g. {dupes[:5]}")
if full - union:
    problems.append(
        f"{len(full - union)} tests in NO shard, e.g. {sorted(full - union)[:5]}"
    )
if union - full:
    problems.append(
        f"{len(union - full)} shard tests not in the full list, e.g. {sorted(union - full)[:5]}"
    )
if problems:
    sys.exit("verify: EXACTLY-ONCE GUARD FAILED (#525):\n  " + "\n  ".join(problems))
print(
    f"exactly-once guard: {len(full)} fast-tier tests, each in exactly one of {len(shard_paths)} shards"
)
PY
    ;;
  *)
    usage
    ;;
esac
