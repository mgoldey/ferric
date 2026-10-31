#!/usr/bin/env bash
# Compare KS energies, bit for bit, between two worktrees across budgets, raw
# storage tiers and thread counts. Prints the counts and every line that differs.
#
# Usage: scripts/compare_ks_bits.sh <base-worktree> <new-worktree>
#
# The table test is copied into a base that lacks it, and removed afterwards.
set -euo pipefail
base=$1 new=$2
src="$new/crates/ferric-scf/tests/ks_bits_table.rs"
dest="$base/crates/ferric-scf/tests/ks_bits_table.rs"
out=$(mktemp -d)
copied=0
if [[ ! -e $dest ]]; then
  cp "$src" "$dest"
  copied=1
fi
trap '[[ $copied == 1 ]] && rm -f "$dest"; rm -rf "$out"' EXIT
run() {
  (cd "$1" && OPENBLAS_NUM_THREADS=1 cargo test -j 4 -p ferric-scf --test ks_bits_table \
    -- --ignored --nocapture --test-threads=1 2>/dev/null) \
    | grep -aoE 'BITS .*' | cut -c1-140 >"$2"
}
run "$base" "$out/base"
run "$new" "$out/new"
total=$(wc -l <"$out/new")
same=$(comm -12 <(sort "$out/base") <(sort "$out/new") | wc -l)
echo "lines: $total  identical: $same"
grep ERR "$out/base" | sed 's/^/base refused: /' | cut -c1-120 || true
grep ERR "$out/new" | sed 's/^/new refused:  /' | cut -c1-120 || true
diff "$out/base" "$out/new" || true
