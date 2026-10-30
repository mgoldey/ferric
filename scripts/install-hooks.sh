#!/usr/bin/env bash
# Install the G1 correctness gate as a git pre-push hook.
#
# Git hooks are NOT tracked by git and are NOT auto-installed by cloning --
# this script is the manual install step. Run it once per checkout/worktree:
#
#   scripts/install-hooks.sh
#
# Hooks are SHARED across worktrees: `git rev-parse --git-path hooks` resolves
# to the *common* .git/hooks dir even from inside a linked worktree (verified
# 2026-07-16), unless core.hooksPath is set. So installing once from any
# checkout wires the hook for every worktree of this repo. This script still
# has to be run explicitly because hooks are local, untracked state.
#
# What gets installed: a thin pre-push hook that gates THE WORKTREE BEING
# PUSHED. git runs pre-push with the current directory at the top of the
# worktree `git push` was invoked in, so the hook resolves that tree with
# `git rev-parse --show-toplevel` and execs ITS scripts/ci-gate.sh (which cds
# to its own repo root). Nothing is baked in at install time.
#
# HISTORY (why this matters): until 2026-08-28 the hook hardcoded the
# INSTALLER's scripts/ci-gate.sh "for a consistent path", and ci-gate.sh cds
# to its own root -- so a push from ~/qc/ferric-<lane> ran the fast gate
# against ~/qc/ferric, i.e. a tree that was not being pushed. A missing gate
# script also `exit 0`'d silently (that rot was hit on 2026-07-18). Both are
# gone: the hook now gates the pushing worktree and BLOCKS if it cannot find
# the gate there (override once with PRE_PUSH_ALLOW_NO_GATE=1).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks)"
# rev-parse --git-path may return a path relative to $REPO_ROOT; normalize.
case "$HOOKS_DIR" in
    /*) : ;;
    *) HOOKS_DIR="$REPO_ROOT/$HOOKS_DIR" ;;
esac

HOOK_PATH="$HOOKS_DIR/pre-push"

mkdir -p "$HOOKS_DIR"

# Overwrite only a hook this script installed (identified by its marker
# line); refuse to clobber a hand-written one. (The previous guard refused
# any existing non-symlink, i.e. its own earlier output -- so the hook could
# never be re-installed to pick up a wrapper fix without rm'ing it first.)
MARKER="Installed by scripts/install-hooks.sh"
if [[ -e "$HOOK_PATH" ]] && ! grep -qF "$MARKER" "$HOOK_PATH"; then
    echo "error: $HOOK_PATH already exists and was not installed by this script." >&2
    echo "       Move it aside first if you want install-hooks.sh to take over," >&2
    echo "       or merge its contents with the gate script manually." >&2
    exit 1
fi

cat > "$HOOK_PATH.tmp" <<'HOOK_EOF'
#!/usr/bin/env bash
# Installed by scripts/install-hooks.sh -- DO NOT hand-edit; edit
# scripts/install-hooks.sh (this wrapper) or scripts/ci-gate.sh (the gate)
# and re-run install-hooks.sh.
#
# git passes ref update info on stdin as:  <local ref> <local sha> <remote ref> <remote sha>
#
# The gate can only ever check THE WORKING TREE IT RUNS IN. So if you push a
# branch that is not the one checked out here, the gate validates code that
# has nothing to do with what you are pushing. On 2026-09-17 that cost an
# hour: `git push origin fix/rohf-guess-selection` was run from a checkout
# sitting on a DIFFERENT branch with uncommitted edits, so the gate spent the
# whole run on unrelated dirty code, failed on it, and blocked two branches it
# had never examined.
#
# We cannot gate a ref that is not checked out, so we REFUSE instead of
# silently gating the wrong thing. Push from that branch's own worktree.
set -uo pipefail

# git sets the cwd of a pre-push hook to the top of the worktree being pushed
# (for linked worktrees too). Resolve it explicitly rather than trusting the
# cwd, so the gate cannot silently run against a different checkout.
TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$TOPLEVEL" ]]; then
    echo "pre-push hook: cannot resolve the worktree top level -- refusing to push ungated." >&2
    echo "               (git rev-parse --show-toplevel failed; PRE_PUSH_ALLOW_NO_GATE=1 to bypass once)" >&2
    [[ "${PRE_PUSH_ALLOW_NO_GATE:-0}" == "1" ]] && exit 0
    exit 1
fi
GATE_SCRIPT="$TOPLEVEL/scripts/ci-gate.sh"

if [[ ! -x "$GATE_SCRIPT" ]]; then
    echo "pre-push hook: $GATE_SCRIPT not found or not executable in the worktree being pushed." >&2
    echo "               Refusing to push ungated. (This tree predates the gate, or scripts/ moved.)" >&2
    echo "               Bypass once with PRE_PUSH_ALLOW_NO_GATE=1 git push, or git push --no-verify." >&2
    [[ "${PRE_PUSH_ALLOW_NO_GATE:-0}" == "1" ]] && exit 0
    exit 1
fi

# Tier selection. The DEFAULT for a push is the LINT tier: clippy on the CHANGED
# crates (--all-targets, so it type-checks their targets; dependents are left to
# CI, see CLIPPY SCOPE in ci-gate.sh), the complexity gate and rustdoc, with BOTH test steps (cargo test and the soft pytest) skipped.
#
# Why tests are not in the default push gate: CI runs the full suite on a
# dedicated machine, in parallel, and a local pre-push run is therefore pure
# serialized cost -- it also holds the cargo lock, blocking every other
# worktree on the box. The lint tier is ~1 minute and catches the classes CI
# is SLOWEST to report. Rustdoc alone is the headline case: ~15s here versus
# ~50min of CI, and four separate rustdoc breaks reached CI on 2026-09-17
# precisely because the all-or-nothing gate was being bypassed wholesale with
# --no-verify. A gate people actually run beats a thorough one they skip.
#
# The soft pytest step is skipped in lint for the same reason, and more so:
# it cannot block a push (it is advisory), CI runs the same binding suite
# ("pytest -- Python bindings, with coverage"), and locally it took 34-55 min
# on a loaded box (2026-10-29) while holding target/release/libferric.so open,
# so no rebuild in that worktree could start until it finished.
#
# Escalate with CI_GATE_TIER (or set the CI_GATE_* knobs directly):
#   lint  (default)  clippy + complexity + rustdoc, no tests
#   fast             lint + cargo tests (slowest binaries deferred) + pytest
#   full             everything, including workspace-wide clippy
# ---- refuse to gate a tree that is not what is being pushed -------------
# Compare each pushed ref's LOCAL SHA against this worktree's HEAD. A match
# means the gate is about to check the right code. A mismatch means it is
# not, and a PASS here would be meaningless.
#
# Deletes (all-zero local sha) are skipped: nothing is being gated.
# Comparing SHAs rather than branch NAMES handles a renamed remote branch
# (`git push origin local:remote`) and a detached HEAD correctly.
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
MISMATCHED=""
while read -r local_ref local_sha remote_ref remote_sha; do
    [[ -z "${local_ref:-}" ]] && continue
    [[ "$local_sha" =~ ^0+$ ]] && continue          # branch deletion
    if [[ -n "$HEAD_SHA" && "$local_sha" != "$HEAD_SHA" ]]; then
        MISMATCHED+="  $local_ref -> ${remote_ref:-?}"$'\n'
    fi
done
if [[ -n "$MISMATCHED" ]]; then
    echo "pre-push hook: REFUSING -- you are pushing ref(s) that are not checked out here." >&2
    echo "" >&2
    echo "  worktree:  $TOPLEVEL" >&2
    echo "  on branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ($HEAD_SHA)" >&2
    echo "  pushing:" >&2
    printf '%s' "$MISMATCHED" >&2
    echo "" >&2
    echo "  The gate can only check THE WORKING TREE IT RUNS IN, so it would" >&2
    echo "  validate this checkout's code and tell you nothing about what you" >&2
    echo "  are actually pushing. A PASS would be meaningless." >&2
    echo "" >&2
    echo "  Push from that branch's own worktree:  git worktree list" >&2
    echo "  Or bypass deliberately:                git push --no-verify" >&2
    exit 1
fi

# Even on the right branch, uncommitted changes mean the gate is checking
# something other than the commits being pushed. That is usually what you
# want locally (you are testing your work in progress), so this WARNS rather
# than refusing -- but it says so, because a PASS on a dirty tree is not a
# statement about the pushed commits.
if ! git diff --quiet HEAD 2>/dev/null; then
    echo "pre-push: NOTE -- working tree is dirty, so the gate is checking your" >&2
    echo "          uncommitted state, not exactly the commits being pushed." >&2
fi

CI_GATE_TIER="${CI_GATE_TIER:-lint}"
case "$CI_GATE_TIER" in
    lint) export CI_GATE_SKIP_TESTS=1; export CI_GATE_SKIP_PYTEST=1; export CI_GATE_FAST=1 ;;
    fast) export CI_GATE_FAST=1 ;;
    full) export CI_GATE_FAST=0 ;;  # ci-gate.sh reads CI_GATE_TIER=full for clippy
    *)
        echo "pre-push hook: unknown CI_GATE_TIER='$CI_GATE_TIER'" >&2
        echo "               expected one of: lint, fast, full" >&2
        exit 1
        ;;
esac

echo "pre-push: gating worktree $TOPLEVEL ($CI_GATE_TIER tier)" >&2
echo "          $GATE_SCRIPT" >&2
if [[ "$CI_GATE_TIER" == "lint" ]]; then
    echo "          Tests (cargo test AND pytest) are NOT run here -- CI runs them." >&2
    echo "          This tier runs clippy (changed crates)/complexity/rustdoc, which CI is slowest to report." >&2
fi
echo "          Escalate:  CI_GATE_TIER=fast git push   (lint + most tests + pytest)" >&2
echo "                     CI_GATE_TIER=full git push   (everything)" >&2
echo "          Skip once: git push --no-verify         (only if you mean it)" >&2
"$GATE_SCRIPT"
exit $?
HOOK_EOF

chmod +x "$HOOK_PATH.tmp"
mv "$HOOK_PATH.tmp" "$HOOK_PATH"

echo "Installed pre-push hook: $HOOK_PATH"
echo "  -> runs: <worktree being pushed>/scripts/ci-gate.sh (resolved at push time)"
echo
echo "This hook is shared across all worktrees of this repo (git hooks live in"
echo "the common .git dir, not per-worktree). Every 'git push' from any"
echo "worktree will now run the gate first."
echo
echo "To bypass once (e.g. box is under heavy load, see scripts/README.md):"
echo "  git push --no-verify"
echo "To uninstall:"
echo "  rm '$HOOK_PATH'"

# ---- pre-COMMIT hooks (bandit / ruff-critical / shellcheck / detect-secrets)
#
# Managed by the pre-commit framework against .pre-commit-config.yaml, not by
# a bespoke wrapper: the framework owns per-hook tool environments (bandit and
# and shellcheck are NOT on the box; pre-commit fetches them pinned),
# runs only against STAGED files, and its generated hook works from linked
# worktrees the same way the pre-push wrapper does (hooks dir is common).
#
# Soft-skip when the binary is missing, matching ci-gate.sh's convention for
# machine-local dev tools: the pre-push gate above is the hard floor, commit
# hooks are an earlier, cheaper tripwire.
echo
# `pre-commit install` hard-refuses when core.hooksPath is set. On this box
# the local config sets it to the COMMON hooks dir -- i.e. the default
# location, a behavioral no-op -- so unsetting it changes nothing and lets
# the install proceed. A hooksPath pointing anywhere ELSE is a real user
# choice we must not override: warn and skip the commit hooks instead.
HOOKS_PATH_CFG="$(git -C "$REPO_ROOT" config --get core.hooksPath || true)"
if [[ -n "$HOOKS_PATH_CFG" ]]; then
    if [[ "$(readlink -f "$HOOKS_PATH_CFG" 2>/dev/null)" == "$(readlink -f "$HOOKS_DIR")" ]]; then
        git -C "$REPO_ROOT" config --unset-all core.hooksPath
        echo "note: removed redundant core.hooksPath (= the default common hooks dir) so pre-commit can install."
    else
        echo "NOTE: core.hooksPath is set to '$HOOKS_PATH_CFG' (not the default hooks dir)."
        echo "      Skipping commit-hook install -- pre-commit refuses under a custom hooksPath,"
        echo "      and overriding your config is not this script's call. Unset it and re-run if wanted."
    fi
fi
if [[ -z "$(git -C "$REPO_ROOT" config --get core.hooksPath || true)" ]] && command -v pre-commit >/dev/null 2>&1; then
    # `pre-commit install` refuses nothing: an existing hand-written
    # pre-commit hook is moved to pre-commit.legacy and still chained, so the
    # marker-guard used for pre-push above is unnecessary here.
    (cd "$REPO_ROOT" && pre-commit install --install-hooks)
    echo "Installed pre-commit hook (bandit -ll, ruff critical subset,"
    echo "shellcheck --severity=error, detect-secrets, merge-conflict/large-file/yaml/toml checks)."
    echo "To bypass once: git commit --no-verify"
else
    echo "NOTE: 'pre-commit' not found -- commit hooks NOT installed (pre-push gate is unaffected)."
    echo "      Install with: uv tool install pre-commit   then re-run scripts/install-hooks.sh"
fi
