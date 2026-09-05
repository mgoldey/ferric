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
# git passes ref update info on stdin; we don't need it -- we always run the
# gate on the working tree of the worktree `git push` was invoked in.
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

# Fast tier: the pre-push hook defers the slowest integration binaries so a
# push is not blocked on the full suite; running scripts/ci-gate.sh by hand
# still runs EVERYTHING. The gate prints exactly which binaries it deferred,
# and its PASS line says "FAST TIER" so a fast run can never be misread as
# full coverage.
export CI_GATE_FAST=1
echo "pre-push: gating worktree $TOPLEVEL ($GATE_SCRIPT, fast tier)..." >&2
echo "          (skip once with: git push --no-verify -- only if you mean it)" >&2
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
