#!/usr/bin/env bash
# Install the G1 correctness gate as a git pre-push hook.
#
# Git hooks are NOT tracked by git and are NOT auto-installed by cloning --
# this script is the manual install step. Run it once per checkout/worktree:
#
#   scripts/install-hooks.sh
#
# WHY per-worktree even though hooks are usually shared: `git rev-parse
# --git-path hooks` resolves to the *common* .git/hooks dir even from inside
# a linked worktree (verified 2026-07-16: this repo's worktrees, e.g.
# .claude/worktrees/g1-ci-gate, all resolve to the same top-level
# /path/to/ferric/.git/hooks -- hooks are shared across ALL worktrees of a
# repo by default, unless core.hooksPath is set). So installing once from
# any checkout wires the hook for every worktree that shares this .git.
# This script still needs to be run explicitly because hooks are local,
# untracked state -- there is no way to make git auto-install them.
#
# What gets installed: a thin pre-push hook that execs scripts/ci-gate.sh
# from the repo's main worktree (not the linked worktree that happened to
# install it), so the gate always runs against a consistent path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks)"
# rev-parse --git-path may return a path relative to $REPO_ROOT; normalize.
case "$HOOKS_DIR" in
    /*) : ;;
    *) HOOKS_DIR="$REPO_ROOT/$HOOKS_DIR" ;;
esac

HOOK_PATH="$HOOKS_DIR/pre-push"
GATE_SCRIPT="$REPO_ROOT/scripts/ci-gate.sh"

mkdir -p "$HOOKS_DIR"

if [[ -e "$HOOK_PATH" && ! -L "$HOOK_PATH" ]]; then
    echo "error: $HOOK_PATH already exists and is not a symlink we manage." >&2
    echo "       Move it aside first if you want install-hooks.sh to take over," >&2
    echo "       or merge its contents with scripts/ci-gate.sh manually." >&2
    exit 1
fi

cat > "$HOOK_PATH.tmp" <<'HOOK_EOF'
#!/usr/bin/env bash
# Installed by scripts/install-hooks.sh -- DO NOT hand-edit, edit
# scripts/ci-gate.sh instead and re-run install-hooks.sh if this wrapper
# itself needs to change.
#
# git passes ref update info on stdin; we don't need it -- we always run the
# full gate on the current working tree before allowing any push to proceed.
set -uo pipefail
GATE_SCRIPT="__GATE_SCRIPT__"

if [[ ! -x "$GATE_SCRIPT" ]]; then
    echo "pre-push hook: $GATE_SCRIPT not found or not executable -- skipping gate." >&2
    echo "                (repo layout changed? re-run scripts/install-hooks.sh)" >&2
    exit 0
fi

echo "pre-push: running correctness gate ($GATE_SCRIPT)..." >&2
echo "          (skip once with: git push --no-verify -- only if you mean it)" >&2
"$GATE_SCRIPT"
exit $?
HOOK_EOF

sed -i "s|__GATE_SCRIPT__|$GATE_SCRIPT|" "$HOOK_PATH.tmp"
chmod +x "$HOOK_PATH.tmp"
mv "$HOOK_PATH.tmp" "$HOOK_PATH"

echo "Installed pre-push hook: $HOOK_PATH"
echo "  -> runs: $GATE_SCRIPT"
echo
echo "This hook is shared across all worktrees of this repo (git hooks live in"
echo "the common .git dir, not per-worktree). Every 'git push' from any"
echo "worktree will now run the gate first."
echo
echo "To bypass once (e.g. box is under heavy load, see scripts/README.md):"
echo "  git push --no-verify"
echo "To uninstall:"
echo "  rm '$HOOK_PATH'"
