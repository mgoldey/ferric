#!/usr/bin/env bash
# Shared cargo-lock helpers. SOURCE this, do not execute it:
#     source "$(dirname "${BASH_SOURCE[0]}")/cargo-lock-lib.sh"
#
# Why this exists: scripts/ci-gate.sh serializes cargo behind
# `flock -w 14400 /tmp/ferric-cargo.lock`. The sccache server that cargo
# auto-spawns INHERITS that lock file descriptor and outlives the cargo run
# that spawned it, so the NEXT build queues behind a daemon that will never
# release -- a silent 4-hour dead wait that reads as "the box is slow".
#
# Hit SIX times on 2026-08-17. The gate got a pre-flight first; ad-hoc scripts
# in scripts/queue/ that call `flock` directly kept hitting it because the fix
# lived only inside ci-gate.sh. Hence this file.
#
# The tell: idle box + lock held + zero rustc.

: "${FERRIC_CARGO_LOCK:=/tmp/ferric-cargo.lock}"
: "${FERRIC_CARGO_LOCK_WAIT_SECS:=14400}"

# Clear a squatting sccache daemon off the cargo lock, if and only if no real
# build is using it. Safe to call unconditionally; no-op when the lock is free
# or genuinely busy.
#
# DISCRIMINATOR (arrived at by testing, not reasoning): parentage does NOT
# work. sccache is a daemon and is ALWAYS reparented to systemd -- INCLUDING
# while actively serving a live build -- so a "PPID 1 means stale" heuristic
# kills running compiles. Verified 2026-08-17 against a live mid-build sccache
# showing ppid=systemd; that first draft would have killed it.
#
# What DOES work: whether a real consumer also holds the lock. A running build
# puts cargo/rustc on it; a QUEUED build puts its own `flock` on it (observed
# directly). sccache ALONE means nothing is building or waiting, so the daemon
# is squatting and is safe to clear. Killing it costs nothing -- sccache is a
# CACHE, its on-disk store survives a restart and cargo respawns the server.
ferric_clear_stale_sccache_lock() {
    local lock="${1:-$FERRIC_CARGO_LOCK}"
    [[ -e "$lock" ]] || return 0
    command -v fuser >/dev/null 2>&1 || return 0

    local holders pid comm killed=0
    local sccache_pids=() has_real_consumer=0
    holders="$(fuser "$lock" 2>/dev/null || true)"
    for pid in $holders; do
        comm="$(ps -o comm= -p "$pid" 2>/dev/null || true)"
        case "$comm" in
            sccache)           sccache_pids+=("$pid") ;;
            cargo|rustc|flock) has_real_consumer=1 ;;
        esac
    done

    [[ $has_real_consumer -eq 1 ]] && return 0

    for pid in "${sccache_pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        echo "   note: releasing stale sccache lock holder (pid $pid, no build in flight)"
        kill "$pid" 2>/dev/null && killed=1
    done
    [[ $killed -eq 1 ]] && sleep 2
    return 0
}

# Run a command under the cargo lock, clearing a stale holder first.
# Usage: ferric_cargo_locked "OPENBLAS_NUM_THREADS=1 cargo test --workspace -j 6"
ferric_cargo_locked() {
    ferric_clear_stale_sccache_lock
    flock -w "$FERRIC_CARGO_LOCK_WAIT_SECS" "$FERRIC_CARGO_LOCK" -c "$*"
}
