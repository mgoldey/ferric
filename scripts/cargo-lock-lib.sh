#!/usr/bin/env bash
# Shared cargo-lock helpers. SOURCE this, do not execute it:
#     source "$(dirname "${BASH_SOURCE[0]}")/cargo-lock-lib.sh"
#
# Why this exists: scripts/ci-gate.sh serializes cargo behind
# `flock -w 14400 /tmp/ferric-cargo.lock`. Before #591 (no -o), the sccache
# server that cargo auto-spawns INHERITED that lock file descriptor and outlives the cargo run
# that spawned it, so the NEXT build queues behind a daemon that will never
# release -- a silent 4-hour dead wait that reads as "the box is slow".
#
# Hit SIX times on 2026-08-17. The gate got a pre-flight first; ad-hoc scripts
# in scripts/queue/ that call `flock` directly kept hitting it because the fix
# lived only inside ci-gate.sh. Hence this file.
#
# The tell: idle box + lock held + zero rustc.
#
# Root fix (2026-10-28): every `flock ... -c` here and in ci-gate.sh uses -o,
# which closes the lock fd in the command, so sccache can no longer inherit it.
# The clearer below stays as a backstop. It used to count every process with
# the file OPEN (fuser), so queued gates' waiting `flock`s blocked it: five
# queued pushes sat behind one squatting sccache for 30+ minutes that way. It
# now looks only at the processes that HOLD the lock (ferric_cargo_lock_holders).

: "${FERRIC_CARGO_LOCK:=/tmp/ferric-cargo.lock}"
: "${FERRIC_CARGO_LOCK_WAIT_SECS:=14400}"

# Print the pids that HOLD the flock on $1, one per line, lowest first.
#
# /proc/locks cannot answer this. It records the pid that CALLED flock(2), and
# for the fd form (`flock -o -w N "$fd"`, ci-gate.sh's run_locked) that is the
# flock(1) program, which exits as soon as the lock is taken; the lock lives on
# in the subshell that owns the fd (measured: /proc/locks named a dead pid).
# A "lock:" line in /proc/PID/fdinfo/FD appears only on an open file that holds
# the lock -- a queued waiter's file shows none -- so the processes listing one
# are the live holders (every process sharing that open file). Other users'
# processes are not readable and do not appear.
ferric_cargo_lock_holders() {
    local lock="${1:-$FERRIC_CARGO_LOCK}" dev ino
    dev="$(stat -L -c '%Hd %Ld' "$lock" 2>/dev/null)" || return 0
    ino="$(stat -L -c '%i' "$lock")"
    # shellcheck disable=SC2086  # split "major minor" into two printf args
    dev="$(printf '%02x:%02x' $dev)"
    # printf is a builtin, so a box with many open fds cannot hit ARG_MAX.
    printf '%s\0' /proc/[0-9]*/fdinfo/* \
        | xargs -0 grep -ls -E "^lock:.* $dev:$ino " 2>/dev/null \
        | cut -d/ -f3 | sort -nu
}

# Clear a squatting sccache daemon off the cargo lock, if and only if sccache
# is ALL that holds it. Safe to call unconditionally; no-op when the lock is
# free or held by anything else.
#
# DISCRIMINATOR (arrived at by testing, not reasoning): parentage does NOT
# work. sccache is a daemon and is ALWAYS reparented to systemd -- INCLUDING
# while actively serving a live build -- so a "PPID 1 means stale" heuristic
# kills running compiles. Verified 2026-08-17 against a live mid-build sccache
# showing ppid=systemd; that first draft would have killed it.
#
# What DOES work: who HOLDS the lock. A build started without -o shares the
# locked open file with its cargo/rustc/flock, so they hold it too; ci-gate.sh's
# run_locked holds it in a bash subshell. sccache ALONE among the holders means
# the build that spawned it is gone and the daemon is squatting. Queued waiters
# do not hold the lock, so they no longer protect a squatter (they used to:
# the old check counted every process with the file open). An sccache that has
# the file open but does NOT hold the lock is serving whoever does, and is left
# alone. Killing a squatter costs nothing -- sccache is a CACHE, its on-disk
# store survives a restart and cargo respawns the server.
ferric_clear_stale_sccache_lock() {
    local lock="${1:-$FERRIC_CARGO_LOCK}"
    [[ -e "$lock" ]] || return 0

    local pid comm killed=0 holders=() sccache_pids=()
    mapfile -t holders < <(ferric_cargo_lock_holders "$lock")
    for pid in "${holders[@]}"; do
        comm="$(ps -o comm= -p "$pid" 2>/dev/null || true)"
        case "$comm" in
            sccache) sccache_pids+=("$pid") ;;
            "")      ;;          # exited while we looked
            *)       return 0 ;; # a real holder: a build, or a gate's subshell
        esac
    done

    for pid in "${sccache_pids[@]:-}"; do
        [[ -n "$pid" ]] || continue
        echo "   note: releasing stale sccache lock holder (pid $pid, no build in flight)"
        kill "$pid" 2>/dev/null && killed=1
    done
    [[ $killed -eq 1 ]] && sleep 2
    return 0
}

# flock's status when the wait expires (its default, 1, looks like a failed build).
: "${FERRIC_CARGO_LOCK_WAIT_EXPIRED_RC:=75}"

# Run a command under the cargo lock, clearing a stale holder first.
# Usage: ferric_cargo_locked "OPENBLAS_NUM_THREADS=1 cargo test --workspace -j 6"
# Returns the command's status, or FERRIC_CARGO_LOCK_WAIT_EXPIRED_RC (75) if the
# lock wait expired and the command never ran. A caller that adds a timeout
# should put it INSIDE, around the command, as ci-gate.sh's run_locked does --
# wrapping this call would spend the timeout queueing.
ferric_cargo_locked() {
    ferric_clear_stale_sccache_lock
    flock -o -w "$FERRIC_CARGO_LOCK_WAIT_SECS" -E "$FERRIC_CARGO_LOCK_WAIT_EXPIRED_RC" \
        "$FERRIC_CARGO_LOCK" -c "$*"
}

# /tmp is wiped at boot and a dead PATH entry falls through to the real cargo without a word, so the
# compile locks and the box lease silently stop applying. Reinstall a missing shim from this tree, and
# warn loudly when this process's cargo would bypass it. Never fails the caller.
ferric_ensure_cargo_shim() { # REPO_ROOT
    local repo="$1" dir="${FERRIC_SHIM_DIR:-/tmp/ferric-cargo-shim}"
    if [[ ! -x "$dir/cargo" ]]; then
        echo "WARNING: the cargo shim is missing from $dir (wiped at boot?); reinstalling." >&2
        "$repo/scripts/install-cargo-shim.sh" >&2 ||
            echo "WARNING: could not reinstall it: cargo here takes no compile lock and ignores box leases." >&2
    fi
    case ":$PATH:" in
        *":$dir:"*) ;;
        *) echo "WARNING: $dir is not on PATH: cargo here takes no compile lock and ignores box leases." >&2 ;;
    esac
    return 0
}
