#!/bin/bash
# Shared memory admission gate for the queue runners.  Source it:
#
#   . scripts/queue/memgate.sh
#   mem_wait 2600 || echo "gave up waiting"
#
# WHY THIS EXISTS
#
# Every runner had its own copy of
#
#     for _ in $(seq 1 90); do
#       [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && break
#       sleep 60
#     done
#
# and that gate has three independent bugs, all of which fired on 2026-07-26
# and stalled the broad set for 18 minutes with an idle 3-wide xargs:
#
#  1. `free -g` TRUNCATES.  6137 MB available prints as 5, so a `-ge 8` test
#     is really "wait for 9 GB".  Up to 1 GB of headroom is invisible.
#
#  2. It gates on TOTAL system availability when the real constraint is the
#     PER-SLOT cgroup cap (`ferric-limited --max=5G`).  A job needing 2.2 GB
#     was refused because the box as a whole was below an 8 GB line that
#     nothing in the job's own budget referred to.
#
#  3. Availability is the WRONG SIGNAL for whether the box is struggling.
#     `MemAvailable` counts reclaimable page cache as unavailable-ish and
#     drops as healthy jobs warm up; it says nothing about whether anything
#     is actually stalling.  The kernel exposes the real answer in
#     /proc/pressure/memory: PSI `some` is the share of wall time at least
#     one task spent blocked on memory.  On the stalled run PSI read 0.00
#     across all three windows while si/so were flat zero -- the box was CPU
#     saturated (run queue 40-51, 98% user), not memory starved.  See
#     [[free-ram-is-not-free-cpu]].
#
# So: gate on need + pressure, in MB, not on a rounded global floor.
#
# mem_wait <need_mb> [max_wait_s] [psi_limit]
#   Waits until BOTH:
#     * MemAvailable >= need_mb + MEMGATE_HEADROOM_MB (default 1024), and
#     * /proc/pressure/memory `some avg10` < psi_limit (default 10.0)
#   Returns 0 when admitted, 1 on timeout (caller decides; every current
#   caller proceeds anyway, matching the old `for` loop's behaviour).
#
# Set MEMGATE_DEBUG=1 to log each poll.

mem_avail_mb() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }

# `some avg10` from PSI: percent of the last 10 s in which at least one task
# was stalled on memory.  Absent (kernel without PSI) -> 0, i.e. never blocks.
mem_psi_some10() {
  awk '/^some/ {sub(/avg10=/, "", $2); print $2; found=1}
       END {if (!found) print "0"}' /proc/pressure/memory 2>/dev/null || echo 0
}

mem_wait() {
  local need_mb="${1:?mem_wait needs a size in MB}"
  local max_wait="${2:-5400}"
  local psi_limit="${3:-10.0}"
  local headroom="${MEMGATE_HEADROOM_MB:-1024}"
  local want=$(( need_mb + headroom ))
  local waited=0 avail psi

  while :; do
    avail=$(mem_avail_mb)
    psi=$(mem_psi_some10)
    if [ "$avail" -ge "$want" ] && awk -v p="$psi" -v l="$psi_limit" \
         'BEGIN{exit !(p < l)}'; then
      [ -n "$MEMGATE_DEBUG" ] && \
        echo "[mem  ] admit: avail=${avail}MB want=${want}MB psi=${psi}"
      return 0
    fi
    if [ "$waited" -ge "$max_wait" ]; then
      echo "[mem  ] TIMEOUT after ${waited}s: avail=${avail}MB want=${want}MB psi=${psi}" >&2
      return 1
    fi
    [ -n "$MEMGATE_DEBUG" ] && \
      echo "[mem  ] wait ${waited}s: avail=${avail}MB want=${want}MB psi=${psi}"
    sleep 30
    waited=$(( waited + 30 ))
  done
}
