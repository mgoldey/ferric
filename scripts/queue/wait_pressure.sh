#!/usr/bin/env bash
# Block until memory PRESSURE has actually drained, not just until load drops.
#
# Load average and `free` both lie about recoverability here: load falls the
# instant the competing job exits, and `free` can show GBs "available" while
# swap is still 100% full and the kernel is refaulting. /proc/pressure/memory
# is the honest signal (see the memory-gates-must-use-psi-not-free convention).
#
# Gate on `full avg10` -- the fraction of the last 10s in which EVERY runnable
# task was stalled on memory. Low avg10 with a still-high avg300 is exactly the
# recovering state we want to wait through: the recent window is what predicts
# whether a new allocation will thrash.
set -u
for _ in $(seq 1 80); do
  FULL10=$(awk '/^full/{for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /proc/pressure/memory)
  SWAPFREE=$(free -m | awk '/^Swap:/{print $4}')
  AVAIL=$(free -g | awk '/^Mem:/{print $7}')
  if awk "BEGIN{exit !($FULL10 < 2.0)}" && [ "$SWAPFREE" -gt 200 ] && [ "$AVAIL" -ge 6 ]; then
    echo "PRESSURE_OK full_avg10=$FULL10 swap_free_mb=$SWAPFREE avail_gb=$AVAIL"
    exit 0
  fi
  sleep 15
done
echo "PRESSURE_STILL_HIGH full_avg10=$FULL10 swap_free_mb=$SWAPFREE avail_gb=$AVAIL"
exit 1
