#!/bin/bash
# Unattended overnight chain: T sweep, then S22, one job at a time.
#
# WHY A CHAIN AND NOT TWO DRIVERS
#
# Both drivers now run RAYON_NUM_THREADS=12 / NPROC=1 -- one job using all 12
# cores. Launching them concurrently would put 24 threads on 12 cores, which is
# the oversubscription behind both of today's box crashes. So they must run
# SEQUENTIALLY, and something has to start the second when the first ends.
# Unattended, that something cannot be a human.
#
# WHAT THIS GUARANTEES
#
# Nothing is lost to a crash. Every ferric job writes its own output file and
# every driver skips jobs whose output is already complete, so a reboot costs
# at most the single in-flight job. That is the whole reason for NPROC=1:
# one job resident means one job at risk.
#
# ORDER: T first (43 jobs, 85 points), then S22 (60 jobs, 60 points). T is
# closer to a publishable result -- it needs a full 24-system curve to settle
# whether its n=4 minimum of 1.433 A survives, the same way B's subset estimate
# moved from 0.835 to 0.893 once all 24 landed. S22 is exploratory: it tests
# whether B's pi-stack advantage (the only real effect found on A24) shows up
# on a benchmark with more pi-stacked cases.
#
# S22 runs SMALLEST FIRST, so if the large systems turn out to be infeasible on
# this box we still wake up with a usable small-to-medium set.
#
# This script is itself restartable: re-running it re-runs both drivers, which
# skip completed work. Safe to launch again after any crash.
cd /home/matt/qc/ferric

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

log "=== overnight chain starting ==="
log "box: $(nproc) cores, $(free -m | awk '/^Mem:/{print $7}') MB available"

log "--- stage 1/2: T sweep (r0 1.40/1.50) ---"
# A T driver may ALREADY be running (this chain is often started while one is
# in flight). Killing it would discard the in-flight job for nothing, and
# starting a second would double the thread count on 12 cores. So: wait for the
# existing one if present, else start our own.
existing=$(pgrep -f 'run_r0tpm\.sh' | head -1)
if [ -n "$existing" ]; then
  log "T driver already running (pid $existing) -- waiting for it"
  while kill -0 "$existing" 2>/dev/null; do sleep 60; done
  log "existing T driver exited"
else
  bash scripts/queue/run_r0tpm.sh
fi
log "T stage complete"

# Between stages: report, and refuse to start stage 2 if the box is unhealthy.
# A crash-loop that keeps relaunching into a sick machine wastes the night.
avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
log "post-T: ${avail} MB available"
if [ "$avail" -lt 4000 ]; then
  log "ABORT: only ${avail} MB free, refusing to start S22 (needs ~6 GB/job)"
  exit 1
fi

log "--- stage 2/2: S22 for B at r0=0.90, smallest first ---"
bash scripts/queue/run_s22b.sh
log "S22 driver exited"

log "=== overnight chain done ==="
