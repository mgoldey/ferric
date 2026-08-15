#!/usr/bin/env bash
# Block until the box is quiet enough for a trustworthy wall-clock measurement.
# Timing on a loaded box measures the scheduler, not the code.
set -u
for _ in $(seq 1 60); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  A=$(free -g | awk '/^Mem:/{print $7}')
  if awk "BEGIN{exit !($L < 4.0)}" && [ "$A" -ge 6 ]; then
    echo "BOX_QUIET load=$L avail_gb=$A"
    exit 0
  fi
  sleep 30
done
echo "BOX_STILL_BUSY load=$(cut -d' ' -f1 /proc/loadavg) avail_gb=$(free -g | awk '/^Mem:/{print $7}')"
exit 1
