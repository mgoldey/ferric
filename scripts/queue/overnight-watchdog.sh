#!/bin/bash
# Restart the overnight chain if it is not running. Idempotent; safe to call
# every few minutes from cron.
#
# WHY: the box crashed twice on 2026-07-27. Unattended, a crash at 02:00 would
# otherwise cost the whole night. The chain and both drivers already skip
# completed jobs, so relaunching is free -- at most the single in-flight job is
# redone.
#
# It deliberately does NOT restart if the chain is already up, and it takes a
# lock so two cron ticks cannot both launch one.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 0

exec 9>/tmp/ferric-overnight-watchdog.lock
flock -n 9 || exit 0

if pgrep -f 'run_overnight\.sh' >/dev/null 2>&1; then
  exit 0
fi

# Do not fight a driver that is still going on its own.
if pgrep -f 'run_r0tpm\.sh|run_s22b\.sh|run_r0tup\.sh' >/dev/null 2>&1; then
  exit 0
fi

# All work done? Then there is nothing to restart, and we should stop
# relaunching a chain that would immediately exit.
remaining=$(python3 - <<'PY' 2>/dev/null
from pathlib import Path
import re
t=Path("benchmarks/grid/toml"); o=Path("benchmarks/grid/out")
n=0
for tag in ("r0Tpm","s22b","r0Tup"):
    for f in t.glob(f"*{tag}*.toml"):
        m=re.search(r'r0_sweep = \[[^\]]*\]', f.read_text())
        if not m: continue
        w=len(re.findall(r'\d+\.\d+', m.group()))
        p=o/f"{f.stem}.out"
        c=p.read_text(errors="ignore").count("Total energy") if p.exists() else 0
        if c<w: n+=1
print(n)
PY
)
[ -z "$remaining" ] && remaining=1
if [ "$remaining" -eq 0 ]; then
  echo "[watchdog $(date +%H:%M)] all work complete, nothing to restart"
  exit 0
fi

echo "[watchdog $(date +%H:%M)] chain not running, $remaining jobs left -- relaunching"
nohup bash scripts/queue/run_overnight.sh \
  > "scripts/queue/overnight_$(date +%m%d_%H%M).log" 2>&1 &
