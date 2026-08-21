#!/bin/bash
# Final GW100 table assembler. Waits until ALL gw workers (both main depth sweeps
# AND the dedicated big-tail worker) have exited, then folds in:
#   1) the validated K-molecule def2-TZVP rows (inject_kmols.py)
#   2) the big-tail worker's rows from its raw driver logs (merge_driver_log.py)
# This sidesteps the shared-results-file write race: out-of-band workers write to
# their own logs, and we merge once nothing is mutating results_*.json anymore.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while systemctl --user list-units --type=scope --no-legend 2>/dev/null | grep -qE 'gw93d-(adz|atz)-|gwbig-'; do
  sleep 60
done

echo "ALL-WORKERS-DONE $(date +%H:%M) — assembling final table"
python3 scripts/gw100/inject_kmols.py

for log in scripts/queue/out/gw_bigtail_atz_*.txt; do
  [ -f "$log" ] && python3 scripts/gw100/merge_driver_log.py "$log" aug-cc-pvtz
done
for log in scripts/queue/out/gw_bigtail_adz_*.txt; do
  [ -f "$log" ] && python3 scripts/gw100/merge_driver_log.py "$log" aug-cc-pvdz
done

echo "ASSEMBLE-COMPLETE $(date +%H:%M)"
python3 - <<'PY'
import json
for b in ('aug-cc-pvdz','aug-cc-pvtz'):
    d=json.load(open(f'scripts/gw100/results_{b}.json'))
    acc=len(d['molecules'])+len(d.get('failed',[]))
    print(f'  {b}: {len(d["molecules"])} conv + {len(d.get("failed",[]))} fail = {acc}/93 | failed {sorted(d.get("failed",[]))}')
PY
