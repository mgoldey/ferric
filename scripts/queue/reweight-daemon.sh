#!/bin/bash
# Keep CPUWeight applied as ferric jobs rotate.
#
# `ferric-limited` runs each job in its own transient systemd scope, and a NEW
# scope starts at the default CPUWeight=100. So a weight set by hand decays the
# moment the runner moves to the next job -- which is why the critical-path
# share kept drifting back after each manual fix.
#
# This re-applies weights every 60 s. Classification is by TOML name, so it
# follows the work rather than any PID:
#   *_r0text_T.toml  -> T extension               (weight 200)
#   *_r0Bmin_B.toml  -> B-min sweep                (weight 100)
#   everything else  -> critical path              (weight 400)
#
# The catch-all is deliberate: the broad set (*_r0tscanB_T.toml) falls through
# to 400 and so automatically outranks both once it starts. That matches the
# 2026-07-26 reprioritization -- breadth (n 7 -> 20) dominates r0 resolution,
# because the optimum's uncertainty is set by system-to-system spread, not by
# grid spacing.
#
# Note the effect is modest, not the nominal 20:1 -- RAYON_NUM_THREADS caps
# each job near 400%, so a 2-job stage cannot exceed ~800% at any weight (see
# docs/queue-chaining-lessons.md, Addendum 3).
while :; do
  for p in $(pgrep -x ferric 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    cg=$(head -1 "/proc/$p/cgroup" 2>/dev/null | cut -d: -f3) || continue
    u=$(basename "$cg")
    case "$u" in *.scope) ;; *) continue;; esac
    # Priority follows what BLOCKS other work, and that changes as stages
    # finish. The T scan is done (2026-07-26 17:36), so the extension and the
    # B-min sweep are now peers -- neither gates the other, and weighting both
    # down leaves nothing prioritized. The extension is the shorter job and
    # completes the T story, so it leads.
    case "$c" in
      *_r0text_T.toml*)  w=200;;   # T extension: finishes the T deliverable
      *_r0Bmin_B.toml*)  w=100;;   # B-min: larger, no downstream dependents
      *)                 w=400;;   # anything still gating (broad set, etc.)
    esac
    cur=$(cat "/sys/fs/cgroup${cg}/cpu.weight" 2>/dev/null)
    [ "$cur" = "$w" ] && continue
    systemctl --user set-property "$u" CPUWeight=$w 2>/dev/null
  done
  sleep 60
done
