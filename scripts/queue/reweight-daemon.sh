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
#   *_r0text_T.toml  -> extension, background      (weight 20)
#   *_r0Bmin_B.toml  -> B-min sweep, background    (weight 20)
#   everything else  -> critical path              (weight 400)
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
    case "$c" in
      *_r0text_T.toml*|*_r0Bmin_B.toml*) w=20;;
      *)                                 w=400;;
    esac
    cur=$(cat "/sys/fs/cgroup${cg}/cpu.weight" 2>/dev/null)
    [ "$cur" = "$w" ] && continue
    systemctl --user set-property "$u" CPUWeight=$w 2>/dev/null
  done
  sleep 60
done
