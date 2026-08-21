#!/usr/bin/env bash
# Interleaved ("paired") variant of the ratio sweep, for a box with OTHER
# people's jobs on it that come and go.
#
# The plain sweep runs 1x6, then 2x3, then 3x2, then 6x1 back to back. If a
# background job starts halfway through, it lands entirely on the later
# configurations and shows up as a fake hybrid loss -- a sequential sweep
# CONFOUNDS configuration with time.
#
# This driver instead runs one ROUND of all four configurations, then repeats
# the whole round N times, and reports the per-configuration MINIMUM across
# rounds. Two properties matter:
#
#   * Interleaving spreads any drifting load across every configuration
#     instead of concentrating it on whichever ran last.
#   * Wall-clock noise from contention is ONE-SIDED (an interfering process can
#     only make a run slower), so the min over rounds is the best available
#     estimate of the contention-free cost. It is still an UPPER bound on the
#     true cost -- report it as such, never as a clean-box number.
set -u
BIN="$1"
TEST="$2"
ROUNDS="${3:-5}"
export OPENBLAS_NUM_THREADS=1
MPIRUN="${MPIRUN:-$(command -v mpirun || echo "$HOME/.local/bin/mpirun")}"
OUT=$(mktemp)

for r in $(seq 1 "$ROUNDS"); do
  for cfg in "1 6" "2 3" "3 2" "6 1"; do
    set -- $cfg
    ranks="$1"; pe="$2"
    line=$(timeout 1800 "$MPIRUN" -np "$ranks" --bind-to core --map-by "slot:PE=$pe" \
             "$BIN" --nocapture --test-threads=1 --ignored "$TEST" 2>&1 \
           | grep -E "RATIO_SWEEP" | head -1)
    if [ -n "$line" ]; then
      secs=$(echo "$line" | grep -o 'secs=[0-9.]*' | cut -d= -f2)
      rss=$(echo "$line" | grep -o 'peak_rss_mib=[0-9.]*' | cut -d= -f2)
      corr=$(echo "$line" | grep -o 'mp2_corr=[-0-9.]*' | cut -d= -f2)
      echo "${ranks}x${pe} $secs $rss $corr" >> "$OUT"
      echo "  round $r  ${ranks}x${pe}  secs=$secs rss=$rss"
    else
      echo "  round $r  ${ranks}x${pe}  FAILED/no output"
    fi
  done
done

echo
echo "=== per-configuration MIN over $ROUNDS rounds (upper bound on clean-box time) ==="
sort -k1,1 "$OUT" | awk '
  { if (!($1 in m) || $2+0 < m[$1]) { m[$1]=$2+0; rss[$1]=$3; corr[$1]=$4 } }
  END { for (k in m) printf "%s min_secs=%.4f peak_rss_mib=%s mp2_corr=%s\n", k, m[k], rss[k], corr[k] }
' | sort -t= -k2 -g
rm -f "$OUT"
