#!/usr/bin/env bash
# Sequential tail-sweep chain: C20 -> C32 -> C48 via run_tail_piece.sh,
# each under its own ferric-limited cap. Meant for nohup/setsid launch.
set -uo pipefail
cd "$(dirname "$0")"
log="$1"
for nc in 20 32 48; do
  if ./run_tail_piece.sh "$nc" "$log"; then
    echo "CHAIN: C$nc ok" >>"$log"
  else
    echo "CHAIN: C$nc FAILED (exit $?)" >>"$log"
  fi
done
echo "CHAIN-DONE" >>"$log"
