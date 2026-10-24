#!/usr/bin/env bash
# Download one artifact of THIS workflow run with stall detection and resume
# (#525). Used by ci.yml's test-shard and malloc-perturb jobs in place of a
# wall-clock-bounded actions/download-artifact.
#
#   scripts/ci-download-artifact.sh <artifact-name> <file-in-artifact> <dest-dir>
#
# Leaves <dest-dir>/<file-in-artifact> and exits 0 only when the file passed
# every integrity check; on any failure it removes the partial file and exits
# non-zero (ci.yml then tries actions/download-artifact once, and the run step
# fails closed if the file is still missing or corrupt).
#
# WHY NOT A WALL CLOCK. Measured on the 7.43 GB nextest-archive (run
# 40111474993): healthy shards download in 77-80 s (~95 MB/s); bad ones crawl
# at ~9 MB/s (831 s) or hang (900 s timeout, then the retry also 900 s). The
# distribution is bimodal, so a THROUGHPUT floor separates the two modes and
# fires in under a minute, where a wall clock has to sit at ~3x the healthy
# time. curl --speed-limit/--speed-time abandons a slow connection; the next
# attempt resumes with an HTTP Range request (-C -), so an abandoned attempt
# loses nothing it already fetched. Even a runner whose EVERY connection is
# slow therefore still finishes, at its own speed, inside the deadline.
#
# Integrity, in order: byte count == the API's size_in_bytes; sha256 of the
# zip == the API's `digest` (the same check actions/download-artifact makes);
# unzip's CRC32; then `zstd -t` on the extracted file (every frame), which is
# what the run step re-checks before it uses the file.
# Existence is never enough: a truncated download leaves a file that only
# fails later, inside `nextest --archive-file`, as "incomplete frame" (seen
# on #168, test-shard 3).
#
# Needs: GITHUB_TOKEN with actions:read, GITHUB_REPOSITORY, GITHUB_RUN_ID,
# and curl, jq, sha256sum, unzip, zstd (all on ubuntu-22.04 runners; zstd is
# apt-installed by the jobs).
#
# Tunables (env), defaults chosen from the numbers above:
#   CI_DL_DEADLINE_S      total time for all attempts            (1200)
#   CI_DL_SPEED_LIMIT     bytes/s floor for one connection   (25000000)
#   CI_DL_SPEED_TIME      seconds below the floor before abort     (45)
#   CI_DL_MAX_STUCK       consecutive attempts that each fetched less
#                         than CI_DL_SPEED_LIMIT bytes (a hang, or a
#                         trickle), before giving up early          (6)
set -euo pipefail

NAME=${1:?artifact name}
FILE=${2:?file inside the artifact}
DEST=${3:?destination directory}

: "${GITHUB_TOKEN:?GITHUB_TOKEN (actions:read) is required}"
: "${GITHUB_REPOSITORY:?}"
: "${GITHUB_RUN_ID:?}"
API=${GITHUB_API_URL:-https://api.github.com}
DEADLINE_S=${CI_DL_DEADLINE_S:-1200}
SPEED_LIMIT=${CI_DL_SPEED_LIMIT:-25000000}
SPEED_TIME=${CI_DL_SPEED_TIME:-45}
MAX_STUCK=${CI_DL_MAX_STUCK:-6}

ZIP="$DEST/.$NAME.download.zip"
OUT="$DEST/$FILE"
START=$SECONDS
elapsed() { echo $((SECONDS - START)); }
fail() {
  echo "::warning::ci-download-artifact: $NAME: $*"
  rm -f "$ZIP" "$OUT"
  exit 1
}
api() {
  curl -fsS --retry 3 --retry-all-errors --connect-timeout 20 --max-time 60 \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" "$@"
}

mkdir -p "$DEST"
rm -f "$OUT"
# Never leave a partial zip (up to 7.4 GB of a ~14 GB-free disk) behind for
# the fallback step, including when the step timeout interrupts us.
trap 'rm -f "$ZIP"' EXIT
trap 'exit 130' INT TERM

# Newest artifact of this name in this run (a re-run of build-tests can leave
# more than one).
META=$(api "$API/repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/artifacts?name=$NAME&per_page=100" \
  | jq -c '[.artifacts[] | select(.expired | not)] | sort_by(.created_at) | last // empty') \
  || fail "listing the run's artifacts failed"
[ -n "$META" ] || fail "no unexpired artifact named $NAME in run $GITHUB_RUN_ID"
ID=$(jq -r .id <<<"$META")
SIZE=$(jq -r .size_in_bytes <<<"$META")
DIGEST=$(jq -r '.digest // ""' <<<"$META")
echo "artifact $NAME id=$ID size=$SIZE digest=${DIGEST:-<none>}"

have() { stat -c %s "$ZIP" 2>/dev/null || echo 0; }

attempt=0
stuck=0
restarts=0
while :; do
  got=$(have)
  if [ "$got" -eq "$SIZE" ]; then
    if [ -z "$DIGEST" ] || [ "sha256:$(sha256sum "$ZIP" | cut -d' ' -f1)" = "$DIGEST" ]; then
      break
    fi
    echo "::warning::$NAME: sha256 mismatch after $(elapsed) s -- discarding and starting over"
    rm -f "$ZIP"
    restarts=$((restarts + 1))
    [ "$restarts" -le 1 ] || fail "sha256 mismatch twice"
    continue
  fi
  if [ "$got" -gt "$SIZE" ]; then
    echo "::warning::$NAME: $got bytes > expected $SIZE -- starting over"
    rm -f "$ZIP"
    continue
  fi

  left=$((DEADLINE_S - $(elapsed)))
  [ "$left" -gt 0 ] || fail "deadline ${DEADLINE_S} s reached at $got/$SIZE bytes after $attempt attempts"
  [ "$stuck" -lt "$MAX_STUCK" ] || fail "$stuck consecutive attempts each fetched < $SPEED_LIMIT bytes ($got/$SIZE bytes) -- the connection is hung"

  attempt=$((attempt + 1))
  # The API answers with a 302 to a short-lived signed blob URL; resolve it
  # fresh for every attempt rather than reusing one that may have expired.
  # Not following it here also keeps the token off the blob host.
  url=$(api -o /dev/null -w '%{redirect_url}' "$API/repos/$GITHUB_REPOSITORY/actions/artifacts/$ID/zip") \
    || { stuck=$((stuck + 1)); echo "attempt $attempt: resolving the download URL failed"; sleep 5; continue; }
  [ -n "$url" ] || { stuck=$((stuck + 1)); echo "attempt $attempt: no redirect URL"; sleep 5; continue; }

  t0=$SECONDS
  rc=0
  curl -fsS -C - -o "$ZIP" \
    --connect-timeout 30 \
    --speed-limit "$SPEED_LIMIT" --speed-time "$SPEED_TIME" \
    --max-time "$left" \
    "$url" || rc=$?
  now=$(have)
  dt=$((SECONDS - t0))
  echo "attempt $attempt: curl exit $rc, +$((now - got)) bytes in ${dt} s ($(( (now - got) / (dt > 0 ? dt : 1) / 1000000 )) MB/s), $now/$SIZE, $(elapsed) s total"
  case "$rc" in
    0) ;;
    # 33: range not satisfiable / server will not resume -> refetch from 0.
    33) rm -f "$ZIP" ;;
    # 28: below the speed floor or out of time -- the stall this exists for.
    28) echo "::notice::$NAME: attempt $attempt below ${SPEED_LIMIT} B/s for ${SPEED_TIME} s -- reconnecting and resuming" ;;
    *) sleep 3 ;;
  esac
  # Progress = at least one second's worth at the floor. A slow-but-moving
  # connection (~9 MB/s gains ~400 MB per attempt) resets this; a hang or a
  # trickle does not, so it gives up after ~MAX_STUCK x SPEED_TIME seconds
  # instead of burning the whole deadline.
  if [ $((now - got)) -ge "$SPEED_LIMIT" ]; then stuck=0; else stuck=$((stuck + 1)); fi
done

echo "downloaded and digest-checked in $(elapsed) s over $attempt attempt(s)"
# unzip checks each member's CRC32; -o so a stale file from a previous step
# cannot make it prompt.
unzip -o -q "$ZIP" "$FILE" -d "$DEST" || fail "unzip failed"
rm -f "$ZIP"
[ -f "$OUT" ] || fail "$FILE is not in the artifact"
zstd -tq "$OUT" || fail "zstd -t failed on $FILE"
echo "$OUT verified in $(elapsed) s total"
