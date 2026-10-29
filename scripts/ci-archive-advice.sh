#!/usr/bin/env bash
# Say WHICH re-run can fix a missing/corrupt build-tests archive (#525).
#
#   scripts/ci-archive-advice.sh <artifact-name> [<downloaded-file>]
#
# Called by ci.yml's test-shard and malloc-perturb jobs only AFTER they have
# already decided to fail on a missing or corrupt archive. It prints one
# ::error:: line of advice and ALWAYS exits 0: it is a diagnostic, and a
# diagnostic that fails must never replace or hide the original error (the
# caller exits 1 itself, whatever this prints).
#
# WHY. build-tests uploads its archives with `retention-days: $RETENTION_DAYS`.
# Once the build is older than that the archive has EXPIRED, and "Re-run failed jobs"
# reuses the old build-tests result, so it can never succeed; only
# "Re-run all jobs" rebuilds the archive (PR #581, run 40579399666, twice).
#
# A LIVE artifact is not proof a job re-run can recover: if the UPLOADED
# archive is corrupt, the re-run downloads the same bad bytes. So first:
#   (0) the downloaded file against what build-tests uploaded
#       (CI_ARCHIVE_DIGESTS, a build-tests job output: {"<name>":
#       {"bytes":N,"sha256":"..."}}). Full byte count AND matching sha256,
#       yet the caller's `zstd -t` failed -> the upload itself is corrupt,
#       "Re-run ALL jobs". Missing, short, or a different hash -> the
#       TRANSFER failed; carry on to (a)-(c) for whether a job re-run can
#       still fetch it. No record (or the hash timed out) -> (a)-(c) as well.
#
# Then, in order of trust:
#   (a) this run's artifact listing (REST API, needs actions:read):
#       an unexpired artifact of this name exists -> transient download
#       failure, "Re-run failed jobs" can work; only expired entries, or none
#       at all -> the archive is gone, "Re-run ALL jobs".
#   (b) if the API call fails: the age of CI_ARCHIVE_UPLOADED_AT (epoch
#       seconds, a build-tests job output) against RETENTION_DAYS less 1 h.
#   (c) neither available: print both cases.
#
# Env: CI_ARCHIVE_DIGESTS (for 0); GITHUB_TOKEN, GITHUB_REPOSITORY,
# GITHUB_RUN_ID (for a), optional GITHUB_API_URL, GITHUB_RUN_ATTEMPT,
# CI_ARCHIVE_UPLOADED_AT (for b);
# CI_ADVICE_NOW overrides `date +%s` (tests). The token reaches curl through
# a process-substitution config (-K), never its argv.
set -uo pipefail

NAME=${1:-archive}
API=${GITHUB_API_URL:-https://api.github.com}
# Must equal every archive upload's retention-days in ci.yml
# (ci_validation_tier_guard checks it).
RETENTION_DAYS=2
EXPIRY_GUARD_S=$((RETENTION_DAYS * 24 * 3600 - 3600))
ATTEMPT=${GITHUB_RUN_ATTEMPT:-1}

RERUN_JOB="This is an ARTIFACT DOWNLOAD failure, not a build or test failure: the archive '$NAME' is still listed for this run, so \"Re-run failed jobs\" (this job) can succeed."
RERUN_ALL="The archive '$NAME' has EXPIRED (build-tests uploads it with retention-days: $RETENTION_DAYS), so \"Re-run failed jobs\" can NEVER succeed for this run: use \"Re-run ALL jobs\" to rebuild it (#525)."

say() { echo "::error::$*"; exit 0; }

# (0) Transfer failure or corrupt upload? Budget: the hash is capped at
# 12 s (a ~0.2 GB archive hashes in well under 1 s), plus (a)'s 31 s,
# inside the callers' `timeout 60`.
FILE=${2:-}
# Set once (0) has shown the TRANSFER failed; until then a job re-run is
# only a hope, and the advice says so.
TRANSFER=0
if [ -n "$FILE" ] && [ -n "${CI_ARCHIVE_DIGESTS:-}" ] && command -v jq >/dev/null; then
  EXP_BYTES=$(jq -r --arg n "$NAME" '.[$n].bytes // empty' <<<"$CI_ARCHIVE_DIGESTS" 2>/dev/null)
  EXP_SHA=$(jq -r --arg n "$NAME" '.[$n].sha256 // empty' <<<"$CI_ARCHIVE_DIGESTS" 2>/dev/null)
  if [[ ${EXP_BYTES:-} =~ ^[0-9]+$ ]]; then
    if [ ! -f "$FILE" ]; then
      echo "::error::TRANSFER failure: $FILE is missing (build-tests uploaded $EXP_BYTES bytes)."
      TRANSFER=1
    else
      GOT=$(stat -c %s "$FILE" 2>/dev/null || echo 0)
      if [ "$GOT" != "$EXP_BYTES" ]; then
        echo "::error::TRANSFER failure: $FILE has $GOT of the $EXP_BYTES bytes build-tests uploaded."
        TRANSFER=1
      elif [[ ${EXP_SHA:-} =~ ^[0-9a-f]{64}$ ]]; then
        HASH=$(timeout 12 sha256sum "$FILE" 2>/dev/null | cut -d' ' -f1)
        if [ "$HASH" = "$EXP_SHA" ]; then
          say "The archive '$NAME' arrived INTACT ($GOT bytes, sha256 equal to what build-tests uploaded) yet fails zstd -t: the UPLOADED archive is corrupt, so \"Re-run failed jobs\" would download the same bad bytes: use \"Re-run ALL jobs\" to rebuild it (#525)."
        elif [ -n "$HASH" ]; then
          echo "::error::TRANSFER failure: $FILE has the full $GOT bytes but a different sha256 from what build-tests uploaded."
          TRANSFER=1
        fi
      fi
    fi
  fi
fi

# (a) The artifact listing.
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ] \
    && command -v jq >/dev/null; then
  # Worst case ~31 s (2 x 15 s + 1 s retry delay; --retry-max-time 30
  # forbids a retry starting later), well inside the callers' `timeout 60`,
  # so a hung API still leaves time for fallback (b) below.
  if LIST=$(curl -fsS --connect-timeout 10 --max-time 15 --retry 1 --retry-max-time 30 \
        -K <(printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN") \
        -H "Accept: application/vnd.github+json" \
        "$API/repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/artifacts?name=$NAME&per_page=100" \
        2>/dev/null) \
     && COUNTS=$(jq -r '[([.artifacts[] | select(.expired | not)] | length),
                         ([.artifacts[] | select(.expired)] | length)] | @tsv' <<<"$LIST" 2>/dev/null) \
     && [[ $COUNTS =~ ^[0-9]+$'\t'[0-9]+$ ]]; then
    live=${COUNTS%%$'\t'*}
    dead=${COUNTS##*$'\t'}
    if [ "$live" -gt 0 ]; then
      [ "$TRANSFER" = 1 ] || RERUN_JOB+=" (Could not check the file against what build-tests uploaded: if a job re-run fails the same way, the upload itself is corrupt and only \"Re-run ALL jobs\" helps.)"
      say "$RERUN_JOB"
    fi
    [ "$dead" -eq 0 ] || say "$RERUN_ALL"
    # Not listed at all. GitHub drops expired artifacts from the listing
    # after a while, so on a re-run attempt (or an old build) that is expiry
    # too; either way nothing is left to download.
    say "No artifact named '$NAME' is listed for run $GITHUB_RUN_ID (attempt $ATTEMPT): it has EXPIRED (retention-days: $RETENTION_DAYS) or was never uploaded, so \"Re-run failed jobs\" can NEVER succeed: use \"Re-run ALL jobs\" to rebuild it (#525)."
  fi
fi

# (b) The build's age.
NOW=${CI_ADVICE_NOW:-$(date +%s)}
UP=${CI_ARCHIVE_UPLOADED_AT:-}
if [[ $UP =~ ^[0-9]+$ ]] && [[ $NOW =~ ^[0-9]+$ ]]; then
  AGE=$((NOW - UP))
  if [ "$AGE" -ge "$EXPIRY_GUARD_S" ]; then
    say "(artifact listing unavailable) build-tests uploaded the archive $((AGE / 3600)) h ago. $RERUN_ALL"
  fi
  MSG="(artifact listing unavailable) build-tests uploaded the archive $((AGE / 60)) min ago, inside its $RETENTION_DAYS-day retention. This is an ARTIFACT DOWNLOAD failure, not a build or test failure: \"Re-run failed jobs\" can succeed."
  [ "$TRANSFER" = 1 ] || MSG+=" (Could not check the file against what build-tests uploaded: if a job re-run fails the same way, the upload itself is corrupt and only \"Re-run ALL jobs\" helps.)"
  say "$MSG"
fi

# (c) Nothing to go on.
say "ARTIFACT DOWNLOAD failure, not a build or test failure (could not tell whether '$NAME' has expired). If build-tests ran less than ~$RETENTION_DAYS days ago, \"Re-run failed jobs\"; otherwise the archive has EXPIRED (retention-days: $RETENTION_DAYS) and only \"Re-run ALL jobs\" works (#525)."
