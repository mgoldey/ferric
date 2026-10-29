#!/usr/bin/env bash
# Say WHICH re-run can fix a missing/corrupt build-tests archive (#525).
#
#   scripts/ci-archive-advice.sh <artifact-name>
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
# HOW, in order of trust:
#   (a) this run's artifact listing (REST API, needs actions:read):
#       an unexpired artifact of this name exists -> transient download
#       failure, "Re-run failed jobs" can work; only expired entries, or none
#       at all -> the archive is gone, "Re-run ALL jobs".
#   (b) if the API call fails: the age of CI_ARCHIVE_UPLOADED_AT (epoch
#       seconds, a build-tests job output) against RETENTION_DAYS less 1 h.
#   (c) neither available: print both cases.
#
# Env: GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID (for a), optional
# GITHUB_API_URL, GITHUB_RUN_ATTEMPT, CI_ARCHIVE_UPLOADED_AT (for b);
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

# (a) The artifact listing.
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ] \
    && command -v jq >/dev/null; then
  if LIST=$(curl -fsS --max-time 20 --retry 2 \
        -K <(printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN") \
        -H "Accept: application/vnd.github+json" \
        "$API/repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/artifacts?name=$NAME&per_page=100" \
        2>/dev/null) \
     && COUNTS=$(jq -r '[([.artifacts[] | select(.expired | not)] | length),
                         ([.artifacts[] | select(.expired)] | length)] | @tsv' <<<"$LIST" 2>/dev/null) \
     && [[ $COUNTS =~ ^[0-9]+$'\t'[0-9]+$ ]]; then
    live=${COUNTS%%$'\t'*}
    dead=${COUNTS##*$'\t'}
    [ "$live" -eq 0 ] || say "$RERUN_JOB"
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
  say "(artifact listing unavailable) build-tests uploaded the archive $((AGE / 60)) min ago, inside its $RETENTION_DAYS-day retention. This is an ARTIFACT DOWNLOAD failure, not a build or test failure: \"Re-run failed jobs\" can succeed."
fi

# (c) Nothing to go on.
say "ARTIFACT DOWNLOAD failure, not a build or test failure (could not tell whether '$NAME' has expired). If build-tests ran less than ~$RETENTION_DAYS days ago, \"Re-run failed jobs\"; otherwise the archive has EXPIRED (retention-days: $RETENTION_DAYS) and only \"Re-run ALL jobs\" works (#525)."
