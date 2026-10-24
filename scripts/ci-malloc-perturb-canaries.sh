#!/usr/bin/env bash
# The ONE list of malloc-perturb canary binaries (nextest binary ids,
# crate::binary). CI's build-tests job archives exactly these into the small
# canary archive, and the malloc-perturb job runs and checks exactly these;
# both read this script, so the archive and the run cannot drift apart.
# Why these binaries: see the comment above the `malloc-perturb` job in
# .github/workflows/ci.yml.
#
#   scripts/ci-malloc-perturb-canaries.sh ids     one binary id per line
#   scripts/ci-malloc-perturb-canaries.sh filter  the nextest filterset that
#                                                 matches exactly these binaries
set -euo pipefail

CANARY_IDS=(
  ferric-integrals::boys_chebyshev_table_edge
  ferric-integrals::tenno_table_edge
  ferric-pbc::pbc_kpoint_ecp_rotation
  ferric-pbc::pbc_krsgdf
  ferric-pbc::pbc_krsgdf_split
  ferric-pbc::pbc_grad_ecp
)

case "${1:-}" in
  ids)
    printf '%s\n' "${CANARY_IDS[@]}"
    ;;
  filter)
    FILTER=""
    for id in "${CANARY_IDS[@]}"; do
      FILTER="${FILTER:+$FILTER | }binary_id(=$id)"
    done
    echo "$FILTER"
    ;;
  *)
    echo "usage: $0 ids|filter" >&2
    exit 2
    ;;
esac
