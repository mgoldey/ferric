#!/usr/bin/env bash
# After C20 finishes: C24, C28 with only the two eps that bracket relerr 1e-3 (1e-2 is
# not bracketed by any affordable eps list; wall is nearly eps-independent so each extra
# eps costs a full integral pass). Repo root.
S=benchmarks/terfc_sparsity_ncp/scripts
while pgrep -f "par.sh 20" > /dev/null; do sleep 20; done
for N in "$@"; do
  bash $S/par.sh $N 1e-4,4e-5 1 1e-4,4e-5,1.6e-5
done
