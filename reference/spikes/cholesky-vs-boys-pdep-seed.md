# Spike: pivoted-Cholesky vs Boys orbitals as the sparse-PDEP screening seed

Research record. Provisional. Two parts, kept separate on purpose:

1. **The 2026-07-16 spike**, ported 2026-10-23 from source commit `14386e506`
   (worktree `.claude/worktrees/agent-ac0af625af17b4f10`, history from before
   the repository rewrite) and its write-up `docs/spikes/cholesky-vs-boys-screening.md`
   (recovered from that worktree's index). Its numbers are quoted unchanged;
   the raw CSV is `cholesky-vs-boys-pdep-seed-2026-07-16.csv` beside this file.
2. **The 2026-10-23 port onto main**, with what the same code measures now.
   Main changed underneath the spike in a way that makes its comparison
   vacuous; see part 2.

Code (feature `spike-cholesky-seed`, off in every shipped build; not reachable
from the CLI, Python or `Chi0Sparsity::parse_config_str`):

* `crates/ferric-rpa/src/cholesky_localize.rs`: `pivoted_cholesky`,
  `cholesky_localize_occupied`, `build_screened_bov_cholesky`.
* `Chi0Sparsity::CholeskyScreened { thresh, dist_cutoff }` (cfg-gated variant)
  and its dispatch arm in `run_pdep_rpa_eigensolve`.
* `crates/ferric-rpa/tests/spike_cholesky_seed.rs`.
* `benchmarks/harness/examples/sparse_scaling.rs` adds `Cholesky-1e-{3,4,5}`
  and `Boys-0` (Boys screening at thresh 0) configs under the same feature.

```
OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --features spike-cholesky-seed \
  --test spike_cholesky_seed -- --nocapture
OPENBLAS_NUM_THREADS=1 FERRIC_SCALING_ONLY=n-hexane \
  cargo run --release -p ferric-benchmarks --features spike-cholesky-seed --example sparse_scaling
```

## Part 1: the 2026-07-16 spike (quoted)

**Question.** Sparse PDEP-RPA kept 98-100% of shell pairs at every
accuracy-preserving Boys threshold. Is the localization method part of that
bottleneck, or is any per-orbital coefficient-magnitude bound loose whatever
orbitals feed it?

**Method.** Pivoted-Cholesky (Aquilante) localization of the active occupied
density `D = C_occ C_occᵀ` via `faer` 0.24.4 `llt_pivoting`, fed into the same
`build_screened_bov` path as Boys. n-hexane and naphthalene, cc-pVDZ /
cc-pVDZ-RI, RHF with DF-J + LinK, frozen core one per heavy atom, PDEP
`trunc_thresh = 1e-4`, Lanczos `conv_thresh = 1e-6`,
`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`. Dense `E_RPA`: n-hexane
-1.1821590, naphthalene -1.4176069.

| system | config | retained_frac | ΔE vs Dense (µHa) | t_pdep (s) |
|---|---|---|---|---|
| n-hexane | Dense | 1.0000 | n/a | 3.25 |
| n-hexane | Boys-1e-3 | 0.9867 | 1.8 | 15.82 |
| n-hexane | Boys-1e-4 | 1.0000 | 0.3 | 22.44 |
| n-hexane | Boys-1e-5 | 1.0000 | 0.1 | 20.36 |
| n-hexane | Cholesky-1e-3 | 0.8978 | 61.9 | 9.93 |
| n-hexane | Cholesky-1e-4 | 0.9844 | 62.9 | 15.05 |
| n-hexane | Cholesky-1e-5 | 1.0000 | 26.9 | 14.40 |
| naphthalene | Dense | 1.0000 | n/a | 6.83 |
| naphthalene | Boys-1e-3 | 0.9817 | 5.3 | 38.80 |
| naphthalene | Boys-1e-4 | 1.0000 | 0.8 | 51.16 |
| naphthalene | Boys-1e-5 | 1.0000 | 0.1 | 53.55 |
| naphthalene | Cholesky-1e-3 | 0.9224 | 18.3 | 26.36 |
| naphthalene | Cholesky-1e-4 | 0.9975 | 26.5 | 31.66 |
| naphthalene | Cholesky-1e-5 | 1.0000 | 23.9 | 36.19 |

RHF (shared, excluded from t_pdep): n-hexane 319 s, naphthalene 444 s. These
timings predate the 2026-10-10 CPU-fan fix and are upper bounds.

Correctness tests at the time (H2O/cc-pVDZ): rank = nocc = 5, `D`
reconstructed to < 1e-10; Cholesky-screened vs Dense 0.27 µHa at thresh 0
and < 3 µHa at thresh 1e-6, both at tight solver settings
(`trunc_thresh = 0`, conv 1e-10).

**Verdict as recorded ("Outcome 2").** Cholesky retention was modestly
tighter than Boys at matched thresholds (89.8-92.2% vs 98.2-98.7% at 1e-3),
nowhere near the ≥10x reduction needed to beat Dense. Every Cholesky config,
including 1e-5 at 100% retention, sat 24-63 µHa from Dense, which the spike
attributed to "the non-orthonormal Cholesky seed interacting with the
production Lanczos/PDEP-truncation path", not to pruning. Cholesky localized
1.3-1.6x faster than Boys, but both stayed 3-8x slower than Dense. Conclusion:
the per-orbital coefficient-magnitude bound, not the orbital generator, is
the bottleneck; next lever a spatial centroid-distance cutoff on `sig_pairs`.
Recommendation: do not default to Cholesky; keep it as an experimental
variant.

## Part 2: the port onto main (2026-10-23)

### What had to change

* `faer` is not a dependency of main and adds roughly 880 lines of
  `Cargo.lock`. The port implements the same diagonal-pivoting outer-product
  algorithm in ndarray (stop when the largest remaining diagonal is
  `<= n·ε·max diag`, the LAPACK `dpstrf` default) and returns the factor in
  the original AO row order, so there is no permutation to undo. Pivot rule
  and result are the same as the spike's up to rounding; the tests check rank,
  reconstruction and pivot order directly.
* `build_screened_bov` and `build_screened_bov_boys` gained `dist_cutoff`
  (the G6 centroid-distance pre-filter, which is the follow-up the spike
  recommended). `CholeskyScreened` carries it too; the spike's numbers
  correspond to `dist_cutoff = ∞`.
* The `"cholesky[:thresh]"` config string and Python/CLI exposure were NOT
  ported: the variant is library-only behind the feature.
* Centroids are normalized by `⟨k|S|k⟩` (the spike used the Euclidean
  `c_kᵀc_k`); they are diagnostic only, because `build_screened_bov`
  recomputes centroids itself.

### Two findings that change how part 1 reads

**(a) The Cholesky orbitals are orthonormal.** Two full-rank factorizations of
the same matrix differ by an orthogonal matrix, so `c_loc = C_occ U` with `U`
orthogonal and `c_locᵀ S c_loc = I`. They are not orthonormal only in the
Euclidean coefficient sense (`c_locᵀ c_loc ≠ I`), which is not the inner
product the RPA code uses. Measured on H2O/cc-pVDZ:
`max|c_locᵀ S c_loc − I| = 2.4e-15`, `max|c_locᵀ c_loc − I| = 0.22`,
`max|ΔD| = 1.1e-16`. The spike's mechanism ("non-orthonormal seed") is
therefore not the explanation of its 24-63 µHa error.

**(b) On main the localization method no longer matters.** Since `a953d4ff5`
(2026-07-28, after the spike) `build_screened_bov` semicanonicalizes the
orbitals it receives: it diagonalizes `C_locᵀ F C_loc` over the WHOLE active
occupied block and screens the eigenvectors. For any orthonormal basis of the
occupied space that rotation lands on the canonical occupied orbitals (up to
sign, for a non-degenerate spectrum). Boys and Cholesky seeds therefore
produce the same screened representation. Measured on H2O/cc-pVDZ
(`h2o_boys_and_cholesky_screens_coincide_after_semicanonicalization`):
identical per-orbital retained aux lists at thresh 1e-3 / 1e-4 / 1e-6
(371 / 386 / 391 of 420 for both) and `eps_loc` equal to each other and to
the canonical occupied energies to 3.9e-14.

Before `a953d4ff5`, the screened matvec used `diag(C_locᵀ F C_loc)` as
orbital energies and dropped the off-diagonal Fock coupling (measured for Boys orbitals at
1.3-2.9% of the diagonal spread, water/alkane_4/alkane_8/benzene at STO-3G, per
the comment in `screen.rs`). That defect depends on
which localized basis is used, it vanishes from the energy at full PDEP rank
(`trunc_thresh = 0`, where the energy is evaluated with the dense `b_ov` in a
complete eigenbasis) and it appears once truncation selects a subspace from
the mis-built matvec. That matches the spike's pattern exactly (error at
`trunc_thresh = 1e-4` even at 100% retention, < 3 µHa at `trunc_thresh = 0`).
It is offered as the more likely cause of the spike's Cholesky error; it was
not tested at the spike's commit, so treat it as a hypothesis.

A consequence for main beyond this spike, also unmeasured here: if the
screened path always screens canonical (delocalized) orbitals, the per-orbital
aux retention cannot benefit from localization at all, whichever method
produced the input orbitals.

### Measured on main with the ported code

H2O/cc-pVDZ, debug-profile test build, `OPENBLAS_NUM_THREADS=1`:

| check | result |
|---|---|
| rank detected | 5 = nocc; pivots (AO) 0, 3, 1, 5, 4 |
| tight (`trunc 0`, conv 1e-10), thresh 0: \|Chol − Dense\|, \|Boys − Dense\| | 9.9e-11, 9.9e-11 |
| tight, thresh 1e-6: retained, \|Chol − Dense\| | 391/420, 7.1e-7 Ha |
| production (`trunc 1e-4`, conv 1e-6), thresh 0: \|Chol − Dense\|, \|Boys − Dense\| | 6.7e-16, 4.4e-16 |

Water is too small to show any truncation effect, so the production row is a
regression check, not a reproduction of the spike's n-hexane/naphthalene
regime.

n-hexane / cc-pVDZ, the spike's driver and settings (`sparse_scaling.rs`, DF-J
+ LinK RHF, frozen core 6, `trunc_thresh 1e-4`, conv 1e-6), main @ `e2e804008`,
release build, `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=3` on a box shared
with other jobs (so timings are noisy). Raw rows:
`cholesky-vs-boys-pdep-seed-2026-10-23-main.csv`.

| config | retained_frac | E_RPA | ΔE vs Dense (µHa) | t_pdep (s) |
|---|---|---|---|---|
| Dense | 1.0000 | -1.1821589845 | n/a | 2.02 |
| Boys-1e-3 | 0.95766 | -1.1821286609 | 30.3 | 36.4 |
| Cholesky-1e-3 | 0.95766 | -1.1821286609 | 30.3 | 26.3 |
| Boys-1e-5 | 1.00000 | -1.1820783730 | 80.6 | 43.2 |
| Cholesky-1e-5 | 1.00000 | -1.1820783730 | 80.6 | 63.1 |

Boys and Cholesky agree to all printed digits and keep the same aux rows, as
part 2(b) predicts. The timing differences between the arms come from the
localizer and box load, not from the screened solve, which is the same.

### Is the identical result an artifact? (checked 2026-10-23)

Identical numbers from two arms is a stop condition, so the three ways it
could be a dead knob were checked:

1. **Does the Cholesky config reach the Cholesky code?** Yes. Mutant M-A
   (the `CholeskyScreened` dispatch arm in `run_pdep_rpa_eigensolve` returns
   an error) fails all three tests that go through `run_pdep_rpa` with
   `CholeskyScreened`; with the arm intact they pass. The arm calls
   `build_screened_bov_cholesky`, which returns the pivots it used (water:
   AO 0, 3, 1, 5, 4).
2. **Do the two localizers produce different orbitals?** Yes, substantially.
   Water/cc-pVDZ (`h2o_localizers_differ_but_semicanonicalize_to_the_same_orbitals`):
   `C_cholᵀ S C_boys` is 0.430 from the nearest signed permutation (0 would
   mean the same orbitals up to order and sign); coefficient fingerprints
   23.371169 (Cholesky) vs -25.960989 (Boys).
3. **Is Cholesky only seeding Boys?** No. `boys_localize` starts its Jacobi
   sweeps from the canonical occupied orbitals (converged in 13 sweeps on
   water); Cholesky output never enters it.

The cause is the semicanonicalization in `build_screened_bov`: after it,
`max|C_chol − C_boys| = 4.3e-15` and `max|C_chol − C_canonical| = 7.7e-15`
(sign-aligned columns). Mutant M-B (semicanonicalization removed, i.e. the
pre-`a953d4ff5` `diag(F_loc)` behaviour) breaks the coincidence test, and it
also reproduces the spike's pattern on water: at production settings
(`trunc 1e-4`, thresh 0, full retention) |Cholesky − Dense| = 6.7 µHa versus
|Boys − Dense| = 0.29 µHa, while at tight full-rank settings both stay below
0.15 µHa (Cholesky 1.5e-7, Boys 8.6e-11). That supports the part-2(b)
explanation of the spike's 24-63 µHa Cholesky error (mis-built matvec
energies, seen only under truncation) over its "non-orthonormal seed"
explanation. It is a water-scale reproduction, not a rerun at the spike's
commit.

### A main-branch observation outside this spike

On main the Boys-screened path is 30.3 µHa (thresh 1e-3) and 80.6 µHa
(thresh 1e-5) from Dense on n-hexane at the production settings above. Notes
on reading those numbers:

* `retained_frac` counts aux rows only. The tile build also drops OBS shell
  pairs below `shell_thresh = thresh/100` (`crates/ferric-rpa/src/screen.rs:269`,
  applied at `:393`), so a retained fraction of 1.0 does not mean the screened
  operator is exact.
* The spike's 1.8 / 0.1 µHa for the same configs are not directly comparable:
  three months of main lie between the two runs (Dense E_RPA moved by
  0.036 µHa; the RHF took 319 s then, 185 s now). No causal link to
  `a953d4ff5` is claimed.
* Boys at 1e-3 now keeps 95.8% of aux rows against 98.7% in the spike. The
  screen now acts on semicanonical (canonical, delocalized) orbitals, not on
  Boys orbitals.
* The eigensolve is not the source. The screened seed has 19 columns (active
  occupied) and naux = 532 = 19 × 28, so the block-Lanczos Krylov space fills
  the whole aux space and the eigensolve is exact for whatever operator it is
  given. Any error is in the screened operator.
* Blind spot: at `trunc_thresh = 0` the energy is evaluated with the dense
  `b_ov` in a complete eigenbasis, so the tight-settings tests in
  `tests/spike_cholesky_seed.rs` cannot see an error in the screened
  operator. Only the production-settings test can.

**Decisive run, 2026-10-23** (n-hexane/cc-pVDZ, main @ `e2e804008`, same
driver and production settings; `Boys-0` = Boys screening at thresh = 0, which
disables both the aux-row and the OBS shell-pair prescreens):

| config | E_RPA | ΔE vs Dense | t_pdep (s) |
|---|---|---|---|
| Dense | -1.1821589845 | n/a | 2.33 |
| Boys-0 | -1.1821589845 | 0 at the printed 1e-10 Ha | 159.1 |

Raw rows: `cholesky-vs-boys-pdep-seed-2026-10-23-main-thresh0.csv`.

The error vanishes at thresh = 0, so the 30.3 / 80.6 µHa above is
**screening error** from the prescreens (mostly the OBS shell-pair cut, since
the 1e-5 run keeps every aux row). It is not an operator defect introduced
by `a953d4ff5`. The error grows from 1e-3 to 1e-5 even though the 1e-5
screen is looser. That is measured and not explained here.

Side observation, not fixed in this PR: in `crates/ferric-rpa/src/lanczos.rs`
`outer_iters = max_krylov / block_size` (`:480`, and `:728` in the targeted
variant) rounds down. On water (naux 84, block 5) the Krylov space therefore
stops at 80 vectors after 16 block iterations. The warning then reports "did
not converge within max_iter=50" (`:417`), which is the wrong count. The water
`run_pdep_rpa` tests in `tests/spike_cholesky_seed.rs` all run with that
unconverged solver (worst residual 1.7e-7 to 2.0e-7).

Mutation ledger for `tests/spike_cholesky_seed.rs` (each mutant run against
the full file, then reverted):

| mutant | tests that fail |
|---|---|
| M-A: `CholeskyScreened` dispatch arm returns `Err` | the 3 `run_pdep_rpa` tests (tight thresh 0, tight 1e-6, production) |
| M-B: no semicanonicalization in `build_screened_bov` | coincidence test; production full-retention test |
| M-C: pivoted Cholesky without the Schur-complement update | all 7 |

## Status

* The spike's Part-1 verdict (Cholesky is not the lever; the bound is) stands
  as a record but its accuracy finding and mechanism are superseded by part 2:
  on main the comparison has no variable left to test.
* Open, not done here: whether the screened path should screen localized
  rather than semicanonicalized orbitals (which needs a Fock-coupled matvec,
  not a diagonal one); why the n-hexane screening error grows from thresh
  1e-3 to 1e-5; the Lanczos `outer_iters` rounding; and a rerun at the spike's commit with the
  semicanonicalization fix applied, to confirm the mechanism at the spike's
  sizes.
