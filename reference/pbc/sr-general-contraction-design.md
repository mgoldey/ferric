# SR 3-centre: general contraction and pair symmetry (design, 2026-09-29)

Research and design only. No crate code was changed. The Python prototypes are in
`reference/pbc/bench/sr_gc/` (`gc_count.py`, `gc_variants.py`, `gc_model.py`, `cint_bench.py`, `bench_sr3.txt`).
The CPU work stopped when the user asked. Everything below marked ESTIMATE is a count times a cost model, not a
ferric timing.

> **Status on main (checked 2026-10-24, when this note was ported from the ferric-pbc worktree).** The note is a
> dated design record and is kept as written below. These parts are now STALE:
>
> - "No crate code was changed" and §1's "Ferric does not exploit the symmetry": Γ s2 landed in c83476b80
>   (2026-09-30; tests `crates/ferric-pbc/tests/pbc_pair_symmetry.rs`). The file:line references in §1 (`rsgdf.rs:1111`,
>   `split.rs:840`, `hcore.rs:894`, ...) point at the pre-s2 code and no longer match.
> - §1 "keep s1 on the k-point path": k-point s2 landed in 33de13d0d (`pbc_kpair_symmetry.rs`).
> - §4 Step 1 "uncommitted, not compiled or measured" and Step 2 "uncommitted; NOT compiled or run": both are
>   merged. The rotation landed opt-in in 0c27f999b and became the default in 282be9cac. The API is now
>   `SrColumnRotation::{Auto (default), Off, On(ColumnRotation)}` (`crates/ferric-pbc/src/sr_rotation.rs`), not
>   `Some(ColumnRotation::new())` / `None`.
> - Step 2 "`build_for_gradient` ... and the k-point builds do not rotate": the Γ forces and stress rotate since
>   922a3b9b6, the k-point energy builds since 6c64a8227, and the k-point forces and stress since 2faa6904d. Only the
>   frozen s1 oracles stay unrotated.
> - §5 item 4 "ω = 0.3, not yet measured": the dry ice ω = 0.3 rotation timing is in `prototypes/pbc/FINDINGS.md`,
>   "Full timing series on libint 2.13.1: MWE, baseline, ω sweep, column rotation" (2026-09-30).
> - SHA: 3fea0374 below = 5ab7c31b0 on main (perf(integrals): stop libint2 copying the erf/erfc Gm evaluator
>   per primitive quartet); 3fea0374 itself is the worktree commit and is not on main.
> - Paths: `pbc_gdf_split.py` is now `prototypes/pbc/pbc_gdf_split.py`. The `sr_gc/` scripts locate it and
>   `cell_facts.py` relative to their own directory.
>
> Still true and re-checked on the port: `gc_count.py` reproduces 19,129,128 (dry ice, ω = 1) and 35,921,128
> (diamond_prim) ordered SR3 triplets. `gc_variants.py` + `gc_count.py dryice 1.0 _rot` gives 10,446,656. `gc_model.py`
> gives the 33.1 s / 15.0 s model CPU of §3. All counts are the ORDERED walk, which ferric now reports as
> `n_sr3_triplets_ordered`; its `n_sr3_triplets` counts the s2-computed triplets. The cost model in
> `bench_sr3.txt` is still the pre-3fea0374 fit that §5 item 1 asks to redo.

## 0. Baseline

- Dry ice, cc-pVDZ / cc-pvdz-ri, Γ RHF, ω = 1, range split λ = 1 (`twocall`). There are 19.1 M SR3 triplets at
  1.45 µs per call on 1 thread (FINDINGS "Why PySCF is faster"), which is about 27.7 s of CPU.
- Commit 3fea0374 removed libint's per-primitive-quartet erf/erfc evaluator copy. The 1-thread erfc 3-centre call on
  the contention mix went from 1875 ns to 1568 ns (−16%). The 6-thread inflation went from 3.28× to 1.05×. All the
  ratios below were built on the pre-patch cost fit. Section 3 says how the patch moves them.
- Cost model. `pbc_triplet_bench` (pre-patch, prebuilt binary, diamond cc-pVDZ / cc-pvdz-ri, 1 thread, load about
  2–4) fits `t_call = 0.779 µs + 0.0713 µs × N_prim`. `N_prim` counts the nonzero-coefficient primitive triples. The
  model uses a per-(l_μ, l_ν, l_P) intercept and a common slope.
- Exact counter. `gc_count.py` is `pbc_gdf_split.count_sr3_ferric` plus per-unit bins. It reproduces ferric's counts
  exactly: 19,129,128 on dry ice (ferric 19.1 M) and 35,921,128 `twocall` on diamond_prim (the FINDINGS value).
- The model predicts 33.1 s for today's dry ice SR3 against 27.7 s measured, 20% high. The main reason is that libint
  also screens primitive pairs at distant images, which the model ignores. Only the model's RATIOS are used below.

## 1. Pair symmetry: ferric computes BOTH (μ, ν@L) and (ν, μ@−L)

Ferric does not exploit the symmetry. Every SR walk enumerates all ORDERED shell pairs and averages the two halves
afterwards:

- `rsgdf.rs:1111`: `Stage::sr_three_index_binned` tasks are `(0..nsh * nsh * rl)`, one per ordered `(i1, i2)`.
  `sr3_pair_image` (`rsgdf.rs:1312`) visits every `L` for each ordered pair.
- `rsgdf/split.rs:840`: the range-split `SplitPlan::sr_three_index_binned` also uses `(0..nsh * nsh * rl)`, with two
  calls per ordered pair (`calls()`, `split.rs:665`).
- `rsgdf.rs:2334`: `symmetrize_pairs` (defined at `rsgdf.rs:1966`) then sets `J3[μν] = J3[νμ] = ½(x + y)` and
  reports `asym_j3`.
- `hcore.rs:894`: `sr_attraction` tasks are `(0..nsh * nsh)`, one per ordered pair over every `L`. `hcore.rs:1267`:
  `symmetrize(&sr.v)` then averages and reports `sr_asymmetry`.
- The same holds for S/T (`hcore.rs:1114`) and for the SR metric (`rsgdf.rs:873`, `split.rs:959`), which is small.

Why the second half is redundant at Γ:

- By translation, `(ν_0 μ_L | P_T) = (μ_0 ν_{−L} | P_{T−L})`.
- The image set is closed under `L → −L`, and `T` runs over the full lattice around the segment midpoint.
- `pair_bound`, the segment test and the radius are all invariant under translation and under a↔b.
- So the (i2, i1) task recomputes the (i1, i2) task's triplets in a different order. The asymmetry that
  `symmetrize_pairs` measures is round-off only.

The range-split kept part is `χ_iχ_j − χ_i^sχ_j^s`, which is symmetric per unordered pair. So the two calls
together, for i1 ≤ i2, give both orders.

For a diagonal shell pair (i1 == i2), `L` and `−L` are also duplicates. `Σ_L B_L(μν) = B_0(μν) + Σ_{L∈half}
[B_L(μν) + B_L(νμ)]`.

PySCF's own count gives the gain: 91.3 M s1 calls against 46.8 M s2 calls, which is 1.95×. The expected gain for
ferric is 2.0× on SR3, hcore SR, S/T and the pair FT (plan item 10).

k-points: the bins are `(L mod mod_l, T mod mod_t)`. `(ν μ)` lands in bin `(−r_L, r_T − r_L)`, so s2 needs that
residue map. Do Γ first and keep s1 on the k-point path until a non-TRIM k-mesh test covers it (see the
"mutation can be an identity at convergence" memory).

## 2. Does libint2 support generally contracted shells?

No, in no version on this box, and the failure is dangerous:

- 2.7.2 (`~/.local/include/libint2/engine.impl.h:177` and `:1116`) and 2.13.1 (the psi4 env,
  `engine.impl.h:191` and `:1203`) both do
  `assert(ncontr() == 1 && "generally-contracted shells are not yet supported")`.
- The compute paths read only `contr[0]` (for example `:1258`, `c0 = sbra1.contr[0].coeff[p]`).
- In a release build (`NDEBUG`) the assert is compiled out, so a shell with ncontr > 1 is SILENTLY WRONG: only
  column 0 is computed, while the buffer is sized for all columns.
- The shim builds exactly one `Contraction` per shell (`shim.cc:291`), which is correct. It should keep that
  invariant and never pass ncontr > 1.
- libint git master (the June 2022 checkout here) is the same. The web shows no GitHub issue or plan beyond the
  assert text.

Why libint cannot share primitives across columns: with `LIBINT2_CONTRACTED_INTS`, the contraction coefficients are
folded into each primitive's prefactor (`pfac *= c0·c1·c2·c3`, `:1263`), and the VRR build sums over primitives
internally. One call therefore yields exactly one contracted column combination.

Two facts about zero coefficients:

- libint already skips zero-coefficient primitive pairs. `max_ln_coeff = ln 0 = −∞` fails the ShellPair screen
  (`shell.h:484`). So the 8 zero primitives stored in ferric's 3s shell and the 3 in its second p shell cost almost
  nothing inside libint.
- Ferric's own `pair_bound` (`rsgdf.rs:690-701`) does include the zero primitives in `pmin` and `pmax`. The counter
  shows that dropping them changes dry ice SR3 by only −0.3% (19.07 M), so that is hygiene, not a speed-up.

## 3. Options

The cc-pVDZ C and O primitive and column structure (bundled JSON):

| shell | primitives | columns | nonzero per column |
|---|---|---|---|
| s | 9 | 3 | 9, 9, 1 |
| p | 4 | 2 | 4, 1 |
| d | 1 | 1 | 1 |

- The single-primitive column is always the most diffuse primitive of that shell.
- At ω = 1, λ = 1 it is the ONLY smooth orbital primitive (threshold `a ≤ 0.5`): C s 0.1596, p 0.1517; O s 0.3023,
  p 0.2753.
- cc-pvdz-ri is fully segmented, with 1 primitive per shell (18 shells per atom). Aux-side general contraction
  therefore buys nothing.

Dry ice SR3 at ω = 1, from the exact counts and the model (1 thread, pre-3fea0374 fit):

| option | calls | primitive triples | model CPU | vs today | notes |
|---|---|---|---|---|---|
| today (segmented, s1) | 19.13 M | 2.38e8 | 33.1 s (overhead 16.1 + primitive 17.0) | 1× | measured 27.7 s |
| drop zero primitives (bounds only) | 19.07 M | 2.38e8 | 33.1 s | 1.00× | integrals bit-identical; screen hygiene |
| **(a)** primitive-level decontraction, contract in ferric | ≥ 1.0e8 (one per unique primitive triple) | 1.0e8 | ≥ 85 s | **≤ 0.4× (slower)** | every call pays the ~0.8 µs overhead (HRR, transform, copies, FFI) |
| **(b)** libint general-contraction shell | — | — | — | — | impossible: assert, silently wrong in release |
| **(c)** general-contraction kernel (libcint), all columns of an (atom, l) | 7.18 M | 1.02e8 | 13.5–23.4 s | **1.4–2.5×** | upper: HRR and transform per call; lower: per column combination (libcint does HRR per column) |
| (c′) share only identical-support columns (1s + 2s) | 13.5 M | 1.23e8 | 20.5 s | 1.6× | |
| **(d)** column rotation to minimal support (below) | 10.45 M | 8.5e7 | 15.0 s | **2.2×** (1.8–2.8) | libint unchanged, no FFI |
| **s2 symmetry** (§1), on any row | ½ | ½ | ½ | **2.0×** | |
| **(d) + s2** | 5.2 M | 4.2e7 | 7.5 s | **4.4×** | |
| (c) + s2 | 3.6 M | 5.1e7 | 6.8–11.7 s | 2.8–4.9× | new C dependency |

### Option (d): column rotation

Within an (atom, l) group, subtract the single-primitive column from the other columns:

```
1s' = 1s − c_1s,9 · e_9
2s' = 2s − c_2s,9 · e_9
p'  = p  − c_p,4  · e_4
```

The span is unchanged, so the transform `R` is block-diagonal and unit-triangular. For C cc-pVDZ its off-diagonal
entries are 0.0032 and 0.58, so it is well conditioned and nothing cancels.

Rotation shortens the long columns (9 → 8 primitives for s, 4 → 3 for p). The bigger effect is that every rotated
column is now WHOLLY compact or wholly smooth at the default split, so the `twocall` pair collapses to one call or
none. That is why the call count falls 1.83× and the primitive-triple count falls 2.8×.

The kept part is bilinear and the compact/smooth split is linear within a fixed primitive set. So J3 is exactly
covariant: `J3 = (R ⊗ R)ᵀ J3_rot`, including the moved G-space parts.

### Why the range is 1.8–2.8×, and how 3fea0374 moves it

- If every call were pure overhead, the gain would be the call ratio, 1.83×. If every call were pure primitive work,
  it would be the primitive-triple ratio, 2.8×.
- The 3fea0374 patch removed a per-primitive-quartet cost. That lowers the slope, not the intercept, so the patch
  pushes (d) toward the low end of 1.8–2.8×.
- It pushes (c) the same way, since (c)'s gain is mostly primitive sharing.

Re-fit the model before quoting (§5, step 1).

### libcint cross-check (short range, unscreened C–O–O molecule, 1 thread, same ferric numbers, erfc ω = 1)

| basis form | µs per column triplet |
|---|---|
| general (nctr 3 / 2 / 1) | 1.07 |
| segmented | 2.36 |
| segmented, zeros dropped | 2.41 |
| rotated (d) | 1.73 |

So general contraction is worth 2.2× inside libcint, and rotation 1.37×, both with no screening and no range split.
In the periodic, range-split case (d) gains more, because it also removes the `twocall` duplication. The general and
segmented integrals agree to 1.5e-15.

### Option (c) in detail: libcint as the SR backend

- **License.** BSD-2-Clause (sunqm/libcint), and it is what PySCF uses.
- **Build.** CMake C library, a few minutes cold. It would be a `cmake` step in a ferric-integrals build script or a
  vendored tarball. It is a new dependency on every CI and wheel path, and the wheel is already at 93% of its size
  cap.
- **FFI.** One `int3c2e_sph(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)` per triplet.
  - Images: write a per-thread `env` copy of the shifted centre coordinates before each call.
  - SR erfc: set `env[PTR_RANGE_OMEGA] = −ω`.
  - It carries its own `CINTOpt` and primitive screening, and no global shared state.
- **Conventions.** Coefficients must be given against normalized primitives. The spherical order for l ≥ 2 is
  m = −l..l, which should match ferric's libint pure order, but must be verified with a mapping test. l = 1 is
  (x, y, z), matching ferric, whose p shells are Cartesian.
- **Scope.** Forces and stress (`deriv.rs`, `strain.rs`, `hcore` SR gradient) would stay on libint. That mixes two
  backends: gradient and energy integrals would differ at about 1e-15, which is fine at the FD tolerances.
- **What it gives.**
  - General contraction, 1.4–2.5×, on top of anything else.
  - A per-thread-clean kernel.
  - Later, a correct point-charge erfc nuclear attraction (`int1e_rinv` with ω < 0). That would remove the
    ζ = 1e16 Gaussian-nucleus workaround for libint's broken `erfc_nuclear`.
- **Complexity.** High: a new dependency, basis-mapping tests, two integral backends, and an ownership question for
  the shim-level reliability conventions (try/catch and status codes).

### Numerics of each option

- s2 changes bits: the average ½(x + y) becomes one of x or y. The difference is about `asym_j3` (round-off).
- (d) changes bits: it is a different association plus a back-transform, about 1 ulp × ‖R‖. It also changes the
  screening decisions, so the difference is at the precision level (1e-13).
- (c) changes bits: a different integral code.
- None of them is bit-identical to today. All of them stay bit-identical ACROSS THREAD COUNTS, because each keeps the
  "task owns its rows, copy not add" construction.

### Memory

- s2 has none.
- (d) needs a per-task scratch for the group pair: `n_grp_i · n_grp_j · naux · R_T`, where a group is at most 3s + 2p.
  That is about 2× today's per-pair scratch.
- (c) needs libcint's per-thread cache, a few MB.

### Composition with the libint threading patch and item 7 (batched call)

- s2 and (d) remove calls and primitives. They are independent of the patch and multiply with it.
- After (d), overhead is 59% of the model (8.9 of 15.0 s), so item 7 (batched shifted-eri3 with a precomputed ket
  ShellPair and no per-call Shell copies) becomes the next lever. (d) keeps the ket pair fixed across all `(P, T)` of
  a `(μ, ν_L)`, which is exactly item 7's batch shape.

## 4. Recommendation: s2 first, then (d) the column rotation; defer libcint

### Step 1: Γ s2 symmetry

**Implemented 2026-09-30 (uncommitted at writing; not yet compiled or measured).** Deviations from the plan below:
diagonal pairs run over ALL `L` and copy their `i ≤ j` elements into both places (no half-space: it would save only
the diagonal share, ~1/(2 nsh) of the work, and needs an `L ↔ −L` image map); the split task evaluates the
orientation with fewer kept calls (compact × split: 1 call, not 2); the SR metric J2 keeps the ordered walk (~0.1% of
SR3 calls); `n_sr3_triplets` now counts computed triplets and `n_sr3_triplets_ordered` (and `sr_walk_counts`'s
`n_sr3_triplets_s1`) keeps benchmarks comparable. Frozen oracles: `RsGdf::build_pair_s1_oracle`,
`hcore::periodic_hcore_pair_s1_oracle`. Tests: `crates/ferric-pbc/tests/pbc_pair_symmetry.rs`. Rust module docs
("Orbital-pair symmetry" in `rsgdf.rs` and `hcore.rs`) are the authority.

Files:

- `rsgdf.rs`: `sr_three_index_binned` task list; `sr3_pair_task` writes both `(a.off+i, b.off+j)` and the transposed
  row.
- `rsgdf/split.rs`: `sr_three_index_binned` and `sr3_task`.
- `hcore.rs`: `sr_attraction` and `pair_task`; also `overlap_kinetic`.

Changes:

- Tasks run over `i1 ≤ i2`. Diagonal pairs take `L = 0` plus a lattice half-space, and add the block plus its
  transpose.
- Each task writes rows `(i1, i2)` and `(i2, i1)`. These are disjoint across tasks, so the COPY construction and
  thread-count bit-identity survive.
- `symmetrize_pairs` stays as a guard. `asym_j3` should become 0 by construction.
- Gate on `bins.mod_l == 1` (Γ). The k-point path keeps s1 until a residue-mapping test on a non-TRIM mesh exists.

Tests:

- **Anchor.** With `sr_screen = false` on the tri 4H s+p and H2O/cc-pVDZ cells, s2 J3 vs today's symmetrised J3
  ≤ 4 ulp relative per element. Derive the bound from the measured `asym_j3`: measure both sides, per the "derive
  thresholds" rule.
- **Energies.** diamond_prim cc-pVDZ vs today ≤ 1e-12 Ha.
- **Counters.** `n_sr3_triplets` halves up to the diagonal: predict it from `sr_walk_counts` first.
- **Mutants.** Each must fail by ≥ 1e-6:
  - drop the transpose write;
  - diagonal pairs over the full L range, which double counts;
  - half-space that omits `L = 0`;
  - on k-points later, a `+r_L` residue in place of `−r_L`, run on a non-TRIM mesh.

### Step 2: column rotation (d), inside the SR3 and hcore-SR walks only

**Implemented 2026-09-30 (uncommitted; NOT compiled or run — the writer had no cargo).** Opt-in
`RsGdfConfig::sr_column_rotation` / `PeriodicHcoreConfig::sr_column_rotation` = `Some(ColumnRotation::new())`,
default `None` (today's build bit for bit). Deviations from the plan below:

- Rotation is done on the element TEMPLATES (`crate::sr_rotation`), and the rotated `PreparedBasis` is rebuilt from
  `cell.mol()`, so it has the parent's atoms, shell order and AO layout (checked). No `cond(R)` error: a column with
  `T_kk < 1e-3` or `|T_ks| > 1e3` is simply left unrotated.
- The back-transform is ONE serial pass over (group, group) pairs AFTER the s2 walk (not per task): the walks are
  unchanged, the output is exactly symmetric and bitwise across threads. Scratch is one group pair (≤ 6×6×naux).
- Screening bounds use the rotated raw coefficients (the size of `T_kk χ'_k` in parent units, as the split pieces
  do), NOT renormalised ones. `gc_variants.py` renormalises, so its counts are an upper bound on the Rust walk.
- Pair images and hcore nucleus candidates come from the parent shells (rotated primitive sets are subsets).
- `build_for_gradient`, the s1 oracles and the k-point builds do not rotate (the first two refuse the flag; the
  k-point builds ignore it).
- Tests: `crates/ferric-pbc/tests/pbc_sr_rotation.rs`. Mutants: `ColumnRotationMutant::{FlipSign, NoRenormalization,
  OneSidedBackTransform, RotateAux}`.
- Counter prediction with s2 (from the saved per-unit counts `units_dryice_1.0_rot.npz` of the Python counter,
  renormalised bounds): dry ice ω = 1 split, ordered 10,446,656 (diagonal 32,448), s2 **5,239,552** computed calls;
  the Rust count should be at or slightly below that (un-renormalised bounds). Unrotated s2 from the same data:
  9.49–9.67 M. Exact Rust numbers: `sr_walk_counts` with the flag on (integral-free).

The SCF, S/T, MO coefficients and every API-visible matrix stay in the original basis.

Design:

- **New `ColumnRotation` in ferric-pbc.** For each (atom, l) group of shells that share an identical exponent list,
  it finds the single-primitive columns and subtracts them from the other columns. The result is rotated shells with
  the zero primitives DROPPED, plus a unit-triangular `R` per group.
- **libint normalization.** libint renormalizes every contraction, so fold its factor into `R`, exactly as `split.rs`
  does with `f = √(s(piece)/s(parent))`.
- **Guard.** Error out, or fall back to identity, if `cond(R) > 1e3`. When nothing is rotatable (STO-3G, segmented
  bases), take the identity path, which is today's code bit for bit.
- **Walks.** `sr_three_index_binned` (unsplit and split) and `hcore::sr_attraction` iterate GROUP pairs. They
  accumulate the rotated blocks in scratch, back-transform with `Rᵢᵀ ⊗ Rⱼᵀ` into the parent rows, then COPY. Each
  task still owns its rows.
- **The split.** The plan's compact/smooth pieces are built from the rotated shells. A rotated column is then wholly
  compact or wholly smooth at λ = 1, ω = 1, so it has one kept call or none.
- **Derivatives.** Plan it now, apply it later. `deriv.rs`, `strain.rs`, the k-point walks, and the pair FT (plan
  item 4) can take the same `R`, since R is constant and commutes with ∂/∂R_A. Every walk must see the SAME
  partition (plan item 8's rule).

Tests (Experimental Protocol):

- **Exactness anchor, written first.** The identity rotation (a basis with nothing to rotate, or a forced-identity
  switch) must be bit-identical to today on J3, V_SR and the energy.
- **Covariance anchor, an independent construction.** With `sr_screen = false`, back-transformed J3 vs today's J3
  ≤ 1e-13 relative (diamond_prim cc-pVDZ, split on and off). Also J3 vs the existing dense-AFT oracle
  (`rsgdf_matches_dense_aft_in_the_trivial_limit`).
- **Energies.** vs today ≤ 1e-11 Ha with screening on, at the build's precision.
- **Artifact hypothesis, stated before measuring.**
  - Correct: the error is at the screening level and shrinks with precision.
  - libint renormalization factor missing: an O(1e-2) relative error, independent of precision.
  - Smooth piece not rotated: an error that depends on ω and does not shrink with precision.
- **Mutants.** Each must miss by ≥ 1e-6:
  - flip the sign of an off-diagonal entry of R;
  - omit the renormalization factor;
  - back-transform with R instead of Rᵀ;
  - rotate the compact piece only.
- **Counter prediction, written down before the run.** Dry ice ω = 1 `n_sr3_triplets` = 10.45 M ± 1% (Python exact
  counter).

### Step 3: item 7, the batched shifted-eri3 call

Do it only if the refit (§5) shows overhead is still more than half the per-call cost after step 2. The test is
blocks bit-identical to the per-call API for every l class up to aux g.

### libcint (c)

Revisit only if, after steps 1 to 3, the SR3 cost per column triplet (1 thread) is still more than 2× PySCF's
0.19 µs (6-thread CPU per column triplet). It is the only option that captures full general contraction, worth
another 1.4–2.5×. It costs a new C dependency and two integral backends.

## 5. What to measure first (no crate edits needed for 1 and 2)

1. **Refit the cost model after 3fea0374.** Rerun
   `OPENBLAS_NUM_THREADS=1 target/release/examples/pbc_triplet_bench --basis cc-pvdz --aux cc-pvdz-ri --reps 10`
   after rebuilding on the patched shim, on a quiet box, as the only job. Put the new per-class table into
   `sr_gc/bench_sr3.txt` and rerun `gc_model.py` for all three variants (`""`, `_nz`, `_rot`). That fixes where
   (d) falls in 1.8–2.8×.
2. **Separate per-call from per-primitive cost within each l class.** Extend `pbc_triplet_bench` with a primitive
   scan (a `--prim-scan` flag):
   - one l class at a time, shells with 1..9 primitives, same geometry;
   - time the slope and intercept per class;
   - template: `pbc_parallel_contention.rs`, for its fixed triple list, min-of-reps and checksum.

   This decides (a) against (c) and (d) quantitatively, and says how much item 7 can save.
3. **After step 1:** dry ice SR3 stage time at 1 and 6 threads, s1 vs s2, with the predicted count ratio written
   down first.
4. **ω = 0.3, not yet measured.** At ω = 0.3, `λω²/2 = 0.045`, so no orbital primitive is smooth and the `twocall`
   collapse does not happen. (d)'s gain there comes from primitive reduction only. Run
   `gc_count.py dryice 0.3 _rot` before claiming anything for small ω (it takes about 30–60 s of CPU; not run here
   because of the CPU stop).

Sources: [libint engine.impl.h (master)](https://github.com/evaleev/libint/blob/master/include/libint2/engine.impl.h),
[libint engine.h](https://github.com/evaleev/libint/blob/master/include/libint2/engine.h).
