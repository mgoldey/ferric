# WIKI APPEND — dRPA on the integral-direct path (2026-09-05, branch feat/drpa-direct-strips)

Append target: `wiki/amplitude-threshold-drpa.md` (new dated section) and a
one-line cross-reference from `wiki/amplitude-threshold-lmp2.md` §31.

## dRPA ported onto the integral-direct assembly (no global B)

`ferric_mp2::drpa_amplitude::amplitude_drpa_direct` (+ `_with_virtuals`
mutation entry): `localized_spaces` front end →
`lmp2_direct::assemble_ragged_direct_local` with scale = 2.0 (the `scale`
parameter put there for exactly this — the Eq-8 mask acts on the SCALED
integrals |B| = |2(ia|jb)|, same as the global path) → the UNCHANGED
masked Riccati fixed point. The solver was extracted VERBATIM from
`amplitude_drpa_from_basis` into a shared `riccati_masked_solve(rg, f_oo,
cfg)` so both paths run literally the same code (all 8 pre-existing
global-path anchors re-run green after the extraction). The global
(naux, no·nv) B and its N⁵ whitening GEMM are never formed on the direct
path; `compute_reference: true` still builds a global B inside the
canonical plasmon reference (documented on the entry point — disable for
a genuinely global-B-free run).

### Anchors (tests/drpa_direct.rs, protocol order — written and passed BEFORE the bench)

Trivial maps = aux_radius 1e6, virt_radius None, ao_tail 0, schwarz_skip
0, batch_merge 1. Debug-opt profile, OPENBLAS_NUM_THREADS=1.

- H2/STO-3G, trivial maps, eps=0: dE vs canonical plasmon = −7.3e-16;
  matches the proof notebook's −0.0126072623 (the cross-artifact lock).
- water/6-31G/cc-pvdz-ri fc=1, trivial maps, eps=0:
  |vs canonical plasmon| = 1.66e-13, |vs existing global-B path| =
  1.77e-13 (121 iters, strips span naux — trivially-trivial asserted).
  The 1.1e-14-class of the 2026-08-16 port anchor is reproduced at the
  same order; the extra ~1e-13 is the domain-fit-at-huge-radius vs
  whitened-Gram B reassociation floor passed first-order through the
  non-variational energy.
- alkane_4 fc=4, eps=1e-3, trivial maps, DIIS(6): direct vs global-B
  path |dE| = 1.08e-13 — the MEASURED reassociation floor at finite eps
  (bar asserted at 1e-8; identical keep_fraction 0.1547 on both paths,
  i.e. zero borderline Eq-8 pattern flips on this system).
- alkane_8 fc=8, coul, eps=1e-3, gate 0.7, production maps
  (r_aux=10 / r_virt=12 / ao_tail=1e-3): map error 8.9e-6 vs eps
  truncation 1.01e-2 (vs canonical plasmon) — 1100× sub-dominant.
  (C8 is below the strip-locality onset — strips 700/75 = naux/nv — so
  only the energy sub-dominance is asserted, per the LMP2 §28 protocol
  note; strip saturation is already established for the SHARED assembly
  in lmp2 §28-§29 and was not re-measured here.)

### Mutation arms (each seen to fail / be loud)

Permanent arms (each_map_gutted_is_loud, alkane_4/eps=0 vs trivial base):
virt_radius=2 → |dE| 6.3e-2; ao_tail=0.3 → |dE| 2.0e-1; schwarz_skip=10
→ Riccati residual goes NaN and the solve HARD-ERRORS (loud, accepted as
such in the arm — never a silent zero; fp_max_iter capped 60 in that arm
so the diverging iteration doesn't spin); aux_radius=4 → |dE| 8.3e-5.

Temporary mutations of the new driver (applied, watched fail, reverted):
- m1 scale 2.0 → 1.0: H2 anchor fails (+9.2e-3), water eps=0 fails
  (+9.7e-2), finite-eps C4 fails (3.2e-1).
- m2 frozen_core+1 into localized_spaces: water eps=0 fails (+4.1e-2).

### Wall-clock, existing global-B dRPA vs direct (release, ferric-limited
4G/3600M, FERRIC_MEM_BUDGET_GB=2, CONTENDED box — ±10%; DIIS(6), gate
0.7, compute_reference off; direct maps = production + skip 1e-5 +
batch_merge 4)

| sys | eps | E(global) | E(direct) | dE | t_global | t_direct |
|-----|-----|-----------|-----------|-----|----------|----------|
| C8  | 1e-3 | −0.8586617084 | −0.8586704384 | 8.7e-6 | 3.20 | 5.47 |
| C8  | 1e-4 | −0.8682389118 | −0.8682337088 | 5.2e-6 | 16.94 | 17.91 |
| C12 | 1e-3 | −1.2773810284 | −1.2773977482 | 1.7e-5 | 7.17 | 6.60 |
| C12 | 1e-4 | −1.2933065098 | −1.2933071267 | 6.2e-7 | 28.66 | 29.80 |

Reading (kept separate): parity at C8-C12, iterations identical (25/27
both paths — the mask, not the assembly, sets solver work). No speedup is
claimed at these sizes: C8-C12 is below the assembly's locality onset and
at eps=1e-4 the ring-product solve (shared by both paths) dominates the
wall. The direct path's payoff at this scale is the MEMORY SHAPE (per-
occupied strips instead of the global B); the wall-clock crossover for
the shared assembly is at ~C20 per lmp2 §31 and was not re-measured for
dRPA. dE between the paths is the locality-map error (trivial-map dE is
1e-13-class, above), sub-dominant to eps truncation in every row.

## LinLCCD: DEFERRED (design/feasibility note, honest verdict)

Verdict: NOT ported this session. The (ia|jb) RHS alone would be a
cosmetic port; the two ladder blocks are the real global-tensor
consumers, and one of them has a STRUCTURAL blocker.

What the current `ferric_cc::linlccd_amplitude` consumes:
1. Global B via `assemble_basis` (RHS (ia|jb) blocks) — directly
   replaceable by `assemble_ragged_direct_local(scale=1.0)`, same as the
   dRPA port. Clean in isolation, pointless alone (see 2).
2. hh ladder OOOO block: built from the FULL AO `eri3_tensor`
   (naux, nao, nao) — LARGER than the global B it would be saving —
   then `transform_3center_oo` + global-metric whitening.
3. pp ladder (Full variant): whitened Bvv (naux, nv²), gathered per pair.

Feasibility per block:
- hh: an occ-occ strip extension is cheap and PSD-safe. Measured basis
  for the size estimate (lmp2 §29 counters): surviving partners per
  occupied saturate at 10.8 (erfc) / 19.7 (coul/1e-3) / 35.2 (coul/1e-4)
  vs 127-199 virtual strip columns — occ-occ columns add ~8-18% strip
  bytes. Only gate-surviving (i,k) columns are needed:
  Boo ≈ naux_strip × (no × partners) — at C48/coul ≈ 1279 rows × 20 cols
  per occupied ≈ 0.2 MB/occupied, 30 MB total, vs the ~3.5 GB global
  Bvv and the (naux, nao²) AO tensor. Whitening Boo with the GLOBAL
  metric keeps oo_g = BoõᵀBoõ an exact Gram → the hh-extended matvec
  stays provably PSD (the proof-notebook license for the CG solve).
- pp: the STRUCTURAL blocker. The solver contract
  (`solve_ragged_with`: matvec MUST be SPD on the pattern) is licensed by
  the notebook's "both ladder blocks are RI Grams of ONE consistent
  whitened B, hence PSD". A domain-local same-kernel fit (the direct
  path's formulation, different V_DD per pair-domain group) does NOT
  produce a single consistent Gram — a pp block assembled that way loses
  the PSD guarantee, and CG convergence with it is unproven. The
  PSD-safe alternative (per-pair on-demand vv columns with FULL aux rows
  + global whitening) keeps the Gram structure but needs per-pair
  d²×d² ladder blocks precomputed before CG (≈12 MB/pair at d≈35,
  hundreds-to-thousands of surviving pairs → tens of GB at C32+), or
  per-iteration recomputation (a naux·d² transform per pair per CG
  iteration — the "rebuilt per pass" defect class, on purpose). Neither
  is clean.
- Route that WOULD be clean, not attempted here: prototype-first
  (Matt's convention) a PSD-verified domain-fitted pp ladder in Python —
  measure the operator's smallest eigenvalue under the domain fit and
  CG behavior on it; if PSD survives empirically with a bounded
  perturbation argument, the Rust port licenses itself. Until that
  measurement exists, porting LinLCCD would swap a memory problem for an
  unproven-solver problem.

Interim recommendation recorded: LinLCCD stays on the global path; its
hh (Hh variant) could drop the AO eri3_tensor for a direct Boo pass as a
separate small PR (PSD-safe per the above), independent of the pp
question.
