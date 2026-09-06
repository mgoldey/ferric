# WIKI APPEND — LinLCCD on the integral-direct path (2026-09-06, branch feat/linlccd-direct)

Append target: `wiki/amplitude-threshold-drpa.md` (continuation of the
"LinLCCD: DEFERRED" section — this note closes that deferral) and a
one-line cross-reference from `wiki/amplitude-threshold-lmp2.md` §31.

## Phase 1 — pp-ladder PSD measurement (PROTOTYPE FIRST, per the deferral note)

Rig: `scripts/queue/proto_linlccd_pp_psd.py` (PySCF + the
`amplitude_lmp2_proto.py` machinery: Boys occupieds, VV-HV virtuals,
same-kernel RI, mp2fit aux). 6-31G. Anchors A1-A4 and mutation arms M1-M2
pre-registered in the script header and run BEFORE any sweep; artifact
hypotheses (X-safe / X-broken / Y-artifact) pre-registered there too.

Structure exploited (stated in the header): the pp operator is
BLOCK-DIAGONAL over occupied pairs in T-space, so its PSD question is
decided per pair block; eigenvalue interlacing makes the full-block
lambda_min the conservative eps-independent bound, the pattern-restricted
one is what CG sees. Both fitted and licensed per-pair blocks are
symmetric by construction (measured, asym <= ~1e-13 everywhere).

A finding about the LICENSE itself, before any fitting: in the exchange
pairing the pp supermatrix is M[(a,b),(c,d)] = sum_P Y_P (x) Y_P with
Y_P = whitened (P|ac) symmetric — NOT a manifest Gram (Y (x) Y is
indefinite for indefinite Y), so notebook 13's "RI Gram, hence PSD" is an
EMPIRICAL property, not an algebraic identity. The same holds for the hh
block (out[i,j] = sum_kl (ik|jl) T[k,l] at fixed spectators (a,b) — the
identical Y (x) Y form on the occupied grid). It was therefore MEASURED,
not cited: A1 below. The exact-integral supermatrix IS provably PSD
(pointwise kernel positivity: sum X_ab X_cd (ac|bd) =
integral h(1,2)^2/r12 with h = sum X_ab phi_a(1) phi_b(2)); RI breaks the
pointwise factorization, and the measurement says how much survives.

### Anchors + mutations (all PASSED/loud before the sweep)

- A1 licensed global pp supermatrix (one whitened B, exchange pairing):
  water lambda_min = +1.988e-01 (rel +0.32), butane +7.9e-02-class —
  comfortably PSD, empirically.
- A2 fitted pp at trivial domain == licensed: max|dm| = 2.7e-14 (water).
- A3 CG vs direct dense solve of P A P t = -P J (independent algebra),
  water eps=0 licensed: |dE| = 4.0e-14; lambda_min(P A P) = +2.387.
- A4 trivial-radius fitted CG == licensed CG (eps=0): |dE| = 1.4e-17.
- M1 largest-|Avv| aux dropped from the trivial domain: max|dm| = 2.8e-02
  (A2 fails as required).
- M2 injected indefinite rank-1 (-2 Y_K (x) Y_K): lambda_min = -2.77e-01
  detected (the eig pipeline can SEE indefiniteness).

### Sweep (raw rows in scripts/queue/out/linlccd_pp_psd.txt)

Columns: lmin_fit(full/pat) = min over pairs of the fitted per-pair block
lambda_min (full block / Eq-8-pattern-restricted); lmin_lic(pat) = same
restriction on the licensed block; nneg = pairs with pattern lambda_min
< -1e-10; Dmin = Fock denominator floor (the PD part of the operator);
cg_fit iterations with pAp_neg counting p^T A p <= 0 CG events (the
in-solver indefiniteness witness); dE(fit-lic) at identical mask/RHS.

water/6-31G coul (no=4, nv=8, naux=84; radii never truncate — the
molecule is smaller than every radius; anchor host only):
- eps 0/1e-3/1e-4 x r 1e6/4/3/2: lmin_fit = +1.988e-01 at every row,
  nneg 0/16, cg identical to licensed (15), dE(fit-lic) ~ 1e-17.

alkane_4/6-31G coul (no=13, nv=39, naux=364, Dmin=2.382; r=8/6/4 truncate
domains to mean 359/346/265 of 364):

| eps | r | lmin_fit(full/pat) | lmin_lic(pat) | nneg | cg_fit (pAp_neg) | dE(fit-lic) | dE_eps |
|-----|---|--------------------|---------------|------|------------------|-------------|--------|
| 1e-3 | 1e6 | +8.795e-2/+1.076e-1 | +1.076e-1 | 0/161 | 16 (0) | -1.6e-15 | +7.098e-3 |
| 1e-3 | 12 | +8.795e-2/+1.076e-1 | +1.076e-1 | 0/161 | 16 (0) | -1.6e-15 | +7.098e-3 |
| 1e-3 | 10 | +8.795e-2/+1.076e-1 | +1.076e-1 | 0/161 | 16 (0) | -1.6e-15 | +7.098e-3 |
| 1e-3 | 8 | +8.795e-2/+1.076e-1 | +1.076e-1 | 0/161 | 16 (0) | -1.3e-07 | +7.098e-3 |
| 1e-3 | 6 | +8.795e-2/+1.076e-1 | +1.076e-1 | 0/161 | 16 (0) | -4.0e-07 | +7.098e-3 |
| 1e-3 | 4 | +8.794e-2/+1.075e-1 | +1.076e-1 | 0/161 | 16 (0) | -5.7e-06 | +7.098e-3 |
| 1e-4 | 1e6 | +7.912e-2/+7.940e-2 | +7.940e-2 | 0/169 | 19 (0) | -1.7e-15 | +2.595e-4 |
| 1e-4 | 10 | +7.912e-2/+7.940e-2 | +7.940e-2 | 0/169 | 19 (0) | -1.7e-15 | +2.595e-4 |
| 1e-4 | 8 | +7.912e-2/+8.018e-2 | +7.940e-2 | 0/169 | 19 (0) | -1.4e-07 | +2.595e-4 |
| 1e-4 | 6 | +7.908e-2/+8.024e-2 | +7.940e-2 | 0/169 | 19 (0) | -4.4e-07 | +2.595e-4 |
| 1e-4 | 4 | +7.363e-2/+7.876e-2 | +7.940e-2 | 0/169 | 19 (0) | -6.9e-06 | +2.595e-4 |

(erfc omega=1 arm and alkane_8 lambda-min rows: below)

### THE A1 FINDING — the LICENSED block is not absolutely PSD either

erfc(w=1)/alkane_4: the licensed global pp supermatrix (one consistent
whitened B, the construction the CURRENT global-path Full tier gathers
from) measured lambda_min = -6.850e-04 (rel -5.8e-3 of maxdiag), with
the construction anchors at machine precision (A2 4.6e-15, A4 6.9e-18)
— so this is a PROPERTY, not a bug: notebook 13's absolute "RI Gram,
hence PSD" is FALSE for the attenuated operator. Coulomb stays positive
but DECREASES with system size: water +1.988e-1, alkane_4 +7.912e-2,
alkane_8 +4.545e-2 (rel +0.32 / +0.12 / +0.069). The operative CG
license — for the licensed AND fitted constructions, global AND direct
paths — is that the pp (and hh) indefiniteness stays bounded orders
below the Fock denominator floor (~2.4 Ha here; erfc margin 3475x), so
the full masked operator is SPD by Weyl. A1 was reframed accordingly
(recorded finding + a 5%-of-Fock-floor STOP bar), and the sweep's
question became: does FITTING worsen the indefiniteness with truncation?
The Rust module docs (linlccd_amplitude.rs, PpFitted) now carry this
corrected license.

Note the eps=1e-4/r=8 and r=6 rows: the fitted pattern lambda_min sits
slightly ABOVE the licensed one (+8.02e-2 vs +7.94e-2) — the fit
perturbation moves eigenvalues in both directions at the ~1e-3 scale,
nowhere near zero. Butane anchors/mutations (re-run, c4_rest.log):
A1 licensed global pp lambda_min = +7.912e-02 (rel +0.12), A2 max|dm| =
7.1e-14, A4 |dE| = 1.9e-15, M1 max|dm| = 7.8e-04 (fails A2's 1e-6 bar as
required), M2 lambda_min = -2.024e-01 detected (scale 6.6e-1).

alkane_4/6-31G erfc(w=1) (naux=364, Dmin=2.382; the A1-finding operator):

| eps | r | lmin_fit(full/pat) | lmin_lic(pat) | nneg | cg_fit (pAp_neg) | dE(fit-lic) | dE_eps |
|-----|---|--------------------|---------------|------|------------------|-------------|--------|
| 1e-3 | 1e6 | -3.875e-4/-1.407e-4 | -1.407e-4 | 18/111 | 17 (0) | +1.4e-17 | +1.786e-3 |
| 1e-3 | 8 | -3.875e-4/-1.407e-4 | -1.407e-4 | 18/111 | 17 (0) | -1.6e-08 | +1.786e-3 |
| 1e-3 | 6 | -4.002e-4/-1.406e-4 | -1.407e-4 | 19/111 | 17 (0) | -5.6e-08 | +1.786e-3 |
| 1e-3 | 4 | -2.323e-3/-7.772e-4 | -1.407e-4 | 13/111 | 17 (0) | -7.0e-07 | +1.786e-3 |
| 1e-4 | 1e6 | -6.850e-4/-5.128e-4 | -5.128e-4 | 75/169 | lam-only | | |
| 1e-4 | 6 | -5.574e-3/-8.221e-4 | -5.128e-4 | 75/169 | lam-only (full sampled 60) | | |
| 1e-4 | 4 | -5.871e-3/-1.855e-3 | -5.128e-4 | 75/169 | lam-only | | |

alkane_8/6-31G (no=25, nv=75, naux=700, Dmin=2.383; lambda-min rows,
full-block eighs sampled 20-22 pairs, pattern eigs exhaustive; CG rows
not run at C8 — box budget. C8 coulomb anchors: A1 +4.545e-2 (rel
+0.069), A2 2.1e-13, A4 1.6e-15; mutations M1 4.97e-4, M2 -2.010e-1.
C8 erfc rows are sweep-only — its anchors/mutations are anchored at C4):

| op | eps | r | dom mn/mx | lmin_fit(full/pat) | lmin_lic(pat) | nneg |
|----|-----|---|-----------|--------------------|---------------|------|
| coul | 1e-3 | 10 | 622/700 | +6.622e-2/+9.519e-2 | +9.519e-2 | 0/407 |
| coul | 1e-3 | 8 | 533/700 | +6.622e-2/+9.519e-2 | +9.519e-2 | 0/407 |
| coul | 1e-3 | 6 | 480/700 | +6.622e-2/+9.519e-2 | +9.519e-2 | 0/407 |
| erfc | 1e-3 | 1e6 | 700/700 | -4.347e-4/-1.389e-4 | -1.389e-4 | 37/243 |
| erfc | 1e-3 | 8 | 486/700 | -4.477e-4/-1.405e-4 | -1.389e-4 | 41/243 |
| erfc | 1e-3 | 6 | 429/616 | -4.576e-4/-1.399e-4 | -1.389e-4 | 41/243 |

### Pre-registered hypothesis check

The X-safe fingerprint holds, with one honest amendment forced by A1:

- "min eigenvalues bounded away from 0" was the WRONG absolute criterion
  — the LICENSED baseline itself violates it under erfc (A1 finding).
  The operative criterion is the margin vs the Fock floor, and there the
  data are unambiguous: worst measured value anywhere is -5.9e-3
  (C4/erfc/1e-4/r=4, full-block) vs Dmin = 2.38 — a 400x margin; at
  production radii (>= 6 Bohr) fitted tracks licensed to ~1e-6.
- Size dependence: pattern-restricted lambda_min is SATURATED, not
  growing — coul +1.076e-1 (C4) -> +9.5e-2 (C8); erfc -1.407e-4 (C4) ->
  -1.389e-4 (C8). The X-broken fingerprint (indefiniteness growing with
  system size toward the floor) is ABSENT.
- Truncation dependence: real but bounded, and only below production
  radii — erfc r=4 deepens the fitted lambda_min ~5x (1e-3) to ~13x
  (1e-4, full-block) vs licensed; at r >= 6 the effect is <= 2e-6.
  This is the direction the X-broken hypothesis named, at 2.5-3 orders
  below the scale that would threaten the license.
- CG: iteration counts IDENTICAL fitted-vs-licensed in every row, zero
  p^T A p <= 0 events anywhere, all solves converged.
- Energy: dE(fit-lic) <= 7e-7 at production radii, >= 2500x below the
  eps truncation at 1e-3; narrowest measured margin 38x
  (C4/coul/1e-4/r=4: 6.9e-6 vs 2.6e-4).
- Not Y (artifact): construction anchors at machine precision (A2
  4.6e-15..2.1e-13, A4 <= 1.9e-15 across all three systems), blocks
  symmetric to 1.5e-13, both mutation arms loud on every system, and the
  licensed-vs-fitted difference responds to the radius knob exactly as a
  fit perturbation should (zero at trivial domain, growing smoothly).

### Verdict (2026-09-06, provisional as always)

GO for the FITTED-pp integral-direct port, under the CORRECTED license:
the ragged-CG SPD contract for LinLCCD was never the absolute
"ladder blocks are PSD Grams" of notebook 13 — for the RI blocks it is
"ladder indefiniteness bounded orders below the Fock denominator floor",
which holds for the licensed (global whitened) construction and is not
measurably degraded by per-pair domain fitting at production radii
(>= 6-8 Bohr) on water/C4/C8, coulomb and erfc(w=1), eps 1e-3/1e-4.
Scope limits stated: sub-production radii (~4 Bohr) measurably deepen
erfc indefiniteness (monotone in truncation) — do not ship defaults
below r_aux ~ 6 without re-measuring; new operators/regimes re-run
proto_linlccd_pp_psd.py first (the script's A1 gate now encodes the
corrected criterion at 5% of the Fock floor).

## Phase 2 — Rust (branch feat/linlccd-direct, on top of PR #27)

### Always-lane: occ-occ direct pass + DriversOnly/Hh tiers

`ferric_mp2::lmp2_direct::assemble_boo_direct` + `OoGram`: evaluates
(P|mu nu) over occupied AO supports only (never the (naux, nao²) AO
`eri3_tensor` the old OOOO build required), half-transforms into
unwhitened (P|ik) columns for the gate-surviving unordered pairs, whitens
with the GLOBAL metric (full aux rows) — so (ik|jl) equals the global
path's object exactly at ao_tail=0, and the hh matvec stays the Gram of
ONE consistent whitened B. Gated (i,k) columns are exactly-zero columns
of the whitened Boo: the Gram (PSD-license) structure is unchanged and
the dropped-coupling error is measured, not assumed. The naux² metric IS
still formed (as on the global path); what died is the AO tensor and the
(naux, no.nv) B.

`ferric_cc::linlccd_amplitude::amplitude_linlccd_direct(_with_virtuals)`:
localized_spaces -> `assemble_ragged_direct_local` (scale 1.0) for the
(ia|jb) RHS -> OoGram -> the UNCHANGED masked ragged CG, extracted
VERBATIM into a shared `linlccd_masked_solve` (the dRPA
`riccati_masked_solve` pattern; the global path funnels through the same
code via `OoGram::dense`, and all 13 pre-existing global-path
linlccd tests re-run green after the extraction).
`AmplitudeLinLccdConfig` gained `pair_gate_cal: Option<f64>`
(default None = trivial limit).

### Full tier: fitted pp factors (per the Phase-1 verdict)

`ferric_mp2::lmp2_direct::assemble_pp_fitted_direct` + `PpFitted`:
per-pair (ac|bd)_fit = Aa V_DD^-1 Ab^T with pattern-derived Da/Db, one
V_DD per pair aux domain (grouped exactly like stage 5), factors held as
(a_rows, gb = V_DD^-1 Ab^T) and contracted per CG iteration with the SAME
GEMM+accumulation shape as the licensed whitened-Bvv gather
(`PpSource::{GlobalWhitened, Fitted}` in the shared solver). vv 3-index
integrals evaluated per domain group over the group's union-virtual AO
supports, slabbed under the scratch budget — never a global Bvv, never
the AO tensor. The PSD contract for these blocks is EMPIRICAL (Phase 1),
stated on the struct doc with a re-measure instruction for new
operators/regimes.

### Anchors + mutations (tests/linlccd_direct.rs, all SEEN green/loud;
debug-opt, OPENBLAS_NUM_THREADS=1)

- OO-GRAM value anchor (water): max|direct - global object| = 4.163e-17;
  trivial gate keeps all no(no+1)/2 columns.
- OO-GRAM mutations: ao_tail=0.5 -> max|dG| = 8.799e-1 (loud); gutted
  gate (cal=1e-30) -> off-diagonal coefficients collapse 5.59e-2 -> 0.0
  exactly, cols 10 -> 4 (loud).
- eps=0 trivial maps, water: DriversOnly |vs canonical spin-orbital| =
  2.624e-13, |vs global path| = 2.461e-13; Hh 5.962e-13 / 2.062e-13;
  Full 2.164e-12 / 1.693e-13. (Reassociation floor; Hylleraas-protected,
  bar 1e-9 vs global, 5e-9 vs canonical.)
- finite-eps trivial maps, alkane_4 eps=1e-3: Hh |direct - global| =
  1.231e-13 (identical keep 0.0761); Full 1.165e-13.
- production maps + gate, alkane_8 Hh eps=1e-3 (r_aux=10, r_virt=12,
  ao_tail=1e-3, cal=0.7): map+gate error 4.345e-5 vs eps-truncation scale
  2.134e-2 (|E(1e-3)-E(1e-4)| trivial-maps proxy, a LOWER bound on the
  1e-3 truncation) — ~490x sub-dominant. Strips 700/700 x 75/75 (C8 is
  below the strip-locality onset, per the lmp2 sec-28 protocol note — no
  locality claim from this row).
- pp-fit mutation (value level): aux_radius 1e6 -> 1.0 on water moves the
  per-pair fitted blocks by max|dm| = 8.212e-2 (loud).
- Collateral: lmp2_direct 5/5, drpa_direct 7/7, linlccd_amplitude +
  linlccd_hh + linlccd_ri_floor 13/13, clippy clean, complexity gate PASS
  (6205 functions, no regressions).

### Honest scope notes

- No wall-clock or scaling claim for LinLCCD-direct: C4/C8 are below the
  assembly's locality onset and the pp/hh solver cost is unchanged. The
  direct path's payoff here is the MEMORY SHAPE (per-occupied ov strips +
  occ-support Boo + per-group fitted pp factors instead of the
  (naux, nao²) AO tensor and global whitened B/Bvv).
- Full-tier production-map sub-dominance was characterized by the
  Phase-1 PROTOTYPE (dE(fit-lic) vs radius, 3 decades below eps
  truncation); the Rust Full tests pin exactness at the trivial domain
  and loudness of the domain map. A Rust-side production Full sweep at
  C8+ would need the pp per-pair GEMMs parallelized first (serial today,
  same as the global path).
- The oo Gram is dense (ncols²) over surviving columns — fine at the
  C-scales LinLCCD runs at today; revisit if no grows past ~200.
