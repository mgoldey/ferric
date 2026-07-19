# A24-subset: RS-MP2-RPA decisive experiment (2026-06-10)

The pre-registered success criterion from
`docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md`:

> B at one fixed ω (0.2–0.3 Bohr⁻¹) beats plain MP2 MAE on the
> dispersion-bound dimers without per-system fitting.

**Verdict (chronological — see sections below): falsified at cc-pVDZ and
aug-cc-pVDZ** (B never beats MP2; basis-incompleteness underbinding dominates),
**criterion MET marginally at aug-cc-pVTZ** (B 0.139 vs MP2 0.143 at ω=0.42,
the predicted π-overbinding mechanism). The naive sum (A) is worse everywhere.

**UPDATE 2026-07-19 — the aTZ "win" is now flagged as likely inside the RI-fit
noise floor.** See "RI-fit noise floor" section below: the 0.004 kcal/mol
(0.143→0.139) effect size the aTZ verdict rests on is comparable to (63%) or
smaller than (4.6× smaller) the empirically-measured RI-fit error on the exact
same operator/aux, depending on which noise probe you compare against. Treat
the aTZ "criterion met" line above as **not yet statistically established**
until re-run with either a tighter aux basis or a non-RI cross-check on the
actual A24 fragments (both out of scope for this pass — see caveats below).

## RI-fit noise floor (2026-07-19)

Before trusting the aTZ verdict above, this section asks: is the 0.004
kcal/mol effect size (MP2 MAE 0.143 vs B MAE 0.139 at ω=0.42) bigger than the
RI-fit error already baked into `E_MP2[Coulomb]` (the term B shares with plain
MP2) at this basis/aux? `docs/VALIDATION.md`'s pre-existing claim was a
generic "~mHa per operator" figure with no system-specific measurement behind
it — this is a direct, empirical bound for the exact aug-cc-pVTZ /
aug-cc-pvtz-rifit combination the aTZ sweep used.

**Pre-registered criterion (written before running the probe):** if either
(a) canonical (exact 4-index) vs RI-MP2[Coulomb] with the actual aug-cc-pvtz-
rifit aux, or (b) RI-MP2[Coulomb] with aug-cc-pvtz-rifit vs a different
independently-optimized aux (def2-tzvpp-rifit) for the same orbital level,
differs by an amount comparable to or larger than 0.004 kcal/mol on a small
test system, that is grounds to call the aTZ win statistically
indistinguishable from RI-fit noise. If both differences are comfortably
smaller (>5× smaller), that's grounds to call the win distinguishable from
noise at this level (though still a small, 1-of-4-systems effect).

**Method:** `crates/ferric-mp2/tests/ri_noise_floor_atz.rs`
(`ri_fit_noise_floor_h2_augccpvtz`, `#[ignore]`d, run via
`cargo test -p ferric-mp2 --release --test ri_noise_floor_atz -- --ignored
--nocapture`). RHF solved once with exact 4-index J/K (no DF-JK SCF confound
— that's a separate, much smaller ~1e-7 Ha convergence-tightness effect
documented in `docs/df-noise-floor-scope.md` (the DF-JK SCF density-level
self-consistency floor, not the RI-MP2 correlation-energy fitting error) and
NOT what this section measures). `E_MP2[Coulomb]` computed three ways on
the same orbitals: `canonical_mp2` (exact, O(N^5)+, `crates/ferric-mp2/src/
canonical.rs`), `ri_mp2` with `aug-cc-pvtz-rifit` (the aux the A24-subset aTZ
sweep used), and `ri_mp2` with `def2-tzvpp-rifit` (a different, reasonable
aux for the same triple-zeta orbital level).

**System:** H2/aug-cc-pVTZ. This is a lower-bound probe, not the A24 systems
themselves — a water/aug-cc-pVTZ probe (closer in size to an actual A24
fragment) was attempted but `canonical_mp2`'s doc comment warns it is
"O(N^5) or worse... not intended for production use on large molecules," and
it did not complete in the compute budget available on a heavily-contended
shared box (one attempt ran 30+ min before being interrupted, a second ran
44+ min; both killed to make room for the decisive, cheap H2 measurement).
This is noted as **incomplete**, not as a negative result — see "What's not
done" below.

**Result (H2/aug-cc-pVTZ, `E_MP2[Coulomb]`):**

| comparison | Δ (Ha) | Δ (kcal/mol) | vs 0.004 kcal/mol effect size |
|---|---|---|---|
| canonical (exact) vs RI(aug-cc-pvtz-rifit) | 3.976e-6 | **0.0025** | 63% of the effect |
| RI(aug-cc-pvtz-rifit) vs RI(def2-tzvpp-rifit) | 2.919e-5 | **0.0183** | 4.6× the effect |

**Reading:** by probe (a), the pure RI-vs-exact fitting error on the operator
that IS `E_MP2[Coulomb]` (shared by both MP2 and B) is already 63% the size of
the entire aTZ effect on a minimal 2-atom system — before any CP/interaction-
energy differencing that could partially but not exactly cancel it across the
5 fragment terms (dimer + 2 monomers + 2 ghost-monomers) that make up a single
CP binding energy. By probe (b), swapping to a different, equally defensible
aux basis moves `E_MP2[Coulomb]` by nearly 5× the claimed effect — i.e. an
aux-basis choice ferric did NOT make (def2-tzvpp-rifit instead of
aug-cc-pvtz-rifit) would by itself swamp the entire "B beats MP2" margin.

**Verdict: the aTZ "criterion met" result is not distinguishable from RI-fit
noise at this level.** Both noise probes are the same order of magnitude as,
or larger than, the 0.004 kcal/mol effect the verdict rests on. This does not
prove B is wrong or that the aTZ π-overbinding mechanism is fake (the physical
argument — MP2 genuinely overbinds ethene dimer near the basis limit, and a
screening correction should help — is independently plausible and matches the
stacked-system aDZ/aTZ results elsewhere in this file). It means the specific
*margin of victory* over MP2 at ω=0.42 cannot currently be trusted to be real
rather than an RI-fit artifact, without either (i) a converged/near-CBS aux
basis for the RI fit, or (ii) an exact non-RI cross-check on the actual A24
fragments (both expensive; out of scope here — canonical_mp2 does not scale
to the A24 fragment sizes, see above).

**What's not done:** a water-sized (or A24-fragment-sized) canonical-vs-RI
comparison, which would directly bound the noise floor on a system the same
order of size as the A24 dimers rather than extrapolating from H2. H2 gives a
real, decisive lower bound (the RI-fit error does not vanish at minimal
system size, and is already comparable to the effect), but a larger-system
number would either sharpen or soften this verdict. Left as future work.

## Setup

- Systems: A24 #2 (H₂O·H₂O), #5 (NH₃·NH₃), #14 (C₂H₄·C₂H₄ C2v), #19 (CH₄·CH₄ D3d).
  Geometries + CCSD(T)/CBS refs from the psi4 A24 database (Řezáč & Hobza).
- cc-pVDZ / cc-pVDZ-RI, full-rank dRPA (trunc_thresh = 0), ω in Å⁻¹.
- Counterpoise via ghost atoms (`@`-notation); non-CP values in parentheses in
  the data file. 4 dimers × 5 ω × 5 fragments = 100 runs (`run_a24.py`).

## CP MAE vs CCSD(T)/CBS (kcal/mol)

| ω (Å⁻¹) | MP2 | naive A | Δ-form B |
|---|---|---|---|
| 0.1 | 0.847 | 0.851 | 0.847 |
| 0.2 | 0.847 | 0.882 | 0.847 |
| 0.3 | 0.847 | 0.956 | 0.849 |
| 0.42 | 0.847 | 1.053 | 0.859 |
| 0.6 | 0.847 | 1.108 | 0.893 |

## Reading

At CP/cc-pVDZ every correlated method *underbinds* these dimers (basis
incompleteness dominates the MAE: water −4.04 vs ref −5.01). The Δ-correction
(dRPA[erf] − dMP2[erf]) is screening — it strictly *removes* long-range binding
— so on systems that already need more binding it can only hurt. The regime
where screening should win is where MP2's tail genuinely overbinds (π-stacks
near the basis-set limit, e.g. benzene dimer at aug-cc-pVTZ); that is outside
DZ scope and would be the only justified revival experiment.

Combined with ACONF (`../gmtkn30/README.md`): no fixed-ω win anywhere tested.
Per the spec's falsifier clause, the method is parked and documented; the code
(driver, limit tests, component diagnostics, erf-safe metric, ghost atoms)
stays — it is validated infrastructure.

## Stacked systems + formulation T (2026-06-10, evening)

Stacked/saddle subset (A24 #22 ethene·ethyne, #23 ethene·ethene D2h, #24
ethyne·ethyne D2h — the MP2-overbinding regime), CP, aug-cc-pVDZ + RI-JK,
comparing MP2 / Δ-form B / coupled-rings T (`stacked_adz_mp2_B_T.txt`):

| ω (Å⁻¹) | MP2 MAE | B MAE | T MAE |
|---|---|---|---|
| 0.2 | 0.071 | 0.071 | 0.159 |
| 0.3 | 0.071 | 0.074 | 0.355 |
| 0.42 | 0.071 | 0.104 | 0.619 |
| 0.6 | 0.071 | 0.245 | 0.888 |

Findings: (i) on the one genuinely MP2-overbound system (#24, −0.031), B at
ω=0.3 improves it (−0.021) but T overshoots past the reference already at
ω=0.2 (+0.086) — T's mixed-ring correction is first-order in v_lr where B's
pure-LR rings are third-order, so **B and T do not share an ω scale**; T's
operating window, if any, is ω ≲ 0.1–0.15 Å⁻¹. (ii) At (a)DZ with CP the
dominant residual is basis-incompleteness underbinding, which any screening
correction worsens. The decisive test for T is stacked systems near the basis
limit (aug-cc-pVTZ run queued).

## aug-cc-pVTZ: success criterion MET (marginally) (2026-06-10, night)

Bound 4-dimer subset, CP, aug-cc-pVTZ (`a24_rsmp2rpa_atz.txt`, `results_atz.json`).
CP MAE vs CCSD(T)/CBS (kcal/mol):

| ω (Å⁻¹) | MP2 | naive A | Δ-form B |
|---|---|---|---|
| 0.1 | 0.145 | 0.142 | 0.145 |
| 0.2 | 0.166 | 0.155 | 0.166 |
| 0.3 | 0.143 | 0.275 | **0.142** |
| 0.42 | 0.143 | 0.452 | **0.139** |
| 0.6 | 0.143 | 0.591 | 0.176 |

First fixed-ω win, at the 2012 erfc-optimal ω=0.42 Å⁻¹ = 0.222 Bohr⁻¹, inside
the pre-registered window. Mechanism as designed: at the basis limit the ethene
dimer overbinds at MP2 (−1.197 vs −1.110) and B halves that error (−1.150 at
ω=0.42) while leaving the H-bonded systems nearly untouched. The effect is
small (~3% of MAE) because only 1 of 4 systems is in the overbinding regime —
stacked π systems at aTZ are the growth case (run in progress).

Caveats (read before quoting):
1. **SCF reproducibility noise on the ω=0.1/0.2 rows.** The ethene fragments
   (368 bf) converged to SCF points differing by 2–4e-5 Ha between runs
   (loose-convergence DIIS path dependence); the CP RHF wobbles ±0.03 kcal/mol
   on those two rows — excluded from conclusions pending rerun. Rows 0.3–0.6
   share identical SCF solutions, and the B-vs-MP2 comparison within every row
   uses the same SCF, so the win rows are internally consistent. Future sweeps
   should pin `[scf] energy_conv`.
2. **This sweep used exact 4-index J/K SCF, not RI-JK** (runner patch race);
   slower but strictly more accurate. The aDZ rows and the stacked-aTZ run use
   RI-JK (def2-universal-jkfit).
3. **All-electron correlation** (frozen_core = 0) vs frozen-core CCSD(T)/CBS
   references — a small protocol mismatch, common to every method column, so it
   largely cancels in the method-vs-method comparison.

## aug-cc-pVTZ stacked: formulation T validated (2026-06-10, late)

#23/#24, CP, aug-cc-pVTZ + RI-JK (`stacked_atz_mp2_B_T.txt`; corrected MAEs —
the runner's printed MAE divides by 3 with 2 systems): at the basis limit both
stacks overbind at MP2 (−0.110/−0.258) and **T at ω=0.2 Å⁻¹ cuts the MAE 2.5×
(0.075 vs MP2 0.184)**, ethene stack to −0.010. B moves ≤0.010 (third-order
correction too weak — the ω-scale law of §5 of the methodology doc). T's
operating window confirmed at ω≈0.2 Å⁻¹; no damping needed (required c ≈ 1).
Verdict: **B for mixed bound systems at ω≈0.42; T for π-stack overbinding at
ω≈0.2** — two regimes, one derivation. Scale-up test (benzene dimer) next.

Per-ω corrected stats (MAE/RMSE, N=2, kcal/mol; full table in
`stacked_atz_mp2_B_T.txt`, raw data `stacked_atz.json`):

| ω (Å⁻¹) | MP2 | Δ-form B | coupled T |
|---|---|---|---|
| 0.1 | 0.184/0.198 | 0.184/0.198 | 0.170/0.185 |
| 0.2 | 0.184/0.198 | 0.183/0.197 | **0.075/0.100** |
| 0.3 | 0.184/0.198 | 0.174/0.189 | 0.124/0.141 |

Caveats: N=2, same-sign errors, both saddle-point stacked geometries (not
minima); all-electron correlation vs frozen-core references.
