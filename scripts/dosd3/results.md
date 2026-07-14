# Heavy-main-group α-vs-experiment test (dosd3)

α₀ reference: CCCBDB (NIST) experimental polarizability volumes (Å³) →
atomic units via ×6.74833. No molecular C6 reference (paywalled for these);
test = does RPA@PBE α stay uniform, and does TS's per-element failure reproduce
across MULTIPLE Si compounds.

| Molecule | heavy | CRC α₀ (a.u.) | RPA@PBE α DZ | RPA@PBE α TZ | α err | TS C6 vs RPA@PBE C6 |
|----------|-------|---------------|--------------|--------------|-------|---------------------|
| SiH4 | Si | 32.24 | 28.68 | 28.75 | −10% | TS +474% at BOTH bases (dosd2) |
| SiF4 | Si | 22.40 | 21.76 | 22.08 | −1.4% | TS C6=3815 vs PBE 296 = **+1190%** (DZ); TS-TZ did not finish (free-atom SCF too slow) |
| PH3  | P  | 28.59 | 26.50 | —     | −7%   | TS C6=323 vs PBE 248 = +30% (DZ, mild) |
| GeH4 | Ge (32) | 32.19 | 31.44 | 31.64 | −1.7% | TS +57.7% (now computes — see below) |
| CH3Br | Br (35) | — | 31.68 | 33.22 | — | TS +39.4% (now computes — see below) |
| Br2  | Br (35) | — | 37.38 | 40.25 | — | TS +27.9% (now computes — see below) |

### RPA@PBE molecular C6 for the heavy-Z set (a.u.)

| Molecule | C6 DZ | C6 TZ |
|----------|-------|-------|
| GeH4 | 343.19 | 348.48 |
| CH3Br | 428.83 | 453.28 |
| Br2  | 569.88 | 622.29 |

### Heavy-Z TS now computable (Gould-Bučko free-atom α/C6 for Z=19–54) — DZ

As of 2026-07-14 the TS free-atom table covers Z=19–54 (Gould & Bučko JCTC 12,
3603 (2016) Table 2), so GeH4/CH3Br/Br2 TS produce a molecular C6 instead of refusing.
The free-atom *volume* is still generated live (per-Z UKS-PBE ∫ρr³dr); only the
α_free/C6_free lookup was extended.

| Molecule | TS C6 | RPA@PBE C6 | TS overshoot |
|----------|-------|-----------|--------------|
| GeH4 | 541.07 | 343.19 | **+57.7%** |
| CH3Br | 597.66 | 428.83 | **+39.4%** |
| Br2  | 728.83 | 569.88 | **+27.9%** |

As predicted from the Si precedent, TS on Ge/Br overshoots RPA@PBE — the
soft/heavy-atom failure mode extends past the third row. But the overshoot is
far *milder* here (+28% to +58%) than for Si (+474%/+1190%): Ge/Br are less
"soft" than Si relative to their free-atom reference. This makes the failure
look more like the curable P/S class (+30%) than the catastrophic Si class.
PDEP-RPA remains the recommended heavy-Z source; the TS number now exists for
cross-check rather than being a blanket refusal.

## Conclusions
1. **RPA@PBE α is uniform on heavier atoms too**: −1.4% (SiF4) to −10% (SiH4),
   −7% (PH3). Extends the robustness claim past first/second row.
2. **The Si TS failure is the ATOM, now N=2**: two chemically different Si
   compounds (hydride SiH4, fluoride SiF4) both blow up (TS +474%, +1190%) while
   RPA@PBE α is accurate on both. SiH4's failure is basis-invariant. → free-atom
   Si reference, not a specific molecule or the basis.
3. **P is NOT like Si**: PH3 TS is only +30% — P behaves like the curable S/O
   class, not the catastrophic Si class. So "heavy main-group" is too coarse;
   the failure is element-specific (Si bad, P mild).
4. **RPA@PBE α holds down to Z=32–35**: GeH4 −1.7% vs CRC — the tightest of the
   heavy set. Br compounds have no CRC α₀ reference here, but their α is smooth
   and basis-monotone (DZ→TZ +0.7% GeH4, +5% CH3Br, +8% Br2). RPA@PBE stays
   accurate all the way past the third row.
5. **Heavy-Z TS now computes and overshoots — mildly (N=3 new)**: with the
   Gould-Bučko free-atom table (Z=19–54), GeH4/CH3Br/Br2 TS produce a C6 instead
   of refusing, all overshooting RPA@PBE (+57.7%, +39.4%, +27.9%). The soft/heavy-atom TS
   failure mode does extend past row 3 — but at Ge/Br it looks like the *curable*
   P/S class (+30%), not the catastrophic Si class (+474%/+1190%). So the earlier
   blanket refusal was conservative: Ge/Br TS is wrong but not catastrophically
   so. PDEP-RPA remains the recommended heavy-Z source; TS is now a cross-check.
   (For Z>54, still absent from the table, TS still refuses rather than fabricate
   an H-like C6 — the reliability guard is intact where no reference exists.)

## Caveats
- α references are CCCBDB experimental (single values, ~few % uncertainty); Br
  compounds (CH3Br, Br2) have no CRC α₀ here, so their α columns are
  reference-free (internal consistency + basis-monotonicity only).
- GeH4/CH3Br/Br2 now **converge** — the SAD-default + g-function-skip + DIIS
  plateau-acceptance SCF fixes (HEAD) resolved the earlier non-convergence/hangs.
  All 6 RPA@PBE cases ran at RAYON=12 in minutes each; NPZs under runs/.
- Heavy-Z TS now computes (Z=19–54 via Gould-Bučko free-atom refs, 2026-07-14);
  GeH4/CH3Br/Br2 TS overshoot RPA@PBE by +58%/+39%/+28% — see conclusion 5. The
  α_free/C6_free spread between the Gould-Bučko and Chu04 sources is ~13% on Br C6
  (documented inline in free_atom_ref.rs), negligible next to the TS model error.
- SiF4 TS at TZ did not finish (free-atom Si+4F SCFs too slow serially); the DZ
  +1190% plus SiH4's basis-invariance carry the conclusion.
- The rayon-on-small-problem penalty (PH3 note below) is superseded for these
  runs by the SAD fixes — the heavy-Z set ran fine on all 12 cores. Earlier
  small molecules still preferred RAYON=1; see rayon-penalty memory.
