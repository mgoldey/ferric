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
| GeH4 | Ge (34) | 32.19 | 31.44 | 31.64 | −1.7% | TS **refused** (Z=32 > 18, not parameterised) |
| CH3Br | Br (35) | — | 31.68 | 33.22 | — | TS **refused** (Z=35 > 18, not parameterised) |
| Br2  | Br (35) | — | 37.38 | 40.25 | — | TS **refused** (Z=35 > 18, not parameterised) |

### RPA@PBE molecular C6 for the heavy-Z set (a.u.)

| Molecule | C6 DZ | C6 TZ |
|----------|-------|-------|
| GeH4 | 343.19 | 348.48 |
| CH3Br | 428.83 | 453.28 |
| Br2  | 569.88 | 622.29 |

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
   and basis-monotone (DZ→TZ +0.7% GeH4, +5% CH3Br, +8% Br2). RPA@PBE is the one
   dispersion source that gives *any* number for these — the physics extends past
   the third row, where TS structurally cannot follow.
5. **TS refuses cleanly for Z>18 — the correct behavior**: GeH4/CH3Br/Br2 TS all
   hit the heavy-atom guard (TS table covers Z=1..18) and refuse rather than
   substitute hydrogen's London frequency (which would silently yield an H-like
   C6). This is the reliability-audit fix working as designed: for Z>18 the only
   valid dispersion source is PDEP-RPA (c6_source="pdep"). The "TS blows up on
   heavy atoms" failure mode is now a hard refusal, not a wrong number.

## Caveats
- α references are CCCBDB experimental (single values, ~few % uncertainty); Br
  compounds (CH3Br, Br2) have no CRC α₀ here, so their α columns are
  reference-free (internal consistency + basis-monotonicity only).
- GeH4/CH3Br/Br2 now **converge** — the SAD-default + g-function-skip + DIIS
  plateau-acceptance SCF fixes (HEAD) resolved the earlier non-convergence/hangs.
  All 6 RPA@PBE cases ran at RAYON=12 in minutes each; NPZs under runs/.
- Heavy-Z TS is not a datapoint but a **refusal** (Z>18 not parameterised) — see
  conclusion 5. No TS C6 number exists for GeH4/CH3Br/Br2 by design.
- SiF4 TS at TZ did not finish (free-atom Si+4F SCFs too slow serially); the DZ
  +1190% plus SiH4's basis-invariance carry the conclusion.
- The rayon-on-small-problem penalty (PH3 note below) is superseded for these
  runs by the SAD fixes — the heavy-Z set ran fine on all 12 cores. Earlier
  small molecules still preferred RAYON=1; see rayon-penalty memory.
