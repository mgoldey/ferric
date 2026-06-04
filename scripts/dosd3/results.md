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
| GeH4 | Ge | 32.19 | non-conv | — | — | KS-PBE SCF oscillates (heavy all-electron Ge); dropped |

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

## Caveats
- α references are CCCBDB experimental (single values, ~few % uncertainty).
- GeH4 (Ge) excluded — KS-PBE non-convergent in ferric (heavy all-electron).
- CH3Br, Br2 excluded — SCF hangs even serially (heavy all-electron Br).
- SiF4 TS at TZ did not finish (free-atom Si+4F SCFs too slow serially); the DZ
  +1190% plus SiH4's basis-invariance carry the conclusion.
- These small molecules require RAYON_NUM_THREADS=1 — the molecular SCF hits the
  same rayon-on-small-problem penalty as free-atom solves (PH3 at RAYON=8 hangs,
  RAYON=1 converges in seconds). See rayon-penalty memory.
