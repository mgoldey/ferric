# 13. Local MP2

Reduced-cost forms of the closed-shell RI-MP2 of chapter 10. Each replaces
the canonical-orbital sum by a calculation over a locality-truncated set of
amplitudes. Locality here is a truncation of what is computed, not a change
of the method: the exact calculation is `method.kind = "rimp2"` with no
`[local]` section, and every local run prints its truncation.

| Section | Topic | Registry id | Status |
|---|---|---|---|
| 13.1 | Amplitude-threshold local MP2 | `local-mp2` (gap) | edit |
| 13.2 | The integral-direct path | none | gap |
| 13.3 | Pair natural orbitals and DLPNO-MP2 | none | gap |

Sections 13.1 and 13.2 are the production local method
(`[local] scheme = "amplitude-threshold"`). Section 13.3 is a library path
with a different truncation, by pair energy and pair natural-orbital
occupation. The two are not compared like for like, and the validation page
reports their recoveries side by side and never ranks them.

Every locality claim in this chapter has a trivial limit, in which the
truncation does nothing, and the method reproduces the exact RI-MP2 energy
there. Those limits are asserted in the committed tests before any
truncated number is quoted.
