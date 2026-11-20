# Local MP2 {#local-mp2}

Two electrons on opposite ends of a long molecule barely correlate. Their
MP2 pair energy falls off like a dispersion energy, as the sixth power of
the distance, while the number of such distant pairs grows with the square
of the molecule's size. Canonical RI-MP2 (chapter @sec:mp2) computes every pair
anyway, because its orbitals spread over the whole molecule and give it no
way to tell a near pair from a far one. Local MP2 localizes the orbitals,
estimates which amplitudes are negligible, and solves the MP2 equations
for the rest.

Locality here truncates what is computed. It does not change the method.
The exact calculation is `method.kind = "rimp2"` with no `[local]` section,
and every local run prints its truncation.

| Section | Topic | Registry id | Status |
|---|---|---|---|
| 13.1 | Amplitude-threshold local MP2 | `local-mp2` (gap) | edit |
| 13.2 | The integral-direct path | none | gap |
| 13.3 | Pair natural orbitals and DLPNO-MP2 | none | gap |
| 13.4 | Amplitude-threshold local OP-MP3 | `local-opmp3` | gap |

Sections @sec:amplitude-lmp2 and @sec:integral-direct are the production local method
(`[local] scheme = "amplitude-threshold"`). It keeps an amplitude when the
integral that drives it exceeds one threshold $\varepsilon$; 13.2 builds the
same problem without ever forming the global three-index tensor. Section
@sec:pno-dlpno is a library path with a different truncation, by pair energy and by
pair natural-orbital occupation. The two families are not compared like for
like: the validation page reports their recoveries side by side and never
ranks them.

Every truncation in this chapter has a trivial limit in which it removes
nothing, and there the method reproduces the exact RI-MP2 energy. The
committed tests assert those limits before any truncated number is quoted.
