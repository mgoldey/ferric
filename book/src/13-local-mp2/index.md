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
| @sec:amplitude-lmp2 | Amplitude-threshold local MP2 | `local-mp2` | ready |
| @sec:integral-direct | The integral-direct path | `local-mp2` | ready |
| @sec:pno-dlpno | Pair natural orbitals and DLPNO-MP2 | none (library only) | ready |
| @sec:local-opmp3 | Amplitude-threshold local OP-MP3 | `local-opmp3` | ready |
| @sec:amplitude-lmp2-combined-truncation | Combining the screen, the embedding and λ-Boys orbitals | `local-mp2` | ready |

Sections @sec:amplitude-lmp2 and @sec:integral-direct are the production local method
(`[local] scheme = "amplitude-threshold"`). It keeps an amplitude when the
integral that drives it exceeds one threshold $\varepsilon$; section
@sec:integral-direct builds the same problem without ever forming the global
three-index tensor. Section @sec:pno-dlpno is a library path with a different
truncation, by pair energy and by pair natural-orbital occupation. The two
families are not compared like for like: the validation page
(`site/src/reference/validation.md`, rows "Local RI-MP2" and "DLPNO-MP2")
reports their recoveries side by side and never ranks them. Section
@sec:local-opmp3 reuses the kept set and the solver of section
@sec:amplitude-lmp2 for the two amplitude orders of OP-MP3 (section
@sec:opmp3), a different method on the same truncation. Section
@sec:amplitude-lmp2-combined-truncation proves what the Fock-propagated
screen, a block embedding and λ-Boys orbitals would each do to the energy
error of a combined scheme that no production code implements.

Every truncation in this chapter has a trivial limit in which it removes
nothing, and there the method reproduces the exact canonical energy of the
same method: RI-MP2 for sections @sec:amplitude-lmp2 to @sec:pno-dlpno,
OP-MP3 for section @sec:local-opmp3. The committed tests assert those limits
before any truncated number is quoted.

Several symbols carry more than one meaning across the chapter:

| Symbol | Meaning |
|---|---|
| $\varepsilon$ | the amplitude threshold. Orbital energies are $\epsilon$ |
| $\kappa$ | the Fock-screen constant (section @sec:amplitude-lmp2-fock-screen) and the candidate-screen constant `virt_schwarz_kappa` (section @sec:integral-direct). The λ-Boys rotation generator is $\eta$ |
| $\mathcal S$ | the swap $t^{ab}_{ij} \leftrightarrow t^{ba}_{ij}$. Plain $S$ is the AO overlap, or a Schur complement where a section says so |
| $\mathcal W$, $G$ | the exchange weight $2 - \mathcal S$; section @sec:local-opmp3 writes it $G$ |
| $P$ | an auxiliary index, or a projector onto kept amplitudes or PNOs where a section defines it |
