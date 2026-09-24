# What is validated

> **Implemented ≠ validated.** Working code is not a checked number.

Read this page before trusting any result. The other pages say what *exists*;
this one says how far each capability's numbers have been checked against an
independent reference, and where they are known to fail.

## Grades

Every CLI `method.kind` except `lmp2`, `lmp2-direct` and `laplace-sos-mp2`
has a grade. The CLI prints the Smoke and Spike grades as a `[warning]` line
at run time; Proven kinds and the three ungraded kinds print nothing. The
per-method table, with Python-only capabilities included, is in
[Capabilities](./capabilities.md).

| Grade | Meaning |
|---|---|
| **Proven** | Total energies (or the stated property) agree at a stated tolerance with an independent reference, pinned by tests. An independent reference is another code (PySCF, MOLGW), a numpy reference, a published value, or an exact limit the method must reduce to. |
| **Proven (narrow)** | As Proven, but only on the stated class of systems, or only against the stated kind of reference (for example exact limits only) |
| **Smoke** | Runs end to end, and its parts or limits are checked, but no independent reference for the headline number, or only one loose one |
| **Spike** | New code with no comparison to a reference yet; for exploration only |
| **not graded** | Dispatched by the CLI but in neither its Proven list nor its warning table, so it prints no warning; treat it as unproven |

**Proven:** `rhf`, `uhf`, `rohf`, `ksdft`, `rimp2`, `mp3`, `att-rimp2`,
`scs-mp2`, `scs-mp2-2terfc`, `laplace-mp2`, `pdep-rpa`, `ccsd`. **Proven
(narrow, exact limits only):** `linlccd`, which has no external reference for
its energy; with the hole–hole ladder off it reduces exactly to RI-MP2, and
with exact integrals its driver terms reproduce canonical MP2. **Proven
(narrow, closed shell):** `tda`, `tddft`, whose lowest five TDA and Casida
roots match PySCF for water, formaldehyde and NH3 at 6-31G and aug-cc-pVDZ
with HF, LDA, PBE and B3LYP (see the anchor below). These print no warning.

**Smoke or Spike**, with the caveat the CLI prints:

| `method.kind` | Grade | What is and is not checked |
|---|---|---|
| `gw` | Smoke | ~5 meV vs MOLGW on one H2O/cc-pVDZ case; most assertions are range bands. Treat results as ±0.3 eV. |
| `bse-tda` | Smoke | Only excitation ordering and a physicality gate; inherits the GW gap error. |
| `tdhf-static-polarizability` | Smoke | Static α at a physical scissor (0.36 Ha) is 5.20 a.u. for water/cc-pVDZ against DOSD 9.64 (−46%); the same kernel gives C6 ~63% low. |
| `rs-mp2-rpa` | Smoke | The ω→0 and ω→∞ limits reduce exactly to MP2 and MP2+dRPA; production ω is unproven on new systems. |
| `mp2-v` | Smoke | VV10 half bit-identical to the ωB97X-V path; no published MP2-V total energy to compare against. Defaults are fitted for aug-cc-pVTZ, no counterpoise, frozen core. |
| `oo-rimp2` | Smoke | Stationarity and vanishing orbital gradient checked; no external absolute reference. |
| `wb97x-l-v` | Smoke | Components and limits checked; no reference for the total energy. |
| `b2plyp`, `dsd-pbep86` | Spike | No comparison to a reference code yet. |

**Not graded** (no warning printed):

| `method.kind` | Grade | What is checked |
|---|---|---|
| `lmp2`, `lmp2-direct` | not graded | ε = 0 reproduces `rimp2`; the measured `lmp2-direct` scaling is under [Known limits](#known-limits-and-negatives). |
| `laplace-sos-mp2` | not graded | With `c_os = 1.0` it reproduces the opposite-spin MP2 energy (internal reference). |

## Anchors

Where numbers are checked, they are checked against external references or
exact limits, not against ferric's own earlier output. "Stated agreement" is
the measured difference recorded in the repository; "test tolerance" is what
the pinning test actually asserts, which is often looser. **Proof** links to
the test file that asserts the row, and, where one exists, the script that
generated its reference data; a row with nothing to link is not graded Proven.

| Capability | System / basis | Reference | Stated agreement | Test tolerance | Proof |
|---|---|---|---:|---:|---|
| UHF and ROHF energies, stability-checked | HO2, NO2, CH2 (triplet), allyl / 6-31G, def2-SVP | PySCF `UHF`/`ROHF` + `stability()` | 4.0e-12 Ha (energy); 3e-7 (⟨S²⟩) | 1e-10 Ha; 1e-6 | [`validation_open_shell_scf.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_open_shell_scf.rs), [`gen_uhf_rohf.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_uhf_rohf.py) |
| Closed-shell (T) | H2O / cc-pVDZ | PySCF `ccsd_t()` | ~1e-6 Ha | 1e-4 Ha | [`ccsd_t_closed_shell.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/src/ccsd_t_closed_shell.rs) (`closed_shell_t_h2o_ccpvdz_matches_pyscf`) |
| G0W0@HF | H2O / cc-pVDZ | MOLGW | ~5 meV | 0.30 eV | [`h2o_g0w0_cohsex.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/h2o_g0w0_cohsex.rs) |
| TDA and Casida TDDFT excitation energies | water, formaldehyde, NH3 / 6-31G, aug-cc-pVDZ | PySCF `tddft.TDA`/`TDDFT`, same RI and grid | HF ≤ 2e-6 eV; LDA/PBE ≤ 2e-5 eV; B3LYP ≤ 6.5e-4 eV | 1e-3 eV | [`validation_tddft.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-tddft/tests/validation_tddft.rs), [`gen_tddft_refs.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_tddft_refs.py) |
| RI-MP2 size-extensivity | H2 dimer at large separation | 2 × monomer | 2e-12 Ha | 1e-7 Ha | [`rimp2_size_extensivity.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/rimp2_size_extensivity.rs) |
| RHF/UHF/ROHF/KS gradients, including density-fitted J/K | water, OH, HO2 / cc-pVDZ, 6-31G | finite differences of the energy; PySCF `df.grad` | 1e-7 to 3e-7 Ha/Bohr (FD); ~1e-10 (PySCF) | 1e-6 Ha/Bohr | [`df_jk_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/df_jk_gradient.rs) |
| COSX exchange, dense-grid limit | water / cc-pVDZ | direct K | 3.3e-7 | — | [`cosx_k_anchors.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/cosx_k_anchors.rs); see [SCF: choosing how exchange is built](../methods/scf.md) |
| COSX SCF energy, (50,110)+fit | water / cc-pVDZ | direct K | 4.9e-6 Ha | — | ″ |
| COSX SCF energy, (50,110)+fit | butane / def2-SVP | direct K | 1.7e-4 Ha | — | ″ |
| COSX SCF energy, (50,110)+fit | butane / def2-TZVP | direct K | 1.2e-4 Ha | — | ″ |
| COSX, open shell | CH3 doublet / cc-pVDZ | direct K | 1.96e-5 Ha (UHF), 1.97e-5 Ha (ROHF) | — | [`k_builder_open_shell.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/k_builder_open_shell.rs) |
| COSX analytic gradient (RHF, RKS, UHF; default overlap fit for RHF/UHF) | water / STO-3G, 6-31G; HO2 / STO-3G | finite differences of the COSX energy | 1.6e-9 to 4.3e-9 Ha/Bohr | 1e-6 / 3e-8 Ha/Bohr | [`cosx_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/cosx_gradient.rs) |

A "—" means the repository states no number for that cell. It is left empty on
purpose rather than filled with an estimate.

Benchmark sweeps (GW100 and others) are kept in the project's working notes
and are not reproduced here. Quoting a benchmark statistic from memory rather
than from the record is the kind of unchecked claim this page exists to
prevent.

## Known limits and negatives

Reported rather than omitted:

- **Analytic Hessians**: not implemented. The mpqc4 libint2 export has no
  second-derivative integrals. Harmonic **frequencies are available** by
  central finite differences of the analytic gradient (`task = "frequencies"`,
  `ferric.run_frequencies`).
- **Local MP2**: `lmp2-direct`'s correlation stage is measured at about
  N<sup>1.24</sup> (erfc) and N<sup>1.4</sup> (Coulomb) on n-alkanes C20–C48
  (6-31G / cc-pVDZ-RI, frozen carbon cores, a calibrated pair gate; fitted to
  three points). That is one family of molecules in one basis: not shown to
  be linear, and not measured on 3-D or diffuse systems. The plain `lmp2`
  path still builds the global 3-index tensor and makes no scaling claim.
  See [The MP2 family](../methods/mp2.md#local-mp2).
- **Laplace SOS-MP2, AO-sparse variant**: the domain truncation is accurate,
  and the radius it needs grows far more slowly than the molecule (chemical
  accuracy at 3 to 5 Bohr on n-alkanes C2 to C12, radius/diameter falling from
  0.52 to 0.17; a 71-atom drug molecule is within 0.05% at 4 Bohr, about 13%
  of its diameter). The algebra is still dense, so
  **no speedup is claimed**. The regression test
  `sos_ao_sparse_truncation_radius_is_transferable_across_sizes` pins the
  STO-3G butane/octane comparison (12 Bohr exact on both; octane worse at
  3 Bohr). The C2-C12 sweep and the drug-molecule figure are measurements,
  not regression tests.
- **TDHF/RPAx C6**: ~63% low regardless of gap. Do not use it for
  dispersion; its static α is not established either (see the grade table).
- **TDDFT / TDA**: closed-shell references only. Functionals without an
  f<sub>xc</sub> kernel (meta-GGA, VV10, range-separated) are refused rather
  than run without it. B3LYP at aug-cc-pVDZ agrees with PySCF to 6.5e-4 eV,
  40× worse than PBE in the same basis; the cause is not yet identified.
- **COSX wins at high angular momentum, not at large system size.**
  Measured on one thread at the default `(50,110)` grid and overlap fit:
  - Against LinK on n-alkanes at def2-SVP it is slower at every size
    measured: 1.59× (C20), 1.09× (C32) and 1.22× (C48), with no trend toward
    parity. At def2-TZVP on C20 it is faster (0.67×); that is the only
    triple-zeta size measured.
  - Against exact direct exchange on butane it is 3.7× slower at def2-TZVP
    (full SCF). At def2-QZVP its K build takes 90 s against about 400 s for
    the direct build's single J+K sweep, at a relative K error of 5.4e-5.
  - Its K build grows as N<sup>1.29</sup>–N<sup>1.32</sup> between C20 and
    C48 at def2-SVP.
  - It applies to the Coulomb operator only: a range-separated functional
    takes its exchange from density-fitted short- and long-range fitters and
    ignores `k_builder = "cosx"`, with a warning.
  - Its analytic gradient is exact for RHF, RKS and UHF with the overlap fit
    off, and for RHF and UHF with the default overlap fit. Fitted COSX with a
    KS functional, UKS, ROHF/ROKS and pruned COSX grids are refused for
    gradient tasks, before the SCF runs.

## Why the distinction is drawn so sharply

A quantum chemistry code can produce a plausible number in many ways that are
wrong:

- a **non-converged SCF** returned as an ordinary result, because convergence is
  a flag rather than an error
- a method missing a **physical term** it does not mention (TDDFT once
  returned KS excitations without its f<sub>xc</sub> kernel)
- a **fallback model** silently substituted for one atom in a molecule, changing
  a partitioning without changing the shape of the output
- a **screening or truncation threshold** that happens to be safe for the test
  system and not for yours

None of these look like failures, and every one of them has occurred in this
codebase. The remedy is to grade each capability separately and say which ones
are checked against ground truth.

## Testing discipline

Some properties are pinned by tests rather than asserted in prose:

- **Bit-identity across thread counts** for reductions and permutations:
  results do not depend on `RAYON_NUM_THREADS`
- **ERI 8-fold permutational symmetry**: verified against the engine, not
  assumed, since the MP2 code relies on it to compute only ~1/8 of quartets
- **Size-extensivity** and **rotational invariance** of total energies
- **Memory guards in both directions**: a starved budget must be refused *and*
  an ample budget must still run, because an over-estimating guard is also a bug

New guards are **mutation-tested**: a deliberate defect is injected and the test
confirmed to fail before the guard is trusted. This has caught guards that
passed while proving nothing.
