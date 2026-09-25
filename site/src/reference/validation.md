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
| `gw` | Smoke | The committed H2O/cc-pVDZ tests compare against MOLGW and PySCF `gw_ac` at 0.2–0.3 eV bars and are `#[ignore]`d; most other assertions are range bands. Treat results as ±0.3 eV. |
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
Proof files named `validation_*.rs` are `#[ignore]`d, so an ordinary
`cargo test` skips them; they run in the weekly `validation` CI job and on
demand (`cargo nextest run --run-ignored only -E 'binary(/^validation_/)'`).

| Capability | System / basis | Reference | Stated agreement | Test tolerance | Proof |
|---|---|---|---:|---:|---|
| UHF and ROHF energies, stability-checked | HO2, NO2, CH2 (triplet), allyl / 6-31G, def2-SVP | PySCF `UHF`/`ROHF` + `stability()` | 4.0e-12 Ha (energy); 3e-7 (⟨S²⟩) | 1e-10 Ha; 1e-6 | [`validation_open_shell_scf.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_open_shell_scf.rs), [`gen_uhf_rohf.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_uhf_rohf.py) |
| SCF stability: lowest orbital-Hessian eigenvalues and verdict (UHF internal, UKS internal, RHF internal, RHF→UHF external) | N2⁺, OH (UHF), water at r(OH) = 0.9572 and 2.0 Å (RHF), NH2 (UKS/PBE) / 6-31G | PySCF dense `gen_g_hop_uhf`/`gen_g_hop_rhf`/`hop_rhf2uhf` Hessians + `stability()` | eigenvalues ≤ 4.5e-10 Ha; energies ≤ 7.1e-12 Ha; stable/unstable verdicts match PySCF; OH (an exact zero mode) is reported MARGINAL by ferric where PySCF reports stable | 1e-8 Ha; 1e-10 Ha | [`validation_scf_stability.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_scf_stability.rs), [`gen_scf_stability.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_scf_stability.py) |
| RHF and UHF with def2 ECPs: energies and analytic gradients | HI, CH3I, RbH, I atom (UHF) / def2-SVP, def2-TZVP; SnH4 / def2-TZVP; HBr (all-electron control) | PySCF `RHF`/`UHF` + analytic gradient, and ORCA RHF (`NoRI`), both fed ferric's basis and ECP | ≤ 2.9e-8 Ha (energy, both codes); ≤ 2.3e-8 Ha/Bohr (gradient); all-electron HBr 1e-11 | 2.5e-7 Ha; 1e-7 Ha/Bohr | [`validation_ecp.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_ecp.rs), [`gen_ecp.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_ecp.py), [`gen_ecp_orca.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_ecp_orca.py) |
| ESP at the nuclei | H2O, CH3OH (RHF), HO2 (UHF) / cc-pVDZ, def2-SVP | PySCF `int1e_rinv` on PySCF's converged density, fed to ferric in ferric's AO order (AO order checked through the overlap matrix) | 4.8e-14 a.u. same density; 4.6e-9 own SCF | 5e-13 a.u.; 5e-8 a.u. | [`validation_density_properties.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_density_properties.rs), [`gen_properties.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_properties.py) |
| Electric field at the nuclei | H2O, CH3OH, HO2 / cc-pVDZ, def2-SVP | PySCF `int1e_iprinv`, same density (checked against a finite difference of the ESP) | 1.5e-13 a.u. same density; 2.7e-9 own SCF | 1e-12 a.u.; 3e-8 a.u. | ″ |
| Becke effective volumes | H2O, CH3OH, HO2 / cc-pVDZ, def2-SVP; free H, C, N, O atoms (UKS-PBE) / aug-cc-pVDZ | numpy on ferric's grid rebuilt from PySCF's radial, Lebedev and AO values, with ferric's Becke partition; free atoms vs PySCF UKS | 7.4e-16 rel. same density; 7.1e-9 (molecules), 4.8e-9 (atoms) own SCF | 1e-13; 5e-8 | ″ |
| Static α (direct RPA with a density-fitted Coulomb kernel, no exchange; not CPHF) | H2O / aug-cc-pVDZ; CH3OH / cc-pVDZ | numpy linear solve on PySCF `int3c2e` with the same aux basis | 4.8e-14 rel. same orbitals; 5.3e-10 rel. own SCF | 5e-13; 5e-9 | [`validation_static_alpha.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-rpa/tests/validation_static_alpha.rs), [`gen_properties.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_properties.py) |
| NPZ export (`[rpa] export_npz`) | water / STO-3G, CLI `pdep-rpa` | `numpy.load` of the CLI's file against the Python bindings | exact (keys, shapes, dtypes, C order, geometry); per-atom and molecular α origin-independent to 3.4e-13 rel. | exact; 1e-11 rel. under translation | [`test_validation_npz_export.py`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-python/tests/test_validation_npz_export.py) |
| KS-DFT energies: open-shell UKS and second-row closed shell | UKS: NH2, CH3, HO2, O2 (triplet) × PBE, B3LYP, ωB97X-V / 6-31G, def2-SVP; RKS: H2S, HCl, SiH4 × PBE, B3LYP / def2-SVP, def2-TZVP | PySCF `UKS`/`RKS` + `stability()`, same grid and density fitting; for UKS ωB97X-V both codes use exact J and range-separated DF-K, whose fitting metrics differ, which sets that functional's larger bar | energy ≤3.1e-12 Ha (PBE, B3LYP), 6e-6 Ha (ωB97X-V); ⟨S²⟩ ≤4.6e-9 (PBE, B3LYP), 1.3e-7 (ωB97X-V) | energy 1e-10 Ha (PBE, B3LYP), 2e-5 Ha (ωB97X-V); ⟨S²⟩ 5e-8, 1e-6 | [`validation_ks_energies.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_ks_energies.rs), [`gen_ks_energies.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_ks_energies.py) |
| Constrained DFT (Becke charge and spin constraints): E(N) − E_unconstrained, λ, and dE/dN = −λ | LiH (Li), HF (F), H2O⁺ (O; charge and spin) / 6-31G, def2-SVP, 3–4 targets each | NWChem 7.2.2 `cdft ... pop becke`, PBE, Becke-partitioned Treutler grid, fed ferric's basis; internal: natural-target limit (λ = 0) and dE/dN = −λ | unconstrained E 1.0e-8 Ha; E(N) − E_unc 2.0e-7 Ha; λ 1.1e-6 (NWChem, grid limit); dE/dN = −λ to 2.3e-12 Ha (Simpson, def2-SVP) | 5e-8 Ha; 1e-6 Ha; 5e-6; 5e-11 Ha | [`validation_cdft.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_cdft.rs), [`gen_cdft.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cdft.py), [NWChem inputs](https://github.com/mgoldey/ferric/tree/main/scripts/validation/nwchem/cdft) |
| IEF-PCM solver and SCF on PySCF's cavity; ferric's own cavity | water, NH3 / STO-3G, cc-pVDZ; ε = 78.4, 4.7 | PySCF `RHF.PCM()` IEF-PCM; its SWIG cavity injected into ferric | charges 1.3e-17, E_pcm 3.5e-18 Ha (solver); total energy 4.8e-12 Ha (SCF); ferric's own cavity 8.5–11.8% weaker | 1e-11 Ha (solver); 1e-10 Ha (SCF); own cavity within [5%, 15%] | [`validation_pcm.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_pcm.rs), [`gen_pcm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_pcm.py) |
| G0W0@HF | H2O / cc-pVDZ | Published MOLGW IP (11.97 eV); PySCF `gw_ac` (IP 12.160 eV) | — | 0.30 eV (IP); 0.20 eV (PySCF LUMO and gap) | [`h2o_g0w0_cohsex.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/h2o_g0w0_cohsex.rs) |
| MP3 (RI integrals) | H2O, NH3 / cc-pVDZ, def2-SVP; all-electron and frozen core | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; two independent numpy MP3 constructions (spin-orbital textbook, closed-shell via PySCF's linear doubles residual), agreeing to 2e-17 | ≤ 7.6e-12 Ha | 1e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| RI-CCD | H2O, NH3 / cc-pVDZ, def2-SVP | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `CCD` on the DF integrals | ≤ 1.9e-11 Ha | 2e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| RI-CCSD, spin-orbital and spin-adapted | H2O, NH3 / cc-pVDZ, def2-SVP; HCN / cc-pVDZ; H2O / aug-cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `CCSD` on the DF integrals | ≤ 4.7e-11 Ha (both solvers) | 2e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| CCSD(T), spin-orbital and spin-adapted (T) | H2O / cc-pVDZ, aug-cc-pVDZ; HCN / cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `ccsd_t` on the DF integrals | (T) ≤ 6.9e-12 Ha; the two (T) codes agree to 8.9e-12 | 1e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| TDA and Casida TDDFT excitation energies | water, formaldehyde, NH3 / 6-31G, aug-cc-pVDZ | PySCF `tddft.TDA`/`TDDFT`, same RI and grid | HF ≤ 2e-6 eV; LDA/PBE ≤ 2e-5 eV; B3LYP ≤ 6.5e-4 eV | 1e-3 eV | [`validation_tddft.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-tddft/tests/validation_tddft.rs), [`gen_tddft_refs.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_tddft_refs.py) |
| RI-MP2 analytic gradient (all electrons, exact-J/K RHF) | distorted H2O, bent HCN / cc-pVDZ (cc-pvdz-ri); distorted NH3 / def2-SVP (def2-svp-rifit) | ORCA `RI-MP2 NoRI NoFrozenCore EnGrad` with the same /C aux; PySCF `DFMP2` 5-point finite differences | ≤ 2.1e-9 Ha/Bohr (PySCF FD); ≤ 5.1e-8 Ha/Bohr (ORCA, its own floor); energies ≤ 1.3e-10 Ha | 2e-8 Ha/Bohr (FD), 2.5e-7 Ha/Bohr (ORCA); 1e-9 Ha | [`validation_rimp2_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_rimp2_gradient.rs), [`gen_rimp2_gradient.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_rimp2_gradient.py) |
| COSMO solvation (RHF, UHF) | H2O, NH3, CH3OH, acetate(−) / STO-3G, cc-pVDZ × ε 4.7, 78.4; HO2 (UHF) / cc-pVDZ | PySCF `pcm.py` COSMO driver with ferric's radii, 110-point cavity and point-charge potential (ferric's model); stock PySCF COSMO for the formulation gap | solvated energy ≤ 1.9e-11 Ha, E_cosmo ≤ 9.7e-11 Ha (ferric's model); 0.05–0.51% from stock PySCF COSMO (point vs Gaussian surface charges) | 1e-10 Ha; 1e-9 Ha; 2% | [`validation_cosmo.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_cosmo.rs), [`gen_cosmo.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cosmo.py) |
| Attenuated RI-MP2 (erfc on 3-center and metric) | H2O, NH3 / aug-cc-pVDZ + aug-cc-pVDZ-RIFIT, ω 0.2, 0.222, 0.42, 1.0 Bohr⁻¹ | numpy RI-MP2 on PySCF `int3c2e`/`int2c2e` under `with_range_coulomb(-ω)` | E_corr ≤ 5.9e-12 Ha at every ω; ω → 0 reproduces Coulomb RI-MP2 to 1.6e-15 | 1e-10 Ha | [`validation_attenuated_mp2.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_attenuated_mp2.rs), [`gen_attenuated_mp2.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_attenuated_mp2.py) |
| QM/MM electrostatic embedding (RHF): energy, embedding shift, QM gradient, MM forces | H2O + 10 charges / cc-pVDZ, aug-cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | PySCF `qmmm.mm_charge` | energy and shift ≤ 4.8e-12 Ha; QM gradient ≤ 1.2e-10, MM forces ≤ 4.4e-12 Ha/Bohr (up to 501 charges) | 5e-11 Ha; 1e-9, 5e-11 Ha/Bohr | [`validation_qmmm.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_qmmm.rs), [`gen_qmmm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_qmmm.py) |
| Gaussian-smeared MM charges (width in Bohr, ζ = 1/width², same as PySCF `radii` with `unit="Bohr"`) | H2O + 10 charges / cc-pVDZ, widths 0.5, 1, 2 Bohr | PySCF `qmmm.mm_charge(radii=)` | energy ≤ 4.1e-12 Ha; QM gradient ≤ 1.2e-10 Ha/Bohr; width → 0 reproduces point charges to 3.0e-12 Ha | 5e-11 Ha; 1e-9 Ha/Bohr | ″ |
| KS QM/MM (exact J/K, matched (75,110) grid; gradient with grid response) | CH3OH + 20 charges / def2-SVP, B3LYP and PBE | PySCF `qmmm.mm_charge` on `dft.RKS`, `grid_response=True` | energy ≤ 4.1e-12 Ha; QM gradient ≤ 2.9e-9, MM forces ≤ 1.8e-10 Ha/Bohr | 1e-10 Ha; 3e-8, 2e-9 Ha/Bohr | ″ |
| External potential in the RI-MP2 analytic gradient | H2O + 10 charges / cc-pVDZ; CH3OH + 20 charges / 6-31G; aux cc-pVDZ-RI | 5-point finite difference of PySCF `DFMP2` on `qmmm.mm_charge` RHF, same aux | energies ≤ 8.4e-12 Ha; RI-MP2 gradient ≤ 4.8e-9 Ha/Bohr vs PySCF FD | 1e-10 Ha; 3e-8 Ha/Bohr | [`validation_qmmm_mp2_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_qmmm_mp2_gradient.rs), [`gen_qmmm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_qmmm.py) |
| RI-MP2 size-extensivity | H2 dimer at large separation | 2 × monomer | 2e-12 Ha | 1e-7 Ha | [`rimp2_size_extensivity.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/rimp2_size_extensivity.rs) |
| RHF/UHF/ROHF/KS gradients, including density-fitted J/K | water, OH, HO2 / cc-pVDZ, 6-31G | finite differences of the energy; PySCF `df.grad` | 1e-7 to 3e-7 Ha/Bohr (FD); ~1e-10 (PySCF) | 1e-6 Ha/Bohr | [`df_jk_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/df_jk_gradient.rs) |
| Harmonic frequencies (FD of analytic gradients; Cartesian Hessian and cm⁻¹) | H2O, NH3 × RHF, PBE, B3LYP / 6-31G, cc-pVDZ; UHF OH, CH3, HO2, UKS-PBE and ROHF CH3 / 6-31G | PySCF same-step FD of analytic gradients (KS with grid response); PySCF analytic `hessian.rhf/uhf/rks/uks`; `thermo.harmonic_analysis` with ferric's masses | same-step FD: 2.4e-7 Ha/Bohr², 7.5e-4 cm⁻¹; HF vs analytic 2.75e-5 Ha/Bohr², 0.10 cm⁻¹ (the 5e-3 Bohr step's truncation); KS vs PySCF's analytic Hessian differs by 5–35 cm⁻¹ because that Hessian has no grid response | 2e-6 Ha/Bohr², 5e-3 cm⁻¹ (same-step); 1e-4 Ha/Bohr², 0.5 cm⁻¹ (HF analytic) | [`validation_frequencies.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_frequencies.rs), [`gen_frequencies.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_frequencies.py) |
| Meta-GGA gradient, closed shell (SCAN, r2SCAN) | H2O, NH3 / 6-31G, def2-SVP; RI-J (default) and exact J | PySCF `RKS` `grid_response=True`, same grid and density fitting; FD of ferric's own energy | ≤ 1.3e-9 Ha/Bohr (def2-SVP 5.0e-10); analytic vs own FD 1.0e-9 | 1e-8 Ha/Bohr | [`validation_mgga_gradients.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_mgga_gradients.rs), [`gen_mgga_gradients.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_mgga_gradients.py) |
| Meta-GGA gradient, open shell (SCAN, r2SCAN) | UKS: HO2, NH2 / 6-31G; ROKS: NH2 / 6-31G | PySCF `UKS` `grid_response=True` + `stability()`; ROKS: central FD of PySCF `ROKS` energy; FD of ferric's own energy | UKS ≤ 6.2e-9 Ha/Bohr; ROKS 2.2e-10 vs the FD; energies ≤ 5.7e-13 Ha | 3e-8 Ha/Bohr (UKS); 1e-7 (ROKS) | [`validation_mgga_gradients.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_mgga_gradients.rs), [`gen_mgga_gradients.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_mgga_gradients.py) |
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
- a method missing a **physical term** it does not mention (for example, TDDFT
  excitations computed without the f<sub>xc</sub> kernel)
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
