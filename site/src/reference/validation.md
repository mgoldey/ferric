# Capabilities and validation

> **Implemented ≠ validated.** Working code is not a checked number.

Read this page before trusting any result. It lists every CLI `method.kind`
(which references it accepts, which tasks it supports, the matching Python
function, an example input) and the Python entry points. Each graded CLI
`method.kind` gets a grade (three kinds are ungraded, and Python entry points
are not graded individually), and the page shows how far each grade's numbers have been checked against an
independent reference and where they are known to fail.

How to read it: find the capability in the [matrix](#cli-methodkind-matrix)
or the [Python entry points](#python-entry-points). A matrix Grade cell links
to the [Anchors](#anchors) table when a row there holds the evidence; the
[Known limits](#known-limits-and-negatives) section records the measured
negatives. Every matrix cell comes from the CLI dispatch in
`crates/ferric-cli/src/lib.rs` or the bindings in
`crates/ferric-python/src/lib.rs`. The page is maintained by hand, so check
those two files if a cell looks wrong. For keyword details see the
[input reference](./input.md); for the example files see the
[examples index](./examples.md).

## Grades

The grades are the ones ferric's CLI uses. Every CLI `method.kind` except
`lmp2`, `lmp2-direct` and `laplace-sos-mp2` has a grade. The CLI prints the
Smoke and Spike grades as a `[warning]` line on stderr at run time; Proven
kinds and the three ungraded kinds print nothing. The warning text is the
`EPISTEMIC_WARNINGS` table in `crates/ferric-cli/src/lib.rs`, and the
"Caveat" column of the matrix condenses it.

| Grade | Meaning |
|---|---|
| **Proven** | Total energies (or the stated property) agree at a stated tolerance with an independent reference, pinned by tests. An independent reference is another code (PySCF, MOLGW), a numpy reference, a published value, or an exact limit the method must reduce to. The CLI groups "Proven" and "Proven (narrow)" together and does not say which kinds are narrow. |
| **Proven (narrow)** | As Proven, but only on the stated class of systems, or only against the stated kind of reference (for example exact limits only) |
| **Smoke** | Runs end to end, and some pieces are checked, but there is no independent reference for the headline number, or only one loose one. The checks are range bands, internal consistency or limits rather than a tight external number. Do not quote a Smoke result as a reference value. |
| **Spike** | Built on new infrastructure and not yet compared against any reference code; for exploration only |
| **not graded** | Dispatched by the CLI but in neither its Proven list nor its warning table, so it prints no warning; treat it as unproven |

Symbols: ✓ = supported. — = not supported (the CLI refuses it with an error).
**FD** = finite differences. Frequencies always take finite differences of
the *analytic* gradient (6N gradient evaluations). ferric has no analytic
Hessians.

## CLI `method.kind` matrix

"Reference" is the SCF reference the CLI actually builds for that kind.
Open-shell support is listed only where the dispatch code handles it (see
[Open shells](#open-shells-in-the-cli) below).

| `method.kind` | Family | Reference (CLI) | Energy | `task = "optimize"` | `task = "frequencies"` | Python | Example | Grade | Caveat |
|---|---|---|---|---|---|---|---|---|---|
| `rhf` | [SCF](../methods/scf.md) | RHF; RKS with `[dft] functional` (refuses multiplicity > 1) | ✓ | ✓ analytic | FD | `run_rhf`, `run_optimize`, `run_frequencies` | `water-rhf.toml` | [Proven](#anchors) | — |
| `uhf` | [SCF](../methods/scf.md) | UHF; UKS with `[dft] functional` | ✓ | ✓ analytic | FD | `run_uhf`, `run_frequencies(reference="uhf", xc=...)` | `h_uhf.toml` | [Proven](#anchors) | — |
| `rohf` | [SCF](../methods/scf.md) | ROHF; ROKS with `[dft] functional` | ✓ | ✓ analytic | FD | `run_rohf`, `run_frequencies(reference="rohf", xc=...)` | — | [Proven](#anchors) | — |
| `ksdft` | [SCF/DFT](../methods/scf.md) | RKS; UKS when multiplicity > 1 | ✓ | ✓ analytic (+ D3(BJ) gradient on the closed-shell optimizer; the open-shell optimizer does not apply D3 and refuses `[dft] dispersion`) | FD (refused with `[dft] dispersion` or `grid_prune`) | `run_dft` / `run_ksdft`, `run_frequencies(xc=...)` | `benzene-dfb3lyp.toml`, `h2-lda-opt.toml` | [Proven](#anchors) | — |
| `rimp2` | [MP2](../methods/mp2.md) | RHF; UHF + unrestricted RI-MP2 when multiplicity > 1 (energy only) | ✓ | ✓ analytic (Z-vector; closed shell only) | — | `run_rimp2` (UHF + UMP2 when multiplicity > 1) | `water-rimp2.toml` | [Proven](#anchors) | — |
| `lmp2` | [MP2](../methods/mp2.md) | RHF (refuses multiplicity > 1) | ✓ | — | — | `run_lmp2` | `water-lmp2.toml` | not graded | ε = 0 reproduces `rimp2`. The canonical RI-MP2 reference and the error against it are opt-in (`[mp2] lmp2_reference = true`, `compute_reference=True`). |
| `lmp2-direct` | [MP2](../methods/mp2.md) | RHF (refuses multiplicity > 1) | ✓ | — | — | `run_lmp2_direct` | `alkane8-lmp2-direct.toml` | not graded | Same as `lmp2`. The opt-in reference forms the global 3-index tensor that this path otherwise avoids. The measured scaling is under [Known limits](#known-limits-and-negatives). |
| `mp3` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_mp3` | `water-mp3.toml` | [Proven](#anchors) | — |
| `oo-rimp2` | [MP2](../methods/mp2.md) | RHF; UHF + unrestricted OO-RI-MP2 when multiplicity > 1 (energy only) | ✓ | — | — | `run_oo_rimp2` (closed shell only) | `water-oo-rimp2.toml` | [Proven (narrow)](#anchors) | Energy matches an independent numpy OO-RI-MP2 to 7.5e-13 Ha (closed-shell H2O and NH3, UHF CH3 / cc-pVDZ) and the closed-shell analytic gradient matches a finite difference of its own energy to 8e-9 Ha/Bohr. ORCA 6.1.1 stops 3.7e-8 to 7.0e-8 Ha above the same minimum. |
| `att-rimp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_attenuated_rimp2` | `water-attmp2.toml` | [Proven](#anchors) | — |
| `mp2-v` | [MP2](../methods/mp2.md) | RHF; UHF when multiplicity > 1 (energy only) | ✓ | — | — | `run_mp2_v` (closed shell only) | `water-mp2v.toml` | Smoke | The VV10 half is bit-identical to the ωB97X-V path. No comparison to any published MP2-V number; no published MP2-V total energy exists to compare against. The defaults are fitted for aug-cc-pVTZ with frozen core and no counterpoise. Open shell is doubly unvalidated. |
| `scs-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2` | `water-scs-mp2.toml` | Proven | — |
| `scs-mp2-2terfc` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2_2terfc` | `water-scs-mp2-2terfc.toml` | Proven | Needs the terfc tables (`FERRIC_TERF_TABLE_DIR`). |
| `laplace-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_mp2` | `water-laplace-rimp2.toml` | Proven | — |
| `laplace-sos-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_sos_mp2` | `water-laplace-sos-mp2.toml` | not graded | With `c_os = 1.0` it reproduces the opposite-spin MP2 energy (internal reference). |
| `pdep-rpa` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS when multiplicity > 1 (energy only) | ✓ | ✓ (closed-shell RHF reference only; `[rpa] xc` is refused; analytic SCF + FD correlation) | — | `run_pdep_rpa` (closed shell only; open-shell U-PDEP-RPA is CLI-only) | `water-pdep-rpa.toml` | Proven | — |
| `rs-mp2-rpa` | [MP2](../methods/mp2.md) / [RPA](../methods/rpa-gw.md) | RHF | ✓ | — | — | `run_rs_mp2_rpa` | `water-rs-mp2-rpa.toml` | Smoke | The ω→0 and ω→∞ limits reduce exactly to MP2 and MP2+dRPA and are Proven. At production ω it is only marginally benchmarked on one small subset and is unproven on new systems. |
| `gw` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS, or ROHF/ROKS with `[gw] reference = "rohf"`, when multiplicity > 1 (energy only) | ✓ (QP energies) | — | — | `run_gw`, `run_u_gw` | `water-g0w0-pbe.toml`, `oh-ugw.toml` | [Smoke](#anchors) | Matches PySCF `gw_ac`/`ugw_ac` only at matched settings (G0W0@HF and @PBE ≤1.3e-7 Ha, evGW ≤5.7e-7, ECP ≤2.5e-6, U-G0W0@UHF ≤7e-6 Ha; the ROHF/ROKS-reference path is not compared against any reference, and it uses the ROHF/ROKS orbital energies directly for both spins, with no semicanonicalization) with `[rpa] n_quad = 100` and `trunc_thresh = 0`; the CLI defaults are coarser (the 20-point grid moves the H2O G0W0@PBE HOMO by 13 meV) and truncation is not validated. See the [G0W0 anchors](#anchors). |
| `bse-tda` | [RPA/GW](../methods/rpa-gw.md) | RHF only (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_bse_tda` | `water-bse-tda.toml` | Smoke | The lowest five singlets match an independent numpy BSE-TDA to 1.8e-10 Ha given the same quasiparticle energies (H2O cc-pVDZ and aug-cc-pVDZ, NH3 and CH2O cc-pVDZ; see [anchors](#anchors)). The quasiparticle energies come from the internal G0W0, which matches PySCF only at `n_quad = 100`, `trunc_thresh = 0`; the defaults are coarser. |
| `tdhf-static-polarizability` | [RPA/GW](../methods/rpa-gw.md) | RKS only (`[rpa] xc` required) | ✓ (static α) | — | — | `run_tdhf_static_polarizability` | `water-tdhf-static-alpha.toml` | Smoke | Static α only, and not established: at a physical scissor (0.36 Ha) it is 5.20 a.u. for water/cc-pVDZ against DOSD 9.64 (−46%). The same kernel gives C6 about 63% low. At `scissor = 0` it can hard-error on a negative α diagonal; set `[gw] scissor` to about 0.3–0.4 Ha. |
| `ccsd` | [CC](../methods/cc.md) | RHF (spin-adapted solver) | ✓ | — | — | `run_ccsd` | `water-ccsd.toml` | [Proven](#anchors) | — |
| `ccd` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_ccd` | `water-ccd.toml` | [Proven (narrow)](#anchors) | RI-CCD, spin-orbital solver. |
| `ccsd(t)` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_ccsd_t` | `water-ccsd-t.toml` | [Proven (narrow)](#anchors) | Spin-adapted CCSD + spin-adapted (T). Prints E_CCSD, E_(T) and the total. |
| `linlccd` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1; open-shell LinLCCD(hh) is library-only) | ✓ | — | — | none (`run_linlccd_amplitude` is the amplitude-threshold variant) | `water-linlccd.toml` | [Proven (narrow)](#anchors) | No other code implements LinLCCD(hh); the energy matches an independent numpy solve on PySCF density-fitted integrals to ≤1.2e-12 Ha (H2O and NH3 through the CLI's closed-shell path; UHF OH through the library-only open-shell path). With the hole–hole ladder off it reduces exactly to RI-MP2, and with exact integrals its driver terms reproduce canonical MP2; size consistency is checked. |
| `linlccd-amplitude` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_linlccd_amplitude` | `water-linlccd-amplitude.toml` | Proven (narrow) | Amplitude-threshold LinLCCD; `[mp2] linlccd_variant`, `linlccd_eps`. ε = 0 reproduces the canonical LinLCCD of the same variant. |
| `drpa` | [RPA/GW](../methods/rpa-gw.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_drpa`, `run_drpa_scan` | `water-drpa.toml` | Proven (narrow) | Amplitude-threshold dRPA; `[mp2] drpa_eps`, `drpa_reference`, `drpa_eps_sweep`. ε = 0 reproduces the canonical plasmon dRPA; finite-ε error is ~linear in ε. |
| `wb97x-l-v` | [CC § ωB97X-L-V](../methods/cc.md#linlccd-and-ωb97x-l-v) | Its own RKS (wB97X-L-V) reference (refuses multiplicity > 1; open shell is library-only) | ✓ | — | — | none | `water-wb97xlv.toml` | Smoke | The pieces and limits are checked separately. No reference value exists for the total energy. |
| `b2plyp` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="b2plyp")` | `water-b2plyp.toml` | Spike | Weighted B88+LYP reference. Not compared with any reference code. |
| `dsd-pbep86` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="dsd-pbep86")` | — | Spike | Weighted PBE+P86 reference. Not compared with any reference code. |
| `tda` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (CIS), or RKS via `[tddft] xc` (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_tddft(method="tda")` | `water-tda.toml` | [Proven (narrow, closed shell)](#anchors) | The lowest five roots match PySCF `TDA` for water, formaldehyde and NH3 at 6-31G and aug-cc-pVDZ with HF, LDA, PBE and B3LYP: at most 6.5e-4 eV (B3LYP), test bar 1e-3 eV. A `[tddft] xc` with no f_xc kernel (meta-GGA, VV10, range-separated) is refused before the SCF. |
| `tddft` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (TDHF), or RKS via `[tddft] xc` (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_tddft(method="casida")` | `water-tddft-pbe.toml` | [Proven (narrow, closed shell)](#anchors) | Same comparison of the lowest five roots against PySCF `TDDFT`, same bar. Same functional refusals as `tda`. |

`task = "optimize"` is accepted only for `rhf`, `ksdft`, `uhf`, `rohf`,
`pdep-rpa` and `rimp2`. `task = "frequencies"` is accepted only for `rhf`,
`ksdft`, `uhf` and `rohf`. On an open-shell molecule both tasks are accepted
only for `uhf`, `rohf` and `ksdft` (UHF/UKS, ROHF/ROKS). Both tasks refuse
`[dft] grid_prune` and `[scf] k_builder = "cosx"`. `[dft] dispersion` is
refused for `frequencies`, and for `optimize` on an open-shell reference: the
CLI threads the D3(BJ) gradient only through its closed-shell optimizer, so
the UKS/ROKS optimizer would walk the uncorrected surface. D3(BJ) itself
depends only on the geometry, and an open-shell `task = "energy"` single
point applies it. Any
other combination exits with an error before the SCF runs.

### Open shells in the CLI

- `uhf` and `rohf` read `[molecule] multiplicity` directly. With
  `[dft] functional` they run UKS and ROKS, for energies, optimizations and
  frequencies.
- `ksdft` with `multiplicity > 1` runs UKS (the `uhf` route with the
  functional set). For ROKS use `rohf` with `[dft] functional`.
- `rhf` refuses `multiplicity > 1` and points to `uhf`/`rohf`.
- `rimp2` and `oo-rimp2` run on the same plain UHF that `kind = "uhf"` runs,
  then take unrestricted RI-MP2 (UMP2, as PySCF `mp.MP2(uhf)`) and
  unrestricted OO-RI-MP2. `[mp2] kappa` is refused on an open shell.
  `task = "energy"` only: there is no unrestricted MP2 nuclear gradient.
- `pdep-rpa`, `gw` and `mp2-v` solve UHF with MOM after 5 iterations when
  `multiplicity > 1`, for `task = "energy"` only. For `pdep-rpa` and `gw`,
  setting `[rpa] xc` makes that reference UKS; `gw` with `[gw] reference =
  "rohf"` uses ROHF (ROKS with `[rpa] xc`) instead. The ROHF-reference path
  is not compared against any reference, and it uses the ROHF orbital
  energies directly for both spins (no semicanonicalization). `mp2-v` does not read
  `[rpa] xc` and stays UHF.
- `linlccd` and `wb97x-l-v` refuse an open-shell molecule; their open-shell
  versions are library-only (`ferric_cc::linlccd_u::u_linlccd`,
  `ferric_cc::double_hybrid::u_solve_wb97x_l_v`).
- `ccd`, `ccsd` and `ccsd(t)` refuse an open-shell molecule, and no open-shell
  CCD, CCSD or CCSD(T) exists in the library either: every CC solver reads
  restricted orbitals.
- Every other kind refuses `multiplicity > 1` with an error before any
  integral is computed.

## Python entry points

These are Python functions without a `method.kind` of their own; some rows
also have a CLI route through a TOML section or task, noted in the row.
Scope is taken from the binding code and its docstrings. They are not graded
individually.

| Capability | Python | Scope (verified in code) |
|---|---|---|
| Open-shell KS frequencies | `run_frequencies(reference="uhf"\|"rohf", xc=...)` | Setting `xc` promotes RHF/UHF/ROHF to RKS/UKS/ROKS. FD Hessian. Also CLI: `kind = "uhf"`/`"rohf"` with `[dft] functional` and `task = "frequencies"`. |
| Transition-state search | `run_saddle` | P-RFO. Closed shell only (refuses multiplicity ≠ 1). Raises if the start has no negative mode. Costs `2(6N+1) + (steps+1)` gradients. |
| Reaction path | `run_irc` | Both IRC branches from a saddle's imaginary mode. Closed shell only. |
| Geometry optimization (Python) | `run_optimize` | RHF only (no `xc` argument). Accepts point charges and a field. |
| QM/MM energy + forces | `QmmmSystem`, `run_qmmm` | `method` = `"rhf"`/`"uhf"`/`"rks"`/`"uks"`. Link atoms, boundary schemes (`keep`/`delete-host`/`rc`/`rcd`), Gaussian-smeared charges, Thole polarizable sites, an optional MM force field (`MmTopology`). See [QM/MM](../using/qmmm.md). The CLI `[qmmm]` section covers fixed point charges only. |
| QM/MM optimization | `run_optimize_qmmm` | Same four methods. `move_mm` = `"none"`/`"all"`/`("within", r)`/`("residues", [...])`. Moving MM atoms requires `mm_topology`. |
| Constrained DFT | `run_cdft`, `CdftConstraint` | UHF, or UKS when `functional` names a libxc functional other than `"HF"` (`None` and `"HF"`, any case, give UHF). Constrained UKS/PBE matches NWChem 7.2.2 (see [anchors](#anchors)); UHF-cDFT has only internal checks, because NWChem cannot run Hartree–Fock cDFT. Fragment `target` is a Becke electron population (`"charge"`: Nα + Nβ; `"spin"`: Nα − Nβ), not a net charge. Raises if the λ loop does not converge. Not graded; see [Constrained DFT](../methods/cdft.md). |
| cDFT electron-transfer coupling | `cdft_coupling` | Wu–Van Voorhis H_ab between two `run_cdft` states, each with one converged `"charge"` constraint, on the same geometry, basis, occupations and Hamiltonian. Not graded. |
| Point charges / uniform field | `point_charges=`, `external_field=` on `run_rhf`/`run_uhf`/`run_rohf`/`run_dft`, `run_optimize`, `run_frequencies`, `run_saddle`, `run_irc`, `run_pdep_rpa` | Bohr and atomic units. `run_rhf` also takes `smeared_charges=`. The MP2/CC drivers take no external-potential arguments. The CLI equivalent is `[external_potential]`, where a `width` makes a charge smeared. |
| D3(BJ) | `run_dft(dispersion="d3bj")`, `d3bj_energy` | Additive, with parameters fitted per functional. |
| Charges | `mulliken_charges`, `lowdin_charges`, `hirshfeld_charges`, `chelpg_charges`, `resp_charges` | Take an `RhfResult` or `DftResult`. Mulliken, Löwdin, CHELPG and RESP are documented closed-shell only. RESP is a single-stage restrained fit, not multi-conformer RESP. |
| Electrostatic potential | `esp_at_atoms`, `esp_at_points` | Evaluated exactly from the density. `esp_at_points` takes (N, 3) points in **Bohr**. |
| Polarizability / moments | `hirshfeld_polarizability`, `orbital_moments`, `density_second_moment` | — |
| ω tuning | `tune_omega` | Range-separation ω for a named functional. |
| Conformer statistics | `boltzmann_weights`, `weighted_stats*`, `ConformerEnsemble` | — |
| Integrals | `compute_eri3`, `compute_eri3_mo`, `compute_metric_2c`, `boys_localize`, `shell_info` | Low-level access. |

Polarizable (Thole) embedding is Python and Rust only. IEF-PCM, UHF
stability descent and terfc-attenuated MP2 are reachable from both: see
`[pcm]`, `[scf] stability_descent` and `[mp2] att_operator` in the
[Input reference](./input.md).

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
| IEF-PCM solver and SCF on PySCF's cavity; ferric's own cavity | water, NH3 / STO-3G, cc-pVDZ; ε = 78.4, 4.7 | PySCF `RHF.PCM()` IEF-PCM; its SWIG cavity injected into ferric | charges 1.3e-17, E_pcm 3.5e-18 Ha (solver); total energy 4.8e-12 Ha (SCF); ferric's own cavity (modified-Bondi radii, 110-point spheres) 0.07–0.61% from PySCF | 1e-11 Ha (solver); 1e-10 Ha (SCF); own cavity ±2% | [`validation_pcm.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_pcm.rs), [`gen_pcm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_pcm.py) |
| Geometry optimization, RHF and RKS | H2O, NH3, CH2O from distorted starts / RHF and B3LYP 6-31G, PBE cc-pVDZ | PySCF analytic gradient (KS with grid response) driven to max abs gradient ≤ 1e-6 Ha/Bohr by scipy BFGS; exact J/K | distances ≤ 2.6e-6 Bohr; angles ≤ 1.1e-4°; optimized energies ≤ 9.3e-13 Ha | 2e-5 Bohr; 1e-3°; 1e-10 Ha | [`validation_geometry_optimization.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_geometry_optimization.rs), [`gen_geometry_optimization.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_geometry_optimization.py) |
| Geometry optimization, UHF, ROHF and UKS | HO2, CH3, NH2 from distorted starts / UHF, ROHF, UKS-PBE 6-31G | PySCF analytic gradient driven to max abs gradient ≤ 1e-6 Ha/Bohr by scipy BFGS; `stability()` at start and end | distances ≤ 2.6e-6 Bohr; angles ≤ 1.1e-4°; optimized energies ≤ 9.3e-13 Ha; ⟨S²⟩ ≤ 4.9e-9 (UHF, UKS; ROHF is a pure spin state and is not checked) | 2e-5 Bohr; 1e-3°; 1e-10 Ha; 5e-8 (⟨S²⟩, UHF and UKS) | [`validation_geometry_optimization.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_geometry_optimization.rs), [`gen_geometry_optimization.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_geometry_optimization.py) |
| G0W0@HF | H2O / cc-pVDZ | Published MOLGW IP (11.97 eV); PySCF `gw_ac` (IP 12.160 eV) | — | 0.30 eV (IP); 0.20 eV (PySCF LUMO and gap) | [`h2o_g0w0_cohsex.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/h2o_g0w0_cohsex.rs) |
| MP3 (RI integrals) | H2O, NH3 / cc-pVDZ, def2-SVP; all-electron and frozen core | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; two independent numpy MP3 constructions (spin-orbital textbook, closed-shell via PySCF's linear doubles residual), agreeing to 2e-17 | ≤ 7.6e-12 Ha | 1e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| RI-CCD | H2O, NH3 / cc-pVDZ, def2-SVP | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `CCD` on the DF integrals | ≤ 1.9e-11 Ha | 2e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| RI-CCSD, spin-orbital and spin-adapted | H2O, NH3 / cc-pVDZ, def2-SVP; HCN / cc-pVDZ; H2O / aug-cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `CCSD` on the DF integrals | ≤ 4.7e-11 Ha (both solvers) | 2e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| CCSD(T), spin-orbital and spin-adapted (T) | H2O / cc-pVDZ, aug-cc-pVDZ; HCN / cc-pVDZ; C2H6 / cc-pVDZ (spin-adapted only) | exact-integral PySCF RHF, then the same aux basis and DF factorization as ferric; PySCF `ccsd_t` on the DF integrals | (T) ≤ 6.9e-12 Ha; the two (T) codes agree to 8.9e-12 | 1e-10 Ha | [`validation_cc.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_cc.rs), [`gen_cc.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cc.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/cc) |
| G0W0 quasiparticle energies, HOMO−2 to LUMO+2, @HF and @PBE (and @HF with one frozen core orbital) | H2O, NH3, N2 / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit); @PBE: H2O / both bases, N2 / cc-pVDZ | PySCF `gw_ac` fed ferric's basis and aux, with the same 100-point frequency grid, Padé nodes, full quasiparticle equation and exact-integral SCF; @PBE references apply ferric's XC density floor (ρ ≤ 1e-10) and evaluate the Padé as the standard Thiele fraction, because PySCF's `pade_thiele_ndarray` applies its last coefficient twice (up to 7.1e-3 Ha at @PBE) | @HF ≤ 1.3e-7 Ha, @PBE ≤ 6.2e-8 Ha; Σc(ef + iω) at the Padé nodes ≤ 8.6e-11 Ha; Σx ≤ 4.9e-9 Ha | 1e-6 Ha (Σc(iω) 1e-9) | [`validation_gw.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/validation_gw.rs), [`gen_gw.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_gw.py) |
| G0W0@HF with ECPs | I2, Xe, Ag2 / aug-cc-pVDZ-PP (def2-tzvp-rifit) | PySCF `gw_ac`, same basis, inline ECP and aux | ≤ 2.5e-6 Ha (ε_mf 2.9e-6 from the ECP SCF offset) | 2e-5 Ha | ″ |
| U-G0W0@UHF | OH, CH3, NH2 / cc-pVDZ, aug-cc-pVDZ; O2 and CH2 triplets / aug-cc-pVDZ | PySCF `ugw_ac` on a stability-checked UHF, with ferric's per-spin Fermi level | ≤ 8.7e-7 Ha; OH/cc-pVDZ 7.0e-6 Ha (its π hole makes the UHF marginally stable) | 3e-5 Ha | ″ |
| COHSEX, evGW₀, evGW (@HF) | COHSEX: H2O, N2 / cc-pVDZ; evGW₀, evGW: H2O / cc-pVDZ | numpy COHSEX on PySCF's density-fitted integrals; evGW₀ and evGW by iterating PySCF's `get_sigma` with ferric's update rule | COHSEX 9.4e-10 Ha; evGW₀/evGW 5.7e-7 Ha | 1e-8 / 5e-6 Ha | ″ |
| U-COHSEX@UHF (static, HF reference, spin-summed W) | OH, CH3, NH2 / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit); O2 and CH2 triplets / aug-cc-pVDZ; H2O / cc-pVDZ singlet UHF against the closed-shell COHSEX | numpy U-COHSEX on PySCF's density-fitted integrals with ferric's aux, W from ε = I + Π_α + Π_β, on the stability-checked UHF of the U-G0W0 row; the singlet UHF numpy reproduces the closed-shell COHSEX reference to 2.1e-9 Ha | ε_qp ≤ 3.3e-9 Ha (open shell); the singlet UHF through `run_u_gw` equals closed-shell COHSEX to 8.1e-10 | 3e-8 Ha (anchor 1e-8) | [`validation_u_cohsex.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/validation_u_cohsex.rs), [`gen_u_cohsex.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_u_cohsex.py), [reference data](https://github.com/mgoldey/ferric/tree/main/testdata/reference/validation/u_cohsex) |
| BSE-TDA singlet excitation energies and oscillator strengths (G0W0@HF, static W, frozen core 0) | H2O / cc-pVDZ (cc-pvdz-ri), aug-cc-pVDZ (aug-cc-pvdz-rifit); NH3, CH2O / cc-pVDZ; lowest five singlets | numpy BSE-TDA on PySCF density-fitted integrals with ferric's aux, W from the static RPA dielectric on HF energies; the kernel is diagonalized on ferric's own quasiparticle energies, because G0W0 energies of core and high virtual orbitals move by up to 0.29 Ha under a 1e-10 relative change of Σc(iω); CIS limit (W replaced by the bare Coulomb interaction, HF energies) against the same numpy build and PySCF `tdscf.TDA` | Ω ≤ 1.8e-10 Ha and oscillator strengths ≤ 2.6e-9 against the kernel on ferric's QP energies; ≤ 2.2e-7 Ha with each side's own QP energies; CIS anchor ≤ 1.6e-9 Ha vs numpy DF-CIS | 2e-9 Ha (Ω), 3e-8 (f), 2e-6 Ha (own QP) | [`validation_bse.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-gw/tests/validation_bse.rs), [`gen_bse.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_bse.py) |
| TDA and Casida TDDFT excitation energies | water, formaldehyde, NH3 / 6-31G, aug-cc-pVDZ | PySCF `tddft.TDA`/`TDDFT`, same RI and grid | HF ≤ 2e-6 eV; LDA/PBE ≤ 2e-5 eV; B3LYP ≤ 6.5e-4 eV | 1e-3 eV | [`validation_tddft.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-tddft/tests/validation_tddft.rs), [`gen_tddft_refs.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_tddft_refs.py) |
| RI-MP2 analytic gradient (all electrons, exact-J/K RHF) | distorted H2O, bent HCN / cc-pVDZ (cc-pvdz-ri); distorted NH3 / def2-SVP (def2-svp-rifit) | ORCA `RI-MP2 NoRI NoFrozenCore EnGrad` with the same /C aux; PySCF `DFMP2` 5-point finite differences | ≤ 2.1e-9 Ha/Bohr (PySCF FD); ≤ 5.1e-8 Ha/Bohr (ORCA, its own floor); energies ≤ 1.3e-10 Ha | 2e-8 Ha/Bohr (FD), 2.5e-7 Ha/Bohr (ORCA); 1e-9 Ha | [`validation_rimp2_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_rimp2_gradient.rs), [`gen_rimp2_gradient.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_rimp2_gradient.py) |
| COSMO solvation (RHF, UHF) | H2O, NH3, CH3OH, acetate(−) / STO-3G, cc-pVDZ × ε 4.7, 78.4; HO2 (UHF) / cc-pVDZ | PySCF `pcm.py` COSMO driver with ferric's radii, 110-point cavity and point-charge potential (ferric's model); stock PySCF COSMO for the formulation gap | solvated energy ≤ 1.9e-11 Ha, E_cosmo ≤ 9.7e-11 Ha (ferric's model); 0.05–0.51% from stock PySCF COSMO (point vs Gaussian surface charges) | 1e-10 Ha; 1e-9 Ha; 2% | [`validation_cosmo.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_cosmo.rs), [`gen_cosmo.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_cosmo.py) |
| Attenuated RI-MP2 (erfc on 3-center and metric) | H2O, NH3 / aug-cc-pVDZ + aug-cc-pVDZ-RIFIT, ω 0.2, 0.222, 0.42, 1.0 Bohr⁻¹ | numpy RI-MP2 on PySCF `int3c2e`/`int2c2e` under `with_range_coulomb(-ω)` | E_corr ≤ 5.9e-12 Ha at every ω; ω → 0 reproduces Coulomb RI-MP2 to 1.6e-15 | 1e-10 Ha | [`validation_attenuated_mp2.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_attenuated_mp2.rs), [`gen_attenuated_mp2.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_attenuated_mp2.py) |
| QM/MM electrostatic embedding (RHF): energy, embedding shift, QM gradient, MM forces | H2O + 10 charges / cc-pVDZ, aug-cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | PySCF `qmmm.mm_charge` | energy and shift ≤ 4.8e-12 Ha; QM gradient ≤ 1.2e-10, MM forces ≤ 4.4e-12 Ha/Bohr (up to 501 charges) | 5e-11 Ha; 1e-9, 5e-11 Ha/Bohr | [`validation_qmmm.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_qmmm.rs), [`gen_qmmm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_qmmm.py) |
| Gaussian-smeared MM charges (width in Bohr, ζ = 1/width², same as PySCF `radii` with `unit="Bohr"`) | H2O + 10 charges / cc-pVDZ, widths 0.5, 1, 2 Bohr | PySCF `qmmm.mm_charge(radii=)` | energy ≤ 4.1e-12 Ha; QM gradient ≤ 1.2e-10 Ha/Bohr; width → 0 reproduces point charges to 3.0e-12 Ha | 5e-11 Ha; 1e-9 Ha/Bohr | ″ |
| KS QM/MM (exact J/K, matched (75,110) grid; gradient with grid response) | CH3OH + 20 charges / def2-SVP, B3LYP and PBE | PySCF `qmmm.mm_charge` on `dft.RKS`, `grid_response=True` | energy ≤ 4.1e-12 Ha; QM gradient ≤ 2.9e-9, MM forces ≤ 1.8e-10 Ha/Bohr | 1e-10 Ha; 3e-8, 2e-9 Ha/Bohr | ″ |
| Thole polarizable embedding (induced point dipoles, exponential Thole damping a = 2.1304): energy, E_pol, induced dipoles, QM gradient, MM rows | H2O + 4 TIP3P waters (α O 0.837, H 0.496 Å³; with and without intramolecular exclusions) / cc-pVDZ, RHF; NH2 + 3 waters / cc-pVDZ, UHF; anchors: one site 30 Å away (classical limit −½α\|E0\|²), α → 0 (PySCF `qmmm.mm_charge`) | numpy induction model on PySCF point-dipole field integrals and `qmmm.mm_charge`, self-consistent in the SCF; gradients by 5-point finite difference | total energy ≤ 3.7e-12 Ha, E_pol ≤ 1.5e-9, induced dipoles ≤ 1.5e-9 a.u., QM and MM gradients ≤ 3.0e-9 Ha/Bohr; far-site E_pol within 7.7e-8 (relative) of −½α\|E0\|² | 5e-11 Ha (total), 1e-8 (E_pol, dipoles), 3e-8 Ha/Bohr | [`validation_thole.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-scf/tests/validation_thole.rs), [`gen_thole.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_thole.py) |
| MM force field (ferric-mm): bond, angle, torsion, Coulomb and LJ energies and gradients, exclusion and 1-4 pair sets | ALA-ALA, ACE-PHE-NME, TRP-PRO-ASP (charge −1), ACE-PHE-NME + 3 TIP3P waters, at the relaxed geometry and two random displacements; one extra case with every torsion phase shifted by 37° | OpenMM Reference platform (double precision), ff14SB and flexible TIP3P, no cutoff; Coulomb and LJ split by evaluating copies of the nonbonded force with epsilons or charges zeroed | energies ≤ 1.75e-12 relative, gradients ≤ 8.5e-11 relative (after OpenMM's Coulomb constant, which differs from ferric's by 6.6e-11 relative); pair sets identical | 1e-11 (energy), 1e-9 (gradient), relative | [`validation_mm.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mm/tests/validation_mm.rs), [`gen_mm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_mm.py) |
| External potential in the RI-MP2 analytic gradient | H2O + 10 charges / cc-pVDZ; CH3OH + 20 charges / 6-31G; aux cc-pVDZ-RI | 5-point finite difference of PySCF `DFMP2` on `qmmm.mm_charge` RHF, same aux | energies ≤ 8.4e-12 Ha; RI-MP2 gradient ≤ 4.8e-9 Ha/Bohr vs PySCF FD | 1e-10 Ha; 3e-8 Ha/Bohr | [`validation_qmmm_mp2_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_qmmm_mp2_gradient.rs), [`gen_qmmm.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_qmmm.py) |
| κ-regularized RI-MP2 (κ = 0.5, 1.1, 2.0 Eh⁻¹; opposite- and same-spin parts) | H2O, CH4 / cc-pVDZ (cc-pvdz-ri) | numpy (1 − exp(−κΔ))² sum on PySCF density-fitted integrals with the same aux basis; κ → ∞ checked against PySCF `DFMP2` | E_corr (OS, SS, total) ≤ 3.3e-12 Ha at κ = 0.5, 1.1, 2.0; κ → ∞ equals RI-MP2 exactly | 3e-11 Ha | [`validation_kappa_mp2.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_kappa_mp2.rs), [`gen_kappa_mp2.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_kappa_mp2.py) |
| LinLCCD(hh) correlation energy | H2O, NH3 (RHF); OH (UHF, stability-checked) / 6-31G, cc-pVDZ (cc-pvdz-ri) | numpy exact linear solve of the hole–hole ladder equations on PySCF density-fitted integrals; ladder off checked against PySCF `DFMP2`/`DFUMP2` | LinLCCD(hh) ≤ 4.4e-13 Ha closed shell, 1.2e-12 Ha OH UHF; ladder off equals RI-MP2 to 1.1e-16 | 1e-11 Ha | [`validation_linlccd.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-cc/tests/validation_linlccd.rs), [`gen_linlccd.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_linlccd.py) |
| OO-RI-MP2 energy and analytic nuclear gradient (all electrons, exact-J/K Fock, RI correlation) | distorted H2O, distorted NH3 (RHF, energy and gradient); CH3 doublet (UHF, energy) / cc-pVDZ (cc-pvdz-ri) | Energy: an independent numpy OO-RI-MP2 on PySCF integrals (exact J/K, the same aux, minimized to max orbital gradient ≤ 1e-10, anchored to PySCF SCF + DF-MP2 at zero rotation); cross-check: ORCA `OO-RI-MP2 NoRI NoFrozenCore` with the same /C aux, which stops 3.7e-8 to 7.0e-8 Ha above the minimum (reference/doubles split off by up to 1.5e-5 Ha). Gradient: 5-point finite difference of ferric's own OO energy; loose cross-checks against the finite difference of ORCA's OO energy and ORCA's analytic `EnGrad` (which misses that finite difference by up to 8e-6 Ha/Bohr) | total 7.5e-13 Ha, reference/doubles split 6.4e-10 Ha vs numpy; gradient 8.0e-9 Ha/Bohr vs own FD, 2.0e-7 vs the FD of ORCA's energy | 1e-11 Ha (total), 5e-9 Ha (split), 2e-7 Ha (vs ORCA); 1e-7 Ha/Bohr (own FD), 1e-6 Ha/Bohr (ORCA FD) | [`validation_oo_rimp2.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-mp2/tests/validation_oo_rimp2.rs), [`gen_oo_rimp2.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_oo_rimp2.py), [ORCA inputs](https://github.com/mgoldey/ferric/tree/main/scripts/validation/orca/oo_rimp2) |
| Open-shell PDEP-RPA correlation energy (UHF reference, full rank, spin-summed dielectric) | OH, CH3, NH2 (doublets), O2 (triplet) / cc-pVDZ (cc-pvdz-ri); OH / aug-cc-pVDZ (aug-cc-pvdz-rifit); H2O / cc-pVDZ through both the restricted and the unrestricted path | PySCF `gw.urpa.URPA` (and `gw.rpa.RPA` for H2O) on the same stability-checked exact-integral UHF, same aux basis and the same 40-point Gauss–Legendre frequency grid; cross-checked by numpy on PySCF's density-fitted integrals; truncation checked against a numpy emulation | ≤ 9.2e-11 Ha (OH/aug-cc-pVDZ); truncated 6.4e-11; U path vs R path at the closed-shell limit 3.2e-12 | 1e-9 Ha (U vs R 3e-11) | [`validation_urpa.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-rpa/tests/validation_urpa.rs), [`gen_urpa.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_urpa.py) |
| Attenuated PDEP-RPA correlation energy (erf and erfc on the 3-center integrals and the metric; closed shell, full rank) | H2O, NH3 / cc-pVDZ (cc-pvdz-ri); H2O / aug-cc-pVDZ (aug-cc-pvdz-rifit); ω 0.2, 0.222, 0.42, 1.0 Bohr⁻¹ | numpy dRPA on PySCF `int3c2e`/`int2c2e` under `with_range_coulomb(±ω)`, the same metric factorization as ferric (eigh with a 1e-10 cutoff for erf, Cholesky for erfc), the same RHF and the same 40-point Gauss–Legendre grid; the Coulomb kernel matches PySCF `gw.rpa.RPA` to 1.6e-14 Ha; erfc(ω → 0) and erf(ω → ∞) reproduce Coulomb RPA | ≤ 5.8e-11 Ha (erf), 3.5e-11 (erfc), 4.5e-11 (Coulomb); the ω → 0 erfc and ω → ∞ erf limits equal Coulomb RPA to 4.9e-11 | 1e-9 Ha (erf), 5e-10 (erfc, Coulomb, limits) | [`validation_attenuated_rpa.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-rpa/tests/validation_attenuated_rpa.rs), [`gen_attenuated_rpa.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_attenuated_rpa.py) |
| RS-MP2-RPA correlation energy, formulations B (`DeltaLr`: E_MP2[Coulomb] + E_dRPA[erf] − 2·E_OS[erf]) and T (`CoupledRings`: E_MP2[Coulomb] + ΔdRPA[Coulomb] − ΔdRPA[erfc]); every reported component | H2O, NH3 / cc-pVDZ (cc-pvdz-ri); H2O / aug-cc-pVDZ (aug-cc-pvdz-rifit); ω 0.222 (= 0.420 Å⁻¹, the default) and 0.42 Bohr⁻¹ | numpy assembly on PySCF `int3c2e`/`int2c2e` under `with_range_coulomb(±ω)`: RI-MP2 spin components and dRPA from the same fitted integrals ferric uses per operator (eigh with a 1e-10 cutoff for erf, Cholesky otherwise), 40-point Gauss–Legendre grid; Coulomb pieces match PySCF `DFMP2` to 3.3e-16 Ha and `gw.rpa.RPA` to 1.7e-14 Ha; the frequency-integrated second-order ring term equals 2·E_OS to 3.2e-15 relative; exact limits B, T(ω → 0) = MP2 and B, T(ω → ∞) = MP2 + ΔdRPA[Coulomb] | every component ≤ 4.7e-11 Ha (MP2 pieces 4.2e-11, dRPA pieces 2.9e-11, B/T e_corr 4.3e-11); ω → 0 and ω → ∞ limits to 4.3e-11 | 5e-10 Ha | [`validation_rs_mp2_rpa.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-rpa/tests/validation_rs_mp2_rpa.rs), [`gen_rs_mp2_rpa.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_rs_mp2_rpa.py) |
| Closed-shell RPA nuclear gradient (`total_rpa_gradient`: central finite difference of E_RHF + E_c^RPA, exact-J/K RHF reference, full rank; all electrons and one frozen core orbital) | distorted H2O, distorted NH3 / cc-pVDZ (cc-pvdz-ri) | finite differences of PySCF exact-integral `RHF` + `gw.rpa.RPA` with the same aux basis and the same 40-point Gauss–Legendre frequency grid (5-point stencil step-converged to 7.4e-10 Ha/Bohr, and the 3-point stencil at ferric's step); 5-point finite difference of ferric's own energy; controls: RHF gradient, E_c-only gradient, all-electron vs frozen core, 6- vs 40-point grid | E_c ≤ 1.5e-13 Ha; gradient ≤ 1.9e-9 Ha/Bohr vs PySCF at the same 3-point step, ≤ 5.7e-8 vs a converged 5-point FD (the 3-point truncation); sum over atoms ≤ 4.8e-9 | 1e-12 Ha (E_c); 2e-8 (same step), 2e-7 (5-point) Ha/Bohr | [`validation_rpa_gradient.rs`](https://github.com/mgoldey/ferric/blob/main/crates/ferric-rpa/tests/validation_rpa_gradient.rs), [`gen_rpa_gradient.py`](https://github.com/mgoldey/ferric/blob/main/scripts/validation/gen_rpa_gradient.py) |
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
  dispersion; its static α is not established either (see the
  [matrix](#cli-methodkind-matrix)).
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

## Related pages

- [Input reference](./input.md): every TOML key
- [Examples](./examples.md): every shipped input file
- [Python bindings](../using/python.md)
