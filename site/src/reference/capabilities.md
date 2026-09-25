# Capabilities

This page lists every CLI `method.kind`: which references it accepts, which
tasks it supports, the matching Python function, an example input, and its
validation grade. Every cell comes from the CLI dispatch in
`crates/ferric-cli/src/lib.rs` or the bindings in
`crates/ferric-python/src/lib.rs`. The page is maintained by hand, so check
those two files if a cell looks wrong. For keyword details see the
[input reference](./input.md); for the example files see the
[examples index](./examples.md).

## How to read the grades

The grades are the ones ferric's CLI uses. It prints a `[warning]` line on
stderr for every `method.kind` that is not Proven. The warning text is the
`EPISTEMIC_WARNINGS` table in `crates/ferric-cli/src/lib.rs`, and the
"Caveat" column below condenses it.

| Grade | Meaning |
|---|---|
| **Proven** | Total energies (or the stated property) agree at a stated tolerance with an independent reference, pinned by tests. An independent reference is another code (PySCF, MOLGW), a numpy reference, a published value, or an exact limit the method must reduce to. "Proven (narrow)" means only on the stated class of systems, or only against the stated kind of reference. The CLI groups "Proven" and "Proven (narrow)" together and does not say which kinds are narrow. See [What is validated](./validation.md) for the anchors. |
| **Smoke** | Runs end to end, and some pieces are checked. The checks are range bands, internal consistency or limits rather than a tight external number. Do not quote a Smoke result as a reference value. |
| **Spike** | Built on new infrastructure and not yet compared against any reference code. |
| **not graded** | The kind is dispatched, but it is neither in the CLI's Proven list nor in its warning table, so it prints no warning. Treat it as unproven. |

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
| `rhf` | [SCF](../methods/scf.md) | RHF; RKS with `[dft] functional` (refuses multiplicity > 1) | ✓ | ✓ analytic | FD | `run_rhf`, `run_optimize`, `run_frequencies` | `water-rhf.toml` | Proven | — |
| `uhf` | [SCF](../methods/scf.md) | UHF; UKS with `[dft] functional` | ✓ | ✓ analytic | FD | `run_uhf`, `run_frequencies(reference="uhf", xc=...)` | `h_uhf.toml` | Proven | — |
| `rohf` | [SCF](../methods/scf.md) | ROHF; ROKS with `[dft] functional` | ✓ | ✓ analytic | FD | `run_rohf`, `run_frequencies(reference="rohf", xc=...)` | — | Proven | — |
| `ksdft` | [SCF/DFT](../methods/scf.md) | RKS; UKS when multiplicity > 1 | ✓ | ✓ analytic (+ D3(BJ) gradient, closed shell only) | FD (refused with `[dft] dispersion` or `grid_prune`) | `run_dft` / `run_ksdft`, `run_frequencies(xc=...)` | `benzene-dfb3lyp.toml`, `h2-lda-opt.toml` | Proven | — |
| `rimp2` | [MP2](../methods/mp2.md) | RHF; UHF + unrestricted RI-MP2 when multiplicity > 1 (energy only) | ✓ | ✓ analytic (Z-vector; closed shell only) | — | `run_rimp2` (UHF + UMP2 when multiplicity > 1) | `water-rimp2.toml` | Proven | — |
| `lmp2` | [MP2](../methods/mp2.md) | RHF (refuses multiplicity > 1) | ✓ | — | — | `run_lmp2` | `water-lmp2.toml` | not graded | ε = 0 reproduces `rimp2`. The canonical RI-MP2 reference and the error against it are opt-in (`[mp2] lmp2_reference = true`, `compute_reference=True`). |
| `lmp2-direct` | [MP2](../methods/mp2.md) | RHF (refuses multiplicity > 1) | ✓ | — | — | `run_lmp2_direct` | `alkane8-lmp2-direct.toml` | not graded | Same as `lmp2`. The opt-in reference forms the global 3-index tensor that this path otherwise avoids. |
| `mp3` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_mp3` | `water-mp3.toml` | Proven | — |
| `oo-rimp2` | [MP2](../methods/mp2.md) | RHF; UHF + unrestricted OO-RI-MP2 when multiplicity > 1 (energy only) | ✓ | — | — | `run_oo_rimp2` (closed shell only) | `water-oo-rimp2.toml` | Smoke | Internally self-consistent (stationary point, vanishing gradient). No external absolute-energy reference exists. |
| `att-rimp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_attenuated_rimp2` | `water-attmp2.toml` | Proven | — |
| `mp2-v` | [MP2](../methods/mp2.md) | RHF; UHF when multiplicity > 1 (energy only) | ✓ | — | — | `run_mp2_v` (closed shell only) | `water-mp2v.toml` | Smoke | No comparison to any published MP2-V number. The defaults are fitted for aug-cc-pVTZ with frozen core. Open shell is doubly unvalidated. |
| `scs-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2` | `water-scs-mp2.toml` | Proven | — |
| `scs-mp2-2terfc` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2_2terfc` | `water-scs-mp2-2terfc.toml` | Proven | Needs the terfc tables (`FERRIC_TERF_TABLE_DIR`). |
| `laplace-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_mp2` | `water-laplace-rimp2.toml` | Proven | — |
| `laplace-sos-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_sos_mp2` | `water-laplace-sos-mp2.toml` | not graded | With `c_os = 1.0` it reproduces the opposite-spin MP2 energy (internal reference). |
| `pdep-rpa` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS when multiplicity > 1 (energy only) | ✓ | ✓ (closed-shell RHF reference only; `[rpa] xc` is refused; analytic SCF + FD correlation) | — | `run_pdep_rpa` | `water-pdep-rpa.toml` | Proven | — |
| `rs-mp2-rpa` | [MP2](../methods/mp2.md) / [RPA](../methods/rpa-gw.md) | RHF | ✓ | — | — | `run_rs_mp2_rpa` | `water-rs-mp2-rpa.toml` | Smoke | The ω→0 and ω→∞ limits are Proven. At production ω it is only marginally benchmarked on one small subset. |
| `gw` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS when multiplicity > 1 (energy only) | ✓ (QP energies) | — | — | `run_gw`, `run_u_gw` | `water-g0w0-pbe.toml`, `oh-ugw.toml` | Smoke | Closed-shell G0W0@HF matches PySCF `gw_ac` to 3.8 meV MAD on 17 GW100 molecules at aug-cc-pVTZ, in a benchmark harness (`benchmarks/harness/examples/gw_xcheck.rs`) that asserts nothing. The committed tests use 0.2–0.3 eV bars. Treat results as ±0.3 eV. |
| `bse-tda` | [RPA/GW](../methods/rpa-gw.md) | RHF only (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_bse_tda` | `water-bse-tda.toml` | Smoke | Only excitation ordering and a physicality gate are checked. The gap error is inherited from GW. |
| `tdhf-static-polarizability` | [RPA/GW](../methods/rpa-gw.md) | RKS only (`[rpa] xc` required) | ✓ (static α) | — | — | `run_tdhf_static_polarizability` | `water-tdhf-static-alpha.toml` | Smoke | Static α only, and not established (−46% against DOSD for water at a physical scissor). The same kernel gives C6 about 63% low. At `scissor = 0` it can hard-error on a negative α diagonal; set `[gw] scissor` to about 0.3–0.4 Ha. |
| `ccsd` | [CC](../methods/cc.md) | RHF (spin-adapted solver) | ✓ | — | — | `run_ccsd` | `water-ccsd.toml` | Proven | — |
| `ccd` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_ccd` | `water-ccd.toml` | Proven (narrow) | RI-CCD, spin-orbital solver. |
| `ccsd(t)` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_ccsd_t` | `water-ccsd-t.toml` | Proven (narrow) | Spin-adapted CCSD + spin-adapted (T). Prints E_CCSD, E_(T) and the total. |
| `linlccd` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1; open-shell LinLCCD(hh) is library-only) | ✓ | — | — | none (`run_linlccd_amplitude` is the amplitude-threshold variant) | `water-linlccd.toml` | Proven (narrow) | No other code implements LinLCCD(hh); the energy matches an independent numpy solve on PySCF density-fitted integrals to ≤1.2e-12 Ha (H2O and NH3 through the CLI's closed-shell path; UHF OH through the library-only open-shell path). With the hole–hole ladder off it reduces exactly to RI-MP2, and with exact integrals its driver terms reproduce canonical MP2; size consistency is checked. |
| `linlccd-amplitude` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_linlccd_amplitude` | `water-linlccd-amplitude.toml` | Proven (narrow) | Amplitude-threshold LinLCCD; `[mp2] linlccd_variant`, `linlccd_eps`. ε = 0 reproduces the canonical LinLCCD of the same variant. |
| `drpa` | [RPA/GW](../methods/rpa-gw.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | `run_drpa`, `run_drpa_scan` | `water-drpa.toml` | Proven (narrow) | Amplitude-threshold dRPA; `[mp2] drpa_eps`, `drpa_reference`, `drpa_eps_sweep`. ε = 0 reproduces the canonical plasmon dRPA; finite-ε error is ~linear in ε. |
| `wb97x-l-v` | [CC § ωB97X-L-V](../methods/cc.md#linlccd-and-ωb97x-l-v) | Its own RKS (wB97X-L-V) reference (refuses multiplicity > 1; open shell is library-only) | ✓ | — | — | none | `water-wb97xlv.toml` | Smoke | The pieces are checked separately. No reference value exists for the total energy. |
| `b2plyp` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="b2plyp")` | `water-b2plyp.toml` | Spike | Weighted B88+LYP reference. Not compared with any reference code. |
| `dsd-pbep86` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="dsd-pbep86")` | — | Spike | Weighted PBE+P86 reference. Not compared with any reference code. |
| `tda` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (CIS), or RKS via `[tddft] xc` (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_tddft(method="tda")` | `water-tda.toml` | Proven (narrow, closed shell) | Matches PySCF `TDA` for water, formaldehyde and NH3 at 6-31G and aug-cc-pVDZ with HF, LDA, PBE and B3LYP: at most 6.5e-4 eV (B3LYP), test bar 1e-3 eV (see [anchors](./validation.md#anchors)). A `[tddft] xc` with no f_xc kernel (meta-GGA, VV10, range-separated) is refused before the SCF. |
| `tddft` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (TDHF), or RKS via `[tddft] xc` (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_tddft(method="casida")` | `water-tddft-pbe.toml` | Proven (narrow, closed shell) | Same comparison against PySCF `TDDFT`, same bar. Same functional refusals as `tda`. |

`task = "optimize"` is accepted only for `rhf`, `ksdft`, `uhf`, `rohf`,
`pdep-rpa` and `rimp2`. `task = "frequencies"` is accepted only for `rhf`,
`ksdft`, `uhf` and `rohf`. On an open-shell molecule both tasks are accepted
only for `uhf`, `rohf` and `ksdft` (UHF/UKS, ROHF/ROKS). Both tasks refuse
`[dft] grid_prune` and `[scf] k_builder = "cosx"`. `[dft] dispersion` is
refused for `frequencies`, and for `optimize` on an open-shell reference. Any
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
  setting `[rpa] xc` makes that reference UKS; `mp2-v` does not read
  `[rpa] xc` and stays UHF.
- `linlccd` and `wb97x-l-v` refuse an open-shell molecule; their open-shell
  versions are library-only (`ferric_cc::linlccd_u::u_linlccd`,
  `ferric_cc::double_hybrid::u_solve_wb97x_l_v`).
- Every other kind refuses `multiplicity > 1` with an error before any
  integral is computed.

## Python-only capabilities

These have no `method.kind`. Scope is taken from the binding code and its
docstrings. None of them is in the CLI grade table.

| Capability | Python | Scope (verified in code) |
|---|---|---|
| Open-shell KS frequencies | `run_frequencies(reference="uhf"\|"rohf", xc=...)` | Setting `xc` promotes RHF/UHF/ROHF to RKS/UKS/ROKS. FD Hessian. |
| Transition-state search | `run_saddle` | P-RFO. Closed shell only (refuses multiplicity ≠ 1). Raises if the start has no negative mode. Costs `2(6N+1) + (steps+1)` gradients. |
| Reaction path | `run_irc` | Both IRC branches from a saddle's imaginary mode. Closed shell only. |
| Geometry optimization (Python) | `run_optimize` | RHF only (no `xc` argument). Accepts point charges and a field. |
| QM/MM energy + forces | `QmmmSystem`, `run_qmmm` | `method` = `"rhf"`/`"uhf"`/`"rks"`/`"uks"`. Link atoms, boundary schemes (`keep`/`delete-host`/`rc`/`rcd`), Gaussian-smeared charges, Thole polarizable sites, an optional MM force field (`MmTopology`). See [QM/MM](../using/qmmm.md). The CLI `[qmmm]` section covers fixed point charges only. |
| QM/MM optimization | `run_optimize_qmmm` | Same four methods. `move_mm` = `"none"`/`"all"`/`("within", r)`/`("residues", [...])`. Moving MM atoms requires `mm_topology`. |
| Constrained DFT | `run_cdft`, `CdftConstraint` | UHF, or UKS when `functional` names a libxc functional other than `"HF"` (`None` and `"HF"`, any case, give UHF) (UKS is smoke-level). Fragment `target` is a Becke electron population (`"charge"`: Nα + Nβ; `"spin"`: Nα − Nβ), not a net charge. Raises if the λ loop does not converge. Not graded; see [Constrained DFT](../methods/cdft.md). |
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

## Related pages

- [What is validated](./validation.md): the numbers behind "Proven"
- [Input reference](./input.md): every TOML key
- [Examples](./examples.md): every shipped input file
- [Python bindings](../using/python.md)
