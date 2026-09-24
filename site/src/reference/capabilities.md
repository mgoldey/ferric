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
| `rhf` | [SCF](../methods/scf.md) | RHF | ✓ | ✓ analytic | FD | `run_rhf`, `run_optimize`, `run_frequencies` | `water-rhf.toml` | Proven | — |
| `uhf` | [SCF](../methods/scf.md) | UHF | ✓ | ✓ analytic | FD | `run_uhf`, `run_frequencies(reference="uhf")` | `h_uhf.toml` | Proven | — |
| `rohf` | [SCF](../methods/scf.md) | ROHF | ✓ | ✓ analytic | FD | `run_rohf`, `run_frequencies(reference="rohf")` | — | Proven | — |
| `ksdft` | [SCF/DFT](../methods/scf.md) | RKS (closed shell) | ✓ | ✓ analytic (+ D3(BJ) gradient) | FD (refused with `[dft] dispersion` or `grid_prune`) | `run_dft` / `run_ksdft`, `run_frequencies(xc=...)` | `benzene-dfb3lyp.toml`, `h2-lda-opt.toml` | Proven | — |
| `rimp2` | [MP2](../methods/mp2.md) | RHF | ✓ | ✓ analytic (Z-vector) | — | `run_rimp2` | `water-rimp2.toml` | Proven | — |
| `lmp2` | [MP2](../methods/mp2.md) | RHF (errors on open shell) | ✓ | — | — | `run_lmp2` | `water-lmp2.toml` | not graded | ε = 0 reproduces `rimp2`. The canonical RI-MP2 reference and the error against it are opt-in (`[mp2] lmp2_reference = true`, `compute_reference=True`). |
| `lmp2-direct` | [MP2](../methods/mp2.md) | RHF (errors on open shell) | ✓ | — | — | `run_lmp2_direct` | `alkane8-lmp2-direct.toml` | not graded | Same as `lmp2`. The opt-in reference forms the global 3-index tensor that this path otherwise avoids. |
| `mp3` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_mp3` | `water-mp3.toml` | Proven | — |
| `oo-rimp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_oo_rimp2` | `water-oo-rimp2.toml` | Smoke | Internally self-consistent (stationary point, vanishing gradient). No external absolute-energy reference exists. |
| `att-rimp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_attenuated_rimp2` | `water-attmp2.toml` | Proven | — |
| `mp2-v` | [MP2](../methods/mp2.md) | RHF; UHF when multiplicity > 1 | ✓ | — | — | `run_mp2_v` (closed shell only) | `water-mp2v.toml` | Smoke | No comparison to any published MP2-V number. The defaults are fitted for aug-cc-pVTZ with frozen core. Open shell is doubly unvalidated. |
| `scs-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2` | `water-scs-mp2.toml` | Proven | — |
| `scs-mp2-2terfc` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_scs_mp2_2terfc` | `water-scs-mp2-2terfc.toml` | Proven | Needs the terfc tables (`FERRIC_TERF_TABLE_DIR`). |
| `laplace-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_mp2` | `water-laplace-rimp2.toml` | Proven | — |
| `laplace-sos-mp2` | [MP2](../methods/mp2.md) | RHF | ✓ | — | — | `run_laplace_sos_mp2` | `water-laplace-sos-mp2.toml` | not graded | With `c_os = 1.0` it reproduces the opposite-spin MP2 energy (internal reference). |
| `pdep-rpa` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS when multiplicity > 1 | ✓ | ✓ (RHF reference only, whatever `[rpa] xc` says; `xc` applies only to `task = "energy"`; analytic SCF + FD correlation) | — | `run_pdep_rpa` | `water-pdep-rpa.toml` | Proven | — |
| `rs-mp2-rpa` | [MP2](../methods/mp2.md) / [RPA](../methods/rpa-gw.md) | RHF | ✓ | — | — | `run_rs_mp2_rpa` | `water-rs-mp2-rpa.toml` | Smoke | The ω→0 and ω→∞ limits are Proven. At production ω it is only marginally benchmarked on one small subset. |
| `gw` | [RPA/GW](../methods/rpa-gw.md) | RHF, or RKS via `[rpa] xc`; UHF/UKS when multiplicity > 1 | ✓ (QP energies) | — | — | `run_gw`, `run_u_gw` | `water-g0w0-pbe.toml`, `oh-ugw.toml` | Smoke | About 5 meV against MOLGW on a single H2O/cc-pVDZ case. Treat results as ±0.3 eV. |
| `bse-tda` | [RPA/GW](../methods/rpa-gw.md) | RHF only (refuses multiplicity > 1) | ✓ (excitations) | — | — | `run_bse_tda` | `water-bse-tda.toml` | Smoke | Only excitation ordering and a physicality gate are checked. The gap error is inherited from GW. |
| `tdhf-static-polarizability` | [RPA/GW](../methods/rpa-gw.md) | RKS only (`[rpa] xc` required) | ✓ (static α) | — | — | `run_tdhf_static_polarizability` | `water-tdhf-static-alpha.toml` (does not run as shipped: set `[gw] scissor`; see [examples](./examples.md)) | Smoke | Static α only, and not established (−46% against DOSD for water at a physical scissor). The same kernel gives C6 about 63% low. At `scissor = 0` it can hard-error on a negative α diagonal; set `[gw] scissor` to about 0.3–0.4 Ha. |
| `ccsd` | [CC](../methods/cc.md) | RHF (spin-adapted solver) | ✓ | — | — | `run_ccsd` | `water-ccsd.toml` (**H2**, not water) | Proven | — |
| `linlccd` | [CC](../methods/cc.md) | RHF only (refuses multiplicity > 1) | ✓ | — | — | none (`run_linlccd_amplitude` is the amplitude-threshold variant) | `water-linlccd.toml` | Proven (narrow, exact limits only) | No external reference for the LinLCCD(hh) energy. With the hole–hole ladder off it reduces exactly to RI-MP2, and with exact integrals its driver terms reproduce canonical MP2; size consistency is checked. |
| `wb97x-l-v` | [CC § ωB97X-L-V](../methods/cc.md#linlccd-and-ωb97x-l-v) | Its own RKS (wB97X-L-V) reference | ✓ | — | — | none | `water-wb97xlv.toml` | Smoke | The pieces are checked separately. No reference value exists for the total energy. |
| `b2plyp` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="b2plyp")` | `water-b2plyp.toml` | Spike | Weighted B88+LYP reference. Not compared with any reference code. |
| `dsd-pbep86` | [CC § double hybrids](../methods/cc.md#mp2-based-double-hybrids) | Its own RKS reference | ✓ | — | — | `run_double_hybrid(kind="dsd-pbep86")` | — | Spike | Weighted PBE+P86 reference. Not compared with any reference code. |
| `tda` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (CIS), or RKS via `[tddft] xc` | ✓ (excitations) | — | — | `run_tddft(method="tda")` | `water-tda.toml` | Spike | CIS on HF is exact. DFT references lack the f_xc kernel. |
| `tddft` | [RPA/GW § TDDFT](../methods/rpa-gw.md) | RHF (TDHF), or RKS via `[tddft] xc` | ✓ (excitations) | — | — | `run_tddft(method="casida")` | `water-tddft-pbe.toml` | Spike | Full Casida equations without f_xc for DFT references. |

`task = "optimize"` is accepted only for `rhf`, `ksdft`, `uhf`, `rohf`,
`pdep-rpa` and `rimp2`. `task = "frequencies"` is accepted only for `rhf`,
`ksdft`, `uhf` and `rohf`. Any other combination exits with an error before
the SCF runs.

### Open shells in the CLI

- `uhf` and `rohf` read `[molecule] multiplicity` directly. `[dft] functional`
  does **not** turn them into UKS/ROKS: the CLI sets an XC functional only
  for `ksdft`.
- `pdep-rpa`, `gw` and `mp2-v` re-solve with UHF (with MOM after 5
  iterations) when `multiplicity > 1`. For `pdep-rpa` and `gw`, setting
  `[rpa] xc` makes that reference UKS; `mp2-v` does not read `[rpa] xc` and
  stays UHF.
- `lmp2`, `lmp2-direct`, `linlccd`, `wb97x-l-v`, `b2plyp`, `dsd-pbep86`,
  `tda`, `tddft`, `bse-tda` and `tdhf-static-polarizability` refuse an
  open-shell input with an error.
- Every other correlated kind (`rimp2`, `mp3`, `oo-rimp2`, `att-rimp2`,
  `scs-mp2`, `scs-mp2-2terfc`, `laplace-mp2`, `laplace-sos-mp2`,
  `rs-mp2-rpa`, `ccsd`) is closed-shell only. It is fed by the shared
  closed-shell `solve_rhf`, which fails on an odd electron count. Give these
  kinds `multiplicity = 1`.
- `ksdft` is closed-shell (RKS) only. From the CLI, UKS/ROKS exist only as
  the reference inside `pdep-rpa`/`gw`. From Python, see below.

## Python-only capabilities

These have no `method.kind`. Scope is taken from the binding code and its
docstrings. None of them is in the CLI grade table.

| Capability | Python | Scope (verified in code) |
|---|---|---|
| CCD | `run_ccd` | RHF reference, RI integrals. |
| CCSD(T) | `run_ccsd_t` | Closed shell. Spin-adapted CCSD amplitudes feed a spin-adapted (T). |
| terfc-attenuated MP2 | `run_terfc_rimp2` | Closed shell. The CLI's `att-rimp2` is the erfc form only. |
| Open-shell KS geometry/frequencies | `run_frequencies(reference="uhf"\|"rohf", xc=...)` | Setting `xc` promotes RHF/UHF/ROHF to RKS/UKS/ROKS. FD Hessian. |
| Transition-state search | `run_saddle` | P-RFO. Closed shell only (refuses multiplicity ≠ 1). Raises if the start has no negative mode. Costs `2(6N+1) + (steps+1)` gradients. |
| SCF stability descent | `run_uhf(stability_descent=True)` | Checks the converged UHF solution's internal stability and, at a saddle, follows the downhill mode and re-converges. The CLI's `[scf] check_stability` only reports a saddle. |
| Reaction path | `run_irc` | Both IRC branches from a saddle's imaginary mode. Closed shell only. |
| Geometry optimization (Python) | `run_optimize` | RHF only (no `xc` argument). Accepts point charges and a field. |
| QM/MM energy + forces | `QmmmSystem`, `run_qmmm` | `method` = `"rhf"`/`"uhf"`/`"rks"`/`"uks"`. Link atoms, boundary schemes (`keep`/`delete-host`/`rc`/`rcd`), Gaussian-smeared charges, Thole polarizable sites, an optional MM force field (`MmTopology`). See [QM/MM](../using/qmmm.md). The CLI `[qmmm]` section covers fixed point charges only. |
| QM/MM optimization | `run_optimize_qmmm` | Same four methods. `move_mm` = `"none"`/`"all"`/`("within", r)`/`("residues", [...])`. Moving MM atoms requires `mm_topology`. |
| IEF-PCM solvation | `run_rhf(solvent=...)`, `run_pdep_rpa(solvent=...)` | A dielectric constant or a solvent name. An unknown name is an error. **No CLI section.** (The CLI has `[cosmo]`, a different, conductor-limit model with no Python argument.) |
| Point charges / uniform field | `point_charges=`, `external_field=` on `run_rhf`/`run_uhf`/`run_rohf`/`run_dft`, `run_optimize`, `run_frequencies`, `run_saddle`, `run_irc`, `run_pdep_rpa` | Bohr and atomic units. `run_rhf` also takes `smeared_charges=`. The MP2/CC drivers take no external-potential arguments. The CLI equivalent is `[external_potential]`. |
| D3(BJ) | `run_dft(dispersion="d3bj")`, `d3bj_energy` | Additive, with parameters fitted per functional. |
| Charges | `mulliken_charges`, `lowdin_charges`, `hirshfeld_charges`, `chelpg_charges`, `resp_charges` | Take an `RhfResult` or `DftResult`. Mulliken, Löwdin, CHELPG and RESP are documented closed-shell only. RESP is a single-stage restrained fit, not multi-conformer RESP. |
| Electrostatic potential | `esp_at_atoms`, `esp_at_points` | Evaluated exactly from the density. `esp_at_points` takes (N, 3) points in **Bohr**. |
| Polarizability / moments | `hirshfeld_polarizability`, `orbital_moments`, `density_second_moment` | — |
| Local correlation research paths | `run_drpa`, `run_drpa_scan`, `run_linlccd_amplitude` | Amplitude-threshold methods. Closed shell. |
| ω tuning | `tune_omega` | Range-separation ω for a named functional. |
| Conformer statistics | `boltzmann_weights`, `weighted_stats*`, `ConformerEnsemble` | — |
| Integrals | `compute_eri3`, `compute_eri3_mo`, `compute_metric_2c`, `boys_localize`, `shell_info` | Low-level access. |

**Not exposed to users at all:** constrained DFT and the cDFT electron-transfer
coupling ([Constrained DFT](../methods/cdft.md)) are Rust-library only,
with no CLI section and no Python function. PCM has no CLI section. Polarizable
(Thole) embedding is Python and Rust only.

## Related pages

- [What is validated](./validation.md): the numbers behind "Proven"
- [Input reference](./input.md): every TOML key
- [Examples](./examples.md): every shipped input file
- [Python bindings](../using/python.md)
