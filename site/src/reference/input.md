# Input reference (TOML)

Every key the `ferric` CLI accepts, section by section.

> **Unknown keys are a hard error.** The config structs in
> `crates/ferric-cli/src/config.rs` are `#[serde(deny_unknown_fields)]`, so a
> misspelled key or section aborts the run before anything is computed. Two
> places are exceptions: `[external_potential]` (and its `point_charges`
> tables) and each `[[scf.ladder]]` rung. Neither is declared strict, so a
> typo there is **silently ignored**.
>
> Most string values go through strict parsers, where an unknown value is an
> error rather than a default. The exceptions are noted per key.

This page is hand-maintained against `crates/ferric-cli/src/config.rs` at
commit `e7e21b24`. Where a default is applied at the point of use rather than
in `config.rs`, it was read from `crates/ferric-cli/src/lib.rs` at the same
commit. If the code and this page disagree, the code wins. For which
`method.kind` values exist and what each supports, see
[Capabilities](./capabilities.md).

Units follow the code: `[molecule]` geometries are Å (XYZ), point charges and
cutoffs named `*_bohr` are Bohr, keys named `*_angstrom` or documented as Å are
Å, and energies are Hartree.

A minimal file:

```toml
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rimp2"
[mp2]
auxbasis = "cc-pvdz-ri"
```

---

## `[molecule]` (required)

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `xyz` | string | **required** | path | Standard XYZ in Å, relative to the working directory. Not read when `[qmmm]` is present (the PQR supplies the geometry). |
| `charge` | integer | `0` | | With `[qmmm]`, applies to the QM region. |
| `multiplicity` | integer | `1` | ≥ 1 | Read by `uhf`, `rohf`, and the UHF fallback of `pdep-rpa`/`gw`/`mp2-v`. Several closed-shell kinds refuse `> 1`. See [open shells](./capabilities.md#open-shells-in-the-cli). |

## `[basis]` (required)

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `name` | string | — | a bundled name (case-insensitive) | `sto-3g`, `6-31g`, `cc-pvdz`, `cc-pvtz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `aug-cc-pvqz`, `aug-cc-pvdz-pp`, `aug-cc-pvtz-pp`, `def2-svp`, `def2-tzvp`, `def2-qzvp`, `cc-pvdz-f12`. |
| `path` | string | — | path to a Gaussian-94 file | Used only if `name` is absent. One of `name`/`path` is required. |

Auxiliary basis names used elsewhere (`auxbasis`, `df_*_aux`) come from the
same table: `cc-pvdz-ri`, `cc-pvtz-rifit`, `aug-cc-pv{d,t,q}z-rifit`,
`def2-svp-rifit`, `def2-tzvp-rifit`, `def2-tzvpp-rifit`, `def2-qzvp-rifit`,
`def2-qzvpp-rifit`, `def2-universal-jkfit`, `cc-pvdz-f12-optri`.
`cc-pvdz-rifit` is an alias of `cc-pvdz-ri`; both names load the same set.

## `[method]` (required)

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `kind` | string | **required** | `rhf` `uhf` `rohf` `ksdft` `rimp2` `lmp2` `lmp2-direct` `mp3` `oo-rimp2` `att-rimp2` `mp2-v` `scs-mp2` `scs-mp2-2terfc` `laplace-mp2` `laplace-sos-mp2` `pdep-rpa` `rs-mp2-rpa` `gw` `bse-tda` `tdhf-static-polarizability` `ccsd` `linlccd` `wb97x-l-v` `b2plyp` `dsd-pbep86` `tda` `tddft` | Any other value is an error. Non-Proven kinds print a `[warning]` grade line on stderr. |
| `task` | string | `"energy"` | `energy` `optimize` `frequencies` | `optimize`: `rhf` `ksdft` `uhf` `rohf` `rimp2` `pdep-rpa` only. `frequencies`: `rhf` `ksdft` `uhf` `rohf` only. |

## `[scf]`

Read by every kind, because every kind runs an SCF first.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `max_iter` | integer | `100` | | |
| `energy_conv` | float | `1e-3` | | **Sanity bound, not a target.** Convergence requires `ΔP_rms < density_conv`, `ΔP_max < 10·density_conv` **and** `ΔE < energy_conv`. `ΔE` floors on the RI noise, so tightening this can make a density-fitted run hit `max_iter`. |
| `density_conv` | float | `1e-6` | | The real convergence signal. |
| `diis_size` | integer | `8` | | |
| `diis` | string | `"pulay"` | `pulay` `adiis` `ediis` (also capitalised) | An unknown value aborts (by panic). |
| `diis_switch_thresh` | float | `1e-1` | | Error level at which ADIIS/EDIIS hand over to Pulay. Ignored for `pulay`. |
| `smearing_sigma` | float | none | Hartree | Fermi–Dirac smearing width. Absent means integer occupations. |
| `guess` | string | `"minao"` | `minao` `sad` `hcore` | **Not strictly parsed.** Only `"hcore"` changes behaviour. `"sad"` and `"minao"` both select the MINAO projection guess, and any other string silently does the same. |
| `soscf` | bool | `false` | | Enables the second-order (Newton) step in the SCF tail. |
| `integral_thresh` | float | `1e-12` | | Integral screening threshold. |
| `screening` | string | `"schwarz"` | `schwarz` `csb` `csam` | `csb` is rigorous and never looser than `schwarz`. `csam` is not a bound. It is refused for erfc (short-range) operators. See [SCF: screening](../methods/scf.md). |
| `k_builder` | string | `"direct"` | `direct` `link` `cosx` | Exchange builder. Ignored with a warning when DF-K is active, for functionals with no exact exchange, or for range-separated functionals. See [SCF: choosing how exchange is built](../methods/scf.md). |
| `cosx_grid` | inline table | `{ radial = 50, angular = 110 }` | `angular` ∈ 6/14/26/50/110/302 | Only with `k_builder = "cosx"`; otherwise it is an error. The inner table is strict. |
| `cosx_overlap_fit` | bool | `true` | | Only with `cosx`; otherwise it is an error. |
| `cosx_backend` | string | `"md3c1e"` | `md3c1e` `cosx-a` | Only with `cosx`; otherwise it is an error. `cosx-a` is the slower cross-check kernel. |
| `cosx_screen_thresh` | float | `1e-7` | ≥ 0 | Only with `cosx` and `md3c1e`. `0` disables the screen. |
| `cosx_half_transform` | string | `"sparse"` | `sparse` `dense` | Only with `cosx`. |
| `df_j_aux` | string | none; `def2-universal-jkfit` for `ksdft`, `pdep-rpa`, `rs-mp2-rpa`, `gw`, `bse-tda`, `tdhf-static-polarizability`, `tda`, `tddft` | aux basis name | RI-J. With neither key set, `rhf`/`uhf`/`rohf` use exact 4-index J/K. |
| `df_k_aux` | string | as `df_j_aux` | aux basis name | RI-K. Use a JK-fit set. |
| `level_shift` | float | `0.0` | Hartree | Virtual-block shift. Left at 0 with a meta-GGA functional, the library applies 0.5. |
| `mom_after_iter` | integer | `0` | | Maximum-overlap occupation pinning after this many iterations. `0` = aufbau throughout. |
| `verbose` | bool | `false` | | One line per SCF iteration. The CLI's `--verbose`/`-v` flag ORs into this. |
| `df_guess` | bool | on | | Two-stage SCF: DF first, then exact. It applies only on the closed-shell, non-laddered path (`rimp2` and the other correlated kinds). `rhf`/`ksdft` use the convergence ladder, which does not compose with it, and `lib.rs` prints a warning there whenever it is enabled, including by default. Mutually exclusive with an explicit `df_increments = true`. |
| `df_guess_aux` | string | `def2-universal-jkfit` | aux basis name | It is an error when `df_guess` is off. |
| `df_increments` | bool | `false` | | DF-corrected incremental Fock SCF. Same scope as `df_guess`, and warned and ignored on `rhf`/`ksdft`. |
| `df_increments_aux` | string | `def2-universal-jkfit` | aux basis name | It is an error when `df_increments` is off. |
| `check_stability` | bool | `false` | | Diagnostic only: warns if the solution is a saddle point and never fails the run. It covers RHF/RKS and UHF/UKS. ROHF/ROKS, range-separated and meta-GGA are skipped with a printed reason. |
| `ladder` | array of tables | built-in ladder | see below | `[[scf.ladder]]` rungs. Read only by `rhf` and `ksdft`. |

### `[[scf.ladder]]` rungs

Each rung overrides the flat `[scf]` settings. The rungs are walked in order,
and the ladder stops at the first converged rung. **This table is not strict:
unknown keys are ignored.**

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `guess` | string | `"sad"` | `sad` `hcore` `sad-smallbasis` | `sad-smallbasis` warns and uses `sad`. An unknown value also warns and uses `sad`. |
| `level_shift` | float | inherits | | |
| `max_iter` | integer | inherits | | |
| `df_j_aux`, `df_k_aux` | string | inherits | | |
| `stall_window` | integer | none | | |
| `divergence_tol` | float | none | | |
| `restart` | bool | `false` | | `true` discards the incoming density. |

## `[dft]`

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `functional` | string | `"LDA"` (for `ksdft`) | an XC name (`LDA`, `PBE`, `B3LYP`, `wB97X-V`, `SCAN`, `r2SCAN`, …) or a libxc name | Read by `ksdft`. `wb97x-l-v` ignores it with a warning. RPA/GW use `[rpa] xc` and TDDFT uses `[tddft] xc` instead. |
| `grid_prune` | string | `"none"` | `none` `off` `flat`; `nwchem` `nwchem-like` `nwchem_like` | Prunes the main grid only. Accepted only with `task = "energy"`. |
| `dispersion` | string | absent | `d3bj`, `d3(bj)`, `d3bj(<functional>)` | Only with `kind = "ksdft"`, and not with `task = "frequencies"`. There is no "off" value; omit the key instead. A functional with no published D3(BJ) fit is an error. |
| `lambda` | float | `0.6` | | Only for `wb97x-l-v`. |
| `omega` | float | `0.1` | Bohr⁻¹ | Only for `wb97x-l-v`. Note the unit differs from `[mp2] omega`. |

## `[mp2]`

This section is shared by the whole MP2 family, `ccsd`, `linlccd`, the double
hybrids and `tda`/`tddft`, all of which read `auxbasis` and `frozen_core` from
here.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `auxbasis` | string | `"cc-pvdz-ri"`; `"cc-pvdz-rifit"` for `tda`/`tddft` | aux basis name | The two defaults name the same bundled set (`cc-pvdz-rifit` is an alias of `cc-pvdz-ri`). |
| `frozen_core` | int, string or bool | `0` | integer ≥ 0, `"auto"`, `"none"`, `true` (= auto), `false` (= 0) | `"auto"` gives the standard small core for this molecule after the ECP is applied, and the run prints the resolved count. |
| `omega` | float | `0.420` | Å⁻¹ | `att-rimp2`, `rs-mp2-rpa`. Ignored with a warning when `attenuator = "terf"`. |
| `kappa` | float | none | κ > 0, Hartree⁻¹ | κ-regularized MP2 for `rimp2`. Absent = plain MP2. |
| `c_os` | float | `1.2` (`scs-mp2`), `1.27` (`scs-mp2-2terfc`), `1.3` (`laplace-sos-mp2`) | | |
| `c_ss` | float | `1/3` (`scs-mp2`), `4.05` (`scs-mp2-2terfc`) | | `laplace-sos-mp2` warns and ignores it. |
| `n_quad` | integer | `7` | `3` `5` `7` | `laplace-mp2`, `laplace-sos-mp2`. Any other value is an error. |
| `sos_formulation` | string | `"mo"` | `mo` `ao` `ao-sparse` | `laplace-sos-mp2`. `mo` and `ao` are exact and agree to round-off. `ao-sparse` is approximate and requires `domain_cutoff_bohr`. |
| `domain_cutoff_bohr` | float | none | > 0, Bohr | Required by `ao-sparse`. An error with the other formulations. |
| `formulation` | string | `"delta-lr"` | `delta-lr` `coupled-rings` | `rs-mp2-rpa`. |
| `attenuator` | string | `"erf"` | `erf` `terf` | `rs-mp2-rpa`. `terf` needs `FERRIC_TERF_TABLE_DIR`. |
| `r0` | float | `1.6828` (= 3.18 Bohr) | Å | `rs-mp2-rpa` with `terf` only. |
| `r0_sweep` | array of floats | none | Å, > 0 | `rs-mp2-rpa` with `terf` only. Reuses one SCF for several r0 values. `r0` is then ignored with a warning. |
| `r0_bonded` | float | `0.75` | Å | `scs-mp2-2terfc`. |
| `r0_nonbonded` | float | `1.05` | Å, > `r0_bonded` | `scs-mp2-2terfc`. |
| `lmp2_eps` | float | `1e-4` | | `lmp2`, `lmp2-direct`. `0` reproduces `rimp2`. |
| `direct_aux_radius` | float | `10.0` | Bohr | `lmp2-direct`. |
| `direct_virt_radius` | float | `12.0` | Bohr | `lmp2-direct`. |
| `direct_ao_tail` | float | `1e-3` | | `lmp2-direct`. `0.0` keeps every shell. |
| `direct_schwarz_skip` | float | `1e-5` | | `lmp2-direct`. Must be `0.0` for terfc operators, or the run errors. |
| `direct_batch_merge` | integer | `4` | ≥ 1 | `lmp2-direct`. |
| `direct_gate_cal` | float | none (gate off) | | `lmp2-direct` pair gate. |
| `direct_virt_schwarz_kappa` | float | none (off) | | `lmp2-direct`. |
| `mp2v_r0` | float | `1.00` | Å, > 0 | `mp2-v`. Also sets the VV10 damping r0. It is correlated with `mp2v_b` in the published fit. |
| `mp2v_b` | float | `11.0` | | `mp2-v`. |
| `mp2v_c` | float | `0.0089` | | `mp2-v`. Fixed in the paper. Changing it leaves the published parameterization. |
| `mp2v_attenuator` | string | `"terfc"` | `terfc` `erfc` | `mp2-v`. `terfc` needs `FERRIC_TERF_TABLE_DIR`. `erfc` is an unparameterized control. |
| `mp2v_omega` | float | linked, ω = 1/(r0√2) | Å⁻¹, > 0 | `mp2-v` with `terfc` only. Setting it leaves the fitted parameterization. |
| `mp2v_vv10_damping` | string | `"terfc"` | `terfc` `none` | `mp2-v`. `none` double-counts short-range correlation. |
| `mp2v_nlc_n_radial` | integer | `50` | > 0 | `mp2-v` VV10 grid. |
| `mp2v_nlc_n_angular` | integer | `50` | > 0 | `mp2-v` VV10 grid (unpruned). |

## `[rpa]`

Read by `pdep-rpa`, `gw`, `bse-tda`, `tdhf-static-polarizability`, and, for
`trunc_thresh` only, `rs-mp2-rpa`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `auxbasis` | string | `"cc-pvdz-ri"` | aux basis name | |
| `frozen_core` | int, string or bool | `0` | as `[mp2]` | |
| `xc` | string | none (HF reference) | XC name | Switches the reference to RKS, or to UKS when open shell. Required by `tdhf-static-polarizability`. `bse-tda` ignores it. |
| `n_quad` | integer | `20` (energy runs); `16` (`task = "optimize"`) | | The Python `run_pdep_rpa` uses 40. Set it explicitly for reproducibility. |
| `quadrature` | string | `"gauss-legendre"` | `gauss-legendre` `gauss_legendre` `gl`; `minimax` `mini-max` `mm`; `chebyshev-tan` `chebyshev_tan` `chebyshev` `ct` | |
| `u0` | float | `0.5` | | **Warn-and-ignore** under `minimax`, which derives its own u₀. |
| `trunc_thresh` | float | `1e-4`; `0.0` (full rank) for `rs-mp2-rpa` | | PDEP truncation. |
| `eigensolver_conv_thresh` | float | `1e-6` (energy); `1e-8` (`optimize`) | | Alias: `davidson_conv_thresh`. |
| `chi0_sparsity` | string | `"dense"` | `dense`, `boys`, `boys:<thresh>`, `auto`, `auto:<cutoff>`, `auto:<cutoff>:<thresh>`, each boys/auto form optionally suffixed `@<radius_bohr>` | |
| `run_diagnostics` | bool | `false` | | |
| `export_eigpot_prefix` | string | none | | Writes `<prefix>_eigpot_NNN.cube`. |
| `export_eigpot_count` | integer | `10` | | Capped at the number of eigenpotentials. |
| `cube_spacing` | float | `0.2` | Bohr | |
| `cube_margin` | float | `4.0` | Bohr | |
| `export_npz` | string | none | path | Turns on the NPZ property bundle. The `compute_*` keys below default to `true` only when this is set. |
| `compute_esp` | bool | `true` | | ESP at the nuclei. |
| `compute_esp_surface` | bool | `false` | | ESP on a vdW shell. |
| `esp_surface_vdw_scale` | float | `1.4` | | |
| `esp_surface_n_angular` | integer | `110` | Lebedev order | |
| `compute_polarizability` | bool | `true` | | |
| `compute_alpha_atomic` | bool | `true` | | Always Becke-partitioned. `c6_partition` does not affect it. |
| `compute_electric_field` | bool | `true` | | |
| `compute_density_matrix` | bool | `true` | | |
| `compute_dipole` | bool | `true` | | |
| `compute_hirshfeld_charges` | bool | `true` | | |
| `compute_lowdin_charges` | bool | `true` | | |
| `compute_mulliken_charges` | bool | `true` | | |
| `compute_chelpg_charges` | bool | `true` | | |
| `compute_resp_charges` | bool | `true` | | |
| `compute_c6` | bool | `true` | | |
| `allow_partial_npz` | bool | `false` | | By default a bundle missing a requested property fails the run. |
| `c6_source` | string | `"ts"` | `ts` `pdep` `mbd` | |
| `c6_partition` | string | `hirshfeld` for `pdep`, `becke` for `ts`/`mbd` | `becke` `hirshfeld` | |

## `[gw]`

Read by `gw`. `bse-tda` and `tdhf-static-polarizability` read `frozen_core`,
and `tdhf-static-polarizability` also reads `scissor`. The `[rpa]` section
supplies the screened interaction.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `method` | string | `"g0w0"` | `g0w0` `cohsex` `evgw0` `evgw` (case-insensitive) | |
| `qp_mos` | `[lo, hi]` | HOMO−2 … LUMO+2 | absolute MO indices, half-open | |
| `max_ev_iter` | integer | `20` | | evGW/evGW0 outer loop. |
| `ev_conv_thresh` | float | `1e-4` | Hartree | |
| `pade_npts` | integer | `0` (= `[rpa] n_quad`) | | |
| `qp_newton_damp` | float | `1.0` | | |
| `frozen_core` | int, string or bool | falls back to `[rpa] frozen_core` | as `[mp2]` | Also overrides the PDEP frozen core, so W and Σ agree. |
| `scissor` | float | `0.0` | Hartree | `tdhf-static-polarizability` only. At `0.0` some molecules hit a negative α diagonal and the run errors. The remedy is about 0.3–0.4 Ha. |

## `[tddft]`

Read by `tda` and `tddft`. Also set `[mp2] auxbasis` (see above).

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `n_roots` | integer | `3` | | |
| `xc` | string | none (HF reference: CIS or TDHF) | XC name | DFT references lack the f_xc kernel. |
| `c_hf` | float | the functional's short-range exact-exchange fraction; `1.0` with no `xc` | | |

## `[optimize]`

Read when `task = "optimize"`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `max_steps` | integer | `100` | | |
| `g_max_thresh` | float | `4.5e-4` | Hartree/Bohr | |
| `g_rms_thresh` | float | `3.0e-4` | Hartree/Bohr | |
| `e_conv` | float | `1e-6` | Hartree | |
| `trust_radius` | float | `0.1` | | Initial step size. |
| `coordinates` | string | `"cartesian"` | `cartesian` `cart`; `internal` `internals` `redundant-internal` `redundant_internal` | |

## `[frequencies]`

Read when `task = "frequencies"`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `delta` | float | `5e-4` | Bohr, finite and > 0 | Central-difference step. Check the printed `Hessian asymmetry`: it is zero in exact arithmetic. |

## `[memory]`

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `budget_gb` | float | auto | finite and > 0 | Precedence: this key, then `FERRIC_MEM_BUDGET_GB`, then the legacy `FERRIC_OOC_BUDGET_GB`/`FERRIC_ERI3_BUDGET_GB`, then 0.8 × available RAM, then 2 GiB. A value of 0, a negative value or NaN is an error; omit the key for auto. It bounds the ledgered allocations, not total process memory. |
| `three_index_budget_gb` | float | — | | Deprecated alias. `budget_gb` wins if both are set. |

## `[output]`

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `json` | string or bool | `<input-stem>.ferric.jsonl` beside the input | a path, `true` (the default path), `false` (off) | **On by default.** See [Run logs](../using/run-logs.md). `--json <path>` and `--no-json` override it. |

## `[qmmm]`

When present, the PQR supplies both the geometry and the MM charges. It
cannot be combined with `[external_potential]`; doing so is an error.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `pqr` | string | **required** | path | Geometry in Å and charges in e. The element is read from the atom name. |
| `qm_indices` | integer array | `[]` | zero-based | Use this, or `qm_seeds` + `qm_radius_angstrom`, but not both. |
| `qm_seeds` | integer array | `[]` | zero-based | |
| `qm_radius_angstrom` | float | none | Å, > 0 | Requires `qm_seeds`. |
| `link_bonds` | array of `[qm, mm]` | `[]` | | Required when the cut crosses a covalent bond. |
| `boundary_scheme` | string | `"delete-host"` | `keep` `delete-host` `rc` `rcd` | Setting a non-default value without `link_bonds` is an error. |

See [QM/MM](../using/qmmm.md). Smeared charges, polarizable sites and MM force
fields are Python only.

## `[cosmo]`

Conductor-like implicit solvent, applied to every SCF variant. There is no
`[pcm]` section: IEF-PCM is Python only (`run_rhf(solvent=...)`).

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `epsilon` | float | **required** when the section is present | finite and > 1 | `Default` in code is 78.39, but serde has no default for this key. |
| `radius_scale` | float | `1.17` | > 0 | Multiplies Bondi radii. |
| `lebedev_order` | integer | `110` | 6/14/26/50/110/302 | |
| `s_matrix_kind` | string | `"GaussianSmeared"` | `GaussianSmeared` `PointCharge` | Serde variant names, case-sensitive. |

## `[external_potential]`

**Not strict: unknown keys are silently ignored.**

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `point_charges` | array of `{ q, x, y, z }` | `[]` | q in e; x, y, z in **Bohr** | Written as `[[external_potential.point_charges]]` tables. |
| `field` | `[Ex, Ey, Ez]` | none | atomic units | Uniform electric field. |

With both empty, the run is identical to a vacuum run.
