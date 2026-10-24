# Input reference (TOML)

Every key the `ferric` CLI accepts, section by section.

> **Unknown keys are a hard error.** The config structs in
> `crates/ferric-cli/src/config.rs` are `#[serde(deny_unknown_fields)]`, so a
> misspelled key or section aborts the run before anything is computed. That
> includes `[external_potential]`, its point charges, and each
> `[[scf.ladder]]` rung.
>
> String values go through strict parsers, where an unknown value is an error
> at load time rather than a default. Most accept any capitalisation; the
> exceptions are noted per key.

This page is hand-maintained against `crates/ferric-cli/src/config.rs` at
commit `4b64e6ce`. Where a default is applied at the point of use rather than
in `config.rs`, it was read from `crates/ferric-cli/src/lib.rs` at the same
commit. If the code and this page disagree, the code wins. For which
`method.kind` values exist and what each supports, see
[Capabilities and validation](./validation.md).

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
| `multiplicity` | integer | `1` | ≥ 1 | Read by `uhf`, `rohf` and `ksdft` (UKS) for every task, and for `task = "energy"` only by `rimp2`/`oo-rimp2` (UHF + unrestricted RI-MP2 / OO-RI-MP2) and the open-shell path of `pdep-rpa`/`gw`/`mp2-v`: UHF, or UKS when `[rpa] xc` is set (`pdep-rpa`, `gw`), or ROHF/ROKS for `gw` with `[gw] reference = "rohf"`; `mp2-v` stays UHF. Every other kind refuses `> 1`. See [open shells](./validation.md#open-shells-in-the-cli). |

## `[basis]` (required)

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `name` | string | — | a bundled name (case-insensitive) | `sto-3g`, `6-31g`, `cc-pvdz`, `cc-pvtz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `aug-cc-pvqz`, `aug-cc-pvdz-pp`, `aug-cc-pvtz-pp`, `def2-svp`, `def2-tzvp`, `def2-qzvp`, `cc-pvdz-f12`. |
| `path` | string | — | path to a Gaussian-94 file | Used only if `name` is absent. One of `name`/`path` is required. |

Auxiliary basis names used elsewhere (`auxbasis`, `df_*_aux`) come from the
same table: `cc-pvdz-ri`, `cc-pvtz-rifit`, `aug-cc-pv{d,t,q}z-rifit`,
`def2-svp-rifit`, `def2-tzvp-rifit`, `def2-tzvpp-rifit`, `def2-qzvp-rifit`,
`def2-qzvpp-rifit`, `def2-universal-jkfit`, `cc-pvtz-jkfit`, `cc-pvqz-jkfit`,
`cc-pvdz-f12-optri`.
`cc-pvdz-rifit` is an alias of `cc-pvdz-ri`; both names load the same set.

## `[method]` (required)

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `kind` | string | **required** | `rhf` `uhf` `rohf` `ksdft` `rimp2` `mp3` `oo-rimp2` `att-rimp2` `mp2-v` `scs-mp2` `scs-mp2-2terfc` `laplace-mp2` `laplace-sos-mp2` `pdep-rpa` `rs-mp2-rpa` `gw` `bse-tda` `tdhf-static-polarizability` `ccsd` `ccd` `ccsd(t)` `linlccd` `drpa` `wb97x-l-v` `b2plyp` `dsd-pbep86` `tda` `tddft` | Any other value is an error. Smoke- and Spike-grade kinds print a `[warning]` grade line on stderr; Proven kinds and the ungraded `laplace-sos-mp2` print none. `rimp2`, `drpa` and `linlccd` name the method and are computed exactly unless [`[local]`](#local) sets a local approximation (see [Capabilities and validation](./validation.md#grades)). |
| `task` | string | `"energy"` | `energy` `optimize` `frequencies` | `optimize`: `rhf` `ksdft` `uhf` `rohf` `rimp2` `pdep-rpa` only. `frequencies`: `rhf` `ksdft` `uhf` `rohf` only. |

## `[scf]`

Read by every kind, because every kind runs an SCF first.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `max_iter` | integer | `200` | | Iteration cap; equals the library (`RhfConfig`) and Python default. |
| `energy_conv` | float | `1e-3` | | **Sanity bound, not a target.** Convergence requires `ΔP_rms < density_conv`, `ΔP_max < 10·density_conv` **and** `ΔE < energy_conv`. `ΔE` has a noise floor that grows with system size, so a tight value is met only by chance and costs iterations; ferric prints a `warning: [scf] energy_conv ...` line when a value you wrote is below 0.3 times the estimated floor (see [sharp bits](../using/sharp-bits.md)); functionals, the LinK / COSX exchange builders, PCM / COSMO and double-hybrid kinds get no warning. |
| `density_conv` | float | `1e-6` | | The real convergence signal. |
| `diis_size` | integer | `8` | | |
| `diis` | string | `"pulay"` | `pulay` `adiis` `ediis` (case-insensitive) | An unknown value is an error when the file is loaded. |
| `diis_switch_thresh` | float | `1e-1` | | Error level at which ADIIS/EDIIS hand over to Pulay. Ignored for `pulay`. If the density stops moving while the error is still above this level, the run is not reported converged: it restarts once on plain Pulay DIIS from a clean history, with a stderr warning (with `inf`, the level is `1e-2`). A switch at or below the commutator floor (about `1e-6`) can trigger the restart on a run that had converged, costing a few iterations. |
| `smearing_sigma` | float | none | Hartree | Fermi–Dirac smearing width. Absent means integer occupations. |
| `guess` | string | `"minao"` | `minao` `sad` `hcore` (case-insensitive) | `"sad"` is an alias of `"minao"` (the MINAO projection guess); the free-atom-SCF SAD guess is not selectable from config. Any other value is an error. |
| `soscf` | bool | `false` | | Enables the second-order (Newton) step in the SCF tail. |
| `aurora` | bool | `false` | | Opt-in AURORA auxiliary-curvature accelerator. Closed-shell RHF only (`kind = "rhf"`, multiplicity 1). Same fixed point as DIIS; a speed-up (fewer J/K builds), not a convergence rescue. |
| `trah_trigger` | float | unset | `> 0` | Opt-in trust-region augmented-Hessian solver, engaged once the DIIS error falls below this value (`rhf` and `uhf`). A convergence rescue: fewer iterations, more wall time. |
| `integral_thresh` | float | `1e-12` | | Integral screening threshold. |
| `eri_precision` | float | `1e-20` | `0` to `1e-8` | libint primitive-screening precision for the SCF J/K integrals. Omitted: `FERRIC_ERI_PRECISION` if set, else `1e-20`. `1e-14` costs up to 6e-9 Ha in E_J for atoms past Ne; `0` disables primitive screening (1.8–5.7× slower per J/K build). |
| `jk_storage` | string | `"auto"` | `auto` `memory` `disk` `direct` | Where the raw RI-J/K three-index tensor lives. `auto` keeps it in memory when `n_aux × n_bf² × 8` bytes fit `[memory] budget_gb` and the shared memory pool, keeps the packed symmetric half (`n_aux × n_bf(n_bf+1)/2 × 8` bytes plus 64 unpacked aux rows of scratch) when only that fits, and otherwise measures this machine (an untimed warm-up block, four timed integral blocks, and a cold write and read of up to 64 MiB in the spill directory) and picks the cheaper of `disk` and `direct` for a 30-pass SCF. A spill directory without room for the packed tensor plus headroom rules `disk` out. `memory` is an error when even the packed half does not fit. `disk` always spills to `$TMPDIR`. `direct` recomputes each block per pass and writes nothing. Omitted: `FERRIC_JK_STORAGE` if set; an invalid value in either place is an error. J from the packed, spilled and recomputed tensors is bit-identical (all three are consumed through the same packed block stream); the unpacked in-core J differs from them in the last bits (summation order, about 1e-12 Ha in the energy). This holds for J only: K and the other consumers of the tensor (the DF-K dressing follows the tensor's own block order, and the packed tier's aux block size differs from the spill file's) can differ between tiers at the level of summation order. The tier chosen by `auto` depends on the budget (default 0.8 × available RAM) and on the free bytes in the shared memory pool, so the same input can pick a different tier on a busier machine or with a different budget; set `memory`, `disk` or `direct` to pin it. Applies to the RI-J tensor, and to the RI-K tensor when both use the same auxiliary basis on one rank; the RI-K dressed tensor follows `[memory] budget_gb`. |
| `screening` | string | `"schwarz"` | `schwarz` `csb` `csam` | `csb` is rigorous and never looser than `schwarz`. `csam` is not a bound. It is refused for erfc (short-range) operators. See [SCF: screening](../methods/scf.md). |
| `k_builder` | string | `"direct"` | `direct` `link` `cosx` | Exchange builder. `cosx` with RI-J active (named, or the Kohn-Sham default) is RIJCOSX: J from RI-J, K from COSX, and the RI-K default is not applied; `cosx` next to an explicitly named `df_k_aux` is an error. `link` is ignored with a warning when DF-J/DF-K is active. Both are ignored with a warning for functionals with no exact exchange and for range-separated functionals. See [SCF: choosing how exchange is built](../methods/scf.md). |
| `cosx_grid` | inline table | `{ radial = 35, angular = 194, prune = "sgx" }` | `angular` ∈ 6/14/26/50/110/194/302/434/590; `prune` ∈ `none` `sgx` `nwchem` | The COSX SCF grid. A table without `prune` is flat; with `prune = "sgx"`, `angular` is the peak order of the pruned rows (50/110/194/302/434/590). Only with `k_builder = "cosx"`; otherwise it is an error. The inner table is strict. |
| `cosx_final_pass` | bool | `true` | | Re-evaluate exchange once on a larger grid at the converged density and report that energy (the SCF-grid energy is printed and logged too). Without `cosx_final_grid` the grid is `{ radial = 50, angular = 302, prune = "sgx" }`. Gradient tasks run without it; ROHF/ROKS skips it with a note (an explicit `true` there is an error). Only with `cosx`. |
| `cosx_final_grid` | inline table | none | as `cosx_grid` | The final-pass grid; setting it turns the pass on (`cosx_final_pass = false` with it is an error). RHF/RKS and UHF/UKS only. Only with `cosx`. |
| `cosx_overlap_fit` | bool | `true` | | Only with `cosx`; otherwise it is an error. |
| `cosx_grid_schedule` | bool | `false` | | Run the first SCF iterations on a coarse pruned `sgx` (25,110) exchange grid and switch to `cosx_grid` once the largest density change falls below 1e-3. Convergence is accepted only on the production grid, DIIS restarts at the switch, and the final-grid pass is unchanged. RHF/RKS and UHF/UKS; ROHF/ROKS refuse it. Only with `cosx`; setting it otherwise is an error. |
| `cosx_backend` | string | `"md3c1e"` | `md3c1e` `cosx-a` | Only with `cosx`; otherwise it is an error. `cosx-a` is the slower cross-check kernel. |
| `cosx_screen_thresh` | float | `1e-7` | ≥ 0 | Only with `cosx` and `md3c1e`. `0` disables the screen. |
| `cosx_half_transform` | string | `"sparse"` | `sparse` `dense` | Only with `cosx`. |
| `cosx_fp64_multiplier` | float | `1e5` with `[gpu] precision = "mixed"` and `cosx-kern` in `mixed_kernels`, else `0` | ≥ 0 | Precision-router threshold: `tau = cosx_fp64_multiplier × cosx_screen_thresh`. A kept (shell pair, sub-batch) unit whose Hölder K bound is below `tau` is counted as f32-eligible; the exchange matrix is unchanged (every unit is still computed in f64), so the key only changes the `route_*` counters. Setting it requires `k_builder = "cosx"`, `[gpu] precision = "mixed"` and `cosx-kern` in `mixed_kernels`, and `cosx-kern` is not in the shipped kernel set, so in this build setting the key is always an error; a value above `0` also needs `md3c1e` and `cosx_screen_thresh > 0`. `1e5` is the multiplier recommended by Laqua, Kussmann and Ochsenfeld, J. Chem. Phys. 154, 214116 (2021). |
| `df_j_aux` | string | none; `def2-universal-jkfit` for `ksdft`, `pdep-rpa`, `rs-mp2-rpa`, `gw`, `bse-tda`, `tdhf-static-polarizability`, `tda`, `tddft` | aux basis name, or `""` | RI-J. With neither key set, `rhf`/`uhf`/`rohf` use exact 4-index J/K. `df_j_aux = ""` selects exact J for the kinds that default to RI-J (the `SCF J/K` log line then reads `RI-JK via` with a blank name). Unlike Python's `run_dft`, the CLI does not accept `"exact"`, `"none"` or `"off"`: any non-empty value is looked up as a basis name, and an unknown one fails the SCF. |
| `df_k_aux` | string | as `df_j_aux` | aux basis name, or `""` | RI-K. Use a JK-fit set. `""` selects exact K. |
| `level_shift` | float | `0.0` | Hartree | Virtual-block shift. Left at 0 with a meta-GGA functional, the library applies 0.5. |
| `mom_after_iter` | integer | `0` | | Maximum-overlap occupation pinning after this many iterations. `0` = aufbau throughout. |
| `verbose` | bool | `false` | | One line per SCF iteration. The CLI's `--verbose`/`-v` flag ORs into this. |
| `df_guess` | bool | on | | Two-stage SCF: DF first, then exact. It applies only on the closed-shell, non-laddered path (`rimp2` and the other correlated kinds). `rhf`/`ksdft` use the convergence ladder, which does not compose with it, and `lib.rs` prints a warning there whenever it is enabled, including by default. Mutually exclusive with an explicit `df_increments = true`. |
| `df_guess_aux` | string | `def2-universal-jkfit` | aux basis name | It is an error when `df_guess` is off. |
| `df_increments` | bool | `false` | | DF-corrected incremental Fock SCF. Same scope as `df_guess`, and warned and ignored on `rhf`/`ksdft`. |
| `df_increments_aux` | string | `def2-universal-jkfit` | aux basis name | It is an error when `df_increments` is off. |
| `check_stability` | bool | `false` | | Diagnostic only: warns if the solution is a saddle point and never fails the run. On an RHF/HF run it reports TWO verdicts — internal (the singlet channel, is this RHF solution an RHF minimum?) and external RHF→UHF (the triplet channel, does breaking spin symmetry lower the energy?) — because a stretched geometry is routinely internally stable and externally a saddle: water / 6-31G at r(OH) = 2.0 Å gives +1.97e-2 and −3.07e-1. An external instability's remedy is to run `kind = "uhf"`, not to re-converge RHF. RKS gets the internal verdict only (no triplet XC kernel, printed as a skip); UHF/UKS get the internal verdict, which already spans the independent α/β rotations. ROHF/ROKS, range-separated and meta-GGA are skipped entirely with a printed reason. |
| `stability_descent` | bool | `false` | | State selection: at a saddle of the orbital Hessian (UHF, or the RHF singlet channel), follow the downhill eigenvector and re-converge, keeping the lowest state. Turns `check_stability` on as well. Same as Python `run_uhf(stability_descent=True)`. Only `kind = "rhf"`, `"uhf"` or `"ksdft"` with `task = "energy"`; any other kind or task is an error. A KS functional with no stability verdict (range-separated, meta-GGA) skips the descent with a printed reason. Costs one Davidson per converged solve plus one SCF per descent. See `examples/o2-uhf-stability-descent.toml`. |
| `symmetry` | bool | `false` | | Point-group-blocked closed-shell RHF: detect the Abelian subgroup, symmetrize and reorient the geometry, and diagonalize the Fock matrix per irrep. Only `kind = "rhf"` (no `[dft] functional`), `task = "energy"`, multiplicity 1, no `[cell]`/`[qmmm]`; anything else is an error (the result lives in a rotated frame, so gradients, optimizers and post-HF would be wrong; UHF/ROHF and Kohn-Sham blocking are not implemented). The energy is that of the SYMMETRIZED geometry; the printout and the run log give the point group, the per-irrep electron counts and the symmetrization shift (at most 1e-4 Bohr). No vector quantity is produced. The blocked SCF is one plain solve: `[[scf.ladder]]`, `df_guess`, `df_increments`, `soscf`, `aurora`, `trah_trigger`, `check_stability`/`stability_descent`, `smearing_sigma`, `mom_after_iter` and `k_builder = "cosx"` are refused, and the default convergence-ladder escalation is not applied. Python: `run_rhf(symmetry=True)` (returns `SymmetricRhfResult`). |
| `irrep_nelec` | table | unset | even counts | Electrons per irrep, `irrep_nelec = { A1 = 4, B2 = 2 }`: pins the occupation instead of aufbau across irreps (unlisted irreps follow aufbau). Requires `symmetry = true`. Python: `run_rhf(irrep_nelec={...})`. |
| `ladder` | array of tables | built-in ladder | see below | `[[scf.ladder]]` rungs. Read only by `rhf` and `ksdft`. |

### `[[scf.ladder]]` rungs

Each rung overrides the flat `[scf]` settings. The rungs are walked in order,
and the ladder stops at the first converged rung. Unknown keys are an error,
as elsewhere.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `guess` | string | `"minao"` | `minao` `sad` `hcore` | As `[scf] guess`: `"sad"` is an alias of `"minao"`, and any other value (including `sad-smallbasis`) is an error. |
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
| `grid_prune` | string | `"none"` | `none` `off` `flat`; `nwchem` `nwchem-like` `nwchem_like` | Prunes the main grid only. Optimize and frequency runs differentiate the pruned energy. |
| `grid_radial` | integer | `75` | > 0 | Radial points per atom on the main grid. Only on a run with a Kohn–Sham grid, and not with `kind = "gw"`. Optimize and frequency runs differentiate this grid. Same as Python `grid_radial=`. |
| `grid_angular` | integer | `110` | `6` `14` `26` `50` `110` `194` `302` `434` `590` | Lebedev order on the main grid. Same scope as `grid_radial`. An unsupported order is an error. Same as Python `grid_angular=`. |
| `dispersion` | string | absent | `d3bj`, `d3(bj)`, `d3bj(<functional>)`, `mbd`, `mbd(<functional>)` (case-insensitive) | Only on a Kohn–Sham SCF (`kind = "ksdft"`, or `rhf`/`uhf`/`rohf` with `functional`). `task = "optimize"` and `task = "frequencies"` run on RKS, UKS and ROKS references; `frequencies` uses finite differences of the KS + dispersion gradient, so `[frequencies] hessian = "analytic"` is an error. There is no "off" value; omit the key instead. A functional with no published D3(BJ) fit or MBD@rsSCS β (PBE, PBE0, HSE06) is an error. |
| `lambda` | float | `0.6` | | Only for `wb97x-l-v`. |
| `omega` | float | `0.1` | Bohr⁻¹ | Only for `wb97x-l-v`. Note the unit differs from `[mp2] omega`. |

## `[mp2]`

This section is shared by the whole MP2 family, `ccsd`, `ccd`, `ccsd(t)`,
`linlccd`, `drpa`, the double hybrids and `tda`/`tddft`, all of which read
`auxbasis` and `frozen_core` from here. `drpa`, `linlccd` and a local `rimp2`
read nothing else from it (plus `linlccd_variant` for `linlccd`); any other
`[mp2]` key on them is an error.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `auxbasis` | string | `"cc-pvdz-ri"`; `"cc-pvdz-rifit"` for `tda`/`tddft` | aux basis name | The two defaults name the same bundled set (`cc-pvdz-rifit` is an alias of `cc-pvdz-ri`). |
| `frozen_core` | int, string or bool | `0` | integer ≥ 0, `"auto"`, `"none"`, `true` (= auto), `false` (= 0) | `"auto"` gives the standard small core for this molecule after the ECP is applied, and the run prints the resolved count. |
| `omega` | float | `0.420` | Å⁻¹ | `att-rimp2` (erfc), `rs-mp2-rpa`. Ignored with a warning when `attenuator = "terf"`. An error on `att-rimp2` with `att_operator = "terfc"`. |
| `kappa` | float | none | κ > 0, Hartree⁻¹ | κ-regularized MP2 for the exact `rimp2`; an error with `[local]`. Absent = plain MP2. |
| `c_os` | float | `1.2` (`scs-mp2`), `1.27` (`scs-mp2-2terfc`), `1.3` (`laplace-sos-mp2`) | | |
| `c_ss` | float | `1/3` (`scs-mp2`), `4.05` (`scs-mp2-2terfc`) | | `laplace-sos-mp2` warns and ignores it. |
| `n_quad` | integer | `7` | `3` `5` `7` | `laplace-mp2`, `laplace-sos-mp2`. Any other value is an error. |
| `sos_formulation` | string | `"mo"` | `mo` `ao` `ao-sparse` | `laplace-sos-mp2`. `mo` and `ao` are exact and agree to round-off. `ao-sparse` is approximate and requires `domain_cutoff_bohr`. |
| `domain_cutoff_bohr` | float | none | > 0, Bohr | Required by `ao-sparse`. An error with the other formulations. |
| `formulation` | string | `"delta-lr"` | `delta-lr` `coupled-rings` | `rs-mp2-rpa`. |
| `attenuator` | string | `"erf"` | `erf` `terf` | `rs-mp2-rpa`. `terf` needs `FERRIC_TERF_TABLE_DIR`. An error on `att-rimp2` (use `att_operator`). |
| `r0` | float | `1.6828` (= 3.18 Bohr) | Å | `rs-mp2-rpa` with `terf` only. An error on `att-rimp2` (use `att_r0`). |
| `terf_omega` | float | linked, ω = 1/(r0√2) | Å⁻¹, > 0 | `rs-mp2-rpa` with `attenuator = "terf"` only (an error elsewhere). Sets the terf/terfc sharpness independently of `r0`. Same as Python `run_rs_mp2_rpa(terf_omega=)`. |
| `att_operator` | string | `"erfc"` | `erfc` `terfc` (case-insensitive) | `att-rimp2` only (an error on any other kind). The short-range operator on the MP2 correlation; the SCF stays Coulomb. `terfc` is the Python `run_terfc_rimp2` and needs `FERRIC_TERF_TABLE_DIR`. |
| `att_r0` | float | `1.05` | Å, > 0 | `att-rimp2` with `att_operator = "terfc"` only; an error with `erfc`. |
| `att_omega` | float | linked, ω = 1/(r0√2) | Å⁻¹, > 0 | `att-rimp2` with `att_operator = "terfc"` only; an error with `erfc` (use `omega`) and on any other kind. Sets the terfc seam sharpness independently of `att_r0`, via `Operator::terfc_with_omega`; r0·ω = `att_r0` × `att_omega`. Needs `FERRIC_TERF_TABLE_DIR`. With `[local]` integral-direct it takes the same `schwarz_skip` cut as the other operators. |
| `r0_sweep` | array of floats | none | Å, > 0 | `rs-mp2-rpa` with `terf` only. Reuses one SCF for several r0 values. `r0` is then ignored with a warning. |
| `r0_bonded` | float | `0.75` | Å | `scs-mp2-2terfc`. |
| `r0_nonbonded` | float | `1.05` | Å, > `r0_bonded` | `scs-mp2-2terfc`. |
| `linlccd_variant` | string | `"hh"` | `hh` `drivers-only` `full` | `linlccd` only (an error on any other kind), exact and local alike: it selects the method. `drivers-only` equals RI-MP2; `full` adds the pp ladder (CCD-like VVVV memory). Any other value is an error. |
| `mp2v_r0` | float | `1.00` | Å, > 0 | `mp2-v`. Also sets the VV10 damping r0. It is correlated with `mp2v_b` in the published fit. |
| `mp2v_b` | float | `11.0` | | `mp2-v`. |
| `mp2v_c` | float | `0.0089` | | `mp2-v`. Fixed in the paper. Changing it leaves the published parameterization. |
| `mp2v_attenuator` | string | `"terfc"` | `terfc` `erfc` | `mp2-v`. `terfc` needs `FERRIC_TERF_TABLE_DIR`. `erfc` is an unparameterized control. |
| `mp2v_omega` | float | linked, ω = 1/(r0√2) | Å⁻¹, > 0 | `mp2-v` with `terfc` only. Setting it leaves the fitted parameterization. |
| `mp2v_vv10_damping` | string | `"terfc"` | `terfc` `none` | `mp2-v`. `none` double-counts short-range correlation. |
| `mp2v_nlc_n_radial` | integer | `50` | > 0 | `mp2-v` VV10 grid. |
| `mp2v_nlc_n_angular` | integer | `50` | > 0 | `mp2-v` VV10 grid (unpruned). |
| `oo_max_iter` | integer | `100` | ≥ 1 | `oo-rimp2` only (closed and open shell). Orbital-optimization iterations. |
| `oo_grad_conv` | float | `1e-4` | > 0 | `oo-rimp2` only. Convergence threshold on the orbital-gradient norm. |
| `oo_level_shift` | float | `0.1` | Hartree, ≥ 0 | `oo-rimp2` only. Level shift on the approximate diagonal orbital Hessian. |
| `oo_diis_size` | integer | `6` | ≥ 1 | `oo-rimp2` only. DIIS subspace for the orbital rotations. The four `oo_*` keys match Python `run_oo_rimp2(max_iter=, grad_conv=, level_shift=, diis_size=)`; on any other kind they are an error. |

## `[local]`

The local approximation of a correlated method. `method.kind` names the
method (`rimp2`, `drpa`, `linlccd` or, integral-direct only, `att-rimp2`); this section says whether and how its
amplitudes are truncated. Without it (or with `scheme = "none"`) the method is
computed exactly. On any other kind the section is an error. The local runs
are closed shell and `task = "energy"` only, and every printout and run-log
`result` carries the model: `"<method> (exact)"` with `"local": null`, or the
scheme, `eps` and kept fraction. See
[The MP2 family](../methods/mp2.md#exact-and-local-mp2) and
[Exact and local correlation](../methods/index.md#exact-and-local-correlation).

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `scheme` | string | `"none"` | `none` `amplitude-threshold` | `amplitude-threshold`: drop pair amplitudes whose localized integral is at or below `eps` (single threshold). Any other value is an error. |
| `eps` | float | **required** with `amplitude-threshold` | ≥ 0, finite | The threshold is part of the model and has no default. `0` keeps every amplitude and reproduces the exact method. An error with `scheme = "none"`. |
| `eps_sweep` | array of floats | none | each ≥ 0 | `drpa` only. Several ε on one SCF and one localized assembly; sorted and de-duplicated, one result block per point. Instead of `eps`, not with it. |
| `reference` | bool | `false` | | Also compute the exact method (canonical RI-MP2, canonical plasmon dRPA, exact LinLCCD) and print the local error against it. Costs the full exact calculation. An error with `scheme = "none"`. |
| `integral_direct` | bool | `false` | | `rimp2` only. The integral-direct local MP2: never forms the global 3-index tensor. |
| `aux_radius` | float | `10.0` | Bohr, > 0 | Integral-direct only (an error otherwise). Aux fit-domain radius. |
| `virt_radius` | float | `12.0` | Bohr, > 0 | Integral-direct only. Virtual domain radius. |
| `ao_tail` | float | `1e-3` | ≥ 0 | Integral-direct only. `0.0` keeps every shell. |
| `schwarz_skip` | float | `1e-5` | ≥ 0 | Integral-direct only. Cauchy-Schwarz cut on the batch 3-index triples: a triple is skipped when `sqrt(max (P\|P)) * sqrt(max (ab\|ab)) < schwarz_skip`, both factors from the run's own operator (Coulomb, erfc or terfc; the terfc factors carry a 1e-5 relative table-error inflation each, so the product is scaled by (1 + 1e-5)²). For terfc it is available for orbital shells up to g, r0 in [0.3, 2.5] Å and r0·ω ≤ 8.9, and errors outside that (set `0.0` to run anyway). See [integral-direct local MP2](../methods/mp2.md). `0.0` evaluates every triple (bit-identical to a run without the cut). |
| `batch_merge` | integer | `4` | ≥ 1 | Integral-direct only. |
| `gate_cal` | float | none (gate off) | > 0 | Integral-direct only. Pair-gate calibration (~0.7 Coulomb, ~0.02 erfc ω = 1). |
| `virt_schwarz_kappa` | float | none (off) | > 0 | Integral-direct only. ε-linked Schwarz virtual-candidate screen. |

## `[rpa]`

Read by `pdep-rpa`, `gw`, `bse-tda`, `tdhf-static-polarizability`, and, for
`trunc_thresh` only, `rs-mp2-rpa`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `auxbasis` | string | `"cc-pvdz-ri"` | aux basis name | |
| `frozen_core` | int, string or bool | `0` | as `[mp2]` | |
| `xc` | string | none (HF reference) | XC name | Switches the reference to RKS, or to UKS when open shell. Required by `tdhf-static-polarizability`. `bse-tda` ignores it. `pdep-rpa` with `task = "optimize"` uses an RHF reference; `xc` applies only to `task = "energy"`. |
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
| `compute_polarizability` | bool | `true` | | `alpha_tensor`, the molecular static α. |
| `compute_alpha_atomic` | bool | `true` | | `alpha_atomic`: the Krishtal–Senet–Van Alsenoy intrinsic per-atom α (JCP 125, 034312 (2006)), always Becke-partitioned; `c6_partition` does not affect it. Charge transfer between atoms is excluded; with `compute_polarizability` also on, the remainder `alpha_ct` = `alpha_tensor` − Σ_A `alpha_atomic` is exported. |
| `compute_electric_field` | bool | `true` | | |
| `compute_density_matrix` | bool | `true` | | |
| `compute_dipole` | bool | `true` | | |
| `compute_hirshfeld_charges` | bool | `true` | | Proatoms are free-atom SCF densities in the molecule's basis and SCF settings, as in Python's `hirshfeld_charges`. |
| `compute_lowdin_charges` | bool | `true` | | |
| `compute_mulliken_charges` | bool | `true` | | |
| `compute_chelpg_charges` | bool | `true` | | |
| `compute_resp_charges` | bool | `true` | | |
| `compute_c6` | bool | `true` | | With `c6_source = "pdep"`, also exports `alpha_ct_dynamic` (nfreq, 3, 3): molecular α(iω) − Σ_A α^A(iω). |
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
| `reference` | string | `"uhf"` | `uhf` `rohf` (case-insensitive) | `gw` on an open-shell molecule only: the reference is UHF or ROHF (UKS or ROKS with `[rpa] xc`). An error on a closed-shell molecule or another kind. Same as Python `run_u_gw(reference=)`. |

## `[tddft]`

Read by `tda` and `tddft`. Also set `[mp2] auxbasis` (see above).

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `n_roots` | integer | `3` | | |
| `xc` | string | none (HF reference: CIS or TDHF) | XC name | Selects the reference functional and the f_xc kernel. Meta-GGA, VV10 and range-separated functionals are refused. |
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
| `hessian` | string | `"auto"` | `auto`; `analytic`; `analytic-moving-grid`; `fd` `finite-difference` | `auto` uses the analytic Hessian for RHF (closed shell), UHF (aufbau occupations) and ROHF (`kind = "rohf"` without `[dft] functional`) with the Coulomb operator and exact four-centre J/K (RHF, UHF and ROHF also with density-fitted J and/or K: `[scf] df_j_aux` and/or `df_k_aux` set, ω = 0 only, no external potential (an ECP is analytic with RI), a fitting-basis metric with no eigenvalue below 1e-10 (RI-K: none below 1e-7 and cond(V) at most 1e8 for `auto`), and RI-K memory that fits the `[memory]` budget (it grows like N⁴: about 10 GB for 30 atoms with 300 basis functions, the same for the two spin channels of UHF and ROHF); no range-separated RI-K, no COSX), a basis up to f functions and a libint2 with second derivatives. For RHF and UHF, ECPs (on the default quadrature backend; `FERRIC_ECP_BACKEND=libecpint` is not analytic), external point charges, Gaussian-smeared external charges and a uniform external field are analytic; for ROHF an external potential is not (an ECP is). Implicit solvent, polarizable embedding, fractional occupations, MOM, constraints and every Kohn-Sham reference (ROKS included) are not analytic for any reference. Anything else uses finite differences. `analytic` is an error where it does not apply. `analytic-moving-grid` is an opt-in that `auto` never selects: the analytic closed-shell Kohn-Sham Hessian on the production (moving) grid, Becke weight response included, so it is the Hessian of the surface the KS gradient differentiates and agrees with `fd` to the finite-difference error. It applies to `kind = "ksdft"` (or `rhf` with `[dft] functional`) on a closed shell, LDA or GGA functionals, pure or global hybrid, with exact or density-fitted J and K. It is an error, before the SCF, for Hartree-Fock, open-shell (UKS/ROKS) references, meta-GGA, VV10, range-separated hybrids, ECPs, external potentials, a pruned `[dft] grid_prune`, `[dft] dispersion`, and what the analytic RHF Hessian refuses (COSX, implicit solvent, fractional occupations, MOM). The output prints `Hessian = analytic`, `analytic-moving-grid` or `finite-difference`; the JSON run log carries a `frequencies` record with the full-precision wavenumbers and Cartesian Hessian. |
| `delta` | float | `5e-3` | Bohr, finite and > 0 | Central-difference step, finite-difference Hessians only. Check the printed `Hessian asymmetry`: it is zero in exact arithmetic. |

## `[memory]`

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `budget_gb` | float | auto | finite and > 0 | Precedence: this key, then `FERRIC_MEM_BUDGET_GB`, then the legacy `FERRIC_OOC_BUDGET_GB`/`FERRIC_ERI3_BUDGET_GB`, then 0.8 × available RAM, then 2 GiB. A value of 0, a negative value or NaN is an error; omit the key for auto. It bounds the ledgered allocations, not total process memory. |
| `three_index_budget_gb` | float | — | | Deprecated alias. `budget_gb` wins if both are set. |

## `[gpu]`

Optional CUDA backend. A default build has no GPU code: there `mode = "on"` is an error and `mode = "auto"` prints a notice and runs on the CPU. A GPU run is deterministic run to run on one device but is not bit-identical to the CPU run (a different summation order, like `FERRIC_BLAS_THREADS` above 1).

| Key | Type | Default | Constraint | Meaning |
|---|---|---|---|---|
| `preset` | string | `"off"` | `off`, `auto`, `on`, `mixed`, `auto-mixed` | One word that sets `mode` and `precision` together; see the presets table below. The root-level key `gpu = "mixed"` is the same as `[gpu] preset = "mixed"` (a file cannot hold both a root `gpu` string and a `[gpu]` table). A `mode`, `precision` or `mixed_kernels` key, or its env var, that disagrees with the preset is an error naming both; one that agrees is kept. `device`, `memory_gb` and `min_flops` combine with any preset. Env: `FERRIC_GPU_PRESET`. |
| `mode` | string | `"off"` | `off`, `auto`, `on` | `auto` uses a device when one is usable and otherwise prints a notice and runs on the CPU; `on` makes an unusable device an error. Env: `FERRIC_GPU`. |
| `device` | integer | 0 | a CUDA ordinal | Which device to use. Env: `FERRIC_GPU_DEVICE`. |
| `memory_gb` | float | 0.8 x free | finite and > 0 | Device-memory pool (decimal GB). A GEMM that does not fit runs on the CPU. Env: `FERRIC_GPU_MEM_GB`. |
| `min_flops` | integer | 549755813888 | | Smallest `2*m*n*k` that an f64 `einsum!` matrix product sends to the device; the default is the rounded-up value of a crossover rule applied to products up to 6144³: the device does not clearly beat 6 CPU cores on a single f64 product at any shape measured (at best a 2-8% median win at 6144³ in two of three runs, a loss in the contested run, and at 4096³ and below the CPU won). Products of 2^39 FLOP or more go to the device, and that region is unmeasured. The device RI-MP2 energy does not consult it. Env: `FERRIC_GPU_MIN_FLOPS`. |
| `precision` | string | `"f64"` | `f64`, `mixed` | `mixed` runs the kernels in `mixed_kernels` with f32 storage and f32 panels accumulated in f64; every other contraction stays f64. Requires `mode` `auto` or `on`. A mixed result is not an f64 result: its error is bounded, measured and documented on the [validation page](validation.md), not validated against a reference code. Env: `FERRIC_GPU_PRECISION`. |
| `mixed_kernels` | string array | the kernels this build ships | `rimp2-energy`, `ccsd-amplitudes`, `dfk-occ`, `cosx-kern`, `dfj-pack` | Which kernels may run in mixed precision; requires `precision = "mixed"`. This build ships `rimp2-energy`: with `precision = "mixed"` the device RI-MP2 energy keeps `B_ov` as f32 and accumulates its panels in f64 (the default `precision` stays `"f64"`, so shipping the kernel changes nothing until you ask). `rimp2-energy` covers the RI-MP2 energy under the Coulomb, erfc and terfc operators, closed-shell and unrestricted (open-shell, one f32 `B_ov` per spin); the erf operator, composite fitted operators, SR-MP2 in RS-MP2+RPA, OO-MP2 and kappa-regularised runs use the f64 device kernel. `dfj-pack` (the device RI-J with the packed raw 3-index tensor resident as f32, f64 accumulation) is named in this build but not shipped. Naming a kernel this build does not ship yet is an error, and a build that ships no mixed kernel refuses `precision = "mixed"` ("no mixed-precision kernel is available in this build"). Kernels that are not names here (SCF diagonalisation, DIIS, metric inverses, GW, grids) never run below f64. Env: `FERRIC_GPU_MIXED_KERNELS` (comma-separated). |

```toml
gpu = "mixed"   # same as [gpu] preset = "mixed": device on, mixed precision for every shipped kernel
```

### GPU presets

A preset fills in every knob below that is not given; the filled-in knobs print with `[source: preset]` in the audit lines.

| Preset | `mode` | `precision` | `mixed_kernels` |
|---|---|---|---|
| `off` | `off` | `f64` | the build default |
| `auto` | `auto` | `f64` | the build default |
| `on` | `on` | `f64` | the build default |
| `mixed` | `on` | `mixed` | every kernel this build ships (`rimp2-energy`: the Coulomb, erfc and terfc RI-MP2 energy, closed-shell and unrestricted) |
| `auto-mixed` | `auto` | `mixed` | every kernel this build ships (`rimp2-energy`: the Coulomb, erfc and terfc RI-MP2 energy, closed-shell and unrestricted) |

`preset = "off"` is the same as no `[gpu]` key. A narrower `mixed_kernels` list is allowed under `mixed` and `auto-mixed`; naming a kernel the build does not ship is an error. Precedence for the preset itself: the command-line flag `ferric --gpu <preset> input.toml` (or `--gpu=<preset>`), then `[gpu] preset` or the root `gpu` key, then `FERRIC_GPU_PRESET`, then `off`. The flag labels its audit line `[source: command line]`; a `mode`, `precision` or `mixed_kernels` key or env var that disagrees with the flag is an error naming both, and `--gpu` given twice, without a value or with an unknown name is refused (exit code 2).

```
ferric --gpu mixed input.toml
```

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

See [QM/MM](../using/qmmm.md). Smeared PQR charges, polarizable sites and MM
force fields are Python only. (Smeared charges outside QM/MM are available
through `[external_potential]` with `width`.)

## `[pcm]`

IEF-PCM implicit solvent (`ferric-pcm`). Absent means vacuum. Honoured on
`task = "energy"` by `rhf`, `uhf`, `rohf`, `ksdft` and `pdep-rpa` (for
`pdep-rpa` the RPA correlation is evaluated on the solvated reference). Any
other kind, a gradient task (no gradient has a PCM term), `[cosmo]` in the
same file, or `pdep-rpa` with `[rpa] export_npz` is an error. The solvent
table and checks are shared with Python `run_rhf(solvent=...)`. See
`examples/water-pcm.toml`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `epsilon` | float | — | finite and > 1 | Dielectric constant. Give exactly one of `epsilon` and `solvent`. |
| `solvent` | string | — | `water` (78.4) `dmso` (46.7) `methanol` (32.6) `ethanol` (24.9) `acetone` (20.7) `dichloromethane`/`dcm` (8.93) `thf` (7.43) `chloroform` (4.71) `toluene` (2.38) `hexane` (1.88), case-insensitive | Dielectric constants at 298 K. An unknown name is an error. |
| `lebedev_order` | integer | `110` | `6` `14` `26` `50` `110` `302` | Tesserae per atomic sphere. |

## `[cosmo]`

Conductor-like implicit solvent, applied to every SCF variant. It cannot be
combined with `[pcm]`.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `epsilon` | float | **required** when the section is present | finite and > 1 | `Default` in code is 78.39, but serde has no default for this key. |
| `radius_scale` | float | `1.17` | > 0 | Multiplies Bondi radii. |
| `lebedev_order` | integer | `110` | 6/14/26/50/110/302 | |
| `s_matrix_kind` | string | `"GaussianSmeared"` | `GaussianSmeared` `PointCharge` | Serde variant names, case-sensitive. |

## `[external_potential]`

Strict, like every other section: an unknown key here or inside a point
charge is an error.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `point_charges` | array of `{ q, x, y, z, width }` | `[]` | q in e; x, y, z in **Bohr**; `width` in **Bohr**, finite and > 0 | Written as `[[external_potential.point_charges]]` tables. `width` is optional: with it the charge is Gaussian-smeared, density ∝ exp(−r²/width²) and potential q·erf(r/width)/r (the Python `smeared_charges=`); without it the charge is a point. See `examples/water-rhf-smeared-charge.toml`. The order of the list changes the one-electron potential only by summation roundoff (reordering 6458 charges: 3e-15 to 1e-14 relative) and ferric does not reorder it; a different SCF iteration count was seen only for the closed-shell direct-J/K path, on rc7 and, on main, with `FERRIC_SCF_INCREMENTAL=1` below the incremental noise floor (see Sharp bits, "`energy_conv` is a sanity bound"). |
| `field` | `[Ex, Ey, Ez]` | none | atomic units | Uniform electric field. |

With both empty, the run is identical to a vacuum run.

## `[cell]`

A 3-D periodic system. When present, the run goes to the periodic drivers
instead of the molecular ones. The `[molecule]` XYZ supplies the atoms of the
reference cell (in Å, like every XYZ). Energies are Hartree per cell.

`method.kind` must be `rhf`, `uhf`, `rohf`, `ksdft`, `rimp2` (periodic MP2)
or `pdep-rpa` (periodic dRPA). `rhf`/`uhf`/`rohf` with `[dft] functional` run
RKS/UKS/ROKS. Any other kind is an error. At the Gamma point (no `kmesh`) RKS, UKS and ROKS also run range-separated (CAM) GGA hybrids with either `jk`: any libxc one is accepted (`HSE06`, or a libxc ID such as `HYB_GGA_XC_CAM_B3LYP` or `HYB_GGA_XC_WB97X`; the names `CAM-B3LYP` and `wB97X` do not resolve); HSE06 and CAM-B3LYP are pinned against PySCF, open-shell wB97X is accepted, not pinned (the attenuated exchange is the `k = 0` point of the k-point range-separated exchange; PySCF Gamma pins agree to 1e-12; with `jk = "rsgdf"` the full-Coulomb fit is built twice, once for the Gamma J and once inside the attenuated exchange, so the build costs about 1.9× a plain Gamma run, and the exchange also contracts a J it discards every iteration; with `jk = "dense"` the tensors take 40 nao⁴ bytes, not 8, and `max_eri_gb` counts all of them); `range_split` with a range-separated hybrid is an error; `task = "optimize"` with HSE06 or CAM-B3LYP runs on the closed-shell RKS and the UKS routes with `jk = "dense"` (the analytic forces of `gamma_rsh::gamma_rks_rsh_gradient` and `gamma_rsh::gamma_uks_rsh_gradient`, the 1×1×1 k-point force on the SCF's own exchange) and is an error with `jk = "rsgdf"` (the Gamma route has no RS-GDF exchange derivative; `kmesh = [1, 1, 1]` is the Gamma point and has the closed-shell RS-GDF force) and on the ROKS route (energy only). A meta-GGA on the Gamma-point UKS route is an error (no Gamma tau term); `kmesh = [1, 1, 1]` is the Gamma point and runs SCAN / r2SCAN UKS. `task = "optimize"` works at the
Gamma point for the SCF routes (RHF, UHF, ROHF, RKS, UKS, ROKS) with either
`jk`: it moves the atoms at a fixed lattice using the analytic periodic
force, and prints the final gradient (Hartree/Bohr per cell). A Gamma-point
RKS with SCAN / r2SCAN / TPSS optimizes too, with `jk = "dense"` only (the
force is the 1×1×1 k-point force fed the Gamma SCF; it builds the 1×1×1
dense kernels, 32 nao⁴ bytes on top of the 8 nao⁴ Gamma ERI, and `max_eri_gb`
counts all of them before the SCF);
`jk = "rsgdf"` with a meta-GGA and `task = "optimize"` is an error. With `kmesh` it
works for `rhf`, `uhf` and `rohf` (no functional) and for Kohn-Sham (`ksdft`,
or `rhf`/`uhf`/`rohf` with `[dft] functional`: RKS, UKS, ROKS) using the analytic
k-point force, with either `jk` on every route except the dense-only cases
named below (range-separated hybrids and SCAN / r2SCAN on the UKS route need
`jk = "dense"`; unsplit RS-GDF: `range_split` with `task = "optimize"` is an
error). The functionals are LDA,
GGA and global hybrids such as PBE0 and B3LYP on every Kohn-Sham route, plus
the pure meta-GGAs SCAN and r2SCAN on the closed-shell route (either `jk`) and
the UKS route (`jk = "dense"`), TPSS on the closed-shell route (`jk = "dense"`), and the range-separated hybrids HSE06 and CAM-B3LYP (`HYB_GGA_XC_CAM_B3LYP`;
other range-separated hybrids such as wB97X are energy only) on the
closed-shell route (either `jk`) and the UKS route (`jk = "dense"`); the force
differentiates the energy's own XC grid (`n_radial`/`n_angular`/`neighbour_cutoff`)
and, for a hybrid, its `exxdiv`. It is an error for range-separated hybrids
on the UKS route with RS-GDF or on the ROKS route, VV10 functionals, TPSS on RS-GDF or an open-shell route, hybrid
meta-GGA, meta-GGA on ROKS and UKS meta-GGA on RS-GDF,
and for `rimp2`/`pdep-rpa`. Cells whose basis carries an ECP are supported by the
k-point force and stress of every route above (RHF, UHF, ROHF, RKS, UKS, ROKS). The lattice is fixed in `task = "optimize"` (no
stress relaxation). The analytic stress of a k-point SCF is printed by a
`task = "energy"` run with `stress = true` (see the key below): `rhf`, `uhf`,
`rohf`, `ksdft` and the `[dft] functional` routes (RKS, UKS, ROKS), either `jk`
except where the `stress` key below says `jk = "dense"`,
the 3×3 tensor in Hartree/Bohr³ and in GPa, and the same two tensors in the
run log's `run_end` record (`extra.stress_hartree_per_bohr3`, `extra.stress_gpa`,
full precision; see [run logs](../using/run-logs.md)). It equals the Python
`with_stress=True` tensor bit for bit (Python: `run_rhf_kpts`, `run_uhf_kpts`,
`run_rks_kpts`, `run_uks_kpts`, `run_roks_kpts`, `run_rohf_kpts`), with the
gate shared: every route runs with either `jk`, the closed-shell SCAN /
r2SCAN and range-separated hybrid (HSE06, CAM-B3LYP) stress included,
except the UKS SCAN / r2SCAN stress, which needs `jk = "dense"` (unsplit
RS-GDF: `range_split` with `stress = true` is an error); there is no
open-shell range-separated stress (UKS: no range-separated stress term;
ROKS: no range-separated derivative). The k-point
force and stress use an explicit
functional allowlist that is separate from the energy gate
(`ferric_pbc::kxcgrad::check_deriv_functional`, per route and J/K, wrapped by
`check_kpoint_deriv_functional`, which the Python
bindings call too). The `run_end` record of an optimization carries the
returned geometry and the gradient there at full precision
(`extra.final_geometry_bohr`, `extra.final_gradient_hartree_per_bohr`). `solve_krks` (every k-point Kohn-Sham energy run, `run_rks_kpts` and the CLI included) clones `S(k)` and `h(k)` transiently during the SCF (`2 N_k n² × 16` bytes, not charged to the budget); the periodic XC Bloch AO cache of the SCF does not follow `[memory] budget_gb`. Each optimizer step runs a fresh SCF from the core guess; at a k-point mesh with `uhf` it can land in a different spin solution between steps, which shows as a visible jump in the step energies (nothing is hidden). The force is taken through the hcore and J/K objects the SCF built, which stay in memory for the force: their bytes are charged against the same memory budget (`[memory] budget_gb` with `jk = "rsgdf"`), and with `jk = "dense"` the kernels count against `max_eri_gb` while they exist. `task = "frequencies"` is an error.

The periodic run reads only `[cell]`, `[scf]` `max_iter`, `[scf]`
`density_conv` (Gamma point) or `energy_conv` (k-point mesh), `[dft]`
`functional`, `[memory]` (RS-GDF only) and `[optimize]`. Any other section
or key is an error. A charged cell is an error.

| Key | Type | Default | Allowed values | Notes |
|---|---|---|---|---|
| `lattice` | 3×3 float array | **required** | rows are the lattice vectors | In `unit`. |
| `unit` | string | `"angstrom"` | `angstrom` `bohr` | Applies to `lattice`, `omega` and `gdf_omega` (as their inverse) and `neighbour_cutoff`. |
| `kmesh` | `[n1, n2, n3]` | none (Gamma point) | each ≥ 1 | Selects the k-point drivers: `rhf`, `uhf`, `rohf` (no functional); Kohn-Sham (`ksdft`, or `rhf`/`uhf`/`rohf` with `[dft] functional`): RKS with LDA, GGA, global-hybrid and SCAN / r2SCAN / TPSS functionals (and range-separated (CAM) GGA hybrids such as `HSE06`, `HYB_GGA_XC_CAM_B3LYP` and `HYB_GGA_XC_WB97X`, with either `jk`; forces and stress for HSE06 and CAM-B3LYP with either `jk`, wB97X energy only), UKS (`ksdft` with multiplicity > 1 included) with the same functionals as RKS except TPSS (SCAN / r2SCAN with `jk = "dense"`, energy, forces and stress; the range-separated hybrids as an energy with either `jk`, any libxc CAM GGA hybrid accepted, HSE06 and CAM-B3LYP pinned, open-shell wB97X accepted, not pinned, and the HSE06 / CAM-B3LYP force with `jk = "dense"`), and ROKS with LDA, GGA, global-hybrid and range-separated hybrid functionals (range-separated: energy only, either `jk`); `rimp2` and `pdep-rpa`. `task = "optimize"` and `stress = true` run with closed-shell HSE06 / CAM-B3LYP (either `jk`) and TPSS and UKS SCAN / r2SCAN on `jk = "dense"`; `task = "optimize"` (not `stress`) runs with UKS HSE06 / CAM-B3LYP on `jk = "dense"`. Not yet supported with `kmesh`, each an error naming the gap: meta-GGAs on ROKS, `stress = true` with a range-separated hybrid on an open-shell route (the UKS stress has no range-separated term), `task = "optimize"` with a range-separated hybrid on the UKS route with `jk = "rsgdf"` or on the ROKS route, a UKS meta-GGA with `jk = "rsgdf"`, VV10 functionals, TPSS on UKS/ROKS, `task = "optimize"` or `stress = true` with TPSS on `jk = "rsgdf"`, meta-GGAs other than the pure SCAN / r2SCAN / TPSS, and `task = "optimize"` for `rimp2`/`pdep-rpa` (see `task`). Open-shell k-point KS and ROHF use `ewald_start`; `[cell] n_radial`/`n_angular`/`neighbour_cutoff` apply to the Kohn-Sham routes and, with a 1×1×1 mesh, the runs equal the Gamma-point ones. |
| `centring` | string | `"gamma"` | `gamma` `mp` | Requires `kmesh`. `gamma` is the Gamma-centred mesh (contains Γ; PySCF `make_kpts(n)`); `mp` is the shifted Monkhorst-Pack mesh (PySCF `with_gamma_point=False`). Energies, `task = "optimize"` forces and `stress = true` all accept either. |
| `exxdiv` | string | `"ewald"` | `ewald` `none` | Exchange G = 0 treatment. |
| `jk` | string | `"dense"` | `dense` `rsgdf` | `dense` is a toy-scale dense AFT tensor. |
| `auxbasis` | string | none | bundled basis name | Required by `jk = "rsgdf"`; an error with `dense`. |
| `range_split` | bool or float | `false` | `false`, `true` (λ = 1), or a number > 0 (λ) | Opt-in RS-GDF range split: moves the short-range blocks whose Fourier transform converges in the long-range G sphere into G space. Orbital primitives with exponent ≤ λω²/2 and aux primitives with exponent ≤ λω² count as smooth, where ω is the RS-GDF split (`gdf_omega`, default 1 Bohr⁻¹; not the `omega` key). λ ≤ 1 adds no G vectors; the energy matches the unsplit build to fitting precision. Requires `jk = "rsgdf"` and `task = "energy"` (Gamma point or `kmesh`). An error with `task = "optimize"`: the optimizer is not wired to the range-split forces. |
| `omega` | float | min(2.5 √π / V^(1/3), 0.9636 × ω_gdf), ω_gdf = `gdf_omega` in Bohr⁻¹ | > 0, in `unit`⁻¹ | Nuclear-attraction Ewald split. Any value gives the same energy to truncation (~1e-11 Ha); the default balances the real- and reciprocal-space nuclear-attraction cost, and its cap keeps the reciprocal-space sphere inside the RS-GDF one (so the cap scales with `gdf_omega`). |
| `gdf_omega` | float or `"auto"` | 1 Bohr⁻¹ | finite, > 0, in `unit`⁻¹; or `"auto"` | RS-GDF Ewald split ω. Any value gives the same energy to the fitting precision; it moves work between the short-range lattice sums (radii ∝ 1/ω) and the long-range G sphere (\|G\| ≤ 2ω √ln(10¹³)). The `range_split` thresholds and the RS-GDF forces follow it. Requires `jk = "rsgdf"` (an error with `dense`). Same as Python `gdf_omega=` (Å⁻¹). `"auto"` (opt-in; the default is unchanged) picks ω from a cost model of the cell: it prices the short-range triplet walk (the build's own screen, sampled) against the long-range G sum at ω = 0.25..2 Bohr⁻¹ and leaves 1 Bohr⁻¹ unless another value is predicted at least 10% cheaper. On the study cells (Gamma RHF, cc-pVDZ / cc-pVDZ-RI, 6 threads) sparse molecular crystals run 1.4-2.4× faster (dry ice 20.0 → 11.8 s) and dense cells keep 1 Bohr⁻¹ at a cost of ~0.2-0.5 s for the choice. Gamma-point `task = "energy"` only; an error with `kmesh` and `task = "optimize"`. The chosen value is printed in the header and on stderr. Python: `ferric.auto_gdf_omega(...)` returns the value to pass as `gdf_omega=`. |
| `sr_column_rotation` | bool | on for a `jk = "rsgdf"` run (Gamma or `kmesh`), off with `jk = "dense"` | `true` `false` | Column rotation of generally contracted shells (the separate s/p contraction columns of cc-pVXZ-style bases) in the short-range walks: the RS-GDF short-range 3-centre sum and the hcore short-range nuclear attraction run on rotated columns and are transformed back exactly (at a `kmesh` the back-transform acts on the short-range residue bins and on `V_SR(k)` before the Bloch phase sums, the long-range and G = 0 parts stay in the parent basis). The energy matches the unrotated build to the screening precision (≤ 1.5e-11 Ha/cell at `gdf_omega` ≥ 0.7 on dry ice and diamond, cc-pVDZ) with fewer short-range integral calls (dry ice 25.9 → 22.4 s, diamond 7.84 → 3.08 s at 6 threads, `gdf_omega = 1`, `range_split = true`). The analytic forces and stress (`task = "optimize"`) differentiate the same rotated walks, with the derivative blocks evaluated on the rotated columns, so optimisations keep the saving; at the Gamma point forces and stress equal the unrotated ones to ≤ 1e-12 (Ha/Bohr, Ha) on the test cells; at a `kmesh` the gap to the unrotated derivative is set by the screen (force 1.5e-10 Ha/Bohr, stress 6.5e-9 Ha on H2 cc-pVDZ at hcore/fit precision 1e-9/1e-10, shrinking with the precision). A basis with nothing to rotate (STO-3G, Pople, segmented) runs the unrotated build bit for bit. Absent, it is on for a `jk = "rsgdf"` run (Gamma point or `kmesh`) and off with `jk = "dense"`; the header line `sr_column_rotation =` says which and why. `false` turns it off everywhere. `true` is an error with `jk = "dense"`. The k-point analytic forces (`task = "optimize"` with `kmesh`) and the library's k-point stress differentiate the rotated walks (the derivative blocks on the rotated columns, the residue weights transformed with `Tᵀ W T`). The stage table counters report how many columns were rotated: `hcore SR rotated columns` and `rsgdf SR3 rotated columns` at the Gamma point, `k hcore SR rotated columns` and `k rsgdf SR3 rotated columns` at a `kmesh` (0 = unrotated). Same as Python `sr_column_rotation=` on the Gamma bindings (`None` = this default). |
| `max_eri_gb` | float | `0.5` | > 0 | Dense tensor cap. An error with `rsgdf`. |
| `ewald_start` | string | `"staged"` | `staged` `direct` | Open-shell SCF with `exxdiv = "ewald"` only. |
| `denominators` | string | **required** for `rimp2`/`pdep-rpa` | `shifted` `unshifted` | An error on other kinds. |
| `frozen_core` | integer | `0` | ≥ 0 | `rimp2`/`pdep-rpa` only. |
| `quad_points` | integer | `40` | ≥ 1 | `pdep-rpa` only. Gamma point needs `jk = "rsgdf"`. |
| `drpa_energy` | string | `"quadrature"` | `quadrature` `plasmon` `second-order` | `pdep-rpa` with `kmesh` only. |
| `grad_conv` | float | `1e-9` | > 0 | k-point SCF orbital-gradient threshold. Requires `kmesh`. |
| `stress` | bool | `false` | `true` `false` | Compute and print the analytic stress σ = (1/Ω) dE/dε of the converged k-point SCF (Hartree/Bohr³ and GPa) and write it to the `run_end` record. Requires `kmesh` and `task = "energy"`; `rhf`, `uhf`, `rohf`, `ksdft` and the `[dft] functional` routes. An error, naming `[cell] stress = true`, with the Gamma point, `task = "optimize"`, `rimp2`/`pdep-rpa`, `range_split`, range-separated hybrids on an open-shell route or other than HSE06 / CAM-B3LYP, TPSS and UKS SCAN/r2SCAN on `jk = "rsgdf"` (closed-shell HSE06 / CAM-B3LYP run with either `jk`; closed-shell TPSS and UKS SCAN/r2SCAN on `jk = "dense"`). ECP cells are supported. ROKS meta-GGA and hybrid meta-GGA are refused earlier by the k-point energy gate (`[cell] kmesh with functional ...`), which does not name `stress`. The run does one SCF through the retained hcore and J/K (the iteration printout of the plain energy run is replaced by the iteration count, energy and stress). The SCF must converge (the stress gate for restricted open shell is loose, 1e-5 in the orbital gradient: keep `grad_conv` well below it). |
| `n_radial` | integer | `75` | | Periodic XC grid. Kohn-Sham only. |
| `n_angular` | integer | `302` | `6` `14` `26` `50` `110` `194` `302` `434` `590` | Periodic XC grid. 434 and 590 are refinement grids; any other order is an error. Kohn-Sham only. |
| `neighbour_cutoff` | float | max(10 Bohr, covering-radius bound) (with `neighbour_switch` W: that value + W) | > 0, in `unit`; ≥ bound + W | Periodic XC grid image cutoff D: image atoms within D of a grid point enter its partition weight. Kohn-Sham only. With the default `neighbour_switch = 0` the cutoff is hard: the energy jumps wherever an image crosses D under a displacement or strain (measured: up to 2.2e-7 Ha on a (20, 50) grid at D = 10 on an H2 cell, and a 3.6e-5 Ha jump on an s+p triclinic cell at D = 10; the size depends on the grid and the cell) (a larger D does not remove the jumps; they get denser), and the analytic force and stress do not see the jumps, so a finite-difference step that straddles one misses by the jump / 2h. The default D also depends on the atom positions in a large sparse cell, so set it explicitly for `task = "optimize"` and for finite-difference checks. |
| `neighbour_switch` | float | `0` (hard cutoff) | ≥ 0, in `unit` | Width W of a smooth taper on the image cutoff: an image at distance d from a grid point takes part with presence 1 for d ≤ D − W, falling to 0 at D (quintic smoothstep), and the analytic force and stress include the taper's derivative. This makes the energy continuous under displacement and strain. The taper costs the energy what lowering the cutoff toward D − W would (the automatic D adds W for that reason). Cost: the automatic D grows by W, so the image neighbours per point rise by about (1 + W/D)³; measured grid build on Li–H (75, 302): 0.29 s hard at D = 10, 0.87 s with W = 2 at the automatic D = 12. `0` is the hard cutoff, unchanged. Kohn-Sham only. Not exposed in the Python bindings. |

Periodic `[scf]` defaults are `max_iter = 200`, `density_conv = 1e-10` and
`energy_conv = 1e-12`, not the molecular ones.
