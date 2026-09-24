# Examples

This page indexes every input file in `examples/`. There are 61 TOML files
and no Python scripts. Run one from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 cargo run --release --bin ferric -- examples/water-rhf.toml
```

## What CI checks

**CI parses every shipped example, but runs only the three listed below.** The test
`all_shipped_examples_parse` (in `crates/ferric-cli/src/config.rs`) loads
every `examples/*.toml` through the strict parser. It also checks that each
file's `[molecule] xyz` path exists. It does not execute the calculation or
compare any energy.

Three examples are also executed by integration tests in
`crates/ferric-cli/tests/`:

| Example | Test | What the test asserts |
|---|---|---|
| `water-rhf.toml` | `epistemic_warning.rs`, `verbose_trace.rs`, `qmmm_reports_the_pqr_as_its_geometry.rs` | The run completes and prints no `[warning]` grade line. |
| `water-tdhf-static-alpha.toml` | `epistemic_warning.rs` | The Smoke-grade warning goes to stderr and not to stdout. |
| `water-qmmm.toml` | `qmmm_reports_the_pqr_as_its_geometry.rs` | The embedded energy `-74.9653197421` appears in the output. |

Every other number in the "Reference in header" column below is copied
verbatim from that file's comment block. **Nothing in CI re-checks it.** Where
the header cites a Rust test as the real reference, that test is the check,
not the example.

## Files that do not run as shipped, or are misleading

- **`water-ccsd.toml` runs H2/STO-3G, not water.** The file name does not
  describe its contents.
- **`water-tdhf-static-alpha.toml` fails** with `unphysical static
  polarizability … alpha_xx = -2.682761e0 (negative)`. It leaves `[gw]
  scissor` at 0, where the RPAx@PBE α tensor has a negative diagonal
  component and the run is refused. Add `scissor = 0.36` (about 0.3–0.4 Ha)
  to its `[gw]` section. Its header's "9.24 vs DOSD 9.64 a.u." is the
  `scissor = 0` value; see
  [RPA and GW](../methods/rpa-gw.md) for the α at a physical scissor.
- **`benzene-dfb3lyp-mpi.toml` contains no MPI settings.** Only the file name
  refers to MPI: the same input is meant to be launched under the MPI build
  (`--features mpi`, which is not published).
- **The `water-rhf-cosx.toml` header conflicts with the code.** It says
  "RHF only (UHF/ROHF ignore k_builder)", but `solve_uhf` and the ROHF solver
  both honour `[scf] k_builder`.
- **Three kinds need `FERRIC_TERF_TABLE_DIR`:** `water-mp2v.toml`,
  `water-scs-mp2-2terfc.toml`, and any `rs-mp2-rpa` run with
  `attenuator = "terf"`. They need the tempered-erfc interpolation tables and
  hard-error without them.

## SCF and DFT

See [SCF and DFT](../methods/scf.md).

| File | System / basis | `kind` / `task` | Reference in header | Notes |
|---|---|---|---|---|
| `water-rhf.toml` | H2O / STO-3G | `rhf` / energy | — | `water-qmmm.toml`'s header gives this geometry's energy as −74.9631468000. The file is commented to demonstrate `screening`. |
| `benzene-rhf.toml` | benzene / cc-pVDZ | `rhf` | — | Exact 4-index J/K. |
| `benzene-rhf-dfj.toml` | benzene / cc-pVDZ | `rhf` | — | RI-J only (`cc-pvdz-ri`). |
| `benzene-rhf-rijk.toml` | benzene / cc-pVDZ | `rhf` | — | RI-JK (`def2-universal-jkfit`). |
| `benzene-rhf-def2.toml` | benzene / def2-SVP | `rhf` | — | |
| `benzene-rhf-def2-rijk.toml` | benzene / def2-SVP | `rhf` | — | RI-JK. |
| `decane-rhf.toml` | decane / STO-3G | `rhf` | — | `k_builder = "link"`. |
| `water-rhf-cosx.toml` | H2O / cc-pVDZ | `rhf` | "E(COSX) − E(direct) is reported in crates/ferric-scf/tests/cosx_scf.rs" | COSX exchange. See the header conflict above. |
| `h_uhf.toml` | H atom / STO-3G, doublet | `uhf` | — | |
| `h2_opt.toml` | stretched H2 / STO-3G | `rhf` / optimize | — | |
| `water-frequencies.toml` | H2O / STO-3G | `rhf` / frequencies | — | FD Hessian. Check `Hessian asymmetry`. |
| `benzene-dfb3lyp.toml` | benzene / def2-SVP | `ksdft` B3LYP | — | RI-JK is on automatically for `ksdft`. |
| `benzene-dfb3lyp-mpi.toml` | benzene / cc-pVDZ | `ksdft` B3LYP | — | See above: no MPI settings in the file. |
| `water-wb97xv.toml` | H2O / cc-pVDZ | `ksdft` wB97X-V | — | |
| `water-pbe-d3bj.toml` | H2O / cc-pVDZ | `ksdft` PBE + D3(BJ) | — | Prints E(KS-DFT) and E(D3BJ) separately. |
| `water-pbe-pruned-grid.toml` | H2O / cc-pVDZ | `ksdft` PBE / energy | "removes ~23% of the grid points" (at 75×110) | Validated in `crates/ferric-dft/tests/grid_prune_live_scf.rs`. |
| `h2-lda-opt.toml` | H2 / STO-3G | `ksdft` LDA / optimize | — | |
| `water-qmmm.toml` | H2O + Na⁺ (PQR) / STO-3G | `rhf` + `[qmmm]` | "vacuum −74.9629466809, embedded −74.9653197421, i.e. −1.489 kcal/mol from the ion at 4 A. Verified against `ferric.run_rhf(point_charges=...)` to all 10 digits." | The vacuum number is at the PQR geometry, not the xyz. The embedded number is asserted by a test. |

## MP2 family

See [The MP2 family](../methods/mp2.md).

| File | System / basis | `kind` | Reference in header | Notes |
|---|---|---|---|---|
| `water-rimp2.toml` | H2O / cc-pVDZ | `rimp2` | — | All-electron. |
| `water-rimp2-frozen-core.toml` | H2O / cc-pVDZ | `rimp2` | Expected log line: "`[ferric] frozen core: 1 orbital(s) frozen from [mp2] frozen_core = "auto"`" | |
| `water-lmp2.toml` | H2O / 6-31G | `lmp2` | — | Prints the canonical RI reference alongside. |
| `alkane8-lmp2-direct.toml` | octane / 6-31G | `lmp2-direct` | — | Every locality knob at its default. |
| `water-mp3.toml` | H2O / cc-pVDZ | `mp3` | — | |
| `water-oo-rimp2.toml` | H2O / cc-pVDZ | `oo-rimp2` | — | Smoke grade. |
| `water-attmp2.toml` | H2O / aug-cc-pVDZ | `att-rimp2` | — | ω = 0.420 Å⁻¹. |
| `water-scs-mp2.toml` | H2O / cc-pVDZ | `scs-mp2` | — | Grimme coefficients (defaults). |
| `water-scs-mp2-2terfc.toml` | H2O / cc-pVDZ | `scs-mp2-2terfc` | — | Thesis defaults r0 = 0.75/1.05 Å. Needs the terf tables. |
| `water-laplace-rimp2.toml` | H2O / cc-pVDZ | `laplace-mp2` | — | |
| `water-laplace-sos-mp2.toml` | H2O / cc-pVDZ | `laplace-sos-mp2` | — | |
| `water-mp2v.toml` | H2O / aug-cc-pVDZ | `mp2-v` | — | The header warns that aDZ is outside the fitted basis. Needs the terf tables. |
| `water-rs-mp2-rpa.toml` | H2O / aug-cc-pVDZ | `rs-mp2-rpa` | — | Smoke grade. |

## Coupled cluster and double hybrids

See [Coupled cluster](../methods/cc.md) and
[RPA and GW § double hybrids](../methods/rpa-gw.md).

| File | System / basis | `kind` | Reference in header | Notes |
|---|---|---|---|---|
| `water-ccsd.toml` | **H2** / STO-3G | `ccsd` | "CCSD correlation energy = −0.02052453 Hartree (exact-integral numpy reference)" (checked by `test_ccsd_h2_sto3g` in `crates/ferric-cc/src/ccsd.rs`) | Misnamed. Aux `def2-qzvpp-rifit`. |
| `water-linlccd.toml` | H2O / 6-31G | `linlccd` | — | |
| `water-wb97xlv.toml` | H2O / 6-31G | `wb97x-l-v` | — | λ = 0.6, ω = 0.1 Bohr⁻¹ (published values). Smoke grade. |
| `water-b2plyp.toml` | H2O / cc-pVDZ | `b2plyp` | — | Spike grade. Aux `cc-pvdz-rifit` (an alias of `cc-pvdz-ri`). |

## RPA, C6 and properties

See [RPA and GW](../methods/rpa-gw.md).

| File | System / basis | `kind` | Reference in header | Notes |
|---|---|---|---|---|
| `water-pdep-rpa.toml` | H2O / cc-pVDZ | `pdep-rpa` | — | Writes eigenpotential cube files. |
| `h2o-pdep-rpa-props.toml` | H2O / cc-pVDZ | `pdep-rpa` | — | NPZ export. |
| `benzene-pdep-rpa.toml` | benzene / def2-SVP | `pdep-rpa` | — | RI-JK SCF. |
| `benzene-pdep-rpa-export.toml` | benzene / cc-pVDZ | `pdep-rpa` | — | Exports cube files. |
| `benzene-rijk-pdep-rpa.toml` | benzene / cc-pVDZ | `pdep-rpa` | — | Full rank (`trunc_thresh = 0`). |
| `water-c6-pdep.toml` | H2O / aug-cc-pVTZ | `pdep-rpa`, `c6_source = "pdep"` | "DOSD molecular reference (Meath/Toulouse): C6(H2O–H2O) = 45.3 a.u." | Compare the printed "molecular C6" line against it, not the sum of the NPZ `c6_iso` entries. Writes to `/tmp`. |
| `argon-c6-rpa-pbe.toml` | Ar / aug-cc-pVTZ | `pdep-rpa` @PBE | "C6(Ar-Ar) = 56.4 a.u. vs DOSD 64.3 (-12%); the full He/Ne/Ar sweep gives mean \|err\| 8.9% at RPA@PBE — vs 39% at RPA@HF" | Writes to `/tmp`. |
| `water-tdhf-static-alpha.toml` | H2O / cc-pVDZ | `tdhf-static-polarizability` @PBE | "measured 9.24 vs DOSD 9.64 a.u. at this same functional/basis via the underlying run_bse_c6_ks spike" | **Does not run as shipped** (`[gw] scissor` is 0; see above). Static α only. Executed by a test (for the warning only). |

## GW and BSE

| File | System / basis | `kind` | Reference in header | Notes |
|---|---|---|---|---|
| `water-g0w0-pbe.toml` | H2O / cc-pVDZ | `gw` G0W0@PBE | "HOMO IP should match … PySCF gw_ac reference (11.1714 eV) to <0.1 eV" (`crates/ferric-gw/tests/g0w0_pbe_h2o.rs`) | |
| `oh-ugw.toml` | OH doublet / cc-pVDZ | `gw` U-G0W0@UHF | "alpha-HOMO IP in the ~13-14 eV window, bracketing experiment 13.02 eV" (`crates/ferric-gw/tests/oh_u_g0w0.rs`) | |
| `water-bse-tda.toml` | H2O / cc-pVDZ | `bse-tda` | "Measured via this exact TOML (2026-07-18): 8.4572 eV", against a PySCF-integral cross-check of 8.46 eV and a sanity window of [5, 12] eV | |
| `water-augccpvdz-bse-tda.toml` | H2O / aug-cc-pVDZ | `bse-tda` | Literature: "aug-cc-pVDZ CCSDT Delta_Evert = 9.279 eV, f = 0.058; CBS TBE = 7.71+-0.02 eV, f = 0.052+-0.001" | No ferric number is recorded. |
| `h2co-bse-tda.toml` | H2CO / cc-pVDZ | `bse-tda` | — | Pilot run. The lowest state is dark (f ≈ 0). |
| `c2h4-bse-tda.toml` | C2H4 / cc-pVDZ | `bse-tda` | — | Pilot run. |
| `formaldehyde-bse-tda-augdz.toml` | H2CO (QUESTDB geometry) / aug-cc-pVDZ | `bse-tda` | Literature: "lowest singlet 1^1A2 (n->pi*, symmetry-forbidden) at 3.966 eV (QUESTDB TBE…)" | The template for the Thiel-set files below. |
| `acetaldehyde-bse-tda-augdz.toml`, `butadiene-bse-tda-augdz.toml`, `cyclopropene-bse-tda-augdz.toml`, `ethylene-bse-tda-augdz.toml`, `furan-bse-tda-augdz.toml`, `glyoxal-bse-tda-augdz.toml`, `pyrazine-bse-tda-augdz.toml`, `pyridine-bse-tda-augdz.toml`, `pyrimidine-bse-tda-augdz.toml` | QUESTDB Thiel-set molecules / aug-cc-pVDZ | `bse-tda` | — | Same setup as the formaldehyde file. The reference values are in `testdata/reference/thiel_set_subset.json`. |

## TDDFT

| File | System / basis | `kind` | Reference in header | Notes |
|---|---|---|---|---|
| `water-tda.toml` | H2O / cc-pVDZ | `tda` (CIS, 5 roots) | — | Default aux `cc-pvdz-rifit` (an alias of `cc-pvdz-ri`). |
| `water-tddft-pbe.toml` | H2O / cc-pVDZ | `tddft` @PBE (5 roots) | — | The f_xc kernel is missing; the run warns. |

See also [Capabilities](./capabilities.md) for the grade of each `kind`,
[Input reference](./input.md) for every key, and
[What is validated](./validation.md).
