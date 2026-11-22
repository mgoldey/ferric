# Architecture and code conventions

This page describes how the crates fit together and the patterns new code
should follow. It exists because most repetition in ferric is not in the
numerical kernels. It is in the plumbing around them: the Python bindings and
the CLI each rebuild the same configs, defaults and error handling by hand.
The rules below keep that from growing, and the roadmap at the end lists the
follow-up refactors that remove what is already there.

## Layering

```
ferric-core
  -> ferric-integrals, ferric-dft, ferric-quadrature, ...
    -> ferric-scf
      -> correlated methods (ferric-mp2, ferric-cc, ferric-rpa, ferric-gw, ferric-tddft, ...)
        -> frontends (ferric-cli, ferric-python)
```

- **Method crates own policy.** Every default (convergence thresholds, auxiliary
  basis names, cross-option rules such as "no RI-K default under COSX") lives
  in the method crate as a named `const` or in a `Default` impl. Frontends do
  not hard-code defaults.
- **Frontends are thin adapters.** The CLI turns TOML into the method crate's
  options type; the Python bindings turn keyword arguments into the same type.
  Neither is a second source of truth. `ferric_scf::options` is the first such
  lowering; new SCF entry points go through it.
- **Python does not reach into ferric-cli** except through `ferric_cli::run`
  (the bundled `ferric` command). CLI helpers may call `process::exit`, which
  would kill the interpreter. Existing uses (`LocalCfg`, `LocalModel`, `LocalScheme`, `Mp2Cfg`) are slated
  to move into ferric-mp2.
- **Python imports nothing above it.** The package under
  `crates/ferric-python/python/ferric/` never imports `smeltery` or
  `attestation`; both depend on ferric, not the other way round.
  `tests/test_python_does_not_import_above.py` scans the package's imports.

## Options and configs

- Inside a method crate, a config is a plain struct with `Default`.
- User input is lowered into it by **one pure function per method family**,
  owned by the method crate. It returns `Result<_, FerricError>`, performs no
  I/O and never exits.
- The lowering destructures the options type exhaustively, so a new option
  field is a compile error until it is lowered. It starts from the config's
  `Default`, so a new *config* field takes its default, exactly as the old
  `..Default::default()` literals did.
- Parse at the boundary. An options type uses enums and validated newtypes
  (`AuxBasis { Off, Named }`, `Stability`, `Guess`, `KBuilderKind`,
  `TrahTrigger::new`), not the strings and bools the frontends receive.
- No sentinel values (`0`, `""`, `0.0`, NaN) for "unset", "auto" or "off".
  Use `Option` or an enum. Existing sentinels in `RhfConfig`
  (`three_index_budget_bytes = 0`, `newton_trigger = 0.0`, the empty-string
  auxiliary basis) are on the roadmap.
- One home per invariant. A rule is checked by one function; other layers call
  it rather than restating it with their own message.
- Per-flavour defaults that differ on purpose (for example `run_rhf` vs
  `run_dft` vs the CLI auxiliary-basis policy) are kept, not merged: each
  frontend sets only the options fields it used to set, and every field it
  leaves `None` takes the method crate's default.
- Every lowering has a test asserting it reproduces the previous construction
  field by field, for defaults and for an all-options case.

## Function signatures

- Order: context first (`ParallelContext`, prepared basis, bounds), then
  inputs, then `&Config` last.
- More than seven parameters means an arguments struct.
- No public function takes two bools, and no bool whose `false` selects a
  different algorithm. Use an enum (for example a three-state
  `Stability { Off, Check, Descend }`).
- `&mut` output parameters are for accumulation into an existing buffer (Fock
  matrices, gradients) and the function name says so (`add_`, `accumulate_`).
  Everything else returns a value.
- Reductions over shells or atoms use rayon `fold`/`reduce` with per-thread
  buffers rather than a shared `Mutex` accumulator.
- Solvers take `&Config` and borrowed problem data. Caches inside shared
  owners use `OnceLock`, never `RefCell`, so `&self` stays observably
  immutable under rayon. A config holds no solver output.
- Names: `prep: &PreparedBasis` for the orbital basis and `aux` for a fitting
  basis (retire `obs`).
- Domain logic lives in the core crates. PyO3 classes for inputs are thin
  handles over `inner`; a wrapper does not compute chemistry (for example the
  Hill formula is `Molecule::hill_formula`).

## Errors

- Library crates return `Result<_, FerricError>`. No new `Result<_, String>`
  across a crate boundary.
- **CLI:** `ferric-cli/src/error.rs` holds `CliError` and the `OrExit`
  extension trait. `OrExit` replaces the hand-written
  `unwrap_or_else(|e| { eprintln!(...); exit(1) })` closures. It is a
  transitional step: run functions should move to `Result<_, CliError>` with
  `?`, so that `process::exit` appears only in `main`/`run`. Its first
  producer is `try_scf_config` (the `[scf]` lowering).
- **Python:** convert a `FerricError` with the crate-local `IntoPyResult`
  extension trait (`.py()?`), not a fresh `map_err` per call site. A blanket
  `From<FerricError> for PyErr` is not possible (orphan rule). Older
  `map_err(make_err)` sites migrate as their files are touched. Exception
  classes only ever become narrower subclasses of the class raised today;
  they are never swapped.

## Python API compatibility

Downstream code (scripts, notebooks, applications such as smeltery) passes
keyword arguments through `**kwargs` and catches exceptions by class, so:

- `run_*` parameter names, order and **default values** are frozen. Additions
  are fine. `crates/ferric-python/tests/test_api_snapshot.py` checks this
  against a committed snapshot; a change to it must be additive unless the
  PR is deliberately breaking.
- Result attribute names and `certificate()` keys are frozen.
- `ferric.pyi` changes in the same commit as the signature it describes.
- Units do not change: Angstrom in, Hartree out.

## Shared structure

- **Traits are small and segregated.** `MethodResult` (energy, certificate,
  timings) and `Reference` (spin, orbitals, eigenvalues, occupations) stay
  separate: an MP2 result is a `MethodResult` but not a `Reference`. Add a
  trait only when a second implementer exists. There is no `trait ScfMethod`:
  the RHF, UHF and ROHF iterations differ numerically.
- **Template method at the driver layer, not the solver layer.** Frontend
  drivers share one skeleton (validate, warn, solve, gradient, wrap) with
  variant hooks; `solve_open_shell_entry` in `ferric-python/src/scf.rs` is the
  first instance.
- **Owners establish invariants once.** `ferric_scf::PreparedSystem`
  (molecule with the basis set's ECP applied, prepared basis, Coulomb
  Schwarz bounds, context) is built by a single
  constructor (`EcpMolecule::new` applies the ECP, `EcpMolecule::prepare`
  takes `PrepareOptions` and builds the basis and bounds) and only
  borrowed afterwards. The ferric-python SCF entry points and correlated
  drivers and the CLI's molecular run functions go through it (roadmap
  items 8, 8b and 8c), and the system owns its screening (its solve
  methods refuse a config whose screening is not `PreparedSystem::screening`). The
  frontend-facing drivers of the CAS, OP-MP3, RI-MP2, CC, GW, PDEP-RPA and
  TDDFT families take `&PreparedSystem` or `&PreparedAux`; most kernels
  underneath still take a loose `&Molecule` and `&PreparedBasis`, so the
  types do not enforce the invariant there yet (item 8d).
- **Units (target):** newtypes at API boundaries only (`Bohr`/`Angstrom` on
  molecule constructors and accessors, `Hartree` on result energies), with
  kernels keeping raw `f64` and arrays. No such newtypes exist yet. Today the
  Python constants live in `ferric.units` and match the Rust values, which mix
  CODATA 2010 (Bohr) and CODATA 2018 (eV).

## Module size

A method family should grow into a directory module with fixed slots
(`config.rs`, `solve.rs`, `result.rs`, `gradient.rs`, tests) rather than a
single file of several thousand lines. Splits are move-only commits with
`pub use` re-exports, kept separate from logic changes.

## Refactoring roadmap

Done:

1. Python API signature/default snapshot test.
2. `ferric_scf::options` lowering used by `run_rhf`, `run_uhf`, `run_rohf`.
3. `CliError` and `OrExit` in ferric-cli.
4. `IntoPyResult` in ferric-python.
5. Optional Python integrations: `ferric.rdkit`, `ferric.serialize`,
   `ferric.units`, public QCSchema molecule helpers, ASE dipole and charges,
   and `Molecule.charge`/`multiplicity` getters.
6a. The `ferric_scf::options` lowering extended to `run_dft`, `run_uks` and
    `run_roks` (oracle-tested against their old `rhf_config_budgeted`
    construction), with the RIJCOSX rule as
    `ferric_scf::options::df_k_aux_or_default`.
6b. The keyword-argument drift between the six SCF entry points filled
    additively (appended kwargs, default `None` = the old behaviour, and in
    `ferric.pyi`) wherever a sibling with the same solver already took the
    knob, and all six bindings call one policy,
    `ferric_scf::options::entry::ScfEntryPoint::options`, which also names
    the knobs each entry point refuses because its solver cannot honour them
    (`ScfEntryPoint::refused`: `diis`, `smearing_sigma`, `aurora` outside
    `solve_rhf`; `diis`/`smearing_sigma`/`soscf` on `run_dft`, whose ladder
    sets them per rung; `trah_trigger` and `stability_descent` on ROHF/ROKS;
    `mom_reference` on RHF/ROHF/DFT/ROKS (`"initial"` is UHF/UKS only);
    open-shell `symmetry`, which needs a result class of its own;
    `functional`, the `grid_*` keywords and `dispersion` on the Hartree-Fock
    entry points, and `dispersion` on UKS/ROKS). That table is exported as
    `ferric._scf_refused()` and checked against every signature
    (`test_scf_entry_point_surface.py`: each knob any of the six takes is
    offered or refused, never both). The `options` oracle tests run the old
    literals against that function.
7. The CLI `[scf]` section lowered through `ferric_scf::options`
   (`try_scf_config` in ferric-cli, oracle-tested against the old
   `RhfConfig` literal), removing the hand-kept "CLI and Python agree"
   parity. It is also the template for 12: a `Result<_, CliError>`
   function with one `.or_exit()` at the call site.
9a. `energy`/`total_energy` aliases on every SCF result class (molecular,
    CDFT and periodic), so callers no longer branch on the result type.
-  API snapshot coverage extends to the pure-Python submodules
   (`serialize`, `rdkit`, `qcschema`, `units`, `molden`, `fcidump`, `viz`,
   and the ASE calculator's properties and defaults).
-  Spin-generic KS gradient core in `ferric-scf/src/ks_gradient.rs`
   (closed, UKS and ROKS share one implementation; public functions are
   shims), and shared reference-SCF/Gamma/k-point helpers in the periodic
   bindings (`PbcArgs` carries its `PbcRole`).
8. `ferric_scf::PreparedSystem`: `PreparedSystem::new(mol, basis)` sets the
   molecule's ECP core counts from the basis (`Molecule::apply_ecp`, now
   total: the basis's `n_core` on each ECP atom, 0 elsewhere, whatever the
   molecule arrived with), prepares the basis and its Coulomb Schwarz
   bounds; `prepare_aux` prepares a fitting basis on the same molecule;
   `solve_rhf`/`solve_uhf`/`solve_rohf`/`solve_rhf_ladder` and
   `solve_reference` (UHF iff multiplicity > 1, converged via
   `ScfResult::into_converged`) run on the borrowed system. In
   ferric-python, `prepare` builds it (and, since 8b, `prepare_aux` builds
   the fitting basis on it) for every driver in
   `mp2.rs`, `cc.rs`, `gw.rs`, `rpa.rs`, `tddft.rs`, `opmp3.rs`, the double
   hybrid in `dft.rs` and `ci.rs`; `response_modes` applies the ECP to its
   molecule without the system (it needs no bounds).
   `crates/ferric-python/tests/test_ecp_correlated_references.py` pins each
   family on HI/def2-SVP. ECP decisions:

   | Drivers | ECP | Pinned against |
   |---|---|---|
   | `run_rimp2` (exact, open-shell, gradient), SCS, Laplace(-SOS), attenuated, MP3, OO-RI-MP2 | applied (fix) | `run_rhf`/`run_uhf` energy; RI-MP2 (also `frozen_core=9`) from numpy on `run_rhf` orbitals |
   | terfc, SCS-2terfc, MP2-V, VV10 variants | applied (fix), same helpers | not run on an ECP system (tables / aug-cc-pVTZ only) |
   | CCD, CCSD, CCSD(T), EOM-CCSD, UCCSD | applied (fix) | EOM `rhf_energy` = `run_rhf`; CCSD/UCCSD vs numpy RI-MP2 ratio |
   | GW, U-GW, BSE-TDA, TDHF polarizability | applied (fix) | `run_rhf`/`run_dft`/`run_uhf`; occupied count 13 |
   | PDEP-RPA, RS-MP2-RPA | applied (fix) | `run_rhf` |
   | `run_tddft` | applied (fix) | numpy RI-CIS on `run_rhf` orbitals |
   | double hybrid | applied (fix) | `run_dft`/`run_uks` KS energy |
   | exact LinLCCD | applied (fix) | `run_rhf` |
   | `tune_omega` | applied (fix) | `run_dft` (neutral) / `run_uks` (cation) at the functional's ω |
   | `run_opmp3` | applied (fix) | `run_rhf`/`run_rohf` |
   | `response_modes` | applied (fix: occupied count) | numpy `alpha_bare` on `run_rhf` orbitals |
   | CAS-CI, CASSCF, NEVPT2 | already applied | `run_rhf` |
   | local MP2, local OP-MP3, `run_drpa`, `run_drpa_scan`, local LinLCCD | refused, unchanged | `pytest.raises` |

   The fixes are a behaviour change only for an ECP basis (the drivers had
   prepared the ECP basis, which carries V_ECP and the reduced nuclear
   charge, on the all-electron molecule: a 54-electron HI at -143.3 Ha
   against `run_rhf`'s -297.2 Ha); non-ECP results are bitwise unchanged.
   The RI-MP2, OO-RI-MP2 and PDEP-RPA analytic gradients on HI match
   central differences within 1e-6 Ha/Bohr (tested; the offset is 3.1e-7,
   as for `run_rhf`). Side effects: the CAS drivers now prepare the basis
   (and compute its Schwarz bounds, with the GIL held) before checking the
   active-electron count, so an input wrong in both reports the basis
   error first; a molecule reused from an ECP run (e.g.
   `run_aimd(...).mol()`) in an all-electron basis is all-electron again.
   The CLI's correlated paths applied the ECP by hand on the one molecule
   they all receive; 8b moved them onto the type.
8b. `PreparedSystem` follow-ups, bitwise unchanged results and messages
    (compared against the previous build on Python SCF/correlated calls and
    CLI inputs, ECP and error cases included):
    - One constructor taking a config: `EcpMolecule::new(mol, basis)` is
      the one place that applies the basis set's ECP, and
      `EcpMolecule::prepare(PrepareOptions { screening, ctx })` the one
      place that prepares the basis and builds the bounds
      (`SchwarzBounds::compute_for_screening`; `Schwarz`, the default, is
      `compute` verbatim). The two stages exist because callers validate
      the ECP molecule before any integral (the CLI's frozen core, the CAS
      core count, `run_uks`/`run_roks`' spin check).
      `PreparedSystem::new` is that pair with `PrepareOptions::default()`;
      `PreparedSystem::screening()` reads the kind back from the bounds
      (which table `compute_for_screening` attached), and
      `PreparedSystem::into_mol` hands the molecule to a result that keeps
      it (`run_dft`).
    - The CLI's molecular path (`ferric-cli/src/lib.rs`, via
      `prepare_system`, which keeps the old error order: basis, then
      `[scf] screening`, then bounds), `cas.rs` and `opmp3.rs` build the
      system; `periodic.rs` keeps its own `apply_ecp`, because its molecule
      is moved into a `Cell` and its integrals are lattice sums (the
      molecular Schwarz bounds it builds are never read: only
      `solve_rhf_injected`'s signature asks for them). The Python periodic
      bindings (`pbc.rs`, through `gamma.rs`'s ECP helper) stay out for the
      same reason.
    - `run_rhf`, `run_uhf`, `run_rohf` (`OpenShellEntry` now holds the
      system), `run_dft`, `run_uks` and `run_roks` build the system; the
      ferric-python SCF helpers they share (`solve_rhf_outcome`,
      `solve_symmetric_outcome`, `solve_rhf_with_gradient`, `scf_gradient`,
      `blocks::{solve_open_shell, solve_rhf_hooked, solve_ks_ladder}`) take
      `&PreparedSystem` instead of loose `(ctx, mol, prep, op, bounds)`.
    - `ferric_ci::driver::rhf_start` is gone; the CLI, the ferric-ci
      examples and `cas_kinds.rs` use `rhf_start_prepared`.
    - Side effects: the CLI's CAS and OP-MP3 runs build the Schwarz bounds
      when they prepare the basis, so before the fitting basis and the
      run-log header rather than after (a bad `auxbasis` pays for a bounds
      pass before it is reported, and the bounds time falls outside the
      run-log window; messages are unchanged).
      `crates/ferric-ci/examples/casscf_ah_bench.rs` now applies the ECP
      for an ECP basis (a fix; on such a basis it is not bitwise identical
      to before).
    - `PreparedSystem::prepare_aux` returns a `PreparedAux<'s>` (derefs to
      `PreparedBasis`) that borrows its system and names it
      (`PreparedAux::system`): it cannot outlive the system, and only a
      system makes one. It does not stop `&*aux` being passed to a loose
      kernel next to another system's basis; that needs item 8c. As before,
      an aux set without ECP data carries the all-electron charge on an ECP
      atom (it matters only to a kernel that reads the aux basis's charges).
-  COSX for `run_uks`/`run_roks`: they now apply
   `df_k_aux_or_default`. Before, they applied the RI-K default even under
   COSX, so a COSX call with HF exchange or a global hybrid (the functionals
   whose exchange COSX takes) that left `df_k_aux` unset or named a basis
   failed with a `df_k_aux` conflict. An explicit opt-out (`df_k_aux=""`)
   already worked, and pure or range-separated functionals ignore COSX, so
   they never hit the conflict.
8c. `PreparedSystem` follow-ups, bitwise unchanged results and messages
    for single-threaded callers (compared against origin/main plus the
    `run_omega_ri` compile fix of #691, on Python calls and CLI inputs, ECP
    and error cases included; with several threads the energies vary run to
    run by about 1e-13 Ha, on main as here):
    - The system owns screening: `PreparedSystem::{solve_rhf, solve_uhf,
      solve_rohf, solve_reference, solve_rhf_ladder}` refuse (a
      `FerricError` naming both kinds) a config, or any rung, whose
      `RhfConfig::screening` is not `PreparedSystem::screening()`, and the
      drivers that build their own config (`rhf_start`, `opmp3_scf`) set it
      from the system. A Python `screening=` keyword therefore has to go
      through `PrepareOptions`.
    - `ferric_ci::driver::rhf_start_prepared` is `rhf_start` again.
    - Drivers taking the system: `ferric_ci::driver::{nevpt2_on_rhf,
      casscf_on_rhf, casci_on_rhf}` (`&PreparedSystem`);
      `opmp3_driver::{opmp3_scf, opmp3_on_reference, run_opmp3}`
      (`&PreparedSystem`, `Opmp3Integrals::Ri(&PreparedAux)`);
      `local_opmp3::{local_opmp3_on_reference, run_local_opmp3}`,
      `rimp2::{ri_mp2_on, ri_mp2_fitted_on}`, `u_rimp2::u_ri_mp2_on`,
      `ccsd_closed_shell_on`, `u_ccsd_on`, `ferric_gw::run_gw_on`,
      `ferric_rpa::run_pdep_rpa_on`, `ferric_tddft::run_tddft_on`
      (`&PreparedAux`, reading the molecule, orbital basis and Coulomb
      operator from `PreparedAux::system`). `PreparedAux::basis_for(sys)` is
      the one check that a fitting basis belongs to a system a kernel also
      holds. The Python and CLI callers use them; the loose functions stay
      as the kernels they wrap. The library `run_opmp3` / `run_local_opmp3`
      now use the system's bounds, context and screening (and its ECP)
      instead of building their own `SchwarzBounds` and context.
    - The CLI's per-method `run_*` functions in `ferric-cli/src/lib.rs`
      (every correlated arm, uhf/rohf/cdft, symmetric RHF, the double
      hybrids, PDEP-RPA, the open-shell/GW reference helpers,
      `preflight_exact_drpa`, `cdft::run`) take `&PreparedSystem` and build
      fitting bases with `sys.prepare_aux` (`correlation_aux` returns a
      `PreparedAux`).
    - Python bindings that applied the ECP by hand go through
      `EcpMolecule`: `run_qmmm` builds a `PreparedSystem`, `run_cdft`
      prepares its `EcpMolecule` stage, and the bindings that need only the
      valence molecule use `EcpMolecule::into_mol` / `system::ecp_molecule`
      (integrals.rs, `AoIntegrals`, properties.rs, response_modes.rs,
      md.rs, geometry.rs, `run_saddle`/`run_irc`, gamma.rs's cell check).
    - `system::{prepare, prepare_ecp, prepare_aux}` build the basis, bounds
      and fitting basis with the GIL released.
    - Also: origin/main did not compile ferric-python (#688's
      `run_omega_ri` called the `prepare_with_aux` helper #690 removed);
      fixed as `prepare` + `prepare_aux` (#691).

15. (first part) Types parsed at the boundary, bitwise unchanged for
    valid input (compared against the 8c build on Python SCF/correlated
    calls and CLI inputs, every `k_builder` spelling and error case
    included):
    - `ferric_scf::options::KBuilderKind { Direct, Link, Cosx }` (named
      `...Kind`: `fock::KBuilder` is the trait). `RhfConfig::k_builder` is
      `Option<KBuilderKind>`; `KBuilderKind::parse` is the one whitelist,
      refusing with the solvers' old text (`FerricError::General`, so
      Python still raises `RuntimeError("General error: unknown k_builder
      ...")`). Parsed once per frontend: the six SCF entry points last of
      their keywords, the correlated entry points when they build their
      reference config (`rhf_config_budgeted` / `rhf_config_conv`),
      `run_cdft` before its config, the CLI in `try_scf_config` as its last
      fallible step. `COSX_K_BUILDER` and the solver-side check are gone;
      `df_k_aux_or_default`, the J/K route, the certificate, `EffectiveJk`
      and the COSX schedule take the enum. `ScfJk` no longer copies the
      builder (`ScfJk::label` takes the config's; `ScfJk::resolve` checks
      the spelling as typed so its conflicts quote it). The checkpoint
      identity keeps the old `Debug` text (`Some("cosx")`).
      Behaviour change, for an unknown spelling only: same class and text,
      but raised at the boundary rather than by the solver, so whatever ran
      in between now runs after it (the pre-SCF noise-floor warning, the
      `with_gradient` / hook / symmetry refusals, in some correlated
      bindings the basis preparation; the CLI's banners, run-log header and
      its "SCF ladder failed: General(...)" wrapper are replaced by
      `error: unknown k_builder ...`).
    - `options::ks_jk_fit_defaults`: the one home of the Kohn-Sham JK-fit
      default and the RIJCOSX rule, used by `ScfEntryPoint::options` and by
      `ksdft_ladder` (which had its own copy).
    - `ri_j_task` is the argument of `ScfOptions::into_rhf_config`, not an
      option.
    - `ferric-python`: `parse_fock_hook` returns one
      `Option<Hook { fock, grad }>` (same messages and order), carried by
      the open-shell entry (`OpenShellEntry`, `run_open_ks`).
    - `ferric.pyi`: the `run_dft` / `run_ksdft` `symmetry=True` overloads
      type `with_gradient` as `Literal[False]` and `dispersion` as `None`.

Next, each as its own PR following the patterns above:

8d. What 8c left loose: (a) the kernels under the drivers still take a
    loose `(mol, prep, dfbs, op)`: `run_casscf`, `run_cas_ci`, `sc_nevpt2`,
    and the RI kernels without a `PreparedAux` entry point (CCD, LinLCCD,
    CCSD(T), EOM-CCSD, MP3, OO-RI-MP2, SCS/Laplace/attenuated/terfc MP2,
    MP2-V, RS-MP2-RPA, BSE-TDA, TDHF polarizability, the fitted and
    energy-only PDEP-RPA routes, the analytic gradients), so
    `PreparedAux: Deref<Target = PreparedBasis>` stays: the CLI's
    correlated arms (37 sites in `ferric-cli/src/lib.rs`) and the Python
    drivers in `mp2.rs`, `cc.rs`, `gw.rs`, `rpa.rs`, `dft.rs`,
    `omega_ri.rs` pass `&aux` to them; (b) the ferric-scf solvers take
    loose arguments, and the CLI arms, the Python block helpers,
    `run_qmmm` and `run_cdft` call them with the system's pieces and a
    config of their own, so the system's screening check does not cover
    them (moving them onto the system's methods would turn a config whose
    screening differs from `[scf] screening` into an error); (c)
    `run_uks`/`run_roks` share the open-shell tail of `run_uhf`/`run_rohf`
    in substance but not in code (they skip the pre-SCF noise-floor
    warning, so merging them is a behaviour change); (d) `FockResponse`
    builds non-Coulomb bounds the system cannot own; (e) the GIL is still
    held while `run_qmmm` builds its system and while the integral
    bindings (integrals.rs, `AoIntegrals`, response_modes.rs) prepare
    their bases; (f) the periodic paths (`ferric-cli/src/periodic.rs`'s
    `apply_ecp`, the Python `pbc.rs` drivers) keep their own preparation,
    their molecule going into a `Cell`.
    Review follow-ups:
    - One public entry per kernel: either the loose kernels become
      `pub(crate)` (their tests move to the `*_on` entries) or the entries
      take the plain names. Settle one naming convention (`_on` means "on
      `aux`'s system" in `ri_mp2_on` / `run_gw_on` but "given a reference"
      in `opmp3_on_reference` / `casscf_on_rhf`; the `run_` prefix is
      mixed; e.g. add `run_pdep_rpa_fitted_on`), and whether an entry takes
      an operator (terfc RI-MP2 cannot use them today).
    - `PreparedSystem` should keep its orbital `BasisSet` (`basis_set()`),
      closing the loose `obs_bs` of `local_opmp3_on_reference` /
      `run_local_opmp3` and the `bs` argument of ~40 CLI functions;
      consider an `ArmCtx` for the CLI arms' shared tail.
    - An `ScfResult` is not tied to its system: `ri_mp2_on(&a_aux, &b_rhf,
      ..)` compiles, and with another nbasis it can panic slicing
      `mos_r()`. The `*_on` / `*_on_rhf` entries should check nbasis and
      geometry (or results should carry a brand).
    - The system's solve methods should take a config type without
      `screening` (parse, don't validate); the refusal then goes.
    - Exit condition for `PreparedAux: Deref`: `*_on` entries (or
      `&PreparedAux` parameters) for the kernels listed in (a), then delete
      `impl Deref` so any remaining `&aux` site fails to compile.
    - Python sites not covered: the per-geometry loops of `run_saddle` /
      `run_irc` (lib.rs) and geometry.rs's own `compute_for_screening` from
      `cfg.screening` (a second home for the screening rule);
      `hirshfeld_polarizability` (properties.rs) and `blocks.rs`'s
      `aux_prep`, which pair a loose fitting basis with an orbital basis.
      integrals.rs's shell info, properties.rs's orbital and second
      moments and one lib.rs helper prepare on the raw molecule on purpose:
      they are purely geometric.
    - ~20 copies in ferric-cli of "section `auxbasis` or
      `DEFAULT_CORRELATION_AUX`, `bundled`, `sys.prepare_aux`":
      `correlation_aux` could take the section's `Option<&str>`.
    - Pre-existing panics (same on main), to be filed separately:
      `density_second_moment` / `orbital_moments` given a result from
      another molecule raise `PanicException` (ndarray shape);
      `hirshfeld_charges(HI, def2-svp, rhf)` panics in `UhfState::new`
      ("end <= axis_len") in the iodine proatom SCF.
9b. A `MethodResult` trait (`total_energy`, `certificate`, `timings`) and a
   declarative macro for the shared Python result getters.
10. A Python exception hierarchy rooted in today's classes.
11. `RhfConfig` in steps, each with a field-by-field oracle test:
    (a) route every literal site through an options lowering, then mark it
    `#[non_exhaustive]`; (b) group fields into sub-configs (`Convergence`,
    `Diis`, `JkRoute`, `Guess`, `Mom`, `Stability`, `Occupation`,
    `Option<KsConfig>`, `Option<CdftConfig>`); (c) replace correlated
    bools and sentinels with sum types (`KMethod { Exact, Ri, Cosx }`,
    `Stability { Off, Check, Descend }`, `LevelShift { Auto, Fixed }`,
    `Mom { Off, After(n), Initial }`, `Guess { Core, Sad, Projected,
    Density }`); (d) move `eri_route` (solver output written through
    `&RhfConfig`) into `ScfResult`, so the config can derive `PartialEq`;
    (e) one `RhfConfig::validate` called at solver entry. Cross-field rules
    then return a typed `OptionError`.
12. CLI run functions return `Result<_, CliError>`; `process::exit` only in
    the binaries' `main`, since Python calls `ferric_cli::run`.
13. Python results store `Arc<ScfResult>` so passing a reference into a
    correlated method does not deep-copy arrays; result classes become
    `frozen`; `ferric.serialize` reads an explicit `MethodResult` schema
    instead of reflection.
14. A shared `ferric-testkit` dev crate, file splits, and a clippy ratchet on
    `too_many_arguments`.
15. (rest) Still stringly or boolean at the boundary, after the first
    part above:
    - `Option<Symmetry { irrep_nelec }>` instead of `symmetry` +
      `irrep_nelec` in `ScfKwargs` / `ScfOptions` / `RhfConfig`. Not done
      because `run_rhf` refuses `irrep_nelec` without `symmetry` at three
      points of its solve path, each reading `RhfConfig` (the
      `with_gradient` refusal in `scf_gradient_config`, the hooked solve's
      `validate_fock_hook`, and `solve_rhf_outcome` after the noise-floor
      warning, each with its own message), while `run_dft` refuses it in
      the binding. Making it unrepresentable after parsing either changes
      which message such a call gets, or moves those three refusals into
      the binding against a parse-time marker; it belongs with 11(b)/(c),
      where `RhfConfig` itself drops the pair.
    - A `Functional` newtype for `xc` (`ScfKwargs::functional`,
      `ScfOptions::xc`, `RhfConfig::xc`). Not contained: the name is
      validated by libxc in the solver (and by `resolve_dispersion` /
      the ladder's grid choice before it), so validating it at the
      boundary moves the unknown-functional error ahead of the dispersion,
      symmetry and gradient refusals, as `KBuilderKind` did for
      `k_builder`, and `RhfConfig::xc` is read as a string across the
      workspace (KS gradients, TDDFT, GW, the CLI's per-kind defaults).
    - One solver-capability table shared by `ScfEntryPoint::refused` and
      the CLI's `validate_open_shell_scf_keys` / `validate_accelerators`.
      Not done: only the open-shell `diis` / `smearing_sigma` refusals are
      solver capabilities on both sides; the CLI's `aurora` and
      `trah_trigger` gates are by `method.kind` (refusing `ksdft`, which
      `run_dft` accepts), so sharing the decision changes CLI behaviour,
      and the messages differ word for word (Python names the entry point
      and the solver field, the CLI the `[scf]` key and the multiplicity).
    - `ScfKwargs` fields grouped (convergence, J/K route, guess,
      environment, accelerators, Kohn-Sham): mechanical, deferred to keep
      this PR reviewable; the oracle tests' `kwargs()` mirror moves with it.
