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
- **Python does not reach into ferric-cli** except through
  `ferric_cli::try_run` (the bundled `ferric` console script, `_cli_main`),
  which turns a returned `CliError` into the CLI's stderr text and
  `SystemExit(status)`, as a command should. Nothing in ferric-cli exits
  below `ferric_cli::run`, the native binaries' entry point.
  Existing uses (`LocalCfg`, `LocalModel`, `LocalScheme`, `Mp2Cfg`) are slated
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
- **CLI:** run functions return `Result<_, CliError>`
  (`ferric-cli/src/error.rs`) and propagate with `?`; `std::process::exit`
  appears only in `ferric_cli::run` (and `ferric-batch`'s `main`). Each
  `CliError` variant fixes the stderr bytes and the exit status:
  `Ferric`/`Msg` print `error: {msg}` (status 1), `Verbatim` prints its
  text as written (a message that does not start with `error: `, or a
  multi-line one; status 1), `Usage` prints its text as written with status
  2. A new error site builds the variant whose output it wants
  (`CliError::msg` / `verbatim` / `usage`; there is no `From<String>`, so a
  site cannot get a prefix by accident) and never prints the error itself.
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

- **Traits are small and segregated.** `MethodResult` (total energy,
  certificate) and `Reference` (spin, orbitals, eigenvalues, occupations) stay
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
3. `CliError` and `OrExit` in ferric-cli (`OrExit` removed again by 12).
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
9b. `MethodResult` (`total_energy`, `certificate` as a `Cow<Certificate>`),
    `pub(crate)` in `ferric-python/src/method_result.rs`, with the helpers
    `total_energy_getter` and `certificate_getter`. Each result class's
    shared getters (`certificate`, `timings`, the `energy`/`total_energy`
    alias, whose docstring comes from `alias_doc!`) are one-line `#[getter]`s
    in its own `#[pymethods]` block over those helpers (or over
    `certificate_dict` / `pbc::timings_dict` of a field). Names, docstrings,
    values and certificate keys are unchanged, and
    `tests/test_result_getters.py` pins the wiring.
    - Its only consumers are those helpers. It lives in ferric-python
      because every implementer is a ferric-python result class (see the
      `stored_method_result!`, `periodic_method_result!` and hand-written
      `impl MethodResult` items); it moves to ferric-core when a kernel or
      CLI result implements it or generic code consumes it. No kernel result
      can today: none carries a certificate, which is built from the result
      plus inputs it does not keep.
    - Left out deliberately: results with no single total energy (CC,
      EOM-CCSD, GW, BSE, TDDFT, OP-MP3, CDFT coupling, MD); task results
      (optimize, saddle, frequencies, QM/MM optimize), which have a total
      energy and a certificate but grade the task, not a method; QM/MM,
      whose `energy` is the embedded QM energy; `SymmetricRhfResult`, which
      has no certificate. These read a stored certificate field.
      `timings` is not in the trait: only periodic results carry it.
    - Not a generating macro: pyo3 0.23 rejects a macro item inside a
      `#[pymethods]` impl, so a macro could only emit a second
      `#[pymethods]` block, which needs pyo3's `multiple-pymethods` feature
      (the `inventory` crate). That was tried and dropped: inventory's
      link-section constructors keep every `#[pymethods]` block alive, so
      the ferric-python lib test binary needs libpython symbols that
      dead-stripping used to remove, and an extension-module build does not
      link libpython; `cargo test -p ferric-python` failed to link.
    - It does not provide the explicit serialize schema item 13 needs: that
      schema is Python-side.
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
      `...Kind`: `fock::KBuilder` is the trait). It lives in `options`
      because the frontends parse into it there, next to the other
      boundary types; it may move to `fock` / `fock_assembly` with 11(b),
      when the J/K route becomes a sub-config. `RhfConfig::k_builder` is
      `Option<KBuilderKind>`; `KBuilderKind::parse` (also `FromStr`) is the
      one whitelist, refusing with the solvers' old text
      (`FerricError::General`, so Python still raises
      `RuntimeError("General error: unknown k_builder ...")`).
      `COSX_K_BUILDER` and the solver-side check are gone;
      `df_k_aux_or_default`, the J/K route, the certificate, `EffectiveJk`
      and the COSX schedule take the enum. `ScfJk` no longer copies the
      builder (`ScfJk::label` takes the config's). The checkpoint identity
      keeps the old `Debug` text (`Some("cosx")`).
    - Where it is parsed: the six SCF entry points last of their keywords;
      the correlated entry points after their keyword checks and their
      system / fitting-basis preparation (`rhf_config_budgeted` /
      `rhf_config_conv`; the CC drivers that prepare inside a GIL-released
      closure through `with_k_builder` there), except the MP2 family,
      which parses it in `mp2_reference_config` before its method
      keywords; `run_cdft` before its config; the CLI in `try_scf_config`
      as its last fallible step (`ScfCfg::k_builder_kind`), after loading
      and the run's validation, naming the file in the form of the
      load-time errors (`error: <file>: [scf] k_builder: unknown k_builder
      ...`).
    - Re-reads of the raw spelling, through the same whitelist
      (`KBuilderKind::spells_cosx`), where a message quotes what was typed
      and runs before the parse: the CLI load-time COSX checks
      (`ScfCfg::k_builder_is_cosx`), `resolve_cosx_kwargs`, the PDEP-RPA
      gradient scope and `ScfJk::resolve`'s conflicts. They go when that
      message constraint does.
    - Behaviour change, for an unknown spelling only: same class and text
      (CLI: the file-naming form above), but raised at the boundary rather than
      by the solver. Checks that ran between the two now run after it: the
      pre-SCF noise-floor warning and `run_dft`'s ladder banner (no longer
      printed), the `with_gradient` / hook / symmetry refusals and the
      libxc functional lookup, the MP2 family's method keywords, and in the
      CLI the system preparation, the banners and run-log header (and the
      per-kind "SCF ladder failed: General(...)" / "General error: ..."
      wrappers). So with two bad inputs the reported
      error can switch, including from a `ValueError` to the
      `RuntimeError`; with one, nothing changes.
      `tests/test_k_builder_parse_order.py` pins both sides of the line.
    - `options::ks_jk_fit_defaults`: the one home of the Kohn-Sham JK-fit
      default of the Python KS entry points (`ScfEntryPoint::options`) and
      `ksdft_ladder` (which had its own copy), with the RIJCOSX rule. The
      CLI's per-kind defaults (`Config::scf_xc_and_aux_defaults`) are
      deliberately separate and share only the RIJCOSX rule
      (`df_k_aux_or_default`).
    - `ri_j_task` is the argument of `ScfOptions::into_rhf_config`, not an
      option.
    - `ferric-python`: `parse_fock_hook` returns one
      `Option<FockHook { fock, grad }>` (same messages and order), carried
      by the open-shell entry (`OpenShellEntry`, `run_open_ks`).
    - `ferric.pyi`: the `run_dft` / `run_ksdft` `symmetry=True` overloads
      type `with_gradient` as `Literal[False]` and `dispersion` as `None`.

12. CLI run functions return `Result<_, CliError>`. `ferric_cli::try_run`
    is the whole CLI in-process: it returns `Ok(())` for a finished run and
    for `--help`/`--version` (text already printed), and every failure as a
    not-yet-printed `CliError`. `ferric_cli::run` (the `ferric`/`ferric-cli`
    binaries) does process init (`init_threading`), calls it, and on an
    error prints it (`CliError::eprint`) and `process::exit`s with
    `exit_code()`; the caller of `try_run` owns process init. Python's
    `_cli_main` (the `ferric` console script; `init_threading` already ran
    at import) prints the error the same way and raises
    `SystemExit(status)`, so the status, stderr and absence of a traceback
    are unchanged but Python shuts down normally (atexit, stream flush;
    `tests/test_cli_main_exit.py`). The value is maintainability and a
    testable CLI (`tests/try_run_in_process.rs`), not a Python crash fix:
    exiting with the status is the console script's correct behaviour.
    - Output is byte-identical: the variants (`Msg`/`Ferric` `error: ...`,
      `Verbatim` as written, `Usage` as written with status 2) reproduce each old
      `eprintln!` + `exit`, multi-line messages (the exact-dRPA memory hint,
      the incomplete-NPZ list) are one `Verbatim`, and nothing runs between an
      error site and the top level, so warnings printed before an error
      still come first. Checked against an origin/main build on 112
      invocations, 90 of them failing (81 with status 1, 9 with status 2:
      usage, config loading, the validation families, ECP, periodic, md,
      thermo/IR, frequencies, optimize, per-arm refusals) and 22 exiting 0
      (`--help`/`--version` and small runs): stdout, stderr, status and
      JSON run logs equal (masks: build identity, the periodic stage-table
      timings, timing keys and record order in the log).
    - The JSON run log is unchanged: its sink is a process static written
      and flushed per record, so returning instead of exiting writes the
      same records (no `run_end` on an error, as before). Returning now runs
      the destructors `exit` skipped (the prepared system, basis handles,
      memory-pool reservations, MD writers); none prints or writes (the MD
      writers flush every frame already).
    - Closures that used to exit became fallible (`.transpose()`, `Ok`
      closures) without moving any check. Two helpers changed shape for
      it: `task_reference_certificate` returns `Result<RefCert, CliError>`
      (`RefCert::Stamped`, or `RefCert::Ungraded` with the warning for a
      failed final-point SCF; `Err`: a dispersion error, which ends the
      run), and `frequencies_maybe_dispersion` wraps its own driver
      errors as `error computing frequencies: ...` so its dispersion errors
      keep their text. One internal-invariant check moved: the periodic
      Gamma-point correlation's `[cell]` denominators lookup (never absent
      after `load_config`) now runs before that route's SCF instead of
      inside its driver.
    - `OrExit` and `die` are gone; no exit is left inside a closure, rayon
      task or handler. `ferric-batch` exits only in `main` (over
      `try_main`); one failing molecule still does not stop the batch,
      which exits 1 at the end, as before. A per-molecule TOML that cannot
      be written is now a setup error (`Failed to write <path>: ...`,
      status 1) instead of a panic. `ferric-batch` does not call
      `init_threading`: it runs no calculation, only child `ferric`
      processes, which initialise themselves.
    - `run`'s dispatch moved into `try_run` with two helpers
      (`parse_args`, `load_and_install`), keeping it under the complexity
      gate's new-function ceiling (CC 131; `run` was 127); the baseline
      was regenerated for the arms that gained `?` branches.
    Follow-ups: `try_run` still installs process-wide, first-call-wins
    state (run-log sink, memory pool, GPU and integral settings), so a
    second in-process run past config loading is not isolated; many sites
    flatten typed errors into `Msg` strings (`CliError::msg`), which a
    later pass could keep as `Ferric`, starting with cas.rs, opmp3.rs,
    periodic.rs and cdft.rs, whose driver errors are all `FerricError`;
    `CliError` construction should move out of `config.rs`
    (`OptimizeCfg::search_options`) and `md.rs` (`md_error`) into their
    callers, so the config layer returns plain errors; `ferric-batch`
    could report a child's error class from its exit status (2 vs 1).
    Done since: `--help` / `--version` are written by
    `info_text::write_info` instead of `eprintln!` / `print!`, same bytes;
    a broken pipe is success (status 0), any other write error a
    `CliError::Msg` (status 1) instead of a panic (101), and
    `CliError::eprint` drops a failed stderr write instead of panicking.
13. Result storage, bitwise unchanged (old-vs-new over the 15
    `result=`-taking functions and the array getters, on two RHF and two
    KS references: 84 values):
    - Survey: no Python API passes a result into a correlated method (those
      run their own SCF), and the functions that take `result=`
      (`DensitySource` in properties.rs, `response_modes`, `cdft_coupling`)
      already borrowed it through a `PyRef`; nothing deep-copied an
      `ScfResult`. The copies were smaller: `RhfResult` / `DftResult` kept
      the density (and `RhfResult` the orbital energies) twice, beside the
      `ScfResult` they came from, and `hirshfeld_charges` cloned the density
      and `RhfConfig` to release the GIL.
    - `RhfResult` / `DftResult` hold `scf_data: Arc<ScfResult>` and
      `scf_config: Arc<RhfConfig>`; `density()` / `orbital_energies()` read
      the `ScfResult`, and `HirshfeldJob` holds clones of the two `Arc`s
      (`DensitySource::shared`, a `SharedReference`). `SymmetricRhfResult`
      keeps no `ScfResult`; its orbital energies are moved out of the solve
      rather than cloned. Kernels still take `&ScfResult`. Getters
      still return fresh, writable numpy copies.
    - Every result class (and `ResponseModes`, `BoysResult`, `IrcBranch`,
      the conformer statistics classes) is `#[pyclass(frozen)]`.
      `FrequencyResult.ir_intensities`, the one `&mut self` result method,
      and `DftResult`'s XC cache use `compute_once::ComputeOnce<T>` (a
      `OnceLock` set only under its `Mutex`, `get_or_try_init` called inside
      `allow_threads` with no Python calls in the closure), so concurrent
      first calls compute once instead of raising
      `RuntimeError("Already borrowed")`. Errors are not cached (as on
      main): after a failed first `ir_intensities`, each waiting thread
      re-runs the six field SCFs in turn. Assignment to a getter still
      raises the same `AttributeError` ("not writable"). Not frozen:
      `ConformerEnsemble` (`set_energy` mutates it) and the `blocks`
      handles `DfJk` and `XcFunctional` (`&mut self` methods). Done since:
      the inputs `BasisSet`, `QmmmSystem`, `MmTopology`, `CdftConstraint`
      and the handles `AoIntegrals`, `Cphf`, `Ucphf` (no setter, `&mut
      self` or `PyRefMut` use) are frozen too (`tests/test_frozen_inputs.py`).
    - `ferric.serialize.RESULT_SCHEMA` (class name -> getters and array
      accessors) replaces the `dir()` scan for ferric classes, generated
      from the reflected field set so the output is byte-identical
      (`tests/test_serialize_schema.py` compares against the old rule kept
      as an oracle, with key orders recorded from the old build in
      `tests/data/serialize_keys.json`). It is not free-form: the drift test
      keeps each entry equal to its class's public getters minus that
      class's names in `ferric.serialize.NOT_SERIALIZED`, the library's
      exclusion set (empty), which is where a getter is kept out on purpose.
      The value filter is unchanged, so list- and dict-valued fields in the
      schema (`frequencies`, `normal_modes`, `weights`, `dm1`...) are never
      emitted, an optional dict (`cosx_final_pass`...) appears only when it
      is `None`, and `DftResult`'s `e_xc` is still evaluated
      (a grid build) by `to_dict`. A ferric class with no entry raises
      `TypeError`; objects ferric does not define (and ferric exceptions)
      are still read by reflection.
    - Measured (naphthalene RHF/cc-pVTZ, 412 basis functions, a 1.36 MB
      density): `hirshfeld_charges(proatom="slater")`, best of 5, 4.09 s
      before and 4.06 s after, peak RSS rising 36 kB and 0 kB over the
      calls; at cc-pVDZ 2.73 s and 2.48 s (a loaded machine, so within
      noise). The removed copies are one `nbf x nbf` f64 array per held
      `RhfResult` / `DftResult` (plus the orbital energies for
      `RhfResult`) and one per `hirshfeld_charges` call: memory, not time.
    - Left: `UhfResult` (and `CdftResult`) copy their arrays out of the
      `ScfResult` into separate fields; nothing consumes them as a
      reference, so they were not changed. Results are not picklable and
      do not support `copy.deepcopy`, before and after (no `__reduce__`).
      `to_dict` should stop evaluating `e_xc` (drop it from the schema) and
      decide what to do with the list/dict fields: visible output changes
      left for their own PR. Done since: ESP (`esp_at_atoms`,
      `esp_at_points`, `esp_on_surface`), CHELPG, RESP, Löwdin, Mulliken
      and `write_cube` prepare and compute with the GIL released
      (`on_density_without_gil`, through `DensitySource::shared`), bitwise
      unchanged (old-vs-new hex floats and file hashes over 12 cases, three
      molecules x two bases x RHF/B3LYP: 108 outputs, 24 of them cube
      files). Still holding it: `orbital_moments`,
      `density_second_moment`, the `eval_*_on_points` functions and
      `hirshfeld_polarizability`.

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
10. A Python exception hierarchy rooted in today's classes.
11. `RhfConfig` in steps, each with a field-by-field oracle test:
    (a) route every literal site through an options lowering, then mark it
    `#[non_exhaustive]`; (b) group fields into sub-configs (`Convergence`,
    `Diis`, `JkRoute` (holding `Option<KBuilderKind>` with `df_j_aux` /
    `df_k_aux`, the request), `Guess`, `Mom`, `Stability`, `Occupation`,
    `Option<KsConfig>`, `Option<CdftConfig>`); (c) replace correlated
    bools and sentinels with sum types (an effective-route
    `KMethod { Exact, Ri, Cosx }` naming what the solver builds, as
    `fock_assembly::JkRoute` / `EffectiveJk` do today, not a second
    request type next to `KBuilderKind`,
    `Stability { Off, Check, Descend }`, `LevelShift { Auto, Fixed }`,
    `Mom { Off, After(n), Initial }`, `Guess { Core, Sad, Projected,
    Density }`); (d) move `eri_route` (solver output written through
    `&RhfConfig`) into `ScfResult`, so the config can derive `PartialEq`;
    (e) one `RhfConfig::validate` called at solver entry. Cross-field rules
    then return a typed `OptionError`.
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
      this PR reviewable; the groups should take 11(b)'s sub-config names,
      and the oracle tests' `kwargs()` mirror moves with it.
    - Design sketch for the symmetry pair: `enum SymmetryRequest { Off,
      On { irrep_nelec } }` in `ScfKwargs` / `ScfOptions` (later
      `RhfConfig`); the lowering returns a typed
      `Refusal::IrrepNelecWithoutSymmetry`, which each entry point renders
      with its own text (`run_dft: irrep_nelec requires symmetry=True`, ...);
      `run_rhf`'s three refusals fold into one `validate` at the point the
      first of them runs today.
    - The "overridden" rule (a pluggable builder is skipped when `k_aux`
      is set, or `j_aux` is set and the builder is not COSX) is still
      written in three places: `scf_jk::effective_jk`,
      `JkRoute::pluggable_k` and `fock_assembly::resolve_k_builder`.
    - The `ferric.pyi` `symmetry=True` overloads still admit keywords the
      blocked solve refuses (`point_charges`, `external_field`,
      `solvent`, `smeared_charges`, `k_builder="cosx"`, `aurora`,
      `stability_descent`); typing them needs an overload per refusal or
      `None`-typed parameters, checked against the binding.
