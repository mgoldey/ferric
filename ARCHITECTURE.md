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
- `RhfConfig` is `#[non_exhaustive]`: outside ferric-scf there is no struct
  literal. Build it with `ScfOptions::into_rhf_config`, a named constructor
  (`RhfConfig::periodic`, `RhfConfig::gradient_reference`) or
  `RhfConfig::default()` plus the `with_*` setters / field assignments.

## Function signatures

- Order: context first (`ParallelContext`, prepared basis, bounds), then
  inputs, then `&Config` last.
- More than seven parameters means an arguments struct. Clippy enforces it
  (`too_many_arguments` is no longer allowed workspace-wide); an existing
  offender carries an item-level allow, and
  `scripts/too_many_arguments_ratchet.py` only lets that count go down.
- Test helpers copied into three or more test files go in `ferric-testkit`
  (a dev-dependency only; see roadmap 14a for what cannot use it).
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
  extension trait (`.py()?`) or `make_err`, not a fresh `map_err` per call
  site. A blanket `From<FerricError> for PyErr` is not possible (orphan
  rule). Both end in `errors::from_core`, the one place that decides the
  exception class (`ferric-python/src/errors.rs`). Exception classes only
  ever become narrower subclasses of the class raised today; they are never
  swapped. Every class derives from `ferric.FerricError(Exception)` and from
  the builtin it raised before, so `except RuntimeError` / `ValueError` /
  `NotImplementedError` handlers are unchanged:

  | class | also | raised for |
  |---|---|---|
  | `ConvergenceError` | `RuntimeError` | `ScfConvergence`, `Convergence` (typed only; also the orbital-optimisation gradient refusal) |
  | `BasisError` | `InputError`, `RuntimeError` | `Basis` (unknown bundled name, unparsable or non-covering basis/ECP); both builtins because `BasisSet.bundled` / `n_functions` raised `ValueError` and the `run_*` drivers `RuntimeError` |
  | `NumericalError` | `RuntimeError` | `Lapack` |
  | `MemoryBudgetError` | `RuntimeError` | `MemoryExhausted` (typed only: `ferric-core`'s `check_alloc` returns `General`, so it is a `ComputationError`) |
  | `UnsupportedError` | `NotImplementedError` | `Unsupported` |
  | `ComputationError` | `RuntimeError` | every other `FerricError` (`General`, `Io`, `Libint`, `XyzParse`) and binding-level runtime failures |
  | `InputError` | `ValueError` | binding-level invalid arguments (`errors::input_err`); a core error passed to it is a `BasisError` if it is `Basis`, else an `InputError` whatever its variant (these sites raised `ValueError`) |
  | periodic drivers | `ValueError` | `pbc.rs::refusal_text` maps `General` and `MemoryExhausted` to `InputError` (`ValueError`, as before) and everything else through `from_core` with the `run_x:` prefix |

  `General` still carries invalid-configuration text (it is a string
  variant), so those stay a `ComputationError`; only a typed variant can be
  told apart reliably. `TypeError`/`KeyError`/`OSError` raised by argument
  extraction or the file helpers stay plain, as does anything raised from
  pure Python. The classes are built with `type()` at module init (pyo3's
  `create_exception!` takes one base), carry `__module__ = "ferric"` and are
  module attributes, so they pickle. Messages are unchanged.

  The decision is `FerricError::category()` (`ErrorCategory`, an exhaustive
  match in ferric-core) mapped by `errors::kind_of`, so a new variant cannot
  be added without choosing its class; `FerricError::samples` (one value per
  variant) feeds the `ferric._raise_core_error` test hook and a unit test.
  `tests/test_exception_sites_are_typed.py` fails on a new bare
  `PyValueError::new_err` / `PyRuntimeError::new_err` outside a short
  allowlist (the GPU/precision settings, `set_blas_threads`, a libint test
  hook).

  **Which variant to return (method authors):** return `Convergence` /
  `ScfConvergence`, `MemoryExhausted`, `Unsupported` (`FerricError::unsupported`)
  or `Basis` for those failures, never `General` with a descriptive string:
  only the typed variant reaches Python as `ConvergenceError`,
  `MemoryBudgetError`, `UnsupportedError` or `BasisError`. `General` ends up
  as `ComputationError`.

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
  frontends reach the correlated kernels only through `*_on` entry points
  taking `&PreparedAux` or `&PreparedSystem`, which also check the
  reference SCF result against the system (item 8d; `PreparedAux` no
  longer dereferences to its fitting basis). The loose kernels under them
  (`&Molecule`, `&PreparedBasis`, ...) stay public for tests, other crates
  and the per-geometry gradient paths.
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
10. The Python exception hierarchy described under Errors (`FerricError`
    root; `ConvergenceError`, `BasisError`, `NumericalError`,
    `MemoryBudgetError`, `UnsupportedError`, `ComputationError`,
    `InputError`), all also subclasses of the builtin they raised before.
    Follow-ups: `General` cannot be split further until invalid-option
    failures get a typed variant (item 11e's `OptionError`); `check_alloc`
    in `ferric-core/src/memory.rs` returns `General` and should return
    `MemoryExhausted` (it changes the message prefix, so it needs its own
    PR). `BasisSet.n_functions` raises `BasisError`; `from_bse_json` raises
    `BasisError` for malformed JSON and `InputError` for an unreadable file.
    `ferric.qcschema.run` now classifies by these classes (a driver basis
    failure is `input_error`, not `execution_error`).
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
      casscf_on_rhf, casci_on_rhf}` (`&PreparedSystem`; renamed
      `nevpt2_on` / `casscf_on` / `casci_on` in 8d);
      `opmp3_driver::{opmp3_scf, opmp3_on_reference, run_opmp3}`
      (`&PreparedSystem`, `Opmp3Integrals::Ri(&PreparedAux)`;
      `opmp3_on_reference` renamed `opmp3_on` in 8d);
      `local_opmp3::{local_opmp3_on_reference, run_local_opmp3}` (renamed
      `local_opmp3_on` in 8d),
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

8d. (first part) The correlated kernels behind `*_on` entry points,
    results and messages bitwise unchanged (compared against an origin/main
    build: 172 CLI inputs, the repository examples plus error and new
    local / open-shell / NPZ-export cases, identical stdout, stderr and
    exit status after masking timings; 152 Python calls, energies and
    gradients as hex floats and arrays by hash, identical except timing
    fields, errors identical in class and message):
    - `PreparedSystem::check_reference(&ScfResult)` (and
      `PreparedAux::check_reference`, the same on `aux.system()`): the one
      check that a reference SCF result fits the system, a
      `FerricError::General` ("the reference SCF result does not belong to
      this system: ...") naming what does not fit: the β fields match
      `ScfResult::spin` (`Restricted`: no β MOs or orbital energies;
      `Unrestricted`: both; `RestrictedOpen`: both or neither), the total
      density is `nbasis × nbasis`, each spin's MO coefficients are
      `nbasis × nbasis` (every solver, linear-dependency removal included,
      pads the MO space to the basis size, and the fixed-width kernels size
      their virtual space from `nbasis`) with one orbital energy per MO,
      and the occupied orbitals of the molecule's electron count and
      multiplicity fit. Two narrower checks sit on top of it:
      `check_restricted_reference` (the closed-shell entries whose kernels
      read `ScfResult::mos_r` and the other restricted accessors, which
      assert the spin: the restricted `ri_mp2*`, MP3, OO-MP2, the double
      hybrid, ω-RI-CDD, Laplace, the RI-MP2 gradient, the amplitude
      LMP2 / dRPA / LinLCCD, CCD, LinLCCD, PDEP-RPA and the RPA gradient)
      and `check_virtual_orbitals` (the entries whose kernels read the
      lowest virtual orbital: ω-RI-CDD, Laplace, PDEP-RPA, RS-MP2-RPA, and
      for closed-shell references GW, BSE-TDA and RPAx). The entries that
      already refused another spin with their own message (attenuated
      MP2-V, CCSD, GW, BSE, RPAx) keep it. ferric-cli's
      `on_entries_never_panic` sweeps every reference-taking entry over
      UHF, ROHF, rectangular-MO, He / STO-3G and H-atom UHF references
      under `catch_unwind`. `PreparedSystem::check_closed_shell_orbitals(mos, eps)` is the
      per-spin part for orbitals that are not an `ScfResult` ("the
      orbitals do not belong ..."). Every `*_on` entry that takes a
      reference calls one of them first: the new ones below (with
      `oo_ri_mp2_gradient_on` checking `OoRiMp2Result::{mos,
      orbital_energies}`) and `ri_mp2_fitted_on` (so `ri_mp2_on`),
      `u_ri_mp2_on`, `ccsd_closed_shell_on`, `u_ccsd_on`, `run_gw_on`,
      `run_pdep_rpa_on`, `run_tddft_on`, `opmp3_on`, `local_opmp3_on`,
      `casscf_on`, `casci_on`, `nevpt2_on`. The two that take none,
      `aux_route_charge_bytes_on` (a budget estimate) and
      `run_wb97x_l_v_on` (it solves its own reference), check nothing.
      `compute_pdep_properties` checks too and returns a typed
      `PropertyBundleError { Config(String), Reference(FerricError) }`: a
      mismatch is a run error (CLI `error: ...`, Python `RuntimeError`),
      while the `c6_source` / `c6_partition` errors keep their text and
      class (CLI `config error: [rpa] ...`, Python `ValueError`). So
      `ri_mp2_on(&a_aux, &b_rhf, ..)` with another basis size is an error
      where the loose kernel panics slicing the MOs (the
      `on_entries_refuse_a_reference_of_another_system` tests in ferric-mp2,
      ferric-cc, ferric-gw, ferric-rpa, ferric-ci and ferric-tddft, the
      mp2 one pinning that the loose `ri_mp2` panics on the same pair, and
      ferric-rpa's `a_reference_of_another_system_is_a_reference_error`
      for the property bundle). An
      `ScfResult` carries no molecule, so a reference of another molecule
      or geometry in a basis of the same size passes. The check cannot be
      reached from Python or the CLI: every binding and CLI arm solves its
      reference on the system it then passes, and no Python driver takes a
      result object as its reference; so no Python test.
    - New entries, each `<kernel>_on(aux, [op,] reference, ..)` next to its
      kernel: ferric-mp2 `amplitude_lmp2_on`, `amplitude_lmp2_direct_on`,
      `mp3_energy_on`, `oo_ri_mp2_on`, `u_oo_ri_mp2_on`,
      `oo_ri_mp2_gradient_on`, `attenuated_ri_mp2_on`, `ri_mp2_with_op_on`,
      `u_ri_mp2_with_op_on`, `ri_mp2_spin_components_any_on`,
      `ri_mp2_spin_components_fitted_on`, `scs_mp2_on`,
      `scs_mp2_2terfc_on`, `att_mp2_vv10_on`, `u_att_mp2_vv10_on`,
      `att_mp2_vv10_scan_on`, `mp2_double_hybrid_on`,
      `u_mp2_double_hybrid_on`, `omega_ri_cdd_mp2_on`, `laplace_ri_mp2_on`,
      `laplace_sos_mp2_on`, `rimp2_gradient_analytical_on`,
      `amplitude_drpa_on`, `amplitude_drpa_scan_timed_on`; ferric-cc
      `ccd_on`, `linlccd_on`, `amplitude_linlccd_on`,
      `ccsd_t_closed_shell_on`, `u_ccsd_t_on`, `eom_ccsd_singlets_on`,
      `run_wb97x_l_v_on`; ferric-rpa `run_pdep_rpa_fitted_on`,
      `run_u_pdep_rpa_fitted_on`, `run_pdep_rpa_energy_on`,
      `aux_route_charge_bytes_on`, `rs_mp2_lr_rpa_on`,
      `rs_mp2_lr_rpa_scan_on`, `rpa_analytic_gradient_on`; ferric-gw
      `run_u_gw_on`, `run_bse_tda_on`, `run_rpax_static_polarizability_on`.
      `property_bundle::PropertyInputs` takes `aux` in place of its loose
      context, molecule, bases and operator.
    - The CLI's correlated arms (`ferric-cli/src/lib.rs`, with `RsScanCtx`
      carrying the `PreparedAux`) and the Python drivers (mp2.rs, cc.rs,
      gw.rs, rpa.rs, dft.rs, omega_ri.rs, pdep_properties.rs' `BundleJob`)
      call them, and `impl Deref for PreparedAux` is gone: a loose fitting
      kernel is reached through `PreparedAux::basis()` only.
    - Naming: `<kernel>_on` takes the `&PreparedAux` (or, for the CAS and
      OP-MP3 drivers that need no fitting basis, the `&PreparedSystem`)
      first and reads the molecule, orbital basis, its basis set, bounds,
      context and Coulomb operator from the system; the reference result
      is an argument and is checked; arguments after it keep the kernel's
      order, except that `run_u_gw_on` takes `vxc_diag` right after the
      reference, as `run_gw_on` does. An entry takes an `op` (right after
      `aux`) only where a caller runs the kernel on another correlation
      operator: `amplitude_lmp2_on` / `amplitude_lmp2_direct_on` (the
      attenuated local MP2), `ri_mp2_spin_components_any_on`, and
      `ri_mp2_with_op_on` / `u_ri_mp2_with_op_on` (terfc RI-MP2). The
      rule: an op-taking entry ends `_with_op_on` exactly when a Coulomb
      `_on` entry of the same kernel exists (`ri_mp2_on`, `u_ri_mp2_on`),
      and plain `_on` otherwise. The entry is the kernel name plus `_on`,
      `run_` prefix and all, except `laplace_sos_mp2_on` for
      `laplace_sos_mp2_with_basis` (the basis set it adds comes from the
      system). Renamed to fit:
      `casscf_on_rhf`, `casci_on_rhf`, `nevpt2_on_rhf` → `casscf_on`,
      `casci_on`, `nevpt2_on`; `opmp3_on_reference` → `opmp3_on`;
      `local_opmp3_on_reference(obs_bs, scf, aux, ..)` →
      `local_opmp3_on(aux, scf, ..)` and `run_local_opmp3` without
      `obs_bs`; the loose `rpa_analytic_gradient_on(mol, obs, dfbs, op,
      bounds, ..)` → `rpa_analytic_gradient_from_rhf`, the name now being
      the entry. No Python name or signature changes.
    - `PreparedSystem::basis_set()`: the system keeps a copy (a clone at
      `prepare`; an `Arc<BasisSet>` would need callers to hand one in, as
      `EcpMolecule` borrows the set) of the `BasisSet` it was prepared from, which the entries of the kernels that
      read shells by element (the amplitude-threshold LMP2 / LinLCCD /
      dRPA, MP2-V, Laplace SOS-MP2, local OP-MP3, the property bundle) read
      instead of a loose `obs_bs`.
    - Not done here: the loose kernels stay `pub` (integration tests,
      benches, ferric-rpa / ferric-gw calling ferric-mp2 / ferric-rpa
      kernels, and the per-geometry gradient paths use them; moving those
      tests onto `*_on` entries would roughly double this diff), and
      ferric-mp2's `total_*_gradient_on` / `optimize_geometry_*_on` keep
      their names although `_on` there means "with the reference on the
      J/K route `scf_jk`": they take loose `BasisSet`s and prepare per
      geometry, and have ~110 call sites outside ferric-mp2.

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
      files). Still holding it among the `DensitySource` consumers:
      `orbital_moments`, `density_second_moment`, the `eval_*_on_points`
      functions, `hirshfeld_polarizability` and `response_modes`
      (response_modes.rs).

11a. `RhfConfig` is `#[non_exhaustive]`, bitwise unchanged results and
    messages (compared against origin/main on Python and CLI runs of
    every SCF family, the correlated drivers, CDFT, QM/MM, periodic
    Gamma, MD, frequencies, optimize, saddle/IRC and TDDFT, error cases
    included). The rule: **no `RhfConfig` (or `UhfConfig` / `RohfConfig`)
    struct literal outside ferric-scf**, so a new field takes its default
    at every external site and adding one breaks nothing outside the
    crate. A config is built from
    - `ScfOptions::into_rhf_config`, for a frontend lowering its own
      keywords: `rhf_config_budgeted` (the correlated Python entry
      points), `run_cdft`, `run_saddle` / `run_irc` / `run_frequencies` /
      `run_optimize` (xc, MM field, PCM), and the ferric-ci RHF start
      (`start_config`, now also carrying the system's screening). A field
      whose `ScfOptions` mapping differs from the old literal is written
      after the lowering (`run_cdft`'s `level_shift`, through the setter:
      `ScfOptions::level_shift` also clears the meta-GGA automatic shift;
      the cDFT knobs, which `ScfOptions` does not carry);
    - two named constructors for the repeated production patterns:
      `RhfConfig::periodic(density_conv, max_iter)` (the core-Hamiltonian
      guess of the Gamma configs in ferric-pbc and the CLI / Python
      periodic helpers) and `RhfConfig::gradient_reference(ext)`
      (`density_conv = GRADIENT_REFERENCE_DENSITY_CONV` = 1e-9 and an
      optional MM field: the gradient / geometry-task reference SCFs of
      ferric-mp2, ferric-rpa and the Python geometry drivers; it leaves
      `ri_j_task` at `Energy`, and geometry drivers add
      `.for_geometry_task()`);
    - `RhfConfig::default()` (or an existing config) plus the `with_*`
      setters, each writing exactly one field (`with_stability` the pair
      it names), `#[must_use]`: next to the existing ten, `with_energy_conv`,
      `with_ri_j_precision`, `with_dft_grid`, `with_guess(Guess)`,
      `with_init_guess_density`, `with_external_potential(impl
      Into<Option<_>>)`, `with_stability(Stability)` and
      `with_mom_after_iter`; or a plain field assignment for everything
      else (the fields stay public). No catch-all builder: the setters
      cover the knobs tests set most; sub-config setters belong to 11(b).
      `xc` / `df_j_aux` / `df_k_aux` keep their `impl Into<String>`
      setters only: an `Option` form would need its own conversion trait
      (`Option<String>` is not `From<&str>`, and a bare `None` would not
      infer), so sites passing an `Option` assign the field.
    - `with_level_shift` and `ScfOptions::level_shift` differ on purpose:
      the setter writes `level_shift` only, so `with_level_shift(0.0)` on
      the defaults keeps the meta-GGA automatic 0.5 Ha shift, while the
      options lowering also clears `meta_gga_auto_level_shift` (an explicit
      0 runs unshifted). A new frontend lowering a user's `level_shift`
      goes through `ScfOptions`; the setter is the literal field write.
    Oracle tests (old literal vs new construction, field by field through
    `Debug`, since `RhfConfig` has no `PartialEq`):
    `rhf::construction_oracle` (both constructors; each setter on its own
    against its one-field struct update, from the defaults and from a base
    where every setter-written field is non-default),
    `options::tests::python_struct_update_sites_lower_through_options`
    (each Python site's old literal against the `ScfOptions` form it now
    uses, for unset and all-set keywords) and ferric-ci's
    `start_config_is_the_old_literal`. The ~1300 test, bench and example
    literals (ferric-scf's own integration tests included: they are
    outside the crate too) were rewritten mechanically: the fields with a
    setter for that value as a setter chain first, then assignments for
    the rest (`let mut`, or, in ~25 argument positions whose fields need
    an assignment, a block expression). The base of a `..base` update, and
    the setter fields, are now evaluated before the assigned fields rather
    than in the literal's order; none of the rewritten sites depends on
    that order (reviewed). The CLI's item-7 oracle `old_literal` became
    the defaults plus one assignment per field, so it no longer fails to
    compile when a field is added; nothing guards that oracle against a
    new field now (`every_rhf_config_field_is_classified` covers the
    `ScfOptions` lowering, not this oracle). Literals remain inside
    ferric-scf (its unit tests, the `options` oracles, `..config.clone()`
    updates in the solvers, the free-atom guess configs).
    Follow-ups:
    - Python `run_cdft` and `run_qmmm` keep the meta-GGA automatic shift
      for an explicit `level_shift=` (unchanged from before 11a), against
      the convention on `RhfConfig::meta_gga_auto_level_shift` that the
      CLI and the six SCF entry points follow. Fixing it changes results
      for a meta-GGA with an explicit shift, so it is its own PR.
    - Move the Python `run_cdft` / geometry keyword → `ScfOptions` mapping
      into ferric-scf, so its oracle tests the code the binding runs
      instead of a hand copy in `options::tests`.
-  Reference SCF reuse across runs (profiling-audit lead 4): CLI
   `[scf] save_reference = "path"` writes the converged reference SCF of a
   correlated kind (`task = "energy"`) as a `ferric_scf::checkpoint` final
   file before the correlated stage (root rank only under MPI);
   `read_reference = "path"` reads one under `ReadMode::Skip` and turns it
   into the result (`Checkpoint::into_result`, new and consuming: the
   stored arrays move out bit for bit, `xc`/`relativistic` are stamped
   from the config, the stored in-core ERI route is recorded back on
   `config.eri_route` so the certificate keeps its `eri` route, and the
   unstored diagnostics are `None`) in place of the solve, so no SCF
   iteration runs. The correlated energy is bitwise the one the saving
   run's reference gives; it equals a fresh run's only at the same
   thresholds, thread count, ERI route and build (a file converged
   tighter than asked is accepted and the run says so). Optional meta
   fields added within format v1 (an older build refuses a newer file on
   the unknown key): `precision_json` (the typed precision record,
   rebuilt by the new `PrecisionRecord::from_json` and accepted only when
   its `Debug` is the stored `precision_detail`, so a mixed RI-J f32
   reference restores), `eri_route` and `writer_threads`. The identity is
   the library's `ScfIdentity::new` on the configuration the solver
   receives, plus two CLI fields pushed onto `fields`
   (`cli_reference_route`: plain / DF-guess + aux / DF-increments + aux /
   UHF / GW UHF-ROHF, and `mpi_ranks`); `identity.rs` only gains
   `FERRIC_GPU_MIXED_K_PANEL` in `ENV_SWITCHES` (it moves the last digits
   of a mixed result). One `ReferenceRoute` in `ferric-cli/src/lib.rs`
   decides both the solve and that field. A mismatch is an error naming
   the fields, never a fallback (as a guess or a fresh SCF); the stored
   `Tr[Dh]` and `Vnn` are rechecked bitwise against this run's integrals.
   A read run's certificate gains the route `scf reference: read from
   <file> (checksum ...)` and its run log records the file's thresholds
   and writer thread count. A dirty or unknown build needs
   `read_reference_other_build = true`. Refused at load time: the SCF
   kinds (their ladder / UHF / ROHF route is not the correlated one), the
   double hybrids, CAS, OP-MP3, other tasks, `[cell]`, both keys at once,
   and `read_reference` with `check_stability` / `stability_descent`.
   Open-shell `pdep-rpa` used to re-solve the same UHF + MOM reference in
   its arm (`run_pdep_rpa_arm`) and so refused `read_reference`; the arm
   now consumes the shared reference (one UHF solve per run instead of
   two; the energy, SCF energy, stdout and certificate are bitwise those
   of the old double solve on OH / STO-3G and 6-31G, OH + PBE and triplet
   CH2 / cc-pVDZ), and `read_reference` is accepted there. Refused before the SCF: a reference
   with the COSX final-grid pass or polarizable embedding (the file
   stores neither), and a save path that is a directory, a symlink or in
   a missing directory. Tests: `ferric-cli/tests/scf_reference_io.rs`,
   `cli_wired_keys_tests::reference_keys_*`, `checkpoint::tests::into_result_*`,
   `provenance::tests::from_json_inverts_to_json`. Python: the drivers
   solve their own reference and take no result object, so the same keys
   are keywords (`save_reference=`, `read_reference=`,
   `read_reference_other_build=`) on `run_rimp2` (exact energy only),
   `run_ccsd` and `run_ccsd_t`, through `ferric_cli::reference_io` (now
   public, with a `Surface` that names the keys per front end) in
   `ferric-python/src/reference_file.rs`; the route field names the Python
   solve (`python solve_rhf` / `python solve_uhf`), so CLI and Python files
   are never read as each other, and a refusal is an `InputError`
   (`tests/test_reference_file.py`). Follow-ups: the other Python
   correlated drivers (each solves its reference its own way); no file
   guess (a mismatched file is refused, not used to seed the SCF); the SCF
   kinds could save once the ladder's final rung is part of the identity.

-  Contraction rotation of generally contracted bases (cc-pVXZ; perf5).
   Phase 1 (#709): `ferric-integrals/src/contraction_rotation.rs`
   (`RotatedBasis`: diffuse projection, then one optimized-general-
   contraction elimination, `T` block diagonal per atom/l/exponent list)
   and the rotated walk of `DirectJK` (`direct_jk/gc_rotation.rs`), switch
   `RhfConfig::gc_rotation` / `[scf] gc_rotation` / `FERRIC_GC_ROTATE`
   (`auto` | `rot1` | `off`; `off` and segmented bases bit for bit).
   Phase 2: the rotated basis, its bounds and engines are cached on the
   geometry's `SchwarzBounds` (`GcSlot`, field `gc`; clones share it, like
   `incore`), built once per geometry instead of once per builder (the
   entry is keyed by `incore_jk::geometry_key`: bounds reused after the
   atoms moved refuse it), and built with no lock held (the rotated
   Schwarz table is rayon work; concurrent first uses keep the first
   insert). The rule for who rotates: a build rotates only where a config
   asked. `GcSlot` has three states: never recorded (fresh bounds: no
   build rotates, `FERRIC_GC_ROTATE` is not read), recorded unset (a
   config left `gc_rotation` unset: the variable, then `auto`) and
   recorded value; the record is shared by clones, last writer wins for
   later resolutions (a builder that already resolved keeps its rotation,
   and a cached level stays cached). Bounds carry their owner geometry
   (`geometry_key`, which hashes atoms, shell layout, exponents and
   contraction coefficients, set by `SchwarzBounds::compute`); a rotation
   for any other geometry is refused before any level is built. Behaviour
   change from phase 1: a standalone `DirectJK` on fresh bounds with
   `FERRIC_GC_ROTATE=rot1` (or unset, i.e. `auto`) used to rotate and now
   runs unrotated; `.with_gc_request(None)` restores the variable-based
   choice. The rotated builds agree on their buffer reservation across MPI
   ranks (`RotatedJk::charge_collective`, a min all-reduce like `df_k`'s
   Cholesky vote) before any reduction, so a refusal on one rank is an
   error on every rank, never a hang. The in-core store is reused only
   when it was built on the requested rotation. The
   SCF drivers record their `RhfConfig::gc_rotation` (`GcSlot::set_request`,
   next to `incore_jk::begin_scf`), and every build on those bounds
   follows it: the SCF's `DirectJK`, `DirectJ`, `DirectK` and in-core
   store (`IncoreEri` holds the rotated ERIs and transforms in `coulomb` /
   `exchange`), the pooled `rhf::build_jk`, `build_jk_with_pool` and its
   homogeneous, spin and multi siblings when they run on the SCF's bounds
   (Newton, TRAH, stability, the Hessian's and properties' CPHF/CPKS, the
   RI-MP2 Z-vector through `ResponseJk::for_scf`, OO-MP2), and the
   gradient's four-centre derivative walk (`gradient::gradient_rotation`:
   the walk on the rotated shells with `D' = Tᵀ D T`, for every
   `twoelectron_*_gradient` wrapper whose operator is the bounds'). Three
   consumers that build their own bounds carry the request over: the KS
   Z-vector (`zvector_ks::ResponseFock` runs its Coulomb builds on the
   SCF's bounds, with the in-core slot detached, when they are plain
   Coulomb Schwarz bounds, and copies the record onto every other
   operator's bounds, `GcSlot::copy_request_from`), the CASSCF direct
   backend (`CasScfConfig::gc_rotation`, the CLI passes `[scf]
   gc_rotation`) and the GW exact exchange (`vxc_diagonal_mo_on_grids`'s
   `gc_rotation`). Everything else that computes its own bounds (the
   Python `jk` / `grad_jk` / `FockResponse` blocks, any other fresh-bounds
   consumer) runs unrotated, bit for bit as before. A pooled build rotates
   only for a pool from `EnginePool::new` (`EnginePool::eri_key`), matching
   its operator and precision; `from_fn` pools run unrotated. Python:
   `gc_rotation=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`,
   `run_uks`, `run_roks` (not `run_ksdft`, frozen); the config spelling is
   strict (`auto` | `rot1` | `off`), the variable ignores case and blanks.
   Tests: `ferric-scf/tests/gc_rotation.rs`,
   `ferric-ci/tests/casscf_direct_gc_rotation.rs`,
   `ferric-mp2/tests/rimp2_gradient_gc_rotation.rs`. Follow-ups: LinK (its
   generic `Bound` and pair tables would need a rotated twin), CFMM near
   field, the Hessian's second-derivative integrals (`hessian.rs` skeleton
   and Fock derivative walks), the Python `scf_*` keyword on the geometry
   and correlated drivers (and a `gc_rotation` on the Python CASSCF
   entries), the engine pool a pooled caller hands in is unused when the
   build rotates (the rotated engines come from the cache), the rotated
   engine pool is a one-slot cache (keyed by operator, precision and
   thread count), and phase 3 of the design (a shared-primitive kernel or
   libcint for the remaining ~1.9x to def2-SVP).

-  Bra-pair-local J/K digest (perf5d). `quartet_scatter.rs`: `JkMode::Pair`
   (one variant for the J+K, open-shell, J-only and K-only jobs, made by a
   `PairSpec` that prepares the densities once per build), the kernel
   `pair_quartet` (const-generic on the ket shell width n4 ∈ {1, 3, 5},
   runtime otherwise, on J, the number of exchange channels and the
   general-density form) and `PairSpec::finish` (`J = Jh + Jhᵀ`,
   `K = Kb + Kbᵀ`, or `Kb + Ktᵀ` for a density not symmetric to roundoff:
   `max|D − Dᵀ| ≤ 1e-14 · max|D|`, non-finite never symmetric, except
   that an incremental `ΔD` is judged against the last full build's
   `max|D|` per channel, which `DirectJK` keeps (`SymScale::Delta`);
   symmetrised explicitly when it passes; UHF checks each spin and runs
   both on the general form if either fails). `scatter_bra_pair_probed`
   calls `begin` / `quartet` / `end` in place of its three digest
   branches; screening, counting and interrupts are shared.
   `PairSpec::sum_groups` is the one pair branch of every builder: half
   partials summed in the usual group order
   (`reduce::ordered_stream_sum_many`), then finished; the caller reduces
   across ranks and adds into its J/K (the add-into contract of
   `build_incremental`). It charges the density copies and half
   accumulators to the memory pool (`gc_rotation::reserve_collective`, on
   every rank or none). The legacy branches stay verbatim per builder:
   they differ in mode, extraction and reduction function
   (`grouped_deterministic_sum*` against `ordered_stream_sum_*`), and a
   generic helper over them would change which reduction a builder calls
   for no gain. Wired into `DirectJK` (closed and open shell; the
   rotated walk follows because its rotated bounds share the parent's
   record), `DirectJ`, `DirectK` and `rhf::build_jk_with_pool`. Switch:
   `JkDigest` (`pair` default | `legacy`, the 8-fold scatter bit for bit),
   `RhfConfig::jk_digest` / `[scf] jk_digest` / `FERRIC_JK_DIGEST`,
   resolved config > env > default per build from a `DigestSlot` on
   `SchwarzBounds` (field `digest`, shared by clones and by the rotated
   bounds). Simpler than `GcSlot`: unrecorded bounds follow the variable
   (there is no "never on fresh bounds" state), so `FERRIC_JK_DIGEST`
   reaches every build in the process. The SCF drivers and
   `df_increments` record the config's value; the KS Z-vector copies it,
   the CASSCF direct backend (`CasScfConfig::jk_digest`, `ScfStart`) and
   the GW exact exchange (`vxc_diagonal_mo_on_grids`) take it, and the CLI
   passes `[scf] jk_digest` to all three (the CLI GW now also passes
   `[scf] gc_rotation`, which its docs already promised; GW takes both as
   `vxc_mo::JkRoute`). In the checkpoint identity as `jk_digest`, the
   resolved digest (Route; the variable is folded in, so it has no
   `ENV_SWITCHES` entry); a file written before the field reads it as
   `legacy`, so a skip from such a file is refused only by a run that uses
   `pair`. Python: the variable only. Tests:
   `ferric-scf/tests/jk_pair_digest.rs` (pair against legacy, all builders
   and bases, symmetric and not, incremental, whole SCF, the config
   reaching the consumers bitwise; each comparison checks through the
   test probe `quartet_scatter::pair_probe` that the pair kernel ran, on
   the expected kernel) and the unit tests in `quartet_scatter.rs`.
   `FERRIC_TIMING` names the K kernel on the `direct-jk build` line and
   reports the serial density preparation and finish on their own line.
   The legacy kernel stays for at least one release after this one, for
   exact reproduction of earlier results and A/B debugging; remove it
   (with `JkDigest`, the switch and the identity field's `legacy` reading)
   once no consumer needs bitwise reproduction of pre-pair results.
   The two gradient routes that re-solve their reference at each geometry
   from reduced settings carry it too: the RI-MP2 family's
   `ScfJk::jk_digest` (built with `ScfJk::exact()` / `ScfJk::rijk(aux)` /
   `with_jk_digest`; `solve_rhf_reference` and the open-shell
   `solve_reference` write it into the reference's config when it is set,
   and the SCF records it on its bounds, which the closed-shell Z-vector
   response reuses; the open-shell response build keeps the legacy
   digest) and the PDEP-RPA gradient's `RpaGradientRoute`
   (`total_rpa_gradient_by` and `optimize_geometry_rpa` take
   `impl Into<RpaGradientRoute>`, so a bare `RpaGradientMethod` still
   means "unset"; the reference, every displaced finite-difference RHF and
   the analytic Z-vector follow it). The CLI fills both from
   `[scf] jk_digest` (PR #717 refused the key there) and warns about the
   `[scf]` keys those routes do not carry (`uncarried_gradient_scf_keys`,
   see 16 under Next); Python leaves them unset. Neither `jk_digest` nor
   `gc_rotation` is an `exact_route_key` any more: they no longer move an
   omitted `[scf] jk` off the RI-JK default (#714 and #717 had added
   them), so they change only the last bits of whatever four-centre builds
   a route runs. A `gc_rotation` on an SCF with no four-centre direct J or
   K build (RI-JK, RIJCOSX, a pure functional on RI-J) warns that it does
   not affect the reference SCF, with the opt-out the kind accepts (`jk =
   "exact"` or `df_*_aux = ""`; not for `gw` on a hybrid, whose exchange
   diagonal is four-centre; `gc_rotation_route_warning`). Tests:
   `ferric-mp2/tests/rimp2_gradient_jk_digest.rs`,
   `ferric-rpa/tests/rpa_gradient_jk_digest.rs` and
   `ferric-cli/tests/jk_digest_gradient_routes_cli.rs` (runs on a memory
   budget below the in-core store, so the reference builds direct).
   Follow-ups: the open-shell
   response build (`build_spin_jk_with_pool_homogeneous`) and the
   block-CPHF `Multi` mode (lanes), LinK's K-only loop (same row
   structure), and packed shell-pair-blocked D/J layouts for the `E34`
   gather.

-  Shared-primitive `(ss|ss)` / `(ss|sp)` kernel on the rotated walk
   (perf5 phase 3, "b2-s"). `ferric-integrals/src/gc_kernel.rs`
   (`GcKernelBasis`): the groups of a `RotatedBasis` (its rotation groups,
   `RotatedBasis::groups`, split into runs of columns sharing a nonzero
   exponent), libint-normalised `n_union × m` coefficient matrices, per
   ordered group pair with `l_a + l_b ≤ 1` the primitive-pair data
   (libint's `Original` screen at the walk's engine precision, the
   same-group s fold), a 7th-order Taylor Boys table, the `(ss|ss)` F0
   kernel and the `l_ab + l_cd = 1` VRR/HRR kernel; `gather` copies one
   column quartet out in libint's layout and function order. Coulomb only
   (the walk checks the bounds' operator). Wiring
   (`ferric-scf/src/direct_jk/gc_kernel.rs`): on the rotated walk of
   `DirectJK` only (`Walk::kernel`: rotated, pair digest, Coulomb, setting
   `auto`), closed and open shell, full and incremental. The bra-pair list
   is reordered group first (`group_first`: the first member of an ordered
   bra group pair pulls the others in after it), striped across MPI ranks
   by whole buckets and partitioned by whole buckets (`Partition::Units`,
   at least `deterministic_group_size` pairs per group), all pure functions
   of the list. `PairSpec::with_gc_kernel` gives every group accumulator a
   `KernelCache` (blocks keyed by the ket group pair, dropped when the bra
   bucket changes); `scatter_bra_pair_probed` asks it for every survivor
   with `l1+l2+l3+l4 ≤ 1` (`PairJk::kernel_quartet`): the group block is
   evaluated on the bucket's first survivor in that ket group pair, the
   column quartet gathered and digested by the same `PairJk::quartet`, in
   the same canonical order; everything else goes to libint unchanged.
   Screening, counting and the reduction are unchanged, so thread-count
   bit-identity holds by construction. Switch: `GcKernel` (`auto` | `off`,
   default `off` for this release), `RhfConfig::gc_kernel` /
   `[scf] gc_kernel` / `FERRIC_GC_KERNEL`, resolved config > env > default
   from a `GcKernelSlot` on `SchwarzBounds` (field `gc_kernel`, shared by
   clones and the rotated bounds, like `digest`), agreed across MPI ranks
   (`GcKernelSlot::mode_collective`); `off` is the walk without the kernel
   bit for bit, its bra-pair order and partition included. Recorded by the
   SCF drivers and `df_increments`, copied by the KS Z-vector, taken by the
   CASSCF direct backend (`CasScfConfig::gc_kernel`, `ScfStart`), GW
   (`JkRoute`) and `mbd_scf`, passed by the CLI, and carried like
   `jk_digest` by the re-solving gradient routes (`ScfJk::gc_kernel` for
   the RI-MP2 family, `RpaGradientRoute::gc_kernel` for `pdep-rpa`, through
   `ferric_rpa::gradient::ReferenceJk` to every displaced and analytic
   reference). It does not move an omitted `jk`, and an explicit key on a
   route with no four-centre build warns with `gc_rotation`'s warning. In the
   checkpoint identity as `gc_kernel`, resolved (Route); a file without it
   reads as `off`. Python: `gc_kernel=` on the entries that take
   `gc_rotation=`. `FERRIC_TIMING` adds the kernel's thread-seconds to the
   group line and a line with its quartets, group evaluations and reuse.
   Tests: the kernel against libint integral by integral
   (`gc_kernel/tests.rs`), `ferric-scf/tests/gc_kernel.rs` (J/K and SCF
   energies against the libint walk, the probe `gc_kernel::kernel_probe`,
   bitwise off / segmented / unrotated / legacy, thread identity), the
   checkpoint identity tests, `tests/mpi_gc_kernel.rs` in the CI `mpirun`
   matrix, the gradient-route tests of the CLI, ferric-mp2 and ferric-rpa;
   the J/K and thread-identity tests also run with 6 reduction groups
   (`reduce::with_target_groups`) so each block cache crosses several bra
   buckets. Mutation-checked (Boys argument, a contraction coefficient, a
   duplicated exponent, the same-group fold, an HRR index and sign, the
   pure-p order, the cache key, a stale cache or `last` memo across
   buckets, a skipped digest, a skipped per-column screen, digesting
   non-surviving column quartets: each fails a test). Measured (book, "The shared-primitive kernel"; 2
   threads, shared box): per build 1.22× on octane cc-pVDZ (paired range
   1.19–1.28), 1.12× butane aug-cc-pVDZ, 1.03× butane cc-pVTZ, def2-SVP
   unchanged (the kernel never runs); 1.43 quartets per group evaluation
   on octane, as in a one-group walk (the bucket partition loses no reuse).
   UHF/ROHF share the wiring (`build_uhf`). The 1.28× projected from the
   prototype was the pooled median of two noisy rounds (1.22 and 1.31); a
   quiet-box rerun of production gave 1.21× with `off` at main's speed.
   Follow-ups: flipping the default to `auto` is item 17 below, with its
   prerequisites; the pooled builds
   (`rhf::build_jk_with_pool`: Newton, stability, CPHF, Z-vector) and
   `DirectJ` / `DirectK` stay on libint and could take the same cache;
   b2-full (p and d classes) needs a kernel that beats libint there, which
   the prototype's generic VRR does not (0.3–0.9×); the per-survivor
   lookup (group and pair indirections, a hash probe) could be hoisted
   per ket group pair.

Next, each as its own PR following the patterns above:

8d. (rest) What 8c and the first part of 8d left loose: (b) the ferric-scf
    solvers take loose arguments, and the CLI arms, the Python block
    helpers, `run_qmmm` and `run_cdft` call them with the system's pieces
    and a config of their own, so the system's screening check does not
    cover them (moving them onto the system's methods would turn a config
    whose screening differs from `[scf] screening` into an error); (c)
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
    - One public entry per kernel: the loose kernels under the `*_on`
      entries are still `pub` (see the first part); make them
      `pub(crate)` crate by crate, moving their tests to the entries.
    - Slim ferric-python's `BundleJob` (pdep_properties.rs) to its
      `PreparedAux`, and remove the CLI's loose `bs` arguments (below).
    - Rename ferric-mp2's `total_*_gradient_on` / `optimize_geometry_*_on`
      (`_on` = "on the J/K route `scf_jk`") so `_on` means one thing.
    - `ScfResult` carries no molecule, so `check_reference` cannot see a
      reference of another molecule or geometry in a basis of the same
      size. `check_reference` should return a `CheckedReference<'s>` that
      the `*_on` entries take, carrying the brand of the system the result
      was solved on (set by the system's solve methods), so such a
      reference is caught at the type level and checked once.
    - The `bs` argument of ~40 CLI functions is `sys.basis_set()` (purely
      mechanical, left to avoid colliding with item 12's rewrite of the
      same functions); consider an `ArmCtx` for the CLI arms' shared tail.
      ferric-python's `BundleJob` (pdep_properties.rs) still carries the context,
      molecule, basis set and operator next to its `PreparedAux`.
    - The system's solve methods should take a config type without
      `screening` (parse, don't validate); the refusal then goes.
      `run_wb97x_l_v_on` passes its caller's `RhfConfig` to the loose
      ladder unchecked, as before.
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
11. `RhfConfig` in steps, each with a field-by-field oracle test
    ((a) is done, as 11a above): (b) group fields into sub-configs (`Convergence`,
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
14a. `ferric-testkit` and a clippy ratchet on `too_many_arguments` (the file
    splits of 14 are NOT done; see 14b below).
    - `crates/ferric-testkit` (`publish = false`, dev-dependency only;
      `scripts/check_testkit_dev_only.py` fails CI if any manifest names it
      under `[dependencies]` or `[build-dependencies]`). It depends on
      `ferric-core` only, so `ferric-core` and the crates below it
      (`ferric-tensors`, `ferric-build-info`, ...) cannot use it (a
      dev-dependency back onto a crate's own dependency would give its unit
      tests two copies of `ferric-core`); everything above can. Helpers that
      need `ferric-integrals` or `ferric-scf` types therefore stay local.
      Contents, each one a helper that was copied verbatim into three or
      more test files: `pool::{pool_lock, CleanSlot}` (the process-global
      memory-pool test slot: binary-wide lock, pool cleared on entry and
      exit), `compare::{max_abs, max_abs_diff}` over `Array2<f64>` (return
      the number; every call site keeps its own tolerance),
      `reference::{num, num_ctx}` (JSON reference-field accessors that name
      the missing field), `molecules::water_xyz` and `paths::{workspace_root, testdata}`
      (the latter two are used by `water_xyz`; ferric-cli's 18 local
      `workspace_root` copies have three different bodies and were not
      migrated) (`testdata/molecules/water.xyz`).
    - Migrated (dev-dependency added, local copies deleted) in ferric-scf,
      ferric-mp2, ferric-rpa, ferric-cc, ferric-gw, ferric-pbc,
      ferric-integrals, ferric-dft, ferric-pcm, ferric-cli, ferric-ci and
      ferric-mm: 226 files. Copies removed: 72 `num`, 51 `max_abs`, 24
      `max_abs_diff`, 22 `global_lock` (now `pool_lock`), 20 `CleanSlot` (struct, `acquire`,
      `Drop`), 15 `water` (the `testdata/molecules/water.xyz` loader). Only semantically
      identical copies, never weaker ones, were replaced (compared after
      comment and whitespace normalisation). Two `max_abs_diff` variants
      were merged: the `assert_eq!(a.dim(), b.dim())` plus zip-fold form
      (ferric-scf) and the `(a - b).iter()` form (ferric-mp2, ferric-scf);
      the testkit asserts dims, so it can only fail louder. The testkit
      norms also return NaN when any entry is NaN (`f64::max` would drop it
      and an all-NaN result would read as 0.0); a `< tol` check on a NaN
      now fails, as does a `> tol` "must matter" check. `pub fn max_abs` in the
      `*_support/mod.rs` and `*_common/mod.rs` test modules became
      `pub use` re-exports so the including binaries are unchanged.
    - Deliberately left: the many `water()` / `h2o()` fixtures with
      inline coordinates (different geometries per test, not copies), the
      `vec_f64` readers (three different bodies), the xorshift `Rng`
      structs (3 crates, method-bearing, different state types),
      `has_deriv2` (needs `ferric-integrals`), `parts_diff` (ferric-pbc
      `kxcgrad_hybrid_support` and `kuxcgrad_support`: a max over
      per-part differences, not a copy of `max_abs_diff`), and in-`src` unit-test
      modules whose helper is also used by non-test code.
    - `too_many_arguments` ratchet. Before: the lint was
      `allow` in `[workspace.lints.clippy]`, so the 594 item-level
      `#[allow(clippy::too_many_arguments)]` were decoration and a new
      9-argument function passed silently; `ferric-gw` carried a
      crate-level `#![allow]` hiding its offenders. Now: the workspace
      entry is gone (lint is warn, so `clippy -D warnings` rejects a new
      offender), the crate-level allow became item-level allows, 77 offenders
      that the old config hid (60 found workspace-wide, 12 in ferric-gw,
      5 under the gpu feature set) got an explicit item-level allow, and
      `scripts/too_many_arguments_ratchet.py` compares the per-crate count of
      those allows to `scripts/too_many_arguments_baseline.json` (670 in
      16 crates: ferric-pbc 172, mp2 132, scf 126, python 90, dft 39, rpa 31,
      cli 22, gw 20, integrals 14, cc 11, benchmarks 3, core 3, mm 2,
      tddft 2, tensors 2, pcm 1). A crate over its baseline fails; a crate
      under it also fails until `--update-baseline` records the gain, and
      `--update-baseline` refuses to raise any number. A crate-level allow
      or an `allow` in a manifest is always an error. Wired into the `lint`
      job in `ci.yml` and into `scripts/ci-gate.sh` (step 3a, never
      soft-skips); tested by
      `crates/ferric-python/tests/test_too_many_arguments_ratchet.py`.
      `ferric-core` appears at 3 because the gpu-feature code was only
      visible under `--features ferric-core/gpu`.
    - Follow-ups: use `#[expect(clippy::too_many_arguments)]` instead of
      `#[allow]` so a fixed offender's attribute becomes a warning (stable
      since Rust 1.81; the manifests declare no `rust-version` and CI uses
      stable, but check any pinned toolchain before the sweep, and the
      ratchet already counts `expect`); shared `vec_f64` / `matrix` reference
      readers (their bodies differ today, so unify them with a decision on
      the panic text); and a `max_abs_diff_at` returning the location of the
      worst entry, for the tests that print it.
14b. (Open) File splits. Files over 3000 lines today (generated tables
    excluded: `ferric-d3/src/tables.rs`, `ferric-quadrature/.../tables.rs`,
    `ferric-integrals/tests/data/yukawa_urr_ref.rs`): ferric-cli
    `config.rs` 14662, `lib.rs` 9824; ferric-scf `rhf.rs` 6726, `uhf.rs`
    4588, `hessian.rs` 4808, `gradient.rs` 3171, `qmmm.rs` 3077,
    `optimize.rs` 3047; ferric-mp2 `lmp2_direct.rs` 5029, `rimp2.rs` 3535,
    `lmp2_amplitude.rs` 3513, `oo_rimp2.rs` 3400, `laplace.rs` 3147;
    ferric-python `pbc.rs` 4597, `blocks.rs` 3292, `geometry.rs` 3176,
    `mp2.rs` 3087; ferric-integrals `three_index_source.rs` 4344,
    `engine.rs` 3882; ferric-pbc `dft.rs` 3615, `rsgdf.rs` 3488, `grad.rs`
    3043; ferric-dft `xc_batch.rs` 3207. Not split here: every one is hot in
    in-flight branches, and a split is move-only with `pub use`
    re-exports, so it goes one file per PR, when nothing else is open
    against that file. Plan for `ferric-cli/src/lib.rs` (`try_run`'s ~700-line dispatch stays in `lib.rs`
    and only calls into these):
    - `args.rs`: `usage_text`, `USAGE_SYNOPSIS`, `version_text`,
      `libint_summary`, `CliArgs`, `parse_args`, `run`, `main`, the
      `SUPPORTED_*` / `EPISTEMIC_WARNINGS` tables and the unsupported-*
      messages.
    - `install.rs`: `load_and_install`, `install_gpu`,
      `install_integral_settings`.
    - `scf_setup.rs`: `apply_scf_jk`, `scf_jk_route_line`,
      `print_df_guess_outcome`, `refuse_dispersion_without_ks`,
      `resolve_grid_prune`, `ScfConfigInputs`, `try_scf_config`,
      `prepare_system`, `resolve_geometry`, `log_jk_path`, with its `scf_config_lowering_tests`.
    - `scf.rs`: `run_rhf`, `run_symmetric_rhf`, `run_ksdft`, `run_uhf*`,
      `run_rohf`, `solve_open_shell_reference`, the certificate printers,
      `print_cosx_final`, `energy_conv_advisory`, `scf_functional`, with `scf_functional_tests` and
      `energy_conv_advisory_tests`.
    - `dispersion.rs`: `DispersionCorrection`, `DispersionGradientModel`,
      `dispersion_*`, `mbd_free_atom_cache`, `print_scf_energy`,
      `log_scf_dispersion`, and the D3 helpers `d3_frequencies`,
      `d3_geometric_correction`, `frequencies_maybe_dispersion`.
    - `mp2_arms.rs`: `LocalPrint` and the local-MP2 JSON/certificate
      printers, `run_rimp2_local`, `run_rimp2`, `run_u_rimp2`, `run_mp3`,
      `run_oo_rimp2`, `run_u_oo_rimp2`, `run_att_rimp2*`, `run_scs_mp2*`,
      `run_mp2_v`, `run_laplace_*`, `run_omega_ri_cdd`, the double-hybrid
      arms (`run_wb97x_l_v_arm`, `run_mp2_double_hybrid_arm`,
      `DhPrint`, `double_hybrid_*`) and the MP2-variant notices, with
      `local_printout_tests`, `mp2v_notice_tests`.
    - `cc_arms.rs`: `closed_shell_cc_config`, `ladder_label`, `run_ccsd`,
      `run_linlccd`, `correlation_aux`, `require_restricted`, `run_ccd`,
      `run_ccsd_t`, `run_u_ccsd`, `run_eom_ccsd`.
    - `rpa_gw_arms.rs`: `run_rs_mp2_rpa*`, `RsScanCtx`,
      `rs_mp2_rpa_notices`, `run_drpa`, `run_exact_drpa`,
      `preflight_exact_drpa`, `run_pdep_rpa_arm`, the PDEP export helpers,
      `run_gw`, `gw_open_shell_reference*`, `run_bse_tda`,
      `run_tdhf_static_polarizability`, `run_tddft_arm`,
      `refuse_tddft_xc_without_kernel`, `tddft_*`, with `rs_mp2_rpa_notice_tests`.
    - `frequencies.rs`: `run_frequencies*`, `print_frequency_table`,
      `frequency_*`, `log_frequencies`, `LastScf` and its tests, the
      gradient-kind tables.
    - `optimize.rs`: `run_optimize*`, `certify_optimization`, `RefCert`,
      `task_reference_certificate*`, `rimp2_gradient_scf_jk`,
      `rpa_gradient_setup`, `att_rimp2_*`, `scs_*_config`,
      `oo_rimp2_config`.
    Do `config.rs` along the same lines (one module per `[section]`
    struct plus its validation) in a second PR, after `lib.rs`, since the
    CLI arms read it everywhere. The unit-test modules move with the code
    they exercise; a `mod` added at a time with `pub(crate) use` keeps every
    path in the arms unchanged.

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

16. The rest of `[scf]` on the two re-solving gradient routes. The
    RI-MP2-family (`rimp2`, `scs-mp2`, `att-rimp2`, `scs-mp2-2terfc`) and
    `pdep-rpa` `optimize` / `frequencies` gradients re-solve their
    reference at each geometry from `RhfConfig::gradient_reference` plus
    the J/K route (`jk` / `jk_aux` for the RI-MP2 family; `pdep-rpa` is
    always four-centre), `jk_digest` and `gc_kernel`. These runs solve no shared SCF
    before the gradient loop (`try_run` dispatches to `run_optimize` /
    `run_frequencies` first), so `gc_rotation`, `k_builder`, `screening`,
    `eri` and `integral_thresh` reach only the certificate's reference SCF
    at the final geometry (`task_reference_certificate`), never the
    energies and gradients; so do the
    convergence and stabilisation settings (`max_iter`, `energy_conv`,
    `level_shift`, DIIS and guess options; `density_conv` is the fixed
    1e-9 of the gradient reference). Today the CLI warns about
    the five J/K keys (`uncarried_gradient_scf_keys`). Carrying them means
    replacing `ScfJk::{jk_digest, gc_kernel}` and `RpaGradientRoute`'s two (`ReferenceJk`) by one
    reference-settings bundle (say `ReferenceScf { gc_rotation, k_builder,
    screening, eri, integral_thresh, jk_digest, gc_kernel, .. }`, applied to the
    per-geometry `RhfConfig` by one method, the same bundle in both
    crates) and checking each key's consumers on these routes:
    `screening` must reach `SchwarzBounds::compute_for_screening` where the
    drivers now call `compute`, which also changes the response and the
    two-electron derivative terms that reuse those bounds; `k_builder`
    needs a decision per builder (COSX is already refused on these
    gradient tasks; a LinK reference would meet the four-centre
    `ResponseJk`, equal only to LinK's screening threshold, so it needs its
    own consistency bar or a refusal); `gc_rotation` reaches the response
    through `GcSlot` once the SCF records it; `eri = "incore"` needs the
    store's budget, which the exact route takes from the environment, not
    `[memory]`; `integral_thresh` changes the response's screening too.
    Each carried key wants a config-vs-variable bitwise test like the
    digest's, and the warning goes once every key is carried (or refused).

17. Flip `[scf] gc_kernel` to `auto` by default (the shared-primitive
    `(ss|ss)` / `(ss|sp)` kernel of the rotated direct walk; see "Done").
    Prerequisites, each its own change first:
    - Memory: cap the per-accumulator block-cache arena and charge it to
      the memory pool (it grows with the ket group pairs one bra bucket
      touches, ~0.5 MB on octane, unbounded in principle); make `pidx`
      sparse (it is `n_sp_groups²` `u32`s, 16 MB at 2000 s/p groups); build
      pair data only for group pairs that hold a significant shell pair
      (today every ordered s/p group pair is set up, most of them empty for
      a large molecule).
    - A speed measurement on heavy atoms and pseudopotential bases (cc-pVTZ
      Cl/Br, cc-pVDZ-PP on Ag/I), paired against `off`: the measured gains
      are alkanes only.
    - Re-measure the cc-pVDZ / aug-cc-pVDZ / cc-pVTZ table on a quiet box.
    Deferred with it, not blocking: the libint-layout helpers (pure-p
    order, function maps) duplicated between `gc_kernel.rs` and the
    digest's layout code could share one place; and the public surface of
    `ferric_integrals::gc_kernel` (`KernelGroup` fields, `PairKind`,
    `boys_for_tests`) can shrink to `pub(crate)` plus what `ferric-scf`
    uses.
