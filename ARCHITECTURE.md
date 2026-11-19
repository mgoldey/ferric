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
  would kill the interpreter. Existing uses (`LocalCfg`, `Mp2Cfg`) are slated
  to move into ferric-mp2.

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
  (`AuxBasis { Off, Named }`, `Stability`, `Guess`, `TrahTrigger::new`), not
  the strings and bools the frontends receive.
- No sentinel values (`0`, `""`, `0.0`, NaN) for "unset", "auto" or "off".
  Use `Option` or an enum. Existing sentinels in `RhfConfig`
  (`three_index_budget_bytes = 0`, `newton_trigger = 0.0`, the empty-string
  auxiliary basis) are on the roadmap.
- One home per invariant. A rule is checked by one function; other layers call
  it rather than restating it with their own message.
- Per-flavour defaults that differ on purpose (for example `run_rhf` vs
  `run_dft` vs the CLI auxiliary-basis policy) are encoded explicitly in the
  lowering, not merged.
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
  `?`, so that `process::exit` appears only in `main`/`run`.
- **Python:** convert a `FerricError` with the crate-local `IntoPyResult`
  extension trait (`.py()?`), not a fresh `map_err` per call site. A blanket
  `From<FerricError> for PyErr` is not possible (orphan rule). Exception
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
- **Owners establish invariants once.** A `PreparedSystem` (molecule with ECP
  applied, prepared basis, operator, Schwarz bounds, context) is built by a
  single constructor and only borrowed afterwards, so a solver cannot receive
  a molecule without its ECP.
- **Units are newtypes at API boundaries only** (`Bohr`/`Angstrom` on molecule
  constructors and accessors, `Hartree` on result energies). Kernels keep raw
  `f64` and arrays. Python constants live in `ferric.units` and match the Rust
  values.

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

Next, each as its own PR following the patterns above:

6. Extend the SCF lowering to `run_dft`/`run_uks`/`run_roks`, with a
   measured-difference test for their `rhf_config_budgeted` defaults; then
   fill the keyword-argument drift between the six SCF entry points
   additively (and in `ferric.pyi`).
7. Lower the CLI `ScfCfg` through `ferric_scf::options`, removing the
   hand-kept "CLI and Python agree" parity.
8. Grow the existing `converged_*` reference-SCF helpers in
   `ferric-python/src/mp2.rs` into a borrowed `PreparedSystem` (molecule,
   prepared basis, operator, bounds, context) used by every correlated
   driver. Whether a driver applies ECPs is a behaviour change and gets its
   own tests.
9. A `MethodResult` trait (`total_energy`, `certificate`, `timings`) and a
   declarative macro for the shared Python result getters, adding
   `energy`/`total_energy` aliases on every SCF result.
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
14. Spin-generic KS gradient wrappers (gated on bit-identical gradients), a
    shared `ferric-testkit` dev crate, table-driven periodic entry points, file
    splits, and a clippy ratchet on `too_many_arguments`.
