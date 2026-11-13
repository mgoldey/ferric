# avogadro-ferric

An [Avogadro 2](https://avogadro.cc) package plugin that runs the
[ferric](https://github.com/mgoldey/ferric) quantum-chemistry engine from
inside Avogadro. It needs Avogadro 2.1.0 or later and Linux x86_64, because
ferric ships wheels for that platform only.

| Avogadro feature | Where | What it does |
|---|---|---|
| Energy model **ferric (QM)** | Extensions > Calculate (Configure… → ferric (QM); Optimize Geometry, Energy, Forces) and the Auto Optimize tool | SCF energy and analytic gradient (RHF, UHF, RKS, UKS) |
| Charge models **ferric Lowdin / Hirshfeld / CHELPG** | Wherever Avogadro offers a charge model, e.g. Apply Colors > By Partial Charge… and electrostatic-potential surface colouring | Partial charges, plus the full QM electrostatic potential |
| Input generator **ferric** | Input > ferric | A ferric CLI input (`job.toml` + `job.xyz`) |
| Extensions > ferric > Single Point | | Total energy, plus partial charges stored as `ferric-<model>` |
| Extensions > ferric > Optimize Geometry | | Relaxed geometry, returned as cjson |
| Extensions > ferric > Frequencies | | Harmonic modes, animated in the Vibrations panel |
| Extensions > ferric > Molecular Orbitals | | Basis set and MOs, so Avogadro draws orbitals |
| Extensions > ferric > Configure… | | Defaults: method, basis, charge model, threads, memory |

Single Point, Optimize Geometry, Frequencies and Molecular Orbitals each open
a dialog for the method (HF or a functional: LDA, PBE, BLYP, BP86, PBE0,
B3LYP, r2SCAN) and the basis, pre-filled from **Configure…**; the choice
applies to that run only. Configure… sets those defaults plus the charge
model, thread count and memory budget. The energy model **ferric (QM)** has no
dialog of its own (Avogadro gives energy models none), so it always uses the
Configure… defaults. Charge and multiplicity come from the molecule
(Avogadro's Molecule Properties). If the multiplicity is above 1, the
calculation runs unrestricted (UHF or UKS).

## Install

The plugin is developed in the ferric repository at `plugins/avogadro-ferric`
and mirrored, on its own, at
[github.com/mgoldey/avogadro-ferric](https://github.com/mgoldey/avogadro-ferric).
Either way, `pixi install` in the plugin directory installs `ferric` from PyPI
(wheels for CPython 3.10 to 3.13, manylinux x86_64).

**From Avogadro's online catalog.** A listing in Avogadro's plugin index has
been submitted. Once it is listed, open Extensions > Manage Plugins… and
install `ferric`. Until then, install by hand.

**By hand.** Copy the plugin into Avogadro's plugin directory, install its
environment, then start (or restart) Avogadro and answer Yes when it asks
"Install updated plugins? avogadro-ferric". The menus appear only after that.
In the Flathub build:

```sh
P=~/.var/app/org.openchemistry.Avogadro2/data/OpenChemistry/Avogadro/plugins
git clone https://github.com/mgoldey/avogadro-ferric.git "$P/avogadro-ferric"
flatpak run --command=pixi org.openchemistry.Avogadro2 install \
    --manifest-path "$P/avogadro-ferric/pyproject.toml"
```

On a native (non-Flatpak) Avogadro the plugin directory is
`~/.local/share/OpenChemistry/Avogadro/plugins/`, and `pixi install
--manifest-path …` runs with your own pixi.

**Using a local ferric build instead of PyPI.** Avogadro runs
`pixi run --as-is avogadro-ferric …` in the plugin directory. Any `ferric`
importable in that environment is the one that gets used. After
`pixi install`, install a wheel built from a checkout into it:

```sh
maturin build --release -m crates/ferric-python/Cargo.toml   # in the ferric checkout
pixi run --manifest-path <plugin>/pyproject.toml python -m pip install --force-reinstall --no-deps target/wheels/ferric-*.whl
```

(Flatpak: the wheel has to be somewhere the sandbox can read, such as the
plugin directory.) `python -c "import ferric; print(ferric.__build__)"`
inside the environment tells you which build is loaded.

**ferric version.** The plugin requires ferric 0.1.0rc11 or newer, the first
PyPI release with `BasisSet.shells()` and `DftResult.mo_coefficients()`, which
Molecular Orbitals needs. With ferric 0.1.0rc10 loaded (for example an older
local build), every other command works, and Molecular Orbitals says what is
missing and does nothing. Releases before rc10 lack other calls the plugin
makes, such as `FrequencyResult.normal_modes`, and are not supported.

## Units and conventions (from the avogadrolibs 2.1.0 source)

- **Energy model.** It replies in kJ/mol and kJ/mol/Å, and the value is the
  gradient dE/dx (Avogadro negates it to get forces). ferric's values are
  converted with ferric's own constants: 1 Hartree = 2625.49962 kJ/mol and
  1 Bohr = 0.52917721092 Å (CODATA 2010, ferric's `crates/ferric-core/src/units.rs`).
- **Charges** are in units of e. **Potentials** are in e/Å, the unit of
  Avogadro's own Σ q/r fallback (`calc/chargemodel.cpp`). That is ferric's
  Hartree/e × 1.8897.
- **`properties.totalEnergy`** is in Hartree; Avogadro labels it
  "Total Energy (Hartree)". Orbital energies are in eV, as the orbital
  table displays them.
- **Vibrations.** `eigenVectors` are Cartesian, mass-unweighted displacement
  vectors, one flat 3N list per mode, scaled to unit length. Avogadro animates
  them as Å displacements and does not normalise them itself. Frequencies are
  in cm⁻¹, with imaginary modes shown as negative. There are no IR intensities,
  because ferric computes no dipole derivatives here.
- **Orbitals.** Avogadro and ferric order and normalise basis functions
  differently. libint2 orders spherical functions m = −l…l; Avogadro uses
  0, +1, −1, +2, −2, …. libint2 gives every Cartesian component the xⁿ
  normalisation; Avogadro normalises each component separately. The plugin
  does not hard-code a table for this. For every shell it measures the map
  between ferric's functions (`eval_ao_on_points`) and a line-by-line port of
  Avogadro's evaluator, and it refuses to export if the fit is not exact.
  s, p, d and f shells are supported (cartesian or spherical), for
  closed-shell RHF/RKS only. Avogadro's d and f conventions are Molden's, and
  ferric's Molden writer (`ferric.molden.ao_map`) builds the same map from
  tables; ferric's test suite checks the two agree shell by shell. The plugin
  keeps its own fit because it supports released ferric wheels
  (`ferric>=0.1.0rc11`), which do not have `ferric.molden`.

## Limits

- **AutoOptimization has a 30-second limit per step.** Avogadro waits that
  long (Qt's `waitForReadyRead()` default) for each energy+gradient reply.
  A reply that arrives later would be read as the answer to the next step, so
  the plugin does not send a late reply (see `energy_model.py`). Use a small
  basis or method for interactive optimisation; ferric's own Optimize
  Geometry command has no time limit.
- **Charges and potentials have a 10-second limit.** Avogadro kills them after
  10 s (`PythonScript::execute()`), including pixi and Python start-up, and a
  plugin cannot change that. Run Single Point first on a molecule that is too
  large for 10 s: it has no limit and fills the result cache (below).
- **No warm start in the energy model.** Its energy and gradient come from
  `run_qmmm`, which has no starting-density argument, so every step starts from
  the MINAO guess. Optimize Geometry (`run_optimize_qmmm`) does warm-start each
  step internally.
- **Closed shell only for charges, ESP and orbitals.** ferric's evaluators
  take RHF/RKS results only. Open-shell molecules get an error message for
  these, and an energy from Single Point.
- Elements H to Kr (Z = 1 to 36). No unit cells.
- Energy and gradient for all four references go through `run_qmmm` with every
  atom in the QM region. With an empty MM region this is ferric's gas-phase
  SCF; the tests check it against `run_rhf` / `run_uhf`. Kohn-Sham steps use
  f64 RI-J storage, because the f32 default jitters the surface by about 0.3 %
  of the gradient. That jitter is enough to fail the finite-difference check.

## Configuration file

Configure… writes `$XDG_CONFIG_HOME/avogadro-ferric/config.json`, or
`$AVOGADRO_FERRIC_CONFIG` if that is set. In the Flatpak this is
`~/.var/app/org.openchemistry.Avogadro2/config/avogadro-ferric/config.json`.
The threads setting sets `RAYON_NUM_THREADS`, and the memory setting sets
`FERRIC_MEM_BUDGET_GB`.

## Result cache

Single Point (closed shell) stores its converged density and the Löwdin,
Hirshfeld and CHELPG charges, the three models in Avogadro's charge menu, in
`$XDG_CACHE_HOME/avogadro-ferric/` (`~/.cache/...`; `$AVOGADRO_FERRIC_CACHE`
overrides). A later charges call for the same molecule and settings returns
the stored charges without an SCF. A potentials call starts its SCF from the
stored density and needs at most 2 iterations. Avogadro picks the surface
points, so the potentials themselves cannot be stored in advance. Molecular
Orbitals also starts from the stored density.

An entry matches only the exact coordinates Avogadro sent, the same
charge, multiplicity, method and basis, and the same ferric build. Moving an
atom or changing the configuration is a miss and recomputes. The charge models
use the configured method and basis, so a Single Point run with a different
method chosen in its dialog does not fill the entry they read. A cache hit
never changes a result; it only skips work. The directory keeps the 32 most
recent entries, and deleting it is always safe.

## Tests

From the plugin directory:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
```

The tests drive `python -m avogadro_ferric` exactly as Avogadro does: one
subprocess per call, JSON on stdin, the text energy protocol for the energy
model. They need an importable `ferric`, and they do not need Avogadro. They
cover:

- energy and gradient consistency by central finite differences of the
  reported kJ/mol energies (RHF, RKS and UHF)
- the absolute energy factor
- charges summing to the total charge
- the ESP far field equal to q/r in e/Å
- generated inputs run by the real ferric CLI, including a check that an
  unknown key fails, so the parse gate is shown to be able to fail
- the optimized geometry is stationary
- the frequencies equal ferric's, and the modes are unit-length, keep the
  centre of mass fixed and are orthogonal in the mass metric
- the exported orbitals, evaluated Avogadro's way, equal
  `ferric.eval_mo_on_points` to 1e-10

`tests/` also pins the measured basis-function maps.

Mutation checks (2026-11-09): each mutation below was applied by itself, and
the test named next to it was run and failed. Every one was caught.

| Mutation | Test that caught it |
|---|---|
| Gradient length factor inverted (× Bohr instead of ÷ Bohr) | the finite-difference test |
| Gradient left in Hartree/Bohr | the finite-difference test |
| Gradient sign flipped | the finite-difference test |
| Energy factor wrong (gradient consistent with it) | the absolute-energy test |
| ESP left in e/Bohr | the ESP-units test |
| ESP points not converted to Bohr | the ESP-units test |
| Modes not normalised | the frequency test |
| Modes left mass-weighted | the frequency test |
| `moCoefficients` written AO-contiguous | the orbital test |
| Per-shell map transposed | the orbital test |
| Orbital energies left in Hartree | the orbital test |
| Kohn-Sham steps on f32 RI-J | the finite-difference test (0.28 % mismatch) |
| Unrestricted runs requested as restricted | the open-shell single-point test |
| Reply numbers formatted with `repr()` | the reply-format test, under numpy 2 |

The `repr()` mutation fails only under numpy 2, where an `np.float64` reprs as
`np.float64(...)`, which Avogadro cannot parse. That bug was found in the
Flatpak smoke run.

## Verification log (2026-11-09, Avogadro 2.1.0 Flathub, avogadrolibs tag 2.1.0)

What was checked in the real application, and how:

- **Install.** The flatpak's bundled pixi 0.82.1 ran `pixi install`
  successfully in the plugin directory, inside the sandbox. It installed
  CPython 3.13, numpy 2.5 and ferric 0.1.0rc10 from PyPI. Every one-shot
  feature was then run inside the sandbox exactly as Avogadro launches it:
  `pixi run --as-is avogadro-ferric <id> … --lang en_US`, with the plugin
  directory as the working directory and JSON on stdin. Single Point,
  charges, potentials, Configure (dialog and save), Frequencies and the input
  generator all returned valid replies. Orbitals reported, as intended, that
  rc10 lacks `BasisSet.shells()`.
- **Energy model, driven by Avogadro itself.** Avogadro was started headless
  (`QT_QPA_PLATFORM=offscreen`, `--rpc-name`, `--skip-dialogs`) and driven
  over its JSON-RPC socket. Avogadro's log shows it launching
  `pixi run --as-is avogadro-ferric ferric --debug --lang en_US` with the
  bootstrap JSON.
  - `listForceFields` listed `ferric (QM)` for water.
  - `energy` and `forces` agreed with ferric called directly
    (PBE/def2-SVP): to 1e-10 kJ/mol for the energy, and to 4e-11 kJ/mol/Å
    for the forces, with forces = −gradient.
  - `optimize` (Avogadro's own optimizer on the ferric surface) converged in
    27 steps and 8 s, to a geometry within 3e-5 Å and 0.005° of ferric's own
    `run_optimize_qmmm` minimum, and 6e-6 kJ/mol from its energy.
- **Not verified in the GUI**:
  - the menu commands, charge models and input generator *inside* Avogadro.
    Package menu commands are not RPC commands, and the offscreen app cannot
    answer the "Install updated plugins?" dialog. Their wire format is checked
    against the source and by the subprocess tests.
  - Avogadro's own TOML parse of this pyproject. For the RPC run, the
    package-cache entry that `PackageManager::saveToCache()` writes was
    written by hand. It came from this pyproject (same keys, metadata copied
    verbatim, as `parsePackage()` does). `Avogadro.conf` was restored
    afterwards, so Avogadro's own install runs at the next launch.
  - orbital rendering. Orbitals were checked against a Python port of
    Avogadro's evaluator, not the C++ one.

- **ferric 0.1.0rc11 from PyPI (2026-11-10).** All 46 plugin tests pass
  against the published rc11 wheel (`dirty: False`, CPython 3.12) in a clean
  venv, the orbital tests included; under rc10 those were skipped.

Manual check (about 5 minutes):

1. Start Avogadro. When it asks "Install updated plugins? avogadro-ferric",
   answer Yes. Then confirm that Extensions > ferric has Single Point,
   Optimize Geometry, Frequencies, Molecular Orbitals and Configure….
2. Build water. Run Extensions > ferric > Single Point. A dialog should show
   the energy (PBE/def2-SVP, about −76.272 Ha). The charges then appear under
   Apply Colors > By Partial Charge… as `ferric-lowdin`.
3. Run Extensions > ferric > Frequencies. Open the Vibrations panel: there
   should be 3 modes. Play the lowest one and check it animates as a bend,
   not a stretch.
4. Run Extensions > Calculate > Configure… and pick "ferric (QM)". Then run
   Optimize Geometry, or drag an H with the Auto Optimize tool. The geometry
   should relax smoothly to r(OH) ≈ 0.975 Å and a 102° angle.
5. Run Input > ferric…, generate and save `job.toml` + `job.xyz` into one
   folder, and run `ferric job.toml` there.
6. Run Extensions > ferric > Molecular Orbitals. The Orbitals panel should
   list the MOs and draw the HOMO.

Removing the install:
- the plugin directory
  `~/.var/app/org.openchemistry.Avogadro2/data/OpenChemistry/Avogadro/plugins/avogadro-ferric`,
  which holds a copy of this directory plus its `.pixi/` environment
  (531 MB) and `pixi.lock`
- pixi's package cache at `~/.var/app/org.openchemistry.Avogadro2/cache/rattler`
  (542 MB; shared by every pixi plugin of this flatpak)
- once Avogadro has registered the plugin, its `[plugins]` entry in
  `~/.var/app/org.openchemistry.Avogadro2/config/OpenChemistry/Avogadro.conf`
