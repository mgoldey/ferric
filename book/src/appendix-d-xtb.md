# Appendix D: The semiempirical interface (xTB)

<!-- section: D; status: gap; registry: none; theory: none -->

ferric can call the GFN family of semiempirical methods through the `xtb`
library, for conformer screening and for relaxing a structure before a
first-principles calculation. The methods themselves are not part of ferric:
ferric passes a molecule across a C interface and takes back an energy, a
gradient and partial charges. This appendix describes that interface. It
derives nothing, and it has no machine-checked equation; the gap status in the
header records that.

| Entry point | Location |
|---|---|
| Rust (calculator) | `ferric-xtb/src/calculator.rs::{XtbCalculator,XtbConfig,XtbMethod,XtbResult}` |
| Rust (one-shot) | `ferric-xtb/src/calculator.rs::{xtb_singlepoint,gfn2_energy}` |
| Rust (drivers) | `ferric-xtb/src/driver.rs::{xtb_energy_gradient,optimize_geometry_xtb,harmonic_frequencies_xtb}` |
| Python | `run_xtb_energy_gradient`, `run_xtb_optimize`, `run_xtb_frequencies` (`crates/ferric-python/ferric.pyi`) |
| CLI | none: no `method.kind` runs xTB |

## D.1 Build and availability

The interface is off unless the `xtb` cargo feature is on. With it off, the
crate has no native dependency and no API surface beyond
`ferric-xtb/src/lib.rs::xtb_available`, which returns `false`. The Python
functions exist in every build and raise `RuntimeError` unless the extension
was built with the feature. The build links libxtb dynamically; the prefix is
`$HOME/.local` unless `XTB_PREFIX` is set, and the loader path must contain it
at run time.

Two build facts matter:

- **Build libxtb at `-O2`.** The crate documentation records that libxtb built
  at `-O3` with gfortran 13.3 returned a GFN1 and GFN2 gradient about 20 times
  too large that disagrees with finite differences, while the energy stayed
  correct. The binding is not at fault, and the same source built at `-O2`
  passes xtb's own unit tests. If gradients look wrong, check how libxtb was
  built before suspecting ferric.
- **Parameter files.** libxtb finds its GFN parameter files through
  `XTBPATH`. ferric sets it, once, to the build-time prefix if the variable is
  unset, and never overwrites a value the caller set.

## D.2 What is computed

| Method | Name in code | Kind |
|---|---|---|
| GFN2-xTB | `XtbMethod::Gfn2` (default) | self-consistent tight-binding |
| GFN1-xTB | `XtbMethod::Gfn1` | self-consistent tight-binding |
| GFN-FF | `XtbMethod::GfnFf` | force field; no self-consistent charges |

`XtbMethod::parse_config_str` accepts `gfn1`, `gfn2` and `gfnff`, with the
hyphenated and `-xtb` spellings, and any other string is an error rather than
a silent default. The settings:

| Setting | Default | Notes |
|---|---|---|
| `accuracy` | 1.0 | xtb's scale, from 1000 (loose) to 0.0001 (tight) |
| `scc_max_iter` | 250 | ignored by GFN-FF |
| `electronic_temp` | 300 K | Fermi smearing; ignored by GFN-FF |
| `verbosity` | 0 | silent |

Everything is in atomic units at the boundary and nothing is rescaled:
coordinates in Bohr in, energy in Hartree, gradient $\partial E/\partial\mathbf R$
in Hartree/Bohr with one row per atom, partial charges in units of $e$.
GFN-FF returns no charges (`XtbResult.charges` is `None`).

The molecule's charge is `mol.charge`. The number of unpaired electrons given
to xtb is `mol.multiplicity - 1`.

**What is refused.** `XtbCalculator::new` returns an error for an empty
molecule, for any ghost atom, for any atom that carries an ECP, and for a
multiplicity below 1. xTB has its own minimal valence basis and its own core
treatment, so a basis-only centre or a reduced-charge ECP atom has no
counterpart, and ferric would otherwise compute a different system from the
one asked for. Any error that libxtb queues (an unsupported element, an SCC
that does not converge) comes back as the library's own message.

## D.3 Drivers

| Task | Function | How |
|---|---|---|
| Energy and gradient | `xtb_energy_gradient` | one calculator per call |
| Geometry optimization | `optimize_geometry_xtb` | ferric's BFGS optimizer, with the xTB energy and gradient as the closure; `OptimizeConfig.max_steps` bounds the geometry steps |
| Harmonic frequencies | `harmonic_frequencies_xtb` | central differences of the xTB gradient, $6N$ gradient evaluations, then ferric's shared mass-weighting and translation and rotation projection |

The optimizer and the frequency code are the ones the SCF drivers use
(`ferric-scf/src/optimize.rs`, `ferric-scf/src/frequencies.rs`); xTB supplies
only the surface. The default finite-difference displacement is the SCF
default, $5\times10^{-3}$ Bohr (`DEFAULT_HESSIAN_DELTA`). The input to the
frequency driver should be an xTB minimum, so relax it with
`optimize_geometry_xtb` first and check the `asymmetry` of the result.

An SCC that fails to converge makes the calculator return an error. In an
optimization or a frequency run that error ends the run: the search does not
continue on a density that is not self-consistent.

## D.4 Threading

libxtb keeps process-global state, so it is not thread-safe, and
`XtbCalculator` is deliberately neither `Send` nor `Sync`. A compile-time guard
in `calculator.rs` fails the build if that stops being true. Three rules
follow.

1. Never drive a calculator from inside a rayon parallel iterator.
2. Drive libxtb from one thread at a time for the life of the process.
3. To screen conformers in parallel, use one process per molecule with
   `OMP_NUM_THREADS=1`.

The crate's own tests take a process-wide lock for the same reason.

## D.5 Evidence

The tests need libxtb and run only with the `xtb` feature
(`crates/ferric-xtb/tests/xtb_singlepoint.rs`,
`crates/ferric-xtb/tests/xtb_opt_freq.rs`). They check:

| Check | Reference | Bound in the test |
|---|---|---|
| GFN2, GFN1 and GFN-FF energies of water | the output of the xtb command-line program on the same geometry | $10^{-8}$ Ha |
| GFN2 gradient of water | the `gradient` file written by the xtb program | $10^{-10}$ Ha/Bohr |
| GFN2 gradient | central difference of xTB's own energy | $10^{-6}$ Ha/Bohr |
| Translation of the molecule | energy unchanged, gradient components summing to zero | $10^{-9}$ Ha and $10^{-8}$ |
| Optimization | water converges to a small gradient; a stationary point does not move; an unconverged SCC raises | as asserted in the test |
| Frequencies | water has six and linear CO2 five translation and rotation modes below 30 cm$^{-1}$; 18 gradient evaluations for water; an Hessian diagonal element agrees with the second difference of the energy | as asserted in the test |

The energy reference is the xtb program itself, so the energy test is a check
of the binding, not of the xTB method. These are the bounds written in the
tests, quoted here as such; this appendix quotes no figure from a run of its
own.

**Grade.** `site/src/reference/validation.md` has no row for xTB, the CLI has
no xTB kind, and the Python optimization result carries a certificate graded
`not graded`. Treat xTB results as unproven in the sense of that page. The
checks above support the binding. They do not support a statement about the
accuracy of GFN-xTB for any chemistry.

## D.6 Known limits (2026-11-15)

- Not in the CLI. Python and Rust only.
- Ghost atoms and ECP atoms are refused (D.2).
- GFN-FF writes a `gfnff_topo` file into the current directory when it loads,
  and the C interface has no switch to prevent it. Run from a scratch
  directory.
- No implicit solvation is exposed, and there is no xTB Hessian other than the
  finite difference of the gradient.
- A libxtb built at `-O3` gives wrong gradients (D.1).
- Periodic cells are not passed: the molecule is created with no lattice.

## D.7 Sources

| Source | Used for |
|---|---|
| `crates/ferric-xtb/src/lib.rs` | the feature gate, the `-O2` finding, the threading rules |
| `crates/ferric-xtb/src/calculator.rs` | the methods, settings, units, refusals and `XTBPATH` |
| `crates/ferric-xtb/src/driver.rs` | the optimization and frequency drivers |
| `crates/ferric-xtb/tests/xtb_singlepoint.rs` | the single-point and gradient bounds |
| `crates/ferric-xtb/tests/xtb_opt_freq.rs` | the optimization and frequency checks |
| `crates/ferric-python/ferric.pyi` | the Python signatures and the certificate grade |
