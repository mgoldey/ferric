# Sharp bits

Things that behave differently from what you might expect, each of which has
cost someone a run. This page follows the pattern of JAX's "Sharp Bits": each
entry says what surprises people, why ferric does it that way, and what to do.

## Set `OPENBLAS_NUM_THREADS=1`

**What happens:** with OpenBLAS threading enabled, jobs slow down, and under some
workloads crash inside LAPACK.

**Why:** ferric parallelizes with rayon at the outer level. OpenBLAS threads
started inside rayon workers oversubscribe the cores and are not safe in every
LAPACK routine.

**Do:** export `OPENBLAS_NUM_THREADS=1` for every run, test and benchmark. For
throughput across many molecules, run many single-threaded processes.

## `converged` is a flag, and it does not mean "ground state"

**What happens:** `run_rhf`, `run_uhf` and `run_rohf` return a result object
even when they hit `max_iter`; they do not raise. (`run_dft`, `run_ksdft` and the
correlated drivers behave differently: they raise if their reference SCF does
not converge.)

**Why:** a non-converged state is still useful for diagnosis, and raising would
discard it.

**Do:** check `result.converged` in Python, or `converged  = true` in CLI
output, before using any number, and treat `False` as "no result". `True` means
the SCF reached *a* stationary point, not necessarily the lowest one; open-shell
molecules especially can land on an excited solution. Comparing UHF with ROHF, or
a different starting guess, is a cheap check. See
[Python bindings](./python.md#check-converged-and-know-what-it-means).

## `energy_conv` is a sanity bound, not a target

**What happens:** setting `[scf] energy_conv` very tight (say `1e-10`) can make
a KS-DFT calculation never converge.

**Why:** convergence is driven by the density criterion (`density_conv`). The
energy change between iterations has a noise floor set by the integration grid
and density fitting, and a tight energy bound can sit below it. The CLI default
is deliberately loose (`1e-3`).

**Do:** tighten `density_conv`, not `energy_conv`.

## Density fitting is on in `run_dft` but off in `run_rhf`

**What happens:** comparing a ferric DFT energy with a code that uses exact
Coulomb gives a discrepancy that grows with molecule size.

**Why:** `run_dft` uses RI-J by default; `run_rhf` builds exact four-centre J
unless you ask for fitting. The difference is the fitting error, not a bug.
The source records it as 0.28 (water), 1.16 (benzene) and 9.5 kcal/mol (a
71-atom drug molecule) at PBE/STO-3G against conventional J.

**Do:** pass `df_j_aux="exact"` to `run_dft` when comparing against an
exact-Coulomb reference, or set the same fitting on both sides.

## Units differ by interface and by accessor

| Quantity | Where | Unit |
|---|---|---|
| Molecule geometry input (`from_xyz`, `from_xyz_string`, `.xyz` files) | Python, CLI | Ångström |
| Coordinates inside the Rust library | Rust | Bohr |
| Range-separation ω (`omega`) | CLI TOML, Python | Å⁻¹ |
| Range-separation ω | Rust configs | Bohr⁻¹ |
| `QmmmSystem(...)` coordinates | Python | Ångström |
| `QmmmSystem.point_charges()` | Python | **Bohr** |
| `QmmmSystem.link_atom_positions()` | Python | **Ångström** |
| Energies | everywhere | Hartree |

**Do:** check the docstring of any coordinate accessor before converting it.
Converting both QM/MM accessors "to be safe" makes one of them wrong by a
factor of 1.89.

## Energies take a `BasisSet`; geometry changes take a basis *name*

**What happens:** `run_rhf(mol, ferric.BasisSet.bundled("sto-3g"))` works, but
`run_optimize` wants the string `"sto-3g"`.

**Why:** entry points that move atoms (optimization, frequencies, saddle search)
build the basis themselves for each geometry, so they take its name.

**Do:** check the signature in [Python bindings](./python.md).

## Not everything is on the CLI

Several capabilities exist only in Python: for example CCD and CCSD(T)
(`run_ccd`, `run_ccsd_t`), transition-state search and IRC (`run_saddle`,
`run_irc`). The
[capability matrix](../reference/capabilities.md) marks each one.

## Examples need the repository

`examples/*.toml` point at molecules under `testdata/`, and the pipeline tools
live under `tools/`. Neither ships in the wheel. Clone the repository and run
from its root; see [Installation](./installation.md#what-the-wheel-does-not-contain).

## `[memory] budget_gb` does not cap the whole process

**What happens:** a job with `budget_gb = 8` can use more than 8 GB.

**Why:** the budget controls how large three-index integral blocks may be.
Other allocations, such as DFT grid work, are not counted against it.

**Do:** on a shared machine, also cap the process externally, for example with
`systemd-run --user --scope -p MemoryMax=12G -- ferric input.toml`.

## Implemented is not validated

A method listed on these pages exists and runs. How closely its numbers have
been checked against an independent reference differs a lot from method to
method. Look up the grade in [What is validated](../reference/validation.md)
before relying on a number.
