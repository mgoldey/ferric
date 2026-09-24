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

**Why:** `run_dft`, `run_ksdft` and the CLI's `ksdft` density-fit Coulomb
(RI-J) with `def2-universal-jkfit` by default, and, for functionals with exact
exchange (hybrids and range-separated hybrids), exchange (RI-K) with the same
basis. `run_rhf` builds exact four-centre J and K unless you pass `df_j_aux`/`df_k_aux`. The
difference is the fitting error that density fitting always carries, not a
bug. Against exact J at PBE/STO-3G it is 0.28 kcal/mol for water, 1.16 for
benzene and 9.5 for a 71-atom drug molecule.

**Do:**

- To compare with an exact-Coulomb code (ORCA with `NORI`, PySCF without
  `density_fit()`), pass `df_j_aux="exact"` to `run_dft` or `run_ksdft`, and
  `df_k_aux="exact"` too for a hybrid. `""`, `"none"`, `"off"` and
  `"conventional"` mean the same. In the CLI, set `[scf] df_j_aux = ""` and
  `df_k_aux = ""`. The CLI's `SCF J/K` log line then reads `RI-JK via` with a
  blank name; the run uses exact J and K.
- Or fit on both sides: `density_fit(auxbasis="def2-universal-jkfit")` in
  PySCF, or `run_rhf(..., df_j_aux="def2-universal-jkfit",
  df_k_aux="def2-universal-jkfit")` in ferric, so J and K are both fitted on
  both sides. `run_rhf` takes a basis name or `""` here, not `"exact"`.
- `run_qmmm` (KS methods), `run_gw` and `run_u_gw` with `xc`, `run_tddft`
  with a functional, `run_tdhf_static_polarizability`, `run_double_hybrid`
  and `run_rs_mp2_rpa` always density-fit their reference SCF with
  `def2-universal-jkfit` and have no opt-out.

## Units differ by interface and by accessor

| Quantity | Where | Unit |
|---|---|---|
| Molecule geometry input (`from_xyz`, `from_xyz_string`, `.xyz` files) | Python, CLI | Ångström |
| Coordinates inside the Rust library | Rust | Bohr |
| Attenuation ω: `[mp2] omega` (`att-rimp2`, `rs-mp2-rpa`), `[mp2] mp2v_omega` (`mp2-v`); `omega=` of `run_attenuated_rimp2`, `run_rs_mp2_rpa`, `run_mp2_v`, and `terf_omega=` of `run_rs_mp2_rpa` | CLI TOML, Python | Å⁻¹ |
| Double-hybrid ω: `[dft] omega` (`wb97x-l-v`) | CLI TOML | **Bohr⁻¹** |
| `tune_omega` (bracket and result); `omega=` of `compute_eri3_mo` and `compute_metric_2c` | Python | **Bohr⁻¹** |
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

The CLI runs the `method.kind`s in the
[capability matrix](../reference/capabilities.md), with `task` = `energy`,
`optimize` or `frequencies`. It also has `[qmmm]`, `[cosmo]`,
`[external_potential]` and `[dft] grid_prune`. These capabilities are
Python-only:

| Capability | Python | What the CLI has instead |
|---|---|---|
| CCD | `run_ccd` | nothing |
| CCSD(T) | `run_ccsd_t` | `ccsd` (CCSD only, no (T)) |
| terfc-attenuated MP2 | `run_terfc_rimp2` | `att-rimp2` (erfc form) |
| Transition-state search, IRC | `run_saddle`, `run_irc` | no `task` for either |
| IEF-PCM solvation | `run_rhf(solvent=...)`, `run_pdep_rpa(solvent=...)` | no `[pcm]` section; `[cosmo]` is the separate conductor-limit model |
| Following an unstable SCF solution downhill | `run_uhf(stability_descent=True)` | `[scf] check_stability` reports a saddle and does not follow it |
| QM/MM MM forces, full-system gradient and optimization, smeared charges, Thole polarization, an MM force field | `run_qmmm`, `run_optimize_qmmm`, `QmmmSystem`, `MmTopology` | `[qmmm]` embeds the QM region in fixed point charges from a PQR file, with link atoms and boundary schemes |
| Constrained DFT and electron-transfer couplings | `run_cdft`, `CdftConstraint`, `cdft_coupling` | nothing |

## Examples need the repository

`examples/*.toml` point at molecules under `testdata/`, and the pipeline tools
live under `tools/`. Neither ships in the wheel. Clone the repository and run
from its root; see [Installation](./installation.md#what-the-wheel-does-not-contain).

## `[memory] budget_gb` does not cap the whole process

**What happens:** a job's resident memory can exceed `budget_gb`, and a job
whose budget is too small usually still runs, slower, rather than failing.
Benzene PBE/def2-SVP with `budget_gb = 0.002` finishes with the same SCF
iterations, a peak RSS of 157 MB, and 11.2 s of wall time instead of 3.5 s.

**Why:** the CLI turns `budget_gb` (or, when unset, 0.8 × available RAM) into
one process-wide ledger. The large, size-dependent allocations reserve their
bytes from it before allocating: three-index RI tensors, the DFT grid's AO
cache, MO-transformed RI blocks, and the large tensors of the MP2, RPA, GW
and CC methods. Two such allocations alive at the same time therefore cannot
each claim the whole budget. Nothing else is charged: basis-sized matrices
(Fock, density, DIIS history), integral engines, BLAS and per-thread scratch,
allocator overhead. The spill path's two scratch blocks are reported but not
refused, and together they can reach about twice the budget.

When a charged allocation does not fit, the result depends on the allocation:

- The SCF's three-index tensor is spilled to a file in `$TMPDIR` (`/tmp` by
  default) and reread on every iteration.
- The DFT grid AO cache is recomputed at every Fock build instead of stored.
  The energy is bit-identical.
- An allocation with no fallback stops the job with an error naming it. For
  some methods this check comes after the SCF. Benzene RI-MP2/def2-SVP with
  `budget_gb = 0.001` completes the SCF, then exits with
  `RI-MP2 MO-side blocks ... requires 0.01 GB; budget is 0.00 GB`.

Some stages print a warning when resident memory passes 110% of the budget.
The warning never stops the run.

From Python, `memory_budget_gb=` sets the same per-allocation limits, but no
shared ledger is installed. Each check compares its own allocation with the
whole budget, not with what the other allocations have left. Two checks
instead subtract the process's current RSS and allow 90% of the remainder: the
KS-DFT grid AO cache (store or recompute) and the UKS Newton/TRAH f<sub>xc</sub>
kernel's second grid cache. The f<sub>xc</sub> check subtracts RSS in the CLI
too.

**Do:** leave room below the machine's real limit for the uncharged part. On
a shared machine, also cap the process externally so a runaway job dies in
its own cgroup, for example with `scripts/ferric-limited -- ferric input.toml`
(defaults `MemoryMax=12G`, `MemoryHigh=10G`, no swap) or
`systemd-run --user --scope -p MemoryMax=12G -- ferric input.toml`.

## Implemented is not validated

A method listed on these pages exists and runs. How closely its numbers have
been checked against an independent reference differs a lot from method to
method. Look up the grade in [What is validated](../reference/validation.md)
before relying on a number.
