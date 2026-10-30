# Sharp bits

Things that behave differently from what you might expect, each of which has
cost someone a run. This page follows the pattern of JAX's "Sharp Bits": each
entry says what surprises people, why ferric does it that way, and what to do.

## Leave `OPENBLAS_NUM_THREADS` unset

**What happens:** the `ferric` CLI and `import ferric` pin OpenBLAS to one
thread when `OPENBLAS_NUM_THREADS` is unset, and honour the variable when it is
set. (`.cargo/config.toml` sets it to 1 for cargo-launched processes, also
only when it is unset.) A value above 1 is therefore a deliberate choice, and
its effect depends on the workload: it can speed up one calculation and slow
another down several-fold.

**Why:** ferric's parallelism comes from rayon. OpenBLAS threads run inside
rayon's worker threads, so the thread count is roughly BLAS threads × rayon
threads. Measured on benzene with the `ferric` CLI, single runs on a heavily
loaded machine with 6 physical cores (12 hardware threads), so read these as
order-of-magnitude:

| `OPENBLAS_NUM_THREADS` | unset | 1 | 6 | 12 |
|---|---|---|---|---|
| RI-JK RHF (`benzene-rhf-def2-rijk.toml`) | 54 s | 31 s | 9.7 s | timed out at 1800 s |
| PDEP-RPA (`benzene-pdep-rpa.toml`) | 5.9 s | 7.2 s | 22.9 s | 30.7 s |

"unset" and 1 run the same configuration, so the gap between those two
columns is the machine's load, not a setting. B3LYP timings were noisy, with
no clear trend. None of these runs crashed.

Three effects matter beyond speed:

- **Oversubscription can stall a run.** Once BLAS threads × rayon threads
  exceed the physical cores, a job can slow by orders of magnitude, as the
  12-thread RI-JK run shows.
- **Results change slightly.** The BLAS thread count changes the order of
  floating-point summation. The RI-JK energy at 6 threads differs from the
  1-thread energy by 6e-6 Ha; expect differences at the 1e-6 to 1e-5 Ha level.
- **LU routines can crash.** Threaded OpenBLAS LU factorization (`dgetrf`)
  can overflow the 2 MB stacks of rayon's worker threads. ferric calls LU
  solves in BSE, PCM, COSMO, MBD and orbital-rotation steps, among others.

**Do:** leave `OPENBLAS_NUM_THREADS` unset, and ferric runs with one BLAS
thread. If you set it higher for a particular job, never above the number of
physical cores, and check the timing and the energy against a one-thread run
of the same input. For throughput across many molecules, run many
single-threaded processes.

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
an SCF run tens of iterations past a settled density, or never converge.

**Why:** convergence is driven by the density criterion (`density_conv`). The
energy change between iterations has a noise floor, and a tight energy bound can
sit below it, so it is met only by chance. For a closed-shell run with exact J
and K the floor comes from the incremental Fock update and its integral
screening (`integral_thresh`). ferric estimates it as
`0.0125 * integral_thresh * nbf^2.25 + 0.5 * eps * |E| * sqrt(nbf)` Ha. The
estimate against the measured RMS of the energy change once the density has
settled:

| system | nbf | estimated floor | measured RMS |
|---|---|---|---|
| water / cc-pVDZ | 24 | 1.6e-11 | 4.5e-12 |
| benzene / cc-pVDZ | 114 | 5.3e-10 | 1.4e-10 |
| alkane C48 / STO-3G | 338 | 6.1e-9 | 2.2e-9 |
| 71-atom drug-like molecule / STO-3G | 239 | 2.8e-9 | 1.9e-9 to 2.7e-9 |

The estimate is good to a factor of a few and errs high for large systems; it
was fitted on systems up to about 340 basis functions, and beyond that it is an
extrapolation. With density-fitted J (and in the open-shell solvers, which
rebuild J and K every iteration) only rounding remains, about
`0.5 * eps * |E| * sqrt(nbf)`: measured 3e-13 for benzene/cc-pVDZ with RI-JK and
1.3e-12 for alkane C20/6-31G with RI-JK. The CLI default is deliberately loose
(`1e-3`).

**What ferric tells you:** when you set `energy_conv` below 0.3 times the
estimated floor, the CLI prints `warning: [scf] energy_conv ... is below the
estimated noise floor of dE ...` to stderr and the Python `run_rhf`, `run_uhf`
and `run_rohf` emit a `RuntimeWarning`, once, before the SCF starts. A default
`energy_conv` is never warned about. Nothing else changes: the threshold is not
clamped and the SCF stops exactly where it otherwise would. A run that ends
without converging on the `dE` criterion alone repeats the statement in
`RhfResult.convergence_summary`. No estimate is made for functionals (grid
noise), the LinK and COSX exchange builders, PCM / COSMO, polarizable embedding
or non-default `screening`: those runs get no warning, which is not a statement
that the tolerance is reachable. The `|E|` in the rounding term is estimated
from the nuclear charges (ECP cores removed, about 5 % for all-electron
molecules, 3.7x high for an iodine ECP case); the point-charge or field energy
of an embedding is not included.

**A symptom of the same floor:** with a large point-charge field (`point_charges=` on `run_rhf`'s closed-shell direct-J/K path) the number of SCF iterations can differ between two orderings of the same charges, because a change of a few ulps in the one-electron potential decides which iteration the `dE` test happens to pass. On rc7 a 6458-charge field took 64 iterations nearest-first and 111 in file order at `energy_conv=1e-10`. The potential itself changes only by roundoff (3e-15 to 1e-14 relative) and the converged energy by about 1e-12 Ha. Main turns the incremental update off when `energy_conv < 1e4 * integral_thresh` or `density_conv < 1e3 * integral_thresh` (1e-8 and 1e-9 at the default `integral_thresh`), where all measured orders took 20 iterations; at the default `energy_conv` and `density_conv` every measured order took the same number (15 iterations on the field and orders tried; the tests do not assert iteration counts, so this is a measurement, not a guarantee for other fields). `run_uhf` and `run_rohf` use a separate open-shell incremental path and `run_dft`'s default RI-J does not use the direct incremental path: for those the order changes the potential by roundoff too, but iteration counts were not measured. ferric does not reorder the charges.

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
[capability matrix](../reference/validation.md#cli-methodkind-matrix), with `task` = `energy`,
`optimize` or `frequencies`. It also has `[qmmm]`, `[cosmo]`, `[pcm]`,
`[external_potential]`, `[scf] stability_descent`, `[mp2] att_operator` and
`[dft] grid_prune`. These capabilities are Python-only:

| Capability | Python | What the CLI has instead |
|---|---|---|
| Transition-state search, IRC | `run_saddle`, `run_irc` | no `task` for either |
| QM/MM MM forces, full-system gradient and optimization, smeared charges, Thole polarization, an MM force field | `run_qmmm`, `run_optimize_qmmm`, `QmmmSystem`, `MmTopology` | `[qmmm]` embeds the QM region in fixed point charges from a PQR file, with link atoms and boundary schemes |
| Constrained DFT and electron-transfer couplings | `run_cdft`, `CdftConstraint`, `cdft_coupling` | nothing |

## Examples need the repository

`examples/*.toml` point at molecules under `testdata/`. Neither ships in the
wheel. Clone the repository and run
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
allocator overhead. The spilled RI-JK path sizes its dressing and read-back
widths to the budget, with floors of a few hundred aux rows. A budget below
those floors overshoots with a warning that says which phase and by how much.

When a charged allocation does not fit, the result depends on the allocation:

- The SCF's three-index tensor is stored whole when `n_aux × n_bf² × 8` bytes
  fit. When only the symmetric half (`n_aux × n_bf(n_bf+1)/2 × 8` bytes, plus
  64 unpacked aux rows of scratch) fits, it is kept in memory as that packed triangle
  and the RI-J build reads the packed rows directly. When even that does not
  fit, it is either spilled to a file in `$TMPDIR` (`/tmp` by default) and
  reread on every iteration, or recomputed on every iteration without
  writing anything. `auto` chooses between those two from a rate measured on
  the machine; `[scf] jk_storage` forces `memory`, `disk` or `direct`.
  Holding the K dressing to a small budget takes several passes over the raw
  tensor, and a run making many says so. At `OPENBLAS_NUM_THREADS=1` the
  dressed tensor and K do not depend on whether they spilled, so running short
  of memory costs IO, not bits. With threaded BLAS that does not hold.
- A dressed tensor that fits the budget but not beside the resident raw tensor
  spills instead of stopping the job.
- A DF-guess pre-stage that runs out of memory is abandoned with a warning, and
  the SCF starts from its usual guess. An explicit `jk_storage = "memory"` that
  does not fit, and any memory failure under MPI, stop the job instead.
- The DFT grid AO cache is recomputed at every Fock build instead of stored.
  The energy is bit-identical.
- An allocation with no fallback stops the job with an error naming it. For
  some methods this check comes after the SCF. Benzene RI-MP2/def2-SVP with
  `budget_gb = 0.001` completes the SCF, then exits with
  `RI-MP2 MO-side blocks ... requires 0.01 GB; budget is 0.00 GB`.

**Changed once for spilled and recomputed RI-JK runs:** adopting the widths above
moved energies by at most 2.84e-13 Ha against the previous build (water/cc-pVDZ
and butane/6-31G, forced to spill or recompute, `OPENBLAS_NUM_THREADS=1`). In-core
and exact runs did not move. Iteration counts matched, except water/cc-pVDZ under
`jk_storage = "direct"`, which went from 13 to 11; that case already varied
between 11 and 13 from run to run at 2 threads on the previous build.

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

## GPU offload is opt-in and may not help

The CUDA backend is not in the PyPI wheel; it needs a source build with the `gpu` feature (see [GPU (CUDA)](installation.md#gpu-cuda)), and even then it is off by default (`FERRIC_GPU=auto` or `[gpu] mode` turns it on). Consumer GeForce cards run FP64 at 1/32 of their FP32 rate or less, so offloading f64 contractions may be slower than the CPU once transfer time is counted; whether it helps depends on the card and the problem, and has to be measured on your own hardware.

The Kohn-Sham XC grid runs on the device only under `mode = "on"` (or `gpu = "on"` / `"mixed"`), never under `auto`: on a GTX 1080 every device XC build took 2.2-3x the CPU time and whole PBE SCFs ran about 2x slower, so `auto` keeps it on the CPU rather than slowing every DFT run on a GPU box.

Device memory is a separate pool (`[gpu] memory_gb`); the host `[memory] budget_gb` is not reduced by it. A contraction that does not fit the device pool runs on the CPU and is counted, not refused. `FERRIC_GPU_TRACE=1` prints each fallback.

`[gpu] precision = "mixed"` changes the numbers, not just the speed: the audit line and `ferric.gpu_status()["precision"]` say which precision a run used, and the per-kernel error map is on the [validation page](../reference/validation.md#mixed-precision-gpu-kernels). The one shipped mixed kernel is `rimp2-energy` (f32-resident `B_ov`; closed-shell and unrestricted RI-MP2 under the Coulomb, erfc and terfc operators); its measured energy errors are 3e-11 to 7e-9 Eh closed-shell Coulomb (total, with the OS and SS parts up to 5e-9 Eh each), 5e-10 to 1.4e-8 Eh unrestricted Coulomb (total, with the αα, ββ and αβ blocks up to 1.1e-8 Eh each) and 7e-11 to 6e-9 Eh for erfc and terfc on the systems listed there, graded as a measured error map, not Proven; a mixed run prints `precision mixed` in its `[ferric] gpu:` line, and its energies are not interchangeable with f64 results in a convergence or reference comparison. A tiny problem still pays the device upload when the mode is on, whatever the precision. `gpu = "mixed"` (or `[gpu] preset = "mixed"`) is `mode = "on"` plus `precision = "mixed"`, so it carries the same change in the numbers; a `mode` or `precision` setting that contradicts the preset is an error, not an override. A `mixed_kernels` list given next to a preset is kept as given instead of the preset's set (it may narrow it); a name this build does not ship is refused, and the opt-in `xc-vxc` is admitted only when the list names it and `mode` resolves to `"on"` (so `gpu = "mixed"` with the list works, `auto-mixed` with it is an error).

## Implemented is not validated

A method listed on these pages exists and runs. How closely its numbers have
been checked against an independent reference differs a lot from method to
method. Look up the grade in [Capabilities and validation](../reference/validation.md)
before relying on a number.
