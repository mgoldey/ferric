# Reproducibility and validation grades {#reproducibility-and-grades}

<!-- status: edit; registry: none -->

A dataset needs two guarantees from the code that produced it. The first is
that the same input gives the same numbers, so that a regenerated sample, or
one computed on another machine, can be compared with the original. The
second is that each number means what its name says, to a stated accuracy.
Without the first, noise from the run enters the labels. Without the second,
a model can fit a quantity that was never computed correctly. This section
covers the first through ferric's threading and precision defaults (@sec:reproducibility-and-grades-bitwise-reproducibility,
@sec:reproducibility-and-grades-precision-convergence-defaults-move), and the second through the validation grades (@sec:reproducibility-and-grades-quantity-carries-grade).

## Bitwise reproducibility {#reproducibility-and-grades-bitwise-reproducibility}

Where does run-to-run variation come from? Floating-point addition is not
associative, so a sum computed in a different order can differ in the last
bits. A parallel reduction that splits its work by thread count changes that
order when the thread count changes. ferric runs OpenBLAS at two thread
counts, and the two behave differently.

| Region | BLAS threads | Set by |
|---|---|---|
| parallel regions (integrals, J/K builds, XC grid, frequency quadrature) | 1, unless `OPENBLAS_NUM_THREADS` is set | leave `OPENBLAS_NUM_THREADS` unset |
| serial sections (Fock diagonalization, DIIS, orthogonalizer, density GEMMs) | the rayon thread count by default | `[parallel] blas_threads`, `ferric.set_blas_threads(n)`, `FERRIC_BLAS_THREADS` |

Reductions in ferric's own parallel loops fold in a fixed ascending order,
so they are bit-identical across `RAYON_NUM_THREADS`
(`site/src/reference/architecture.md`). OpenBLAS splits its reductions by
thread count, so the serial sections are not: two runs with different serial
BLAS counts differ by about 1e-13 Ha (`site/src/using/sharp-bits.md`).

| Goal | Setting |
|---|---|
| bit-identical across thread counts (same binary, same CPU family) | `[parallel] blas_threads = 1` or `ferric.set_blas_threads(1)` |
| many molecules at once | many single-threaded processes (`RAYON_NUM_THREADS=1`) |
| several jobs from Python threads in one process | `ferric.set_blas_threads(1)` first; the BLAS count is process-wide |

Two per-property tests pin this for the features of this chapter:
`ferric-rpa/tests/p5_parallel_determinism.rs::esp_at_atoms_bit_identical_across_thread_counts`
and
`ferric-rpa/tests/p5_parallel_determinism.rs::becke_dynamic_polarizability_bit_identical_across_thread_counts`.

Bit-identical across thread counts is not bit-identical across builds or
CPUs. A different compiler, OpenBLAS build or instruction set changes the
kernels themselves. Record `certificate.build.commit` with each sample, and
expect last-digit differences between OpenBLAS builds.

## Precision and convergence defaults that move labels {#reproducibility-and-grades-precision-convergence-defaults-move}

A few defaults are chosen for single-point speed. They change labels by more
than rounding, so a dataset should set them on purpose.

| Default | Value | Effect on a dataset |
|---|---|---|
| RI-J tensor storage, single points | f32 (`[scf] ri_j_precision`) | the energy surface jitters by up to about 1e-6 Ha between nearby geometries (`site/src/reference/input.md`) |
| RI-J tensor storage, `optimize` / `frequencies` tasks | f64 | smooth surface |
| Python `run_rhf` `density_conv` | 1e-6, with the largest element gated at 10× | sets the density, hence every property's, convergence |
| Python `run_rhf` `energy_conv` | 1e-3 | a sanity bound, not the convergence target |
| CLI `pdep-rpa` reference | RI-JK (`def2-universal-jkfit`) | differs from Python's exact-J/K default by the fitting error (section @sec:npz-bundle-every-array-computed) |

For energy and force labels on a set of nearby geometries (a scan, an MD
trajectory, a conformer set), set `ri_j_precision = "f64"` explicitly. The
f32 default is chosen for single points, and its jitter is the size of the
energy differences such a dataset is meant to resolve.

## Which quantity carries which grade {#reproducibility-and-grades-quantity-carries-grade}

A label without a grade is unsafe to train on. A model cannot tell a value
that agrees with an independent code to $10^{-13}$ from one that has only
been checked for internal consistency, and it will fit both with the same
confidence. The certificate grades the capability's energy (section @sec:python-cube-provenance-certificate-provenance-label).
For the features of this chapter, the evidence is the anchor row on the
validation page, summarized here.

| Quantity | Evidence | Agreement | Kind of check |
|---|---|---|---|
| ESP, field at nuclei | PySCF integrals | 4.8e-14 / 1.5e-13 a.u. same density | independent reference |
| Hirshfeld charges | numpy on ferric's grid with PySCF densities and proatoms | 1.9e-13 $e$ same density; grid ≤ 2.0e-4 $e$ | independent reference |
| Mulliken, Löwdin | PySCF integrals contracted with the exported density, in the NPZ row | integral level ≤ 2.8e-14 | independent reference |
| CHELPG, RESP | charge sum, mirror symmetry, reproduction of the surface ESP | — | invariants only |
| static and dynamic α, per-atom α, $C_6$ | numpy Casida sums on PySCF integrals | ≤ 6.2e-14 rel. same orbitals | independent reference; method grade `rpa-polarizability-c6` |
| $\alpha_{\mathrm{CT}}$ | identity with α − Σα^A; independent charge-flow form | exact; 1.1e-4 (grid error of the lab dipole) | identity plus reference |
| PDEP eigenvectors | PySCF generalized dielectric eigenproblem | metric 4.3e-12, projector 2.6e-7 | independent reference |
| TS $C_6$ | recomputed from the exported α(iu) and the TS free-atom table | exact | consistency |

The energies themselves carry the capability grade of their `method.kind`
(appendix @sec:appendix-c): the PDEP-RPA energy (`pdep-rpa`) is Proven, and
BSE-TDA and the statically screened TDHF α (`bse-tda`,
`tdhf-static-polarizability`) are Smoke.

All figures are from `site/src/reference/validation.md`, "Anchors" and
"CLI `method.kind` matrix". As in @sec:electrostatics-and-charges-well-these-checked, "same density" isolates ferric's
integrals and contractions, and the "own SCF" figures quoted in sections
@sec:electrostatics-and-charges and @sec:polarizability-pdep-c6 add ferric's
SCF convergence.

The last column says how much a row proves. An independent reference
computes the same definition by another route, so it catches a wrong
formula. An invariant or a consistency check catches only errors that break
the invariant; a CHELPG fit with a wrong lattice would still sum to the
molecular charge.

These grades say ferric computes the stated definition. They do not say the
definition is a good label. A direct-RPA polarizability on an RHF reference,
or a Mulliken charge in a diffuse basis, can be computed to $10^{-13}$ and
still be a poor target. That choice belongs to the model builder.

## Sources

| Source | Used for |
|---|---|
| `site/src/using/sharp-bits.md`, `site/src/using/python.md` | the two BLAS counts and their effect |
| `site/src/reference/architecture.md`, "Determinism" | the fixed reduction order |
| `crates/ferric-core/src/blas_threads.rs` | the setters and their precedence |
| `site/src/reference/input.md`, `[scf] ri_j_precision` | the f32/f64 defaults and the jitter figure |
| `crates/ferric-python/python/ferric/ferric.pyi`, `run_rhf` | the Python SCF thresholds |
| `site/src/reference/validation.md` | every figure in @sec:reproducibility-and-grades-quantity-carries-grade |

Changes from the sources: none.
