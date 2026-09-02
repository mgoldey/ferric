# SCF and DFT

Ground-state self-consistent field methods, plus analytical nuclear gradients.

## Hartree–Fock

- **RHF** (closed-shell) with DIIS, Schwarz/QQR screening, and a choice of
  direct or density-fitted (RI-J / RI-K) Fock builds
- **UHF / ROHF** (open-shell) with per-spin DIIS, virtual-space level shifting,
  augmented-Hessian Newton, and **Maximum-Overlap-Method (MOM)** orbital
  tracking for near-degenerate cases

The convergence machinery is not decoration. Heavy atoms and near-degenerate
frontier orbitals genuinely break plain DIIS; the virtual-block level shift and
MOM exist because specific systems failed without them.

## Kohn–Sham DFT

Closed- and open-shell (RKS / UKS / ROKS) via **libxc**:

- LDA, GGA, hybrid, and range-separated-hybrid functionals — LDA, PBE, B3LYP,
  ωB97X-V
- **Becke–Lebedev** grids, with pruning
- **VV10** nonlocal correlation

## Gradients

Analytical nuclear gradients for RHF, UHF, ROHF and KS-DFT — **including grid
response** — validated against finite differences.

Hessians are **not** implemented. The mpqc4 libint2 export does not carry
second-derivative integrals, so the CPKS machinery is stubbed with explicit
TODOs rather than silently absent. See [Installation](../using/installation.md).

## Convergence

A few things worth knowing before debugging a stubborn SCF:

**Check `converged`.** These routines return a result whether or not they
converged; a non-converged SCF is a result with `converged = false`, not an
error. Downstream code that ignores the flag will happily consume a
half-converged density.

**Multiple solutions are real.** For systems like alkane chains, different
initial guesses converge to genuinely different SCF solutions — not a
convergence failure but a different basin. The guess picks the basin.

**Density-fitting has a noise floor.** DF-JK introduces an error floor that
makes energy-based convergence criteria below roughly 1e-9 meaningless; SCF
gates on the density RMS change instead.

**Near-linear-dependence.** Diffuse (aug-) basis sets on close-packed systems
can drive the overlap matrix near-singular; the canonical-orthogonalization
threshold is tunable via `FERRIC_LINDEP_THRESH`.

## Screening

- **QQR** (Maurer, Lambrecht & Ochsenfeld 2012) — distance-including screening
  that is tighter than Schwarz alone
- **LinK** (Ochsenfeld, White & Head-Gordon 1998) — linear-scaling exchange via
  significant-pair lists

## Determinism

The Fock build's reduction folds partial matrices in a **strict ascending group
order**, independent of thread count and of the memory band width. Results are
bit-identical across `RAYON_NUM_THREADS` — a property pinned by tests, not just
intended.

This matters more than it might seem: a tree-fold reduction would be equally
*deterministic* but would produce *different* bits, since floating-point
addition is not associative. The ascending order is load-bearing.
