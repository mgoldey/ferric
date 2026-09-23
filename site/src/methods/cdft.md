# Constrained DFT

Charge- and spin-constrained DFT, and the electron-transfer couplings that
follow from it.

## Run it

**Rust library only.** There is no CLI section and no Python function. The
entry points are `ferric_scf::cdft_driver::solve_cdft_uhf` (constrained UHF/UKS
on one or more fragment charge or spin constraints) and
`ferric_scf::cdft_coupling::coupling_hab` for the coupling between two
converged diabatic states. See the [Rust API](../reference/api.md).

## Accuracy

No external reference value is stated for these energies or couplings. The
tests check the coupling kernel on synthetic matrices and He₂⁺ identities
(`ferric-scf/tests/cdft_coupling.rs`), probe HeNe⁺ over a distance series
(`cdft_coupling_hene.rs`), and check exact identities on LiH/def2-SVP
(`cdft_uhf.rs`): the constraint is satisfied, λ = 0 reproduces plain UHF, and
the constraint composes with an external point charge. Treat results as unvalidated; see
[Capabilities](../reference/capabilities.md).

## The response connection

A cDFT constraint couples a Lagrange multiplier \\( \lambda \\) to a
fragment-weighted density operator. The derivative

\\[ \frac{\partial N}{\partial \lambda} \\]

— how much charge moves per unit constraint potential — **is a susceptibility**.
So cDFT probes the same object as [RPA and GW](./rpa-gw.md) and
[attenuated MP2](./mp2.md), through a different coupling.

## Implementation

- **Fragment charge and spin constraints** via a grid-Becke weight operator
- A **nested Lagrange-multiplier solve** (Wu–Van Voorhis): an inner SCF at fixed
  \\( \lambda \\), an outer Newton iteration on \\( \lambda \\) itself

The nesting is what makes cDFT more expensive than a plain SCF — each outer step
is a full converged inner solve.

## Electron-transfer coupling

Once you have two charge-localized diabatic states, the coupling
\\( H_{ab} \\) between them follows from a **non-orthogonal determinant
overlap**, computed via Löwdin biorthogonalization.

That gives the matrix element governing electron-transfer rates in Marcus
theory, from states that are constructed rather than guessed.

## A caveat

The `cdft_lambda_tol` convergence tolerance interacts with the coupling
calculation in a way worth checking: a loosely converged \\( \lambda \\)
produces diabatic states that are not quite the ones you asked for, and
\\( H_{ab} \\) inherits that error. Tighten it before trusting a coupling.

## Cite

Wu & Van Voorhis 2006 (electron-transfer coupling from cDFT); Becke 1988
(the fragment weight partition). Full entries in
[References](../reference/references.md).
