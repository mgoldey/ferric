# Constrained DFT

Charge- and spin-constrained DFT, and the electron-transfer couplings that
follow from it.

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
