# Constrained DFT

Charge- and spin-constrained DFT, and the electron-transfer couplings that
follow from it.

## Run it

**Python and the Rust library; there is no CLI section.**

- `run_cdft(mol, basis_set, constraints, functional=None, ...)` returns a
  `CdftResult`: a constrained UHF solve, or UKS when `functional` names a
  libxc functional other than `"HF"` (`None` and `"HF"`, any case, give UHF).
- `CdftConstraint(atoms, target, kind="charge")` defines one fragment
  constraint. `atoms` are 0-based atom indices. `target` is the electron
  **population** on the fragment, the Becke-weighted trace
  \\( \mathrm{Tr}[W D] \\), not a net charge: \\( N_\alpha + N_\beta \\) for
  `kind="charge"` and \\( N_\alpha - N_\beta \\) for `kind="spin"`. A neutral
  He atom has a charge population of 2.0; He⁺ has 1.0.
- `cdft_coupling(state_a, state_b)` returns a `CdftCouplingResult` with the
  Wu–Van Voorhis coupling `h_ab`, the determinant overlap `s_ab` and the two
  diabat energies `e_a`, `e_b`.

In Rust the entry points are `ferric_scf::cdft_driver::solve_cdft_uhf` and
`ferric_scf::cdft_coupling::coupling_hab`; see the
[Rust API](../reference/api.md).

This example reproduces the HeNe⁺ constrained solution that
`crates/ferric-scf/tests/cdft_outer_loop.rs` pins (E = −130.4021906 Ha,
λ = −2.754 Ha per electron), with the He fragment held at 2.0 electrons:

<!-- doctest: atol=1e-5 -->
```python
import ferric

# HeNe+ doublet at 2.0 Å; hold the He atom (index 0) at 2.0 electrons.
mol = ferric.Molecule.from_xyz_string(
    "2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 2.0\n", 1, 2
)
r = ferric.run_cdft(
    mol,
    ferric.BasisSet.bundled("def2-svp"),
    [ferric.CdftConstraint([0], 2.0, kind="charge")],
    guess="hcore",
    level_shift=0.5,
    max_iter=400,
    lambda_tol=1e-5,
    max_outer=40,
    stability_descent=False,
    grid_radial=99,
    grid_angular=302,
)
print(r.converged)
print(f"{r.energy:.6f}")
print(f"{r.populations[0]:.5f}")
```

```text
True
-130.402190
2.00000
```

`r.energy` is the ordinary UHF/UKS energy at the constrained density, without
the constraint term. `r.lambdas` holds the multipliers (Ha per electron) and
`r.weight_matrix(i)` the AO weight operator of constraint `i`.

What to know before using it:

- **A returned result has a converged λ loop.** When the outer loop exceeds
  `max_outer`, `run_cdft` raises `RuntimeError`. The inner SCF at the final λ
  can still be unconverged, so check `r.converged`: it is true only when the
  inner SCF converged and every constraint is met to `lambda_tol`.
- **`stability_descent` defaults to True here** (in `run_uhf` it defaults to
  False). After the λ loop converges, a constrained saddle point is followed
  downhill to a lower state that still meets the constraint. The example turns
  it off to reproduce the pinned solution; with it on, this HeNe⁺ case ends
  about 0.0245 Ha lower. The descent is skipped for a KS reference.
- **The weight grid defaults to 99 × 302.** It must resolve populations below
  `lambda_tol`; 75 × 110 resolves them only to about 1e-4. Setting
  `grid_radial` or `grid_angular` uses that grid for the XC quadrature too.
- **UKS-cDFT is smoke-level.** The tests validate the UHF path; a libxc
  `functional=` runs UKS with no validated reference.
- **One constraint is the tested case.** With several constraints the outer
  loop is a plain k × k Newton step, without the single-constraint bracket
  safeguard.
- **`cdft_coupling` has strict preconditions** and raises `ValueError` when
  one fails. Each state carries exactly one `kind="charge"` constraint, both
  are converged, and both come from the same molecule, geometry, basis,
  charge and multiplicity **and the same Hamiltonian**: functional,
  `df_j_aux`/`df_k_aux`, `k_builder`, XC grid, point charges and external
  field. Two states that are the same determinant (\\( |S_{ab}| \to 1 \\))
  also raise. The sign of `h_ab` is a determinant-phase convention; compare
  \\( |H_{ab}| \\).

## Accuracy

No external reference value is stated for these energies or couplings. The
tests check the coupling kernel on synthetic matrices and He₂⁺ identities
(`ferric-scf/tests/cdft_coupling.rs`), probe HeNe⁺ over a distance series
(`cdft_coupling_hene.rs`), and check exact identities on LiH/def2-SVP
(`cdft_uhf.rs`): the constraint is satisfied, λ = 0 reproduces plain UHF, and
the constraint composes with an external point charge. The Python tests
(`crates/ferric-python/tests/test_cdft.py`) rerun those configurations
through the bindings and check `cdft_coupling` against an independent
transition-density construction. Treat results as unvalidated; see
[Capabilities and validation](../reference/validation.md#python-entry-points).

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

The λ convergence tolerance (`lambda_tol` in Python, `cdft_lambda_tol` in
Rust, default 1e-5 electrons) interacts with the coupling calculation in a
way worth checking: a loosely converged \\( \lambda \\) produces diabatic
states that are not quite the ones you asked for, and \\( H_{ab} \\) inherits
that error. Tighten it before trusting a coupling. The exception is a
constraint whose population barely responds to \\( \lambda \\), such as He₂⁺
at the localized-hole plateau: there the tolerance has to be loosened to match
that flatness (the tests use 1e-2), or the Newton step drives
\\( \lambda \\) off a cliff.

## Cite

Wu & Van Voorhis 2006 (electron-transfer coupling from cDFT); Becke 1988
(the fragment weight partition). Full entries in
[References](../reference/references.md).
