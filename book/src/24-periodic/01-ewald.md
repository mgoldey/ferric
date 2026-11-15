# 24.1 Lattices, Ewald summation and the Madelung constant

<!-- section: 24.1; status: ready; registry: pbc-ewald-pairft-stress;
     theory: crates/ferric-pbc/theory.md#"1. Ewald nuclear repulsion and the Madelung constant" -->

A periodic energy has to be summed over an infinite lattice, and the bare
Coulomb sum converges only conditionally. The Ewald split writes
$1/r = \operatorname{erfc}(\eta r)/r + \operatorname{erf}(\eta r)/r$, sums the
short-range part in real space and the smooth long-range part in reciprocal
space, and adds two constants that make the total independent of $\eta$. The
same split, with the same bookkeeping of the $\mathbf G = 0$ term, recurs in
the nuclear attraction (section 24.3) and the density fit (section 24.5).

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathbf a_i$, $\mathbf b_i$ | lattice and reciprocal rows, $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$ | `Cell` lattice, reciprocal |
| $\eta$ | Ewald splitting parameter | `omega` |
| $Z_j$, $\mathbf R_j$ | point charges and positions | nuclear charges |
| $S(\mathbf G)$ | structure factor | |
| $v_M$ | Madelung constant | `madelung` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/ewald.rs::{ewald_point_charges,ewald_nuclear_repulsion,madelung_constant}` |
| Mesh constant | `ferric-pbc/src/kpts.rs::madelung` |
| Python | every periodic result exposes `e_nuc` and `madelung` |

<!-- include: crates/ferric-pbc/theory.md#"1. Ewald nuclear repulsion and the Madelung constant" -->

## 24.1.1 Verified equations

> **Verified equation 24.1** · the Ewald sum is independent of $\eta$
>
> The $\eta$-derivatives of the short-range, long-range, self and background
> terms cancel per $\mathbf G \neq 0$ and at $\mathbf G = 0$; a transcribed
> Ewald sum on a charged triclinic toy cell is $\eta$-independent to
> $10^{-12}$, and without the background term it is not.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1, claim `ewald-omega` |
> | Negative control | the runner's mutation: background $-\pi Q^2/(2\Omega\eta^2)$ without the 2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::ewald_point_charges` |
> | Code test | `ewald_matches_pyscf_and_is_omega_independent` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.ewald.omega-independence; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=ewald-omega; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::ewald_point_charges; rust_test=ferric-pbc/tests/pbc_stage0.rs::ewald_matches_pyscf_and_is_omega_independent; registry=pbc-ewald-pairft-stress -->

> **Verified equation 24.2** · the Madelung constant
>
> $v_M = -2E$ for one unit charge per cell with its background reproduces the
> rock-salt constant $1.747564594633182$ and the simple-cubic
> $v_Ma = 2.8372974794806$, independent of $\eta$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1, claim `madelung` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::madelung_constant` |
> | Code test | `madelung_matches_pyscf` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.ewald.madelung; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=madelung; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::madelung_constant; rust_test=ferric-pbc/tests/pbc_stage0.rs::madelung_matches_pyscf; registry=pbc-ewald-pairft-stress -->

## 24.1.2 The algorithm

| Step | Quantity | Cost |
|---|---|---|
| 1 | cutoffs $r_\text{cut} = s/\eta$, $G_\text{cut} = 2\eta s$, $s = \sqrt{\ln(1/\epsilon)} + 1$ | |
| 2 | real-space sum of $Z_iZ_j\operatorname{erfc}(\eta d)/d$ over pairs and translations within $r_\text{cut}$ | $N_\text{atom}^2 N_{\mathbf L}$ |
| 3 | reciprocal sum of $e^{-G^2/4\eta^2}|S(\mathbf G)|^2/G^2$ within $G_\text{cut}$ | $N_\text{atom}N_{\mathbf G}$ |
| 4 | self term $-\eta\pi^{-1/2}\sum Z_j^2$ and background $-\pi Q^2/(2\Omega\eta^2)$ | |

All sums are Neumaier-compensated. The Madelung constant is evaluated at
$\eta = \sqrt\pi/\Omega^{1/3}$; for a mesh it is the constant of the
$\operatorname{diag}(N)$ supercell (`KPointMesh::madelung`, PySCF
`tools.pbc.madelung(cell, kpts)`).

## 24.1.3 Numerical anchors

From the tests named in `crates/ferric-pbc/theory.md` §1.

| Quantity | Reference | Bar | Test |
|---|---|---|---|
| $E_{nn}$ on three charged nuclear cells | PySCF `energy_nuc` | $10^{-10}$ Ha; $\eta$-spread $< 10^{-12}$ at $\eta = 0.8, 1.5, 3$ | `crates/ferric-pbc/tests/pbc_stage0.rs` `ewald_matches_pyscf_and_is_omega_independent` |
| $v_M$ | PySCF and the simple-cubic textbook value | $10^{-11}$ | `madelung_matches_pyscf` |
| rock-salt $E = -8M/a$ | $M = 1.747564594633182$ | $\eta$-spread $< 10^{-12}$ | `ewald_rock_salt_madelung_is_textbook_and_omega_independent` |

## 24.1.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §1 | the sum, its $\eta$-independence, the gradient |
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S1 | the proof |
| P. P. Ewald, *Ann. Phys.* **369**, 253 (1921) | the method |
| `site/src/methods/pbc.md` | the user guide |
