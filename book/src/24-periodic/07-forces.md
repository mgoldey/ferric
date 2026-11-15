# 24.7 Nuclear forces

<!-- section: 24.7; status: edit; registry: pbc-hf, pbc-ks;
     theory: crates/ferric-pbc/theory.md#"6. Refusal gates and tolerances of the open-shell force and stress paths" -->

Moving atom $A$ in a periodic cell moves its nucleus, every basis function
centred on it and, for Kohn-Sham, every grid point homed on it, in every
lattice image. The force is the derivative of the energy of sections
24.1-24.6 at the converged density. Each term of the energy gives one term
of the force, and the Ewald and $\mathbf G = 0$ bookkeeping of sections
24.1-24.4 carries over unchanged.

The forces drive `task = "optimize"`: atoms move, the lattice is fixed.

| Symbol | Meaning | Code name |
|---|---|---|
| $D_\sigma$, $D = D_\alpha + D_\beta$ | converged spin densities | |
| $W = \sum_\sigma D_\sigma F_\sigma D_\sigma$ | energy-weighted density | |
| $M$ | the matrix contracted with $\partial S/\partial R_A$ | |
| $\alpha$ | exact-exchange fraction (1 for HF) | |
| $Q_{\mu\nu}(\mathbf G)$ | bra-centre derivative of the pair transform | `pair_ft_deriv_chunked` |

| Entry point | Location |
|---|---|
| Gamma, RHF and RKS | `ferric-pbc/src/grad.rs::gamma_rhf_gradient` |
| Gamma, UHF and UKS | `ferric-pbc/src/grad.rs::gamma_uhf_gradient` |
| $k$-points | `ferric-pbc/src/kgrad.rs::kpoint_rhf_gradient` and siblings |
| CLI | `task = "optimize"` with a `[cell]` section |
| Python | `with_gradient=True` on the SCF entry points; `gradient()` in Hartree/Bohr per cell |

## 24.7.1 The force, term by term

At fixed $D_\sigma$ (`grad.rs` module documentation):

| Energy term | Force term |
|---|---|
| kinetic, ECP | lattice-summed shifted one-electron derivative blocks |
| SR nuclear attraction (erfc) | three-centre derivative; the nuclear centre by translation invariance, $-(\text{bra} + \text{ket})$ |
| LR nuclear attraction | basis: $-(2/\Omega)\sum_{\mathbf G}v_\eta\operatorname{Re}[\rho_A'^*S(\mathbf G)]$; nucleus: the structure-factor derivative |
| two-electron (pure AFT) | $(8/\Omega)\sum_{\mathbf G}(4\pi/G^2)\operatorname{Re}\sum_{\mu\in A,\nu}Q^*_{\mu\nu}Z_{\mu\nu}$, $Z = \tfrac12D\rho(\mathbf G) - \tfrac\alpha2\sum_\sigma D_\sigma P(\mathbf G)D_\sigma$ |
| overlap coupling | $\sum M\,\partial S/\partial R_A$, $M = -W + c_0Z_\text{tot}D - \alpha v_M\sum_\sigma D_\sigma SD_\sigma$ |
| Ewald $E_{nn}$ | SR erfc plus LR structure factor (section 24.1) |
| XC | AO term, grid-point motion and the partition-weight derivative |

Two entries of $M$ come from the periodic bookkeeping:

- $c_0Z_\text{tot}D$: the $\mathbf G = 0$ term of the nuclear attraction,
  $c_0Z_\text{tot}S$ (section 24.3), depends on the geometry only through
  $S$.
- $-\alpha v_M\sum_\sigma D_\sigma SD_\sigma$: the explicit $S$ dependence of
  the Madelung energy, per spin. Because $D_\sigma SD_\sigma = D_\sigma$, it
  cancels the $-\alpha v_M$ shift of the occupied levels inside $W$. Forces
  for `ewald` and `none` therefore agree. The total-density form
  $-\tfrac{\alpha v_M}2DSD$ cancels only for a closed shell.

## 24.7.2 Verified equations

> **Verified equation 24.14** · the Madelung terms cancel in the force
>
> With the per-spin overlap term $-\alpha v_M\sum_\sigma D_\sigma SD_\sigma$,
> the overlap-coupled force is the same for `exxdiv = ewald` and `none`, for
> closed and open shells; with the total-density form it differs on an open
> shell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `madelung-force` |
> | Negative control | the claim itself evaluates the total-density form and requires it to fail on the open shell |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/grad.rs::gamma_uhf_gradient` |
> | Code test | `h2_ewald_and_none_forces_agree` in `crates/ferric-pbc/tests/pbc_grad.rs` |
>
> <!-- verified: id=pbc.grad.madelung; script=scripts/pbc_kpoint_corr_sympy.py; claim=madelung-force; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/grad.rs::gamma_uhf_gradient; rust_test=ferric-pbc/tests/pbc_grad.rs::h2_ewald_and_none_forces_agree; registry=pbc-hf -->

The Ewald nuclear gradient is part of section 24.1's
`ewald_nuclear_gradient_parts`, checked against PySCF and finite differences
in `ewald_gradient_matches_pyscf_grad_nuc_and_fd`.

There is no committed derivation of the full electronic force (the
two-electron, nuclear-attraction and XC terms above) in `theory.md` and no
proof script for it. The forces are anchored by finite differences of
ferric's own energy instead (24.7.4). That is why this section's status is
`edit`.

## 24.7.3 Refusals

Open-shell forces refuse a state that is not stationary instead of returning
a number:

<!-- include: crates/ferric-pbc/theory.md#"6. Refusal gates and tolerances of the open-shell force and stress paths" -->

## 24.7.4 Numerical anchors

| Quantity | Bar | Test |
|---|---|---|
| H2 force against finite differences of ferric's own energy, both `exxdiv` | $10^{-7}$ Ha/Bohr; net force below $10^{-10}$ | `crates/ferric-pbc/tests/pbc_grad.rs` `h2_force_matches_fd_of_own_energy_both_exxdiv` |
| `ewald` against `none` forces | $10^{-12}$ Ha/Bohr | `h2_ewald_and_none_forces_agree` |
| each deliberate force mutant | detected above $10^{-3}$ | `h2_fd_anchor_catches_each_mutant` |
| triclinic s/p cell against finite differences | $1.5\times10^{-8}$ Ha/Bohr | `triclinic_sp_force_matches_fd_of_own_energy` |
| $1\times1\times3$ mesh force against the supercell force on one copy | | `crates/ferric-pbc/tests/pbc_kgrad.rs` `h2_1x1x3_force_is_the_supercell_force_on_one_copy` |

## 24.7.5 Limits

- The lattice is fixed; there is no lattice-vector relaxation.
  `task = "frequencies"` is an error for a cell.
- VV10 is refused on every periodic derivative path.
- `range_split` is refused with `task = "optimize"`.

## 24.7.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/src/grad.rs`, module documentation | the force terms |
| `crates/ferric-pbc/theory.md`, §1 (Gradient), §6 | the Ewald gradient, the refusal gates |
| `scripts/pbc_kpoint_corr_sympy.py` | the Madelung cancellation |
