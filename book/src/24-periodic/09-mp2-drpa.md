# 24.9 Periodic MP2 and dRPA

<!-- section: 24.9; status: ready; registry: pbc-mp2, pbc-drpa;
     theory: crates/ferric-pbc/theory.md#"9. Periodic MP2 and dRPA" -->

Periodic MP2 and direct RPA reuse the molecular kernels of chapters 10, 11
and 18. At the Gamma point the orbitals are real, and the only new ingredients
are the integrals and the orbital energies. On a mesh the integrals carry
crystal momentum: an excitation $i \to a$ moves momentum $k_a - k_i$, and
only pairs of excitations with opposite momenta couple.

Two periodic facts carry the section:

1. An occupied-virtual pair density has no $\mathbf G = 0$ component.
   The dropped-$\mathbf G = 0$ kernel of the SCF therefore needs no
   correction in correlation.
2. The exchange divergence reaches correlation only through the occupied
   orbital energies. The caller chooses the denominators, and the
   conversion between the two choices is exact.

| Symbol | Meaning | Code name |
|---|---|---|
| $B_{k,ia}$ | fitted integrals in the occupied-virtual block | `b_ov_from_ao_b` |
| $\Pi(q, iu)$ | density response at momentum $q$ | |
| $V^{k_ik_jk_a}_{iajb}$ | $(ik_i\,ak_a|jk_j\,bk_b)$, $k_b = k_i + k_j - k_a$ | |
| $u_0$ | quadrature scale, 0.5 | |

| Entry point | Location |
|---|---|
| Gamma MP2 | `ferric-pbc/src/mp2.rs::gamma_mp2` |
| Gamma dRPA | `ferric-pbc/src/drpa.rs::gamma_drpa` |
| $k$-point MP2, dRPA | `ferric-pbc/src/kcorr.rs::{kpoint_mp2,kpoint_drpa}` |
| CLI | `[cell]` with `method.kind = "rimp2"` or `"pdep-rpa"` |
| Python | `run_mp2_gamma`, `run_drpa_gamma`, `run_mp2_kpts`, `run_drpa_kpts` |

<!-- include: crates/ferric-pbc/theory.md#"9. Periodic MP2 and dRPA" -->

## 24.9.1 Verified equations

> **Verified equation 24.18** · occupied-virtual pair densities are neutral
>
> For $S$-orthonormal orbitals, $C_o^{\mathsf H}S(k)C_v = 0$ at every $k$,
> so $\int\phi_i^*\phi_a = 0$ and the $\mathbf G = 0$ component of every
> occupied-virtual pair density vanishes.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `ov-neutral` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/mp2.rs::b_ov_from_ao_b` |
> | Code test | `mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit` in `crates/ferric-pbc/tests/pbc_mp2.rs` |
>
> <!-- verified: id=pbc.corr.ov; script=scripts/pbc_kpoint_corr_sympy.py; claim=ov-neutral; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/mp2.rs::b_ov_from_ao_b; rust_test=ferric-pbc/tests/pbc_mp2.rs::mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit; registry=pbc-mp2 -->

> **Verified equation 24.19** · k-point MP2 is the supercell MP2
>
> With $k_b = k_i + k_j - k_a$, the kernel at $\mathbf G + q$,
> $q = k_a - k_i$, and the factor $1/N_k^3$, the $k$-point MP2 energy per
> cell equals the closed-shell MP2 of the $\operatorname{diag}(N)$
> supercell per cell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kmp2-supercell` |
> | Negative control | the runner's mutation: $k_b = k_i - k_j + k_a$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::kpoint_mp2` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.kmp2; script=scripts/pbc_kpoint_corr_sympy.py; claim=kmp2-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::kpoint_mp2; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-mp2 -->

> **Verified equation 24.20** · k-point dRPA is the supercell dRPA
>
> $\Pi(q, iu)$ with the factor $4/N_k$ and
> $E_c = N_k^{-1}\sum_q(2\pi)^{-1}\int[\ln\det(1 + \Pi) - \operatorname{tr}\Pi]\,du$
> equals the supercell's plasmon-formula dRPA per cell.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kdrpa-supercell` |
> | Negative control | the runner's mutation: $\Pi$ without the $1/N_k$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::kpoint_drpa` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.kdrpa; script=scripts/pbc_kpoint_corr_sympy.py; claim=kdrpa-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::kpoint_drpa; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-drpa -->

> **Verified equation 24.21** · realification
>
> For $B_q = X + iY$, the real
> $B_R = \begin{pmatrix}X & -Y\\ Y & X\end{pmatrix}$ with every excitation
> energy duplicated gives a real response whose spectrum is that of
> $\Pi(q)$ with each eigenvalue twice, so $E_q = \tfrac12E[\Pi_R]$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `realify` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kcorr.rs::drpa_q_realified` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_kcorr.rs` |
>
> <!-- verified: id=pbc.corr.realify; script=scripts/pbc_kpoint_corr_sympy.py; claim=realify; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kcorr.rs::drpa_q_realified; rust_test=ferric-pbc/tests/pbc_kcorr.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-drpa -->

> **Verified equation 24.22** · the second-order dRPA term is direct MP2
>
> $\int_0^\infty e_1e_2/((u^2 + e_1^2)(u^2 + e_2^2))\,du = \pi/(2(e_1 + e_2))$,
> so the $O(\Pi^2)$ term of the dRPA energy is
> $\sum 2|V|^2/(\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b)$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `drpa-second-order` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/drpa.rs::gamma_drpa` |
> | Code test | `drpa_second_order_term_is_direct_mp2_from_gamma_mp2` in `crates/ferric-pbc/tests/pbc_drpa.rs` |
>
> <!-- verified: id=pbc.corr.second; script=scripts/pbc_kpoint_corr_sympy.py; claim=drpa-second-order; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/drpa.rs::gamma_drpa; rust_test=ferric-pbc/tests/pbc_drpa.rs::drpa_second_order_term_is_direct_mp2_from_gamma_mp2; registry=pbc-drpa -->

The denominator conversion rests on box 24.7 (section 24.4): `ewald`
lowers every occupied level by $v_M$ at fixed orbitals.

## 24.9.2 The algorithm

| Step | Gamma | $k$-points |
|---|---|---|
| 1 | the SCF's RS-GDF $B$, transformed to $B_{k,ia}$ | the pair tensors of each class $q$ |
| 2 | occupied energies shifted by `occupied_shift` | the same |
| 3 | MP2: `ferric_mp2::rimp2::spin_components_from_b_ov` | MP2: loop over $(k_i, k_j, k_a)$, $k_b$ by momentum conservation |
| 4 | dRPA: `ferric_rpa::run_pdep_rpa_from_parts`, full rank | dRPA: per $q$, realified, through the Gamma pipeline |

The denominator conversion is one match:

<!-- excerpt: ferric-pbc/src/mp2.rs::occupied_shift -->
```rust
    match (reference_exxdiv, denominators) {
        (ExxDiv::Ewald, Mp2Denominators::MadelungShifted) => 0.0,
        (ExxDiv::None, Mp2Denominators::MadelungShifted) => -madelung,
        (ExxDiv::None, Mp2Denominators::Unshifted) => 0.0,
        (ExxDiv::Ewald, Mp2Denominators::Unshifted) => madelung,
    }
```

| Symbol | Code |
|---|---|
| the SCF's `exxdiv` | `reference_exxdiv` |
| the requested convention | `denominators` |
| $v_M$ | `madelung` |

## 24.9.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] denominators`, `denominators=` | `"madelung-shifted"`, `"unshifted"` | none (required) | the occupied orbital energies in the denominators |
| `frozen_core` | integer | 0 | core orbitals excluded from correlation |
| `[cell] quad_points` | positive integer | 40 | dRPA Gauss-Legendre points |
| `[cell] drpa_energy` | `"quadrature"`, `"plasmon"`, `"second-order"` | `"quadrature"` | the dRPA energy route |

## 24.9.4 Numerical anchors

| Quantity | Bar | Test |
|---|---|---|
| $k$-point MP2 and plasmon dRPA against the explicit supercell | $10^{-11}$ Ha | `crates/ferric-pbc/tests/pbc_kcorr.rs` `h2_1x1x3_mesh_is_the_gamma_supercell` |
| momentum, conjugation, Madelung, occupied-$k$ and normalization mutants | each detected | `supercell_anchor_catches_momentum_conj_madelung_occupied_k_and_normalisation_mutants` |
| Gamma MP2 against PySCF | | `crates/ferric-pbc/tests/pbc_mp2.rs` `h2_sto3g_a4_matches_pinned_pyscf_rmp2` |
| $k$-point MP2 against PySCF `KMP2`, both `exxdiv` | | `kmp2_h2_1x1x2_matches_pinned_pyscf_kmp2_both_exxdiv` |

## 24.9.5 Limits

- Closed shell only on these routes. Open-shell UMP2 and URPA at Gamma are
  library-only.
- Every $k$ must hold the same number of occupied levels; the global aufbau
  of section 24.6 can violate this, and the run is then refused.
- The $q = 0$ head correction is not implemented, so the energy converges
  with the mesh at the rate of the uncorrected sum.

## 24.9.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §9 | the equations |
| `scripts/pbc_kpoint_corr_sympy.py` | the proof |
| PySCF `pyscf.pbc.mp.kmp2` | the $k$-point MP2 conventions |
