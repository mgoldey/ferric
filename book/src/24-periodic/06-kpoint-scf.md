# 24.6 k-point Hartree-Fock and Kohn-Sham

<!-- section: 24.6; status: edit; registry: pbc-hf, pbc-ks;
     theory: crates/ferric-pbc/theory.md#"8. k-point Hartree-Fock and Kohn-Sham" -->

A Gamma-point calculation samples the Brillouin zone at one point. That
is exact only when the cell is large enough that the bands are flat. A
$k$-point mesh samples $N_k$ points instead. Each point gets its own complex
Fock matrix, and the Coulomb and exchange terms couple the points. The
energy is reported per primitive cell.

The organizing fact of this section: a Gamma-centred
$N_1\times N_2\times N_3$ mesh is the Gamma calculation of the
$\operatorname{diag}(N)$ supercell, term by term. That identity is the
main test of the $k$-point code, and the proof below checks it.

| Symbol | Meaning | Code name |
|---|---|---|
| $k$, $k'$ | mesh points | `KPointMesh::kpts` |
| $N_k$ | mesh size | `nk` |
| $q = k' - k$ | momentum transfer of an exchange pair | `q` class |
| $\chi_{\mu k}$ | Bloch AO | |
| $D(k)$ | density, $2C_o(k)C_o(k)^{\mathsf H}$ | `dms[k]` |
| $v_M$ | Madelung constant of the $\operatorname{diag}(N)$ supercell | `KPointMesh::madelung` |

| Entry point | Location |
|---|---|
| Mesh | `ferric-pbc/src/kpts.rs::{monkhorst_pack,madelung}` |
| RHF | `ferric-pbc/src/kscf.rs::{solve_krhf,solve_k_closed_shell}` |
| UHF | `ferric-pbc/src/kuscf.rs::solve_kuhf` |
| ROHF, ROKS | `ferric-pbc/src/kroks.rs::solve_krohf` |
| RKS | `ferric-pbc/src/kdft.rs::solve_krks` |
| CLI | `[cell] kmesh = [n1, n2, n3]` with `method.kind` `rhf`, `uhf`, `rohf` or `ksdft` |
| Python | `run_rhf_kpts`, `run_uhf_kpts`, `run_rohf_kpts`, `run_rks_kpts`, `run_uks_kpts`, `run_roks_kpts` |

<!-- include: crates/ferric-pbc/theory.md#"8. k-point Hartree-Fock and Kohn-Sham" -->

## 24.6.1 Verified equations

> **Verified equation 24.11** · the mesh and its phases
>
> The Gamma-centred and Monkhorst-Pack numerators give the PySCF fractional
> points; for odd $N$ the two meshes are the same set modulo the reciprocal
> lattice. Phases built from the integer numerators satisfy
> $X(-k) = X(k)^*$ for every one-particle matrix.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `mesh` |
> | Negative control | none in the script; the code test pins the points to PySCF |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kpts.rs::monkhorst_pack` |
> | Code test | `monkhorst_pack_points_and_mesh_madelung_match_pyscf` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.kscf.mesh; script=scripts/pbc_kpoint_corr_sympy.py; claim=mesh; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kpts.rs::monkhorst_pack; rust_test=ferric-pbc/tests/pbc_krhf.rs::monkhorst_pack_points_and_mesh_madelung_match_pyscf; registry=pbc-hf -->

> **Verified equation 24.12** · the closed-shell k-mesh is the Gamma supercell
>
> $J(k)$ with the kernel at $\mathbf G$, $K(k)$ with the kernel at
> $\mathbf G + q$ and the factor $1/N_k$, the Madelung term
> $v_MS(k)D(k)S(k)$ with no $1/N_k$, and the two-electron energy
> $N_k^{-1}\sum_k\tfrac12\operatorname{Re}\operatorname{tr}[(J - \tfrac12K)(k)D(k)]$
> reproduce the supercell's Gamma $J$, $K$ and two-electron energy per cell.
> The one-electron term maps the same way through
> $X(k) = \sum_{\mathbf L}e^{ik\cdot\mathbf L}X^\text{sc}_{(\mu0),(\nu\mathbf L)}$
> (claim `mesh`).
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `krhf-supercell` |
> | Negative control | the runner's two mutations: the Madelung term divided by $N_k$, and the exchange kernel taken at $\mathbf G$ instead of $\mathbf G + q$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kscf.rs::solve_k_closed_shell` |
> | Code test | `h2_1x1x3_mesh_is_the_gamma_supercell` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.kscf.supercell; script=scripts/pbc_kpoint_corr_sympy.py; claim=krhf-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kscf.rs::solve_k_closed_shell; rust_test=ferric-pbc/tests/pbc_krhf.rs::h2_1x1x3_mesh_is_the_gamma_supercell; registry=pbc-hf -->

> **Verified equation 24.13** · the open-shell k-mesh is the Gamma supercell
>
> Per spin, $F_\sigma(k) = h + J[D_\alpha + D_\beta] - K[D_\sigma]$ with the
> Madelung term at coefficient 1 inside $K[D_\sigma]$, and the per-spin
> energy sum, reproduce the supercell's unrestricted Gamma energy.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `kuhf-supercell` |
> | Negative control | the `krhf-supercell` mutations act on the shared builder |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kuscf.rs::solve_kuhf` |
> | Code test | `crates/ferric-pbc/tests/pbc_kuhf.rs` |
>
> <!-- verified: id=pbc.kscf.uhf; script=scripts/pbc_kpoint_corr_sympy.py; claim=kuhf-supercell; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kuscf.rs::solve_kuhf; registry=pbc-hf -->

The Madelung shift of the occupied levels, which explains why `none` and
`ewald` share their densities, is box 24.7 in section 24.4.

The Kohn-Sham energy functional and Fock matrix use the same $J$, $K$ and
Madelung conventions with exchange scaled by $a$. The periodic exchange-
correlation quadrature (the SSF-partitioned atom-centred grid of
`dft.rs`) has no proof script. That is why `pbc-ks` stays a registry gap
and this section's status is `edit`.

## 24.6.2 The algorithm

| Step | What | Function |
|---|---|---|
| 1 | mesh, phases, supercell Madelung constant | `ferric-pbc/src/kpts.rs::{monkhorst_pack,madelung}` |
| 2 | $S(k)$, $h(k)$, orthogonalizer at `lindep` $10^{-6}$ | `complex_canonical_orthogonalizer` |
| 3 | $J(k)$, $K(k)$ from the pair transforms (dense) or the $k$-point fit (`rsgdf`) | `ferric-pbc/src/kdense_aft.rs::build`, `KRsGdf` |
| 4 | diagonalize one of each $\{k, -k\}$; $C(-k) = C(k)^*$ | `ferric-pbc/src/kscf.rs::diagonalize_all` |
| 5 | occupy the $N_kN_e/2$ lowest levels over the whole mesh; a gap below `min_gap` is an error | `ferric-pbc/src/kscf.rs::aufbau` |
| 6 | DIIS on the stacked commutators, real weights | `KDiis::step` |

The energy is assembled in one loop:

<!-- excerpt: ferric-pbc/src/kscf.rs::solve_k_closed_shell -->
```rust
        let mut e_elec = 0.0;
        for k in 0..nk {
            let hf = &inj.h[k] + &f_nox[k];
            // Re tr[(h + F) D] = Re Σ_mn (h+F)_mn D_nm
            let mut tr = 0.0;
            for m in 0..n {
                for nu in 0..n {
                    tr += (hf[(m, nu)] * dm[k][(nu, m)]).re;
                }
            }
            e_elec += 0.5 * tr;
        }
        let energy = e_elec * inv_nk + e_xc + inj.vnn;
```

| Symbol | Code |
|---|---|
| $h(k) + F(k)$, $F$ without $V_{xc}$ | `hf` (`f_nox`) |
| $E_{xc}$ (0 for HF) | `e_xc` |
| $D(k)$ | `dm[k]` |
| $N_k^{-1}$ | `inv_nk` |
| $E_{nn}$ | `inj.vnn` |

## 24.6.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] kmesh`, `kmesh=` | three positive integers | none (Gamma point) | the mesh |
| `[cell] centring` | `"gamma"`, `"mp"` | `"gamma"` | Gamma-centred or Monkhorst-Pack; only the Gamma-centred mesh has the supercell anchor |
| `[cell] exxdiv` | `"ewald"`, `"none"` | `"ewald"` | the Madelung term of section 24.4 |
| `[cell] ewald_start` | `"staged"`, `"direct"` | `"staged"` | open shells: converge with `none`, then restart with `ewald` |
| `[cell] jk` | `"dense"`, `"rsgdf"` | `"dense"` | section 24.5 |

## 24.6.4 Numerical anchors

| Quantity | Bar | Test |
|---|---|---|
| $1\times1\times3$ and $2\times2\times2$ meshes against the explicit supercell, both `exxdiv` | $10^{-11}$ Ha | `crates/ferric-pbc/tests/pbc_krhf.rs` `h2_1x1x3_mesh_is_the_gamma_supercell`, `h2_2x2x2_mesh_is_the_gamma_supercell` |
| PySCF `KRHF` with AFTDF | $10^{-9}$ Ha | `h2_1x1x2_matches_pinned_pyscf_krhf_aftdf` |
| exchange kernel without the $q$ shift, a phase sign flip, the primitive-cell Madelung constant | each detected | `mutant_kernel_without_q_shift_is_caught`, `mutant_phase_sign_in_pair_ft_is_caught`, `mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact` |

## 24.6.5 Limits

- Insulators only: the global aufbau refuses a converged gap below
  `min_gap`. There is no smearing.
- A shifted (Monkhorst-Pack, even $N$) mesh is a twisted-boundary supercell
  and has no supercell anchor; it is pinned to PySCF instead.

## 24.6.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §8 | the equations and the supercell identity |
| `scripts/pbc_kpoint_corr_sympy.py` | the proof |
| PySCF `pyscf.pbc.scf.khf` | the conventions (`get_occ`, `_ewald_exxdiv_for_G0`) |
