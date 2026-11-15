# 24.2 Fourier transform of lattice-summed Gaussian pairs

<!-- section: 24.2; status: ready; registry: pbc-ewald-pairft-stress;
     theory: crates/ferric-pbc/theory.md#"2. Fourier transform of lattice-summed Gaussian pairs" -->

Every reciprocal-space term of the chapter contracts one object: the Fourier
transform of the product of a basis function in the home cell and a
lattice-summed partner. The Coulomb tensor of the dense builder, the long-range
nuclear attraction, the long-range part of the density fit, the $k$-point
kernels and the strain derivative all read it. At $\mathbf G = 0$ it is the
lattice overlap.

| Symbol | Meaning | Code name |
|---|---|---|
| $P_{\mu\nu}(\mathbf G)$ | pair transform, PySCF `ft_aopair` sign | `pair_ft` output |
| $\alpha, \beta$, $p = \alpha + \beta$ | primitive exponents | |
| $\mathbf P_c$ | product centre | |
| $E_t^{ij}$ | McMurchie–Davidson coefficients | `ferric_integrals::md3c1e::e_table` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/pair_ft.rs::{pair_ft,pair_ft_chunked}` |
| Plan and kernels | `crates/ferric-pbc/src/pair_ft/plan.rs` (`walk`, `site_terms`, `f_rows`) |
| $k$-point residues | `crates/ferric-pbc/src/pair_ft/residues.rs` |

<!-- include: crates/ferric-pbc/theory.md#"2. Fourier transform of lattice-summed Gaussian pairs" -->

## 24.2.1 Verified equation

> **Verified equation 24.3** · the McMurchie–Davidson pair transform
>
> The Gaussian product theorem holds as an exact exponent identity, and the
> expansion $\prod_d\sum_tE_t^{i_dj_d}(-iG_d)^t$ for
> $(l_a, l_b) = (0,0), (1,0), (1,1), (2,0)$ equals an mpmath quadrature of the
> complex one-dimensional Fourier integral.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S2, claim `pairft-md` |
> | Negative control | none in the script; the code tests compare with PySCF |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/pair_ft.rs::pair_ft` |
> | Code test | `pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.pairft.md-expansion; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=pairft-md; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/pair_ft.rs::pair_ft; rust_test=ferric-pbc/tests/pbc_stage0.rs::pair_ft_at_nonzero_g_matches_pyscf_ft_aopair; registry=pbc-ewald-pairft-stress -->

## 24.2.2 The algorithm

| Step | Quantity | Cost |
|---|---|---|
| 1 | pair images $\mathbf L$ whose overlap Gaussian $e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}$ passes the screen | $N_\text{shell}^2N_{\mathbf L}$ |
| 2 | per pair and image: prefactor, $E_t$ tables, phase $e^{-i\mathbf G\cdot\mathbf P_c}$ | |
| 3 | per $\mathbf G$ chunk: rows $\sum_tE_t(-iG_d)^t$, accumulate into $P_{\mu\nu}(\mathbf G)$ | $N_\text{pair}N_{\mathbf L}N_{\mathbf G}$ |

The transform is streamed in $\mathbf G$ chunks (`pair_ft_chunked`); no caller
holds all $\mathbf G$ at once. At a $k$-point mesh the images are binned by
residue $\mathbf L \bmod N$, and one residue-resolved transform per
momentum-transfer class gives $P^{k'}(\mathbf G + q)$ for every $k'$ by a phase
sum (section 24.6).

## 24.2.3 Numerical anchors

| Quantity | Reference | Bar | Test |
|---|---|---|---|
| $P_{\mu\nu}(0)$, every element | an independent libint2 lattice overlap | $10^{-10}$ | `crates/ferric-pbc/tests/pbc_stage0.rs` `pair_ft_at_g0_equals_independent_libint_lattice_overlap` |
| $P_{\mu\nu}(\mathbf G)$ for s, p, pure d at lattice and off-lattice $\mathbf G$ | PySCF `ft_aopair` | $10^{-10}$ | `pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` |

## 24.2.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §2 | the transform and its coefficients |
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S2 | the proof |
| L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) | the Hermite expansion |
