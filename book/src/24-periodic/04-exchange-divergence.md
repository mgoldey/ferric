# 24.4 The exchange divergence

<!-- section: 24.4; status: ready; registry: pbc-hf;
     theory: crates/ferric-pbc/theory.md#"4. Exchange divergence (`exxdiv`)" -->

Dropping $\mathbf G = 0$ is exact for the Coulomb energy of a neutral cell:
the electronic and nuclear divergences cancel. Exchange has no such partner.
Its $\mathbf G = 0$ term is integrable in the infinite crystal, but a finite
sampling of it (one cell at $\Gamma$, or a mesh) misses a finite piece that
vanishes only as $1/L$ in the cell or supercell size. The Madelung correction
adds that piece back for a probe charge in a uniform background, and the
error then falls as $L^{-3}$.

| Symbol | Meaning | Code name |
|---|---|---|
| $v_M$ | Madelung constant of the cell, or of the $\operatorname{diag}(N)$ supercell for a mesh | `madelung`, `KPointMesh::madelung` |
| `exxdiv` | `"ewald"` (correction on) or `"none"` | `ExxDiv::{Ewald,None}` |
| $\alpha$ | exact-exchange fraction, 1 for HF | `exact_exchange` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/kpts.rs::madelung`, `ferric-pbc/src/ewald.rs::madelung_constant` |
| K builders that carry the term | `ferric-pbc/src/rsgdf.rs::k_builder`, `ferric-pbc/src/kdense_aft.rs::build` |
| CLI | `[cell] exxdiv = "ewald"` (default) or `"none"` |
| Python | `exxdiv=` on every periodic SCF entry point |

<!-- include: crates/ferric-pbc/theory.md#"4. Exchange divergence (`exxdiv`)" -->

## 24.4.1 What the term does

The term lives inside the exchange builder, $K \mathrel{+}= v_MSDS$, linear in
the density it is given. Three consequences follow, each proven below or in
section 24.6:

| Consequence | Where |
|---|---|
| At fixed orbitals every occupied level moves by $-v_M$ (times $\alpha$ in Kohn–Sham), virtual levels and orbitals do not move; $\Delta E = -\alpha v_MN_e/2$ per cell | Verified equations 24.6 and 24.7 |
| On a mesh, the constant is the supercell's and is added at every $k$ with no $1/N_k$ | Verified equation 24.12 |
| Per spin it is $v_MSD_\sigma S$ with coefficient 1; the total-density form is wrong for open shells | Verified equations 24.13 and 24.14 |

Because the orbitals do not move, `none` and `ewald` share their stationary
densities. A state with a hole shallower than $v_M$ below the Fermi level can
be self-consistent under `ewald` only. The open-shell drivers therefore
converge with `none` first and continue with `ewald`
(`EwaldStart::Staged`, the default), and report each spin's gap against $v_M$.

## 24.4.2 Verified equations

> **Verified equation 24.6** · the closed-shell energy shift
>
> For an idempotent closed-shell density, $DSD = 2D$, so
> $\Delta E = -\tfrac14v_M\operatorname{tr}(DSDS) = -\tfrac12Nv_M$ and
> $\partial\Delta E/\partial D = -\tfrac12v_MSDS$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S4, claim `exxdiv-shift` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/rsgdf.rs::k_builder` |
> | Code test | `h2_rhf_ewald_minus_none_is_the_madelung_strain` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.exxdiv.energy-shift; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=exxdiv-shift; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/rsgdf.rs::k_builder; rust_test=ferric-pbc/tests/pbc_stress.rs::h2_rhf_ewald_minus_none_is_the_madelung_strain; registry=pbc-hf -->

> **Verified equation 24.7** · the occupied levels move, the orbitals do not
>
> For S-orthonormal orbitals at any $k$,
> $C^{\mathsf H}(-\tfrac12v_MSDS)C = -v_M\operatorname{diag}(1_\text{occ}, 0_\text{virt})$,
> and the energy per cell moves by $-v_MN_e/2$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_kpoint_corr_sympy.py`, claim `madelung-shift` |
> | Negative control | the runner's mutation: the Madelung term divided by $N_k$ (fails `krhf-supercell`) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/kpts.rs::madelung`, `ferric-pbc/src/kdense_aft.rs::build` |
> | Code test | `mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact` in `crates/ferric-pbc/tests/pbc_krhf.rs` |
>
> <!-- verified: id=pbc.exxdiv.level-shift; script=scripts/pbc_kpoint_corr_sympy.py; claim=madelung-shift; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/kpts.rs::madelung; rust_test=ferric-pbc/tests/pbc_krhf.rs::mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact; registry=pbc-hf -->

## 24.4.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cell] exxdiv`, `exxdiv=` | `"ewald"`, `"none"` (strict) | `"ewald"` | adds $v_MSDS$ to every exchange build |
| `ewald_start=` (open-shell Python drivers) | `"staged"`, `"direct"` | `"staged"` | converge with `none` first, then `ewald` |

Correlation methods read the reference's `exxdiv` as a stated input, because
the occupied orbital energies differ by $v_M$ between the two (section 24.9).

## 24.4.4 Numerical anchors

| Quantity | Test |
|---|---|
| $E_\text{none} - E_\text{ewald} - n_\text{occ}v_M$ on the $k$-point supercell anchors, $< 10^{-11}$ | `crates/ferric-pbc/tests/pbc_krhf.rs` (inside `supercell_anchor`) |
| PySCF energies with both `exxdiv` values | `crates/ferric-pbc/tests/pbc_dense_aft_scf.rs` |

## 24.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §4 and §8 | the correction and its per-spin and per-$k$ forms |
| `scripts/pbc_ewald_pairft_stress_sympy.py` part S4, `scripts/pbc_kpoint_corr_sympy.py` | the proofs |
| J. Paier, R. Hirschl, M. Marsman, G. Kresse, *J. Chem. Phys.* **122**, 234102 (2005) | the probe-charge correction |
