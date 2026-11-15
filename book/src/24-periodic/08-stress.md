# 24.8 The stress tensor

<!-- section: 24.8; status: ready; registry: pbc-ewald-pairft-stress;
     theory: crates/ferric-pbc/theory.md#"5. Stress tensor (strain derivative)" -->

The stress is the derivative of the energy per cell with respect to a
homogeneous strain of the lattice and the atoms together. Fractional
coordinates and Miller indices stay fixed. It is the quantity a
lattice optimizer would follow, and its trace gives the pressure. ferric
computes it analytically. There is no lattice relaxation driver: the
stress is reported, not used.

Every term of the periodic energy depends on the strain in one of three
ways:

- through the volume $\Omega$;
- through $G^2$ in a reciprocal kernel;
- through a real-space distance or a pair centre.

The structure factor and every $\mathbf G\cdot\mathbf X$ phase are
strain-invariant. Section 24.8 derives the three pieces that are specific
to a periodic code: the Ewald nuclear energy, the Madelung constant and the
pair transform. The overlap-coupled terms reuse the force machinery of
section 24.7.

| Symbol | Meaning | Code name |
|---|---|---|
| $\varepsilon_{ab}$ | homogeneous strain, $\mathbf r \to (1 + \varepsilon)\mathbf r$ | |
| $\sigma_{ab} = \Omega^{-1}dE/d\varepsilon_{ab}$ | stress, Hartree/Bohr$^3$, not symmetrized | `stress()` |
| $p = -\operatorname{tr}\sigma/3$ | pressure | |
| $k(G^2) = e^{-G^2/4\eta^2}/G^2$ | Ewald LR kernel | |

| Entry point | Location |
|---|---|
| Ewald strain | `ferric-pbc/src/ewald.rs::{ewald_nuclear_strain,madelung_strain}` |
| Pair-transform strain | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
| Gamma assembly | `ferric-pbc/src/stress.rs::gamma_rhf_stress` and siblings |
| $k$-point assembly | `ferric-pbc/src/kstress.rs::kpoint_rhf_stress` and siblings |
| Python | `with_stress=True` on the SCF entry points |
| CLI | `[cell] stress = true`, which requires `kmesh` |

<!-- include: crates/ferric-pbc/theory.md#"5. Stress tensor (strain derivative)" -->

## 24.8.1 Verified equations

> **Verified equation 24.15** · the Ewald strain
>
> The SR virial, the LR term $-\delta_{ab}E_\text{LR}$ plus the $G^2$
> derivative of the kernel, and $-\delta_{ab}E_\text{bg}$ equal central
> finite differences of a pure-Python Ewald at frozen index sets.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `ewald-strain` |
> | Negative control | the runner's mutation: the $1/G^2$ term dropped from $k'$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::ewald_nuclear_strain` |
> | Code test | `ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.ewald; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=ewald-strain; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::ewald_nuclear_strain; rust_test=ferric-pbc/tests/pbc_stress.rs::ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous; registry=pbc-ewald-pairft-stress -->

> **Verified equation 24.16** · the strain derivative of a pair transform
>
> $dP/d\varepsilon_{ab} = [Q_a + iG_a(\alpha/p)P](\mathbf A - \mathbf B')_b
> + G_aG_bP/(2p) - G_aP^g_b$ equals the direct derivative of the strained
> closed form for s|s and p$_x$|s at exact rational points.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `pairft-strain` |
> | Negative control | none in the script; the code test compares against finite differences at fixed Miller index |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
> | Code test | `pair_ft_strain_matches_fd_at_fixed_miller_index` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.pairft; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=pairft-strain; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked; rust_test=ferric-pbc/tests/pbc_stress.rs::pair_ft_strain_matches_fd_at_fixed_miller_index; registry=pbc-ewald-pairft-stress -->

> **Verified equation 24.17** · the Madelung constant is homogeneous
>
> $v_M$ is homogeneous of degree $-1$ in lengths, so
> $\operatorname{tr}\,dv_M/d\varepsilon = -v_M$.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, claim `madelung-euler` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/ewald.rs::madelung_strain` |
> | Code test | `h2_rhf_ewald_minus_none_is_the_madelung_strain` in `crates/ferric-pbc/tests/pbc_stress.rs` |
>
> <!-- verified: id=pbc.stress.madelung; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=madelung-euler; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/ewald.rs::madelung_strain; rust_test=ferric-pbc/tests/pbc_stress.rs::h2_rhf_ewald_minus_none_is_the_madelung_strain; registry=pbc-ewald-pairft-stress -->

## 24.8.2 The algorithm

| Step | Term | Function |
|---|---|---|
| 1 | Ewald nuclear strain | `ferric-pbc/src/ewald.rs::ewald_nuclear_strain` |
| 2 | each reciprocal energy: $-\delta_{ab}E + \Omega^{-1}\sum_{\mathbf G}[v'(-2G_aG_b)\Phi + v\,d\Phi]$, with $d\Phi$ from the pair-transform strain | `ferric-pbc/src/pair_ft.rs::pair_ft_strain_chunked` |
| 3 | overlap-coupled terms: the force's matrix $M$ contracted with per-image pair vectors | `ferric-pbc/src/stress.rs::gamma_rhf_stress` |
| 4 | Madelung: $-\tfrac\alpha2\sum_\sigma\operatorname{tr}(D_\sigma SD_\sigma S)\,dv_M/d\varepsilon$ | `ferric-pbc/src/ewald.rs::madelung_strain` |
| 5 | XC, ECP and RS-GDF terms | listed in the `stress.rs` module documentation |

Truncated sums are frozen as integer index sets at the reference cell
(`Cell::strained`), so the stress is the exact derivative of the energy
that was computed, not of a different truncation.

## 24.8.3 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `with_stress=` (Python) | `True`, `False` | `False` | Gamma point or mesh |
| `[cell] stress` (CLI) | `true`, `false` | `false` | prints the stress; requires `kmesh` |

## 24.8.4 Numerical anchors

| Quantity | Bar | Test |
|---|---|---|
| Ewald strain against finite differences; at $\eta = 1$ and $2.3$; Euler identity | $10^{-8}$; $10^{-10}$; $10^{-12}$ | `crates/ferric-pbc/tests/pbc_stress.rs` `ewald_strain_matches_fd_is_omega_independent_and_madelung_is_homogeneous` |
| H2 RHF stress, all nine components, both `exxdiv` | $10^{-7}$ Ha/Bohr$^3$ against finite differences of ferric's own energy | `h2_rhf_stress_matches_fd_of_own_energy_all_nine_both_exxdiv` |
| `ewald` minus `none` stress equals the Madelung strain term | $10^{-9}$ | `h2_rhf_ewald_minus_none_is_the_madelung_strain` |
| one-point mesh stress against the Gamma stress | | `crates/ferric-pbc/tests/pbc_kstress.rs` `one_point_mesh_rhf_stress_is_the_gamma_stress` |

## 24.8.5 Limits

- No lattice-vector relaxation.
- Gamma-point stress is Python-only; the CLI `stress = true` requires
  `kmesh`.
- VV10 is refused. Per-functional support is listed in the input reference.

## 24.8.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §5 | the strain derivatives |
| `scripts/pbc_ewald_pairft_stress_sympy.py` | the proof (S5) |
| O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) | the stress theorem |
