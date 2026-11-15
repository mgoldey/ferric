# 24.3 Ewald-split nuclear attraction

<!-- section: 24.3; status: ready; registry: pbc-ewald-pairft-stress;
     theory: crates/ferric-pbc/theory.md#"3. Ewald-split nuclear attraction (`hcore`)" -->

The attraction between an electron pair density and the lattice of nuclei
diverges at $\mathbf G = 0$ exactly as the electron repulsion does, with the
opposite sign. ferric drops $\mathbf G = 0$ from both, the convention that
cancels in a neutral cell. The attraction is split like the Ewald sum: a
short-range erfc part in real space through the three-centre engine, a
long-range part in reciprocal space through the pair transform of section
24.2, and one constant that removes the $\mathbf G = 0$ average the real-space
part carries.

| Symbol | Meaning | Code name |
|---|---|---|
| $V_\text{SR}$, $V_\text{LR}$, $V_\text{G0}$ | the three parts | `periodic_hcore` |
| $\eta$ | the nuclear split | `PeriodicHcoreConfig::omega` |
| $c_0 = \pi/(\eta^2\Omega)$ | the $\mathbf G = 0$ constant | `c0` |
| $\zeta = 10^{16}$ | exponent of the Gaussian nuclei in $V_\text{SR}$ | |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/hcore.rs::{periodic_hcore,reciprocal_nuclear}` |
| Oracle | `ferric-pbc/src/hcore.rs::pure_aft_nuclear` ($\eta \to \infty$) |
| $k$-points | `crates/ferric-pbc/src/hcore/kpoint.rs` |

<!-- include: crates/ferric-pbc/theory.md#"3. Ewald-split nuclear attraction (`hcore`)" -->

## 24.3.1 Verified equations

> **Verified equation 24.4** · the erfc and erf parts add to the bare kernel
>
> $\operatorname{FT}[\operatorname{erfc}(\eta r)/r] = (4\pi/G^2)(1 - e^{-G^2/4\eta^2})$,
> so per $\mathbf G \neq 0$ the short- and long-range parts are the bare
> $-4\pi Z/(\Omega G^2)$ kernel.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3, claim `hcore-split` |
> | Negative control | none in the script; the code test checks that $V_\text{G0}$ is visible |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/hcore.rs::reciprocal_nuclear` |
> | Code test | `periodic_hcore_is_omega_independent_triclinic_sp` in `crates/ferric-pbc/tests/pbc_hcore.rs` |
>
> <!-- verified: id=pbc.hcore.split; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=hcore-split; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/hcore.rs::reciprocal_nuclear; rust_test=ferric-pbc/tests/pbc_hcore.rs::periodic_hcore_is_omega_independent_triclinic_sp; registry=pbc-ewald-pairft-stress -->

> **Verified equation 24.5** · the $\mathbf G = 0$ constant
>
> $\int\operatorname{erfc}(\eta r)/r\,d^3r = \pi/\eta^2$, so the real-space
> erfc sum carries the average $-\pi Z_\text{tot}/(\eta^2\Omega)$ times the
> overlap, which $V_\text{G0} = +c_0Z_\text{tot}S$ removes; the total is the
> pure reciprocal attraction with $\mathbf G = 0$ dropped.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3, claim `hcore-g0` |
> | Negative control | none in the script |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/hcore.rs::periodic_hcore` |
> | Code test | `periodic_hcore_is_omega_independent_triclinic_sp` in `crates/ferric-pbc/tests/pbc_hcore.rs` |
>
> <!-- verified: id=pbc.hcore.g0; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=hcore-g0; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/hcore.rs::periodic_hcore; rust_test=ferric-pbc/tests/pbc_hcore.rs::periodic_hcore_is_omega_independent_triclinic_sp; registry=pbc-ewald-pairft-stress -->

## 24.3.2 The algorithm

| Step | Quantity | Cost |
|---|---|---|
| 1 | $S$, $T$ by lattice-summed shifted one-electron blocks | $N_\text{shell}^2N_{\mathbf L}$ |
| 2 | $V_\text{SR}$: erfc three-centre integrals of $\langle\mu_0|\cdot|\nu_{\mathbf L}\rangle$ with Gaussian nuclei at $\mathbf R_C + \mathbf M$, screened per triplet | $N_\text{shell}^2N_{\mathbf L}N_\text{nuc}$ |
| 3 | $V_\text{LR}$ on the half $\mathbf G$ sphere (weight 2): $-(2/\Omega)\sum v_\eta(\mathbf G)\operatorname{Re}[P^*_{\mu\nu}S(\mathbf G)]$ | $N_\text{pair}N_{\mathbf G}$ |
| 4 | $V_\text{G0} = c_0Z_\text{tot}S$ | $n_\text{ao}^2$ |

libint2's `erf_nuclear` and `erfc_nuclear` kernels are not used: they are
wrong in the libint2 versions ferric builds against. The Gaussian nuclei of
exponent $10^{16}$ go through the ordinary erfc three-centre engine instead;
the finite-nucleus error is about $\pi Z\rho(\mathbf R_C)/\zeta$.

The default split is $\eta = \min(2.5\sqrt\pi/\Omega^{1/3}, \eta_\text{cap})$
(`site/src/methods/pbc.md`); any value gives the same matrix up to
truncation.

## 24.3.3 Numerical anchors

| Quantity | Reference | Test |
|---|---|---|
| $V(\eta)$ at three $\eta$ | the pure-AFT oracle `pure_aft_nuclear` | `crates/ferric-pbc/tests/pbc_hcore.rs` `periodic_hcore_is_omega_independent_triclinic_sp` |
| one s primitive | the closed form $V_\text{erf} = -(2/\sqrt\pi)(1/\eta^2 + 1/p + 1/\zeta)^{-1/2}$ | `single_s_primitive_matches_closed_form` |
| erf + erfc on a Gaussian nucleus | the point nucleus | `gaussian_nucleus_erf_plus_erfc_is_point_nuclear` |

## 24.3.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §3 | the split and its constant |
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S3 | the proof |
| Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020) | PySCF's periodic conventions |
