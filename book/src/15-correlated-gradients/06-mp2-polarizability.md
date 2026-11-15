# 15.6 MP2 static polarizability

<!-- section: 15.6; status: ready; registry: mp2-polarizability;
     theory: crates/ferric-mp2/theory.md#"10. Static polarizability" -->

The relaxed density of section 15.1 is the derivative of the energy with
respect to any one-electron perturbation, so its dipole is the MP2 dipole,
and the field derivative of that dipole is the static polarizability. ferric
computes it two ways, both library only: by finite field with the SCF
re-converged at each field, and by a central difference of the relaxed
density along an analytic first-order reference. The derivation is
`crates/ferric-mp2/theory.md` section 10; `scripts/mp2_polarizability_sympy.py`
proves the relaxed dipole, the analytic polarizability and the
frozen-amplitude C6 on a model with the SCF re-solved at every field.

| Symbol | Meaning | Code name |
|---|---|---|
| $r_d$ | dipole integrals, axis $d$ | `dip_ao` |
| $P^{\mathrm{rel}}$ | relaxed RI-MP2 AO density | `static_relaxed_density_ao` |
| $U^x$ | dipole CPHF response | |
| $\partial F$ | first-order Fock response | `first_order_fock_response` |
| $h$ | field step | `LR_FIELD_STEP` ($10^{-4}$) |

| Entry point | Location |
|---|---|
| Rust, analytic | `ferric-mp2/src/cpks_polar.rs::{mp2_polarizability_analytic,analytic_alpha_relaxed}` |
| Rust, finite field | `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static` |
| CLI, Python | none |

## 15.6.1 The relaxed dipole

$$ \mu_d = -\operatorname{Tr}[P^{\mathrm{rel}}r_d] + \sum_AZ_AR_{A,d}, \qquad P^{\mathrm{rel}} = 2\delta_{ij} + (P + P^{\mathsf T}) + z, \tag{15.15} $$

with $z$ from the Z-vector of section 15.1 and right-hand side
$X_{ai} = L_{ai} + [(2J - K)[P + P^{\mathsf T}]]_{ai}$. This $z$ equals the
gradient path's, and $\mu_d(0) = -\partial E/\partial F_d$.

> **Verified equation 15.15** · the relaxed dipole is the field derivative
>
> For $h \to h + \lambda R$, $dE/d\lambda = \operatorname{Tr}[(D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}})R]$ for the `solve_zvector` relaxed density (part G1).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part G |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/zvector.rs::{solve_zvector,build_relaxed_density_ao}` |
> | Code test | `cpks_static_relaxed_dipole_vs_pyscf` in `crates/ferric-mp2/tests/cpks_polar.rs` |
>
> <!-- verified: id=mp2.polar.relaxed-dipole; script=scripts/mp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/zvector.rs::{solve_zvector,build_relaxed_density_ao}; rust_test=ferric-mp2/tests/cpks_polar.rs::cpks_static_relaxed_dipole_vs_pyscf -->

The script proves the identity for the `solve_zvector` density. The
polarizability path builds its $z$ with a different right-hand side, the
four-term `build_lagrangian` loop $L$ plus $(2J - K)[P + P^{\mathsf T}]$; at
canonical orbitals $L_{ck} = I_{kc} - I_{ck}$, so it is the same $z$.

> **Verified equation 15.15b** · `static_relaxed_density_ao` is the gradient's relaxed density
>
> The right-hand side $L + (2J - K)[P + P^{\mathsf T}]_{vo}$ equals the
> `solve_zvector` right-hand side and the two $z$ agree to $10^{-14}$ (claim
> P1); $-\operatorname{Tr}[P^{\mathrm{rel}}r_d]$ equals $-dE/dF_d$ with the RHF
> re-solved at each field (claim P2).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_polarizability_sympy.py`, claims P1, P2 |
> | Negative control | the runner flips the sign of one `build_lagrangian` term (P1), and drops the $(2J - K)[P + P^{\mathsf T}]$ term (P2) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/cpks_polar.rs::static_relaxed_density_ao` |
> | Code test | `cpks_static_relaxed_dipole_vs_pyscf` in `crates/ferric-mp2/tests/cpks_polar.rs` |
>
> <!-- verified: id=mp2.polar.static-density; script=scripts/mp2_polarizability_sympy.py; claim=P1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/cpks_polar.rs::static_relaxed_density_ao; rust_test=ferric-mp2/tests/cpks_polar.rs::cpks_static_relaxed_dipole_vs_pyscf; registry=mp2-polarizability -->

## 15.6.2 The analytic polarizability

Per field axis $x$: solve the dipole CPHF $(A + B)U^x = -r^x_{vo}$, form

$$ \partial F = r^x + J[\partial D] - \tfrac12K[\partial D], \qquad \partial D = 2(\mathbf C_vU^x\mathbf C_o^{\mathsf T} + \mathbf C_oU^{x\mathsf T}\mathbf C_v^{\mathsf T}), $$

take the canonical orbitals of $F \pm h\,\partial F$ as the reference at
field $\pm h$, and

$$ \alpha_{xy} = -\operatorname{Tr}\Big[\frac{P^{\mathrm{rel}}(+h) - P^{\mathrm{rel}}(-h)}{2h}\,r_y\Big] . \tag{15.16} $$

The first-order reference has the occupied space, orbital energies and
$oo$/$vv$ Fock blocks of the field-$h$ SCF to $O(h^2)$, and no SCF is
re-converged, so the difference carries no SCF noise. The tensor is then
symmetrized.

<!-- excerpt: ferric-mp2/src/cpks_polar.rs::analytic_alpha_relaxed -->
```rust
let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
tensor[i][j] = avg;
tensor[j][i] = avg;
```

| Symbol | Code |
|---|---|
| $\alpha_{xy}$ from (15.16), before symmetrization | `tensor`, from `relaxed_alpha_tensors` |

The difference itself is in `relaxed_alpha_tensors`: `dd_rel` is
$\big(P^{\mathrm{rel}}(+h) - P^{\mathrm{rel}}(-h)\big)/2h$ and
`a_mp2[x][y] = -(&dd_rel * &dip_ao[y]).sum()`; `dd_hf` and `a_hf` are the
same for the SCF density.

**Exactness anchor.** The same difference of the SCF density reproduces the
CPHF $\alpha^{\mathrm{HF}} = 4\mu^{\mathsf T}(A + B)^{-1}\mu$.

> **Verified equation 15.16** · the analytic polarizability
>
> With no correlation the coded tensor is the CPHF $\alpha^{\mathrm{HF}}$
> exactly (claim P0). With correlation the $h \to 0$ limit of (15.16) equals
> $-\partial^2(E_{\mathrm{RHF}} + E^{(2)})/\partial F_x\partial F_y$ from a
> finite-field energy Hessian with the RHF re-solved, for $xx$, $yy$ and $xy$,
> and the $h = 10^{-4}$ value differs only by its own $O(h^2)$ term (claim P3).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_polarizability_sympy.py`, claims P0, P3 |
> | Negative control | the runner doubles $J[\partial D]$ in the first-order Fock response |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/cpks_polar.rs::{analytic_alpha_relaxed,first_order_fock_response,reference_at_field}` |
> | Code test | `relaxed_alpha_matches_pyscf_energy_hessian` in `crates/ferric-mp2/tests/cpks_polar.rs` |
>
> <!-- verified: id=mp2.polar.analytic; script=scripts/mp2_polarizability_sympy.py; claim=P3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/cpks_polar.rs::{analytic_alpha_relaxed,first_order_fock_response,reference_at_field}; rust_test=ferric-mp2/tests/cpks_polar.rs::relaxed_alpha_matches_pyscf_energy_hessian; registry=mp2-polarizability -->

**HF dynamic response and the frozen-amplitude C6.**
`dynamic_cphf_alpha_iw` returns
$\alpha(iu) = 4\mu^{\mathsf T}[(A-B)(A+B) + u^2]^{-1}(A-B)\mu$, the TDHF
response at imaginary frequency; `frozen_mp2_c6_molecular` rescales its
isotropic shape to the static MP2 magnitude,
$\alpha^{\mathrm{MP2}}(iu) = s\,\alpha^{\mathrm{HF}}(iu)$ with
$s = \alpha^{\mathrm{MP2}}(0)/\alpha^{\mathrm{HF}}(0)$, and integrates
$C_6 = \tfrac3\pi\sum_kw_k\,\alpha^{\mathrm{MP2}}(iu_k)^2$.

> **Verified equation 15.17** · HF $\alpha(iu)$ and the frozen-amplitude C6
>
> $\tfrac12$`compute_az_product` $+ \Delta$ is the $(A+B)$ of `build_apb_amb`;
> the coded $\alpha(iu)$ equals a complex TDHF linear-response solve at
> $u = 0, 0.3, 1.7$ and its $u = 0$ value is the static CPHF $\alpha$ (claim
> P4); the rescaled C6 equals the closed-form pole sum
> $\tfrac32 s^2\sum_{mn}f_mf_n/(\Omega_m\Omega_n(\Omega_m + \Omega_n))$ (claim P5).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_polarizability_sympy.py`, claims P4, P5 |
> | Negative control | the runner replaces $u^2$ by $u$ in the system matrix (P4), and uses $3/(2\pi)$ (P5) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/cpks_polar.rs::{dynamic_cphf_alpha_iw,frozen_mp2_c6_molecular}` |
> | Code test | `cpks_dynamic_alpha_w0_matches_static` in `crates/ferric-mp2/tests/cpks_polar.rs` |
>
> <!-- verified: id=mp2.polar.c6-frozen; script=scripts/mp2_polarizability_sympy.py; claim=P5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/cpks_polar.rs::{dynamic_cphf_alpha_iw,frozen_mp2_c6_molecular}; rust_test=ferric-mp2/tests/cpks_polar.rs::cpks_dynamic_alpha_w0_matches_static; registry=mp2-polarizability -->

**Operators.** The SCF response terms (CPHF, $\partial F$, the
$(2J - K)[P + P^{\mathsf T}]$ term, the Z-vector Hessian) use the operator the
RHF was converged with; the MP2 amplitudes and Lagrangian use the MP2
operator. With a Coulomb RHF and an $\operatorname{erfc}$ MP2 operator the
result is the polarizability of attenuated MP2.

## 15.6.3 Refused inputs

A non-restricted reference and `frozen_core > 0` (there is no core-active
relaxation block on this path).

## 15.6.4 Numerical anchors

From `site/src/reference/validation.md` (the RI-MP2 polarizability rows).

| Quantity | Systems | Error | Test |
|---|---|---|---|
| analytic relaxed $\alpha$ vs PySCF finite-field $-d^2E/dF^2$ (RHF + DF-MP2, same aux basis) | H2O / STO-3G, cc-pVDZ; NH3 in a generic orientation / cc-pVDZ | $\le 2.2\times10^{-6}$ relative | `relaxed_alpha_matches_pyscf_energy_hessian` in `crates/ferric-mp2/tests/cpks_polar.rs` |
| HF anchor: difference of the SCF density vs CPHF | the same | $4\times10^{-9}$ relative | the same file |
| relaxed $\mu_z$ vs PySCF finite field | water / STO-3G, cc-pVDZ | $3\times10^{-8}$, $2\times10^{-8}$ a.u. | `crates/ferric-mp2/tests/cpks_polar.rs` |
| finite-field relaxed $\alpha$ | water / STO-3G (diagonal), cc-pVDZ ($\alpha_{zz}$) | $3\times10^{-6}$, $1.4\times10^{-5}$ a.u. | `crates/ferric-mp2/tests/ff_polar.rs` |

## 15.6.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, "10. Static polarizability" | (15.15) to (15.17), the response matrices |
| `scripts/mp2_polarizability_sympy.py`, claims P0 to P5 | the proofs |
| `scripts/cpks/ff_dfmp2_dipole_alpha.py` | the PySCF finite-field reference |

**Known limits (2026-11-15).** Library only, closed shell, no frozen core.
The frozen-amplitude C6 takes its frequency dependence from HF; it has no
MP2 correction to the shape.
