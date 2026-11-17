# MP2 static polarizability {#mp2-polarizability}

<!-- status: ready; registry: mp2-polarizability;
     theory: crates/ferric-mp2/theory.md#"10. Static polarizability" -->

Put a molecule in a uniform electric field and its electrons shift: the
dipole changes in proportion to the field, and the proportionality constant
is the static polarizability. Hartree–Fock gets the shift of the mean-field
density. Correlation changes it, because the correlated electrons respond to
the field and to each other's response. In energy terms the polarizability
is a second derivative, $\alpha_{de} = -\partial^2E/\partial F_d\partial F_e$,
so for MP2 it is the field derivative of the MP2 dipole. The relaxed density
of section @sec:lagrangian-zvector already gives that dipole: a field enters as a one-electron
perturbation, and the relaxed density's contraction with any one-electron
perturbation is the energy derivative.

ferric computes it two ways, both library only: by finite field with the SCF
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

## The relaxed dipole {#mp2-polarizability-relaxed-dipole}

A uniform field $F$ along axis $d$ enters the core Hamiltonian as
$h \to h + F\,r_d$, with $r_d$ the AO dipole integrals about the origin and
the electron charge $-1$. Then

$$ \mu_d = -\frac{\partial E}{\partial F_d}, \qquad \alpha_{de} = -\frac{\partial^2E}{\partial F_d\,\partial F_e} = \frac{\partial\mu_d}{\partial F_e}, \qquad E = E_{\mathrm{RHF}} + E^{(2)} . $$

By section @sec:lagrangian-zvector-relaxed-density with $R = r_d$, the first derivative is a trace with the
relaxed density:

$$ \mu_d = -\operatorname{Tr}[P^{\mathrm{rel}}r_d] + \sum_AZ_AR_{A,d}, \qquad P^{\mathrm{rel}} = 2\delta_{ij} + (P + P^{\mathsf T}) + z, \label{eq:mp2-polar-relaxed-dipole} $$

with $z$ from the Z-vector of section @sec:lagrangian-zvector and right-hand side
$X_{ai} = L_{ai} + [(2J - K)[P + P^{\mathsf T}]]_{ai}$. This $z$ equals the
gradient path's, and $\mu_d(0) = -\partial E/\partial F_d$. The nuclear term
does not depend on the field, so only the electronic part contributes to
$\alpha$.

> **Verified equation @eq:mp2-polar-relaxed-dipole** · the relaxed dipole is the field derivative
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
polarizability path builds its $z$ with a different right-hand side. Its
$L$ is `build_lagrangian`:

$$ L_{ck} = \sum_jP_{kj}F_{cj} + \sum_bP_{cb}F_{bk} + 2\sum_{jab}\tilde t_{kj}^{ab}(ca|jb) - 2\sum_{ijb}\tilde t_{ij}^{cb}(ik|jb), $$

with $\tilde t = 2t - t^{\mathrm{swap}}$, written in the code as four loops (the
two $\tilde t$ terms each appear twice, once per amplitude slot). At
canonical orbitals the $PF$ terms vanish, because $F_{cj} = F_{bk} = 0$
between occupied and virtual orbitals, and $L_{ck} = I_{kc} - I_{ck}$ with the
$I$ of `build_imat_ri`. So it is the same $z$.

> **Verified equation {@eq:mp2-polar-relaxed-dipole}b** · `static_relaxed_density_ao` is the gradient's relaxed density
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

The finite-field driver, `ferric-mp2/src/ff_polar.rs::mp2_polarizability_static`,
adds $\mp h\,r_e$ to the core Hamiltonian, re-converges the RHF at each sign,
builds this density on each, and takes the central difference of $\mu_d$,
then symmetrizes; its default step is $10^{-3}$. Each re-converged SCF
brings its own convergence noise into the difference.

## The analytic polarizability {#mp2-polarizability-analytic-polarizability}

The analytic route differentiates the relaxed density along a path that
needs no SCF. Per field axis $x$:

1. Solve the dipole CPHF $(A + B)U^x = -r^x_{vo}$, the first-order orbital
   response to the field, and form the first-order Fock response
   $$ \partial F = r^x + J[\partial D] - \tfrac12K[\partial D], \qquad \partial D = 2(\mathbf C_vU^x\mathbf C_o^{\mathsf T} + \mathbf C_oU^{x\mathsf T}\mathbf C_v^{\mathsf T}) . $$
2. Take the canonical orbitals of $F \pm h\,\partial F$ as the reference at
   field $\pm h$.
3. Difference the relaxed density of @sec:mp2-polarizability-relaxed-dipole on the two references:

$$ \alpha_{xy} = -\operatorname{Tr}\Big[\frac{P^{\mathrm{rel}}(+h) - P^{\mathrm{rel}}(-h)}{2h}\,r_y\Big] . \label{eq:mp2-polar-analytic} $$

This is $\partial\mu_y/\partial F_x$ by (@eq:mp2-polar-relaxed-dipole), since the nuclear term drops
out.

Why is step 2 good enough? The first-order reference has the occupied space,
orbital energies and $oo$/$vv$ Fock blocks of the field-$h$ SCF to $O(h^2)$.
The off-diagonal $oo$/$vv$ Fock response is re-canonicalized away by the
diagonalization. The $O(h^2)$ error is even in $h$, so it cancels in the
central difference, and the difference carries only its own $O(h^2)$
truncation. No SCF is re-converged, so the difference carries no SCF noise.
The tensor is then symmetrized.

A hand assembly $\partial P + \partial z + 2U$ with the canonical-denominator
$\partial t$ would miss two contributions that the reference construction
carries: the off-diagonal $oo$/$vv$ Fock response, and the rotation
$\Theta D^{\mathrm{corr}} + D^{\mathrm{corr}}\Theta^{\mathsf T}$ of
$D^{\mathrm{corr}} = P + P^{\mathsf T} + z$. On water/STO-3G they contribute
0.0009 and $-0.0054$ to $\alpha_{zz}$, and 0.048 and $-0.040$ to
$\alpha_{yy}$.

The difference itself is in `ferric-mp2/src/cpks_polar.rs::relaxed_alpha_tensors`:

<!-- excerpt: ferric-mp2/src/cpks_polar.rs::relaxed_alpha_tensors -->
```rust
let dd_hf = (&d_hf[0] - &d_hf[1]) / (2.0 * h);
let dd_rel = (&d_rel[0] - &d_rel[1]) / (2.0 * h);
for y in 0..3 {
    a_hf[x][y] = -(&dd_hf * &dip_ao[y]).sum();
    a_mp2[x][y] = -(&dd_rel * &dip_ao[y]).sum();
}
```

| Code | Equation |
|---|---|
| `d_rel[0]`, `d_rel[1]` | $P^{\mathrm{rel}}(+h)$, $P^{\mathrm{rel}}(-h)$ from `static_relaxed_density_ao` on `reference_at_field` |
| `dd_rel` | $\big(P^{\mathrm{rel}}(+h) - P^{\mathrm{rel}}(-h)\big)/2h$ |
| `a_mp2[x][y]` | $\alpha_{xy}$ of (@eq:mp2-polar-analytic), before symmetrization |
| `dd_hf`, `a_hf` | the same for the SCF density |

`ferric-mp2/src/cpks_polar.rs::analytic_alpha_relaxed` then symmetrizes:

<!-- excerpt: ferric-mp2/src/cpks_polar.rs::analytic_alpha_relaxed -->
```rust
let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
tensor[i][j] = avg;
tensor[j][i] = avg;
```

**Exactness anchor.** The same difference of the SCF density reproduces the
CPHF polarizability. Follow it through: the SCF density moves by
$\partial D = 2(\mathbf C_vU^x\mathbf C_o^{\mathsf T} + \text{h.c.})$, so
$-\operatorname{Tr}[\partial D\,r_y] = -4\sum_{ai}(r_y)_{ai}U^x_{ai}$, and
substituting $U^x = -(A + B)^{-1}r^x_{vo}$ gives

$$ \alpha^{\mathrm{HF}}_{xy} = 4\,\mu_x^{\mathsf T}(A + B)^{-1}\mu_y, \qquad \mu_{d,ai} = (\mathbf C_v^{\mathsf T}r_d\mathbf C_o)_{ai} . $$

With no correlation ($t = 0$) the MP2 tensor equals it exactly.

> **Verified equation @eq:mp2-polar-analytic** · the analytic polarizability
>
> With no correlation the coded tensor is the CPHF $\alpha^{\mathrm{HF}}$
> exactly (claim P0). With correlation the $h \to 0$ limit of (@eq:mp2-polar-analytic) equals
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

On the script's model P3 holds to $6\times10^{-9}$ in the $h \to 0$ limit,
and the $h = 10^{-4}$ value is off by its own $6\times10^{-7}$ $O(h^2)$
term.

### HF dynamic response and the frozen-amplitude C6

A $C_6$ dispersion coefficient needs the polarizability at imaginary
frequency, $\alpha(iu)$, not only at $u = 0$ (section @sec:casimir-polder-c6).
`dynamic_cphf_alpha_iw` returns
$\alpha(iu) = 4\mu^{\mathsf T}[(A-B)(A+B) + u^2]^{-1}(A-B)\mu$, the TDHF
response at imaginary frequency. Diagonalize
$\boldsymbol\Omega^2 = (A-B)^{1/2}(A+B)(A-B)^{1/2}$ and it becomes a sum over
excitation energies $\Omega_n$ with oscillator strengths $f_n$,
$\alpha^{\mathrm{HF}}_{\mathrm{iso}}(iu) = \sum_nf_n/(\Omega_n^2 + u^2)$. At
$u = 0$ the matrix product collapses to $(A + B)^{-1}$, the static CPHF
$\alpha$ of the anchor above.

This path has no MP2 $\alpha(iu)$. `frozen_mp2_c6_molecular` borrows the
HF frequency shape and rescales its isotropic part to the static MP2
magnitude,
$\alpha^{\mathrm{MP2}}(iu) = s\,\alpha^{\mathrm{HF}}(iu)$ with
$s = \alpha^{\mathrm{MP2}}(0)/\alpha^{\mathrm{HF}}(0)$ ($s = 1$ if
$|\alpha^{\mathrm{HF}}_{\mathrm{iso}}| \le 10^{-12}$), and integrates
$C_6 = \tfrac3\pi\sum_kw_k\,\alpha^{\mathrm{MP2}}(iu_k)^2$. It reproduces the
static MP2 $\alpha$ at $u = 0$ by construction. The numerator of $s$ uses
four-center $J/K$ and the denominator the RI-factor $(A + B)$, so $s$ also
absorbs the RI error of the HF static term.

> **Verified equation {#eq:mp2-polar-c6-frozen}** · HF $\alpha(iu)$ and the frozen-amplitude C6
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
operator. The rule is the one of section @sec:rimp2-gradient-attenuated-erfc-ri-mp2: each term takes the
operator of what it differentiates. With a Coulomb RHF and an
$\operatorname{erfc}$ MP2 operator the result is the polarizability of
attenuated MP2, checked against a 5-point finite field of that energy
(`attenuated_relaxed_alpha_matches_energy_finite_field`).

## Refused inputs {#mp2-polarizability-refused-inputs}

A non-restricted reference and `frozen_core > 0` (there is no core-active
relaxation block on this path).

## Numerical anchors {#mp2-polarizability-numerical-anchors}

From `site/src/reference/validation.md` (the RI-MP2 polarizability rows).

| Quantity | Systems | Error | Test |
|---|---|---|---|
| analytic relaxed $\alpha$ vs PySCF finite-field $-d^2E/dF^2$ (RHF + DF-MP2, same aux basis) | H2O / STO-3G, cc-pVDZ; NH3 in a generic orientation / cc-pVDZ | $\le 2.2\times10^{-6}$ relative | `relaxed_alpha_matches_pyscf_energy_hessian` in `crates/ferric-mp2/tests/cpks_polar.rs` |
| HF anchor: difference of the SCF density vs CPHF | the same | $4\times10^{-9}$ relative | the same file |
| relaxed $\mu_z$ vs PySCF finite field | water / STO-3G, cc-pVDZ | $3\times10^{-8}$, $2\times10^{-8}$ a.u. | `crates/ferric-mp2/tests/cpks_polar.rs` |
| finite-field relaxed $\alpha$ | water / STO-3G (diagonal), cc-pVDZ ($\alpha_{zz}$) | $3\times10^{-6}$, $1.4\times10^{-5}$ a.u. | `crates/ferric-mp2/tests/ff_polar.rs` |

## Sources {#mp2-polarizability-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, "10. Static polarizability" | (@eq:mp2-polar-relaxed-dipole) to (@eq:mp2-polar-c6-frozen), the response matrices |
| `scripts/mp2_polarizability_sympy.py`, claims P0 to P5 | the proofs |
| `scripts/cpks/ff_dfmp2_dipole_alpha.py` | the PySCF finite-field reference |

**Known limits (2026-11-15).** Library only, closed shell, no frozen core.
The frozen-amplitude C6 takes its frequency dependence from HF; it has no
MP2 correction to the shape.
