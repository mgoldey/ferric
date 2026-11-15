# 21.1 Static and dynamic polarizabilities from the dielectric modes

<!-- section: 21.1; status: ready; registry: rpa-polarizability-c6;
     theory: crates/ferric-rpa/theory.md#"Dispersion: Casimir-Polder C6" -->

A uniform field $F$ along $x$ perturbs every orbital pair $i\to a$ by
$-F\,\mu_{x,ia}$. In direct RPA the electrons respond to the field and to each
other's induced Coulomb potential, with no exchange kernel, so the response
is the same dielectric matrix that chapter 18 builds for the correlation
energy. The molecular polarizability, static or at an imaginary frequency
$iu$, is therefore a by-product of that matrix: one linear solve against
$\tilde{\boldsymbol\varepsilon}(iu)$ per axis, in the auxiliary basis, never in
the $n_on_v$ space.

This is not the response of section 19.9. That section screens an exchange-type
kernel and is Smoke grade; this one is direct RPA, whose polarizability is
the static limit of the same $\alpha(iu)$ that section 25.2 integrates into
$C_6$.

| Symbol | Meaning | Code name |
|---|---|---|
| $\Delta_i^a$ | $\epsilon_a - \epsilon_i$ | `e_ia` |
| $g_{ia}(u)$ | $\Delta_i^a/(u^2 + (\Delta_i^a)^2)$; $1/\Delta_i^a$ at $u = 0$ | `g`, `inv_de` |
| $\mu_{d,ia}$ | MO dipole block, $d = x,y,z$, about the origin $\mathbf 0$ | `mu_flat` |
| $B^P_{ia}$ | dressed three-index tensor, occupied–virtual block | `b_ov` |
| $\tilde{\boldsymbol\varepsilon}(iu)$ | $\mathbf 1 + 4\,B\,\mathrm{diag}(g)\,B^{\mathsf T}$ (closed shell) | `eps_mat` |
| $w_d$ | $B\,(g\circ\mu_d)$ | `w_vec` |
| $\lambda_\xi$, $U_\xi$ | dielectric eigenvalues and eigenvectors at $iu$ | `ResponseModes::eigenvalues`, the coefficients |
| $s_{\xi,d}$ | field coupling of mode $\xi$, $4\,U_\xi\!\cdot w_d$ | `field_couplings` |

| Entry point | Location |
|---|---|
| Static | `ferric-rpa/src/properties.rs::{pdep_polarizability_static,pdep_polarizability_static_unrestricted}` |
| Dynamic | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
| Modes | `ferric-rpa/src/response_modes.rs::{response_modes,polarizability_from_modes}` |
| CLI | `method.kind = "pdep-rpa"` with `[rpa] compute_polarizability` (on by default under `export_npz`); `[rpa] compute_c6` for the frequency grid |
| Python | `run_pdep_rpa(properties=True)` giving `alpha_tensor`; `response_modes(..., n_modes=, omega=)` |

## 21.1.1 The dielectric form

In singlet direct RPA, $\mathbf A - \mathbf B = \Delta$ and
$\mathbf A + \mathbf B = \Delta + 4\mathbf K$ with $K_{ia,jb} = (ia|jb) =
\sum_PB^P_{ia}B^P_{jb}$, so the response formula of section 19.9 gives

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\big[\Delta(\Delta + 4\mathbf K) + u^2\big]^{-1}\Delta\,\mu_e . \tag{21.1} $$

The Woodbury identity turns the $n_on_v$-sized inverse into one of size
$N_{\rm aux}$:

$$ \alpha_{de}(iu) = 4\,\mu_d^{\mathsf T}\mathrm{diag}(g)\,\mu_e - 16\,w_d^{\mathsf T}\,\tilde{\boldsymbol\varepsilon}(iu)^{-1}\,w_e , \qquad w_d = B\,(g\circ\mu_d) . \tag{21.2} $$

The first term is the polarizability of independent particles, and the
second is the screening by the induced density. At $u = 0$ every $g_{ia}$ is
$1/\Delta_i^a$ and (21.2) is the static polarizability.

<!-- excerpt: ferric-rpa/src/properties.rs::pdep_polarizability_static -->
```rust
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare = mu_flat[i].dot(&mu_flat_inv[j]); // = μ^i^T D^{-1} μ^j
            let coupled = w_vec[i].dot(&y_vec[j]);
            tensor[i][j] = 4.0 * bare - 16.0 * coupled;
        }
    }
```

| Symbol | Variable |
|---|---|
| $\mu_d^{\mathsf T}\mathrm{diag}(g)\,\mu_e$ | `bare` (here with $g = 1/\Delta$) |
| $w_d^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}w_e$ | `coupled`, with `y_vec` $= \tilde{\boldsymbol\varepsilon}^{-1}w$ from `solve_dielectric_3` |
| 4, 16 | the prefactor of the independent-particle term (2 for spin, 2 for the pair of poles), and 16 = 4 (that prefactor) $\times$ 4 (the dielectric's $4BD^{-1}B^{\mathsf T}$) |

The tensor is then symmetrized and its principal values taken. The molecular
dipole is the whole-molecule lab-frame dipole on both sides, so the tensor is
origin-independent.

> **Verified equation 21.1** · the dielectric form equals the direct response solve
>
> The dielectric form (21.2) equals an independent direct solve of (21.1) in
> the $n_on_v$ space, for the full tensor, at $u=0$ and $u=0.9$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M1 |
> | Negative control | the runner changes the screening prefactor 16 to 4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties.rs::pdep_polarizability_static` |
> | Code test | `alpha_row` in `crates/ferric-rpa/tests/validation_static_alpha.rs` |
>
> <!-- verified: id=pol.dielectric-form; script=scripts/rpa_response_modes_sympy.py; claim=M1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties.rs::pdep_polarizability_static; rust_test=ferric-rpa/tests/validation_static_alpha.rs::alpha_row; registry=rpa-polarizability-c6 -->

The same response is the sum over the direct-RPA (Casida) excitations,
$\alpha(iu) = \sum_nf_n/(\Omega_n^2 + u^2)$, with $\Omega_n^2$ the eigenvalues of
$\Delta^{1/2}(\Delta+4\mathbf K)\Delta^{1/2}$ and $f_n$ the oscillator weights
of the transition dipole. It is the independent route of the validation
below.

> **Verified equation 21.2** · the kernel against the sum over excitations
>
> The coded kernel $4\,m\cdot(g\circ\mu) - 16\,(B(g\circ m))^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}(B(g\circ\mu))$
> equals the sum over the closed-shell dRPA excitations for any bra $m$, at
> $u = 0$, $0.4$ and $1.7$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_c6_polarizability_sympy.py`, check `kernel-vs-casida` |
> | Negative control | the runner changes the kernel's screening term 16 to 4 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha` |
> | Code test | `crates/ferric-rpa/tests/validation_pdep_c6.rs` |
>
> <!-- verified: id=pol.kernel-vs-casida; script=scripts/rpa_c6_polarizability_sympy.py; claim=kernel-vs-casida; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties/per_atom.rs::closed_per_atom_alpha; registry=rpa-polarizability-c6 -->

## 21.1.2 The mode form

Diagonalize $\tilde{\boldsymbol\varepsilon}(iu) = \sum_\xi\lambda_\xi U_\xi U_\xi^{\mathsf T}$. Then
$\tilde{\boldsymbol\varepsilon}^{-1} = \sum_\xi U_\xi U_\xi^{\mathsf T}/\lambda_\xi$ and (21.2) becomes

$$ \alpha(iu) = \alpha_{\rm bare}(iu) - \sum_\xi\frac{s_\xi s_\xi^{\mathsf T}}{\lambda_\xi},\qquad s_{\xi,d} = 4\,U_\xi\cdot w_d,\quad \alpha_{\rm bare} = 4\,\mu^{\mathsf T}\mathrm{diag}(g)\,\mu . \tag{21.3} $$

Each mode is a charge density in the auxiliary basis; $s_\xi$ is its Coulomb
overlap with the density that a unit field induces in independent particles,
and $s_\xi s_\xi^{\mathsf T}/\lambda_\xi$ is how much that mode screens
the field. This is the form `ResponseModes::polarizability_from_modes`
evaluates. It is exact when every mode is kept, and a truncation to the
leading modes otherwise, because the modes are ordered by $\lambda_\xi$, the
strength of the screening.

> **Verified equation 21.3** · the mode expansion
>
> With every mode kept, $\alpha_{\rm bare} - \sum_\xi s_\xi s_\xi^{\mathsf T}/\lambda_\xi$
> equals (21.2); keeping 3 of 7 modes does not.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M2 |
> | Negative control | the 3-of-7 truncation must differ; the runner flips the sign of the mode sum |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/response_modes.rs::polarizability_from_modes` |
> | Code test | `full_rank_modes_reconstruct_the_static_polarizability` in `crates/ferric-rpa/tests/response_modes.rs` |
>
> <!-- verified: id=pol.mode-expansion; script=scripts/rpa_response_modes_sympy.py; claim=M2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/response_modes.rs::polarizability_from_modes; rust_test=ferric-rpa/tests/response_modes.rs::full_rank_modes_reconstruct_the_static_polarizability; registry=rpa-polarizability-c6 -->

At $u=0$ the modes are the PDEP eigenpotentials of section 18.3 (test
`leading_static_modes_are_the_pdep_eigenpotentials`). The physical coefficients
$c_\xi = M^{\mathsf T}U_\xi$ obey $(V + \Pi(iu))c_\xi = \lambda_\xi Vc_\xi$ with
$c_\xi^{\mathsf T}Vc_\eta = \delta_{\xi\eta}$, and an added density
$\rho_\xi$ is screened self-consistently to $\rho_\xi/\lambda_\xi$. A mode is
a charge density $\sum_Pc_P\chi_P(\mathbf r)$, and its potential is the
Coulomb potential of that density: reading the coefficients as a potential
is wrong.

## 21.1.3 Open shell

With different $\alpha$ and $\beta$ orbitals the screening is shared and the
bare terms are per spin:

$$ \tilde{\boldsymbol\varepsilon}_U = \mathbf 1 + 2\sum_\sigma B_\sigma\,\mathrm{diag}(g_\sigma)B_\sigma^{\mathsf T},\qquad \alpha_{de} = \sum_\sigma 2\,\mu_{\sigma,d}^{\mathsf T}\mathrm{diag}(g_\sigma)\mu_{\sigma,e} - 4\,w_d^{\mathsf T}\tilde{\boldsymbol\varepsilon}_U^{-1}w_e,\quad w_d = \sum_\sigma B_\sigma(g_\sigma\circ\mu_{\sigma,d}) , \tag{21.4} $$

which is (21.2) when the spins coincide: $w$ doubles, so the screening term
$4\,(2w_\sigma)^{\mathsf T}\tilde{\boldsymbol\varepsilon}^{-1}(2w_\sigma) = 16\,w_\sigma^{\mathsf T}\cdots$.
ROHF and ROKS references are semi-canonicalized per spin first, as in
section 19.7.

> **Verified equation 21.4** · the open-shell form against the spin-orbital response
>
> For unequal $\alpha$ and $\beta$ spaces (2 occupied and 4 virtual, 1
> occupied and 5 virtual), (21.4) equals a direct solve of the spin-orbital
> direct-RPA response, $2\mu_{\rm so}^{\mathsf T}[\Delta(\Delta+2\mathbf K_{\rm so})+u^2]^{-1}\Delta\mu_{\rm so}$,
> at $u = 0$ and $0.9$.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_response_modes_sympy.py`, claim M4 |
> | Negative control | leaving the $\beta$ channel out of $w$ must miss; the runner changes the screening prefactor 4 to 16 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/properties.rs::molecular_dynamic_polarizability` |
> | Code test | `crates/ferric-rpa/tests/u_pdep_rpa.rs` |
>
> <!-- verified: id=pol.open-shell; script=scripts/rpa_response_modes_sympy.py; claim=M4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/properties.rs::molecular_dynamic_polarizability; registry=rpa-polarizability-c6 -->

## 21.1.4 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | reference (RHF or RKS), all occupied active (no frozen core: the response is on every occupied orbital) | | the caller |
| 2 | $B^P_{ia}$ and the gaps, the same intermediates as RPA | as RI-MP2 | `compute_rpa_intermediates` (ferric-mp2) |
| 3 | $\mu_{d,ia}$ from the AO dipole and the MO coefficients | $O(N^2 n_on_v)$ | `oneelectron::dipole` |
| 4 | $\tilde{\boldsymbol\varepsilon}(iu)$, one dense matrix per frequency | $N_{\rm aux}^2n_on_v$ | `pdep_polarizability_static`, `molecular_dynamic_polarizability` |
| 5 | one SPD solve for the three right-hand sides $w_d$ | $N_{\rm aux}^3$ | `solve_dielectric_3` |
| 6 | the tensor, symmetrized; principal values by a $3\times3$ eigensolve | $O(1)$ | `eig3_sym` |
| 6' | the mode form: a dense eigensolve of $\tilde{\boldsymbol\varepsilon}$, the leading $M$ modes and their couplings | $N_{\rm aux}^3$ | `response_modes` |

The dynamic driver loops over the imaginary-frequency nodes in parallel with
one BLAS thread per worker; the memory pre-flight charges `b_ov`, the scaled
copy `b_scaled` and `eps_mat` per spin, and the open-shell branch also charges
the second $N_{\rm aux}^2$ buffer that the sum of the two spins' products
needs. A solve of a non-positive-definite $\tilde{\boldsymbol\varepsilon}$
(a NaN dielectric) returns an error; it never panics inside the parallel
region.

## 21.1.5 Options and limits

| Option | Effect |
|---|---|
| `[rpa] n_quad`, `quadrature`, `u0` | the frequency grid of the dynamic tensor, and of $C_6$ |
| `memory_budget_gb` | the pre-flight ceiling |
| `n_modes`, `omega` (`response_modes`) | how many leading modes, and the imaginary frequency |

**Known limits (2026-11-15).** Direct RPA: the kernel has no exchange, so the
static tensor is not the CPHF polarizability. `response_modes` is closed shell
only, and refuses a negative or non-finite $u$ and `n_modes = 0`.

## 21.1.6 Numerical anchors

From `site/src/reference/validation.md` (the static $\alpha$ and the dynamic
$\alpha(iu)$ rows of the properties table):

| Quantity | Systems | Error | Test |
|---|---|---|---|
| static $\alpha$ against a numpy linear solve on PySCF `int3c2e`, same auxiliary basis | H2O / aug-cc-pVDZ; CH3OH / cc-pVDZ | $4.8\times10^{-14}$ relative on the same orbitals; $5.3\times10^{-10}$ with ferric's own SCF | `crates/ferric-rpa/tests/validation_static_alpha.rs` |
| $\alpha(iu)$ against a numpy sum over Casida excitations at ferric's 20 Gauss–Legendre nodes | H2O, N2 / aug-cc-pVDZ | $4.9\times10^{-14}$ relative on the same orbitals | `crates/ferric-rpa/tests/validation_pdep_c6.rs` |

## 21.1.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-rpa/theory.md`, "Dispersion: Casimir-Polder C6" | (21.1) to (21.4), the per-atom kernel |
| `crates/ferric-rpa/src/response_modes.rs`, module documentation | the mode form (21.3) and the meaning of a mode |
| `scripts/rpa_response_modes_sympy.py`, claims M1 to M4 | (21.2) to (21.4) |
| `scripts/rpa_c6_polarizability_sympy.py`, check `kernel-vs-casida` | the sum-over-excitations route |
