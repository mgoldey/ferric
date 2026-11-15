# 19.9 The statically screened TDHF polarizability

<!-- section: 19.9; status: ready; registry: tdhf-static-polarizability;
     theory: crates/ferric-gw/theory.md#"12. TDHF/RPAx polarizability" -->

The same static screened interaction as section 19.8, used in the full
response problem instead of the Tamm–Dancoff one: both the excitation block
$\mathbf A$ and the de-excitation block $\mathbf B$ carry a screened
exchange-type term, and the static polarizability follows from the linear
response of the dipole. With $W$ replaced by $v$ at a Hartree–Fock reference
the kernel is TDHF and the polarizability is the coupled-perturbed HF one; the
shipped kernel keeps the screened $W$, so it is a statically screened RPAx,
not TDHF.

This section is Smoke grade and says why. The implementation is checked
against independent assemblies of the same equations, but the physics is not
established: the polarizability is 46% low against the dipole oscillator
strength distribution (DOSD) reference on water.

| Symbol | Meaning | Code name |
|---|---|---|
| $\Delta_i^a$ | orbital-energy gap on the response diagonal, KS energies plus `scissor` on the virtuals | `eps` |
| $(\mathbf A\pm\mathbf B)_{ia,jb}$ | screened response matrices | `apb`, `amb` |
| $\mu_{x,ia}$ | occupied–virtual dipole blocks | `mu[d]` |
| $\alpha_{xy}$ | static polarizability tensor, a.u. | `RpaxStaticPolarizabilityResult::tensor` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/bse.rs::{run_rpax_static_polarizability,rpax_static_polarizability_with_screening}` |
| Dynamic, library only | `ferric-gw/src/bse.rs::{run_bse_c6,run_bse_c6_ks}` |
| CLI | `method.kind = "tdhf-static-polarizability"`; `[rpa] xc` is required; `[gw] scissor`; example `examples/water-tdhf-static-alpha.toml` |
| Python | `run_tdhf_static_polarizability(xc=, scissor=)` |

## 19.9.1 The response matrices

<!-- include: crates/ferric-gw/theory.md#"12. TDHF/RPAx polarizability" -->

The assembly follows the singlet formulas row by row:

<!-- excerpt: ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening -->
```rust
                let coul = bare(i, a_loc, j, b_loc);
                let w_abij = screened(a_loc, b_loc, i, j);
                let w_ibaj = screened(i, b_loc, a_loc, j);
                apb_row[jb] = 4.0 * coul - w_abij - w_ibaj;
                amb_row[jb] = w_ibaj - w_abij;
```

| Symbol | Variable |
|---|---|
| $(ia|jb)$ | `coul` |
| $(ab|W|ij)$ | `w_abij` |
| $(ib|W|ja)$ | `w_ibaj` |
| $(\mathbf A+\mathbf B)_{ia,jb}$ without the diagonal | `apb_row[jb]` |
| $(\mathbf A-\mathbf B)_{ia,jb}$ without the diagonal | `amb_row[jb]` |

> **Verified equation 19.16** · the response matrices are the spin adaptation of the spin-orbital RPAx matrices
>
> With the Hartree term $(ia|jb)$ between every spin pair and the screened
> terms $(ab|W|ij)$ and $(ib|W|ja)$ between equal spins only, rotating the
> spin index by $\tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ gives
> the singlet $\mathbf A+\mathbf B = \Delta + 4(ia|jb) - (ab|W|ij) - (ib|W|ja)$
> and $\mathbf A-\mathbf B = \Delta - (ab|W|ij) + (ib|W|ja)$, with no
> singlet–triplet coupling. Checked numerically against spin-orbital matrices
> with $W$ from a matrix inverse.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_bse_rpax_sympy.py`, claims S1, R1 |
> | Negative control | $\mathbf A+\mathbf B$ and $\mathbf A-\mathbf B$ must differ; the runner changes the Hartree prefactor 4 to 2 and swaps the two terms of $\mathbf A-\mathbf B$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening` |
> | Code test | `rpax_alpha_h2o_aug_cc_pvdz_rhf_vs_numpy_and_cphf` in `crates/ferric-gw/tests/validation_rpax_alpha.rs` |
>
> <!-- verified: id=rpax.matrices; script=scripts/gw_bse_rpax_sympy.py; claim=R1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening; rust_test=ferric-gw/tests/validation_rpax_alpha.rs::rpax_alpha_h2o_aug_cc_pvdz_rhf_vs_numpy_and_cphf; registry=tdhf-static-polarizability -->

## 19.9.2 The response

The code solves one linear system per Cartesian axis,
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)]\,t = (\mathbf A-\mathbf B)\mu$,
and sets $\alpha_{xy} = 4\mu_x^{\mathsf T}t_y$:

<!-- excerpt: ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening -->
```rust
    let sysm = amb.dot(&apb);
```

The product of the two matrices is not symmetric; its operator
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)]^{-1}(\mathbf A-\mathbf B)$ is the symmetric
$(\mathbf A+\mathbf B)^{-1}$ at $u = 0$, so the static $\alpha$ depends on
$\mathbf A+\mathbf B$ alone. The dynamic library drivers add $u^2$ to the
product. The reversed order, $\mu^{\mathsf T}(\mathbf A-\mathbf B)[\cdots]^{-1}\mu$, is a
different number.

> **Verified equation 19.17** · the response solve against the full spin-orbital response equation
>
> Solving $\begin{pmatrix}\mathbf A-iu&\mathbf B\\\mathbf B&\mathbf A+iu\end{pmatrix}\binom XY = \binom{\mu}{\mu}$
> in the full spin-orbital space, with $\alpha = \mu\cdot(X+Y)$, gives the
> code's $4\mu^{\mathsf T}[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2]^{-1}(\mathbf A-\mathbf B)\mu$
> at $u = 0$ and $u = 0.7$; at $u=0$ it is $4\mu^{\mathsf T}(\mathbf A+\mathbf B)^{-1}\mu$.
> The scalar reduction $X+Y = 2m(a-b)/((a-b)(a+b)+u^2)$ is symbolic.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_bse_rpax_sympy.py`, claims S3, R2; and claim 7 of `scripts/gw_self_energy_sympy.py` for the sum over states |
> | Negative control | the reversed order must differ; $(\mathbf A+\mathbf B)$ in the numerator of the scalar form must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening` |
>
> <!-- verified: id=rpax.response; script=scripts/gw_bse_rpax_sympy.py; claim=R2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening; rust_test=ferric-gw/tests/validation_rpax_alpha.rs::rpax_alpha_h2o_aug_cc_pvdz_rhf_vs_numpy_and_cphf; registry=tdhf-static-polarizability -->

> **Verified equation 19.18** · the bare-exchange limit is the coupled-perturbed HF polarizability
>
> With $W\to v$ at an HF reference, the static $\alpha$ equals the central
> finite-field derivative of the dipole of a self-consistent RHF in the same
> orthonormal MO basis ($h = h_0 - F\,r$, $F = h + J - \tfrac12K$,
> $D = 2C_oC_o^{\mathsf T}$, SCF to $10^{-15}$). This is the one claim of the
> section that tests the physics of the kernel, not only its algebra.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_bse_rpax_sympy.py`, claim R3 |
> | Negative control | a Hartree prefactor 2 with no exchange must miss the finite-field number |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening` |
> | Code test | `rpax_alpha_nh3_aug_cc_pvdz_rhf_vs_numpy_and_cphf` in `crates/ferric-gw/tests/validation_rpax_alpha.rs` |
>
> <!-- verified: id=rpax.cphf-limit; script=scripts/gw_bse_rpax_sympy.py; claim=R3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening; rust_test=ferric-gw/tests/validation_rpax_alpha.rs::rpax_alpha_nh3_aug_cc_pvdz_rhf_vs_numpy_and_cphf; registry=tdhf-static-polarizability -->

## 19.9.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | KS reference; `scissor` added to every virtual energy | | the caller; `eps[p] += scissor` |
| 2 | PDEP-RPA on the same reference (skipped for `RpaxScreening::Bare`) | chapter 18 | `run_pdep_rpa` |
| 3 | $B$, redressed modes, $B^\xi$ | section 19.2 | `build_full_b`, `project_b_into_pdep` |
| 4 | dense $\mathbf A+\mathbf B$ and $\mathbf A-\mathbf B$, $n = n_on_v$ | $n^2$ entries | `rpax_static_polarizability_with_screening` |
| 5 | $\mu$ from the AO dipole, one $n\times n$ solve per axis | $n^3$ | the same function |
| 6 | refuse a non-positive $\alpha$ diagonal | $O(1)$ | `check_alpha_diagonal_positive` |

Three $n\times n$ buffers are live at the peak (`apb`, `amb` and the product);
the matrices are a hard pool charge and a pre-flight refuses a system that
does not fit.

## 19.9.4 Options and limits

| Option | Effect |
|---|---|
| `[gw] scissor` | adds a constant to every virtual KS energy; 0.3 to 0.4 Ha for small molecules |
| `[rpa] xc` | required: the CLI refuses an HF reference |
| `trunc_thresh` | the CLI default is $10^{-4}$; the validation runs at 0 |

**Why the diagonal guard exists.** At `scissor = 0` the kernel can give a
negative $\alpha$ diagonal even for small closed-shell molecules at
equilibrium geometry (water/cc-pVDZ $\alpha_{xx} = -2.68$ a.u.), an excitonic
instability of pairing the screened exchange kernel with the uncorrected KS
gap. The isotropic average can look right while the tensor is unphysical, so
the library refuses rather than warn: an independent kernel reproduces the
negative diagonal, so it is not an assembly error. A nonzero scissor, or a
diagonal from G0W0 energies, removes it.

**Known limits (2026-11-15).** Static $\alpha$ only. The dynamic drivers
`run_bse_c6` and `run_bse_c6_ks` are deliberately library-only: $C_6$ from this
kernel stays about 63% low regardless of the HOMO–LUMO gap, worse than the
PDEP-RPA route of section 21.1. Closed shell only. The grade is Smoke.

## 19.9.5 Numerical anchors

From `site/src/reference/validation.md` (the RPAx row of the anchors table and
the `tdhf-static-polarizability` row of the capability table):

| Quantity | Systems | Result | Test |
|---|---|---|---|
| $\alpha$ against a numpy assembly of the same kernel (full 3×3 tensor) | H2O, NH3 / aug-cc-pVDZ at RHF, scissor 0; H2O / cc-pVDZ at PBE, scissor 0.36 Ha | $\le 5.3\times10^{-9}$ relative | `crates/ferric-gw/tests/validation_rpax_alpha.rs` |
| bare-exchange limit against the same numpy build | same | $\le 5.4\times10^{-9}$ relative | same |
| lowest eigenvalues of $\mathbf A\pm\mathbf B$ | same | $\le 7.5\times10^{-10}$ Ha | same |
| physical accuracy: water / cc-pVDZ at PBE, scissor 0.36 Ha | $\alpha_{\rm iso}$ 5.20 a.u. against DOSD 9.64 | 46% low | `examples/water-tdhf-static-alpha.toml` |

## 19.9.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, "TDHF/RPAx polarizability" | the matrices and the response |
| `scripts/gw_bse_rpax_sympy.py`, claims S1, S3, R1 to R3 | the proofs |
| `scripts/gw_self_energy_sympy.py`, claim 7 | the sum-over-states form |
