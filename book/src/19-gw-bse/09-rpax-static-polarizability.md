# The statically screened TDHF polarizability {#rpax-static-polarizability}

<!-- status: ready; registry: tdhf-static-polarizability -->

A static electric field pushes the electrons one way and the nuclei the
other. The induced dipole per unit field is the polarizability, and in
linear response it comes from the same particle–hole matrices as the
excitation energies: a field mixes the excited states into the ground state,
each weighted by its transition dipole and its energy. This section takes
the static screened interaction of section @sec:bse-tda and uses it in the full
response problem instead of the Tamm–Dancoff one. Both the excitation block
$\mathbf A$ and the de-excitation block $\mathbf B$ then carry a screened
exchange-type term. With $W$ replaced by $v$ at a Hartree–Fock reference the
kernel is TDHF and the polarizability is the coupled-perturbed HF one; the
shipped kernel keeps the screened $W$, so it is a statically screened RPAx.

This section is Smoke grade. The implementation is
checked against independent assemblies of the same equations, but the
physics is not established: the polarizability is 46% low against the dipole
oscillator strength distribution (DOSD) reference on water.

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

## The response matrices {#rpax-static-polarizability-response-matrices}

**In spin orbitals.** The construction is the one of section @sec:bse-tda, now with
the de-excitation block. The Hartree term $(ia|jb)$ couples every spin pair,
and the screened terms couple equal spins only:

$$ A_{i_\sigma a_\sigma,j_\tau b_\tau} = \Delta_i^a\delta + (ia|jb) - \delta_{\sigma\tau}(ab|W|ij), \qquad B_{i_\sigma a_\sigma,j_\tau b_\tau} = (ia|jb) - \delta_{\sigma\tau}(ib|W|ja) . $$

**Spin-adapt.** The spin rotation of section @sec:bse-tda adds the $\alpha\alpha$ and
$\alpha\beta$ blocks for the singlet. In $\mathbf A$ and in $\mathbf B$ the
Hartree term appears in both blocks, so the singlet carries $2(ia|jb)$ in
each, and $4(ia|jb)$ in their sum. The singlet block is

$$ (\mathbf A+\mathbf B)_{ia,jb} = \Delta_i^a\delta + 4(ia|jb) - (ab|W|ij) - (ib|W|ja), \qquad (\mathbf A-\mathbf B)_{ia,jb} = \Delta_i^a\delta + (ib|W|ja) - (ab|W|ij) . $$

The Hartree term cancels from $\mathbf A-\mathbf B$. The triplets are block
diagonal, with $4\to0$ in the Hartree term of $\mathbf A+\mathbf B$. Claim S1
is the symbolic spin algebra, and claim R1 the numerical check against the
spin-orbital matrices built from $W = B^{\mathsf T}\tilde\varepsilon^{-1}B$.
The dynamic library drivers `ferric-gw/src/bse.rs::run_bse_c6` and
`run_bse_c6_ks` use the same matrices.

The assembly in `ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening`
follows the singlet formulas row by row:

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

The diagonal $\Delta_i^a$ is added to both rows after the loop. `scissor`
adds a constant to every virtual energy before the assembly
(`eps[p] += scissor`).

> **Verified equation {#eq:rpax-matrices}** · the response matrices are the spin adaptation of the spin-orbital RPAx matrices
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

## The response {#rpax-static-polarizability-response}

**The equations.** Apply a perturbation $-\mathbf F\cdot\hat{\boldsymbol\mu}$ at
imaginary frequency $\omega = iu$. The first-order amplitudes solve

$$ \begin{pmatrix}\mathbf A-\omega&\mathbf B\\\mathbf B&\mathbf A+\omega\end{pmatrix}\binom{X}{Y} = \binom{\boldsymbol\mu}{\boldsymbol\mu}F , $$

and the induced dipole is $\boldsymbol\mu\cdot(X+Y)$.

**Reduce to one block.** Subtract the second block row from the first:

$$ (\mathbf A-\mathbf B)(X-Y) - \omega(X+Y) = 0 \quad\Rightarrow\quad X - Y = \omega(\mathbf A-\mathbf B)^{-1}(X+Y) . $$

Add the two rows instead:

$$ (\mathbf A+\mathbf B)(X+Y) - \omega(X-Y) = 2\boldsymbol\mu F . $$

Substitute $X - Y$ from the first result:

$$ \big[(\mathbf A+\mathbf B) - \omega^2(\mathbf A-\mathbf B)^{-1}\big](X+Y) = 2\boldsymbol\mu F . $$

**Go to the singlet space.** At $\omega = iu$, $-\omega^2 = u^2$. In the
rotated spin basis the spin-orbital dipole is $\boldsymbol\mu_{\rm so} = \sqrt2\,\boldsymbol\mu$,
so $X+Y$ carries one $\sqrt2$ from the right-hand side and the dot product
with $\boldsymbol\mu_{\rm so}$ another. Together with the 2 of the
right-hand side that gives a factor 4. Factor $(\mathbf A-\mathbf B)^{-1}$
out of the bracket on the left, $(\mathbf A+\mathbf B) + u^2(\mathbf A-\mathbf B)^{-1} = (\mathbf A-\mathbf B)^{-1}\big[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2\big]$,
and invert:

$$ \alpha_{xy}(iu) = 4\,\mu_x^{\mathsf T}\big[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2\big]^{-1}(\mathbf A-\mathbf B)\,\mu_y = 4\,\mu_x^{\mathsf T}\big[(\mathbf A+\mathbf B) + u^2(\mathbf A-\mathbf B)^{-1}\big]^{-1}\mu_y . $$

**Check the static limit.** At $u = 0$ the right-hand form is
$\alpha = 4\mu^{\mathsf T}(\mathbf A+\mathbf B)^{-1}\mu$: the static
polarizability depends on $\mathbf A+\mathbf B$ alone. The operator
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)+u^2]^{-1}(\mathbf A-\mathbf B)$ is
symmetric and equals the sum over states
$2\sum_n|\langle0|\mathbf r|n\rangle|^2\,\Omega_n/(\Omega_n^2+u^2)$, with
$\Omega_n^2$ the eigenvalues of $(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)$
(claim 7 of `scripts/gw_self_energy_sympy.py`): the sum over excited
states of the opening paragraph.

**The code** solves one linear system per Cartesian axis,
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)]\,t = (\mathbf A-\mathbf B)\mu$,
and sets $\alpha_{xy} = 4\mu_x^{\mathsf T}t_y$:

<!-- excerpt: ferric-gw/src/bse.rs::rpax_static_polarizability_with_screening -->
```rust
    let sysm = amb.dot(&apb);
    // α_ij(0) = 4 μ_iᵀ sysm⁻¹ (A−B) μ_j, full 3×3 tensor (not just the
    // isotropic average run_bse_c6_ks reports).
    let mut t: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::zeros(n));
    for d in 0..3 {
        let rhs = amb.dot(&mu[d]);
        t[d] = sysm
            .solve(&rhs)
            .map_err(|e| FerricError::Lapack(format!("RPAx static α(0) solve: {e}")))?;
    }
```

| Code | Equation |
|---|---|
| `sysm` | $(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)$ |
| `rhs` | $(\mathbf A-\mathbf B)\mu_d$ |
| `t[d]` | $t_d$; then `tensor[i][j] = 4.0 * mu[i].dot(&t[j])` |

The product of the two matrices is not symmetric, but
$[(\mathbf A-\mathbf B)(\mathbf A+\mathbf B)]^{-1}(\mathbf A-\mathbf B)$ is,
and at $u = 0$ it equals $(\mathbf A+\mathbf B)^{-1}$. The dynamic
library drivers add $u^2$ to the product. The order matters: the reversed
form $\mu^{\mathsf T}(\mathbf A-\mathbf B)[\cdots]^{-1}\mu$ is a different
number. Claim S3 is the scalar algebra of the reduction, and claim R2 solves
the full $2N\times2N$ spin-orbital system at $u = 0$ and $u = 0.7$ and gets
the code's number.

> **Verified equation {#eq:rpax-response}** · the response solve against the full spin-orbital response equation
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

**Check the bare-exchange limit.** With $W \to v$ at a Hartree–Fock reference
(`RpaxScreening::Bare`) the kernel is TDHF, and $\alpha(0)$ must be the
coupled-perturbed HF polarizability, the derivative of the SCF dipole with
respect to a field. Claim R3 tests this against an independent route: a
self-consistent RHF in the same orthonormal MO basis, with
$h = h_0 - F\,r$ and $F[D] = h + J - \tfrac12K$, differentiated by a central
finite field. The two agree to the finite-difference error (a relative
$10^{-6}$ bar in the script).

> **Verified equation {#eq:rpax-cphf-limit}** · the bare-exchange limit is the coupled-perturbed HF polarizability
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

A negative $\alpha$ diagonal is refused
(`ferric-gw/src/bse.rs::check_alpha_diagonal_positive`, section @sec:rpax-static-polarizability-options-limits).

## The algorithm {#rpax-static-polarizability-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | KS reference; `scissor` added to every virtual energy | | the caller; `eps[p] += scissor` |
| 2 | PDEP-RPA on the same reference (skipped for `RpaxScreening::Bare`) | chapter @sec:rpa | `run_pdep_rpa` |
| 3 | $B$, redressed modes, $B^\xi$ | section @sec:screened-interaction | `build_full_b`, `project_b_into_pdep` |
| 4 | dense $\mathbf A+\mathbf B$ and $\mathbf A-\mathbf B$, $n = n_on_v$ | $n^2$ entries | `rpax_static_polarizability_with_screening` |
| 5 | $\mu$ from the AO dipole, one $n\times n$ solve per axis | $n^3$ | the same function |
| 6 | refuse a non-positive $\alpha$ diagonal | $O(1)$ | `check_alpha_diagonal_positive` |

Three $n\times n$ buffers are live at the peak (`apb`, `amb` and the product).
The matrices are a hard pool charge, and a pre-flight refuses a system that
does not fit.

## Options and limits {#rpax-static-polarizability-options-limits}

| Option | Effect |
|---|---|
| `[gw] scissor` | adds a constant to every virtual KS energy; 0.3 to 0.4 Ha for small molecules |
| `[rpa] xc` | required: the CLI refuses an HF reference |
| `trunc_thresh` | the CLI default is $10^{-4}$; the validation runs at 0 |

**Why the diagonal guard exists.** At `scissor = 0` the kernel can give a
negative $\alpha$ diagonal even for small closed-shell molecules at
equilibrium geometry (water/cc-pVDZ $\alpha_{xx} = -2.68$ a.u.). This is an
excitonic instability: the screened exchange kernel paired with the
uncorrected KS gap. In the static limit $\alpha = 4\mu^{\mathsf T}(\mathbf A+\mathbf B)^{-1}\mu$,
so a negative diagonal element means $\mathbf A+\mathbf B$ is not positive
definite. The isotropic average can look right while the tensor is
unphysical, so the library refuses instead of warning. An independent kernel
reproduces the negative diagonal, so it is not an assembly error. A nonzero
scissor, or a diagonal from G0W0 energies, removes it.

**Known limits (2026-11-15).** Static $\alpha$ only. The dynamic drivers
`run_bse_c6` and `run_bse_c6_ks` are deliberately library-only: $C_6$ from this
kernel stays about 63% low regardless of the HOMO–LUMO gap, worse than the
PDEP-RPA route of section @sec:polarizabilities-from-dielectric-modes. Closed shell only. The grade is Smoke.

## Numerical anchors {#rpax-static-polarizability-numerical-anchors}

From `site/src/reference/validation.md` (the RPAx row of the anchors table and
the `tdhf-static-polarizability` row of the capability table):

| Quantity | Systems | Result | Test |
|---|---|---|---|
| $\alpha$ against a numpy assembly of the same kernel (full 3×3 tensor) | H2O, NH3 / aug-cc-pVDZ at RHF, scissor 0; H2O / cc-pVDZ at PBE, scissor 0.36 Ha | $\le 5.3\times10^{-9}$ relative | `crates/ferric-gw/tests/validation_rpax_alpha.rs` |
| bare-exchange limit against the same numpy build | same | $\le 5.4\times10^{-9}$ relative | same |
| lowest eigenvalues of $\mathbf A\pm\mathbf B$ | same | $\le 7.5\times10^{-10}$ Ha | same |
| physical accuracy: water / cc-pVDZ at PBE, scissor 0.36 Ha | $\alpha_{\rm iso}$ 5.20 a.u. against DOSD 9.64 | 46% low | `examples/water-tdhf-static-alpha.toml` |

Further tests are in `crates/ferric-gw/tests/rpax_*.rs`.

The physical-accuracy row and the scissor-0 refusal, from the `589b50832`
build. The script cuts the error message at 200 characters.

```python
# Water / cc-pVDZ at PBE, scissor 0.36 Ha: statically screened RPAx alpha tensor and alpha_iso (DOSD 9.64 a.u.).
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

r = ferric.run_tdhf_static_polarizability(mol, bs, aux, xc="PBE", scissor=0.36, trunc_thresh=0.0)
print("alpha tensor (a.u.):")
print(r.tensor)
print(f"alpha_iso = {r.iso:.3f} a.u.; DOSD 9.64; ratio {r.iso/9.64:.3f}")
try:
    ferric.run_tdhf_static_polarizability(mol, bs, aux, xc="PBE", scissor=0.0, trunc_thresh=0.0)
    print("scissor 0: no error raised")
except Exception as e:
    print("scissor 0 raised:", type(e).__name__, str(e)[:200])
```

```text
alpha tensor (a.u.):
[[3.17316347e+00 4.29357137e-16 9.80918785e-16]
 [4.29357137e-16 7.12027031e+00 1.76776903e-14]
 [9.80918785e-16 1.70340909e-14 5.31800015e+00]]
alpha_iso = 5.204 a.u.; DOSD 9.64; ratio 0.540
scissor 0 raised: RuntimeError General error: run_rpax_static_polarizability: unphysical static polarizability -- every Cartesian diagonal element must be strictly positive, but alpha_xx = -2.682779e0 (negative). A non-positive alp
```

## Sources {#rpax-static-polarizability-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_bse_rpax_sympy.py`, claims S1, S3, R1 to R3 | the proofs |
| `scripts/gw_self_energy_sympy.py`, claim 7 | the sum-over-states form |
