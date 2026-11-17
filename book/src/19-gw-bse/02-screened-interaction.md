# The screened interaction on the imaginary axis; PDEP as $W$ {#screened-interaction}

<!-- status: ready; registry: gw -->

Put a test charge into a molecule. The electrons polarize around it, and the
induced charge partly cancels the field of the test charge far away. The
interaction that a second charge feels is the bare Coulomb interaction
weakened by that response. This is the screened interaction $W$: the bare
interaction divided by the dielectric function. The response takes time to
build, so $W$ depends on frequency.

Chapter @sec:rpa builds the dielectric matrix to compute the RPA energy. GW needs
its inverse. Section @sec:pdep diagonalizes the static dielectric matrix to find
the projective dielectric eigenpotentials (PDEP). The GW code uses that
eigenbasis as a fixed basis in which to build $W$ at every frequency.

| Symbol | Meaning | Code name |
|---|---|---|
| $\Pi_{PQ}(iu)$, $\tilde{\boldsymbol\varepsilon}(iu)$ | independent-particle response and dielectric matrix | `ferric-rpa/src/sternheimer.rs::dielectric_matrix_from_projection` |
| $\lambda_\xi(0)$, $U_\xi$ | static eigenvalues and (dressed) eigenvectors | `PdepRpaResult::{eigenvalues_static, eigenpotentials}` |
| $w_\xi = 1/\lambda_\xi(0) - 1$ | static weights | `static_weights` |
| $\tilde W_d(iu_k)$ | $N_{\rm pdep}\times N_{\rm pdep}$ matrix per node | `PdepRpaResult::inv_dielectric_freq` |
| $B^\xi_{pq}$ | $B$ projected on mode $\xi$ | `MProj`, from `project_b_into_pdep` |
| $W_{pq}(iu_k)$ | screened matrix element | `screened_elements` |

| Entry point | Location |
|---|---|
| Rust | `ferric-gw/src/w_pdep.rs::{redress_eigenpotentials,redress_with_check,static_weights}`, `ferric-gw/src/cohsex.rs::project_b_into_pdep` |
| Frequency stack | `ferric-rpa/src/energy.rs::eval_inv_dielectric_matrices` |
| Options | `[rpa] n_quad`, `trunc_thresh`, `quadrature`, `u0` |

## The dielectric matrix and its single-pole structure {#screened-interaction-dielectric-matrix-single-pole}

**The bare interaction is the identity.** In the dressed auxiliary basis of
section @sec:ri-factors-and-exchange the Coulomb integral is $(pq|rs) = \sum_P B^P_{pq}\,1\,B^P_{rs}$.
The matrix between the two $B$ factors is the bare interaction, and it is the
unit matrix. Every screened quantity below is a matrix in the same place.

**The response.** The closed-shell independent-particle response and the
dielectric matrix are (equation @eq:pdep-1)

$$ \Pi_{PQ}(iu) = 4 \sum_{ia} B^P_{ia}B^Q_{ia}\,\frac{\Delta_i^a}{u^2 + (\Delta_i^a)^2}, \qquad \tilde{\boldsymbol\varepsilon}(iu) = \mathbf 1 + \Pi(iu), \qquad \Delta_i^a = \epsilon_a - \epsilon_i . $$

The prefactor 4 is 2 for spin times 2 for the resonant and antiresonant
terms (`ferric-rpa/src/sternheimer.rs::build_scale_factors`,
`dielectric_matrix_from_projection`). $\Pi(iu)$ is positive semidefinite on
the imaginary axis.

**The screened interaction.** $W$ is the bare interaction divided by the
dielectric matrix. The bare interaction is $\mathbf 1$, so
$\tilde W = \tilde{\boldsymbol\varepsilon}^{-1}$. Multiply
$\tilde{\boldsymbol\varepsilon}\tilde W = \mathbf 1$ out and move $\Pi\tilde W$
to the right:

$$ \tilde W = \mathbf 1 - \Pi\,\tilde W . $$

This is the Dyson equation $W = v + v\chi_0W$ in the dressed basis, with
$v = \mathbf 1$ and $\chi_0 = -\Pi$. The static bare part $\mathbf 1$ of
$\tilde W$ gives the exchange self-energy $\Sigma^x$ of section @sec:ri-factors-and-exchange. The code
carries only the dynamic part:

$$ \tilde W_d(iu) = \tilde{\boldsymbol\varepsilon}^{-1}(iu) - \mathbf 1 = -\Pi(iu)\,\big[\mathbf 1 + \Pi(iu)\big]^{-1} . $$

**Check two limits.** With no response, $\Pi = 0$, $\tilde W_d = 0$: the
correlation self-energy vanishes and GW reduces to the exchange of section
@sec:ri-factors-and-exchange ($W \to v$). For a weak response, $\tilde W_d \approx -\Pi$ to first
order: screening lowers the interaction. In general
$\tilde{\boldsymbol\varepsilon} \ge \mathbf 1$, so
$\tilde{\boldsymbol\varepsilon}^{-1} \le \mathbf 1$ and $\tilde W_d$ is
negative definite on the imaginary axis.

**One particle–hole pair.** Take one $(ia)$ pair with $B^P_{ia} = b$ and
$\Delta_i^a = e$. Then $\Pi = 4b^2e/(u^2+e^2)$ is a scalar, and

$$ \tilde W_d(iu) = -\frac{\Pi}{1+\Pi} = -\frac{4b^2 e}{u^2 + e^2 + 4b^2 e} = -\frac{2\Omega s}{u^2 + \Omega^2}, \qquad \Omega^2 = e(e + 4b^2), \quad s = \frac{2b^2e}{\Omega} . $$

Compare with the singlet direct-RPA matrices of one pair, $\mathbf A = e + 2K$
and $\mathbf B = 2K$ with $K = b^2$. Their product is
$(\mathbf A-\mathbf B)(\mathbf A+\mathbf B) = e(e+4b^2) = \Omega^2$: the pole
of $W$ sits at the direct-RPA excitation energy (section @sec:drpa-three-ways-plasmon-formula). In general
$W$ is a sum of such poles, one per RPA excitation $s$ with eigenvector
$\rho_s$:

$$ \tilde W_d(iu) = -\sum_s \frac{2\Omega_s\,\rho_s\rho_s^{\mathsf T}}{u^2+\Omega_s^2} . $$

> **Verified equation {#eq:gw-single-pole-w}** · one particle–hole pair gives a single-pole $W$
>
> With the closed-shell prefactor 4, $\tilde\varepsilon^{-1} - 1$ of one pair
> $(ia)$ with $B^P_{ia} = b$ and $\Delta_i^a = e$ is
> $-2\Omega s/(u^2+\Omega^2)$, with $\Omega^2 = e(e+4b^2) = (\mathbf A-\mathbf B)(\mathbf A+\mathbf B)$
> of singlet direct RPA. The unrestricted per-spin prefactor 2 reproduces it
> when $\alpha = \beta$; one spin alone does not.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 1 |
> | Negative control | prefactor 2 on one spin only is not the closed-shell pole; the runner turns the first spin's prefactor into 1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/sternheimer.rs::build_scale_factors`, `ferric-rpa/src/energy.rs::eval_inv_dielectric_matrices` |
> | Code test | `g0w0_hf_h2o_vs_pyscf_gw_ac` in `crates/ferric-gw/tests/validation_gw.rs` (end to end through $W$) |
>
> <!-- verified: id=gw.single-pole-w; script=scripts/gw_self_energy_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/sternheimer.rs::build_scale_factors; rust_test=ferric-gw/tests/validation_gw.rs::g0w0_hf_h2o_vs_pyscf_gw_ac; registry=gw -->

The one-pair model is the analytic reference for sections @sec:correlation-self-energy to @sec:continuation-and-qp-equation: its
$W(iu)$ is a single Lorentzian, so the frequency integral of the self-energy
has a closed form.

## PDEP as the basis for $W$ {#screened-interaction-pdep-basis}

**The fixed basis.** Diagonalize the static matrix,
$\tilde{\boldsymbol\varepsilon}(0)\,U_\xi = \lambda_\xi(0)\,U_\xi$. The GW code
works in the fixed eigenbasis $\{U_\xi\}$. At every quadrature node it builds
the *full* $N_{\rm pdep}\times N_{\rm pdep}$ matrix

$$ \tilde W_d(iu_k) = \big[U^{\mathsf T}\tilde{\boldsymbol\varepsilon}(iu_k)\,U\big]^{-1} - \mathbf 1 $$

(`ferric-rpa/src/energy.rs::eval_inv_dielectric_matrices`, stored in
`PdepRpaResult.inv_dielectric_freq`). At $u = 0$ the matrix is diagonal, with
the static weights

$$ w_\xi = \frac{1}{\lambda_\xi(0)} - 1 $$

(`ferric-gw/src/w_pdep.rs::static_weights`). A mode with $\lambda_\xi(0)$
close to 1 is a potential the molecule barely screens, and its weight is
close to 0.

**Redressing.** The PDEP step of chapter @sec:rpa returns its eigenpotentials in
raw auxiliary coefficients, $c = M_{\rm pdep}^{\mathsf T}U$. The GW code needs
the vectors in the dressed basis of its own $B$, so
`ferric-gw/src/w_pdep.rs::redress_eigenpotentials` solves
$M^{\mathsf T}\hat U = c$. When $M = L^{-1}$ (Coulomb and erfc) the solve is
triangular. When $M$ is the symmetric $V^{-1/2}$ of the erf operator, it is an
eigendecomposition pseudo-inverse that drops the null modes. Any other shape
of $M$ is an error. `redress_with_check` returns the largest deviation of
$\|\hat U_\xi\|^2$ from 1, and `run_gw` prints it. The tests are in
`crates/ferric-gw/tests/redress_recovers_dressed_eigenvectors.rs`.

**Projection.** Every self-energy and kernel contraction uses $B$ rotated
into the mode basis:

$$ B^\xi_{pq} = \sum_P \hat U^P_\xi\, B^P_{pq} . $$

`ferric-gw/src/cohsex.rs::project_b_into_pdep` does this as one GEMM over the
whole active square. It is the largest single matrix product in the GW code,
it is rebuilt in every evGW iteration, and it is a hard memory charge
(`guard_m_proj`): $N_{\rm pdep}\,n_{\rm act}^2$ doubles beside $B$ itself,
which is $N_{\rm aux}\,n_{\rm act}^2$ at full rank.

**Check completeness.** For a complete orthonormal $\hat U$, insert
$\sum_\xi \hat U_\xi\hat U_\xi^{\mathsf T} = \mathbf 1$ between the two $B$
factors of $(pq|rs)$. The result is
$\sum_\xi B^\xi_{pq}B^\xi_{rs} = (pq|rs)$. The static screened integral then
splits into bare and dynamic parts,

$$ (pq|rs) + \sum_\xi w_\xi B^\xi_{pq}B^\xi_{rs} = \sum_\xi \lambda_\xi(0)^{-1} B^\xi_{pq}B^\xi_{rs} , $$

because $1 + w_\xi = \lambda_\xi(0)^{-1}$. A truncated $\hat U$ does not
satisfy the first identity. The code always uses the left-hand form, with the
bare part from the full $B$, so it stays correct when modes are dropped.

> **Verified equation {#eq:gw-pdep-complete}** · a complete PDEP basis reproduces the bare integrals, and the screened kernel is its weighted sum
>
> For a complete orthonormal $\hat U$,
> $\sum_\xi B^\xi_{pq}B^\xi_{rs} = (pq|rs)$, so
> $(pq|rs) + \sum_\xi w_\xi B^\xi_{pq}B^\xi_{rs} = \sum_\xi\lambda_\xi(0)^{-1}B^\xi_{pq}B^\xi_{rs}$.
> A truncated $\hat U$ does not satisfy the first identity.
>
> | | |
> |---|---|
> | Proof | `scripts/gw_self_energy_sympy.py`, claim 8 |
> | Negative control | a two-mode truncation of a three-mode exact rational basis must miss $(pq|rs)$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-gw/src/cohsex.rs::project_b_into_pdep`, `ferric-gw/src/w_pdep.rs::static_weights` |
> | Code test | `coulomb_redress_recovers_dressed_eigenvectors_and_matches_prefix_pipeline` in `crates/ferric-gw/tests/redress_recovers_dressed_eigenvectors.rs` |
>
> <!-- verified: id=gw.pdep-complete; script=scripts/gw_self_energy_sympy.py; claim=8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-gw/src/cohsex.rs::project_b_into_pdep; rust_test=ferric-gw/tests/redress_recovers_dressed_eigenvectors.rs::coulomb_redress_recovers_dressed_eigenvectors_and_matches_prefix_pipeline; registry=gw -->

**The screened matrix element.** The self-energy of state $p$ needs, for each
state $q$ and node $k$, the dynamic interaction of the pair density
$\phi_p\phi_q$ with itself. In the mode basis that is a quadratic form:

$$ W_{pq}(iu_k) = \sum_{\xi\eta} B^\xi_{pq}\,[\tilde W_d(iu_k)]_{\xi\eta}\,B^\eta_{pq} . $$

For fixed $p$, collect $B^\xi_{pq}$ into a matrix $V$ of shape
$N_{\rm pdep}\times n_{\rm act}$. Then all $q$ at one node come from one
GEMM, $\tilde W_d(iu_k)\,V$, followed by a column-wise dot product with $V$.
This is the loop of `ferric-gw/src/sigma.rs::screened_elements`:

<!-- excerpt: ferric-gw/src/sigma.rs::screened_elements -->
```rust
    for (k, wt) in w_tilde.iter().enumerate() {
        let wv = with_blas_threads(opt_in_blas_threads(), || wt.dot(&v)); // (m_modes, n_act)
        let col = (&v_owned * &wv).sum_axis(ndarray::Axis(0)); // (n_act,)
        w_nk.column_mut(k).assign(&col);
    }
```

| Code | Equation |
|---|---|
| `wt` | $\tilde W_d(iu_k)$ |
| `v`, `v_owned` | $B^\xi_{pq}$ for fixed $p$, all $\xi$ and $q$ |
| `wv` | $\sum_\eta[\tilde W_d(iu_k)]_{\xi\eta}B^\eta_{pq}$ |
| `col` | $W_{pq}(iu_k)$ for all $q$ |

## The algorithm {#screened-interaction-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | PDEP-RPA run: static modes and, at each node $u_k$, the full $\tilde W_d(iu_k)$ | chapter @sec:rpa, plus $N_{\rm pdep}^3$ per node for the inverse | `run_pdep_rpa`, `eval_inv_dielectric_matrices` |
| 2 | redress $U$ into the dressed basis of $B$ | $N_{\rm aux}^2N_{\rm pdep}$ | `redress_with_check` |
| 3 | $B^\xi_{pq}$ | $N_{\rm pdep}N_{\rm aux}n_{\rm act}^2$ | `project_b_into_pdep` |
| 4 | per state $p$ and node $k$: $W_{pq}(iu_k) = (B^{\bullet}_{pq})^{\mathsf T}\tilde W_d(iu_k)B^{\bullet}_{pq}$ for all $q$ | one GEMM per node, $N_{\rm pdep}^2n_{\rm act}$ | `screened_elements` |

**Why the full matrix at every node.** Chapter @sec:rpa also computes the
per-frequency scalar eigenvalues $\lambda_\xi(iu)$ for the energy. Those live
in a $u$-dependent rotated eigenbasis, so they are valid only for
basis-invariant traces such as the RPA energy. The self-energy contracts
$\tilde W_d$ with specific $B^\xi_{pq}$ vectors in the fixed static basis, so it
needs the full matrix in that basis. The diagonal form
$\sum_\xi\big(1/\lambda_\xi(iu_k) - 1\big)$ is therefore never formed for
$\Sigma^c$.

## Options that change the equation {#screened-interaction-options-change-equation}

The only GW-specific use of PDEP is the basis $\{\hat U_\xi\}$ above.
`run_gw` and `run_u_gw` force the inverse-dielectric stack on
(`need_inv_dielectric_freq = true`, through `with_inv_dielectric`).

| Option | Effect on $W$ |
|---|---|
| `trunc_thresh` | drops modes with $\lambda_\xi(0)$ within the threshold of 1, which removes their rows of $B^\xi_{pq}$ and of $\tilde W_d$; $\Sigma^x$ is unchanged because it uses the full $B$ |
| `n_quad`, `quadrature`, `u0` | the frequency grid of the $\tilde W_d$ stack; the self-energy integral is evaluated on it |
| `frozen_core` | must agree between $W$ and $\Sigma$; the CLI passes `[gw] frozen_core` to both |

**Known limits (2026-11-15).** Every external comparison in the validation
tables runs at `trunc_thresh = 0`, where the PDEP basis is the full RI
dielectric eigenbasis and $W$ is the ordinary RI $W$. Truncated GW is not
validated. At the defaults (`n_quad = 40`, `trunc_thresh = 1e-4`) the HOMO
and LUMO stay within 10 meV of the matched settings.

## Sources {#screened-interaction-sources}

| Source | What it gives |
|---|---|
| `scripts/gw_self_energy_sympy.py`, claims 1 and 8 | the proofs |
| Wilson, Gygi and Galli, *Phys. Rev. B* **78**, 113303 (2008) | PDEP |
