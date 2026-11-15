# 19.2 The screened interaction on the imaginary axis; PDEP as $W$

<!-- section: 19.2; status: ready; registry: gw;
     theory: crates/ferric-gw/theory.md#"2. Screened interaction on the imaginary axis" -->

The screened interaction is the bare Coulomb interaction divided by the
dielectric function. In the dressed auxiliary basis the bare interaction is
the identity, so $W = \tilde{\boldsymbol\varepsilon}^{-1}$ and only its
dynamic part, $\tilde W_d = \tilde{\boldsymbol\varepsilon}^{-1} - \mathbf 1$,
enters the correlation self-energy; the identity part is the exchange of
section 19.1. Section 18.3 diagonalizes $\tilde{\boldsymbol\varepsilon}(0)$ to
find the projective dielectric eigenpotentials (PDEP). The GW code uses that
eigenbasis as a fixed basis in which to build $\tilde W_d$ at every
frequency.

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

## 19.2.1 The dielectric matrix and its single-pole structure

<!-- include: crates/ferric-gw/theory.md#"2. Screened interaction on the imaginary axis" -->

> **Verified equation 19.2** · one particle–hole pair gives a single-pole $W$
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

<!-- include: crates/ferric-gw/theory.md#"3. Single-pole structure" -->

This one-pair model is the analytic reference for sections 19.3 and 19.4: its
$W(iu)$ is a single Lorentzian, so the frequency integral of the self-energy
has a closed form.

## 19.2.2 PDEP as the basis for $W$

<!-- include: crates/ferric-gw/theory.md#"10. PDEP as W" -->

The PDEP step of chapter 18 returns its eigenpotentials in raw auxiliary
coefficients, $c = M_{\rm pdep}^{\mathsf T}U$. The GW code needs the vectors in
the dressed basis of its own $B$, so `redress_eigenpotentials` solves
$M^{\mathsf T}\hat U = c$: a triangular solve when $M = L^{-1}$ (Coulomb and
erfc), and an eigendecomposition pseudo-inverse when $M$ is the symmetric
$V^{-1/2}$ of the erf operator, dropping its null modes; any other shape is
an error. `redress_with_check` returns the largest deviation of
$\|\hat U_\xi\|^2$ from 1, and `run_gw` prints it.

The projection $B^\xi_{pq} = \sum_P\hat U^P_\xi B^P_{pq}$ is one GEMM over the
whole active square. It is the largest single matrix product in the GW code,
it is rebuilt in every evGW iteration, and it is a hard memory charge:
$N_{\rm pdep}\,n_{\rm act}^2$ doubles beside $B$ itself.

> **Verified equation 19.3** · a complete PDEP basis reproduces the bare integrals, and the screened kernel is its weighted sum
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

## 19.2.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | PDEP-RPA run: static modes and, at each node $u_k$, the full $\tilde W_d(iu_k)$ | chapter 18, plus $N_{\rm pdep}^3$ per node for the inverse | `run_pdep_rpa`, `eval_inv_dielectric_matrices` |
| 2 | redress $U$ into the dressed basis of $B$ | $N_{\rm aux}^2N_{\rm pdep}$ | `redress_with_check` |
| 3 | $B^\xi_{pq}$ | $N_{\rm pdep}N_{\rm aux}n_{\rm act}^2$ | `project_b_into_pdep` |
| 4 | per state $p$ and node $k$: $W_{pq}(iu_k) = (B^{\bullet}_{pq})^{\mathsf T}\tilde W_d(iu_k)B^{\bullet}_{pq}$ for all $q$ | one GEMM per node, $N_{\rm pdep}^2n_{\rm act}$ | `screened_elements` |

The per-frequency scalar eigenvalues $\lambda_\xi(iu)$ that chapter 18 uses for
the energy live in an $u$-dependent rotated eigenbasis, so they are valid
only for basis-invariant traces. The self-energy contracts the matrix
$\tilde W_d$ with specific $B^\xi_{pq}$ vectors and so needs the full matrix in
the fixed static basis. The diagonal weights $1/\lambda_\xi(iu_k) - 1$ are
therefore never formed for $\Sigma^c$.

## 19.2.4 Options that change the equation

| Option | Effect on $W$ |
|---|---|
| `trunc_thresh` | drops modes with $\lambda_\xi(0)$ within the threshold of 1; $\Sigma^x$ is unchanged because it uses the full $B$ |
| `n_quad`, `quadrature`, `u0` | the frequency grid of the $\tilde W_d$ stack; the self-energy integral is evaluated on it |
| `frozen_core` | must agree between $W$ and $\Sigma$; the CLI passes `[gw] frozen_core` to both |

**Known limits (2026-11-15).** Every external comparison in the validation
tables runs at `trunc_thresh = 0`, where the PDEP basis is the full RI
dielectric eigenbasis and $W$ is the ordinary RI $W$. Truncated GW is not
validated. At the defaults (`n_quad = 40`, `trunc_thresh = 1e-4`) the HOMO
and LUMO stay within 10 meV of the matched settings.

## 19.2.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-gw/theory.md`, sections "Screened interaction on the imaginary axis", "Single-pole structure", "PDEP as W" | boxes 19.2 and 19.3 and the text |
| `scripts/gw_self_energy_sympy.py`, claims 1 and 8 | the proofs |
| Wilson, Gygi and Galli, *Phys. Rev. B* **78**, 113303 (2008) | PDEP |
