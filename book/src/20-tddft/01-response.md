# 20.1 Response matrices, TDA and the Casida equations

<!-- section: 20.1; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"1. Response matrices" -->

For a closed-shell reference, linear response couples single excitations
$i \to a$ through two $d\times d$ matrices, $d = n_on_v$: $\mathbf A$
(excitation–excitation) and $\mathbf B$ (excitation–de-excitation). The
Tamm–Dancoff approximation (TDA) drops $\mathbf B$; full response (Casida)
keeps it. For an HF reference these are CIS and TDHF.

| Symbol | Meaning | Code name |
|---|---|---|
| $ia$ | compound index, `i * nvir + a` | `ia` |
| $\Delta_i^a$ | $\varepsilon_a - \varepsilon_i$ | the diagonal of `a`, `diag` |
| $c_J$ | Coulomb coefficient, 2 singlet, 0 triplet | `coulomb_factor` |
| $c_{\rm sr}, c_{\rm lr}, \omega$ | exact-exchange fractions and range separation | `ExchangeMix` |
| $K_{ia,jb}$ | XC kernel block | added by `add_fxc_block` |
| $B^P_{pq}$ | Coulomb-metric RI tensors ($ov$, $oo$, $vv$) | `RiTensors { b_ov, .. }` |
| $\Omega_n$, $X_n$, $(X+Y)_n$ | excitation energy and amplitudes | `omega`, `amplitudes` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{run_tddft,build_a_matrix,build_b_matrix,solve_tda,solve_casida}`, `ferric-tddft/src/davidson.rs::{davidson_tda,davidson_casida}` |
| CLI | `method.kind = "tda"` or `"tddft"`, `[tddft]` section |
| Python | `run_tddft(method="tda" \| "casida")` |

## 20.1.1 The matrices

<!-- include: crates/ferric-tddft/theory.md#"1. Response matrices" -->

$$ A_{ia,jb} = \delta_{ij}\delta_{ab}\Delta_i^a + 2(ia|jb) - c_{\rm HF}(ij|ab) + K_{ia,jb},\qquad
B_{ia,jb} = 2(ia|jb) - c_{\rm HF}(ib|ja) + K_{ia,jb}. \tag{20.1} $$

The 2 is the singlet spin adaptation. Every integral is RI in the Coulomb
metric, so the Coulomb part is $2\,B_{ov}^{\mathsf T}B_{ov}$ in both matrices;
the exchange in $\mathbf A$ is read from $B_{oo}^{\mathsf T}B_{vv}$, and that
in $\mathbf B$ is the $(ib, ja)$ element of the same $B_{ov}^{\mathsf T}B_{ov}$.

<!-- excerpt: ferric-tddft/src/tddft.rs::build_a_matrix -->
```rust
// Coulomb: c_J (ia|jb) = c_J B_ovᵀ B_ov.
if c_j != 0.0 {
    let coulomb = ri.b_ov.t().dot(&ri.b_ov);
    a += &(c_j * &coulomb);
}
```

> **Verified equation 20.1** · the singlet matrices are the spin adaptation of the spin-orbital ones
>
> Symbolically, $U = \tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ on
> the spin index block-diagonalizes the closed-shell spin-orbital $\mathbf A$
> and $\mathbf B$ into singlet ($c_J = 2$, kernel $f_{\alpha\alpha}+f_{\alpha\beta}$)
> and triplet ($c_J = 0$, kernel $f_{\alpha\alpha}-f_{\alpha\beta}$) blocks.
> Numerically, on a random closed-shell model with RI tensors for Coulomb and
> erf and a grid kernel, the spin-orbital $\mathbf A$ and $\mathbf B$ built
> from the textbook definition equal, after $U$, the code's singlet and
> triplet blocks for HF, a global hybrid and an RSH; and for an HF model,
> $\langle\Phi_i^a|H - E_0|\Phi_j^b\rangle$ evaluated over Jordan–Wigner
> determinants equals the spin-orbital $\mathbf A$ elementwise.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A1, B1, B2 |
> | Negative control | the runner sets the singlet `coulomb_factor` to 1 and reads $(ia|jb)$ for the $\mathbf B$ exchange (B1 fails both times) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::{build_a_matrix,build_b_matrix,coulomb_factor}` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft.rs` (against PySCF `TDA` and `TDDFT`) |
>
> <!-- verified: id=td.matrices.spin-adaptation; script=scripts/tddft_response_sympy.py; claim=B1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::build_a_matrix; rust_test=ferric-tddft/tests/validation_tddft.rs::tda_matches_the_validated_library_tda_path; registry=tddft -->

## 20.1.2 TDA and the Casida reduction

<!-- include: crates/ferric-tddft/theory.md#"2. Tamm-Dancoff approximation" -->

<!-- include: crates/ferric-tddft/theory.md#"3. Casida equations" -->

TDA is $\mathbf A X = \Omega X$, diagonalized by `solve_tda`. The full problem

$$ \begin{pmatrix}\mathbf A & \mathbf B\\ \mathbf B & \mathbf A\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix}
= \Omega\begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}\begin{pmatrix}X\\ Y\end{pmatrix} \tag{20.2} $$

reduces, when $\mathbf A - \mathbf B$ is positive definite, to the symmetric
problem

$$ T(\mathbf A+\mathbf B)T\,Z = \Omega^2 Z,\qquad T = (\mathbf A-\mathbf B)^{1/2},\qquad
(X+Y)_n = \Omega_n^{-1/2}\,TZ_n , \tag{20.3} $$

where the factor $\Omega_n^{-1/2}$ gives $|X|^2 - |Y|^2 = 1$.

<!-- excerpt: ferric-tddft/src/tddft.rs::solve_casida -->
```rust
// M = (A−B)^{1/2} (A+B) (A−B)^{1/2}
let m = amb_sqrt.dot(&a_plus_b).dot(&amb_sqrt);
```

A negative eigenvalue of $\mathbf A-\mathbf B$, or an $\Omega^2$ below
$-10^{-10}$ Ha$^2$, is an instability of the reference and an error, not a
clamped root. Both paths return the lowest `n_roots` roots above
$10^{-10}$ Ha.

> **Verified equation 20.3** · the Casida reduction and its normalization
>
> Symbolically, for the scalar case the eigenvalues of
> $\begin{pmatrix}a&b\\-b&-a\end{pmatrix}$ are $\pm\sqrt{(a-b)(a+b)}$ and
> $X+Y = \Omega^{-1/2}\sqrt{a-b}\,z$ gives $X^2 - Y^2 = 1$. Numerically, the
> positive eigenvalues of the full non-Hermitian matrix equal the code's
> $\Omega$, its eigenvector normalized to $X\cdot X - Y\cdot Y = 1$ gives the
> code's $X+Y$ up to sign, and the TDA roots are the $X$-block roots with
> $\mathbf B = 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims A3, B3, B4 |
> | Negative control | the runner scales $X+Y$ by $1/\Omega$ instead of $\Omega^{-1/2}$ in `solve_casida` (B4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::{solve_tda,solve_casida}` |
> | Code test | `casida_matches_dense_solve` in `crates/ferric-tddft/src/davidson.rs`; `crates/ferric-tddft/tests/validation_tddft.rs` |
>
> <!-- verified: id=td.casida.reduction; script=scripts/tddft_response_sympy.py; claim=B4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::solve_casida; rust_test=ferric-tddft/src/davidson.rs::casida_matches_dense_solve; registry=tddft -->

## 20.1.3 The matrix-free solvers

<!-- include: crates/ferric-tddft/theory.md#"12. Matrix-free Davidson solvers" -->

The dense matrices cost $d^2 \sim N^4$ memory and each `eigh` $N^6$ time.
`TddftSolver::Auto` keeps the dense path when $d \le 200$ or
$4n_{\rm roots} \ge d$ and otherwise runs Davidson, which needs only products
by $\mathbf A$ (TDA) or by $\mathbf A\pm\mathbf B$ (Casida) and the diagonal
$D_{ia} = \Delta_i^a$ as preconditioner.

For TDA the correction to a Ritz pair $(\theta, x)$ with residual
$r = \mathbf Ax - \theta x$ is Olsen's,

$$ t = Pr - \epsilon Px,\qquad P = (\theta - D)^{-1},\qquad \epsilon = \frac{x^{\mathsf T}Pr}{x^{\mathsf T}Px}, \tag{20.4} $$

so $t \perp x$; the plain $Pr$ stalls on a root that lies on a block of equal
orbital differences.

For Casida one subspace $V$ carries $X+Y$ and $X-Y$. With
$M_\pm = V^{\mathsf T}(\mathbf A\pm\mathbf B)V$ and $L = M_-^{1/2}$, the
Galerkin conditions give the subspace image of equation 20.3,

$$ LM_+L\,z = \Omega^2z,\qquad u = \Omega^{-1/2}Lz,\qquad v = M_+u/\Omega , \tag{20.5} $$

with $u^{\mathsf T}v = 1$, the normalization of section 20.1.2:

<!-- excerpt: ferric-tddft/src/davidson.rs::casida_ritz -->
```rust
let om = om2[j].sqrt();
let uk = l_half.dot(&z.column(j)) / om.sqrt();
let vk = mp.dot(&uk) / om;
```

The reported residual is $\sqrt{(\lVert r_+\rVert^2 + \lVert r_-\rVert^2)/2}$
with $r_+ = (\mathbf A+\mathbf B)(X+Y) - \Omega(X-Y)$ and
$r_- = (\mathbf A-\mathbf B)(X-Y) - \Omega(X+Y)$, which is the residual of
equation 20.2 in the $X$, $Y$ normalization.

| Setting | Default |
|---|---|
| guesses | lowest $\max(2n_{\rm roots}, n_{\rm roots}+8)$ diagonal elements, plus any tied with the last, each with a small dense part |
| `conv_tol` | $10^{-6}$ on the residual norm |
| `max_iter` | 100; an unconverged root is an error |
| subspace before collapse | $\max(n_{\rm guess} + 10n_{\rm roots}, 50)$ |

These solvers have no proof script. The unit tests `tda_matches_dense_eigh`,
`casida_matches_dense_solve`, `root_in_a_block_no_guess_touches_is_found`,
`unconverged_root_is_an_error` and
`non_positive_a_minus_b_is_an_rpa_instability` in
`crates/ferric-tddft/src/davidson.rs` compare them with the dense solve,
and the range-separated, triplet and open-shell validation tests run both
solvers against PySCF.

## 20.1.4 Numerical anchors

From `site/src/reference/validation.md`, "TDA and Casida TDDFT excitation energies".

| Quantity | Value | Test |
|---|---|---|
| lowest five roots vs PySCF, HF | $\le 2\times10^{-6}$ eV | `crates/ferric-tddft/tests/validation_tddft.rs` |
| LDA, PBE | $\le 1.2\times10^{-5}$ eV | the same |
| B3LYP, default pruned grid | $\le 8.1\times10^{-4}$ eV | the same |
| B3LYP, the reference's flat grid | $\le 2.8\times10^{-6}$ eV | the same |
| r2SCAN, water/6-31G, flat 75 × 110 grid | $2.1\times10^{-6}$ eV | the same |

## 20.1.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, sections 1–3, 6 and 12 | the matrices, the reductions, the cost and the Davidson solvers |
| `crates/ferric-dft/theory.md`, "$f_{xc}$ response kernels" | the kernel block $K$ (section 9.6) |
