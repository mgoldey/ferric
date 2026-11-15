# 9.7 Constrained DFT and the electron-transfer coupling

<!-- section: 9.7; status: ready; registry: cdft;
     theory: crates/ferric-dft/theory.md#"Constrained DFT and the diabatic coupling" -->

Constrained DFT finds the lowest state whose Becke population on a group of
atoms (a charge, or a spin difference) equals a target. It adds a
Lagrange-multiplier potential to the Fock matrix and solves for the
multiplier in an outer loop. Two such states, each with its charge
localized on a different fragment, are diabatic states, and the
Wu–Van Voorhis coupling between them is the electron-transfer matrix
element.

| Symbol | Meaning | Code name |
|---|---|---|
| $w_C(\mathbf r)$ | population function of the group $C$, $\sum_{A\in C}w^A$ | |
| $W^C$ | its AO matrix on the cDFT grid | `w` |
| $N_C$, $N^{\rm target}_C$ | population, target | `population`, `target` |
| $\lambda_C$ | multiplier (Hartree per electron) | `lambda` |
| $c(\lambda)$ | residual $N - N^{\rm target}$ | `c` |
| $V(\lambda)$ | $E + \lambda c$ | `v` |
| $M_\sigma$, $s_i$, $d_i$ | occupied-overlap matrix, its singular values, paired diagonal of an operator | `m`, `s_vals`, `paired_diagonal` |
| $F_X$, $S_{ab}$, $H_{ab}$ | $E_X + \lambda_XN_X$, determinant overlap, coupling | `f_a`, `s_ab`, `h_ab` |

| Entry point | Location |
|---|---|
| Rust | `ferric-dft/src/cdft.rs::{build_weight_matrix,population}`, `ferric-scf/src/cdft_driver.rs::{solve_cdft_uhf,solve_cdft_uhf_seeded}`, `ferric-scf/src/cdft_coupling.rs::coupling_hab` |
| CLI | `[cdft]` on `kind = "uhf"`, `task = "energy"` |
| Python | `run_cdft`, `CdftConstraint`, `cdft_coupling` |

## 9.7.1 Equations

<!-- include: crates/ferric-dft/theory.md#"Constrained DFT and the diabatic coupling" -->

> **Verified equation 9.11** · the Fock shift and $dE/dN = -\lambda$
>
> For a model UHF energy (random $h$, a small 8-fold-symmetric two-electron
> tensor) solved self-consistently with the code's shift
> ($F_\alpha \mathrel{+}= \lambda W$, $F_\beta \mathrel{\pm}= \lambda W$),
> central differences in $\lambda$ give $dV/d\lambda = N - N^{\rm target}$
> to $5\times10^{-11}$ and $dE/dN = -\lambda$ to $10^{-5}$ or better, for
> Total and SpinDiff constraints. The route uses only converged SCF
> solutions, not the derivation of the shift.
>
> | | |
> |---|---|
> | Proof | `scripts/cdft_constraint_coupling_sympy.py`, claim envelope (check 4) |
> | Negative control | the runner flips the shift to $F \mathrel{-}= \lambda W$: $dV/d\lambda$ misses by $0.11$ and $dE/dN = +\lambda$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/cdft_driver.rs::solve_cdft_uhf`, `ferric-dft/src/cdft.rs::population` |
> | Code test | `crates/ferric-scf/tests/validation_cdft.rs` (dE/dN = −λ by Simpson's rule) |
>
> <!-- verified: id=ks.cdft.envelope; script=scripts/cdft_constraint_coupling_sympy.py; claim=envelope; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/cdft_driver.rs::solve_cdft_uhf; registry=cdft -->

> **Verified equation 9.12** · the one-body element between non-orthogonal determinants
>
> `cross_one_body` (SVD pairing, cofactor sum by prefix and suffix
> products, the other spin's determinant) equals the Fock-space matrix
> element $\langle\Psi_a\vert\sum_{pq}O_{pq}a^\dagger_pa_q\vert\Psi_b\rangle$
> computed by brute force over occupation strings in a Löwdin-orthonormalized
> basis, after the SVD phase $\det U\det V$ per spin, to $1.3\times10^{-15}$:
> generic, with one zero singular value (where the element is nonzero), and
> with two; the determinant overlap likewise.
>
> | | |
> |---|---|
> | Proof | `scripts/cdft_constraint_coupling_sympy.py`, claim cofactor (check 1) |
> | Negative control | the runner pairs each spin's cofactor sum with its own determinant instead of the other spin's |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/cdft_coupling.rs::{cross_one_body,biorth_pairing}` |
> | Code test | `crates/ferric-scf/tests/validation_cdft_et.rs` (every NWChem `et` ingredient) |
>
> <!-- verified: id=ks.cdft.cofactor; script=scripts/cdft_constraint_coupling_sympy.py; claim=cofactor; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/cdft_coupling.rs::cross_one_body; registry=cdft -->

> **Verified equation 9.13** · the coupling: orthogonalization and offset invariance
>
> Symbolically, the off-diagonal element of $S^{-1/2}HS^{-1/2}$ for the
> $2\times2$ problem is $(H_{\rm raw} - \tfrac12(E_a + E_b)S_{ab})/(1 - S_{ab}^2)$.
> Numerically, for two random UHF states with different MO sets,
> `coupling_hab` is unchanged when both weight operators become $W - cS$;
> the form with $E_X$ in place of $F_X$ changes by $0.23$ Ha.
>
> | | |
> |---|---|
> | Proof | `scripts/cdft_constraint_coupling_sympy.py`, claims lowdin and invariance (checks 2, 3) |
> | Negative control | the runner replaces $F_b$ by $E_b$ in $H_{\rm raw}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/cdft_coupling.rs::coupling_hab` |
> | Code test | `crates/ferric-scf/tests/validation_cdft_et.rs` |
>
> <!-- verified: id=ks.cdft.coupling; script=scripts/cdft_constraint_coupling_sympy.py; claim=invariance; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/cdft_coupling.rs::coupling_hab; registry=cdft -->

<!-- excerpt: ferric-scf/src/cdft_coupling.rs::coupling_hab -->
```rust
let f_a = e_a + state_a.lambda * n_a;
let f_b = e_b + state_b.lambda * n_b;
let h_raw =
    0.5 * ((f_b * s_ab - state_b.lambda * w_b_elem) + (f_a * s_ab - state_a.lambda * w_a_elem));
```

| Symbol | Code |
|---|---|
| $N_X$ (own population) | `n_a`, `n_b` |
| $\langle a\vert\hat w_X\vert b\rangle$ | `w_a_elem`, `w_b_elem` |
| $S_{ab}$ | `s_ab` |

## 9.7.2 The outer loop

| Step | What | Function |
|---|---|---|
| 1 | $W^C$ on its own grid (default $99\times302$) | `build_weight_matrix` |
| 2 | inner UKS (or UHF) at fixed $\lambda$ with the shifted Fock | `solve_uhf_fockmod` |
| 3 | $c(\lambda)$; a constraint counts as met only once $\lambda$ has been applied, unless $c$ is already at the population floor | `solve_scalar` |
| 4 | one constraint: second solve at $\lambda + 10^{-3}$, finite-difference slope, Newton step clamped to the trust radius (at most 1 Ha/e) | `solve_scalar` |
| 5 | trust the pair only if $V(\lambda + h) - V(\lambda) = h(c_\lambda + c_{\lambda+h})/2$ to the trapezoid error (same SCF branch); otherwise step toward lower $\lvert c\rvert$ | `solve_scalar` |
| 6 | keep a sign-change bracket and bisect when a step leaves it; backtrack after an unconverged inner SCF | `Bracket` |
| 7 | several constraints: a plain $k\times k$ finite-difference Newton step | `solve_cdft_uhf` |
| 8 | optional stability descent from a constrained saddle (default on in Python, skipped for UKS) | |

$c(\lambda)$ can be a near-step function: on HeNe$^+$ the populations
reachable at fixed $\lambda$ form two flat branches with a gap between them,
which is why each Newton pair is checked against $dV/d\lambda = c$ and why
the bracket compares residuals, not $\lambda$ (the well deepens as $\lambda$
goes negative). The module documentation of `cdft_driver.rs` records the
measurements.

**Reproducibility.** The constrained populations are defined by the Becke
partition with ferric's Bragg–Slater radii, including its own He, Ne and Ar
values (section 9.2); a population from another code's partition differs.
The sign of $S_{ab}$ and $H_{ab}$ follows the SVD phase convention;
$\lvert H_{ab}\rvert$ does not depend on it.

## 9.7.3 Options

| Option (Python `run_cdft`) | Effect | Default |
|---|---|---|
| `constraints` | list of `CdftConstraint` (atoms, `"charge"` or `"spin"`, target population) | |
| `functional` | `None` or `"HF"` = UHF; any other = UKS | UHF |
| `lambda_tol`, `max_outer` | outer convergence (electrons), iteration cap | $10^{-5}$, 30 |
| `stability_descent` | descend from a constrained saddle | on |
| `grid_radial`, `grid_angular` | the weight grid | 99, 302 |

`cdft_coupling` takes two converged single-constraint charge states on the
same geometry, basis and Hamiltonian; spin constraints are not supported
there. A loose `lambda_tol` passes its error into $H_{ab}$
(`site/src/methods/cdft.md`, "Tolerance and coupling error").

## 9.7.4 Numerical anchors

From `site/src/reference/validation.md`, the constrained-DFT rows.

| Quantity | Systems | Agreement | Test |
|---|---|---|---|
| $E(N) - E_{\rm unc}$, $\lambda$ vs NWChem 7.2.2 (PBE, Becke populations) | LiH, HF, H2O$^+$ / 6-31G, def2-SVP | $2.0\times10^{-7}$ Ha; $1.1\times10^{-6}$ | `crates/ferric-scf/tests/validation_cdft.rs` |
| $dE/dN = -\lambda$ (Simpson) | the same, def2-SVP | $2.3\times10^{-12}$ Ha | the same |
| state selection seeded from NWChem orbitals | HeNe$^+$, LiH$^+$ / def2-SVP | $\le 7.7\times10^{-10}$ Ha | `crates/ferric-scf/tests/validation_cdft_state.rs` |
| coupling kernel vs NWChem `et` ingredients | He$_2^+$ / def2-SVP, aug-cc-pVDZ | $\le 5.9\times10^{-11}$ Ha | `crates/ferric-scf/tests/validation_cdft_et.rs` |
| $\lvert V\rvert$ end to end, KS-PBE diabats | the same, 2.5 to 3.5 Å | $\le 8.3\times10^{-4}$ relative | the same |

## 9.7.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Constrained DFT and the diabatic coupling" | the equations |
| module documentation of `crates/ferric-scf/src/cdft_driver.rs` and `crates/ferric-scf/src/cdft_coupling.rs` | the outer-loop measurements; the derivation of the invariant form |
| `site/src/methods/cdft.md` | how to run it, accuracy, tolerances |
| Wu and Van Voorhis, J. Chem. Phys. 125, 164105 (2006); Kaduk, Kowalczyk and Van Voorhis, Chem. Rev. 112, 321 (2012) | the method and the coupling |
