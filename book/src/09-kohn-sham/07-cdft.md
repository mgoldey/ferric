# 9.7 Constrained DFT and the electron-transfer coupling

<!-- section: 9.7; status: ready; registry: cdft;
     theory: crates/ferric-dft/theory.md#"Constrained DFT and the diabatic coupling" -->

An electron moving from a donor to an acceptor passes between two states:
one with the charge on the donor, one with it on the acceptor. An unconstrained
SCF finds the ground state of the whole system, which can be a mixture of
the two. Constrained DFT finds each one directly: the lowest state whose
population on a chosen group of atoms equals a target. The Wu–Van Voorhis
coupling between two such states is the electron-transfer matrix element.

The plan: in 9.7.1, write the population of a group of atoms as a matrix
trace, add it to the energy with a Lagrange multiplier, derive the shifted
Fock matrix and the two derivatives the solver relies on, and compute the
coupling between two constrained determinants; in 9.7.2, solve for the
multiplier in an outer loop.

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

The code lives in `ferric-dft/src/cdft.rs`, `ferric-scf/src/cdft_driver.rs`
and `ferric-scf/src/cdft_coupling.rs`; the proof script is
`scripts/cdft_constraint_coupling_sympy.py`.

## 9.7.1 Equations

**The population as a trace.** "How many electrons are on fragment $C$"
needs a rule for dividing space. ferric uses the Becke partition of section
9.2.2: $w_C(\mathbf r) = \sum_{A\in C}w^A(\mathbf r)$, the share of each point
that belongs to the atoms of $C$. Integrate it against the density on a grid
of its own (default $99\times302$). The density is
$\sum_{\mu\nu}D_{\mu\nu}\chi_\mu\chi_\nu$, so the integral is a trace:

$$ W^C_{\mu\nu} = \sum_g w_g\,w_C(\mathbf r_g)\,\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g), \qquad N_C = \mathrm{Tr}[W^C(D^\alpha \pm D^\beta)] , $$

with $+$ for a charge constraint (`SpinChannel::Total`) and $-$ for a spin
constraint (`SpinChannel::SpinDiff`). `ferric-dft/src/cdft.rs::build_weight_matrix`
builds $W^C$ the way section 9.3 builds the LDA part of $V_{xc}$: scale the AO
columns by $w_gw_C(\mathbf r_g)$, then one GEMM.
`ferric-dft/src/cdft.rs::population` is the trace:

<!-- excerpt: ferric-dft/src/cdft.rs::population -->
```rust
let d = match spin {
    SpinChannel::Total => d_a[(i, j)] + d_b[(i, j)],
    SpinChannel::SpinDiff => d_a[(i, j)] - d_b[(i, j)],
};
// Tr[W D] = Σ_ij W_ij D_ji ; W and D symmetric ⇒ Σ_ij W_ij D_ij.
acc += w[(i, j)] * d;
```

**The constrained energy.** Look for the lowest energy at which $N_C$ equals
its target. Add the constraint with a multiplier $\lambda_C$:

$$ V(\boldsymbol\lambda) = E[\rho] + \sum_C\lambda_C\,(N_C[\rho] - N^{\rm target}_C) . $$

The constrained state makes $V$ stationary in $\rho$ and in
$\boldsymbol\lambda$. Stationarity in $\lambda$ is the constraint itself.
Stationarity in $\rho$ at fixed $\lambda$ is an ordinary UKS problem with one
extra term in the energy. That term is linear in the densities, so its
derivative is a constant matrix: $\partial(\lambda N_C)/\partial D^\alpha = \lambda W$
and $\partial(\lambda N_C)/\partial D^\beta = \pm\lambda W$. The Fock matrices
shift by

$$ \text{Total: } F_\alpha \mathrel{+}= \lambda W,\ F_\beta \mathrel{+}= \lambda W;\qquad \text{SpinDiff: } F_\alpha \mathrel{+}= \lambda W,\ F_\beta \mathrel{-}= \lambda W . $$

In `ferric-scf/src/cdft_driver.rs::solve_cdft_uhf_seeded` this is the Fock
modifier passed to the inner solve (`solve_uhf_fockmod`), per constraint:

<!-- excerpt: ferric-scf/src/cdft_driver.rs::solve_cdft_uhf_seeded -->
```rust
SpinChannel::Total => {
    // same potential to both spins
    let lw = l * &w_mats[ci];
    *f_a += &lw;
    *f_b += &lw;
}
SpinChannel::SpinDiff => {
    let lw = l * &w_mats[ci];
    *f_a += &lw;
    *f_b -= &lw;
}
```

A negative $\lambda$ lowers the potential on $C$, deepens the well there and
draws electrons in. The outer loop solves
$c(\lambda) = N[\rho_\lambda] - N^{\rm target} = 0$.

**Two derivatives.** Both follow from stationarity in $\rho$, the
Hellmann–Feynman argument.

- Differentiate $V(\lambda) = E[\rho_\lambda] + \lambda c(\lambda)$ along the
  solution. The density changes with $\lambda$, but $\partial V/\partial\rho = 0$
  at the solution, so only the explicit $\lambda$ survives:
  $$ \frac{dV}{d\lambda} = c(\lambda) . $$
- Treat the target as a parameter. At a solution $c = 0$, so $E = V$.
  Differentiate $V$ with respect to $N^{\rm target}$: the $\rho$ term vanishes
  by stationarity, the $\lambda$ term is multiplied by $c = 0$, and the
  explicit dependence is $-\lambda$:
  $$ \frac{dE}{dN^{\rm target}} = -\lambda . $$

So $-\lambda$ is the change in energy per electron added to the fragment.

On the lowest solution $V(\lambda) = \min_\rho(E + \lambda c)$ is a minimum
of functions linear in $\lambda$, so it is concave. Its derivative $c$ can
only decrease: $dc/d\lambda \le 0$. The reported energy is $E$ at the
constrained density, without $\lambda c$.

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

**The coupling between two diabatic states.** Take two converged
constrained UHF determinants, $\Psi_a$ (charge on the donor) and $\Psi_b$
(charge on the acceptor). They have different orbitals, so they are not
orthogonal, and matrix elements between them need the generalized
Slater–Condon rules. Start with the occupied-orbital overlap per spin and
its singular value decomposition (`ferric-scf/src/cdft_coupling.rs::biorth_pairing`):

$$ M_\sigma = \mathbf C^{\sigma\,\mathsf T}_{a,o}\, S\,\mathbf C^\sigma_{b,o} = U_\sigma\Sigma_\sigma V_\sigma^{\mathsf T} . $$

Rotate each state's occupied orbitals, $\tilde{\mathbf C}_a = \mathbf C_aU$ and
$\tilde{\mathbf C}_b = \mathbf C_bV$. In the rotated orbitals the overlap is
diagonal, $\tilde M_\sigma = \Sigma_\sigma = \mathrm{diag}(s^\sigma_i)$.
The determinant overlap is then a product, and for a one-body operator
$\hat O$ with paired diagonal
$d^\sigma_i = \tilde{\mathbf c}^{\sigma\,\mathsf T}_{a,i}\,O\,\tilde{\mathbf c}^\sigma_{b,i}$,

$$ \langle\tilde\Psi_a|\tilde\Psi_b\rangle = \prod_\sigma\prod_i s^\sigma_i, \qquad \langle\tilde\Psi_a|\hat O|\tilde\Psi_b\rangle = \sum_\sigma\Bigl[\sum_i d^\sigma_i\prod_{j\ne i}s^\sigma_j\Bigr]\prod_j s^{\sigma'}_j , $$

where $\sigma'$ is the other spin. Where does the one-body formula come
from? Per spin, the matrix element is $\frac{d}{dt}\det(M + tO)$ at $t = 0$.
By Jacobi's formula that is $\mathrm{tr}(\mathrm{adj}(M)\,O)$, and the
adjugate of a diagonal matrix is diagonal with entries $\prod_{j\ne i}s_j$.
The operator acts on one spin at a time, so the product rule over the two
spin determinants gives the sum over $\sigma$, each term times the other
spin's determinant.

The formula has no division. It holds with any number of zero singular
values. Two diabatic states with the transferred electron in different
orbitals can have a zero $s_i$. The code
builds the cofactor sum from prefix and suffix products
(`ferric-scf/src/cdft_coupling.rs::cofactor_sum`):

<!-- excerpt: ferric-scf/src/cdft_coupling.rs::cofactor_sum -->
```rust
let mut suffix = vec![1.0; n + 1];
for i in (0..n).rev() {
    suffix[i] = suffix[i + 1] * s_vals[i];
}
let mut prefix = 1.0;
let mut total = 0.0;
for i in 0..n {
    total += d[i] * prefix * suffix[i + 1];
    prefix *= s_vals[i];
}
```

| Symbol | Code |
|---|---|
| $\prod_{j > i}s_j$ | `suffix[i + 1]` |
| $\prod_{j < i}s_j$ | `prefix` at step $i$ |
| $\sum_id_i\prod_{j\ne i}s_j$ | `total` |

and `ferric-scf/src/cdft_coupling.rs::cross_one_body` combines the two spins:

<!-- excerpt: ferric-scf/src/cdft_coupling.rs::cross_one_body -->
```rust
let alpha = cofactor_sum(&pair_alpha.s_vals, &paired_diagonal(op_ao, pair_alpha));
let beta = cofactor_sum(&pair_beta.s_vals, &paired_diagonal(op_ao, pair_beta));
alpha * pair_beta.det_m + beta * pair_alpha.det_m
```

The rotated determinants differ from the original ones by the phase
$\det U_\sigma\det V_\sigma$ per spin, so $S_{ab}$ and $H_{ab}$ carry that sign
convention. $\lvert H_{ab}\rvert$ does not depend on it.

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

**The Hamiltonian element.** Wu and Van Voorhis evaluate it from the
state energies and one-body elements only. Each converged state is treated as an eigenfunction of
$\hat H + \lambda_X\hat w_X$, the Hamiltonian it was solved with, with
eigenvalue $F_X = E_X + \lambda_XN_X$. Here
$N_X = \langle\Psi_X|\hat w_X|\Psi_X\rangle$ comes from the state's own
orbitals (`own_population`). Then
$\langle a|\hat H|b\rangle = F_bS_{ab} - \lambda_b\langle a|\hat w_b|b\rangle$
acting to the right, and the same with $a$ to the left. The code takes the
average of the two:

$$ H_{\rm raw} = \tfrac12\bigl[(F_b S_{ab} - \lambda_b\langle a|\hat w_b|b\rangle) + (F_a S_{ab} - \lambda_a\langle a|\hat w_a|b\rangle)\bigr] . $$

The two states are not orthogonal, so $H_{\rm raw}$ is not yet a coupling.
Orthogonalize symmetrically. For the $2\times2$ problem
$H = \bigl[\begin{smallmatrix}E_a & H_{\rm raw}\\ H_{\rm raw} & E_b\end{smallmatrix}\bigr]$,
$S = \bigl[\begin{smallmatrix}1 & S_{ab}\\ S_{ab} & 1\end{smallmatrix}\bigr]$,
write $S^{-1/2} = p\,I + q\,\sigma_x$ with
$p = \tfrac12[(1+S_{ab})^{-1/2} + (1-S_{ab})^{-1/2}]$ and
$q = \tfrac12[(1+S_{ab})^{-1/2} - (1-S_{ab})^{-1/2}]$. The off-diagonal
element of $S^{-1/2}HS^{-1/2}$ is $(p^2+q^2)H_{\rm raw} + pq(E_a + E_b)$.
Multiply out: $p^2 + q^2 = 1/(1 - S_{ab}^2)$ and
$pq = -S_{ab}/[2(1 - S_{ab}^2)]$, so

$$ H_{ab} = \frac{H_{\rm raw} - \tfrac12(E_a+E_b)S_{ab}}{1 - S_{ab}^2} . $$

At $S_{ab} = 0$ this is $H_{\rm raw}$. When $\lvert 1 - S_{ab}^2\rvert < 10^{-10}$
the two states are the same state and `coupling_hab` returns $E_a$.

**Offset invariance.** Any constant added to the weight operator,
$W \to W - cS$ (the target moves with it, $N \to N - cN_e$), must leave the
coupling unchanged. Each $\lambda_X$ in $H_{\rm raw}$ multiplies
$N_XS_{ab} - \langle a|\hat w_X|b\rangle$, the matrix element of
$\hat w_X - N_X$. Under the shift, $\langle a|\hat w|b\rangle$ moves by
$-cN_eS_{ab}$ and $N_X$ by $-cN_e$, so the combination does not move. With
$E_X$ in place of $F_X$ the $N_XS_{ab}$ term is missing and the coupling is
not invariant. The coupling takes charge constraints only, where the same
$\hat w$ acts on both spins.

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

In `coupling_hab`, after the pairings, the overlap and the two cross
elements:

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

The Rust tests are `crates/ferric-scf/tests/cdft_outer_loop.rs` (the
HeNe$^+$ outer loop), `crates/ferric-scf/tests/cdft_uhf.rs`, and
`validation_cdft_et.rs`, which compares the He$_2^+$ coupling with NWChem's
direct $\langle A|\hat H|B\rangle$.

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

Step 5 uses the first derivative of 9.7.1. On one smooth SCF branch,
$dV/d\lambda = c$, so $V(\lambda + h) - V(\lambda) = \int c\,d\lambda$, which the
trapezoid rule approximates to $O(h^3)$. If the inner SCF at $\lambda + h$ fell
into a different basin, $V$ jumps by the energy gap between the two
solutions, and the mismatch is that gap. `hf_mismatch` in
`ferric-scf/src/cdft_driver.rs` is that test:

<!-- excerpt: ferric-scf/src/cdft_driver.rs::hf_mismatch -->
```rust
((b.v - a.v) - 0.5 * h * (a.c + b.c)).abs()
```

A pair is trusted when the mismatch is at most `HF_MISMATCH_TOL`
$= 10^{-6}$ Ha. Measured on the HeNe$^+$ and He$_2^+$ runs behind it,
same-basin pairs fall between $1.8\times10^{-14}$ and $7.0\times10^{-9}$ Ha and
cross-basin pairs between $5.5\times10^{-5}$ and $0.35$ Ha. $10^{-6}$ is 140
times the largest same-basin value and 55 times below the smallest
cross-basin one. One pair, at $2.7\times10^{-7}$ Ha on a steep stretch of
$c(\lambda)$ ($dc/d\lambda = -6.6$), belongs to neither group.

Why not trust every slope? $c(\lambda)$ can be a near-step function. On
HeNe$^+$ the populations reachable at fixed $\lambda$ form two flat branches
with a gap between them. A finite-difference probe that straddles the gap
divides a change of state by $10^{-3}$. That is why each Newton pair is
checked against $dV/d\lambda = c$, and why the bracket compares residuals,
not $\lambda$ (the well deepens as $\lambda$ goes negative). The module
documentation of `cdft_driver.rs` records the measurements.

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
| `crates/ferric-dft/theory.md`, "Constrained DFT and the diabatic coupling" | the reference statement of the equations |
| module documentation of `crates/ferric-scf/src/cdft_driver.rs` and `crates/ferric-scf/src/cdft_coupling.rs` | the outer-loop measurements; the derivation of the invariant form |
| `site/src/methods/cdft.md` | how to run it, accuracy, tolerances |
| Wu and Van Voorhis, J. Chem. Phys. 125, 164105 (2006); Kaduk, Kowalczyk and Van Voorhis, Chem. Rev. 112, 321 (2012) | the method and the coupling |
