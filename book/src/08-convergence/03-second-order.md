# 8.3 Second-order steps: Newton, augmented Hessian, trust region

<!-- section: 8.3; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Orbital Hessians: Newton, augmented Hessian, trust region" -->

When DIIS stalls, the solvers can take steps from the orbital Hessian
instead. All three methods here solve a shifted Newton equation
$(\mathbf H - \mu)\kappa = -g$ and differ in how they choose $\mu$. Each one
needs the product $\mathbf H\kappa$, built from one $J$/$K$ build per product.

| Symbol | Meaning | Code name |
|---|---|---|
| $\kappa_{ai}$ | occupied–virtual rotation, $C \leftarrow CU(\kappa)$ | `k`, `kappa` |
| $U(\kappa)$ | Cayley unitary $(1 - \kappa/2)^{-1}(1 + \kappa/2)$ | `cayley_unitary` |
| $g$ | the code's gradient, $f_{ai}$ (RHF) | `g` |
| $\mathbf H\kappa$ | the code's Hessian product | `hessian_matvec_jk` (RHF), `hessian_matvec` (UHF) |
| $\Delta_{\rm tr}$ | trust radius | the TRAH radius |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf_newton.rs::{rhf_newton_step,hessian_matvec_jk}`, `ferric-scf/src/uhf_newton.rs::{uhf_newton_step,hessian_matvec}`, `ferric-scf/src/rohf_newton.rs::rohf_newton_step`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step`, `ferric-scf/src/trah.rs::solve_trust_region`, `ferric-scf/src/cayley.rs::cayley_unitary` |
| CLI | `[scf] soscf = true` (sets `newton_trigger` to $10^{-3}$), `[scf] trah_trigger` |
| Python | `soscf=`, `trah_trigger=` on the SCF entry points |
| Rust only | `RhfConfig::newton_trigger` at other values, `RhfConfig::ah_trigger` (ROHF) |

## 8.3.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Orbital Hessians: Newton, augmented Hessian, trust region" -->

> **Verified equation 8.5** · orbital gradients and Hessian products
>
> For RHF, UHF and ROHF the code's gradient and Hessian product equal
> fourth-order central differences of the energy $E(CU(t\kappa))$, at the
> scales stated in the included derivation: a quarter of the RHF derivatives,
> half of the UHF ones, the exact ROHF ones from `rohf_trah`. The RHF product
> is exact only at canonical orbitals unless the off-diagonal gap correction
> is added.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claims C1, H1 to H6 |
> | Negative control | the response density factor 2 → 1 or the exchange weight $-\tfrac12 \to -1$ must fail H2; $J[\delta D^\alpha + \delta D^\beta] \to J[\delta D^\sigma]$ must fail H4; the ROHF $\tfrac12 b \to b$ must fail H5 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rhf_newton.rs::hessian_matvec_jk`, `ferric-scf/src/uhf_newton.rs::hessian_matvec`, `ferric-scf/src/rohf_trah.rs::rohf_exact_hessian_matvec` |
>
> <!-- verified: id=scf.orbital-hessian; script=scripts/scf_orbital_hessian_sympy.py; claim=H2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf_newton.rs::hessian_matvec_jk, ferric-scf/src/uhf_newton.rs::hessian_matvec, ferric-scf/src/rohf_trah.rs::rohf_exact_hessian_matvec; registry=rhf -->

> **Verified equation 8.6** · the augmented-Hessian step
>
> The lowest eigenpair of the augmented matrix gives
> $(\mathbf H - \mu)\kappa = -g$ and the predicted change
> $\tfrac12(g^{\mathsf T}\kappa + \mu\lvert\kappa\rvert^2)$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claim T1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/trah.rs::solve_trust_region`, `ferric-scf/src/rohf_ah.rs::rohf_ah_step` |
>
> <!-- verified: id=scf.ah-step; script=scripts/scf_orbital_hessian_sympy.py; claim=T1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/trah.rs::solve_trust_region, ferric-scf/src/rohf_ah.rs::rohf_ah_step; registry=rhf -->

## 8.3.2 The rotation

The Cayley unitary needs one matrix inverse:

<!-- excerpt: ferric-scf/src/cayley.rs::cayley_unitary -->
```rust
let mut a = Array2::<f64>::eye(n);
    a.scaled_add(-0.5, kappa);
    let mut u = with_blas_threads(opt_in_blas_threads(), || a.inv())
        .map_err(|e| FerricError::Lapack(format!("Cayley inverse: {e}")))?;
    u *= 2.0;
    for i in 0..n {
        u[(i, i)] -= 1.0;
    }
```

| Code | Equation |
|---|---|
| `a` | $1 - \kappa/2$ |
| `u` after `*= 2.0` and the diagonal $-1$ | $2(1 - \kappa/2)^{-1} - 1 = (1 - \kappa/2)^{-1}(1 + \kappa/2)$ |

$U$ is exactly orthogonal for any antisymmetric $\kappa$ and agrees with
$e^\kappa$ through second order (claim C1), so a converged Newton iteration
does not depend on which of the two maps is used.

## 8.3.3 The three steps

| Method | $\mu$ chosen by | Step limit | Solver | Who has it |
|---|---|---|---|---|
| Newton | 0 (PCG on $\mathbf H\kappa = -g$), ROHF truncates at negative curvature | $\max\lvert\kappa\rvert \le 0.2$ (0.1 ROHF), at most 20 PCG iterations | `rhf_newton_step`, `uhf_newton_step`, `rohf_newton_step` | RHF, UHF, ROHF via `newton_trigger` (`soscf`) |
| augmented Hessian | the lowest eigenvalue of the augmented matrix, $\alpha = 1$ | none beyond the eigenproblem | block Davidson | ROHF via `ah_trigger` (library only) |
| TRAH | the augmented eigenproblem, $\alpha$ adjusted until $\lvert\kappa\rvert$ fits the radius | $\Delta_{\rm tr}$, from 0.4, at most 2.0 | `solve_trust_region` | RHF, UHF via `trah_trigger` |

The TRAH radius update:

| $\rho = \Delta E/\Delta E_{\rm pred}$ | Step | Radius |
|---|---|---|
| $\le 0$ | rejected | $\times 0.7$ |
| $(0, 0.25]$ | accepted | $\times 0.7$ |
| $(0.25, 0.75]$ | accepted | unchanged |
| $> 0.75$ | accepted | $\times 1.2$, at most 2.0 |

Below $10^{-4}$ the region has collapsed and DIIS takes over.

## 8.3.4 Numerical anchors

The second-order solvers are tested by the energies they converge to and by
their smoke and anchor tests in `crates/ferric-scf/tests/`
(`rhf_newton_smoke.rs`, `uhf_newton_smoke.rs`, `rohf_newton_smoke.rs`,
`trah_converges.rs`, `trah_hard_case.rs`). The Hessian products themselves are
checked against finite differences of the energy by the proof above, which
is the anchor that a scale or sign error in the product would fail.

## 8.3.5 Doc fixes needed

| Text | Code |
|---|---|
| `rohf_newton.rs` module doc says the step is a DIIS | it is preconditioned conjugate gradient |
| `rohf_ah.rs` calls its eigenproblem "generalized" | it is a standard symmetric eigenproblem at $\alpha = 1$ |
| the TRAH module table rejects at $\rho < 0$ | the code rejects at $\rho \le 0$ (`rho <= self.cfg.rho_reject` in `crates/ferric-scf/src/trah.rs`) |
| the RHF Newton product is documented as exact and FD-validated | it is exact at canonical orbitals only; off them it needs `offdiag_gap_correction` |
| the Davidson `converged` flag and the `newton_trigger` docs | state the condition the code tests |
| stale text in `crates/ferric-scf/src/rohf.rs` describing the Newton route | update to the current PCG route |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Orbital Hessians: Newton, augmented Hessian, trust region" | the equations |
| `scripts/scf_orbital_hessian_sympy.py` | verified equations 8.5 and 8.6 |
| `crates/ferric-scf/src/trah.rs` | the radius table and the $\alpha$ search |
