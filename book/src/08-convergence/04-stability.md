# 8.4 Stability analysis

<!-- section: 8.4; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"SCF stability analysis" -->

A converged SCF is a stationary point of the energy, not necessarily a
minimum. Stability analysis computes the lowest eigenvalues of the orbital
Hessian at the converged orbitals. A negative one means a lower solution
exists along its eigenvector: of the same kind (internal) or with broken spin
symmetry (external, RHF → UHF). The descent then follows it.

| Symbol | Meaning | Code name |
|---|---|---|
| $\lambda_{\min}$ | lowest eigenvalue of the stability operator | `lowest_eigenvalue` |
| noise floor | $\max(10^{-6}, \text{residual})$ | `noise_floor` |
| $\lambda_T$ | lowest eigenvalue of the triplet (RHF → UHF) operator | the external result |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/stability.rs::{rhf_internal_stability,rhf_external_stability,uhf_internal_stability}`, `ferric-scf/src/uhf.rs::stability_descent`, `ferric-scf/src/rhf.rs::rhf_stability_descent` |
| CLI | `[scf] check_stability`, `stability_descent` |
| Python | `stability_descent=` on the SCF entry points |

## 8.4.1 The equations

<!-- include: crates/ferric-scf/theory.md#"SCF stability analysis" -->

> **Verified equation 8.7** · the singlet and triplet channels
>
> At an RHF point the UHF Hessian product in the direction
> $\kappa^\alpha = \kappa^\beta$ is the RHF singlet operator and in the
> direction $\kappa^\alpha = -\kappa^\beta$ the triplet operator; the UHF
> energy along $(v, -v)$ has curvature $4\lambda_T$.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_orbital_hessian_sympy.py`, claims S1 and S2 |
> | Negative control | the triplet exchange term set to zero must fail S1 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/stability.rs::{rhf_external_stability,uhf_internal_stability}` |
>
> <!-- verified: id=scf.stability-channels; script=scripts/scf_orbital_hessian_sympy.py; claim=S1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/stability.rs::{rhf_external_stability,uhf_internal_stability}; registry=uhf -->

## 8.4.2 The verdict

The verdict is four-way and is what callers branch on:

<!-- excerpt: ferric-scf/src/stability.rs::verdict -->
```rust
if !self.converged {
            StabilityVerdict::Indeterminate
        } else if self.is_marginal() {
            StabilityVerdict::Marginal
        } else if self.lowest_eigenvalue > 0.0 {
            StabilityVerdict::Stable
        } else {
            StabilityVerdict::Unstable
        }
```

| Code | Meaning |
|---|---|
| `converged` | the Davidson solve reached its residual |
| `is_marginal()` | $\lvert\lambda_{\min}\rvert \le$ noise floor: neither a proven minimum nor a proven saddle |
| `lowest_eigenvalue > 0.0` | a local minimum in the directions analysed |

The eigenvalues carry the code's Hessian scale (a quarter of the RHF energy
curvature, half of the UHF one), so a threshold on $\lambda$ is not a
threshold in Hartree per radian squared.

## 8.4.3 The descent

| Step | What |
|---|---|
| 1 | the lowest eigenvector $v$ |
| 2 | orbitals rotated by $\epsilon v$, $\epsilon \in \{0.4, 0.8, 1.2\}$ |
| 3 | each start reconverged; the lowest energy below the incumbent kept |
| 4 | repeated up to three rounds, with a warning if still descending |

Real-to-complex, UHF-to-GHF and ROHF stability are not implemented.

## 8.4.4 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| lowest Hessian eigenvalues, UHF/UKS internal, RHF internal and external: N2⁺, OH, water at 0.9572 and 2.0 Å, NH2 / 6-31G | PySCF dense Hessians and `stability()` | ≤ 4.5e-10 Ha | `crates/ferric-scf/tests/validation_scf_stability.rs` |
| triplet operator against the UHF Hessian at the RHF point | an independent construction | ≤ 2.3e-13 Ha | `crates/ferric-scf/tests/scf_stability_external.rs` |
| water at 2.0 Å: internal and external $\lambda_{\min}$ | PySCF | internal STABLE (+1.97e-2), external UNSTABLE (−3.07e-1), verdicts match | `crates/ferric-scf/tests/validation_scf_stability.rs` |
| convergence ladder with descent: N2 at 1.60 Å, CuCN, Cr(CO)6 / def2-SVP | PySCF `.newton()` from four guesses | energy ≤ 8.8e-10 Ha; plain DIIS lands on the N2 saddle 1.906e-2 Ha higher | `crates/ferric-scf/tests/validation_scf_ladder.rs` |

<!-- figures: site/src/reference/validation.md "SCF stability: lowest orbital-Hessian eigenvalues and verdict", "SCF convergence ladder with RHF internal stability descent" -->

OH is an exact zero mode; ferric reports it MARGINAL where PySCF reports
stable, which is the intended reading of the noise floor. Kohn–Sham references
are refused for the external channel, which would need the triplet XC kernel.

## 8.4.5 Doc fixes needed

| Text | Code |
|---|---|
| eigenvalues described as Hartree per radian squared | they are the quarter-scale (RHF) or half-scale (UHF) Newton operator's |
| the `check_stability` doc | name both the internal and the external result it fills |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "SCF stability analysis" | the operators |
| `scripts/scf_orbital_hessian_sympy.py` | verified equation 8.7 |
| `site/src/reference/validation.md` | the anchors |
