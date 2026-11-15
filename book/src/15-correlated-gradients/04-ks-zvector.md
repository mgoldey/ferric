# 15.4 Kohn–Sham Z-vector for post-SCF corrections

<!-- section: 15.4; status: ready; registry: ks-zvector;
     theory: crates/ferric-scf/theory.md#"Kohn-Sham Z-vector for post-SCF corrections" -->

A correction computed after a Kohn–Sham SCF, such as a many-body dispersion
energy evaluated from the converged density, depends on the orbitals and
needs the same orbital relaxation term as a correlated method. ferric
provides it as one module, `ferric_scf::zvector_ks`, for RKS, UKS and ROKS
references. Its only production caller is the MBD@rsSCS gradient. The
derivation is `crates/ferric-scf/theory.md`, "Kohn-Sham Z-vector for post-SCF
corrections", and the proof is `scripts/ks_zvector_sympy.py`.

| Symbol | Meaning | Code name |
|---|---|---|
| $Q(R, D)$ | the post-SCF quantity | |
| $V$ | $\partial Q/\partial D$ in the AO basis, symmetric | `v_ao` |
| $H$ | KS orbital Hessian, the Jacobian of $F_{ai} = 0$ | the Hessian product |
| $Z$ | the multipliers | `z` |
| $g(D, W)$ | the SCF gradient expression at density $D$ and energy-weighted density $W$ | `ks_gradient_closed_for_density` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/zvector_ks.rs::{relaxation_gradient_closed,relaxation_gradient_unrestricted,relaxation_gradient_roks}` |
| Caller | `crates/ferric-rpa/src/dispersion/mbd_scf.rs` (MBD@rsSCS exact gradient) |
| CLI, Python | through `[dft] dispersion = "mbd"` with `task = "optimize"`, and `run_dft(dispersion="mbd", with_gradient=True)` |

## 15.4.1 The equations

The total derivative splits into a fixed-orbital part, which the caller
supplies, and the relaxation:

$$ \frac{dQ}{dR} = \frac{\partial Q}{\partial R}\bigg|_{\text{occupied orbitals fixed}} + \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial R}, \qquad H Z = -4V_{vo}, \quad V_{vo} = \mathbf C_v^{\mathsf T}V\mathbf C_o . \tag{15.10} $$

"Fixed" means the occupied orbitals follow their basis functions and stay
orthonormal, $\delta\mathbf C_o = -\tfrac12\mathbf C_o S^x_{oo}$. The factor 4
is the closed-shell convention: a rotation $\kappa$ changes the density by
$2\mathbf C(\kappa + \kappa^{\mathsf T})\mathbf C^{\mathsf T}$, so
$dQ = 4\sum\kappa_{ai}V_{ai}$.

**The contraction without new derivative integrals.** Because
$F_{ai} = \tfrac14\partial E/\partial\kappa_{ai}$,

$$ \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial R} = \frac14\,\frac{d}{dt}\,g\big(D(t),\; W + t\,\dot W\big)\Big|_{t=0}, \qquad \delta D_Z = 2(\mathbf C_vZ\mathbf C_o^{\mathsf T} + \mathbf C_oZ^{\mathsf T}\mathbf C_v^{\mathsf T}), \tag{15.11} $$

with $\dot D(0) = \delta D_Z$, $\dot W = \tfrac12(\delta D_ZFD + DF\delta D_Z + D\,\delta F\,D)$ and
$\delta F$ the response Fock matrix of $\delta D_Z$. The code takes a central
difference in $t$ of the SCF gradient routine itself, with $D(t)$ built from
occupied orbitals rotated by $tZ$ and re-orthonormalized (an exact
projector). That projector has an odd $t^3$ term, so the central difference
carries an $O(t^2)$ error in every part of $g$, not only the XC part; at the
chosen step, $t\max|\delta D_Z| = 10^{-4}$, it is about $10^{-8}$ relative.

<!-- excerpt: ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed -->
```rust
let rhs = -4.0 * cv.t().dot(v_ao).dot(&co);
```

| Symbol | Code |
|---|---|
| $-4V_{vo}$ | `rhs` |
| $V$ | `v_ao` |
| $\mathbf C_v, \mathbf C_o$ | `cv`, `co` |

The contraction (15.11) is returned as `0.25 * (&gp - &gm) / (2.0 * t)`,
with `gp` and `gm` the SCF gradient expression at $\pm t$.

> **Verified equation 15.10** · the Z-vector relaxation term
>
> For RKS, the caller's fixed-orbital part plus $\sum Z_{ai}\,\partial F_{ai}/\partial x$,
> with $HZ = -4V_{vo}$ and $\partial F_{ai}/\partial x$ taken along the explicit
> fixed path, equals a Richardson difference of $Q$ with the SCF re-solved, on
> a model with $x$-dependent $S$, $h$, factorized integrals and an LDA-type XC
> kernel. At $Q = E_{\rm SCF}$ the right-hand side vanishes and $Z = 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/ks_zvector_sympy.py`, claims Z0 and Z1 |
> | Negative control | the runner changes $-4V_{vo}$ to $-2V_{vo}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed` |
> | Code test | `mbd_exact_gradient_matches_fd_of_full_scf_pipeline` in `crates/ferric-rpa/tests/mbd_scf_gradient.rs` |
>
> <!-- verified: id=ks.zvector; script=scripts/ks_zvector_sympy.py; claim=Z1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed; rust_test=ferric-rpa/tests/mbd_scf_gradient.rs::mbd_exact_gradient_matches_fd_of_full_scf_pipeline; registry=ks-zvector -->

> **Verified equation 15.11** · the contraction by a central difference
>
> $\tfrac14[g(+t) - g(-t)]/(2t)$ at the rotated projector and $W + t\dot W$
> equals $\sum Z_{ai}\,\partial F_{ai}/\partial x$; halving $t$ divides the
> error by 4, with and without the XC term.
>
> | | |
> |---|---|
> | Proof | `scripts/ks_zvector_sympy.py`, claim Z2 |
> | Negative control | the runner changes the factor $\tfrac14$ to $\tfrac12$, and drops $D\,\delta F\,D$ from $\dot W$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed` |
>
> <!-- verified: id=ks.zvector.contraction; script=scripts/ks_zvector_sympy.py; claim=Z2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed; rust_test=ferric-rpa/tests/mbd_scf_gradient.rs::mbd_exact_gradient_matches_fd_of_full_scf_pipeline; registry=ks-zvector -->

> **Verified equations 15.10 and 15.11, open shell** · the UKS and ROKS blocks
>
> UKS with $HZ^\sigma = -2V^\sigma_{vo}$ and the factor $\tfrac12$, and ROKS
> with $HZ = -(4V_{vc}, 2V_{vo}, 2V_{oc})$, the three-block product and the
> closed-open cross term of the fixed path, each against a Richardson
> difference of $Q$ with the SCF re-solved; $\alpha = \beta$ reproduces RKS.
>
> | | |
> |---|---|
> | Proof | `scripts/ks_zvector_sympy.py`, claims Z3 and Z4 |
> | Negative control | the runner changes the UKS $-2V_{vo}$ to $-4V_{vo}$, the ROKS $4V_{vc}$ to $2V_{vc}$, and drops the cross term of $\dot W$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/zvector_ks.rs::{relaxation_gradient_unrestricted,relaxation_gradient_roks}` |
> | Code test | `roks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient` in `crates/ferric-rpa/tests/mbd_scf_gradient_roks.rs` |
>
> <!-- verified: id=ks.zvector.open-shell; script=scripts/ks_zvector_sympy.py; claim=Z4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/zvector_ks.rs::{relaxation_gradient_unrestricted,relaxation_gradient_roks}; rust_test=ferric-rpa/tests/mbd_scf_gradient_roks.rs::roks_path_on_a_closed_shell_reproduces_the_closed_shell_gradient; registry=ks-zvector -->

## 15.4.2 The solver

Preconditioned conjugate gradient with the orbital-energy gap as
preconditioner, converged when $\max|r| \le 10^{-10}\max|\mathrm{rhs}|$,
with an error after 200 iterations. When the curvature along a search
direction is numerically zero, the solve is accepted only if the residual is
already below $10^{-8}\max|\mathrm{rhs}|$, and is an error otherwise. The
Hessian product uses the same $J$, $K$ and XC kernel routes as the SCF (RI-J,
fitted or COSX exchange, LDA, GGA, hybrid, range-separated hybrid and τ-only
meta-GGA kernels).

| Reference | Rotations | Right-hand side | Contraction factor |
|---|---|---|---|
| RKS | $vo$ | $-4V_{vo}$ | $\tfrac14$ |
| UKS | $vo$ per spin, coupled through $\delta J$ of the total density | $-2V_{vo}$ per spin | $\tfrac12$ |
| ROKS | $vc$, $vo$, $oc$ | $-(4V_{vc}, 2V_{vo}, 2V_{oc})$ | $\tfrac12$ |

For a linear ²Π radical the Hessian has a null mode (rotation about the
molecular axis mixes the degenerate orbitals); the open-shell solvers detect
it and project it out.

## 15.4.3 Refused inputs

Each refusal returns a reason rather than an approximate answer
(`unsupported_reason`, `unsupported_reason_unrestricted`,
`unsupported_reason_roks`):

| Input | Why |
|---|---|
| HF reference (no functional) | the module is KS only |
| VV10 | no VV10 kernel in the Hessian |
| an `xc_omega` override | not threaded into the Hessian |
| COSMO, PCM, polarizable embedding, cDFT | their response is not in $H$ |
| fractional occupation, smearing | the density is not a projector |
| an unconverged SCF | $F_{ai} \ne 0$, so the term is not a derivative |
| MOM spin densities (UKS) | not the aufbau projectors of the MOs |

## 15.4.4 Numerical anchors

From `site/src/reference/validation.md` (the MBD@rsSCS gradient rows and the
meta-GGA response row). Each compares the exact gradient with a central
finite difference ($h = 10^{-3}$ unless stated) of the full SCF + MBD
pipeline.

| Reference | Systems | Exact gradient vs FD (Ha/Bohr) | Tests |
|---|---|---|---|
| RKS | H2O / STO-3G PBE; H2O / 6-31G (PBE, PBE0, HSE06, PBE + RI-J); NH3 / 6-31G | $8.0\times10^{-10}$; $\le 3.5\times10^{-9}$; $2.0\times10^{-10}$ | `crates/ferric-rpa/tests/mbd_scf_gradient.rs` |
| UKS | NH2, OH, O2 / 6-31G (PBE, PBE0, HSE06) | $\le 1.9\times10^{-9}$ (exact $J/K$) | `crates/ferric-rpa/tests/mbd_scf_gradient_uks.rs` |
| ROKS | HCO, NH2, CH2, O2 / 6-31G | $\le 2.1\times10^{-11}$ (PBE, PBE0) | `crates/ferric-rpa/tests/mbd_scf_gradient_roks.rs` |
| τ-only meta-GGA (r2SCAN) | water, NH2, HCO, CH2 / STO-3G | $8.0\times10^{-10}$ RKS, $1.1\times10^{-11}$ UKS, $2.0\times10^{-11}$ ROKS | the same three |

The relaxation term alone (of size $10^{-6}$ to $10^{-5}$ Ha/Bohr on these
systems) agrees with the difference of the relaxed and unrelaxed finite
differences to $\le 3.5\times10^{-9}$ (RKS). Run through the UKS or ROKS
path, a closed shell reproduces the RKS gradient to $3.0\times10^{-14}$ and
$2.7\times10^{-14}$.

## 15.4.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-scf/theory.md`, "Kohn-Sham Z-vector for post-SCF corrections" | (15.10) to (15.12), the solve |
| `scripts/ks_zvector_sympy.py`, claims Z0 to Z4 | the proofs |
| module documentation of `crates/ferric-scf/src/zvector_ks.rs` | the scope and refusals |
| section 15.1 | the Lagrangian argument |

**Known limits (2026-11-15).** One production caller (MBD@rsSCS). HF references, VV10, solvation, embedding and
cDFT are refused.
