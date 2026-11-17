# Kohn–Sham Z-vector for post-SCF corrections {#ks-zvector}

<!-- status: ready; registry: ks-zvector;
     theory: crates/ferric-scf/theory.md#"Kohn-Sham Z-vector for post-SCF corrections" -->

A many-body dispersion energy evaluated from a converged Kohn–Sham density
has the same problem as MP2 on Hartree–Fock orbitals. It depends on the
orbitals through the density, but the orbitals were chosen to minimize the
Kohn–Sham energy, not the dispersion energy. When a nucleus moves, the
orbitals relax, the density changes, and the dispersion energy changes with
it at first order. Its gradient needs the same orbital relaxation term as a
correlated method, with the Kohn–Sham orbital Hessian in place of the
Hartree–Fock one.

ferric provides it as one module, `ferric_scf::zvector_ks`, for RKS, UKS and
ROKS references. Its only production caller is the MBD@rsSCS gradient. The
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

## The equations {#ks-zvector-equations}

### The split

The total derivative splits into a fixed-orbital part, which the caller
supplies, and the relaxation:

$$ \frac{dQ}{dR} = \frac{\partial Q}{\partial R}\bigg|_{\text{occupied orbitals fixed}} + \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial R}, \qquad H Z = -4V_{vo}, \quad V_{vo} = \mathbf C_v^{\mathsf T}V\mathbf C_o . \label{eq:ks-zvector-1} $$

"Fixed" means the occupied orbitals follow their basis functions and stay
orthonormal, $\delta\mathbf C_o = -\tfrac12\mathbf C_o S^x_{oo}$, and the
virtual space follows as the orthogonal complement. Along that path the
Brillouin condition is not kept, so the orbitals must also rotate by some
$\kappa^x$, and that rotation is what the relaxation term accounts for.

Where does the factor 4 come from? It is the closed-shell occupation. A
rotation $\kappa$ changes the density $D = 2\mathbf C_o\mathbf C_o^{\mathsf T}$
by $2\mathbf C(\kappa + \kappa^{\mathsf T})\mathbf C^{\mathsf T}$, where
$\kappa + \kappa^{\mathsf T}$ has $\kappa_{ai}$ in both off-diagonal corners.
Contract with the symmetric $V$: $dQ = \operatorname{Tr}[V\,\delta D] = 4\sum\kappa_{ai}V_{ai}$.
The same count gives $dE = 4\sum\kappa_{ai}F_{ai}$. Then the argument of
@sec:lagrangian-zvector-lagrangian, with $\partial Q/\partial\kappa = 4V_{vo}$ and $H$ symmetric, gives
$HZ = -4V_{vo}$.

### The Hessian and $f_{xc}$

The Hessian product applied to a trial $x$ is

$$ (Hx)_{ai} = (F_{vv}x - xF_{oo})_{ai} + \big(\mathbf C_v^{\mathsf T}\,\delta F[\delta D(x)]\,\mathbf C_o\big)_{ai}, \qquad \delta F[\delta D] = \delta J - \tfrac12\big(c_{\mathrm{sr}}\delta K_{\mathrm{sr}} + c_{\mathrm{lr}}\delta K_{\mathrm{lr}}\big) + \delta V_{xc} . $$

The first term is the orbital-energy difference in a non-canonical
representation. The second is the response of the Kohn–Sham Fock matrix to
the density change: the Coulomb response, the exact-exchange response scaled
by the functional's short- and long-range fractions, and the XC response
$\delta V_{xc}$. For a closed shell $\delta V_{xc}$ is the spin kernel
$f_{xc}$ applied to $\delta D_\alpha = \delta D_\beta = \tfrac12\delta D$. The
$f_{xc}$ term is what distinguishes this Hessian from the Hartree–Fock one of
section @sec:lagrangian-zvector. It carries the second derivative of the XC functional, so the
kernel for a GGA includes the $\sigma = |\nabla\rho|^2$ couplings and the
kernel for a τ-only meta-GGA the $\tau$ couplings.

### The contraction without new derivative integrals

The relaxation term needs $\partial F_{ai}/\partial R$, which would normally
need new derivative integrals. The code avoids them. Since
$F_{ai} = \tfrac14\partial E/\partial\kappa_{ai}$, contract with $Z$ and
exchange the order of the two derivatives. The derivative along $Z$ is a
derivative in $t$ at orbitals rotated by $tZ$, and the nuclear derivative of
$E$ along the fixed path is the ordinary SCF gradient expression $g(D, W)$:

$$ \sum_{ai}Z_{ai}\frac{\partial F_{ai}}{\partial R} = \frac14\,\frac{d}{dt}\,g\big(D(t),\; W + t\,\dot W\big)\Big|_{t=0}, \qquad \delta D_Z = 2(\mathbf C_vZ\mathbf C_o^{\mathsf T} + \mathbf C_oZ^{\mathsf T}\mathbf C_v^{\mathsf T}), \label{eq:ks-zvector-contraction} $$

with $\dot D(0) = \delta D_Z$, $\dot W = \tfrac12(\delta D_ZFD + DF\delta D_Z + D\,\delta F\,D)$ and
$\delta F$ the response Fock matrix of $\delta D_Z$. $\dot W$ is the
derivative of $W = \tfrac12DFD$ by the product rule. $g$ is linear in $W$, so
$W + t\dot W$ is exact to the order needed.

The code takes a central difference in $t$ of the SCF gradient routine
itself, with $D(t)$ built from occupied orbitals rotated by $tZ$ and
re-orthonormalized:

$$ D(t) = 2\,\mathbf C_o(t)\mathbf C_o(t)^{\mathsf T}, \qquad \mathbf C_o(t) = (\mathbf C_o + t\,\mathbf C_vZ)\,(\mathbf 1 + t^2Z^{\mathsf T}Z)^{-1/2} . $$

That is an exact projector, positive semidefinite, which the fitted-exchange
gradient needs; $D \pm t\,\delta D_Z$ is not. The projector has an odd $t^3$
term, $-t^3(\mathbf C_vZZ^{\mathsf T}Z\mathbf C_o^{\mathsf T} + \text{h.c.})$, from
the $(\mathbf 1 + t^2Z^{\mathsf T}Z)^{-1/2}$ factor, and its even $t^2$ term
enters the bilinear two-electron part. So the central difference carries an
$O(t^2)$ error in every part of $g$, not only the XC part. At the chosen
step, $t\max|\delta D_Z| = 10^{-4}$ (`T_SCALE`), it is about $10^{-8}$
relative.

The contraction uses the same routine, $J/K$ route and XC grid response as
the SCF gradient, so it differentiates exactly the energy the SCF converged.

<!-- excerpt: ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed -->
```rust
let rhs = -4.0 * cv.t().dot(v_ao).dot(&co);
```

| Symbol | Code |
|---|---|
| $-4V_{vo}$ | `rhs` |
| $V$ | `v_ao` |
| $\mathbf C_v, \mathbf C_o$ | `cv`, `co` |

After the solve, the contraction is three lines of the same function:

<!-- excerpt: ferric-scf/src/zvector_ks.rs::relaxation_gradient_closed -->
```rust
let gp = g_at(1.0)?;
let gm = g_at(-1.0)?;
0.25 * (&gp - &gm) / (2.0 * t)
```

| Code | Equation |
|---|---|
| `g_at(±1.0)` | $g\big(D(\pm t), W \pm t\dot W\big)$ from `ks_gradient_closed_for_density` |
| `0.25 * (&gp - &gm) / (2.0 * t)` | $\tfrac14$ times the central difference of (@eq:ks-zvector-contraction) |

> **Verified equation @eq:ks-zvector-1** · the Z-vector relaxation term
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

The limit in the box's last sentence is the exactness anchor: with
$Q = E_{\mathrm{SCF}}$, $V_{vo} = F_{vo} = 0$ at convergence, so $Z = 0$ and
the caller's part alone is the SCF gradient.

> **Verified equation @eq:ks-zvector-contraction** · the contraction by a central difference
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

### Open shells

The occupation factor changes with the reference. UKS has unit occupations,
so a rotation in spin $\sigma$ changes $D_\sigma$ by
$\mathbf C_{\sigma v}x^\sigma\mathbf C_{\sigma o}^{\mathsf T} + \text{h.c.}$ and
$dE = 2\sum_\sigma\kappa^\sigma\cdot F^\sigma_{vo}$. The right-hand side is
$-2V^\sigma_{vo}$ per spin and the contraction factor is $\tfrac12$. The two
spins couple through $\delta J$ of the total density change; $\delta K$ and
$f_{xc}$ are spin-resolved. For $\alpha = \beta$ each spin block is half the
closed-shell $Z$ and the relaxation is the RKS one.

ROKS has one set of orbitals split into closed, open and virtual blocks.
Three rotation blocks change the densities: $\kappa_{vc}$ moves both spins,
$\kappa_{vo}$ only $\alpha$, $\kappa_{oc}$ only $\beta$. Since $\kappa_{vc}$
moves both spins of $D_{\mathrm{total}}$, its block of the right-hand side
carries the closed-shell factor 4 and the other two carry 2. The fixed path
re-orthonormalizes closed and open orbitals as one set, which gives the
caller's part a closed–open cross term,
$\delta D^{\mathrm{fix}}_\beta = -P_cS^xP_c - \tfrac12(P_oS^xP_c + P_cS^xP_o)$.
That cross term is part of the derivative because $V_{co} \ne 0$.

> **Verified equations @eq:ks-zvector-1 and @eq:ks-zvector-contraction, open shell** · the UKS and ROKS blocks
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

## The solver {#ks-zvector-solver}

Preconditioned conjugate gradient with the orbital-energy gap as
preconditioner, started from $\mathrm{rhs}/\mathrm{gap}$, converged when
$\max|r| \le 10^{-10}\max|\mathrm{rhs}|$, with an error after 200
iterations. A non-positive gap is refused: the reference is then not an
aufbau minimum. When the curvature along a search direction is numerically
zero, the solve is accepted only if the residual is already below
$10^{-8}\max|\mathrm{rhs}|$, and is an error otherwise.

The Hessian product uses exact four-center $J$ and $K$ even when the SCF
fitted them, with the XC kernel $f_{xc}$ for LDA, GGA, hybrid,
range-separated hybrid and τ-only meta-GGA functionals. The contraction
(@eq:ks-zvector-contraction) uses the SCF's own route (RI-J, fitted or COSX exchange). The
response operator is linear, so it is applied to $\delta D/\max|\delta D|$
and rescaled: the $J/K$ builders screen against an absolute threshold, and
the trial densities are small.

| Reference | Rotations | Right-hand side | Contraction factor |
|---|---|---|---|
| RKS | $vo$ | $-4V_{vo}$ | $\tfrac14$ |
| UKS | $vo$ per spin, coupled through $\delta J$ of the total density | $-2V_{vo}$ per spin | $\tfrac12$ |
| ROKS | $vc$, $vo$, $oc$ | $-(4V_{vc}, 2V_{vo}, 2V_{oc})$ | $\tfrac12$ |

For a linear ²Π radical the Hessian has a null mode: rotation about the
molecular axis mixes the degenerate orbitals without changing the energy.
The open-shell solvers detect it and project it out. The closed-shell solve
has no such projection, so a closed-shell reference with a near-null mode
ends in the curvature error.

## Refused inputs {#ks-zvector-refused-inputs}

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

## Numerical anchors {#ks-zvector-numerical-anchors}

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
differences to $\le 3.5\times10^{-9}$ (RKS). The relaxation is two to
three orders of magnitude larger than that agreement, so the tests resolve
it. Run through the UKS or ROKS path, a closed shell reproduces
the RKS gradient to $3.0\times10^{-14}$ and $2.7\times10^{-14}$.

## Sources {#ks-zvector-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-scf/theory.md`, "Kohn-Sham Z-vector for post-SCF corrections" | (@eq:ks-zvector-1) to (@eq:rpa-grad-densities), the solve |
| `scripts/ks_zvector_sympy.py`, claims Z0 to Z4 | the proofs |
| module documentation of `crates/ferric-scf/src/zvector_ks.rs` | the scope and refusals |
| section @sec:lagrangian-zvector | the Lagrangian argument |

**Known limits (2026-11-15).** One production caller (MBD@rsSCS). HF references, VV10, solvation, embedding and
cDFT are refused.
