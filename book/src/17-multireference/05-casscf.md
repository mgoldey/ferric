# Two-step CASSCF {#casscf}

<!-- status: ready; registry: casscf;
     theory: crates/ferric-ci/theory.md#"The two-step method", "Second-order check" -->

A CAS-CI on RHF orbitals uses orbitals shaped for one determinant. The RHF
virtuals describe an added electron in the field of $N$ others, not the
antibonding partner a stretched bond needs, so the active space can hold the
right configurations in the wrong orbitals. CASSCF minimizes the CAS-CI
energy over the orbitals as well as the CI vector, so the core, active and virtual orbitals become the ones the correlated
wavefunction prefers. ferric does it in two steps per macro iteration: a
CAS-CI in the current orbitals (sections @sec:determinants–@sec:rdms), then an orbital step
$C \leftarrow C\exp(K)$ from the gradient at those fixed RDMs.

The orbital gradient is a generalized Fock matrix built from $\gamma$,
$\Gamma$ and the core and active Fock matrices. The step is an
augmented-Hessian Newton step on the exact fixed-RDM orbital Hessian
(default) or an inverse BFGS step. The section builds, in order: the energy
at fixed orbitals, the gradient, the two steps, and the second-order check
that runs after convergence.

| Symbol | Meaning | Code name |
|---|---|---|
| $F^I$, $F^A$ | inactive and active Fock matrices | `active_fock` |
| $F_{pq}$ | generalized Fock matrix | `CasScfProblem::generalized_fock` |
| $\kappa_{pq}$, $K$ | rotation parameters, $K_{pq} = \kappa_{pq} = -K_{qp}$ for $p > q$ | `rotate`, `expm_antisymmetric` |
| $g$ | $\partial E/\partial\kappa = 2(F_{pq} - F_{qp})$ over core-active, core-virtual and active-virtual pairs | `CasScfPoint` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/casscf.rs::{run_casscf,optimize,CasScfProblem}`, `ferric-ci/src/casscf_newton.rs::{orbital_hessian,ah_davidson}`, `ferric-ci/src/casscf_stability.rs::stability` |
| CLI | `method.kind = "casscf"` with `[cas]`, `[cas.ci]` (`examples/water-casscf.toml`, `examples/o2-casscf-triplet.toml`) |
| Python | `run_casscf` |

## Energy, gradient and step {#casscf-energy-gradient-step}

The orbitals are $C = [C_c\,|\,C_a\,|\,C_v]$: $n_c$ doubly occupied core
orbitals $i,j$, $n_a$ active $t,u,v,w$, virtual $a$, and $p,q,r,s$ over
all MOs. The orbitals are real, there is a single root, and the RDMs are
the spin-summed ones of section @sec:rdms. The reference implementation is
`scripts/prototypes/casscf_proto.py` with
`scripts/prototypes/casscf_derivation.md`.

### The energy at fixed orbitals

The core electrons fill $C_c$ twice, so their density is
$D^c = 2C_cC_c^\mathsf{T}$. The active electrons have the density
$D^a = C_a\gamma C_a^\mathsf{T}$. Each density gives a Fock-like field:

$$
F^I = h + J[D^c] - \tfrac12 K[D^c],\qquad
F^A = J[D^a] - \tfrac12 K[D^a],\qquad
E_\text{core} = E_\text{nuc} + \tfrac12\operatorname{Tr}[D^c(h + F^I)].
$$

$F^I$ is what an electron feels from the nuclei and the core electrons, and
$E_\text{core}$ is the closed-shell Hartree–Fock energy expression of
chapter @sec:hartree-fock applied to the core alone. The CAS-CI runs on
$h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ and $E_\text{core}$
(`run_cas_ci_on_integrals`), the folding of section @sec:determinants-hamiltonian. With no
core, $F^I = h$ and $E_\text{core} = E_\text{nuc}$. In
`ferric-ci/src/casscf.rs::inactive`, with the $n_c = 0$ branch and the
factored $J/K$ call elided:

```rust
let dc = 2.0 * cc.dot(&cc.t());                                     // D^c = 2 C_c C_cᵀ
// ...
let fi = &self.h_ao + &j - &(0.5 * &k);                             // F^I = h + J − ½K
let e_core = self.e_nuc + 0.5 * (&dc * &(&self.h_ao + &fi)).sum();  // E_nuc + ½ Tr[D^c (h + F^I)]
```

The element-wise product summed is the trace because both factors are
symmetric.

### Where the integrals come from

`CasScfConfig::integrals` (`casscf_ints.rs`) selects the route.

- **Auto** (the default) takes the dense route when its $n_\text{bas}^4$
  block and transients fit in half of the memory budget, and the direct
  route otherwise.
- **Direct** builds $J$ and $K$ with ferric-scf's direct builder
  (`build_jk_with_pool`, Schwarz and density screened), and $(pu|vw)$ for
  all $p$ by a direct half transform. Per canonical bra shell pair, every
  ket shell pair with $Q_\text{bra}Q_\text{ket} \ge$ `thresh` is contracted
  with $C_a$ on both ket indices,
  $X_{\mu\nu,vw} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)C_{\lambda v}C_{\sigma w}$,
  and then $(pu|vw) = \sum_{\mu\nu}C_{\mu p}C_{\nu u}X_{\mu\nu,vw}$ by
  GEMMs. Peak memory is $X$, $n_\text{bas}^2n_a^2$ doubles.
- **Dense** holds the $n_\text{bas}^4$ AO block, contracts $J$ and $K$
  from it and transforms it by four quarter transforms
  (`integrals::transform_eri_mixed`). It is the test oracle:
  in `tests/casscf_direct.rs`, at `thresh = 0` the routes agree to
  $10^{-12}$ in $F$, $(pu|vw)$, $E$ and $g$.
- **RI** (`integrals = "ri"`, a bundled `auxbasis`, `src/ri.rs`) is a
  different Hamiltonian, the density-fitted one:
  $(\mu\nu|\lambda\sigma) \approx \sum_P B^P_{\mu\nu}B^P_{\lambda\sigma}$
  with $B^P_{\mu\nu} = \sum_Q M_{PQ}(Q|\mu\nu)$, $M = L^{-1}$ and $L$ the
  Cholesky factor of the Coulomb metric $V_{PQ} = (P|Q)$, so
  $M^\mathsf{T}M = V^{-1}$ (the construction of section @sec:closed-shell-ri-mp2-density-fitting-making-affordable). It holds
  $B$, $N_\text{aux}n_\text{bas}^2$ doubles, and nothing of size
  $n_\text{bas}^4$.

On the RI route a density given in factored form $D = XWX^\mathsf{T}$ (the
core density $2C_cC_c^\mathsf{T}$ and the active density
$C_a\gamma C_a^\mathsf{T}$) gives $J = B^\mathsf{T}(B\,\mathrm{vec}D)$ and
$K = \sum_P (B_PX)W(B_PX)^\mathsf{T}$ as one GEMM over $(P, j)$. A general
$D$ (the AH Hessian's trial densities) is summed over aux blocks,
$K = \sum_P B_PDB_P$. The MO blocks are half transforms
$(P|pq) = C_l^\mathsf{T}B_PC_r$ followed by one GEMM over $P$:
$(pu|vw) = (P|pu)^\mathsf{T}(P|vw)$, and the AH occupied kernel's
$(pq|vw)$ and $(pu|qv)$ the same way, so the AH optimizer runs unchanged.
SC-NEVPT2 on an RI CASSCF reference takes its MO blocks and $J/K$ from the
same $B$. The RI route is never chosen by `auto`. It is anchored against the
dense route fed the reconstructed block $B^\mathsf{T}B$
(`tests/casscf_ri.rs`), which must agree to rounding: only the order of the
sums differs.

### The orbital gradient

Rotate the orbitals by $C' = C\exp(K)$, with $K_{pq} = \kappa_{pq} = -K_{qp}$
for $p > q$. Define the generalized Fock matrix in the MO basis, with $F^I$
and $F^A$ transformed by $C$:

$$
F_{pi} = 2(F^I_{pi} + F^A_{pi}),\qquad
F_{pt} = \sum_u F^I_{pu}\gamma_{ut} + \sum_{uvw}(pu|vw)\,\Gamma_{tuvw},\qquad
F_{pa} = 0 .
$$

Column $q$ of $F$ is built from what orbital $q$ holds. A core orbital holds two electrons that feel the full
field of the core and active electrons, so its column is twice
$F^I + F^A$. An active orbital's column weights the one-electron field by
$\gamma$ and the two-electron interaction by $\Gamma$. A virtual orbital is
empty and its column is zero.

The gradient is $\partial E/\partial\kappa_{pq} = 2(F_{pq} - F_{qp})$, so
it vanishes when $F$ is symmetric (the generalized Brillouin theorem). The
nonredundant pairs and their reductions:

- core-active $(t,i)$: $2(F_{ti} - F_{it})$;
- core-virtual $(a,i)$: $F_{ia} = 0$ because $a$ is a virtual column, so
  the gradient is $2F_{ai} = 4(F^I_{ai} + F^A_{ai})$;
- active-virtual $(a,t)$: $F_{ta} = 0$ for the same reason, so the gradient
  is $2F_{at}$.

The code builds $F$ in `ferric-ci/src/casscf.rs::generalized_fock`, with a
comment and the two shape conversions to ndarray views elided:

```rust
for p in 0..nmo {
    for i in 0..nc {
        f[(p, i)] = 2.0 * (fi[(p, i)] + fa[(p, i)]);       // F_pi = 2(F^I + F^A)_pi
    }
}
let one = fi.slice(s![.., nc..nc + na]).dot(gamma);        // Σ_u F^I_pu γ_ut
// ...
let two = p_mat.dot(&g2.t());                              // Σ_uvw (pu|vw) Γ_tuvw
f.slice_mut(s![.., nc..nc + na]).assign(&(&one + &two));   // F_pt
let gmat = 2.0 * (&f - &f.t());                            // 2(F − Fᵀ)
```

Points about it:

- **The virtual columns are never written.** `f` starts at zero, which is
  $F_{pa} = 0$.
- **The two-electron part is one GEMM.** `p_mat` is $(pu|vw)$ as an
  $n_\text{mo}\times n_a^3$ matrix and `g2` is $\Gamma$ as $n_a\times n_a^3$,
  so their product contracts $uvw$ in one call.
- **The whole antisymmetric matrix is formed.** The packed gradient picks
  the nonredundant pairs from it; the active-active block is the redundant
  part, discussed next.

### The redundant active-active block

Rotations inside the active space change nothing at a CI eigenvector,
because CAS-CI already includes every determinant of the active orbitals.
The code needs to know how far from zero that block is when the CI vector
is not exact. For any normalized $c$ with residual $r = Hc - \theta c$ and
the antisymmetric operator $A = E_{pq} - E_{qp}$,
$2(F_{pq} - F_{qp}) = \langle c|[H, A]|c\rangle$. Expand the commutator and
substitute $Hc = \theta c + r$:

$$
c^\mathsf{T}HAc = (\theta c + r)^\mathsf{T}Ac = \theta\,c^\mathsf{T}Ac + r^\mathsf{T}Ac,
\qquad
c^\mathsf{T}AHc = \theta\,c^\mathsf{T}Ac + c^\mathsf{T}Ar .
$$

$c^\mathsf{T}Ac = 0$ because $A$ is antisymmetric, and
$r^\mathsf{T}Ac = c^\mathsf{T}A^\mathsf{T}r = -c^\mathsf{T}Ar$. Subtract:

$$
2(F_{pq} - F_{qp}) = \langle c|[H, E_{pq} - E_{qp}]|c\rangle = -2\,c^\mathsf{T}(E_{pq}-E_{qp})\,r .
$$

So a redundant element is bounded by $8\lVert r\rVert$, using
$\lVert E_{pq}\rVert \le 2$ for $p\ne q$. It is zero at an exact root and
first order in the CI residual otherwise.
`scripts/ci_rdm_casscf_gradient_sympy.py` checks both statements on a
six-orbital model with random N-representable RDMs: `casscf-gradient`, the
central difference of $E(C\exp K)$ at fixed RDMs against $2(F_{pq}-F_{qp})$
for every pair and the two non-redundant reductions; `redundant-residual`,
the active-active identity.

> **Verified equation {#eq:ci-casscf-gradient}** · the orbital gradient is $2(F - F^\mathsf{T})$
>
> On a six-orbital model (one core, three active, two virtual orbitals) with
> random N-representable RDMs from a random CI vector, the central
> difference of the CASSCF energy at fixed RDMs along every rotation
> $p > q$ of $C\exp K$ equals $2(F_{pq} - F_{qp})$ with
> $F_{pi} = 2(F^I + F^A)_{pi}$,
> $F_{pt} = \sum_u F^I_{pu}\gamma_{ut} + \sum_{uvw}(pu|vw)\Gamma_{tuvw}$ and
> $F_{pa} = 0$; the core-virtual and active-virtual blocks reduce to
> $4(F^I + F^A)_{ai}$ and $2F_{at}$. For active $t, u$,
> $2(F_{tu} - F_{ut}) = -2c^\mathsf{T}(E_{tu} - E_{ut})r$ with
> $r = Hc - \langle c|H|c\rangle c$, so the active-active gradient vanishes
> at a CI eigenvector.
>
> | | |
> |---|---|
> | Proof | `scripts/ci_rdm_casscf_gradient_sympy.py`, checks `casscf-gradient`, `redundant-residual` |
> | Negative control | the runner's mutations: the factor 2 of $F_{pi}$ set to 1; the exchange of $F^A$ dropped. A core factor of 1 is also a printed negative control |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/casscf.rs::{generalized_fock,active_fock,gradient_at_fixed_rdms}` |
> | Code test | `orbital_gradient_matches_fd_h2o_cas44_631g` in `crates/ferric-ci/tests/casscf.rs` |
>
> <!-- verified: id=ci.casscf.gradient; script=scripts/ci_rdm_casscf_gradient_sympy.py; claim=casscf-gradient; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/casscf.rs::{generalized_fock,active_fock,gradient_at_fixed_rdms}; rust_test=ferric-ci/tests/casscf.rs::orbital_gradient_matches_fd_h2o_cas44_631g; registry=casscf -->

The box is at fixed RDMs, which is what the formula claims: at a converged
CI root the RDM response does not enter the gradient, because the CI energy
is stationary in $c$. The Rust test checks the analytic gradient against a
finite difference of the CI-relaxed energy per rotation class. The
augmented-Hessian products and the step control are not covered by a proof
script; their anchors are the PySCF comparisons below.

### The BFGS step

`optimizer = "bfgs"` takes inverse BFGS steps in the current orbital frame,
with no transport of earlier steps. The initial and reset inverse Hessian is
$1/\max(|H_\text{diag}|, 0.05)$, from the approximate diagonal of Chaban,
Schmidt and Gordon, Theor. Chem. Acc. 97, 88 (1997), with
$F^T = F^I + F^A$:

| Pair | $H_\text{diag}$ |
|---|---|
| $(a,i)$ | $4(F^T_{aa} - F^T_{ii})$ |
| $(a,t)$ | $2\gamma_{tt}F^T_{aa} - 2F_{tt}$ |
| $(t,i)$ | $4(F^T_{tt} - F^T_{ii}) + 2\gamma_{tt}F^T_{ii} - 2F_{tt}$ |

Its median relative error against the finite-difference diagonal is 24-28 %
(prototype measurement), so it serves as a preconditioner and not as a
Hessian. The update is skipped when $s^\mathsf{T}y \le 10^{-14}$. The
inverse resets to the diagonal when the energy rises by more than
$10^{-10}$ Ha. $\max_k|s_k|$ is capped at 0.2.

$\exp(K)$ (`expm_antisymmetric`) is a Taylor series after scaling $K$ to
$\infty$-norm $\le \tfrac12$, followed by squaring: $\exp(K) =
(\exp(K/2^m))^{2^m}$. For a real antisymmetric $K$ the result is orthogonal
to rounding, so the rotated orbitals stay orthonormal.

Convergence: $\lVert g\rVert_2 < 10^{-7}$ and $|\Delta E| < 10^{-12}$ Ha at
the same iteration (defaults). Every CI solve must reach the CI residual
threshold ($10^{-10}$ by default). The gradient error is first order in that
residual (the bound above), so a solve that misses it is an error, not a
warning.

### The augmented-Hessian step

`optimizer = "ah"` (`src/casscf_newton.rs`, the default) uses the exact
orbital Hessian at fixed RDMs, applied as a product. Chapter @sec:second-order describes
the augmented-Hessian and trust-region machinery for SCF; here are the
CASSCF ingredients.

**Frame derivatives.** For $C(\varepsilon) = C\exp(\varepsilon X)$, every
quantity above in the rotated orbitals has the frame derivative

$$
\begin{aligned}
dF^I &= F^IX - XF^I + C^\mathsf{T}(J - \tfrac12K)[dD^c]C, &
dD^c &= 2(CX)_cC_c^\mathsf{T} + \text{transpose},\\
dF^A &= F^AX - XF^A + C^\mathsf{T}(J - \tfrac12K)[dD^a]C, &
dD^a &= (CX)_a\gamma C_a^\mathsf{T} + \text{transpose},
\end{aligned}
$$

and $d(pu|vw) = \sum_r X_{rp}(ru|vw) + X_{ru}(pr|vw) + X_{rv}(pu|rw) +
X_{rw}(pu|rv)$. The first two terms of $dF$ rotate the matrix with the
frame; the $J/K$ term is the response of the field to the changed density.
$H_fx = \operatorname{pack}2(dF - dF^\mathsf{T})$ is the derivative of the
fixed-RDM gradient along $x$ in the rotated frame.

**The $\Gamma$ contraction.** The $(pr|vw)$ and $(pu|rw)$ terms are
contracted with $\Gamma$ once per macro iteration into

$$
W_{(p,t),(r,s)} = \sum_{vw}(pr|vw)\Gamma_{tsvw}
 + \sum_{uw}(pu|rw)(\Gamma_{tusw} + \Gamma_{tuws}),
$$

the first sum being the active-active coupling through $(pr|vw)$.

**The symmetric Hessian.** Rotations do not commute:
$e^{\varepsilon X}e^{\delta A} = \exp(\varepsilon X + \delta A +
\tfrac12\varepsilon\delta[X, A] + \dots)$. The frame derivative therefore
differs from the Hessian of $E(C\exp K)$ at $K = 0$ by a commutator with
the gradient. The symmetric Hessian is
$Hx = H_fx - \tfrac12\operatorname{pack}[G, X]$ with
$G = 2(F - F^\mathsf{T})$.

**The occupied kernel.** Both density responses have the form
$dD = C(Y + Y^\mathsf{T})C^\mathsf{T}$ with $Y$ nonzero only in occupied
columns ($Y_{ri} = 2X_{ri}$ for core $i$, $Y_{rt} = (X_a\gamma)_{rt}$ for
active $t$). So in the MO basis
$C^\mathsf{T}(J - \tfrac12K)[dD]C_\text{occ} = L\,\mathrm{vec}(Y)$ with
the occupied kernel

$$
L_{(pj),(rk)} = 2(pj|rk) - \tfrac12(pr|jk) - \tfrac12(pk|rj),
$$

$j, k$ over the core and active orbitals. $(pq|jk)$ and $(pj|qk)$ are built
once per macro iteration. On the dense route one first quarter transform is
shared by both chains, the $(pj|qk)$ chain contracting $\nu$ with
$C_\text{occ}$ second through the $(\mu\nu|\lambda\sigma) = (\mu\nu|\sigma\lambda)$
symmetry of the stored block (`transform_ppaa_papa`). On the direct route
it is one half transform per bra shell pair, all ket pairs. Their active
blocks give $W$ and $\sum(pu|vw)\Gamma$. A Hessian product is then one GEMM
of $L$ ($(n_\text{mo}n_\text{occ})^2$) with two columns. The AO form, one
$J/K$ pass over $dD^c$ and $dD^a$ per product (`HessianProducts::AoJk`), is
kept as its oracle.

**The step.** The step is the lowest eigenpair of
$\begin{pmatrix}0 & g^\mathsf{T}\\ g & H\end{pmatrix}$, found by Davidson
with the preconditioner $\max(|H_\text{diag}|, \texttt{hdiag\_floor}) -
\lambda$ and at most 40 products. Then $s = v/v_0$, stopped at
$\lVert(H - \lambda)s + g\rVert \le \eta\lVert g\rVert$ with
$\eta = \min(0.1, \lVert g\rVert)$ by default (`CasScfConfig::ah_forcing`;
`AhForcing::Constant` holds $\eta$ fixed). The step is then scaled to a
trust radius (2-norm; start 0.4, at most 1.0) and to `max_step`
($\infty$-norm).

**Step control.** A step that raises $E$ by more than $10^{-10}$ Ha is
halved and evaluated again, a CI solve each time. The radius halves below an
actual/predicted ratio of 0.25 and doubles above 0.75 for a step at the
boundary. The predicted change is $g\cdot s + \tfrac12 s\cdot Hs/\beta$,
with $\beta$ from the next paragraph.

**Curvature stretch.** The Hessian lacks the CI response, which lowers the
curvature (section @sec:casscf-second-order-check shows why it can only lower it). Near
convergence the plain AH step undershoots by a factor of about 1.6, which
makes convergence linear, and on N2 CAS(10,10) it stalls by orders of
magnitude along one direction. Each accepted step therefore measures the
ratio of the model to the actual curvature along itself,

$$
\beta = \frac{s\cdot Hs}{2(\Delta E - g\cdot s)}
$$

(100 when the actual curvature is not positive), and the next AH step is
multiplied by $\beta$ clamped to $[1, 100]$ before the trust and
`max_step` scaling (`curvature_stretch`). The prototype's `scale` variant
used the energy ratio clamped to $[1, 2]$ instead; in Rust that rule
stalled N2 CAS(10,10). Multi-secant SR1 corrections of $H$ from earlier
steps were tried and were unstable for a memory of three or more steps
(water 62 iterations with 16 rejected steps).

### Open-shell active spaces

$H$ is spin-free, so the energy and the gradient above depend on the
spin-summed $\gamma$ and $\Gamma$ only, for any sector
$(n_\alpha, n_\beta)$. The CI root is the lowest of that $M_s$ sector, and
the orbitals stay one restricted set with a doubly occupied core. Swapping
$n_\alpha$ and $n_\beta$ gives the same energy and RDMs ($M_s \to -M_s$).

### Canonical orbitals

`canonicalize` (`CasScfConfig::canonicalize`) rotates to a standard
orbital set. With $F^T = F^I + F^A$ in the MO basis, $U$ is block diagonal:
the eigenvectors of the core block of $F^T$ (ascending), of $\gamma$
(occupations descending) and of the virtual block of $F^T$ (ascending).
$C' = CU$, and each column is signed by its first AO coefficient within a
relative $10^{-6}$ of the largest magnitude (the first index breaks ties
between symmetry-equivalent atoms). Every rotation in $U$ is redundant, so
$E$ and $\lVert g\rVert$ are unchanged. The CAS-CI is solved again in $C'$,
which carries the CI vector into the new orbitals. $D^c$ is unchanged by
the core rotation and $D^a = C_a\gamma C_a^\mathsf{T}$ by the active
rotation, so $F^I$ and $F^A$ in $C'$ are the rotated matrices up to the CI
error of the second solve. `mo_energy` is the diagonal of $F^T$ in $C'$.

### Limits of the method as built

Lowest root, no state averaging. The direct route recomputes every shell
quartet each macro iteration, in three passes: $J/K[D^c]$, $(pu|vw)$ and
$J/K[D^a]$. The start is a restricted RHF, so the electron count must be
even. There is no orbital-CI coupling: both optimizers are two-step, and
the augmented-Hessian step uses the orbital Hessian at fixed RDMs.

## The macro iteration {#casscf-macro-iteration}

| Step | Quantity | Function |
|---|---|---|
| 1 | $F^I$, $E_\text{core}$, $h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ by the dense, direct or RI route | `CasScfProblem::evaluate` |
| 2 | CAS-CI with RDMs; a root that misses the CI threshold is an error | `run_cas_ci_on_integrals` |
| 3 | $F^A$, the generalized Fock matrix, $g$ | `generalized_fock` |
| 4 | converged if $\lVert g\rVert_2 <$ `grad_tol` and $|\Delta E| <$ `energy_tol` at the same iteration | `optimize` |
| 5 | step $s$: AH Davidson on the fixed-RDM Hessian with trust radius and curvature stretch, or inverse BFGS | `ah_davidson`, `optimize` |
| 6 | $C \leftarrow C\exp(K(s))$ | `rotate`, `expm_antisymmetric` |
| 7 | after convergence, optional canonicalization and second-order check | `canonicalize`, `stability` |

## The second-order check {#casscf-second-order-check}

Both optimizers stop at a point where the gradient vanishes. That can be a
minimum or a saddle. The sign of the lowest eigenvalue of the orbital
Hessian decides, but it has to be the Hessian of the energy with the CI
vector relaxed at every orbital set, because that is the function CASSCF
minimizes. `src/casscf_stability.rs` (`[cas] stability`, 2026-11-14)
computes it after convergence; chapter @sec:stability does the same for SCF.

**The coupled Hessian.** Write the CI vector as
$c(z) = (c+z)/\lVert c+z\rVert$ with $z\perp c$, about the CI root $c$
($Hc = \theta c$). The Hessian of $E(\kappa, z)$ at the converged point has
the blocks

$$
H_{oo}x = \text{the fixed-RDM orbital Hessian (AH step, above)},\qquad
H_{co}x = 2P\,dH(x)\,c,\qquad H_{cc} = 2P(H-\theta)P,\qquad
H_{oc}z = g(\gamma_T, \Gamma_T),
$$

with $P = 1 - cc^\mathsf{T}$ and $dH(x)$ the frame derivative of the
active-space Hamiltonian along the rotation $x$: $dh_{tu} = (dF^I)_{tu}$
and $d(tu|vw) = \sum_r X_{rt}(ru|vw) + X_{ru}(tr|vw) + X_{rv}(tu|rw) +
X_{rw}(tu|rv)$. $\gamma_T$ and $\Gamma_T$ are the symmetrized transition
RDMs of $c$ and $z$ ($\gamma_T = \langle z|E|c\rangle + \langle c|E|z\rangle$,
`spin_summed_transition_rdms`), and $g(\gamma, \Gamma)$ is the part of the
orbital gradient linear in the RDMs:
$F_{pi} = 2F^A[\gamma_T]_{pi}$,
$F_{pt} = \sum_u F^I_{pu}\gamma_{T,ut} + \sum_{uvw}(pu|vw)\Gamma_{T,tuvw}$,
$g = 2(F - F^\mathsf{T})$. The gradient is $\langle c|dH(x)|c\rangle$, so
$x\cdot g(\gamma_T,\Gamma_T) = 2\langle z|dH(x)|c\rangle$ and
$H_{oc} = H_{co}^\mathsf{T}$.

**Relaxing the CI vector.** Minimize the quadratic model
$\tfrac12 x^\mathsf{T}H_{oo}x + z^\mathsf{T}H_{co}x + \tfrac12 z^\mathsf{T}H_{cc}z$
over $z$ for fixed $x$. The stationarity condition is
$H_{cc}z = -H_{co}x$, which after dividing by 2 reads
$(H-\theta)\,z = -P\,dH(x)\,c$ with $z\perp c$. Substitute $z(x)$ back. The
energy with the CI vector relaxed at every orbital set,
$E_\text{rel}(\kappa) = \min_z E(\kappa, z)$, has the Schur complement as
its Hessian:

$$
H_\text{rel}\,x = H_{oo}x + H_{oc}\,z(x),\qquad (H-\theta)\,z = -P\,dH(x)\,c,\quad z\perp c .
$$

Since $H_{oc}z(x) = -H_{oc}H_{cc}^{-1}H_{co}x$ and $H_{cc}$ is positive
semidefinite on the complement of $c$ (because $\theta$ is the lowest
root), the correction is negative semidefinite. The relaxed Hessian is
therefore never stiffer than $H_{oo}$. This is why the CI response matters
for the sign: without it the check reads the fixed-RDM Hessian, which is
stiffer, and can call a saddle a minimum. It is also the curvature the AH
step misses (section @sec:casscf-energy-gradient-step).

**The solve.** Each product is one CI linear solve: conjugate gradients on
the complement of $c$, preconditioner $P(\operatorname{diag}H - \theta)^{-1}P$,
relative residual $10^{-10}$. $H - \theta \ge 0$ on the complement because
$\theta$ is the lowest root, and a non-positive curvature is an error. This
is the Hessian that the finite difference of the analytic CASSCF gradient
measures (the CI re-solved at each displaced orbital set). The lowest
eigenpair comes from the block Davidson of
`ferric_scf::stability::davidson_lowest` (residual $10^{-5}$, noise floor
$10^{-6}$ Ha), preconditioned by `approx_hessian_diag`.

**Verdicts.** `UNSTABLE` if $\lambda < -10^{-6}$, `MARGINAL` if
$|\lambda| \le 10^{-6}$, `INDETERMINATE` if Davidson or a CI solve missed
its threshold, else `STABLE`. `follow` rotates the orbitals by $0.3$ along
the eigenvector (the sign with the lower energy, halving up to 3 times when
neither lowers it), re-optimizes and checks again, at most 5 times; the
energy before the first step is kept on the result.

**The certificate** (`src/certificate.rs`). An `UNSTABLE` verdict is
evidence that the point is not the CASSCF minimum the PySCF anchors
certify, so it caps the grade to not graded and removes the anchor from
scope, like an unconverged run. `MARGINAL` and `INDETERMINATE` add a warning
only. The check's summary is recorded as the route `stability`, and an
SC-NEVPT2 certificate carries its reference's.

The check costs about as much as the CASSCF itself, so it is off by
default.

## Options {#casscf-options}

| Option (`[cas]`; Python) | Values | Default | Effect |
|---|---|---|---|
| `n_active`, `n_elec_active` | integer, $[n_\alpha, n_\beta]$ | required | active space and sector |
| `n_core` | integer | the electrons the active space leaves, halved | doubly occupied core |
| `grad_tol`; `casscf_grad_tol` | | $10^{-7}$ | gradient-norm threshold |
| `energy_tol`; `casscf_energy_tol` | Ha | $10^{-12}$ | energy-change threshold |
| `max_iter`; `casscf_max_iter` | integer | 300 | macro iterations |
| `max_step`; `casscf_max_step` | rad | 0.2 | cap on the largest step element |
| `optimizer`; `casscf_optimizer` | `ah`, `bfgs` | `ah` | orbital step |
| `integrals`; `casscf_integrals` | `auto`, `direct`, `dense`, `ri` | `auto` | integral route; `ri` needs `auxbasis` and is a different Hamiltonian |
| `integral_thresh`; `casscf_integral_thresh` | | $10^{-12}$ | Schwarz threshold of `direct` |
| `stability`; `casscf_stability` | `off`, `check`, `follow` | `off` | second-order check |
| `canonicalize` | bool | false | canonical core and virtual, natural active orbitals |
| `[cas.ci] conv_thresh`; `ci_conv_thresh` | residual | $10^{-10}$ | CI root acceptance |
| `[cas.ci] max_subspace`, `max_iter` | integer | 50, 200 | CI Davidson |

## Numerical anchors {#casscf-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| $E$ vs PySCF `mcscf.CASSCF` on ferric's AO integrals | H2O CAS(4,4) / 6-31G | $\le 1.5\times10^{-11}$ Ha | `crates/ferric-ci/tests/casscf_vs_pyscf.rs` |
| the same | N2 CAS(6,6) at 1.0977 and 1.6 Å, O2 CAS(8,6) $(5,3)$ / 6-31G | $\le 5.8\times10^{-13}$ Ha | `crates/ferric-ci/tests/casscf_vs_pyscf.rs` |
| full space: $E$ vs `run_cas_ci` | H2O ($M_s$ = 0, 1), N2, O2 ($M_s$ = 1) / STO-3G | $\le 10^{-13}$ Ha | `crates/ferric-ci/tests/casscf.rs` |
| full space: $2(F - F^\mathsf{T})$ | same | $\le 7.7\times10^{-11}$ | `crates/ferric-ci/tests/casscf.rs` |
| analytic vs finite-difference gradient | | $\le 4.6\times10^{-10}$ | `crates/ferric-ci/tests/casscf.rs` |
| canonicalization: change of $E$ | | $\le 5.7\times10^{-14}$ Ha | `crates/ferric-ci/tests/casscf_canonical.rs` |
| RI route vs PySCF `density_fit()` CASSCF | the three 6-31G systems, def2-universal-jkfit | $\le 1.2\times10^{-12}$ Ha | `crates/ferric-ci/tests/casscf_ri.rs` |

When every orbital is active there is no core, no virtual space and no
nonredundant rotation, so CASSCF is full CI. The two full-space rows check
that limit: the energy equals `run_cas_ci`, and $2(F - F^\mathsf{T})$ is
zero to the CI residual, as the redundant-block identity of @sec:casscf-energy-gradient-step
requires.

## What is proven and what is not {#casscf-proven}

The anchors table above summarizes the records below. They are grouped by
what they test: the method on the dense route, the integral routes, the
optimizers, the stationary points they reach, and the second-order check.

### Anchors on the dense route {#casscf-proven-dense-anchors}

`crates/ferric-ci/tests/casscf.rs` (measured 2026-11-11) uses
`testdata/molecules/water.xyz` and `n2.xyz`, 6-31G, a ferric exact-J/K RHF
start, and a Davidson threshold of $10^{-10}$ ($10^{-11}$ for the
finite-difference anchor).

**Anchor 1: active space = full space.** In STO-3G, CASSCF with every
orbital active is compared with `run_cas_ci` over the full window in the
same $(n_\alpha, n_\beta)$ sector, and the RDMs of the sector rebuild the
energy (the bar of `tests/casci_rdms.rs`). The lowest root of the H2O
$M_s = 1$ sector is a triplet above the singlet ground state, so a CI run
in another sector would miss `run_cas_ci` by the gap.

| System | Sector | $N_\text{det}$ | $\Delta E$ (Ha) | max $\lvert 2(F - F^\mathsf{T})\rvert$ | bar $8\lVert r\rVert$ + rounding |
|---|---|---:|---:|---:|---:|
| H2O | (5, 5) | 441 | 9.9e-14 | 7.6e-11 | 1.9e-9 |
| H2O | (6, 4) | 245 | 2.8e-14 | 7.7e-11 | 1.2e-9 |
| N2 | (7, 7) | 14400 | 9.9e-14 | 6.2e-12 | 4.5e-10 |
| O2 | (9, 7) | 1200 | -2.8e-14 | 1.8e-11 | 3.6e-10 |

**Anchor 2: the analytic gradient.** At RHF orbitals rotated by a random
$\kappa$ (scale 0.05), the analytic gradient is compared with a
Richardson-extrapolated 5-point central difference ($h = 10^{-3}$ and
$h/2$), as the maximum over each rotation class, with bar $10^{-8}$:

| System | Row | core-active | core-virtual | active-virtual |
|---|---|---:|---:|---:|
| H2O CAS(4,4) | max $\lvert g\rvert$ | 6.28 | 5.58 | 0.834 |
| | FD($h$) − FD($h/2$) | 2.0e-7 | 1.7e-10 | 1.2e-10 |
| | analytic − FD | 2.4e-10 | 1.2e-10 | 1.5e-10 |
| N2 CAS(6,6) | max $\lvert g\rvert$ | 2.07 | 6.98 | 0.800 |
| | FD($h$) − FD($h/2$) | 1.4e-9 | 2.0e-10 | 1.9e-10 |
| | analytic − FD | 1.8e-10 | 2.0e-10 | 1.6e-10 |
| H2O CAS(4,4), (3, 1) | max $\lvert g\rvert$ | 4.41 | 5.62 | 0.942 |
| | FD($h$) − FD($h/2$) | 3.7e-8 | 2.5e-10 | 1.3e-10 |
| | analytic − FD | 2.6e-10 | 1.8e-10 | 1.1e-10 |
| O2 CAS(8,6), (5, 3) | max $\lvert g\rvert$ | 2.47 | 9.13 | 0.991 |
| | FD($h$) − FD($h/2$) | 2.5e-9 | 4.1e-10 | 5.3e-10 |
| | analytic − FD | 4.2e-10 | 4.6e-10 | 4.2e-10 |

The finite difference runs in the same CI sector as the analytic gradient,
so it cannot see a CI run in the wrong sector. Anchor 1 and the PySCF
comparison can.

**Anchor 3: the prototype.** The converged energies are compared with the
committed values of `scripts/prototypes/casscf_proto.py`, with bar
$(\lVert g\rVert^2 + \lVert g_\text{proto}\rVert^2)/(2\lambda_\text{min}) + 8\varepsilon|E|\sqrt{N_\text{det}}$,
where $\lambda_\text{min}$ is the lowest eigenvalue of the
finite-difference CI-relaxed orbital Hessian at the converged orbitals. The
first term is the energy error of two points near the same minimum: at
distance $\delta$ from the minimum, $E - E_\text{min} \approx
\tfrac12\lambda\delta^2$ and $\lVert g\rVert \approx \lambda\delta$, so
$E - E_\text{min} \approx \lVert g\rVert^2/(2\lambda)$.

| Quantity | H2O CAS(4,4)/6-31G | N2 CAS(6,6)/6-31G |
|---|---:|---:|
| $E$ (Ha) | -76.0369690965018 | -109.0155467897005 |
| $E$ prototype (Ha) | -76.0369690965018 | -109.0155467897007 |
| $\Delta E$ (Ha) | -4.3e-14 | 2.3e-13 |
| bar (Ha) | 1.1e-12 | 3.9e-12 |
| $\lVert g\rVert$ / prototype | 4.2e-8 / 9.5e-8 | 3.4e-8 / 3.4e-8 |
| $\lambda_\text{min}$ (Ha) / prototype note | 0.0195 / 0.0195 | 0.0402 / 0.0402 |
| macro iterations / prototype | 60 / 62 | 21 / 21 |

The prototype's energies match `pyscf.mcscf.CASSCF` on ferric's AO
integrals to $1.4\times10^{-12}$ Ha (H2O) and $1.8\times10^{-13}$ Ha (N2).

**Anchor 4: PySCF.** `crates/ferric-ci/tests/casscf_vs_pyscf.rs` compares
the converged energies with `pyscf.mcscf.CASSCF` run on ferric's AO
integrals from PySCF's own RHF orbitals. The references come from
`scripts/gen_casscf_refs.py` (PySCF 2.13.1, `conv_tol` $10^{-12}$,
`conv_tol_grad` $10^{-6}$, FCI `conv_tol` $10^{-14}$) and live in
`testdata/reference/casscf_*_pyscf.json`. ferric runs from its own RHF with
`canonicalize`. The bar is
$(\lVert g\rVert^2 + \lVert g_\text{ref}\rVert^2)/(2\lambda_\text{min})$ +
`conv_tol` + rounding, with $\lVert g_\text{ref}\rVert$ twice PySCF's packed
norm. Natural occupations and the sorted eigenvalues of the core and
virtual blocks of $F^I + F^A$ are compared but not gated: they move to
first order in the orbital error, which PySCF's gradient sets.

| System | $E$ (Ha) | $\Delta E$ vs PySCF (Ha) | bar (Ha) | $\lVert g\rVert$ / $2\lVert g_\text{PySCF}\rVert$ | $\lambda_\text{min}$ | iterations | natural occupations | max $\Delta$ occupation / core / virtual $\varepsilon$ |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| H2O CAS(4,4) | -76.0369690965018 | -1.5e-11 | 1.0e-10 | 4.2e-8 / 2.0e-6 | 0.0195 | 60 | 1.978, 1.975, 0.024, 0.023 | 2.2e-7 / 1.6e-5 / 1.6e-6 |
| N2 CAS(6,6), 1.0977 Å | -109.0155467897006 | -9.9e-14 | 5.2e-12 | 3.4e-8 / 1.5e-7 | 0.0402 | 21 | 1.981, 1.935 (2), 0.065 (2), 0.019 | 8.9e-8 / 8.7e-7 / 1.9e-7 |
| N2 CAS(6,6), 1.6 Å | -108.8482292478625 | -5.8e-13 | 5.1e-12 | 7.8e-8 / 2.5e-7 | 0.1485 | 12 | 1.897, 1.723 (2), 0.277 (2), 0.102 | 7.0e-7 / 2.1e-7 / 6.1e-8 |
| O2 CAS(8,6), (5, 3) | -149.6367298537082 | -2.6e-13 | 4.0e-12 | 5.9e-8 / 1.3e-7 | 0.0958 | 14 | 1.960 (2), 1.955, 1.039 (2), 0.046 | 5.5e-8 / 1.9e-7 / 9.3e-8 |

All four reach the same minimum as PySCF. The test also asserts that the
point is a minimum: the lowest eigenvalue of the central-difference
CI-relaxed orbital Hessian at $h = 10^{-4}$ must exceed
$\tfrac23\lvert\lambda(2h) - \lambda(h)\rvert + n_\text{pair}\,\delta g/h$,
with $\delta g = 8r + 4\varepsilon n_\text{mo}^2 \max\lvert F\rvert$ and $r$
the CI convergence threshold. The measured $\lambda(h)$ and $\lambda(2h)$
agree to six digits: 0.019457 (bar $4.4\times10^{-4}$), 0.040210
($8.4\times10^{-4}$), 0.148488 ($8.4\times10^{-4}$) and 0.095834
($8.4\times10^{-4}$) for the four rows in order, so no anchor is a saddle.
PySCF's O2 state has $\langle S^2\rangle = 2$. PySCF's gradient sets the
H2O bar: `conv_tol_grad` $10^{-7}$ does not converge H2O in 300 macro
iterations.

**Canonicalization.** `crates/ferric-ci/tests/casscf_canonical.rs` starts
from ferric's converged orbitals $C_0$, forms the canonical orbitals $C_1$
and solves the CI again in $C_1$. With
$\delta\gamma = \gamma_1 - U_a^\mathsf{T}\gamma_0U_a$, and the bars derived
in the test's module documentation:

| Quantity | H2O CAS(4,4) | N2 CAS(6,6), 1.6 Å | O2 CAS(8,6), (5, 3) |
|---|---:|---:|---:|
| $\lvert E_1 - E_0\rvert$ (bar) | 1.4e-14 (8.1e-11) | 5.7e-14 (5.7e-11) | 2.8e-14 (1.1e-10) |
| $\lvert E_\text{RDM} - E_1\rvert$ (bar) | 1.4e-14 (1.4e-11) | 2.8e-14 (3.1e-11) | 2.8e-14 (7.6e-11) |
| $\lVert H_1c - E_1c\rVert$ of the returned CI vector | 1.2e-11 | 2.7e-11 | 7.0e-11 |
| max $\lvert\delta\gamma\rvert$ (bar $2N(\lVert r_0\rVert+\lVert r_1\rVert)/\Delta$) | 4.4e-11 (8.2e-10) | 5.2e-12 (9.8e-9) | 2.4e-11 (7.5e-9) |
| max off-diagonal $F^T$, core and virtual (bar) | 9.1e-13 (9.2e-11) | 3.7e-13 (2.1e-11) | 4.5e-11 (6.3e-11) |
| idempotence, worst $\lvert O_{pq}\rvert$ / bar; unconstrained pairs | 0.50; 0 | 0.46; 8 | 0.50; 0 |

The test therefore covers the energy, the RDM energy and the CI
eigenvector in the new orbitals, the natural orbitals, the diagonal core
and virtual Fock blocks, and idempotence.

**Mutations of the gradient** (applied to `src/casscf.rs` one at a time,
2026-11-11; max $\lvert$analytic − FD$\rvert$ per class, core-active /
core-virtual / active-virtual):

| Mutation | H2O | N2 | Anchor 1 | Anchor 3 |
|---|---|---|---|---|
| $F_{pi} = 2F^I_{pi}$ ($F^A$ dropped) | 1.94 / 1.57 / 1.5e-10 | 1.39 / 1.34 / 1.6e-10 | pass (no core) | not converged in 300 |
| $\Gamma$ term of $F_{pt}$ dropped | 0.567 / 1.2e-10 / 1.91 | 1.17 / 2.0e-10 / 1.71 | fail (6.7, 1.7) | not converged in 300 |
| core-virtual gradient sign flipped | 2.4e-10 / 11.2 / 1.5e-10 | 1.8e-10 / 14.0 / 1.6e-10 | pass (no core) | not converged in 300 |

Each mutation fails the classes whose formula it changes and leaves the
others at the unmutated error.

**Mutations of canonicalization and of the open-shell path** (applied to
`src/casscf.rs` one at a time, 2026-11-11):

| Mutation | Result |
|---|---|
| active orbitals rotated, CI vector and RDMs of the old orbitals kept | `casscf_canonical.rs` fails for all three systems: $\lvert E_\text{RDM} - E_1\rvert$ = 3.9e-2 / 0.29 / 0.16 Ha, $\lVert H_1c - E_1c\rVert$ = 0.31 / 0.63 / 0.52, max $\lvert\delta\gamma\rvert$ = 1.7e-2 / 1.3e-10 / 5.1e-3 (H2O / N2 1.6 Å / O2) |
| $F^A$ dropped from the canonical Fock ($F^I$ only) | `casscf_canonical.rs` fails for all three: max off-diagonal $F^T$ 0.26 / 0.20 / 0.37 against bars near 1e-10; the idempotence check passes (both calls use the same matrix) |
| CI in the sector $(n_\alpha - 1, n_\beta + 1)$ when $n_\alpha \ne n_\beta$ | H2O (6, 4) full space fails, $\Delta E$ = -0.398 Ha (the singlet). O2 (9, 7) full space, O2 against PySCF and the H2O (3, 1) finite-difference check pass: the lowest root of the O2 $M_s = 0$ sector is the $M_s = 0$ component of the same triplet, and the finite difference runs in the same wrong sector. The O2 finite-difference test errors (a Davidson solve in the near-degenerate $M_s = 0$ sector misses its threshold) and the O2 canonicalization test panics on the CI vector length; neither is an assertion |

The spin-sector mutation is caught only where the lowest state of the
requested sector is not the lowest state of the neighbouring sector. The
H2O (6, 4) full-space case is that check.

### The direct and RI routes {#casscf-proven-routes}

**Direct against dense** (2026-11-13, `crates/ferric-ci/tests/casscf_direct.rs`).
H2O CAS(4,4), N2 CAS(6,6) and O2 CAS(8,6) $(5, 3)$ on 6-31G, at RHF
orbitals rotated by a random $\kappa$ (scale 0.05), CI threshold
$10^{-12}$. The direct route is compared with the dense oracle at
`thresh = 0` and at the default $10^{-12}$. The bars on $F$ and
$(pu|vw)$ are a priori: the summation rounding of both routes plus the
screened quartets, carried through the coefficients by their column
1-norms. The bars on $E$ and $g$ propagate the measured $\delta F$ and
$\delta P$, including the CI response through the dense CI gap. $(pu|vw)$
is checked before any CI runs on it.

| System | thresh | $\delta F$ (bar) | $\delta(pu\vert vw)$ (bar) | $\lvert\Delta E\rvert$ (bar) | $\lvert\Delta g\rvert$ (bar) |
|---|---:|---:|---:|---:|---:|
| H2O CAS(4,4) | 0 | 9.4e-14 (1.1e-8) | 1.5e-14 (7.7e-11) | 2.8e-14 (1.8e-11) | 2.5e-13 (1.5e-6) |
| H2O CAS(4,4) | 1e-12 | 9.4e-14 (2.6e-8) | 1.5e-14 (7.8e-10) | 2.8e-14 (1.8e-11) | 2.5e-13 (1.5e-6) |
| N2 CAS(6,6) | 0 | 2.3e-13 (1.4e-6) | 1.7e-13 (6.5e-9) | 2.6e-13 (2.7e-10) | 7.7e-13 (7.9e-4) |
| N2 CAS(6,6) | 1e-12 | 2.3e-13 (1.6e-6) | 1.7e-13 (5.6e-8) | 2.6e-13 (2.7e-10) | 7.7e-13 (7.9e-4) |
| O2 CAS(8,6) | 0 | 3.3e-13 (2.6e-7) | 7.3e-14 (8.4e-10) | 4.3e-13 (1.6e-10) | 1.0e-12 (1.3e-3) |

The gradient bars are loose because they multiply worst-case element
bounds over every rotation class. The measured differences are at
rounding.

Mutations (one at a time, 2026-11-13; each fails on an assertion):

| Mutation | Caught by |
|---|---|
| $F^A = J - \tfrac14K$ (half the exchange in the active Fock) | `casscf.rs`: analytic vs FD, core-active 0.245 (H2O) against $10^{-8}$; prototype energy $\lvert\Delta E\rvert$ = 4.7e-2 Ha (H2O). `casscf_direct.rs` passes: both routes share the formula |
| direct $K$ halved | `casscf_direct.rs`, 5 of 6 tests: $\delta F$ = 2.32 against $\le 1.6\times10^{-6}$ |
| direct $(pu\vert vw)$ without the $s_3 \ne s_4$ ket mirror | `casscf_direct.rs`, 5 of 6 tests: $\delta(pu\vert vw)$ = 0.86 to 8.5 against $\le 5.6\times10^{-8}$ |
| dense $K$ contracted with $D^\mathsf{T}$ (`column(l)`) | unit test `dense_jk_gemv_matches_the_scalar_loops` (nonsymmetric $D$): $\lvert\Delta K\rvert$ = 0.84 against 5.3e-11 |

The sixth `casscf_direct.rs` test checks that a negative or NaN threshold
is refused.

**RI against dense fed $B^\mathsf{T}B$** (2026-11-14,
`crates/ferric-ci/tests/casscf_ri.rs`). Water CAS(4,4), N2 CAS(6,6) and O2
CAS(8,6) $(5, 3)$ on 6-31G with def2-universal-jkfit ($n_\text{aux}$ 113,
154, 154). The anchor feeds the dense route the fitted integrals
$B^\mathsf{T}B$, at RHF orbitals rotated by a random $\kappa$ (scale 0.05).
The bars are a priori as for the direct route, with the RI chain lengths
($n_\text{bas}^2 + n_\text{aux}$ terms in $J/K$, $4n_\text{bas} + n_\text{aux}$
in the MO blocks) and $e_\text{abs} = \max \sum_P |B_{P,mn}||B_{P,ls}|$:

| System | $\delta F$ (bar) | $\delta(pu\vert vw)$ (bar) | $\lvert\Delta E\rvert$ (bar) | $\lvert\Delta g\rvert$ (bar) | $\lvert\delta(Hx)\rvert$ kernel / AO $J/K$ (bar) |
|---|---:|---:|---:|---:|---:|
| H2O CAS(4,4) | 2.5e-14 (1.8e-8) | 4.0e-15 (2.5e-10) | 2.8e-14 (1.3e-11) | 2.7e-13 (3.9e-7) | 3.6e-15 / 4.0e-15 (3.4e-6) |
| N2 CAS(6,6) | 8.5e-14 (2.0e-6) | 6.2e-14 (2.0e-8) | 4.3e-14 (1.2e-10) | 1.9e-13 (2.9e-4) | 2.8e-14 / 2.7e-14 (7.9e-4) |
| O2 CAS(8,6) | 3.2e-14 (3.9e-7) | 4.7e-15 (2.6e-9) | 1.1e-13 (5.1e-11) | 1.4e-13 (8.4e-5) | 6.7e-15 / 7.3e-15 (1.2e-4) |

SC-NEVPT2 on the converged RI CASSCF, with the canonical orbitals used as
given on both routes, agrees with the dense route within
$2.5\times10^{-16}$ in every class (water, N2). Re-diagonalizing the core
and virtual blocks instead (`canonicalize = true`) moves N2's $S_{rs}$ by
$1.6\times10^{-8}$ between the two routes and by $1.2\times10^{-7}$ from the
as-given value. N2's virtual $\pi$ pairs are degenerate, the eigensolver's
rotation within them follows $10^{-14}$ differences in $F$, and the
strongly contracted energy is not invariant to that rotation. The
comparisons below therefore fix the orbitals.

**RI against PySCF `density_fit()`.** The references come from
`scripts/gen_casscf_ri_refs.py`, with PySCF built on ferric's basis data;
PySCF runs CASSCF from its own DF RHF, ferric from its own.

| System | $E_\text{RI}$ (Ha) | − PySCF DF (bar) | $E_\text{RI} - E_\text{exact}$ | RI-JK RHF error |
|---|---:|---:|---:|---:|
| H2O CAS(4,4) | -76.036953811359 | -1.1e-12 (2.7e-9) | 1.53e-5 | 1.05e-5 |
| N2 CAS(6,6) | -109.015504086513 | -5.7e-14 (6.8e-9) | 4.27e-5 | 4.38e-5 |
| O2 CAS(8,6) | -149.636648921281 | -2.6e-13 (1.4e-8) | 8.09e-5 | 6.01e-5 |

The RI CASSCF error is of the size of the RI-JK RHF error with the same
auxiliary basis (both codes give the same RHF error to the digits shown).
At ferric's RI orbitals the fixed-orbital CAS-CI agrees with PySCF DF
CAS-CI to $4\times10^{-13}$. The SC-NEVPT2 classes (water, N2) agree to
$1.1\times10^{-13}$ except N2's $S_{rs}$: $3.8\times10^{-8}$ Ha, inside the
$8.7\times10^{-8}$ Ha spread of PySCF's own $S_{rs}$ over seven repeats on
identical orbitals (identical CAS-CI energies; every other class repeats
to $\le 3.4\times10^{-15}$). The generator records that spread and the
test bar adds it.

Mutations (one at a time, 2026-11-14; each fails on an assertion, and the
source is restored and checked by checksum):

| Mutation | Caught by |
|---|---|
| half the exchange in the RI active Fock ($F^A = J - \tfrac14K$ on the RI route only) | `casscf_ri.rs`, 8 of 12 tests: $\delta F$ = 0.41 / 0.31 / 0.36 (H2O / N2 / O2) against bars $\le 2.0\times10^{-6}$; the PySCF and NEVPT2 tests fail their convergence assertions. `casscf.rs`, `casscf_direct.rs`, `casscf_newton.rs` pass (they do not run the RI route) |
| RI Hessian products return $(pu\vert qv)$ in place of $(pq\vert uv)$ | `casscf_ri.rs`, 8 of 12 tests: $\lvert\delta(Hx)\rvert$ = 0.89 / 2.8 / 3.6 against bars $\le 7.9\times10^{-4}$; the AH optimizer then fails to converge in the PySCF and NEVPT2 tests |

### Timings {#casscf-proven-timings}

**RI against dense** (2026-11-14). Benzene CAS(6,6), 18 core orbitals,
CLI defaults (AH optimizer, exact-J/K RHF start), `budget_gb = 10`,
def2-universal-jkfit, `OPENBLAS_NUM_THREADS=1`, 12 rayon threads. At
cc-pVDZ (114 basis functions) the dense and RI routes ran paired and
interleaved (A B B A A B). CPU is user + system time; the box load average
was 5.2 to 18.4 during the runs.

| Route | Wall (s), 3 runs | mean ± sd (s) | CPU (s), 3 runs | mean ± sd (s) | macro iterations | peak RSS |
|---|---|---:|---|---:|---:|---:|
| dense | 30.1, 29.5, 32.7 | 30.8 ± 1.7 | 301.6, 298.7, 304.5 | 301.6 ± 2.9 | 19 | 2.35 GB |
| RI | 20.0, 20.5, 27.4 | 22.6 ± 4.1 | 196.6, 201.7, 211.0 | 203.1 ± 7.3 | 15 | 0.70 to 0.75 GB |

Four of the six runs started above load 12. The CPU difference (98 s) is
more than ten times either route's spread; the wall difference is not, so
only the CPU and memory figures are claims. Both include the RHF (about
10 s, identical on both routes). The two routes reach different stationary
points from the same RHF start: dense $-230.7814586093$ Ha (the point of
section @sec:casscf-proven-stationary), RI $-230.7872547148$ Ha. The RI
point lies $8.0\times10^{-5}$ Ha above PySCF's exact-integral
$-230.7873347336$, an RI-error-sized offset, so it is the RI image of
PySCF's lower minimum. Per macro iteration (CASSCF phase totals,
`FERRIC_TIMING=1`, first run of each): dense 18.7 s / 19 = 0.98 s, RI
8.4 s / 15 = 0.56 s. The occupied-kernel integrals of the AH Hessian take
0.68 s (dense) and 0.39 s (RI) per build.

At cc-pVTZ (264 basis functions) the dense route is refused before it runs
(a 50.5 GB plan against the 10.7 GB budget). The RI route under
`scripts/ferric-limited` (12 GB cgroup) converges in 18 macro iterations to
$-230.8442716629$ Ha, $\lVert g\rVert = 2.3\times10^{-8}$: wall 147 s, CPU
1397 s, peak RSS 2.80 GB, load average 16.8 to 17.2. The exact-J/K RHF is
77 s of it and the CASSCF 67 s.

**The dense route** (2026-11-13). Benzene CAS(6,6)/cc-pVDZ,
`testdata/molecules/benzene.xyz`, 114 basis functions, 18 core orbitals,
ferric RHF start, default settings (`integrals = "auto"` resolves to dense
at `budget_gb = 10`). CLI wall times, paired and interleaved
(A B B A A B), 12 rayon threads, box load average 2.5 to 10.4:

| Build | Wall (s), 3 runs | mean ± sd (s) | macro iterations | wall per iteration (s) | peak RSS |
|---|---|---:|---:|---:|---:|
| dense scalar $J/K$, single-threaded transform | 117.3, 110.5, 110.3 | 112.7 ± 4.0 | 90 | 1.25 ± 0.04 | 1.79 GB |
| dense route | 37.4, 35.9, 35.6 | 36.3 ± 1.0 | 89 | 0.41 ± 0.01 | 1.78 GB |

Both reach $E = -230.7814586093$ Ha. The wall times include the RHF (about
9 s). The iteration counts differ by one because the $J/K$ sums round
differently. Phase totals of the dense route (`FERRIC_TIMING=1`, the
median run) and of one direct-route run (`thresh` $10^{-12}$, 88
iterations, peak RSS 0.56 GB):

| Phase | dense (s) | busy cores | direct (s) | busy cores |
|---|---:|---:|---:|---:|
| inactive Fock ($J/K$ of $D^c$) | 6.0 | 11.3 | 93.2 | 11.2 |
| $(pu\vert vw)$ transform | 7.0 | 11.5 | 169.6 | 11.2 |
| CI solve | 2.4 | 11.7 | 1.6 | 11 |
| active Fock ($J/K$ of $D^a$) | 6.3 | 11.2 | 93.0 | 11.2 |
| BFGS update + step | 3.7 | 1.0 | 3.8 | 1.0 |
| other | 0.1 | | 0.1 | |
| total, per iteration | 25.5, 0.29 | | 361.4, 4.1 | |

### The optimizers {#casscf-proven-optimizers}

**Iteration counts** (2026-11-13). Macro iterations (one CAS-CI each) and
Hessian-vector products, from `scripts/prototypes/casscf_newton_proto.py`
on ferric's integrals and RHF start; PySCF 2.13.1 `mcscf.CASSCF`
(`conv_tol` $10^{-12}$, `conv_tol_grad` $10^{-6}$, FCI `conv_tol`
$10^{-14}$, the settings of `scripts/gen_casscf_refs.py`) from ferric's RHF
and from its own. Water, N2 and O2 as in `tests/casscf.rs` (6-31G); benzene
CAS(6,6)/cc-pVDZ with 18 core orbitals; N2 CAS(10,10)/cc-pVDZ with 2 core
orbitals.

| System | BFGS | AH | AH, ratio stretch | AH, η = 0.1 | AH, ratio stretch, η = 0.3 | Rust AH (shipped) | PySCF macro (from ferric / own RHF) | PySCF micro, inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| water CAS(4,4) | 61 | 21 / 197 | 22 / 190 | 27 / 134 | 27 / 111 | 22 | 19 / 19 | 67, 312 |
| N2 CAS(6,6) | 21 | 10 / 57 | 11 / 55 | 12 / 32 | 13 / 24 | 11 | 8 / 8 | 25, 95 |
| O2 CAS(8,6) $M_s = 1$ | 14 | 7 / 31 | 8 / 35 | 8 / 19 | 10 / 15 | 8 | 8 / 8 | 23, 82 |
| benzene CAS(6,6) | 90 | 27 / 470 | 19 / 352 | 29 / 314 | 28 / 223 | 18 | 16 / 19 | 57, 285 |
| N2 CAS(10,10) | 36 | 39 / 343 | 26 / 191 | | 29 / 79 | 25 | 15 / 15 | 51, 230 |

Prototype entries are macro iterations / Hessian products. "AH" is the
step without a stretch; "ratio stretch" multiplies an unclipped step by
the previous energy ratio clamped to $[1, 2]$, with forcing
$\eta = \min(0.1, \lVert g\rVert)$; the η columns stop the Davidson at the
constant relative residual η. "Rust AH" is the shipped step with the
curvature stretch (macro iterations; BFGS in Rust takes 60, 21, 14, 89 and
36). A PySCF macro iteration contains several orbital micro-steps with
approximate CI updates, so its macro count is not the CI-solve count;
ferric's two-step AH needs 1.2 to 1.7 times PySCF's macro count. All AH
variants on benzene stay at ferric's minimum $-230.7814586093$ Ha; none
reaches PySCF's $-230.7873347336$. The $\eta = 0.1$ runs (with and without
the stretch) stop at $-230.7814587989$, $1.9\times10^{-7}$ Ha lower with
$\lVert g\rVert = 8\times10^{-8}$, a nearby stationary point that was not
examined further.

**AH anchors** (`crates/ferric-ci/tests/casscf_newton.rs`,
`crates/ferric-ci/tests/casscf_direct.rs`, 2026-11-13). 6-31G, RHF
orbitals rotated by a random $\kappa$ (scale 0.05), random unit $x$.

| Check | Measured | Bar |
|---|---|---|
| $H_fx$ vs the central difference of the fixed-RDM gradient at $h/2 = 5\times10^{-4}$ (water, N2, O2) | worst error 0.50 of the bar | $2\lvert\Delta\rvert/3$ plus rounding |
| $H_fx$ vs the Richardson extrapolate | $1.5\times10^{-11}$, $1.2\times10^{-10}$, $2.3\times10^{-11}$ | |
| symmetry $\lvert y^\mathsf{T}Hx - x^\mathsf{T}Hy\rvert$ | $7\times10^{-16}$ to $4\times10^{-15}$ | $3$–$6\times10^{-11}$ |
| the frame derivative alone, asymmetry | 0.06–0.79 | |
| AH vs BFGS minimum, $\lvert\Delta E\rvert$ | water $1.4\times10^{-13}$, O2 $1.4\times10^{-13}$ | $1.1\times10^{-12}$, $2.9\times10^{-12}$ |
| direct vs dense $Hx$ (O2, N2 at `thresh` $10^{-12}$) | $8.4\times10^{-14}$, $2.6\times10^{-13}$ | a priori $2.8\times10^{-6}$ to $2.2\times10^{-3}$ |

The FD error is 0.50 of the bar because it is $|\Delta|/3$, the leading
truncation term, against a bar of $2|\Delta|/3$: $H_fx$ matches the
Richardson extrapolate. With the default switched to AH every existing
CASSCF test passes (`casscf.rs`, `casscf_vs_pyscf.rs`,
`casscf_canonical.rs`, `casscf_direct.rs`, `cas_ecp.rs`): water, N2, O2 and
stretched N2 against PySCF within $1.5\times10^{-11}$, $4\times10^{-14}$,
$3\times10^{-13}$ and $5\times10^{-13}$ Ha in 22, 11, 8 and 7 iterations.

Mutations, each failing on an assertion: the active-active coupling term of
$W$ with its sign flipped (`&b - &a`) fails all three FD anchors and both
same-minimum tests (AH no longer converges); dropping the
$-\tfrac12[G, X]$ correction fails the three symmetry assertions; $K$
replaced by $J$ in `dense_jk_multi` fails its unit test; the direct
$(pu|qv)$ scaled by $1 + 10^{-4}$ fails the water direct-vs-dense test
($3.4\times10^{-5}$ against $2.8\times10^{-6}$; $1 + 10^{-6}$ is below the
bar); the CLI `optimizer` key ignored fails `cli_keys_reach_the_config`.

**AH timing** (2026-11-13, AO $J/K$ products, build c74da329d). CLI wall
times, RHF included, paired and interleaved: base build 1db597bf6, build
c74da329d with `optimizer = "bfgs"` (then the default) and with `"ah"`, in
the order base, bfgs, ah, ah, bfgs, base, base, bfgs, ah; 12 rayon
threads, default `[memory]`, dense route. The box was shared with other
jobs; the load average of each set is given.

| System, load | base (s) | BFGS (s) | AH (s) | iterations base / BFGS / AH |
|---|---|---|---|---|
| benzene CAS(6,6), load 15.5-20.5 | 63.47, 65.66, 64.23: 64.5 ± 1.1 | 63.27, 56.84, 65.13: 61.7 ± 4.4 | 87.04, 83.69, 85.09: 85.3 ± 1.7 | 89 / 89 / 18 |
| N2 CAS(10,10), load 17-20 | 28.70, 28.50, 28.84: 28.7 ± 0.2 | 28.58, 30.43, 29.03: 29.4 ± 1.0 | 20.53, 19.60, 20.72: 20.3 ± 0.6 | 36 / 36 / 25 |
| benzene CAS(6,6), load 5.6-15.5 | 37.70, 36.76, 36.26: 36.9 ± 0.7 | 37.04, 35.61, 35.54: 36.1 ± 0.9 | 48.72, 47.65, 47.91: 48.1 ± 0.6 (ratio stretch) | 89 / 89 / 18 |

Every run reaches the energy of the base build to the printed 10 decimals
(benzene $-230.7814586093$, N2 $-109.1294674756$ Ha). BFGS in this build
takes the same iterations as the base build. AH costs about 1.3 times the
BFGS wall time on benzene, where one Hessian product (a $J/K$ pass over the
dense block) is about a third of a macro iteration and each AH step takes
10 to 20 products. It saves about 30 % on N2 CAS(10,10), where the CI solve
dominates. The third row uses the prototype's ratio stretch instead of the
shipped curvature stretch, at the same 18 iterations. With the ratio
stretch N2 CAS(10,10) did not converge: it passed the $-109.1294674756$ Ha
point with $\lVert g\rVert \approx 5\times10^{-4}$ and steps of $10^{-4}$,
actual/predicted ratios of $10^5$ to $10^6$, then descended to
$-109.1347620086$ Ha, 5.3 mHa lower, and stalled there at
$\lVert g\rVert = 9.2\times10^{-5}$ after 1000 macro iterations. That point
is the minimum of section @sec:casscf-proven-stationary.

**Davidson forcing** (2026-11-13). Harness `crates/ferric-ci/examples/casscf_ah_bench.rs`
(exact-J/K RHF of `driver::rhf_start`, `CasScfConfig` defaults,
`FERRIC_TIMING=1` for the product count); base build 8833755a8 (the same
harness without the forcing argument) in a separate target directory; 12
rayon threads, dense route. Wall times are of `run_casscf` only.
Iterations / Hessian products of the AH step, from single runs (identical
on repeats):

| System | base AH (quadratic) | constant $\eta = 0.1$ | $\eta = 0.3$ | $\eta = 0.5$ |
|---|---:|---:|---:|---:|
| water CAS(4,4) | 22 / 193 | 24 / 133 | 28 / 112 | 31 / 101 |
| N2 CAS(6,6) | 11 / 55 | 11 / 27 | 13 / 23 | 16 / 21 |
| O2 CAS(8,6) $M_s = 1$ | 8 / 35 | 8 / 17 | 10 / 15 | 12 / 13 |
| N2 CAS(10,10) | 25 / 176 | 25 / 80 | 29 / 65 | |
| benzene CAS(6,6) | 18 / 335 | 24 / 235 | 30 / 230 | |

Anchor: `AhForcing::Quadratic` in the build with the forcing option and the
AO $J/K$ products reproduces 8833755a8's AH run on all five systems in the
final energy to the last printed digit (17 significant), $\lVert g\rVert$,
the iteration count and the product count. Constant forcing alone left
benzene slower than BFGS (CASSCF 33.5 s at $\eta = 0.1$, 38.6 s at 0.3,
BFGS 27.4 s, single runs at load 7 to 13).

**The occupied kernel** removes the $J/K$ pass from the product. Its
anchors (`crates/ferric-ci/tests/casscf_newton.rs`, dense route, rotated
RHF orbitals, unit $x$):

| Check | Measured (water, N2, O2) | Bar |
|---|---|---|
| kernel vs AO $J/K$ product | $3.6\times10^{-15}$, $1.6\times10^{-14}$, $3.6\times10^{-15}$ | a priori $2.9\times10^{-6}$, $6.7\times10^{-4}$, $1.1\times10^{-4}$ |
| `transform_ppaa_papa` vs two `transform_eri_mixed` chains: $(pq\vert vw)$ | bitwise | |
| the same: $(pu\vert qv)$ (water, 5 columns) | $8.1\times10^{-16}$ | $1.1\times10^{-10}$ |
| direct vs dense $Hx$, both on the kernel | $8.5\times10^{-14}$, $2.6\times10^{-13}$, $2.5\times10^{-13}$ | $1.1\times10^{-6}$, $3.7\times10^{-5}$, $2.1\times10^{-3}$ |

Benzene phases (one run, load 9 to 12): orbital Hessian integrals 0.71 s
per macro iteration, products 1.4 s in total for 349 (3.4 s while the
product GEMM ran at the serial-section BLAS thread count). With products
this cheap, constant forcing costs more in extra macro iterations (each
with a Hessian integral build) than it saves: benzene CASSCF
$31.6 \pm 8.9$ s at $\eta = 0.1$ against $27.3 \pm 6.0$ s quadratic, so
quadratic forcing stays the default. At $\eta = 0.1$ benzene stops at
$-230.7814587989$ Ha, the nearby stationary point of the iteration-count
comparison above.

Mutations, each failing on an assertion: $-\tfrac12(pr|jk) \to
+\tfrac12(pr|jk)$ in $L$ (kernel vs AO $J/K$ 1.2 to 1.6 against bars
$\le 6.7\times10^{-4}$, the FD anchors, the water same-minimum tests); the
$(pu|qv)$ active block taken at $u$ instead of $n_c + u$ (kernel tests
0.30 to 1.09, FD anchors, all four same-minimum tests); the axis swap of
`transform_ppaa_papa` dropped (1.01 against $1.1\times10^{-10}$); the two
outputs of the shared-first-quarter transform swapped (bitwise check); the
direct $(pu|qv)$ scaled by $1 + 10^{-4}$ (direct vs dense $Hx$, water
$7.7\times10^{-5}$ and O2 $1.1\times10^{-4}$ against $1.1\times10^{-6}$ and
$3.7\times10^{-5}$).

Wall times with the kernel (`run_casscf`, s), paired and interleaved. The
small systems: 10 repetitions in alternating order (base BFGS, base AH,
new AH; then reversed), load 13 to 17. N2 CAS(10,10) and benzene: three
repetitions in the orders (bB bA nA nE), (nE nA bA bB), (bB nA nE bA)
(b base, n this build, B BFGS, A AH quadratic, E AH $\eta = 0.1$):

| System | base BFGS | base AH | AH, kernel | iterations BFGS / AH | load |
|---|---|---|---|---|---|
| water | 0.478 ± 0.048 | 0.240 ± 0.057 | 0.202 ± 0.040 | 61 / 22 | 13-14 |
| N2 CAS(6,6) | 0.859 ± 0.247 | 0.498 ± 0.158 | 0.474 ± 0.144 | 21 / 11 | 14-17 |
| O2 CAS(8,6) | 0.343 ± 0.096 | 0.234 ± 0.087 | 0.264 ± 0.096 | 14 / 8 | 13-15 |
| N2 CAS(10,10) | 27.23, 26.70, 26.90: 26.95 ± 0.27 | 18.62, 18.12, 18.57: 18.44 ± 0.28 | 18.63, 19.79, 18.56: 18.99 ± 0.69 | 36 / 25 | 15-24 |
| benzene CAS(6,6) | 38.75, 47.11, 41.27: 42.4 ± 4.3 | 62.19, 39.46, 56.15: 52.6 ± 11.8 | 30.76, 20.33, 30.75: 27.3 ± 6.0 | 89 / 19 | 8-25 |

The benzene load swung between 8 and 25 within the set; within each
repetition the kernel AH run was faster than the base BFGS run (30.8/38.8,
20.3/47.1, 30.8/41.3 s). On N2 CAS(10,10) the CI solve dominates and the
kernel changes nothing measurable. O2 base AH and kernel AH are inside the
noise; both beat BFGS in fewer iterations. BFGS in this build reproduces
the base BFGS run (energy, $\lVert g\rVert$, iterations) on all five
systems. With AH the default these pass: ferric-ci `--lib`, `casscf`,
`casscf_vs_pyscf`, `casscf_canonical`, `casscf_direct`, `casscf_newton`,
`cas_ecp`, `nevpt2`; ferric-cli `cas_kinds`; and the Python
`test_casscf.py`, `test_cas_cli.py`, `test_nevpt2.py`, `test_casci.py`
(slow markers included) and the `.pyi` coverage tests.

### Two stationary points examined {#casscf-proven-stationary}

**N2 CAS(10,10)/cc-pVDZ** (2026-11-13). The harness
`crates/ferric-ci/examples/casscf_stationary_points.rs` runs BFGS from the RHF orbitals to
$\lVert g\rVert < 10^{-9}$ and canonicalizes (natural active orbitals). It
builds the CI-relaxed orbital Hessian by central differences of the
analytic gradient ($h = 10^{-4}$, all 212 pairs, symmetrized) and takes its
lowest eigenvalues. It classifies each active natural orbital by its
dominant $|m|$ (Fourier power on circles about the bond axis, 64 points)
and its inversion parity $\langle\phi(r), \phi(-r)\rangle/\langle\phi,\phi\rangle$
about the bond midpoint. While the lowest eigenvalue is below $-10^{-4}$,
it rotates the orbitals by 0.3 along the eigenvector and converges BFGS
again.

| Point | $E$ (Ha) | $\lVert g\rVert$ | lowest Hessian eigenvalues | natural occupations | active orbitals beyond the 8 valence |
|---|---:|---:|---|---|---|
| start (= BFGS, AH, PySCF) | -109.1294674756 | 9.4e-10 | -0.00999, -0.00814, 0.00000, 0.00263 | 1.9938 1.9803 1.9790 1.9422 1.9407 0.0679 0.0641 0.0203 0.0076 0.0040 | $\sigma_g$, one component of a $\pi_u$ pair |
| one step down | -109.1300807964 | 7.8e-10 | -0.00457, 0.00000, 0.00219 | 1.9933 1.9811 1.9782 1.9418 1.9404 0.0681 0.0643 0.0203 0.0085 0.0039 | $\sigma$ orbitals of mixed parity (0.03 to 2.7), one $\pi_u$ component |
| two steps down | -109.1347620089 | 6.2e-10 | 0.01087, 0.01087, 0.01928, 0.02000 | 1.9905 1.9811 1.9795 1.9423 1.9423 0.0636 0.0636 0.0202 0.0100 0.0070 | $\sigma_g$, $\sigma_u$ (parities $\pm1.000$) |

The point that BFGS, AH (both forcings) and PySCF reach is a second-order
saddle. Its active space holds a single component of a degenerate $\pi_u$
pair, which breaks the axial symmetry: the valence $\pi$ occupations differ
by $1.5\times10^{-3}$ and $3.8\times10^{-3}$. The zero eigenvalue is the
rotation of that component about the axis. Two steps down the negative
modes reach a local minimum 5.29 mHa lower whose extra active orbitals are
one $\sigma_g$ and one $\sigma_u$, with exactly degenerate $\pi$ pairs. This
is the $-109.1347620086$ Ha point the ratio stretch descended to, to
$3\times10^{-10}$ Ha. The optimizers are first-order convergent to
stationary points; the second-order check reports this point as a saddle
and `follow` descends from it (section @sec:casscf-proven-second-order).
Why the RHF start leads to the saddle was not examined.

**Benzene CAS(6,6)/cc-pVDZ.** `pyscf.mcscf.CASSCF` (PySCF 2.13.1) on
ferric's AO integrals reaches $E = -230.7873347336$ Ha, 5.876 mHa below
ferric, from ferric's RHF orbitals (36 macro iterations) and from PySCF's
own RHF. Each code's energy functional reproduces the other's at the
other's orbitals:

| Orbitals | ferric $E$ (Ha) | ferric $\lVert g\rVert$ | PySCF CASCI $E$ (Ha) | PySCF $\lVert g\rVert$ |
|---|---:|---:|---:|---:|
| ferric's converged | -230.7814586093 | 4.8e-8 | -230.7814586093 | 2.4e-8 |
| PySCF's converged | -230.7873347336 | 1.0e-6 | -230.7873347336 | |

Both points are stationary in both codes. The lowest eigenvalues of the
CI-relaxed orbital Hessian come from Lanczos on central-difference
gradient products ($h = 10^{-4}$), applied to $D^{-1/2}HD^{-1/2}$ with $D$
the approximate diagonal floored at 0.05, which has the inertia of $H$:

| Point | Lanczos steps | lowest three | Ritz residual of the lowest |
|---|---:|---|---:|
| ferric's | 80 | 0.3645, 0.5082, 0.5132 | 2.9e-11 |
| PySCF's | 40 | 0.3999, 0.5125, 0.5143 | not recorded |

Ferric's point is therefore a local minimum. Under $z \to -z$ (the
molecular plane) every orbital in all three sets (RHF, ferric, PySCF) stays
purely $\sigma$ or purely $\pi$. Both active spaces hold two $\sigma$ and
four $\pi$ orbitals, with the sixth $\pi$ ($a_{2u}$) in the core. In the
RHF the $a_{2u}$ $\pi$ ($-0.498$ Ha) lies below the $e_{2g}$ $\sigma$ pair
($-0.491$ Ha), and the 18-orbital core splits that pair. Natural
occupations: ferric 1.981, 1.938, 1.936, 0.064, 0.063, 0.019; PySCF 1.982,
1.941, 1.936, 0.065, 0.058, 0.018. The two codes sit at two minima of the
same $2\sigma + 4\pi$ active-space class, and neither is the all-$\pi$
CAS(6,6). A $\pi$ CAS(6,6) needs an active window placed on the six $\pi$
orbitals, which neither `[cas]` nor `run_casscf` offers: the active
orbitals are the `n_active` orbitals after the first `n_core` in the RHF
order.

### The second-order check {#casscf-proven-second-order}

Measured 2026-11-14 in `crates/ferric-ci/tests/casscf_stability.rs`,
`crates/ferric-ci/tests/casscf_vs_pyscf.rs` and
`crates/ferric-ci/tests/casscf_stability_saddle.rs` (release,
`OPENBLAS_NUM_THREADS=1`).

**Products.** $H_\text{rel}x$ is compared with the central difference $D$
of the CI-relaxed analytic gradient along a random unit vector
($h = 10^{-3}$, $D(h/2)$), with bar
$\tfrac23|D(h) - D(h/2)| + 2\delta g/h + \lVert g\rVert$, on 6-31G:

| System | max $\lvert H_\text{rel}x - D\rvert$ | bar | max $\lvert H_{oc}z\rvert$ | asymmetry $\lvert y\cdot Hx - x\cdot Hy\rvert$ |
|---|---:|---:|---:|---:|
| H2O CAS(4,4) | 1.6e-6 | 4.9e-6 | 0.217 | 8.3e-14 |
| H2O CAS(4,4), RI (def2-universal-jkfit) | 1.6e-6 | 4.9e-6 | 0.217 | 8.6e-14 |
| N2 CAS(6,6) | 1.7e-6 | 5.0e-6 | 0.132 | 2.5e-13 |
| O2 CAS(8,6) $(5,3)$ | 4.0e-6 | 9.5e-6 | 0.072 | 4.9e-13 |

The product error is the $O(h^2)$ truncation of $D(h/2)$, one third of
$|D(h) - D(h/2)|$ in every row; the CI part it resolves is $10^4$ to $10^5$
times larger. The trivial limit is one determinant (H2O CAS(2,1) above 4
core orbitals): the CI complement is empty and $H_\text{rel}x = H_{oo}x$
exactly. Its check is `MARGINAL`, $\lambda = -2.4\times10^{-15}$, because
the doubly occupied active orbital makes the core-active rotations
redundant.

**The lowest eigenvalue at the four PySCF anchors**, against the
finite-difference CI-relaxed Hessian of `casscf_vs_pyscf.rs` (all pairs,
$h = 10^{-4}$):

| System, 6-31G | analytic $\lambda_\text{min}$ | FD $\lambda_\text{min}$ | difference | bar | products | Davidson iterations |
|---|---:|---:|---:|---:|---:|---:|
| H2O CAS(4,4) | 0.01945655 | 0.01945655 | 2.1e-10 | 4.4e-4 | 64 | 11 |
| N2 CAS(6,6), 1.0977 Å | 0.04020959 | 0.04020959 | 3.4e-10 | 8.4e-4 | 70 | 12 |
| N2 CAS(6,6), 1.6 Å | 0.14848830 | 0.14848829 | 1.0e-9 | 8.4e-4 | 75 | 13 |
| O2 CAS(8,6) $(5,3)$ | 0.09583425 | 0.09583427 | 1.7e-8 | 8.4e-4 | 74 | 17 |

All four are minima (`STABLE`). The bar here is the error bound of the full
finite-difference Hessian (truncation plus $n_\text{pair}\,\delta g/h$,
$4\times10^{-4}$ to $8\times10^{-4}$), set by that construction and loose
for this purpose; the product tests, at bars near $5\times10^{-6}$, are
what resolve the CI response.

**cc-pVDZ**, with the eigenpair checked by the central difference along
$v$ (bar in `casscf_stability_saddle.rs`):

| Point | $E$ (Ha) | $\lambda_\text{min}$, CI-relaxed | FD Rayleigh along $v$ | max $\lvert Dv - \lambda v\rvert$ (bar) | $\lambda_\text{min}$, fixed RDMs | verdict |
|---|---:|---:|---:|---:|---:|---|
| N2 CAS(10,10), RHF start | -109.1294674756 | -9.994836e-3 | -9.99482e-3 | 1.6e-7 (1.9e-6) | +1.3506e-2 | `UNSTABLE` |
| N2 CAS(10,10), `follow` (2 steps) | -109.1347620089 | +1.086752e-2 | +1.086752e-2 | 1.2e-8 (1.8e-6) | +3.0470e-2 | `STABLE` |
| benzene CAS(6,6), dense endpoint | -230.7814586093 | +1.072679e-3 | +1.07269e-3 | 1.1e-8 (2.2e-6) | +2.340e-3 | `STABLE` |

The N2 eigenvalues agree with the full finite-difference Hessian of the
stationary-point table ($-0.00999$ at the saddle, $0.01087$ at the
minimum). `follow` takes two steps and reaches $-109.1347620089$ Ha,
$2.9\times10^{-11}$ Ha from the BFGS reference minimum (bar
$(\lVert g\rVert^2 + \lVert g_\text{ref}\rVert^2)/(2\lambda) + 8\varepsilon|E|\sqrt{N_\text{det}}
= 4.9\times10^{-11}$) and 5.2945 mHa below the saddle.

The benzene dense endpoint, 5.876 mHa above PySCF's point, is a
positive-definite local minimum of the CI-relaxed Hessian,
$\lambda_\text{min} = 1.07\times10^{-3}$ Ha. `follow` takes no step there.
The 0.3645 of the Lanczos table is an eigenvalue of the scaled
$D^{-1/2}HD^{-1/2}$, which has the inertia of $H$; both say minimum.
Reaching PySCF's minimum needs a different start, not a second-order
check.

The CI response decides the sign. The fixed-RDM Hessian $H_{oo}$ is
positive at the N2 saddle ($+0.0135$), so a check without the CI response
would call the saddle a minimum. It is stiffer everywhere
($H_\text{rel} \le H_{oo}$): 0.0305 against 0.0109 at the N2 minimum,
0.00234 against 0.00107 for benzene.

**Cost** (CLI `FERRIC_TIMING=1` phase totals, release,
`OPENBLAS_NUM_THREADS=1` with the default rayon pool, 3 runs each under
`box-lease`, load average 7 to 14 on the 6-core box from other sessions;
mean ± sd):

| System, cc-pVDZ | determinants | products | check (s) | rest of the CASSCF (s) | check / rest |
|---|---:|---:|---:|---:|---:|
| benzene CAS(6,6) | 400 | 178 | 8.58 ± 0.15 | 19.03 ± 0.18 | 0.45 ± 0.01 |
| N2 CAS(10,10) | 63,504 | 97 | 35.19 ± 0.59 | 11.12 ± 0.51 | 3.17 ± 0.10 |

On benzene the CASSCF is dominated by its orbital-Hessian integrals; on N2
by the CI, and each product of the check is a CI linear solve costing about
as much as one CI root solve of a macro iteration (0.36 s against 0.43 s).
Because the check costs about as much as the CASSCF or more,
`[cas] stability` defaults to `"off"`.

**Mutations** (2026-11-14, one at a time, each failing on an assertion,
then restored):

| Mutation | Fails | Detail |
|---|---|---|
| drop the CI response ($H_\text{rel} \to H_{oo}$) | all four product tests; `casscf_vs_pyscf` N2 1.6 Å and O2; the N2 CAS(10,10) saddle test | product error 0.07 to 0.22 against bars near 5e-6. Against the finite-difference $\lambda_\text{min}$: O2 0.1105 vs 0.0958 and N2 1.6 Å 0.1584 vs 0.1485 miss the bar; N2 1.0977 Å (0.04063 vs 0.04021, diff 4.2e-4 under the bar 8.4e-4) and H2O (lowest mode not CI-coupled) pass it. N2 CAS(10,10) changes sign: $+0.0135$, `STABLE`, so the saddle goes undetected |
| add $F^I$ to the RDM-linear gradient ($\gamma_T \to \gamma_T + 1$) | the single-determinant anchor, the product tests, check-vs-follow | the anchor sees it at 6e-17 only (the added term is gradient-sized at a converged single determinant); the products miss by 0.9 to 1.5 |
| drop the certificate cap on `UNSTABLE` | `certificate::tests::a_saddle_is_capped_and_leaves_the_anchor_scope` | grade Proven instead of not graded |

## Sources {#casscf-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Two-step CASSCF" | the method and the measurement records of section @sec:casscf-proven |
| Chaban, Schmidt and Gordon, TCA 97, 88 (1997) | the approximate diagonal Hessian |
| `scripts/prototypes/casscf_proto.py`, `casscf_derivation.md` | the reference implementation |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `casscf-gradient`, `redundant-residual` |

**Known limits (2026-11-15).** Lowest root, no state averaging; one
restricted orbital set with a doubly occupied core, from an exact-J/K RHF
with an even electron count. Both optimizers are two-step: no orbital–CI
coupling in the step. An unconverged run or a saddle found by the check
caps the certificate to not graded.
