# 17.5 Two-step CASSCF

<!-- section: 17.5; status: ready; registry: casscf;
     theory: crates/ferric-ci/theory.md#"The two-step method", "Second-order check" -->

A CAS-CI on RHF orbitals uses orbitals shaped for one determinant. The RHF
virtuals describe an added electron in the field of $N$ others, not the
antibonding partner a stretched bond needs, so the active space can hold the
right configurations in the wrong orbitals. CASSCF minimizes the CAS-CI
energy over the orbitals as well as the CI vector, so the core, active and virtual orbitals become the ones the correlated
wavefunction prefers. ferric does it in two steps per macro iteration: a
CAS-CI in the current orbitals (sections 17.1–17.4), then an orbital step
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

## 17.5.1 Energy, gradient and step

The orbitals are $C = [C_c\,|\,C_a\,|\,C_v]$: $n_c$ doubly occupied core
orbitals $i,j$, $n_a$ active $t,u,v,w$, virtual $a$, and $p,q,r,s$ over
all MOs. The orbitals are real, there is a single root, and the RDMs are
the spin-summed ones of section 17.4. The reference implementation is
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
chapter 7 applied to the core alone. The CAS-CI runs on
$h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ and $E_\text{core}$
(`run_cas_ci_on_integrals`), the folding of section 17.1.2. With no
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
  $M^\mathsf{T}M = V^{-1}$ (the construction of section 10.1.2). It holds
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

> **Verified equation 17.5** · the orbital gradient is $2(F - F^\mathsf{T})$
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
orbital Hessian at fixed RDMs, applied as a product. Chapter 8.3 describes
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
curvature (section 17.5.3 shows why it can only lower it). Near
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

## 17.5.2 The macro iteration

| Step | Quantity | Function |
|---|---|---|
| 1 | $F^I$, $E_\text{core}$, $h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ by the dense, direct or RI route | `CasScfProblem::evaluate` |
| 2 | CAS-CI with RDMs; a root that misses the CI threshold is an error | `run_cas_ci_on_integrals` |
| 3 | $F^A$, the generalized Fock matrix, $g$ | `generalized_fock` |
| 4 | converged if $\lVert g\rVert_2 <$ `grad_tol` and $|\Delta E| <$ `energy_tol` at the same iteration | `optimize` |
| 5 | step $s$: AH Davidson on the fixed-RDM Hessian with trust radius and curvature stretch, or inverse BFGS | `ah_davidson`, `optimize` |
| 6 | $C \leftarrow C\exp(K(s))$ | `rotate`, `expm_antisymmetric` |
| 7 | after convergence, optional canonicalization and second-order check | `canonicalize`, `stability` |

## 17.5.3 The second-order check

Both optimizers stop at a point where the gradient vanishes. That can be a
minimum or a saddle. The sign of the lowest eigenvalue of the orbital
Hessian decides, but it has to be the Hessian of the energy with the CI
vector relaxed at every orbital set, because that is the function CASSCF
minimizes. `src/casscf_stability.rs` (`[cas] stability`, 2026-11-14)
computes it after convergence; chapter 8.4 does the same for SCF.

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
step misses (section 17.5.1).

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

## 17.5.4 Options

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

## 17.5.5 Numerical anchors

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
zero to the CI residual, as the redundant-block identity of 17.5.1
requires.

## 17.5.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Two-step CASSCF" | the method; its measurement records follow the method text there |
| Chaban, Schmidt and Gordon, TCA 97, 88 (1997) | the approximate diagonal Hessian |
| `scripts/prototypes/casscf_proto.py`, `casscf_derivation.md` | the reference implementation |
| `scripts/ci_rdm_casscf_gradient_sympy.py` | `casscf-gradient`, `redundant-residual` |

**Known limits (2026-11-15).** Lowest root, no state averaging; one
restricted orbital set with a doubly occupied core, from an exact-J/K RHF
with an even electron count. Both optimizers are two-step: no orbital–CI
coupling in the step. An unconverged run or a saddle found by the check
caps the certificate to not graded.
