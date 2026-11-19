# Combining the screen, the embedding and λ-Boys orbitals {#amplitude-lmp2-combined-truncation}

<!-- status: ready; registry: local-mp2 -->

The Fock-propagated screen of section @sec:amplitude-lmp2-fock-screen
removes most of the error that the off-diagonal Fock coupling leaves in a
fixed mask, and two other remedies attack the same coupling. Wang, Ling,
Shi, Shen, Wang, Yang, Li and Head-Gordon (JCTC 21, 10910 (2025), WLLH25
below) correct the dropped amplitudes with a per-pair block solve, and
their λ-Boys orbitals make the occupied Fock block more nearly diagonal.
A combined production scheme would use all three: λ-Boys occupied
orbitals (WLLH25 Eqs. 8–9, after Subotnik and Head-Gordon, JCP 122, 034109
(2005)), the WLLH25 block-Kapuy embedding (Eqs. 43–49), and the screen
extended with occupied coupling. This section proves what each does to the
energy. The proofs, and a dense
cross-check on the LMP2 block structure (several pairs, shared virtuals,
non-zero $f_{oo}$ and $F_{vv}$ off-diagonals), are in
`scripts/lmp2_combined_sympy.py`. Each claim carries a status. No
production code implements the occupied extension, the
Hylleraas-evaluated embedding or the embedded Galerkin solve below. They
are statements about the equations. Only λ-Boys and the
virtual-only screen are in the production code (sections
@sec:amplitude-lmp2-lambda-boys and @sec:amplitude-lmp2-fock-screen).

| Claim | Statement | Status |
|---|---|---|
| P1 | masked error is the Schur norm of the dropped exact amplitudes; two-sided bound; nesting | proven under A1–A3 |
| P2 | the occupied-extended screen estimate is one Jacobi step, second order | proven |
| P3 | the embedding residual is linear in the inter-pair coupling; $E_H(t_{\mathrm{emb}})$ is an upper bound | proven |
| P3 | the WLLH25 linear energy $E_L$ is an upper bound; the embedding always beats the plain mask | disproved |
| P3 | the embedded Galerkin energy $E_G$ lies between $E^*$ and both arms | proven by construction; production cost open |
| P4 | the λ-Boys gradient and Jacobi angle; monotone sweeps | proven; global optimality open |
| P4 | $\Phi$ non-increasing in $\lambda$ | proven under the global-optimum assumption |
| P6 | the trivial limits and the rotation invariance of $E^*$ | proven |
| P6 | size consistency | proven under the fragment assumption |

**Setting and assumptions.**

- (A1) The occupied and virtual orbitals are orthonormal, so the amplitude
  space carries the Euclidean metric and $\mathcal A$ of
  (@eq:amplitude-lmp2-1) is symmetric. With
  $\epsilon_{\mathrm{LUMO}} > \epsilon_{\mathrm{HOMO}}$ it is positive
  definite, its spectrum being $\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$.
- (A2) The retained pattern $K$ and its complement $D$ are closed under the
  virtual swap $\mathcal S$ and the pair transpose
  $\mathcal P: (ij,ab)\mapsto(ji,ba)$. The Eq-8 mask and the screen satisfy this by
  construction.
- (A3) The reference $E^*$ is the exact minimum over a fixed universe of
  amplitudes with fixed integrals $J$: the global RI fit, or the union of
  candidate squares. The domain-local fit of section
  @sec:integral-direct-domain-local-fit changes $J$ and is outside these
  statements.

Write $\mathcal W = 2 - \mathcal S$ (eigenvalues 1 and 3, commuting with
$\mathcal A$), $\Delta = \operatorname{diag}\mathcal A$,
$\mathcal A' = \mathcal A - \Delta$, and

$$ E_H(t) = \langle\mathcal W t,\ 2J + \mathcal A t\rangle, \qquad t^* = -\mathcal A^{-1}J, \qquad E_H(t) - E^* = (t-t^*)^{\mathsf T}\mathcal W\mathcal A\,(t-t^*) \ge 0 . $$

$E_H$ is a minimum principle over the whole universe.

## P1. The truncation error (proven under A1–A3) {#amplitude-lmp2-combined-p1}

The masked solve is the Galerkin solution $\mathcal A_{KK}t_K = -J_K$,
$t_D = 0$. Let $R_D = J_D + \mathcal A_{DK}t_K$ be the dropped-block
residual and
$\Sigma = \mathcal A_{DD} - \mathcal A_{DK}\mathcal A_{KK}^{-1}\mathcal A_{KD}$
the Schur complement. By (A2) $\mathcal W$ is block diagonal,
$\mathcal W_D$ commutes with $\Sigma$, and

$$ E_K - E^* = R_D^{\mathsf T}\,\mathcal W_D\,\Sigma^{-1}R_D = t^{*\mathsf T}_D\,\mathcal W_D\,\Sigma\,t^*_D, \qquad t^*_D = -\Sigma^{-1}R_D , \label{eq:lmp2-combined-truncation-error} $$
$$ \frac{\lVert R_D\rVert^2}{\lambda_{\max}(\mathcal A)} \;\le\; E_K - E^* \;\le\; \frac{3\,\lVert R_D\rVert^2}{\lambda_{\min}(\mathcal A)} . $$

The bounds hold because $\Sigma^{-1} = (\mathcal A^{-1})_{DD}$ has its
spectrum in $[\lambda_{\max}^{-1}, \lambda_{\min}^{-1}]$. Two corollaries
follow:

- The masked energy is an upper bound, $E_K \ge E^*$: it recovers less
  correlation energy.
- $K_1 \subseteq K_2 \Rightarrow E^* \le E_{K_2} \le E_{K_1}$. So any screen
  that only adds elements to the Eq-8 pattern can never raise the energy
  error, for every $\kappa$ and with or without the occupied term.

The bracket is a computable a-posteriori certificate for any arm that stops
at a Galerkin solution: $R_D$ costs one matvec over the universe. The CG
error of section @sec:amplitude-lmp2-accurate-solve adds a non-negative term
on top, so the bound survives inexact solves.

> **Verified equation @eq:lmp2-combined-truncation-error** · the truncation error, its two-sided bound and nesting
>
> $E_K - E^* = R_D^{\mathsf T}\mathcal W_D\Sigma^{-1}R_D = t^{*\mathsf T}_D\mathcal W_D\Sigma t^*_D$
> in symbols on a 2+1 block model with a generic symmetric matrix, and
> numerically on random instances with the LMP2 block structure, where the
> two-sided bound holds and a screened pattern is nested in Eq 8 and no
> worse (P1).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P1 |
> | Negative control | the runner drops $\mathcal A_{DK}$ from the Schur complement ("P1: residual identity") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::pair_blocks_screened` |
>
> <!-- verified: id=lmp2.combined.truncation-error; script=scripts/lmp2_combined_sympy.py; claim=P1: upper bound; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::pair_blocks_screened; registry=local-mp2 -->

## P2. The screen's estimate is one Jacobi step (proven) {#amplitude-lmp2-combined-p2}

With $t^0 = -J/\Delta$, the occupied-extended dressed integral is

$$ \tilde g_{ij} = g_{ij} + F'\,t^0_{ij} + t^0_{ij}\,F' - \sum_{k\ne i} f_{ik}\,t^0_{kj} - \sum_{k\ne j} t^0_{ik}\,f_{kj} \;=\; \big(J + \mathcal A' t^0\big)_{ij} . $$

Apply $t^* = t^0 - \Delta^{-1}\mathcal A' t^*$ twice. Exactly,

$$ t^* + \Delta^{-1}\tilde g = (\Delta^{-1}\mathcal A')^2\,t^*, \qquad
\big|t^*_e + \tilde g_e/\Delta_e\big| \le \rho^2\,\lVert t^*\rVert, \quad \rho = \lVert\Delta^{-1}\mathcal A'\rVert_2 \le \frac{\lVert\mathcal A'\rVert_2}{\Delta_{\min}} . \label{eq:lmp2-combined-jacobi-remainder} $$

So the estimate $-\tilde g/\Delta$ is exact through first order in the Fock
coupling, and the estimator error $t^* + \Delta^{-1}\tilde g$ is second
order. It has two sources:

- (i) Paths of length two: a dropped element reached from the integrals
  only through another element with $J = 0$. A three-element chain has
  $\tilde g_3 = 0$ and $t^*_3 = O(\rho^2)$, so the estimate is zero there.
- (ii) The deviation of the kept solution from $t^0$,
  $t_K - t^0_K = -\mathcal A_{KK}^{-1}\mathcal A'_{KK}t^0_K$, which enters
  $R_D$ only through $\mathcal A'_{DK}$ and which $\tilde g_D$, built on
  $t^0$, does not contain.

The committed virtual-only screen drops the occupied terms, which are first
order in $\lVert f'_{oo}\rVert$. The error of its estimate is therefore
$O(\rho)$, not $O(\rho^2)$. On the model, halving the coupling divides the
error of the occupied-extended estimate by 4.0 and that of the
virtual-only estimate by 2.0.

**The estimate is second order; the mask's miss is not.** Everything above
is about the estimator. A mask at a fixed threshold drops every element
whose estimate falls below $\kappa\varepsilon$, and those elements carry
exact amplitudes of $O(\mu)$ in the coupling strength $\mu$. Unless the mask covers the
whole support of the occupied coupling term $f'_{oo}\,t$, the norm of the
dropped exact amplitudes therefore stays first order, for both screens.
Measured 2026-11-17 on octane / 6-31G at $\varepsilon = 10^{-6}$, with all
off-diagonal Fock coupling scaled by $\mu$:

| quantity | occupied-extended | virtual-only |
|---|---|---|
| estimator error $\lVert t_{\mathrm{est}} - t^*\rVert$, no threshold | $\approx 0.038\,\mu^2$, slope 1.98 | $\approx 0.049\,\mu$, slope 1.00–1.07 |
| norm of the dropped exact amplitudes, $\kappa$ = 1, 0.1, 0.01 | slope $\approx 1$ | slope $\approx 1$ |
| miss coefficient, occupied-extended relative to virtual-only | 3% lower ($\kappa$ = 1), 12% lower ($\kappa$ = 0.1), 21% lower ($\kappa$ = 0.01) | |
| kept amplitudes at production $\kappa = 1$ | 0.05% more | |

Both estimator errors are exactly 0 at $\mu = 0$. The occupied extension
makes the estimate second order, but at a fixed threshold it lowers the
dropped-amplitude norm only by the coefficients in the table.

Since $\Sigma = \Delta_D + O(\rho\,\Delta)$, the leading-order error
estimate is the Kapuy energy of the dressed integrals over the dropped set,

$$ \delta E_{\mathrm{est}} = \sum_{e\in D}\frac{\tilde g_e\,(\mathcal W\tilde g)_e}{\Delta_e} = \sum_{ij}\sum_{(a,b)\in D_{ij}}\frac{\tilde g^{ij}_{ab}\,(2\tilde g^{ij}_{ab} - \tilde g^{ij}_{ba})}{\Delta^{ij}_{ab}} , $$

with relative error $O(\rho)$ of either sign. It is an estimate, not a
bound, and it sees only the candidate squares the assembly evaluates.

> **Verified equation @eq:lmp2-combined-jacobi-remainder** · the occupied-extended estimate is one Jacobi step with a second-order estimator error
>
> $t^* + \Delta^{-1}\tilde g = (\Delta^{-1}\mathcal A')^2t^*$ in symbols on a
> generic 3×3 model and numerically on the block model, with the elementwise
> bound; the explicit tensor form of $\tilde g$ equals $J + \mathcal A't^0$
> in symbols ($n_o = n_v = 2$) and numerically; halving the coupling divides
> the occupied-extended estimator error by about 4 and the virtual-only one
> by about 2 (P2).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P2 |
> | Negative control | the runner removes the occupied term $\sum_{k\ne i} f_{ik}t^0_{kj}$ from $\tilde g$ ("P2: Jacobi") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::fock_dressed_gate` (the virtual-only form) |
>
> <!-- verified: id=lmp2.combined.jacobi-remainder; script=scripts/lmp2_combined_sympy.py; claim=P2: remainder identity (numeric); runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::fock_dressed_gate; registry=local-mp2 -->

## P3. The WLLH25 embedding (identities proven; upper bound of the WLLH25 linear energy disproved) {#amplitude-lmp2-combined-p3}

Split $\mathcal A = \mathcal B + \mathcal C$. $\mathcal B$ is $\mathcal A$
with the occupied off-diagonals removed: block diagonal over pairs, the
per-pair Sylvester operator of WLLH25 Eq. 45.
$\mathcal C = -(f'_{oo}\oplus f'_{oo})\otimes 1_{vv}$ is the inter-pair
coupling, $\lVert\mathcal C\rVert_2 = 2\lVert f'_{oo}\rVert_2$. WLLH25
solves $\tau = -\mathcal B^{-1}J$ per pair, keeps $\tau_D$, and solves the
kept block with the shifted right-hand side $m = \mathcal B_{KD}\tau_D$:
$\mathcal A_{KK}t_K = -J_K - \mathcal B_{KD}\tau_D$,
$t_{\mathrm{emb}} = (t_K, \tau_D)$. For any split, with
$\varrho = J + \mathcal A\,t_{\mathrm{emb}}$,

$$ \varrho_K = \mathcal C_{KD}\,\tau_D, \qquad
\varrho_D = \mathcal C_{DK}t_K + \mathcal C_{DD}\tau_D - \mathcal B_{DK}\mathcal B_{KK}^{-1}\mathcal C_{KK}t_K . \label{eq:lmp2-combined-embedding-residual} $$

So $\mathcal C = 0 \Rightarrow t_{\mathrm{emb}} = t^*$ exactly, and the
whole residual the embedding leaves is linear in the inter-pair occupied
Fock coupling it sets to zero. That is the probe's "$f_{ik}$ zeroed, WLLH
exact" anchor of section @sec:amplitude-lmp2-fock-screen-measured, now a
theorem. What it cancels exactly is the intra-pair part $\mathcal B_{DK}$
of the coupling that drives $R_D$ in P1, including the virtual Fock
coupling the screen also targets. Two energies can be formed:

- **Full Hylleraas** $E_H(t_{\mathrm{emb}})$ (the probe's "H" column):
  $E_H(t_{\mathrm{emb}}) - E^* = \varrho^{\mathsf T}\mathcal W\mathcal A^{-1}\varrho \ge 0$,
  an upper bound with error $O(\lVert\mathcal C\rVert^2)$ (ratio 4.03 when
  $f'_{oo}$ is halved), and the certificate
  $E_H(t_{\mathrm{emb}}) - E^* \le 3\lVert\varrho\rVert^2/\lambda_{\min}(\mathcal A)$.
  This is proven. It costs one matvec over $K\cup D$.
- **WLLH25 Eq 41 at convergence**,
  $E_L = \langle\mathcal W t_{\mathrm{emb}}, J\rangle$ (the "lin" column):
  $E_L - E^* = E_H - E^* - \langle\mathcal W t_{\mathrm{emb}},\varrho\rangle$,
  first order in $\mathcal C$ with no fixed sign (ratio 2.03). On the model
  it fell below $E^*$ in 10 of 28 instances, so it is not an upper
  bound.

Neither energy is guaranteed to beat the plain mask:
$E_H(t_{\mathrm{emb}}) > E_K$ in 4 of 28 model instances, so that
guarantee is disproved. The fix is a Galerkin step over the enlarged space
$K \oplus \operatorname{span}\{\tau^{(ij)}_D\}$, one contracted dropped
vector per pair:

$$ E_G = \min_{t_K,\ \alpha_{ij}} E_H\Big(t_K + \sum_{ij}\alpha_{ij}\,\tau^{(ij)}_D\Big), \qquad E^* \le E_G \le \min\big(E_K,\ E_H(t_{\mathrm{emb}})\big) . $$

The bound holds because the family contains $\alpha = 0$ and, at
$\alpha = 1$, the best $t_K$ for the fixed $\tau_D$. $E_G$ is exact at
$\mathcal C = 0$ and size-consistent (P6). The $\tau^{(ij)}_D$ enter a CG
solve as extra unknowns. The bound holds by construction; the cost in
production is unmeasured. One global $\alpha$ instead of one per pair is not size-consistent:
on the model it is off by $+2.2\times10^{-6}$ Ha for two non-interacting
fragments.

> **Verified equation @eq:lmp2-combined-embedding-residual** · the embedding residual is linear in the inter-pair coupling
>
> $\varrho_K = \mathcal C_{KD}\tau_D$ for a generic split, $\mathcal C = 0$
> gives $t_{\mathrm{emb}} = t^*$, $E_H(t_{\mathrm{emb}})$ keeps the upper
> bound and its certificate, $E_L$ undershoots $E^*$ on some instances, and
> halving $\lVert\mathcal C\rVert$ divides the $E_H$ error by about 4 and
> the $E_L$ error by about 2 (P3).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P3 |
> | Negative control | the runner flips the sign of the embedding right-hand side ("P3: exact when f_oo diagonal") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/ragged.rs::matvec_indexed` (the operator $\mathcal A$; the embedding has no production code) |
>
> <!-- verified: id=lmp2.combined.embedding-residual; script=scripts/lmp2_combined_sympy.py; claim=P3: rho_K; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/ragged.rs::matvec_indexed; registry=local-mp2 -->

## P4. λ-Boys (proven; global optimality open) {#amplitude-lmp2-combined-p4}

Maximize the λ-Boys objective, Eq. 9 of WLLH25. Its Eq. 8 is the same up
to the rotation-invariant $\sum_i\langle i|r^2|i\rangle$ and $\lVert f_{oo}\rVert_F^2$.

$$ G(U) = \sum_{x=1}^{3}\sum_i (r^x_{ii})^2 + \lambda\sum_i f_{ii}^2 , \qquad \Phi = \sum_{i\ne j} f_{ij}^2 = \lVert f_{oo}\rVert_F^2 - \sum_i f_{ii}^2 . $$

For $U = 1 + \eta$, $\eta^{\mathsf T} = -\eta$, and any symmetric $X$,
$\partial_{\eta_{pq}}\sum_i X_{ii}^2 = 4X_{pq}(X_{qq} - X_{pp})$, so

$$ \frac{\partial G}{\partial\eta_{pq}} = 4\sum_{X}w_X\,X_{pq}\,(X_{qq} - X_{pp}), \qquad (X, w_X) \in \{(r^x, 1),\ (f_{oo}, \lambda)\} , \label{eq:lmp2-combined-lambda-boys} $$

and stationarity is
$\sum_x r^x_{pq}(r^x_{pp} - r^x_{qq}) + \lambda f_{pq}(f_{pp} - f_{qq}) = 0$
for all $p<q$. A 2×2 rotation by $\theta$ gives, with
$d_X = (X_{pp} - X_{qq})/2$ and $c_X = X_{pq}$,

$$ G(\theta) = \mathrm{const} + a\cos4\theta + b\sin4\theta, \qquad a = \sum_X w_X(d_X^2 - c_X^2), \quad b = 2\sum_X w_X d_Xc_X , $$

maximized at $4\theta = \operatorname{atan2}(b, a)$. λ-Boys is therefore
Boys with a fourth "dipole component" $\sqrt\lambda\,f_{oo}$, and each
Jacobi step is the exact maximum in its plane, so $G$ rises monotonically.
The paper's remark that Jacobi sweeps converge less easily concerns rate,
not the step. Each step maximizes only within its plane, so a converged
sweep is a stationary point that is a maximum in every 2×2 plane, not
necessarily a global maximum; global optimality is an open question.

If $U_\lambda$ is a global optimum, summing the optimality inequalities at
$\lambda_1 < \lambda_2$ gives $(\lambda_2 - \lambda_1)(\Phi_1 - \Phi_2) \ge 0$,
so $\Phi$ is non-increasing in $\lambda$. That holds under the
global-optimum assumption, and is seen on the model with warm starts. λ enters P1–P3 through

$$ \lVert\mathcal C\rVert_2 = 2\lVert f'_{oo}\rVert_2 \le 2\sqrt{\Phi} , $$

the occupied part of $\mathcal A'$ in P2's $\rho$, and the embedding
residual of P3:
$\lVert\varrho\rVert \le 2\sqrt\Phi\,\big[2\lVert\tau_D\rVert + (1+\gamma)\lVert t_K\rVert\big]$,
$\gamma = \lVert\mathcal B_{DK}\mathcal B_{KK}^{-1}\rVert_2$, hence
$E_H(t_{\mathrm{emb}}) - E^* \le 12\,\Phi\,[2\lVert\tau_D\rVert + (1+\gamma)\lVert t_K\rVert]^2/\lambda_{\min}(\mathcal A)$.
This bound is not size-extensive. It fixes the scaling with $\Phi$, not the
magnitude.

> **Verified equation @eq:lmp2-combined-lambda-boys** · the λ-Boys gradient and Jacobi angle
>
> The gradient formula, the 2×2 Jacobi form $G(\theta)$ in symbols, and a
> numerical Jacobi λ-Boys sweep that is stationary and rises monotonically
> (P4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_combined_sympy.py`, P4 |
> | Negative control | the runner drops the factor 4 from the gradient ("P4: gradient") and sets the $f_{oo}$ weight to 0 ("P4: stationarity") |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::localized_spaces` (plain Boys, $\lambda = 0$; λ-Boys is section @sec:amplitude-lmp2-lambda-boys) |
>
> <!-- verified: id=lmp2.combined.lambda-boys; script=scripts/lmp2_combined_sympy.py; claim=P4: gradient; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::localized_spaces; registry=local-mp2 -->

## P5. Composition {#amplitude-lmp2-combined-p5}

| piece | error term it attacks | effect on the bound |
|---|---|---|
| screen (virtual-only) | $\mathcal B$-driven part of $t^*_D$, by moving it into $K$ | $E^* \le E_{K'} \le E_K$ (P1) |
| screen (+ occupied) | also the first-order $\mathcal C$-driven part ($\varrho_D \approx \mathcal C_{DK}t_K$) | same |
| embedding | $\mathcal B_{KD}$ coupling of the remaining $D$, exactly | $E_H(t_{\mathrm{emb}})$ an upper bound; $E_L$ no bound |
| λ-Boys | $\lVert\mathcal C\rVert \le 2\sqrt\Phi$: the embedding's whole residual and the occupied part of the screen's | none directly; it changes $J$, hence $K$ |

The screen and the embedding overlap in what they target: both act on the
large dropped exact amplitudes. The screen removes them from $D$, and the
embedding approximates them by $\tau_D$. After a screen the embedding has
less left to recover, consistent with the probe's further factor 1.2–4.8
instead of 1.6–5.6. The occupied extension and λ-Boys overlap too: both reduce
the $\mathcal C$-driven error, one by keeping those amplitudes and the
other by shrinking $\mathcal C$. At a fixed threshold the occupied
extension lowers the dropped-amplitude norm only modestly (P2).

The combination is an upper bound when every piece ends in a Galerkin
solve over a subspace of the universe (plain, screened or
embedded-Galerkin, $E_G$), or when the embedding is scored with the full
Hylleraas form. With the WLLH25 linear energy it is not, and a combined
method using it needs the a-posteriori estimate of P2 and P3 to be trusted.
λ trades $\Phi$ against locality: larger $\lambda$ spreads $J$ and enlarges
$K$ at fixed $\varepsilon$. Where the optimum lies is an open, empirical question.

## P6. Limits and invariances {#amplitude-lmp2-combined-p6}

- $\varepsilon \to 0$: the Eq-8 mask keeps everything, the screen is inert, and
  $D = \emptyset$ makes $\tau_D$ and $m$ vanish. Every combination is exact
  RI-MP2 in any orthonormal orbitals.
- $\kappa \to \infty$, embedding off: the screen adds nothing and the
  pattern is the Eq-8 mask of section @sec:amplitude-lmp2-algorithm.
- $\lambda = 0$: $G$ is the Boys functional, by definition.
- The untruncated $E^* = -J^{\mathsf T}\mathcal W\mathcal A^{-1}J$ is
  invariant under orthogonal $U_o$, $U_v$:
  $Q = U_o\otimes U_o\otimes U_v\otimes U_v$ is orthogonal,
  $J\to Q^{\mathsf T}J$, $\mathcal A\to Q^{\mathsf T}\mathcal AQ$, and $Q$
  commutes with $\mathcal S$. The truncated energies are not invariant.
- Size-consistency, under an assumption on the orbitals. Take non-interacting
  fragments, with every cross-fragment $(ia|jb)$ and Fock element exactly
  zero, and fragment-pure orbitals identical to the isolated fragments'.
  Then $\mathcal A$ preserves the fragment label of each index, $J$ lives
  only in the all-A and all-B sectors, and the screen's $\mathcal A' t^0$
  and the per-pair $\mathcal B^{-1}$ stay in them. Every arm (plain,
  screen, $E_H(t_{\mathrm{emb}})$, $E_L$, $E_G$ with per-pair $\alpha$)
  gives $E(A{+}B) = E(A) + E(B)$. The localizer half of the assumption
  holds for Jacobi λ-Boys: on a cross-fragment plane $c_X = 0$ for every
  $X$, so $b = 0$, $a > 0$ and $\theta = 0$ exactly. Fragment orbitals are
  never mixed, and the intra-fragment rotations run in the same order as
  in the isolated run, so the result matches it up to the stopping
  tolerance. A localizer that can stop at a saddle (PySCF's default Boys
  optimizer was observed to) gives no such guarantee. The same argument
  needs the VV-HV construction, the candidate squares and the pair gate to
  be fragment-local.

The script checks each limit numerically on the block model: at
$\varepsilon = 0$ every arm gives $E^*$, at $\kappa \to \infty$ the screened
pattern is the Eq-8 pattern, rotating the orbitals leaves $E^*$ unchanged
(mutation "P6: rotational invariance"), and non-interacting fragments give
$E(A{+}B) = E(A) + E(B)$ to a relative $10^{-11}$ for every arm.

## Sources

| Source | Used for |
|---|---|
| `scripts/lmp2_combined_sympy.py`, P1 to P6 | verified equations @eq:lmp2-combined-truncation-error, @eq:lmp2-combined-jacobi-remainder, @eq:lmp2-combined-embedding-residual and @eq:lmp2-combined-lambda-boys |
| `crates/ferric-mp2/tests/lmp2_fock_screen_probe.rs` | the dense-oracle comparison of the screen and the embedding (section @sec:amplitude-lmp2-fock-screen-measured) |
| Wang, Ling, Shi, Shen, Wang, Yang, Li and Head-Gordon, JCTC 21, 10910 (2025) (WLLH25) | the block-Kapuy embedding (Eqs. 43–49) and the λ-Boys objective (Eqs. 8–9) |
| Subotnik and Head-Gordon, JCP 122, 034109 (2005) | the earlier work the λ-Boys objective follows |
