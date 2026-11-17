# RI-MP2 gradient with exact and RI-JK references; SCS-MP2 {#rimp2-gradient}

<!-- status: ready; registry: ri-mp2-gradient, scs-mp2-gradient, att-rimp2-gradient, terfc-rimp2-gradient, scs-2terfc-gradient;
     theory: crates/ferric-mp2/theory.md#"9. Analytic RI-MP2 gradient and Z-vector", "9.1 Fitted (RI-JK) reference", "9.2 Spin-component-scaled (SCS-MP2) gradient", "9.4 Attenuated (erfc) RI-MP2 gradient", "9.5 terfc-attenuated RI-MP2 gradient", "9.6 SCS-MP2(2terfc) gradient" -->

The RI-MP2 energy depends on the nuclear positions in three ways. The basis
functions and auxiliary functions sit on the nuclei and move with them. The
Hartree–Fock orbitals re-mix as the nuclei move. And the orbital energies in
the MP2 denominators shift with the Fock matrix. A gradient has to collect
all three. Section @sec:lagrangian-zvector showed how the Lagrangian turns the orbital part into
one Z-vector solve; this section supplies the MP2-specific pieces: the
densities that say how the energy responds to each orbital and integral, the
right-hand side $X$, and the assembly.

The same code gives three further gradients. On a fitted (RI-JK) reference
the four-center $J$ and $K$ are replaced by their fits. For SCS-MP2 the
amplitude weight $2t - \bar t$ is replaced by a spin-scaled one. For the
attenuated and terfc variants the correlation uses a different operator
from the reference, and each term takes the operator of what it
differentiates. The subsections go in that order: densities and the
assembly (@sec:rimp2-gradient-densities-right-hand-side–@sec:rimp2-gradient-algorithm), the fitted reference (@sec:rimp2-gradient-fitted-ri-jk-reference), SCS-MP2 (@sec:rimp2-gradient-scs-mp2),
the attenuated kernels (@sec:rimp2-gradient-attenuated-erfc-ri-mp2).

| Symbol | Meaning | Code name |
|---|---|---|
| $t_{ij}^{ab}$ | $(ia\vert jb)/\Delta_{ij}^{ab}$, with $\Delta_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$ as written in `theory.md` | `t2` |
| $\tilde T$ | amplitude weight, $2t - \bar t$ for MP2 | |
| $x^P_{ia}$ | $\sum_{jb}\tilde T_{ij}^{ab}B^P_{jb}$ | `x_ov` |
| $P_{ij}, P_{ab}$ | unrelaxed occupied and virtual densities | `p_oo`, `p_vv` |
| $I$ | Lagrangian matrix | `imat` |
| $G[D]$ | $J[D] - \tfrac12K[D]$, four-center or fitted | |

The denominator sign follows `crates/ferric-mp2/theory.md`, where $\Delta$
is negative; the book's positive-gap convention (chapter @sec:mp2) is the negative
of it. With this sign $t$ is positive where $(ia|jb)$ is negative, and
$E^{(2)} = \sum K\tilde T$ with $K_{ij}^{ab} = (ia|jb)$ is negative.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,relaxed_ri_gradient,scs_mp2_gradient_analytical,total_rimp2_gradient}` |
| Rust (attenuated) | `ferric-mp2/src/att_rimp2_gradient.rs::{att_rimp2_gradient_analytical,total_att_rimp2_gradient_on,terfc_rimp2_gradient_analytical,total_terfc_rimp2_gradient_on}` |
| Rust (SCS-MP2(2terfc)) | `ferric-mp2/src/scs_2terfc_gradient.rs::{scs_mp2_2terfc_gradient_analytical,total_scs_mp2_2terfc_gradient_on}` |
| CLI | `method.kind = "rimp2"`, `"scs-mp2"` or `"att-rimp2"` with `task = "optimize"` |
| Python | `run_rimp2(with_gradient=True)`, `run_optimize(method="rimp2")`, `run_optimize(method="scs-mp2")`, `run_optimize(method="att-rimp2")` |

## Densities and the right-hand side {#rimp2-gradient-densities-right-hand-side}

### The unrelaxed densities

The MP2 energy depends on the orbital energies only through the
denominators. Differentiate $E^{(2)} = \sum K\tilde T$ with respect to one
denominator: $\tilde T$ is $K/\Delta$ times fixed weights, so
$\partial E/\partial\Delta_{ij}^{ab} = -t_{ij}^{ab}\tilde T_{ij}^{ab}$. An
occupied orbital energy $\varepsilon_k$ enters $\Delta_{ij}^{ab}$ with a plus
sign whenever $i = k$ or $j = k$. Sum those two cases with the chain rule.
The pair symmetry $t_{ij}^{ab} = t_{ji}^{ba}$ makes them equal, so

$$ \frac{\partial E^{(2)}}{\partial\varepsilon_k} = 2P_{kk}, \qquad \frac{\partial E^{(2)}}{\partial\varepsilon_c} = 2P_{cc}, $$

with the diagonals of

$$ P_{ij} = -\sum_{kab}t_{ik}^{ab}\tilde T_{jk}^{ab}, \qquad P_{ab} = \sum_{ijc}t_{ij}^{ac}\tilde T_{ij}^{bc} . \label{eq:rimp2-gradient-1} $$

The off-diagonal elements are the derivatives with respect to the
off-diagonal Fock elements, $F_{ij}$ and $F_{ab}$, which would couple the
amplitudes in a non-canonical basis (section @sec:amplitude-lmp2-amplitude-equations-energy). $P_{ij}$ is the
depletion of the occupied orbitals: for MP2 its diagonal is negative
($\sum_{ab}t^{ab}t^{ba} \le \sum_{ab}(t^{ab})^2$ by Cauchy–Schwarz, so
$2t - \bar t$ weighted by $t$ sums to a positive number), and correlation
moves electron density out of them. $P_{ab}$ is the matching population of
the virtuals, with a positive diagonal by the same inequality.

### The Lagrangian matrix

The energy also depends on the orbitals through the fitted integrals
$B^P_{ia}$, each of which carries one occupied and one virtual orbital
coefficient. The derivative of $E^{(2)}$ with respect to $B$ is carried by
$x^P_{ia} = \sum_{jb}\tilde T_{ij}^{ab}B^P_{jb}$. Contract $x$ back with the
full-MO factor $B^P_{pq}$ and you get the response of the energy to mixing
any orbital $q$ into an occupied orbital $i$ or a virtual orbital $a$:

$$ I_{qi} = -2\sum_{Pa}x^P_{ia}B^P_{qa}, \qquad I_{qa} = -2\sum_{Pi}x^P_{ia}B^P_{qi} . \label{eq:rimp2-gradient-2} $$

This is PySCF's conventional `Imat` with the four-index integrals replaced by
$\sum_P B^P B^P$: the $(r,s)$ legs of the two-particle density collapse onto
$x$, and no four-index tensor is formed. The factor 2 is the closed-shell
weight of both spins. Frozen-core columns are zero (the core is absent from
the amplitudes); frozen-core rows are not.

### The right-hand side

A rotation $\kappa_{ai}$ adds $\kappa\,C_a$ to orbital $i$ and subtracts
$\kappa\,C_i$ from orbital $a$. Through $B$, that changes the energy by the
difference $I_{ia} - I_{ai}$. The rotation also changes the Hartree–Fock
density, which changes the Fock matrix by $(2J - K)$ of the density change,
which changes the orbital energies that the correlation density weights.
Together:

$$ X_{ai} = \big[\mathbf C^{\mathsf T}(2J - K)[D^{\mathrm{corr}}]\,\mathbf C\big]_{ai} + I_{ia} - I_{ai}, \qquad D^{\mathrm{corr}} = \mathbf C\,(P + P^{\mathsf T} + D_{kI})\,\mathbf C^{\mathsf T} . \label{eq:mp2-grad-assembly} $$

With a frozen core the correlation density gains the core-active block
$D_{kI} = I_{kI}/(\varepsilon_k - \varepsilon_I)$
(`ferric-mp2/src/zvector.rs::frozen_core_block`). The MP2 energy is not
stationary under core–active rotations, and because both orbitals are
occupied the response is diagonal: it needs no CPHF, only a division. A
degenerate core and active orbital (gap below $10^{-8}$) makes the
frozen-core gradient undefined, and the code returns an error. $X$, $z$ and
the energy-weighted density run over every occupied orbital. The Z-vector
equation is (@eq:zvec-hessian-product).

### The gradient

With $z$ from (@eq:zvec-hessian-product), the assembled derivative has five parts:

| Part | Density or weight | Derivative integral |
|---|---|---|
| one-electron | $D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}}$ | $h'$ |
| overlap | $\mathrm{im1} - \zeta - v^{\mathrm{s1occ}}$ | $S'$ |
| separable two-electron | $(D^{\mathrm{HF}}, D^{\mathrm{HF}} + 2D^{\mathrm{corr}}_{\mathrm{rel}})$ | $(\mu\nu\vert\lambda\sigma)'$ |
| three-center RI response | $4y^P_{ia}$ | $(P\vert ia)'$ |
| two-center RI response | $-2\sum_{ia}(V^{-1}u)^P_{ia}y^Q_{ia}$ | $V'_{PQ}$ |

Here $D^{\mathrm{corr}}_{\mathrm{rel}} = D^{\mathrm{corr}} + D^z$, $y = V^{-1/2}x$
and $u^P_{ia} = (P|ia)$. The overlap weight has three pieces, all from
orthonormality, the $W$ of (@eq:lagrangian-zvector-1):

- $\mathrm{im1} = \mathbf C\tilde I\mathbf C^{\mathsf T}$, where $\tilde I$ is
  $I$ with its virtual–occupied block replaced by the transpose of its
  occupied–virtual block.
- $\zeta$, the orbital-energy-weighted relaxed density:
  $\tfrac12(\varepsilon_p + \varepsilon_q)D^{\mathrm{corr}}_{\mathrm{rel},pq}$
  in the MO basis, except $\varepsilon_i D^{\mathrm{corr}}_{\mathrm{rel},ai}$ on
  the occupied–virtual blocks, plus
  the Hartree–Fock $\sum_i 2\varepsilon_i\mathbf C_i\mathbf C_i^{\mathsf T}$.
- $v^{\mathrm{s1occ}} = P_{\mathrm{occ}}(2J - K)[D^{\mathrm{corr}}_{\mathrm{rel}}]P_{\mathrm{occ}}$,
  with $P_{\mathrm{occ}} = \mathbf C_o\mathbf C_o^{\mathsf T}$: the Fock
  matrix depends on $D^{\mathrm{HF}}$, which moves with the occupied orbitals
  as the overlap changes.

The separable two-electron part is the bilinear form
$B(D_1, D_2) = \tfrac12\operatorname{Tr}[D_1(J - \tfrac12K)[D_2]]$
differentiated through the integrals only. The $J$ and $K$ of
$2D^{\mathrm{corr}}_{\mathrm{rel}}$ come from linearity: twice those of
$D^{\mathrm{corr}}$ (built for the right-hand side) plus twice those of the
converged $D^z$ (returned by the Z-vector solve). The sum differs from a
direct build only by which quartets the absolute screen
($\text{bound}\cdot\max|D| < 10^{-12}$) drops in each of the builds.

> **Verified equation @eq:mp2-grad-assembly** · the assembled gradient
>
> On a model where $S$, $h$, the four-center ERIs, the RI factors and the
> metric all depend on $x$, the five parts above equal a Richardson
> difference of the re-solved $E_{\mathrm{HF}} + E^{(2)}$ for MP2 and for
> SCS-MP2 $(6/5, 1/3)$, with and without a frozen core.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part G |
> | Negative control | the runner's mutations of the right-hand-side sign of $I_{ai}$, the SCS density weight and the $\zeta$ occupied-virtual weight each turn G red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,relaxed_ri_gradient,integral_response_gradient_3c2c}`, `ferric-mp2/src/zvector.rs::{build_imat_ri,frozen_core_block}` |
> | Code test | `rimp2_gradient_h2o_ccpvdz_vs_orca_and_pyscf` in `crates/ferric-mp2/tests/validation_rimp2_gradient.rs` |
>
> <!-- verified: id=mp2.grad.assembly; script=scripts/mp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,relaxed_ri_gradient,integral_response_gradient_3c2c}; rust_test=ferric-mp2/tests/validation_rimp2_gradient.rs::rimp2_gradient_h2o_ccpvdz_vs_orca_and_pyscf; registry=ri-mp2-gradient -->

The script models the RI terms at the MO level. The AO bookkeeping in
`ferric-mp2/src/gradient.rs::integral_response_gradient_3c2c` (its $2y$ and
symmetrized $\Gamma$) is checked only by the Rust tests against PySCF and
ORCA.

**Translational invariance.** Sum the gradient over all atoms for one
Cartesian direction. A rigid translation of the molecule moves every basis
and auxiliary function together and leaves every integral unchanged, so the
centre derivatives of each integral sum to zero. Each of the five parts is a
fixed density contracted with derivative integrals, so each part sums to
zero on its own, and so does the gradient. The Rust tests check this sum
(`test_analytical_gradient_translational_invariance` in
`crates/ferric-mp2/src/gradient.rs`).

### The overlap weight in code

`ferric-mp2/src/gradient.rs::mp2_relaxed_lagrangian_gradient` adds the three
overlap pieces into one weight and contracts it once, then forms the second
density of the bilinear two-electron term:

<!-- excerpt: ferric-mp2/src/gradient.rs::mp2_relaxed_lagrangian_gradient -->
```rust
let w_overlap = &im1 - &zeta_ao - &vhf_s1occ;
grad += &overlap_deriv_contract(obs, &w_overlap)?;
```

| Code | Equation |
|---|---|
| `im1` | $\mathbf C\tilde I\mathbf C^{\mathsf T}$ |
| `zeta_ao` | $\zeta$, including the Hartree–Fock energy-weighted density |
| `vhf_s1occ` | $P_{\mathrm{occ}}(2J - K)[D^{\mathrm{corr}}_{\mathrm{rel}}]P_{\mathrm{occ}}$ |
| `overlap_deriv_contract(obs, M)` | $\sum_{\mu\nu}S'_{\mu\nu}(M + M^{\mathsf T})_{\mu\nu}$ |

`im1` is not symmetric, so the contraction symmetrizes it. The
`(2J − K)` in `vhf_s1occ` is built as `2.0 * j_corr - k_corr` from the
`corr_jk` pair of the Z-vector solution: PySCF's `get_veff` of the doubled
density $2D^{\mathrm{corr}}_{\mathrm{rel}}$ is $J[2D] - \tfrac12K[2D]$, which
equals $2J[D] - K[D]$.

## The algorithm {#rimp2-gradient-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF, four-center or fitted | `solve_rhf` |
| 2 | $B$, $t$, $x$, $P$ | `ferric-mp2/src/rimp2.rs::compute_mp2_intermediates_ov_only` (the RI-dRPA gradient fills the same struct through `Mp2Intermediates::from_ri_densities`, section @sec:rpa-gradient) |
| 3 | $I$, frozen-core block | `build_imat_ri`, `frozen_core_block` |
| 4 | $(J, K)[D^{\mathrm{corr}}]$, $X$, then $z$ | `solve_zvector_timed` |
| 5 | one-electron, overlap and separable two-electron terms | `mp2_relaxed_lagrangian_gradient` |
| 6 | three- and two-center RI response | `integral_response_gradient_3c2c` |
| 7 | add the RHF gradient | `total_rimp2_gradient` |

Step 2 builds only the occupied–virtual intermediates: the gradient never
reads $B^P_{ij}$ or $B^P_{ab}$, so the $(n_{\mathrm{aux}}, n_v^2)$ block is
never formed. Step 3 forms $I$ as matrix products over fixed blocks of rows,
so the result does not depend on the number of threads.

## The fitted (RI-JK) reference {#rimp2-gradient-fitted-ri-jk-reference}

With `[scf] jk = "rijk"` (Python `scf_jk="rijk"`) the reference minimizes the
fitted energy
$E_{\mathrm{HF}} = \operatorname{Tr}[hD] + Q(D) + V_{nn}$, with
$Q(D) = \tfrac12\operatorname{Tr}[D\,G^{\mathrm{fit}}(D)]$. Its Fock matrix is
$h + G^{\mathrm{fit}}(D)$, because $G^{\mathrm{fit}}$ is linear and
self-adjoint: both fits use the same factor on each side, so
$\operatorname{Tr}[D_1G(D_2)] = \operatorname{Tr}[D_2G(D_1)]$. The MP2
Lagrangian meets the reference only through $F$: the orbital energies, the
condition $F_{ai} = 0$ that $z$ enforces, and orthonormality. So the
derivation is unchanged, and each four-center object is replaced by its fit:

| Term | Four-center | Fitted |
|---|---|---|
| right-hand side | $(2J - K)[D^{\mathrm{corr}}]$ | $2G^{\mathrm{fit}}(D^{\mathrm{corr}})$ |
| Hessian coupling | $(4J - K - K^{\mathsf T})[D^z]$ | $4G^{\mathrm{fit}}(D^z)$ |
| potential term | $(J - \tfrac12K)[2D^{\mathrm{corr}}_{\mathrm{rel}}]$ | $G^{\mathrm{fit}}(2D^{\mathrm{corr}}_{\mathrm{rel}})$ |
| separable two-electron | $\Gamma(D_1, D_2)\,(\mu\nu\vert\lambda\sigma)'$ | $\partial_R B(D_1, D_2)$ at fixed $D_1, D_2$ |

The fitted separable term is the derivative of a bilinear form at fixed
densities. ferric already has the derivative of the quadratic form $Q$ (the
RI-J and RI-K gradients of section @sec:df-jk-gradient), so it rewrites $B$ in terms of $Q$.
Put $u = \tfrac12(D_1 + D_2)$ and $v = \tfrac12(D_1 - D_2)$, expand
$Q(u) - Q(v)$, and use self-adjointness to combine the cross terms:

$$ Q(u) - Q(v) = \tfrac12\operatorname{Tr}\big[(u - v)\,G(u + v)\big] = \tfrac12\operatorname{Tr}\big[D_2\,G(D_1)\big] = B(D_1, D_2) . $$

With $D_1 = D^{\mathrm{HF}}$ and $D_2 = D^{\mathrm{HF}} + 2D^{\mathrm{corr}}_{\mathrm{rel}}$,
$u = D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}}$ and
$v = -D^{\mathrm{corr}}_{\mathrm{rel}}$. The RI-K gradient factors its density
as $XX^{\mathsf T}$, so it needs a positive semidefinite density. $v$ is
always indefinite, and $u$ is when a natural occupation of the relaxed
density is negative. Split such a density by eigenvalue sign, $D = P - N$
with $P, N \succeq 0$. Expanding $Q(P \pm N)$ gives
$Q(P) + Q(N) \pm 2B(P, N)$; eliminate $B(P, N)$ between the two signs:

$$ Q_K(P - N) = 2Q_K(P) + 2Q_K(N) - Q_K(P + N) . $$

Both identities hold for any symmetric bilinear form. The MP2 integral
response on its own RI basis does not change. The result is the gradient of
the fitted energy, not an approximation to the four-center gradient.

**Which reference the CLI uses.** With `[scf] jk` omitted, a closed-shell
MP2-family run takes the fitted reference on the JK set matched to the
orbital basis, for `energy` and for geometry tasks alike
(`Config::omitted_scf_jk` in `crates/ferric-cli/src/config.rs`). The
open-shell `rimp2`, `att-rimp2` and `scs-mp2-2terfc` gradient tasks take it
too (section @sec:u-rimp2-gradient). Any other open-shell run, or an input that sets a key only
the four-center route reads, keeps the four-center reference. Python
`run_optimize(method="rimp2")` resolves an unset `scf_jk` by the same rule;
`run_rimp2` defaults to `"exact"`. The fitted gradient's peak memory is about
twice that of the four-center route (2.1 times at dodecane/cc-pVDZ, 8.3
against 4.0 GB), so the gradient declares that peak before the reference SCF
and refuses a budget below it.

## SCS-MP2 {#rimp2-gradient-scs-mp2}

Spin-component scaling (section @sec:scs-and-kappa) weights the opposite-spin and
same-spin energies separately. Write both in the amplitudes of this
section, $E_{\mathrm{OS}} = \sum Kt$ and $E_{\mathrm{SS}} = \sum K(t - \bar t)$,
and collect the coefficient of $K$:
$E_{\mathrm{SCS}} = c_{\mathrm{OS}}E_{\mathrm{OS}} + c_{\mathrm{SS}}E_{\mathrm{SS}} = \sum K\tilde T$ with

$$ \tilde T = (c_{\mathrm{OS}} + c_{\mathrm{SS}})\,t - c_{\mathrm{SS}}\,\bar t, \qquad \bar t_{ij}^{ab} = t_{ij}^{ba}, \label{eq:mp2-grad-scs-weights} $$

which is $2t - \bar t$ at $(1, 1)$. The gradient uses the amplitudes in two
places, and both carry over with $\tilde T$ in place of $2t - \bar t$. In
the integral derivative, $\tilde T$ enters $x$, and through $x$ the
Lagrangian matrix and the RI response. In the denominator derivative it
enters the second slot of (@eq:rimp2-gradient-1).

The densities need no new code. $P(t, \tilde T)$ is linear in its second
slot, so expand $\tilde T$ and write
$P(t, \bar t)$ through the MP2 density
$P^{\mathrm{MP2}} = 2P(t, t) - P(t, \bar t)$:

$$ P(t, \tilde T) = (c_{\mathrm{OS}} + c_{\mathrm{SS}})P^{\mathrm{OS}} - c_{\mathrm{SS}}\big(2P^{\mathrm{OS}} - P^{\mathrm{MP2}}\big) = c_{\mathrm{SS}}P^{\mathrm{MP2}} + (c_{\mathrm{OS}} - c_{\mathrm{SS}})P^{\mathrm{OS}}, \qquad P^{\mathrm{OS}} = P(t, t) . $$

Nothing else changes. `ferric-mp2/src/rimp2.rs::apply_spin_scaling` applies
the density part as the last line says:

<!-- excerpt: ferric-mp2/src/rimp2.rs::apply_spin_scaling -->
```rust
self.p_oo = c_ss * &self.p_oo + (c_os - c_ss) * &os_oo;
self.p_vv = c_ss * &self.p_vv + (c_os - c_ss) * &os_vv;
```

| Symbol | Code |
|---|---|
| $c_{\mathrm{OS}}, c_{\mathrm{SS}}$ | `c_os`, `c_ss` |
| $P^{\mathrm{MP2}}$ (before scaling) | `self.p_oo`, `self.p_vv` |
| $P^{\mathrm{OS}}$ | `os_oo`, `os_vv` |

The $x$ part is `ferric-mp2/src/zvector.rs::build_x_ov_weighted`, which
forms $\tilde T$ for one occupied $i$ at a time as
`d * t_ij_ab - e * t_ij_ba`, with $(d, e) = (2, 1)$ for MP2. At $(1, 1)$,
`apply_spin_scaling` returns without touching the intermediates, so the
RI-MP2 gradient goes through the same code bit for bit
(`scs_unit_scaling_is_the_rimp2_gradient_bit_for_bit`).

> **Verified equation @eq:mp2-grad-scs-weights** · the SCS weights
>
> $\partial E/\partial K_{ij}^{ab}$ over both slots of a symmetric integral
> is $2\tilde T$, and the orbital-energy derivatives are the diagonals of
> $P + P^{\mathsf T}$ with $\tilde T$ in the second slot.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part S (symbolic weights) and part G (the full SCS gradient against finite differences) |
> | Negative control | the runner's mutation $(c_{\mathrm{OS}} - c_{\mathrm{SS}}) \to c_{\mathrm{OS}}$ in the SCS density weight turns G red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::apply_spin_scaling`, `ferric-mp2/src/zvector.rs::build_x_ov_weighted` |
> | Code test | `scs_gradient_matches_fd_water_ccpvdz` in `crates/ferric-mp2/tests/scs_mp2_gradient.rs` |
>
> <!-- verified: id=mp2.grad.scs-weights; script=scripts/mp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::apply_spin_scaling; rust_test=ferric-mp2/tests/scs_mp2_gradient.rs::scs_gradient_matches_fd_water_ccpvdz; registry=scs-mp2-gradient -->

## Attenuated (erfc) RI-MP2 {#rimp2-gradient-attenuated-erfc-ri-mp2}

The attenuated energy is $E_{\mathrm{HF}}[1/r] + s\,E_c[\operatorname{erfc}(\omega r)/r]$.
The reference is Coulomb RHF. The correlation uses $\operatorname{erfc}(\omega r)/r$
in both the $(P\vert ia)$ integrals and the metric $V_{PQ}$. Two operators
now appear in one Lagrangian, and each term takes the operator of what it
differentiates:

| Term | Operator |
|---|---|
| amplitudes, $x$, $P$, $I$ (@eq:rimp2-gradient-1, @eq:rimp2-gradient-2) | erfc |
| three- and two-center RI response | erfc |
| $(2J - K)[D^{\mathrm{corr}}]$ in (@eq:mp2-grad-assembly), the Z-vector Hessian, the separable two-electron term | Coulomb |

Why does the right-hand side mix them? The amplitudes and $I$ differentiate
$E_c$, which is built on erfc. The $(2J - K)$ term, the Hessian and the
separable term come from the Hartree–Fock energy and its stationarity
condition $F_{ai}[1/r] = 0$, which are Coulomb. The orbital energies in
$E_c$'s denominators are Coulomb Fock eigenvalues, so their response is
Coulomb too. In the code the correlation operator is the `op` argument and
the four-center terms run on `bounds.op`, the operator of the reference's
Schwarz bounds and $J/K$ route. The scaling $s$ enters as
$c_{\mathrm{OS}} = c_{\mathrm{SS}} = s$ in (@eq:mp2-grad-scs-weights). A fitted reference
changes only the Coulomb rows, as in @sec:rimp2-gradient-fitted-ri-jk-reference.

**The $\omega \to 0$ limit.** Expand the kernel:
$\operatorname{erfc}(\omega r)/r = 1/r - 2\omega/\sqrt\pi + O(\omega^3r^2)$.
At $\omega = 0$ the erfc integrals are the Coulomb ones and the gradient is
the RI-MP2 gradient. To first order the kernel shifts by a constant. That
shift moves $(P|\mu\nu)$ by an amount proportional to $S_{\mu\nu}$, which is
zero in $(P|ia)$ because $S_{ia} = 0$, and moves $V$ by a rank-one term. At
$\omega = 10^{-7}$ Bohr⁻¹ the two gradients agree to $1.6\times10^{-13}$
Ha/Bohr (`att_gradient_matches_rimp2_in_the_trivial_limit`). The QQR-3
screened energy (`screen_thresh`) is not differentiated and is refused for
gradient tasks. No erf or erfc nuclear attraction enters: $h$ is the Coulomb
core Hamiltonian. The derivation and the operator table are in `theory.md`
section 9.4. `scripts/att_rimp2_gradient_sympy.py` proves the erfc chain
rule and the $\omega \to 0$ limit, and checks the operator split on a
one-dimensional s-Gaussian model against finite differences.

**terfc.** With `att_operator = "terfc"` the correlation kernel is
$\operatorname{terfc}(r; r_0, \omega)/r$ and there is no scaling. The
operator table above holds with terfc in place of erfc (`theory.md` section
9.5). The 3c/2c derivative integrals come from the terf table engine: the
McMurchie-Davidson pass with the Hermite derivative
$2a\,E^{i+1,j}_t - i\,E^{i-1,j}_t$, $R$ one order higher, terfc formed as
Coulomb minus terf in $R$, and the third centre by translational invariance
(`crates/ferric-integrals/theory.md`, "Derivative integrals of terf and
terfc"). At $r_0 \to 0$ terfc becomes erfc, and terf is even in $r_0$, so the
gap closes as $r_0^2$. At $r_0 = 10^{-6}$ Bohr the two gradients agree to
$1.2\times10^{-14}$ Ha/Bohr
(`terfc_gradient_matches_erfc_in_the_r0_to_zero_limit`). The open-shell terfc
gradient is section @sec:u-rimp2-gradient-terfc-scs-mp2-2terfc.

> **Verified equation {@eq:mp2-grad-scs-weights}b** · the terfc RI-MP2 gradient
>
> On a one-dimensional model with real $s$-Gaussian Coulomb and terfc
> integrals (the terfc closed forms checked against a $k$-space quadrature of
> the kernel's Fourier transform), the analytic Lagrangian equals a
> Richardson difference of the re-solved $E_{\mathrm{HF}} + E_c[\mathrm{terfc}]$
> at $\omega r_0 = 1/\sqrt2$ and 4, with and without a frozen core.
>
> | | |
> |---|---|
> | Proof | `scripts/terfc_rimp2_gradient_sympy.py`, parts K, H and L |
> | Negative control | the runner's mutations (metric derivative dropped or on Coulomb, four-centre term or Z-vector coupling on terfc, terfc = C + terf, the $-i\,E^{i-1,j}$ term dropped) each turn K, H or L red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/att_rimp2_gradient.rs::terfc_rimp2_gradient_analytical` |
> | Code test | `terfc_gradient_matches_fd_water` in `crates/ferric-mp2/tests/terfc_rimp2_gradient_fd.rs` |
>
> <!-- verified: id=mp2.grad.terfc; script=scripts/terfc_rimp2_gradient_sympy.py; claim=L; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/att_rimp2_gradient.rs::terfc_rimp2_gradient_analytical; rust_test=ferric-mp2/tests/terfc_rimp2_gradient_fd.rs::terfc_gradient_matches_fd_water; registry=terfc-rimp2-gradient -->

**SCS-MP2(2terfc).** The energy is
$E_{\mathrm{HF}} + c_{\mathrm{OS}} E_{\mathrm{OS}}[\tau_1] + c_{\mathrm{SS}}\,(E_{\mathrm{SS}}[\tau_2] - E_{\mathrm{SS}}[\tau_1])$,
with $\tau_k = \operatorname{terfc}(r; r_0^{(k)})/r$ and $r_0^{(1)} < r_0^{(2)}$.
Regroup it by operator: it is the sum of two spin-scaled terfc RI-MP2 terms,
$\tau_1$ with weights $(c_{\mathrm{OS}}, -c_{\mathrm{SS}})$ and $\tau_2$ with
$(0, c_{\mathrm{SS}})$. Negative and zero weights are allowed; nothing in
@sec:rimp2-gradient-scs-mp2 assumes positive ones. Each term builds its amplitudes, $x$,
unrelaxed density and RI response on its own operator, exactly as the
single-operator gradient does. Every first variation of $E_c$ at fixed
orbitals is linear in each term's $I$ and $P$, and both terms share the
Coulomb orbital Hessian, so the code sums them and solves one Coulomb
Z-vector for both (`theory.md` section 9.6). It builds one term's
intermediates, keeps its $I$, $P$ and RI response, drops the rest and moves
on, so the memory peak is one term's.

Two anchors check the assembly. With $r_0^{(1)} = r_0^{(2)}$ the same-spin
terms cancel and the gradient is the scaled-opposite-spin terfc gradient
(equal to $1.3\times10^{-15}$ Ha/Bohr; the gap at the published cutoffs is
$1.4\times10^{-3}$). One Z-vector on the sums equals the sum of the two
single-term gradients, each with its own Z-vector, to $1.3\times10^{-13}$
(`scs_2terfc_one_zvector_is_the_sum_of_its_terms`). Neither anchor can see a
term on the wrong cutoff or a weight on the wrong spin block, since both
read the same term list. Finite differences of the energy can.

> **Verified equation {@eq:mp2-grad-scs-weights}c** · the SCS-MP2(2terfc) gradient
>
> On the one-dimensional model of 15.8b with two terfc cutoffs, the
> Lagrangian with summed $I$, $P$ and one Z-vector equals a Richardson
> difference of the re-solved SCS-MP2(2terfc) energy, with and without a
> frozen core.
>
> | | |
> |---|---|
> | Proof | `scripts/scs_2terfc_gradient_sympy.py`, part C (and part L for the $r_0^{(1)} = r_0^{(2)}$ limit) |
> | Negative control | the runner's mutations (term 2 on $r_0^{(1)}$, the Z-vector coupling on the RI kernel, one term's $I$ dropped from the sum) each turn C red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/scs_2terfc_gradient.rs::scs_mp2_2terfc_gradient_analytical` |
> | Code test | `scs_2terfc_gradient_matches_fd_water` in `crates/ferric-mp2/tests/scs_2terfc_gradient_fd.rs` |
>
> <!-- verified: id=mp2.grad.scs-2terfc; script=scripts/scs_2terfc_gradient_sympy.py; claim=C; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/scs_2terfc_gradient.rs::scs_mp2_2terfc_gradient_analytical; rust_test=ferric-mp2/tests/scs_2terfc_gradient_fd.rs::scs_2terfc_gradient_matches_fd_water; registry=scs-2terfc-gradient -->

## Reproducibility and memory {#rimp2-gradient-reproducibility-memory}

The Z-vector assumes a stationary RHF, so the gradient error follows the
SCF density convergence. Tighten `scf_density_conv` for finite-difference work. An
unconverged Z-vector is an error, not a gradient. The fitted reference costs
a few full-rank RI-K gradient channels ($u$ and $v$ of @sec:rimp2-gradient-fitted-ri-jk-reference are full
rank, where the SCF gradient's density has occupied rank). Its declared
peak memory is checked against the budget before the reference SCF
(@sec:rimp2-gradient-fitted-ri-jk-reference).

## Options {#rimp2-gradient-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[scf] jk`, `scf_jk=` | `"exact"`, `"rijk"` | closed-shell MP2 family in the CLI and `run_optimize`: `"rijk"` on the matched JK set, for every task; `run_rimp2`: `"exact"` | the reference and every $J$, $K$ of the gradient |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | adds the core-active block |
| `[mp2] c_os`, `c_ss` (`scs-mp2`) | floats | 6/5, 1/3 (`ScsMp2Config` default) | the weights of (@eq:mp2-grad-scs-weights) |
| `[mp2] omega`, `omega=` (`att-rimp2`) | float, Å⁻¹ | 0.420 | the erfc range parameter |
| `[mp2] att_operator`, `att_operator=` (`att-rimp2`) | `"erfc"`, `"terfc"` | `"erfc"` | the correlation kernel |
| `[mp2] att_r0`, `r0=` (`att-rimp2`, terfc) | float, Å | the basis default (`terfc_r0.rs`) | the terfc cutoff |
| `[mp2] r0_bonded`, `r0_nonbonded`, `c_os`, `c_ss` (`scs-mp2-2terfc`) | floats, Å | 0.75, 1.05, 1.27, 4.05 | $r_0^{(1)}$, $r_0^{(2)}$ and the weights of the SCS-MP2(2terfc) energy |

`scf_jk="rijcosx"`, `k_builder="cosx"`, `local=`, the attenuation keys,
`kappa` and an open-shell molecule are refused by `run_rimp2` with
`with_gradient=True`.

## Numerical anchors {#rimp2-gradient-numerical-anchors}

From `site/src/reference/validation.md` and the tests named.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| RI-MP2 gradient vs PySCF 5-point FD ($h = 2\times10^{-3}$) | H2O, HCN / cc-pVDZ; NH3 / def2-SVP | $\le 2.1\times10^{-9}$ Ha/Bohr | `crates/ferric-mp2/tests/validation_rimp2_gradient.rs` |
| RI-MP2 gradient vs ORCA 6.1.1 (NoRI, no frozen core) | the same | $\le 5.1\times10^{-8}$ Ha/Bohr | the same |
| frozen core vs FD | H2O, HCN, NH3 | $4.9\times10^{-8}$, $1.3\times10^{-8}$, $1.7\times10^{-8}$ | the same |
| fitted-reference gradient vs FD of the fitted energy | water, methane, water with frozen core | test tolerance | `crates/ferric-mp2/tests/rimp2_gradient_rijk.rs` |
| SCS-MP2 gradient vs PySCF 5-point FD of DF-MP2 $E_\mathrm{OS}$, $E_\mathrm{SS}$ | H2O / cc-pVDZ, NH3 / def2-SVP; $(6/5, 1/3)$, $(1, 0)$, $(1, 1)$; frozen core | $\le 2.5\times10^{-9}$ Ha/Bohr | `crates/ferric-mp2/tests/validation_scs_mp2_gradient.rs` |
| att-RI-MP2 gradient vs Richardson FD of `attenuated_ri_mp2` | H2O, CH3F (frozen core 0 and 2) / cc-pVDZ, CH3CHO / 6-31G\*, H2O on RI-JK; $\omega = 0.420$ and 1.0 Å⁻¹ | $\le 2.5\times10^{-9}$ Ha/Bohr (bar about $2\times10^{-7}$) | `crates/ferric-mp2/tests/att_rimp2_gradient_fd.rs` |
| terfc RI-MP2 gradient vs Richardson FD of `ri_mp2` under terfc | H2O / cc-pVDZ ($\omega r_0 = 1/\sqrt2$ and 4), H2O on RI-JK with frozen core 1, CH3F (frozen core 2) / cc-pVDZ, CH3CHO / 6-31G\*; $r_0 = 1.05$ Å | $\le 1.8\times10^{-9}$ Ha/Bohr (bar about $2\times10^{-7}$) | `crates/ferric-mp2/tests/terfc_rimp2_gradient_fd.rs` |
| terf/terfc derivative integrals vs libint2 (Coulomb through the same kernel), FD and C − terf | water / cc-pVTZ and cc-pVDZ, every shell class | $3.8\times10^{-14}$, $2.6\times10^{-11}$, $2.1\times10^{-14}$ | `crates/ferric-integrals/tests/terfc_deriv_integrals.rs` |
| SCS-MP2(2terfc) gradient vs Richardson FD of `scs_mp2_2terfc` | H2O / cc-pVDZ, H2O on RI-JK with frozen core 1, CH3F (frozen core 2) / cc-pVDZ; published parameters | $\le 1.6\times10^{-9}$ Ha/Bohr (bar about $2\times10^{-7}$) | `crates/ferric-mp2/tests/scs_2terfc_gradient_fd.rs` |
| SCS-MP2 gradient vs central FD of `scs_mp2` | H2O / STO-3G and cc-pVDZ, NH3 / 6-31G; $(6/5, 1/3)$ and $(1, 0)$; frozen core; exact and RI-JK | $\le 5.7\times10^{-8}$ Ha/Bohr (bar $2\times10^{-7}$) | `crates/ferric-mp2/tests/scs_mp2_gradient.rs` |

The terfc and SCS-MP2(2terfc) tests also check the translational sum: below
$7\times10^{-13}$ and $4\times10^{-13}$ Ha/Bohr.

## Sources {#rimp2-gradient-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, sections 9, 9.1, 9.2, 9.4, 9.5 | the derivation, the fitted-term table, the RI-JK shift measurements |
| `scripts/mp2_gradient_sympy.py`, parts G and S | the proofs |
| `scripts/att_rimp2_gradient_sympy.py` | the erfc chain rule, the $\omega \to 0$ limit, the operator split |
| `scripts/terfc_rimp2_gradient_sympy.py` | the terfc closed forms and derivatives vs $k$-space quadrature, the Hermite derivative, the operator split, the $r_0 \to 0$ limit |
| PySCF `grad/mp2.py` | the convention the right-hand side follows (cited in `theory.md`) |

**Known limits (2026-11-16).** The functions of this section are closed
shell; the unrestricted gradients are section @sec:u-rimp2-gradient. The SCS-MP2 gradient is
Proven (narrow): closed shell, Coulomb operator, exact or RI-JK J/K, frozen
core 0 or 1. The attenuated RI-MP2 gradient is Proven (narrow): erfc and
terfc, closed shell, exact or RI-JK J/K (the open-shell erfc and terfc
gradients are section @sec:u-rimp2-gradient); its frequencies are central
differences of it and are not graded. The AO-level RI bookkeeping is not
covered by a script.
