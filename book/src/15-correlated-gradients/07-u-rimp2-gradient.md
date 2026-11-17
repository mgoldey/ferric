# Unrestricted RI-MP2 gradient {#u-rimp2-gradient}

<!-- status: ready; registry: u-ri-mp2-gradient, u-att-rimp2-gradient, scs-2terfc-gradient -->

A radical or a triplet has different numbers of $\alpha$ and $\beta$
electrons, and on a UHF reference the two spins get different orbitals. When
a nucleus moves, both sets relax, and they relax together: the $\alpha$
electrons feel the Coulomb field of the $\beta$ density and the reverse.
The unrestricted RI-MP2 gradient is the closed-shell Lagrangian of section
@sec:rimp2-gradient written once per spin. Two things are new. The opposite-spin amplitudes
enter both spins' energy weights, and the UHF orbital Hessian couples the two
spins' Z-vectors through the Coulomb term. Everything else is the closed-shell
piece built from one spin, with the factor that counted both spins halved.

The section derives the energy weights (@sec:u-rimp2-gradient-energy-weights), the per-spin Lagrangian and
the coupled Z-vector (@sec:u-rimp2-gradient-per-spin-lagrangian-coupled) and the assembly (@sec:u-rimp2-gradient-assembly), checks the
closed-shell limit (@sec:u-rimp2-gradient-closed-shell-limit), and then extends the result to the attenuated
kernels (@sec:u-rimp2-gradient-attenuated-erfc-variant–@sec:u-rimp2-gradient-terfc-scs-mp2-2terfc).

| Symbol | Meaning | Code name |
|---|---|---|
| $B^\sigma_{P,ia}$ | dressed RI factors of spin $\sigma$ (capital indices for $\beta$) | `inter_a.b_ov`, `inter_b.b_ov` |
| $t^{\alpha\alpha}, t^{\beta\beta}, t^{\alpha\beta}$ | amplitudes, $t^{\alpha\beta}$ stored $[i,J,a,B]$ | `t_aa`, `t_bb`, `t_ab` |
| $x^\sigma$ | energy weight, $\partial E^{(2)}/\partial B^\sigma = 2x^\sigma$ | `x_a`, `x_b` |
| $I^\sigma$ | Lagrangian matrix of spin $\sigma$ | `imat[s]` |
| $P^\sigma$, $z^\sigma$ | unrelaxed density and Z-vector of spin $\sigma$ | `p_mo[s]`, `zv.z[s]` |
| $v^\sigma[X]$ | $J[X^\alpha + X^\beta] - K[X^\sigma]$ | `veff_vo` |

The denominators follow `crates/ferric-mp2/theory.md`: $\Delta$ positive with
the minus sign written into the amplitudes.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,total_u_rimp2_gradient_on}` |
| Rust (attenuated) | `ferric-mp2/src/att_rimp2_gradient.rs::{u_att_rimp2_gradient_analytical,total_u_att_rimp2_gradient_on,u_terfc_rimp2_gradient_analytical,total_u_terfc_rimp2_gradient_on}`, `ferric-mp2/src/scs_2terfc_gradient.rs::{u_scs_mp2_2terfc_gradient_analytical,total_u_scs_mp2_2terfc_gradient_on}` |
| CLI | `method.kind = "rimp2"`, `"att-rimp2"` (erfc or terfc) or `"scs-mp2-2terfc"` with `multiplicity > 1` and `task = "optimize"` or `"frequencies"` (UHF reference; with `[scf] jk` omitted the reference is RI-JK on the matched JK-fit set) |
| Python | `run_optimize` or `run_frequencies` with `method="rimp2"`, `"att-rimp2"` or `"scs-mp2-2terfc"` on a UHF reference |

## Energy weights {#u-rimp2-gradient-energy-weights}

With $t^{\alpha\alpha}_{ijab} = -[(ia|jb) - (ib|ja)]/\Delta_{ij}^{ab}$ and
$t^{\alpha\beta}_{iJaB} = -(ia|JB)/\Delta_{iJ}^{aB}$, at fixed orbital energies

$$ x^\alpha_{P,ia} = \sum_{jb}t^{\alpha\alpha}_{ijab}B^\alpha_{P,jb} + \sum_{JB}t^{\alpha\beta}_{iJaB}B^\beta_{P,JB}, \qquad x^\beta_{P,JB} = \sum_{KC}t^{\beta\beta}_{JKBC}B^\beta_{P,KC} + \sum_{ia}t^{\alpha\beta}_{iJaB}B^\alpha_{P,ia} . \label{eq:mp2-u-grad-weights} $$

The $\alpha\beta$ block appears in both weights: an opposite-spin pair
$(i\alpha, J\beta)$ depends on $B^\alpha$ through the $ia$ half of $(ia|JB)$
and on $B^\beta$ through the $JB$ half, so differentiating it gives one term in each spin's weight.
Same-spin pairs contribute only to their own spin.

A consistency check follows from the degree. $E^{(2)}$ is homogeneous of
degree 4 in $B$: each term is an integral times an amplitude, each amplitude
is linear in the integrals, and each integral is quadratic in $B$. Euler's
theorem gives
$\sum_\sigma B^\sigma\cdot\partial E^{(2)}/\partial B^\sigma = 4E^{(2)}$, and
with $\partial E^{(2)}/\partial B^\sigma = 2x^\sigma$,
$\sum_\sigma B^\sigma\cdot x^\sigma = 2E^{(2)}$. In the closed-shell limit
$B^\sigma = B$ and $x^\sigma = x$, and the identity becomes
$B\cdot x = E^{(2)}$, the relation section @sec:rpa-gradient-lagrangian-z-vector-frozen uses.

`u_energy_weights` builds $x^\alpha$ one occupied $i$ at a time from
$t^{\alpha\alpha}_{i\cdot\cdot\cdot}$ and $t^{\alpha\beta}_{i\cdot\cdot\cdot}$, and
$x^\beta$ one $J$ at a time from $t^{\beta\beta}_{J\cdot\cdot\cdot}$ and the
$J$ slice of $t^{\alpha\beta}$ with its axes permuted to $[B,i,a]$.

> **Verified equation @eq:mp2-u-grad-weights** · both spins' energy weights
>
> With $B^\sigma$ free symbols (one auxiliary function, two occupied and two
> virtual orbitals per spin, fixed orbital energies), sympy differentiates
> the U-MP2 energy and recovers $2x^\alpha$ and $2x^\beta$ of (@eq:mp2-u-grad-weights).
>
> | | |
> |---|---|
> | Proof | `scripts/u_rimp2_gradient_sympy.py`, part S |
> | Negative control | the runner sets the $\alpha\beta$ weight in $x^\beta$ to 0: S turns red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2_gradient.rs::u_energy_weights` |
> | Code test | `energy_weights_at_alpha_equals_beta_are_the_closed_weights` in `crates/ferric-mp2/src/u_rimp2_gradient.rs` |
>
> <!-- verified: id=mp2.u.grad.weights; script=scripts/u_rimp2_gradient_sympy.py; claim=S; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2_gradient.rs::u_energy_weights; rust_test=ferric-mp2/src/u_rimp2_gradient.rs::energy_weights_at_alpha_equals_beta_are_the_closed_weights; registry=u-ri-mp2-gradient -->

## Per-spin Lagrangian and the coupled Z-vector {#u-rimp2-gradient-per-spin-lagrangian-coupled}

$$ I^\sigma_{qi} = -\sum_{Pa}x^\sigma_{P,ia}B^\sigma_{P,qa}, \qquad I^\sigma_{qa} = -\sum_{Pi}x^\sigma_{P,ia}B^\sigma_{P,qi}, \qquad X^\sigma_{ai} = \big[C_\sigma^{\mathsf T}v^\sigma[P]\,C_\sigma\big]_{ai} + I^\sigma_{ia} - I^\sigma_{ai} , \label{eq:u-rimp2-gradient-2} $$

with $P^\sigma$ the symmetrized unrelaxed spin density on its occupied and
virtual blocks and, on a frozen core, $P^\sigma_{kI} = I^\sigma_{kI}/(\varepsilon_k - \varepsilon_I)$.
The closed-shell builder `build_imat_ri` carries a factor $-2$ that is the
weight of both spins; one spin's weight is half of it:

<!-- excerpt: ferric-mp2/src/u_rimp2_gradient.rs::u_term_parts -->
```rust
out.push(0.5 * crate::zvector::build_imat_ri(x, &b_full, &spins[s].orb));
```

Why one half? In the closed-shell $I$ of (@eq:rimp2-gradient-2) the factor 2 counts the two
spins of a doubly occupied orbital; a spin orbital is singly occupied.

The Z-vector couples the spins through Coulomb only. A rotation in spin
$\sigma$ changes $D^\sigma$; the Coulomb potential of that change acts on
both spins, while exchange acts only between electrons of the same spin:

$$ (\varepsilon^\sigma_a - \varepsilon^\sigma_i)\,z^\sigma_{ai} + \big[C_\sigma^{\mathsf T}\big(J[D^{z,\alpha} + D^{z,\beta}] - K[D^{z,\sigma}]\big)C_\sigma\big]_{ai} = -X^\sigma_{ai}, \qquad D^{z,\sigma} = C_{v\sigma}z^\sigma C_{o\sigma}^{\mathsf T} + \text{h.c.} \label{eq:u-rimp2-gradient-3} $$

$i$ runs over every occupied orbital, frozen or not. `solve_u_zvector`
starts from $z^\sigma = -X^\sigma/(\varepsilon_a - \varepsilon_i)$, iterates the
pair with one DIIS of eight vectors, and stops at $\max|r| < 10^{-8}$; 50 iterations without
convergence, or a gap $\varepsilon_a - \varepsilon_i \le 10^{-12}$, is an error.
Before each $J/K$ build the pair is scaled by the power of two that brings
$\max|z|$ into $(\tfrac12, 1]$, so the screening sees a unit-scale density;
the scale is exact.

<!-- excerpt: ferric-mp2/src/u_rimp2_gradient.rs::u_terms_gradient_timed -->
```rust
-(v[s][(a, i)] + imat[s][(i, a_mo)] - imat[s][(a_mo, i)])
```

| Symbol | Code |
|---|---|
| $-X^\sigma_{ai}$ | the right-hand side `rhs[s]` |
| $[C^{\mathsf T}v^\sigma[P]C]_{ai}$ | `v[s][(a, i)]` (`veff_vo` of the $J/K$ of both $P^\sigma$) |
| $I^\sigma_{ia}$, $I^\sigma_{ai}$ | `imat[s][(i, a_mo)]`, `imat[s][(a_mo, i)]` |

## Assembly {#u-rimp2-gradient-assembly}

With $P^\sigma_{\mathrm{rel}} = P^\sigma + z^\sigma$ on both occupied–virtual
blocks and $D^\sigma$ the UHF spin densities,

$$ \frac{dE}{dx} = \operatorname{Tr}\big[(D + P_{\mathrm{rel}})_{\mathrm T}h'\big] + \sum_{\mu\nu}\Big(\sum_\sigma\big[\mathrm{im1}^\sigma - \zeta^\sigma - W^\sigma_{\mathrm{HF}} - P^\sigma_{\mathrm{occ}}v^\sigma[P_{\mathrm{rel}}]P^\sigma_{\mathrm{occ}}\big]\Big)_{\mu\nu}S'_{\mu\nu} + \partial_xU\big((D^\alpha, D^\beta), (D^\alpha + 2P^\alpha_{\mathrm{rel}}, D^\beta + 2P^\beta_{\mathrm{rel}})\big) + \sum_\sigma R^\sigma_{\mathrm{RI}} , \label{eq:mp2-u-grad-assembly} $$

$$ U(A, B) = \tfrac12\operatorname{Tr}\big[A_{\mathrm T}J[B_{\mathrm T}]\big] - \tfrac12\sum_\sigma\operatorname{Tr}\big[A^\sigma K[B^\sigma]\big], $$

differentiated through the integrals only. In the code:

| Term | Built as |
|---|---|
| $\mathrm{im1}^\sigma$ | $I^\sigma$ with its virtual–occupied block replaced by the transpose of its occupied–virtual block |
| $\zeta^\sigma$ | $\tfrac12(\varepsilon_p + \varepsilon_q)P^\sigma_{\mathrm{rel},pq}$ on the oo and vv blocks, $\varepsilon_iP^\sigma_{\mathrm{rel},ai}$ on both ov blocks |
| $W^\sigma_{\mathrm{HF}}$ | `build_energy_weighted_density_uhf` (unit occupation) |
| $P^\sigma_{\mathrm{occ}}v^\sigma[P_{\mathrm{rel}}]P^\sigma_{\mathrm{occ}}$ | `vhf_s1occ`, with $J/K$ of $P_{\mathrm{rel}}$ by linearity (unrelaxed plus $D^z$) |
| $\partial_xU$ | `two_electron_gradient_bilinear_open`, which dispatches on the reference's route: `ferric-scf/src/gradient.rs::twoelectron_gradient_bilinear_uhf` (four-centre) or `ferric-scf/src/response_jk.rs::fitted_bilinear_gradient_open` (RI-JK, the $Q(u) - Q(v)$ split of section @sec:rimp2-gradient-fitted-ri-jk-reference per spin) |
| $R^\sigma_{\mathrm{RI}}$ | `ri_response_gradient_channels`, one channel per spin with $y^\sigma = V^{-1/2}x^\sigma$ and weight 1 |

The closed-shell RI response has weight 2 against this weight 1 per spin.

> **Verified equation @eq:mp2-u-grad-assembly** · the assembled gradient
>
> On a model whose overlap, core Hamiltonian, four-centre integrals and RI
> factors and metric all move with one parameter $x$, a UHF with
> $n_\alpha = 3$, $n_\beta = 2$ solved to $10^{-13}$ at every $x$, (@eq:mp2-u-grad-assembly) with
> (@eq:u-rimp2-gradient-2)–(@eq:u-rimp2-gradient-3) equals the Richardson derivative of the re-solved
> $E_{\mathrm{UHF}} + E^{(2)}$, with and without a frozen core (G1), and
> $\operatorname{Tr}[(D + P_{\mathrm{rel}})R]$ equals $dE/d\lambda$ for
> $h \to h + \lambda R$ (G2). The closed-shell limit reproduces the restricted
> assembly of section @sec:rimp2-gradient (part C).
>
> | | |
> |---|---|
> | Proof | `scripts/u_rimp2_gradient_sympy.py`, parts G and C |
> | Negative control | the runner drops the $\beta$ Z-vector, drops the opposite-spin Coulomb coupling of (@eq:u-rimp2-gradient-3), and doubles the RI weight per spin; each turns G red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,solve_u_zvector,unrelaxed_density_mo,veff_vo,response_density}` |
> | Code test | `o2_triplet_matches_fd_four_centre` in `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` |
>
> <!-- verified: id=mp2.u.grad.assembly; script=scripts/u_rimp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,solve_u_zvector,unrelaxed_density_mo,veff_vo,response_density}; rust_test=ferric-mp2/tests/u_rimp2_gradient_fd.rs::o2_triplet_matches_fd_four_centre; registry=u-ri-mp2-gradient -->

The model covers the RI terms at the MO level; the AO bookkeeping of
`ri_response_gradient_channels` is checked only by the Rust tests.

### Memory {#u-rimp2-gradient-memory}

On the RI-JK route the peak is declared by
`ferric-mp2/src/gradient_memory.rs::fitted_gradient_plan_unrestricted` and
checked before the UHF, as for the closed shell (section
@sec:rimp2-gradient-fitted-peak-memory). Its terms are the closed-shell ones
resolved by spin: the three amplitude blocks $t^{\alpha\alpha}$,
$t^{\beta\beta}$, $t^{\alpha\beta}$, two $B^\sigma$ and two $x^\sigma$, one
full-MO $b^\sigma(P|pq)$ at a time for $I^\sigma$, and four exchange-channel
densities $u^\alpha$, $u^\beta$, $v^\alpha$, $v^\beta$ in the two-electron
derivative. Each indefinite density costs at most $2n_{\mathrm{ao}}^2$ in
$\sum_sm_s^2$, so the bound is $8\,n_{\mathrm{ao}}^2$ in place of the closed
shell's $4\,n_{\mathrm{ao}}^2$ (`OPEN_EXCHANGE_CHANNEL_RANK2_BOUND`).

## The closed-shell limit {#u-rimp2-gradient-closed-shell-limit}

Set the two spins' orbitals equal and every piece must reduce to the
restricted one. For $\alpha \equiv \beta$: $t^{\alpha\alpha} = t - \bar t$ and
$t^{\alpha\beta} = t$, so $x^\sigma$ is the closed-shell $x$; $I^\sigma = I/2$,
$P^\alpha + P^\beta = P + P^{\mathsf T}$, $X^\sigma = X/2$, $z^\sigma = z/2$,
$U$ is the closed-shell bilinear form, and the two weight-1 RI channels sum to
the closed-shell weight-2 channel. In the coupled Z-vector equation the opposite-spin Coulomb term
$J[D^{z,\alpha} + D^{z,\beta}]$ with $z^\sigma = z/2$ is $J$ of the closed-shell
$D^z$, so dropping it would break this limit; the runner's mutation that
drops it turns part G red. A closed-shell RHF written as a UHF gives
the restricted gradient on both $J/K$ routes
(`closed_shell_limit_matches_restricted_four_centre`,
`closed_shell_limit_matches_restricted_rijk`, and the frozen-core variant).

## What is anchored {#u-rimp2-gradient-anchored}

The proof scripts first. In part G of `scripts/u_rimp2_gradient_sympy.py`
the assembled gradient matches the Richardson derivative of the re-solved
$E_{\mathrm{UHF}} + E^{(2)}$ ($n_\alpha = 3$, $n_\beta = 2$) to
$1.5\times10^{-11}$ relative, with and without a frozen core, and
$\operatorname{Tr}[(D + P_{\mathrm{rel}})_{\mathrm T}R]$ matches
$dE/d\lambda$ for $h \to h + \lambda R$ to $10^{-11}$. In part C the model
reproduces the closed-shell assembly to $2\times10^{-16}$.

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| analytic vs Richardson FD of ferric's own U-RI-MP2 energy, four-centre and RI-JK | OH, NH2 doublets, O2 triplet / 6-31G; OH frozen core 1 | ≤ 3.1e-9 Ha/Bohr (O2), 1.8e-9 (NH2), 8e-10 (OH); frozen core 2.5e-10 | 2.1e-7 to 2.3e-7 Ha/Bohr, derived in the test header | `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` |
| closed-shell limit vs the restricted analytic gradient | H2O / 6-31G, both routes, frozen core | ≤ 3.1e-15 Ha/Bohr | 1e-10 Ha/Bohr | `crates/ferric-mp2/src/u_rimp2_gradient.rs` |

There is no external reference: PySCF's UMP2 gradient uses conventional
integrals, not density fitting. Figures from `site/src/reference/validation.md`.

## The attenuated (erfc) variant {#u-rimp2-gradient-attenuated-erfc-variant}

The open-shell attenuated gradient (`att-rimp2` at multiplicity > 1, with
`task = "optimize"` or `"frequencies"`) is
`ferric-mp2/src/att_rimp2_gradient.rs::u_att_rimp2_gradient_analytical`, with
`total_u_att_rimp2_gradient_on` adding the UHF part. Both run over
`ferric-mp2/src/u_rimp2_gradient.rs::u_correlated_gradient_timed`.

On a Coulomb UHF with orbitals $C_\sigma$ and energies $\varepsilon^\sigma$
the energy is

$$ E = E_{\mathrm{UHF}}[1/r] + s\,\big(E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}\big)[\hat w], \qquad \hat w = \frac{\operatorname{erfc}(\omega r)}{r}, $$

with the spin blocks of @sec:u-rimp2-gradient-energy-weights built from
$B^\sigma_{P,ia} = \sum_Q[V_{\hat w}^{-1/2}]_{PQ}(Q|ia)_{\hat w}$: both the
3-index integrals and the metric are under erfc. The Lagrangian is the
unrestricted one with $E^{(2)} \to s\,E_c[\hat w]$:

$$ \mathcal L = E_{\mathrm{UHF}}[1/r] + s\,E_c[\hat w;\,C_\alpha, C_\beta, \varepsilon] + \sum_\sigma\sum_{ai} z^\sigma_{ai}F^\sigma_{ai}[1/r] - \sum_\sigma\operatorname{Tr}\big[W^\sigma(C_\sigma^{\mathsf T}SC_\sigma - 1)\big] . $$

As in the closed-shell case of section @sec:rimp2-gradient, each term takes the
operator of what it differentiates:

| Term | Differentiates | Operator |
|---|---|---|
| $t^{\sigma\sigma'}$, $P^\sigma$, $x^\sigma$, $I^\sigma$ (through $b^\sigma(P\vert pq)$) | $s\,E_c$ | erfc |
| $v^\sigma[P]$ in $X^\sigma$ | $\varepsilon^\sigma$ in $E_c$, through $F^\sigma[1/r]$ | Coulomb |
| the Z-vector coupling $J[D^{z,\alpha} + D^{z,\beta}] - K[D^{z,\sigma}]$ | the stationarity $F^\sigma_{ai}[1/r]$ | Coulomb |
| $P_{\mathrm{occ}}v^\sigma[P_{\mathrm{rel}}]P_{\mathrm{occ}}$, $\partial_xU$ | $E_{\mathrm{UHF}}$ and $F^\sigma[1/r]$ | Coulomb |
| $R^\sigma_{\mathrm{RI}}$ | $E_c$ at fixed $C$ | erfc |

The scaling enters only through $E_c$. Every first variation of $E_c$ is
linear in $x^\sigma$ and in the unrelaxed $P^\sigma_{oo}$, $P^\sigma_{vv}$:
$I^\sigma$, the frozen-core block $I^\sigma_{kI}/(\varepsilon_k - \varepsilon_I)$,
$X^\sigma$, $z^\sigma$ and the RI response are all built from them. So the
code multiplies $x^\sigma$ and the unrelaxed densities by $s$ and leaves the
rest of the assembly unchanged. The correlation operator is `UCorrelation::op`
(`ferric-mp2/src/u_rimp2_gradient.rs::UCorrelation`): the amplitudes, the
$b^\sigma$ of $I^\sigma$ and the 3c/2c response. The Coulomb terms run on the
reference's $J/K$ route and on `bounds.op` (the two-electron derivative).
Both $J/K$ routes are supported.

**Two limits.** For $\alpha \equiv \beta$ the reduction of
@sec:u-rimp2-gradient-closed-shell-limit ($x^\sigma = x$, $I^\sigma = I/2$,
$z^\sigma = z/2$) holds for any operator and scaling, so the open-shell
gradient equals the closed-shell attenuated gradient. As $\omega \to 0$ the
erfc integrals tend to the Coulomb ones with a first-order constant-kernel
shift that vanishes in $(P|ia)$ (section
@sec:rimp2-gradient-attenuated-erfc-ri-mp2), so the gradient tends to the
U-RI-MP2 gradient.

`scripts/u_att_rimp2_gradient_sympy.py` proves both. Part C is the
closed-shell limit and part Z the $\omega \to 0$ limit. In part U, on a 1-D
model with real $s$-Gaussian Coulomb and erfc integrals ($n_\alpha = 3$,
$n_\beta = 2$), the analytic gradient equals Richardson differences of the
re-solved $E_{\mathrm{UHF}} + s\,E_c[\hat w]$ for $s = 1$ and 1.2, with and
without a frozen core. Part K checks the model's closed forms against a
$t$-quadrature of the erfc kernel and their derivatives against differences.

ROHF is refused for the reason given under Known limits. Open-shell
`scs-mp2` (Coulomb) has no gradient: the CLI refuses it for `optimize` and
`frequencies`.

| Check | System | Measured | Bar | Test |
|---|---|---|---|---|
| analytic vs Richardson FD of PySCF's UHF + `DFUMP2` energy on an erfc fit (`scripts/validation/gen_u_att_rimp2_gradient_pyscf.py`) | OH, NH2 doublets / cc-pVDZ + cc-pVDZ-RI, ω = 0.420 Å⁻¹ | ≤ 4.0e-10 Ha/Bohr (OH), 1.7e-10 (NH2) | about 2.8e-7 Ha/Bohr | `crates/ferric-mp2/tests/validation_u_att_rimp2_gradient_pyscf.rs` |
| analytic vs FD of ferric's own `attenuated_ri_mp2` on a UHF | OH, NH2 / 6-31G + cc-pVDZ-RI; OH at ω = 1.0 Å⁻¹, s = 1.2, frozen core 1; OH on RI-JK | ≤ 4.8e-9 Ha/Bohr | 2.1e-7 Ha/Bohr | `crates/ferric-mp2/tests/u_att_rimp2_gradient_fd.rs` |
| closed-shell limit vs the restricted attenuated gradient (`open_shell_att_gradient_matches_closed_in_the_trivial_limit*`) | H2O / 6-31G, both routes, frozen core; ω = 1 Å⁻¹, s = 1.2 | ≤ 2.2e-15 Ha/Bohr | 1e-10 Ha/Bohr, from the two Z-vector residuals | `crates/ferric-mp2/src/att_rimp2_gradient.rs` |
| ω → 0 vs the Coulomb U-RI-MP2 gradient on the same UHF (`open_shell_att_gradient_is_u_rimp2_as_omega_goes_to_zero`) | OH, ω = 1e-7 Bohr⁻¹ | 1.7e-13 Ha/Bohr | 1e-10 Ha/Bohr | `crates/ferric-mp2/src/att_rimp2_gradient.rs` |

## terfc and SCS-MP2(2terfc) {#u-rimp2-gradient-terfc-scs-mp2-2terfc}

The open-shell terfc gradient (`att-rimp2` with `att_operator = "terfc"` at
multiplicity > 1) is the @sec:u-rimp2-gradient-attenuated-erfc-variant Lagrangian with
$\hat w = \tau$, the terfc kernel of section
@sec:rimp2-gradient-attenuated-erfc-ri-mp2, and $s = 1$; the operator table
holds row by row with terfc in place of erfc. The open-shell SCS-MP2(2terfc)
gradient (`scs-mp2-2terfc` at multiplicity > 1) sums two such terms, as in
section @sec:rimp2-gradient. The code is
`ferric-mp2/src/att_rimp2_gradient.rs::u_terfc_rimp2_gradient_analytical`
(with `total_u_terfc_rimp2_gradient_on`) and
`ferric-mp2/src/scs_2terfc_gradient.rs::u_scs_mp2_2terfc_gradient_analytical`
(with `total_u_scs_mp2_2terfc_gradient_on`), both over
`ferric-mp2/src/u_rimp2_gradient.rs::u_terms_gradient_timed`. Both run with
`task = "optimize"` or `"frequencies"` in the CLI and through `run_optimize` or
`run_frequencies` in Python.

**Spin-component weights.** On a UHF the opposite-spin energy is
$E_{\alpha\beta}$ and the same-spin energy is
$E_{\alpha\alpha} + E_{\beta\beta}$. A term
$E_c[\tau; a, b] = a\,E_{\alpha\beta} + b\,(E_{\alpha\alpha} + E_{\beta\beta})$
has, at fixed orbital energies, $\partial E_c/\partial B^\sigma = 2x^\sigma$
with

$$ x^{\alpha}_{P,ia} = b\sum_{jb} t^{\alpha\alpha}_{ijab}B^{\alpha}_{P,jb} + a\sum_{JB} t^{\alpha\beta}_{iJaB}B^{\beta}_{P,JB}, $$

and $x^\beta$ likewise: (@eq:mp2-u-grad-weights) with $b$ on the same-spin
sum and $a$ on the opposite-spin sum. The denominator derivatives are the
unrelaxed densities with $b$ on the same-spin Gram products and $a$ on the
opposite-spin ones:

$$ P^{\alpha}_{ij} = -\tfrac12 b\sum t^{\alpha\alpha}_{ikab}t^{\alpha\alpha}_{jkab} - a\sum t^{\alpha\beta}_{iKaB}t^{\alpha\beta}_{jKaB}, \qquad P^{\alpha}_{ab} = \tfrac12 b\sum t^{\alpha\alpha}_{ijac}t^{\alpha\alpha}_{ijbc} + a\sum t^{\alpha\beta}_{iJaC}t^{\alpha\beta}_{iJbC} . $$

The spin blocks carry no cross terms, so the weights split cleanly
(`ferric-mp2/src/u_rimp2_gradient.rs::u_energy_weights_scaled`,
`ferric-mp2/src/u_rimp2.rs::build_u_mp2_density_scaled`). At $a = b$ the code
keeps the uniform-scale path of @sec:u-rimp2-gradient-attenuated-erfc-variant,
unchanged bit for bit. For SCS-MP2(2terfc) the same-spin products carry
$c_{\mathrm{SS}}$ and the opposite-spin products carry $c_{\mathrm{OS}}$, each
term runs its RI response on its own cutoff, and one spin-coupled Coulomb
Z-vector closes the summed $I^\sigma$ and $P^\sigma$.

**Closed-shell limit.** For $\alpha \equiv \beta$,
$E_{\alpha\beta} = E_{\mathrm{OS}}$ and
$E_{\alpha\alpha} + E_{\beta\beta} = E_{\mathrm{SS}}$, and the reduction of
@sec:u-rimp2-gradient-closed-shell-limit holds term by term.

> **Verified equation {@eq:mp2-u-grad-assembly}b** · the open-shell SCS-MP2(2terfc) gradient
>
> On the one-dimensional model of 15.8b with a UHF reference (three
> $\alpha$, two $\beta$ electrons), the Lagrangian with spin-resolved
> weights and one coupled Z-vector equals a Richardson difference of the
> re-solved unrestricted SCS-MP2(2terfc) energy. At $\alpha = \beta$ it
> reduces to the closed-shell Lagrangian of 15.8c.
>
> | | |
> |---|---|
> | Proof | `scripts/scs_2terfc_gradient_sympy.py`, parts U and L (part S proves the spin-resolved weights) |
> | Negative control | the runner's mutations (same-spin and opposite-spin weights swapped in $x^\sigma$ or in the densities, the $\beta$ Z-vector dropped) each turn U or S red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/scs_2terfc_gradient.rs::u_scs_mp2_2terfc_gradient_analytical` |
> | Code test | `u_scs_2terfc_oh_doublet_matches_fd_frozen_core` in `crates/ferric-mp2/tests/u_terfc_gradient_fd.rs` |
>
> <!-- verified: id=mp2.u.grad.scs-2terfc; script=scripts/scs_2terfc_gradient_sympy.py; claim=U; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/scs_2terfc_gradient.rs::u_scs_mp2_2terfc_gradient_analytical; rust_test=ferric-mp2/tests/u_terfc_gradient_fd.rs::u_scs_2terfc_oh_doublet_matches_fd_frozen_core; registry=scs-2terfc-gradient -->

| Check | System | Measured | Bar | Test |
|---|---|---|---|---|
| terfc: analytic vs Richardson FD of ferric's own `u_ri_mp2` (terfc) on a UHF | OH / 6-31G + cc-pVDZ-RI (four-centre); NH2 on RI-JK, frozen core 1 | ≤ 3.9e-9 Ha/Bohr (NH2, RI-JK) | at least 2.1e-7 Ha/Bohr | `crates/ferric-mp2/tests/u_terfc_gradient_fd.rs` |
| SCS-MP2(2terfc): analytic vs Richardson FD of `scs_mp2_2terfc` on a UHF | OH frozen core 1; NH2 four-centre and RI-JK; 6-31G + cc-pVDZ-RI | ≤ 1.2e-9 Ha/Bohr (NH2, RI-JK) | at least 2.1e-7 Ha/Bohr | `crates/ferric-mp2/tests/u_terfc_gradient_fd.rs` |
| closed-shell limit vs the restricted gradient (`open_shell_terfc_gradients_match_closed_in_the_trivial_limit*`) | H2O, both routes, frozen core | terfc 8.9e-16, SCS-MP2(2terfc) 2.6e-15 Ha/Bohr (four-centre); 3.0e-15 and 3.1e-15 (RI-JK, frozen core) | 1e-10 Ha/Bohr | `crates/ferric-mp2/src/scs_2terfc_gradient.rs` |
| one Z-vector vs the sum of single-term gradients (`open_shell_scs_2terfc_one_zvector_is_the_sum_of_its_terms`) | OH | 7.6e-14 Ha/Bohr, correlation part 2.0e-2 | 1e-10 Ha/Bohr | `crates/ferric-mp2/src/scs_2terfc_gradient.rs` |

On the 1-D model of `scripts/scs_2terfc_gradient_sympy.py`, the open-shell
assembly at $n_\alpha = n_\beta$ equals the closed one to $6.6\times10^{-15}$
(part L). At $n_\alpha = 3$, $n_\beta = 2$ the analytic gradients of
SCS-MP2(2terfc) (frozen core 0 and 1) and terfc RI-MP2 equal Richardson
differences of the re-solved energy to $1.4\times10^{-10}$ relative (part U).
Check S3 of part S proves the spin-resolved weights that
`u_energy_weights_scaled` and `build_u_mp2_density_scaled` build.

## Known limits (2026-11-16)

- UHF reference only. A ROHF reference is refused (`ROHF_REFUSAL`): it is not
  stationary under independent $\alpha$/$\beta$ rotations and its MP2 runs on
  semicanonical orbitals, so it needs the ROHF Z-vector and the
  semicanonicalization response, neither of which is built.
- No COSX reference, no X2C, at least one $\beta$ electron. The operators are
  Coulomb, erfc and terfc.
- SCS-MP2(2terfc) is the only spin-scaled unrestricted gradient. There is no
  open-shell `scs-mp2` or OO-RI-MP2 gradient.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "15. Unrestricted RI-MP2 nuclear gradient" | equations @eq:mp2-u-grad-weights–@eq:mp2-u-grad-assembly |
| `crates/ferric-mp2/src/u_rimp2_gradient.rs` | implementation |
| `scripts/u_rimp2_gradient_sympy.py` | proofs |
| `crates/ferric-mp2/theory.md`, "15.1 Open-shell attenuated (erfc) RI-MP2 gradient"; `scripts/u_att_rimp2_gradient_sympy.py` | section @sec:u-rimp2-gradient-attenuated-erfc-variant |
| `crates/ferric-mp2/theory.md`, "15.2 Open-shell terfc and SCS-MP2(2terfc) gradients"; `scripts/scs_2terfc_gradient_sympy.py` | section @sec:u-rimp2-gradient-terfc-scs-mp2-2terfc |
| `site/src/reference/validation.md`, `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` | anchors |
