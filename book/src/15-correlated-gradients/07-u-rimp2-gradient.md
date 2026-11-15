# 15.7 Unrestricted RI-MP2 gradient

<!-- section: 15.7; status: ready; registry: u-ri-mp2-gradient;
     theory: crates/ferric-mp2/theory.md#"15. Unrestricted RI-MP2 nuclear gradient" -->

The unrestricted RI-MP2 gradient is the closed-shell Lagrangian of section
15.2 written once per spin. Two things are new. The opposite-spin amplitudes
enter both spins' energy weights, and the UHF orbital Hessian couples the two
spins' Z-vectors through the Coulomb term. Everything else is the closed-shell
piece built from one spin, with the factor that counted both spins halved.

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
| CLI | `method.kind = "rimp2"` with `multiplicity > 1` and `task = "optimize"` (UHF reference; with `[scf] jk` omitted the reference is RI-JK on the matched JK-fit set) |
| Python | `run_optimize(method="rimp2")` on a UHF reference |

## 15.7.1 Energy weights

With $t^{\alpha\alpha}_{ijab} = -[(ia|jb) - (ib|ja)]/\Delta_{ij}^{ab}$ and
$t^{\alpha\beta}_{iJaB} = -(ia|JB)/\Delta_{iJ}^{aB}$, at fixed orbital energies

$$ x^\alpha_{P,ia} = \sum_{jb}t^{\alpha\alpha}_{ijab}B^\alpha_{P,jb} + \sum_{JB}t^{\alpha\beta}_{iJaB}B^\beta_{P,JB}, \qquad x^\beta_{P,JB} = \sum_{KC}t^{\beta\beta}_{JKBC}B^\beta_{P,KC} + \sum_{ia}t^{\alpha\beta}_{iJaB}B^\alpha_{P,ia} . \tag{15.12} $$

The $\alpha\beta$ block appears in both weights. $E^{(2)}$ is homogeneous of
degree 4 in $B$, so $\sum_\sigma B^\sigma\cdot x^\sigma = 2E^{(2)}$.
`u_energy_weights` builds $x^\alpha$ one occupied $i$ at a time from
$t^{\alpha\alpha}_{i\cdot\cdot\cdot}$ and $t^{\alpha\beta}_{i\cdot\cdot\cdot}$, and
$x^\beta$ one $J$ at a time from $t^{\beta\beta}_{J\cdot\cdot\cdot}$ and the
$J$ slice of $t^{\alpha\beta}$ with its axes permuted to $[B,i,a]$.

> **Verified equation 15.12** · both spins' energy weights
>
> With $B^\sigma$ free symbols (one auxiliary function, two occupied and two
> virtual orbitals per spin, fixed orbital energies), sympy differentiates
> the U-MP2 energy and recovers $2x^\alpha$ and $2x^\beta$ of (15.12).
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

## 15.7.2 Per-spin Lagrangian and the coupled Z-vector

$$ I^\sigma_{qi} = -\sum_{Pa}x^\sigma_{P,ia}B^\sigma_{P,qa}, \qquad I^\sigma_{qa} = -\sum_{Pi}x^\sigma_{P,ia}B^\sigma_{P,qi}, \qquad X^\sigma_{ai} = \big[C_\sigma^{\mathsf T}v^\sigma[P]\,C_\sigma\big]_{ai} + I^\sigma_{ia} - I^\sigma_{ai} , \tag{15.13} $$

with $P^\sigma$ the symmetrized unrelaxed spin density on its occupied and
virtual blocks and, on a frozen core, $P^\sigma_{kI} = I^\sigma_{kI}/(\varepsilon_k - \varepsilon_I)$.
The closed-shell builder `build_imat_ri` carries a factor $-2$ that is the
weight of both spins; one spin's weight is half of it:

<!-- excerpt: ferric-mp2/src/u_rimp2_gradient.rs::u_rimp2_gradient_timed -->
```rust
out.push(0.5 * crate::zvector::build_imat_ri(x, &b_full, &spins[s].orb));
```

The Z-vector couples the spins through Coulomb only:

$$ (\varepsilon^\sigma_a - \varepsilon^\sigma_i)\,z^\sigma_{ai} + \big[C_\sigma^{\mathsf T}\big(J[D^{z,\alpha} + D^{z,\beta}] - K[D^{z,\sigma}]\big)C_\sigma\big]_{ai} = -X^\sigma_{ai}, \qquad D^{z,\sigma} = C_{v\sigma}z^\sigma C_{o\sigma}^{\mathsf T} + \text{h.c.} \tag{15.14} $$

$i$ runs over every occupied orbital, frozen or not. `solve_u_zvector`
starts from $z^\sigma = -X^\sigma/(\varepsilon_a - \varepsilon_i)$, iterates the
pair with one DIIS, and stops at $\max|r| < 10^{-8}$; 50 iterations without
convergence, or a gap $\varepsilon_a - \varepsilon_i \le 10^{-12}$, is an error.
Before each $J/K$ build the pair is scaled by the power of two that brings
$\max|z|$ into $(\tfrac12, 1]$, so the screening sees a unit-scale density;
the scale is exact.

<!-- excerpt: ferric-mp2/src/u_rimp2_gradient.rs::u_rimp2_gradient_timed -->
```rust
-(v[s][(a, i)] + imat[s][(i, a_mo)] - imat[s][(a_mo, i)])
```

| Symbol | Code |
|---|---|
| $-X^\sigma_{ai}$ | the right-hand side `rhs[s]` |
| $[C^{\mathsf T}v^\sigma[P]C]_{ai}$ | `v[s][(a, i)]` (`veff_vo` of the $J/K$ of both $P^\sigma$) |
| $I^\sigma_{ia}$, $I^\sigma_{ai}$ | `imat[s][(i, a_mo)]`, `imat[s][(a_mo, i)]` |

## 15.7.3 Assembly

With $P^\sigma_{\mathrm{rel}} = P^\sigma + z^\sigma$ on both occupied–virtual
blocks and $D^\sigma$ the UHF spin densities,

$$ \frac{dE}{dx} = \operatorname{Tr}\big[(D + P_{\mathrm{rel}})_{\mathrm T}h'\big] + \sum_{\mu\nu}\Big(\sum_\sigma\big[\mathrm{im1}^\sigma - \zeta^\sigma - W^\sigma_{\mathrm{HF}} - P^\sigma_{\mathrm{occ}}v^\sigma[P_{\mathrm{rel}}]P^\sigma_{\mathrm{occ}}\big]\Big)_{\mu\nu}S'_{\mu\nu} + \partial_xU\big((D^\alpha, D^\beta), (D^\alpha + 2P^\alpha_{\mathrm{rel}}, D^\beta + 2P^\beta_{\mathrm{rel}})\big) + \sum_\sigma R^\sigma_{\mathrm{RI}} , \tag{15.15} $$

$$ U(A, B) = \tfrac12\operatorname{Tr}\big[A_{\mathrm T}J[B_{\mathrm T}]\big] - \tfrac12\sum_\sigma\operatorname{Tr}\big[A^\sigma K[B^\sigma]\big], $$

differentiated through the integrals only. In the code:

| Term | Built as |
|---|---|
| $\mathrm{im1}^\sigma$ | $I^\sigma$ with its virtual–occupied block replaced by the transpose of its occupied–virtual block |
| $\zeta^\sigma$ | $\tfrac12(\varepsilon_p + \varepsilon_q)P^\sigma_{\mathrm{rel},pq}$ on the oo and vv blocks, $\varepsilon_iP^\sigma_{\mathrm{rel},ai}$ on both ov blocks |
| $W^\sigma_{\mathrm{HF}}$ | `build_energy_weighted_density_uhf` (unit occupation) |
| $P^\sigma_{\mathrm{occ}}v^\sigma[P_{\mathrm{rel}}]P^\sigma_{\mathrm{occ}}$ | `vhf_s1occ`, with $J/K$ of $P_{\mathrm{rel}}$ by linearity (unrelaxed plus $D^z$) |
| $\partial_xU$ | `two_electron_gradient_bilinear_open` (four-centre or fitted route of the reference) |
| $R^\sigma_{\mathrm{RI}}$ | `ri_response_gradient_channels`, one channel per spin with $y^\sigma = V^{-1/2}x^\sigma$ and weight 1 |

The closed-shell RI response has weight 2 against this weight 1 per spin.

> **Verified equation 15.15** · the assembled gradient
>
> On a model whose overlap, core Hamiltonian, four-centre integrals and RI
> factors and metric all move with one parameter $x$, a UHF with
> $n_\alpha = 3$, $n_\beta = 2$ solved to $10^{-13}$ at every $x$, (15.15) with
> (15.13)–(15.14) equals the Richardson derivative of the re-solved
> $E_{\mathrm{UHF}} + E^{(2)}$, with and without a frozen core (G1), and
> $\operatorname{Tr}[(D + P_{\mathrm{rel}})R]$ equals $dE/d\lambda$ for
> $h \to h + \lambda R$ (G2). The closed-shell limit reproduces the restricted
> assembly of section 15.2 (part C).
>
> | | |
> |---|---|
> | Proof | `scripts/u_rimp2_gradient_sympy.py`, parts G and C |
> | Negative control | the runner drops the $\beta$ Z-vector, drops the opposite-spin Coulomb coupling of (15.14), and doubles the RI weight per spin; each turns G red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,solve_u_zvector,unrelaxed_density_mo,veff_vo,response_density}` |
> | Code test | `o2_triplet_matches_fd_four_centre` in `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` |
>
> <!-- verified: id=mp2.u.grad.assembly; script=scripts/u_rimp2_gradient_sympy.py; claim=G; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2_gradient.rs::{u_rimp2_gradient_analytical,solve_u_zvector,unrelaxed_density_mo,veff_vo,response_density}; rust_test=ferric-mp2/tests/u_rimp2_gradient_fd.rs::o2_triplet_matches_fd_four_centre; registry=u-ri-mp2-gradient -->

The model covers the RI terms at the MO level; the AO bookkeeping of
`ri_response_gradient_channels` is checked only by the Rust tests.

## 15.7.4 The closed-shell limit

For $\alpha \equiv \beta$: $t^{\alpha\alpha} = t - \bar t$ and
$t^{\alpha\beta} = t$, so $x^\sigma$ is the closed-shell $x$; $I^\sigma = I/2$,
$P^\alpha + P^\beta = P + P^{\mathsf T}$, $X^\sigma = X/2$, $z^\sigma = z/2$,
$U$ is the closed-shell bilinear form, and the two weight-1 RI channels sum to
the closed-shell weight-2 channel. A closed-shell RHF written as a UHF gives
the restricted gradient on both $J/K$ routes
(`closed_shell_limit_matches_restricted_four_centre`,
`closed_shell_limit_matches_restricted_rijk`, and the frozen-core variant).

## 15.7.5 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| analytic vs Richardson FD of ferric's own U-RI-MP2 energy, four-centre and RI-JK | OH, NH2 doublets, O2 triplet / 6-31G; OH frozen core 1 | ≤ 3.1e-9 Ha/Bohr (O2), 1.8e-9 (NH2), 8e-10 (OH); frozen core 2.5e-10 | 2.1e-7 to 2.3e-7 Ha/Bohr, derived in the test header | `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` |
| closed-shell limit vs the restricted analytic gradient | H2O / 6-31G, both routes, frozen core | ≤ 3.1e-15 Ha/Bohr | 1e-10 Ha/Bohr | `crates/ferric-mp2/src/u_rimp2_gradient.rs` |

There is no external reference: PySCF's UMP2 gradient uses conventional
integrals, not density fitting. Figures from `site/src/reference/validation.md`.

## Known limits (2026-11-15)

- UHF reference only. A ROHF reference is refused (`ROHF_REFUSAL`): it is not
  stationary under independent $\alpha$/$\beta$ rotations and its MP2 runs on
  semicanonical orbitals, so it needs the ROHF Z-vector and the
  semicanonicalization response, neither of which is built.
- No COSX reference, no X2C, Coulomb operator only, at least one $\beta$
  electron.
- No SCS or OO variant of the unrestricted gradient.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "15. Unrestricted RI-MP2 nuclear gradient" | equations 15.12–15.15 |
| `crates/ferric-mp2/src/u_rimp2_gradient.rs` | implementation |
| `scripts/u_rimp2_gradient_sympy.py` | proofs |
| `site/src/reference/validation.md`, `crates/ferric-mp2/tests/u_rimp2_gradient_fd.rs` | anchors |
