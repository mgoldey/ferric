# 15.2 RI-MP2 gradient with exact and RI-JK references; SCS-MP2

<!-- section: 15.2; status: ready; registry: ri-mp2-gradient, scs-mp2-gradient, att-rimp2-gradient, terfc-rimp2-gradient;
     theory: crates/ferric-mp2/theory.md#"9. Analytic RI-MP2 gradient and Z-vector", "9.1 Fitted (RI-JK) reference", "9.2 Spin-component-scaled (SCS-MP2) gradient", "9.4 Attenuated (erfc) RI-MP2 gradient", "9.5 terfc-attenuated RI-MP2 gradient" -->

The closed-shell RI-MP2 gradient is the Lagrangian of section 15.1 with the
MP2 right-hand side, assembled as one-electron, overlap, two-electron and
RI-response terms. The same code gives the gradient on a fitted (RI-JK)
reference, where four-center $J$ and $K$ are replaced by their fits, and the
SCS-MP2 gradient, where the amplitude weight $2t - \bar t$ is replaced by a
spin-scaled one. This section gives the densities, the substitutions and
what each anchor tests.

| Symbol | Meaning | Code name |
|---|---|---|
| $t_{ij}^{ab}$ | $(ia\vert jb)/\Delta_{ij}^{ab}$, with $\Delta_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$ as written in `theory.md` | `t2` |
| $\tilde T$ | amplitude weight, $2t - \bar t$ for MP2 | |
| $x^P_{ia}$ | $\sum_{jb}\tilde T_{ij}^{ab}B^P_{jb}$ | `x_ov` |
| $P_{ij}, P_{ab}$ | unrelaxed occupied and virtual densities | `p_oo`, `p_vv` |
| $I$ | Lagrangian matrix | `imat` |
| $G[D]$ | $J[D] - \tfrac12K[D]$, four-center or fitted | |

The denominator sign follows `crates/ferric-mp2/theory.md`, where $\Delta$
is negative; the book's positive-gap convention is the negative of it.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/gradient.rs::{rimp2_gradient_analytical,relaxed_ri_gradient,scs_mp2_gradient_analytical,total_rimp2_gradient}` |
| Rust (attenuated) | `ferric-mp2/src/att_rimp2_gradient.rs::{att_rimp2_gradient_analytical,total_att_rimp2_gradient_on,terfc_rimp2_gradient_analytical,total_terfc_rimp2_gradient_on}` |
| CLI | `method.kind = "rimp2"`, `"scs-mp2"` or `"att-rimp2"` with `task = "optimize"` |
| Python | `run_rimp2(with_gradient=True)`, `run_optimize(method="rimp2")`, `run_optimize(method="scs-mp2")`, `run_optimize(method="att-rimp2")` |

## 15.2.1 Densities and the right-hand side

$$ P_{ij} = -\sum_{kab}t_{ik}^{ab}\tilde T_{jk}^{ab}, \qquad P_{ab} = \sum_{ijc}t_{ij}^{ac}\tilde T_{ij}^{bc}, \tag{15.5} $$

$$ I_{qi} = -2\sum_{Pa}x^P_{ia}B^P_{qa}, \qquad I_{qa} = -2\sum_{Pi}x^P_{ia}B^P_{qi}, \tag{15.6} $$

$$ X_{ai} = \big[\mathbf C^{\mathsf T}(2J - K)[D^{\mathrm{corr}}]\,\mathbf C\big]_{ai} + I_{ia} - I_{ai}, \qquad D^{\mathrm{corr}} = \mathbf C\,(P + P^{\mathsf T} + D_{kI})\,\mathbf C^{\mathsf T} . \tag{15.7} $$

With a frozen core the correlation density gains the core-active block
$D_{kI} = I_{kI}/(\varepsilon_k - \varepsilon_I)$ (`frozen_core_block`), and
$X$, $z$ and the energy-weighted density run over every occupied orbital.
The Z-vector equation is (15.3).

**The gradient.** The assembled derivative has five parts:

| Part | Density or weight | Derivative integral |
|---|---|---|
| one-electron | $D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}}$ | $h'$ |
| overlap | $\mathrm{im1} - \zeta - v^{\mathrm{s1occ}}$ | $S'$ |
| separable two-electron | $(D^{\mathrm{HF}}, D^{\mathrm{HF}} + 2D^{\mathrm{corr}}_{\mathrm{rel}})$ | $(\mu\nu\vert\lambda\sigma)'$ |
| three-center RI response | $4y^P_{ia}$ | $(P\vert ia)'$ |
| two-center RI response | $-2\sum_{ia}(V^{-1}u)^P_{ia}y^Q_{ia}$ | $V'_{PQ}$ |

The terms are defined in `theory.md` section 9; $y = V^{-1/2}x$. The $J$ and
$K$ of $2D^{\mathrm{corr}}_{\mathrm{rel}}$ come from linearity: twice those of
$D^{\mathrm{corr}}$ (built for the right-hand side) plus twice those of the
converged $D^z$ (returned by the Z-vector solve).

> **Verified equation 15.7** · the assembled gradient
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
`integral_response_gradient_3c2c` (its $2y$ and symmetrized $\Gamma$) is
checked only by the Rust tests against PySCF and ORCA.

## 15.2.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF, four-center or fitted | `solve_rhf` |
| 2 | $B$, $t$, $x$, $P$ | `Mp2Intermediates::from_ri_densities` and the RI-MP2 energy pass |
| 3 | $I$, frozen-core block | `build_imat_ri`, `frozen_core_block` |
| 4 | $(J, K)[D^{\mathrm{corr}}]$, $X$, then $z$ | `solve_zvector_timed` |
| 5 | one-electron, overlap and separable two-electron terms | `mp2_relaxed_lagrangian_gradient` |
| 6 | three- and two-center RI response | `integral_response_gradient_3c2c` |
| 7 | add the RHF gradient | `total_rimp2_gradient` |

## 15.2.3 The fitted (RI-JK) reference

With `[scf] jk = "rijk"` (Python `scf_jk="rijk"`) the reference minimizes the
fitted energy, whose Fock matrix is $h + G^{\mathrm{fit}}(D)$. The MP2
Lagrangian meets the reference only through $F$, so the derivation is
unchanged and each four-center object is replaced by its fit:

| Term | Four-center | Fitted |
|---|---|---|
| right-hand side | $(2J - K)[D^{\mathrm{corr}}]$ | $2G^{\mathrm{fit}}(D^{\mathrm{corr}})$ |
| Hessian coupling | $(4J - K - K^{\mathsf T})[D^z]$ | $4G^{\mathrm{fit}}(D^z)$ |
| potential term | $(J - \tfrac12K)[2D^{\mathrm{corr}}_{\mathrm{rel}}]$ | $G^{\mathrm{fit}}(2D^{\mathrm{corr}}_{\mathrm{rel}})$ |
| separable two-electron | $\Gamma(D_1, D_2)\,(\mu\nu\vert\lambda\sigma)'$ | $\partial_R B(D_1, D_2)$ at fixed $D_1, D_2$ |

The fitted separable term uses $B(D_1, D_2) = Q(u) - Q(v)$ with
$u = \tfrac12(D_1 + D_2)$ and $v = \tfrac12(D_1 - D_2)$, so it is assembled
from the RI-J and RI-K gradients of a quadratic form. The RI-K gradient
factors its density as $XX^{\mathsf T}$; an indefinite density is split by
eigenvalue sign. The MP2 integral response on its own RI basis does not
change. The result is the gradient of the fitted energy, not an
approximation to the four-center gradient.

**Which reference the CLI uses.** With `[scf] jk` omitted, a closed-shell
MP2-family energy runs on the fitted reference with the JK set matched to the
orbital basis. Any task other than `energy`, an open shell, or a key only the
four-center route reads keeps the four-center reference
(`Config::omitted_scf_jk` in `crates/ferric-cli/src/config.rs`). The stated
reason is memory: the fitted gradient's peak memory is about twice that of
the four-center route.

## 15.2.4 SCS-MP2

$E_{\mathrm{SCS}} = c_{\mathrm{OS}}E_{\mathrm{OS}} + c_{\mathrm{SS}}E_{\mathrm{SS}} = \sum K\tilde T$ with

$$ \tilde T = (c_{\mathrm{OS}} + c_{\mathrm{SS}})\,t - c_{\mathrm{SS}}\,\bar t, \qquad \bar t_{ij}^{ab} = t_{ij}^{ba}, \tag{15.8} $$

which is $2t - \bar t$ at $(1, 1)$. $\tilde T$ enters $x$ (and through $x$
the Lagrangian matrix and the RI response) and the second slot of (15.5).
By bilinearity the densities are
$c_{\mathrm{SS}}P^{\mathrm{MP2}} + (c_{\mathrm{OS}} - c_{\mathrm{SS}})P^{\mathrm{OS}}$,
with $P^{\mathrm{OS}} = P(t, t)$. Nothing else changes.

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

At $(1, 1)$, `apply_spin_scaling` returns without touching the
intermediates, so the RI-MP2 gradient goes through the same code bit for bit
(`scs_unit_scaling_is_the_rimp2_gradient_bit_for_bit`).

> **Verified equation 15.8** · the SCS weights
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

## 15.2.5 Attenuated (erfc) RI-MP2

The attenuated energy is $E_{\mathrm{HF}}[1/r] + s\,E_c[\operatorname{erfc}(\omega r)/r]$.
The reference is Coulomb RHF. The correlation uses $\operatorname{erfc}(\omega r)/r$
in both the $(P\vert ia)$ integrals and the metric $V_{PQ}$. The gradient is
the Lagrangian of this section with the operator chosen per term:

| Term | Operator |
|---|---|
| amplitudes, $x$, $P$, $I$ (15.5, 15.6) | erfc |
| three- and two-center RI response | erfc |
| $(2J - K)[D^{\mathrm{corr}}]$ in (15.7), the Z-vector Hessian, the separable two-electron term | Coulomb |

The Coulomb terms come from the HF stationarity condition, so they keep the
reference operator. The scaling $s$ enters as $c_{\mathrm{OS}} = c_{\mathrm{SS}} = s$
in (15.8). At $\omega \to 0$ the erfc operator becomes Coulomb and the
gradient becomes the RI-MP2 gradient: at $\omega = 10^{-7}$ Bohr⁻¹ the two
agree to $1.6\times10^{-13}$ Ha/Bohr (`att_gradient_matches_rimp2_in_the_trivial_limit`).
The QQR-3 screened energy (`screen_thresh`) is refused for gradient tasks.
The derivation and the operator table are in `theory.md` section 9.4.
`scripts/att_rimp2_gradient_sympy.py` proves the erfc chain rule and the
$\omega \to 0$ limit, and checks the operator split on a one-dimensional
s-Gaussian model against finite differences.

**terfc.** With `att_operator = "terfc"` the correlation kernel is
$\operatorname{terfc}(r; r_0, \omega)/r$ and there is no scaling. The
operator table above holds with terfc in place of erfc (`theory.md` section
9.5). The 3c/2c derivative integrals come from the terf table engine: the
McMurchie-Davidson pass with the Hermite derivative
$2a\,E^{i+1,j}_t - i\,E^{i-1,j}_t$, $R$ one order higher, terfc formed as
Coulomb minus terf in $R$, and the third centre by translational invariance
(`crates/ferric-integrals/theory.md`, "Derivative integrals of terf and
terfc"). At $r_0 \to 0$ terfc becomes erfc: at $r_0 = 10^{-6}$ Bohr the two
gradients agree to $1.2\times10^{-14}$ Ha/Bohr
(`terfc_gradient_matches_erfc_in_the_r0_to_zero_limit`). The terfc gradient
is closed shell only; open-shell terfc and `scs-mp2-2terfc` are refused for
gradient tasks.

> **Verified equation 15.8b** · the terfc RI-MP2 gradient
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

## 15.2.6 Reproducibility and memory

The Z-vector assumes a stationary RHF, so the gradient error follows the
SCF density convergence; tighten `scf_density_conv` for finite-difference
work. An unconverged Z-vector is an error, not a gradient. The fitted
reference costs a few full-rank RI-K gradient channels, which is the memory
reason the CLI keeps the four-center reference for geometry tasks.

## 15.2.7 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[scf] jk`, `scf_jk=` | `"exact"`, `"rijk"` | four-center for geometry tasks; Python four-center | the reference and every $J$, $K$ of the gradient |
| `[mp2] frozen_core`, `frozen_core=` | integer | 0 | adds the core-active block |
| `[mp2] c_os`, `c_ss` (`scs-mp2`) | floats | 6/5, 1/3 (`ScsMp2Config` default) | the weights of (15.8) |
| `[mp2] omega`, `omega=` (`att-rimp2`) | float, Å⁻¹ | 0.420 | the erfc range parameter |
| `[mp2] att_operator`, `att_operator=` (`att-rimp2`) | `"erfc"`, `"terfc"` | `"erfc"` | the correlation kernel; terfc is closed shell only |
| `[mp2] att_r0`, `r0=` (`att-rimp2`, terfc) | float, Å | the basis default (`terfc_r0.rs`) | the terfc cutoff |

`scf_jk="rijcosx"`, `k_builder="cosx"`, `local=`, the attenuation keys,
`kappa` and an open-shell molecule are refused with `with_gradient=True`.

## 15.2.8 Numerical anchors

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
| SCS-MP2 gradient vs central FD of `scs_mp2` | H2O / STO-3G and cc-pVDZ, NH3 / 6-31G; $(6/5, 1/3)$ and $(1, 0)$; frozen core; exact and RI-JK | $\le 5.7\times10^{-8}$ Ha/Bohr (bar $2\times10^{-7}$) | `crates/ferric-mp2/tests/scs_mp2_gradient.rs` |

## 15.2.9 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, sections 9, 9.1, 9.2, 9.4, 9.5 | the derivation, the fitted-term table, the RI-JK shift measurements |
| `scripts/mp2_gradient_sympy.py`, parts G and S | the proofs |
| `scripts/att_rimp2_gradient_sympy.py` | the erfc chain rule, the $\omega \to 0$ limit, the operator split |
| `scripts/terfc_rimp2_gradient_sympy.py` | the terfc closed forms and derivatives vs $k$-space quadrature, the Hermite derivative, the operator split, the $r_0 \to 0$ limit |
| PySCF `grad/mp2.py` | the convention the right-hand side follows (cited in `theory.md`) |

**Known limits (2026-11-16).** Closed shell only; the unrestricted RI-MP2
is energy only. The SCS-MP2 gradient is Proven (narrow): closed shell, Coulomb operator,
exact or RI-JK J/K, frozen core 0 or 1. The attenuated RI-MP2 gradient is
Proven (narrow): erfc and terfc, closed shell, exact or RI-JK J/K (the
open-shell erfc gradient is section 15.7; open-shell terfc is refused); its
frequencies are central differences of it and are not graded. The AO-level RI
bookkeeping is not covered by a script.
