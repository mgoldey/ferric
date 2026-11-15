# 15.3 OO-RI-MP2 gradient

<!-- section: 15.3; status: ready; registry: oo-rimp2-gradient;
     theory: crates/ferric-mp2/theory.md#"9.3 OO-RI-MP2 nuclear gradient" -->

Orbital-optimized RI-MP2 (chapter 13) makes the energy stationary in every
orbital rotation. The Z-vector of section 15.1 exists to supply that
stationarity, so at converged OO orbitals $z = 0$ and no response equation is
solved. The rest of the RI-MP2 assembly of section 15.2 carries over, with
two changes that come from the orbitals being non-canonical.

| Symbol | Meaning | Code name |
|---|---|---|
| $W$ | unrelaxed $(P + P^{\mathsf T})_{oo} \oplus (P + P^{\mathsf T})_{vv}$ in the semicanonical frame | `dm1mo` |
| $F$ | MO Fock matrix of the OO orbitals; $F_{ia} \ne 0$ | `f_mo` |
| $\zeta$ | energy-weighted density | `zeta_ao` |
| $v^{\mathrm{s1occ}}$ | $P_{\mathrm{occ}}\,G[2\mathbf CW\mathbf C^{\mathsf T}]\,P_{\mathrm{occ}}$ | `vhf_s1occ` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/oo_rimp2_gradient.rs::oo_ri_mp2_gradient` |
| CLI | `method.kind = "oo-rimp2"` with `task = "optimize"` |
| Python | `run_oo_rimp2(with_gradient=True)`, `run_optimize(method="oo-rimp2")` |

## 15.3.1 The zero-response gradient

The gradient is the assembly of section 15.2 with $z = 0$,
$D^{\mathrm{corr}} = \mathbf CW\mathbf C^{\mathsf T}$, and

$$ \zeta^{\mathrm{MO}} = \tfrac12\big(FW + (FW)^{\mathsf T}\big), \qquad \zeta^{\mathrm{MO}}_{ia} = \zeta^{\mathrm{MO}}_{ai} = (FW)_{ia}, \qquad \zeta = \mathbf C\zeta^{\mathrm{MO}}\mathbf C^{\mathsf T} + 2\,\mathbf C_oF_{oo}\mathbf C_o^{\mathsf T} . \tag{15.9} $$

In canonical orbitals $FW$ is $\varepsilon_pW_{pq}$, and (15.9) becomes the
$\tfrac12(\varepsilon_p + \varepsilon_q)$ weight of RI-MP2. The occupied-virtual
entry is one-sided, $(FW)_{ia}$, which is the same choice as the $\tilde I$
of the RI-MP2 assembly.

The term $v^{\mathrm{s1occ}}$ is present at $z = 0$. It is the response of
$F(D^{\mathrm{HF}})$ to the overlap connection, not a CPHF term; omitting it
breaks the gradient.

<!-- excerpt: ferric-mp2/src/oo_rimp2_gradient.rs::energy_weighted_density -->
```rust
let f_dot_p = f_mo.dot(dm1mo);
let mut zeta_mo = 0.5 * (&f_dot_p + &f_dot_p.t());
```

| Symbol | Code |
|---|---|
| $FW$ | `f_dot_p` |
| $\zeta^{\mathrm{MO}}$ | `zeta_mo` |
| $W$ | `dm1mo` |

> **Verified equation 15.9** · the $z = 0$ gradient
>
> On the model of section 15.2, with OO-MP2 re-converged to
> $\max|g| < 10^{-11}$ at every displacement, the $z = 0$ gradient with (15.9)
> and $v^{\mathrm{s1occ}}$ equals the finite difference of the OO-MP2 energy.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part O |
> | Negative control | the runner's mutation that drops $v^{\mathrm{s1occ}}$ from the overlap weight turns O red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/oo_rimp2_gradient.rs::{oo_ri_mp2_gradient,energy_weighted_density,vhf_s1occ}` |
> | Code test | `oo_rimp2_h2o_ccpvdz_gradient_self_fd` in `crates/ferric-mp2/tests/validation_oo_rimp2.rs` |
>
> <!-- verified: id=mp2.grad.oo-zero-response; script=scripts/mp2_gradient_sympy.py; claim=O; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/oo_rimp2_gradient.rs::{oo_ri_mp2_gradient,energy_weighted_density,vhf_s1occ}; rust_test=ferric-mp2/tests/validation_oo_rimp2.rs::oo_rimp2_h2o_ccpvdz_gradient_self_fd; registry=oo-rimp2-gradient -->

## 15.3.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | converged OO-RI-MP2 orbitals and their $J/K$ route (four-center or the reference's RI-JK fit) | `oo_ri_mp2` |
| 2 | semicanonical frame: diagonalize $F_{oo}$ and $F_{vv}$ separately | `semicanonical_frame` |
| 3 | $B$, $t$, $P$ in that frame | `compute_b_full_mo`, `build_mp2_density` |
| 4 | $\zeta$, $v^{\mathrm{s1occ}}$ | `energy_weighted_density`, `vhf_s1occ` |
| 5 | one-electron, overlap, two-electron and RI-response terms | as in section 15.2 |

The amplitudes must be built in the semicanonical frame: there $t$ has the
diagonal denominator of the energy expression the OO solver made
stationary. The two $J/K$ builds (of $D^{\mathrm{HF}}$ and of
$2\mathbf CW\mathbf C^{\mathsf T}$) reuse the solver's when the densities
agree.

## 15.3.3 Options and refusals

| Option | Effect |
|---|---|
| `[scf] jk`, `scf_jk=` | the gradient uses the route the orbitals were optimized on |
| `frozen_core=` | must be 0 with `with_gradient=True` (Python refuses otherwise) |

The frozen core is refused because core rotations are not optimized, so
$z = 0$ is not exact there. The unrestricted OO-RI-MP2 is energy only.

## 15.3.4 Numerical anchors

From the `oo-rimp2` row of `site/src/reference/validation.md`.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| analytic gradient vs FD of its own energy | H2O, NH3 / cc-pVDZ | $8\times10^{-9}$ Ha/Bohr | `crates/ferric-mp2/tests/validation_oo_rimp2.rs` |
| energy vs an independent numpy OO-RI-MP2 | H2O, NH3, CH3 (UHF) / cc-pVDZ | $7.5\times10^{-13}$ Ha | the same |

The gradient is not graded; the energy is Proven (narrow).

## 15.3.5 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, section 9.3 | (15.9) and $v^{\mathrm{s1occ}}$ |
| module documentation of `crates/ferric-mp2/src/oo_rimp2_gradient.rs` | the no-Z-vector argument |
| `scripts/mp2_gradient_sympy.py`, part O | the proof |

**Known limits (2026-11-15).** Closed shell, frozen core 0. Not graded.
