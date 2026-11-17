# OO-RI-MP2 gradient {#oo-rimp2-gradient}

<!-- status: ready; registry: oo-rimp2-gradient -->

Orbital-optimized RI-MP2 (section @sec:oo-rimp2) rotates the orbitals until the MP2
energy itself is stationary. That removes the reason the Z-vector exists.
Section @sec:lagrangian-zvector introduced $z$ to account for the first-order change of the
energy when the orbitals relax; at converged OO orbitals that change is zero
in every direction. So $z = 0$ and no response equation is solved. The rest
of the RI-MP2 assembly of section @sec:rimp2-gradient carries over, with two changes that
come from the orbitals being non-canonical.

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

## The zero-response gradient {#oo-rimp2-gradient-zero-response-gradient}

Why is the OO point stationary in *every* rotation, and not only in the
occupied–virtual ones the optimizer moves? Two facts combine. The optimizer
drives the occupied–virtual orbital gradient to zero. And the functional of
section @sec:oo-rimp2-functional evaluates the MP2 part in the semicanonical frame, so it is
invariant to occupied–occupied and virtual–virtual rotations: its gradient in
those directions is zero identically. With $\partial E/\partial\kappa = 0$ in
all directions, the right-hand side of @sec:lagrangian-zvector-lagrangian vanishes and the Lagrangian
needs no $z$. The functional's invariance matters here. A functional that is
not invariant in the occupied–occupied and virtual–virtual blocks would leave
a non-zero gradient there at its converged point, and a $z = 0$ gradient
would then be wrong.

The gradient is the assembly of section @sec:rimp2-gradient with $z = 0$,
$D^{\mathrm{corr}} = \mathbf CW\mathbf C^{\mathsf T}$, and

$$ \zeta^{\mathrm{MO}} = \tfrac12\big(FW + (FW)^{\mathsf T}\big), \qquad \zeta^{\mathrm{MO}}_{ia} = \zeta^{\mathrm{MO}}_{ai} = (FW)_{ia}, \qquad \zeta = \mathbf C\zeta^{\mathrm{MO}}\mathbf C^{\mathsf T} + 2\,\mathbf C_oF_{oo}\mathbf C_o^{\mathsf T} . \label{eq:mp2-grad-oo-zero-response} $$

Check the canonical limit. With $F$ diagonal, $(FW)_{pq} = \varepsilon_pW_{pq}$,
and the symmetric part is $\tfrac12(\varepsilon_p + \varepsilon_q)W_{pq}$, the
$\zeta$ weight of RI-MP2 in section @sec:rimp2-gradient. The second term becomes
$2\sum_i\varepsilon_i\mathbf C_i\mathbf C_i^{\mathsf T}$, the Hartree–Fock
energy-weighted density.

The occupied–virtual entry is one-sided, $(FW)_{ia}$, which is the same choice
as the $\tilde I$ of the RI-MP2 assembly. At a stationary point the
occupied–virtual block of the Lagrangian can be taken in either
representation, but it must be the same one in $\zeta$ and in $\tilde I$.
After orbital optimization $F_{ia} \ne 0$, so the symmetric form
$\tfrac12(FW + WF)_{ia}$ differs from $(FW)_{ia}$, and mixing the two costs
$6\times10^{-6}$ to $3\times10^{-5}$ in the gradient (module documentation of
`oo_rimp2_gradient.rs`).

The term $v^{\mathrm{s1occ}}$ is present at $z = 0$. It is the response of
$F(D^{\mathrm{HF}})$ to the overlap connection, not a CPHF term: the MP2
energy depends on the Fock matrix, the Fock matrix depends on
$D^{\mathrm{HF}}$, and $D^{\mathrm{HF}}$ moves with the occupied orbitals when
the overlap changes, with or without a Z-vector. Omitting it breaks the
gradient.

The MO part of $\zeta$ comes from `ferric-mp2/src/oo_rimp2_gradient.rs::energy_weighted_density`:

<!-- excerpt: ferric-mp2/src/oo_rimp2_gradient.rs::energy_weighted_density -->
```rust
let f_dot_p = f_mo.dot(dm1mo);
let mut zeta_mo = 0.5 * (&f_dot_p + &f_dot_p.t());
for i in 0..nocc_total {
    for a in nocc_total..nmo {
        zeta_mo[(i, a)] = f_dot_p[(i, a)];
        zeta_mo[(a, i)] = f_dot_p[(i, a)];
    }
}
```

| Code | Equation |
|---|---|
| `f_dot_p` | $FW$ |
| `zeta_mo` before the loop | $\tfrac12\big(FW + (FW)^{\mathsf T}\big)$ |
| the loop | the one-sided $\zeta^{\mathrm{MO}}_{ia} = \zeta^{\mathrm{MO}}_{ai} = (FW)_{ia}$ |
| `dm1mo` | $W$ |

The function then transforms to the AO basis and adds
$2\mathbf C_oF_{oo}\mathbf C_o^{\mathsf T}$ over the full occupied block, which
is not diagonal between core and active orbitals when `frozen_core > 0`.

> **Verified equation @eq:mp2-grad-oo-zero-response** · the $z = 0$ gradient
>
> On the model of section @sec:rimp2-gradient, with OO-MP2 re-converged to
> $\max|g| < 10^{-11}$ at every displacement, the $z = 0$ gradient with (@eq:mp2-grad-oo-zero-response)
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

The model agrees with the finite difference to $8\times10^{-12}$ relative.

## The algorithm {#oo-rimp2-gradient-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | converged OO-RI-MP2 orbitals and their $J/K$ route (four-center or the reference's RI-JK fit) | `oo_ri_mp2` |
| 2 | semicanonical frame: diagonalize $F_{oo}$ and $F_{vv}$ separately | `semicanonical_frame` |
| 3 | $B$, $t$, $P$ in that frame | `compute_b_full_mo`, `build_mp2_density` |
| 4 | $\zeta$, $v^{\mathrm{s1occ}}$ | `energy_weighted_density`, `vhf_s1occ` |
| 5 | one-electron, overlap, two-electron and RI-response terms | as in section @sec:rimp2-gradient |

The amplitudes must be built in the semicanonical frame: there $t$ has the
diagonal denominator of the energy expression the OO solver made
stationary. The two $J/K$ builds (of $D^{\mathrm{HF}}$ and of
$2\mathbf CW\mathbf C^{\mathsf T}$) reuse the solver's when the densities
agree. The two-electron derivative and the RI 3c/2c terms are the plain
RI-MP2 ones, because the OO energy depends on the integrals and on $B$
exactly as plain RI-MP2 does.

## Options and refusals {#oo-rimp2-gradient-options-refusals}

| Option | Effect |
|---|---|
| `[scf] jk`, `scf_jk=` | the gradient uses the route the orbitals were optimized on |
| `frozen_core=` | must be 0 with `with_gradient=True` (Python refuses otherwise) |

The frozen core is refused because core rotations are not optimized: the
point is not stationary in the core–active and core–virtual directions, so
$z = 0$ is not exact there. The unrestricted OO-RI-MP2 is energy only.

## Numerical anchors {#oo-rimp2-gradient-numerical-anchors}

From the `oo-rimp2` row of `site/src/reference/validation.md`.

| Quantity | Systems | Error | Test |
|---|---|---|---|
| analytic gradient vs FD of its own energy | H2O, NH3 / cc-pVDZ | $8\times10^{-9}$ Ha/Bohr | `crates/ferric-mp2/tests/validation_oo_rimp2.rs` |
| analytic gradient vs Richardson FD, RI-JK reference | H2O / cc-pVDZ, NH3 / 6-31G | $\le 1.8\times10^{-9}$ Ha/Bohr (bar $2\times10^{-8}$) | `crates/ferric-mp2/tests/oo_rimp2_gradient_fd.rs` |
| analytic gradient vs Richardson FD, exact J/K | NH3 / 6-31G | $6.6\times10^{-10}$ Ha/Bohr | the same |
| energy vs an independent numpy OO-RI-MP2 | H2O, NH3, CH3 (UHF) / cc-pVDZ | $7.5\times10^{-13}$ Ha | the same |

The finite differences here re-run the RHF and the full orbital optimization
at each displaced geometry. Differencing at frozen orbitals would test a
different function. The gradient is Proven (narrow): closed shell, frozen
core 0, exact or RI-JK J/K, two molecules at double-zeta size or smaller. No
independent code has an orbital-optimized MP2 gradient to compare with
(PySCF has no OMP2); the independent evidence is ORCA's analytic gradient
(loose, $3\times10^{-5}$) and the finite differences of ferric's own energy.
The energy is Proven (narrow).

## Sources {#oo-rimp2-gradient-sources}

| Source | What it gives |
|---|---|
| module documentation of `crates/ferric-mp2/src/oo_rimp2_gradient.rs` | the no-Z-vector argument |
| `scripts/mp2_gradient_sympy.py`, part O | the proof |

**Known limits (2026-11-15).** Closed shell, frozen core 0. Frequencies,
central differences of this gradient, are not graded.
