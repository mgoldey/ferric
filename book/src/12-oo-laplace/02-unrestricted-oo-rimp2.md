# 12.2 Unrestricted orbital-optimized RI-MP2

<!-- section: 12.2; status: verify; registry: u-oo-rimp2;
     theory: crates/ferric-mp2/theory.md#"14. Unrestricted OO-RI-MP2 orbital gradient", "7. Orbital-optimized RI-MP2" -->

*Notation follows `book/notation.md`: $\beta$ orbitals carry an overbar
($\bar\imath, \bar a$), and $\sigma$ stands for either spin. `theory.md`
section 14 writes $\beta$ indices as capitals instead.*

The unrestricted driver minimizes the UHF plus U-RI-MP2 energy over two
independent rotations, one per spin. Its loop has the shape of section 12.1.
It differs in four places:

- the gradient has a Coulomb coupling between the spins (12.6);
- there is no J/K route choice;
- the safeguard against a failed backtrack is different;
- the convergence test is different.

| Index | Range | Code name |
|---|---|---|
| $i, j, k$; $\bar\imath, \bar\jmath, \bar k$ | active $\alpha$; active $\beta$ occupied | `0..nocc_a`; `0..nocc_b`, offset `first_occ` |
| $a, b, c$; $\bar a, \bar b, \bar c$ | $\alpha$; $\beta$ virtual | `0..nvir_a`; `0..nvir_b` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/u_oo_rimp2.rs::u_oo_ri_mp2`; configuration `UOoRiMp2Config` |
| CLI | `method.kind = "oo-rimp2"` on an open-shell molecule, which arrives with a UHF reference (`solve_open_shell_reference`); same `[mp2]` keys as section 12.1 (`run_oo_rimp2` in `crates/ferric-cli/src/lib.rs` dispatches) |
| Python | none: `run_oo_rimp2` is closed shell only |

The reference must be UHF or ROHF; a restricted reference is refused
(`check_u_oo_reference`). Energy only: there is no unrestricted OO nuclear
gradient.

## 12.2.1 The functional and the rotation

With $D^\sigma = C^\sigma_o C^{\sigma\mathsf T}_o$ and
$F^\sigma = h + J[D^\alpha + D^\beta] - K[D^\sigma]$,

$$ E(C^\alpha, C^\beta) = \tfrac12\sum_\sigma \operatorname{Tr}\big[D^\sigma(h + F^\sigma)\big] + V_{nn} + E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}, \tag{12.5} $$

with the three U-RI-MP2 terms of section 10.2 evaluated in each spin's
semicanonical frame. The code builds both $D^\sigma$ and the energy in
`ferric-mp2/src/u_oo_rimp2.rs::compute_uhf_energy`, with exact four-centre
`DirectJ` and `DirectK` at a $10^{-12}$ threshold. Each spin rotates as
$C^\sigma \leftarrow C^\sigma U(\kappa^\sigma)$ with the Cayley form (12.2).
$\kappa^\alpha$ and $\kappa^\beta$ are independent.

## 12.2.2 The orbital gradient

$$ g^\sigma_{ai} = g^\sigma_{\mathrm{MP2},ai} + 2F^\sigma_{ai} + 2\big[C^{\sigma\mathsf T}\big(J[P^\alpha + P^\beta] - K[P^\sigma]\big)C^\sigma\big]_{ai} + 2\big(F^\sigma W^\sigma - W^\sigma F^\sigma\big)_{ai}, \qquad P^\sigma = C^\sigma W^\sigma C^{\sigma\mathsf T}. \tag{12.6} $$

All quantities are in the semicanonical frame of spin $\sigma$. The result is
rotated back as in section 12.1.

- **The HF term** is $2F^\sigma_{ai}$, half the closed-shell $4F_{ai}$,
  because each spin carries one electron per orbital
  (`add_hf_gradient`).
- **$W^\sigma$** is the symmetrized unrelaxed U-MP2 density
  $\tfrac12(P^\sigma_{\mathrm{MO}} + P^{\sigma\mathsf T}_{\mathrm{MO}})$ on
  the occupied and virtual blocks
  (`ferric-mp2/src/u_rimp2.rs::build_u_mp2_density`, embedded by
  `embed_u_density`). In $\alpha$:
  $$ P_{ij} = -\tfrac12\sum_{kab} t^{ab}_{ik}t^{ab}_{jk} - \sum_{\bar k a\bar b} t^{a\bar b}_{i\bar k}t^{a\bar b}_{j\bar k}, \qquad P_{ab} = \tfrac12\sum_{ijc} t^{ac}_{ij}t^{bc}_{ij} + \sum_{i\bar\jmath\bar c} t^{a\bar c}_{i\bar\jmath}t^{b\bar c}_{i\bar\jmath}, \tag{12.7} $$
  with antisymmetric same-spin amplitudes. The $\beta$ blocks follow by
  exchanging the spins.
- **The Coulomb term** couples the spins: $J$ is built once from
  $P^\alpha + P^\beta$, and $K$ separately for each spin.
- **The integral response** $g^\sigma_{\mathrm{MP2}}$ is
  `ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient`. Its doc comment
  writes it per spin as
  $\sum_{P,x}B^P_{cx}\Gamma_k[P,x] - \tfrac12\sum_{ijy}N_{ij}[c,y](ik|jy)$,
  plus the $\alpha\beta$ blocks.

> **Unverified equation 12.6** · unrestricted OO-MP2 orbital gradient
>
> Equation (12.6), with $g^\sigma_{\mathrm{MP2}}$ the derivative of the
> U-MP2 energy through its integrals at fixed amplitudes, equals
> $\partial E/\partial\kappa^\sigma_{ai}$ of (12.5).
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_gradient_sympy.py`, part U: the derived gradient against a Richardson finite difference of the spin-orbital energy on a 3$\alpha$/2$\beta$ model, at a point rotated off both the UHF solution and the semicanonical frame; agreement $5\times10^{-12}$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Missing | part U computes the integral term from its own derivation; it does not transcribe the loop of `compute_u_mp2_orbital_gradient`. The HF and Fock-response terms (`u_gradient_solver_frame`) are not transcribed either. That link is the open item behind the registry gap. |
> | Code | `ferric-mp2/src/u_oo_rimp2.rs::u_gradient_solver_frame`, `ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient` |
> | Code test | `u_oo_gradient_matches_fd_on_and_off_reference_oh_631g` in `crates/ferric-mp2/src/u_oo_rimp2.rs`: the production gradient against a finite difference of the production energy on OH/6-31G, at rotations of norm 0, 0.05 and 0.2 |
>
> <!-- unverified: id=oo.u.gradient; todo=transcribe compute_u_mp2_orbital_gradient and u_gradient_solver_frame into scripts/mp2_gradient_sympy.py part U -->

The Rust test and part U together cover the equation from both sides. The
test shows that the code's gradient is the derivative of the code's energy.
The script shows that the written equation is the derivative of the written
energy. What is not shown is that the code computes the written equation term
by term.

## 12.2.3 The algorithm

The loop follows the table in section 12.1.4. The differences, read from
`u_oo_ri_mp2`:

| Rule | Closed shell (12.1) | Unrestricted |
|---|---|---|
| Step | one $\kappa$ | $\kappa^\sigma_{ai} = -g^\sigma_{ai}/(\varepsilon^\sigma_a - \varepsilon^\sigma_i + \mu)$, one $\mu$ shared by both spins |
| Cap | 0.3 rad | `max_kappa` (default 0.3 rad), applied to each spin separately |
| DIIS | on $C$ | one DIIS per spin, each followed by Löwdin orthonormalization |
| Backtrack accepted | $E \le E_{\mathrm{old}} + 10^{-12}$ | same |
| No halving accepted | takes the last halved step | keeps the old point, doubles $\mu$ up to `MU_MAX` = 5.0 Ha, resets DIIS; after `STUCK_LIMIT` = 3 such iterations it returns unconverged |
| Gradient norm | $\lVert g\rVert_2$ | $\big(\lVert g^\alpha\rVert_2^2 + \lVert g^\beta\rVert_2^2\big)^{1/2}$ |
| Energy-change stop | recomputes the gradient at the new point, needs $< 10\,$`grad_conv` | uses the gradient norm of the previous point, needs $< 10\,$`grad_conv`; no recompute |
| J and K | the reference's route (`OoJk`) | always exact four-centre `DirectJ`/`DirectK` |

A system with no $\beta$ electrons ($n_{\bar o} = 0$) has a zero $\beta$
gradient and skips the $\beta$ exchange build.

## 12.2.4 The kernel

From `u_gradient_solver_frame`, the Fock response added to each spin's
gradient:

Abridged from `ferric-mp2/src/u_oo_rimp2.rs::u_gradient_solver_frame` (not a verbatim excerpt):
```rust
let g_mo = c.t().dot(&(&j_tot - k)).dot(c);
let fw = f_mo.dot(w);
let wf = w.dot(f_mo);
// ...
g[(a, i)] += 2.0 * g_mo[(a_mo, i)] + 2.0 * (fw[(a_mo, i)] - wf[(a_mo, i)]);
```

| Code | Equation |
|---|---|
| `j_tot` | $J[P^\alpha + P^\beta]$ |
| `k` | $K[P^\sigma]$ |
| `w` | $W^\sigma$ |
| `f_mo` | $F^\sigma$ in the semicanonical frame of $\sigma$ |
| `g` on entry | $g^\sigma_{\mathrm{MP2}} + 2F^\sigma$ from `add_hf_gradient` |

## 12.2.5 Memory

- The three amplitude blocks $t^{\alpha\alpha}$, $t^{\beta\beta}$ and
  $t^{\alpha\bar\beta}$ stay resident for the whole loop.
  `check_u_amplitude_alloc` refuses a run whose blocks exceed the budget.
- The two full MO three-index tensors are built together;
  `check_u_bfull_alloc` guards them.
- Tests on both guards: `check_u_amplitude_alloc_rejects_realistic_large_scale`
  and `check_u_bfull_alloc_rejects_large_and_accepts_small_scale`.

## 12.2.6 Options

`UOoRiMp2Config` has the fields and defaults of section 12.1.7, plus
`max_kappa` (0.3) and `memory_budget_bytes`. `frozen_core` applies to both
spins (`active_occ` per spin). An external potential and ECPs are folded into
$h$ by `hcore_ecp_with_external`. A relativistic reference is refused.

## 12.2.7 Anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| UHF CH3/cc-pVDZ total energy | independent numpy OO-RI-MP2 | within the $7.5\times10^{-13}$ Ha figure the validation page states for all three systems | `u_oo_rimp2_ch3_ccpvdz_energy_vs_orca` in `crates/ferric-mp2/tests/validation_oo_rimp2.rs` |
| H2 singlet through the unrestricted driver | closed-shell OO-RI-MP2 | $10^{-8}$ Ha asserted | `u_oo_rimp2_matches_closed_shell_on_h2` |
| Converged OH/6-31G point is stationary | $g^\alpha = g^\beta = 0$ | $10^{-10}$ asserted | `u_oo_converged_point_is_stationary_oh_631g` |

<!-- figures: site/src/reference/validation.md row "oo-rimp2" -->

The validation page lists the unrestricted path under the `oo-rimp2` row as
energy only. The registry keeps `u-oo-rimp2` as a gap until the
transcription in the box above lands.

**Known limits (2026-11-15).**

- J and K are always exact four-centre, whatever route the reference SCF used.
- There is no nuclear gradient and no Python entry point.
- The convergence recheck does not recompute the gradient (table above).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, sections 7 and 14 | equations 12.5 and 12.6 |
| `crates/ferric-mp2/src/u_oo_rimp2.rs`, `crates/ferric-mp2/src/u_rimp2.rs` | the loop rules, the densities (12.7), the kernel |
| `scripts/mp2_gradient_sympy.py`, part U | the proof cited in the box |
| U. Bozkaya, J. Chem. Phys. 139, 154105 (2013) | the unrestricted functional, as cited in the code |
| `site/src/reference/validation.md` | the anchor figures |

Changes from the sources:

- $\beta$ indices are written with overbars, not capitals.
- The $\mu$ doubling with its three-strike exit, and the unrestricted
  energy-change stop that skips the gradient recompute, are stated from the
  loop, as in `theory.md` section 7.
