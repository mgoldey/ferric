# 12.2 Unrestricted orbital-optimized RI-MP2

<!-- section: 12.2; status: verify; registry: u-oo-rimp2;
     theory: crates/ferric-mp2/theory.md#"14. Unrestricted OO-RI-MP2 orbital gradient", "14.1 The integral-response loop", "7. Orbital-optimized RI-MP2" -->

*Notation follows `book/notation.md`: $\beta$ orbitals carry an overbar
($\bar\imath, \bar a$), and $\sigma$ stands for either spin. `theory.md`
section 14 writes $\beta$ indices as capitals instead.*

In an open-shell molecule the $\alpha$ and $\beta$ electrons feel different
exchange fields, so a UHF reference gives each spin its own orbitals. When
MP2 correlation is added, each set relaxes in its own way. The unrestricted
driver therefore minimizes the UHF plus U-RI-MP2 energy over two independent
rotations, one per spin. The two sets still interact: the Coulomb repulsion
sees the total density, so a rotation of the $\alpha$ orbitals changes the
field the $\beta$ electrons move in.

The loop has the shape of section 12.1. It differs in four places:

- the gradient has a Coulomb coupling between the spins (12.6);
- J and K are built per spin (`UOoJk`);
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

Each spin has its own density $D^\sigma = C^\sigma_o C^{\sigma\mathsf T}_o$,
with no factor 2 because each orbital holds one electron. Its Fock matrix
$F^\sigma = h + J[D^\alpha + D^\beta] - K[D^\sigma]$ has Coulomb from both
spins and exchange from its own spin only. The functional is

$$ E(C^\alpha, C^\beta) = \tfrac12\sum_\sigma \operatorname{Tr}\big[D^\sigma(h + F^\sigma)\big] + V_{nn} + E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}, \tag{12.5} $$

with the three U-RI-MP2 terms of section 10.2 evaluated in each spin's
semicanonical frame. The argument of section 12.1.1 applies to each spin
separately, so (12.5) is invariant to occupied–occupied and
virtual–virtual rotations within either spin.

The code builds both $D^\sigma$ and the energy in
`ferric-mp2/src/u_oo_rimp2.rs::compute_uhf_energy`. J and K follow the
reference's route: four-centre `DirectJ` and `DirectK` at a $10^{-12}$
threshold, or the reference's RI-JK fit when it fitted J and K on one
auxiliary basis (`UOoJk::for_reference`). Each spin rotates as
$C^\sigma \leftarrow C^\sigma U(\kappa^\sigma)$ with the Cayley form (12.2).
$\kappa^\alpha$ and $\kappa^\beta$ are independent.

## 12.2.2 The orbital gradient

The derivation follows section 12.1.3 term by term. A rotation
$\kappa^\sigma_{ck}$ moves $\delta C^\sigma_k = C^\sigma_c$ and
$\delta C^\sigma_c = -C^\sigma_k$, and touches only spin $\sigma$.

- **The HF term.** $\delta D^\sigma = C^\sigma_cC^{\sigma\mathsf T}_k + C^\sigma_kC^{\sigma\mathsf T}_c$
  has no factor 2, so $\operatorname{Tr}[F^\sigma\delta D^\sigma] = 2F^\sigma_{ck}$:
  half the closed-shell $4F_{ai}$ (`add_hf_gradient`).
- **The Fock response.** Each $F^\tau$ depends on $D^\sigma$ through $J$,
  and $F^\sigma$ also through $K$. Contracting with the per-spin MP2
  densities and using
  $\sum_\tau\operatorname{Tr}[P^\tau\,\mathcal G^\tau[\delta D]] = \operatorname{Tr}[\delta D\,\mathcal G[P]]$
  collects all the $J$ pieces into one build on $P^\alpha + P^\beta$.
- **The integral response** comes from the rotation derivative of the
  integrals, now in three spin blocks.

The result:

$$ g^\sigma_{ai} = g^\sigma_{\mathrm{MP2},ai} + 2F^\sigma_{ai} + 2\big[C^{\sigma\mathsf T}\big(J[P^\alpha + P^\beta] - K[P^\sigma]\big)C^\sigma\big]_{ai} + 2\big(F^\sigma W^\sigma - W^\sigma F^\sigma\big)_{ai}, \qquad P^\sigma = C^\sigma W^\sigma C^{\sigma\mathsf T}. \tag{12.6} $$

All quantities are in the semicanonical frame of spin $\sigma$. The result is
rotated back as in section 12.1.

**The density $W^\sigma$.** It is the symmetrized unrelaxed U-MP2 density
$\tfrac12(P^\sigma_{\mathrm{MO}} + P^{\sigma\mathsf T}_{\mathrm{MO}})$ on the
occupied and virtual blocks
(`ferric-mp2/src/u_rimp2.rs::build_u_mp2_density`, embedded by
`embed_u_density`). In $\alpha$:

$$ P_{ij} = -\tfrac12\sum_{kab} t^{ab}_{ik}t^{ab}_{jk} - \sum_{\bar k a\bar b} t^{a\bar b}_{i\bar k}t^{a\bar b}_{j\bar k}, \qquad P_{ab} = \tfrac12\sum_{ijc} t^{ac}_{ij}t^{bc}_{ij} + \sum_{i\bar\jmath\bar c} t^{a\bar c}_{i\bar\jmath}t^{b\bar c}_{i\bar\jmath}, \tag{12.7} $$

with antisymmetric same-spin amplitudes. The $\tfrac12$ on the same-spin
sums corrects for counting each antisymmetric pair twice; the
opposite-spin sums have distinguishable partners and no $\tfrac12$. The
$\beta$ blocks follow by exchanging the spins.

**The Coulomb coupling.** $J$ is built once from $P^\alpha + P^\beta$, and
$K$ separately for each spin. Nowhere else in (12.6) does one
spin's correlation density enter the other spin's gradient directly.

**The integral response.** $g^\sigma_{\mathrm{MP2}}$ is
`ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient`. Per spin it is
$\sum_{P,x}B^P_{cx}\Gamma_k[P,x] - \tfrac12\sum_{ijy}N_{ij}[c,y](ik|jy)$ plus
the $\alpha\bar\beta$ blocks. For antisymmetric amplitudes the pair
combination is $N_{ij} = 4t_{ij}$ and $\Gamma_k = 2\sum_jB_jt_{kj}^{\mathsf T}$.
Written out for $\alpha$:

$$ g^{\alpha\alpha}_{ck} = 2\sum_{jab}t^{ab}_{kj}(ca|jb) - 2\sum_{ijb}t^{cb}_{ij}(ik|jb), \qquad
g^{\alpha\bar\beta}_{ck} = 2\sum_{\bar\jmath a\bar b}t^{a\bar b}_{k\bar\jmath}(ca|\bar\jmath\bar b) - 2\sum_{i\bar\jmath\bar b}t^{c\bar b}_{i\bar\jmath}(ik|\bar\jmath\bar b). \tag{12.8} $$

Where the two factors of 2 come from:

- In $g^{\alpha\alpha}$, the derivative of $(ia|jb)$ has two occupied terms
  ($i = k$ and $j = k$) and two virtual terms. With
  $t_{ijab} = t_{jiba}$ each pair is folded into one term with a factor 2.
  In the code it is the $\tfrac12$ of `pair_gamma_rows` times the 4 of the
  pair combination.
- In $g^{\alpha\bar\beta}$, the opposite-spin energy is
  $\sum(ia|\bar\jmath\bar b)^2/D$, and differentiating the square gives the
  factor 2 (the chain-rule factor of `theory.md` section 14.1). In the code
  it is the explicit factor 2 on the cross columns.

Check the closed-shell limit. With $C^\alpha = C^\beta$ and
$n_\alpha = n_\beta$ the kernel gives identical $\alpha$ and $\beta$
blocks, and the closed-shell gradient is $g = g^\alpha + g^\beta$. With
$B = 0$ the kernel returns zero. Both are checked in the script's
`closed-shell-anchor`.

> **Verified equation 12.6** · unrestricted OO-MP2 orbital gradient
>
> The integral-response kernel and `u_gradient_solver_frame` are transcribed
> loop by loop. The kernel equals (12.8) and the section-14 derivation to
> $6\times10^{-17}$ relative (bar $10^{-13}$), and each block equals a
> Richardson difference of its block energy at fixed denominators to
> $7\times10^{-13}$ relative (bar $10^{-9}$), with and without one frozen
> core orbital. The assembled gradient (12.6) equals a Richardson difference
> of the spin-orbital energy (12.5) at non-stationary, non-semicanonical
> orbitals of a 3$\alpha$/2$\beta$ model to $1.4\times10^{-11}$ relative (bar
> $10^{-9}$). `scripts/mp2_gradient_sympy.py`, part U, checks the derived
> form of the same equation independently.
>
> | | |
> |---|---|
> | Proof | `scripts/u_oo_rimp2_response_sympy.py`, checks `pair-combination`, `chain-rule-factors`, `kernel-equals-derivation`, `kernel-vs-fd`, `kernel-vs-fd-frozen-core`, `assembled-vs-fd`, `closed-shell-anchor` |
> | Negative control | the runner sets the same-spin $\Gamma$ weight to 1, the $\alpha\bar\beta$ factor to 1, flips the sign of the same-spin and of the cross $(ik\vert jb)$ term, drops the frozen-core row offset, uses $u_o$ for $u_o^{\mathsf T}$, halves the HF term, flips a sign in the pair combination; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_oo_rimp2.rs::u_gradient_solver_frame`, `ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient` |
> | Code test | `u_oo_gradient_matches_fd_on_and_off_reference_oh_631g` in `crates/ferric-mp2/src/u_oo_rimp2.rs`: the production gradient against a finite difference of the production energy on OH/6-31G, at rotations of norm 0, 0.05 and 0.2 |
>
> <!-- verified: id=oo.u.gradient; script=scripts/u_oo_rimp2_response_sympy.py; claim=assembled-vs-fd; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_oo_rimp2.rs::u_gradient_solver_frame, ferric-mp2/src/u_rimp2.rs::compute_u_mp2_orbital_gradient; rust_test=ferric-mp2/src/u_oo_rimp2.rs::u_oo_gradient_matches_fd_on_and_off_reference_oh_631g; registry=u-oo-rimp2 -->

The two checks cover different things. The Rust test shows that the code's
gradient is the derivative of the code's energy. The script shows that the
code's loop computes (12.6) term by term and that (12.6) is the derivative
of (12.5).

**Frozen core.** The kernel returns only the active columns.
`embed_active_columns` places them at MO columns $n_c$ to $n_{\mathrm{occ}}$
and leaves the core columns zero before `add_hf_gradient` adds the HF term.
The Fock response needs no change, because $W^\sigma$ is zero on the core
rows and columns. The back-rotation and the step use the active columns
only.

## 12.2.3 The algorithm

The loop follows the table in section 12.1.4. The differences, read from
`u_oo_ri_mp2`:

| Rule | Closed shell (12.1) | Unrestricted |
|---|---|---|
| Step | one $\kappa$ | $\kappa^\sigma_{ai} = -g^\sigma_{ai}/(\varepsilon^\sigma_a - \varepsilon^\sigma_i + \mu)$, one $\mu$ shared by both spins |
| Cap | 0.3 rad | `max_kappa` (default 0.3 rad), applied to each spin separately |
| DIIS | on $C$ | one DIIS per spin, each followed by Löwdin orthonormalization |
| Backtrack accepted | $E \le E_{\mathrm{old}} + 10^{-12}$ | same; the halvings scale both spins together |
| No halving accepted | takes the last halved step | keeps the old point, doubles $\mu$ up to `MU_MAX` = 5.0 Ha, resets DIIS; after `STUCK_LIMIT` = 3 such iterations it returns unconverged |
| Gradient norm | $\lVert g\rVert_2$ | $\big(\lVert g^\alpha\rVert_2^2 + \lVert g^\beta\rVert_2^2\big)^{1/2}$ |
| Energy-change stop | recomputes the gradient at the new point, needs $< 10\,$`grad_conv` | uses the gradient norm of the previous point, needs $< 10\,$`grad_conv`; no recompute |
| J and K | the reference's route (`OoJk`) | the reference's route per spin (`UOoJk`) |

Why does the unrestricted driver double $\mu$ instead of taking a tiny
step? A larger $\mu$ shortens every element of the next step, as
$\kappa \propto 1/(\text{gap} + \mu)$, and weights them more evenly across
small and large gaps. The doubling never reverses within a run, and an
accepted step resets the strike count.

A system with no $\beta$ electrons ($n_{\bar o} = 0$) has a zero $\beta$
gradient and skips the $\beta$ exchange build.

## 12.2.4 The kernel

The Fock response is the third and fourth terms of (12.6). Below is the
closure `add_response` inside
`ferric-mp2/src/u_oo_rimp2.rs::u_gradient_solver_frame`, with the closure
parameters dropped and the comments added. It runs once per spin, with that
spin's `c`, `k`, `w` and `f_mo`; `j_tot` is shared:

```rust
let g_mo = c.t().dot(&(&j_tot - k)).dot(c);  // C^T (J[P^a + P^b] - K[P^s]) C
let fw = f_mo.dot(w);                        // F^s W^s
let wf = w.dot(f_mo);                        // W^s F^s
let (nvir, nocc_cols) = g.dim();
for a in 0..nvir {
    let a_mo = nocc_total + a;
    for i in 0..nocc_cols {
        g[(a, i)] += 2.0 * g_mo[(a_mo, i)] + 2.0 * (fw[(a_mo, i)] - wf[(a_mo, i)]);
    }
}
```

| Code | Equation |
|---|---|
| `j_tot` | $J[P^\alpha + P^\beta]$, from `jk.spin_jk` |
| `k` | $K[P^\sigma]$ |
| `w` | $W^\sigma$ |
| `f_mo` | $F^\sigma$ in the semicanonical frame of $\sigma$ |
| `g` on entry | $g^\sigma_{\mathrm{MP2}} + 2F^\sigma$ from `add_hf_gradient` |

Compare with the closed-shell kernel in 12.1.5. The exchange weight is $1$
here against $\tfrac12$ there, because each spin's exchange acts only within
that spin. The column index `i` starts at 0, not at `first_occ`, because
the kernel's output has already been placed in MO columns.

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

The H2 row is the closed-shell limit of 12.2.2 run through the production
code: a singlet through the unrestricted driver must land on the
closed-shell minimum. The validation page lists the unrestricted path under
the `oo-rimp2` row as energy only.

**Known limits (2026-11-15).**

- There is no nuclear gradient and no Python entry point.
- The convergence recheck does not recompute the gradient (table above).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, sections 7 and 14 | equations 12.5 and 12.6, the frozen-core embedding |
| `crates/ferric-mp2/src/u_oo_rimp2.rs`, `crates/ferric-mp2/src/u_rimp2.rs` | the loop rules, the densities (12.7), the kernel |
| `scripts/u_oo_rimp2_response_sympy.py`; `scripts/mp2_gradient_sympy.py`, part U | the proofs cited in the box |
| U. Bozkaya, J. Chem. Phys. 139, 154105 (2013) | the unrestricted functional, as cited in the code |
| `site/src/reference/validation.md` | the anchor figures |

Changes from the sources:

- $\beta$ indices are written with overbars, not capitals.
- The $\mu$ doubling with its three-strike exit, and the unrestricted
  energy-change stop that skips the gradient recompute, are stated from the
  loop, as in `theory.md` section 7.
