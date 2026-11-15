# 26.1 Second order in a split interaction

<!-- section: 26.1; status: edit; registry: rs-mp2-rpa;
     theory: crates/ferric-rpa/theory.md#"Range-separated MP2 + RPA" -->

Section 10.7 showed that splitting the operator does not split the MP2
energy. This section states what that means for a method that uses MP2 at
short range and the dRPA at long range, and fixes the rule that the two
formulations of section 26.4 follow: correct full-Coulomb MP2 only by terms
that start at third order.

| Symbol | Meaning | Code name |
|---|---|---|
| $(ia\vert jb)_v$ | integral of operator $v \in \{\mathrm C, \mathrm{sr}, \mathrm{lr}\}$, from its own RI fit | `RpaIntermediates::b_ov` per operator |
| $E_{\mathrm{OS}}[v]$, $E_{\mathrm{SS}}[v]$ | MP2 spin components with $v$ | `SpinComponents::{e_os,e_ss}` |
| $E_{\mathrm{dRPA}}[v]$ | dRPA correlation energy with $v$ | `PdepRpaResult::e_rpa` |
| $X$ | the cross term (10.18) | not returned |

## 26.1.1 The split energy

With $(ia|jb)_v = g^v_{ijab}$ and its exchange partner $g^{v\times}_{ijab} = g^v_{ijba}$,

$$ E_{\mathrm{OS}}[v] = -\sum_{ijab}\frac{(g^v_{ijab})^2}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{SS}}[v] = -\sum_{ijab}\frac{g^v_{ijab}\big(g^v_{ijab} - g^{v\times}_{ijab}\big)}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{MP2}}[v] = E_{\mathrm{OS}}[v] + E_{\mathrm{SS}}[v] . \tag{26.1} $$

$E_{\mathrm{OS}} + E_{\mathrm{SS}}$ is the closed-shell energy (10.3) and the
spin-orbital $-\tfrac14\sum|\langle ij||ab\rangle|^2/\Delta$. For
$g^{\mathrm C} = g^{\mathrm{sr}} + g^{\mathrm{lr}}$, equation (10.18) gives

$$ E_{\mathrm{MP2}}[\mathrm C] = E_{\mathrm{MP2}}[\mathrm{sr}] + E_{\mathrm{MP2}}[\mathrm{lr}] + X . \tag{26.2} $$

$X$ is second order and mixes the two ranges: one interaction of the pair
is short-range, the other long-range.

## 26.1.2 Where the dRPA starts

The dRPA energy is a sum of ring diagrams, one per order in the interaction.
Scale every integral by $\lambda$ and expand:
$E_{\mathrm{dRPA}}[\lambda v] = \sum_{n \ge 2}\lambda^nE^{(n)}_{\mathrm{dRPA}}[v]$.
In the drCCD form (18.2), in canonical orbitals,

$$ \Delta\circ\mathbf T_1 = -\mathbf K, \qquad \Delta\circ\mathbf T_2 = -(\mathbf K\mathbf T_1 + \mathbf T_1\mathbf K), \qquad E^{(n+1)}_{\mathrm{dRPA}} = \tfrac12\sum_{ia,jb}K_{ia,jb}(\mathbf T_n)_{ia,jb}, \tag{26.3} $$

with $\Delta_{ia,jb} = \Delta_i^a + \Delta_j^b$. The second-order ring is the
direct part of MP2 and contains no exchange:

$$ E^{(2)}_{\mathrm{dRPA}}[v] = -\sum_{ijab}\frac{2\,(g^v_{ijab})^2}{\Delta_{ij}^{ab}} = 2E_{\mathrm{OS}}[v] . \tag{26.4} $$

> **Verified equation 26.4** · the second-order ring is twice the opposite-spin MP2 energy
>
> On a random many-mode model ($n_o = 3$, $n_v = 4$, RI factors of rank 6 and 5),
> the first drCCD iterate of (26.3) gives $E^{(2)} = 2E_{\mathrm{OS}}$ to
> $10^{-13}$ relative, with $E_{\mathrm{OS}}$ computed by the loop of
> `pair_energy`. The ring orders 2 and 3 agree with the Taylor coefficients
> of the plasmon energy (18.1), taken by a Cauchy integral in complex
> $\lambda$; $E_{\mathrm{OS}} + E_{\mathrm{SS}}$ agrees with the spin-orbital
> sum.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_rs_mp2_rpa_sympy.py`, checks `ring2-equals-2eos`, `ring-orders-vs-plasmon`, `mp2-vs-spin-orbital` |
> | Negative control | the runner weights the opposite-spin term 2 in the `pair_energy` transcription, and flips the exchange sign; each turns its check red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::spin_components_from_b_ov`, `ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa` |
> | Code test | `rs_mp2_rpa_h2o_cc_pvdz_vs_numpy` in `crates/ferric-rpa/tests/validation_rs_mp2_rpa.rs` (the frequency-integrated second-order ring term equals $2E_{\mathrm{OS}}$ to $3.2\times10^{-15}$ relative, `site/src/reference/validation.md`) |
>
> <!-- verified: id=rsrpa.ring2-2eos; script=scripts/rpa_rs_mp2_rpa_sympy.py; claim=ring2-equals-2eos; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::spin_components_from_b_ov, ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa; rust_test=ferric-rpa/tests/validation_rs_mp2_rpa.rs::rs_mp2_rpa_h2o_cc_pvdz_vs_numpy; registry=rs-mp2-rpa -->

In one mode, with coupling $k$ and excitation energy $\Delta$, (18.1) gives

$$ E_{\mathrm{dRPA}}(k) = \tfrac12\Big(\sqrt{\Delta(\Delta + 2k)} - \Delta - k\Big) = -\frac{k^2}{4\Delta} + \frac{k^3}{4\Delta^2} + O(k^4) . \tag{26.5} $$

> **Verified equation 26.5** · the one-mode ring series
>
> $\tfrac12\big(\sqrt{\Delta(\Delta + 2\lambda k)} - \Delta - \lambda k\big)$ has
> $\lambda^2$ coefficient $-k^2/(4\Delta)$ and $\lambda^3$ coefficient
> $k^3/(4\Delta^2)$ symbolically.
>
> | | |
> |---|---|
> | Proof | `scripts/rpa_rs_mp2_rpa_sympy.py`, check `one-mode-series` |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa` |
>
> <!-- verified: id=rsrpa.one-mode-series; script=scripts/rpa_rs_mp2_rpa_sympy.py; claim=one-mode-series; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-rpa/src/rs_mp2_rpa.rs::rs_mp2_lr_rpa; registry=rs-mp2-rpa -->

## 26.1.3 The rule

The bracket $\Delta E_{\mathrm{dRPA}}[v] = E_{\mathrm{dRPA}}[v] - 2E_{\mathrm{OS}}[v]$
therefore starts at third order. Any sum of such brackets added to
$E_{\mathrm{MP2}}[\mathrm C]$ leaves the second-order energy equal to full
MP2, exchange and $X$ included. Both formulations of section 26.4 are built
this way.

The construction that ignores the rule is the naive sum, called formulation
A in the code:

$$ E^{\mathrm A}_c = E_{\mathrm{MP2}}[\mathrm{sr}] + E_{\mathrm{dRPA}}[\mathrm{lr}] . \tag{26.6} $$

By (26.2) and (26.4), its second-order term is
$E_{\mathrm{MP2}}[\mathrm{sr}] + 2E_{\mathrm{OS}}[\mathrm{lr}] = E_{\mathrm{MP2}}[\mathrm C] - X - \big(E_{\mathrm{MP2}}[\mathrm{lr}] - 2E_{\mathrm{OS}}[\mathrm{lr}]\big)$.
It loses the cross term and the long-range second-order exchange.
`rs_mp2_lr_rpa` reports $E^{\mathrm A}_c$ as `e_corr_naive` with the
`delta-lr` formulation, as a diagnostic that makes the loss visible; it is
never the returned energy.

## Known limits (2026-11-15)

- No entry point returns $X$; section 10.7 has the only smoke check of the
  three-operator decomposition.
- In RI each operator has its own fit, so (26.2) holds for the fitted
  integrals only up to the differences of three fitting errors (section
  10.7.3). The rule of 26.1.3 does not depend on it: each bracket uses one
  operator's integrals throughout.

## Sources

| Source | Used for |
|---|---|
| book section 10.7 | (26.2) |
| `crates/ferric-rpa/theory.md`, "Range-separated MP2 + RPA" | (26.1), (26.4), the rule |
| `scripts/rpa_rs_mp2_rpa_sympy.py` | (26.3), (26.5), the verified boxes |
| `crates/ferric-rpa/src/rs_mp2_rpa.rs`, `RsMp2RpaResult` | formulation A |

Changes from the sources: `theory.md` and the script write the denominator
$D = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b = -\Delta$
with positive-signed energies over $D$; this section uses $\Delta$. The
script's `b2 = 2.0 * k`, with `k` the $(ia|jb)$ matrix, is this book's
$\mathbf K$.
