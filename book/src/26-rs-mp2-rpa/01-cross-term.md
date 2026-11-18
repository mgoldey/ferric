# Second order in a split interaction {#cross-term}

<!-- status: edit; registry: rs-mp2-rpa -->

Split the Coulomb interaction into a short-range and a long-range part and
compute MP2 with each. The two energies do not add up to MP2 with the full
interaction. MP2 is quadratic in the interaction, so a pair of electrons
that interacts once at short range and once at long range contributes a
term that belongs to neither part (section @sec:range-split-cross-term).
A method that uses MP2 at short range and the dRPA at long range has to
keep that term. This section fixes the rule that the two formulations of
section @sec:two-formulations follow: correct full-Coulomb MP2 only by
terms that start at third order.

| Symbol | Meaning | Code name |
|---|---|---|
| $(ia\vert jb)_v$ | integral of operator $v \in \{\mathrm C, \mathrm{sr}, \mathrm{lr}\}$, from its own RI fit | `RpaIntermediates::b_ov` per operator |
| $E_{\mathrm{OS}}[v]$, $E_{\mathrm{SS}}[v]$ | MP2 spin components with $v$ | `SpinComponents::{e_os,e_ss}` |
| $E_{\mathrm{dRPA}}[v]$ | dRPA correlation energy with $v$ | `PdepRpaResult::e_rpa` |
| $X$ | the cross term (@eq:mp2-rs-cross-term) | not returned |

## The split energy {#cross-term-split-energy}

With $(ia|jb)_v = g^v_{ijab}$ and its exchange partner $g^{v\times}_{ijab} = g^v_{ijba}$,

$$ E_{\mathrm{OS}}[v] = -\sum_{ijab}\frac{(g^v_{ijab})^2}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{SS}}[v] = -\sum_{ijab}\frac{g^v_{ijab}\big(g^v_{ijab} - g^{v\times}_{ijab}\big)}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{MP2}}[v] = E_{\mathrm{OS}}[v] + E_{\mathrm{SS}}[v] . \label{eq:cross-term-1} $$

$E_{\mathrm{OS}} + E_{\mathrm{SS}}$ is the closed-shell energy (@eq:closed-shell-ri-mp2-3) and the
spin-orbital $-\tfrac14\sum|\langle ij||ab\rangle|^2/\Delta$. For
$g^{\mathrm C} = g^{\mathrm{sr}} + g^{\mathrm{lr}}$, equation (@eq:mp2-rs-cross-term) gives

$$ E_{\mathrm{MP2}}[\mathrm C] = E_{\mathrm{MP2}}[\mathrm{sr}] + E_{\mathrm{MP2}}[\mathrm{lr}] + X . \label{eq:cross-term-2} $$

$X$ is second order and mixes the two ranges: one interaction of the pair
is short-range, the other long-range. The dissertation (section 7.2.3)
writes the energy under $H = H_0 + \lambda V_{\mathrm{sr}} + \mu V_{\mathrm{lr}}$
as a double perturbation series: $X$ is its $E^{(1,1)}$ term, and each
energy term of MP$n$ has $2^n$ range contributions.

## Where the dRPA starts {#cross-term-drpa-starts}

The dRPA energy is a sum of ring diagrams, one per order in the interaction.
Scale every integral by $\lambda$ and expand:
$E_{\mathrm{dRPA}}[\lambda v] = \sum_{n \ge 2}\lambda^nE^{(n)}_{\mathrm{dRPA}}[v]$.
In the drCCD form (@eq:drpa-three-ways-2), in canonical orbitals,

$$ \Delta\circ\mathbf T_1 = -\mathbf K, \qquad \Delta\circ\mathbf T_2 = -(\mathbf K\mathbf T_1 + \mathbf T_1\mathbf K), \qquad E^{(n+1)}_{\mathrm{dRPA}} = \tfrac12\sum_{ia,jb}K_{ia,jb}(\mathbf T_n)_{ia,jb}, \label{eq:cross-term-3} $$

with $\Delta_{ia,jb} = \Delta_i^a + \Delta_j^b$. The second-order ring is the
direct part of MP2 and contains no exchange:

$$ E^{(2)}_{\mathrm{dRPA}}[v] = -\sum_{ijab}\frac{2\,(g^v_{ijab})^2}{\Delta_{ij}^{ab}} = 2E_{\mathrm{OS}}[v] . \label{eq:rsrpa-ring2-2eos} $$

> **Verified equation @eq:rsrpa-ring2-2eos** · the second-order ring is twice the opposite-spin MP2 energy
>
> On a random many-mode model ($n_o = 3$, $n_v = 4$, RI factors of rank 6 and 5),
> the first drCCD iterate of (@eq:cross-term-3) gives $E^{(2)} = 2E_{\mathrm{OS}}$ to
> $10^{-13}$ relative, with $E_{\mathrm{OS}}$ computed by the loop of
> `pair_energy`. The ring orders 2 and 3 agree with the Taylor coefficients
> of the plasmon energy (@eq:rpa-drpa-three-routes), taken by a Cauchy integral in complex
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

In one mode, with coupling $k$ and excitation energy $\Delta$, (@eq:rpa-drpa-three-routes) gives

$$ E_{\mathrm{dRPA}}(k) = \tfrac12\Big(\sqrt{\Delta(\Delta + 2k)} - \Delta - k\Big) = -\frac{k^2}{4\Delta} + \frac{k^3}{4\Delta^2} + O(k^4) . \label{eq:rsrpa-one-mode-series} $$

> **Verified equation @eq:rsrpa-one-mode-series** · the one-mode ring series
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

## The rule {#cross-term-rule}

The bracket $\Delta E_{\mathrm{dRPA}}[v] = E_{\mathrm{dRPA}}[v] - 2E_{\mathrm{OS}}[v]$
therefore starts at third order. Any sum of such brackets added to
$E_{\mathrm{MP2}}[\mathrm C]$ leaves the second-order energy equal to full
MP2, exchange and $X$ included. Both formulations of section @sec:two-formulations are built
this way.

The construction that ignores the rule is the naive sum, called formulation
A in the code:

$$ E^{\mathrm A}_c = E_{\mathrm{MP2}}[\mathrm{sr}] + E_{\mathrm{dRPA}}[\mathrm{lr}] . \label{eq:cross-term-6} $$

By (@eq:cross-term-2) and (@eq:rsrpa-ring2-2eos), its second-order term is
$E_{\mathrm{MP2}}[\mathrm{sr}] + 2E_{\mathrm{OS}}[\mathrm{lr}] = E_{\mathrm{MP2}}[\mathrm C] - X - \big(E_{\mathrm{MP2}}[\mathrm{lr}] - 2E_{\mathrm{OS}}[\mathrm{lr}]\big)$.
It loses the cross term and the long-range second-order exchange.
`rs_mp2_lr_rpa` reports $E^{\mathrm A}_c$ as `e_corr_naive` with the
`delta-lr` formulation, as a diagnostic that makes the loss visible; it is
never the returned energy.

## Known limits (2026-11-17)

- No entry point returns $X$; section @sec:range-split-cross-term has the only smoke check of the
  three-operator decomposition.
- With erf each operator has its own fit, so (@eq:cross-term-2) holds for the fitted
  integrals only up to the differences of three fitting errors (section
  @sec:range-split-cross-term-code). With terf the long range is the
  Coulomb fit minus the terfc fit (section @sec:two-formulations-terf-signed),
  so it holds exactly for the fitted integrals. The rule of
  @sec:cross-term-rule does not depend on either: each bracket uses one
  operator's integrals throughout.

## Sources

| Source | Used for |
|---|---|
| book section @sec:range-split-cross-term | (@eq:cross-term-2) |
| `scripts/rpa_rs_mp2_rpa_sympy.py` | (@eq:cross-term-3), (@eq:rsrpa-one-mode-series), the verified boxes |
| `crates/ferric-rpa/src/rs_mp2_rpa.rs`, `RsMp2RpaResult` | formulation A |

Changes from the sources: the script writes the denominator
$D = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b = -\Delta$
with positive-signed energies over $D$; this section uses $\Delta$. The
script's `b2 = 2.0 * k`, with `k` the $(ia|jb)$ matrix, is this book's
$\mathbf K$.
