# The integral-direct path {#integral-direct}

<!-- status: ready; registry: local-mp2 -->

The method of section @sec:amplitude-lmp2 builds the global three-index tensor, of shape
$(N_{\mathrm{aux}}, n_on_v)$, before it truncates anything. That tensor is
$O(N^3)$ in memory, whitening it is $O(N^4)$, and the Gram product that
turns it into pair integrals, $J_{ij} = \tilde B_i^{\mathsf T}\tilde B_j$
for every pair, is $O(N^5)$ in time. Dropping amplitudes afterwards cannot reduce a cost that has already
been paid. Yet a charge distribution $\phi_i\phi_a$ from a localized pair
lives near $i$ and $a$, and only auxiliary functions nearby can describe
it. The integral-direct path uses this. It evaluates only the three-index
integrals that the retained pairs need, half-transforms them into one small
strip per occupied orbital, and fits each pair in a local auxiliary basis.
The solve and the energy are the unchanged ragged machinery of section
@sec:amplitude-lmp2.

The section derives the local fit (@sec:integral-direct-domain-local-fit), walks through the stages and
the truncation each locality map makes (@sec:integral-direct-algorithm-each-map-truncates), and then states the
exactness anchor, the cost of the stages and what is measured.

The fit identities are machine-checked below. The locality maps are
definitions, not equations, and the trivial-limit tests check them.

| Symbol | Meaning | Code name |
|---|---|---|
| $D_i$ | auxiliary fit domain of occupied $i$: aux shells within `aux_radius` of its Boys centroid | `aux_dom` |
| $V_i$ | virtual domain of $i$: virtuals within `virt_radius` of its centroid | `virt_dom` |
| $\tilde D_i,\ \tilde V_i$ | unions over $i$'s surviving partners | strip rows and columns |
| $D_{ij} = D_i\cup D_j$ | pair fit domain | `pair_dom` |
| $C_{ij} = V_i\cup V_j$ | candidate virtuals of a pair | `pair_cands` |
| $A_i$ | strip of occupied $i$: raw $(P\vert ia)$ for $P\in D_{ij}$, $a\in C_{ij}$, unwhitened | `Strip` |
| $V_{DD}$ | the metric restricted to $D_{ij}$ | `vdd` |
| $W_k = L^{-1}A_k$ | whitened strip, $V_{DD} = LL^{\mathsf T}$ | `ws` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/lmp2_direct.rs::amplitude_lmp2_direct`, `ferric-mp2/src/lmp2_direct.rs::assemble_ragged_direct_local` |
| CLI | `[local] scheme = "amplitude-threshold"`, `eps`, `integral_direct = true`, with `aux_radius`, `virt_radius`, `ao_tail`, `schwarz_skip`, `eri3_pass`, `batch_merge`, `gate_cal`, `virt_schwarz_kappa`, `pair_skip`, `fock_screen` (`examples/alkane8-rimp2-local-direct.toml`) |
| Python | `run_rimp2(..., local="amplitude-threshold", eps=..., integral_direct=True)` |

## The domain-local fit {#integral-direct-domain-local-fit}

Density fitting (section @sec:closed-shell-ri-mp2) expands $\phi_i\phi_a$ in the auxiliary
functions. The best coefficients in the Coulomb metric are
$c = V^{-1}A$, where $A_{P,a} = (P\vert ia)$ and $V_{PQ} = (P\vert Q)$.
For pair $(i,j)$, restrict that fit to the functions in $D_{ij}$, the
auxiliary shells near either orbital. Substitute the restricted
coefficients into the Coulomb product of the two fitted densities:

$$ J_{ia,jb} = \sum_{P,Q\in D_{ij}} (P\vert ia)\,[V_{DD}^{-1}]_{PQ}\,(Q\vert jb) = (W_i^{\mathsf T}W_j)_{ab}, \qquad V_{DD} = LL^{\mathsf T},\quad W_k = L^{-1}A_k . \label{eq:lmp2-domain-fit} $$

The right-hand form follows in one line. Factor $V_{DD} = LL^{\mathsf T}$,
so $V_{DD}^{-1} = L^{-\mathsf T}L^{-1}$, and group each $L^{-1}$ with its
strip: $A_i^{\mathsf T}L^{-\mathsf T}L^{-1}A_j = (L^{-1}A_i)^{\mathsf T}(L^{-1}A_j)$.
That is the same Cholesky whitening as section @sec:closed-shell-ri-mp2, applied to one domain.

Check the limit. When $D_{ij}$ holds every auxiliary function, $V_{DD} = V$
and (@eq:lmp2-domain-fit) is the global RI integral of equation (@eq:closed-shell-ri-mp2-4).

Why one domain for the whole pair, rather than one per orbital? Dunlap's
robust form, $A_i^{\mathsf T}c_j + c_i^{\mathsf T}A_j - c_i^{\mathsf T}V_{DD}c_j$
with $c = V_{DD}^{-1}A$, is the fit whose error is quadratic in the error of
the fitted densities. When one domain and one kernel serve both orbitals,
substitute $c$: each of the three terms equals $A_i^{\mathsf T}V_{DD}^{-1}A_j$,
so the sum is the Gram form (@eq:lmp2-domain-fit). The metric and the three-index
integrals use the same operator, which is why the two forms coincide and
why the fit error is quadratic, as in section @sec:closed-shell-ri-mp2.

> **Verified equation @eq:lmp2-domain-fit** · the whitened Gram product and the robust form
>
> With $V = LL^{\mathsf T}$ for a generic lower-triangular $3\times3$ factor,
> $W_i^{\mathsf T}W_j = A_i^{\mathsf T}V^{-1}A_j$ in symbols (claim L1), and
> the robust form $A_i^{\mathsf T}c_j + c_i^{\mathsf T}A_j - c_i^{\mathsf T}Vc_j$ with $c = V^{-1}A$ equals
> the Gram form (claim L2), so the Dunlap correction cancels identically in one
> shared domain.
>
> | | |
> |---|---|
> | Proof | `scripts/local_mp2_fit_pno_sympy.py`, claims L1 and L2 |
> | Negative control | the runner replaces $W_i^{\mathsf T}W_j$ by $W_i^{\mathsf T}W_i$ (claim L1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_direct.rs::{fit_domain_group,whiten,chol_lower}` |
> | Code test | `trivial_limit_matches_global_domain_fit_and_canonical` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.domain-fit; script=scripts/local_mp2_fit_pno_sympy.py; claim=L1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_direct.rs::{fit_domain_group,whiten,chol_lower}; rust_test=ferric-mp2/tests/lmp2_direct.rs::trivial_limit_matches_global_domain_fit_and_canonical -->

**What the restriction does to the values.** A fit in fewer functions
describes each density less completely. For a diagonal element this has a
sign. Write $\bar D$ for the auxiliary functions outside $D$, $S$ for the
Schur complement $V_{\bar D\bar D} - V_{\bar DD}V_{DD}^{-1}V_{D\bar D}$ and
$y = A_{\bar D} - V_{\bar DD}V_{DD}^{-1}A_D$. Block elimination of $V$ gives

$$ A^{\mathsf T}V^{-1}A - A_D^{\mathsf T}V_{DD}^{-1}A_D = y^{\mathsf T}S^{-1}y \ \ge\ 0 , \label{eq:lmp2-aux-restriction} $$

because $S$ is positive definite. So each fitted $(ia\vert ia)$ can only
fall when the domain shrinks. Off-diagonal elements have no such sign, and
neither has the energy error. The fit radius changes $J$ values, so the
one-sided argument of @sec:amplitude-lmp2-amplitude-equations-energy does not cover it.

> **Verified equation @eq:lmp2-aux-restriction** · restricting the fit domain lowers each fitted diagonal
>
> The block-elimination identity in symbols for a generic symmetric
> $3\times3$ metric with a two-function domain, and numerically on a $4+3$
> block with two densities, where the left side is also positive
> semidefinite (claim L6).
>
> | | |
> |---|---|
> | Proof | `scripts/local_mp2_fit_pno_sympy.py`, claim L6 |
> | Negative control | the runner drops $V_{\bar DD}V_{DD}^{-1}A_D$ from $y$ (claim L6 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_direct.rs::fit_domain_group` |
>
> <!-- verified: id=lmp2.aux-restriction; script=scripts/local_mp2_fit_pno_sympy.py; claim=L6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_direct.rs::fit_domain_group; registry=local-mp2 -->

**The code.** `ferric-mp2/src/lmp2_direct.rs::fit_domain_group` handles all
pairs that share one domain $D_{ij}$. It factors $V_{DD}$ once, whitens
every distinct occupied of the group with one triangular solve (their
strips side by side as the right-hand sides), and then
forms one product per pair, with no explicit inverse. The product is the
whole of the pair step. The excerpt is copied unchanged:

<!-- excerpt: ferric-mp2/src/lmp2_direct.rs::fit_domain_group -->
```rust
let g = if i == j {
    w_i.t().dot(&w_i)
} else {
    w_i.t().dot(&ws.select(j, cij))
};
```

| Code | Equation |
|---|---|
| `w_i` | $W_i$ restricted to the candidate columns $C_{ij}$ |
| `ws.select(j, cij)` | $W_j$ on the same columns |
| `g` | $J_{ia,jb}$ over $a,b\in C_{ij}$, passed to the $\varepsilon$ mask of section @sec:amplitude-lmp2 (`mask_pair`) |

Grouping by domain pays because pairs repeat domains, so one Cholesky
factor serves several pairs (`fit_pairs`). On C20–C48 there were 6–9 times
more surviving pairs than distinct pair domains, measured 2026-09-05 at the
earlier maps (virtual radius 12 Bohr, AO tail $10^{-3}$, pair gate on). At
the current defaults (pair gate off, so every pair is fitted) the ratio is
smaller and falls with length: 1891 pairs in 251 groups on C20 and 4753 in
1031 on C32 (6-31G, cc-pVDZ-RI, $\varepsilon = 10^{-4}$, measured
2026-11-24). On C32, 778 of those groups, 79% of the stage's thread time,
produce no pair block at all: every element of their pairs falls under the
mask. Over the whole chain the empty pairs dominate: 2628 of 4753 pairs on
C32 and 14196 of 18721 on C64 keep no element, and fitting them is most of
the ungated cost. C64 takes 215.6 s ungated and 37.6 s when the pair gate
at cal 5.6 skips 13860 of them, with a bit-identical energy (2026-11-24,
provisional; the table and the interaction-energy check are in section
@sec:amplitude-lmp2-algorithm, step 5). The gate stays off by default.

## The algorithm and what each map truncates {#integral-direct-algorithm-each-map-truncates}

| Stage | Work | Function |
|---|---|---|
| 1 | the pair gate of section @sec:amplitude-lmp2-algorithm (off by default), giving surviving pairs and per-occupied partner lists | `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep` |
| 2 | the maps: $D_i$, $V_i$ and each orbital's AO-shell support (`ao_tail`, on the per-shell maximum $\lvert C\rvert$; truncation is exactly zeroing small coefficients) | `ferric-mp2/src/lmp2_direct.rs::build_locality_maps` |
| 3 | occupieds grouped by nearest atom to their Boys centroid; per batch, evaluate only the $(\mu\nu\vert P)$ shell triples needed and half-transform into per-occupied strips $A_i$ over $\tilde D_i\times\tilde V_i$ | `ferric-mp2/src/lmp2_direct.rs::collect_strips`, `ferric-mp2/src/lmp2_direct.rs::eval_batch` |
| 4 | the metric, only for $(P\vert Q)$ shell pairs inside some $\tilde D_i^2$. This is exact for the domain-local fit, which never reads a metric element outside a pair domain | `ferric-mp2/src/lmp2_direct.rs::strips_and_metric` |
| 5 | per pair-domain group: Cholesky once, whiten each distinct occupied once, $J_{ij} = W_i^{\mathsf T}W_j$, the Eq-8 mask and the Fock-propagated screen, the `PairBlock` | `ferric-mp2/src/lmp2_direct.rs::fit_pairs` |
| 6 | PCG, tolerance policy and Hylleraas energy, unchanged | section @sec:amplitude-lmp2 |

**Strips.** A pair $(i,j)$ reads rows $D_i\cup D_j$ and columns
$V_i\cup V_j$ from the two strips $A_i$ and $A_j$. So a strip must cover
the union over all of an orbital's partners, $\tilde D_i\times\tilde V_i$,
and then one strip serves every pair that $i$ belongs to. The strips are
unwhitened: they hold the same raw values as the global tensor, only for
the rows and columns that survive. Whitening happens per group in stage 5,
because $L$ depends on the pair domain.

**Two kinds of map.** As in @sec:amplitude-lmp2-options-change-equation, a map either removes amplitudes and
leaves every kept $J$ value alone, or it changes the $J$ values.

- *Pattern maps* can only raise the energy relative to the same run without
  them, up to the solver error of @sec:amplitude-lmp2-accurate-solve, because $E_H$ is then minimized
  over fewer amplitudes on a swap-closed pattern.
  - The pair gate (`gate_cal`), as in section @sec:amplitude-lmp2-algorithm.
  - The virtual domains (`virt_radius`). The candidate set
    $C_{ij} = V_i\cup V_j$ restricts rows and columns of pair $(i,j)$ alike,
    so the pattern stays swap-closed, and selecting columns of $W_i$ and
    $W_j$ leaves each kept product unchanged.
  - The candidate screen (`virt_schwarz_kappa` $=\kappa$). With $\varepsilon > 0$
    it removes $a$ from $C_{ij}$ unless $q_{ia}q^{\max}_j \ge \kappa\varepsilon$
    or $q_{ja}q^{\max}_i \ge \kappa\varepsilon$, where
    $q_{ia} = \sqrt{A_{ia}^{\mathsf T}V_{D_iD_i}^{-1}A_{ia}}$ is the fitted
    diagonal in $i$'s own domain. On a global Gram fit it is the exact
    Cauchy–Schwarz screen of @sec:amplitude-lmp2-algorithm for $\kappa \le 1$. On the domain-local fit
    it is not a proven bound. It was measured escape-free at
    $\kappa \le 1$, both operators, on C8 only (provisional). A
    $\kappa$ above 1 drops elements that would pass the Eq-8 test. It
    bounds the bare integrals only, so it is refused with the Fock screen on
    (section @sec:amplitude-lmp2-fock-screen-candidates); set
    `fock_screen = false` to use it.
- *Value maps* change the integrals, and their energy error has no proven
  sign.
  - The auxiliary domains (`aux_radius`), @sec:integral-direct-domain-local-fit.
  - The AO support (`ao_tail`). An AO shell enters an orbital's support
    when its largest $\lvert C\rvert$ is at least the threshold. Leaving a
    shell out is the same as setting those coefficients to zero in the half
    transform, applied alike to both strips of a pair. The Fock blocks are
    untouched.
  - The triple cut (`schwarz_skip`). A batch triple is skipped when
    $\sqrt{\max(P\vert P)}\,Q(\mu\nu)\,w(\mu\nu)$ falls below the
    threshold, with $Q(\mu\nu) = \sqrt{(\mu\nu\vert\mu\nu)}$, both
    factors from the run's operator, and
    $w(\mu\nu) = \max\bigl(1,\ c^{o}_{\mu} c^{v}_{\nu} + c^{o}_{\nu} c^{v}_{\mu}\bigr)$,
    where $c^{o}$ and $c^{v}$ are the largest $\lvert C\rvert$ on the shell
    over the active occupieds and over the virtuals. A stored triple enters
    $(P\vert ia)$ as $(C_{\mu i}C_{\nu a} + C_{\nu i}C_{\mu a})(P\vert\mu\nu)$,
    so the cut bounds each skipped triple's contribution to every
    transformed entry, not only the raw integral. The aux factor is the
    maximum over the shell, so the cut stays conservative at shell
    granularity. The floor at 1 keeps the cut no looser than a cut on the
    raw integral.

    A cut on the raw integral alone ($w = 1$) is not a bound on what the
    integral feeds. In a near-linearly-dependent basis the localized
    virtuals carry large, cancelling coefficients: in n-C$_{12}$H$_{26}$ /
    aug-cc-pVDZ the largest $\lvert C_v\rvert$ is 119, and that cut moved
    $E_\text{corr}$ by $-91$ mHa, to 79 mHa below canonical RI-MP2. With
    the weight it moves $E_\text{corr}$ by about $10^{-6}$ Ha (measured
    2026-11-25; `schwarz_cut_bounds_the_transformed_entries`,
    `production_schwarz_cut_is_harmless_in_a_diffuse_basis`).

The pair candidates of stage 5 come from `ferric-mp2/src/lmp2_direct.rs::pair_candidates`.
The body of its per-pair closure, unchanged except for the
comments:

```rust
let cij = sorted_union([virt_dom[i].iter().copied(), virt_dom[j].iter().copied()]); // C_ij = V_i ∪ V_j
match (qvirt, kappa_eps) {
    (Some((q, qmax)), Some(th)) => cij                         // κ screen on, th = κ ε
        .into_iter()
        .filter(|&a| {
            let qia = q[i][strips[i].col_pos[a]];               // q_ia
            let qja = q[j][strips[j].col_pos[a]];               // q_ja
            qia * qmax[j] >= th || qja * qmax[i] >= th          // keep a if either bound reaches κ ε
        })
        .collect(),
    _ => cij,                                                   // screen off or ε = 0
}
```

The screen only ever shrinks $V_i\cup V_j$. So every candidate still lies
in both strips' columns, and a missing column position is a construction
error that panics.

**Variants of stage 3.** Two passes build the same strips to round-off.
The production pass (`Eri3Pass::AuxMajor`, CLI `[local] eri3_pass =
"aux-major"`) takes one task per auxiliary shell over the union of the
supports that need it, so each needed triple is evaluated once. The other
(`Eri3Pass::Batches`, `eri3_pass = "batches"`) evaluates per occupied batch,
so a triple that neighbouring batches also need is evaluated once per batch.
`batch_merge` belongs to that second pass: it merges several nearest-atom
batches along the molecule's long axis, which evaluates fewer triples at
the cost of larger scratch slabs, and agrees with $k=1$ to the
float-reassociation floor, not bitwise, because the per-occupied transform
runs over a larger batch union. The aux-major pass has no batches, so it runs
at `batch_merge = 1` and a larger explicit value is an error that says to set
`eri3_pass = "batches"`; choosing `batches` without `batch_merge` uses 4.

Measured against the batches pass at merge 4 (release, 6 cores, 6-31G /
cc-pVDZ-RI, ε = 1e-4, production maps, 2 repeats; energies identical to 12
digits, peak resident set unchanged):

| alkane | stage 3, batches → aux-major | LMP2 wall | shell triples |
|---|---|---|---|
| C12 | 1.35 → 0.27 s | 3.20 → 2.14 s | 11.2 M → 1.2 M |
| C20 | 5.17 → 1.19 s | 13.5 → 9.4 s | 41.8 M → 3.4 M |
| C32 | 15.6 → 5.6 s | 50.2 → 39.0 s | 122 M → 8.9 M |

Under the MPI rank stripe (2 ranks, C20) aux-major also wins, 11.5-12.9 s
against 16.9-18.6 s.

## How the two kinds of map combine at ε = 0 {#integral-direct-map-errors-eps0}

At $\varepsilon = 0$ the Eq-8 mask keeps every amplitude. Any difference
from canonical RI-MP2 then comes from the maps. A pattern map and a value map
leave different fingerprints on the energy, and with both on, the two errors
do not add.

Write the Hylleraas functional as $E(t) = 2\langle t, J\rangle + \langle t, \mathcal A t\rangle$,
with $\mathcal A$ the positive definite Fock superoperator of @sec:amplitude-lmp2
and $J$ the fitted pair integrals. The exchange weight $2-\mathcal S$ of
(@eq:lmp2-hylleraas) is absorbed into the inner product: it commutes with
$\mathcal A$ and with a swap-closed projection, so it changes no step
below. Its minimum is $E^\star = -\langle J, \mathcal A^{-1}J\rangle$
at $t^\star = -\mathcal A^{-1}J$.

**A pattern map.** Split the amplitudes into a kept block $K$ and a dropped
block $D$, solve on $K$ only, $t_K = -\mathcal A_{KK}^{-1}J_K$, and eliminate
the $D$ block of $\mathcal A$. With
$S = \mathcal A_{DD} - \mathcal A_{DK}\mathcal A_{KK}^{-1}\mathcal A_{KD}$,

$$ E_K - E^\star = t^{\star\mathsf T}_D\, S\, t^\star_D \ \ge\ 0 . \label{eq:lmp2-map-pattern} $$

$S$ is positive definite because $\mathcal A$ is, so the error of
`virt_radius` is never negative. It is second order in the dropped exact
amplitudes. Check the limit: with nothing dropped, $D$ is empty and the error
is zero.

**A value map.** `aux_radius`, `ao_tail` and `schwarz_skip` replace $J$ by
$J + \delta J$ and leave the pattern and $\mathcal A$ alone. `ao_tail` zeroes
coefficients only inside the strip transform; the Fock blocks use the full
orbitals. Expand $E^\star(J+\delta J) = -\langle J+\delta J, \mathcal A^{-1}(J+\delta J)\rangle$
and use $t^\star = -\mathcal A^{-1}J$:

$$ E^\star(J + \delta J) - E^\star(J) = 2\langle t^\star, \delta J\rangle - \langle \delta J, \mathcal A^{-1}\delta J\rangle . \label{eq:lmp2-map-value} $$

The leading term is linear in $\delta J$ and has the sign of
$\langle t^\star, \delta J\rangle$, which no map controls. A negative
$\varepsilon = 0$ error therefore comes from a value map. Truncating an
orbital's AO expansion also leaves the occupied and virtual spaces of the
strips no longer orthogonal, so a fitted $(ia)$ density can carry a small
net charge (the mechanism of section @sec:local-laplace-sos-truncations).
Three things bound it here. The cut is on coefficient magnitude, not
distance, so only small coefficients go. The damage is confined to $J$:
the Fock blocks and the mask use the full orbitals. And `probe_tail_mass`
reports the coefficient mass each orbital loses. Its measured size is the
`ao_tail` row of the table below.

**Both on.** Solve the restricted problem with the perturbed integrals and
subtract $E^\star$. Adding and subtracting $E_K(J)$ splits the result into
(@eq:lmp2-map-pattern) and (@eq:lmp2-map-value) restricted to $K$:

$$ E_K(J + \delta J) - E^\star = t^{\star\mathsf T}_D S\, t^\star_D + 2\langle t_K, \delta J_K\rangle - \langle \delta J_K, \mathcal A_{KK}^{-1}\delta J_K\rangle . \label{eq:lmp2-map-both} $$

The value map is charged only on the kept block, and against $t_K$, not
$t^\star_K$. So the error with both maps on is not the sum of the two errors
measured one map at a time. A small combined error can be a positive pattern
error cancelling a value error of the other sign. Such a cancellation depends
on the geometry, so it does not carry over from one conformer to the next.
To attribute an error, switch one map on at a time from the trivial limit,
and also switch one map off at a time from the production setting. The
ignored probe `probe_map_ablation` in `crates/ferric-mp2/tests/lmp2_map_ablation.rs`
does both, and `probe_tail_mass` reports the coefficient mass `ao_tail` drops
per orbital.

> **Verified equations @eq:lmp2-map-pattern, @eq:lmp2-map-value and @eq:lmp2-map-both** · how pattern and value maps enter the energy
>
> On a generic symmetric $2+1$ block, $E_K - E^\star$ equals the Schur form
> (claim 1). On a generic $3\times3$ model the value-map expansion is exact
> through both orders (claim 2). On the $2+1$ block, both maps together give
> the kept-block form (claim 3). A random rational positive definite model
> checks all three by brute-force solves. It finds value-map errors of both
> signs and a combined error that differs from the one-at-a-time sum.
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_direct_maps_sympy.py`, claims 1, 2 and 3 |
> | Negative control | the runner flips the sign of the Schur coupling term (claim 1 fails), drops the quadratic term (claim 2 fails), and charges the value map on $t^\star_K$ (claim 3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_direct.rs::{build_locality_maps,pair_candidates}` |
> | Code test | `trivial_maps_match_canonical_aconf20_sto3g` in `crates/ferric-mp2/tests/lmp2_map_ablation.rs` |
>
> <!-- verified: id=lmp2.direct-maps; script=scripts/lmp2_direct_maps_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_direct.rs::{build_locality_maps,pair_candidates}; rust_test=ferric-mp2/tests/lmp2_map_ablation.rs::trivial_maps_match_canonical_aconf20_sto3g -->

The proof covers the algebra of a fixed $\mathcal A$ and $J$. It does not
bound the size of $\delta J$ for any map. The sizes come from measurement.
Measured 2026-11-17 (provisional) with `probe_map_ablation`, 6-31G with
cc-pVDZ-RI at $\varepsilon = 0$, one map on at a time from the trivial
limit, error against canonical RI-MP2 in Ha:

| Map | n-C$_{20}$H$_{42}$, extended | A202, a folded C$_{20}$H$_{42}$ conformer | adamantane |
|---|---|---|---|
| `ao_tail` $10^{-3}$ | $-1.4\times10^{-4}$ | $+9.4\times10^{-5}$ | $-4.55\times10^{-5}$ |
| `virt_radius` 12 Bohr | $+9.7\times10^{-5}$ | $+2.5\times10^{-5}$ | |
| `aux_radius` 10 Bohr | about $10^{-6}$ | | |
| `schwarz_skip` $10^{-5}$ (unweighted cut, before 2026-11-25) | about $10^{-7}$ | | |

`virt_radius`, a pattern map, is positive in every column, as
(@eq:lmp2-map-pattern) requires. `ao_tail`, a value map, changes sign with
the geometry. The two cancel on the extended chain and add on the folded
conformer, which put up to 0.38 kJ/mol into the conformer energies of 13
ACONF20 conformers. At `virt_radius` 20 Bohr and `ao_tail` $10^{-4}$ that
error is 0.0031 kJ/mol, which is why those are the defaults.

## The exactness anchor and the artifact hypothesis {#integral-direct-exactness-anchor-artifact-hypothesis}

The repository's experimental protocol requires both of these to be stated
before any sweep is run, and the module documentation of `lmp2_direct.rs`
records them that way.

**Trivial limit.** Set every map to a no-op: `virt_radius = None`,
`ao_tail = 0`, `aux_radius` at least $10^5$ Bohr, no gate,
`schwarz_skip = 0`, and $\varepsilon = 0$. Then the energy must reproduce
the global-tensor domain-fit path of the same radius, and canonical RI-MP2,
to the CG tolerance. The test has one mutation arm per map, so each map is
shown to change the energy when it is broken.

n-Butane in cc-pVDZ shows the trivial limit and then the default maps
(re-run 2026-11-18 at the current defaults: λ-Boys λ = 10, Fock screen κ = 1 on both paths). Python has no way to pass `virt_radius = None`, so
the first run keeps the 20 Bohr default; butane is 10.5 Bohr between its
farthest atoms, so that radius drops nothing. The trivial run matches exact
RI-MP2 to $2.71\times10^{-13}$ Ha. With the default maps (`virt_radius` 20,
`ao_tail` $10^{-4}$) the direct recovery agrees with the in-core one to
$7\times10^{-4}$ percentage points at each $\varepsilon$:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/alkane_4.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

exact = ferric.run_rimp2(mol, bs, aux, frozen_core=0).mp2_corr
trivial = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=0.0,
                           integral_direct=True, aux_radius=1e5, ao_tail=0.0, schwarz_skip=0.0,
                           batch_merge=1).mp2_corr
print(f"exact {exact:.12f}  direct trivial maps {trivial:.12f}  diff {trivial - exact:+.2e} Ha")
for eps in (1e-5, 1e-4, 1e-3):
    incore = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=eps).mp2_corr
    direct = ferric.run_rimp2(mol, bs, aux, frozen_core=0, local="amplitude-threshold", eps=eps,
                              integral_direct=True).mp2_corr
    print(f"eps={eps:7.0e}  in-core {100 * incore / exact:8.4f}%  direct (default maps) {100 * direct / exact:8.4f}%")
```

```text
exact -0.599645225098  direct trivial maps -0.599645225098  diff +2.71e-13 Ha
eps=  1e-05  in-core  99.9990%  direct (default maps)  99.9991%
eps=  1e-04  in-core  99.8929%  direct (default maps)  99.8931%
eps=  1e-03  in-core  96.5805%  direct (default maps)  96.5812%
```

**Artifact hypothesis.** The two outcomes of a size sweep are predicted in
advance:

- If the locality is real, then at fixed radii and thresholds the extra
  error against the $\varepsilon$-only path stays flat per CH$_2$ as an
  alkane grows (it is size-extensive), and the strip sizes saturate past
  the onset.
- If a map is mis-constructed by mixing two orbital index sets, exactness
  will instead demand radii that track the molecular diameter. A domain
  mask that mixed two index sets did this in an AO-sparse Laplace
  implementation (section @sec:local-laplace-sos).

The two predictions differ, so the sweep can tell them apart. Measured
2026-09-05 at the earlier maps (virtual radius 12 Bohr, AO tail
$10^{-3}$; provisional): the strip maxima are flat from C16 to C32 (1092
rows and 127 columns with erfc), the pair domains are flat, and the strip
means still grow by about 6% per 12 carbons. That is the first prediction,
with the union tail not yet past the onset at C32. At the current maps the
extra error is $+0.9$ to $+3.2\times10^{-6}$ Ha from C12 to C20 (section
@sec:amplitude-lmp2-fock-screen-default): it grows with the chain, as a
size-extensive error must, and three sizes are too few to test whether the
amount per CH$_2$ is constant.

## Proving a pair empty {#integral-direct-pair-skip}

With an attenuated operator most distant pairs end empty: every element of
the fitted $J_{ij}$ falls below $\varepsilon$, the mask keeps nothing, and the
pair adds nothing to the energy. Stage 5 still fits them. A pair costs its
share of the group's Cholesky and whitening and one Gram product, and the
product is thrown away. The skip removes those pairs before the fit, by
proof: it bounds every element the retention test reads from data stage 5
already holds, and drops the pair only when the bound is below the
threshold. A skipped pair is one whose blocks the mask would have removed,
so the energy is bitwise the energy without the skip.

**What has to be bounded.** The retention test of a pair reads
$g = s\,J_{ia,jb}$ for $a,b\in C_{ij}$, with $s$ the spin-scale of the mask,
and, with the Fock screen on, the dressed gate $\tilde g$ of section
@sec:amplitude-lmp2-fock-screen, compared with $\kappa\varepsilon$. The $J$
it reads is the computed one. Write
$a = A_i[D, a]$ and $b = A_j[D, b]$ for the strip columns on
$D = D_{ij}$, as stage 3 produced them, and $V = V_{DD}$ for the metric
block. Then $J_{ab} = a^{\mathsf T}V^{-1}b$. The strips already carry the
AO tail and the Schwarz skip, so a bound on this product covers both maps
with no tail term of its own: nothing below treats a strip column as the
integral of a function.

**Splitting the domain.** Split $D$ into $A = D_i$ and $B = D_j\setminus
D_i$. Scale each block by its own metric, $\alpha = \mathrm{diag}(V_{AA},
V_{BB})^{-1/2}a$ and likewise $\beta$, so that

$$ V = \mathrm{diag}(V_{AA},V_{BB})^{1/2}\,(I + X)\,\mathrm{diag}(V_{AA},V_{BB})^{1/2},\qquad X = \begin{pmatrix}0 & Y\\ Y^{\mathsf T} & 0\end{pmatrix},\qquad Y = V_{AA}^{-1/2}V_{AB}V_{BB}^{-1/2}. $$

Then $J = \alpha^{\mathsf T}(I+X)^{-1}\beta$. Expand the inverse once,
$(I+X)^{-1} = I - X + X^2(I+X)^{-1}$, which is an identity for any $X$
with $I+X$ invertible:

$$ J = \alpha_A\!\cdot\beta_A + \alpha_B\!\cdot\beta_B - \bigl(\alpha_A^{\mathsf T}Y\beta_B + \alpha_B^{\mathsf T}Y^{\mathsf T}\beta_A\bigr) + \alpha^{\mathsf T}X^2(I+X)^{-1}\beta . $$

The eigenvalues of $X$ are $\pm$ the singular values of $Y$, so with
$\gamma = \lVert Y\rVert_2 < 1$ the last term is at most
$\gamma^2/(1-\gamma)\,\lVert\alpha\rVert\lVert\beta\rVert$. Take
$u_A\ge\lVert\alpha_A\rVert$, $u_B\ge\lVert\alpha_B\rVert$ and likewise
$v_A$, $v_B$, all maximised over the columns. Every element obeys

$$ \lvert J_{ab}\rvert \le u_Av_A + u_Bv_B + \gamma\,(u_Av_B + u_Bv_A) + \frac{\gamma^2}{1-\gamma}\sqrt{u_A^2+u_B^2}\,\sqrt{v_A^2+v_B^2}. \label{eq:lmp2-pair-skip-bound} $$

Check the limit $V_{AB} = 0$: $\gamma = 0$, the fit separates, and the
bound is $u_Av_A + u_Bv_B$, the Cauchy–Schwarz bound on each block.

**Where each factor comes from.** None of them needs the pair's own
Cholesky.

- $u_A$ is $\max_a\lVert V_{D_iD_i}^{-1/2}a_{D_i}\rVert$, occupied $i$'s
  fitted norm on its own domain, which the κ screen already computes.
- $v_B$ is $j$'s own-domain norm. It bounds $\lVert V_{BB}^{-1/2}b_B\rVert$
  because $B\subseteq D_j$ and a fit on fewer functions has the smaller
  diagonal (@eq:lmp2-aux-restriction).
- $u_B$ bounds $\lVert V_{BB}^{-1/2}a_B\rVert$. Pad $a_B$ with zeros to
  all of $D_j$; by the same monotonicity its fitted norm on $D_j$ is at
  least as large, $\lVert L_j^{-1}(a_B, 0)\rVert$, with $L_j$ the Cholesky
  factor of $V_{D_jD_j}$. Two bounds on that are cheap: $\lVert
  a_B\rVert/\sqrt{\lambda_j}$, and the shell-by-shell triangle inequality
  $\sum_{P\in B}\lVert L_j^{-1}[:,P]\rVert_F\,\lVert a_P\rVert$. The code
  takes the smaller. $v_A$ is the same with $L_i$ and $j$'s strip rows on
  $D_i$.
- $\gamma \le \lVert V_{AB}\rVert_F/\sqrt{\lambda_i\lambda_j}$, or with one
  side whitened shell by shell, $\gamma\le\sum_{P\in A}\lVert
  L_i^{-1}[:,P]\rVert_F\lVert V_{PB}\rVert_F/\sqrt{\lambda_j}$ and its
  mirror. The code takes the smallest of the three.
- $\lambda_k$ floors $\lambda_{\min}(V_{D_kD_k})$, and by Cauchy interlacing
  every principal sub-block, $A = D_i$ and $B\subseteq D_j$ among them. Two
  floors are proved and the larger is used. The trace floor
  $\tfrac12/\lVert L_k^{-1}\rVert_F^2$ follows from
  $\operatorname{tr}V^{-1} = \lVert L^{-1}\rVert_F^2 \ge 1/\lambda_{\min}$;
  the $\tfrac12$ covers the rounding of the computed $L^{-1}$, and the
  floor is refused (nothing is proved for that occupied) when $16\,d\,u\,\kappa
  > \tfrac14$, with $u$ the unit round-off and $\kappa\le\lVert
  V\rVert_F\lVert L^{-1}\rVert_F^2$. The trace counts every small
  eigenvalue, so it can sit a factor $d$ below $\lambda_{\min}$. The
  second floor is tighter: power iteration on $V^{-1}$ estimates
  $\lambda_{\min}$ from above, and $\sigma = \lambda_{\mathrm{est}}/2$
  (then $/10$) is accepted only if the Cholesky of $V - (\sigma + c)I$
  succeeds, with $c = 2\,(\gamma_{d+1}/(1-\gamma_{d+1})\operatorname{tr}V
  + 4u\max_k v_{kk})$. By Rump's theorem (BIT 46 (2006) 433) a completed
  floating-point Cholesky of $M - cI$ proves $M$ positive definite, so
  $\lambda_{\min}(V) > \sigma$.

The norms $\lVert a_B\rVert$, $\lVert b_A\rVert$ and $\lVert
V_{AB}\rVert_F$ come from per-shell tables: the squared strip norm of each
occupied on each auxiliary shell, and the squared Frobenius norm of each
metric shell block. A shell block that stage 4 never computed has no table
entry, and a pair that would need one is not bounded.

**The computed $J$.** Stage 5 forms $J$ in floating point. Its Cholesky
has backward error of order $d\,u\,\lVert V\rVert$, so the computed product
differs from the exact one by at most
$16\,d\,u\,\kappa_D\,\lVert u\rVert\lVert v\rVert/(1-\gamma)$, with $\kappa_D
\le \lVert V_{DD}\rVert_F/((1-\gamma)\lambda_{\min})$. The code adds this
allowance, and a relative slack of $10^{-9}$, to (@eq:lmp2-pair-skip-bound)
before it compares. The own-domain norms $u_A$ and $v_B$ are themselves
computed through a Cholesky factor, so each is multiplied by
$1 + 16\,d_k\,u\,\kappa_k$ before it enters the bound
(`ferric-mp2/src/lmp2_pair_skip.rs::norm_inflation`); each shell block norm
of $L_k^{-1}$ gets the same factor plus the absolute allowance
$16\,d_k\,u\,\kappa_k\lVert L_k^{-1}\rVert_F$. The exact quantity
bounded is the one stage 5 would form from the stored strips and metric
blocks; those are the same arrays the fit reads.

**The Fock-dressed gate.** If every element obeys $\lvert g_{cb}\rvert\le
G$, the dressed gate $\tilde g = g + F't_0 + t_0F'$ with $t_0 = -g/\Delta$
obeys $\lvert\tilde g_{ab}\rvert \le G(1 + w_a + w_b) \le G(1 + 2w_{\max})$,
where $w_a = \sum_{c\neq a}\max(\lvert F_{ac}\rvert,\lvert
F_{ca}\rvert)/D_c$ over the full virtual space and $D_c = f_{cc} + \min
f_{vv} - 2\max f_{oo}$. These are the weights of
`ferric-mp2/src/ragged.rs::fock_bound_weights` at a unit Schwarz factor,
and the test is `ferric-mp2/src/ragged.rs::fock_bound_keeps` on a constant
factor $\sqrt G$, the same function the κ screen uses. A pair is skipped
when $G\le\varepsilon$ and $G(1+2w_{\max})\le\kappa\varepsilon$.

**The code.** `ferric-mp2/src/lmp2_pair_skip.rs::fitted_bound` evaluates
(@eq:lmp2-pair-skip-bound). The excerpt is copied with the field
destructuring removed:

<!-- excerpt: ferric-mp2/src/lmp2_pair_skip.rs::fitted_bound -->
```rust
let finite = [u_a, u_b, v_a, v_b, gamma].iter().all(|x| x.is_finite());
if !finite || !(gamma < 1.0) {
    return None; // no convergent remainder: nothing is proved
}
// gamma^2/(1-gamma) ||u|| ||v||
let rem = gamma * gamma / (1.0 - gamma) * u_a.hypot(u_b) * v_a.hypot(v_b);
// uA vA + uB vB + gamma (uA vB + uB vA) + remainder
Some(u_a * v_a + u_b * v_b + gamma * (u_a * v_b + u_b * v_a) + rem)
```

- `ferric-mp2/src/lmp2_direct.rs::prove_empty_pairs` builds the tables and
  the floors once per occupied and evaluates the bound for every pair
  before stage 5. Per occupied it costs one $L^{-1}$, one shifted
  Cholesky, forty power-iteration steps and one whitening of the whole
  strip; per pair, sums over the shells of the two domains.
- A skipped pair keeps its place in its group and pushes an empty block.
  The group's union of candidates and its whitening are unchanged, so the
  pairs that remain get bit-identical $J$. A group whose pairs are all
  skipped is not fitted at all.
- `pair_skip = "audit"` evaluates the bound on every pair, skips nothing,
  and records for each fitted pair the ratio of the largest computed
  element to the bound.

**Why only attenuated operators.** The bound holds for any
positive-definite kernel. For erfc the Fourier transform
$4\pi(1-e^{-k^2/4\omega^2})/k^2$ is positive, so the metric is positive
definite and the fit is a projection (claim 3 of the proof); terfc keeps
that property, terf does not and is refused. The bound proves a pair only
when $\gamma < 1$, which needs the metric coupling between the two
domains to be small next to the domains' smallest eigenvalues. With
Coulomb, $V_{AB}$ decays as a power of the distance and $\gamma$ stays
above 1 at chemical sizes. `pair_skip = "auto"`, the default, therefore
turns the skip on for erfc and terfc and off for Coulomb.

> **Verified equation @eq:lmp2-pair-skip-bound** · the provable-empty pair bound
>
> The block-Neumann split in symbols on a $1+1$ block (claim 1); Schur
> monotonicity, which makes $v_B$ a bound (claim 2); the erfc Fourier
> transform is positive (claim 3); the bound against a direct dense solve
> on random and adversarial SPD instances, where the adversarial one needs
> the $\gamma^2$ remainder (claim 4); the eigenvalue floor (claim 5); and
> the dressed-gate bound on random Fock matrices (claim 6); the
> shift-verified eigenvalue floor (claim 7); and the shell-wise
> alternatives for $u_B$, $v_A$ and $\gamma$ (claim 8).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_attenuated_pair_skip_sympy.py`, claims 1 to 8 |
> | Negative control | the runner drops the $\gamma^2$ remainder (claim 4 fails), and six more mutations |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_pair_skip.rs::{fitted_bound,eigen_floor,verified_floor,proved_empty}` |
> | Code test | `pair_skip_is_bitwise_and_its_audit_holds_under_erfc` in `crates/ferric-mp2/src/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.pair-skip; script=scripts/lmp2_attenuated_pair_skip_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_pair_skip.rs::{fitted_bound,eigen_floor,verified_floor,proved_empty}; registry=local-mp2 -->

**What is measured.** On n-alkanes with 6-31G / cc-pVDZ-RI, $\varepsilon =
10^{-4}$, the production maps, the Fock screen on and erfc at $\omega =
0.2223$ Bohr$^{-1}$ (0.42 Å$^{-1}$, the CLI default), measured 2026-11-25 on
one node with 6 threads (provisional; raw data and drivers in the wiki note
`lmp2-attenuated-pair-skip-2026-11-24`):

| alkane | pairs | empty | proved empty | LMP2 wall, skip off → on | stage 5, off → on |
|---|---|---|---|---|---|
| C$_{20}$ | 1891 | 971 | 13 | 6.65 → 7.10 s | 3.83 → 4.33 s |
| C$_{32}$ | 4753 | 3197 | 835 | 28.4 → 26.4 s | 19.6 → 17.5 s |
| C$_{48}$ | 10585 | 8181 | 3931 | 92.6 → 66.8 s | 61.8 → 36.0 s |

Each entry is the mean of two runs; the correlation energy has the same bits
with the skip on and off in every run. At $\omega = 1$ Bohr$^{-1}$ (one run
each) the skip proves 1025 of 3514 empty pairs at C$_{32}$ and 4329 of 8690
at C$_{48}$, where the LMP2 wall drops from 92.3 to 59.9 s. The bound itself
costs 0.5 s at C$_{20}$ and 1.5 to 1.8 s at C$_{48}$; at C$_{20}$, where it
proves almost nothing, that is a 7% loss.

Pairs end empty from about 10 Bohr between the Boys centroids, and every
pair past 20 Bohr is empty. The proof starts near 42 Bohr (37 Bohr at
$\omega = 1$). Between 30 and 45 Bohr the bound is finite but held up by
the cross terms $u_Av_A$ and $u_Bv_B$: a strip's rows on the other
occupied's domain, against an own-domain norm of about 0.4. Past 55 Bohr
the floating-point allowance dominates it.

The audit mode measured the bound against the computed elements on alkanes
C$_8$ to C$_{32}$, the water, methane and ethylene dimers, naphthalene, an
all-trans C$_{32}$H$_{34}$ polyene, and the water dimer and octane in
aug-cc-pVDZ / aug-cc-pVDZ-RI, at both $\omega$. Over those 22 runs the
largest ratio of a computed element to its bound is $4.75\times10^{-3}$, and
no pair proved empty kept a block. Among pairs whose bound is within a
factor 10 of $\varepsilon$, the largest ratio is $1.65\times10^{-4}$. The
aug-cc-pVDZ runs are too small for the skip to prove anything, so its reach
with diffuse functions is not measured.

## Memory and determinism {#integral-direct-memory-determinism}

- The peak three-index working set is the strips. It is reported as
  `DirectStats::strip_bytes_total`, with mean and maximum strip rows and
  columns (`strip_rows_mean`, `strip_cols_max`). The strip sizes are the
  locality claim, so the code reports them rather than leaving them to be
  inferred.
- The stage-3 integral slabs of the batch pass have one cap per node
  (`DirectConfig::slab_budget_bytes`, 64 MiB by default), shared by the
  batch workers. Each worker slabs under the cap divided by the number of
  workers, so the co-resident slabs never exceed it; a per-worker budget
  would not bound the total. Under an MPI stripe each rank takes the cap
  divided by the ranks on its node (all ranks when the launcher does not
  say), so the node's slabs do not grow with the rank count. The slab size
  changes the strips only at round-off. Smaller slabs cost no time: at
  C20 (6-31G, 6 threads) stage 3 took 4.9–5.9 s at every cap from 1 MiB to 2 GiB, while its
  resident set fell from 2.3 GB at 2 GiB to 0.35 GB at 64 MiB.
- `DirectConfig::scratch_budget_bytes` (2 GiB by default) is the cap of the
  other stage-3 and stage-5 scratch: the aux-major pass, the κ screen's
  own-domain store and the fitted pair assembly. It also bounds the slab
  cap from above.
- The opt-in exact reference still forms the global tensor. Only the
  reference makes a run not reduced-cost.

## Options that change the equation {#integral-direct-options-change-equation}

| Option | Trivial limit | CLI default | Effect |
|---|---|---|---|
| `aux_radius` | at least $10^5$ Bohr (global fit) | 10.0 Bohr | auxiliary fit domain |
| `virt_radius` | all virtuals | 20.0 Bohr | virtual domain on dipole centroids |
| `ao_tail` | 0.0 keeps every shell | 1e-4 | AO-shell support threshold |
| `schwarz_skip` | 0.0 evaluates every triple (bit-identical to a run without the cut) | 1e-5 | Cauchy–Schwarz cut on batch triples, $\sqrt{\max(P\vert P)}\,Q(\mu\nu)\,w(\mu\nu) <$ the threshold, with both factors from the run's operator and $w$ the orbital-coefficient weight above |
| `gate_cal` | none (gate off) | none | the pair gate of equation @eq:amplitude-lmp2-4 |
| `virt_schwarz_kappa` | none (off) | none | $\varepsilon$-linked virtual-candidate screen: keep $a$ if $q_{ia}q^{\max}_j \ge \kappa\varepsilon$ or the mirror |
| `fock_screen` | `false` (plain Eq-8 mask) | `true` (κ = 1) | the Fock-propagated screen of section @sec:amplitude-lmp2-fock-screen; inert at $\varepsilon = 0$ |
| `batch_merge` | 1 | 1 (aux-major); 4 if `eri3_pass = "batches"` | batches merged per integral pass; only the batches pass has batches |
| `eri3_pass` | batches | aux-major | stage-3 integral pass |
| `pair_skip` | `"off"` | `"auto"` (on for erfc and terfc, off for Coulomb) | skips, by proof, pairs whose retained set is empty (section @sec:integral-direct-pair-skip); the energy is bitwise unchanged, so it changes the cost, never the equation |

The first six rows, `batch_merge` and `eri3_pass` are fields of
`ferric-mp2/src/lmp2_direct.rs::DirectConfig`. `gate_cal` and `fock_screen`
are fields of `ferric-mp2/src/lmp2_amplitude.rs::AmplitudeLmp2Config`
(`pair_gate_cal`, `fock_screen`), whose Rust default is `None` for both.
The `DirectConfig` defaults are the trivial limits of its rows except
`aux_radius_bohr`, which defaults to 10.0 Bohr as in the CLI. They differ from the CLI defaults for `virt_radius`, `ao_tail`,
`schwarz_skip`, `batch_merge` and `eri3_pass`. `DirectConfig::production()` holds the
CLI defaults, and the CLI `[local]` section and Python
`run_rimp2(integral_direct=True)` both build their configs from it, so the
two entry points cannot drift apart. The CLI defaults are in
`site/src/reference/input.md`, section `[local]`.

## What is measured {#integral-direct-measured}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| $\varepsilon = 0$, trivial maps, integral-direct: H2O, n-butane / cc-pVDZ, 6-31G (cc-pvdz-ri) | PySCF `DFMP2`, same auxiliary basis | at most $1.38\times10^{-11}$ Ha over the 24 comparisons of the validation page, bar $10^{-10}$ Ha | `crates/ferric-mp2/tests/validation_lmp2_amplitude.rs`, `crates/ferric-mp2/tests/lmp2_direct.rs` |
| finite $\varepsilon$, n-butane / cc-pVDZ | PySCF `DFMP2` | recovers 99.999%, 99.89%, 96.6% at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ at the default maps (virtual radius 20 Bohr, AO tail 1e-4, no pair gate, Fock screen κ = 1), against 99.999%, 99.89%, 96.6% in core (λ-Boys, λ = 10, Fock screen κ = 1 on both) | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

On this molecule the integral-direct defaults and the in-core path, both
with the Fock screen, agree to 0.001 percentage points at $\varepsilon =
10^{-5}$, $10^{-4}$, $10^{-3}$ (all electrons; 2026-11-18).

**Cost and scaling (provisional).** On n-alkanes C20, C32 and C48 (6-31G
with cc-pVDZ-RI, frozen carbon cores, a calibrated pair gate of 0.7 for
Coulomb and 0.02 for erfc with $\omega = 1.0$ Bohr$^{-1}$, every other map at its
default, Fock screen $\kappa = 1$ and λ-Boys $\lambda = 10$; 2026-11-18),
the correlation stage (assembly and solve) grew as about $N^{1.45}$ with
the Coulomb kernel and $N^{1.35}$ with the erfc kernel at $\varepsilon =
10^{-4}$, and as $N^{1.61}$ and $N^{1.55}$ at $\varepsilon = 10^{-3}$,
fitted to the three sizes. The slope falls from the first size interval to
the second (Coulomb at $10^{-4}$: 1.56 from C20 to C32, 1.31 from C32 to
C48), so the large-$N$ exponent is not settled. With the Fock screen and
λ-Boys both off, the time is between 2% higher and 13% lower at each size
(3 to 9% lower at C48), with no trend in $N$; one repeat per point
resolves the exponent to about $\pm 0.1$, and the two settings do not
differ by more than that. These exponents replace the ones measured
2026-09-07 at the earlier maps ($N^{1.24}$ erfc, $N^{1.39}$ to $N^{1.40}$
Coulomb, with a crossover against canonical RI-MP2 near C20); those
describe the earlier maps only, and no crossover has been measured at the
current defaults.

The limits of that measurement:

- It is a fit to **three points** on **one family of molecules in one
  basis**. It is not shown to be linear.
- It is not measured on three-dimensional or diffuse systems.
- It excludes the dense $O(N^3)$ VV-HV virtual-space build (`t_spaces_s`),
  whose cost the fitted exponents do not cover. `virtual_space = "local"`
  (section @sec:amplitude-lmp2-algorithm, step 2) is the opt-in build that
  removes the dense steps; it is not part of this measurement.

Local MP2 without `integral_direct` still builds the global tensor and
makes no scaling claim. The measurement runs the CLI (`rimp2` and
`att-rimp2` with `[local] integral_direct = true`).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_direct.rs` module documentation, `fit_domain_group`, `DirectConfig` | the pipeline, the maps, the exactness anchor and the artifact hypothesis |
| `crates/ferric-mp2/tests/lmp2_map_ablation.rs` (`probe_map_ablation`), measured 2026-11-17 | the per-map $\varepsilon = 0$ errors |
| `scripts/local_mp2_fit_pno_sympy.py`, claims L1, L2, L6 | verified equations @eq:lmp2-domain-fit and @eq:lmp2-aux-restriction |
| `scripts/lmp2_direct_maps_sympy.py`, claims 1 to 3 | verified equations @eq:lmp2-map-pattern, @eq:lmp2-map-value, @eq:lmp2-map-both |
| `scripts/lmp2_attenuated_pair_skip_sympy.py`, claims 1 to 8 | verified equation @eq:lmp2-pair-skip-bound |
| `crates/ferric-mp2/src/lmp2_pair_skip.rs`, `crates/ferric-mp2/src/lmp2_direct.rs::prove_empty_pairs` | the pair skip and its audit |
| S. M. Rump, "Verification of positive definiteness", BIT Numer. Math. 46 (2006) 433 | the shifted-Cholesky eigenvalue floor |
| `crates/ferric-mp2/tests/lmp2_map_ablation.rs` | the $\varepsilon = 0$ anchors at octane and C$_{20}$H$_{42}$ size, and the ablation probes |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the measured scaling |
| `site/src/reference/input.md`, `[local]` | the CLI defaults |
| `site/src/reference/validation.md` | the anchor row and the known limits |
