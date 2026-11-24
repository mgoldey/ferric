# Significant pair lists {#pair-lists}

<!-- status: edit; registry: none -->

Exchange couples two charge clouds through the density matrix. The element
$K_{\mu\lambda}$ collects $(\mu\nu|\lambda\sigma)\,D_{\nu\sigma}$ over $\nu$
and $\sigma$. The integral is negligible unless $\chi_\mu$ sits near
$\chi_\nu$ and $\chi_\lambda$ near $\chi_\sigma$, and in a molecule with a
gap $D_{\nu\sigma}$ decays with the distance between $\nu$ and $\sigma$. So
for each $\mu$ only the $\lambda$ near it contribute, and the number of
quartets that matter grows linearly with the size of the molecule. LinK
builds exchange in time linear in system size by walking only shell pairs
that can matter.

Two lists supply those pairs: one from the geometry, built once, and one from
the density, rebuilt every SCF iteration. The plan: derive the condition each
list tests, then read how `LinkK` combines them with a per-quartet screen.

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/pairs.rs::{SignificantPairs,DensityPairs,intersect_sorted}` |
| Consumer | `ferric-scf/src/link_k.rs::LinkK` |

## Geometry pairs {#pair-lists-geometry-pairs}

A quartet $(ij|kl)$ is computed only if $Q_{ij}Q_{kl}|D| \ge \tau$ for the
density element it multiplies, where $\tau$ is the threshold. To get a
condition on the pair $(i, j)$ alone, replace the other two factors by their
largest values, $Q_{kl} \le Q_{\max}$ and $|D| \le D_{\max}$. A pair can
contribute only if

$$ Q_{ij}\,Q_{\max}\,D_{\max} \ge \tau . $$

`ferric-scf/src/pairs.rs::SignificantPairs::build` keeps the pairs with

$$ Q_{ij}\, Q_{\max} > \tau, \qquad Q_{\max} = \max_{kl} Q_{kl} , $$

the density-free test of PySCF's `q_cond`. It is the necessary condition only
when $D_{\max} \le 1$, and section @sec:pair-lists-open-point-geometry-list
shows that this fails for every closed-shell molecule.

The pair factor is read back from the bound as
$Q_{ij} = \texttt{estimate}(i, j, i, j)^{1/2}$, so the list works with any
`Bound`. Here is the test, with the row loop elided and comments added:

```rust
let q = |i: usize, j: usize| bound.estimate(i, j, i, j).sqrt(); // Q_ij from (ij|ij)
let q_max = (0..nshells)
    .flat_map(|i| (0..nshells).map(move |j| (i, j)))
    .map(|(i, j)| q(i, j))
    .fold(0.0f64, f64::max);                                     // Q_max over all pairs
// ... for each i, for each j in 0..nshells ...
if q(i, j) * q_max > threshold {                                 // Q_ij Q_max > tau
    row.push(j);
}
```

Each row is sorted ascending because $j$ runs in order.

## Density pairs {#pair-lists-density-pairs}

A density block $D_{j\sigma}$ enters exchange only through quartets
$(ij|\sigma l)$, each bounded by $Q_{ij}Q_{\sigma l}|D_{j\sigma}|$. The bra
pair contains $j$ and the ket pair contains $\sigma$. With
$q_{\max}(x) = \max_y Q_{xy}$, the largest factor of any pair containing
$x$, we have $Q_{ij} \le q_{\max}(j)$ and $Q_{\sigma l} \le q_{\max}(\sigma)$.
A block can contribute only if

$$ \max|D_{j\sigma}|\; q_{\max}(j)\; q_{\max}(\sigma) > \tau , $$

where the maximum runs over the elements of the shell block. It is the
tightest necessary condition at pair granularity
(`ferric-scf/src/pairs.rs::DensityPairs::build`):

```rust
if dmax * qmax[j] * qmax[sigma] > threshold {  // max|D_{j sigma}| q_max(j) q_max(sigma) > tau
    row.push(sigma);
}
```

It prunes little. $q_{\max}(x)$ is the shell's own factor $Q_{xx}$ (12 of 12
shells on water and 54 of 54 on butane, def2-SVP, from PySCF diagonals
checked for this section), which lies between 0.60 and 2.19. The product
$q_{\max}(j)q_{\max}(\sigma)$ is therefore of order one for every pair, and
the test asks only whether the density block is above about $\tau$. Tightening the list does not help, because the
bound is attained. Let $i^*$ and $l^*$ be the partners that realize
$q_{\max}(j)$ and $q_{\max}(\sigma)$. The quartet $(i^*j|\sigma l^*)$ has
exactly the Schwarz estimate $q_{\max}(j)\,q_{\max}(\sigma)\max|D_{j\sigma}|$,
so any smaller pair-level test would drop it while its estimate is above
$\tau$. The locality is recovered per
quartet, where `ferric-scf/src/link_k.rs::LinkK`'s `build` screens on the
density blocks of the four shells in hand (`DensityScreen::FourPairK`). The
list bounds the ket loop; the quartet screen bounds the work.

Both lists are sorted ascending. `pairs.rs` also provides `intersect_sorted`,
a linear merge intersection of two sorted slices. The ket loop of `LinkK`
merges the density lists inline (below) and does not call it.

## How LinK uses the lists {#pair-lists-link-uses-lists}

| Step | List | Bound |
|---|---|---|
| 1 | bra pairs $(i, j)$, $j \le i$, from `SignificantPairs` | $Q_{ij}Q_{\max} > \tau$ |
| 2 | ket shells $k$: the union of the density partners of $i$ and of $j$ from `DensityPairs` | $\max\vert D_{xk}\vert\, q_{\max}(x)\, q_{\max}(k) > \tau$ for $x = i$ or $j$ |
| 3 | ket partners $l$ of $k$ from `SignificantPairs` | $Q_{kl}Q_{\max} > \tau$ |
| 4 | each quartet | Schwarz (or the `Bound` in use) times the four-pair density key, $\ge \tau$ |

Why a union in step 2? A quartet $(ij|kl)$ feeds $K$ through the four
exchange blocks $D_{ik}$, $D_{il}$, $D_{jk}$ and $D_{jl}$ (the 8-fold
permutational images of $K_{\mu\lambda} \mathrel{+}= (\mu\nu|\lambda\sigma)D_{\nu\sigma}$).
A ket shell $k$ is needed whenever $D_{ik}$ or $D_{jk}$ is significant, and
both orderings of the ket pair are visited, which covers $D_{il}$ and $D_{jl}$.
The four-pair density key of step 4 is the largest of the shell-blocked
$\max|D|$ over those four blocks.

Here is the inner loop of `LinkK`'s `build`, with the merge walk, the
canonical-ordering and deduplication checks and the scatter elided:

```rust
let dp_ish = dp.partners(ish);                   // density partners of i
let dp_jsh = dp.partners(jsh);                   // density partners of j
// ... sorted-merge walk: each ksh in dp_ish ∪ dp_jsh, once ...
for &lsh in self.sp.partners(ksh) {              // geometry partners of k
    // ... order to (cs1 cs2|cs3 cs4); skip unless this (ish, jsh) owns it ...
    if self.bound.estimate(cs1, cs2, cs3, cs4)   // Q_12 Q_34 (or CSB, QQR)
        * screen.dmax(cs1, cs2, cs3, cs4)        // max(d13, d14, d23, d24)
        < thresh
    {
        continue;
    }
    if let Some(q) = engine.compute_quartet(self.prep, cs1, cs2, cs3, cs4) {
        // ... k_local[(mu, la)] += d[(nu, sg)] * v, and the permutational images ...
    }
}
```

Three points about it:

- **The work unit is a bra pair.** The significant $(i, j)$ pairs with
  $j \le i$ are the parallel tasks, so each task carries the $k, l$ loops for
  one bra pair.
- **Each quartet has one owner.** After ordering to $(cs_1 cs_2|cs_3 cs_4)$,
  only the task whose bra pair is $(cs_1, cs_2)$ computes it, and a bit set
  over $(cs_3, cs_4)$ removes repeats within the task.
- **The key uses four pairings, not six.** A builder that also forms J from
  the same integral needs $D_{12}$ and $D_{34}$ in its key. LinK builds K
  only, so it can screen on the four exchange blocks and skip more.

The lists use the same threshold as the per-quartet screen. A pair a list
drops is then one no surviving quartet could need, so the lists change the
cost and not the result. At threshold $0$ every pair with a positive bound is
kept, which is the trivial limit
(`crates/ferric-scf/tests/screening_exactness.rs`).

## An open point in the geometry list {#pair-lists-open-point-geometry-list}

The geometry list drops the density factor, which is the necessary condition
only when every density element satisfies $|D| \le 1$
(section @sec:pair-lists-geometry-pairs). With the closed-shell density
$D = 2CC^{\mathsf T}$ that fails in every molecule: $\max|D| = 2.14$ on water
and $2.08$ on butane (def2-SVP, PySCF, on the core $s$ functions). So the
list is stricter than the necessary condition by a factor of about 2 in
every run: a pair with $\tau/D_{\max} < Q_{ij}Q_{\max} \le \tau$ is dropped
although a quartet through it can pass the per-quartet screen. The effect on
$K$ has not been measured: no committed test compares LinK with and without
the factor $D_{\max}$ in this test. The LinK anchors below, including the
threshold sweep on alkane_8, pass on the systems they run.

## Numerical anchors {#pair-lists-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| LinK SCF energy against direct | `crates/ferric-scf/tests/link_scf_anchor.rs` | per test |
| LinK K at a threshold sweep against the dense K (alkane_8, Schwarz and QQR) | `crates/ferric-scf/tests/screening_exactness.rs` | per test |
| the screen is reachable: some quartets are skipped | `crates/ferric-scf/tests/link_screen_reachability.rs` | per test |
| `intersect_sorted` | unit tests in `crates/ferric-scf/src/pairs.rs` | exact |

## Sources

- C. Ochsenfeld, C. A. White and M. Head-Gordon, *J. Chem. Phys.* **109**,
  1663 (1998).
