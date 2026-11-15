# 5.3 Significant pair lists

<!-- section: 5.3; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Significant pair lists" -->

LinK builds exchange in time linear in system size by walking only shell
pairs that can matter. Two lists supply those pairs: one from the geometry,
built once, and one from the density, rebuilt every SCF iteration.

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/pairs.rs::{SignificantPairs,DensityPairs,intersect_sorted}` |
| Consumer | `ferric-scf/src/link_k.rs::LinkK` |

<!-- include: crates/ferric-integrals/theory.md#"Significant pair lists" -->

## 5.3.1 How LinK uses the lists

| Step | List | Bound |
|---|---|---|
| 1 | bra pairs $(i, j)$ from `SignificantPairs` | $Q_{ij}Q_{\max} > \tau$ |
| 2 | for each $j$, the density partners $\sigma$ from `DensityPairs` | $\max\vert D_{j\sigma}\vert\, q_{\max}(j)\, q_{\max}(\sigma) > \tau$ |
| 3 | ket pairs $(\sigma, l)$, the intersection of the two lists | `intersect_sorted` |
| 4 | each quartet | Schwarz (or the `Bound` in use) times the four-pair density key |

The lists use the same threshold as the per-quartet screen. At threshold $0$
every pair with a positive bound is kept, which is the trivial limit
(`crates/ferric-scf/tests/screening_exactness.rs`).

## 5.3.2 An open point in the geometry list

The geometry list drops the density factor, which is the necessary condition
only when every density element satisfies $|D| \le 1$ (theory section above).
The book records this as derived, not measured: no committed test sweeps a
basis with density elements above 1 against the direct exchange. The LinK
anchors below pass on the systems they run.

## 5.3.3 Numerical anchors

| What | Test | Bar |
|---|---|---|
| LinK SCF energy against direct | `crates/ferric-scf/tests/link_scf_anchor.rs` | per test |
| LinK K at a threshold sweep against the dense K (alkane_8, Schwarz and QQR) | `crates/ferric-scf/tests/screening_exactness.rs` | per test |
| the screen is reachable: some quartets are actually skipped | `crates/ferric-scf/tests/link_screen_reachability.rs` | per test |
| `intersect_sorted` | unit tests in `crates/ferric-scf/src/pairs.rs` | exact |

## Sources

- C. Ochsenfeld, C. A. White and M. Head-Gordon, *J. Chem. Phys.* **109**,
  1663 (1998).
