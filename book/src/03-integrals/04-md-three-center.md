# 3.4 A McMurchie–Davidson engine for three-center one-electron integrals

<!-- section: 3.4; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"McMurchie-Davidson three-center one-electron integrals" -->

Seminumerical exchange (COSX) needs, at every grid point, the potential matrix
of a unit charge at that point. libint2 can produce it one point at a time as
a nuclear-attraction integral, but the cost is the per-call setup. The Rust
engine in `md3c1e.rs` builds the parts that do not depend on the point once per
shell pair and runs the point-dependent recursion over tiles of points.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/md3c1e.rs::{Md3c1e,e_table,boys}`, `ferric-integrals/src/md3c1e.rs::pair_block` |
| COSX caller | `ferric-integrals/src/cosx_a.rs::a_matrix_at_point` |
| single precision | `ferric-integrals/src/md3c1e/f32_block.rs::pair_block_f32` |

<!-- include: crates/ferric-integrals/theory.md#"McMurchie-Davidson three-center one-electron integrals" -->

## 3.4.1 The algorithm

| Step | Depends on the point | Function |
|---|---|---|
| 1. Primitive pairs: $p$, $\mathbf P$, $E^{00}_0$; pairs with a prefactor below $10^{-30}$ are skipped | no | `prim_pair_setup` |
| 2. Hermite coefficients $E^{ij}_t$ per Cartesian direction, and the list of nonzero $E_tE_uE_v$ products | no | `e_table`, `prim_pair_setup` |
| 3. Boys values $F_n(T)$ for each point of a tile | yes | `boys` |
| 4. $R^0_{tuv}$ by the recursion, over the tile | yes | `pair_block` |
| 5. Contract $E\,E\,E\,R$, transform to pure functions | yes | `pair_block` |

The prefactor skip is an underflow guard: the skipped primitive pairs are
bounded by $10^{-30}$, far below any tolerance.

## 3.4.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| `pair_block` against the libint2 path of `a_matrix_at_point`, per shell-pair block | `crates/ferric-integrals/tests/md3c1e_matches_cosx_a.rs` | $10^{-12}\max(1, \max\vert A\vert)$ |
| FMA and portable arithmetic agree | same file | $10^{-13}$ |
| single-precision error budget | `crates/ferric-integrals/tests/md3c1e_f32_error.rs` | per test |

## Sources

- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
- F. Neese, F. Wennmohs, A. Hansen and U. Becker, *Chem. Phys.* **356**, 98
  (2009) (COSX).
