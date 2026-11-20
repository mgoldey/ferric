# A McMurchie–Davidson engine for three-center one-electron integrals {#md-three-center}

<!-- status: edit; registry: none -->

Seminumerical exchange (COSX) computes the exchange matrix by putting one of
the two electrons on a grid. At each grid point $\mathbf r_g$ it needs the
electrostatic potential that a unit charge at $\mathbf r_g$ produces in every
orbital product $\chi_\mu\chi_\nu$:

$$ A^{g}_{\mu\nu} = \int \frac{\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)}{|\mathbf r - \mathbf r_g|}\, d\mathbf r . $$

This is a nuclear-attraction integral with the nucleus replaced by a grid
point and the sign of a repulsion. libint2 can produce it one point at a
time, but each call repeats the work that does not depend on the point, so
the repeated work grows with the number of grid points. The Rust engine in
`md3c1e.rs` builds the parts that do not depend on the point once per shell
pair and runs the point-dependent recursion over tiles of points.

The plan: expand the orbital product in Hermite Gaussians (the $E$
coefficients, which do not depend on the grid point); apply the Coulomb kernel
to each Hermite Gaussian (the $R$ integrals, which do); evaluate the Boys
function that seeds $R$; then read the loop that runs $R$ over eight points at
a time.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/md3c1e.rs::{Md3c1e,e_table,boys}`, `ferric-integrals/src/md3c1e.rs::pair_block` |
| COSX caller | `ferric-integrals/src/cosx_a.rs::a_matrix_at_point` |
| single precision | `ferric-integrals/src/md3c1e/f32_block.rs::pair_block_f32` |

## The algorithm {#md-three-center-algorithm}

### The product as Hermite Gaussians

Take two primitives of exponents $a$, $b$ on centres $A$, $B$, with
$p = a + b$ and $\mathbf P = (a\mathbf A + b\mathbf B)/p$. The Gaussian product
theorem (section @sec:one-electron) turns the two Gaussians into one on $\mathbf P$, but the
polynomial factors $x_A^i x_B^j$ are still centred on $A$ and $B$. McMurchie
and Davidson expand the whole product, one Cartesian direction at a time, in
Hermite Gaussians on $\mathbf P$:

$$ x_A^{i} x_B^{j}\, e^{-a x_A^2 - b x_B^2} = \sum_{t=0}^{i+j} E^{ij}_t\, \Lambda_t(x), \qquad \Lambda_t(x) = \Big(\frac{\partial}{\partial P_x}\Big)^{t} e^{-p x_P^2} . $$

The coefficients $E^{ij}_t$ follow from one property of $\Lambda_t$:

$$ x_P\, \Lambda_t = \frac{1}{2p}\Lambda_{t+1} + t\,\Lambda_{t-1} . $$

For $t = 0$ it reads $\partial_{P_x}e^{-px_P^2} = 2p\,x_P\,e^{-px_P^2}$.
Differentiate both sides with respect to $P_x$, using
$\partial x_P/\partial P_x = -1$, and the identity for $t$ becomes the
identity for $t + 1$.

Now raise $i$ by one. Write $x_A = x_P + X_{PA}$ with $X_{PA} = P_x - A_x$,
multiply the expansion by $x_A$, and use the property on the $x_P$ part:

$$ x_A \sum_t E^{ij}_t\Lambda_t = \sum_t E^{ij}_t\Big[\frac{1}{2p}\Lambda_{t+1} + t\,\Lambda_{t-1} + X_{PA}\Lambda_t\Big] . $$

Collect the coefficient of $\Lambda_t$ on the right. That is the recursion
(one Cartesian direction, $X_{AB} = A_x - B_x$):

$$ E^{00}_0 = e^{-ab X_{AB}^2/p}, \qquad E^{i+1,j}_t = \frac{1}{2p}E^{ij}_{t-1} + X_{PA}E^{ij}_t + (t+1)E^{ij}_{t+1}, \qquad X_{PA} = -\frac{b}{p}X_{AB}, $$

and the same with $X_{PB} = (a/p)X_{AB}$ for $j+1$
(`ferric-integrals/src/md3c1e.rs::e_table`). The start value $E^{00}_0$ is the
Gaussian product prefactor. Check the one-centre limit $A = B$: then
$X_{AB} = X_{PA} = 0$, $E^{00}_0 = 1$, and $E^{10}_1 = 1/(2p)$ with
$E^{10}_0 = 0$. So $x_A e^{-px_A^2} = \Lambda_1/(2p)$, which is
$\partial_{P_x} e^{-px_P^2} = 2p\,x_P e^{-px_P^2}$ divided by $2p$.

### The Coulomb kernel on a Hermite Gaussian

The potential of a spherical Gaussian at a point $\mathbf C$ is

$$ \int \frac{e^{-p|\mathbf r - \mathbf P|^2}}{|\mathbf r - \mathbf C|}\,d\mathbf r = \frac{2\pi}{p}\,F_0\big(p|\mathbf P - \mathbf C|^2\big), $$

the $q \to \infty$ limit of the $(ss|ss)$ formula of section @sec:two-electron. A Hermite
Gaussian is a $\mathbf P$-derivative of the spherical one, and $\mathbf P$
does not appear in $1/|\mathbf r - \mathbf C|$, so the derivatives move
outside the integral. Every Hermite term becomes a derivative of the
spherical result:

$$ A^g = \frac{2\pi}{p} \sum_{tuv} E^{i_xj_x}_t E^{i_yj_y}_u E^{i_zj_z}_v\, R^0_{tuv}(p, \mathbf P - \mathbf r_g), $$

where $(i_x, i_y, i_z)$ and $(j_x, j_y, j_z)$ are the Cartesian powers of the
two primitives, and

$$ R^n_{000} = (-2p)^n F_n(T), \quad T = p|\mathbf P - \mathbf r_g|^2, \qquad R^{n}_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X_{PC} R^{n+1}_{tuv} , $$

and likewise in $u$, $v$, with $X_{PC} = P_x - (r_g)_x$. The superscript $n$
counts how many times $F_0$ has been differentiated: $dF_n/dT = -F_{n+1}$, and
each $\mathbf P$-derivative of $T$ brings a factor $2p X_{PC}$. Check the first
step: $R^0_{100} = X_{PC} R^1_{000} = -2p X_{PC} F_1(T)$, and differentiating
$F_0(p|\mathbf P - \mathbf C|^2)$ directly with respect to $P_x$ gives
$F_0'(T)\cdot 2pX_{PC} = -2pX_{PC}F_1(T)$, the same.

The $E$ coefficients do not depend on the grid point. Only $R$ does, through
$T$ and the three components of $\mathbf P - \mathbf r_g$, and its recursion is
a chain of multiply-adds. So the code builds $E$ once per primitive pair and
runs $R$ over a tile of 8 points (`TILE`) in lock step
(`ferric-integrals/src/md3c1e.rs::Md3c1e::pair_block`; the COSX caller is
`ferric-integrals/src/cosx_a.rs::a_matrix_at_point`).

### The Boys function evaluator

The Boys function $F_n(T)$ seeds the recursion. Section @sec:two-electron-boys
defines it and derives the three facts the evaluator uses: the stable
downward recursion, the Taylor series from $dF_n/dT = -F_{n+1}$, and the
asymptotic form of $F_0$.

The code (`ferric-integrals/src/md3c1e.rs::boys`, `BOYS_T_SWITCH`) combines
them. Below $T = 35$ it takes a
10-term Taylor expansion about the nearest node of a table on
$T_k = k/8$, giving $F_{n_{\max}}$, then runs the downward recursion for the
lower orders. At $T \ge 35$ it uses the asymptotic $F_0$ and the upward
recursion. With nodes $1/8$ apart, $|\delta| \le 1/16$, so the first omitted
Taylor term is below $(1/16)^{10}/10! \approx 2.5 \times 10^{-19}$ relative.
The switch is at 35 rather than the common 25 because $\operatorname{erfc}(5)$
is $1.5 \times 10^{-12}$, while $\operatorname{erfc}(\sqrt{35}) \approx 6 \times 10^{-17}$.
At $T = 0$ (a grid point on the product centre) nothing divides by $T$: the
table node $k = 0$ holds $F_n(0) = 1/(2n+1)$, and the Taylor sum with
$\delta = 0$ returns it.

### The tile loop

Here is the $R$ recursion in `build_r_tile`, the function the tile kernel
calls once per primitive pair and tile, unchanged except for the added
comments:

```rust
let mut fac = 1.0_f64;
for n in 0..=l {                                     // seed R^n_000
    let dst = &mut r[prog.base[n] * TILE..prog.base[n] * TILE + TILE];
    let src = &boys[n * TILE..n * TILE + TILE];
    for g in 0..TILE {
        dst[g] = fac * src[g];                       // (-2p)^n F_n(T_g)
    }
    fac *= -2.0 * p;
}
for st in &prog.steps {                              // one R^n_{t+1,u,v} per step
    let s = st.src as usize * TILE;
    let a: [f64; TILE] = r[s..s + TILE].try_into().expect("tile");
    let pcd = &pc[st.dir as usize];                  // X_PC (or Y, Z) per point
    let d = st.dst as usize * TILE;
    if st.src2 == NONE {
        let dst = &mut r[d..d + TILE];
        for g in 0..TILE {
            dst[g] = pcd[g] * a[g];                  // X_PC R^{n+1}_{0uv}
        }
    } else {
        let s2 = st.src2 as usize * TILE;
        let a2: [f64; TILE] = r[s2..s2 + TILE].try_into().expect("tile");
        let c = st.coef;
        let dst = &mut r[d..d + TILE];
        for g in 0..TILE {
            dst[g] = fma::<FMA>(c, a2[g], pcd[g] * a[g]);  // t R^{n+1}_{t-1} + X_PC R^{n+1}_t
        }
    }
}
```

Four points about it:

- **The recursion is a program, built once.** `prog.steps` is a list of
  (destination, source, second source, coefficient, direction) entries, built
  at first use for each total order $l = l_a + l_b \le 8$
  (`build_r_program`). A target $(t,u,v)$ is built by the recursion in $t$
  if $t > 0$, else in $u$, else in $v$.
  At run time the loop only indexes; there is no branching on $(t,u,v)$.
- **The grid point is the innermost index.** Every array has `TILE` = 8
  contiguous entries, one per grid point, and each line of arithmetic is the
  same operation on all eight. The compiler turns these into SIMD
  instructions, and an AVX2+FMA variant is selected at run time.
- **The sign is repulsive with no flip.** The expansion of $A^g$ has no
  overall minus sign, so `pair_block` returns $+1/|\mathbf r - \mathbf r_g|$
  directly. Raw libint2 nuclear attraction has the opposite sign, and the
  libint2 path `ferric-integrals/src/cosx_a.rs::a_matrix_at_point` negates its
  block. A caller that switches from that path to this one must not negate
  again.
- **FMA changes only the rounding.** `fma::<true>` computes
  $c\cdot a_2 + x$ with one rounding, `fma::<false>` with two. The tests hold
  the two within $10^{-13}$ (@sec:md-three-center-numerical-anchors).

| Step | Depends on the point | Function |
|---|---|---|
| 1. Primitive pairs: $p$, $\mathbf P$, $E^{00}_0$; pairs with a prefactor below $10^{-30}$ are skipped | no | `prim_pair_setup` |
| 2. Hermite coefficients $E^{ij}_t$ per Cartesian direction, and the list of nonzero $E_tE_uE_v$ products | no | `e_table`, `prim_pair_setup` |
| 3. Boys values $F_n(T)$ for each point of a tile | yes | `boys` |
| 4. $R^0_{tuv}$ by the recursion, over the tile | yes | `pair_block` |
| 5. Contract $E\,E\,E\,R$, transform to pure functions | yes | `pair_block` |

The prefactor skip is an underflow guard: the skipped primitive pairs are
bounded by $10^{-30}$, far below any tolerance.

### Design choices

**Why not loop over points inside libint2?** libint2's per-primitive work runs
inside its loop over charges, so asking it for many points repeats steps 1
and 2 for every point. Here steps 1 and 2 run once per primitive pair, and
only steps 3 to 5 run per point.

**Why fold the products of $E$ into a list?** `prim_pair_setup` multiplies
$2\pi/p$, the two contraction coefficients and $E_tE_uE_v$ for every
Cartesian component pair and keeps only
the nonzero products, with the index of the $R$ they multiply. Step 5 is then
one multiply-add per stored coefficient, per point.

## Numerical anchors {#md-three-center-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| `pair_block` against the libint2 path of `a_matrix_at_point`, per shell-pair block | `crates/ferric-integrals/tests/md3c1e_matches_cosx_a.rs` | $10^{-12}\max(1, \max\vert A\vert)$ |
| FMA and portable arithmetic agree | same file | $10^{-13}$ |
| single-precision error budget | `crates/ferric-integrals/tests/md3c1e_f32_error.rs` | per test |

The engine has an independent oracle: libint2's nuclear-attraction integral,
evaluated one point at a time through `a_matrix_at_point`. No proof script
covers the recursions. For this section, the $E$ recursion was checked against
the direct expansion of $x_A^ix_B^je^{-ax_A^2-bx_B^2}$ for $i, j \le 2$, the
first $R$ step against a numerical derivative of $F_0$, and the Boys
identities against quadrature of the definition (section
@sec:two-electron-boys).

## Known limits

Checked 2026-11-19.

- Shells up to $g$ ($l \le 4$, `MAX_L`); `Md3c1e::new` returns an error for
  a higher $l$ and for a pure $p$ shell.
- The Hermite program covers total order $l_a + l_b \le 8$, which is the
  pair of two $g$ shells.

## Sources

- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
- F. Neese, F. Wennmohs, A. Hansen and U. Becker, *Chem. Phys.* **356**, 98
  (2009) (COSX).
