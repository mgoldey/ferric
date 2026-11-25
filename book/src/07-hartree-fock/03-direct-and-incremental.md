# Direct and incremental J/K builds {#direct-and-incremental}

<!-- status: ready; registry: rhf, uhf, rohf -->

$J$ and $K$ need the four-centre integrals $(\mu\nu|\lambda\sigma)$, about
$N^4/8$ distinct ones, which for a few hundred basis functions is already too
many to store: at $N = 300$ that is $1.0\times10^9$ integrals, 8.1 GB of
f64. The default route therefore computes the integrals every
iteration and contracts each block with the density as soon as it exists:
a direct build, with nothing fitted, so $J$ and $K$ are exact to the
screening threshold. Most integrals between functions on distant atoms are
tiny, and a cheap bound (Cauchy–Schwarz) identifies them without computing
them. Late in an SCF the density hardly changes, so building $J$ and $K$ from
the change of the density lets the screen drop most quartets; that is the
incremental build.

This section states the contraction, its 8-fold symmetry and the screen
(@sec:direct-and-incremental-equations, @sec:direct-and-incremental-algorithm), when the incremental build runs and what it costs in
reproducibility (@sec:direct-and-incremental-incremental-build-runs, @sec:direct-and-incremental-reproducibility), and the anchors and options.

| Symbol | Meaning | Code name |
|---|---|---|
| $s_1 \ge s_2$, $s_3$, $s_4$ | shell indices of a canonical quartet | `s1`, `s2`, `s3`, `s4` |
| $Q_{st}$ | Schwarz factor of the shell pair $(s, t)$ | `q_table[(s1, s2)]` |
| $d_{\max}$ | density bound for the quartet | `screen.dmax(s1, s2, s3, s4)` |
| $\tau_{\rm int}$ | `[scf] integral_thresh`, default $10^{-12}$ | `thresh` |
| $\Delta D$ | change of the density since the last build | `delta_d` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/direct_jk.rs::DirectJK`, `ferric-scf/src/quartet_scatter.rs::scatter_bra_pair` |
| CLI | the default when `df_j_aux`, `df_k_aux` and `k_builder` are unset under `kind = "rhf"`, `"uhf"`, `"rohf"` |
| Python | `run_rhf`, `run_uhf`, `run_rohf` with no `df_*_aux` |

## The equations {#direct-and-incremental-equations}

### Unique quartets

Real integrals have 8-fold symmetry:
$(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda) = (\lambda\sigma|\mu\nu)$
and the combinations of these swaps. So each distinct integral should be
computed once and used for every place it appears in $J$ and $K$.

The direct builder (`DirectJK`, `crates/ferric-scf/src/direct_jk.rs`) loops
over canonical shell quartets,

$$ s_1 \ge s_2, \qquad s_3 \le s_1, \qquad s_4 \le \begin{cases} s_2 & s_3 = s_1 \\ s_3 & \text{otherwise}, \end{cases} $$

and computes the full block $(\mu\nu|\lambda\sigma)$ with
$\mu\in s_1, \nu\in s_2, \lambda\in s_3, \sigma\in s_4$. Read the conditions
as an ordering of shell pairs. The bra pair has $s_1 \ge s_2$. If
$s_3 < s_1$, every ket pair $(s_3, s_4)$ with $s_4 \le s_3$ comes before the
bra pair in lexicographic order. If $s_3 = s_1$, the cap $s_4 \le s_2$ keeps
the ket pair at or before the bra pair. So each unordered pair of shell pairs
is visited once, which removes the bra–ket swap.

Each element $v = (\mu\nu|\lambda\sigma)$ of a computed block then goes to
every distinct image of the symmetry. Define three flags,
$\delta_{12} = [s_1 \ne s_2]$, $\delta_{34} = [s_3 \ne s_4]$ and
$\delta_{\rm bk} = [(s_1,s_2) \ne (s_3,s_4)]$. The element adds

$$ J_{\mu\nu} \mathrel{+}= v\,D_{\lambda\sigma}, \qquad K_{\mu\lambda} \mathrel{+}= v\,D_{\nu\sigma}, $$

and, when the flag of the image is set, the same with $\mu\leftrightarrow\nu$
($\delta_{12}$), $\lambda\leftrightarrow\sigma$ ($\delta_{34}$), both, and the
bra–ket swap with its three sub-images ($\delta_{\rm bk}$). That is up to
eight index images, each with one add to $J$ and one to $K$: up to 16 adds
per element (`scatter_bra_pair`, `scatter_bra_pair_probed`, and the blocked
`digest_j`/`digest_k` for blocks of 16 or more integrals in
`crates/ferric-scf/src/quartet_scatter.rs`).

Why is there no degeneracy factor? A flag is clear exactly when its swap maps
the shell quartet onto itself. Then the block already contains both orderings
of the functions (for $s_1 = s_2$ it holds $(\mu\nu|\cdot)$ and
$(\nu\mu|\cdot)$ as separate elements), so adding the image again would count
it twice. Because each distinct shell-index image is added once, the sum over
unique quartets reproduces

$$ J[D]_{\mu\nu} = \sum_{\lambda\sigma}(\mu\nu|\lambda\sigma)D_{\lambda\sigma}, \qquad K[D]_{\mu\nu} = \sum_{\lambda\sigma}(\mu\lambda|\nu\sigma)D_{\lambda\sigma} $$

exactly for a symmetric $D$.

### Screening

The Cauchy–Schwarz inequality for the Coulomb inner product bounds each
integral by two diagonal ones,
$|(\mu\nu|\lambda\sigma)| \le \sqrt{(\mu\nu|\mu\nu)}\sqrt{(\lambda\sigma|\lambda\sigma)}$.
Take the largest value over each shell pair,

$$ Q_{st} = \max_{a\in s,\,b\in t}\sqrt{|(ab|ab)|}, $$

so that every integral of a quartet is at most $Q_{s_1s_2}Q_{s_3s_4}$. The
table $Q$ costs $O(N^2)$ diagonal integrals once per geometry (section @sec:schwarz).

Each add above is an integral times a density element. Let $d_{\max}$ be the
largest shell-block maximum $\max|D_{\mu\nu}|$ over the six pairings of the
four shells (`DensityScreen::SixPair`; the open shell uses
$|D^\alpha| + |D^\beta|$). Then every add of the quartet is at most
$Q_{s_1s_2}Q_{s_3s_4}d_{\max}$ in magnitude, and the quartet is skipped when

$$ Q_{s_1s_2}\,Q_{s_3s_4}\,d_{\max} < \tau_{\rm int}, $$

with $\tau_{\rm int}$ = `integral_thresh` (default $10^{-12}$). Each dropped
contribution to any element of $J$ or $K$ is therefore below $\tau_{\rm int}$.
This bounds each dropped term, not their sum: an element of $J$ that collects
many dropped quartets can be off by more than $\tau_{\rm int}$.

`[scf] screening = "csb"` replaces the bound by
$\min(Q_{12}Q_{34}, M_{13}M_{24}, M_{14}M_{23})$, with
$M_{st} = \max_{a\in s,\,b\in t}(aa|bb)^{1/2}$ the shell-pair maximum of
the repulsion between the densities of two functions (section
@sec:schwarz-combined-schwarz-bound derives it); it is never looser than the
Schwarz product. `"csam"` multiplies the Schwarz product by
$\max(X_{13}X_{24}, X_{14}X_{23})^{1/2}$, with the ratio
$X_{st} = \max_{a\in s,\,b\in t}|(aa|bb)|/\sqrt{|(aa|aa)||(bb|bb)|} \le 1$
(Thompson and Ochsenfeld, J. Chem. Phys. 147, 144101 (2017), Eqs. 9 to 12,
ported from Psi4 in `crates/ferric-integrals/src/csam.rs`). That product is
an estimate, not a bound: it can fall below the integral it screens.

### Fock matrix

$F = h + J - \tfrac12 c_xK$ ($c_x = 1$ for Hartree–Fock),
$F^\sigma = h + J[D^\alpha + D^\beta] - c_xK[D^\sigma]$ for UHF, and
$E = \tfrac12\operatorname{tr}[D(h+F)] + E_{nn}$ (section @sec:energy-functionals).

### Incremental builds

$J$ and $K$ are linear in $D$. Write $\Delta D_n = D_n - D_{n-1}$ for the
change of the density since the last build. Then

$$ J[D_n] = J[D_{n-1}] + J[\Delta D_n], \qquad K[D_n] = K[D_{n-1}] + K[\Delta D_n] . $$

The closed-shell loop of `solve_rhf` keeps $J$ and $K$ (not $F$) in buffers
and adds $J[\Delta D]$, $K[\Delta D]$ built by `DirectJK::build_incremental`.
$F$ itself cannot be updated this way in a Kohn–Sham run, because the
exchange–correlation potential it carries is not linear in $D$; $F$ is
assembled from the buffers each iteration (section @sec:energy-functionals).

The screen is the same absolute test with $d_{\max}$ taken from $\Delta D$.
As the SCF converges $\Delta D$ shrinks, $d_{\max}$ with it, and more
quartets fall below $\tau_{\rm int}$. That is where the saving comes from.

The price is that screening errors accumulate. Write $\tilde J[\cdot]$ for the
screened build. After a full build at $D_r$ and $k$ increments, the buffer
holds $\tilde J[D_r] + \sum_{l=1}^{k}\tilde J[\Delta D_{r+l}]$. Subtract the
exact $J[D_{r+k}] = J[D_r] + \sum_l J[\Delta D_{r+l}]$ and use linearity:

$$ \text{buffer} - J[D_{r+k}] = \big(\tilde J[D_r] - J[D_r]\big) + \sum_{l=1}^{k}\big(\tilde J[\Delta D_{r+l}] - J[\Delta D_{r+l}]\big) . $$

Each bracket is the sum of the terms one screened build dropped, each below
$\tau_{\rm int}$. So the error grows with the number of screened builds
since the last full one. The buffers are rebuilt from the full density on
iterations $1, 9, 17, \dots$ (every 8), so at most seven increments follow
each full build. No other event (a DIIS reset, MOM, a level-shift change)
forces a rebuild.

The scheme is on only when it can pay and its error cannot reach the
convergence target; section @sec:direct-and-incremental-incremental-build-runs
lists the conditions. The environment variable `FERRIC_SCF_INCREMENTAL`
forces it on or off.

### The in-core store

`[scf] eri` (`crates/ferric-scf/src/incore_jk.rs`) takes the other path for
small systems. Below about 200 functions the unscreened integrals of one
geometry are stored as the packed lower triangle of the pair matrix, and $J$
and $K$ come from matrix products. `"auto"` takes the store when it fits a
quarter of the memory budget and both $J$ and $K$ are four-centre. The
incremental build is off with the store: a stored-integral build costs the
same for $\Delta D$ as for $D$, so there is nothing to save.

> **Verified equation {#eq:scf-incremental-jk}** · incremental builds
>
> $J[D_n] = J[D_{n-1}] + J[\Delta D_n]$ and the same for $K$; accumulating
> increments with a full rebuild every 8 iterations reproduces $J[D]$ and
> $K[D]$ of the final density.
>
> | | |
> |---|---|
> | Proof | `scripts/scf_jk_builders_sympy.py`, claim N9 |
> | Negative control | dropping the accumulation (`jb = j_exact(dd)`) must fail N9 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/direct_jk.rs::DirectJK` |
>
> <!-- verified: id=scf.incremental-jk; script=scripts/scf_jk_builders_sympy.py; claim=N9; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/direct_jk.rs::DirectJK; registry=rhf -->

## The algorithm {#direct-and-incremental-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | Schwarz table $Q$ once per geometry | `ferric-scf/src/screening.rs::SchwarzBounds` |
| 2 | shell-block maxima of $|D|$ (or $|\Delta D|$) | `ferric-scf/src/quartet_scatter.rs::DensityScreen` |
| 3 | loop over canonical quartets, skip below the bound | `ferric-scf/src/quartet_scatter.rs::scatter_bra_pair_probed` |
| 4 | one libint2 block per surviving quartet | `ferric-integrals/src/engine.rs::Engine` |
| 5 | scatter to every distinct image in $J$ and $K$ | `ferric-scf/src/quartet_scatter.rs::digest_j` and its siblings |

One call of `scatter_bra_pair_probed` handles one bra pair $(s_1, s_2)$ and
loops over its ket pairs. That bra pair is the unit of parallel work. The
quartet is canonical in the fourth index,

<!-- excerpt: ferric-scf/src/quartet_scatter.rs::scatter_bra_pair_probed -->
```rust
let s4max = if s3 == s1 { s2 } else { s3 };
        for s4 in 0..=s4max {
```

and skipped by one comparison, with `bound` the Schwarz product
$Q_{s_1s_2}Q_{s_3s_4}$ unless a tighter `[scf] screening` bound is selected:

<!-- excerpt: ferric-scf/src/quartet_scatter.rs::scatter_bra_pair_probed -->
```rust
if bound * dmax < thresh {
                continue;
            }
```

| Code | Equation |
|---|---|
| `bound` | $Q_{s_1s_2}Q_{s_3s_4}$, or the CSB minimum |
| `dmax` | $d_{\max}$ over the six pairings of the four shells |
| `thresh` | $\tau_{\rm int}$ |

Each element of a computed block goes to at most 16 places in $J$ and $K$,
with no degeneracy factor, because the canonical loop visits each distinct
shell quartet once and the scatter writes each distinct index image once.
A quartet of four $s$ shells, one function each, shows the images without
inner loops. Here it is from the same
function, with the probe timer and the bra–ket half shortened:

```rust
let v = unsafe { *q.get_unchecked(0) };   // v = (μν|λσ)
mode.add_j(o1, o2, d, o3, o4, v);         // J_μν += v D_λσ
mode.add_k(o1, o3, d, o2, o4, v);         // K_μλ += v D_νσ
if sym12 {                                // δ12: μ ↔ ν
    mode.add_j(o2, o1, d, o3, o4, v);     // J_νμ += v D_λσ
    mode.add_k(o2, o3, d, o1, o4, v);     // K_νλ += v D_μσ
}
if sym34 {                                // δ34: λ ↔ σ
    mode.add_j(o1, o2, d, o4, o3, v);     // J_μν += v D_σλ
    mode.add_k(o1, o4, d, o2, o3, v);     // K_μσ += v D_νλ
}
if sym12 && sym34 {                       // both swaps
    mode.add_j(o2, o1, d, o4, o3, v);
    mode.add_k(o2, o4, d, o1, o3, v);
}
if sym1234 {                              // δbk: (μν) ↔ (λσ)
    // ... the same four pairs with bra and ket exchanged ...
}
```

- The four `add_j` calls of the bra half land on $J_{\mu\nu}$ and
  $J_{\nu\mu}$, two each, through $D_{\lambda\sigma}$ and
  $D_{\sigma\lambda}$. For a symmetric $D$ both elements receive the same
  total, so $J$ comes out symmetric.
- `mode` decides what an add does. The `JkMode` variants accumulate $J$
  only, $K$ only, both, the UHF triple $J$, $K_\alpha$, $K_\beta$ from
  $D_\alpha$ and $D_\beta$, or many densities at once for the response
  equations; the loop above is shared.
- Blocks of 16 or more integrals go through the blocked `digest_j` and
  `digest_k`, which apply the same images as matrix operations.

## The bra-pair-local digest {#direct-and-incremental-pair-digest}

The scatter above writes every element to up to 16 places scattered over
$J$ and $K$. The default digest (`[scf] jk_digest = "pair"`,
`ferric-scf/src/quartet_scatter.rs::PairSpec`) computes the same sums with
writes that stay near the bra pair. For one bra pair $(s_1, s_2)$ it keeps
three accumulators: a bra Coulomb block, a half Coulomb matrix
$J^{\rm h}$ and a bra-row exchange matrix $K^{\rm b}$.

- **Bra Coulomb block.** For every computed ket pair, each bra element
  gets a dot product with $E^{34}_{\lambda\sigma} = D_{\lambda\sigma} +
  \delta_{34} D_{\sigma\lambda}$. The block goes into $J^{\rm h}$ once,
  after the last ket pair.
- **Ket Coulomb block.** For a distinct bra and ket ($\delta_{\rm bk}$),
  the ket block $\sum_{\mu\nu} v\, E^{12}_{\mu\nu}$, with $E^{12} =
  D_{\mu\nu} + \delta_{12} D_{\nu\mu}$, goes into $J^{\rm h}$ once per
  quartet.
- **Exchange.** Only the four bra-row images are written:
  $K^{\rm b}_{\mu\lambda} \mathrel{+}= w\, v D_{\nu\sigma}$, and the
  $\delta_{12}$, $\delta_{34}$ images, all in rows $\mu \in s_1$ and $\nu
  \in s_2$. The working set is the $n_1 + n_2$ rows of $D$ and $K^{\rm b}$.

Once per build, after the group partials are summed in their usual order,

$$ J = J^{\rm h} + (J^{\rm h})^{\mathsf T}, \qquad K = K^{\rm b} + (K^{\rm b})^{\mathsf T}. $$

For a symmetric $D$ the four bra–ket-swapped exchange images, $K_{\lambda\mu}
\mathrel{+}= v D_{\sigma\nu}$ and the rest, are the transposes of the
bra-row images. So the transpose in the last step supplies them. The
transpose also doubles anything that is its own image, so three degenerate
cases take a factor ½:

| Case | Weight | Why |
|---|---|---|
| $s_1 = s_2$ | ½ on the bra block in $J^{\rm h}$ | the block holds both $(\mu\nu|\cdot)$ and $(\nu\mu|\cdot)$ |
| $s_3 = s_4$ | ½ on the ket block in $J^{\rm h}$ | likewise for $(\cdot|\lambda\sigma)$ |
| $(s_3,s_4) = (s_1,s_2)$ | $w = ½$ on $K^{\rm b}$ | there are no swapped images to supply |

A density that is not symmetric takes the general form instead. The swapped
images are accumulated, transposed, into a second bra-row matrix $K^{\rm t}$
against $D^{\mathsf T}$, and $K = K^{\rm b} + (K^{\rm t})^{\mathsf T}$ with
$w = 1$ everywhere. Each build checks its exchange densities once:
$\max|D - D^{\mathsf T}| \le 10^{-14} \max(\max|D|, 1)$ counts as symmetric,
and such a density is symmetrised explicitly to $\frac12(D + D^{\mathsf
T})$, so $K$ comes out exactly symmetric. The floor of 1 keeps the
incremental build's $\Delta D$ on the symmetric form: a difference of two
$O(1)$ densities carries their absolute rounding asymmetry. UHF checks each
spin, and if either is not symmetric both run the general form.

The integrals, the screen, the work partition and the fold order do not
change, so the result still does not depend on the thread count. Only the
bracketing of each element's sum differs from the scatter, by about
$10^{-14}$ relative in $J$ and $K$ and $10^{-12}$ Ha in SCF energies
(`crates/ferric-scf/tests/jk_pair_digest.rs`). `jk_digest = "legacy"`, or
`FERRIC_JK_DIGEST=legacy`, restores the scatter bit for bit. The innermost
loop is specialised at compile time for ket shells of 1, 3 and 5 functions
(`ferric-scf/src/quartet_scatter.rs::pair_quartet`).

Measured runs, not tests (octane, RHF, `eri = "direct"`, 2 threads,
`FERRIC_TIMING=1`, mean of 10 builds over two runs each, interleaved, on a
shared 4-core box):

| Basis | digest thread-s per build, legacy → pair | J/K build wall, legacy → pair | SCF energy shift |
|---|---|---|---|
| cc-pVDZ (rotated) | 2.54 → 0.86 (2.9×) | 5.40 → 4.28 s (1.26×) | −5.7e-13 Ha |
| def2-SVP | 2.24 → 0.85 (2.6×) | 3.44 → 2.83 s (1.22×) | −1.1e-13 Ha |

## Generally contracted bases {#direct-and-incremental-general-contraction}

libint takes one segmented shell per contraction column, and stock libint
cannot evaluate a generally contracted shell. A correlation-consistent basis
stores several columns over one exponent list: cc-pVDZ carbon has s columns
with 9, 9 and 1 non-zero coefficients over the same 9 exponents, and the
single column is the most diffuse exponent, which also appears in the two
long columns. The direct builder would recompute the shared primitives for
each column, and the diffuse primitive inside the core column keeps every
tight × diffuse primitive pair alive at any distance, since libint's
primitive-pair screen uses the coefficients only.

`DirectJK` therefore builds $J$ and $K$ on an exactly equivalent rotated
basis (`ferric-integrals/src/contraction_rotation.rs::RotatedBasis`). Within
each group of shells on one atom with the same $l$ and the same exponent
list, it first subtracts the single-primitive columns from the long ones,
$k' = k - (c_{kq}/c_{sq})\,s$. That stage is shared with the periodic
short-range walks. Then it recombines the remaining long columns so that
each one loses the primitives the others keep as pivots, the optimized
general contraction of Hashimoto, Hirao and Tatewaki (Chem. Phys. Lett.
243, 190 (1995)). cc-pVDZ carbon becomes s 7, 7, 1 and p 3, 1, and hydrogen
s 3, 1. Write the parent functions as $\chi = T\chi'$. $T$ is block
diagonal, the identity outside the rotated groups, with a condition number of
1.8 for carbon s. A group whose block would exceed 100 keeps the first stage
only, and a group whose first-stage block already exceeds 100 is not rotated.
Then, for any density,

$$ D' = T^{\mathsf T} D\,T, \qquad J[D] = T\,J'[D']\,T^{\mathsf T}, \qquad K[D] = T\,K'[D']\,T^{\mathsf T}, $$

where $J'$, $K'$ are the direct builds on the rotated shells. The Schwarz
table, the CSB or CSAM table when one is selected, the density screen, the
engines and their shell-pair cache all belong to the rotated shells. The
incremental build applies the same formula to $\Delta D$. The two transforms
are serial; in the measured runs below they took under 2 ms per build on
octane. A basis with nothing to
rotate (def2, Pople, STO-3G, pcseg) never enters this path.

The rotated basis, its Schwarz table and its engines are built once per
geometry and kept with the geometry's screening bounds
(`ferric-scf/src/direct_jk/gc_rotation.rs::GcSlot`), so every four-centre
build on that geometry shares them. The rotation covers the SCF builds:
`solve_rhf` on the exact four-centre route (RHF and hybrid RKS), the
combined open-shell build of `solve_uhf` and `solve_rohf` (and their
Kohn–Sham hybrids with $\omega = 0$), the exact builds of the DF-increments
SCF, the separate `DirectJ` and `DirectK` builders (the $J$ or $K$ half when
the other half is fitted, LinK or COSX), and the in-core store, which then
holds the integrals of the rotated shells. It also covers the builds that
follow an SCF on the same geometry: the Newton and TRAH steps, the stability
analyses, the CPHF of the Hessian and of the response properties, the
Z-vector and the OO-MP2 $J/K$, all through `rhf::build_jk_with_pool` and its
siblings. The SCF records its setting on the bounds, so these builds run the
same rotation as the SCF. Three consumers that build their own bounds copy
the setting over: the Kohn–Sham Z-vector response (`zvector_ks`), the
direct CASSCF integrals (`CasScfConfig::gc_rotation`) and the exact exchange
of the GW $V_{xc}$. Bounds that no SCF configured, such as those of the
Python `jk`, `grad_jk` and `FockResponse` blocks, never rotate, and
`FERRIC_GC_ROTATE` does not apply to them. The analytic gradient's four-centre derivative
integrals are those of the rotated shells, contracted with the rotated
densities: $T$ does not depend on the geometry and keeps every shell on its
atom, so this is the same gradient. LinK, CFMM, the Hessian's
second-derivative integrals and the periodic code use the unrotated basis.
`[scf] gc_rotation`
(`RhfConfig::gc_rotation`, spelled exactly) selects the rotation, then
`FERRIC_GC_ROTATE` when the key is unset (case and blanks ignored), then
`auto`: `auto` runs both stages, `rot1` the first
stage only, and `off` the unrotated builder bit for bit; any other value is
an error (`ferric-scf/src/direct_jk/gc_rotation.rs::GcRotate`).

The timings below are measured runs, not tests. They used `FERRIC_TIMING=1`,
`[scf] eri = "direct"`, `energy_conv`
1e-10 and `density_conv` 1e-8 (every build full), 4 rayon threads, on a box
shared with three other agents' builds (load average 7 to 8 on 4 cores).
Wall times varied by a factor of 3 between repeats, so the process CPU time
is the steadier figure.

| System (functions) | Threads | Per-build wall, off → auto (s) | libint thread-s per build, off → auto | Quartets per build, off → auto | SCF CPU time, off → auto (s) |
|---|---|---|---|---|---|
| hexane / cc-pVDZ (154) | 4 | 4.8–8.5 → 1.4–4.1 | 16.1–28.5 → 4.0–11.8 | 4.41 M → 3.65 M | 113–126 → 52–54 |
| hexane / cc-pVDZ (154) | 1 | 9.2–9.8 → 4.2–4.2 | 7.7–8.2 → 2.9–3.0 | 4.41 M → 3.65 M | 110–111 → 49–51 |
| octane / cc-pVDZ (202) | 4 | 17.3–20.2 → 5.9–6.8 | 57.0–66.8 → 16.3–18.8 | 10.87 M → 8.13 M | 292–296 → 120–122 |
| hexane / def2-SVP (154) | 4 | 2.7–2.8 → 2.6–2.8 | 6.2–6.5 → 6.2–6.6 | 3.63 M → 3.63 M | 35–37 → 36–36 |

Two or three runs per row, interleaved. The single-thread rows are the least
disturbed by the other jobs: 2.2 to 2.4 times faster per build, and 2.6 to
2.8 times less libint time. Stage 1 alone (`rot1`) on octane took 6.9 s
per build and 140 s of CPU time, against 6.8 s and 122 s for both stages.
def2-SVP has nothing to rotate, so `auto` runs the same code as `off` and
gives the same energy bit for bit. In a measured run (not a test), dodecane / cc-pVDZ at the default SCF
settings (incremental builds, 10 iterations) took 260 s of wall time and
516 s of CPU time with `off`, and 128 s and 195 s with `auto`; def2-SVP took
70 s and 141 s.

A second set of measured runs (not tests) covers the builds that follow the
SCF. Same settings, 2 rayon threads, load average about 2 (the runs' own two
threads). Butane / cc-pVDZ (106 functions) with `check_stability = true`: the
RHF internal and external Davidson solves made 435 pooled $J/K$ builds with
`off` and 425 with `auto`, at a median 1.51 s and 0.73 s each, and the whole
run took 685 s and 322 s. The rotated basis was set up once for the
geometry, in 7 ms, and shared by the SCF and all the stability builds. In the
same session the per-build SCF wall went from 5.49 s to 2.51 s on hexane and
from 13.5 s to 5.6 s on octane.

## When the incremental build runs {#direct-and-incremental-incremental-build-runs}

The incremental build is closed-shell, direct-only and conditional:

| Condition | Source |
|---|---|
| no density fitting, no pluggable K builder | the route decision in `ferric-scf/src/fock_assembly.rs::resolve_jk_route` |
| no in-core integral store | `ferric-scf/src/direct_jk.rs::DirectJK` (`incremental_pays`) |
| `energy_conv` $\ge 10^4\,\tau_{\rm int}$ and `density_conv` $\ge 10^3\,\tau_{\rm int}$ | `ferric-scf/src/direct_jk.rs::closed_shell_incremental_enabled` |
| open shell | only with `FERRIC_SCF_UHF_INCREMENTAL=1` |

The full rebuild every 8 iterations is
`ferric-scf/src/driver.rs::INCREMENTAL_FULL_REBUILD_EVERY`, one value for
the three drivers; nothing else, including a DIIS reset, forces one.

Why is the open shell off by default? The doc comment of
`ferric-scf/src/direct_jk.rs::open_shell_incremental_enabled` records the
measurement: the open-shell increments reproduce a full
rebuild, but the screen started rejecting quartets only once $\max|\Delta D|$
fell below about $10^{-8}$, which the measured runs reached only in their
last iteration or two. The ratios of full to incremental quartet counts it
records run from 0.987 to 1.046 (CH3 and alkane cations). For the closed
shell the same comment records the incremental build computing 0.73 to 0.86
times the quartets of the full one on alkane chains at STO-3G and 6-31G, and
no change on water and methane in cc-pVDZ.

## Reproducibility {#direct-and-incremental-reproducibility}

The incremental build changes the floating-point order of the sum relative to
a full build: $J[D_{n-1}] + J[\Delta D_n]$ is not bitwise $J[D_n]$, and the
screen drops a different set of quartets on $\Delta D$ than on $D$. The two
agree to the screening threshold, and the rebuild every 8 iterations keeps the
screened increments from accumulating without bound (the error expression in
@sec:direct-and-incremental-equations has at most eight screened builds in it).

The contraction rotation of @sec:direct-and-incremental-general-contraction
changes the bits too: the rotated shells have their own Schwarz table, so the
screen keeps a slightly different quartet set, and the transforms reassociate
the sums. In `crates/ferric-scf/tests/gc_rotation.rs`, $J$ and $K$ at a
fixed density differ from the unrotated builder's by at most 4.9e-12 (water
with a ghost neon in cc-pVDZ; the set includes Cu, ECP, mixed and Cartesian
bases) and converged energies by at most 1.2e-12 Ha. At threshold 0 the
rotated build equals the dense $T J' T^{\mathsf T}$ to rounding. Energy
differences grow with the number of screened quartets: the measured octane /
cc-pVDZ run below moved by 8.3e-12 Ha. The rotated builder is bitwise
identical across thread counts, and `off` reproduces the unrotated builder
bit for bit.

## Numerical anchors {#direct-and-incremental-numerical-anchors}

The direct builder is the exact reference against which every other route in
this chapter is measured. Its own anchors are the energy anchors of section
@sec:energy-functionals (1.8e-12 Ha RHF, 4.0e-12 Ha UHF and ROHF against PySCF).

<!-- figures: site/src/reference/validation.md "UHF and ROHF energies, stability-checked" -->

The incremental path is compared with a full rebuild every iteration, with
the in-core store off, on water, methane and hexane in cc-pVDZ
(`incremental_fock_matches_full_rebuild_*` in `crates/ferric-scf/src/rhf.rs`).

## Options that change the equation {#direct-and-incremental-options-change-equation}

| Option | Effect |
|---|---|
| `[scf] integral_thresh` | $\tau_{\rm int}$ |
| `[scf] screening = "csb"` | the tighter rigorous bound $\min(Q_{12}Q_{34}, M_{13}M_{24}, M_{14}M_{23})$ |
| `[scf] screening = "csam"` | the Schwarz product times $\max(X_{13}X_{24}, X_{14}X_{23})^{1/2}$; an estimate, not a bound |
| `[scf] eri = "incore"` / `"auto"` | stores the integrals once and builds $J$, $K$ by matrix products (`crates/ferric-scf/src/incore_jk.rs`) |
| `[scf] gc_rotation` (or `FERRIC_GC_ROTATE`) = `auto` (default) / `rot1` / `off` | the internal contraction rotation of generally contracted bases (@sec:direct-and-incremental-general-contraction); `off` is the unrotated builder |
| `[scf] jk_digest` (or `FERRIC_JK_DIGEST`) = `pair` (default) / `legacy` | how each computed block is contracted into $J$ and $K$ (@sec:direct-and-incremental-pair-digest); `legacy` is the 8-fold scatter bit for bit |

## Sources

| Source | Used for |
|---|---|
| `scripts/scf_jk_builders_sympy.py` | verified equation @eq:scf-incremental-jk |
| `crates/ferric-scf/src/quartet_scatter.rs` | the loop, the screen, the scatter |
| `crates/ferric-scf/src/direct_jk.rs` | the incremental gates and the open-shell measurement note |
| `crates/ferric-integrals/src/contraction_rotation.rs`, `crates/ferric-scf/src/direct_jk/gc_rotation.rs` | the contraction rotation and its use in the direct builder |
| `crates/ferric-scf/tests/gc_rotation.rs` | rotated against unrotated $J$, $K$ and energies; bitwise checks |
