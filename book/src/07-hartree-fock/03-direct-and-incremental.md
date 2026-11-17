# Direct and incremental J/K builds {#direct-and-incremental}

<!-- status: ready; registry: rhf, uhf, rohf -->

$J$ and $K$ need the four-centre integrals $(\mu\nu|\lambda\sigma)$, about
$N^4/8$ distinct ones, which for a few hundred basis functions is already too
many to store. The default route therefore computes the integrals every
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
| $Q_{PQ}$ | Schwarz factor of a shell pair | `q_table[(s1, s2)]` |
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

$$ Q_{PQ} = \max_{a\in P,\,b\in Q}\sqrt{|(ab|ab)|}, $$

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
$\min(Q_{12}Q_{34}, M_{13}M_{24}, M_{14}M_{23})$, which is never looser than
the Schwarz product; `"csam"` is not a rigorous bound.

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
convergence target:

- the direct route (no density fitting, no pluggable K, no in-core store);
- `energy_conv` $\ge 10^4\,\tau_{\rm int}$ and `density_conv` $\ge 10^3\,\tau_{\rm int}$
  (`closed_shell_incremental_enabled` in `crates/ferric-scf/src/direct_jk.rs`;
  the environment variable `FERRIC_SCF_INCREMENTAL` forces it on or off);
- the builder has no in-core store (`DirectJK::incremental_pays`).

The open-shell loops build increments only under
`FERRIC_SCF_UHF_INCREMENTAL=1`.

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

## When the incremental build runs {#direct-and-incremental-incremental-build-runs}

The incremental build is closed-shell, direct-only and conditional:

| Condition | Source |
|---|---|
| no density fitting, no pluggable K builder | the route decision in `ferric-scf/src/fock_assembly.rs::resolve_jk_route` |
| no in-core integral store | `ferric-scf/src/direct_jk.rs::DirectJK` (`incremental_pays`) |
| `energy_conv` $\ge 10^4\,\tau_{\rm int}$ and `density_conv` $\ge 10^3\,\tau_{\rm int}$ | `ferric-scf/src/direct_jk.rs::closed_shell_incremental_enabled` |
| open shell | only with `FERRIC_SCF_UHF_INCREMENTAL=1` |

The full rebuild every 8 iterations is a function-local constant of
`solve_rhf`; nothing else, including a DIIS reset, forces one.

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

## Numerical anchors {#direct-and-incremental-numerical-anchors}

The direct builder is the exact reference against which every other route in
this chapter is measured. Its own anchors are the energy anchors of section
@sec:energy-functionals (1.8e-12 Ha RHF, 4.0e-12 Ha UHF and ROHF against PySCF).

<!-- figures: site/src/reference/validation.md "UHF and ROHF energies, stability-checked" -->

**Open item.** The unit tests
`incremental_fock_matches_full_rebuild_water_ccpvdz` and
`incremental_fock_matches_full_rebuild_methane_ccpvdz` in
`crates/ferric-scf/src/rhf.rs` no longer exercise the incremental path. Since
the in-core store became the default for small bases, `incremental_pays()`
is false for water and methane in cc-pVDZ, so both tests compare the full
build with itself. They need the store disabled to test what their names say.

## Options that change the equation {#direct-and-incremental-options-change-equation}

| Option | Effect |
|---|---|
| `[scf] integral_thresh` | $\tau_{\rm int}$ |
| `[scf] screening = "csb"` | the tighter rigorous bound $\min(Q_{12}Q_{34}, M_{13}M_{24}, M_{14}M_{23})$ |
| `[scf] screening = "csam"` | a non-rigorous refinement of the Schwarz bound |
| `[scf] eri = "incore"` / `"auto"` | stores the integrals once and builds $J$, $K$ by matrix products (`crates/ferric-scf/src/incore_jk.rs`) |

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Direct and incremental J/K builds" | the equations |
| `scripts/scf_jk_builders_sympy.py` | verified equation @eq:scf-incremental-jk |
| `crates/ferric-scf/src/quartet_scatter.rs` | the loop, the screen, the scatter |
| `crates/ferric-scf/src/direct_jk.rs` | the incremental gates and the open-shell measurement note |
