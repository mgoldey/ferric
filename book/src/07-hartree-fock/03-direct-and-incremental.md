# 7.3 Direct and incremental J/K builds

<!-- section: 7.3; status: ready; registry: rhf, uhf, rohf;
     theory: crates/ferric-scf/theory.md#"Direct and incremental J/K builds" -->

The default Hartree–Fock route computes every needed four-centre integral in
each iteration and contracts it with the density at once. Nothing is fitted,
so $J$ and $K$ are exact to the screening threshold. This section states the
contraction, the screen and the incremental build.

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

## 7.3.1 The equations

<!-- include: crates/ferric-scf/theory.md#"Direct and incremental J/K builds" -->

> **Verified equation 7.3** · incremental builds
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

## 7.3.2 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | Schwarz table $Q$ once per geometry | `ferric-scf/src/screening.rs::SchwarzBounds` |
| 2 | shell-block maxima of $|D|$ (or $|\Delta D|$) | `ferric-scf/src/quartet_scatter.rs::DensityScreen` |
| 3 | loop over canonical quartets, skip below the bound | `ferric-scf/src/quartet_scatter.rs::scatter_bra_pair_probed` |
| 4 | one libint2 block per surviving quartet | `ferric-integrals/src/engine.rs::Engine` |
| 5 | scatter to every distinct image in $J$ and $K$ | `ferric-scf/src/quartet_scatter.rs::digest_j` and its siblings |

The quartet is canonical in the fourth index,

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

## 7.3.3 When the incremental build runs

The incremental build is closed-shell, direct-only and conditional:

| Condition | Source |
|---|---|
| no density fitting, no pluggable K builder | the route decision in `ferric-scf/src/fock_assembly.rs::resolve_jk_route` |
| no in-core integral store | `ferric-scf/src/direct_jk.rs::DirectJK` (`incremental_pays`) |
| `energy_conv` $\ge 10^4\,\tau_{\rm int}$ and `density_conv` $\ge 10^3\,\tau_{\rm int}$ | `ferric-scf/src/direct_jk.rs::closed_shell_incremental_enabled` |
| open shell | only with `FERRIC_SCF_UHF_INCREMENTAL=1` |

The full rebuild every 8 iterations is a function-local constant of
`solve_rhf`; nothing else, including a DIIS reset, forces one.

## 7.3.4 Reproducibility

The incremental build changes the floating-point order of the sum relative to
a full build: $J[D_{n-1}] + J[\Delta D_n]$ is not bitwise $J[D_n]$, and the
screen drops a different set of quartets on $\Delta D$ than on $D$. The two
agree to the screening threshold, and the rebuild every 8 iterations keeps the
screened increments from accumulating without bound.

## 7.3.5 Numerical anchors

The direct builder is the exact reference against which every other route in
this chapter is measured. Its own anchors are the energy anchors of section
7.1 (1.8e-12 Ha RHF, 4.0e-12 Ha UHF and ROHF against PySCF).

<!-- figures: site/src/reference/validation.md "UHF and ROHF energies, stability-checked" -->

**Open item.** The unit tests
`incremental_fock_matches_full_rebuild_water_ccpvdz` and
`incremental_fock_matches_full_rebuild_methane_ccpvdz` in
`crates/ferric-scf/src/rhf.rs` no longer exercise the incremental path. Since
the in-core store became the default for small bases, `incremental_pays()`
is false for water and methane in cc-pVDZ, so both tests compare the full
build with itself. They need the store disabled to test what their names say.

## 7.3.6 Options that change the equation

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
| `scripts/scf_jk_builders_sympy.py` | verified equation 7.3 |
| `crates/ferric-scf/src/quartet_scatter.rs` | the loop, the screen, the scatter |
