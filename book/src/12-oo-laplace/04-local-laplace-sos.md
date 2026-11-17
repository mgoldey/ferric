# Localized Laplace SOS-MP2 {#local-laplace-sos}

<!-- status: gap; registry: laplace-sos-mp2 -->

*Notation follows `book/notation.md`. $t$ is the Laplace variable. The
pseudo-densities are underlined as in section @sec:laplace-mp2.*

Dynamic correlation is short-ranged. Two electrons on opposite ends of a
long alkane barely correlate, and an occupied orbital on one bond barely
excites into a virtual orbital on a distant bond. Canonical orbitals hide
this, because each spreads over the whole molecule. Localized orbitals
expose it: in a localized basis the integral $(ia|P)$ for an orbital $i$
and a distant virtual $a$ is small, and so is its contribution to
$E_{\mathrm{OS}}$. A method that drops those contributions does less work
for the same answer, up to a controlled error.

The SOS energy of section @sec:laplace-mp2 is invariant to orthogonal rotations among
the occupied orbitals and among the virtual orbitals. Writing it in
localized orbitals therefore changes nothing until terms are dropped. Two
formulations drop terms by distance:

- `ao-sparse` cuts the pseudo-density by distance. The cut changes the
  energy but not the cost.
- `local` drops whole (occupied, virtual) pairs. That sets the cost.

A distance cut does nothing until the molecule is longer than the cutoff.
If every localized virtual lies within the radius of every localized
occupied orbital, every pair is kept and the cost is that of the MO path.
Savings, and errors, appear only past that size.

Both are approximations. A run's certificate marks them not graded.

| Index | Range | Code name |
|---|---|---|
| $i, j$ | Boys-localized active occupied orbitals (LMOs) | `n_lmo` |
| $a, b$ | VV-HV localized virtual orbitals (LVOs) | `n_lvo` |
| $V_i$ | virtual domain of LMO $i$ | `LocalDomains::vir[i]` |
| $T = \{(i,a): a \in V_i\}$ | kept pairs | `t_size` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/laplace.rs::laplace_sos_mp2_with_basis` with `SosFormulation::AoSparse(r)` or `SosFormulation::Local(r)`; kernel `ferric-mp2/src/laplace_local.rs::sos_local_energy` |
| CLI | `method.kind = "laplace-sos-mp2"`, `[mp2] sos_formulation = "ao-sparse"` or `"local"`, `domain_cutoff_bohr` (Bohr, required) |
| Python | `run_laplace_sos_mp2(..., formulation="ao-sparse" or "local", domain_cutoff_bohr=r)` |

The formulation string is parsed strictly by `SosFormulation::parse_config_str`:

- `"ao-sparse"` and `"local"` require a cutoff, which must be finite and
  positive.
- A cutoff passed with `"mo"` or `"ao"` is an error.
- An unknown string is an error.
- `local` also needs the orbital `BasisSet`, to build the localized
  virtuals; `laplace_sos_mp2` without it refuses `local`.

## The rotated energy {#local-laplace-sos-rotated-energy}

Let $L = C_oU$ be the Boys orbitals, with $U = C_o^{\mathsf T}SL$, and let
$C_{vl} = C_vR$ be the VV-HV virtuals, with $R = C_v^{\mathsf T}SC_{vl}$.
Both $U$ and $R$ are orthogonal: the localized orbitals span exactly the
canonical occupied and virtual spaces.

Start from $J_{PQ}(t) = \sum_{ia}B^P_{ia}e^{t\varepsilon_i}e^{-t\varepsilon_a}B^Q_{ia}$
of section @sec:laplace-mp2-factorized-energy. Invert the two rotations: $C_o = LU^{\mathsf T}$ and
$C_v = C_{vl}R^{\mathsf T}$, so a canonical factor is a rotated localized
one, $B^P_{ia} = \sum_{i'a'}U_{ii'}R_{aa'}\hat H^P_{i'a'}$. Substitute and
move the sums over the canonical $i$ and $a$ inside. Each meets only its
two rotation coefficients and its exponential, and becomes a matrix in the
localized indices:

$$ E(t) = U^{\mathsf T}\operatorname{diag}(e^{+t\varepsilon_o})\,U, \qquad Q(t) = R^{\mathsf T}\operatorname{diag}(e^{-t\varepsilon_v})\,R . $$

These are the pseudo-densities of @sec:laplace-mp2-ao-form-coulomb-term written in the localized bases.
With $\hat H^P_{ia}$ the metric-whitened integral $(P|ia)$ in the localized
orbitals,

$$ Z_{PQ}(t) = \sum_{ij}\sum_{ab} E_{ij}(t)\,\hat H^P_{ia}\,Q_{ab}(t)\,\hat H^Q_{jb}, \qquad E_{\mathrm{OS}} \approx -\sum_k w_k\,\lVert Z(t_k)\rVert_F^2 . \label{eq:laplace-local-rotated} $$

$Z$ is the $J$ of (@eq:laplace-factorization) written in the rotated bases, element by element,
so (@eq:laplace-local-rotated) is exact before any truncation. `rotated_exp` forms $E(t)$ and
$Q(t)$, and `point_energy` forms $\lVert Z\rVert_F^2$.

What changed is the shape. In canonical orbitals the exponential weight is
a scalar per orbital. In localized orbitals it is a matrix, $E_{ij}(t)$ and
$Q_{ab}(t)$, that couples localized orbitals to each other. A localized
orbital is a mixture of canonical orbitals with different energies, so it
has no single energy to put in the exponential. Pairing a scalar
$e^{t\varepsilon_i}$ with a localized orbital is therefore wrong, and the
box below names it as the negative control.

> **Unverified equation @eq:laplace-local-rotated** · the SOS energy in localized orbitals
>
> For orthogonal $U$, $R$ and no truncation, (@eq:laplace-local-rotated) equals the
> opposite-spin term of (@eq:laplace-factorization).
>
> | | |
> |---|---|
> | Proof | none committed. To add: a claim in `scripts/mp2_variant_identities.py` that rotates the model of claim 3 by random orthogonal $U$, $R$ and checks $\lVert Z\rVert_F^2 = \sum J^2$; negative control: the scalar weight $e^{t\varepsilon_i}$ paired with localized coefficients (the fixed 2026-07-27 defect of `ao-sparse`) must differ. |
> | Code | `ferric-mp2/src/laplace_local.rs::point_energy`, `ferric-mp2/src/laplace_local.rs::rotated_exp` |
> | Code test | `sos_local_infinite_cutoff_reproduces_mo` in `crates/ferric-mp2/src/laplace.rs`: vacuous radii reproduce `compute_sos_mo` on water/6-31G, butane/STO-3G and butane/6-31G with and without frozen core, bar $10^{-9}$, and every $(i,a)$ is kept |
>
> <!-- unverified: id=laplace.local.rotated; todo=add rotated-basis claim to scripts/mp2_variant_identities.py; rust_test=ferric-mp2/src/laplace.rs::sos_local_infinite_cutoff_reproduces_mo -->

The substitution above was checked numerically for this section, on a
random model with random orthogonal $U$ and $R$ ($Z = J$ elementwise to
rounding), and the scalar-weight control differs. That check is not
committed, so the box stays unverified. The Rust test is the exactness
anchor: with vacuous radii the production code must give the `mo` energy.

## The truncations of `local` {#local-laplace-sos-truncations}

Each truncation is exact as its threshold goes to its vacuous value.

| Label | What is kept | Default from `LocalRadii::from_cutoff(r)` | Sets |
|---|---|---|---|
| T1 `occ` | AO shells within `occ` Bohr of LMO $i$'s centroid, in $L_{\mu i}$ | $\max(r, 12)$ (`OCC_FLOOR`) | integral work only |
| TA `pair_tau` | the AO index $\lambda$ of $X^P_{i\lambda} = \sum_\mu L_{\mu i}(P\vert\mu\lambda)$ for shells Schwarz-significant with a shell of LMO $i$'s support | $10^{-10}$ (`PAIR_TAU`) | integral work only |
| T2 `vir` | $V_i$: LVOs with centroid within `vir` Bohr of LMO $i$'s centroid | $r$ | $\lvert T\rvert$, the cost and the memory |
| T3 | $E_{ij}$ when $V_i$ and $V_j$ share an LVO | follows from T2 | — |
| TE `energy_tau` | $(i,a)$ with $(ia\vert ia)/(f_{aa} - f_{ii}) \ge$ `energy_tau` | 0 (off); library only | $\lvert T\rvert$ |

Why T3 follows from T2: in (@eq:laplace-local-rotated), $E_{ij}$ multiplies
$\hat H_{ia}Q_{ab}\hat H_{jb}$ with $a \in V_i$ and $b \in V_j$. The code
keeps $E_{ij}$ for the pairs whose domains overlap, the neighbour lists
`dom.nbrs[i]`.

Two notes on the table:

- **TA is a significance rule, not a radius.** Cutting the AO index $\lambda$ by
  distance breaks $\langle i|a\rangle = 0$: the truncated orbital is no
  longer orthogonal to the virtuals. The fitted $(ia)$ density then carries
  a net charge, and the $1/R$ tail of the Coulomb operator turns that charge
  into an overestimate of $\lvert E_{\mathrm{OS}}\rvert$. The module
  documentation of `crates/ferric-mp2/src/laplace_local.rs` records this
  for a radius cut on $\lambda$, and the same failure when the virtual side
  is kept in AOs. An exact orbital pair $(i,a)$ is orthogonal, so
  $\int\phi_i\phi_a = 0$ and the pair density has no charge. `local`
  therefore drops only whole $(i,a)$ pairs.
- **Cost.** Per point, the code does one
  $N_{\mathrm{aux}}\times\lvert T\rvert\times N_{\mathrm{aux}}$ GEMM for $Z$.
  The whitening $\hat H = L_V^{-1}H$ is done once, outside the point loop.
  Memory is $N_{\mathrm{aux}}\lvert T\rvert$. At a fixed radius
  $\lvert T\rvert = \sum_i\lvert V_i\rvert$ grows linearly with system size, against
  $n_on_v$ for the MO path. The global Coulomb metric keeps the
  $N_{\mathrm{aux}}^2$ factor dense, so the cost stays cubic. Boys and VV-HV
  are dense $N^3$-class steps.

## The `ao-sparse` formulation {#local-laplace-sos-formulation}

`ao-sparse` (`compute_sos_ao` with a cutoff) works as follows:

- **Domains.** It Boys-localizes the occupied orbitals. Each LMO gets a
  domain: the AO shells whose atom lies within $r$ of its centroid
  (`ferric-mp2/src/boys.rs::build_domains`).
- **The occupied pseudo-density.** In localized orbitals the exponential
  weight is the matrix of @sec:local-laplace-sos-rotated-energy. The code builds it as the matrix
  exponential of the localized occupied Fock block,
  $\underline P(t) = L\,e^{+tF_{\mathrm{loc}}}L^{\mathsf T}$. An element
  $\underline P_{\mu\nu}$ is kept only when $\mu$ and $\nu$ share a domain
  (`build_pseudo_density_occ_sparse`).
- **The virtual pseudo-density** is restricted to the union of the domains.
- **Cost.** Every tensor stays $N_{\mathrm{aux}}N^2$, so the cutoff changes
  the energy and not the cost.

Exactness depends on the matrix weight. The doc comment on
`build_pseudo_density_occ_sparse` records a defect fixed on 2026-07-27: the
code paired canonical coefficients and the scalar $e^{t\varepsilon_i}$ with
the domain of the $i$-th Boys orbital. Those two index sets do not
correspond, so the code built a different matrix, not a truncated one, and
exactness required every domain to cover the molecule.
`localized_pseudo_density_matches_canonical_when_full` in
`crates/ferric-mp2/src/boys.rs` asserts that full domains reproduce the
canonical $\underline P(t)$.

## The kernel {#local-laplace-sos-kernel}

`ferric-mp2/src/laplace_local.rs::point_energy`, shown whole, with comments
added. The double loop assembles $W = \hat H\,(E\otimes Q)^{\mathsf T}$ one
block pair at a time, over the domain neighbours $j$ of each $i$; the last
two lines form $Z$ and its squared norm:

```rust
fn point_energy(h_hat: &Array2<f64>, dom: &LocalDomains, e: &Array2<f64>, q: &Array2<f64>) -> f64 {
    let mut w = Array2::<f64>::zeros(h_hat.dim());
    for (i, vi) in dom.vir.iter().enumerate() {            // LMO i, its domain V_i
        let (ci0, ci1) = (dom.t_off[i], dom.t_off[i + 1]); // columns of (i, a in V_i)
        for &j in &dom.nbrs[i] {                           // j kept by T3
            let vj = &dom.vir[j];
            let eij = e[(i, j)];                           // E_ij(t)
            let g = Array2::from_shape_fn((vj.len(), vi.len()), |(x, y)| eij * q[(vj[x], vi[y])]); // E_ij Q_ba
            let hj = h_hat.slice(s![.., dom.t_off[j]..dom.t_off[j + 1]]); // H^Q_jb, b in V_j
            let mut wi = w.slice_mut(s![.., ci0..ci1]);
            general_mat_mul(1.0, &hj, &g, 1.0, &mut wi);  // W^Q_ia += sum_b H^Q_jb E_ij Q_ba
        }
    }
    let z = h_hat.dot(&w.t());                             // Z_PQ = sum_ia H^P_ia W^Q_ia
    z.iter().map(|&x| x * x).sum()                         // ||Z||_F^2
}
```

| Code | Equation |
|---|---|
| `e`, `q` | $E(t)$, $Q(t)$ |
| `h_hat` | $\hat H$, columns ordered by $T$ |
| `vi`, `vj` | $V_i$, $V_j$ |
| `dom.nbrs[i]` | the $j$ kept by T3 |
| `z` | $Z(t)$ of (@eq:laplace-local-rotated) |

Three points about it:

- **The unit of work** is one block pair $(i,j)$: a GEMM of
  $\lvert V_j\rvert$ columns of $\hat H$ against a
  $\lvert V_j\rvert\times\lvert V_i\rvert$ block of $Q$ scaled by $E_{ij}$.
- **The truncation is the loop bounds.** Dropped pairs never appear as
  columns of `h_hat`; nothing is computed and then zeroed.
- **The final GEMM** is the $N_{\mathrm{aux}}\times\lvert T\rvert\times N_{\mathrm{aux}}$
  product of @sec:local-laplace-sos-truncations, and it sets the per-point cost.

The caller multiplies by $-w_k$ and sums over points in parallel.

## Anchors and measurements {#local-laplace-sos-anchors-measurements}

| Check | Result | Test |
|---|---|---|
| `local`, vacuous radii | reproduces `mo`, bar $10^{-9}$ | `sos_local_infinite_cutoff_reproduces_mo` |
| `local`, energy cut at $10^{-300}$ | keeps every column, reproduces `mo` at $10^{-9}$; $10^{-4}$ drops columns and changes the energy | `sos_local_energy_cut_trivial_limit_and_teeth` |
| `local` on octane/6-31G at $r$ = 4, 6, 9 Bohr | each radius truncates, $\lvert T\rvert$ grows with $r$, the error shrinks | `sos_local_radius_restricts_cost_and_converges` |
| `ao-sparse`, vacuous cutoff | reproduces `compute_sos_mo`, across the AO/MO construction boundary, on water and butane | `sos_ao_sparse_infinite_cutoff_reproduces_mo` |
| `ao-sparse` at 12 Bohr, STO-3G | butane relative error below $10^{-9}$, octane below $10^{-6}$ | `sos_ao_sparse_truncation_radius_is_transferable_across_sizes` |

The first, second and fourth rows are the exactness anchors: each
truncation at its vacuous value must give the `mo` energy. The fourth is
the stronger test, because it compares two independent constructions, an
AO pseudo-density trace and an MO Gram product. The second row also shows
that the energy cut has an effect when it is not vacuous, so the anchor can
fail.

The alkane sweeps of both formulations, with errors, $\lvert T\rvert$ and
timings at C20 to C48, are on `site/src/methods/mp2.md` ("Laplace
formulations"). They are single runs and no figure from them is restated
here. The validation page caps both formulations to not graded.

**Known limits (2026-11-15).**

- No derivation or sympy claim covers (@eq:laplace-local-rotated) or the truncation rules. This
  section's status is gap for that reason.
- The global Coulomb metric keeps the cost cubic; there are no local fits
  and no local metric.
- Boys localization runs at most 200 sweeps. `local` records
  `boys_converged` in its statistics but does not refuse an unconverged
  localization. `ao-sparse` does not check it either. An unconverged
  rotation is still orthogonal, so the vacuous limit stays exact, but the
  truncation then acts on less local orbitals.
- `energy_tau` (TE) is reachable from Rust only.
- Closed shell, energy only.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/laplace_local.rs`, module documentation and `LocalRadii` | equation @eq:laplace-local-rotated, the truncation table, the failed constructions |
| `crates/ferric-mp2/src/boys.rs`, `build_pseudo_density_occ_sparse` | the `ao-sparse` construction and its fixed defect |
| `crates/ferric-mp2/src/laplace.rs`, tests listed above | the anchors |
| `site/src/methods/mp2.md`, "Laplace formulations" | where the measurements live |

Changes from the sources: `crates/ferric-mp2/theory.md` section 8 lists
`Mo`, `Ao` and `AoSparse` only. This section adds `Local` from the code. The
module documentation writes the energy as
$\lVert L_V^{-1}\tilde Z L_V^{-\mathsf T}\rVert_F^2$ with unwhitened
integrals; the code whitens the integrals first. The two are the same
number. The substitution that derives (@eq:laplace-local-rotated) in @sec:local-laplace-sos-rotated-energy is written out here
and was checked numerically; it is not in a committed script.
