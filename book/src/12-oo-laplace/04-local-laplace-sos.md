# 12.4 Localized Laplace SOS-MP2

<!-- section: 12.4; status: gap; registry: laplace-sos-mp2;
     theory: crates/ferric-mp2/theory.md#"8. Laplace-transform MP2" -->

*Notation follows `book/notation.md`. $t$ is the Laplace variable. The
pseudo-densities are underlined as in section 12.3.*

The SOS energy of section 12.3 is invariant to orthogonal rotations among
the occupied orbitals and among the virtual orbitals. Writing it in
localized orbitals changes nothing until terms are dropped. Two
formulations drop terms by distance:

- `ao-sparse` cuts the pseudo-density by distance. The cut changes the
  energy but not the cost.
- `local` drops whole (occupied, virtual) pairs. That sets the cost.

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

## 12.4.1 The rotated energy

Let $L = C_oU$ be the Boys orbitals, with $U = C_o^{\mathsf T}SL$, and let
$C_{vl} = C_vR$ be the VV-HV virtuals, with $R = C_v^{\mathsf T}SC_{vl}$.
Both $U$ and $R$ are orthogonal. Define

$$ E(t) = U^{\mathsf T}\operatorname{diag}(e^{+t\varepsilon_o})\,U, \qquad Q(t) = R^{\mathsf T}\operatorname{diag}(e^{-t\varepsilon_v})\,R . $$

With $\hat H^P_{ia}$ the metric-whitened integral $(P|ia)$ in the localized
orbitals,

$$ Z_{PQ}(t) = \sum_{ij}\sum_{ab} E_{ij}(t)\,\hat H^P_{ia}\,Q_{ab}(t)\,\hat H^Q_{jb}, \qquad E_{\mathrm{OS}} \approx -\sum_k w_k\,\lVert Z(t_k)\rVert_F^2 . \tag{12.13} $$

Here $Z$ is the $J$ of (12.10) written in the rotated bases, so (12.13) is
exact before any truncation. `rotated_exp` forms $E(t)$ and $Q(t)$, and
`point_energy` forms $\lVert Z\rVert_F^2$.

> **Unverified equation 12.13** · the SOS energy in localized orbitals
>
> For orthogonal $U$, $R$ and no truncation, (12.13) equals the
> opposite-spin term of (12.10).
>
> | | |
> |---|---|
> | Proof | none committed. To add: a claim in `scripts/mp2_variant_identities.py` that rotates the model of claim 3 by random orthogonal $U$, $R$ and checks $\lVert Z\rVert_F^2 = \sum J^2$; negative control: the scalar weight $e^{t\varepsilon_i}$ paired with localized coefficients (the fixed 2026-07-27 defect of `ao-sparse`) must differ. |
> | Code | `ferric-mp2/src/laplace_local.rs::point_energy`, `ferric-mp2/src/laplace_local.rs::rotated_exp` |
> | Code test | `sos_local_infinite_cutoff_reproduces_mo` in `crates/ferric-mp2/src/laplace.rs`: vacuous radii reproduce `compute_sos_mo` on water/6-31G, butane/STO-3G and butane/6-31G with and without frozen core, bar $10^{-9}$, and every $(i,a)$ is kept |
>
> <!-- unverified: id=laplace.local.rotated; todo=add rotated-basis claim to scripts/mp2_variant_identities.py; rust_test=ferric-mp2/src/laplace.rs::sos_local_infinite_cutoff_reproduces_mo -->

## 12.4.2 The truncations of `local`

Each truncation is exact as its threshold goes to its vacuous value.

| Label | What is kept | Default from `LocalRadii::from_cutoff(r)` | Sets |
|---|---|---|---|
| T1 `occ` | AO shells within `occ` Bohr of LMO $i$'s centroid, in $L_{\mu i}$ | $\max(r, 12)$ (`OCC_FLOOR`) | integral work only |
| TA `pair_tau` | the AO index $\lambda$ of $X^P_{i\lambda} = \sum_\mu L_{\mu i}(P\vert\mu\lambda)$ for shells Schwarz-significant with a shell of LMO $i$'s support | $10^{-10}$ (`PAIR_TAU`) | integral work only |
| T2 `vir` | $V_i$: LVOs with centroid within `vir` Bohr of LMO $i$'s centroid | $r$ | $\lvert T\rvert$, the cost and the memory |
| T3 | $E_{ij}$ when $V_i$ and $V_j$ share an LVO | follows from T2 | — |
| TE `energy_tau` | $(i,a)$ with $(ia\vert ia)/(f_{aa} - f_{ii}) \ge$ `energy_tau` | 0 (off); library only | $\lvert T\rvert$ |

Two notes on the table:

- **TA is not a radius, deliberately.** The module documentation of
  `crates/ferric-mp2/src/laplace_local.rs` records an earlier version that
  cut the AO index $\lambda$ by distance. That cut breaks $\langle i|a\rangle = 0$.
  The fitted $(ia)$ density then carries a charge, and the $1/R$ tail turns
  that charge into an overestimate of $\lvert E_{\mathrm{OS}}\rvert$. The
  same failure appears when the virtual side is kept in AOs. `local`
  therefore drops only whole $(i,a)$ pairs, each a chargeless density.
- **Cost.** Per point, the code does one
  $N_{\mathrm{aux}}\times\lvert T\rvert\times N_{\mathrm{aux}}$ GEMM for $Z$.
  The whitening $\hat H = L_V^{-1}H$ is done once, outside the point loop.
  Memory is $N_{\mathrm{aux}}\lvert T\rvert$. The global Coulomb metric keeps
  the $N_{\mathrm{aux}}^2$ factor dense, so the cost stays cubic. Boys and
  VV-HV are dense $N^3$-class steps.

## 12.4.3 The `ao-sparse` formulation

`ao-sparse` (`compute_sos_ao` with a cutoff) works as follows:

- **Domains.** It Boys-localizes the occupied orbitals. Each LMO gets a
  domain: the AO shells whose atom lies within $r$ of its centroid
  (`ferric-mp2/src/boys.rs::build_domains`).
- **The occupied pseudo-density.** It is built with the matrix exponential
  of the localized occupied Fock block,
  $\underline P(t) = L\,e^{+tF_{\mathrm{loc}}}L^{\mathsf T}$. An element
  $\underline P_{\mu\nu}$ is kept only when $\mu$ and $\nu$ share a domain
  (`build_pseudo_density_occ_sparse`).
- **The virtual pseudo-density** is restricted to the union of the domains.
- **Cost.** Every tensor stays $N_{\mathrm{aux}}N^2$, so the cutoff changes
  the energy and not the cost.

The doc comment on `build_pseudo_density_occ_sparse` records a defect fixed
on 2026-07-27. The earlier code paired canonical coefficients and the scalar
$e^{t\varepsilon_i}$ with the domain of the $i$-th Boys orbital. Those two
index sets do not correspond, so the code built a different matrix, not a
truncated one. Exactness then required every domain to cover the molecule.
`localized_pseudo_density_matches_canonical_when_full` in
`crates/ferric-mp2/src/boys.rs` now asserts that full domains reproduce the
canonical $\underline P(t)$.

## 12.4.4 The kernel

From `ferric-mp2/src/laplace_local.rs::point_energy`. The inner loop
assembles $W = \hat H\,(E\otimes Q)^{\mathsf T}$ one block pair at a time,
over the domain neighbours $j$ of each $i$:

Abridged from `ferric-mp2/src/laplace_local.rs::point_energy` (not a verbatim excerpt):
```rust
let eij = e[(i, j)];
let g = Array2::from_shape_fn((vj.len(), vi.len()), |(x, y)| eij * q[(vj[x], vi[y])]);
let hj = h_hat.slice(s![.., dom.t_off[j]..dom.t_off[j + 1]]);
let mut wi = w.slice_mut(s![.., ci0..ci1]);
general_mat_mul(1.0, &hj, &g, 1.0, &mut wi);
// ...
let z = h_hat.dot(&w.t());
z.iter().map(|&x| x * x).sum()
```

| Code | Equation |
|---|---|
| `e`, `q` | $E(t)$, $Q(t)$ |
| `h_hat` | $\hat H$, columns ordered by $T$ |
| `vi`, `vj` | $V_i$, $V_j$ |
| `dom.nbrs[i]` | the $j$ kept by T3 |
| `z` | $Z(t)$ of (12.13) |

The caller multiplies by $-w_k$ and sums over points in parallel.

## 12.4.5 Anchors and measurements

| Check | Result | Test |
|---|---|---|
| `local`, vacuous radii | reproduces `mo`, bar $10^{-9}$ | `sos_local_infinite_cutoff_reproduces_mo` |
| `local`, energy cut at $10^{-300}$ | keeps every column, reproduces `mo` at $10^{-9}$; $10^{-4}$ drops columns and changes the energy | `sos_local_energy_cut_trivial_limit_and_teeth` |
| `local` on octane/6-31G at $r$ = 4, 6, 9 Bohr | each radius truncates, $\lvert T\rvert$ grows with $r$, the error shrinks | `sos_local_radius_restricts_cost_and_converges` |
| `ao-sparse`, vacuous cutoff | reproduces `compute_sos_mo`, across the AO/MO construction boundary, on water and butane | `sos_ao_sparse_infinite_cutoff_reproduces_mo` |
| `ao-sparse` at 12 Bohr, STO-3G | butane relative error below $10^{-9}$, octane below $10^{-6}$ | `sos_ao_sparse_truncation_radius_is_transferable_across_sizes` |

The alkane sweeps of both formulations, with errors, $\lvert T\rvert$ and
timings at C20 to C48, are on `site/src/methods/mp2.md` ("Laplace
formulations"). They are single runs and no figure from them is restated
here. The validation page caps both formulations to not graded.

**Known limits (2026-11-15).**

- No derivation or sympy claim covers (12.13) or the truncation rules. This
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
| `crates/ferric-mp2/src/laplace_local.rs`, module documentation and `LocalRadii` | equation 12.13, the truncation table, the failed constructions |
| `crates/ferric-mp2/src/boys.rs`, `build_pseudo_density_occ_sparse` | the `ao-sparse` construction and its fixed defect |
| `crates/ferric-mp2/src/laplace.rs`, tests listed above | the anchors |
| `site/src/methods/mp2.md`, "Laplace formulations" | where the measurements live |

Changes from the sources: `crates/ferric-mp2/theory.md` section 8 lists
`Mo`, `Ao` and `AoSparse` only. This section adds `Local` from the code. The
module documentation writes the energy as
$\lVert L_V^{-1}\tilde Z L_V^{-\mathsf T}\rVert_F^2$ with unwhitened
integrals; the code whitens the integrals first. The two are the same
number.
