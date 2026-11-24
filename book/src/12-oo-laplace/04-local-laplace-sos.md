# Localized Laplace SOS-MP2 {#local-laplace-sos}

<!-- status: gap; registry: laplace-sos-mp2-local -->

*Notation follows chapter @sec:notation. $t$ is the Laplace variable. The
pseudo-densities are underlined as in section @sec:laplace-mp2.*

Dynamic correlation is short-ranged. Two electrons on opposite ends of a
long alkane barely correlate, and an occupied orbital on one bond barely
excites into a virtual orbital on a distant bond. Canonical orbitals hide
this, because each spreads over the whole molecule. Localized orbitals
expose it: in a localized basis the integral $(ia|P)$ for an orbital $i$
and a distant virtual $a$ is small, and so is its contribution to
$E_{\mathrm{OS}}$. A method that drops those contributions does less work
for the same answer, with an error that shrinks as the radius grows. The
error is measured, not bounded.

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

The two orbital spaces come from chapter @sec:local-mp2:

- **Boys orbitals.** The active occupied orbitals rotated to minimize the
  summed spread $\sum_i\langle i|r^2|i\rangle - \lvert\langle i|\mathbf r|i\rangle\rvert^2$,
  by Jacobi sweeps. Both formulations call plain Boys,
  `boys_localize_checked(.., 200)`, not the λ-Boys of
  section @sec:amplitude-lmp2-lambda-boys that chapter @sec:local-mp2 uses by default. A localization
  that has not converged in 200 sweeps is an error.
- **VV-HV virtuals.** An orthonormal localized basis of the canonical
  virtual space, built from valence virtuals (projected minimal-basis
  functions) and hard virtuals, as listed in section @sec:amplitude-lmp2-algorithm. Only
  `local` uses them.

| Index | Range | Code name |
|---|---|---|
| $i, j$ | Boys-localized active occupied orbitals (LMOs), plain Boys | `n_lmo` |
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

Let $C_{ol} = C_oU$ be the Boys orbitals, with $U = C_o^{\mathsf T}SC_{ol}$,
and let $C_{vl} = C_vR$ be the VV-HV virtuals, with
$R = C_v^{\mathsf T}SC_{vl}$. Both $U$ and $R$ are orthogonal: the
localized orbitals span exactly the canonical occupied and virtual spaces,
so $C_o = C_{ol}U^{\mathsf T}$ and $C_v = C_{vl}R^{\mathsf T}$.

Start from $J_{PQ}(t) = \sum_{ia}B^P_{ia}e^{t\varepsilon_i}e^{-t\varepsilon_a}B^Q_{ia}$
of section @sec:laplace-mp2-factorized-energy, with $B^P_{ia}$ the whitened factor in
canonical orbitals. Let $\hat B^P_{ia}$ be the same factor in the
localized orbitals. The derivation takes four steps.

1. **Back-rotate the factor.** $B^P_{ia}$ is linear in the orbital
   coefficients of $i$ and of $a$. Substituting $C_o = C_{ol}U^{\mathsf T}$
   and $C_v = C_{vl}R^{\mathsf T}$ gives
   $B^P_{ia} = \sum_{i'a'}U_{ii'}R_{aa'}\hat B^P_{i'a'}$.
2. **Substitute into $J$.** Each of the two factors brings its own pair of
   localized indices:
   $J_{PQ} = \sum_{ia}\sum_{i'a'j'b'}U_{ii'}R_{aa'}\hat B^P_{i'a'}\,e^{t\varepsilon_i}e^{-t\varepsilon_a}\,U_{ij'}R_{ab'}\hat B^Q_{j'b'}$.
3. **Move the canonical sums inside.** The sum over $i$ meets only
   $U_{ii'}U_{ij'}e^{t\varepsilon_i}$, and the sum over $a$ only
   $R_{aa'}R_{ab'}e^{-t\varepsilon_a}$. Each is a matrix in the localized
   indices:

   $$ \underline P^{\mathrm{loc}}(t) = U^{\mathsf T}\operatorname{diag}(e^{+t\varepsilon_o})\,U, \qquad \underline Q^{\mathrm{loc}}(t) = R^{\mathsf T}\operatorname{diag}(e^{-t\varepsilon_v})\,R . \label{eq:laplace-local-weights} $$

4. **Rename** the primed indices:

$$ Z_{PQ}(t) = \sum_{ij}\sum_{ab} \underline P^{\mathrm{loc}}_{ij}(t)\,\hat B^P_{ia}\,\underline Q^{\mathrm{loc}}_{ab}(t)\,\hat B^Q_{jb} = J_{PQ}(t), \qquad E_{\mathrm{OS}} \approx -\sum_k w_k\,\lVert Z(t_k)\rVert_F^2 . \label{eq:laplace-local-rotated} $$

$Z$ is $J$ element by element, so (@eq:laplace-local-rotated) is exact before any
truncation. The two matrices of (@eq:laplace-local-weights) are the pseudo-densities of
section @sec:laplace-mp2-ao-form-coulomb-term, written in the localized bases.

The whitening matrix may differ between the two sides. `local` whitens
with $M = L^{-1}$ of section @sec:four-step-algorithm, the inverse Cholesky
factor of the metric. Any $M$ with $MVM^{\mathsf T} = 1$ differs from
$V^{-1/2}$ by an orthogonal matrix on the auxiliary index, and
$\lVert Z\rVert_F$ does not see it. `rotated_exp` forms the two weights,
and `point_energy` forms $\lVert Z\rVert_F^2$.

What changed is the shape. In canonical orbitals the exponential weight is
a scalar per orbital. In localized orbitals it is a matrix that couples
localized orbitals to each other. A localized orbital is a mixture of
canonical orbitals with different energies, so it has no single energy to
put in the exponential. The matrix is an exponential of the localized Fock
block: with $F_{\mathrm{loc}} = U^{\mathsf T}\operatorname{diag}(\varepsilon_o)U$,
the columns of $U^{\mathsf T}$ are its eigenvectors with eigenvalues
$\varepsilon_k$, so

$$ \underline P^{\mathrm{loc}}(t) = e^{\,tF_{\mathrm{loc}}} . \label{eq:laplace-local-expm} $$

Pairing a scalar $e^{t\varepsilon_i}$, or $e^{tf_{ii}}$, with a localized
orbital drops the off-diagonal part of this matrix and changes the energy.
The box below names it as a control.

> **Verified equation @eq:laplace-local-rotated** · the SOS energy in localized orbitals
>
> For orthogonal $U$, $R$ and no truncation, (@eq:laplace-local-rotated) equals the
> opposite-spin term of (@eq:laplace-factorization), and the occupied weight is
> (@eq:laplace-local-expm). Symbolically for two occupied, two virtual and two
> auxiliary functions with rotations in Cayley form; numerically, the
> transcribed `point_energy` on Cholesky-whitened localized integrals
> equals $\lVert J\rVert_F^2$ from canonical orbitals whitened by
> $V^{-1/2}$, to $1.4\times10^{-15}$ relative.
>
> | | |
> |---|---|
> | Proof | `scripts/laplace_local_sympy.py`, checks 1 to 3 |
> | Negative control | the scalar weight $\operatorname{diag}(e^{tf_{ii}})$ in place of $\underline P^{\mathrm{loc}}(t)$ changes the energy by 5 % (check 4) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`; mutations: $R^{\mathsf T}DR \to RDR^{\mathsf T}$ in `rotated_exp` and in the symbolic $\underline Q^{\mathrm{loc}}$, partners restricted to $j = i$, whitening by $L^{\mathsf T}$ |
> | Code | `ferric-mp2/src/laplace_local.rs::point_energy`, `ferric-mp2/src/laplace_local.rs::rotated_exp` |
> | Code test | `sos_local_infinite_cutoff_reproduces_mo` in `crates/ferric-mp2/src/laplace.rs`: vacuous radii reproduce `compute_sos_mo` on water/6-31G, butane/STO-3G and butane/6-31G with and without frozen core, bar $10^{-9}$, and every $(i,a)$ is kept |
>
> <!-- verified: id=laplace.local.rotated; script=scripts/laplace_local_sympy.py; claim=rotated-equals-canonical; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace_local.rs::point_energy, ferric-mp2/src/laplace_local.rs::rotated_exp; rust_test=ferric-mp2/src/laplace.rs::sos_local_infinite_cutoff_reproduces_mo -->

The Rust test is the exactness anchor in production code: with vacuous
radii the code must give the `mo` energy. The same anchor from Python, on
water / 6-31G (the `3f612e0b3` build), at a 1000 Bohr radius:

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("6-31g")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

mo = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0, formulation="mo").e_os
for f in ("local", "ao-sparse"):
    r = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0, formulation=f, domain_cutoff_bohr=1.0e3)
    print(f"{f:9s} r=1000 Bohr reproduces mo:", abs(r.e_os - mo) < 1e-12)
```

```text
local     r=1000 Bohr reproduces mo: True
ao-sparse r=1000 Bohr reproduces mo: True
```

## The truncations of `local` {#local-laplace-sos-truncations}

Each truncation is exact at its vacuous value.

| Label | What is kept | Default from `LocalRadii::from_cutoff(r)` | Vacuous value | Sets |
|---|---|---|---|---|
| T1 `occ` | AO shells within `occ` Bohr of LMO $i$'s centroid, in the coefficients $(C_{ol})_{\mu i}$ | $\max(r, 12)$ (`OCC_FLOOR`) | larger than the molecule | integral work |
| TA `pair_tau` | the AO index $\lambda$ of $X^P_{i\lambda} = \sum_\mu (C_{ol})_{\mu i}(P\vert\mu\lambda)$ for shells Schwarz-significant with a shell of LMO $i$'s support | $10^{-10}$ (`PAIR_TAU`) | 0 | integral work |
| T2 `vir` | $V_i$: LVOs with centroid within `vir` Bohr of LMO $i$'s centroid | $r$ | larger than the molecule | $\lvert T\rvert$, the cost and the memory |
| T3 | $\underline P^{\mathrm{loc}}_{ij}$ only for partners: $V_i$ and $V_j$ share an LVO | tied to T2 | T2 vacuous | the pair loop |
| TE `energy_tau` | $(i,a)$ with $(ia\vert ia)/(f_{aa} - f_{ii}) \ge$ `energy_tau`, $f_{ii} = \sum_k U_{ki}^2\varepsilon_k$, $f_{aa} = \sum_b R_{ba}^2\varepsilon_b$ | 0 (off); library only | 0 | $\lvert T\rvert$ |

T3 is a truncation of its own, not a consequence of T2. In
(@eq:laplace-local-rotated), a non-partner $j$ of $i$ contributes
$\underline P^{\mathrm{loc}}_{ij}\sum_{a\in V_i}\sum_{b\in V_j}\hat B_{ia}\underline Q^{\mathrm{loc}}_{ab}\hat B_{jb}$
with $a \ne b$ throughout, since the domains share no LVO. Those terms
vanish only when $\underline Q^{\mathrm{loc}}(t)$ is diagonal, which it is
for canonical virtuals and is not for VV-HV virtuals. The code keeps
$\underline P^{\mathrm{loc}}_{ij}$ for partners only (the neighbour lists
`dom.nbrs[i]`), and so drops them. When T2 is vacuous every pair of
domains shares every LVO, and T3 drops nothing.
`scripts/laplace_local_sympy.py` checks both statements on its model
(check 6): with T2 finite, T3 moves the energy by $10^{-2}$ relative when
$R$ is a rotation, and by nothing when $R = 1$.

> **Verified table** · the truncations at their vacuous values
>
> Transcriptions of `occ_half_transform`, `build_local_domains`, `partners`
> and `energy_cut` reproduce the untruncated energy when each threshold is
> vacuous (T1 and TA with every shell kept; TE at $10^{-300}$), and each
> changes the energy when it is not (checks 5 to 8).
>
> | | |
> |---|---|
> | Proof | `scripts/laplace_local_sympy.py`, checks 5 to 8 |
> | Negative control | a finite radius or cut must change the energy; T3 with $\underline Q^{\mathrm{loc}}$ diagonal must not |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`; mutation: the TE denominator $f_{aa} - f_{ii} \to f_{ii} - f_{aa}$ |
> | Code | `ferric-mp2/src/laplace_local.rs::partners`, `ferric-mp2/src/laplace_local.rs::energy_cut` |
> | Code test | `sos_local_energy_cut_trivial_limit_and_teeth` and `sos_local_radius_restricts_cost_and_converges` in `crates/ferric-mp2/src/laplace.rs` |
>
> <!-- verified: id=laplace.local.truncations; script=scripts/laplace_local_sympy.py; claim=t3-is-an-extra-truncation; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/laplace_local.rs::partners, ferric-mp2/src/laplace_local.rs::energy_cut; rust_test=ferric-mp2/src/laplace.rs::sos_local_energy_cut_trivial_limit_and_teeth -->

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
  The whitening $\hat B = L^{-1}H$ is done once, outside the point loop, with
  $H$ the unwhitened localized integrals: one Cholesky of the metric and one
  triangular solve into $H$ (`ferric-mp2/src/rimp2.rs::cholesky_whiten`), no
  explicit inverse. At C48 / 6-31G (cc-pVDZ-RI, 4060 functions, `local` at
  8 Bohr) it takes 1.9 s of a 43 s SOS stage (2026-11-21). Memory is
  $N_{\mathrm{aux}}\lvert T\rvert$. At a fixed radius
  $\lvert T\rvert = \sum_i\lvert V_i\rvert$ grows linearly with system size, against
  $n_on_v$ for the MO path. The global Coulomb metric keeps the
  $N_{\mathrm{aux}}^2$ factor dense, so the cost stays cubic. Boys and VV-HV
  are dense $N^3$-class steps.

How large the error is at a given radius, on octane / 6-31G with all
electrons (the `3f612e0b3` build; the molecule is about 20 Bohr long):

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/alkane_8.xyz")
bs = ferric.BasisSet.bundled("6-31g")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")

mo = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0, frozen_core=0, formulation="mo").e_os
for r in (4.0, 6.0, 9.0):
    e = ferric.run_laplace_sos_mp2(
        mol, bs, aux, c_os=1.0, frozen_core=0, formulation="local", domain_cutoff_bohr=r
    ).e_os
    print(f"local r = {r:3.0f} Bohr: relative error {abs(e - mo) / abs(mo):.2e}")
```

```text
local r =   4 Bohr: relative error 4.68e-02
local r =   6 Bohr: relative error 6.87e-03
local r =   9 Bohr: relative error 3.21e-04
```

The statistics line the code prints to stderr gives $\lvert T\rvert$ = 613,
1204 and 1693 of $33 \times 75 = 2475$ pairs at the three radii.

## The `ao-sparse` formulation {#local-laplace-sos-formulation}

`ao-sparse` (`compute_sos_ao` with a cutoff) works as follows:

- **Domains.** It Boys-localizes the occupied orbitals. Each LMO gets a
  domain: the AO shells whose centre lies within $r$ of its centroid
  (`ferric-mp2/src/boys.rs::build_domains`).
- **The occupied pseudo-density.** By (@eq:laplace-local-expm), the canonical pseudo-density
  of (@eq:laplace-ao-trace) is, in the Boys orbitals,

  $$ \underline P(t) = C_o\operatorname{diag}(e^{t\varepsilon_o})C_o^{\mathsf T} = C_{ol}\,U^{\mathsf T}\operatorname{diag}(e^{t\varepsilon_o})U\,C_{ol}^{\mathsf T} = C_{ol}\,e^{\,tF_{\mathrm{loc}}}\,C_{ol}^{\mathsf T} . \label{eq:laplace-local-pseudo-density} $$

  The code forms the right-hand side with $F_{\mathrm{loc}} = C_{ol}^{\mathsf T}FC_{ol}$
  and keeps an element $\underline P_{\mu\nu}$ only when $\mu$ and $\nu$
  share a domain (`build_pseudo_density_occ_sparse`). Every kept element
  is the exact element; the mask acts on the final matrix.
- **The virtual pseudo-density** is the canonical
  $\underline Q(t)$ restricted to rows and columns in the union of the
  domains (`build_pseudo_density_vir_sparse`).
- **Cost.** Every tensor stays $N_{\mathrm{aux}}N^2$, so the cutoff changes
  the energy and not the cost.

The weight must be the matrix exponential. A scalar $e^{t\varepsilon_i}$
paired with canonical coefficients and the domain of the $i$-th Boys
orbital mixes two index sets and builds a different matrix, not a
truncated one; exactness would then need every domain to cover the
molecule. `localized_pseudo_density_matches_canonical_when_full` in
`crates/ferric-mp2/src/boys.rs` guards the construction with full domains.

> **Verified equation @eq:laplace-local-pseudo-density** · the masked localized pseudo-density
>
> With full domains, the transcribed `ao-sparse` energy equals the
> canonical one. At a finite cutoff every kept element of
> `build_pseudo_density_occ_sparse` equals the canonical
> $\underline P_{\mu\nu}(t)$ to $10^{-15}$ relative, and the cutoff changes
> the energy.
>
> | | |
> |---|---|
> | Proof | `scripts/laplace_local_sympy.py`, checks 2 and 9 to 11 |
> | Negative control | the per-orbital scalar construction (canonical $C$, $e^{t\varepsilon_i}$, domain of Boys orbital $i$) misses the kept elements by 87 % (check 10) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`; mutation: $e^{tF_{\mathrm{loc}}} \to \operatorname{diag}(e^{tf_{ii}})$ |
> | Code | `ferric-mp2/src/boys.rs::build_pseudo_density_occ_sparse`, `ferric-mp2/src/boys.rs::build_pseudo_density_vir_sparse` |
> | Code test | `localized_pseudo_density_matches_canonical_when_full` in `crates/ferric-mp2/src/boys.rs`; `sos_ao_sparse_infinite_cutoff_reproduces_mo` in `crates/ferric-mp2/src/laplace.rs` |
>
> <!-- verified: id=laplace.local.ao-sparse; script=scripts/laplace_local_sympy.py; claim=ao-sparse-retained-elements-exact; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/boys.rs::build_pseudo_density_occ_sparse, ferric-mp2/src/boys.rs::build_pseudo_density_vir_sparse; rust_test=ferric-mp2/src/boys.rs::localized_pseudo_density_matches_canonical_when_full -->

## The kernel {#local-laplace-sos-kernel}

`ferric-mp2/src/laplace_local.rs::point_energy`, shown whole, with comments
added. The double loop assembles
$W = \hat B\,(\underline P^{\mathrm{loc}}\otimes\underline Q^{\mathrm{loc}})^{\mathsf T}$
one block pair at a time, over the partners $j$ of each $i$; the last two
lines form $Z$ and its squared norm:

```rust
fn point_energy(h_hat: &Array2<f64>, dom: &LocalDomains, e: &Array2<f64>, q: &Array2<f64>) -> f64 {
    let mut w = Array2::<f64>::zeros(h_hat.dim());
    for (i, vi) in dom.vir.iter().enumerate() {            // LMO i, its domain V_i
        let (ci0, ci1) = (dom.t_off[i], dom.t_off[i + 1]); // columns of (i, a in V_i)
        for &j in &dom.nbrs[i] {                           // j kept by T3
            let vj = &dom.vir[j];
            let eij = e[(i, j)];                           // P^loc_ij(t)
            let g = Array2::from_shape_fn((vj.len(), vi.len()), |(x, y)| eij * q[(vj[x], vi[y])]); // P^loc_ij Q^loc_ba
            let hj = h_hat.slice(s![.., dom.t_off[j]..dom.t_off[j + 1]]); // B^Q_jb, b in V_j
            let mut wi = w.slice_mut(s![.., ci0..ci1]);
            general_mat_mul(1.0, &hj, &g, 1.0, &mut wi);  // W^Q_ia += sum_b B^Q_jb P^loc_ij Q^loc_ba
        }
    }
    let z = h_hat.dot(&w.t());                             // Z_PQ = sum_ia B^P_ia W^Q_ia
    z.iter().map(|&x| x * x).sum()                         // ||Z||_F^2
}
```

| Code | Equation |
|---|---|
| `e`, `q` | $\underline P^{\mathrm{loc}}(t)$, $\underline Q^{\mathrm{loc}}(t)$ |
| `h_hat` | $\hat B$, columns ordered by $T$ |
| `vi`, `vj` | $V_i$, $V_j$ |
| `dom.nbrs[i]` | the partners $j$ kept by T3 |
| `z` | $Z(t)$ of (@eq:laplace-local-rotated) |

Three points about it:

- **The unit of work** is one block pair $(i,j)$: a GEMM of
  $\lvert V_j\rvert$ columns of $\hat B$ against a
  $\lvert V_j\rvert\times\lvert V_i\rvert$ block of $\underline Q^{\mathrm{loc}}$
  scaled by $\underline P^{\mathrm{loc}}_{ij}$.
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
formulations"). They are single runs. The validation page caps both
formulations to not graded.

**Known limits (2026-11-19).**

- The error of a truncation is measured, not bounded. No estimate of it is
  computed during a run.
- The global Coulomb metric keeps the cost cubic; there are no local fits
  and no local metric.
- Both formulations use plain Boys, not λ-Boys.
- `energy_tau` (TE) is reachable from Rust only.
- Closed shell, energy only.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/laplace_local.rs`, module documentation, `LocalRadii` and `sos_local_energy` | equation @eq:laplace-local-rotated, the truncation table, the failed constructions |
| `crates/ferric-mp2/src/boys.rs`, `build_pseudo_density_occ_sparse` and `build_pseudo_density_vir_sparse` | the `ao-sparse` construction and its guard |
| `crates/ferric-mp2/src/laplace.rs`, tests listed above | the anchors |
| `scripts/laplace_local_sympy.py` | the proofs in the three boxes |
| `site/src/methods/mp2.md`, "Laplace formulations" | where the sweeps live |

Section @sec:laplace-mp2 lists `Mo`, `Ao` and `AoSparse` only; this section adds
`Local` from the code. The module documentation writes the energy as
$\lVert L_V^{-1}\tilde Z L_V^{-\mathsf T}\rVert_F^2$, with $L_V$ its name for the
Cholesky factor $L$ and unwhitened integrals; the
code whitens the integrals first. The two are the same number.
