# 13.1 Amplitude-threshold local MP2

<!-- section: 13.1; status: ready; registry: local-mp2;
     theory: crates/ferric-mp2/theory.md#"12.1 Amplitude equations in localized orbitals" -->

The MP2 amplitude $t_{ij}^{ab}$ measures how much two electrons in orbitals
$i$ and $j$ correlate by jumping together into $a$ and $b$. What drives the
jump is the integral $(ia|jb)$, the Coulomb interaction of the charge
distributions $\phi_i\phi_a$ and $\phi_j\phi_b$. In canonical orbitals
every orbital spreads over the whole molecule, so almost every $(ia|jb)$ is
nonzero and canonical MP2 has to sum all of them.

Localize the orbitals and that changes. Each $\phi_i$ now sits on a bond or
a lone pair, and so does each local virtual $\phi_a$. If $a$ is far from
$i$, the product $\phi_i\phi_a$ is close to zero everywhere, so $(ia|jb)$ is
too. In a long molecule most $(i,a)$ combinations are far apart, and most
amplitudes are negligible.

The amplitude-threshold scheme of Wang, Aldossary, Shi, Liu, Li and
Head-Gordon (J. Chem. Theory Comput. 19, 7577, 2023) acts on this directly.
It keeps an amplitude only when the integral that drives it exceeds one
threshold $\varepsilon$, and solves the MP2 equations for the kept set only.
Three facts frame the rest of the section:

- $\varepsilon = 0$ keeps everything and gives canonical RI-MP2 exactly.
- The energy error has one sign: the truncated energy is never below the
  canonical one (§13.1.1 shows why).
- `eps` has no default. It is part of the model, like the basis set.

Notation:

| Symbol | Meaning | In the code |
|---|---|---|
| $i, j, k$ | active occupied orbitals, Boys-localized | `no` of them; `LocalizedSpaces::c_locc` |
| $a, b$ | virtual orbitals of the VV-HV localized space | `nv` of them; `VvHv::c_vloc` |
| $t_{ij}^{ab}$ | first-order amplitudes | one `Array2` block per ordered pair, `PairBlock` |
| $J_{ia,jb}$ | $(ia\vert jb)$ in the localized basis | `PairBlock::j_blk` |
| $f_{ik}$, $F_{ab}$ | occupied and virtual blocks of the Fock matrix in the localized bases; **not diagonal** | `f_oo`, `f_vv` |
| $\Delta_{ij}^{ab}$ | $F_{aa}+F_{bb}-f_{ii}-f_{jj}$, positive | `PairBlock::denom` |
| $\varepsilon$ | the threshold | `AmplitudeLmp2Config::eps` |

Entry points:
- Rust: `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2`. Use
  `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2_with_virtuals` when you
  supply your own virtual space.
- CLI: `method.kind = "rimp2"` with `[local] scheme = "amplitude-threshold"`
  and `eps = ...` (`examples/water-rimp2-local.toml`).
- Python: `run_rimp2(..., local="amplitude-threshold", eps=...)`.

The method is closed shell and computes energies only.

## 13.1.1 The amplitude equations and the energy

In canonical orbitals MP2 needs no equations: $t_{ij}^{ab} = -J_{ia,jb}/\Delta_{ij}^{ab}$,
one division per amplitude. Localized orbitals are not eigenfunctions of the
Fock operator, so this shortcut fails. The Fock matrix in the local basis
has off-diagonal elements, and they couple amplitudes to each other.

To see the coupling, write the first-order condition for pair $(i,j)$ as a
matrix in $a,b$. The Fock operator acts on each of the four orbital labels:

- on the particle $a$: $F_{ac}$ moves amplitude from $c$ to $a$, giving
  $F\,t_{ij}$;
- on the particle $b$: likewise, giving $t_{ij}\,F$;
- on the hole $i$: $f_{ik}$ moves the hole from $k$ to $i$, so pair
  $(i,j)$ receives amplitude from pair $(k,j)$;
- on the hole $j$: likewise, from $(i,k)$.

Holes enter with the opposite sign to particles. The residual is

$$ (\mathcal A\,t)_{ij} = F\,t_{ij} + t_{ij}\,F - \sum_k\big(f_{ik}\,t_{kj} + f_{kj}\,t_{ik}\big), \qquad
R = J + \mathcal A\,t = 0, \qquad t^{*} = -\mathcal A^{-1}J . \tag{13.1} $$

Check the canonical limit. With $F$ and $f$ diagonal, $\mathcal A$
multiplies each amplitude by
$F_{aa} + F_{bb} - f_{ii} - f_{jj} = \Delta_{ij}^{ab}$, and $R = 0$ gives back
$t = -J/\Delta$ and the energy of equation (10.3).

The code applies $\mathcal A$ without ever forming it. This is the body of
`ferric-mp2/src/ragged.rs::matvec_indexed`, run in parallel over output
pairs; only the flop counter has been removed:

```rust
let mut r = pb.fvv_aa.dot(&t[p_idx]);        // F t_ij     (particle a)
r += &t[p_idx].dot(&pb.fvv_bb);              //   + t_ij F (particle b)
for &q in &rg.by_j[&pb.j] {                  // pairs (k, j): same j
    let f = f_oo[(pb.i, rg.pairs[q].i)];     //   f_ik
    if f == 0.0 { continue; }
    gather_into(&mut r, &rg.pairs[q], pb, -f, &t[q], ..);   // - f_ik t_kj
}
for &q in &rg.by_i[&pb.i] {                  // pairs (i, k): same i
    let f = f_oo[(rg.pairs[q].j, pb.j)];     //   f_kj
    if f == 0.0 { continue; }
    gather_into(&mut r, &rg.pairs[q], pb, -f, &t[q], ..);   // - f_kj t_ik
}
apply_pattern(&mut r, &pb.pat, nb);          // zero the dropped amplitudes
```

The four lines of (13.1) map onto these lines one for one. Two details are
not in the equation:

- **Pair blocks have different shapes.** Pair $(i,j)$ only stores the
  virtuals in its own domain, `da` × `db`. `gather_into` moves amplitudes
  between two blocks whose virtual lists overlap only in part, matching
  them by index. "Ragged" in the file name refers to these unequal blocks.
- **`apply_pattern` makes the truncation exact.** The operator is projected
  onto the kept amplitudes. Dropped amplitudes are not approximated; they
  are zero, and the equations hold exactly on what remains.

### Two energy formulas, and why the code uses the longer one

With $\tilde t = (2-\mathcal S)\,t$, where $\mathcal S$ swaps
$t^{ab}_{ij} \leftrightarrow t^{ba}_{ij}$ (the same $2(ia|jb) - (ib|ja)$
combination as in equation 10.3), there are two ways to write the energy:

$$ E_{\mathrm{lin}}(t) = \langle\tilde t,\,J\rangle = \sum_{ijab}\big(2t_{ij}^{ab} - t_{ij}^{ba}\big)J_{ia,jb}, \qquad
E_H(t) = \langle\tilde t,\;2J + \mathcal A\,t\rangle . \tag{13.2} $$

At the exact solution they agree, since $\mathcal A t^* = -J$ turns
$2J + \mathcal A t^*$ into $J$. They differ in how they respond to an
amplitude that is slightly wrong, and an iterative solver only ever returns
slightly wrong amplitudes.

Put $t = t^* + d$, where $d$ is the solver's error, and expand. For
$E_{\mathrm{lin}}$ this is one line:

$$ E_{\mathrm{lin}}(t^*+d) = E^* + \langle (2-\mathcal S)\,d,\; J\rangle . $$

The error is linear in $d$. For $E_H$, expand the product. Two cross terms
appear: $\langle (2-\mathcal S)t^*, \mathcal A d\rangle$ and
$\langle (2-\mathcal S)d, J\rangle$. Move $\mathcal A$ to the left,
which needs $\mathcal A$ symmetric and commuting with $\mathcal S$, and use
$\mathcal A t^* = -J$. The first cross term becomes
$-\langle (2-\mathcal S) J, d\rangle$, which cancels the second. What is
left is

$$ E_H(t^*+d) = E^* + \langle (2-\mathcal S)\,d,\; \mathcal A\,d\rangle . $$

The error is quadratic in $d$. An amplitude error of order $\delta$ gives
an $E_H$ error of order $\delta^2$ and an $E_{\mathrm{lin}}$ error of order
$\delta$. That is why the code reports $E_H$. It keeps $E_{\mathrm{lin}}$ as an independent check of the
same solution.

The same expansion gives the sign of the truncation error. $(2-\mathcal S)\mathcal A$
is positive definite, so $E_H$ is a convex quadratic whose minimum over all
amplitudes is the canonical energy $E^*$. Truncation minimizes $E_H$ over a
subset of the amplitudes, and a minimum over fewer variables cannot be
lower. So $E_{\varepsilon} \ge E^*$ for every $\varepsilon$.

> **Verified equation 13.2** · the two energy forms and the stationarity of $E_H$
>
> Along a line $t = t^* + s\,d$ through the solution, $E_H$ has no term linear in
> $s$ while $E_{\mathrm{lin}}$ does, and $E_H(t^*) = E_{\mathrm{lin}}(t^*)$ (claim 1).
> "Residual folded in" is the identity $E_{\mathrm{lin}} + \langle\tilde t,R\rangle = \langle\tilde t, 2J+\mathcal At\rangle$
> (claim 2). On a one-pair model with a swap-closed retained pattern the
> stationarity holds, and on a pattern that keeps $(a,b)$ but drops its
> partner $(b,a)$ it does not: a first-order term survives (claim 5). The
> sign of the exchange term in the code's energy sum is pinned to
> $\sum(2t_{ab}-t_{ba})J_{ab}$, since $2+\mathcal S$ has the same spectrum and would pass
> the rest (claim 4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claims 1, 2, 4 and 5 |
> | Negative control | the runner flips the exchange sign of the transcribed energy sum (claim 4) and drops the 2 on $J$ in the Hylleraas form so a closed mask has a first-order term (claim 5) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::{hylleraas_energy,hylleraas_energy_and_residual,linear_energy}`, `ferric-mp2/src/ragged.rs::check_swap_closed` |
> | Code test | `hylleraas_is_stationary` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.hylleraas; script=scripts/lmp2_hylleraas_vs_wshg23_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::{hylleraas_energy,hylleraas_energy_and_residual,linear_energy},ferric-mp2/src/ragged.rs::check_swap_closed; rust_test=ferric-mp2/tests/lmp2_direct.rs::hylleraas_is_stationary -->

### The kept set must be closed under the swap

The cancellation above assumed that
$\mathcal S$ commutes with the projection onto the kept amplitudes. Suppose
the mask keeps $t^{ab}_{ij}$ but drops its partner $t^{ba}_{ij}$. Then
$\mathcal S$ maps a kept amplitude to a dropped one, the two cross terms no
longer cancel, and a term linear in $d$ comes back. The energy error goes
from second order to first. Claim 5 of the proof script builds exactly this
case and shows the linear term.

So the retention test has to be symmetric. The test in §13.1.3 is symmetric
by construction, and the code does not rely on that alone:
`check_swap_closed` asserts the property, and `hylleraas_energy` calls it.
On a one-sided pattern it returns an error, never a number
(`hylleraas_energy_rejects_a_one_sided_pattern` in
`crates/ferric-mp2/tests/lmp2_direct.rs`).

## 13.1.2 How accurate is the solve?

The amplitudes come from preconditioned conjugate gradients on
$\mathcal A t = -J$, with the diagonal $\Delta_{ij}^{ab}$ as preconditioner.
CG stops at some residual $R = J + \mathcal A t$. The question is how much
energy error that leaves. The expansion above already answers it.

Since $R = J + \mathcal A(t^* + d) = \mathcal A d$, the error is
$d = \mathcal A^{-1}R$. Substitute into the $E_H$ expansion:

$$ E_H(t) - E^{*} = \langle(2-\mathcal S)\,\mathcal A^{-1}R,\;R\rangle \;\le\; \frac{3\,\lVert R\rVert^2}{\lambda_{\min}(\mathcal A)}, \qquad
\lambda_{\min}(\mathcal A) \ge 2\,(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}}) . \tag{13.3} $$

There are three steps from the identity to the bound:

1. $\lVert 2-\mathcal S\rVert = 3$: $\mathcal S$ is a swap, so its
   eigenvalues are $\pm 1$, and $2-\mathcal S$ has eigenvalues 1 and 3.
2. $\lVert \mathcal A^{-1}\rVert = 1/\lambda_{\min}(\mathcal A)$.
3. Bound $\lambda_{\min}$ without computing it. Before projection,
   $\mathcal A$ is a sum of Fock operators acting on separate indices, so
   its eigenvalues are $\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$, with
   the $\epsilon$ taken from diagonalizing `f_vv` and `f_oo`. The smallest is
   $2(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}})$. Projecting onto
   the kept amplitudes compresses the operator, and by Cauchy interlacing a
   compression's smallest eigenvalue can only be larger.

The bound needs only the orbital gap, which is cheap and guaranteed. The CG
iteration also produces Lanczos estimates of $\lambda_{\min}$ for free, but
they converge to it from above. Dividing by an overestimate would make the
"bound" too small, so the code does not use them.

> **Verified equation 13.3** · the solver-error bound
>
> For a one-pair, two-virtual model with $\mathcal A$ symmetric positive
> definite, $\mathcal S$ commuting with $\mathcal A$ and $(2-\mathcal S)\mathcal A$ symmetric:
> $E_H(t^*+d) - E^* = \langle(2-\mathcal S)\mathcal A^{-1}R,R\rangle$ and
> $\lVert R\rVert^2/\lambda_{\max} \le E_H - E^* \le 3\lVert R\rVert^2/\lambda_{\min}$,
> checked on five directions $d$ of a concrete rational model
> (claim 6, which uses the structure of claim 4).
>
> | | |
> |---|---|
> | Proof | `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claim 6 |
> | Negative control | none for this claim; the claim-4 structure checks it relies on carry the controls of equation 13.2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/lmp2_amplitude.rs::{cg_energy_error_bound,operator_lambda_min_lower_bound}` |
> | Code test | `default_rtol_energy_is_within_its_bound` in `crates/ferric-mp2/tests/lmp2_direct.rs` |
>
> <!-- verified: id=lmp2.cg-bound; script=scripts/lmp2_hylleraas_vs_wshg23_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/lmp2_amplitude.rs::{cg_energy_error_bound,operator_lambda_min_lower_bound}; rust_test=ferric-mp2/tests/lmp2_direct.rs::default_rtol_energy_is_within_its_bound -->

What is proven and what is not: the script proves the identity and the bound
on a small model. It does not prove the interlacing step for the production
operator. The code applies that step with the gap of the localized-space
Fock blocks, and the code test checks the reported bound against a dense
solve.

**How the code uses the bound.**

1. The solve first stops at a relative residual of $10^{-6}$
   (`LMP2_CG_RTOL_DEFAULT`) and evaluates (13.3).
2. If the bound is above $10^{-10}$ Ha (`LMP2_CG_ENERGY_ERROR_TARGET`), CG
   continues on the same Krylov sequence, without restarting. It stops at
   the residual that puts the bound at a quarter of the target, but never
   goes below $10^{-11}$ (`LMP2_CG_RTOL_STRICT`).
3. With no positive gap the bound is undefined, and the solve runs at
   $10^{-11}$.

Two details make the bound reliable:

- $R$ is the true residual $\lVert J+\mathcal At\rVert$, taken from the
  energy's own matrix-vector product. It is not the residual CG updates
  recursively, which can drift from the true one.
- Running out of iterations at either stage is an error, not a warning.

The bound covers the solver only. It says nothing about the error from
dropping amplitudes with $\varepsilon$, which is the subject of the rest of
this chapter.

## 13.1.3 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | RHF reference: canonical orbitals and Fock matrix | `ferric-scf/src/rhf.rs::solve_rhf` |
| 2 | virtual space: valence virtuals from projected minimal-basis functions, hard virtuals from the rest | `ferric-mp2/src/lmp2_amplitude.rs::build_vvhv` |
| 3 | Boys-localized active occupieds, their centroids and spreads, and the localized Fock blocks $f$ and $F$ | `ferric-mp2/src/lmp2_amplitude.rs::localized_spaces` |
| 4 | three-index factors in the localized basis, and the metric | `ferric-mp2/src/lmp2_amplitude.rs::assemble_basis` |
| 5 | integral-free pair gate | `ferric-mp2/src/lmp2_amplitude.rs::pair_gate_keep` |
| 6 | per unique pair: $J_{ij}$ by a Gram product of whitened factors over Schwarz-screened candidates, or a domain-local fit; the $\varepsilon$ mask; the ragged block | `ferric-mp2/src/lmp2_amplitude.rs::assemble_ragged_direct_aux`, `ferric-mp2/src/ragged.rs::pair_block_from_g_cand` |
| 7 | PCG on the ragged blocks under the tolerance policy | `ferric-mp2/src/lmp2_amplitude.rs::solve_under_rtol_policy` |
| 8 | $E_H$ and the true residual | `ferric-mp2/src/lmp2_amplitude.rs::hylleraas_energy_and_residual` |

**Step 2, the virtual space (VV-HV).** Valence virtuals are the minimal-basis
(STO-3G) functions projected into the working basis and orthogonalized to the
occupied space, $n_{\mathrm{val}} = n_{\mathrm{min}} - n_{\mathrm{occ}}$ of them, Boys-localized. The remaining
$n_{\mathrm{AO}} - n_{\mathrm{occ}} - n_{\mathrm{val}}$ hard virtuals come from the unit AO vectors with
the occupied and valence parts projected out. Candidates are chosen by
pivoted Cholesky on their overlap, weighted by the inverse spread so compact
ones are taken first, then Löwdin orthonormalized, then
pseudo-canonicalized within each atom's block of the Fock matrix, which cannot
delocalize them. This selection differs from the paper's weighted symmetric
orthogonalization of the full redundant set. The selected set spans the same
space, so energy anchors are invariant to it, and only the compactness of the
domains depends on it.

**Step 5, the gate.** An off-diagonal pair $(i,j)$ is skipped before any
integral work when a London-type estimate falls below a threshold linked to
$\varepsilon$:

$$ \mathrm{cal}\,\frac{(\sigma_i\sigma_j)^3}{R_{ij}^6} < 10^{-2}\,\varepsilon , \tag{13.4} $$

with $R_{ij}$ the distance between Boys centroids and $\sigma_i$ the orbital
spread. Diagonal pairs are never gated, and $\varepsilon = 0$ makes the gate
inert. The constant `cal` is `pair_gate_cal`, off by default; the calibrated
values from the alkane series are about 0.7 for the Coulomb operator and about
0.02 for erfc at $\omega = 1$.

**Step 6, the Eq-8 mask.** Element $(a,b)$ of pair $(i,j)$ is retained when
$\lvert J_{ia,jb}\rvert > \varepsilon$ or $\lvert J_{ib,ja}\rvert > \varepsilon$ (or $\varepsilon = 0$). The test
is symmetric in $(a,b)$, so the pattern is swap-closed by construction. A pair block
holds the rows and columns that have any retained element: those are the pair
domains, equal for rows and columns on a swap-closed pattern, and
`check_swap_closed` requires it. The retained values are the exact $J$ values,
so there is no second approximation inside the block.

The pair kernel is the construction of the block from a candidate square.

<!-- excerpt: ferric-mp2/src/ragged.rs::pair_block_from_g_cand_gated -->
```rust
if eps == 0.0 || jd > eps || kd > eps {
    any_a[a] = true;
    any_b[b] = true;
    any = true;
}
```

| Code | Equation |
|---|---|
| `jd` | $\lvert J_{ia,jb}\rvert$, from the gate quantity `gate[(a, b)]` |
| `kd` | $\lvert J_{ib,ja}\rvert$, the same block transposed, `gate[(b, a)]` |
| `eps` | $\varepsilon$ |
| `any_a`, `any_b` | rows and columns that enter the pair domain |

## 13.1.4 Reproducibility and memory

- The ragged matrix-vector product is rayon-parallel over output pairs, with the
  same arithmetic order per block in every schedule, so it is bit-identical.
- The default assembly builds the global three-index tensor
  $(N_{\mathrm{aux}}, n_on_v)$ once (`ferric-mp2/src/rimp2.rs::eri3_mo_ov_blocked`). So
  amplitude-threshold LMP2 without `integral_direct` is not integral-direct and
  carries no scaling claim. The integral-direct path of section 13.2 removes that
  tensor.
- The exact reference that measures the truncation error is opt-in
  (`compute_reference`, CLI `[local] reference = true`). It is a full canonical
  RI-MP2 that forms the same global tensor, so a run with it on is not
  reduced-cost. Without it the printout says the canonical energy was not
  computed.

## 13.1.5 Options that change the equation

| Option | Effect | Where |
|---|---|---|
| `eps` | the threshold; no default | this section |
| `frozen_core` | removes the lowest occupied orbitals from $i,j$ through `active_occ` | section 10.1 |
| `pair_gate_cal` | enables the gate (13.4); off by default | this section |
| `fit_radius_bohr` | per-pair domain-local RI fit in place of the global-metric fit | section 13.2 |
| `aux_tail_frac` | per-pair auxiliary truncation, rigorously bounded per element by $f\varepsilon$; inactive at $\varepsilon = 0$ | `ferric-mp2/src/lmp2_amplitude.rs::AmplitudeLmp2Config` |
| `cg_rtol`, `cg_max_iter` | Rust only: an explicit tolerance disables the tightening; neither is a CLI or Python option | this section |
| operator | Coulomb or erfc, in the metric and the three-index integrals | chapter 4 |

## 13.1.6 Numerical anchors

The exactness anchor is the first check and the only one with a tolerance. At
$\varepsilon = 0$ the localized-orbital ragged solve reproduces canonical
RI-MP2, which is an independent construction that shares only the RI
integrals, and a construction mutation (dropping one hard virtual) breaks it
(`mutated_virtual_space_fails_the_anchor`).

| Check | Reference | Agreement | Test |
|---|---|---|---|
| $\varepsilon = 0$, every locality map at its no-op limit, in-core (global and per-pair domain fit) and integral-direct: H2O, n-butane / cc-pVDZ, 6-31G (cc-pvdz-ri), frozen core 0 and one per heavy atom | PySCF `DFMP2` with the same auxiliary basis, checked against a dense numpy sum to $10^{-12}$ | $E_{\mathrm{corr}}$ at most $1.38\times10^{-11}$ Ha over 24 comparisons, at the default PCG stop; bar $10^{-10}$ Ha | `crates/ferric-mp2/tests/validation_lmp2_amplitude.rs`, `eps_zero_matches_canonical_ri_mp2` in `crates/ferric-mp2/tests/lmp2_amplitude.rs` |
| finite $\varepsilon$ on n-butane / cc-pVDZ, all electrons | PySCF `DFMP2` (canonical) | recovers 99.98%, 99.67%, 95.0% of the DF-MP2 correlation energy in core at $\varepsilon = 10^{-5}$, $10^{-4}$, $10^{-3}$ with production maps; the error is one-sided and has no tolerance | same |

<!-- figures: site/src/reference/validation.md "Local RI-MP2 (`rimp2` with `[local] scheme = \"amplitude-threshold\"`)" -->

ORCA DLPNO-MP2 on the same molecules is recorded on the validation page as a
ballpark only. It truncates by pair-natural-orbital occupation, not by an
amplitude threshold, so the two recoveries are not compared.

**The error is superlinear in $\varepsilon$.** The recoveries above fall from
99.98% to 99.67% to 95.0% as $\varepsilon$ grows by factors of ten, so the
missing correlation energy grows faster than linearly in $\varepsilon$. There
is no tolerance on it, and users who need a target accuracy choose $\varepsilon$
by running a series.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/lmp2_amplitude.rs`, `crates/ferric-mp2/src/ragged.rs` | the algorithm, the policy and the mask |
| `scripts/lmp2_hylleraas_vs_wshg23_sympy.py`, claims 1, 2, 4, 5, 6 | verified equations 13.2 and 13.3 |
| Wang, Aldossary, Shi, Liu, Li and Head-Gordon, J. Chem. Theory Comput. 19, 7577 (2023) | the single-threshold scheme and the Hylleraas stopping rule, as cited in the module documentation |
| `site/src/methods/mp2.md`, "Exact and local MP2" | the user-facing description and the certificate fields |
| `site/src/reference/validation.md` | the anchor row |
