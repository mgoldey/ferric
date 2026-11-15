# 13.1 Amplitude-threshold local MP2

<!-- section: 13.1; status: ready; registry: local-mp2;
     theory: crates/ferric-mp2/theory.md#"12.1 Amplitude equations in localized orbitals" -->

Canonical RI-MP2 sums over every $(i,j,a,b)$. In a localized basis most of
those amplitudes are small, because an occupied orbital $i$ only interacts
strongly with virtuals near it. The amplitude-threshold scheme of Wang,
Aldossary, Shi, Liu, Li and Head-Gordon (J. Chem. Theory Comput. 19, 7577,
2023) keeps an amplitude only when the integral that drives it exceeds a
single threshold $\varepsilon$, and solves the MP2 amplitude equations on the
retained set. The error is one-sided and shrinks to zero with $\varepsilon$.

| Index | Range | Code name |
|---|---|---|
| $i, j$ | active occupied orbitals, Boys-localized | `no` of them; `LocalizedSpaces::c_locc` |
| $a, b$ | virtual orbitals of the VV-HV localized space | `nv` of them; `VvHv::c_vloc` |
| $t_{ij}^{ab}$ | first-order amplitudes | one `Array2` block per ordered pair, `PairBlock` |
| $J_{ia,jb}$ | $(ia\vert jb)$ in the localized basis | `PairBlock::j_blk` |
| $f_{ik}$, $F_{ab}$ | occupied and virtual blocks of the SCF Fock matrix in the localized bases, not diagonal | `f_oo`, `f_vv` |
| $\Delta_{ij}^{ab}$ | $F_{aa}+F_{bb}-f_{ii}-f_{jj}$, positive | `PairBlock::denom` |
| $\varepsilon$ | the threshold | `AmplitudeLmp2Config::eps` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2`, with `ferric-mp2/src/lmp2_amplitude.rs::amplitude_lmp2_with_virtuals` as the entry that takes a caller-supplied virtual space |
| CLI | `method.kind = "rimp2"` with `[local] scheme = "amplitude-threshold"`, `eps = ...` (`examples/water-rimp2-local.toml`) |
| Python | `run_rimp2(..., local="amplitude-threshold", eps=...)` |

The method is closed shell and energy only. `eps` has no default: it is part
of the model. `eps = 0` keeps every amplitude and reproduces canonical RI-MP2.

## 13.1.1 The amplitude equations and the energy

For canonical orbitals the MP2 amplitudes are $t_{ij}^{ab} = -J_{ia,jb}/\Delta_{ij}^{ab}$.
In a localized basis the Fock blocks are not diagonal, and the amplitudes
solve a linear system. With the Fock superoperator $\mathcal A$ and the
exchange swap $\mathcal S: t^{ab}_{ij}\mapsto t^{ba}_{ij}$,

$$ (\mathcal A\,t)_{ij} = F\,t_{ij} + t_{ij}\,F - \sum_k\big(f_{ik}\,t_{kj} + f_{kj}\,t_{ik}\big), \qquad
R = J + \mathcal A\,t = 0, \qquad t^{*} = -\mathcal A^{-1}J . \tag{13.1} $$

The code's matrix-vector product (`ferric-mp2/src/ragged.rs::matvec_indexed`)
is exactly this: for each pair block, the two virtual-Fock products, then the two
occupied-coupling sums over the pairs that share $j$ and $i$, then projection
onto the retained pattern. For canonical orbitals $\mathcal A$ is diagonal with
entries $\Delta_{ij}^{ab}$, which gives $t = -J/\Delta$ and the energy of
equation (10.3).

The energy has two forms that agree at the solution. With
$\tilde t = (2-\mathcal S)\,t$ and the sums running over the retained pattern:

$$ E_{\mathrm{lin}}(t) = \langle\tilde t,\,J\rangle = \sum_{ijab}\big(2t_{ij}^{ab} - t_{ij}^{ba}\big)J_{ia,jb}, \qquad
E_H(t) = \langle\tilde t,\;2J + \mathcal A\,t\rangle . \tag{13.2} $$

$E_H$ is the Hylleraas functional with the residual folded in:
$E_H = E_{\mathrm{lin}} + \langle\tilde t, R\rangle$. The code evaluates $E_H$ and
treats $E_{\mathrm{lin}}$ as an oracle. The reason is the error order. At a
residual $R$ the linear form errs at first order in $\lVert R\rVert$, and
$E_H$ at second.

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

**Why the pattern must be swap-closed.** The energy couples $t^{ab}_{ij}$ with
$t^{ba}_{ij}$, so the retained pattern must contain an element's swap partner,
or the stationarity of $E_H$ is lost and the energy error becomes first order.
The test that decides retention is symmetric by construction (section 13.1.3),
and `check_swap_closed` asserts it. `hylleraas_energy` calls it and errors
instead of returning a non-stationary number
(`hylleraas_energy_rejects_a_one_sided_pattern` in
`crates/ferric-mp2/tests/lmp2_direct.rs`).

## 13.1.2 The solver error is bounded

The solution is found by preconditioned conjugate gradients on $\mathcal At = -J$,
with the diagonal $\Delta_{ij}^{ab}$ as preconditioner. With
$\mathcal A$ symmetric positive definite on the retained pattern and
$\mathcal S$ commuting with it, the error of $E_H$ at a residual $R$ is

$$ E_H(t) - E^{*} = \langle(2-\mathcal S)\,\mathcal A^{-1}R,\;R\rangle \;\le\; \frac{3\,\lVert R\rVert^2}{\lambda_{\min}(\mathcal A)}, \qquad
\lambda_{\min}(\mathcal A) \ge 2\,(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}}) , \tag{13.3} $$

since $\lVert 2-\mathcal S\rVert = 3$ ($\mathcal S^2 = 1$). The second inequality is a
Cauchy interlacing bound: the unprojected operator has eigenvalues
$\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$, the pattern projection is a
compression, and the extreme eigenvalues are read from the eigenvalues of
`f_oo` and `f_vv`. The Lanczos Ritz values from the CG are not used, because
the smallest Ritz value is an upper bound on $\lambda_{\min}$.

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

The script proves the bound on a model; it does not prove the interlacing step
for the production operator. The code applies it with $\lambda_{\min}$ from
$\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}}$ of the localized-space Fock blocks, and the
test above checks the reported bound against a dense solve.

**The default policy.** The solve stops at a relative residual of $10^{-6}$
(`LMP2_CG_RTOL_DEFAULT`). It then evaluates (13.3) at the achieved residual, and
if the bound exceeds $10^{-10}$ Ha (`LMP2_CG_ENERGY_ERROR_TARGET`) it continues,
on the same Krylov sequence, to the residual that puts the bound at a
quarter of the target, never below $10^{-11}$ (`LMP2_CG_RTOL_STRICT`). With no
positive orbital gap the bound is undefined and the solve runs at
$10^{-11}$. The bound is evaluated on the true residual $\lVert J+\mathcal At\rVert$ from the
energy's own matrix-vector product, not on the solver's recursive residual,
and running out of iterations at either stage is an error. The bound limits
the solver error only, not the truncation error of $\varepsilon$.

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
