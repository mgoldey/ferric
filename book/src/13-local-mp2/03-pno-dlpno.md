# Pair natural orbitals and DLPNO-MP2 {#pno-dlpno}

<!-- status: ready; registry: local-mp2;
     theory: crates/ferric-mp2/theory.md#"12.5 DLPNO-MP2" -->

A pair of occupied orbitals $(i,j)$ correlates by exciting into pairs of
virtuals, but it does not use the virtual space evenly. Two electrons in a
C–H bond mostly excite into the antibonding and polarization functions on
that bond. The amplitude matrix $T^{ij}_{ab}$ of the pair is therefore close
to low rank: a few virtual directions carry most of its weight. Pair natural orbitals (PNOs) are those directions, found by
diagonalizing a density built from the pair's own amplitudes. Keeping only
the important ones compresses the pair's virtual space.

The domain-based local pair natural orbital (DLPNO) idea truncates MP2 in
two places. Pairs whose estimated correlation energy is small are dropped.
Within each retained pair, the virtual space is compressed to its PNOs.
ferric's version is a library path. It is separate from the
amplitude-threshold method of sections @sec:amplitude-lmp2 and @sec:integral-direct, with a different
truncation, and the two are not compared like for like.

The section covers the pair screen (@sec:pno-dlpno-pair-screening), the PNOs (@sec:pno-dlpno-pair-natural-orbitals), the energy
in the PNO basis (@sec:pno-dlpno-energy-pno-basis), the two entry points (@sec:pno-dlpno-algorithm-entry-points), and then the
exactness contracts and the measured numbers.

The derivation is `crates/ferric-mp2/theory.md` §12.5, read from
`crates/ferric-mp2/src/dlpno_mp2.rs`, `local_pno.rs`,
`pair_energy_screen.rs` and `dlpno_localized.rs`. The pair-energy sum, the
PNO energy and the pair density are machine-checked below. The
PNO-truncation error itself has no bound.

| Symbol | Meaning | Code name |
|---|---|---|
| $e_{ij}$ | estimated pair energy, mirror weight included | `PairEnergies::e` |
| $G^{ij}_{ab} = (ia\vert jb)$ | the pair's integral block, $n_v\times n_v$ | `g_ij` |
| $T^{ij}_{ab} = -G^{ij}_{ab}/\Delta^{ab}_{ij}$ | first-order amplitudes, from canonical orbitals | `amp` |
| $\mathcal D^{ij} = T T^{\mathsf T} + T^{\mathsf T}T$ | pair density | `d_sym` |
| $n_k^{ij}$, $Q$ | PNO occupation numbers, and the $n_v\times n_{\mathrm{pno}}$ transform with orthonormal columns | `PnoPair::occupations`, `transform` |
| $t_{\mathrm{cut,pairs}}$, $t_{\mathrm{cut,PNO}}$ | the two thresholds | `DlpnoConfig::t_cut_pairs`, `t_cut_pno` |
| $\mathrm{fac}$ | 1 for $i=j$, 2 for $i<j$ | `fac` |

| Entry point | Location |
|---|---|
| Rust, pair-driven | `ferric-mp2/src/dlpno_mp2.rs::dlpno_mp2_from_b_ov` |
| Rust, dense-$g$ reference | `ferric-mp2/src/dlpno_mp2.rs::dlpno_mp2_spin_components` |
| Rust, PNOs | `ferric-mp2/src/local_pno.rs::{build_pno_transforms,pno_for_pair}` |
| Rust, localized occupieds | `ferric-mp2/src/dlpno_localized.rs::dlpno_mp2_localized_from_b_ov` |
| CLI, Python | none: library only |

MP2 is the natural place to build this stack. The first-order amplitudes
$T^{ij}_{ab} = G^{ij}_{ab}/(\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b)$ are
exactly what PNOs are built from, so the pair density needs no earlier
correlated calculation.

## Pair screening {#pno-dlpno-pair-screening}

Which pairs can be dropped? The screen estimates each pair's energy and
drops the small ones. The estimator is the MP2 pair energy itself, with
$\Delta^{ab}_{ij}>0$ and the mirror weight $\mathrm{fac}$:

$$ e_{ij} = -\,\mathrm{fac}\sum_{ab}\frac{(ia\vert jb)\,\big[\,2(ia\vert jb) - (ib\vert ja)\,\big]}{\Delta^{ab}_{ij}}, \qquad \sum_{i\le j}e_{ij} = E^{(2)} . \label{eq:dlpno-pair-weight} $$

The sum rule follows from the symmetry of the full sum over ordered pairs:
the $(j,i)$ term equals the $(i,j)$ term, so
$\sum_{ij}e_{ij} = \sum_ie_{ii} + 2\sum_{i<j}e_{ij}$, and the weight 2 on an
off-diagonal pair stands in for its mirror.

Each pair energy is also nonpositive. $\Delta^{ab}_{ij}$ is symmetric in
$a$ and $b$, so split the numerator and symmetrize the exchange part over
$a\leftrightarrow b$. With $g_{ab} = (ia\vert jb)$:

$$ \sum_{ab}\frac{g_{ab}(2g_{ab} - g_{ba})}{\Delta_{ab}}
= \sum_{ab}\frac{g_{ab}^2}{\Delta_{ab}} + \frac12\sum_{ab}\frac{(g_{ab} - g_{ba})^2}{\Delta_{ab}} \ \ge\ 0 , $$

so $e_{ij} \le 0$. The two terms are the opposite-spin and same-spin parts
of the pair.

A pair is retained when $\lvert e_{ij}\rvert \ge t_{\mathrm{cut,pairs}}$.
Diagonal pairs are always retained: they carry the bulk of the
correlation, and dropping one has nothing to do with distance. Because the
criterion is an energy, the threshold means the same thing on water as on
a protein. In canonical orbitals $e_{ij}$ is the pair's exact contribution
to (@eq:dlpno-pair-weight), so dropping pairs raises the energy by exactly the sum of their
$\lvert e_{ij}\rvert$. That is at most the number of dropped pairs times
the threshold. A distance cutoff in Bohr (`pair_cutoff`) is the
alternative; when both are set, the energy criterion wins.

> **Verified equation @eq:dlpno-pair-weight** · the unique-pair sum and its weight
>
> $\sum_{ij}e_{ij} = \sum_ie_{ii} + 2\sum_{i<j}e_{ij}$, so the pair energies with
> the mirror factor applied sum to the full correlation energy. The same holds for
> the opposite-spin and same-spin parts separately.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 (unique-pair checks) |
> | Negative control | an off-diagonal weight of 1 must differ (as for equation @eq:mp2-cs-unique-pairs) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/dlpno_mp2.rs::{pair_energy_in_pno_basis,estimate_pair_energies_from_b_ov}` |
> | Code test | `off_diagonal_pairs_carry_the_mirror_weight_of_two` in `crates/ferric-mp2/src/dlpno_mp2.rs` |
>
> <!-- verified: id=dlpno.pair-weight; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/dlpno_mp2.rs::{pair_energy_in_pno_basis,estimate_pair_energies_from_b_ov}; rust_test=ferric-mp2/src/dlpno_mp2.rs::off_diagonal_pairs_carry_the_mirror_weight_of_two -->

The proof covers the sum rule. The sign of each $e_{ij}$ is the two-line
argument above, which the script does not check.

The weight is applied inside the one energy kernel, so no caller can get it
wrong independently. A wrong weight doubles or halves the off-diagonal
correlation energy and still gives a plausible number.

**The screen costs the canonical calculation.** The estimator needs every
pair's $(ia\vert jb)$ block, so it costs $O(n_o^2n_v^2N_{\mathrm{aux}})$,
the same as canonical RI-MP2. Screening reduces what is solved afterwards,
not what the screen costs. With canonical occupieds the estimated energies
of distant pairs are not small, because canonical orbitals spread over the
whole molecule. On the alkane series the screen
retains essentially every pair. The localized path of section @sec:pno-dlpno-algorithm-entry-points exists
for that reason.

## Pair natural orbitals {#pno-dlpno-pair-natural-orbitals}

For each retained pair, build a density from its first-order amplitudes and
diagonalize it:

$$ \mathcal D^{ij} = T^{ij}\,(T^{ij})^{\mathsf T} + (T^{ij})^{\mathsf T}T^{ij}, \qquad \mathcal D^{ij}\,q_k = n_k\,q_k, \qquad Q = [\,q_k : n_k \ge t_{\mathrm{cut,PNO}}\,] . \label{eq:dlpno-pair-density} $$

Why both terms? The row index of $T^{ij}$ is the virtual of electron $i$
and the column index that of electron $j$. $TT^{\mathsf T}$ measures how
much each row direction is used, $T^{\mathsf T}T$ each column direction.
For an off-diagonal pair $T^{ij}$ is not symmetric, and $TT^{\mathsf T}$
alone would favour the index $a$ over $b$. The sum treats the two
electrons alike.

**What the occupations measure.** Both terms are Gram matrices, so
$\mathcal D^{ij}$ is symmetric and positive semidefinite and every $n_k$ is
nonnegative. Its trace is $2\lVert T\rVert_F^2$. Let $P = QQ^{\mathsf T}$
be the projector onto the kept PNOs. The weight of the discarded
occupations is the trace of $\mathcal D^{ij}$ over the complement. Expand
both terms with the cyclic property of the trace and
$(1-P)^2 = 1-P$:

$$ \sum_{k\ \mathrm{dropped}} n_k = \operatorname{tr}\big[(1-P)\,\mathcal D^{ij}\big]
= \big\lVert (1-P)\,T\big\rVert_F^2 + \big\lVert T\,(1-P)\big\rVert_F^2 . $$

So the discarded weight is the squared amplitude norm that falls outside
the kept space on the first index plus the same on the second. For a
fixed number of kept vectors, the eigenvectors with the largest $n_k$
maximize $\operatorname{tr}(P\mathcal D)$ (Ky Fan's maximum principle), so
the PNOs are the subspace of that size that loses the least amplitude
norm. Amplitude norm is not energy, and nothing here bounds the energy
error of the truncation. The code records the discarded weight per pair
as `PnoPair::discarded_weight`.

Check the limit. At $t_{\mathrm{cut,PNO}} = 0$ every vector is kept, $Q$ is
a square orthogonal matrix, and nothing is lost.

> **Verified equation @eq:dlpno-pair-density** · the pair density
>
> $\mathcal D^{ij}$ is symmetric and positive semidefinite, and its trace is
> $2\lVert T\rVert_F^2$ (claim L4).
>
> | | |
> |---|---|
> | Proof | `scripts/local_mp2_fit_pno_sympy.py`, claim L4 |
> | Negative control | none for this claim; the energy claim below carries the mutations |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/local_pno.rs::pno_for_pair` |
> | Code test | `pno_zero_threshold_keeps_everything` in `crates/ferric-mp2/src/local_pno.rs` |
>
> <!-- verified: id=dlpno.pair-density; script=scripts/local_mp2_fit_pno_sympy.py; claim=L4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/local_pno.rs::pno_for_pair; rust_test=ferric-mp2/src/local_pno.rs::pno_zero_threshold_keeps_everything -->

The script proves symmetry, semidefiniteness and the trace. The
discarded-weight identity and the Ky Fan step are not in it; they were
checked numerically on random amplitude matrices for this section.

**The code.** This is the core of `ferric-mp2/src/local_pno.rs::pno_for_pair`,
with the shape check, the eigen-solver error mapping and the assembly of
the result removed:

```rust
let d = t.dot(&t.t()) + t.t().dot(t);                  // T T^T + T^T T
let mut d_sym = Array2::<f64>::zeros((nvir, nvir));
for a in 0..nvir {
    for b in 0..nvir {
        d_sym[(a, b)] = 0.5 * (d[(a, b)] + d[(b, a)]); // symmetrize against rounding
    }
}
// ... eigh, then `order` = eigenvalue indices, largest first ...
let keep: Vec<usize> = order
    .iter()
    .copied()
    .filter(|&k| eigs[k].abs() >= t_cut_pno)           // n_k >= t_cut,PNO
    .collect();
let keep = if keep.is_empty() {
    vec![order[0]]                                     // never empty: keep the largest n_k
} else {
    keep
};
```

Points about the code:

- **The test uses $\lvert n_k\rvert$.** The occupations are nonnegative in
  exact arithmetic, so the absolute value only matters for eigenvalues
  that rounding pushes slightly below zero.
- **A pair is never emptied.** If no occupation passes, the largest is
  kept. An empty virtual space would set the pair's correlation to zero
  without any warning, instead of approximating it.
- **The occupations are not normalized.** The code symmetrizes the density
  and does not divide by its trace, so `t_cut_pno` thresholds the raw
  eigenvalues, in the units of amplitude squared, whose sum is
  $2\lVert T\rVert_F^2$.

**PNOs versus OSVs.** `ferric-rpa/src/pno.rs` builds orbital-specific
virtuals: one set per occupied orbital, from the diagonal pair density
$\mathcal D^{ii}$. PNOs diagonalize $\mathcal D^{ij}$ for each retained
pair, one virtual set per pair, which adapts to what each pair uses.

## The energy in the PNO basis {#pno-dlpno-energy-pno-basis}

The canonical denominator $\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j$
assumes the virtuals diagonalize the Fock matrix. PNOs do not. The kernel
therefore semicanonicalizes, in three operations: rotate the integral
blocks into the PNOs, build the virtual Fock matrix in the pair's PNO basis
and diagonalize it, and carry the integrals into that eigenbasis:

$$ f^{\mathrm{PNO}} = Q^{\mathsf T}\mathrm{diag}(\epsilon_v)\,Q = U\,\mathrm{diag}(\tilde\epsilon)\,U^{\mathsf T}, \qquad
\tilde G = U^{\mathsf T}Q^{\mathsf T}\,G^{ij}\,QU, \qquad \tilde G^{\mathrm x} = U^{\mathsf T}Q^{\mathsf T}\,(G^{ij})^{\mathsf T}\,QU , \label{eq:dlpno-pno-energy} $$

$$ E^{ij}_{\mathrm{OS}} = -\,\mathrm{fac}\sum_{ab}\frac{\tilde G_{ab}^{\,2}}{\tilde\Delta_{ab}}, \qquad
E^{ij}_{\mathrm{SS}} = -\,\mathrm{fac}\sum_{ab}\frac{\tilde G_{ab}\,\big(\tilde G_{ab}-\tilde G^{\mathrm x}_{ab}\big)}{\tilde\Delta_{ab}}, \qquad
\tilde\Delta_{ab} = \tilde\epsilon_a+\tilde\epsilon_b-\epsilon_i-\epsilon_j . $$

$\tilde G^{\mathrm x}$ is the rotated $(ib\vert ja)$, which is the transpose
of the block $(ia\vert jb)$. In the new basis the virtual Fock matrix is
diagonal again, so the pair energy has the canonical form of equation
(@eq:mp2-cs-spin-adaptation), with $\tilde\epsilon$ in place of the virtual orbital energies.

Check the limit. With every PNO kept, $Q$ is square and orthogonal, and
$QU$ is orthogonal too. Then $f^{\mathrm{PNO}}$ has the same eigenvalues as
$\mathrm{diag}(\epsilon_v)$, and for distinct orbital energies $QU$ maps
the PNOs back onto the canonical virtuals up to order and sign. The pair energy is the canonical one.

The shortcut of keeping only the diagonal $f_{aa} = \sum_cQ_{ca}^2\epsilon_c$
is wrong, and it fails without any error. At zero truncation $Q$ is a
square orthogonal rotation, and a rotated diagonal matrix is not diagonal,
so the diagonal alone gives wrong denominators even when nothing is
truncated.

> **Verified equation @eq:dlpno-pno-energy** · the full-rank PNO energy is the canonical energy
>
> With all $n_v$ PNOs kept and the semicanonicalization of the code, the pair
> energy equals the canonical pair energy (claim L3). With only the diagonal of
> $f^{\mathrm{PNO}}$ it does not.
>
> | | |
> |---|---|
> | Proof | `scripts/local_mp2_fit_pno_sympy.py`, claim L3 |
> | Negative control | the runner replaces the semicanonicalization by the diagonal of $f^{\mathrm{PNO}}$ (claim L3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/dlpno_mp2.rs::pair_energy_in_pno_basis` |
> | Code test | `dlpno_mp2_exact_at_zero_truncation` in `crates/ferric-mp2/src/dlpno_mp2.rs` |
>
> <!-- verified: id=dlpno.pno-energy; script=scripts/local_mp2_fit_pno_sympy.py; claim=L3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/dlpno_mp2.rs::pair_energy_in_pno_basis; rust_test=ferric-mp2/src/dlpno_mp2.rs::dlpno_mp2_exact_at_zero_truncation -->

The code, from `ferric-mp2/src/dlpno_mp2.rs::pair_energy_in_pno_basis` with
the shape check, the error mapping, the accumulator declarations and the
comments of the source removed:

```rust
let fac = if i == j { 1.0 } else { 2.0 };                // mirror weight
let t_ij = q.t().dot(&g_ij).dot(q);                      // Q^T G Q
let t_ji = q.t().dot(&g_ji).dot(q);                      // Q^T G^T Q
// ... f_pno[(a, b)] = sum_c q[(c, a)] q[(c, b)] eps[c] : Q^T diag(eps_v) Q ...
let (e_pno, u) = ferric_core::linalg::eigh_dc(&f_pno, ferric_core::linalg::Uplo::Upper)?;
let t_ij = u.t().dot(&t_ij).dot(&u);                     // G~
let t_ji = u.t().dot(&t_ji).dot(&u);                     // G~x
for a in 0..npno {
    for b in 0..npno {
        let d = e_i + e_j - e_pno[a] - e_pno[b];         // -Delta~_ab
        let iajb = t_ij[(a, b)];
        let ibja = t_ji[(a, b)];
        e_os += fac * iajb * iajb / d;
        e_ss += fac * iajb * (iajb - ibja) / d;
    }
}
```

As in the RI-MP2 pair kernel of section @sec:closed-shell-ri-mp2, `d` is $-\tilde\Delta$ and
the sums carry no leading minus; the two signs cancel.

This is the energy of the truncated problem evaluated with the
PNO-projected integrals. It is not a Hylleraas functional. There is no
iteration and no amplitude solve inside a pair, because the canonical
occupied Fock matrix is diagonal.

## The algorithm and its two entry points {#pno-dlpno-algorithm-entry-points}

| Step | Quantity | Function |
|---|---|---|
| 1 | $e_{ij}$ for every pair from $B$, one $n_v\times n_v$ block at a time | `ferric-mp2/src/dlpno_mp2.rs::estimate_pair_energies_from_b_ov` |
| 2 | domains: keep pairs with $\lvert e_{ij}\rvert\ge t_{\mathrm{cut,pairs}}$; diagonals always | `ferric-mp2/src/pair_energy_screen.rs::build_pair_domains_by_energy` |
| 3 | per retained pair: $T$, then the PNOs, built and dropped inside the loop | `ferric-mp2/src/local_pno.rs::pno_for_pair` |
| 4 | per retained pair: $G^{ij} = B_i^{\mathsf T}B_j$, the energy (@eq:dlpno-pno-energy). The same-spin partner $(ib\vert ja)$ is this block's transpose, so no second contraction | `ferric-mp2/src/dlpno_mp2.rs::pair_energy_in_pno_basis` |
| 5 | sum, and report pair retention, virtual retention and the largest discarded weight | `ferric-mp2/src/dlpno_mp2.rs::DlpnoDiagnostics` |

Build on `dlpno_mp2_from_b_ov`. The older `dlpno_mp2_spin_components`
takes as input the dense matrix $g$ of shape $(n_on_v)\times(n_on_v)$. That
is a structural defect: production RI-MP2 never forms $g$, and the
allocation comes before every threshold, so no screen can remove it. Its
size is $(n_on_v)^2\times 8$ bytes, against $6\,n_v^2\times 8$ bytes per
worker for the pair-driven path (`ferric-mp2/src/dlpno_mp2.rs::dense_g_bytes`
and `pair_driven_working_bytes`). At $n_o = 170$, $n_v = 760$, the shape of
the documented drug-scale case, the dense matrix is 133.5 GB and one block
is 4.6 MB. The dense entry point is kept for two jobs: it is the reference
the pair-driven path is anchored to, and it accepts a non-Gram $g$, which
the pair-driven path cannot consume. The robust Dunlap fit gives such a
$g$, with $(ib\vert ja)$ unequal to the transpose of the $(ia\vert jb)$
block.

**Scope.** The pair-driven path assumes $(ia\vert jb) = B_i^{\mathsf T}B_j$,
the Gram product of a factorized $B$. That holds for everything `ri_mp2`
produces and fails for the robust three-term $G$. A non-Gram $G$ fed here
would give a wrong same-spin energy with no error, so the restriction is
stated here and not left implicit.

**Localized occupieds.** `ferric-mp2/src/dlpno_localized.rs::dlpno_mp2_localized_from_b_ov`
addresses the weak canonical screen of @sec:pno-dlpno-pair-screening:

1. Rotate the occupieds to Boys-localized orbitals. This rotates $B$
   exactly and makes the occupied Fock block non-diagonal.
2. Screen pairs with the diagonal-$f$ pair energy.
3. Solve the coupled amplitude equations of equation @eq:amplitude-lmp2-1, with canonical
   virtuals, by Jacobi iteration over the retained pairs only.

This path screens pairs only. It has no PNO or PAO truncation, because the
occupied coupling sums amplitude blocks of different pairs, and that needs
one shared virtual basis. The screen still costs
$O(n_o^2n_v^2N_{\mathrm{aux}})$. An opt-in add-back
(`add_back_truncation`) adds the screened pairs' estimated energy to the
result. The estimate is not a bound, so it can over-correct.

## Exactness and determinism {#pno-dlpno-exactness-determinism}

- With `DlpnoConfig::exact()` (`t_cut_pairs = 0`, `t_cut_pno = 0`, no
  distance cutoff), the pair-driven path reproduces the dense-$g$ path bit
  for bit when fed the same blocks (`pair_driven_matches_dense`), and end
  to end to rounding. At zero truncation $Q$ is a square orthogonal rotation
  that cancels (`dlpno_mp2_exact_at_zero_truncation`), and the pair-driven
  path equals `spin_components_from_b_ov` (`pair_driven_exact_at_zero_truncation`).
- With `t_cut_pairs = 0` the localized solve equals canonical RI-MP2,
  because the energy is invariant to an occupied rotation.
- The default `DlpnoConfig` has pair-energy screening on at
  `DEFAULT_T_CUT_PAIRS` ($10^{-5}$ Ha), and PNO truncation and the distance
  cutoff off. The pair screen is a measured small effect at that
  threshold, and virtual truncation has a measured dead zone and cliff.

## Numerical anchors {#pno-dlpno-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Untruncated DLPNO (`DlpnoConfig::exact()`) and canonical RI-MP2: n-butane, n-hexane, n-octane / cc-pVDZ (cc-pvdz-ri), frozen core 0 and one per carbon | PySCF `DFMP2` with the same auxiliary basis, checked against a dense numpy sum to $10^{-12}$ | at most $3.6\times10^{-11}$ Ha over 12 comparisons | `crates/ferric-mp2/tests/validation_dlpno_mp2.rs` |
| PNO truncation sweep, recovery of ferric's own canonical RI-MP2, all electrons, C4 / C6 / C8 | the same | 99.98 / 99.95 / 99.92% at $t_{\mathrm{cut,PNO}} = 10^{-8}$; 99.75 / 99.48 / 99.10% at $10^{-7}$; 97.2 / 95.3 / 93.4% at $10^{-6}$; monotone in both thresholds and never above 100% | same |
| Localized pair screen, untruncated: water, n-butane, n-hexane, n-octane | ferric's canonical RI-MP2 | 3.2e-12 Ha (water); 1.9e-12, 3.9e-12, 5.9e-12 Ha (C4, C6, C8); the residual is the Jacobi tolerance | `crates/ferric-mp2/tests/validation_dlpno_localized.rs` |
| Localized pair retention at $t_{\mathrm{cut,pairs}} = 10^{-4}$, all electrons, C4 / C6 / C8 | ferric's canonical RI-MP2 | retention 0.667 / 0.511 / 0.410, recovering 99.90 / 99.81 / 99.76% (errors 0.58 / 1.7 / 2.8 mHa, one-sided); frozen core 0.901 / 0.716 / 0.585, recovering 99.93 / 99.84 / 99.80% | same |

<!-- figures: site/src/reference/validation.md "DLPNO-MP2, pair-driven library path (`dlpno_mp2_from_b_ov`; library only, canonical occupieds)"; site/src/reference/validation.md "DLPNO-MP2 pair screen on Boys-localized occupieds" -->

**Reading the numbers.** ORCA's DLPNO-MP2 recoveries are recorded on the
validation page side by side with ferric's and never ranked. The two codes
localize differently and threshold differently: ferric's pair-driven PNOs
are built per canonical occupied pair, while ORCA localizes the occupieds
and truncates PAO domains. ORCA's all-electron columns are also lopsided,
since ORCA applies a separate core threshold. There is no DLPNO-CCSD or
DLPNO-CCSD(T) driver, so ORCA's CCSD(T) figure is context only.

**Known limits.** The canonical pair screen is uninformative for n-butane
to n-octane. With Boys-localized occupieds, the screen drops pairs only
below the locality onset of those chains. The pair retention figures are
measured for these chains only, and the all-electron figure includes core
pairs that the canonical screen also drops. No speedup is claimed: the
pair-driven path removes a memory wall, and whether it is also faster is
unmeasured. The module documentation of `dlpno_mp2.rs` states the same.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/dlpno_mp2.rs`, `local_pno.rs`, `pair_energy_screen.rs`, `dlpno_localized.rs` module documentation and kernels | every definition, the exactness contracts and the scope |
| `scripts/mp2_variant_identities.py`, claim 1; `scripts/local_mp2_fit_pno_sympy.py`, claims L3 and L4 | verified equations @eq:dlpno-pair-weight to @eq:dlpno-pno-energy |
| `site/src/reference/validation.md` | the anchor rows and the known limits |
