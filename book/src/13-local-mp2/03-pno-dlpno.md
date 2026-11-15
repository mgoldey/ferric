# 13.3 Pair natural orbitals and DLPNO-MP2

<!-- section: 13.3; status: ready; registry: local-mp2;
     theory: crates/ferric-mp2/theory.md#"12.5 DLPNO-MP2" -->

The domain-based local pair natural orbital (DLPNO) idea truncates MP2 in two
places. Pairs of occupied orbitals whose estimated correlation energy is small are
dropped. Within each retained pair, the virtual space is compressed to the pair
natural orbitals (PNOs) that carry the pair's correlation. ferric's version is a
library path. It is separate from the amplitude-threshold method of sections
13.1 and 13.2, with a different truncation, and the two are not compared like for
like.

The derivation is `crates/ferric-mp2/theory.md` §12.5, read from
`crates/ferric-mp2/src/dlpno_mp2.rs`, `local_pno.rs`,
`pair_energy_screen.rs` and `dlpno_localized.rs`. The pair-energy sum, the
PNO energy and the pair density are machine-checked below. The PNO-truncation
error itself has no bound.

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

MP2 is the right place to build the stack: the first-order amplitudes
$T^{ij}_{ab} = G^{ij}_{ab}/(\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b)$ are exactly
what PNOs are built from, so the pair density needs no earlier correlated
calculation.

## 13.3.1 Pair screening

The estimator is the MP2 pair energy itself. With $\Delta^{ab}_{ij}>0$ and the
mirror weight $\mathrm{fac}$:

$$ e_{ij} = -\,\mathrm{fac}\sum_{ab}\frac{(ia\vert jb)\,\big[\,2(ia\vert jb) - (ib\vert ja)\,\big]}{\Delta^{ab}_{ij}}, \qquad \sum_{i\le j}e_{ij} = E^{(2)} . \tag{13.6} $$

A pair is retained when $\lvert e_{ij}\rvert \ge t_{\mathrm{cut,pairs}}$; diagonal pairs are always
retained, because they carry the bulk of the correlation and dropping them is not
a locality approximation but an error. The criterion is an energy, so the
threshold means the same thing on water as on a protein, and the dropped energy is
at most the number of dropped pairs times the threshold. A distance cutoff in Bohr
(`pair_cutoff`) is the alternative; when both are set, the energy criterion wins.

> **Verified equation 13.6** · the unique-pair sum and its weight
>
> $\sum_{ij}e_{ij} = \sum_ie_{ii} + 2\sum_{i<j}e_{ij}$, so the pair energies with
> the mirror factor applied sum to the full correlation energy. The same holds for
> the opposite-spin and same-spin parts separately.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1 (unique-pair checks) |
> | Negative control | an off-diagonal weight of 1 must differ (as for equation 10.6) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/dlpno_mp2.rs::{pair_energy_in_pno_basis,estimate_pair_energies_from_b_ov}` |
> | Code test | `off_diagonal_pairs_carry_the_mirror_weight_of_two` in `crates/ferric-mp2/src/dlpno_mp2.rs` |
>
> <!-- verified: id=dlpno.pair-weight; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/dlpno_mp2.rs::{pair_energy_in_pno_basis,estimate_pair_energies_from_b_ov}; rust_test=ferric-mp2/src/dlpno_mp2.rs::off_diagonal_pairs_carry_the_mirror_weight_of_two -->

The weight is applied inside the one energy kernel, so no caller can get it wrong
independently. Getting it wrong doubles or halves the correlation energy while
giving a plausible number.

**The screen costs the canonical calculation.** The estimator needs every pair's
$(ia\vert jb)$ block, so it costs $O(n_o^2n_v^2N_{\mathrm{aux}})$, the same as canonical
RI-MP2. Screening reduces what is solved afterwards, not what the screen costs.
With canonical occupieds the estimated energies of distant pairs are not small, and
the screen on the alkane series retains essentially every pair. The localized path
of section 13.3.4 exists for that reason.

## 13.3.2 Pair natural orbitals

For each retained pair, the pair density is built from its first-order amplitudes and
diagonalized:

$$ \mathcal D^{ij} = T^{ij}\,(T^{ij})^{\mathsf T} + (T^{ij})^{\mathsf T}T^{ij}, \qquad \mathcal D^{ij}\,q_k = n_k\,q_k, \qquad Q = [\,q_k : n_k \ge t_{\mathrm{cut,PNO}}\,] . \tag{13.7} $$

Both terms are needed for an off-diagonal pair, because $T^{ij}$ is not symmetric
when $i\ne j$ and $TT^{\mathsf T}$ alone would privilege the index $a$ over $b$. A
pair is never emptied: if no occupation passes, the largest is kept, because an
empty virtual space would silently zero the pair's correlation rather than
approximate it. The weight of the discarded occupations is recorded per pair as
`PnoPair::discarded_weight`.

> **Verified equation 13.7** · the pair density
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

**Occupations are not normalized.** The module documentation of `local_pno.rs`
describes the pair density as "trace-normalized". The code symmetrizes it and does
not normalize it: `t_cut_pno` thresholds the raw eigenvalues of $TT^{\mathsf T}+T^{\mathsf T}T$,
whose sum is $2\lVert T\rVert_F^2$. This section follows the code.

**PNOs versus OSVs.** `ferric-rpa/src/pno.rs` builds orbital-specific virtuals, one set
per occupied orbital from the diagonal pair density $\mathcal D^{ii}$. PNOs diagonalize
the off-diagonal $\mathcal D^{ij}$, one virtual set per pair, which is more adaptive.

## 13.3.3 The energy in the PNO basis

The PNOs are not eigenvectors of the virtual Fock matrix, so the canonical
denominator is invalid in the PNO basis. The kernel semicanonicalizes: rotate the
integral blocks, build the virtual Fock matrix in the pair's PNO basis, diagonalize it,
and carry the integrals into that basis:

$$ f^{\mathrm{PNO}} = Q^{\mathsf T}\mathrm{diag}(\epsilon_v)\,Q = U\,\mathrm{diag}(\tilde\epsilon)\,U^{\mathsf T}, \qquad
\tilde G = U^{\mathsf T}Q^{\mathsf T}\,G^{ij}\,QU, \qquad \tilde G^{\mathrm x} = U^{\mathsf T}Q^{\mathsf T}\,(G^{ij})^{\mathsf T}\,QU , \tag{13.8} $$

$$ E^{ij}_{\mathrm{OS}} = -\,\mathrm{fac}\sum_{ab}\frac{\tilde G_{ab}^{\,2}}{\tilde\Delta_{ab}}, \qquad
E^{ij}_{\mathrm{SS}} = -\,\mathrm{fac}\sum_{ab}\frac{\tilde G_{ab}\,\big(\tilde G_{ab}-\tilde G^{\mathrm x}_{ab}\big)}{\tilde\Delta_{ab}}, \qquad
\tilde\Delta_{ab} = \tilde\epsilon_a+\tilde\epsilon_b-\epsilon_i-\epsilon_j . $$

$\tilde G^{\mathrm x}$ is the rotated $(ib\vert ja)$, which is the transpose of the block $(ia\vert jb)$.
Taking only the diagonal $f_{aa} = \sum_cQ_{ca}^2\epsilon_c$ instead is wrong and fails silently: at zero truncation $Q$
is a square orthogonal rotation, and a rotated diagonal matrix is not diagonal.

> **Verified equation 13.8** · the full-rank PNO energy is the canonical energy
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

This is the energy of the truncated problem evaluated with the PNO-projected
integrals. It is not a Hylleraas functional: there is no iteration and no
amplitude solve inside a pair, since the canonical occupied Fock matrix is
diagonal.

## 13.3.4 The algorithm and its two entry points

| Step | Quantity | Function |
|---|---|---|
| 1 | $e_{ij}$ for every pair from $B$, one $n_v\times n_v$ block at a time | `ferric-mp2/src/dlpno_mp2.rs::estimate_pair_energies_from_b_ov` |
| 2 | domains: keep pairs with $\lvert e_{ij}\rvert\ge t_{\mathrm{cut,pairs}}$; diagonals always | `ferric-mp2/src/pair_energy_screen.rs::build_pair_domains_by_energy` |
| 3 | per retained pair: $T$, then the PNOs, built and dropped inside the loop | `ferric-mp2/src/local_pno.rs::pno_for_pair` |
| 4 | per retained pair: $G^{ij} = B_i^{\mathsf T}B_j$, the energy (13.8). The same-spin partner $(ib\vert ja)$ is this block's transpose, so no second contraction | `ferric-mp2/src/dlpno_mp2.rs::pair_energy_in_pno_basis` |
| 5 | sum, and report pair retention, virtual retention and the largest discarded weight | `ferric-mp2/src/dlpno_mp2.rs::DlpnoDiagnostics` |

`dlpno_mp2_from_b_ov` is the one to build on. The older
`dlpno_mp2_spin_components` takes the dense matrix $g$ of shape
$(n_on_v)\times(n_on_v)$ as input, which is a structural defect: production RI-MP2
never forms it, and the allocation precedes every threshold. Its size is
$(n_on_v)^2\times 8$ bytes, against $6\,n_v^2\times 8$ bytes per worker for the
pair-driven path (`ferric-mp2/src/dlpno_mp2.rs::dense_g_bytes` and
`pair_driven_working_bytes`). At $n_o = 170$, $n_v = 760$, the shape of the
documented drug-scale case, the dense matrix is 133.5 GB and one block is 4.6 MB.
The dense entry point is kept as the reference the pair-driven path is anchored to
and for a non-Gram $g$ (the robust Dunlap fit has $(ib\vert ja)$ unequal to this
block's transpose), which the pair-driven path cannot consume.

**Scope.** The pair-driven path assumes $(ia\vert jb) = B_i^{\mathsf T}B_j$, the Gram
product of a factorized $B$. That is true for everything `ri_mp2` produces and
false for the robust three-term $G$. Feeding a non-Gram $G$ here would give a wrong
same-spin energy with no error, so the restriction is stated, not assumed.

**Localized occupieds.** `ferric-mp2/src/dlpno_localized.rs::dlpno_mp2_localized_from_b_ov`
rotates the occupieds to Boys-localized orbitals, which rotates $B$ exactly and
makes the occupied Fock block non-diagonal. The amplitudes then solve the coupled
equations of equation 13.1 with canonical virtuals, by Jacobi iteration over the
retained pairs only. The pair screen uses the diagonal-$f$ pair energy. It does
pair screening only: there is no PNO or PAO truncation, because the coupling sums
amplitude blocks of different pairs and needs a shared virtual basis. The screen
still costs $O(n_o^2n_v^2N_{\mathrm{aux}})$. An opt-in add-back
(`add_back_truncation`) adds the screened pairs' estimated energy to the result. The
estimate is not a bound, so it can over-correct.

## 13.3.5 Exactness and determinism

- With `DlpnoConfig::exact()` (`t_cut_pairs = 0`, `t_cut_pno = 0`, no distance
  cutoff) the pair-driven path reproduces the dense-$g$ path bit for bit when fed the
  same blocks (`pair_driven_matches_dense`), and end to end to rounding. At zero
  truncation $Q$ is a square orthogonal rotation that cancels
  (`dlpno_mp2_exact_at_zero_truncation`), and the pair-driven path equals
  `spin_components_from_b_ov` (`pair_driven_exact_at_zero_truncation`).
- With `t_cut_pairs = 0` the localized solve equals canonical RI-MP2, because the
  energy is invariant to an occupied rotation.
- The default `DlpnoConfig` has pair-energy screening on at `DEFAULT_T_CUT_PAIRS`
  ($10^{-5}$ Ha) and PNO truncation and the distance cutoff off. The pair screen is
  measurably a small effect at that threshold, and virtual truncation has a measured
  dead zone and cliff.

## 13.3.6 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Untruncated DLPNO (`DlpnoConfig::exact()`) and canonical RI-MP2: n-butane, n-hexane, n-octane / cc-pVDZ (cc-pvdz-ri), frozen core 0 and one per carbon | PySCF `DFMP2` with the same auxiliary basis, checked against a dense numpy sum to $10^{-12}$ | at most $3.6\times10^{-11}$ Ha over 12 comparisons | `crates/ferric-mp2/tests/validation_dlpno_mp2.rs` |
| PNO truncation sweep, recovery of ferric's own canonical RI-MP2, all electrons, C4 / C6 / C8 | the same | 99.98 / 99.95 / 99.92% at $t_{\mathrm{cut,PNO}} = 10^{-8}$; 99.75 / 99.48 / 99.10% at $10^{-7}$; 97.2 / 95.3 / 93.4% at $10^{-6}$; monotone in both thresholds and never above 100% | same |
| Localized pair screen, untruncated: water, n-butane, n-hexane, n-octane | ferric's canonical RI-MP2 | 3.2e-12 Ha (water); 1.9e-12, 3.9e-12, 5.9e-12 Ha (C4, C6, C8); the residual is the Jacobi tolerance | `crates/ferric-mp2/tests/validation_dlpno_localized.rs` |
| Localized pair retention at $t_{\mathrm{cut,pairs}} = 10^{-4}$, all electrons, C4 / C6 / C8 | ferric's canonical RI-MP2 | retention 0.667 / 0.511 / 0.410, recovering 99.90 / 99.81 / 99.76% (errors 0.58 / 1.7 / 2.8 mHa, one-sided); frozen core 0.901 / 0.716 / 0.585, recovering 99.93 / 99.84 / 99.80% | same |

<!-- figures: site/src/reference/validation.md "DLPNO-MP2, pair-driven library path (`dlpno_mp2_from_b_ov`; library only, canonical occupieds)"; site/src/reference/validation.md "DLPNO-MP2 pair screen on Boys-localized occupieds" -->

**Reading the numbers.** ORCA's DLPNO-MP2 recoveries are recorded on the validation
page side by side with ferric's and never ranked, because the two codes localize
differently and threshold differently: ferric's pair-driven PNOs are built per
canonical occupied pair, and ORCA localizes the occupieds and truncates PAO
domains. Its all-electron columns are also lopsided, since ORCA applies a separate
core threshold. There is no DLPNO-CCSD or DLPNO-CCSD(T) driver, so ORCA's
CCSD(T) figure is context only.

**Known limits.** The canonical pair screen is uninformative for n-butane to
n-octane, and with Boys-localized occupieds the screen drops pairs only below the
locality onset of those chains. The pair retention figures are measured for these
chains only, and the all-electron figure includes core pairs that the canonical
screen also drops. No speedup is claimed: the pair-driven path removes a memory
wall, and whether it is also faster is unmeasured. The module documentation says
this in capitals, and this section repeats it.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/src/dlpno_mp2.rs`, `local_pno.rs`, `pair_energy_screen.rs`, `dlpno_localized.rs` module documentation and kernels | every definition, the exactness contracts and the scope |
| `scripts/mp2_variant_identities.py`, claim 1; `scripts/local_mp2_fit_pno_sympy.py`, claims L3 and L4 | verified equations 13.6 to 13.8 |
| `site/src/reference/validation.md` | the anchor rows and the known limits |
