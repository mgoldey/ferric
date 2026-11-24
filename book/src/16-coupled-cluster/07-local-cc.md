# Local coupled cluster: amplitude threshold and DLPNO {#local-cc}

<!-- status: edit; registry: linlccd, ccsd-t -->

In a localized basis most coupled-cluster amplitudes of a large molecule
connect orbitals that are far apart, and their driving integrals are close
to zero. Two local approximations sit on top of the canonical solvers of
this chapter to exploit that. The amplitude threshold, the `[local]` section
of `linlccd`, keeps the amplitudes whose driving integral exceeds $\epsilon$
and solves the spatial LinLCCD equations of section @sec:linlccd on that pattern.
The DLPNO pieces, all library-only, compress the virtual space per pair
instead: a LinLCCD(hh) iteration closed in pair natural orbitals, a pair
screen with a PNO energy for CCSD, and a triple screen with triple natural
orbitals for (T).

Each approximation has a trivial limit in which it reproduces the canonical
method, and that limit is how each is anchored. The section takes them in
order:

1. the amplitude threshold (@sec:local-cc-amplitude-threshold-linlccd);
2. DLPNO-LinLCCD(hh) (@sec:local-cc-dlpno-linlccd-hh), the CCSD pair screen (@sec:local-cc-dlpno-ccsd-pair-screen) and the (T)
   triple screen (@sec:local-cc-dlpno-t-triple-screen);
3. the trivial limits and the measured errors (@sec:local-cc-trivial-limits, @sec:local-cc-numerical-anchors).

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/linlccd_amplitude.rs::{amplitude_linlccd,amplitude_linlccd_direct}`, `ferric-cc/src/dlpno_linlccd.rs::dlpno_linlccd_hh`, `ferric-cc/src/dlpno_ccsd_virtual.rs::pno_ccsd_energy`, `ferric-cc/src/dlpno_ccsd_t.rs::{triple_is_retained,screened_triple_energy}` |
| CLI | `method.kind = "linlccd"` with `[local] scheme = "amplitude-threshold"`, `eps` (`examples/water-linlccd-local.toml`); the DLPNO pieces have no CLI kind |
| Python | `run_linlccd(local="amplitude-threshold", eps=)` |

## Amplitude-threshold LinLCCD {#local-cc-amplitude-threshold-linlccd}

`linlccd_amplitude.rs` works closed shell, in Boys-localized occupied
orbitals and VV-HV localized virtuals (`ferric_mp2::lmp2_amplitude`, section
@sec:amplitude-lmp2). The amplitude pattern keeps $(i,a,j,b)$ if and only if

$$|(ia|jb)| > \epsilon \quad\text{or}\quad |(ib|ja)| > \epsilon .$$

The test is the swap-closed one of `ferric_mp2::lmp2_amplitude`: the energy
pairs $T_{iajb}$ with $T_{ibja}$, so the pattern keeps both or neither.

**The masked solve.** The spatial equations of section @sec:linlccd are solved on
the pattern by pattern-projected conjugate gradient on the ragged per-pair
blocks (`ferric-cc/src/linlccd_amplitude.rs::linlccd_masked_solve`):

- the Fock part acts in the localized basis, where it is not diagonal;
- the hh ladder is a gather over source pairs with coefficient $(ik|jl)$;
- for `Full`, the pp block $\sum_P B^P_{ac} B^P_{bd}$ is restricted to each
  pair's virtual domain.

The energy is the linear form $\sum (ia|jb)(2T - \tilde T)$ over the
pattern, with $\tilde T_{iajb} = T_{ibja}$. Its error is therefore first
order in the CG residual, and the residual tolerance is tight
(`cg_rtol` default $10^{-11}$).

**The trivial limit.** $\epsilon = 0$ keeps every amplitude, the masked
equations are the full spatial equations, and the solve reproduces exact
LinLCCD of the same variant.

**The pair gate.** The optional integral-free pair gate (`pair_gate_cal`,
direct path only) drops pair $(i,j)$, $i\ne j$, before any integral is
formed, when

$$ c\,(s_i s_j)^3/R_{ij}^6 < 10^{-2}\epsilon , $$

with $s$ the orbital spreads and $R_{ij}$ the distance between the orbital
centres. The $R^{-6}$ is the decay of the dispersion interaction between two
distant localized pairs.

> **Verified equation {#eq:cc-local-amplitude-spatial}** · the spatial equations the masked solve uses
>
> The spatial closed-shell LinLCCD equations, with hole–hole coefficient
> $(ik|jl)$ and energy $\sum(ia|jb)(2T_{iajb} - T_{ibja})$, give the
> spin-orbital energy and amplitudes for all three variants. At
> $\epsilon = 0$ the pattern is complete and `linlccd_masked_solve` solves
> exactly these equations.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claims 6 and 7 |
> | Negative control | claim 7: $(ij|kl)$ for $(ik|jl)$ in the spatial hh term |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/linlccd_amplitude.rs::{linlccd_masked_solve,amplitude_linlccd}` |
> | Code test | `amplitude_linlccd_nh3_ccpvdz_vs_numpy` in `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
>
> <!-- verified: id=cc.local.amplitude-spatial; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/linlccd_amplitude.rs::{linlccd_masked_solve,amplitude_linlccd}; rust_test=ferric-cc/tests/validation_linlccd_amplitude.rs::amplitude_linlccd_nh3_ccpvdz_vs_numpy; registry=linlccd -->

What the box does not cover is the truncation itself: which amplitudes the
mask drops and how the error grows with $\epsilon$. That is measured, not
derived (anchors below).

## DLPNO-LinLCCD(hh) {#local-cc-dlpno-linlccd-hh}

`ferric-cc/src/dlpno_linlccd.rs::dlpno_linlccd_hh` closes the LinLCCD(hh)
iteration in per-pair PNO bases. The pair natural orbitals of pair $(i,j)$
are the virtual combinations its MP2 amplitudes occupy; keeping only the
strongly occupied ones gives each pair a small virtual space of its own.
Spin-orbital indices are used throughout, so pairs run over $2n_o$ occupied
spin orbitals.

**PNOs.** From the canonical MP2 guess
$t_{ij}^{ab} = -\langle ij\|ab\rangle/\Delta_{ij}^{ab}$, each retained pair
gets the pair density (`ferric_mp2::local_pno::pno_for_pair`)

$$D^{ij} = T^{ij} T^{ij\,\mathsf T} + T^{ij\,\mathsf T} T^{ij},\qquad (T^{ij})_{ab} = t_{ij}^{ab},$$

and keeps the eigenvectors with $|n| \ge t_{\mathrm{cut,pno}}$
(`local_pno.rs`); if none survives, the largest one is kept. $D^{ij}$
carries no $1/(1+\delta_{ij})$ normalization and, in the spin-orbital basis,
no closed-shell $\tilde T = 2T - T^{\mathsf T}$, so $t_{\mathrm{cut,pno}}$
values are not interchangeable with ORCA's `TCutPNO`.

**Semicanonicalization.** The kept columns $Q_{ij}$ do not diagonalize the
Fock operator. Diagonalize its projection,
$F^{ij} = Q_{ij}^{\mathsf T}\,\mathrm{diag}(\varepsilon_v)\,Q_{ij}$, and
rotate, $\tilde Q_{ij} = Q_{ij} U$. This gives PNO energies
$\tilde\varepsilon^{ij}_{\tilde a}$ (`PairPnoBasis::build`,
`dlpno_ccsd_virtual.rs`), so the Fock part of the pair's equations is
diagonal again and the Jacobi update applies. The denominators are
$\tilde\Delta^{ij}_{\tilde a\tilde b} = \tilde\varepsilon^{ij}_{\tilde a}+\tilde\varepsilon^{ij}_{\tilde b}-\varepsilon_i-\varepsilon_j$
(`pno_denominators`, `dlpno_ccsd_kernel.rs`, which stores $-\tilde\Delta$).

**Iteration.** The hole–hole ladder couples pair $(i,j)$ to every pair
$(k,l)$, whose amplitude lives in a different PNO basis. Express
$t^{kl}$ in the canonical virtuals,
$\tilde Q_{kl}\tilde t^{kl}\tilde Q_{kl}^{\mathsf T}$, and project both
virtual indices onto pair $(i,j)$'s basis. With the overlap
$S^{ij,kl} = \tilde Q_{ij}^{\mathsf T}\tilde Q_{kl}$ (`pair_overlap`,
`dlpno_ccsd_kernel.rs`) this is $S^{ij,kl}\,\tilde t^{kl}\,(S^{ij,kl})^{\mathsf T}$,
and the iteration is

$$\tilde R^{ij} = \tilde v^{ij} + \tfrac12\sum_{kl}\langle kl\|ij\rangle\, S^{ij,kl}\,\tilde t^{kl}\,(S^{ij,kl})^{\mathsf T},\qquad
\tilde t^{ij} \leftarrow -\tilde R^{ij} / \tilde\Delta^{ij},$$

$$E_c = \tfrac14\sum_{i\le j} w_{ij} \sum_{\tilde a\tilde b} \tilde v^{ij}_{\tilde a\tilde b}\tilde t^{ij}_{\tilde a\tilde b},\qquad w_{ii}=1,\ w_{i<j}=2.$$

Only the $i\le j$ block is stored, so the energy counts each $i<j$ pair
twice; $t_{lk} = -t_{kl}$ with $a,b$ in place (`so_oriented_amp`) supplies
the other orientation inside the ladder.

**The trivial limit.** At $t_{\mathrm{cut,pno}} = 0$ every $\tilde Q_{ij}$
is square and orthogonal, so every $S^{ij,kl}$ is orthogonal, the projection
loses nothing, and the iteration reproduces `linlccd(.., Hh)`.

## DLPNO-CCSD pair screen and PNO energy {#local-cc-dlpno-ccsd-pair-screen}

**Pair lists** (`ferric_mp2`). Two criteria build the retained pairs
$i\le j$; diagonal pairs are always kept:

- distance (`pair_domains::build_pair_domains`, `pair_domains.rs`): keep
  $(i,j)$ if $|\mathbf r_i - \mathbf r_j| \le R_{\mathrm{pair}}$ (Boys
  centres, Bohr);
- estimated MP2 pair energy (`pair_energy_screen`): keep $(i,j)$ if
  $|e_{ij}| \ge T_{\mathrm{cut,pairs}}$, with
  $e_{ij} = -(2-\delta_{ij})\sum_{ab} (2(ia|jb) - (ib|ja))(ia|jb)/\Delta_{ij}^{ab}$.

$e_{ij}$ is the pair's share of the closed-shell MP2 energy (@eq:closed-shell-ri-mp2-3), with the
factor $2-\delta_{ij}$ counting the $(i,j)$ and $(j,i)$ terms of an
off-diagonal pair together.

`dlpno_ccsd::apply_pair_mask` (`dlpno_ccsd.rs`) zeroes $t_{ij}$ and $t_{ji}$
of every screened pair in the closed-shell CCSD amplitudes; with complete
domains it is the identity.

**PNO basis** (`ferric-cc/src/dlpno_ccsd_virtual.rs::pno_ccsd_energy`): the
same `pno_for_pair` criterion on the closed-shell spatial $T^{ij}$, the same
semicanonicalization, and the closed-shell energy

$$ E = \sum_{ij}\sum_{\tilde a\tilde b}\tilde\tau^{ij}_{\tilde a\tilde b}\big(2\tilde g^{ij}_{\tilde a\tilde b} - \tilde x^{ij}_{\tilde a\tilde b}\big), $$

with $g_{ab} = (ia|jb)$ and $x_{ab} = (ib|ja)$ rotated by $\tilde Q_{ij}$.
It is the energy of section @sec:closed-shell-ccsd with each pair's virtual indices in its own
basis. `dlpno_ccsd_kernel.rs` evaluates individual residual contractions in
PNO bases with $S$ inserted; a closed PNO-basis CCSD iteration does not
exist.

## DLPNO-(T) triple screen and TNOs {#local-cc-dlpno-t-triple-screen}

**Triple list** (`dlpno_ccsd_t.rs`). A banded triple $i\le j\le k$ is kept
if any two indices coincide, or else if all three pairwise Boys distances
are $\le R_{\mathrm{triple}}$ (`triple_is_retained`). The predicate depends
only on the multiset $\{i,j,k\}$, so the retained triples keep the dense
weights $m\in\{1,3,6\}$ of section @sec:triples-closed-shell-t, and

$$E_{(T)} = \tfrac13\sum_{\text{kept}} m\, f(i,j,k)$$

(`screened_triple_energy`). Only $m=6$ triples can be screened.

> **Verified equation {#eq:cc-local-triple-banding}** · the occupied banding of the triple screen
>
> On random closed-shell amplitudes and integrals, the per-triple sum
> $f(i,j,k)$ is invariant over the six orderings of its labels, and the
> $m$-weighted sum over $i\le j\le k$ divided by 3 equals the unrestricted
> sum over 3. A predicate on the multiset $\{i,j,k\}$ therefore keeps or
> drops a whole orbit of orderings, and the dense weights stay valid on the
> retained triples.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claims 2 and 3 |
> | Negative control | claim 5: unit weights and the spin-orbital banding |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/dlpno_ccsd_t.rs::{screened_triple_energy,triple_is_retained,occ_triple_multiplicity}` |
> | Code test | `complete_domains_reproduce_the_unrestricted_count` in `crates/ferric-cc/src/dlpno_ccsd_t.rs` |
>
> <!-- verified: id=cc.local.triple-banding; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/dlpno_ccsd_t.rs::{screened_triple_energy,triple_is_retained,occ_triple_multiplicity}; rust_test=ferric-cc/src/dlpno_ccsd_t.rs::complete_domains_reproduce_the_unrestricted_count; registry=ccsd-t -->

**TNOs** (`dlpno_ccsd_t_virtual.rs`). A triple needs one virtual space that
serves all three of its pairs. Each constituent pair
$(p,q)\subset\{i,j,k\}$ gets a PNO basis $P^{pq}$ at $t_{\mathrm{cut,tno}}$
(the same `pno_for_pair` criterion). The union of their spans is the range
of the projector sum

$$M^{ijk} = \sum_{(p,q)} P^{pq} P^{pq\,\mathsf T},$$

and the code takes the eigenvectors with eigenvalue $> 10^{-8}$
(`UNION_RANK_TOL`). A direction orthogonal to every $P^{pq}$ has eigenvalue
0. A direction inside the union has a positive eigenvalue, but it need not
be large: two one-dimensional subspaces at angle $\theta$ give eigenvalues
$1\pm\cos\theta$. The cut therefore drops a direction of the union only when
two PNO sets nearly coincide. The pairs are summed in sorted order, so the
basis is bit-identical for all orderings of $(i,j,k)$. The union is then
semicanonicalized as for PNOs, and the per-triple $-\sum \tilde W V/\Delta$
is evaluated with every virtual index (free and summed) projected onto that
basis (`dlpno_ccsd_t_kernel::triple_contribution_in_tno`). The kernel
returns the unweighted $f(i,j,k)$; $m$ and the divisor 3 are applied only in
`screened_triple_energy`.

**The trivial limit.** At $t_{\mathrm{cut,tno}} = 0$ every $P^{pq}$ is
square and orthogonal, $M^{ijk}$ is a multiple of the identity, every
transform is square orthogonal, and the dense $f$ is recovered.

## Trivial limits {#local-cc-trivial-limits}

Each approximation is checked first where it does nothing.

| Approximation | Trivial limit | Test |
|---|---|---|
| amplitude threshold | $\epsilon = 0$ keeps every amplitude | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| DLPNO-LinLCCD(hh) | $t_{\rm cut,pno} = 0$: every $S^{ij,kl}$ orthogonal | `stage1_hh_ladder_is_exact_at_zero_truncation` in `crates/ferric-cc/src/dlpno_linlccd.rs` |
| DLPNO-CCSD pair mask | complete domains: the mask is the identity | `complete_mask_leaves_ccsd_energy_bit_identical` in `crates/ferric-cc/tests/dlpno_ccsd_energy.rs` |
| DLPNO-(T) screen | infinite cutoff: every triple kept | `screened_energy_is_identity_when_complete` in `crates/ferric-cc/src/dlpno_ccsd_t.rs` |

Each also has a test that the approximation is not inert at a finite
cutoff (`stage1_truncation_changes_the_ladder`, `screening_is_not_inert`,
`screening_changes_the_energy`). A limit test alone would pass for an
approximation that was never applied; the pair of tests rules that out.

## Numerical anchors {#local-cc-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| amplitude-threshold LinLCCD at $\epsilon = 0$ vs numpy exact solves | H2O, NH3 / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ (drivers-only), $2.8\times10^{-13}$ (hh), $2.0\times10^{-13}$ Ha (full) | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| the integral-direct path at $\epsilon = 0$ | same | $\le 3.0\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| error at finite $\epsilon$, hh | NH3 / cc-pVDZ | $+1.6\times10^{-13}$ ($10^{-6}$), $+2.7\times10^{-9}$ ($10^{-5}$), $+1.07\times10^{-5}$ Ha ($10^{-4}$) | `finite_eps_error_is_one_sided_and_monotone_nh3_ccpvdz` in `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |

The finite-$\epsilon$ errors are positive (the truncated energy lies above
the exact one) and grow by three to four orders of magnitude per decade of
$\epsilon$ on this system.

The DLPNO pieces carry no row in the validation page: there is no
DLPNO-CCSD or DLPNO-(T) energy driver, only the screens, the PNO energy and
the kernels above.

## Sources {#local-cc-sources}

| Source | What it gives |
|---|---|
| `site/src/methods/cc.md`, "Local LinLCCD" | the user guide of the `[local]` path |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claims 2, 3, 6 and 7 |

**Known limits (2026-11-15).** Closed shell and energy only. The PNO
iteration of DLPNO-LinLCCD(hh), the TNO union and the pair screen have no
proof script; their evidence is the trivial-limit and non-inertness tests.
No DLPNO piece has a CLI kind or a Python binding, and no speedup is claimed
for any of them.
