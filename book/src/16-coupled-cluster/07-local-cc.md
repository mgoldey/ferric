# 16.7 Local coupled cluster: amplitude threshold and DLPNO

<!-- section: 16.7; status: edit; registry: linlccd, ccsd-t;
     theory: crates/ferric-cc/theory.md#"Amplitude-threshold LinLCCD", "DLPNO-LinLCCD(hh)", "DLPNO-CCSD: pair screen and PNO substrate", "DLPNO-CCSD(T): triple screen and TNOs" -->

Two local approximations sit on top of the canonical solvers of this
chapter. The amplitude threshold, the `[local]` section of `linlccd`, keeps
the amplitudes whose driving integral exceeds $\epsilon$ and solves the
spatial LinLCCD equations of section 16.5 on that pattern. The DLPNO pieces
are library-only: a LinLCCD(hh) iteration closed in pair natural orbitals,
a pair screen with a PNO energy for CCSD, and a triple screen with triple
natural orbitals for (T). Each has a trivial limit in which it reproduces
the canonical method, and that limit is how each is anchored.

| Entry point | Location |
|---|---|
| Rust | `ferric-cc/src/linlccd_amplitude.rs::{amplitude_linlccd,amplitude_linlccd_direct}`, `ferric-cc/src/dlpno_linlccd.rs::dlpno_linlccd_hh`, `ferric-cc/src/dlpno_ccsd_virtual.rs::pno_ccsd_energy`, `ferric-cc/src/dlpno_ccsd_t.rs::{triple_is_retained,screened_triple_energy}` |
| CLI | `method.kind = "linlccd"` with `[local] scheme = "amplitude-threshold"`, `eps` (`examples/water-linlccd-local.toml`); the DLPNO pieces have no CLI kind |
| Python | `run_linlccd(local="amplitude-threshold", eps=)` |

## 16.7.1 Amplitude-threshold LinLCCD

<!-- include: crates/ferric-cc/theory.md#"Amplitude-threshold LinLCCD" -->

> **Verified equation 16.10** · the spatial equations the masked solve uses
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

## 16.7.2 DLPNO-LinLCCD(hh)

<!-- include: crates/ferric-cc/theory.md#"DLPNO-LinLCCD(hh)" -->

## 16.7.3 DLPNO-CCSD pair screen and PNO energy

<!-- include: crates/ferric-cc/theory.md#"DLPNO-CCSD: pair screen and PNO substrate" -->

## 16.7.4 DLPNO-(T) triple screen and TNOs

<!-- include: crates/ferric-cc/theory.md#"DLPNO-CCSD(T): triple screen and TNOs" -->

> **Verified equation 16.11** · the occupied banding of the triple screen
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

## 16.7.5 Trivial limits

Each approximation is checked first where it does nothing.

| Approximation | Trivial limit | Test |
|---|---|---|
| amplitude threshold | $\epsilon = 0$ keeps every amplitude | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| DLPNO-LinLCCD(hh) | $t_{\rm cut,pno} = 0$: every $S^{ij,kl}$ orthogonal | `stage1_hh_ladder_is_exact_at_zero_truncation` in `crates/ferric-cc/src/dlpno_linlccd.rs` |
| DLPNO-CCSD pair mask | complete domains: the mask is the identity | `complete_mask_leaves_ccsd_energy_bit_identical` in `crates/ferric-cc/tests/dlpno_ccsd_energy.rs` |
| DLPNO-(T) screen | infinite cutoff: every triple kept | `screened_energy_is_identity_when_complete` in `crates/ferric-cc/src/dlpno_ccsd_t.rs` |

Each also has a test that the approximation is not inert at a finite
cutoff (`stage1_truncation_changes_the_ladder`, `screening_is_not_inert`,
`screening_changes_the_energy`).

## 16.7.6 Numerical anchors

| Quantity | Systems | Error | Test |
|---|---|---|---|
| amplitude-threshold LinLCCD at $\epsilon = 0$ vs numpy exact solves | H2O, NH3 / 6-31G, cc-pVDZ | $\le 4.4\times10^{-13}$ (drivers-only), $2.8\times10^{-13}$ (hh), $2.0\times10^{-13}$ Ha (full) | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| the integral-direct path at $\epsilon = 0$ | same | $\le 3.0\times10^{-13}$ Ha | `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |
| error at finite $\epsilon$, hh | NH3 / cc-pVDZ | $+1.6\times10^{-13}$ ($10^{-6}$), $+2.7\times10^{-9}$ ($10^{-5}$), $+1.07\times10^{-5}$ Ha ($10^{-4}$) | `finite_eps_error_is_one_sided_and_monotone_nh3_ccpvdz` in `crates/ferric-cc/tests/validation_linlccd_amplitude.rs` |

The DLPNO pieces carry no row in the validation page: there is no
DLPNO-CCSD or DLPNO-(T) energy driver, only the screens, the PNO energy and
the kernels above.

## 16.7.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-cc/theory.md`, the four local sections | the masks, the PNO and TNO constructions |
| `site/src/methods/cc.md`, "Local LinLCCD" | the user guide of the `[local]` path |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | claims 2, 3, 6 and 7 |

**Known limits (2026-11-15).** Closed shell and energy only. The PNO
iteration of DLPNO-LinLCCD(hh), the TNO union and the pair screen have no
proof script; their evidence is the trivial-limit and non-inertness tests.
No DLPNO piece has a CLI kind or a Python binding, and no speedup is claimed
for any of them.
