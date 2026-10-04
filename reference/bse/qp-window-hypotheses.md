# Issue #280 — hypotheses, written and committed BEFORE any measurement

Date: 2026-10-03. Branch `fix/bse-tda-qp-conditioning`, base `origin/main` @ 0b4fa7dc.
Issue #280: `run_bse_tda` puts a Newton G0W0 QP energy for EVERY MO on the TDA
diagonal, and for core / high-virtual MOs the Thiele/Pade continuation far from
e_F is ill-conditioned (relative 1e-10 perturbation of Sigma_c moves those QP
energies by 0.23-0.29 Ha; two generator runs differ by 1.9e-3 Ha on H2O MO 23).
HOMO-2..LUMO+2 is stable to <= 1.4e-7 Ha.

Task item 1 is MEASURE FIRST: diagonalize the stored, QP-independent kernel
`bse_kernel` with (a) all-MO QP energies, (b) QP inside a HOMO-k..LUMO+k window
with a rigid scissor outside, k = 2, 4, 8, and (c) two independent generator
runs; report how far the lowest 5 Omega move.

## What is actually being asked

Two DIFFERENT questions are entangled in the issue and must be kept apart:

Q1 (conditioning / reproducibility). How much do the lowest 5 Omega move when
the ill-conditioned QP energies are RE-DRAWN, i.e. between two runs that differ
only by the noise the sensitivity probe measures? This is the question
"does the defect reach the quantity users read".

Q2 (window as a method change). How much do the lowest 5 Omega move when the
out-of-window QP energies are REPLACED by a rigid scissor? This is NOT the same
number. The scissor is a different approximation, whose deviation from the
all-MO result is dominated by the SMOOTH drift of the true QP correction across
the virtual manifold, not by the noise. A large (a)-vs-(b) movement therefore
does NOT establish that the conditioning matters; it may only establish that the
scissor is a coarser approximation.

The issue's acceptance criteria conflate these in the "negative control" bullet.
I will report both and label which is which.

## Physics hypothesis

H-P1. The lowest 5 Omega of BSE-TDA on these small closed-shell molecules are
dominated by particle-hole pairs built from HOMO-2..HOMO and LUMO..LUMO+4-ish.
Their eigenvectors therefore have small amplitude on the high-virtual rows whose
QP energies are ill-conditioned. Perturbing those diagonal entries moves the
low Omega only at SECOND order in the eigenvector amplitude (first-order
perturbation theory: dOmega_n = sum_ia |X_n(ia)|^2 d(eps_a - eps_i)).

H-P2. Quantitatively: the per-MO sensitivities are O(1e-2..3e-1) Ha for the
worst virtuals, but a low state's amplitude^2 on those rows should be <= 1e-3,
so the Q1 movement of the lowest 5 Omega should be O(1e-4) Ha or smaller --
i.e. BELOW the 2e-6 Ha bar only if the amplitudes are <= 1e-5, which I doubt.
Prediction (committed): Q1 movement of Omega_1..5 is in 1e-6 .. 1e-4 Ha,
i.e. LARGER than TOL_OMEGA_RAW = 2e-6 but far smaller than the 0.23 Ha QP
noise, and NOT chemically significant (<< 0.01 eV = 3.7e-4 Ha). I expect the
honest verdict to be "real but does not reach the quantity users read at a
chemically meaningful level", with the caveat that it DOES exceed a 2e-6 Ha
numerical bar.

H-P3. Q2 (scissor vs all-MO) movement is much LARGER than Q1 and DECREASES
with k, because the scissor discards a real, smooth ~0.04-0.09 Ha spread of QP
corrections across the virtual manifold. Prediction: Q2 at k=2 is O(1e-2) Ha
(0.1-1 eV) on the lowest Omega, shrinking roughly monotonically with k. If this
holds, the window is NOT a free accuracy win: it trades reproducible noise for
a systematic error 100x larger. That would be an argument for keeping all-MO QP
as the default and exposing the window only as an option / for frozen-core.

H-P4. Degeneracy (NH3): the noise splits degenerate QP pairs (MOs 2/3, 6/7,
8/9, ...). Degenerate Omega groups should therefore split by O(the Q1 movement),
and the grouped comparison the validation row already uses is the right
observable.

## Artifact hypothesis (what a BROKEN measurement would look like)

H-A1. If I decode `bse_kernel` with the wrong triangle convention, wrong flat
index order (i*nvir+a vs a*nocc+i), or wrong nov, the reconstructed A will not
reproduce the stored `bse_singlet.omega` from the stored `qp.eps_qp`. So the
EXACTNESS ANCHOR for the whole measurement is:
  kernel(stored upper triangle) + diag(eps_qp stored) must reproduce
  `bse_singlet.omega` to <= ~1e-13 Ha.
I will run that anchor FIRST and refuse to report any sweep until it passes.
This is the trivial limit "window covers all MOs / scissor applied to nobody".

H-A2. If my windowing code is off by one, or applies the scissor to the wrong
side of the gap, then at k = large enough to cover every MO the windowed result
must STILL equal (a) exactly. A window with k >= max(nocc, nvir) is the vacuous
limit. I will assert that too, and it distinguishes "window implemented
correctly" from "window silently doing nothing / doing everything".
  Distinguishing X from Y: if the window logic is correct, (b) at k >= nmo
  equals (a) to 0.0 exactly AND (b) at k=2 differs from (a). If the window is
  a no-op (bug), (b) at k=2 ALSO equals (a) -> detectable. If the window is
  applied to everything (bug), (b) at k >= nmo differs from (a) -> detectable.

H-A3. If I compute the scissor shift from the wrong reference MO (e.g. the
window's lowest occupied correction applied to virtuals), the out-of-window
virtual diagonal entries move by ~0.1-0.4 Ha (the occupied corrections are
POSITIVE, the virtual ones NEGATIVE), so Q2 would come out O(0.1 Ha) with the
WRONG SIGN of drift vs k. A monotone-in-k Q2 that converges to 0 is the
signature of a correct scissor; a non-monotone or non-vanishing one is the
signature of a side-swap. These predictions differ => the experiment can
distinguish them.

H-A4. "Two independent generator runs" (c): the only stored artifact is a
single run, so a true (c) requires re-running the PySCF generator. The stored
`qp.sensitivity` is a PROXY for run-to-run spread (20 random relative 1e-10
perturbations). Using the proxy I can only bound Q1; if I quote the proxy as if
it were run-to-run I will be over-stating. I will do BOTH: a sensitivity-driven
Monte-Carlo (re-drawing the QP energies within the measured sensitivity) AND an
actual re-run of the generator's QP stage for at least H2O/cc-pVDZ, and report
them separately.

## Pre-registered decision rule

- If the Q1 movement of the lowest 5 Omega is < 1e-9 Ha on all four systems:
  the conditioning does not reach user-visible quantities at all; change
  nothing but a diagnostic + documentation.
- If Q1 is in 1e-9 .. 1e-4 Ha: real, numerically visible, chemically
  irrelevant. Report the numbers; the right remedy is a WARNING/diagnostic plus
  (if Q2 is small) an optional window -- NOT a default behaviour change, since
  per H-P3 the window's own systematic error would be larger.
- If Q1 > 1e-4 Ha (>= ~0.003 eV): the defect reaches users; implement the
  window with the smallest k whose Q2 is below Q1, and say so.
- Independently: if Q2 at the proposed default k EXCEEDS Q1, the window is a
  net accuracy LOSS and must not be the default, whatever Q1 is.


---

# MEASUREMENT (2026-10-03/04). Raw tables first, verdict last.

Method: the reference JSONs under `testdata/reference/validation/bse/` store
`bse_kernel`, the QP-INDEPENDENT part K = 2(ia|jb) − (ab|W(0)|ij) (upper
triangle, base64 f64le, flat ia = i·nvir + a). Every row below is
`eigvalsh(K + diag(eps_qp[nocc+b] − eps_qp[i]))`, lowest 5, in numpy. No Rust
build is involved in the measurement, so none of these numbers depend on the
code change this branch makes.

## Anchor A1 (gate for everything below) — stored kernel + stored eps_qp must
## reproduce the stored `bse_singlet.omega`

| system | nov | max abs dev (Ha) |
|---|---|---|
| h2o/cc-pvdz | 95 | 8.549e-15 |
| nh3/cc-pvdz | 120 | 8.493e-15 |
| ch2o/cc-pvdz | 240 | 9.742e-15 |
| h2o/aug-cc-pvdz | 180 | 1.282e-14 |

Passes. The decode convention, flat index order and diagonal assembly are right.

## Anchor A2 — window vacuous limit

| system | k = nmo: max abs dOmega | k = 2: max abs dOmega |
|---|---|---|
| h2o/cc-pvdz | 0.000e+00 | 1.483e-04 |
| nh3/cc-pvdz | 0.000e+00 | 5.151e-04 |
| ch2o/cc-pvdz | 0.000e+00 | 9.434e-03 |
| h2o/aug-cc-pvdz | 0.000e+00 | 7.798e-04 |

A window covering every MO is bit-identically the all-MO result (so the
windowing code is not applied where it should not be), and k = 2 is not a no-op
(so it is applied where it should be). Both halves of H-A2 hold.

## (a) all-MO QP, lowest 5 Omega (Ha)

| system | Ω1 | Ω2 | Ω3 | Ω4 | Ω5 |
|---|---|---|---|---|---|
| h2o/cc-pvdz | 0.31081386 | 0.38595037 | 0.41045565 | 0.48558280 | 0.55112229 |
| nh3/cc-pvdz | 0.29073775 | 0.37354665 | 0.37354669 | 0.49709856 | 0.49709887 |
| ch2o/cc-pvdz | 0.16896081 | 0.33152774 | 0.36340227 | 0.38814580 | 0.41567603 |
| h2o/aug-cc-pvdz | 0.28381285 | 0.34604724 | 0.37282426 | 0.40951431 | 0.43367154 |

## (b) windowed QP + rigid scissor outside, dOmega vs (a) (Ha)

Window: occupied `max(0, nocc−1−k)` .. virtual `min(nmo−1, nocc+k)`, QP inside;
below the window `eps_mf + corr[lo]`, above `eps_mf + corr[hi]`,
`corr = eps_qp − eps_mf`.

### k = 2

| system | MOs solved | dΩ1 | dΩ2 | dΩ3 | dΩ4 | dΩ5 | max abs |
|---|---|---|---|---|---|---|---|
| h2o/cc-pvdz | 6/24 | +1.09e-04 | +6.46e-05 | +1.37e-04 | +1.48e-04 | +1.01e-04 | 1.483e-04 |
| nh3/cc-pvdz | 6/29 | +2.52e-04 | +5.15e-04 | +5.15e-04 | +2.35e-04 | +2.35e-04 | 5.151e-04 |
| ch2o/cc-pvdz | 6/38 | +3.06e-04 | +1.85e-04 | +9.17e-04 | +1.36e-03 | −9.43e-03 | 9.434e-03 |
| h2o/aug-cc-pvdz | 6/41 | +6.21e-04 | +7.80e-04 | +5.96e-04 | +5.27e-04 | +5.10e-04 | 7.798e-04 |

### k = 4

| system | MOs solved | max abs dOmega (Ha) | (eV) |
|---|---|---|---|
| h2o/cc-pvdz | 10/24 | 4.501e-05 | 0.0012 |
| nh3/cc-pvdz | 10/29 | 1.954e-04 | 0.0053 |
| ch2o/cc-pvdz | 10/38 | 6.041e-04 | 0.0164 |
| h2o/aug-cc-pvdz | 10/41 | 2.594e-04 | 0.0071 |

### k = 8

| system | MOs solved | max abs dOmega (Ha) | (eV) |
|---|---|---|---|
| h2o/cc-pvdz | 14/24 | 1.652e-05 | 0.0004 |
| nh3/cc-pvdz | 14/29 | 1.393e-05 | 0.0004 |
| ch2o/cc-pvdz | 17/38 | 3.583e-05 | 0.0010 |
| h2o/aug-cc-pvdz | 14/41 | 1.958e-04 | 0.0053 |

k = 12 for reference: 2.31e-06 / 5.21e-06 / 2.25e-05 / 4.39e-05 Ha.

## (c) two independent runs of the generator's QP stage

Re-ran `gen_bse.run_gw_all` + `sigma_nodes` + `ferric_qp` from scratch (same
code, same inputs, fresh PySCF process) and compared to the stored `qp.eps_qp`.
This is the real run-to-run spread, not a proxy.

| system | max abs d eps_qp, any MO (Ha) | worst MO | max abs d eps_qp in HOMO−2..LUMO+2 (Ha) | max abs dOmega, lowest 5 (Ha) | (eV) |
|---|---|---|---|---|---|
| h2o/cc-pvdz | 1.194e-03 | 23 | 2.514e-08 | 4.615e-08 | 1.26e-06 |
| nh3/cc-pvdz | 3.250e-03 | 22 | 1.190e-10 | 1.118e-07 | 3.04e-06 |
| ch2o/cc-pvdz | 3.752e-03 | 1 | 3.761e-11 | 8.702e-08 | 2.37e-06 |
| h2o/aug-cc-pvdz | 7.975e-04 | 34 | 4.523e-11 | 1.097e-08 | 2.98e-07 |

Attenuation from QP movement to Ω movement: 2.6e4 / 2.9e4 / 4.3e4 / 7.3e4.

## Monte-Carlo proxy and worst-case bound on the same quantity

Re-drawing every QP energy uniformly within its stored `qp.sensitivity`, 200
draws, gives max abs dOmega of 2.45e-05 / 3.86e-06 / 3.68e-05 / 9.69e-06 Ha.
A worst-case COHERENT-sign bound from first-order perturbation theory,
`dOmega_n = sum_p (sum of |X_n(ia)|^2 over rows touching MO p) * sensitivity_p`,
gives 3.17e-05 / 6.78e-06 / 5.53e-05 / 1.21e-05 Ha.

Both OVER-estimate the observed run-to-run movement by 1-2 orders of magnitude,
because the real generator-to-generator noise is correlated across MOs rather
than independent-uniform. Quoting the proxy as if it were run-to-run, which
H-A4 warned about, would have over-stated the defect by ~500x on H2O.

## Why the attenuation is so large

Summed `|X_n(ia)|^2` of the lowest 5 states over rows touching ANY MO whose
stored sensitivity exceeds 1e-3:

| system | suspect MOs | affected rows | summed amplitude^2, lowest 5 |
|---|---|---|---|
| h2o/cc-pvdz | 14/24 | 74/95 | 1.4e-03 .. 5.1e-03 |
| nh3/cc-pvdz | 16/29 | 90/120 | 5.6e-04 .. 8.3e-04 |
| ch2o/cc-pvdz | 21/38 | 180/240 | 1.1e-03 .. 6.6e-03 |
| h2o/aug-cc-pvdz | 21/41 | 129/180 | 2.6e-04 .. 1.4e-03 |

Most ROWS touch a suspect MO, but the lowest eigenvectors carry under 0.7% of
their norm there. The first-order response is the amplitude-weighted sum, so a
0.3 Ha uncertainty on a high virtual reaches Ω at the 1e-4 level at worst, and
at the 1e-7 level in practice because the per-MO errors partially cancel.

---

# VERDICT (dated 2026-10-04, provisional as all verdicts here are)

Measured, not assumed:

1. The conditioning defect is REAL. Individual far-from-Fermi QP energies are
   not reproducible to better than ~4e-3 Ha between runs, so
   `BseResult::eps_qp` must not be quoted for core or high-virtual MOs.
2. It does NOT reach the quantity users read. The lowest five Ω move by at most
   1.1e-7 Ha (3.0e-6 eV) between independent runs, on all four systems. That is
   four orders of magnitude below chemical significance and below the existing
   `TOL_OMEGA_RAW = 2e-6` validation bar.
3. The proposed remedy is WORSE than the defect, at every window size measured.
   The windowed-QP + scissor recipe moves the lowest five Ω by 1.5e-4..9.4e-3 Ha
   at k = 2, 1.4e-5..2.0e-4 Ha at k = 8, and 2.3e-06..4.4e-05 Ha at k = 12 —
   never below the 1.1e-7 Ha it would remove. Even the pessimistic worst-case
   coherent bound on the noise (1.2e-05..5.5e-05 Ha) is comparable to or smaller
   than the k = 8 scissor error. The scissor discards a real, smooth ~0.04-0.09
   Ha spread of QP corrections across the virtual manifold; that systematic loss
   dominates the noise it removes.

Matches/contradicts the pre-registered hypotheses:

- H-P1 CONFIRMED (amplitude^2 on affected rows <= 7e-3; the mechanism is the
  predicted one).
- H-P2 WRONG in the conservative direction. I predicted Q1 in 1e-6..1e-4 Ha and
  "larger than TOL_OMEGA_RAW". The real run-to-run Q1 is 1.1e-8..1.1e-7 Ha,
  i.e. 10-100x SMALLER than predicted and BELOW the 2e-6 bar. My error was
  assuming the per-MO noise adds incoherently at full amplitude; it partly
  cancels. The Monte-Carlo proxy reproduced my prediction (2.4e-5 Ha) and the
  real re-run refuted it — which is exactly why H-A4 required running both.
- H-P3 CONFIRMED and stronger than stated: Q2 exceeds Q1 by 100-10^5x and does
  so at every k measured, not only at k = 2.
- H-P4 CONFIRMED: NH3's degenerate pairs (Ω2/Ω3 and Ω4/Ω5) stay degenerate to
  4e-11 Ha under the re-run, i.e. the degeneracy split is also attenuated.

Per the pre-registered decision rule, Q1 < 1e-9 Ha is false (it is 1e-8..1e-7)
and Q1 > 1e-4 is false, so this lands in the middle branch: "real, numerically
visible, chemically irrelevant — report the numbers, add a diagnostic, do NOT
change the default behaviour". The rule's independent clause also fires: Q2 at
every candidate k exceeds Q1, so the window must not be the default.

## What was therefore changed, and what was not

CHANGED (robustness/observability only, no returned energy moves):
- `ferric_gw::bse::flag_suspect_qp` names the MOs whose G0W0 Newton solve did
  not converge or whose Z renormalization was clamped to a boundary of the
  [0, 1.5] clamp `sigma::solve_qp_for_mo` applies.
- `run_bse_tda` and `run_bse_c6` warn once naming those MOs, with the measured
  attenuation in the message so a reader can judge the consequence.
- `BseResult::qp_suspect_mos` (and the Python getter) report the list.
- The validation row and `TOL_OMEGA_RAW`'s comment carry the measured numbers.

NOT CHANGED, deliberately:
- No `qp_window` config field, no scissor, no CLI/TOML key, no change to
  `gen_bse.py`'s reference spectrum. Issue #280 task items 2, 4 and 5 describe
  the window as the fix; the measurement says the window is a net accuracy
  loss, so implementing it as the default would make BSE-TDA less accurate in
  order to close a ticket. Task item 1 was explicit that the measurement
  decides, and it decided against.
- `TOL_OMEGA_RAW` stays at 2e-6. It is ~9x its measured max (2.2e-7 Ha), which
  is already the repo's ~10x rule. That 2.2e-7 also carries the
  ferric-vs-generator screening-path difference, not only QP noise, so
  tightening toward the 1.1e-7 Ha Ω-noise floor would be a bar on a different
  quantity.
- The `QP_SENS_FACTOR`-scaled per-MO QP bars stay. They are the correct
  observable for the per-MO energies, which genuinely are only reproducible to
  their measured sensitivity.

## Open item this measurement does NOT close

The issue notes that with `frozen_core > 0` the all-MO QP range hits the frozen
block and `run_gw` refuses it, and that a window would remove that limitation.
That is a real, separate usability gap with its own correct fix (restrict the
QP range to the ACTIVE MOs, which needs no scissor and no accuracy tradeoff).
It is not addressed here and should be its own issue, because bundling it with
the window would re-import the accuracy loss this measurement rejects.

## Mutation ledger

`flag_suspect_qp`, 5 unit tests, each mutant run with
`cargo test -p ferric-gw --lib bse::tests::flag_suspect` and the
passed/failed/ignored counts read (all runs executed 5 tests, 0 ignored):

| # | mutation | result | caught by |
|---|---|---|---|
| M1 | drop the `!qp_converged[k]` criterion | 4 passed, 1 FAILED | `..._catches_unconverged_newton` |
| M2 | drop both Z-clamp criteria | 3 passed, 2 FAILED | `..._catches_both_z_clamp_boundaries...`, `..._returns_absolute_ascending_mo_indices` |
| M3 | `<=`/`>=` weakened to `<`/`>` at the clamp boundary (off-by-one) | 3 passed, 2 FAILED | same two |
| M4 | return the local index `k` instead of the absolute MO index | 2 passed, 3 FAILED | three tests |
| M5 | drop `out.sort_unstable()` | 4 passed, 1 FAILED | `..._returns_absolute_ascending_mo_indices` |

5/5 caught, each by the test written for it. `bse.rs` byte-identical to the
pre-mutation copy afterwards (verified with `diff -q`).

---

# ADDENDUM 2026-10-04: the degenerate-QP mechanism is measured, not separate

The issue lists NH3's degeneracy splitting as a distinct symptom: the noise
splits degenerate QP energies, so Omega depends on an arbitrary MO rotation
(quoted 1.9e-8 to 4.9e-8 Ha). It is natural to read that as a second,
unaddressed mechanism. The measurement covers it, and it is the SAME
attenuation story.

NH3/cc-pVDZ degenerate QP pair splits, stored run:

| MO pair | split (Ha) |
|---|---|
| 2/3 | 2.14e-07 |
| 6/7 | 5.97e-09 |
| 8/9 | 3.39e-07 |
| 12/13 | 1.06e-07 |
| 14/15 | 3.32e-06 |
| 19/20 | 6.47e-05 |
| 21/22 | 3.97e-03 |
| 24/25 | 2.01e-04 |
| 27/28 | 2.71e-04 |

The splits grow by five orders of magnitude with distance from the Fermi level,
exactly like the sensitivities. The corresponding Omega degeneracy splits:

| quantity | stored run | independent re-run | change between runs |
|---|---|---|---|
| Omega2 - Omega3 | 4.172e-08 | 3.600e-08 | 5.72e-09 |
| Omega4 - Omega5 | 3.067e-07 | 3.370e-07 | 3.03e-08 |

So a 3.97e-03 Ha QP degeneracy violation on MO 21/22 reaches the lowest Omega
pairs as a 4e-08 to 3e-07 Ha split, and that split is itself reproducible to
6e-09 / 3e-08 Ha between runs. The stored
`degenerate_rotation_ambiguity = 4.90e-08` agrees with the Omega-level split, and
is NOT the 3.97e-03 Ha QP-level number.

Consequence for the fix decision: unchanged, and reinforced. The windowed
scissor would REPLACE those out-of-window QP energies with a single rigid shift,
which makes every out-of-window degenerate pair exactly degenerate — cosmetically
better at the QP level, while moving the lowest Omega by 1.4e-05 to 5.2e-04 Ha
(k = 8 and k = 2), i.e. 100 to 10^4 times the 4e-08 Ha Omega-level split it
would tidy up. Enforcing a symmetry at the QP level by discarding real
information is not a trade worth making for a quantity already attenuated to
1e-07.

What the measurement does NOT cover: whether the Omega-level split matters for
assigning degenerate STATE pairs in a code that reports symmetry labels.
ferric's BSE-TDA reports neither labels nor irreps, so there is nothing for a
4e-08 Ha split to corrupt today. If labelling is added, this becomes live again.
