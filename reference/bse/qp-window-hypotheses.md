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

