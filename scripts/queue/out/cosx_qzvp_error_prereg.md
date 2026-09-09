# Pre-registration: COSX accuracy at def2-QZVP (butane)

Written and committed BEFORE any number in this lane was taken.
Date: 2026-09-08. Branch `meas/cosx-qzvp-error`, worktree `.claude/worktrees/cosx`.

## The gap this closes

`wiki/cosx-scaling-whitepaper.md` §3.1 reports COSX's A-build at 1.51x a LinK K
build at butane/def2-QZVP, and §3.3 reports a full COSX K there at 137.04 s.
Both are COST numbers. COSX's ACCURACY at QZVP has never been measured. The only
accuracy points on record are:

| system / basis | SCF dE (COSX - exact) |
|---|---|
| water / cc-pVDZ | +4.9e-6 Ha |
| butane / def2-TZVP | -1.2e-4 Ha |

So the headline QZVP ratio compares an approximate builder of UNKNOWN error
against an exact one. That is not a defensible speedup claim.

## What will be measured

Butane (`testdata/molecules/alkane_4.xyz`), def2-QZVP, COSX in its production
configuration: `k_builder="cosx"`, `cosx_backend="md3c1e"`, sparse half
transforms, `cosx_screen_thresh = 1e-7`, grid (50,110), `cosx_overlap_fit=true`.

1. `max|K_cosx - K_exact|` on a converged density, alongside `max|K_exact|`.
2. SCF energy error vs the direct builder, same convergence settings, with
   iteration counts for both.
3. Grid sensitivity: the same SCF error at (25,50) and (75,194).
4. Overlap fit ON vs OFF at QZVP.

## Reference choice (decided before measuring)

`K_exact` is the **direct four-centre builder** (`ferric_scf::rhf::build_jk` at
`integral_thresh` 1e-12), NOT LinK.

This is not a stylistic preference. `scripts/queue/out/cosx_scaling_results.md`
incidental finding #2 records that standalone LinK on this branch differs from
the direct K by 2.784e-3 (max element) on butane/def2-SVP, where COSX's own
error against the direct K on the same density is 2.9e-4 — i.e. LinK is ~10x
FURTHER from exact than the quantity being measured. Using LinK as the reference
would report LinK's error, not COSX's. The `COSX_FK_LINK` path in
`cosx_full_k.rs` is therefore NOT the measurement instrument here; a direct-K
block is added to that harness for this lane.

## Predictions (stated before any run)

### P1 — K-matrix error at QZVP

**Prediction: max|K_cosx - K_direct| lands in 1e-4 .. 3e-3, i.e. LARGER in
absolute terms than the def2-SVP 2.9e-4 figure, and I expect it in the upper
half of that range.**

Reasoning: the COSX error is a *quadrature* error on
`K_uv = sum_g w_g X_u(g) A_v(g)`. Its size is set by how well a fixed Becke grid
integrates the AO products it is fed. Going SVP -> TZVP -> QZVP raises L_max from
2 to 3 to 4 and adds tight, high-exponent primitives. Both changes make the
integrand more oscillatory and more sharply peaked near nuclei, and the radial
grid is NOT refined to compensate — (50,110) is fixed. A quadrature rule held
fixed while the integrand's angular order rises must lose accuracy. There is
also a pure magnitude effect: more basis functions means larger K elements, so
some growth is trivial. This is exactly why `max|K|` must be reported alongside;
the relative figure `max|dK| / max|K|` is the transferable one.

### P2 — SCF energy error at QZVP vs the TZVP -1.2e-4 Ha

**Prediction: |dE| GROWS relative to TZVP. Point estimate: -2e-4 to -1e-3 Ha,
same sign (COSX below direct), most likely ~-3e-4.**

Reasoning: same argument as P1 — fixed grid, higher angular momentum. The
observed trend already points this way: +4.9e-6 (cc-pVDZ, water, 1 heavy atom)
-> -1.2e-4 (def2-TZVP, butane, 4 heavy atoms). Part of that jump is size (4x the
atoms), part is basis. QZVP adds another L unit on the same molecule, so the
size term is held fixed and only the basis term moves. A factor of a few is the
honest expectation; an order of magnitude would surprise me.

**The sign matters.** The TZVP point is negative (COSX below exact), which is
NOT variationally protected — a seminumerical K is not a variational
approximation to the exact K, so the SCF can sit below the exact answer. If
QZVP comes out POSITIVE while TZVP was negative, that is a sign flip on a
one-basis change and I would treat it as suspicious rather than physical, and
re-check the density and convergence before reporting it.

### P3 — is (50,110) still adequate at QZ?

**Prediction: NO, or marginally. I expect (75,194) to move the QZVP energy by
more than chemical accuracy (>1e-3 Ha would be a strong "no"; 1e-4..1e-3 is
"the default is the error"), and I expect the (25,50) -> (50,110) -> (75,194)
sequence to be MONOTONE and NOT converged at (50,110).**

The interesting falsifiable half: if (50,110) and (75,194) agree to <1e-5 Ha at
QZVP, the default IS converged at quadruple zeta and P1/P2's whole
"finer basis needs finer grid" story is wrong. I would accept that.

### P4 — the overlap fit at QZVP

**Prediction: the fit still HELPS at QZVP, but by less than its water headline.
I expect fit-OFF to be worse than fit-ON by a factor of ~2-10 in |dE|, not the
16.9x/24.2x seen on water.**

Reasoning: the fit is a near-field correction that repairs the numerical overlap
`S_num` against the analytic `S`. Its measured benefit is strongly
system-dependent and already known NOT to transfer: 16.9x/24.2x on water, but
5.0x/1.1x on methane and 0.5-0.9x on ethane — i.e. on ethane the fit HURTS.
Butane is the ethane-like end of that set (a chain of CH2/CH3 with no lone
pairs), so a fit that is neutral-to-harmful at QZVP is a live possibility, and
would be a real finding: **if fit-OFF beats fit-ON at QZVP on butane, the
`overlap_fit = true` default is wrong for hydrocarbon chains at high L** and the
paper must say so.

I explicitly flag that P4's prediction ("helps") and the alternative it is most
worried about ("hurts") are BOTH plausible from the existing water/methane/
ethane spread. This experiment can distinguish them; that is why it is worth
running.

## Artifact hypothesis (what a BROKEN measurement looks like)

Per the repo's experimental protocol, the artifact hypothesis is written next to
the physics hypothesis, before measuring. If any of these fire, the number is
discarded, not interpreted:

* **A1 — contaminated timing/energy from memory pressure.** `/proc/pressure/memory`
  `full avg10` nonzero before OR after a run. QZVP butane is nbf ~528 and the
  direct J+K sweep is L=4 quartets; a reclaim storm would silently inflate wall
  time and (via a spilled/thrashed run) risk a partial result. Every cell prints
  PSI before and after; nonzero => discard and re-run.
* **A2 — cpu != wall.** Runs are pinned to ONE thread
  (`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1`). If cpu/wall departs from ~1.0
  the process was contended or multi-threaded and the cell is void.
* **A3 — comparing across different densities.** The K error must be evaluated
  on ONE density contracted by both builders. If `K_cosx` and `K_direct` were
  built from densities from different SCFs, the "error" is a density difference,
  not a quadrature error. Mitigation: both builds happen in ONE process on the
  same `d`, or the density is round-tripped through the harness's own save/load.
* **A4 — an unconverged SCF reported as an energy.** `COSX_FK_SCF_SAVE_UNCONVERGED`
  exists in this harness for window-chaining; an unconverged density silently
  used as "the" density would fake an error of arbitrary size. Every reported
  SCF must print `converged=true` and its iteration count.
* **A5 — the config didn't take.** ferric's CLI is `deny_unknown_fields` and
  errors on a cosx knob with the wrong `k_builder`, so a typo hard-errors rather
  than silently defaulting. But a grid change that does NOT change the point
  count, or a `cosx_overlap_fit=false` run that produces a BIT-IDENTICAL energy
  to fit=true, means the knob did not reach the builder. Point counts and
  energies must differ across the grid arm; identical energies across arms =
  broken configuration, not a null result.
* **A6 — "too clean".** If the three grid arms give energies agreeing to
  machine precision, or the fit-on/fit-off pair agrees to 1e-12, that is
  arithmetic, not chemistry (see A5). Suspicious tidiness triggers an audit of
  whether the knob is wired, not a write-up.

## What would change the recommendation

* (75,194) - (50,110) > 1e-3 Ha at QZVP => the default grid is NOT adequate at
  quadruple zeta and the paper must qualify its QZVP cost claims, because a
  fair comparison would have to price the finer grid COSX actually needs.
* fit-OFF better than fit-ON at QZVP => `overlap_fit = true` is the wrong
  default for this regime.
* Either outcome makes the §3.1 "1.51x" headline conditional. A cost ratio
  against an exact builder is only meaningful once the approximate builder's
  error is stated next to it.

## Scope limit (stated up front)

ONE molecule (butane), ONE basis (def2-QZVP). Butane is a saturated hydrocarbon
with no lone pairs, no heteroatoms, no diffuse functions. Whatever comes out is
a single point, not a QZVP-wide verdict, and will be reported as such. In
particular a good result here does NOT license "COSX is accurate at QZVP".
