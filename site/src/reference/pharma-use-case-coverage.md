# Pharma use-case coverage, measured 2026-09-19

What the golden paths actually cover against the six use cases Matt named, and
where the one real gap is. Measured by grepping `tools/`, `experiments/` and
`crates/ferric-python/` on `origin/main` at `1ad0c84b`, not by reading the
pipeline docs — the docs describe intent, the grep describes code.

## METHOD, and one correction

First pass used `grep -rlE 'MMFF\|UFF'` with escaped alternation inside `-E`,
which matches the literal string `MMFF|UFF` and therefore **nothing**. It
reported four capabilities as absent that are all present (ff-minimize, xtb,
toxicology, substitution). Caught by running the pattern against a term known
to be there — the discipline in `grep-patterns-manufacture-false-results`.

The table below is the corrected pass. `files` counts files matching the
pattern anywhere under those three trees.

## COVERAGE

| use case | files | where | status |
|---|---:|---|---|
| docking | 33 | `tools/docking/` (vina + meeko) | **covered** |
| geometry opt (pose) | 7 | `tools/active_site/pose_relaxation.py` | **covered** |
| minima with FF | 16 | `tools/docking/vina_dock.py`, `tools/campaign/strain.py` (MMFF/UFF) | **covered** |
| minima with xtb | 27 | `tools/campaign/xtb_engine.py` | **covered** |
| common substitutions | — | `tools/pipeline/substitution.py`, `tools/isomers/substitutional.py` (#96) | **covered** |
| toxicology | 21 | `tools/tox/alerts.py`, `tools/tox/model.py` | **covered** |
| binding energy in site | 4 | `tools/active_site/prescreen.py`, `binding_energy.py` | **covered** |
| **transition state** | **1** | — | **GAP** |

The single transition-state hit is `tools/campaign/tests/test_strain_and_fit.py`
matching `dimer_` incidentally. There is no saddle-point search.

## THE GAP: transition-state search

What exists already, which is most of the machinery:

- `crates/ferric-scf/src/optimize.rs` — BFGS driver, minimizes; `run_bfgs`
  plus a coordinate-vector core `optimize_coordinates`
- `crates/ferric-scf/src/frequencies.rs` — `harmonic_frequencies` and
  `frequencies_from_cartesian_hessian`, i.e. a finite-difference Hessian built
  from analytic gradients, plus `atom_masses` and `detect_linear`
- Analytic nuclear gradients for RHF/UHF/ROHF and KS-DFT including meta-GGA

What is missing is the **saddle step itself**: P-RFO (partitioned rational
function optimization) follows the eigenvector with the *negative* Hessian
eigenvalue UPHILL while minimizing along all the others. BFGS cannot do this —
it is built to descend, and its Hessian update is kept positive definite on
purpose.

Sketch of what a `saddle.rs` needs, in dependency order:

1. Cartesian Hessian at the current geometry — `frequencies.rs` already
   produces one; it must be callable mid-optimization, not only at a minimum.
2. Project out translations and rotations (6, or 5 if linear —
   `detect_linear` exists) before diagonalizing, or the near-zero modes
   contaminate the eigenvector selection.
3. Mode selection: follow the lowest eigenvalue by default; a
   `follow_mode: usize` knob for when the lowest is not the reaction
   coordinate.
4. The P-RFO step: two separate RFO partitions, one maximizing along the
   followed mode, one minimizing in its orthogonal complement.
5. Hessian update between steps (Bofill is the usual choice for saddles —
   it does NOT preserve positive definiteness, which is the point).
6. Convergence: gradient norm AND **exactly one** imaginary frequency. A
   "converged" saddle with zero or two imaginary modes is not a transition
   state, and this must be a hard check, not a warning.

Cost note: step 1 is the expensive part. The Hessian is finite-differenced
from analytic gradients, so it is 6N gradient evaluations per Hessian (MEASURED
via `n_gradient_evaluations`: H2 = 12, water = 18 — exactly 6N, not 6N+1).
Rebuilding it every step is not affordable past a handful of atoms, which is
why step 5 matters.

## NOT A BLOCKER FOR THE OTHER FIVE

Every other use case has working code. The pipeline-level gaps recorded
elsewhere — that `context["geometry"]` is written by nothing, so tiers 3/4
score a gas-phase conformer rather than the docked pose — are about *wiring*,
not missing capability, and are tracked separately.

## COVERAGE, complete (updated 2026-09-19 after the viz work)

Every named use case now has code, a MEASURED cost, and a plot:

| use case | code | cost | plot |
|---|---|---|---|
| docking geom opt | yes | 1e-5 s/pose | `pose_ensemble`, `funnel_survival` |
| minima with FF | yes | 1e-3 s/pose | `tier_comparison` |
| minima with xtb | yes | 5e-1 s/pose | `tier_comparison` |
| transition state | yes | 2x6N + n_steps grads | `energy_profile` (barrier annotated) |
| common substitutions | yes | 2.8 ms enumerate, 214 ms embed | `site_substituent_heatmap`, `grid_with_scores` |
| toxicology | yes | 9.4 ms/molecule | `liability_profile` |
| binding energy in site | yes | tier 3/4 above | `site_substituent_heatmap` |

The costs are per-item; the campaign-level shares (cheap 0.8%, dock 73%,
xtb 4%, DFT 22%) are in the golden-path note, and they are the number that
should drive optimization decisions -- not the per-call cost.

**What "has a plot" does NOT mean.** The binding-energy row has a plot and a
cost and still cannot produce a trustworthy RANKING: all four pose protocols
are closed (RESULTS.md M4-M13) and the best available ddE noise is ~4.07
kcal/mol against effects of 1-2. `site_substituent_heatmap(noise_floor=...)`
greys out every cell inside that limit precisely so a figure cannot imply
otherwise.

## VISUALIZATION

Was absent entirely (`find tools experiments -iname '*vis*' -o -iname '*plot*'
-o -iname '*render*'` returned nothing). `tools/viz/` now covers energy
profiles, funnels, tier comparisons and 2-D depictions with substitution
highlighting. The energy-profile plot is what a transition-state search would
report against once it exists.
