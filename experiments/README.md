# `experiments/` — campaigns, not libraries

This directory holds **specific scientific investigations**. Each subdirectory is
one campaign: its hypotheses, its drivers, its results, and its evidence.

## The boundary

| | `tools/` | `experiments/` |
|---|---|---|
| contains | reusable machinery | one campaign's hypotheses and findings |
| knows about | molecules in general | *this* molecule, *this* target |
| lifetime | as long as it is useful | permanent record of what was measured |
| when it changes | to fix or extend a capability | when new measurements land |

A rule that keeps the split honest: **`tools/` must not import from
`experiments/`**, and must not contain a named molecule, target, or hypothesis.
The reverse direction is expected — a campaign imports the machinery it needs.

`tools/` docstrings *do* cite campaign measurements ("measured 2026-08-29 on the
danuglipron ensemble: ..."), and that is deliberate. A library rule justified by
a real observation is far more useful than an unattributed assertion, and the
citation is provenance, not a dependency.

## Layout of a campaign

```
experiments/<name>/
  PLAN.md       pre-registered design: hypotheses, arms, exactness anchors, and
                the artifact hypothesis stated BEFORE measuring. Not updated
                with results -- rewriting a prediction after seeing the outcome
                is what a pre-registration exists to prevent.
  RESULTS.md    measurements, kept separate from interpretation. Read first.
  README.md     how to re-run it, and how to read its gates.
  design.py     this campaign's hypothesis set (structures, constraints).
  run_*.py      the drivers.
  tests/        tests of THIS campaign's designs, not of the machinery.
  out/          gitignored scratch; evidence promoted with `git add -f`.
```

## Current campaigns

- **`danuglipron/`** — can conformers or structural changes to danuglipron
  (PF-06882961) reduce toxicity liability while keeping GLP-1R active-site fit?
  Reached a definitive **negative**: candidate pose generation failed (no
  generated conformer reaches the bound pose within 2.0 Å), so no fit ranking is
  licensed. The strain result is independent of that and stands. See its
  `RESULTS.md`.
