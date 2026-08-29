# `scripts/danuglipron/` — toxicity-reduction campaign

Experiment asking whether other conformers or structural modifications of
danuglipron (PF-06882961) reduce toxicity liability while keeping GLP-1R
active-site fit.

| file | what |
|---|---|
| `PLAN.md` | The pre-registered design: hypotheses, arms, exactness anchors, and the **artifact hypothesis stated before measuring**. Deliberately NOT updated with results — rewriting a prediction after seeing the outcome is what a pre-registration exists to prevent. |
| `RESULTS.md` | Measurements, kept separate from interpretation. Read this first. |
| `run_arm_a_free.py` | Arm A: relaxes the 20 committed conformers in vacuum, finds the free-solution reference. |
| `run_fit.py` | Arms A/B/D: embed → align → score every candidate in the pocket field, then run the precision check and the metric gate. |
| `out/` | Gitignored scratch. Evidence promoted with `git add -f`. |

## Running it

Both drivers need `xtb` on `PATH` with libxtb resolvable from the multiarch
subdir, and `pdb2pqr30` for the pocket charges:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib:$LD_LIBRARY_PATH
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python scripts/danuglipron/run_arm_a_free.py   # ~3.5 min
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python scripts/danuglipron/run_fit.py          # ~25 min at N_CONF=40
```

`OMP_NUM_THREADS=1` is not optional: **libxtb is not thread-safe** (process-global
state). The engine parallelizes across processes, never threads.

`run_fit.py` refuses to optimize until `verify_xtb_build()` passes. gfortran 13.3
miscompiles xtb 6.7.1's GFN gradients at `-O3` while leaving **energies
byte-identical**, so only a geometry optimization can detect a bad build.

## Reading the output correctly

The drivers print two gates before any table, and **both must pass before any
ranking in that table means anything**:

1. **PRECISION CHECK** — is the candidate-to-candidate range larger than the
   resolution limit (2σ on the standard error of a difference)? Uses the **SEM**,
   which falls as 1/√n — never the pose range, which *grows* with n.
2. **METRIC GATE** — do the pharmacophore-breaking negative controls score
   clearly worse than the parent? A metric that cannot separate a
   known-inactive control licenses no ranking.

As of 2026-08-29 the metric gate **FAILS** (see `RESULTS.md` §M3), so the
candidate table is printed for transparency but **no candidate is recommended**.
A gate failure is a result, not a bug to be worked around.

## Where the code lives

`tools/tox` (external toxicity providers), `tools/morph` (analogue design +
embedding), `tools/campaign` (xtb engine, strain, fit, alignment, ranking).
Run their tests with:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/ -q
```
