"""Danuglipron toxicity-reduction campaign: strain, fit, and liability.

See `experiments/danuglipron/PLAN.md` for the experiment design. This package holds
the measurement drivers:

- `xtb_engine.py` — GFN2-xTB via the `xtb` CLI (separate processes, because
  libxtb is not thread-safe). The cheap tier: conformer energies and relaxations.
- `strain.py`     — Arm A: bound-vs-free conformer strain penalty.
- `fit.py`        — Arms A/B: pocket electrostatic complementarity for a pose.
- `rank.py`       — Arm D: Pareto ranking over (liability, fit loss, strain).
"""
