"""Visualization for pipeline outputs: energy plots and molecule depictions.

Two modules, split by what they need:

- `energy_plots` — matplotlib figures for funnels, reaction/scan paths,
  tier-vs-tier comparisons, (substituent, site) ddE heatmaps, pose ensembles
  and liability profiles. Needs matplotlib only.
- `molecules` — 2-D structure depictions with substitution highlighting. Needs
  RDKit, which lives in the `docking` extra.
- `report` — `campaign_report(...)` builds the whole figure set from what a
  substitution campaign produces, and attaches the caveats the measurements
  demand. Start here; it composes `energy_plots` (NOT `molecules`, whose
  depictions need a structure rather than a campaign's numbers).

All three are PRESENTATION ONLY. Neither computes a chemical quantity, and neither
re-derives a value it was handed: a plotting layer that quietly recomputes
something becomes a second, unvalidated implementation of it.

The shared convention across all of them, and the reason most of the code here is
input validation rather than drawing: an unevaluated result is rendered as a
GAP, never as zero. A figure is read faster than a table and trusted more, so
"this tier did not run" and "this tier returned 0.0" must not look alike.
"""

from __future__ import annotations

__all__ = ["energy_plots", "molecules", "report"]
