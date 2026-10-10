#!/usr/bin/env python3
"""Write the pre-registered arm / b grids (see PREREGISTRATION.md)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R0S = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]  # Table-1 r0 points (Angstrom)
# r0*omega: None = published linked width (1/sqrt2, omega=None path);
# 2.0 = intermediate (Dutoi-safe bound 2.07); 4.0 = sharp.
SHARP = [None, 2.0, 4.0]
arms = [{"r0": r, "r0omega": w} for w in SHARP for r in R0S]
(HERE / "arms_primary.json").write_text(json.dumps({"arms": arms}, indent=1))
bs = [round(5.0 + 0.25 * i, 2) for i in range(int((20.0 - 5.0) / 0.25) + 1)]
(HERE / "bgrid.json").write_text(json.dumps({"bs": bs}))
(HERE / "arms_smoke.json").write_text(
    json.dumps({"arms": [{"r0": 1.0, "r0omega": None}, {"r0": 1.0, "r0omega": 4.0}]})
)
(HERE / "bgrid_smoke.json").write_text(json.dumps({"bs": [8.0, 11.0, 14.0]}))
print(len(arms), "arms;", len(bs), "b values", bs[0], "..", bs[-1])
