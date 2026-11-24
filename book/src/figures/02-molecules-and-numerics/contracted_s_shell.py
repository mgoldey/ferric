"""Figure for book section 2.2: the STO-3G hydrogen 1s contraction.

Reads the bundled BSE-JSON set crates/ferric-core/src/basis/bundled/sto-3g.json,
renormalizes the contraction to unit self-overlap exactly as
ferric-core/src/basis.rs::renormalize_contraction does, and plots the radial
function against the hydrogen-atom 1s function exp(-r)/sqrt(pi). No ferric run.

    python3 book/src/figures/02-molecules-and-numerics/contracted_s_shell.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "ferric-book"
matplotlib.rcParams["svg.fonttype"] = "none"

here = Path(__file__).resolve().parent
root = here.parents[3]
data = json.loads(
    (root / "crates/ferric-core/src/basis/bundled/sto-3g.json").read_text()
)
shell = data["elements"]["1"]["electron_shells"][0]
alpha = np.array([float(x) for x in shell["exponents"]])
coef = np.array([float(x) for x in shell["coefficients"][0]])
lmom = shell["angular_momentum"][0]
assert lmom == 0

# Self-overlap of unit-normalized primitives: (2 sqrt(a b)/(a+b))^(l+3/2).
ov = (2 * np.sqrt(np.outer(alpha, alpha)) / np.add.outer(alpha, alpha)) ** (lmom + 1.5)
s_raw = coef @ ov @ coef
coef = coef / np.sqrt(s_raw)
assert abs(coef @ ov @ coef - 1.0) < 1e-12

r = np.linspace(0.0, 6.0, 1201)
prim = (2 * alpha[:, None] / np.pi) ** 0.75 * np.exp(-alpha[:, None] * r**2)
contracted = (coef[:, None] * prim).sum(axis=0)
hydrogen = np.exp(-r) / np.sqrt(np.pi)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
for ax, log in zip(axes, (False, True)):
    for k in range(len(alpha)):
        ax.plot(r, coef[k] * prim[k], lw=0.9, ls="--", label=f"c·g, α = {alpha[k]:.3f}")
    ax.plot(r, contracted, lw=2.0, color="black", label="contraction")
    ax.plot(r, hydrogen, lw=1.2, color="tab:red", label="exp(-r)/√π (exact H 1s)")
    ax.set_xlabel("r (Bohr)")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_title("log scale: tail")
    else:
        ax.set_title("linear scale: no cusp at r = 0")
        ax.set_ylabel("radial function (Bohr$^{-3/2}$)")
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle(
    "STO-3G hydrogen 1s, unit self-overlap after renormalization "
    f"(raw sum c c S = {s_raw:.6f})",
    fontsize=8,
)
fig.tight_layout()
fig.savefig(
    here / "contracted_s_shell.svg",
    format="svg",
    metadata={
        "Date": None,
        "Title": "STO-3G H 1s contraction (analytic, no ferric run)",
    },
)
