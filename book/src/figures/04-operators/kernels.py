"""Figure 4.1: the attenuated kernels r*w(r) for erf/erfc and terf/terfc.

Analytic only (no ferric run). Writes kernels.svg next to this file.
Run: uv run --no-sync python book/src/figures/04-operators/kernels.py
"""
# SPDX-License-Identifier: MIT OR Apache-2.0

from math import erf, exp, pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "ferric-book"

R0 = 1.0  # Bohr; terf width tied by the curvature constraint
OMEGA_T = 1.0 / (R0 * sqrt(2.0))
OMEGA_E = OMEGA_T  # erf at the same omega, for comparison


def terf(r: float, r0: float, w: float) -> float:
    return 0.5 * (erf(w * (r - r0)) + erf(w * (r + r0)))


def terf_over_r_at_0(r0: float, w: float) -> float:
    """lim r->0 of terf(r)/r = (2w/sqrt(pi)) exp(-w^2 r0^2)."""
    return 2.0 * w / sqrt(pi) * exp(-((w * r0) ** 2))


def main() -> None:
    rs = [i * 0.02 for i in range(0, 301)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax.plot(rs, [erf(OMEGA_E * r) for r in rs], label=r"erf$(\omega r)$")
    ax.plot(rs, [1 - erf(OMEGA_E * r) for r in rs], "--", label=r"erfc$(\omega r)$")
    ax.plot(rs, [terf(r, R0, OMEGA_T) for r in rs], label=r"terf$(r; r_0)$")
    ax.plot(rs, [1 - terf(r, R0, OMEGA_T) for r in rs], "--", label=r"terfc$(r; r_0)$")
    ax.axvline(R0, color="0.6", lw=0.8)
    ax.text(R0 + 0.05, 0.05, r"$r_0$", color="0.4")
    ax.set_xlabel(r"$r$ (Bohr)")
    ax.set_ylabel(r"$r\,\hat w(r)$")
    ax.set_title("each pair sums to 1", fontsize=10)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)

    # Long-range kernels themselves, normalized to their value at r = 0.
    small = [i * 0.01 for i in range(1, 201)]
    e0 = 2.0 * OMEGA_E / sqrt(pi)
    t0 = terf_over_r_at_0(R0, OMEGA_T)
    bx.plot(
        small, [erf(OMEGA_E * r) / r / e0 for r in small], label=r"erf$(\omega r)/r$"
    )
    bx.plot(
        small, [terf(r, R0, OMEGA_T) / r / t0 for r in small], label=r"terf$(r; r_0)/r$"
    )
    bx.set_xlabel(r"$r$ (Bohr)")
    bx.set_ylabel(r"$\hat w(r) / \hat w(0)$")
    bx.set_title(r"terf$/r$ is flat to $O(r^4)$ at the origin", fontsize=10)
    bx.set_xlim(0, 2)
    bx.legend(frameon=False, fontsize=8)
    fig.suptitle(r"$r_0 = 1$ Bohr, $\omega = 1/(\sqrt{2}\,r_0)$", fontsize=10)
    fig.tight_layout()
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(out, metadata={"Date": None, "Title": "kernels.py (analytic)"})


if __name__ == "__main__":
    main()
