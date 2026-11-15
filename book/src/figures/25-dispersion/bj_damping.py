"""Figure 25.1: Becke-Johnson damping of the D3 two-body term.

Analytic only (no ferric run). For an element pair the BJ radius is
R0 = sqrt(3 q_AB), q_AB = sqrt(Q_A Q_B), Q_A = sqrt(Z_A) <r^4>/<r^2>_A / 2
(ferric-d3 lib.rs sqrt_q, with <r^4>/<r^2> read from the committed table
R4R2 in crates/ferric-d3/src/tables.rs), and f = a1 R0 + a2 with the PBE
parameters of crates/ferric-d3/src/params.rs (a1 = 0.4289, a2 = 4.4407 Bohr).
The plot shows the damped multipliers R^n / (R^n + f^n), n = 6 and 8, that
turn the bare C6/R^6 and C8/R^8 into the BJ energy, for H-H, C-C and C-H.

Writes bj_damping.svg next to this file.
Run: python book/src/figures/25-dispersion/bj_damping.py
"""
# SPDX-License-Identifier: MIT OR Apache-2.0

import re
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ferric-book"

ROOT = Path(__file__).resolve().parents[4]
TABLES = ROOT / "crates" / "ferric-d3" / "src" / "tables.rs"
A1, A2 = 0.4289, 4.4407  # PBE row of params.rs
BOHR_PER_ANG = 1.0 / 0.529177210903


def r4r2() -> list[float]:
    src = TABLES.read_text()
    body = re.search(r"pub static R4R2: \[f64; 118\] = \[(.*?)\];", src, re.S)
    return [float(v) for v in re.findall(r"[-+]?\d+\.\d*(?:e[-+]?\d+)?", body.group(1))]


def damping_radius(za: int, zb: int, table: list[float]) -> float:
    qa = 0.5 * table[za - 1] * sqrt(za)
    qb = 0.5 * table[zb - 1] * sqrt(zb)
    return A1 * sqrt(3.0 * sqrt(qa * qb)) + A2


def main() -> None:
    table = r4r2()
    rs = [0.5 + 11.5 * k / 600 for k in range(601)]  # Bohr
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (za, zb, name), color in zip(
        [(1, 1, "H–H"), (6, 1, "C–H"), (6, 6, "C–C")], ["C0", "C1", "C2"]
    ):
        f = damping_radius(za, zb, table)
        ax.plot(
            rs,
            [r**6 / (r**6 + f**6) for r in rs],
            color=color,
            label=f"{name}, $n$ = 6 ($f$ = {f:.2f} bohr)",
        )
        ax.plot(
            rs,
            [r**8 / (r**8 + f**8) for r in rs],
            "--",
            color=color,
            label=f"{name}, $n$ = 8",
        )
    ax.set_xlabel("R / bohr")
    ax.set_ylabel(r"$R^n/(R^n + f^n)$")
    ax.set_xlim(0.5, 12.0)
    ax.set_ylim(-0.02, 1.02)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda b: b / BOHR_PER_ANG, lambda a: a * BOHR_PER_ANG)
    )
    secax.set_xlabel("R / Å")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = Path(__file__).with_suffix(".svg")
    fig.savefig(out, metadata={"Date": None, "Title": "bj_damping (analytic, PBE)"})


if __name__ == "__main__":
    main()
