"""Figure 9.1: the default Kohn-Sham grid of one O and one H atom.

The Treutler-Ahlrichs M4 radial shells and the NWChem-region angular order of
each shell on the default production grid (75 x 110, NwchemRadial pruning:
66 shells for O, 50 for H). The grid is transcribed from
crates/ferric-dft/src/radial.rs (treutler_ahlrichs_m4) and
crates/ferric-dft/src/prune.rs (region_orders, angular_orders_for_atom,
radial_shells); the script asserts that the transcription reproduces
ferric.dft_grid_point_count for water with the weight cut off, so the figure
cannot drift from the code silently.

Writes atomic_grid.svg next to this file and records ferric.__build__ in its
title. Run (a second):
  box-lease run --label book-fig -- uv run --no-sync python book/src/figures/09-kohn-sham/atomic_grid.py
"""
# SPDX-License-Identifier: MIT OR Apache-2.0

import math
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

import ferric  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "ferric-book"

ROOT = Path(__file__).resolve().parents[4]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
ANGSTROM_TO_BOHR = 1.0 / 0.529_177_210_92

# radial.rs TA_XI, becke.rs BRAGG_ANGSTROM, prune.rs region_alphas (H, O only)
XI = {1: 0.8, 8: 0.9}
BRAGG = {1: 0.35, 8: 0.60}
ALPHAS = {1: (0.25, 0.5, 1.0, 4.5), 8: (0.1667, 0.5, 0.9, 3.5)}
# prune.rs region_orders(110): NWChem [50, 86, 86, 110, 86], inner -> 26, rest snapped up
ORDERS_110 = (26, 110, 110, 110, 110)


def ta_m4(xi, n):
    rs = []
    for k in range(1, n + 1):
        x = math.cos(math.pi * k / (n + 1))
        rs.append(-(xi / math.log(2)) * (1 + x) ** 0.6 * math.log((1 - x) / 2))
    return rs[::-1]


def radial_shells(z, n_radial=75):
    m = 50 if z <= 2 else 66 if z <= 10 else 75
    return max((n_radial * m + 37) // 75, 1) if m != 75 else n_radial


def shells(z):
    rs = ta_m4(XI[z], radial_shells(z))
    r_atom = BRAGG[z] * ANGSTROM_TO_BOHR
    return [(r, ORDERS_110[sum(r / r_atom > a for a in ALPHAS[z])]) for r in rs]


def main() -> None:
    o, h = shells(8), shells(1)
    mine = sum(n for _, n in o) + 2 * sum(n for _, n in h)
    mol = ferric.Molecule.from_xyz(str(WATER))
    code = ferric.dft_grid_point_count(mol, grid_weight_cut=0.0)
    flat = ferric.dft_grid_point_count(mol, grid_prune="none", grid_weight_cut=0.0)
    if mine != code:
        raise SystemExit(f"transcribed grid has {mine} points, ferric builds {code}")
    print(f"water: pruned {code} points (flat 75 x 110: {flat})")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for label, pts, z, mk, dy in (
        ("O (66 shells)", o, 8, "o", 0),
        ("H (50 shells)", h, 1, "s", 6),
    ):
        ax.semilogx(
            [r for r, _ in pts],
            [n + dy for _, n in pts],
            mk,
            ms=3.5,
            label=label,
            mfc="none",
        )
        r_atom = BRAGG[z] * ANGSTROM_TO_BOHR
        for a in ALPHAS[z]:
            ax.axvline(a * r_atom, color="C0" if z == 8 else "C1", lw=0.6, ls=":")
    ax.set_xlabel("radius $r$ of the shell (Bohr)")
    ax.set_ylabel("Lebedev order of the shell")
    ax.set_yticks([26, 110])
    ax.set_ylim(0, 135)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    ax.set_title(
        f"default grid: water has {code} points pruned, {flat} flat (weight cut off); "
        "dotted: region boundaries",
        fontsize=9,
    )
    ax.text(
        0.02,
        0.06,
        "H points offset by +6 for legibility",
        transform=ax.transAxes,
        fontsize=7,
    )
    fig.tight_layout()
    b = ferric.__build__
    title = (
        f"atomic_grid.py; ferric {b['git_sha'][:12]}{' dirty' if b['dirty'] else ''}"
    )
    fig.savefig(
        Path(__file__).with_suffix(".svg"), metadata={"Date": None, "Title": title}
    )


if __name__ == "__main__":
    main()
