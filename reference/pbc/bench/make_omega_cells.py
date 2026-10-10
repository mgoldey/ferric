"""Extra Gamma-point cells for the RS-GDF omega study (issue #227): ionic, heavier-element and
sparse atomic crystals plus an expanded dry ice. Writes <name>.xyz (Angstrom) and <name>.lattice
(Bohr rows) next to this script, the same convention as make_geometries.py (whose cells --
diamond_prim, diamond_conv, dryice -- are copied here verbatim).

Run: python reference/pbc/bench/make_omega_cells.py   (pure numpy)
"""

from __future__ import annotations

import os

import numpy as np

BOHR = 0.52917721092
HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, symbols, cart_ang, lattice_ang, comment):
    with open(os.path.join(HERE, name + ".xyz"), "w") as f:
        f.write(f"{len(symbols)}\n{comment}; lattice (Bohr rows) in {name}.lattice\n")
        for s, r in zip(symbols, cart_ang):
            f.write(f"{s:2s} {r[0]:22.15f} {r[1]:22.15f} {r[2]:22.15f}\n")
    with open(os.path.join(HERE, name + ".lattice"), "w") as f:
        for row in np.asarray(lattice_ang) / BOHR:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")


def fcc_prim(a):
    return 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)


def rocksalt(name, cat, an, a):
    lat = fcc_prim(a)
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    write(
        name, [cat, an], frac @ lat, lat, f"{cat}{an} rocksalt a={a} A, primitive cell"
    )


def main():
    rocksalt("lif_prim", "Li", "F", 4.026)
    rocksalt("mgo_prim", "Mg", "O", 4.212)
    # silicon, diamond structure, a = 5.431 A
    a = 5.431
    lat = fcc_prim(a)
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    write(
        "si_prim", ["Si", "Si"], frac @ lat, lat, "Si diamond a=5.431 A, primitive cell"
    )
    # solid argon, fcc, a = 5.256 A (4 K), conventional cubic cell (4 atoms)
    a = 5.256
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    write(
        "ar_conv",
        ["Ar"] * 4,
        fcc * a,
        a * np.eye(3),
        "Ar fcc a=5.256 A, conventional cubic cell",
    )
    # dry ice with the LATTICE scaled by s and the CO2 molecules kept rigid (centres scale, bond
    # vectors do not): a density axis that leaves the chemistry fixed.
    s = 1.15
    lines = open(os.path.join(HERE, "dryice.xyz")).read().split("\n")
    n = int(lines[0])
    sym, xyz = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        sym.append(p[0])
        xyz.append([float(v) for v in p[1:4]])
    xyz = np.array(xyz)
    L = np.loadtxt(os.path.join(HERE, "dryice.lattice")) * BOHR
    out = xyz.copy()
    cs = [i for i in range(n) if sym[i] == "C"]
    for i in range(n):
        if sym[i] != "O":
            out[i] = None
            continue
        best = None
        for c in cs:
            for t in np.ndindex(3, 3, 3):
                d = xyz[i] - xyz[c] - (np.array(t) - 1) @ L
                if best is None or np.linalg.norm(d) < best[0]:
                    best = (np.linalg.norm(d), c, d)
        out[i] = s * xyz[best[1]] + best[2]
    for c in cs:
        out[c] = s * xyz[c]
    write("dryice_x115", sym, out, L * s, "dry ice, lattice x1.15, rigid CO2")


if __name__ == "__main__":
    main()
