#!/usr/bin/env python3
"""Generate physically correct n-alkane geometries (C_n H_{2n+2}).

Replaces the previous testdata/molecules/alkane_*.xyz files, which were
ENTIRELY PLANAR (every z = 0) with 90/180 degree bond angles instead of the
tetrahedral 109.47, and H-H contacts at 1.540 A (normal >= 1.8 A). Each carbon
was a flat cross with two hydrogens at 180 degrees -- impossible for sp3 carbon.
That bad geometry produced near-degenerate frontier sigma states and multiple
close-lying SCF solutions, which manifested as spurious "SCF convergence
problems" (C16 failing to converge in 100 iterations, C20 needing 425).

Builds the standard all-anti (zig-zag) conformer:
  - C-C 1.526 A, C-H 1.094 A
  - tetrahedral angles (109.47 deg)
  - backbone zig-zags in the xy-plane; methylene H's straddle +/- z

Usage:  python3 scripts/gen_alkane_geometries.py [--check-only]
"""
import argparse
import math
import os

CC = 1.526          # C-C bond length (A)
CH = 1.094          # C-H bond length (A)
TET = math.radians(109.47)

HERE = os.path.dirname(os.path.abspath(__file__))
MOLDIR = os.path.join(HERE, "..", "testdata", "molecules")


def backbone(n):
    """All-anti carbon backbone zig-zagging in the xy-plane."""
    half = TET / 2.0
    dx = CC * math.sin(half)      # advance along the chain axis
    dy = CC * math.cos(half)      # alternating transverse displacement
    return [(i * dx, (i % 2) * dy, 0.0) for i in range(n)]


def unit(v):
    m = math.sqrt(sum(c * c for c in v))
    return tuple(c / m for c in v)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(v, s):
    return tuple(c * s for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def methylene_hydrogens(c_prev, c, c_next):
    """Two H on an interior carbon: bisector in-plane, H's out of plane (+/- z)."""
    b1 = unit(sub(c_prev, c))
    b2 = unit(sub(c_next, c))
    bisect = unit(add(b1, b2))          # points "inward"; H's go opposite
    perp = unit(cross(b1, b2))          # normal to the C-C-C plane
    # H directions: -bisector tilted +/- out of plane by half the H-C-H angle.
    half = TET / 2.0
    d1 = unit(add(scale(bisect, -math.cos(half)), scale(perp, math.sin(half))))
    d2 = unit(add(scale(bisect, -math.cos(half)), scale(perp, -math.sin(half))))
    return [add(c, scale(d1, CH)), add(c, scale(d2, CH))]


def terminal_hydrogens(c, c_nbr, n_h=3):
    """n_h hydrogens on a terminal carbon, staggered about the C-C axis."""
    axis = unit(sub(c, c_nbr))          # points away from the chain
    # Build an orthonormal frame perpendicular to `axis`.
    tmp = (0.0, 0.0, 1.0)
    if abs(axis[2]) > 0.9:
        tmp = (1.0, 0.0, 0.0)
    u = unit(cross(axis, tmp))
    v = cross(axis, u)
    beta = math.pi - TET                # tilt from the axis
    out = []
    for k in range(n_h):
        phi = 2.0 * math.pi * k / n_h
        radial = add(scale(u, math.cos(phi)), scale(v, math.sin(phi)))
        d = unit(add(scale(axis, math.cos(beta)), scale(radial, math.sin(beta))))
        out.append(add(c, scale(d, CH)))
    return out


def build(n):
    cs = backbone(n)
    atoms = [("C", p) for p in cs]
    if n == 1:
        # Methane: 4 H tetrahedrally about the origin.
        for p in terminal_hydrogens(cs[0], (cs[0][0] - 1.0, cs[0][1], cs[0][2]), 4):
            atoms.append(("H", p))
        return atoms
    for i, c in enumerate(cs):
        if i == 0:
            hs = terminal_hydrogens(c, cs[1], 3)
        elif i == n - 1:
            hs = terminal_hydrogens(c, cs[n - 2], 3)
        else:
            hs = methylene_hydrogens(cs[i - 1], c, cs[i + 1])
        atoms.extend(("H", p) for p in hs)
    return atoms


def validate(n, atoms):
    """Sanity checks: formula, planarity, and closest contacts."""
    nc = sum(1 for s, _ in atoms if s == "C")
    nh = sum(1 for s, _ in atoms if s == "H")
    assert nc == n, f"C{n}: got {nc} carbons"
    assert nh == 2 * n + 2, f"C{n}: expected {2*n+2} H, got {nh}"
    zs = [p[2] for _, p in atoms]
    nonplanar = max(zs) - min(zs)
    dmin = min(
        math.dist(atoms[i][1], atoms[j][1])
        for i in range(len(atoms)) for j in range(i + 1, len(atoms))
    )
    assert nonplanar > 0.5, f"C{n}: still planar (z-spread {nonplanar:.3f})"
    assert dmin > 1.0, f"C{n}: atoms too close ({dmin:.3f} A)"
    return nonplanar, dmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="validate without writing files")
    args = ap.parse_args()

    for n in range(1, 21):
        atoms = build(n)
        spread, dmin = validate(n, atoms)
        path = os.path.join(MOLDIR, f"alkane_{n}.xyz")
        if not args.check_only:
            with open(path, "w") as fh:
                fh.write(f"{len(atoms)}\n")
                fh.write(f"n-alkane C{n}H{2*n+2}, all-anti conformer "
                         f"(C-C {CC} A, C-H {CH} A, tetrahedral)\n")
                for sym, (x, y, z) in atoms:
                    fh.write(f"{sym:<3}{x:>12.6f}{y:>12.6f}{z:>12.6f}\n")
        print(f"C{n:<3} atoms={len(atoms):<3} z-spread={spread:.3f}  min-dist={dmin:.3f}")


if __name__ == "__main__":
    main()
