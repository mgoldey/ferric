"""
Generate ferric's Lebedev-Laikov parameter tables and their PySCF reference.

Source: PySCF's compiled Lebedev-Laikov generator,
`pyscf.dft.gen_grid.MakeAngularGrid(n)` (PySCF is Apache-2.0, compatible with
ferric's MIT OR Apache-2.0). Only its OUTPUT is used: the nodes and weights of
each rule are grouped into O_h orbits and one (class, weight, parameters) row
is written per orbit. No code is copied. The values are the published
parameters of Lebedev & Laikov, Doklady Math. 59, 477 (1999).

Writes two files:

* `crates/ferric-quadrature/src/lebedev/tables.rs` -- the orbit parameters
  of every order ferric's hand-written arms in `lebedev.rs` do not cover
  (`HAND_WRITTEN` below), as const arrays expanded at runtime by the
  `class_*` generators.
* `testdata/reference/lebedev_pyscf_orbits.json` -- for EVERY order, PySCF's
  node set reduced to one canonical representative per orbit
  (|x| >= |y| >= |z|, from PySCF's own points, not from the classification
  above), with the orbit's weight and point count. A rule whose points form
  whole O_h orbits with one weight per orbit is fully described by that list,
  so comparing it is comparing the sorted point and weight sets; the script
  asserts that precondition for every PySCF rule.

PySCF's weights already sum to 1 (ferric's normalisation), so they are used
unscaled.

Usage (from the repo root, with the venv's PySCF), then format the table:
  python scripts/gen_lebedev_tables.py
  cargo fmt -p ferric-quadrature
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf.dft import gen_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_OUT = os.path.join(ROOT, "crates/ferric-quadrature/src/lebedev/tables.rs")
JSON_OUT = os.path.join(ROOT, "testdata/reference/lebedev_pyscf_orbits.json")

# Orders lebedev.rs builds with hand-written match arms.
HAND_WRITTEN = {6, 14, 26, 50, 110, 194, 302, 434, 590}
# Same-orbit tolerance on canonical coordinates. Distinct orbits of every rule
# differ by > 1e-6; one orbit's points agree to a few ulp.
TOL = 1e-12
ORBIT_SIZE = {"a1": 6, "a2": 8, "a3": 12, "b": 24, "c": 24, "d": 48}


def degree_by_order():
    return {n: d for d, n in gen_grid.LEBEDEV_ORDER.items() if n > 1}


def orbits(order):
    """PySCF's rule `order` as a list of (canonical (a,b,c), weight, count)."""
    g = gen_grid.MakeAngularGrid(order)
    assert g.shape == (order, 4), g.shape
    canon = -np.sort(-np.abs(g[:, :3]), axis=1)
    groups = defaultdict(list)
    for row, w in zip(canon, g[:, 3]):
        key = tuple(np.round(row / TOL).astype(np.int64))
        groups[key].append((row, w))
    out = []
    for members in groups.values():
        rows = np.array([m[0] for m in members])
        ws = np.array([m[1] for m in members])
        assert np.ptp(rows, axis=0).max() < 1e-15, (order, rows)
        assert ws.min() == ws.max(), (order, ws)
        out.append((tuple(float(v) for v in rows[0]), float(ws[0]), len(members)))
    out.sort()
    assert sum(o[2] for o in out) == order
    assert abs(sum(o[1] * o[2] for o in out) - 1.0) < 1e-14, order
    return out


def close(x, y):
    return abs(x - y) < TOL


def classify(abc):
    """Lebedev class of a canonical representative a >= b >= c >= 0."""
    a, b, c = abc
    if close(b, 0.0):
        assert close(a, 1.0), abc
        return "a1"
    if close(a, b) and close(b, c):
        return "a2"
    if close(c, 0.0):
        return "a3" if close(a, b) else "c"
    if close(a, b) or close(b, c):
        return "b"
    return "d"


def parameters(cls, abc):
    """Generator parameters of one orbit, chosen so that the generator's
    recomputed coordinate (sqrt of 1 minus the others' squares) lands closest
    to PySCF's value."""
    a, b, c = abc
    if cls == "b":
        # (p, p, q) with q = sqrt(1 - 2 p^2): p is the repeated value.
        return (b,)
    if cls == "c":
        # (p, q, 0) with q = sqrt(1 - p^2): either value can be p.
        p, _ = min(
            [(a, b), (b, a)],
            key=lambda o: (abs(math.sqrt(max(0.0, 1.0 - o[0] ** 2)) - o[1]), o),
        )
        return (p,)
    if cls == "d":
        # (p, q, r) with r = sqrt(1 - p^2 - q^2): pick which value is r.
        options = [((b, c), a), ((a, c), b), ((a, b), c)]

        def err(opt):
            (p, q), r = opt
            return abs(math.sqrt(max(0.0, 1.0 - p * p - q * q)) - r)

        return min(options, key=lambda o: (err(o), o))[0]
    return ()


def lit(x):
    s = repr(float(x))
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def rust_rule(order, degree, orbs):
    by = defaultdict(list)
    for abc, w, count in orbs:
        cls = classify(abc)
        assert count == ORBIT_SIZE[cls], (order, cls, abc, count)
        by[cls].append((w, parameters(cls, abc)))
    for k in ("a1", "a2", "a3"):
        assert len(by[k]) <= 1, (order, k)

    def single(k):
        return f"Some({lit(by[k][0][0])})" if by[k] else "None"

    def rows(k):
        return ", ".join(f"[{', '.join(lit(v) for v in (w, *p))}]" for w, p in by[k])

    counts = "+".join(f"{len(by[k])}{k}" for k in ORBIT_SIZE if by[k])
    return (
        f"    // {order} points, degree {degree}: {counts}.\n"
        f"    Rule {{\n"
        f"        order: {order},\n"
        f"        degree: {degree},\n"
        f"        a1: {single('a1')},\n"
        f"        a2: {single('a2')},\n"
        f"        a3: {single('a3')},\n"
        f"        b: &[{rows('b')}],\n"
        f"        c: &[{rows('c')}],\n"
        f"        d: &[{rows('d')}],\n"
        f"    }},\n"
    )


def main():
    degrees = degree_by_order()
    all_orders = sorted(degrees)
    reference = {
        "source": f"pyscf.dft.gen_grid.MakeAngularGrid, PySCF {pyscf_version}",
        "generator": "scripts/gen_lebedev_tables.py",
        "layout": "per order: degree and orbits [a, b, c, weight, count] with a >= b >= c >= 0 "
        "taken from PySCF's points; weights sum to 1",
        "orders": {},
    }
    table_orders = [n for n in all_orders if n not in HAND_WRITTEN]
    rust_rules = []
    for n in all_orders:
        orbs = orbits(n)
        reference["orders"][str(n)] = {
            "degree": degrees[n],
            "orbits": [[*abc, w, count] for abc, w, count in orbs],
        }
        if n in table_orders:
            rust_rules.append(rust_rule(n, degrees[n], orbs))

    header = (
        "//! Lebedev-Laikov orbit parameters for the orders `lebedev.rs` does not\n"
        "//! build by hand. GENERATED by `scripts/gen_lebedev_tables.py` from the\n"
        f"//! output of PySCF {pyscf_version}'s `pyscf.dft.gen_grid.MakeAngularGrid`\n"
        "//! (provenance and licence in the `lebedev` module docs); do not edit.\n"
        "//!\n"
        "//! Rows: `b`, `c` = `[weight, p]`; `d` = `[weight, p, q]` (see `class_b`,\n"
        "//! `class_c`, `class_d`). Weights sum to 1.\n\n"
        "use super::Rule;\n\n"
        f"pub(super) const RULES: [Rule; {len(rust_rules)}] = [\n"
    )
    with open(RUST_OUT, "w") as f:
        f.write(header + "".join(rust_rules) + "];\n")
    with open(JSON_OUT, "w") as f:
        json.dump(reference, f, indent=None, separators=(",", ":"))
        f.write("\n")
    print(
        f"wrote {RUST_OUT} ({len(rust_rules)} rules) and {JSON_OUT} ({len(all_orders)} orders)"
    )


if __name__ == "__main__":
    main()
