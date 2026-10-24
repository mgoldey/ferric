"""Write the real-size benchmark geometries (FINDINGS "Real-size benchmark plan — 2026-09-25").

For every cell this writes, next to this script:
  <name>.xyz          atoms of the reference cell, ANGSTROM (ferric's XYZ convention; [molecule] xyz)
  <name>.lattice      lattice vectors as ROWS, BOHR, 17 significant digits (the [cell] lattice, unit = "bohr")

Both codes read these two files: the ferric TOMLs embed the .lattice rows, and pyscf_bench.py
reads the .xyz + .lattice and converts Angstrom -> Bohr with ferric's constant 1/0.52917721092
(== pyscf.data.nist.BOHR), so the two codes see bit-identical nuclear positions.

Structures (experimental; see the FINDINGS section for the citations and the caveats):
  diamond  Fd-3m, a = 3.567 A. Primitive fcc cell (2 C), conventional cube (8 C) and the
           2x2x2 supercell of the primitive cell (16 C; its Gamma point == the 2x2x2 k-mesh
           of the primitive cell, an exact identity used as a correctness anchor at scale).
  dry ice  CO2, Pa-3, a = 5.624 A, C on 4a (0,0,0), O on 8c (x,x,x) with x = 0.1185
           (Simon & Peters 1980, 150 K). 4 molecules / 12 atoms per cube, and the 1x1x2
           supercell (24 atoms).

It also writes the ferric CLI inputs toml/<run>.toml (the RUNS table below; [cell] jk = "rsgdf",
unit = "bohr", lattice copied from <cell>.lattice). Run from the worktree root: the TOMLs name the
XYZ by a repo-relative path.

Run:  python reference/pbc/bench/make_geometries.py   (pure numpy, < 1 s)
"""

from __future__ import annotations

import os

import numpy as np

BOHR = 0.52917721092  # Angstrom per Bohr, ferric-core mol.rs and pyscf.data.nist
HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, symbols, frac, lattice_ang, comment):
    lat = np.asarray(lattice_ang, float)
    cart = np.asarray(frac, float) @ lat  # Angstrom
    # sanity: no two atoms (incl. images) closer than 0.9 A
    dmin = np.inf
    for i in range(len(cart)):
        for j in range(len(cart)):
            for n in np.ndindex(3, 3, 3):
                t = (np.array(n) - 1) @ lat
                if i == j and not any(np.array(n) - 1):
                    continue
                dmin = min(dmin, np.linalg.norm(cart[i] - cart[j] - t))
    assert dmin > 0.9, (name, dmin)
    with open(os.path.join(HERE, name + ".xyz"), "w") as f:
        f.write(f"{len(symbols)}\n{comment}; lattice (Bohr rows) in {name}.lattice\n")
        for s, r in zip(symbols, cart):
            f.write(f"{s:2s} {r[0]:22.15f} {r[1]:22.15f} {r[2]:22.15f}\n")
    with open(os.path.join(HERE, name + ".lattice"), "w") as f:
        for row in lat / BOHR:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
    return dmin


def diamond():
    a = 3.567
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    out = {}
    # primitive: C at 0 and (1/4,1/4,1/4) conventional = (1/4,1/4,1/4) in primitive fractions
    out["diamond_prim"] = (["C", "C"], [[0, 0, 0], [0.25, 0.25, 0.25]], prim)
    fcc = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    basis = [[0, 0, 0], [0.25, 0.25, 0.25]]
    frac = [np.add(t, b) % 1.0 for t in fcc for b in basis]
    out["diamond_conv"] = (["C"] * 8, frac, a * np.eye(3))
    # 2x2x2 supercell of the primitive cell
    fr, sy = [], []
    for n in np.ndindex(2, 2, 2):
        for b in [[0, 0, 0], [0.25, 0.25, 0.25]]:
            fr.append((np.array(n) + b) / 2.0)
            sy.append("C")
    out["diamond_prim222"] = (sy, fr, 2.0 * prim)
    return out


def dry_ice():
    a, x = 5.624, 0.1185
    c = [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0]]
    o8c = [
        [x, x, x],
        [-x + 0.5, -x, x + 0.5],
        [-x, x + 0.5, -x + 0.5],
        [x + 0.5, -x + 0.5, -x],
        [-x, -x, -x],
        [x + 0.5, x, -x + 0.5],
        [x, -x + 0.5, x + 0.5],
        [-x + 0.5, x + 0.5, x],
    ]
    frac = [np.array(p) % 1.0 for p in c + o8c]
    sy = ["C"] * 4 + ["O"] * 8
    lat = a * np.eye(3)
    # every C must have exactly two O at the C=O distance sqrt(3) x a
    cart = np.array(frac) @ lat
    rco = np.sqrt(3) * x * a
    for i in range(4):
        d = []
        for j in range(4, 12):
            for n in np.ndindex(3, 3, 3):
                d.append(np.linalg.norm(cart[j] + (np.array(n) - 1) @ lat - cart[i]))
        d = np.sort(d)
        assert abs(d[0] - rco) < 1e-9 and abs(d[1] - rco) < 1e-9 and d[2] > 2.5, (
            i,
            d[:3],
        )
    out = {"dryice": (sy, frac, lat)}
    sup = np.diag([1.0, 1.0, 2.0])
    fr2 = [(np.array(f) + [0, 0, k]) @ np.linalg.inv(sup) for k in (0, 1) for f in frac]
    out["dryice_112"] = (sy * 2, fr2, sup @ lat)
    return out


# run name -> (cell, basis, aux, kind, functional, kmesh, note)
RUNS = {
    # ferric-side preflight: seconds-scale, pinned against PySCF (FINDINGS benchmark plan)
    "pre_diamond_prim_sto3g_ri_rhf": (
        "diamond_prim",
        "sto-3g",
        "cc-pvdz-ri",
        "rhf",
        None,
        None,
        "PREFLIGHT. PySCF 2.13 GDF p1e-12: -74.0034040291 (0/112 aux dropped)",
    ),
    "pre_diamond_prim_sto3g_jk_rhf": (
        "diamond_prim",
        "sto-3g",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "PREFLIGHT (exercises l = 4 aux). PySCF 2.13 GDF p1e-12, eig metric: "
        "-74.0023846489 (21/150 aux dropped)",
    ),
    "diamond_prim_rhf": (
        "diamond_prim",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "Gamma, 2 C, nao 28",
    ),
    "diamond_prim_k222_rhf": (
        "diamond_prim",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        [2, 2, 2],
        "2x2x2 k-mesh; E/cell must equal diamond_prim222_rhf E/8 (exact identity)",
    ),
    "diamond_conv_rhf": (
        "diamond_conv",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "Gamma, 8 C, nao 112",
    ),
    "diamond_conv_pbe": (
        "diamond_conv",
        "cc-pvdz",
        "def2-universal-jkfit",
        "ksdft",
        "PBE",
        None,
        "Gamma PBE, SSF 75x302",
    ),
    "dryice_rhf": (
        "dryice",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "Gamma, 4 CO2, nao 168",
    ),
    "dryice_pbe": (
        "dryice",
        "cc-pvdz",
        "def2-universal-jkfit",
        "ksdft",
        "PBE",
        None,
        "Gamma PBE, SSF 75x302",
    ),
    "diamond_prim222_rhf": (
        "diamond_prim222",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "Gamma, 16 C, nao 224 (= the k222 anchor)",
    ),
    "dryice_112_rhf": (
        "dryice_112",
        "cc-pvdz",
        "def2-universal-jkfit",
        "rhf",
        None,
        None,
        "Gamma, 8 CO2, nao 336 (top size)",
    ),
}


def write_tomls(budget_gb=10):
    os.makedirs(os.path.join(HERE, "toml"), exist_ok=True)
    for run, (cell, basis, aux, kind, xc, kmesh, note) in RUNS.items():
        rows = open(os.path.join(HERE, cell + ".lattice")).read().split("\n")
        rows = [
            [repr(float(v)) for v in r.split()] for r in rows if r.strip()
        ]  # always a float literal
        lat = ", ".join("[" + ", ".join(r) + "]" for r in rows)
        t = [
            f"# ferric PBC real-size benchmark: {run}. {note}.",
            "# Generated by reference/pbc/bench/make_geometries.py; run via run_ferric_bench.sh from the worktree root.",
            "# Energies are Hartree PER CELL. RS-GDF at its fixed defaults (omega 1, precision 1e-13, lindep 1e-10);",
            "# hcore precision 1e-14 with omega = min(2.5 sqrt(pi)/V^(1/3), 0.9636) (PeriodicHcoreConfig::for_cell). Nothing here is tuned for speed.",
            "",
            "[molecule]",
            f'xyz = "reference/pbc/bench/{cell}.xyz"',
            "",
            "[basis]",
            f'name = "{basis}"',
            "",
            "[method]",
            f'kind = "{kind}"',
            "",
        ]
        if xc:
            t += ["[dft]", f'functional = "{xc}"', ""]
        t += [
            "[cell]",
            'unit = "bohr"',
            f"lattice = [{lat}]",
            'exxdiv = "ewald"',
            'jk = "rsgdf"',
        ]
        # The SR column rotation became the default after these runs were planned (282be9cac). The
        # non-preflight runs are the UNROTATED baseline, so they pin it off; the preflights keep the default.
        pin = [] if run.startswith("pre_") else ["sr_column_rotation = false"]
        if kmesh:
            t += [f'auxbasis = "{aux}"', f"kmesh = {kmesh}", 'centring = "gamma"']
            t += [
                '# The unrotated baseline (the rotation is on by default at a kmesh with jk = "rsgdf").'
            ] + pin
        else:
            if pin:
                t += [
                    "# The unrotated baseline (the rotation is on by default for this run)."
                ] + pin
            t += [f'auxbasis = "{aux}"']
        if xc:
            t += ["n_radial = 75", "n_angular = 302"]
        t += ["", "[memory]", f"budget_gb = {budget_gb}", ""]
        with open(os.path.join(HERE, "toml", run + ".toml"), "w") as f:
            f.write("\n".join(t))


if __name__ == "__main__":
    for name, (sy, fr, lat) in {**diamond(), **dry_ice()}.items():
        dmin = write(name, sy, fr, lat, f"ferric PBC benchmark cell {name}")
        print(
            f"{name:18s} natoms {len(sy):3d}  volume {abs(np.linalg.det(lat)) / BOHR**3:10.3f} Bohr^3  "
            f"min interatomic {dmin:.4f} A"
        )
    write_tomls()
    print(f"wrote {len(RUNS)} TOMLs under {os.path.join(HERE, 'toml')}")
