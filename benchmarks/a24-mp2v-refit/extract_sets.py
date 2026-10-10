#!/usr/bin/env python3
"""Extract A24 / S22 / S66 geometries + CCSD(T)/CBS references from the psi4
database modules into a small committed JSON, recording provenance.

Usage:  extract_sets.py <dir containing A24.py S22.py S66.py> [out.json]

Reference sources (read from the psi4 modules' own comments, not assumed):
  A24  BIND[...]       Rezac & Hobza, JCTC 9, 2151 (2013), dx.doi.org/10.1021/ct400057w
  S22  BIND_S22B[...]  S22B, Marshall/Sherrill/Hobza revision (CCSD(T)/CBS, JCP 135, 194102 (2011))
  S66  BIND[...]       Rezac, Riley, Hobza, JCTC 7, 2427 (2011) (CCSD(T)/CBS)
Geometries are the psi4 qcdb.Molecule dimer blocks, split on "--" into the two
fragments, Angstrom, as published. No re-optimisation.
"""

import json
import re
import sys
from pathlib import Path

SETS = {
    "a24": ("A24.py", r"BIND\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?[\d.]+)"),
    "s22": (
        "S22.py",
        r"BIND_S22B\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?[\d.]+)",
    ),
    "s66": (
        "S66.py",
        r"BIND\['%s-%s'\s*%\s*\(dbse,\s*'(\d+)'\s*\)\]\s*=\s*(-?[\d.]+)",
    ),
}
GEO_PAT = (
    r"GEOS\['%s-%s-dimer' % \(dbse, '?(\d+)'?\)\] = qcdb\.Molecule\(\"\"\"(.*?)\"\"\"\)"
)
ELEM = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Ar", "Cl", "S", "P", "Si"}


def parse(path, bind_re):
    src = Path(path).read_text()
    geos = {}
    for m in re.finditer(GEO_PAT, src, re.S):
        idx, body = int(m.group(1)), m.group(2)
        frags = []
        for part in body.split("--"):
            atoms = []
            for line in part.splitlines():
                t = line.split()
                if len(t) == 4 and t[0].capitalize() in ELEM:
                    atoms.append(
                        [t[0].capitalize(), float(t[1]), float(t[2]), float(t[3])]
                    )
            if atoms:
                frags.append(atoms)
        assert len(frags) == 2, f"{path} {idx}: {len(frags)} fragments"
        geos[idx] = frags
    refs = {int(m.group(1)): float(m.group(2)) for m in re.finditer(bind_re, src)}
    assert set(geos) == set(refs), (
        f"{path}: geometry ids {sorted(geos)} != reference ids {sorted(refs)}"
    )
    return geos, refs, None


def main():
    d = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "sets.json"
    res = {}
    for name, (fn, rx) in SETS.items():
        geos, refs, sha = parse(d / fn, rx)
        res[name] = {
            "source_file": fn,
            "n": len(geos),
            "systems": {
                str(i): {"frag_a": geos[i][0], "frag_b": geos[i][1], "ref": refs[i]}
                for i in sorted(geos)
            },
        }
    out.write_text(json.dumps(res, indent=0))
    print({k: v["n"] for k, v in res.items()}, "->", out)


if __name__ == "__main__":
    main()
