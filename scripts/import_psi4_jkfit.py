#!/usr/bin/env python3
"""Import Weigend's aug-cc-pVXZ-JKFIT sets, which the Basis Set Exchange lacks.

BSE (package 0.12 and its REST API, checked 2026-10-29) carries cc-pVXZ-JKFIT
but no aug- variant. The data are Weigend's (PCCP 4, 4285 (2002)), distributed
via Molpro's ``weigend_jkfit.libmol`` (11 Feb 2010) in both Psi4
(``psi4/share/psi4/basis/aug-cc-pv?z-jkfit.gbs``) and PySCF
(``pyscf/gto/basis/aug-cc-pv?z-jkfit.dat``). Per that header the DZ sets are
the TZ sets truncated (not published) and Ne/Ar/Kr are extrapolated.

This script converts Psi4's Gaussian-94 file (fetched at a pinned commit) to
the BSE-JSON layout of ``crates/ferric-core/src/basis/bundled/`` with the
digit strings untouched, then cross-checks it shell by shell against PySCF's
independent copy (every exponent and coefficient parses to the bit-identical double). It
refuses to write a file whose two sources disagree.

    python -I scripts/import_psi4_jkfit.py --psi4-dir DIR --sha SHA

where DIR holds ``aug-cc-pv{d,t,q}z-jkfit.gbs`` downloaded from
``https://raw.githubusercontent.com/psi4/psi4/<SHA>/psi4/share/psi4/basis/``.
Run with an interpreter that has PySCF (the repo's .venv does).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyscf import gto
from pyscf.data.elements import ELEMENTS

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ROOT / "crates/ferric-core/src/basis/bundled"
PROVENANCE = ROOT / "crates/ferric-core/src/basis/provenance.json"
EXTRAPOLATED = [10, 18, 36]  # Ne, Ar, Kr: extrapolated, not published
L_OF = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}
NAMES = {"aug-cc-pvdz-jkfit": "aug-cc-pVDZ-JKFIT", "aug-cc-pvtz-jkfit": "aug-cc-pVTZ-JKFIT",
         "aug-cc-pvqz-jkfit": "aug-cc-pVQZ-JKFIT"}  # fmt: skip


def parse_gbs(text: str) -> dict[int, list[tuple[int, list[str], list[str]]]]:
    """Gaussian-94 text -> {Z: [(l, exponents, coefficients)]}, strings kept."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("!")]
    assert lines[0].lower() == "spherical", "expected a 'spherical' Psi4 header"
    out: dict[int, list] = {}
    i, z = 1, None
    while i < len(lines):
        tok = lines[i].split()
        if tok[0] == "****":
            z = None
        elif z is None:
            z = ELEMENTS.index(tok[0].capitalize())
            out[z] = []
        else:
            ang, nprim = L_OF[tok[0].upper()], int(tok[1])
            prims = [lines[i + 1 + k].replace("D", "E").split() for k in range(nprim)]
            assert all(len(p) == 2 for p in prims), f"Z={z}: not a 1-column shell"
            out[z].append((ang, [p[0] for p in prims], [p[1] for p in prims]))
            i += nprim
        i += 1
    return out


def same_as_pyscf(stem: str, data: dict) -> list[str]:
    """Disagreements between the Psi4-derived data and PySCF's own copy."""
    problems = []
    for z, shells in data.items():
        sym = ELEMENTS[z]
        ref = gto.basis.load(stem, sym)
        mine = sorted(
            (ang, tuple(map(float, e)), tuple(map(float, c))) for ang, e, c in shells
        )
        theirs = sorted(
            (b[0], tuple(p[0] for p in b[1:]), tuple(p[1] for p in b[1:])) for b in ref
        )
        if len(mine) != len(theirs):
            problems.append(f"{stem} {sym}: {len(mine)} shells vs PySCF {len(theirs)}")
            continue
        for a, b in zip(mine, theirs):
            vals = list(zip(a[1] + a[2], b[1] + b[2]))
            if a[0] != b[0] or len(a[1]) != len(b[1]):
                problems.append(f"{stem} {sym}: shell shape differs")
            elif any(
                x != y for x, y in vals
            ):  # exact: same digits parse to the same double
                problems.append(f"{stem} {sym} l={a[0]}: values differ")
    return problems


def to_bse_json(stem: str, data: dict, sha: str) -> dict:
    url = f"https://github.com/psi4/psi4/blob/{sha}/psi4/share/psi4/basis/{stem}.gbs"
    elements = {
        str(z): {
            "electron_shells": [
                {
                    "function_type": "gto_spherical" if ang >= 2 else "gto",
                    "region": "",
                    "angular_momentum": [ang],
                    "exponents": e,
                    "coefficients": [c],
                }
                for ang, e, c in shells
            ]
        }
        for z, shells in sorted(data.items())
    }
    return {
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "revision_description": (
            f"NOT from the Basis Set Exchange (it has no aug-cc-pVXZ-JKFIT). Data "
            f"from Psi4 {url} (Molpro weigend_jkfit.libmol, 11 Feb 2010), checked "
            "equal to PySCF's copy by scripts/import_psi4_jkfit.py. Ne/Ar/Kr are "
            "extrapolated; the DZ set is the TZ set truncated (per that header)."
        ),
        "revision_date": "2010-02-11",
        "elements": elements,
        "version": "psi4-" + sha[:12],
        "function_types": ["gto", "gto_spherical"],
        "names": [NAMES[stem]],
        "tags": [],
        "family": "dunning",
        "description": NAMES[stem] + " (Weigend 2002) J/K fitting",
        "role": "jkfit",
        "auxiliaries": {},
        "name": NAMES[stem],
    }


def provenance_record(stem: str, data: dict, sha: str) -> dict:
    rec = {
        "file": f"{stem}.json",
        "source": "psi4",
        "not_in_bse": True,
        "name": NAMES[stem],
        "psi4_commit": sha,
        "url": f"https://github.com/psi4/psi4/blob/{sha}/psi4/share/psi4/basis/{stem}.gbs",
        "origin": "Molpro weigend_jkfit.libmol, 11 February 2010",
        "reference": "F. Weigend, Phys. Chem. Chem. Phys. 4, 4285 (2002)",
        "cross_check": "bit-identical to PySCF's aug-cc-pv?z-jkfit.dat (import_psi4_jkfit.py)",
        "role": "jkfit",
        "elements": sorted(data),
        "extrapolated_elements": [z for z in EXTRAPOLATED if z in data],
    }
    if stem == "aug-cc-pvdz-jkfit":
        rec["truncated_from"] = (
            "aug-cc-pVTZ-JKFIT: not published; formed by dropping the highest "
            "angular momentum of the TZ set (per the Molpro/Psi4/PySCF header)"
        )
    return rec


def write_atomic(path: Path, text: str) -> None:
    """Write via a sibling temp file and rename, so a crash leaves old or new."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def update_provenance(records: list[dict]) -> None:
    prov = json.loads(PROVENANCE.read_text())
    files = {r["file"] for r in records}
    prov["sets"] = [r for r in prov["sets"] if r["file"] not in files] + records
    write_atomic(PROVENANCE, json.dumps(prov, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psi4-dir", type=Path, required=True)
    ap.add_argument("--sha", required=True, help="psi4 commit the files came from")
    args = ap.parse_args()
    # All-or-nothing: every set is converted and cross-checked in memory, and
    # nothing (JSON or provenance) is written unless the whole batch passes.
    # Writing as we go left an earlier set's JSON on disk with no provenance
    # record whenever a later set failed.
    bad, docs, records = [], {}, []
    for stem in NAMES:
        data = parse_gbs((args.psi4_dir / f"{stem}.gbs").read_text())
        problems = same_as_pyscf(stem, data)
        print(
            f"{stem}: {len(data)} elements, PySCF cross-check: {problems or 'identical'}"
        )
        if problems:
            bad += problems
            continue
        docs[stem] = to_bse_json(stem, data, args.sha)
        records.append(provenance_record(stem, data, args.sha))
    if bad:
        print("ERROR: cross-check failed; nothing written")
        return 1
    for stem, doc in docs.items():
        write_atomic(BUNDLED / f"{stem}.json", json.dumps(doc, indent=1) + "\n")
    update_provenance(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
