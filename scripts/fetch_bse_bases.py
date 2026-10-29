#!/usr/bin/env python3
"""Fetch ferric's bundled basis sets from the Basis Set Exchange (BSE).

This is the single source of truth for which BSE set backs which file in
``crates/ferric-core/src/basis/bundled/`` and for the provenance record
``crates/ferric-core/src/basis/provenance.json`` (BSE package version, each
set's BSE name, BSE data version, revision date and element list).

Run it with the pinned BSE release so a re-fetch is reproducible::

    uvx --from basis_set_exchange==0.12 python scripts/fetch_bse_bases.py
    uvx --from basis_set_exchange==0.12 python scripts/fetch_bse_bases.py --check

``--check`` writes nothing: it reports, per file, whether the shipped JSON
already matches BSE element for element and which elements a re-fetch would
add or change. A changed shell on an element the file already had is an
error, because that would silently move every reference energy computed with
it.

The one deliberate departure from BSE data is ``SPHERICAL_OVERRIDE``: the
6-311G family is made spherical on every element (Gaussian's convention).

``def2-ecp.json`` is not in the table: it is a hand-trimmed iodine-only ECP
consumed by ``ecp.rs``, not a basis set, and it has its own tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import basis_set_exchange as bse

ROOT = Path(__file__).resolve().parent.parent
BASIS_DIR = ROOT / "crates" / "ferric-core" / "src" / "basis"
BUNDLED = BASIS_DIR / "bundled"
PROVENANCE = BASIS_DIR / "provenance.json"

# file stem in bundled/  ->  BSE basis name
SETS: dict[str, str] = {
    # Pople. "*" and "+" are kept out of file names; the (d,p) spelling is
    # the file stem, and bundled() accepts both spellings.
    "sto-3g": "STO-3G",
    "6-31g": "6-31G",
    "6-31g_d": "6-31G*",
    "6-31g_d_p": "6-31G**",
    "6-31+g_d": "6-31+G*",
    "6-31+g_d_p": "6-31+G**",
    "6-31++g_d_p": "6-31++G**",
    "6-311g_d_p": "6-311G**",
    "6-311++g_d_p": "6-311++G**",
    # Dunning, all-electron
    "cc-pvdz": "cc-pVDZ",
    "cc-pvtz": "cc-pVTZ",
    "cc-pvqz": "cc-pVQZ",
    "aug-cc-pvdz": "aug-cc-pVDZ",
    "aug-cc-pvtz": "aug-cc-pVTZ",
    "aug-cc-pvqz": "aug-cc-pVQZ",
    "cc-pvdz-f12": "cc-pVDZ-F12",
    "cc-pvdz-f12-optri": "cc-pVDZ-F12-OPTRI",
    # Dunning, pseudopotential
    "cc-pvdz-pp": "cc-pVDZ-PP",
    "cc-pvtz-pp": "cc-pVTZ-PP",
    "cc-pvqz-pp": "cc-pVQZ-PP",
    "aug-cc-pvdz-pp": "aug-cc-pVDZ-PP",
    "aug-cc-pvtz-pp": "aug-cc-pVTZ-PP",
    "aug-cc-pvqz-pp": "aug-cc-pVQZ-PP",
    # Dunning RI / JK fitting
    "cc-pvdz-ri": "cc-pVDZ-RIFIT",
    "cc-pvtz-rifit": "cc-pVTZ-RIFIT",
    "cc-pvqz-rifit": "cc-pVQZ-RIFIT",
    "aug-cc-pvdz-rifit": "aug-cc-pVDZ-RIFIT",
    "aug-cc-pvtz-rifit": "aug-cc-pVTZ-RIFIT",
    "aug-cc-pvqz-rifit": "aug-cc-pVQZ-RIFIT",
    "cc-pvdz-pp-rifit": "cc-pVDZ-PP-RIFIT",
    "cc-pvtz-pp-rifit": "cc-pVTZ-PP-RIFIT",
    "cc-pvqz-pp-rifit": "cc-pVQZ-PP-RIFIT",
    "aug-cc-pvdz-pp-rifit": "aug-cc-pVDZ-PP-RIFIT",
    "aug-cc-pvtz-pp-rifit": "aug-cc-pVTZ-PP-RIFIT",
    "aug-cc-pvqz-pp-rifit": "aug-cc-pVQZ-PP-RIFIT",
    "cc-pvtz-jkfit": "cc-pVTZ-JKFIT",
    "cc-pvqz-jkfit": "cc-pVQZ-JKFIT",
    # Karlsruhe
    "def2-svp": "def2-SVP",
    "def2-svpd": "def2-SVPD",
    "def2-tzvp": "def2-TZVP",
    "def2-tzvpp": "def2-TZVPP",
    "def2-tzvpd": "def2-TZVPD",
    "def2-tzvppd": "def2-TZVPPD",
    "def2-qzvp": "def2-QZVP",
    "def2-qzvpp": "def2-QZVPP",
    "def2-qzvppd": "def2-QZVPPD",
    "def2-svp-rifit": "def2-SVP-RIFIT",
    "def2-svpd-rifit": "def2-SVPD-RIFIT",
    "def2-tzvp-rifit": "def2-TZVP-RIFIT",
    "def2-tzvpp-rifit": "def2-TZVPP-RIFIT",
    "def2-tzvpd-rifit": "def2-TZVPD-RIFIT",
    "def2-tzvppd-rifit": "def2-TZVPPD-RIFIT",
    "def2-qzvp-rifit": "def2-QZVP-RIFIT",
    "def2-qzvpp-rifit": "def2-QZVPP-RIFIT",
    "def2-qzvppd-rifit": "def2-QZVPPD-RIFIT",
    "def2-universal-jkfit": "def2-universal-JKFIT",
    # Jensen segmented polarization-consistent
    "pcseg-0": "pcseg-0",
    "pcseg-1": "pcseg-1",
    "pcseg-2": "pcseg-2",
    "aug-pcseg-1": "aug-pcseg-1",
    "aug-pcseg-2": "aug-pcseg-2",
}


# PP sets cover only the heavy elements (Z >= 29) upstream. Following the
# Psi4 convention (and what the shipped aug-cc-pVDZ-PP already did for
# H/C/Al/Cl), the light elements are filled from the matching all-electron
# set, so e.g. HBr/cc-pVDZ-PP is one basis. The PP-RIFIT sets get the matching
# all-electron RIFIT on exactly those filled elements. Z 29-38 under a
# PP-RIFIT stays absent upstream and therefore errors at lookup.
FILL_BELOW = 29
FILL_FROM: dict[str, str] = {
    "cc-pvdz-pp": "cc-pVDZ",
    "cc-pvtz-pp": "cc-pVTZ",
    "cc-pvqz-pp": "cc-pVQZ",
    "aug-cc-pvdz-pp": "aug-cc-pVDZ",
    "aug-cc-pvtz-pp": "aug-cc-pVTZ",
    "aug-cc-pvqz-pp": "aug-cc-pVQZ",
    "cc-pvdz-pp-rifit": "cc-pVDZ-RIFIT",
    "cc-pvtz-pp-rifit": "cc-pVTZ-RIFIT",
    "cc-pvqz-pp-rifit": "cc-pVQZ-RIFIT",
    "aug-cc-pvdz-pp-rifit": "aug-cc-pVDZ-RIFIT",
    "aug-cc-pvtz-pp-rifit": "aug-cc-pVTZ-RIFIT",
    "aug-cc-pvqz-pp-rifit": "aug-cc-pVQZ-RIFIT",
}

# THE one place ferric deliberately departs from BSE's function_type.
#
# ferric follows Gaussian's convention for the Pople sets. The 6-31G family
# stays Cartesian 6D, which is what BSE says too. The 6-311G family is
# spherical 5D on EVERY element. BSE types the Na-Ar d shells of 6-311G**
# and 6-311++G** gto_cartesian but every other element spherical, so a
# molecule such as CH3Cl would carry 5D on C and 6D on Cl, which no other code
# builds. Every l >= 2 shell of these files is therefore rewritten to
# gto_spherical, at fetch time AND on the shipped file. The provenance record
# carries a function_type_override entry that names the changed elements and
# this reason. scripts/gen_pyscf_basis_coverage_refs.py reads that entry, so
# its PySCF references use the same convention (cart=False).
SPHERICAL_OVERRIDE: frozenset[str] = frozenset({"6-311g_d_p", "6-311++g_d_p"})
SPHERICAL_OVERRIDE_REASON = "Gaussian convention; BSE marks Na-Ar Cartesian"

REL_TOL = 1e-12


def force_spherical(data: dict) -> list[int]:
    """Make every l >= 2 shell gto_spherical; return the elements changed."""
    changed = set()
    for z, el in data["elements"].items():
        for sh in el.get("electron_shells", []):
            if (
                max(sh["angular_momentum"]) >= 2
                and sh["function_type"] != "gto_spherical"
            ):
                sh["function_type"] = "gto_spherical"
                changed.add(int(z))
    data["function_types"] = sorted(
        {
            sh["function_type"]
            for el in data["elements"].values()
            for sh in el.get("electron_shells", [])
        }
    )
    return sorted(changed)


def _floats(xs: list[str]) -> list[float]:
    return [float(x) for x in xs]


def _close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(
        abs(x - y) <= REL_TOL * max(abs(x), abs(y), 1e-300) for x, y in zip(a, b)
    )


def _shell_key(sh: dict) -> tuple:
    ls = tuple(sh["angular_momentum"])
    # ferric reads l >= 2 as spherical only for gto_spherical; l < 2 is the
    # same function either way, so "gto" vs "gto_spherical" there is cosmetic.
    pure = sh.get("function_type") == "gto_spherical" and max(ls) >= 2
    return ls, pure


def element_equal(a: dict, b: dict) -> bool:
    """Numerical equality of one element's shells and ECP (formatting-blind)."""
    sa, sb = a.get("electron_shells", []), b.get("electron_shells", [])
    if len(sa) != len(sb) or a.get("ecp_electrons") != b.get("ecp_electrons"):
        return False
    for x, y in zip(sa, sb):
        if _shell_key(x) != _shell_key(y):
            return False
        if not _close(_floats(x["exponents"]), _floats(y["exponents"])):
            return False
        if len(x["coefficients"]) != len(y["coefficients"]):
            return False
        for cx, cy in zip(x["coefficients"], y["coefficients"]):
            if not _close(_floats(cx), _floats(cy)):
                return False
    pa, pb = a.get("ecp_potentials") or [], b.get("ecp_potentials") or []
    key = lambda p: (tuple(p["angular_momentum"]), tuple(p["r_exponents"]))  # noqa: E731
    pa, pb = sorted(pa, key=key), sorted(pb, key=key)
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if key(x) != key(y):
            return False
        if not _close(
            _floats(x["gaussian_exponents"]), _floats(y["gaussian_exponents"])
        ):
            return False
        if not _close(_floats(x["coefficients"][0]), _floats(y["coefficients"][0])):
            return False
    return True


def fetch(stem: str, name: str) -> tuple[dict, dict]:
    data = json.loads(bse.get_basis(name, fmt="json", header=False))
    record = {
        "file": f"{stem}.json",
        "bse_name": data["name"],
        "bse_data_version": data["version"],
        "revision_date": data["revision_date"],
        "revision_description": data["revision_description"],
        "role": data.get("role"),
        "bse_elements": sorted(int(z) for z in data["elements"]),
    }
    if stem in FILL_FROM:
        src_name = FILL_FROM[stem]
        src = json.loads(bse.get_basis(src_name, fmt="json", header=False))
        filled = []
        for z, el in src["elements"].items():
            if int(z) < FILL_BELOW and z not in data["elements"]:
                data["elements"][z] = el
                filled.append(int(z))
        record["filled_from"] = {"bse_name": src["name"], "elements": sorted(filled)}
    if stem in SPHERICAL_OVERRIDE:
        record["function_type_override"] = {
            "l_ge_2": "gto_spherical",
            "elements_changed": force_spherical(data),
            "reason": SPHERICAL_OVERRIDE_REASON,
        }
    return data, record


def merge(stem: str, fresh: dict) -> tuple[dict, list[int], list[int], list[int]]:
    """Merge a fresh fetch into the shipped file without touching shipped data.

    Elements the shipped file already has are kept byte-for-byte (several were
    hand-added and are documented in their revision_description); elements
    only upstream has are appended. Returns (merged, added, kept_extra,
    changed) where kept_extra are shipped elements upstream lacks and changed
    are shipped elements that numerically DISAGREE with upstream.
    """
    path = BUNDLED / f"{stem}.json"
    if not path.exists():
        return fresh, sorted(int(z) for z in fresh["elements"]), [], []
    old = json.loads(path.read_text())
    if stem in SPHERICAL_OVERRIDE:
        force_spherical(old)  # idempotent; see SPHERICAL_OVERRIDE
    oe, fe = old["elements"], fresh["elements"]
    added = sorted((z for z in fe if z not in oe), key=int)
    extra = sorted(int(z) for z in oe if z not in fe)
    changed = sorted(int(z) for z in oe if z in fe and not element_equal(oe[z], fe[z]))
    for z in added:
        oe[z] = fe[z]
    old["elements"] = dict(sorted(oe.items(), key=lambda kv: int(kv[0])))
    return old, [int(z) for z in added], extra, changed


def kept_non_bse() -> list[dict]:
    if not PROVENANCE.exists():
        return []
    return [r for r in json.loads(PROVENANCE.read_text())["sets"] if r.get("source")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    records = []
    bad = False
    for stem, name in SETS.items():
        fresh, rec = fetch(stem, name)
        is_new = not (BUNDLED / f"{stem}.json").exists()
        merged, added, extra, changed = merge(stem, fresh)
        status = "new" if is_new else ("updated" if added else "same")
        if changed:
            status, bad = "CHANGED", True
        print(
            f"{stem:22s} {status:7s} n_el={len(merged['elements']):3d} "
            f"added={added} shipped_only={extra} changed={changed}"
        )
        rec["elements"] = sorted(int(z) for z in merged["elements"])
        if extra:
            rec["shipped_only_elements"] = extra
        records.append(rec)
        path = BUNDLED / f"{stem}.json"
        text = json.dumps(merged, indent=1, ensure_ascii=False) + "\n"
        # Written when new, extended, or rewritten by SPHERICAL_OVERRIDE.
        if not args.check and (is_new or text != path.read_text()):
            path.write_text(text)

    if not args.check:
        prov = {
            "source": "Basis Set Exchange, https://www.basissetexchange.org",
            "bse_package_version": bse.version(),
            "fetched_by": "scripts/fetch_bse_bases.py",
            "fetched_on": str(date.today()),
            "note": (
                "Shipped elements are never rewritten by a re-fetch; "
                "shipped_only_elements are local extensions documented in each "
                "file's revision_description. def2-ecp.json is not listed: it "
                "is an iodine-only def2 ECP used by ecp.rs, not a basis set. "
                "The one deliberate departure from BSE data: every l >= 2 "
                "shell of the 6-311G family is spherical (see "
                "function_type_override and SPHERICAL_OVERRIDE in the fetch "
                "script)."
            ),
            # Sets imported from elsewhere (scripts/import_psi4_jkfit.py)
            # keep their own records.
            "sets": records + kept_non_bse(),
        }
        PROVENANCE.write_text(json.dumps(prov, indent=1) + "\n")
    if bad:
        print("ERROR: shipped elements numerically disagree with upstream")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
