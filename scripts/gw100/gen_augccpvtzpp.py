#!/usr/bin/env python3
"""Generate the bundled merged aug-cc-pVTZ-PP JSON for ferric.

Mirrors the existing aug-cc-pvdz-pp.json: a MolSSI-BSE-schema file covering
Z = {1, 6, 13, 17, 47, 53, 54}.

  - Heavy PP elements 47 (Ag), 53 (I), 54 (Xe): pulled from BSE `aug-cc-pvtz-pp`.
    These carry electron_shells + ecp_potentials + ecp_electrons (28-core).
  - Light elements 1 (H), 6 (C), 13 (Al), 17 (Cl): pulled from plain BSE
    `aug-cc-pvtz` (electron_shells only, no ECP).

ferric's parse_bse_json renormalizes contractions to unit self-overlap, so raw
BSE coefficients are used as-is.

Run:  python3 scripts/gw100/gen_augccpvtzpp.py
Writes: crates/ferric-core/src/basis/bundled/aug-cc-pvtz-pp.json
"""
import json
import os

import basis_set_exchange as bse

HEAVY = [47, 53, 54]        # Ag, I, Xe -- from aug-cc-pvtz-pp (with ECP)
LIGHT = [1, 6, 13, 17]      # H, C, Al, Cl -- from plain aug-cc-pvtz (no ECP)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "crates", "ferric-core", "src", "basis", "bundled",
                   "aug-cc-pvtz-pp.json")


def fetch_elements(name, zlist):
    """Return {str(z): element_dict} from a BSE basis, one Z at a time so a
    missing element surfaces clearly rather than aborting the whole fetch."""
    out = {}
    for z in zlist:
        d = bse.get_basis(name, elements=[z], fmt=None)
        els = d["elements"]
        key = str(z)
        if key not in els:
            raise SystemExit(f"FAILED to fetch Z={z} from {name!r}")
        out[key] = els[key]
    return out


def main():
    heavy = fetch_elements("aug-cc-pvtz-pp", HEAVY)
    light = fetch_elements("aug-cc-pvtz", LIGHT)

    elements = {}
    elements.update(light)
    elements.update(heavy)

    # sanity: heavy elements must carry the 28-core ECP
    for z in HEAVY:
        el = elements[str(z)]
        assert "ecp_potentials" in el and el.get("ecp_electrons") == 28, (
            f"Z={z} missing 28-core ECP: {list(el.keys())}, "
            f"ecp_electrons={el.get('ecp_electrons')}"
        )
    # sanity: light elements must NOT carry an ECP
    for z in LIGHT:
        el = elements[str(z)]
        assert "ecp_potentials" not in el and "ecp_electrons" not in el, (
            f"Z={z} unexpectedly carries an ECP"
        )

    doc = {
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "revision_description": (
            "Merged aug-cc-pVTZ-PP (heavy, ECP) + aug-cc-pVTZ (light) "
            "for GW100 ECP molecules"
        ),
        "revision_date": "2010-05-27",
        "elements": elements,
        "version": "0",
        "function_types": ["gto", "gto_spherical", "scalar_ecp"],
        "names": ["aug-cc-pVTZ-PP"],
        "tags": [],
        "family": "dunning_pp",
        "description": "aug-cc-pVTZ-PP",
        "role": "orbital",
        "auxiliaries": {
            "rifit": "aug-cc-pvtz-pp-rifit",
            "optri": "aug-cc-pvtz-pp-optri",
        },
        "name": (
            "aug-cc-pVTZ-PP (heavy: I/Xe/Ag with ECP; "
            "light: aug-cc-pVTZ for H/C/Al/Cl)"
        ),
    }

    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")
    for z in sorted(elements, key=int):
        el = elements[z]
        print(f"  Z={z:>2}  nshells={len(el.get('electron_shells', [])):>2}  "
              f"has_ecp={'ecp_potentials' in el}  "
              f"ecp_electrons={el.get('ecp_electrons')}")


if __name__ == "__main__":
    main()
