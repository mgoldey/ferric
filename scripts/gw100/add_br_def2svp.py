#!/usr/bin/env python3
"""Idempotently add the Br (Z=35) element block to ferric's bundled def2-svp.json.

def2-SVP covers Br as an ALL-ELECTRON basis (no ECP). This script fetches Br's
block from basis-set-exchange (offline) and merges it into the bundled JSON under
the "35" key, matching the existing MolSSI-BSE-schema element-block format used
for Z=1..18 (which carry `region` + `references`).

Re-runnable: writes only the "35" key, leaving every other element and all
top-level metadata untouched. If "35" already matches the BSE block, it is a no-op.

Usage:
    python3 scripts/gw100/add_br_def2svp.py
"""
import json
import sys
from pathlib import Path

import basis_set_exchange as bse

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "crates/ferric-core/src/basis/bundled/def2-svp.json"
Z = 35
KEY = str(Z)


def main() -> int:
    with TARGET.open() as fh:
        bundled = json.load(fh)

    # Fetch Br block from BSE (fmt=None -> native MolSSI-BSE dict, string coeffs).
    br_full = bse.get_basis("def2-svp", elements=[Z], fmt=None)
    br_block = br_full["elements"][KEY]

    # def2-SVP Br is all-electron: refuse to proceed if BSE ever hands us an ECP.
    if "ecp_potentials" in br_block or "ecp_electrons" in br_block:
        print("ERROR: BSE returned an ECP for Br in def2-svp; refusing.", file=sys.stderr)
        return 1

    if bundled["elements"].get(KEY) == br_block:
        print(f"No change: Z={Z} already present and identical.")
        return 0

    existed = KEY in bundled["elements"]
    bundled["elements"][KEY] = br_block

    # Match the file's existing formatting exactly: 1-space indent, NO trailing
    # newline (verified: json.dumps(indent=1) reproduces every other line byte-for-
    # byte, and the original file has no terminating newline).
    with TARGET.open("w") as fh:
        fh.write(json.dumps(bundled, indent=1))

    n_shells = len(br_block["electron_shells"])
    verb = "Updated" if existed else "Added"
    print(f"{verb} Z={Z} (Br): {n_shells} electron_shells, all-electron (no ECP).")
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
