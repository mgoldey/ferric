#!/usr/bin/env python3
"""Emit BSE-JSON element blocks for def2-SVP + def2-ECP heavy atoms (I, Xe),
in the exact format ferric's parse_bse_json / parse_ecp_block expect, and merge
them into crates/ferric-core/src/basis/bundled/def2-svp.json.

The bundled def2-svp.json (a Turbomole-raw BSE download) lacks Z>=37; ferric's
parser renormalizes each contraction to unit self-overlap, so emitting PySCF's
(already libcint-normalized) contraction coefficients is convention-safe: both
end up unit-normalized and identical to what libint builds internally.

ECP encoding matches the bundled def2-ecp.json convention:
  - the local channel (PySCF l=-1) is emitted with the actual max projector l+1
    (e.g. I: projectors l=0,1,2 -> local l=3);
  - r_exponents is the literal r-power (the PySCF power-index, 2 for def2 terms).

Run: python3 scripts/gw100/gen_def2svp_ecp_blocks.py
"""
import json
import os
from pyscf import gto

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.normpath(
    os.path.join(HERE, "..", "..", "crates", "ferric-core", "src", "basis", "bundled", "def2-svp.json")
)

ELEMENTS = {"53": "I", "54": "Xe"}


def basis_block_to_shells(b):
    """PySCF basis [[l, [exp, c0, c1...], ...], ...] -> list of BSE electron_shells."""
    shells = []
    for blk in b:
        l = int(blk[0])
        rows = blk[1:]
        exps = [str(r[0]) for r in rows]
        ncol = len(rows[0]) - 1
        coeffs = [[str(r[1 + c]) for r in rows] for c in range(ncol)]
        shells.append({
            "function_type": "gto_spherical",
            "angular_momentum": [l],
            "exponents": exps,
            "coefficients": coeffs,
        })
    return shells


def ecp_to_potentials(e):
    """PySCF ecp [ncore, [[l, [terms_by_rpower]], ...]] -> (ncore, BSE ecp_potentials)."""
    ncore = int(e[0])
    channels = e[1]
    proj_ls = [int(c[0]) for c in channels if int(c[0]) >= 0]
    local_l = max(proj_ls) + 1
    pots = []
    for c in channels:
        l = int(c[0])
        out_l = local_l if l == -1 else l
        ams, r_exps, gexps, coefs = [], [], [], []
        for r_power, terms in enumerate(c[1]):
            for (zeta, d) in terms:
                ams.append(out_l)
                r_exps.append(int(r_power))
                gexps.append(str(zeta))
                coefs.append(str(d))
        if not ams:
            continue
        pots.append({
            "angular_momentum": [out_l],
            "r_exponents": r_exps,
            "gaussian_exponents": gexps,
            "coefficients": [coefs],
        })
    # Sort local channel first to match bundled def2-ecp.json convention.
    pots.sort(key=lambda p: 0 if p["angular_momentum"][0] == local_l else p["angular_momentum"][0] + 1)
    return ncore, pots


def main():
    with open(BUNDLED) as f:
        data = json.load(f)
    for z, sym in ELEMENTS.items():
        b = gto.basis.load("def2-svp", sym)
        e = gto.basis.load_ecp("def2-svp", sym)
        ncore, pots = ecp_to_potentials(e)
        elem = {
            "electron_shells": basis_block_to_shells(b),
            "ecp_electrons": ncore,
            "ecp_potentials": pots,
        }
        data["elements"][z] = elem
        print(f"Z={z} {sym}: {len(elem['electron_shells'])} shells, "
              f"ncore={ncore}, {len(pots)} ecp channels")
    with open(BUNDLED, "w") as f:
        json.dump(data, f)
    print(f"merged into {BUNDLED}")


if __name__ == "__main__":
    main()
