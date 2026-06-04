#!/usr/bin/env python3
"""Emit TOMLs for the TS-stress benchmark: 10 molecules x {rpa_pbe, ts} x
{aug-cc-pVDZ, aug-cc-pVTZ}. (RPA@HF dropped — the DOSD study already showed it's
uniformly ~40% low; here we contrast the GOOD method RPA@PBE against TS on
TS's predicted failure modes.)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"

# molecule key -> (xyz filename, charge, multiplicity)
MOLS = {
    "so2": ("so2.xyz", 0, 1), "cs2": ("cs2.xyz", 0, 1),
    "cos": ("cos.xyz", 0, 1), "n2o": ("n2o.xyz", 0, 1),
    "cl2": ("cl2.xyz", 0, 1), "hbr": ("hbr.xyz", 0, 1),
    "sih4": ("sih4.xyz", 0, 1), "ccl4": ("ccl4.xyz", 0, 1),
    "ch3oh": ("ch3oh.xyz", 0, 1), "ch3och3": ("ch3och3.xyz", 0, 1),
}
BASES = {
    "augccpvdz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "augccpvtz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit"),
}
METHODS = {
    "rpa_pbe": ("PBE", "pdep"),
    "ts": ("PBE", "ts"),
}


def toml_for(mol, bkey, mkey, npz):
    xyz, charge, mult = MOLS[mol]
    obs, aux = BASES[bkey]
    xc, c6src = METHODS[mkey]
    lines = [
        "[molecule]", f'xyz = "testdata/molecules/{xyz}"',
        f"charge = {charge}", f"multiplicity = {mult}", "",
        "[basis]", f'name = "{obs}"', "",
        "[method]", 'kind = "pdep-rpa"', 'task = "energy"', "",
        "[rpa]", f'auxbasis = "{aux}"', f'xc = "{xc}"',
        "n_quad = 40", 'quadrature = "gauss-legendre"',
        "frozen_core = 0", "trunc_thresh = 0.0",
        f'export_npz = "{npz}"', "compute_c6 = true",
        f'c6_source = "{c6src}"',
        # Becke partition: avoids needing element-specific TS free-atom tables
        # for heavy atoms, and the molecular C6 is partition-independent anyway.
        'c6_partition = "becke"', "",
    ]
    return "\n".join(lines)


def main():
    for bkey in BASES:
        outdir = RUNS / bkey
        outdir.mkdir(parents=True, exist_ok=True)
        for mol in MOLS:
            for mkey in METHODS:
                npz = outdir / f"{mol}_{mkey}.npz"
                (outdir / f"{mol}_{mkey}.toml").write_text(
                    toml_for(mol, bkey, mkey, str(npz)))
    n = len(BASES) * len(MOLS) * len(METHODS)
    print(f"wrote {n} TOMLs under {RUNS}")


if __name__ == "__main__":
    main()
