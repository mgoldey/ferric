#!/usr/bin/env python3
"""Deterministically emit ferric-cli TOMLs for the DOSD C6 sweep.
One TOML per (molecule, method, basis). TS is computed in the same run config
family but with c6_source='ts' on the PBE reference (see run_sweep)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"

# molecule key -> (xyz filename, charge, multiplicity)
MOLS = {
    "h2": ("h2.xyz", 0, 1), "n2": ("n2.xyz", 0, 1), "co": ("co.xyz", 0, 1),
    "water": ("water.xyz", 0, 1), "nh3": ("nh3.xyz", 0, 1),
    "ch4": ("methane.xyz", 0, 1), "co2": ("co2.xyz", 0, 1),
    "c2h2": ("c2h2.xyz", 0, 1), "c2h4": ("c2h4.xyz", 0, 1),
    "c2h6": ("c2h6.xyz", 0, 1), "hf": ("hf.xyz", 0, 1),
    "hcl": ("hcl.xyz", 0, 1), "h2s": ("h2s.xyz", 0, 1),
    "benzene": ("benzene.xyz", 0, 1), "o2": ("o2.xyz", 0, 3),
}
BASES = {  # basis key -> (orbital basis, rifit aux)
    "augccpvdz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "augccpvtz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit"),
}
# method key -> (xc or None, c6_source)
METHODS = {
    "rpa_pbe": ("PBE", "pdep"),
    "rpa_hf":  (None,  "pdep"),
    "ts":      ("PBE", "ts"),   # TS from the PBE reference
}


def toml_for(mol, bkey, mkey, npz_path):
    xyz, charge, mult = MOLS[mol]
    obs, aux = BASES[bkey]
    xc, c6src = METHODS[mkey]
    lines = [
        "[molecule]",
        f'xyz = "testdata/molecules/{xyz}"',
        f"charge = {charge}",
        f"multiplicity = {mult}",
        "",
        "[basis]",
        f'name = "{obs}"',
        "",
        "[method]",
        'kind = "pdep-rpa"',
        'task = "energy"',
        "",
        "[rpa]",
        f'auxbasis = "{aux}"',
    ]
    if xc is not None:
        lines.append(f'xc = "{xc}"')
    lines += [
        "n_quad = 40",
        'quadrature = "gauss-legendre"',
        "frozen_core = 0",
        "trunc_thresh = 0.0",
        f'export_npz = "{npz_path}"',
        "compute_c6 = true",
        f'c6_source = "{c6src}"',
        'c6_partition = "hirshfeld"',
        "",
    ]
    return "\n".join(lines)


def main():
    for bkey in BASES:
        outdir = RUNS / bkey
        outdir.mkdir(parents=True, exist_ok=True)
        for mol in MOLS:
            for mkey in METHODS:
                npz = outdir / f"{mol}_{mkey}.npz"
                toml = toml_for(mol, bkey, mkey, str(npz))
                (outdir / f"{mol}_{mkey}.toml").write_text(toml)
    n = len(BASES) * len(MOLS) * len(METHODS)
    print(f"wrote {n} TOMLs under {RUNS}")


if __name__ == "__main__":
    main()
