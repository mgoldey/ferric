#!/usr/bin/env python3
"""Generate crates/ferric-d3/testdata/d3bj_atm_reference.json.

Reference: the `dftd3` Python package (simple-dftd3), an independent Fortran
implementation. For each system and functional it records the D3(BJ) energy
and gradient with the Axilrod-Teller-Muto three-body term OFF (s9 = 0) and ON
(s9 = 1), using explicitly passed (s6, s8, a1, a2) so the parameters are pinned
in the file rather than looked up by name.

Consumed by crates/ferric-d3/tests/vs_reference_dftd3_atm.rs.

Run (needs the dftd3 package; heavy python goes through the compile lock):
    flock -o /tmp/ferric-compile.lock python3 scripts/gen_d3_atm_reference.py
"""

import json
import pathlib

import dftd3
import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO_BODY = ROOT / "crates/ferric-d3/testdata/d3bj_reference.json"
OUT = ROOT / "crates/ferric-d3/testdata/d3bj_atm_reference.json"

FUNCTIONALS = {
    "pbe": (1.0, 0.7875, 0.4289, 4.4407),
    "pbe0": (1.0, 1.2177, 0.4145, 4.8593),
    "b3lyp": (1.0, 1.9889, 0.3981, 4.4211),
}

# Systems beyond the two-body reference file, chosen so the three-body term is
# well above rounding: a compact argon trimer, a mixed-element fragment with
# spread coordination numbers (same as tests/gradient_vs_fd.rs::mixed), and a
# bromine-containing cluster for the heavy-element part of the vdW-radius table.
EXTRA = {
    "argon_trimer": (
        [18, 18, 18],
        [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [3.5, 6.062177826491071, 0.0]],
    ),
    "mixed_cf2onh": (
        [6, 9, 9, 8, 1, 7],
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.3, 0.1],
            [-1.2, 2.2, -0.3],
            [-1.0, -1.1, 2.3],
            [-0.5, -1.9, -1.6],
            [1.1, -2.6, 3.4],
        ],
    ),
    "chbr3": (
        [6, 1, 35, 35, 35],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.06],
            [3.43, 0.0, -1.21],
            [-1.715, 2.97, -1.21],
            [-1.715, -2.97, -1.21],
        ],
    ),
}


def evaluate(numbers, coords, params, s9):
    s6, s8, a1, a2 = params
    model = DispersionModel(np.array(numbers), np.array(coords, dtype=float))
    p = RationalDampingParam(s6=s6, s8=s8, s9=s9, a1=a1, a2=a2, alp=14.0)
    r = model.get_dispersion(p, grad=True)
    return float(r["energy"]), np.asarray(r["gradient"]).tolist()


def main():
    systems = {}
    two_body = json.loads(TWO_BODY.read_text())["systems"]
    for name, sys_ in two_body.items():
        systems[name] = (sys_["numbers"], sys_["coords_bohr"])
    systems.update(EXTRA)

    out = {
        "source": f"simple-dftd3 (dftd3 python package) {dftd3.__version__}",
        "generator": "scripts/gen_d3_atm_reference.py",
        "note": "Hartree, Bohr, Hartree/Bohr. RationalDampingParam(s6,s8,a1,a2,"
        "s9,alp=14) passed explicitly; s9=0 is two-body only, s9=1 adds ATM "
        "(zero damping, rs9=4/3, alp+2=16).",
        "systems": {},
    }
    for name, (numbers, coords) in systems.items():
        entry = {"numbers": numbers, "coords_bohr": coords, "functionals": {}}
        for fname, params in FUNCTIONALS.items():
            e0, g0 = evaluate(numbers, coords, params, 0.0)
            e1, g1 = evaluate(numbers, coords, params, 1.0)
            entry["functionals"][fname] = {
                "params": dict(zip(("s6", "s8", "a1", "a2"), params)),
                "s9_0": {"energy": e0, "gradient": g0},
                "s9_1": {"energy": e1, "gradient": g1},
            }
            print(f"{name:14s} {fname:6s} E3 = {e1 - e0: .6e}")
        out["systems"][name] = entry
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
