#!/usr/bin/env python3
"""Generate crates/ferric-d3/testdata/d3bj_params_snapshot.json.

Reference: simple-dftd3 (the `dftd3` Python package). For every functional
name ferric's D3(BJ) table accepts (`ferric_d3::d3bj_tabulated_functionals`),
record the upstream rational-damping (s6, s8, a1, a2) and its DOI, read from
the package's `parameters.toml`.

The TOML is the Python side's copy of the table; the numbers ferric is
compared against must also be the ones the compiled Fortran library uses. So
each entry is cross-checked: the two-body energy of a test molecule with
`RationalDampingParam(method=<key>, atm=False)` (compiled table) must equal the
energy with the TOML values passed explicitly. A mismatch aborts.

Functionals ferric refuses because upstream has NO rational-damping entry are
listed under `no_upstream_bj`; for each one the generator asserts the compiled
library also refuses it.

Consumed by crates/ferric-d3/tests/params_snapshot.rs.

Run (needs the dftd3 package; python goes through the compile lock):
    flock -o /tmp/ferric-compile.lock python3 scripts/gen_d3bj_param_snapshot.py
"""

import json
import pathlib
import sys

import dftd3
import numpy as np
import tomllib
from dftd3.interface import DispersionModel, RationalDampingParam

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "crates/ferric-d3/testdata/d3bj_params_snapshot.json"

# ferric lookup key -> simple-dftd3 parameter key. Keys are ferric's
# normalised spelling (lowercase, no '-', '_' or ' '); aliases map to the same
# upstream entry.
FERRIC_TO_UPSTREAM = {
    "blyp": "blyp",
    "bp86": "bp",
    "bp": "bp",
    "b97d": "b97d",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "revpbe": "revpbe",
    "b3lyp": "b3lyp",
    "tpss": "tpss",
    "tpss0": "tpss0",
    "b2plyp": "b2plyp",
    "bhlyp": "bhlyp",
    "bhandhlyp": "bhlyp",
    "scan": "scan",
    "r2scan": "r2scan",
    "wb97x": "wb97x",
}

# Names ferric rejects by name because upstream has no D3(BJ) fit for them.
NO_UPSTREAM_BJ = ["m062x"]

# Water dimer-like fragment with spread coordination numbers (Bohr).
NUMBERS = np.array([8, 1, 1, 8, 1, 1, 6, 1])
COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.43, 1.11],
        [0.0, -1.43, 1.11],
        [5.4, 0.2, 0.3],
        [6.1, 1.6, -0.8],
        [6.0, -1.5, -0.5],
        [-2.1, 4.9, -3.2],
        [-3.9, 5.3, -2.4],
    ]
)


def two_body(param):
    return DispersionModel(NUMBERS, COORDS).get_dispersion(param, grad=False)["energy"]


def load_toml():
    """The package's parameters.toml, read with the stdlib parser (the
    package's own loader needs tomli/tomlkit/toml, which are optional)."""
    path = pathlib.Path(dftd3.__file__).with_name("parameters.toml")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def bj_entry(toml, key):
    """Upstream d3.bj entry for `key` with the [default.parameter] values
    applied, as dftd3.parameters._get_params does."""
    row = dict(toml["default"]["parameter"]["d3"]["bj"])
    row.update(toml["parameter"][key]["d3"]["bj"])
    return row


def main():
    toml = load_toml()
    params = {}
    for ferric_key, up_key in FERRIC_TO_UPSTREAM.items():
        meta = bj_entry(toml, up_key)
        row = {
            "upstream_key": up_key,
            "s6": float(meta["s6"]),
            "s8": float(meta["s8"]),
            "a1": float(meta["a1"]),
            "a2": float(meta["a2"]),
            "doi": meta.get("doi"),
        }
        e_lib = two_body(RationalDampingParam(method=up_key, atm=False))
        e_toml = two_body(
            RationalDampingParam(
                s6=row["s6"], s8=row["s8"], a1=row["a1"], a2=row["a2"], s9=0.0
            )
        )
        if abs(e_lib - e_toml) > 1e-14 * max(1.0, abs(e_lib)):
            sys.exit(
                f"{up_key}: compiled table and parameters.toml disagree "
                f"({e_lib!r} vs {e_toml!r})"
            )
        params[ferric_key] = row

    for name in NO_UPSTREAM_BJ:
        if "bj" in toml["parameter"].get(name, {}).get("d3", {}):
            sys.exit(f"{name}: parameters.toml now HAS a d3.bj entry")
        try:
            RationalDampingParam(method=name, atm=False)
        except RuntimeError as err:
            print(f"{name}: compiled library refuses it ({err})")
        else:
            sys.exit(f"{name}: compiled library now HAS a rational-damping entry")

    out = {
        "source": f"simple-dftd3 (dftd3 python package) {dftd3.__version__}",
        "generator": "scripts/gen_d3bj_param_snapshot.py",
        "note": "rational damping (s6, s8, a1, a2), keyed by ferric's normalised "
        "functional name; s9 is not recorded (ferric loads named functionals "
        "with ATM off, as load_rational_damping(atm=False) does)",
        "params": params,
        "no_upstream_bj": NO_UPSTREAM_BJ,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(params)} names, {len(NO_UPSTREAM_BJ)} refused)")


if __name__ == "__main__":
    main()
