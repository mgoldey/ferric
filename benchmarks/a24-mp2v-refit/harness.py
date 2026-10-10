#!/usr/bin/env python3
"""MP2-V refit harness: raw fragment energies for (r0, omega, b) scans.

For each (system, fragment) ONE SCF, then per (r0, omega) arm ONE attenuated
MP2, then ALL b values from one shared VV10 pair sum (ferric
`run_mp2_v_scan`). Raw per-fragment energies are written to JSON; interaction
energies / fits / validation are computed from that JSON by analyze.py with no
recomputation.

Fragments per system (frozen core on; ghost atoms via '@'):
  dimer, mA_cp (A + ghost B), mB_cp (ghost A + B)   -> counterpoise-corrected
  mA, mB (monomer in its own basis)                  -> non-CP (published convention)

Usage:
  harness.py --set a24 --basis atz --arms arms_primary.json --bs bgrid.json \
      --out results/a24_atz.json [--systems 1,2,3] [--nlc 50,50] [--coulomb-mp2]

arms JSON: {"arms": [{"r0": 1.00, "r0omega": null}, {"r0": 1.00, "r0omega": 4.0}, ...]}
  r0 in Angstrom; r0omega = r0*omega (dimensionless); null = published linked
  width (omega=None path, byte-identical to the published method).
bs JSON: {"bs": [..]} or a plain list.

Resumable: fragments already present in --out (same arms/bs/nlc/basis) are
skipped. Run under scripts/ferric-limited with OPENBLAS_NUM_THREADS=1.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = 627.509474  # Hartree -> kcal/mol (same constant as the A24 scripts)

BASES = {
    "adz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "atz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit"),
}
ZSYM = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18,
}  # fmt: skip


def n_frozen_core(atoms):
    """Frozen-core orbital count over REAL atoms only: 0 for H/He, 1 for Li-Ne,
    5 for Na-Ar ([Ne] core). Ghost atoms ('@X') carry no electrons => 0.
    (scripts/scan_a24_aqz_terfc_r0_mae.py counts Ar as 1 -- a known defect for
    A24 #20/#21; this function is the corrected rule.)"""
    n = 0
    for s, *_ in atoms:
        if s.startswith("@"):
            continue
        z = ZSYM[s]
        n += 0 if z <= 2 else (1 if z <= 10 else 5)
    return n


def ghost(atoms):
    return [["@" + s, x, y, z] for s, x, y, z in atoms]


def fragments(sysrec):
    a, b = sysrec["frag_a"], sysrec["frag_b"]
    return {
        "dimer": a + b,
        "mA_cp": a + ghost(b),
        "mB_cp": ghost(a) + b,
        "mA": a,
        "mB": b,
    }


def xyz_text(atoms, comment=""):
    lines = [str(len(atoms)), comment]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in atoms]
    return "\n".join(lines) + "\n"


def load_sets():
    return json.loads((HERE / "sets.json").read_text())


def omega_from(r0, r0omega):
    return None if r0omega is None else r0omega / r0


def run_fragment(ferric, atoms, basis, arms, bs, nlc, coulomb, df_exact=False):
    obs_name, aux_name = BASES[basis]
    mol = ferric.Molecule.from_xyz_string(xyz_text(atoms), 0, 1)
    obs = ferric.BasisSet.bundled(obs_name)
    aux = ferric.BasisSet.bundled(aux_name)
    t0 = time.time()
    kw = dict(
        frozen_core=n_frozen_core(atoms),
        nlc_grid=tuple(nlc),
        include_coulomb_mp2=coulomb,
    )
    if df_exact:
        kw.update(df_j_aux="exact", df_k_aux="exact")
    r = ferric.run_mp2_v_scan(
        mol,
        obs,
        aux,
        [(a["r0"], omega_from(a["r0"], a["r0omega"])) for a in arms],
        list(bs),
        **kw,
    )
    r["frozen_core"] = kw["frozen_core"]
    r["seconds"] = time.time() - t0
    return r


def atomic_write(path, obj):
    tmp = str(path) + f".tmp{os.getpid()}"
    Path(tmp).write_text(json.dumps(obj))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["a24", "s22", "s66"])
    ap.add_argument("--basis", required=True, choices=list(BASES))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--bs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default="")
    ap.add_argument("--nlc", default="50,50")
    ap.add_argument("--coulomb-mp2", action="store_true")
    ap.add_argument("--fragments", default="dimer,mA_cp,mB_cp,mA,mB")
    a = ap.parse_args(argv)

    import ferric  # imported late so --help works without the extension

    arms = json.loads(Path(a.arms).read_text())["arms"]
    bs = json.loads(Path(a.bs).read_text())
    bs = bs["bs"] if isinstance(bs, dict) else bs
    nlc = [int(x) for x in a.nlc.split(",")]
    sets = load_sets()[a.set]["systems"]
    ids = [int(x) for x in a.systems.split(",")] if a.systems else sorted(map(int, sets))
    want = a.fragments.split(",")

    out = Path(a.out)
    meta = {
        "set": a.set,
        "basis": a.basis,
        "arms": arms,
        "bs": bs,
        "nlc": nlc,
        "coulomb_mp2": a.coulomb_mp2,
        "build": getattr(ferric, "build_info", lambda: {})(),
    }
    if out.exists():
        db = json.loads(out.read_text())
        for k in ("set", "basis", "arms", "bs", "nlc", "coulomb_mp2"):
            if db["meta"][k] != meta[k]:
                sys.exit(f"{out}: existing meta[{k}] differs; refusing to mix")
    else:
        db = {"meta": meta, "fragments": {}}
    for i in ids:
        frs = fragments(sets[str(i)])
        for tag in want:
            key = f"{a.set}-{i:02d}|{tag}"
            if key in db["fragments"]:
                continue
            r = run_fragment(ferric, frs[tag], a.basis, arms, bs, nlc, a.coulomb_mp2)
            db["fragments"][key] = r
            atomic_write(out, db)
            print(
                f"{key} natoms={len(frs[tag])} fc={r['frozen_core']} "
                f"npts={r['n_nlc_points']} {r['seconds']:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
