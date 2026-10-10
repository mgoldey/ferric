#!/usr/bin/env python3
"""Stretched-dimer geometries (rigid monomers) and the HF/aTZ geometry check
against the paper SI table '3' (SCF/aV5Z non-CP). See PREREGISTRATION.md Addendum A.

  G1: monomer B translated so the COM-COM vector is scaled by f.
  G2: monomer B translated so the closest-atom-pair vector is scaled by f.

Usage: stretch.py hf --systems 2,4,5,8,19 --factors 1.0 [--mode G1|G2] --out x.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

K = harness.K
FACTORS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
MASS = {"H": 1.00794, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994,
        "F": 18.9984, "Ar": 39.948}  # fmt: skip


def com(frag):
    m = np.array([MASS[a[0]] for a in frag])
    x = np.array([a[1:] for a in frag], float)
    return (m[:, None] * x).sum(0) / m.sum()


def shift_vec(fa, fb, mode):
    if mode == "G1":
        return com(fb) - com(fa)
    xa = np.array([a[1:] for a in fa], float)
    xb = np.array([a[1:] for a in fb], float)
    d = xb[None, :, :] - xa[:, None, :]
    r = np.linalg.norm(d, axis=2)
    i, j = np.unravel_index(np.argmin(r), r.shape)
    return d[i, j]


def stretched(sysrec, f, mode):
    fa, fb = sysrec["frag_a"], sysrec["frag_b"]
    s = (f - 1.0) * shift_vec(fa, fb, mode)
    fb2 = [[a[0], *(np.array(a[1:]) + s).tolist()] for a in fb]
    return fa, fb2


def frags_for(sysrec, f, mode):
    fa, fb = stretched(sysrec, f, mode)
    return harness.fragments({"frag_a": fa, "frag_b": fb})


def paper(key, sid):
    d = json.loads((HERE / "paper_si" / "a21x12.json").read_text())
    return d[key]["rows"][str(sid)]["v"]


def hf_energy(ferric, atoms, basis, exact=False):
    obs = ferric.BasisSet.bundled(harness.BASES[basis][0])
    mol = ferric.Molecule.from_xyz_string(harness.xyz_text(atoms), 0, 1)
    aux = "exact" if exact else "def2-universal-jkfit"
    r = ferric.run_rhf(
        mol, obs, df_j_aux=aux, df_k_aux=aux, energy_conv=1e-10, density_conv=1e-9,
        max_iter=400,
    )  # fmt: skip
    assert r.converged
    return r.energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hf"])
    ap.add_argument("--systems", default="2,4,5,8,19")
    ap.add_argument("--factors", default="1.0")
    ap.add_argument("--mode", default="G1")
    ap.add_argument("--basis", default="atz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import ferric

    sets = harness.load_sets()["a24"]["systems"]
    res = {}
    for sid in [int(x) for x in a.systems.split(",")]:
        rec = sets[str(sid)]
        res[sid] = {}
        mono = {}
        for tag in ("mA", "mB"):
            mono[tag] = hf_energy(ferric, frags_for(rec, 1.0, a.mode)[tag], a.basis)
        for f in [float(x) for x in a.factors.split(",")]:
            d = hf_energy(ferric, frags_for(rec, f, a.mode)["dimer"], a.basis)
            e = (d - mono["mA"] - mono["mB"]) * K
            p = paper("3", sid)[FACTORS.index(f)]
            res[sid][f] = {"hf_ncp": e, "paper_s3": p, "dev": e - p}
            print(
                f"{sid} f={f} {a.mode} HF/{a.basis} nonCP {e:+.4f} paper {p:+.4f} dev {e - p:+.4f}",
                flush=True,
            )
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
