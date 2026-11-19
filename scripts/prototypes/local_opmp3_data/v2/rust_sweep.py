"""Rust local OP-MP3 (ferric.run_opmp3(local="amplitude-threshold")) error vs
the canonical RI OP-MP3 of the same reference, for ferric's current maps
("new": Fock screen kappa = 1, lambda-Boys 10) and the earlier ones ("old":
plain amplitude mask, plain Boys), HF and PBE0 references.

    PYTHONPATH=<shim to this checkout's build> python rust_sweep.py OUT.json

One canonical run per (system, reference); every local run repeats the SCF
(same settings, so the reference is the same to SCF convergence).
"""

import json
import sys
import time
from pathlib import Path

import ferric

ROOT = Path(__file__).resolve().parents[4]
M = ROOT / "testdata" / "molecules"
# (name, xyz, frozen core = heavy atoms, as in the prototype tables)
SYSTEMS = [
    ("butane", M / "alkane_4.xyz", 4),
    ("octane", M / "alkane_8.xyz", 8),
    ("naphthalene", M / "scaling" / "naphthalene.xyz", 10),
]
REFERENCES = ["hf", "pbe0"]
EPS = [1e-4, 1e-5, 1e-6, 1e-7]
MAPS = {
    "new": {},
    "old": {"fock_screen": False, "lambda_boys": 0.0},
}
SCF = {"scf_energy_conv": 1e-12, "scf_density_conv": 1e-10}


def main(out):
    bs = ferric.BasisSet.bundled("6-31g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    rows = []
    for name, xyz, fc in SYSTEMS:
        mol = ferric.Molecule.from_xyz(str(xyz))
        for ref in REFERENCES:
            t = time.perf_counter()
            can = ferric.run_opmp3(
                mol, bs, reference=ref, auxbasis=aux, frozen_core=fc, **SCF
            )
            t_can = time.perf_counter() - t
            ec = dict(can.correlation)
            for maps, kw in MAPS.items():
                for eps in EPS:
                    t = time.perf_counter()
                    r = ferric.run_opmp3(
                        mol,
                        bs,
                        reference=ref,
                        auxbasis=aux,
                        local="amplitude-threshold",
                        eps=eps,
                        frozen_core=fc,
                        **kw,
                        **SCF,
                    )
                    wall = time.perf_counter() - t
                    loc = r.local
                    row = {
                        "system": name,
                        "reference": ref,
                        "maps": maps,
                        "frozen_core": fc,
                        "eps": eps,
                        "keep": loc["keep_fraction"],
                        "n_pairs": loc["n_pairs"],
                        "lambda_boys": loc["lambda_boys"],
                        "cg_iterations": loc["cg_iterations"],
                        "cg_relres": loc["cg_relres"],
                        "e3_gap": loc["e3"] - loc["e3_2n1"],
                        "e_nbs_diff": r.e_nbs - can.e_nbs,
                        "err": {k: v - ec[k] for k, v in r.correlation.items()},
                        "corr_canonical": ec,
                        "wall_s": wall,
                        "wall_canonical_s": t_can,
                        "build": ferric.__build__,
                    }
                    rows.append(row)
                    e = row["err"]
                    print(
                        f"{name:12s} {ref:5s} {maps:4s} eps={eps:.0e} "
                        f"keep={row['keep']:.3f} dE2={e['mp2']:+.2e} "
                        f"dOP={e['op-mp3']:+.2e} gap={row['e3_gap']:+.1e} "
                        f"{wall:.1f}s",
                        flush=True,
                    )
                    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
