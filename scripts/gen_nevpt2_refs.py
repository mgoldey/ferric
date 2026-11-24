#!/usr/bin/env python
"""SC-NEVPT2 references for crates/ferric-ci/tests/nevpt2.rs.

Per system (``scripts/prototypes/nevpt2_proto.py`` systems h2o, n2,
n2-stretched):

* ``ferric_casscf``: ferric's own CASSCF (``run_casscf``, ``casscf_grad_tol``
  1e-8, ``canonicalize=True``): orbitals, CI vector, energy, gradient norm.
* ``prototype``: the numpy prototype's Wick construction (``nevpt2_wick``)
  on exactly those orbitals and that CI vector, with no further
  canonicalization (``build_ref(..., canon=False)``: orbital energies are
  the diagonal of ``F^I + F^A``). The Rust test feeds the same inputs to
  ``sc_nevpt2`` with ``canonicalize = false``, so the two differ by rounding
  only (anchor 1).
* ``pyscf``: ``pyscf.mrpt.NEVPT`` per class on PySCF's own CASSCF on
  ferric's AO integrals, and the ingredients of the bar of a comparison
  between two independently converged CASSCF references
  (``nevpt2_proto.compare_ref`` with FD): ``|dE_class/dkappa|`` over every
  nonredundant rotation (CI re-solved, orbitals re-canonicalized), PySCF's
  orbital-gradient norm, the lowest eigenvalue of the CI-relaxed orbital
  Hessian, and the spread of each class under rotations inside degenerate
  core or virtual groups (anchor 2). The Rust test runs its own CASSCF and
  adds its own gradient norm.

Usage (needs PySCF; one heavy op at a time on the shared box; the FD bars
take about 9 minutes):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 PYTHONPATH=<shim> \\
    /home/matt/qc/ferric/.venv/bin/python scripts/gen_nevpt2_refs.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pyscf

import ferric

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "reference"
sys.path.insert(0, str(ROOT / "scripts" / "prototypes"))
import nevpt2_proto as nv  # noqa: E402

GRAD_TOL = 1e-8
SYSTEMS = ("h2o", "n2", "n2-stretched")


def by_name(d: dict) -> dict:
    return {nv.PYSCF_NAME[k]: float(v) for k, v in d.items()}


def xyz(mol) -> str:
    sym, xyzs = mol.symbols(), np.asarray(mol.coords())
    lines = [str(len(sym)), "angstrom"]
    lines += [f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(sym, xyzs)]
    return "\n".join(lines) + "\n"


def one(label: str) -> dict:
    mol, basis, na, ne, nc = nv.system(label)
    bs = ferric.BasisSet.bundled(basis)
    ints = ferric.AoIntegrals(mol, bs)
    kw = dict(casscf_grad_tol=GRAD_TOL, canonicalize=True)
    if nc is not None:
        kw["n_core"] = nc
    r = ferric.run_casscf(mol, bs, na, ne, **kw)
    if not r.converged:
        raise RuntimeError(f"{label}: ferric CASSCF did not converge")
    ref, chk = nv.ref_from_casscf(ints, r, ne, canon=False)
    wick = by_name(nv.nevpt2_wick(ref))
    nv.log(f"{label}: prototype on ferric canonical orbitals {wick}; checks {chk}")

    row = nv.compare_ref(label, fd=True)
    if not nv.ref_ok(row):
        raise RuntimeError(f"{label}: prototype vs PySCF fails its own bars")
    c = np.asarray(r.mo_coeff)
    return {
        "system": label,
        "basis": basis,
        "xyz": xyz(mol),
        "n_core": int(r.n_core),
        "n_active": na,
        "n_elec_active": list(ne),
        "ferric_casscf": {
            "casscf_grad_tol": GRAD_TOL,
            "e_total": r.e_total,
            "grad_norm": r.grad_norm,
            "mo_coeff_shape": list(c.shape),
            "mo_coeff": c.ravel().tolist(),
            "ci_vector": np.asarray(r.ci_vector).ravel().tolist(),
        },
        "prototype": wick,
        "prototype_checks": {k: float(v) for k, v in chk.items()},
        "pyscf": {
            "classes": by_name(row["own"]),
            "e_casscf": float(row["e_p"]),
            "grad_norm": float(row["g_p"]),
            "lambda_min": float(row["lam"]),
            "dE_dkappa": by_name(row["grads"]),
            "degenerate_spread": by_name(row["spread"]),
            "prototype_on_ferric_casscf": by_name(row["wick"]),
            "ferric_grad_norm_at_prototype_run": float(row["g_f_proto"]),
        },
    }


def main() -> int:
    out = {
        "generator": "scripts/gen_nevpt2_refs.py",
        "date": datetime.date.today().isoformat(),
        "pyscf_version": pyscf.__version__,
        "ferric_build": str(ferric.__build__),
        "systems": [one(s) for s in SYSTEMS],
    }
    path = OUT / "nevpt2_refs.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    nv.log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
