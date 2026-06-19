#!/usr/bin/env python3
"""Generate a PySCF ECP-matrix reference for ferric's libecpint shim (Unit 2).

Dumps, for a single iodine atom with def2-SVP + def2-ECP, a self-contained JSON:
  - the Cartesian Gaussian shells with libecpint-ready contraction coefficients
    (libcint bas_ctr_coeff x gto_norm: primitive normalization folded in, since
    libecpint applies no internal normalization, giving the bare-Cartesian
    convention libcint uses with cart=True);
  - the ECP semilocal expansion (per-primitive l, n=r_power, exponent, coef);
  - the reference SPHERICAL ECPscalar matrix (mol.intor('ECPscalar')).

ferric's tests/ecp_matrix.rs feeds the shells+ecp to the libecpint shim, applies
the per-shell Cartesian->spherical transform, and compares the resulting V_ECP
element-by-element against `ref_matrix` (the spherical reference -- the
production convention, matching libint's spherical AO basis).

Verified recipe: c2s^T @ (gto_norm-folded libecpint Cartesian) @ c2s == spherical
ECPscalar to ~1e-17 (see commit message / report).

Run:  python3 scripts/gw100/gen_ecp_ref.py
Out:  testdata/reference/iodine_def2svp_ecpscalar.json
"""
import json
import os
import numpy as np
from pyscf import gto

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "testdata", "reference",
                                    "iodine_def2svp_ecpscalar.json"))


def main():
    # Spherical reference (production convention). The Gaussian shells we hand the
    # shim are Cartesian (libecpint emits Cartesian), and the per-shell c2s
    # transform maps them to this spherical reference.
    mol = gto.M(atom="I 0.0 0.0 0.0", basis="def2-svp", ecp="def2-svp",
                spin=1, charge=0, unit="Bohr", cart=False)

    shells = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        es = mol.bas_exp(ib)
        cs = mol.bas_ctr_coeff(ib).ravel()
        # libcint stores ctr_coef with the primitive norm split out; fold gto_norm
        # back in so libecpint (which does no internal normalization) produces the
        # same bare-Cartesian integrals libcint uses under cart=True.
        full = [float(c * gto.gto_norm(l, a)) for a, c in zip(es, cs)]
        shells.append({
            "l": l,
            "center": [0.0, 0.0, 0.0],
            "exps": [float(x) for x in es],
            "coefs": full,
        })

    # ECP semilocal expansion, flattened per primitive.
    # gto.basis.load_ecp returns [n_core, [ [l, [ [], [], terms_for_n=2, ... ]], ... ]].
    ecp_raw = gto.basis.load_ecp("def2-svp", "I")
    n_core = int(ecp_raw[0])
    ams, ns, exps, coefs = [], [], [], []
    for l_block in ecp_raw[1]:
        l = int(l_block[0])  # -1 means local; libecpint wants the actual max-l, see below
        for n_power, terms in enumerate(l_block[1]):
            for (zeta, d) in terms:
                ams.append(l)
                ns.append(int(n_power))
                exps.append(float(zeta))
                coefs.append(float(d))

    # PySCF encodes the local channel as l = -1. libecpint instead treats the
    # MAXIMUM angular momentum as local, so remap -1 -> (max projector l + 1).
    proj_max = max(a for a in ams if a >= 0)
    local_l = proj_max + 1
    ams = [local_l if a == -1 else a for a in ams]

    ecp = {
        "n_core": n_core,
        "center": [0.0, 0.0, 0.0],
        "ams": ams,
        "ns": ns,
        "exps": exps,
        "coefs": coefs,
    }

    vecp = mol.intor("ECPscalar")  # spherical
    nsph = vecp.shape[0]
    ncart = sum((s["l"] + 1) * (s["l"] + 2) // 2 for s in shells)
    ref = {
        "description": "iodine def2-SVP + def2-ECP, spherical ECPscalar matrix",
        "nsph": int(nsph),
        "ncart": int(ncart),
        "shells": shells,
        "ecp": ecp,
        "ref_matrix": [[float(v) for v in row] for row in vecp],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(ref, f)
    print(f"wrote {OUT}")
    print(f"  nsph={nsph} ncart={ncart}  n_core={n_core}  necp_terms={len(ams)}  local_l={local_l}")
    print(f"  trace(ECPscalar)={np.trace(vecp):.10f}  max|V|={np.max(np.abs(vecp)):.10f}")


if __name__ == "__main__":
    main()
