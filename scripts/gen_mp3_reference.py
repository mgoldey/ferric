import os
import sys, json
sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))
import numpy as np
from pyscf import gto, scf, ao2mo

def mp3_spinorbital(atom, basis, name):
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom')
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mo = mf.mo_coeff; nmo = mo.shape[1]
    no = mol.nelectron  # occupied spin orbitals
    nso = 2 * nmo
    eps = np.repeat(mf.mo_energy, 2)
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, mo), nmo)   # chemist (pq|rs), spatial
    eri_phys = eri_mo.transpose(0, 2, 1, 3)                 # <pq|rs> spatial
    gs = np.zeros((nso, nso, nso, nso))
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    if p % 2 == r % 2 and q % 2 == s % 2:
                        gs[p, q, r, s] = eri_phys[p//2, q//2, r//2, s//2]
    asym = gs - gs.transpose(0, 1, 3, 2)                    # <pq||rs>
    o = slice(0, no); v = slice(no, nso)
    D = (eps[o, None, None, None] + eps[None, o, None, None]
         - eps[None, None, v, None] - eps[None, None, None, v])
    t = asym[o, o, v, v] / D
    e_mp2 = 0.25 * np.einsum('ijab,ijab->', t, asym[o, o, v, v])
    e_pp = 0.125 * np.einsum('ijab,abcd,ijcd->', t, asym[v, v, v, v], t)
    e_hh = 0.125 * np.einsum('ijab,klij,klab->', t, asym[o, o, o, o], t)
    e_ph = np.einsum('ijab,kbcj,ikac->', t, asym[o, v, v, o], t)
    e_mp3 = e_pp + e_hh + e_ph
    rec = {
        "molecule": name, "basis": basis, "method": "mp3_spinorbital",
        "rhf_energy": float(mf.e_tot),
        "mp2_corr": float(e_mp2), "mp3_corr": float(e_mp3),
        "mp3_total_corr": float(e_mp2 + e_mp3),
        "total_energy": float(mf.e_tot + e_mp2 + e_mp3),
        "e_pp": float(e_pp), "e_hh": float(e_hh), "e_ph": float(e_ph),
        "nbasis": int(nmo), "nocc": int(mol.nelectron // 2),
    }
    return rec

cases = [
    ("H 0 0 0; H 0 0 0.74", "sto-3g", "h2", "h2_sto-3g_mp3.json"),
    ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "cc-pvdz", "h2o", "h2o_cc-pvdz_mp3.json"),
]
for atom, basis, name, fname in cases:
    rec = mp3_spinorbital(atom, basis, name)
    with open(f"testdata/reference/{fname}", "w") as f:
        json.dump(rec, f, indent=2)
    print(fname, "MP3=", rec["mp3_corr"], "MP2=", rec["mp2_corr"], "ph=", rec["e_ph"])
