"""Generate PySCF TDA reference data for the ferric TDA-DFT spike.

Uses PySCF's `tdscf.rhf.get_ab` to build the EXACT dense A matrix and
diagonalizes it directly, rather than running the Davidson solver. Two reasons:

  1. ferric's spike solves the dense eigenproblem exactly, so a dense PySCF
     reference is like-for-like -- no iterative-solver tolerance enters the
     comparison on either side.
  2. PySCF's own Davidson failed to converge all 8 requested states for
     water/cc-pVDZ/LDA at conv_tol=1e-9, which would otherwise have silently
     contaminated the reference.

Emits, per (functional, basis): all excitation energies (eV), length-gauge
oscillator strengths, and the dominant (i,a) particle-hole pair of each state
-- so ferric can match STATES by character, not by sorted energy.

The PySCF side uses EXACT 4-index ERIs (no density fitting) and the PySCF
default grid; ferric uses RI and its own Becke-Lebedev grid. Residual
disagreement from those two choices is real and must be reported, not tuned away.
"""
import json
import sys
import numpy as np
from pyscf import gto, scf, dft, tdscf

HARTREE2EV = 27.211386245988

WATER = """
O 0.0000 0.0000 0.1173
H 0.0000 0.7572 -0.4692
H 0.0000 -0.7572 -0.4692
"""

NSTATES = 8


def run(basis, xc):
    mol = gto.M(atom=WATER, basis=basis, unit="Angstrom", verbose=0, spin=0, charge=0)
    if xc is None:
        mf = scf.RHF(mol)
        label = "HF"
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
        label = xc
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged, f"{basis}/{label} SCF not converged"

    nocc = int((mf.mo_occ > 0).sum())
    nvir = mf.mo_occ.size - nocc
    n = nocc * nvir

    # Exact dense TDA A matrix, indexed [i, a, j, b].
    a, _b = tdscf.rhf.get_ab(mf)
    amat = a.reshape(n, n)
    asym = np.abs(amat - amat.T).max() / max(np.abs(amat).max(), 1e-30)
    assert asym < 1e-9, f"PySCF A matrix not symmetric: {asym:e}"
    amat = 0.5 * (amat + amat.T)
    w, v = np.linalg.eigh(amat)

    # Length-gauge oscillator strengths, same convention as ferric's
    # tda_oscillator_strengths (and as PySCF's TDA.oscillator_strength):
    #   <0|r|n> = sqrt(2) sum_ia X_n(ia) <i|r|a>;  f = (2/3) w |mu|^2
    orbo = mf.mo_coeff[:, mf.mo_occ > 0]
    orbv = mf.mo_coeff[:, mf.mo_occ == 0]
    with mol.with_common_orig((0, 0, 0)):
        mu_ao = mol.intor("int1e_r", comp=3)
    dip_ia = np.einsum("pi,xpq,qa->xia", orbo, mu_ao, orbv)

    states = []
    for k in range(min(NSTATES, n)):
        x = v[:, k].reshape(nocc, nvir)
        mu = np.sqrt(2.0) * np.einsum("ia,xia->x", x, dip_ia)
        f = (2.0 / 3.0) * w[k] * float(mu @ mu)
        idx = int(np.argmax(x * x))
        i, aa = divmod(idx, nvir)
        states.append({
            "omega_ev": float(w[k] * HARTREE2EV),
            "omega_ha": float(w[k]),
            "osc_length": f,
            "transition_dipole": [float(t) for t in mu],
            "dominant_ia": [i, aa],
            "dominant_weight": float((x * x).ravel()[idx] / (x * x).sum()),
        })

    return {
        "basis": basis,
        "xc": label,
        "e_scf": float(mf.e_tot),
        "nocc": nocc,
        "nvir": nvir,
        "n_ia": n,
        "mo_energy": [float(x) for x in mf.mo_energy],
        "states": states,
        "pyscf_version": __import__("pyscf").__version__,
        "note": "DENSE get_ab A-matrix eigh (no Davidson); exact 4-index ERIs; "
                "PySCF default grid; length-gauge oscillator strengths",
    }


if __name__ == "__main__":
    out = {}
    for basis in ["sto-3g", "cc-pvdz"]:
        for xc in [None, "lda,vwn", "pbe,pbe", "b3lyp"]:
            key = f"{basis}__{xc if xc else 'HF'}"
            print(f"running {key}", file=sys.stderr)
            out[key] = run(basis, xc)
            e = [f"{s['omega_ev']:.4f}" for s in out[key]["states"][:4]]
            print(f"  E_scf={out[key]['e_scf']:.10f}  first 4 (eV): {e}", file=sys.stderr)
    path = sys.argv[1] if len(sys.argv) > 1 else "tda_refs.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}", file=sys.stderr)
