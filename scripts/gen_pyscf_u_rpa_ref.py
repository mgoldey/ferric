"""
Generate PySCF RI-RPA reference values for ferric's U-PDEP-RPA (C4).

Builds α and β χ₀ tensors separately, forms ε̃ = I + Π_α + Π_β at each
Gauss-Legendre quadrature point, integrates the trace-log:

    E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))]

with λ_α the eigenvalues of ε̃(iω_k). This matches ferric's convention
exactly (see [[pyscf-ri-rpa-convention]] and [[ri-rpa-spin-factor]]).

Writes JSON to testdata/reference/{label}_{basis}_u-rpa.json with
keys {e_rpa, scf_energy, total_energy, basis, aux_basis, method,
mult, n_quad, u0}.

Usage:
    python scripts/gen_pyscf_u_rpa_ref.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/matt/qc/pyscf")
from pyscf import df, gto, scf
from scipy.linalg import eigh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(ROOT, "testdata/reference"), exist_ok=True)


def gauss_legendre_freq(n: int, u0: float):
    """Quadrature on [0, ∞) via Gauss-Legendre on [-1, 1] with the
    PySCF convention ω = u0 (1+x)/(1-x).  Matches ferric's
    QuadratureScheme::GaussLegendre default (u0 = 0.5)."""
    x, w = np.polynomial.legendre.leggauss(n)
    omegas = u0 * (1 + x) / (1 - x)
    weights = 2 * u0 * w / (1 - x) ** 2
    return omegas, weights


def build_b_ov_spin(mol, mf, dfbs, spin: int):
    """Build B^P_{ia} = V^{-1/2}(P|ia) for one spin channel of a UHF result.

    NOTE on eps source: we use `eigh(F_σ, S)` directly rather than
    `mf.mo_energy[σ]`. PySCF post-processes mo_energy for some UHF/UKS
    convergence paths (notably the H-atom-doublet edge case) in ways that
    do not match the raw Fock spectrum. For RI-RPA we need the *true*
    Koopmans energies that go with the converged Fock operator, which is
    `eigh(F_σ, S)`.  Ferric uses the raw spectrum, so apples-to-apples
    requires we do the same here.
    """
    aux = df.addons.make_auxbasis(mol, mf=False) if dfbs is None else dfbs
    auxmol = df.addons.make_auxmol(mol, auxbasis=aux)
    nao = mol.nao
    naux = auxmol.nao

    # (P|μν) in AO basis
    pmunu = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    pmunu = pmunu.reshape(nao, nao, naux).transpose(2, 0, 1)  # (naux, nao, nao)

    # V_2c = (P|Q)
    v2c = auxmol.intor("int2c2e")
    w, u = eigh(v2c)
    vinv = u @ np.diag(1.0 / np.sqrt(w)) @ u.T

    # Use eigh(F, S) — see note above.
    S = mol.intor("int1e_ovlp")
    F = mf.get_fock()
    eps_real, C_real = eigh(F[spin], S)

    occ = mf.mo_occ[spin]
    # Map mo_occ onto the eigh ordering: lowest |nocc_σ| eigenvalues = occupied.
    nocc = int(round(occ.sum()))
    occ_idx = np.arange(nocc)
    vir_idx = np.arange(nocc, len(eps_real))
    Co = C_real[:, occ_idx]
    Cv = C_real[:, vir_idx]
    eps_o = eps_real[occ_idx]
    eps_v = eps_real[vir_idx]

    # (P|ia) = Σ_{μν} C_μi (P|μν) C_νa
    pmuv_mo = np.einsum("Pmn,mi,na->Pia", pmunu, Co, Cv, optimize=True)
    b_ov = np.einsum("PQ,Qia->Pia", vinv, pmuv_mo).reshape(naux, -1)
    return b_ov, eps_o, eps_v


def u_rpa_energy(mol, mf, dfbs, n_quad: int = 20, u0: float = 0.5) -> float:
    """E_c^U-RPA via spin-summed dielectric, evaluated on Gauss-Legendre grid."""
    omegas, weights = gauss_legendre_freq(n_quad, u0)
    bA, eo_a, ev_a = build_b_ov_spin(mol, mf, dfbs, 0)
    bB, eo_b, ev_b = build_b_ov_spin(mol, mf, dfbs, 1)
    naux = bA.shape[0]

    def pi_sigma(b, eo, ev, omega):
        # Π_σ_{PQ} = 2 Σ_ia B^P_ia · e_ia/(ω²+e_ia²) · B^Q_ia
        nocc = len(eo)
        nvir = len(ev)
        e_ia = (ev[None, :] - eo[:, None]).ravel()  # (nov,)
        factor = 2.0 * e_ia / (omega * omega + e_ia * e_ia)
        bs = b * np.sqrt(factor)[None, :]
        return bs @ bs.T

    e_c = 0.0
    for w, omega in zip(weights, omegas):
        eps_mat = np.eye(naux) + pi_sigma(bA, eo_a, ev_a, omega) \
                                + pi_sigma(bB, eo_b, ev_b, omega)
        lam, _ = eigh(eps_mat)
        contrib = np.sum(np.log(lam) + (1.0 - lam))
        e_c += w * contrib
    return e_c / (2.0 * np.pi)


def make_uhf(atom: str, basis: str, spin_2s: int):
    mol = gto.M(atom=atom, basis=basis, unit="angstrom",
                charge=0, spin=spin_2s, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"UHF did not converge for {atom}")
    return mol, mf


def make_uhf_charged(atom: str, basis: str, charge: int, spin_2s: int):
    mol = gto.M(atom=atom, basis=basis, unit="angstrom",
                charge=charge, spin=spin_2s, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"UHF did not converge for {atom} charge={charge}")
    return mol, mf


cases = [
    # (output stub, atom, basis, aux_basis, 2S, mult, charge)
    ("h_cc-pvdz_u-rpa",  "H 0 0 0",          "cc-pvdz", "cc-pvdz-ri", 1, 2, 0),
    ("oh_cc-pvdz_u-rpa", "O 0 0 0; H 0 0 0.97", "cc-pvdz", "cc-pvdz-ri", 1, 2, 0),
    # Cation comparison vs ferric GW100 driver — same geometry (Bohr→Angstrom-equivalent
    # taken from the driver's literal angstrom values, since ferric parse_xyz treats the
    # XYZ block as angstroms by default).
    ("h2o_cation_cc-pvdz_u-rpa",
     "O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
     "cc-pvdz", "cc-pvdz-ri", 1, 2, 1),
    ("h2o_cation_aug-cc-pvtz_u-rpa",
     "O 0.0 0.0 0.117790; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161",
     "aug-cc-pvtz", "aug-cc-pvtz-ri", 1, 2, 1),
]

for stub, atom, basis, aux, spin_2s, mult, charge in cases:
    mol, mf = make_uhf_charged(atom, basis, charge, spin_2s)
    e_c = u_rpa_energy(mol, mf, aux, n_quad=20, u0=0.5)
    out = {
        "atom": atom,
        "basis": basis,
        "aux_basis": aux,
        "mult": mult,
        "scf_method": "uhf",
        "scf_energy": float(mf.e_tot),
        "e_rpa": float(e_c),
        "total_energy": float(mf.e_tot + e_c),
        "n_quad": 20,
        "u0": 0.5,
        "method": "u-pdep-rpa",
    }
    print(f"{stub:35s} E_scf={mf.e_tot:.10f}  E_c={e_c:.10f}  E_tot={mf.e_tot+e_c:.10f}")
    with open(os.path.join(ROOT, f"testdata/reference/{stub}.json"), "w") as f:
        json.dump(out, f, indent=2)

print("Wrote testdata/reference/*_u-rpa.json")
