"""PySCF references for the second set of building-block ports in examples/python/.

The ports are the Phase 1 exit test's second set (wiki/moonshot-reference-code.md,
Pillar 2); each reference here is PySCF's own implementation of the method:

* ``delta_scf_from_blocks.py``: O/N 1s core-hole Delta-SCF, cc-pVDZ. The
  neutral RHF, then the doublet cation's UHF under ``scf.addons.mom_occ``,
  started from the neutral orbitals with the beta 1s electron removed.
* ``cphf_polarizability_from_blocks.py``: the static RHF polarizability of
  ``pyscf.prop.polarizability.rhf`` (PySCF's ``scf.cphf.solve`` on
  ``int1e_r`` with the RHF response function), cc-pVDZ, dipole origin at 0,
  as ``alpha``; and ``alpha_dense``, the same CPHF equations solved by a dense
  linear solve of the orbital Hessian PySCF's response function builds.
  MEASURED 2026-11-13 (water): the Krylov solve stops 2.5e-7 a.u. from the
  dense solution at tol 1e-11 and at 1e-13 alike.
* ``fno_mp2_from_blocks.py``: ``DFMP2.make_fno`` virtual natural occupations
  (the eigenvalues of the vv block of ``make_rdm1``) and the all-electron
  DF-MP2 correlation energy in the semicanonical FNOs for several numbers of
  kept virtuals, cc-pVDZ / cc-pVDZ-RI.

Every orbital and auxiliary basis is ferric's bundled BSE JSON handed to
PySCF (``ferric_basis`` from gen_pyscf_block_ports_refs.py). Each code
converges its own SCF. Exact J/K everywhere but the DF-MP2 integrals.

Usage: python scripts/gen_pyscf_block_ports_set2_refs.py
Writes testdata/reference/block_ports_set2_pyscf.json
"""

import json
import os
import sys

import numpy as np
from pyscf import df, gto, lib, scf
from pyscf.mp import dfmp2
from pyscf.scf import cphf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_pyscf_block_ports_refs import ROOT, converged, ferric_basis  # noqa: E402

OUT = os.path.join(ROOT, "testdata", "reference", "block_ports_set2_pyscf.json")
MOLECULES = ("water", "nh3")


def molecule(name: str, basis: str, charge: int = 0, spin: int = 0) -> gto.Mole:
    with open(os.path.join(ROOT, "testdata", "molecules", f"{name}.xyz")) as f:
        lines = f.read().splitlines()[2:]
    return gto.M(
        atom="\n".join(line for line in lines if line.strip()),
        basis=ferric_basis(basis),
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=0,
    )


def delta_scf(name: str, hole: int = 0) -> dict:
    neutral = converged(scf.RHF(molecule(name, "cc-pvdz")))
    mol = molecule(name, "cc-pvdz", charge=1, spin=1)
    nocc = neutral.mol.nelectron // 2
    occ_a = np.zeros(neutral.mo_coeff.shape[1])
    occ_a[:nocc] = 1.0
    occ_b = occ_a.copy()
    occ_b[hole] = 0.0
    mo0 = (neutral.mo_coeff, neutral.mo_coeff)
    ion = scf.addons.mom_occ(scf.UHF(mol), mo0, (occ_a, occ_b))
    ion = converged(ion, dm0=ion.make_rdm1(mo0, (occ_a, occ_b)))
    return {
        "basis": "cc-pvdz",
        "hole": hole,
        "e_neutral": neutral.e_tot,
        "e_cation": ion.e_tot,
        "ip": ion.e_tot - neutral.e_tot,
    }


def polarizability(name: str) -> dict:
    mf = converged(scf.RHF(molecule(name, "cc-pvdz")))
    occ = mf.mo_occ > 0
    orbo, orbv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    nocc = orbo.shape[1]
    with mf.mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mf.mol.intor_symmetric("int1e_r", comp=3)
    h1 = lib.einsum("xpq,pa,qi->xai", r, orbv, orbo)
    vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)

    def fvind(mo1):
        mo1 = mo1.reshape(-1, orbv.shape[1], nocc)
        dm1 = lib.einsum("xai,pa,qi->xpq", mo1, orbv, 2.0 * orbo)
        v1 = vresp(dm1 + dm1.transpose(0, 2, 1))
        return lib.einsum("xpq,pa,qi->xai", v1, orbv, orbo)

    def alpha_of(mo1):
        e2 = lib.einsum("xai,yai->xy", h1, mo1)
        return -2.0 * (e2 + e2.T)

    mo1 = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, h1, max_cycle=100, tol=1e-11)[0]
    # The same equations solved densely: the orbital Hessian is fvind applied
    # to every unit rotation plus the orbital-energy differences.
    nvir = orbv.shape[1]
    e_ai = mf.mo_energy[~occ][:, None] - mf.mo_energy[occ][None, :]
    unit = np.eye(nvir * nocc).reshape(-1, nvir, nocc)
    hess = fvind(unit).reshape(nvir * nocc, nvir * nocc).T + np.diag(e_ai.ravel())
    dense = np.linalg.solve(hess, -h1.reshape(3, -1).T).T.reshape(3, nvir, nocc)
    return {
        "basis": "cc-pvdz",
        "rhf_energy": mf.e_tot,
        "alpha": alpha_of(mo1).tolist(),
        "alpha_dense": alpha_of(dense).tolist(),
    }


def fno_mp2(name: str) -> dict:
    mf = converged(scf.RHF(molecule(name, "cc-pvdz")))
    aux = ferric_basis("cc-pvdz-ri")
    pt = dfmp2.DFMP2(mf)
    pt.with_df = df.DF(mf.mol, auxbasis=aux)
    e_full = pt.kernel()[0]
    nocc = pt.nocc
    occupations = np.linalg.eigvalsh(pt.make_rdm1(with_frozen=False)[nocc:, nocc:])
    nvir = pt.nmo - nocc
    keeps = (nvir // 4, nvir // 2, nvir)
    e_keep = {}
    for keep in keeps:
        frozen, no_coeff = pt.make_fno(nvir_act=keep)
        sub = dfmp2.DFMP2(mf, frozen=frozen, mo_coeff=no_coeff)
        sub.with_df = df.DF(mf.mol, auxbasis=aux)
        e_keep[str(keep)] = sub.kernel()[0]
    return {
        "basis": "cc-pvdz",
        "auxbasis": "cc-pvdz-ri",
        "rhf_energy": mf.e_tot,
        "e_corr": e_full,
        "occupations": sorted(occupations.tolist(), reverse=True),
        "e_corr_keep": e_keep,
    }


def main() -> None:
    import pyscf

    ref = {
        "generator": "scripts/gen_pyscf_block_ports_set2_refs.py",
        "pyscf_version": pyscf.__version__,
        "molecules": {},
    }
    for name in MOLECULES:
        ref["molecules"][name] = {
            "delta_scf": delta_scf(name),
            "polarizability": polarizability(name),
            "fno_mp2": fno_mp2(name),
        }
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=2)
        f.write("\n")
    print(json.dumps(ref, indent=2))


if __name__ == "__main__":
    main()
