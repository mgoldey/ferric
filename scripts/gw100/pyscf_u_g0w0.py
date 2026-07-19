#!/usr/bin/env python3
"""PySCF open-shell G0W0@UHF on a small doublet radical — the independent
reference half of the U-GW validation gate (ferric's `run_u_gw`).

Apples-to-apples: PySCF is fed the EXACT same bundled orbital + RI-aux JSON that
ferric uses (default cc-pVDZ + cc-pVDZ-RI), so the orbital basis and the density
fitting aux are bit-identical between the two codes. ferric uses PDEP-as-W;
PySCF uses spin-unrestricted analytic continuation (`ugw_ac.UGWAC`). Agreement of
the α-HOMO and β-HOMO quasiparticle energies proves ferric's open-shell
self-energy (u_sigma.rs) and QP layer reproduce a standard implementation, not
merely a Koopmans-window sanity band.

Prints one parseable line:
    PYSCF <ip_a_ev> <ip_b_ev> <koop_a_ev> <koop_b_ev> <nelec> <e_uhf>
where ip_a/ip_b are −ε_QP of the α/β HOMO.

Usage: pyscf_u_g0w0.py <file.xyz> <charge> <spin_2s> [orb.json] [aux.json|auxname]

Note: reads geometry in Angstrom from an .xyz-style file (line 1 = natoms,
line 2 = comment, remaining = element x y z).
"""
import json
import sys

from pyscf import gto, scf
from pyscf.data.elements import charge as sym2charge
from pyscf.gw.ugw_ac import UGWAC

HARTREE2EV = 27.211386245988

DEFAULT_ORB = (
    sys.path[0] + "/../../crates/ferric-core/src/basis/bundled/cc-pvdz.json"
)
DEFAULT_AUX = (
    sys.path[0] + "/../../crates/ferric-core/src/basis/bundled/cc-pvdz-ri.json"
)


def bse_to_pyscf_basis(elem):
    """Convert a BSE 'complete'-schema element block to PySCF internal basis."""
    out = []
    for sh in elem.get("electron_shells", []):
        ams = sh["angular_momentum"]
        exps = [float(x) for x in sh["exponents"]]
        cols = [[float(x) for x in col] for col in sh["coefficients"]]
        if len(ams) == 1:
            l = ams[0]
            block = [l]
            for i, e in enumerate(exps):
                block.append([e] + [col[i] for col in cols])
            out.append(block)
        else:  # SP-style: one column per angular momentum
            for k, l in enumerate(ams):
                block = [l]
                for i, e in enumerate(exps):
                    block.append([e, cols[k][i]])
                out.append(block)
    return out


def main():
    xyz_path = sys.argv[1]
    charge = int(sys.argv[2])
    spin = int(sys.argv[3])  # 2S = n_alpha - n_beta
    orb_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_ORB
    aux_arg = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_AUX

    orb_bundle = json.load(open(orb_path))["elements"]

    lines = open(xyz_path).read().splitlines()
    nat = int(lines[0].split()[0])
    atom = "\n".join(lines[2 : 2 + nat])
    syms = sorted({ln.split()[0] for ln in lines[2 : 2 + nat]})

    basis = {s: bse_to_pyscf_basis(orb_bundle[str(sym2charge(s))]) for s in syms}

    mol = gto.M(
        atom=atom, basis=basis, charge=charge, spin=spin,
        unit="Angstrom", verbose=0,
    )

    if aux_arg.endswith(".json"):
        aux_bundle = json.load(open(aux_arg))["elements"]
        aux = {s: bse_to_pyscf_basis(aux_bundle[str(sym2charge(s))]) for s in syms}
    else:
        aux = aux_arg

    mf = scf.UHF(mol).density_fit(auxbasis=aux)
    mf.kernel()
    assert mf.converged, "UHF did not converge"

    nocc_a, nocc_b = mol.nelec  # (n_alpha, n_beta)
    homo_a, homo_b = nocc_a - 1, nocc_b - 1
    koop_a = -mf.mo_energy[0][homo_a] * HARTREE2EV
    koop_b = -mf.mo_energy[1][homo_b] * HARTREE2EV

    gw = UGWAC(mf)
    gw.orbs = range(mol.nao_nr())
    gw.kernel()
    ip_a = -gw.mo_energy[0][homo_a] * HARTREE2EV
    ip_b = -gw.mo_energy[1][homo_b] * HARTREE2EV

    print(f"PYSCF {ip_a:.4f} {ip_b:.4f} {koop_a:.4f} {koop_b:.4f} "
          f"{mol.nelectron} {mf.e_tot:.6f}")


if __name__ == "__main__":
    main()
