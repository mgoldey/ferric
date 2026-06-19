#!/usr/bin/env python3
"""PySCF G0W0@HF on an ECP molecule — the reference half of the GW-through-ECP
validation gate (spec 2026-06-17-gw100-ecp-molecules.md).

Apples-to-apples: PySCF is fed the EXACT same bundled aug-cc-pVDZ-PP JSON that
ferric uses (basis + inline ECP), so the orbital basis and the ECP are
bit-identical between the two codes. Density-fitted with def2-tzvp correlation
aux (mirrors ferric's def2-tzvp-rifit pairing). ferric uses PDEP-as-W; PySCF
uses analytic continuation (gw_ac). Agreement proves the V_ECP /
reduced-electron-count path flows correctly into the GW intermediates.

Prints one parseable line:  PYSCF <ip_g0w0_ev> <ip_koopmans_ev> <nelec> <e_rhf>

Usage: pyscf_g0w0_ecp.py <file.xyz> [bundled_json] [auxbasis]
"""
import json
import sys

import numpy as np
from pyscf import df, gto, scf
from pyscf.data.elements import _symbol, charge as sym2charge
from pyscf.gw.gw_ac import GWAC

HARTREE2EV = 27.211386245988

# Repo-relative default to the bundled basis ferric compiles in.
DEFAULT_JSON = (
    sys.path[0]
    + "/../../crates/ferric-core/src/basis/bundled/aug-cc-pvdz-pp.json"
)


def bse_to_pyscf_basis(elem):
    """Convert a BSE 'complete'-schema element block to PySCF internal basis
    (list of [l, [exp, c1, c2, ...], ...])."""
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


def bse_to_pyscf_ecp(elem):
    """Convert a BSE ecp_potentials block to PySCF internal ECP format:
        [n_core, [[l, [terms_r0, terms_r1, ...]], ...]]
    where the index into the inner list is the literal power of r (0..6) and
    terms_rN is [[gexp, coef], ...]. The local channel (max l) uses l = -1."""
    n_core = elem["ecp_electrons"]
    pots = elem["ecp_potentials"]
    max_l = max(p["angular_momentum"][0] for p in pots)
    chans = []
    for p in pots:
        l = p["angular_momentum"][0]
        l_key = -1 if l == max_l else l
        r_exps = p["r_exponents"]
        gexps = [float(x) for x in p["gaussian_exponents"]]
        coefs = [float(x) for x in p["coefficients"][0]]
        rmax = max(r_exps)
        by_r = [[] for _ in range(rmax + 1)]
        for r, g, c in zip(r_exps, gexps, coefs):
            by_r[r].append([g, c])
        chans.append([l_key, by_r])
    chans.sort(key=lambda x: (x[0] != -1, x[0]))  # local (-1) first
    return [n_core, chans]


def main():
    xyz_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JSON
    auxbasis = sys.argv[3] if len(sys.argv) > 3 else "def2-tzvp-ri"

    bundle = json.load(open(json_path))
    by_z = bundle["elements"]

    lines = open(xyz_path).read().splitlines()
    nat = int(lines[0].split()[0])
    atom = "\n".join(lines[2 : 2 + nat])
    syms = sorted({ln.split()[0] for ln in lines[2 : 2 + nat]})

    basis = {}
    ecp = {}
    for s in syms:
        z = str(sym2charge(s))
        elem = by_z[z]
        basis[s] = bse_to_pyscf_basis(elem)
        if "ecp_electrons" in elem:
            ecp[s] = bse_to_pyscf_ecp(elem)

    mol = gto.M(
        atom=atom, basis=basis, ecp=ecp if ecp else None,
        unit="Angstrom", verbose=0,
    )

    # Correlation RI aux: load the bundled def2-tzvp-rifit JSON so PySCF and
    # ferric share the IDENTICAL aux (PySCF's internal def2-tzvp-ri lacks I/Xe/Ag).
    if auxbasis.endswith(".json"):
        auxbundle = json.load(open(auxbasis))["elements"]
        aux = {s: bse_to_pyscf_basis(auxbundle[str(sym2charge(s))]) for s in syms}
    else:
        aux = auxbasis

    mf = scf.RHF(mol).density_fit(auxbasis=aux)
    mf.kernel()
    nocc = mol.nelectron // 2
    homo = nocc - 1
    ip_koop = -mf.mo_energy[homo] * HARTREE2EV

    gw = GWAC(mf)
    gw.orbs = range(mol.nao_nr())
    gw.kernel()
    ip_g0w0 = -gw.mo_energy[homo] * HARTREE2EV

    print(f"PYSCF {ip_g0w0:.4f} {ip_koop:.4f} {mol.nelectron} {mf.e_tot:.6f}")


if __name__ == "__main__":
    main()
