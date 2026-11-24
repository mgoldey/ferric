#!/usr/bin/env python
"""PySCF CASSCF references on ferric's AO integrals.

Feeds ``crates/ferric-ci/tests/casscf_vs_pyscf.rs``. For each system,
``pyscf.mcscf.CASSCF`` runs on hcore, overlap, ERIs and nuclear repulsion
taken from ``ferric.AoIntegrals``, starting from PySCF's own RHF orbitals on
those integrals (``init_guess="1e"``; the minao guess assumes PySCF's AO
order). Only integral values are shared: ferric's side starts from ferric's
RHF in the Rust test, and PySCF orbitals never reach ferric.

Recorded per system: the settings below, PySCF's converged energy and
packed gradient norm, the natural occupations (eigenvalues of the
spin-summed active 1-RDM), the eigenvalues of the core and virtual blocks of
``F^I + F^A`` (``mc.get_fock()`` in the converged MOs), ``<S^2>``, the RHF
energy on the same integrals, and a fingerprint of the integrals so a drift in
ferric's integrals is visible.

PySCF must report convergence; the script exits nonzero otherwise.

Usage (one heavy op at a time on the shared box):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 PYTHONPATH=<shim> \\
    /home/matt/qc/ferric/.venv/bin/python scripts/gen_casscf_refs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci import spin_op

import ferric

ROOT = Path(__file__).resolve().parents[1]
MOLS = ROOT / "testdata" / "molecules"
OUT = ROOT / "testdata" / "reference"

BASIS = "6-31g"
RHF_CONV_TOL = 1e-12
RHF_CONV_TOL_GRAD = 1e-8
MC_CONV_TOL = 1e-12
MC_CONV_TOL_GRAD = 1e-6  # 1e-7 does not converge H2O in 300 macro iterations
MC_MAX_CYCLE_MACRO = 300
FCI_CONV_TOL = 1e-14

N2_STRETCHED = "2\nN2 stretched to 1.6 Angstrom\nN 0.0 0.0 0.0\nN 0.0 0.0 1.6\n"

# name, xyz source (file name or inline text), n_core, n_active, (n_alpha, n_beta)
SYSTEMS = (
    ("h2o_cas44_631g", "water.xyz", 3, 4, (2, 2)),
    ("n2_cas66_631g", "n2.xyz", 4, 6, (3, 3)),
    ("n2_r1.6_cas66_631g", N2_STRETCHED, 4, 6, (3, 3)),
    ("o2_triplet_cas86_631g", "o2.xyz", 4, 6, (5, 3)),
)


def xyz_text(src: str) -> str:
    return src if "\n" in src else (MOLS / src).read_text()


def pyscf_rhf_on_ferric_integrals(fmol, ints):
    atoms = list(zip(fmol.symbols(), fmol.coords_bohr()))
    pmol = gto.M(atom=atoms, unit="Bohr", basis=BASIS, verbose=0)
    nao = ints.nao
    assert pmol.nao_nr() == nao, (pmol.nao_nr(), nao)
    h, s, enuc = ints.hcore(), ints.overlap(), ints.nuclear_repulsion()
    pmol.incore_anyway = True
    mf = scf.RHF(pmol)
    mf.get_hcore = lambda *a: h
    mf.get_ovlp = lambda *a: s
    mf.energy_nuc = lambda *a: enuc
    mf._eri = ao2mo.restore(8, ints.eri(), nao)
    mf.init_guess = "1e"
    mf.conv_tol = RHF_CONV_TOL
    mf.conv_tol_grad = RHF_CONV_TOL_GRAD
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF on ferric integrals did not converge")
    return mf, h, s


def run(name, src, n_core, n_active, nelecas):
    text = xyz_text(src)
    fmol = ferric.Molecule.from_xyz_string(text)
    ints = ferric.AoIntegrals(fmol, ferric.BasisSet.bundled(BASIS))
    mf, h, s = pyscf_rhf_on_ferric_integrals(fmol, ints)
    assert 2 * n_core + sum(nelecas) == fmol.nelec()

    mc = mcscf.CASSCF(mf, n_active, nelecas)
    mc.verbose = 0
    mc.conv_tol = MC_CONV_TOL
    mc.conv_tol_grad = MC_CONV_TOL_GRAD
    mc.max_cycle_macro = MC_MAX_CYCLE_MACRO
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    mc.kernel()
    if not mc.converged:
        raise RuntimeError(f"{name}: pyscf.mcscf.CASSCF did not converge")
    g = float(np.linalg.norm(mc.get_grad()))

    mo = mc.mo_coeff
    dm1 = mc.fcisolver.make_rdm1(mc.ci, n_active, nelecas)
    occ = np.sort(np.linalg.eigvalsh(dm1))[::-1]
    f_mo = mo.T @ mc.get_fock() @ mo
    nc, na = n_core, n_active
    eps_core = np.sort(np.linalg.eigvalsh(f_mo[:nc, :nc]))
    eps_virt = np.sort(np.linalg.eigvalsh(f_mo[nc + na :, nc + na :]))
    s2 = float(spin_op.spin_square0(mc.ci, n_active, nelecas)[0])
    eri = ints.eri()
    row = {
        "system": name,
        "xyz": text,
        "xyz_source": src if "\n" not in src else "inline",
        "basis": BASIS,
        "n_core": n_core,
        "n_active": n_active,
        "n_elec_active": list(nelecas),
        "e_total": float(mc.e_tot),
        "grad_norm_pyscf": g,
        "converged": bool(mc.converged),
        "natural_occupations": occ.tolist(),
        "core_fock_eigenvalues": eps_core.tolist(),
        "virtual_fock_eigenvalues": eps_virt.tolist(),
        "s2": s2,
        "e_rhf_pyscf_same_integrals": float(mf.e_tot),
        "integral_fingerprint": {
            "hcore_sum": float(h.sum()),
            "overlap_sum": float(s.sum()),
            "eri_norm": float(np.linalg.norm(eri)),
            "e_nuc": float(ints.nuclear_repulsion()),
        },
        "settings": {
            "start": "PySCF RHF on ferric AO integrals, init_guess=1e",
            "rhf_conv_tol": RHF_CONV_TOL,
            "rhf_conv_tol_grad": RHF_CONV_TOL_GRAD,
            "mc_conv_tol": MC_CONV_TOL,
            "mc_conv_tol_grad": MC_CONV_TOL_GRAD,
            "mc_max_cycle_macro": MC_MAX_CYCLE_MACRO,
            "fcisolver": "direct_spin1",
            "fcisolver_conv_tol": FCI_CONV_TOL,
            "fock_for_eigenvalues": "mc.get_fock() (F^I + F^A) in mc.mo_coeff",
        },
        "provenance": {
            "pyscf_version": pyscf.__version__,
            "ferric_commit": ferric.build_info()["commit"],
            "ferric_dirty": ferric.build_info()["dirty"],
            "generator": "scripts/gen_casscf_refs.py",
        },
    }
    print(
        f"{name}: E {mc.e_tot:.13f} |g| {g:.1e} occ {np.round(occ, 6).tolist()} "
        f"S2 {s2:.2e} macro converged {mc.converged}",
        flush=True,
    )
    return row


def main() -> int:
    print(f"ferric {ferric.build_info()['commit']} pyscf {pyscf.__version__}")
    for name, src, nc, na, ne in SYSTEMS:
        row = run(name, src, nc, na, ne)
        path = OUT / f"casscf_{name}_pyscf.json"
        path.write_text(json.dumps(row, indent=1) + "\n")
        print(f"  wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
