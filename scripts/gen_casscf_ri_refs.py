#!/usr/bin/env python
"""PySCF density-fitted CASSCF / SC-NEVPT2 references for the RI route.

Feeds ``crates/ferric-ci/tests/casscf_ri.rs``. Per system (6-31G,
def2-universal-jkfit), PySCF runs on a molecule built from ferric's own basis
data (``BasisSet.shells``, orbital and auxiliary basis alike). PySCF
stable-sorts an atom's shells by angular momentum while ferric keeps the file
order (s, sp -> s p, ...), so the script maps ferric's AO order onto PySCF's
and checks the map on the exact core Hamiltonian and ERIs (``delta_hcore``,
``delta_eri``: their max differences, which also set the test's integral
bar). Orbitals passed to PySCF are permuted with the same map; stored
orbitals are in ferric's order.

Recorded:

* ``pyscf_casscf``: ``scf.RHF(mol).density_fit(aux)`` then ``mcscf.CASSCF``
  (DF throughout) from PySCF's own orbitals: energy, gradient norm, settings.
* ``ferric_ri``: ferric's ``run_casscf(casscf_integrals="ri")`` (canonical):
  energy, gradient norm, orbitals.
* ``pyscf_fixed_orbitals``: PySCF DF CAS-CI on ferric's RI orbitals and
  ``mrpt.NEVPT`` per class on it (DF integrals, ``mrpt.dfnevpt2._ERIS``).
* the exact-integral PySCF CASSCF energy (``e_exact_casscf``) and the RI-JK
  RHF error of both codes (``rhf_ri_error_*``), for the RI-error report.

Usage (one heavy op at a time on the shared box):
  box-lease exec -- env OPENBLAS_NUM_THREADS=1 PYTHONPATH=<shim> \\
    /home/matt/qc/ferric/.venv/bin/python scripts/gen_casscf_ri_refs.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pyscf
from pyscf import fci, gto, mcscf, mrpt, scf
from pyscf.mrpt import dfnevpt2
from pyscf.mrpt import nevpt2 as nv2

import ferric

ROOT = Path(__file__).resolve().parents[1]
MOLS = ROOT / "testdata" / "molecules"
OUT = ROOT / "testdata" / "reference" / "casscf_ri_pyscf.json"

BASIS = "6-31g"
AUX = "def2-universal-jkfit"
MC_CONV_TOL = 1e-12
MC_CONV_TOL_GRAD = 1e-6
FCI_CONV_TOL = 1e-14
# PySCF repeats of the fixed-orbital CAS-CI + SC-NEVPT2: N2's Srs differs
# between repeats on identical orbitals by up to ~1e-7 Ha (labels whose norm
# is rounding-level), so the spread is recorded and enters the test bar.
NEVPT_REPEATS = 7

# name, xyz file, n_core, n_active, (n_alpha, n_beta)
SYSTEMS = (
    ("h2o_cas44_631g", "water.xyz", 3, 4, (2, 2)),
    ("n2_cas66_631g", "n2.xyz", 4, 6, (3, 3)),
    ("o2_triplet_cas86_631g", "o2.xyz", 4, 6, (5, 3)),
)


def ferric_basis(bs, zs):
    """PySCF basis dict from a ferric ``BasisSet``, in ferric's shell order."""
    out = {}
    for z in sorted(set(zs)):
        out[gto.mole._symbol(z)] = [
            [ang, *[[e, c] for e, c in zip(exps, coefs)]]
            for ang, _pure, exps, coefs in bs.shells(z)
        ]
    return out


def ferric_ao_order(pmol, basis):
    """perm[i] = PySCF AO index of ferric AO i (PySCF stable-sorts shells by l)."""
    perm, off = [], 0
    for ia in range(pmol.natm):
        ls = [sh[0] for sh in basis[pmol.atom_pure_symbol(ia)]]
        start, pos = {}, off
        for k in sorted(range(len(ls)), key=lambda k: ls[k]):
            start[k] = pos
            pos += 2 * ls[k] + 1
        for k in range(len(ls)):
            perm.extend(range(start[k], start[k] + 2 * ls[k] + 1))
        off = pos
    assert sorted(perm) == list(range(pmol.nao_nr()))
    return np.array(perm)


def pyscf_mol(fmol, bs, aux_bs):
    """PySCF molecule on ferric's basis data, and the ferric -> PySCF AO map."""
    zs = [gto.charge(s) for s in fmol.symbols()]
    atoms = list(zip(fmol.symbols(), fmol.coords_bohr()))
    basis = ferric_basis(bs, zs)
    pmol = gto.M(atom=atoms, unit="Bohr", basis=basis, verbose=0)
    return pmol, ferric_ao_order(pmol, basis), ferric_basis(aux_bs, zs)


def rhf(pmol, aux=None):
    mf = scf.RHF(pmol)
    if aux:
        mf = mf.density_fit(auxbasis=aux)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    assert mf.converged
    return mf


def nevpt_classes(mc):
    """``mrpt.NEVPT`` per class on DF integrals (the steps of NEVPT.kernel).

    The orbitals are used as given (ferric's canonical CASSCF orbitals, the
    active ones natural), with orbital energies the diagonal of PySCF's
    generalized Fock matrix: no re-diagonalization, which would rotate N2's
    degenerate pairs arbitrarily, and SC-NEVPT2 is not invariant to that.
    """
    nev = mrpt.NEVPT(mc)
    nev.verbose = 0
    ci = mc.ci
    fock = mc.get_fock(mo_coeff=mc.mo_coeff, ci=ci)
    nev.mo_coeff = mc.mo_coeff
    nev.mo_energy = np.einsum("pi,pq,qi->i", mc.mo_coeff, fock, mc.mo_coeff)
    nev.ci = ci
    ncas, nelecas, ncore = nev.ncas, nev.nelecas, nev.ncore
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelecas)
    dms = {"1": dm1, "2": dm2, "3": dm3, "4": None}
    eris = dfnevpt2._ERIS(nev, nev.mo_coeff, mc._scf.with_df)
    li = (
        fci.cistring.gen_linkstr_index(range(ncas), nelecas[0]),
        fci.cistring.gen_linkstr_index(range(ncas), nelecas[1]),
    )
    aaaa = eris["ppaa"][ncore : ncore + ncas, ncore : ncore + ncas].copy()
    dms["f3ca"] = nv2._contract4pdm("NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelecas, li)
    dms["f3ac"] = nv2._contract4pdm("NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelecas, li)
    return {
        "Sr": float(nv2.Sr(nev, ci, dms, eris)[1]),
        "Si": float(nv2.Si(nev, ci, dms, eris)[1]),
        "Sijrs": float(nv2.Sijrs(nev, eris)[1]),
        "Sijr": float(nv2.Sijr(nev, dms, eris)[1]),
        "Srsi": float(nv2.Srsi(nev, dms, eris)[1]),
        "Srs": float(nv2.Srs(nev, dms, eris)[1]),
        "Sij": float(nv2.Sij(nev, dms, eris)[1]),
        "Sir": float(nv2.Sir(nev, dms, eris)[1]),
    }


def casscf(mf, n_active, nelecas):
    mc = mcscf.CASSCF(mf, n_active, nelecas)
    mc.verbose = 0
    mc.conv_tol = MC_CONV_TOL
    mc.conv_tol_grad = MC_CONV_TOL_GRAD
    mc.max_cycle_macro = 300
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    mc.kernel()
    if not mc.converged:
        raise RuntimeError("pyscf CASSCF did not converge")
    return mc


def one(name, src, n_core, n_active, nelecas) -> dict:
    text = (MOLS / src).read_text()
    fmol = ferric.Molecule.from_xyz_string(text)
    bs = ferric.BasisSet.bundled(BASIS)
    ints = ferric.AoIntegrals(fmol, bs)
    pmol, perm, aux = pyscf_mol(fmol, bs, ferric.BasisSet.bundled(AUX))
    ix = np.ix_(perm, perm)
    d_h = float(
        np.max(
            np.abs(
                np.asarray(ints.hcore())
                - (pmol.intor("int1e_kin") + pmol.intor("int1e_nuc"))[ix]
            )
        )
    )
    d_eri = float(
        np.max(
            np.abs(
                np.asarray(ints.eri()).reshape([pmol.nao] * 4)
                - pmol.intor("int2e")[np.ix_(perm, perm, perm, perm)]
            )
        )
    )
    assert d_h < 1e-10 and d_eri < 1e-10, (d_h, d_eri)

    mf_df = rhf(pmol, aux)
    mc = casscf(mf_df, n_active, nelecas)
    g = float(np.linalg.norm(mc.get_grad()))
    mf = rhf(pmol)
    e_exact = float(casscf(mf, n_active, nelecas).e_tot)

    r = ferric.run_casscf(
        fmol,
        bs,
        n_active,
        nelecas,
        n_core=n_core,
        canonicalize=True,
        casscf_integrals="ri",
        casscf_auxbasis=AUX,
    )
    assert r.converged and r.integrals == "ri" and r.auxbasis == AUX
    c = np.asarray(r.mo_coeff)
    c_p = np.empty_like(c)
    c_p[perm] = c

    runs = []
    for _ in range(NEVPT_REPEATS if nelecas[0] == nelecas[1] else 1):
        ci = mcscf.CASCI(mf_df, n_active, nelecas)
        ci.verbose = 0
        ci.fcisolver.conv_tol = FCI_CONV_TOL
        ci.kernel(c_p)
        runs.append(
            (float(ci.e_tot), nevpt_classes(ci) if nelecas[0] == nelecas[1] else {})
        )
    fixed = {"e_casci": runs[0][0]}
    if nelecas[0] == nelecas[1]:
        names = runs[0][1]
        fixed["nevpt2_classes"] = {
            k: float(np.median([r[1][k] for r in runs])) for k in names
        }
        fixed["nevpt2_class_spread"] = {
            k: float(np.ptp([r[1][k] for r in runs])) for k in names
        }
        fixed["nevpt2_repeats"] = len(runs)

    e_rhf_f = ferric.run_rhf(fmol, bs).energy
    e_rhf_f_ri = ferric.run_rhf(
        fmol, bs, df_j_aux=AUX, df_k_aux=AUX, ri_j_precision="f64", ri_k_precision="f64"
    ).energy
    row = {
        "name": name,
        "xyz": text,
        "basis": BASIS,
        "auxbasis": AUX,
        "n_core": n_core,
        "n_active": n_active,
        "n_elec_active": list(nelecas),
        "delta_hcore": d_h,
        "delta_eri": d_eri,
        "pyscf_casscf": {
            "e_total": float(mc.e_tot),
            "grad_norm": g,
            "conv_tol": MC_CONV_TOL,
            "conv_tol_grad": MC_CONV_TOL_GRAD,
        },
        "ferric_ri": {
            "e_total": float(r.e_total),
            "grad_norm": float(r.grad_norm),
            "mo_coeff_shape": list(c.shape),
            "mo_coeff": c.ravel().tolist(),
        },
        "pyscf_fixed_orbitals": fixed,
        "e_exact_casscf": e_exact,
        "rhf_ri_error_pyscf": float(mf_df.e_tot - mf.e_tot),
        "rhf_ri_error_ferric": float(e_rhf_f_ri - e_rhf_f),
    }
    print(
        f"{name}: ferric RI {r.e_total:.12f} PySCF DF {mc.e_tot:.12f} "
        f"diff {r.e_total - mc.e_tot:.2e}; RI-exact CASSCF {mc.e_tot - e_exact:.3e}; "
        f"RI-JK RHF error PySCF {row['rhf_ri_error_pyscf']:.3e} ferric "
        f"{row['rhf_ri_error_ferric']:.3e}; fixed-orbital CAS-CI PySCF {ci.e_tot:.12f}",
        flush=True,
    )
    return row


def main() -> int:
    out = {
        "generator": "scripts/gen_casscf_ri_refs.py",
        "date": datetime.date.today().isoformat(),
        "pyscf_version": pyscf.__version__,
        "ferric_build": str(ferric.__build__),
        "systems": [one(*s) for s in SYSTEMS],
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
