"""Core-hole Delta-SCF written on ferric's blocks: initial maximum overlap.

Delta-SCF (Bagus, Phys. Rev. 139, A619 (1965)) takes an excitation or
ionization energy as the difference of two SCF energies, the second one
converged on a non-aufbau occupation. Here the second state is the O 1s
core-ionized cation, for the O 1s binding energy.

The non-aufbau state is held by the initial maximum overlap method (Barca,
Gilbert, Gill, JCTC 14, 1501 (2018)): every iteration occupies, per spin, the
orbitals that overlap most with the STARTING occupied set, PySCF's
``scf.addons.mom_occ``. The start is ``dm0``: the neutral RHF orbitals with
one beta 1s electron removed, and ``run_uhf(dm0=, mom_reference="initial")``
runs the SCF. Aufbau would refill the 1s hole on the first iteration and
converge to the valence-hole cation, 19.5 Ha lower.

``uhf_check`` rebuilds the UHF energy and the plain orbital gradient from
``AoIntegrals.veff_polarized``. ``mom_uhf`` is the independent check: the
maximum overlap method of Gilbert, Besley, Gill (JPCA 112, 13164 (2008)),
whose reference is the PREVIOUS iteration's occupied set, as a numpy UHF on
veff_polarized with its own DIIS, from the same start. J and K are exact.
Bohr and Hartree.

Run as a script: the O 1s binding energy of water/cc-pVDZ, asserting it equals
``mom_uhf`` to 3e-11 Ha and the PySCF reference (``scf.addons.mom_occ``) to
1e-9 Ha.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_set2_pyscf.json"

EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def uhf_check(ints: ferric.AoIntegrals, d_a: np.ndarray, d_b: np.ndarray):
    """(plain UHF energy, max |F_s D_s S - S D_s F_s|) of a density pair."""
    h, s = ints.hcore(), ints.overlap()
    e2, v_a, v_b = ints.veff_polarized(d_a, d_b)
    energy = float(np.sum((d_a + d_b) * h)) + e2 + ints.nuclear_repulsion()
    grad = 0.0
    for d, v in ((d_a, v_a), (d_b, v_b)):
        f = h + v
        grad = max(grad, float(np.abs(f @ d @ s - s @ d @ f).max()))
    return energy, grad


def mom_uhf(ints, c0, occ, e_conv=1e-11, d_conv=1e-9, max_iter=200, diis_size=8):
    """UHF energy by MOM + DIIS in numpy from the occupied columns ``occ`` of ``c0``."""
    h, s = ints.hcore(), ints.overlap()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    c_occ = [c0[:, o] for o in occ]
    d = [co @ co.T for co in c_occ]
    focks, errors, e_old = [], [], 0.0
    for _ in range(max_iter):
        e2, *v = ints.veff_polarized(*d)
        f = [h + vs for vs in v]
        energy = float(np.sum((d[0] + d[1]) * h)) + e2 + ints.nuclear_repulsion()
        err = np.concatenate(
            [(x.T @ (fs @ ds @ s - s @ ds @ fs) @ x).ravel() for fs, ds in zip(f, d)]
        )
        focks, errors = (focks + [f])[-diis_size:], (errors + [err])[-diis_size:]
        n = len(focks)
        b = -np.ones((n + 1, n + 1))
        b[n, n] = 0.0
        b[:n, :n] = [[ei @ ej for ej in errors] for ei in errors]
        coef = np.linalg.solve(b, np.r_[np.zeros(n), -1.0])[:n]
        d_new = []
        for spin in range(2):
            fs = sum(ci * fk[spin] for ci, fk in zip(coef, focks))
            _, cs = np.linalg.eigh(x.T @ fs @ x)
            cs = x @ cs
            # MOM: keep the orbitals that overlap most with the last occupied set.
            p = np.sum(np.abs(c_occ[spin].T @ s @ cs), axis=0)
            c_occ[spin] = cs[:, np.sort(np.argsort(-p)[: c_occ[spin].shape[1]])]
            d_new.append(c_occ[spin] @ c_occ[spin].T)
        change = max(np.abs(dn - ds).max() for dn, ds in zip(d_new, d))
        d = d_new
        if abs(energy - e_old) < e_conv and change < d_conv:
            return uhf_check(ints, *d)[0]
        e_old = energy
    raise RuntimeError(f"MOM-UHF did not converge in {max_iter} iterations")


def core_ionization(xyz: Path, basis: str, hole: int = 0, mom: bool = False) -> dict:
    """Delta-SCF binding energy of the closed-shell orbital ``hole`` (Hartree).

    mom=False runs ferric's initial MOM (run_uhf); mom=True runs ``mom_uhf``
    from the same start instead (the independent check). Returns the neutral
    and cation energies and the binding energy ``ip`` and, for run_uhf, the
    cation's rebuilt UHF energy ``e_check`` and orbital gradient ``grad``.
    """
    bs = ferric.BasisSet.bundled(basis)
    neutral = ferric.run_rhf(
        ferric.Molecule.from_xyz(str(xyz)), bs, **EXACT_JK, **TIGHT
    )
    cation = ferric.Molecule.from_xyz(str(xyz), charge=1, multiplicity=2)
    ints = ferric.AoIntegrals(cation, bs)
    nocc = (ints.nelec + 1) // 2
    c = neutral.mo_coefficients()
    occ = (list(range(nocc)), [i for i in range(nocc) if i != hole])
    out = {"e_neutral": neutral.energy}
    if mom:
        out["e_cation"] = mom_uhf(ints, c, occ)
    else:
        dm0 = tuple(c[:, o] @ c[:, o].T for o in occ)
        ion = ferric.run_uhf(
            cation, bs, dm0=dm0, mom_reference="initial", **EXACT_JK, **TIGHT
        )
        assert ion.converged
        out["e_cation"] = ion.energy
        out["e_check"], out["grad"] = uhf_check(
            ints, ion.density_alpha(), ion.density_beta()
        )
    out["ip"] = out["e_cation"] - neutral.energy
    return out


def main() -> None:
    ferric_mom = core_ionization(WATER, "cc-pvdz")
    numpy_mom = core_ionization(WATER, "cc-pvdz", mom=True)
    e, ip = ferric_mom["e_cation"], ferric_mom["ip"]
    print(f"neutral RHF          {ferric_mom['e_neutral']:.10f}")
    print(f"O 1s cation, ferric  {e:.10f}   rebuilt {ferric_mom['e_check']:.10f}")
    print(f"orbital gradient     {ferric_mom['grad']:.1e}")
    print(f"O 1s IP, ferric      {ip:.10f} Ha = {ip * 27.211386:.4f} eV")
    print(
        f"O 1s IP, numpy MOM   {numpy_mom['ip']:.10f}   diff {ip - numpy_mom['ip']:.1e}"
    )
    assert abs(ip - numpy_mom["ip"]) < 3e-11
    if PYSCF_REF.exists():
        ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["delta_scf"]
        diff = ip - ref["ip"]
        print(f"O 1s IP, PySCF       {ref['ip']:.10f}   diff {diff:.1e}")
        assert abs(diff) < 1e-9


if __name__ == "__main__":
    main()
