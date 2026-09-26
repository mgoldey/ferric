"""PySCF references for the VALIDATION.md row "CAS-CI".

Consumer: crates/ferric-ci/tests/validation_casci.rs.
Output:   testdata/reference/validation/casci/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-ci/src/{lib,integrals}.rs, not
assumed)
---------------------------------------------------------------------------
`ferric_ci::run_cas_ci(mol, prep, rhf, CasCiConfig { n_active,
n_elec_active: (na, nb), active_start, .. })` on a converged restricted RHF
(ferric's `solve_rhf` default: exact four-centre J/K). The active orbitals are
the canonical RHF MOs `active_start .. active_start + n_active` (energy order);
MOs `0 .. active_start` are doubly occupied inactive orbitals. The active-space
integrals come from a dense EXACT AO ERI tensor (no density fitting, no
screening beyond libint2's 1e-14 precision), and

    e_core  = E_nuc + sum_i [2 h_ii + sum_j (2 (ii|jj) - (ij|ji))]    (inactive)
    h_eff   = h + sum_i [2 (pq|ii) - (pi|iq)]                          (active)
    e_total = e_core + lowest eigenvalue of the active CI Hamiltonian

over the FULL Ms = (na - nb)/2 determinant space (every S with that Ms; no
spin penalty, no root following). `CasCiResult::e_active` is equal to
`e_total` (core included, see its doc comment), so the active-space energy is
formed on the Rust side as `e_total - e_core`.

REFERENCE
---------
* RHF: PySCF `scf.RHF` (no point group, default guess — as ferric), exact
  integrals, ferric's basis JSON and Bohr geometry, conv_tol 1e-12. The state
  is PINNED: it must equal the D2h symmetry-adapted aufbau RHF to 1e-9
  (otherwise refused). Internal stability is recorded, not followed: at
  r = 2.20 A the symmetric N2 RHF is a saddle toward a symmetry-broken RHF
  0.19 Ha lower (energy recorded as `symmetry_broken_rhf_energy`). The CAS is
  built on the symmetric RHF — the conventional CAS reference, and the state
  ferric's `solve_rhf` reaches from its default guess (ferric has no RHF
  instability following, so the broken state would need a seeded density).
* CAS-CI: PySCF `mcscf.CASCI(mf, ncas, nelecas)` with the default orbital
  window, ncore = (N - nelecas)/2 — the SAME rule as ferric's `active_start`
  — on the canonical RHF orbitals; `fcisolver = fci.direct_spin1` (lowest
  Ms = 0 root of any S, as ferric), conv_tol 1e-12. Recorded: e_tot, e_cas
  (= e_tot - ecore), ecore from `get_h1eff()`, <S^2>, N_det.
* Anchor (independent of the Davidson / direct-CI path): the full dense
  active-space Hamiltonian from `fci.direct_spin1.pspace` over ALL
  determinants, diagonalized with numpy; its lowest eigenvalue + ecore must
  equal e_tot to 1e-10 (`dense_anchor_diff`).
* Window guard: the orbital-energy gaps at both edges of every window used
  are recorded, and a window that cuts through a (near-)degenerate set
  (gap < 1e-4 Ha) is REFUSED — there the orbital choice inside the window is
  arbitrary and the two codes need not agree.

Blocks per system:
  `casci`          the plan's window.
  `casci_shifted`  the same n_active moved one orbital DOWN (active_start - 1,
                   two more active electrons) — the negative-control window.

SYSTEMS
  n2_r1.10 / cc-pVDZ   CAS(6,6)  ncore 4  (near equilibrium)
  n2_r2.20 / cc-pVDZ   CAS(6,6)  ncore 4  (stretched; strongly multireference)
  h2o      / 6-31G     CAS(4,4)  ncore 3

Run (light; seconds):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_casci.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "casci"
ROW_NAME = "CAS-CI"
SYSTEMS = {  # system -> (xyz stem, basis, ncas, nelecas)
    "n2_r1.10": ("n2_r1.10", "cc-pvdz", 6, 6),
    "n2_r2.20": ("n2_r2.20", "cc-pvdz", 6, 6),
    "h2o": ("h2o", "6-31g", 4, 4),
}
SCF_CONV = {"conv_tol": 1e-12, "conv_tol_grad": 1e-9}
FCI_CONV_TOL = 1e-12
TOL_DENSE_ANCHOR = 1e-10
MIN_EDGE_GAP = 1e-4


def rhf(mol, symbols, coords, basis):
    """Plain (no point-group) RHF from PySCF's default guess, as ferric runs it.

    The state is pinned, not followed: it must equal the D2h symmetry-adapted
    aufbau RHF (the symmetric reference a CAS is built on). Its internal
    stability is recorded; for stretched N2 it is a saddle toward a
    symmetry-broken RHF, which is recorded (energy) but NOT used — see the
    module docstring.
    """
    import numpy as np
    from pyscf import gto, scf

    def converge(m, dm0=None):
        m.conv_tol = SCF_CONV["conv_tol"]
        m.conv_tol_grad = SCF_CONV["conv_tol_grad"]
        m.max_cycle = 300
        m.verbose = 0
        m.kernel(dm0=dm0)
        if not m.converged:
            raise RuntimeError(f"{type(m).__name__} did not converge")
        return m

    mf = converge(scf.RHF(mol))
    pyscf_basis, _ = common.pyscf_basis(basis, symbols)
    molsym = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=pyscf_basis,
        symmetry=True,
        verbose=0,
    )
    msym = converge(scf.RHF(molsym))
    d_sym = float(mf.e_tot - msym.e_tot)
    if abs(d_sym) > 1e-9:
        raise RuntimeError(
            f"default-guess RHF {mf.e_tot:.10f} is not the symmetry-adapted aufbau "
            f"RHF {msym.e_tot:.10f} ({d_sym:.2e})"
        )
    mo_i = mf.stability(internal=True, external=False)[0]
    stable = bool(np.allclose(mo_i, mf.mo_coeff))
    rec = {
        "energy": float(mf.e_tot),
        "symmetry_adapted_energy": float(msym.e_tot),
        "point_group": molsym.topgroup,
        "irrep_nelec": {k: int(v) for k, v in msym.get_irrep_nelec().items()},
        "internal_stable": stable,
    }
    if not stable:
        # Follow the instability once, for the record only.
        # Its own soft mode keeps PySCF's convergence flag off at 1e-9 gradient;
        # the energy is settled to ~1e-12, which is all that is recorded.
        broken = scf.RHF(mol).newton()
        broken.conv_tol, broken.max_cycle, broken.verbose = 1e-12, 100, 0
        broken.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
        rec["symmetry_broken_rhf_energy"] = float(broken.e_tot)
        rec["note"] = (
            "internally unstable toward a symmetry-broken RHF (energy recorded); "
            "the CAS-CI uses the symmetric RHF, which ferric and PySCF both reach "
            "from their default guesses"
        )
    return mf, rec


def window_gaps(mo_energy, start, ncas):
    lo = float(mo_energy[start] - mo_energy[start - 1]) if start > 0 else None
    end = start + ncas
    hi = float(mo_energy[end] - mo_energy[end - 1]) if end < len(mo_energy) else None
    for name, g in (("lower", lo), ("upper", hi)):
        if g is not None and g < MIN_EDGE_GAP:
            raise RuntimeError(
                f"window [{start}, {end}) cuts a near-degenerate set at its {name} "
                f"edge (gap {g:.2e} Ha): the orbital choice is arbitrary there"
            )
    return {"lower_edge_gap": lo, "upper_edge_gap": hi}


def casci_block(mf, ncas, nelecas) -> dict:
    import numpy as np
    from pyscf import ao2mo, fci, mcscf

    mol = mf.mol
    ncore = (mol.nelectron - nelecas) // 2
    assert 2 * ncore + nelecas == mol.nelectron
    gaps = window_gaps(mf.mo_energy, ncore, ncas)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = fci.direct_spin1.FCI(mol)
    mc.fcisolver.conv_tol = FCI_CONV_TOL
    mc.fcisolver.max_cycle = 500
    mc.verbose = 0
    e_tot, e_cas, ci, _, _ = mc.kernel()
    assert mc.ncore == ncore, f"PySCF ncore {mc.ncore} != {ncore}"
    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    na = nb = nelecas // 2
    s2, mult = mc.fcisolver.spin_square(ci, ncas, (na, nb))

    # Independent anchor: dense H over every determinant, numpy eigh.
    from math import comb

    ndet = comb(ncas, na) * comb(ncas, nb)
    h2 = ao2mo.restore(1, h2eff, ncas)
    _, hdense = fci.direct_spin1.pspace(h1eff, h2, ncas, (na, nb), np=ndet)
    assert hdense.shape == (ndet, ndet), hdense.shape
    e_dense = float(np.linalg.eigvalsh(hdense)[0]) + float(ecore)
    d_anchor = e_dense - float(e_tot)
    assert abs(d_anchor) < TOL_DENSE_ANCHOR, f"dense anchor {d_anchor:.2e}"
    return {
        "ncore": int(ncore),
        "active_start": int(ncore),
        "n_active": int(ncas),
        "n_elec_active": [int(na), int(nb)],
        "n_determinants": int(ndet),
        "e_total": float(e_tot),
        "e_active": float(e_cas),
        "e_core": float(ecore),
        "e_corr_vs_rhf": float(e_tot - mf.e_tot),
        "s_squared": float(s2),
        "window_mo_energies": [
            float(x) for x in mf.mo_energy[max(ncore - 1, 0) : ncore + ncas + 1]
        ],
        **gaps,
        "dense_anchor_diff": d_anchor,
        "max_abs_ci_coeff": float(np.max(np.abs(ci))),
    }


def gen(system) -> Path:
    import numpy
    import pyscf

    stem, basis, ncas, nelecas = SYSTEMS[system]
    xyz = common.MOL_DIR / f"{stem}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis)
    ll = common.check_basis_like_for_like(mol, basis, symbols)
    mf, rhf_rec = rhf(mol, symbols, coords, basis)
    main = casci_block(mf, ncas, nelecas)
    shifted = casci_block(mf, ncas, nelecas + 2)
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rhf": rhf_rec,
        "casci": main,
        "casci_shifted": shifted,
    }
    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "scf": "scf.RHF exact integrals + stability(internal=True) loop",
            "casci": "mcscf.CASCI(mf, ncas, nelecas), default window "
            "ncore=(N-nelecas)/2, fcisolver=fci.direct_spin1 (lowest Ms=0 root)",
            "anchor": "dense fci.direct_spin1.pspace over all determinants + numpy eigvalsh",
            "fci_conv_tol": FCI_CONV_TOL,
            "numpy": numpy.__version__,
        },
        basis_name=basis,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=None,
        frozen_core={
            "casci": f"{main['ncore']} inactive doubly occupied (the CAS core)",
            "casci_shifted": f"{shifted['ncore']} inactive doubly occupied",
        },
        scf_conv=SCF_CONV,
        stability={"rhf_internal_stable": rhf_rec["internal_stable"]},
        generator="scripts/validation/gen_casci.py",
        extra={"basis_self_check": ll},
    )
    path = common.write_reference(ROW, system, basis, payload)
    for key in ("casci", "casci_shifted"):
        b = payload[key]
        print(
            f"{system:9s} {key:13s} CAS({sum(b['n_elec_active'])},{b['n_active']}) "
            f"start {b['active_start']} ndet {b['n_determinants']:4d} "
            f"E_RHF {rhf_rec['energy']:.10f} E_CAS {b['e_total']:.10f} "
            f"e_core {b['e_core']:.10f} corr {b['e_corr_vs_rhf']:+.6e} "
            f"<S2> {b['s_squared']:.2e} gaps {b['lower_edge_gap']:.3e}/{b['upper_edge_gap']:.3e} "
            f"anchor {b['dense_anchor_diff']:.1e}",
            flush=True,
        )
    print(f"{system:9s} RHF {rhf_rec}")
    d = payload["casci"]["e_total"] - payload["casci_shifted"]["e_total"]
    print(f"{system:9s} window shift moves E_CAS by {d:+.3e} Ha")
    return path


def main(argv) -> int:
    want = argv or list(SYSTEMS)
    unknown = set(want) - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    n = 0
    for s in SYSTEMS:
        if s in want:
            gen(s)
            n += 1
    print(f"GEN_CASCI_DONE written={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
