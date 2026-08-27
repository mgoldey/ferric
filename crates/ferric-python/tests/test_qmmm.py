"""Python surface of the QM/MM layer: Molecule geometry getters, QmmmSystem
partitioning (link atoms + boundary charge schemes), and run_qmmm.

The numbers here are pinned to the SAME PySCF references the Rust tests use
(testdata/reference/*_qmmm_*.json), so the binding is checked against an
independent code, not against itself.
"""
import json
import math
import os

import numpy as np
import pytest

import ferric

from conftest import BOHR_PER_ANGSTROM, WATER_ANGSTROM, WATER_SYMBOLS, water_xyz_string

REFDIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata", "reference")


def _load(name):
    with open(os.path.join(REFDIR, name)) as f:
        return json.load(f)


# ── Molecule getters ──


def test_molecule_coords_and_symbols_round_trip_angstrom():
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    assert mol.symbols() == WATER_SYMBOLS
    coords = mol.coords()
    assert len(coords) == 3
    for got, want in zip(coords, WATER_ANGSTROM):
        assert len(got) == 3
        # Å -> Bohr -> Å round trip is ~1 ulp, not bit-exact (see the
        # ConformerEnsemble.coordinates() docs).
        assert np.allclose(got, want, atol=1e-13, rtol=0)


def test_molecule_coords_bohr_are_the_internal_values():
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    bohr = mol.coords_bohr()
    ang = mol.coords()
    for b, a in zip(bohr, ang):
        for k in range(3):
            assert b[k] == pytest.approx(a[k] * BOHR_PER_ANGSTROM, abs=1e-13)
    # And the NRE from those Bohr coordinates is ferric's own NRE — proves
    # they really are the internal Bohr values, not a second conversion.
    from conftest import nuclear_repulsion_from_bohr

    assert nuclear_repulsion_from_bohr(bohr, [8, 1, 1]) == pytest.approx(mol.nuclear_repulsion(), abs=1e-10)


def test_optimized_geometry_is_retrievable():
    # The gap that blocked tools/active_site/pose_relaxation.py: after
    # run_optimize the relaxed geometry must be readable, and must differ
    # from the (deliberately stretched) start.
    mol = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.8\n")
    r = ferric.run_optimize(mol, "sto-3g", max_steps=50)
    assert r.converged
    c = r.mol().coords()
    assert r.mol().symbols() == ["H", "H"]
    d = math.dist(c[0], c[1])
    assert 0.70 < d < 0.75, f"H2/STO-3G should relax to ~0.71 Å, got {d}"


# ── QmmmSystem ──


def _water_ref_system(ref):
    """Full structure = the reference QM atoms, then its MM charges as atoms."""
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X"] * len(ref["mm_charges"])
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [99.0] * len(ref["atoms"]) + [c["q"] for c in ref["mm_charges"]]
    return ferric.QmmmSystem(
        symbols, coords_ang, charges,
        qm_indices=list(range(len(ref["atoms"]))),
        charge=ref["charge"], multiplicity=ref["multiplicity"],
    )


def test_qmmm_system_partitions_and_exposes_point_charges_in_bohr():
    ref = _load("water_sto-3g_qmmm_two_charges.json")
    sys = _water_ref_system(ref)
    assert sys.qm_indices() == [0, 1, 2]
    assert sys.mm_indices() == [3, 4]
    assert sys.qm_atom_count() == 3
    assert sys.natoms() == 5
    mol = sys.qm_molecule()
    assert mol.natoms() == 3
    assert mol.symbols() == ["O", "H", "H"]
    pcs = sys.point_charges()
    assert len(pcs) == 2
    for (q, x, y, z), c in zip(pcs, ref["mm_charges"]):
        assert q == c["q"]
        # Bohr out, to ~1 ulp of the Å round trip.
        assert np.allclose([x, y, z], c["xyz_bohr"], atol=1e-12, rtol=0)
    # The QM atoms' own MM charges (99.0) must not leak into the potential.
    assert all(abs(q) < 2 for q, *_ in pcs)


def test_qmmm_system_by_radius_selects_neighbours():
    ref = _load("water_sto-3g_qmmm_two_charges.json")
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X", "X"]
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [0, 0, 0, 1.0, -1.0]
    # Seed on O with a 1.2 Å radius: catches both H (0.96 Å), not the charges.
    sys = ferric.QmmmSystem(symbols, coords_ang, charges, qm_seeds=[0], qm_radius_angstrom=1.2)
    assert sys.qm_indices() == [0, 1, 2]
    assert sys.mm_indices() == [3, 4]


def test_qmmm_system_residue_ids_selects_whole_residues():
    ref = _load("water_sto-3g_qmmm_two_charges.json")
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X", "X"]
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [0, 0, 0, 1.0, -1.0]
    # residue_ids = one atom per residue must reproduce the by-atom result
    # (the exactness anchor): same seeds/radius, same split.
    sys = ferric.QmmmSystem(
        symbols, coords_ang, charges, qm_seeds=[0], qm_radius_angstrom=1.2,
        residue_ids=[0, 1, 2, 3, 4],
    )
    assert sys.qm_indices() == [0, 1, 2]
    assert sys.mm_indices() == [3, 4]


def test_qmmm_system_residue_ids_and_qm_indices_conflict():
    ref = _load("water_sto-3g_qmmm_two_charges.json")
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X", "X"]
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [0, 0, 0, 1.0, -1.0]
    with pytest.raises(ValueError):
        ferric.QmmmSystem(
            symbols, coords_ang, charges, qm_indices=[0, 1, 2],
            residue_ids=[0, 1, 2, 3, 4],
        )


def test_qmmm_system_point_charges_feed_run_rhf_to_the_pyscf_reference():
    ref = _load("water_sto-3g_qmmm_plus_lonepair.json")
    sys = _water_ref_system(ref)
    bs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_rhf(sys.qm_molecule(), bs, point_charges=sys.point_charges(), density_conv=1e-10)
    assert r.converged
    # 5e-8: the known ~2.4e-8 ferric-below-PySCF mean-field floor on water/STO-3G.
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)


def test_run_qmmm_matches_pyscf_energy_forces_and_gradient():
    ref = _load("water_sto-3g_qmmm_offaxis.json")
    sys = _water_ref_system(ref)
    r = ferric.run_qmmm(sys, "sto-3g", density_conv=1e-10)
    assert r.converged
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)

    qm_grad = r.qm_gradient()
    assert qm_grad.shape == (3, 3)
    assert np.allclose(qm_grad, np.array(ref["qm_gradient"]), atol=1e-6, rtol=0)

    mm_f = r.mm_forces()
    assert mm_f.shape == (3, 3)
    # ferric reports the FORCE; PySCF's grad_*_mm is dE/dR.
    assert np.allclose(mm_f, -np.array(ref["mm_gradient"]), atol=1e-6, rtol=0)

    full = r.full_gradient()
    assert full.shape == (6, 3)  # 3 QM atoms + 3 MM sites
    # No link atoms here: QM rows pass through, MM rows are -F.
    assert np.allclose(full[:3], qm_grad, atol=0, rtol=0)
    assert np.allclose(full[3:], -mm_f, atol=0, rtol=0)


def test_run_qmmm_uhf_doublet_matches_pyscf():
    ref = _load("oh_sto-3g_uqmmm_plus_lonepair.json")
    sys = _water_ref_system(ref)
    r = ferric.run_qmmm(sys, "sto-3g", method="uhf", density_conv=1e-10)
    assert r.converged
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)
    assert np.allclose(r.mm_forces(), -np.array(ref["mm_gradient"]), atol=1e-6, rtol=0)


# ── KS-DFT (RKS/UKS) embedding ──


def _water_dft_ref_system(ref):
    """Full structure = the reference QM atoms, then its MM charges as atoms.
    Same shape as `_water_ref_system` but for the *_qmmm_dft_*.json refs,
    which key the energy fields as e_total/e_gas_phase instead of
    energy/energy_gas_phase.
    """
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X"] * len(ref["mm_charges"])
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [99.0] * len(ref["atoms"]) + [c["q"] for c in ref["mm_charges"]]
    return ferric.QmmmSystem(
        symbols, coords_ang, charges,
        qm_indices=list(range(len(ref["atoms"]))),
        charge=ref["charge"], multiplicity=ref["multiplicity"],
    )


def test_run_qmmm_rks_pbe_matches_pyscf():
    ref = _load("water_sto-3g_qmmm_dft_pbe.json")
    sys = _water_dft_ref_system(ref)
    r = ferric.run_qmmm(sys, "sto-3g", method="rks", xc="PBE", density_conv=1e-8)
    assert r.converged
    # 2e-5: the same PBE/cc-pVDZ absolute-energy bar dft_pbe.rs uses (see
    # crates/ferric-scf/tests/qmmm_dft_vs_pyscf.rs for the measured floor).
    assert r.energy == pytest.approx(ref["e_total"], abs=2e-5)


def test_run_qmmm_uks_pbe_matches_pyscf():
    ref = _load("oh_sto-3g_uqmmm_dft_pbe.json")
    sys = _water_dft_ref_system(ref)
    r = ferric.run_qmmm(sys, "sto-3g", method="uks", xc="PBE", density_conv=1e-8)
    assert r.converged
    assert r.energy == pytest.approx(ref["e_total"], abs=2e-5)


def test_run_qmmm_rejects_xc_without_ks_method():
    ref = _load("water_sto-3g_qmmm_dft_pbe.json")
    sys = _water_dft_ref_system(ref)
    with pytest.raises(ValueError):
        ferric.run_qmmm(sys, "sto-3g", method="rhf", xc="PBE")


# ── Link atoms + boundary schemes ──

_CC = 1.53
_CH = 1.09


def _ethane():
    """Staggered ethane in Å; indices 0=C0, 1=C1, H's alternate (2,4,6 on C0; 3,5,7 on C1)."""
    th = math.radians(109.5)
    s, c = math.sin(th), math.cos(th)
    symbols = ["C", "C"]
    coords = [(0.0, 0.0, 0.0), (0.0, 0.0, _CC)]
    charges = [-0.1, -0.1]
    for k in range(3):
        phi = 2 * math.pi * k / 3
        symbols += ["H", "H"]
        coords += [(_CH * s * math.cos(phi), _CH * s * math.sin(phi), _CH * c),
                   (_CH * s * math.cos(phi), _CH * s * math.sin(phi), _CC - _CH * c)]
        charges += [0.033, 0.033]
    bonds = [(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
    return symbols, coords, charges, bonds


def test_link_atom_caps_the_cut_bond():
    symbols, coords, charges, bonds = _ethane()
    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    assert sys.qm_atom_count() == 4
    mol = sys.qm_molecule()
    assert mol.natoms() == 5
    assert mol.symbols()[-1] == "H"
    links = sys.link_atom_positions()
    assert len(links) == 1
    # Default scale 1.09/1.53 puts the cap at 1.09 Å from C0 along +z.
    assert np.allclose(links[0], [0.0, 0.0, 1.09], atol=1e-12)
    # Same number the Rust side reports, in Å here.
    assert sys.min_link_to_charge_distance() == pytest.approx(_CC - 1.09, abs=1e-12)


def test_boundary_scheme_strings_are_strict():
    symbols, coords, charges, bonds = _ethane()
    base = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    for name, n_charges in [("keep", 4), ("delete-host", 3), ("rc", 6), ("rcd", 6)]:
        s = base.with_boundary_charges(bonds, name)
        assert len(s.point_charges()) == n_charges, name
        assert s.boundary_scheme() == name
    with pytest.raises(ValueError, match="boundary"):
        base.with_boundary_charges(bonds, "Keep")  # case-sensitive: no silent default
    with pytest.raises(ValueError):
        base.with_boundary_charges(bonds, "z1")


def test_rcd_conserves_charge_and_moves_the_nearest_charge_away():
    symbols, coords, charges, bonds = _ethane()
    base = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    rcd = base.with_boundary_charges(bonds, "rcd")
    q_base = sum(q for q, *_ in base.point_charges())
    q_rcd = sum(q for q, *_ in rcd.point_charges())
    assert q_rcd == pytest.approx(q_base, abs=1e-14)
    assert rcd.min_link_to_charge_distance() > base.min_link_to_charge_distance()


def test_run_qmmm_full_gradient_covers_the_whole_structure_across_a_cut():
    symbols, coords, charges, bonds = _ethane()
    sys = (ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6])
           .with_link_atoms(bonds)
           .with_boundary_charges(bonds, "rcd"))
    r = ferric.run_qmmm(sys, "sto-3g")
    assert r.converged
    assert r.qm_gradient().shape == (5, 3)   # 4 QM + 1 link
    assert r.mm_forces().shape == (6, 3)     # 3 M2 + 3 midpoints
    full = r.full_gradient()
    assert full.shape == (8, 3)
    assert np.all(np.isfinite(full))
    # The link row was projected: the frontier C0 and host C1 rows both carry it.
    assert abs(full[1, 2]) > 0
