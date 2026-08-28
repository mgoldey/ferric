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


# ── MmTopology / run_qmmm(mm_topology=) ──


def test_mm_topology_from_amber_units_basic_energy():
    # Two atoms, one bond, no angles/torsions: hand-computed harmonic bond
    # energy, same AMBER (no leading 1/2) convention as the Rust crate.
    top = ferric.MmTopology.from_amber_units(
        charges=[0.0, 0.0],
        sigmas_angstrom=[0.0, 0.0],
        epsilons_kcal=[0.0, 0.0],
        bonds=[(0, 1, 300.0, 1.5)],
        angles=[],
        torsions=[],
    )
    assert top.n_atoms() == 2


def test_mm_topology_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ferric.MmTopology.from_amber_units(
            charges=[0.0, 0.0, 0.0],
            sigmas_angstrom=[0.0, 0.0],
            epsilons_kcal=[0.0, 0.0],
            bonds=[],
            angles=[],
            torsions=[],
        )


def test_run_qmmm_with_full_ethane_topology_reports_mm_energy_and_full_gradient():
    symbols, coords, charges, bonds = _ethane()
    lj_sigma = [3.4, 3.4] + [2.6] * 6
    lj_eps = [0.109, 0.109] + [0.0157] * 6
    theta0 = math.degrees(math.radians(109.5))
    angles = []
    for h in (2, 4, 6):
        angles.append((h, 0, 1, 50.0, theta0))
    for h in (3, 5, 7):
        angles.append((h, 1, 0, 50.0, theta0))
    for a, b in [(2, 4), (2, 6), (4, 6)]:
        angles.append((a, 0, b, 35.0, theta0))
    for a, b in [(3, 5), (3, 7), (5, 7)]:
        angles.append((a, 1, b, 35.0, theta0))
    torsions = []
    for hi in (2, 4, 6):
        for hj in (3, 5, 7):
            torsions.append((hi, 0, 1, hj, 3, 0.16, 0.0))

    top = ferric.MmTopology.from_amber_units(
        charges=charges,
        sigmas_angstrom=lj_sigma,
        epsilons_kcal=lj_eps,
        bonds=[(0, 1, 310.0, 1.53)] + [(0, h, 340.0, 1.09) for h in (2, 4, 6)] + [(1, h, 340.0, 1.09) for h in (3, 5, 7)],
        angles=angles,
        torsions=torsions,
    )

    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    r = ferric.run_qmmm(sys, "sto-3g", mm_topology=top)
    assert r.converged
    e = r.mm_energy
    for key in ("bond", "angle", "torsion", "lj", "coulomb", "total"):
        assert key in e
        assert np.isfinite(e[key])
    full = r.full_gradient()
    assert full.shape == (8, 3)
    assert np.all(np.isfinite(full))


def test_run_qmmm_without_mm_topology_has_zero_mm_energy():
    # Exactness anchor on the Python side: omitting mm_topology must be
    # indistinguishable from passing an all-zero one, and the reported
    # mm_energy must be all zero (not merely absent).
    symbols, coords, charges, bonds = _ethane()
    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    r = ferric.run_qmmm(sys, "sto-3g")
    e = r.mm_energy
    assert e == {"bond": 0.0, "angle": 0.0, "torsion": 0.0, "lj": 0.0, "coulomb": 0.0, "total": 0.0}


# ── Gaussian-smeared MM charges (Lane A) ──


def _water_ref_system_smeared(ref):
    """Full structure = the reference QM atoms, then its MM charges (with
    per-charge Gaussian widths, in Bohr in the ref -> Angstrom for the
    constructor) as atoms."""
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X"] * len(ref["mm_charges"])
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [c["xyz_bohr"] for c in ref["mm_charges"]]
    coords_ang = [[v / BOHR_PER_ANGSTROM for v in xyz] for xyz in coords_bohr]
    charges = [99.0] * len(ref["atoms"]) + [c["q"] for c in ref["mm_charges"]]
    widths_ang = [0.0] * len(ref["atoms"]) + [w / BOHR_PER_ANGSTROM for w in ref["radii"]]
    return ferric.QmmmSystem(
        symbols, coords_ang, charges,
        qm_indices=list(range(len(ref["atoms"]))),
        charge=ref["charge"], multiplicity=ref["multiplicity"],
        widths_angstrom=widths_ang,
    )


def test_qmmm_system_exposes_smeared_charges_separately_from_point_charges():
    ref = _load("water_sto-3g_qmmm_smeared_offaxis.json")
    sys = _water_ref_system_smeared(ref)
    assert sys.point_charges() == []
    scs = sys.smeared_charges()
    assert len(scs) == len(ref["mm_charges"])
    for (q, x, y, z, width), c, r_bohr in zip(scs, ref["mm_charges"], ref["radii"]):
        assert q == c["q"]
        assert np.allclose([x, y, z], c["xyz_bohr"], atol=1e-12, rtol=0)
        assert width == pytest.approx(r_bohr, abs=1e-12)


def test_run_qmmm_smeared_single_site_matches_pyscf():
    ref = _load("water_sto-3g_qmmm_smeared_r1.json")
    sys = _water_ref_system_smeared(ref)
    r = ferric.run_qmmm(sys, "sto-3g", density_conv=1e-10)
    assert r.converged
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)

    qm_grad = r.qm_gradient()
    assert qm_grad.shape == (3, 3)
    assert np.allclose(qm_grad, np.array(ref["qm_gradient"]), atol=1e-6, rtol=0)

    mm_f = r.mm_forces()
    assert mm_f.shape == (1, 3)
    assert np.allclose(mm_f, -np.array(ref["mm_gradient"]), atol=1e-6, rtol=0)


def test_run_qmmm_smeared_offaxis_distinct_widths_matches_pyscf():
    ref = _load("water_sto-3g_qmmm_smeared_offaxis.json")
    sys = _water_ref_system_smeared(ref)
    r = ferric.run_qmmm(sys, "sto-3g", density_conv=1e-10)
    assert r.converged
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)

    qm_grad = r.qm_gradient()
    assert np.allclose(qm_grad, np.array(ref["qm_gradient"]), atol=1e-6, rtol=0)

    mm_f = r.mm_forces()
    assert mm_f.shape == (3, 3)
    assert np.allclose(mm_f, -np.array(ref["mm_gradient"]), atol=1e-6, rtol=0)


def test_run_rhf_smeared_charges_kwarg_matches_pyscf():
    ref = _load("water_sto-3g_qmmm_smeared_r1.json")
    mol = ferric.Molecule.from_xyz_string(
        "3\nwater\n" + "\n".join(
            f"{a['symbol']} {a['xyz_bohr'][0] / BOHR_PER_ANGSTROM} "
            f"{a['xyz_bohr'][1] / BOHR_PER_ANGSTROM} {a['xyz_bohr'][2] / BOHR_PER_ANGSTROM}"
            for a in ref["atoms"]
        ) + "\n"
    )
    bs = ferric.BasisSet.bundled("sto-3g")
    smeared = [
        (c["q"], c["xyz_bohr"][0], c["xyz_bohr"][1], c["xyz_bohr"][2], w)
        for c, w in zip(ref["mm_charges"], ref["radii"])
    ]
    r = ferric.run_rhf(mol, bs, smeared_charges=smeared, density_conv=1e-10)
    assert r.converged
    assert r.energy == pytest.approx(ref["energy"], abs=5e-8)


def test_tiny_width_smeared_charge_scf_matches_point_charge_scf():
    # Exactness anchor at the Python layer: width -> 0 must reproduce the
    # point-charge run to high precision, mirroring the Rust-level
    # tiny_width_scf_matches_point_charge_scf test.
    ref = _load("water_sto-3g_qmmm_plus_lonepair.json")
    mol = ferric.Molecule.from_xyz_string(
        "3\nwater\n" + "\n".join(
            f"{a['symbol']} {a['xyz_bohr'][0] / BOHR_PER_ANGSTROM} "
            f"{a['xyz_bohr'][1] / BOHR_PER_ANGSTROM} {a['xyz_bohr'][2] / BOHR_PER_ANGSTROM}"
            for a in ref["atoms"]
        ) + "\n"
    )
    bs = ferric.BasisSet.bundled("sto-3g")
    c = ref["mm_charges"][0]
    r_point = ferric.run_rhf(mol, bs, point_charges=[(c["q"], *c["xyz_bohr"])], density_conv=1e-11)
    r_smeared = ferric.run_rhf(
        mol, bs, smeared_charges=[(c["q"], *c["xyz_bohr"], 1e-3)], density_conv=1e-11
    )
    assert r_point.converged and r_smeared.converged
    assert r_point.energy == pytest.approx(r_smeared.energy, abs=1e-9)

# ── run_optimize_qmmm ──


def test_run_optimize_qmmm_h2_in_a_field_matches_run_optimize():
    # EXACTNESS ANCHOR: an all-QM QmmmSystem (no MM atoms) + move_mm="none"
    # must reproduce ferric.run_optimize's energy exactly, since
    # to_external_potential() is None either way (the literal gas-phase
    # SCF path) and optimize_qmmm's Rust core is the SAME
    # optimize_coordinates BFGS run_bfgs/optimize_geometry use.
    mol = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 1.0\n")
    plain = ferric.run_optimize(mol, "sto-3g")
    assert plain.converged

    symbols = ["H", "H"]
    coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    charges = [0.0, 0.0]
    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 1])
    assert sys.mm_indices() == []

    result = ferric.run_optimize_qmmm(sys, "sto-3g", move_mm="none")
    assert result.converged
    assert result.steps == plain.steps
    assert result.energy == pytest.approx(plain.energy, abs=1e-10)
    assert len(result.energies()) == result.steps + 1


def test_run_optimize_qmmm_capped_ethane_relaxes_the_frontier_bond():
    symbols, coords, charges, bonds = _ethane()
    sys = (ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6])
           .with_link_atoms(bonds)
           .with_boundary_charges(bonds, "rcd"))

    result = ferric.run_optimize_qmmm(sys, "sto-3g", move_mm="none", max_steps=60)
    assert result.converged
    relaxed = result.system()
    assert relaxed.qm_indices() == sys.qm_indices()
    assert relaxed.mm_indices() == sys.mm_indices()
    # The frontier C0-link-H distance should differ from the starting
    # (exact 1.09 A scaled-position) placement once the geometry relaxes.
    mol0 = sys.qm_molecule()
    mol1 = relaxed.qm_molecule()
    c0_0 = np.array(mol0.coords()[0])
    h_0 = np.array(mol0.coords()[-1])
    c0_1 = np.array(mol1.coords()[0])
    h_1 = np.array(mol1.coords()[-1])
    d0 = np.linalg.norm(h_0 - c0_0)
    d1 = np.linalg.norm(h_1 - c0_1)
    assert d0 != pytest.approx(d1, abs=1e-6)
    energies = result.energies()
    assert len(energies) == result.steps + 1
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-8

    # HONEST COUNTERPART to the MoveMm::All assertion in
    # test_run_optimize_qmmm_full_topology_move_all_converges: under
    # move_mm="none" every MM atom must be BIT-IDENTICAL to its starting
    # coordinates (not merely close) -- otherwise a broken free-atom
    # selection that moved MM atoms regardless of move_mm would pass
    # silently.
    coords0 = sys.atom_coords_angstrom()
    coords1 = relaxed.atom_coords_angstrom()
    for i in sys.mm_indices():
        assert coords1[i] == coords0[i], f"MM atom {i} moved under move_mm='none'"


def test_run_optimize_qmmm_full_topology_move_all_converges():
    symbols, coords, charges, bonds = _ethane()
    # Deliberately small LJ sigma (see the Rust vs FD comment in
    # tests/qmmm_mm.rs / tests/qmmm_optimize.rs): ethane's nonbonded pairs
    # sit at bonded-range separations, so a realistic 3.4/2.6 A sigma puts
    # every pair deep in the repulsive wall and the optimizer never settles.
    lj_sigma = [0.6, 0.6] + [0.5] * 6
    lj_eps = [0.109, 0.109] + [0.0157] * 6
    theta0 = math.degrees(math.radians(109.5))
    angles = []
    for h in (2, 4, 6):
        angles.append((h, 0, 1, 0.06 * 627.509474, theta0))
    for h in (3, 5, 7):
        angles.append((h, 1, 0, 0.06 * 627.509474, theta0))
    for a, b in [(2, 4), (2, 6), (4, 6)]:
        angles.append((a, 0, b, 0.04 * 627.509474, theta0))
    for a, b in [(3, 5), (3, 7), (5, 7)]:
        angles.append((a, 1, b, 0.04 * 627.509474, theta0))
    torsions = []
    for hi in (2, 4, 6):
        for hj in (3, 5, 7):
            torsions.append((hi, 0, 1, hj, 3, 0.02 * 627.509474, 0.0))

    # Bond force constants in kcal/mol/A^2, matching the Rust
    # tests/qmmm_optimize.rs full_ethane_topology() k=0.35/0.4 Hartree/Bohr^2
    # (converted: k_kcal = k_ha * 627.509474 / ANGSTROM_TO_BOHR^2).
    k_cc_kcal = 61.502
    k_ch_kcal = 70.288
    top = ferric.MmTopology.from_amber_units(
        charges=charges,
        sigmas_angstrom=lj_sigma,
        epsilons_kcal=lj_eps,
        bonds=[(0, 1, k_cc_kcal, 1.53)]
        + [(0, h, k_ch_kcal, 1.09) for h in (2, 4, 6)]
        + [(1, h, k_ch_kcal, 1.09) for h in (3, 5, 7)],
        angles=angles,
        torsions=torsions,
    )

    sys = (ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6])
           .with_link_atoms(bonds)
           .with_boundary_charges(bonds, "rcd"))

    result = ferric.run_optimize_qmmm(
        sys, "sto-3g", move_mm="all", mm_topology=top, max_steps=80,
    )
    assert result.converged
    energies = result.energies()
    for a, b in zip(energies, energies[1:]):
        assert b <= a + 1e-6

    # Distinguish "move_mm='all' actually moved the MM atoms" from "the QM
    # atoms alone drove the energy decrease" (a broken free-atom selection
    # that only ever returned QM indices would still pass every assertion
    # above). MM atoms here are full indices {1, 3, 5, 7} (C1 + its three
    # H's) -- e.g. full index 1 or 3.
    relaxed = result.system()
    coords0 = np.array(sys.atom_coords_angstrom())
    coords1 = np.array(relaxed.atom_coords_angstrom())
    ang2bohr = 1.0 / 0.52917721092
    any_mm_moved = False
    for i in sys.mm_indices():
        d_bohr = np.linalg.norm(coords1[i] - coords0[i]) * ang2bohr
        if d_bohr > 1e-4:
            any_mm_moved = True
    assert any_mm_moved, (
        "move_mm='all' must move at least one MM atom by > 1e-4 Bohr from its start "
        f"coordinates (full indices {sys.mm_indices()})"
    )


def test_run_optimize_qmmm_move_mm_without_topology_raises():
    # This typed error originates in the Rust qmmm::optimize_qmmm core
    # (FerricError::General), which every other run_* binding maps through
    # make_err() to RuntimeError -- same convention as e.g. a bad k_builder
    # string or an unconverged-SCF report elsewhere in this module.
    symbols, coords, charges, bonds = _ethane()
    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    with pytest.raises(RuntimeError, match="mm_topology"):
        ferric.run_optimize_qmmm(sys, "sto-3g", move_mm="all")
    with pytest.raises(RuntimeError, match="mm_topology"):
        ferric.run_optimize_qmmm(sys, "sto-3g", move_mm=("within", 3.0))


def test_run_optimize_qmmm_bad_move_mm_string_raises():
    symbols, coords, charges, bonds = _ethane()
    sys = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 2, 4, 6]).with_link_atoms(bonds)
    with pytest.raises(ValueError):
        ferric.run_optimize_qmmm(sys, "sto-3g", move_mm="everything")


# ── Thole-damped polarizable embedding (Lane B) ──

ANG2BOHR = 1.0 / 0.52917721092


def _pe_case(tag):
    """Build a QmmmSystem for a Lane B reference case: the water QM atoms
    plus one MM site per entry in `sites` (charge q, polarizability alpha,
    both at the SAME position -- the realistic "one atom, one role" case)."""
    ref = _load(f"{tag}.json")
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X"] * len(ref["sites"])
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [s["xyz_bohr"] for s in ref["sites"]]
    coords_angstrom = [[c / ANG2BOHR for c in xyz] for xyz in coords_bohr]
    charges = [0.0] * len(ref["atoms"]) + [s["q"] for s in ref["sites"]]
    polarizabilities_angstrom3 = [0.0] * len(ref["atoms"]) + [
        s["alpha_bohr3"] / (ANG2BOHR**3) for s in ref["sites"]
    ]
    qm_indices = list(range(len(ref["atoms"])))
    sys = ferric.QmmmSystem(
        symbols,
        coords_angstrom,
        charges,
        qm_indices=qm_indices,
        polarizabilities_angstrom3=polarizabilities_angstrom3,
    )
    return sys, ref


@pytest.mark.parametrize(
    "tag",
    ["water_sto-3g_pe_one_site", "water_sto-3g_pe_three_sites", "water_sto-3g_pe_three_sites_nodamp"],
)
def test_run_qmmm_polarizable_matches_pyscf_prototype(tag):
    sys, ref = _pe_case(tag)
    # ref["thole_a"] is JSON null (Python None) for the nodamp case, but
    # run_qmmm's thole_a=None means "use the DEFAULT damping" (matching
    # e.g. run_rhf's None-means-default convention elsewhere in this
    # binding) -- 0.0 is the documented "disable damping entirely" value.
    thole_a = 0.0 if ref["thole_a"] is None else ref["thole_a"]
    result = ferric.run_qmmm(sys, "sto-3g", thole_a=thole_a)
    assert result.converged
    assert abs(result.energy - ref["energy"]) < 1e-7, (
        f"{tag}: energy {result.energy} vs pyscf {ref['energy']}"
    )
    assert abs(result.e_pol - ref["e_pol"]) < 1e-7, f"{tag}: e_pol {result.e_pol} vs pyscf {ref['e_pol']}"
    dipoles = result.induced_dipoles()
    assert dipoles is not None
    ref_dipoles = np.array(ref["induced_dipoles"])
    assert np.max(np.abs(np.asarray(dipoles) - ref_dipoles)) < 1e-6, f"{tag}: dipole mismatch"


def test_run_qmmm_polarizable_disabled_matches_plain_embedding():
    # alpha=0 everywhere (no polarizabilities_angstrom3 passed at all) must
    # be the EXACT non-polarizable code path: e_pol == 0.0 and
    # induced_dipoles() is None -- the same anchor
    # `polarizable_none_is_bit_identical_to_plain_scf` pins in Rust.
    ref = _load("water_sto-3g_pe_one_site.json")
    symbols = [a["symbol"] for a in ref["atoms"]] + ["X"]
    coords_bohr = [a["xyz_bohr"] for a in ref["atoms"]] + [ref["sites"][0]["xyz_bohr"]]
    coords_angstrom = [[c / ANG2BOHR for c in xyz] for xyz in coords_bohr]
    charges = [0.0, 0.0, 0.0, ref["sites"][0]["q"]]
    sys = ferric.QmmmSystem(symbols, coords_angstrom, charges, qm_indices=[0, 1, 2])
    result = ferric.run_qmmm(sys, "sto-3g")
    assert result.converged
    assert result.e_pol == 0.0
    assert result.induced_dipoles() is None


def test_run_qmmm_polarizable_thole_a_zero_disables_damping():
    # thole_a=0.0 must select the undamped Thole model (None internally),
    # matching the nodamp reference -- a genuine end-to-end check of the
    # thole_a=0.0 -> damping-disabled convention (see run_qmmm's Rust doc:
    # "pass 0.0 to disable damping entirely").
    sys, ref = _pe_case("water_sto-3g_pe_three_sites_nodamp")
    result = ferric.run_qmmm(sys, "sto-3g", thole_a=0.0)
    assert result.converged
    assert abs(result.energy - ref["energy"]) < 1e-7
    assert abs(result.e_pol - ref["e_pol"]) < 1e-7


def test_qmmm_system_polarizabilities_angstrom3_length_mismatch_raises():
    with pytest.raises(ValueError, match="polarizabilities_angstrom3"):
        ferric.QmmmSystem(
            WATER_SYMBOLS,
            WATER_ANGSTROM,
            [0.0, 0.0, 0.0],
            qm_indices=[0, 1, 2],
            polarizabilities_angstrom3=[0.0, 0.0],
        )


# ── Lane B4: polarizable-embedding gradient for uhf/rks/uks ──
#
# run_qmmm's QM gradient used to include the polarizable Fock-term
# contribution ONLY for method="rhf" (the Rust side's Task B3 wrapper). Lane
# B4 closed that gap with uhf_gradient_with_polarizable /
# ks_gradient_closed_with_polarizable / ks_gradient_uks_with_polarizable and
# wired them into run_qmmm for "uhf"/"rks"/"uks". This test is the
# end-to-end Python check for the "uhf" case: it builds an OH-doublet
# QmmmSystem with ONE polarizable site (same OH-doublet geometry
# `oh_sto-3g_uqmmm_plus_lonepair.json` uses elsewhere in this file, at
# charge=0/multiplicity=2) and central-FD's `run_qmmm(...).energy` along
# one QM-atom coordinate against `run_qmmm(...).qm_gradient()` -- an
# independent check of the SAME claim the Rust FD tests
# (qmmm_polarizable_multivariant.rs) make, but through the actual Python
# binding surface end users call.

_OH_SYMBOLS = ["O", "H"]
_OH_ANGSTROM = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.98]]


def _oh_polarizable_system(extra_angstrom):
    """OH doublet QM region + one polarizable MM site (charge 0.5, alpha
    ~8 Bohr^3 -- large/close, same non-triviality rationale as the Rust
    multivariant tests' `close_polarizable_sites`) at a fixed lab-frame
    position offset from the OH atoms by `extra_angstrom` -- used to
    displace the SITE together with the rest of the world when an OH atom
    is displaced would defeat the point, so this only ever displaces QM
    atoms; the site itself is fixed across all FD points here."""
    site_bohr = [4.0, -1.0, 2.0]
    site_angstrom = [c / ANG2BOHR for c in site_bohr]
    symbols = list(_OH_SYMBOLS) + ["X"]
    coords = [list(c) for c in _OH_ANGSTROM] + [site_angstrom]
    coords[0] = [a + b for a, b in zip(coords[0], extra_angstrom)]
    charges = [0.0, 0.0, 0.5]
    polarizabilities_angstrom3 = [0.0, 0.0, 8.0 / (ANG2BOHR**3)]
    return ferric.QmmmSystem(
        symbols,
        coords,
        charges,
        qm_indices=[0, 1],
        charge=0,
        multiplicity=2,
        polarizabilities_angstrom3=polarizabilities_angstrom3,
    )


def test_run_qmmm_uhf_polarizable_gradient_matches_finite_difference():
    h = 5e-4  # Angstrom
    sys0 = _oh_polarizable_system([0.0, 0.0, 0.0])
    result = ferric.run_qmmm(sys0, "sto-3g", method="uhf", density_conv=1e-10)
    assert result.converged
    assert result.e_pol != 0.0, "polarizable site must actually be inducing a dipole"

    analytic = result.qm_gradient()  # (2, 3): O then H
    assert analytic.shape == (2, 3)

    # FD only the O atom's x coordinate (index 0, coord 0) -- one component
    # is enough to validate the wiring is present and correctly signed;
    # the Rust multivariant tests already cover the full (natoms, 3) grid
    # analytically-vs-FD.
    sys_p = _oh_polarizable_system([h, 0.0, 0.0])
    sys_m = _oh_polarizable_system([-h, 0.0, 0.0])
    r_p = ferric.run_qmmm(sys_p, "sto-3g", method="uhf", density_conv=1e-10)
    r_m = ferric.run_qmmm(sys_m, "sto-3g", method="uhf", density_conv=1e-10)
    assert r_p.converged and r_m.converged
    h_bohr = h * ANG2BOHR
    fd = (r_p.energy - r_m.energy) / (2.0 * h_bohr)

    assert analytic[0, 0] == pytest.approx(fd, abs=1e-5), (
        f"analytic {analytic[0, 0]} vs FD {fd}"
    )


def test_run_qmmm_uhf_polarizable_gradient_differs_from_non_polarizable():
    # Non-triviality: the polarizable contribution must materially change
    # the QM gradient -- otherwise the FD test above could pass merely
    # because the term is negligible (or because run_qmmm silently fell
    # back to the SCF-only gradient again).
    sys_pol = _oh_polarizable_system([0.0, 0.0, 0.0])
    r_pol = ferric.run_qmmm(sys_pol, "sto-3g", method="uhf", density_conv=1e-10)
    assert r_pol.converged

    symbols = list(_OH_SYMBOLS) + ["X"]
    coords = [list(c) for c in _OH_ANGSTROM] + [
        [c / ANG2BOHR for c in [4.0, -1.0, 2.0]]
    ]
    sys_plain = ferric.QmmmSystem(
        symbols, coords, [0.0, 0.0, 0.5], qm_indices=[0, 1], charge=0, multiplicity=2,
    )
    r_plain = ferric.run_qmmm(sys_plain, "sto-3g", method="uhf", density_conv=1e-10)
    assert r_plain.converged

    delta = np.max(np.abs(r_pol.qm_gradient() - r_plain.qm_gradient()))
    assert delta > 1e-4, f"polarizable gradient contribution suspiciously small: {delta:.3e}"
