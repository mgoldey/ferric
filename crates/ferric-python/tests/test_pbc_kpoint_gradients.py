"""k-point analytic forces and stress through the Python surface.

`with_gradient=True` / `with_stress=True` on run_rhf_kpts and run_uhf_kpts expose
the Rust `ferric_pbc::kgrad` / `ferric_pbc::kstress` entry points
(`result.gradient()`, `result.stress()`), differentiating the hcore output and
J/K object the SCF itself ran on. The Rust suites (pbc_kgrad.rs, pbc_kstress.rs,
pbc_kkeep.rs) hold the per-term anchors and mutants; this file checks the
BINDING, with the units of the Gamma bindings (gradient Hartree/Bohr per cell,
stress Hartree/Bohr^3, atoms and lattice in Angstrom in, Bohr derivatives out).

* EXACTNESS ANCHORS, independent constructions:
  - analytic gradient vs a central FD of the binding's OWN k-point energies
    (displacement in Bohr written into the Angstrom XYZ: a unit slip misses by
    1.89x, asserted non-trivial |g| > 1e-3);
  - a 1x1x1 mesh equals the Gamma bindings' gradient and stress;
  - the 1x1x3 mesh force on atom A equals the Gamma force on ONE copy of A in
    the explicit 3x supercell (not the sum over copies);
  - Omega tr(sigma) = dE/de for the isotropic strain of lattice and atoms.
* Flags off: gradient() / stress() are None; the derivatives do not move the
  energy.
* Monkhorst-Pack meshes (even ones are twisted supercells, residue moduli 2N)
  against FD of the binding's own energies, both exxdiv.
* RS-GDF with a ROTATING basis (cc-pVDZ: the rotated-column counters are > 0),
  RS-GDF UHF, and a 1x1x3 mesh, against FD.
* Refusals: range_split with forces/stress (as Gamma), k-point KS stress (the
  k-point KS forces are in test_pbc_kpoint_ks_gradients.py).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ferric

# 1 / ferric's ANGSTROM_TO_BOHR, read back from the binding (1 Angstrom in Bohr).
_PROBE = ferric.Molecule.from_xyz_string(
    "1\nunit probe\nH 1.0 0.0 0.0\n", multiplicity=2
)
BOHR_IN_ANGSTROM = 1.0 / _PROBE.coords_bohr()[0][0]

# Off-axis so every Cartesian component of the force is non-trivial.
H2_ATOMS = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5)]
H2_A = 4.0
H3_ATOMS = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (1.6, 0.9, 0.7)]
H3_A = 4.5
OMEGA = 0.8 / BOHR_IN_ANGSTROM
FD_H = 1e-4
FD_BAR = 1e-6
SCF = dict(energy_conv=1e-13, grad_conv=1e-10, max_iter=400)

PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.1688554"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}


def _mol(atoms_bohr, multiplicity=1):
    lines = [str(len(atoms_bohr)), "pbc k-point gradient test (Angstrom)"]
    for x, y, z in atoms_bohr:
        lines.append(
            f"H {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return ferric.Molecule.from_xyz_string(
        "\n".join(lines) + "\n", multiplicity=multiplicity
    )


def _lat(a_bohr, c_bohr=None):
    a = a_bohr * BOHR_IN_ANGSTROM
    c = (c_bohr if c_bohr is not None else a_bohr) * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]]


def _displaced(atoms, i, x, h):
    out = [list(p) for p in atoms]
    out[i][x] += h
    return out


def _fd(energy, atoms, comps, h):
    return {
        (i, x): (
            energy(_displaced(atoms, i, x, h)) - energy(_displaced(atoms, i, x, -h))
        )
        / (2.0 * h)
        for i, x in comps
    }


def _assert_matches_fd(g, fd, tag):
    assert np.abs(g).max() > 1e-3, f"{tag}: trivial gradient, FD check is vacuous"
    worst = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    assert worst < FD_BAR, f"{tag}: analytic vs FD max miss {worst:.3e}"


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


def _krhf(atoms, basis, mesh=(1, 1, 3), **kw):
    return ferric.run_rhf_kpts(
        _mol(atoms), _lat(H2_A), basis, mesh, omega=OMEGA, **SCF, **kw
    )


def _kuhf(atoms, basis, mesh=(1, 1, 3), **kw):
    return ferric.run_uhf_kpts(
        _mol(atoms, 2), _lat(H3_A), basis, mesh, omega=OMEGA, **SCF, **kw
    )


def test_derivatives_are_none_unless_requested_and_do_not_move_the_energy(basis):
    r = _krhf(H2_ATOMS, basis)
    assert r.gradient() is None and r.stress() is None
    g = _krhf(H2_ATOMS, basis, with_gradient=True)
    assert g.gradient() is not None and g.stress() is None
    s = _krhf(H2_ATOMS, basis, with_stress=True)
    assert s.gradient() is None and s.stress() is not None
    assert abs(r.energy - g.energy) < 1e-12 and abs(r.energy - s.energy) < 1e-12
    u = _kuhf(H3_ATOMS, basis)
    assert u.gradient() is None and u.stress() is None


@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_krhf_gradient_matches_fd_of_own_k_energy(basis, exx):
    r = _krhf(H2_ATOMS, basis, exxdiv=exx, with_gradient=True)
    g = r.gradient()
    assert g.shape == (2, 3)
    assert np.abs(g.sum(axis=0)).max() < 1e-8
    comps = [(0, 0), (0, 2), (1, 1), (1, 2)]
    fd = _fd(lambda at: _krhf(at, basis, exxdiv=exx).energy, H2_ATOMS, comps, FD_H)
    _assert_matches_fd(g, fd, f"KRHF {exx}")


def test_krhf_stress_trace_matches_isotropic_scaling(basis):
    r = _krhf(H2_ATOMS, basis, with_stress=True)
    sigma = r.stress()
    assert sigma.shape == (3, 3)
    assert np.abs(sigma - sigma.T).max() < 1e-8

    def e_scaled(s):
        f = 1.0 + s
        atoms = [tuple(f * c for c in p) for p in H2_ATOMS]
        return ferric.run_rhf_kpts(
            _mol(atoms), _lat(f * H2_A), basis, (1, 1, 3), omega=OMEGA, **SCF
        ).energy

    de = (e_scaled(FD_H) - e_scaled(-FD_H)) / (2.0 * FD_H)
    analytic = H2_A**3 * np.trace(sigma)
    assert abs(analytic) > 1e-3, "trivial stress trace; the check is vacuous"
    assert abs(analytic - de) < FD_BAR, f"Omega tr(sigma) {analytic!r} vs FD {de!r}"


@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_one_point_mesh_equals_the_gamma_bindings(basis, exx):
    k = _krhf(
        H2_ATOMS,
        basis,
        mesh=(1, 1, 1),
        exxdiv=exx,
        with_gradient=True,
        with_stress=True,
    )
    g = ferric.run_rhf_gamma(
        _mol(H2_ATOMS),
        _lat(H2_A),
        basis,
        exxdiv=exx,
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        with_stress=True,
    )
    assert np.abs(g.gradient()).max() > 1e-3
    assert np.abs(k.gradient() - g.gradient()).max() < 1e-7
    assert np.abs(k.stress() - g.stress()).max() < 1e-7


def test_k_mesh_force_is_the_gamma_force_on_one_supercell_copy(basis):
    k = _krhf(H2_ATOMS, basis, with_gradient=True)
    n = 3
    sc_atoms = [(x, y, z + c * H2_A) for c in range(n) for (x, y, z) in H2_ATOMS]
    g = ferric.run_rhf_gamma(
        _mol(sc_atoms),
        _lat(H2_A, n * H2_A),
        basis,
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
    ).gradient()
    # every copy of an atom carries the same force; the k force is ONE copy's
    for c in range(n):
        assert np.abs(g[2 * c : 2 * c + 2] - k.gradient()).max() < 1e-7, f"copy {c}"


def test_kuhf_gradient_and_stress_match_fd_staged_ewald(basis):
    r = _kuhf(H3_ATOMS, basis, with_gradient=True, with_stress=True)
    assert r.converged and r.ewald_start == "staged"
    g = r.gradient()
    assert g.shape == (3, 3)
    assert np.abs(g.sum(axis=0)).max() < 1e-8
    comps = [(0, 0), (1, 2), (2, 1)]
    fd = _fd(lambda at: _kuhf(at, basis).energy, H3_ATOMS, comps, FD_H)
    _assert_matches_fd(g, fd, "KUHF")
    sigma = r.stress()

    def e_scaled(s):
        f = 1.0 + s
        atoms = [tuple(f * c for c in p) for p in H3_ATOMS]
        return ferric.run_uhf_kpts(
            _mol(atoms, 2), _lat(f * H3_A), basis, (1, 1, 3), omega=OMEGA, **SCF
        ).energy

    de = (e_scaled(FD_H) - e_scaled(-FD_H)) / (2.0 * FD_H)
    analytic = H3_A**3 * np.trace(sigma)
    assert abs(analytic) > 1e-3
    assert abs(analytic - de) < FD_BAR, f"Omega tr(sigma) {analytic!r} vs FD {de!r}"


def test_krhf_rsgdf_gradient_matches_fd_of_own_energy(basis):
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _krhf(
        H2_ATOMS, basis, mesh=(1, 1, 2), with_gradient=True, with_stress=True, **kw
    )
    g = r.gradient()
    assert g.shape == (2, 3) and np.isfinite(g).all()
    assert r.stress().shape == (3, 3)
    comps = [(0, 0), (1, 2)]
    fd = _fd(
        lambda at: _krhf(at, basis, mesh=(1, 1, 2), **kw).energy,
        H2_ATOMS,
        comps,
        FD_H,
    )
    _assert_matches_fd(g, fd, "KRHF rsgdf")


def test_refusals(basis):
    with pytest.raises(ValueError, match="range_split"):
        _krhf(
            H2_ATOMS,
            basis,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            range_split=True,
            with_gradient=True,
        )
    with pytest.raises(ValueError, match="with_stress is not implemented for k-point"):
        ferric.run_rks_kpts(
            _mol(H2_ATOMS), _lat(H2_A), basis, "LDA", (1, 1, 2), with_stress=True
        )


@pytest.mark.parametrize("mesh", [(1, 1, 2), (2, 2, 2)])
@pytest.mark.parametrize("exx", ["none", "ewald"])
def test_monkhorst_pack_gradient_matches_fd_of_own_energy(basis, mesh, exx):
    kw = dict(centring="mp", exxdiv=exx)
    r = _krhf(H2_ATOMS, basis, mesh=mesh, with_gradient=True, **kw)
    assert r.centring == "mp"
    comps = [(0, 0), (0, 2), (1, 1), (1, 2)]
    fd = _fd(lambda at: _krhf(at, basis, mesh=mesh, **kw).energy, H2_ATOMS, comps, FD_H)
    _assert_matches_fd(r.gradient(), fd, f"KRHF mp {mesh} {exx}")


def test_krhf_rsgdf_1x1x3_force_and_stress_match_fd(basis):
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _krhf(H2_ATOMS, basis, with_gradient=True, with_stress=True, **kw)
    fd = _fd(
        lambda at: _krhf(at, basis, **kw).energy,
        H2_ATOMS,
        [(0, 0), (0, 2), (1, 1)],
        FD_H,
    )
    _assert_matches_fd(r.gradient(), fd, "KRHF rsgdf 1x1x3")

    def e_scaled(s):
        f = 1.0 + s
        atoms = [tuple(f * c for c in p) for p in H2_ATOMS]
        return ferric.run_rhf_kpts(
            _mol(atoms), _lat(f * H2_A), basis, (1, 1, 3), omega=OMEGA, **SCF, **kw
        ).energy

    de = (e_scaled(FD_H) - e_scaled(-FD_H)) / (2.0 * FD_H)
    analytic = H2_A**3 * np.trace(r.stress())
    assert abs(analytic) > 1e-3
    assert abs(analytic - de) < FD_BAR, f"Omega tr(sigma) {analytic!r} vs FD {de!r}"


def test_kuhf_rsgdf_gradient_matches_fd(basis):
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _kuhf(H3_ATOMS, basis, with_gradient=True, with_stress=True, **kw)
    assert r.converged and r.stress().shape == (3, 3)
    fd = _fd(
        lambda at: _kuhf(at, basis, **kw).energy,
        H3_ATOMS,
        [(0, 0), (1, 2), (2, 1)],
        FD_H,
    )
    _assert_matches_fd(r.gradient(), fd, "KUHF rsgdf 1x1x3")


def test_rotating_basis_rsgdf_force_matches_fd():
    """cc-pVDZ H has a generally contracted s shell: the hcore SR walk and the
    RS-GDF SR 3-centre sum run column-rotated (counters > 0), and the force
    differentiates those rotated walks."""
    ccpvdz = ferric.BasisSet.bundled("cc-pvdz")
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")

    def run(at, **extra):
        return ferric.run_rhf_kpts(
            _mol(at), _lat(H2_A), ccpvdz, (1, 1, 2), omega=OMEGA, **SCF, **kw, **extra
        )

    r = run(H2_ATOMS, with_gradient=True, with_stress=True)
    counters = r.timings["counters"]
    assert counters["k hcore SR rotated columns"] > 0, counters
    assert counters["k rsgdf SR3 rotated columns"] > 0, counters
    fd = _fd(lambda at: run(at).energy, H2_ATOMS, [(0, 0), (0, 2), (1, 1)], FD_H)
    _assert_matches_fd(r.gradient(), fd, "KRHF rsgdf cc-pVDZ (rotated)")
