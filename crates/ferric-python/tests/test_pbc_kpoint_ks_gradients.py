"""k-point closed-shell KS analytic forces through the Python surface.

`run_rks_kpts(..., with_gradient=True)` exposes `ferric_pbc::kxcgrad::
kpoint_rks_gradient` (LDA, GGA and global hybrids) on the hcore output and J/K
object the SCF itself ran on (`solve_krks_keep`), with the XC term on the
energy's own grid. The Rust suites (pbc_kxcgrad.rs, pbc_kxcgrad_hybrid.rs,
pbc_kkeep.rs) hold the per-term anchors and mutants and the Fock guard (a
wrong functional or exxdiv cannot be passed through the binding: it takes both
from the energy call); this file checks the BINDING, with the units of the
Gamma bindings (Hartree/Bohr per cell, atoms and lattice in Angstrom in).

The XC energy has a hard neighbour cutoff, so every energy in a comparison
uses ONE explicit `neighbour_cutoff` (the default depends on the cell and makes
the energy jump by one step under a displacement).

* EXACTNESS ANCHORS, independent constructions:
  - analytic force vs a central FD of the binding's OWN k-point energies
    (displacement in Bohr written into the Angstrom XYZ: a unit slip misses by
    1.89x, asserted non-trivial |g| > 1e-3);
  - a 1x1x1 mesh equals the Gamma KS binding's force;
  - the 1x1x3 mesh force on atom A equals the Gamma force on ONE copy of A in
    the explicit 3x supercell.
* Monkhorst-Pack mesh, a hybrid, RS-GDF J/K against FD.
* Flags off: gradient() is None and the energy is bit-identical with the flag.
* Refusals with exact wording: stress, range_split with a force.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402,F401  (fixtures re-exported)
    BOHR_IN_ANGSTROM,
    H2_A,
    H2_ATOMS,
    OMEGA,
    _fd,
    _lat,
    _mol,
    basis,
)

FD_H = 1e-4
FD_BAR = 2e-6
# 10 Bohr, written in Angstrom (the binding's unit for neighbour_cutoff).
CUTOFF = 10.0 * BOHR_IN_ANGSTROM
GRID = dict(n_radial=40, n_angular=110, neighbour_cutoff=CUTOFF)
SCF = dict(energy_conv=1e-13, grad_conv=1e-10, max_iter=400)


def _krks(atoms, basis, functional="PBE", mesh=(1, 1, 3), **kw):
    return ferric.run_rks_kpts(
        _mol(atoms), _lat(H2_A), basis, functional, mesh, omega=OMEGA, **GRID, **SCF, **kw
    )


def _assert_force_matches_fd(g, atoms, energy, tag, comps=((0, 0), (0, 2), (1, 1), (1, 2))):
    assert g.shape == (2, 3) and np.isfinite(g).all()
    assert np.abs(g).max() > 1e-3, f"{tag}: trivial gradient, FD check is vacuous"
    assert np.abs(g.sum(axis=0)).max() < 1e-6, f"{tag}: net force on the cell"
    fd = _fd(energy, atoms, comps, FD_H)
    worst = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    assert worst < FD_BAR, f"{tag}: analytic vs FD max miss {worst:.3e}"


def test_gradient_is_none_unless_requested_and_does_not_move_the_energy(basis):
    off = _krks(H2_ATOMS, basis, "PBE0")
    on = _krks(H2_ATOMS, basis, "PBE0", with_gradient=True)
    assert off.gradient() is None
    assert on.gradient() is not None
    assert on.energy == off.energy  # bit-identical: the same SCF path


@pytest.mark.parametrize(
    "functional, exx",
    [("LDA", "ewald"), ("PBE", "ewald"), ("PBE", "none"), ("PBE0", "ewald"), ("PBE0", "none")],
)
def test_force_matches_fd_of_own_k_energy(basis, functional, exx):
    r = _krks(H2_ATOMS, basis, functional, exxdiv=exx, with_gradient=True)
    assert r.converged
    _assert_force_matches_fd(
        r.gradient(),
        H2_ATOMS,
        lambda at: _krks(at, basis, functional, exxdiv=exx).energy,
        f"{functional} {exx}",
    )


@pytest.mark.parametrize("functional", ["PBE", "PBE0"])
def test_one_point_mesh_equals_the_gamma_ks_binding(basis, functional):
    k = _krks(H2_ATOMS, basis, functional, mesh=(1, 1, 1), with_gradient=True)
    g = ferric.run_rks_gamma(
        _mol(H2_ATOMS),
        _lat(H2_A),
        basis,
        functional,
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        **GRID,
    )
    assert np.abs(g.gradient()).max() > 1e-3
    assert np.abs(k.gradient() - g.gradient()).max() < 1e-6


def test_k_mesh_force_is_the_gamma_force_on_one_supercell_copy(basis):
    k = _krks(H2_ATOMS, basis, "PBE0", with_gradient=True)
    n = 3
    sc_atoms = [(x, y, z + c * H2_A) for c in range(n) for (x, y, z) in H2_ATOMS]
    g = ferric.run_rks_gamma(
        _mol(sc_atoms),
        _lat(H2_A, n * H2_A),
        basis,
        "PBE0",
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        **GRID,
    ).gradient()
    # every copy of an atom carries the same force; the k force is ONE copy's
    for c in range(n):
        assert np.abs(g[2 * c : 2 * c + 2] - k.gradient()).max() < 1e-6, f"copy {c}"


def test_monkhorst_pack_hybrid_force_matches_fd(basis):
    kw = dict(mesh=(1, 1, 2), centring="mp")
    r = _krks(H2_ATOMS, basis, "PBE0", with_gradient=True, **kw)
    assert r.centring == "mp"
    _assert_force_matches_fd(
        r.gradient(),
        H2_ATOMS,
        lambda at: _krks(at, basis, "PBE0", **kw).energy,
        "MP PBE0",
    )


def test_rsgdf_hybrid_force_matches_fd(basis):
    kw = dict(mesh=(1, 1, 2), jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _krks(H2_ATOMS, basis, "PBE0", with_gradient=True, **kw)
    assert r.jk == "rsgdf"
    _assert_force_matches_fd(
        r.gradient(),
        H2_ATOMS,
        lambda at: _krks(at, basis, "PBE0", **kw).energy,
        "RS-GDF PBE0",
        comps=((0, 0), (1, 2)),
    )


def test_memory_budget_smaller_than_the_retained_objects_is_a_typed_error(basis):
    # The retained hcore and J/K count against memory_budget_gb; a budget that
    # covers only them leaves ~1 byte for the derivative, which must fail its
    # ledger with a typed error instead of running unbounded (budget 0 would
    # mean "unset").
    with pytest.raises((ValueError, RuntimeError), match="budget|memory"):
        _krks(
            H2_ATOMS,
            basis,
            "PBE",
            with_gradient=True,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            memory_budget_gb=1e-9,
        )


def test_refusals_have_exact_wording(basis):
    with pytest.raises(
        ValueError,
        match=r"run_rks_kpts: with_stress is not implemented for k-point Kohn-Sham "
        r"\(the analytic k-point stress exists for run_rhf_kpts / run_uhf_kpts and, "
        r"at the Gamma point, run_rks_gamma\); with_gradient=True is supported",
    ):
        _krks(H2_ATOMS, basis, "PBE", with_stress=True)
    with pytest.raises(ValueError, match="range_split"):
        _krks(
            H2_ATOMS,
            basis,
            "PBE",
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            range_split=True,
            with_gradient=True,
        )
    with pytest.raises(ValueError, match="meta-GGA"):
        _krks(H2_ATOMS, basis, "SCAN", with_gradient=True)
    with pytest.raises(ValueError, match="range-separated"):
        _krks(H2_ATOMS, basis, "HSE06", with_gradient=True)
