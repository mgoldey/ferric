"""Gamma-point analytic forces and stress through the Python surface.

`with_gradient=True` / `with_stress=True` on run_rhf_gamma, run_uhf_gamma,
run_rohf_gamma, run_rks_gamma, run_uks_gamma and run_roks_gamma expose the
Rust `ferric_pbc::grad` / `ferric_pbc::stress` entry points
(`result.gradient()`, `result.stress()`).

What is anchored against what (the Rust suites pbc_grad*.rs / pbc_stress.rs
hold the per-term anchors and mutants; this file checks the BINDING):
* EXACTNESS ANCHOR: analytic gradient vs a central finite difference of the
  binding's OWN energies. The displacement is taken in Bohr and written into
  the Angstrom XYZ the binding reads (the binding converts back), so the FD
  is dE/dR in Hartree/Bohr, the unit `gradient()` reports. A unit slip (Å vs
  Bohr) would miss by a factor 1.89; a sign slip by 2|g|; the gradients are
  asserted non-trivial (|g| > 1e-3) so neither can hide in a small number.
* Translation invariance: sum over atoms of dE/dR ~ 0 (a sanity bound only).
* Stress: Omega * tr(sigma) = dE/de for the isotropic strain (1 + e) of the
  lattice rows AND the atoms (fixed fractional coordinates), FD of the
  binding's energies at a fixed Ewald split omega.
* Flags off: gradient() / stress() are None.

Bars: the Rust suites pass at 1e-7 (FD_BAR) for these systems; 1e-6 here
leaves room for the Å round trip and the fresh (re-selected) lattice sums
of the scaled cells, while every mutant in the Rust suites misses by > 1e-4.
UKS uses h = 5e-5 Bohr: at h >= 1e-4 displaced grid points can cross the
hard neighbour mask and the energy jumps (pbc_grad_open.rs module doc).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR

# Off-axis so every Cartesian component of the force is non-trivial
# (pbc_grad_open.rs GRAD_H2_ATOMS / H3_ATOMS, Bohr).
H2_ATOMS = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5)]
H2_A = 4.0
H3_ATOMS = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (1.6, 0.9, 0.7)]
H3_A = 4.5
H_ATOM = [(0.3, 0.2, 0.1)]
# Fixed nuclear-attraction Ewald split (0.8 Bohr^-1, in 1/Angstrom): the
# default depends on the volume, so the stress FD pins it.
OMEGA = 0.8 / BOHR_IN_ANGSTROM
FD_H = 1e-4
FD_H_KS = 5e-5
FD_BAR = 1e-6
NET_FORCE_BAR = 1e-8
# The Rust anchor grid: (40, 50), SSF, D = 8 Bohr.
KS_GRID = dict(n_radial=40, n_angular=50, neighbour_cutoff=8.0 * BOHR_IN_ANGSTROM)

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
    lines = [str(len(atoms_bohr)), "pbc gradient test (Angstrom)"]
    for x, y, z in atoms_bohr:
        lines.append(
            f"H {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return ferric.Molecule.from_xyz_string(
        "\n".join(lines) + "\n", multiplicity=multiplicity
    )


def _cubic(a_bohr):
    a = a_bohr * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


def _displaced(atoms, i, x, h):
    out = [list(p) for p in atoms]
    out[i][x] += h
    return out


def _fd(energy, atoms, comps, h):
    """Central FD dE/dR_{i,x} (Hartree/Bohr) for each (i, x) in comps."""
    return {
        (i, x): (
            energy(_displaced(atoms, i, x, h)) - energy(_displaced(atoms, i, x, -h))
        )
        / (2.0 * h)
        for i, x in comps
    }


def _assert_matches_fd(g, fd, tag):
    assert np.abs(g).max() > 1e-3, (
        f"{tag}: gradient is trivial, the FD check is vacuous"
    )
    worst = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    assert worst < FD_BAR, f"{tag}: analytic vs FD max miss {worst:.3e}"


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


def _rhf(atoms, basis, **kw):
    return ferric.run_rhf_gamma(_mol(atoms), _cubic(H2_A), basis, omega=OMEGA, **kw)


@pytest.fixture(scope="module")
def rhf_dense(basis):
    return _rhf(H2_ATOMS, basis, with_gradient=True, with_stress=True)


# ── flags off ──


def test_derivatives_are_none_unless_requested(basis):
    r = _rhf(H2_ATOMS, basis)
    assert r.gradient() is None and r.stress() is None
    u = ferric.run_uhf_gamma(_mol(H_ATOM, 2), _cubic(H2_A), basis)
    assert u.gradient() is None and u.stress() is None
    only_g = _rhf(H2_ATOMS, basis, with_gradient=True)
    assert only_g.gradient() is not None and only_g.stress() is None
    only_s = _rhf(H2_ATOMS, basis, with_stress=True)
    assert only_s.gradient() is None and only_s.stress() is not None


def test_derivatives_do_not_change_the_energy(basis, rhf_dense):
    # The derivatives run after the SCF; parallel reductions may still move
    # the last bits between two runs, so not exact equality.
    assert abs(_rhf(H2_ATOMS, basis).energy - rhf_dense.energy) < 1e-12


# ── RHF, dense AFT ──


def test_rhf_dense_gradient_matches_fd_of_own_energy(basis, rhf_dense):
    g = rhf_dense.gradient()
    assert g.shape == (2, 3)
    comps = [(i, x) for i in range(2) for x in range(3)]
    fd = _fd(lambda at: _rhf(at, basis).energy, H2_ATOMS, comps, FD_H)
    _assert_matches_fd(g, fd, "RHF dense")


def test_rhf_dense_net_force_vanishes(rhf_dense):
    net = np.abs(rhf_dense.gradient().sum(axis=0)).max()
    assert net < NET_FORCE_BAR, f"sum of forces {net:.3e}"


def test_rhf_dense_stress_trace_matches_isotropic_scaling(basis, rhf_dense):
    sigma = rhf_dense.stress()
    assert sigma.shape == (3, 3)
    # HF with frozen index sets is rotation invariant: sigma symmetric.
    assert np.abs(sigma - sigma.T).max() < 1e-8

    def e_scaled(s):
        f = 1.0 + s
        atoms = [tuple(f * c for c in p) for p in H2_ATOMS]
        return ferric.run_rhf_gamma(
            _mol(atoms), _cubic(f * H2_A), basis, omega=OMEGA
        ).energy

    de = (e_scaled(FD_H) - e_scaled(-FD_H)) / (2.0 * FD_H)
    volume = H2_A**3
    analytic = volume * np.trace(sigma)
    assert abs(analytic) > 1e-3, "trivial stress trace; the check is vacuous"
    assert abs(analytic - de) < FD_BAR, f"Omega tr(sigma) {analytic!r} vs FD {de!r}"


# ── RHF, RS-GDF ──


def test_rhf_rsgdf_gradient_matches_fd_of_own_energy(basis, rhf_dense):
    def run(at, **kw):
        return _rhf(at, basis, jk="rsgdf", auxbasis="cc-pvdz-ri", **kw)

    r = run(H2_ATOMS, with_gradient=True, with_stress=True)
    g = r.gradient()
    assert g.shape == (2, 3) and np.isfinite(g).all()
    assert r.stress().shape == (3, 3)
    assert np.abs(g.sum(axis=0)).max() < NET_FORCE_BAR
    # The fitted force is close to (not equal to) the dense one.
    assert np.abs(g - rhf_dense.gradient()).max() < 1e-3
    comps = [(0, 0), (1, 2)]
    fd = _fd(lambda at: run(at).energy, H2_ATOMS, comps, FD_H)
    _assert_matches_fd(g, fd, "RHF rsgdf")


# ── UKS PBE, dense AFT ──


def test_uks_pbe_gradient_matches_fd_of_own_energy(basis):
    def run(at, **kw):
        return ferric.run_uks_gamma(
            _mol(at, 2), _cubic(H3_A), basis, "PBE", omega=OMEGA, **KS_GRID, **kw
        )

    r = run(H3_ATOMS, with_gradient=True)
    assert r.converged
    g = r.gradient()
    assert g.shape == (3, 3)
    assert np.abs(g.sum(axis=0)).max() < NET_FORCE_BAR
    comps = [(0, 0), (2, 1), (1, 2)]
    fd = _fd(lambda at: run(at).energy, H3_ATOMS, comps, FD_H_KS)
    _assert_matches_fd(g, fd, "UKS PBE")


# ── refusals ──


@pytest.mark.parametrize("fn", ["run_rohf_gamma", "run_roks_gamma"])
def test_restricted_open_rsgdf_stress_is_refused_before_any_work(basis, fn):
    args = (_mol(H_ATOM, 2), _cubic(H2_A), basis)
    if fn == "run_roks_gamma":
        args = args + ("LDA",)
    with pytest.raises(ValueError, match="dense-AFT J/K only"):
        getattr(ferric, fn)(*args, jk="rsgdf", auxbasis="cc-pvdz-ri", with_stress=True)
