"""`with_gradient=True` on run_rhf / run_uhf / run_rohf / run_rimp2 / run_oo_rimp2.

Anchors (independent constructions):

* BITWISE: run_rhf / run_uhf gradients equal the all-QM run_qmmm
  ``qm_gradient()``, a second binding that solves the same SCF config and
  calls ``ferric_scf::gradient::rhf_gradient`` / ``uhf_gradient`` directly.
  run_rohf, run_rimp2 and run_oo_rimp2 have no second Python route to their
  Rust gradient, so they are anchored by finite differences only.
* Central finite differences of each binding's OWN energy, on a geometry
  with no symmetry (every component non-trivial), for the plain SCFs, RI-J/K,
  point charges + field, IEF-PCM, RI-MP2 and OO-RI-MP2.
* Refusals raise before the SCF; without the flag ``gradient()`` is None.

FD bar. Central difference at step h: error = h^2 |E'''| / 6 + delta_E / h.
With h = 1e-3 Bohr and |E'''| <= 1 Ha/Bohr^3 for these bond-length
displacements the truncation term is <= 1.7e-7; delta_E (SCF converged to
density_conv 1e-9, energy error second order in it, integral screening
1e-12) is <= 1e-11, so delta_E / h <= 1e-8. FD_BAR = 1e-6 is ~5x the sum.
The measured misses are printed with -s.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

FD_H = 1e-3
FD_BAR = 1e-6
TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}
# OO-MP2: the zero-response gradient holds at the stationary point; a loose
# grad_conv leaves a residual linear in it (oo_rimp2_gradient.rs tests use
# the same settings).
OO_TIGHT = {"grad_conv": 1e-8, "scf_density_conv": 1e-10, "oo_max_iter": 200}

# Water and NH2 off any symmetry element (Angstrom).
WATER = (
    ["O", "H", "H"],
    [(0.01, 0.02, 0.117), (0.03, 0.757, -0.467), (-0.02, -0.75, -0.45)],
)
NH2 = (["N", "H", "H"], [(0.0, 0.01, 0.0), (0.02, 0.80, 0.60), (-0.03, -0.81, 0.58)])
ANG = 1.0 / 0.52917721092  # ferric's ANGSTROM_TO_BOHR


def _mol(atoms, multiplicity=1, bohr=None):
    symbols, coords = atoms
    if bohr is None:
        return ferric.Molecule.from_arrays(symbols, coords, multiplicity=multiplicity)
    return ferric.Molecule.from_arrays(
        symbols, bohr, units="bohr", multiplicity=multiplicity
    )


def _fd_gradient(energy, atoms, multiplicity=1):
    """Central FD dE/dR (natoms x 3, Hartree/Bohr) of energy(Molecule)."""
    base = np.asarray(atoms[1], dtype=float) * ANG
    g = np.zeros_like(base)
    for i in range(base.shape[0]):
        for x in range(3):
            plus, minus = base.copy(), base.copy()
            plus[i, x] += FD_H
            minus[i, x] -= FD_H
            e_p = energy(_mol(atoms, multiplicity, plus))
            e_m = energy(_mol(atoms, multiplicity, minus))
            g[i, x] = (e_p - e_m) / (2.0 * FD_H)
    return g


def _assert_fd(analytic, fd, tag):
    assert analytic.shape == fd.shape
    assert np.abs(analytic).max() > 1e-3, (
        f"{tag}: trivial gradient, FD check is vacuous"
    )
    miss = np.abs(analytic - fd).max()
    print(f"{tag}: max |analytic - FD| = {miss:.2e}")
    assert miss < FD_BAR, f"{tag}: analytic vs FD max miss {miss:.3e}"


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def b631g():
    return ferric.BasisSet.bundled("6-31g")


# ── bitwise: the binding returns the Rust gradient ──


@pytest.mark.parametrize(
    "atoms, mult, method, run",
    [(WATER, 1, "rhf", "run_rhf"), (NH2, 2, "uhf", "run_uhf")],
    ids=["rhf", "uhf"],
)
def test_gradient_equals_the_rust_gradient_bitwise(atoms, mult, method, run, b631g):
    symbols, coords = atoms
    r = getattr(ferric, run)(
        _mol(atoms, mult), b631g, density_conv=1e-9, with_gradient=True
    )
    system = ferric.QmmmSystem(
        symbols,
        coords,
        [0.0] * len(symbols),
        qm_indices=list(range(len(symbols))),
        multiplicity=mult,
    )
    q = ferric.run_qmmm(system, "6-31g", method=method, density_conv=1e-9)
    assert r.energy == q.energy
    assert np.array_equal(r.gradient(), q.qm_gradient())


# ── finite differences of each binding's own energy ──


@pytest.mark.parametrize(
    "run, atoms, mult, extra",
    [
        ("run_rhf", WATER, 1, {}),
        (
            "run_rhf",
            WATER,
            1,
            {
                "df_j_aux": "def2-universal-jkfit",
                "df_k_aux": "def2-universal-jkfit",
                "ri_j_precision": "f64",
            },
        ),
        ("run_uhf", NH2, 2, {}),
        ("run_rohf", NH2, 2, {}),
    ],
    ids=["rhf", "rhf-rijk", "uhf", "rohf"],
)
def test_scf_gradient_matches_fd_of_its_energy(run, atoms, mult, extra, b631g):
    fn = getattr(ferric, run)
    kw = {**TIGHT, **extra}
    r = fn(_mol(atoms, mult), b631g, with_gradient=True, **kw)
    fd = _fd_gradient(lambda m: fn(m, b631g, **kw).energy, atoms, mult)
    _assert_fd(r.gradient(), fd, f"{run} {extra}")


def test_rhf_gradient_with_point_charges_and_field_matches_fd(b631g):
    kw = {
        **TIGHT,
        "point_charges": [(0.4, 3.0, 1.0, -2.0), (-0.4, -2.5, 2.0, 3.0)],
        "external_field": (0.01, -0.005, 0.02),
    }
    r = ferric.run_rhf(_mol(WATER), b631g, with_gradient=True, **kw)
    fd = _fd_gradient(lambda m: ferric.run_rhf(m, b631g, **kw).energy, WATER)
    _assert_fd(r.gradient(), fd, "rhf point charges + field")


def test_rhf_gradient_with_pcm_matches_fd(sto3g):
    kw = {**TIGHT, "solvent": "water"}
    r = ferric.run_rhf(_mol(WATER), sto3g, with_gradient=True, **kw)
    vac = ferric.run_rhf(_mol(WATER), sto3g, with_gradient=True, **TIGHT)
    assert np.abs(r.gradient() - vac.gradient()).max() > 1e-4, (
        "PCM term is not in the gradient"
    )
    fd = _fd_gradient(lambda m: ferric.run_rhf(m, sto3g, **kw).energy, WATER)
    _assert_fd(r.gradient(), fd, "rhf pcm")


def test_rimp2_gradient_matches_fd_of_its_energy():
    obs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    kw = {"scf_energy_conv": 1e-10, "scf_density_conv": 1e-10}
    r = ferric.run_rimp2(_mol(WATER), obs, aux, with_gradient=True, **kw)
    fd = _fd_gradient(lambda m: ferric.run_rimp2(m, obs, aux, **kw).total_energy, WATER)
    _assert_fd(r.gradient(), fd, "rimp2")


def test_rimp2_frozen_core_gradient_matches_fd():
    obs = ferric.BasisSet.bundled("6-31g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    kw = {"scf_energy_conv": 1e-10, "scf_density_conv": 1e-10, "frozen_core": 1}
    r = ferric.run_rimp2(_mol(WATER), obs, aux, with_gradient=True, **kw)
    fd = _fd_gradient(lambda m: ferric.run_rimp2(m, obs, aux, **kw).total_energy, WATER)
    _assert_fd(r.gradient(), fd, "rimp2 frozen core")


def test_oo_rimp2_gradient_matches_fd_of_its_energy(sto3g):
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    r = ferric.run_oo_rimp2(_mol(WATER), sto3g, aux, with_gradient=True, **OO_TIGHT)
    assert r.converged
    fd = _fd_gradient(
        lambda m: ferric.run_oo_rimp2(m, sto3g, aux, **OO_TIGHT).total_energy, WATER
    )
    _assert_fd(r.gradient(), fd, "oo-rimp2")


# ── flags, refusals ──


def test_gradient_is_none_without_the_flag_and_the_energy_is_unchanged(sto3g):
    plain = ferric.run_rhf(_mol(WATER), sto3g, **TIGHT)
    assert plain.gradient() is None
    assert ferric.run_uhf(_mol(NH2, 2), sto3g).gradient() is None
    with_g = ferric.run_rhf(_mol(WATER), sto3g, with_gradient=True, **TIGHT)
    assert with_g.energy == plain.energy
    assert with_g.gradient().shape == (3, 3)
    # A molecule in vacuum: the net force vanishes.
    assert np.abs(with_g.gradient().sum(axis=0)).max() < 1e-8


def test_cosx_gradient_run_drops_the_final_grid_pass(sto3g):
    r = ferric.run_rhf(_mol(WATER), sto3g, k_builder="cosx", with_gradient=True)
    assert r.cosx_final_pass is None
    assert (
        ferric.run_rhf(_mol(WATER), sto3g, k_builder="cosx").cosx_final_pass is not None
    )


@pytest.mark.parametrize(
    "run, atoms, mult", [("run_rhf", WATER, 1), ("run_uhf", NH2, 2)]
)
def test_gradient_run_resolves_an_unset_ri_j_precision_to_f64(run, atoms, mult, sto3g):
    fn = getattr(ferric, run)
    kw = {"df_j_aux": "def2-universal-jkfit"}

    def ri_j(**extra):
        r = fn(_mol(atoms, mult), sto3g, **kw, **extra)
        (detail,) = [
            a["detail"] for a in r.certificate["approximations"] if a["name"] == "ri-j"
        ]
        return detail.split(",")[1].split()[0]

    assert ri_j() == "f32"
    assert ri_j(with_gradient=True) == "f64"
    assert ri_j(with_gradient=True, ri_j_precision="f32") == "f32"


def _hook(d):
    return 0.0, np.zeros_like(d)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"symmetry": True}, "symmetry"),
        ({"fock_hook": _hook}, "fock_hook"),
        ({"smearing_sigma": 0.01}, "smearing_sigma"),
        ({"mom_after_iter": 2}, "mom_after_iter"),
    ],
    ids=["symmetry", "fock_hook", "smearing", "mom"],
)
def test_rhf_gradient_refusals(kw, match, sto3g):
    with pytest.raises(ValueError, match=match):
        ferric.run_rhf(_mol(WATER), sto3g, with_gradient=True, **kw)


def test_open_shell_gradient_refuses_a_fock_hook(sto3g):
    def hook(da, db):
        return 0.0, np.zeros_like(da), np.zeros_like(db)

    for fn in (ferric.run_uhf, ferric.run_rohf):
        with pytest.raises(ValueError, match="fock_hook"):
            fn(_mol(NH2, 2), sto3g, fock_hook=hook, with_gradient=True)


def test_gradient_of_an_unconverged_scf_raises(sto3g):
    with pytest.warns(RuntimeWarning):
        assert not ferric.run_rhf(
            _mol(WATER), sto3g, max_iter=2, density_conv=1e-12
        ).converged
    with pytest.raises(RuntimeError, match="(?i)converge"):
        ferric.run_rhf(
            _mol(WATER), sto3g, max_iter=2, density_conv=1e-12, with_gradient=True
        )


def test_ghost_atom_gradient_surfaces_the_rust_refusal(sto3g):
    mol = ferric.Molecule.from_arrays(
        ["O", "H", "H", "He"],
        [*WATER[1], (3.0, 0.0, 0.0)],
        ghost=[False, False, False, True],
    )
    with pytest.raises(RuntimeError, match="ghost"):
        ferric.run_rhf(mol, sto3g, with_gradient=True)


@pytest.mark.parametrize(
    "atoms, mult, kw, match",
    [
        (NH2, 2, {}, "closed-shell"),
        (WATER, 1, {"local": "amplitude-threshold", "eps": 1e-6}, "local"),
        (WATER, 1, {"kappa": 1.0}, "kappa"),
        (WATER, 1, {"scf_jk": "rijk"}, "exact"),
    ],
    ids=["open-shell", "local", "kappa", "rijk"],
)
def test_rimp2_gradient_refusals(atoms, mult, kw, match, sto3g):
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    with pytest.raises(ValueError, match=match):
        ferric.run_rimp2(_mol(atoms, mult), sto3g, aux, with_gradient=True, **kw)


def test_oo_rimp2_gradient_refusals(sto3g):
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    with pytest.raises(ValueError, match="frozen_core"):
        ferric.run_oo_rimp2(_mol(WATER), sto3g, aux, frozen_core=1, with_gradient=True)
    with pytest.raises(RuntimeError, match="did not converge"):
        ferric.run_oo_rimp2(_mol(WATER), sto3g, aux, oo_max_iter=1, with_gradient=True)
