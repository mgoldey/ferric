"""`fock_hook` symmetry tolerance, the hooked `run_dft` ladder and `dm0=`.

Anchors:

* a hook V that is symmetric up to round-off is symmetrized as ½(V + Vᵀ)
  before use, so it gives the energy of its symmetrized form exactly; the
  acceptance bound scales with ‖V‖ (4 n ε ‖V‖_F), so a μ = 1e6 projector
  passes and an asymmetry above the bound is still rejected;
* a zero hook on the hooked ladder is bit-identical to the plain ladder when
  rung 0 runs out of iterations and rung 1 converges;
* `dm0` equal to the converged density of an identical run restarts at the
  solution: same energy within energy_conv, in at most two iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9, "ri_j_precision": "f64"}
EPS = np.finfo(float).eps


def _water(basis: str, charge: int = 0, multiplicity: int = 1):
    mol = ferric.Molecule.from_xyz_string(WATER, charge=charge, multiplicity=multiplicity)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _projector_hook(ints, mu: float, symmetrize: bool):
    """V = μ S C Cᵀ S over the two lowest core-Hamiltonian orbitals.

    The raw product is symmetric in exact arithmetic and asymmetric by
    round-off, which is how the projection-embedding level shift is built.
    The energy is evaluated with the symmetrized V either way, so the two
    hooks differ only in the V they return.
    """
    s = ints.overlap()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    _, c = np.linalg.eigh(x.T @ ints.hcore() @ x)
    c = x @ c[:, :2]
    v_raw = mu * (s @ c @ c.T @ s)
    v_sym = 0.5 * (v_raw + v_raw.T)
    v = v_sym if symmetrize else v_raw
    return (lambda d: (float(np.sum(d * v_sym)), v)), v


# ── fock_hook symmetry ──


def test_roundoff_asymmetric_hook_equals_its_symmetrized_form():
    mol, bs, ints = _water("cc-pvdz", charge=2)
    raw, v_raw = _projector_hook(ints, 1e6, symmetrize=False)
    sym, _ = _projector_hook(ints, 1e6, symmetrize=True)
    asym = np.abs(v_raw - v_raw.T).max()
    # The case the old absolute 1e-10 bound rejected.
    assert asym > 1e-10
    assert asym < 4 * v_raw.shape[0] * EPS * np.linalg.norm(v_raw)
    e_raw = ferric.run_rhf(mol, bs, fock_hook=raw, **TIGHT)
    e_sym = ferric.run_rhf(mol, bs, fock_hook=sym, **TIGHT)
    assert e_raw.converged and e_sym.converged
    # Exact: the driver forms ½(V + Vᵀ) with the same two flops numpy does.
    assert e_raw.energy == e_sym.energy


def test_asymmetry_above_the_roundoff_bound_is_rejected():
    mol, bs, ints = _water("sto-3g")
    _, v = _projector_hook(ints, 1e6, symmetrize=True)
    n = v.shape[0]
    tol = 4 * n * EPS * np.linalg.norm(v)
    skew = np.zeros_like(v)
    skew[0, 1], skew[1, 0] = 5.0 * tol, -5.0 * tol
    v_bad = v + skew

    def hook(d):
        return float(np.sum(d * v_bad)), v_bad

    with pytest.raises(ValueError, match="non-symmetric V.*round-off bound"):
        ferric.run_rhf(mol, bs, fock_hook=hook)


def test_open_shell_hook_accepts_roundoff_asymmetry():
    mol, bs, ints = _water("sto-3g", charge=1, multiplicity=2)
    raw, _ = _projector_hook(ints, 1e6, symmetrize=False)
    sym, _ = _projector_hook(ints, 1e6, symmetrize=True)

    def spin(h):
        def hook(d_a, d_b):
            e, v = h(d_a + d_b)
            return e, v, v

        return hook

    e_raw = ferric.run_uhf(mol, bs, fock_hook=spin(raw), **TIGHT).energy
    e_sym = ferric.run_uhf(mol, bs, fock_hook=spin(sym), **TIGHT).energy
    assert e_raw == e_sym


# ── hooked run_dft ladder ──


def _zero_hook(d):
    return 0.0, np.zeros_like(d)


def test_hooked_run_dft_continues_the_ladder():
    # max_iter caps rung 0 only; rung 1 (ADIIS, 60 iterations) converges.
    mol, bs, _ = _water("sto-3g")
    assert ferric.run_dft(mol, bs, functional="PBE", **TIGHT).iterations > 3
    kw = {"functional": "PBE", "max_iter": 3, **TIGHT}
    plain = ferric.run_dft(mol, bs, **kw)
    hooked = ferric.run_dft(mol, bs, fock_hook=_zero_hook, **kw)
    assert hooked.converged
    assert hooked.iterations == plain.iterations
    assert hooked.total_energy == plain.total_energy


# ── dm0 ──


@pytest.mark.parametrize("basis", ["cc-pvdz", "aug-cc-pvdz"])
def test_rhf_dm0_from_a_converged_run_restarts_at_the_solution(basis):
    mol, bs, _ = _water(basis)
    first = ferric.run_rhf(mol, bs, **TIGHT)
    again = ferric.run_rhf(mol, bs, dm0=first.density(), **TIGHT)
    assert again.converged
    assert first.iterations > 2
    assert again.iterations <= 2
    assert abs(again.energy - first.energy) < TIGHT["energy_conv"]


def test_dft_dm0_from_a_converged_run_restarts_at_the_solution():
    mol, bs, _ = _water("cc-pvdz")
    first = ferric.run_dft(mol, bs, functional="PBE", **TIGHT)
    again = ferric.run_dft(mol, bs, functional="PBE", dm0=first.density(), **TIGHT)
    assert first.iterations > 2
    assert again.iterations <= 2
    assert abs(again.total_energy - first.total_energy) < TIGHT["energy_conv"]


def test_dm0_accepts_a_mixture_of_densities():
    # A damped mixture keeps tr(D S) = N exactly; the bound must admit it.
    mol, bs, _ = _water("cc-pvdz")
    hf = ferric.run_rhf(mol, bs, **TIGHT)
    ks = ferric.run_dft(mol, bs, functional="PBE", **TIGHT)
    mix = 0.3 * hf.density() + 0.7 * ks.density()
    r = ferric.run_rhf(mol, bs, dm0=mix, **TIGHT)
    assert abs(r.energy - hf.energy) < 1e-9


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda d: d[:-1, :-1], "shape"),
        (lambda d: d + np.triu(np.full_like(d, 1e-6), 1), "not symmetric"),
        (lambda d: np.where(np.eye(len(d), dtype=bool), np.nan, d), "NaN"),
        (lambda d: 1.0000001 * d, "electrons"),
    ],
    ids=["shape", "asymmetric", "nan", "electron-count"],
)
def test_bad_dm0_is_rejected(make, match):
    mol, bs, _ = _water("sto-3g")
    d = ferric.run_rhf(mol, bs).density()
    with pytest.raises(ValueError, match=match):
        ferric.run_rhf(mol, bs, dm0=make(d))
    with pytest.raises(ValueError, match=match):
        ferric.run_dft(mol, bs, functional="PBE", dm0=make(d))


def test_dm0_of_another_charge_state_is_rejected():
    # The embedding mistake: the full molecule's density for subsystem A.
    mol, bs, _ = _water("sto-3g")
    cation, _, _ = _water("sto-3g", charge=2)
    d = ferric.run_rhf(mol, bs).density()
    with pytest.raises(ValueError, match="system has 8"):
        ferric.run_rhf(cation, bs, dm0=d)


@pytest.mark.parametrize("kwargs", [{"guess": "hcore"}, {"guess_basis": "sto-3g"}])
def test_dm0_with_another_guess_is_rejected(kwargs):
    mol, bs, _ = _water("cc-pvdz")
    d = ferric.run_rhf(mol, bs).density()
    with pytest.raises(ValueError, match="dm0 and"):
        ferric.run_rhf(mol, bs, dm0=d, **kwargs)
