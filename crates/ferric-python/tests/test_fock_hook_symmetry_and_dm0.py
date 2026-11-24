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

import warnings

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9, "ri_j_precision": "f64"}
EPS = np.finfo(float).eps


def _water(basis: str, charge: int = 0, multiplicity: int = 1):
    mol = ferric.Molecule.from_xyz_string(
        WATER, charge=charge, multiplicity=multiplicity
    )
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


def test_hooked_run_dft_ladder_keeps_the_first_rungs_guess_basis_report():
    # Rung 0 runs the secondary-basis guess and stops at max_iter; the rung
    # that converges starts from its density and runs no guess of its own.
    mol, bs, _ = _water("6-31g")
    kw = {"functional": "PBE", "max_iter": 1, "guess_basis": "sto-3g", **TIGHT}
    plain = ferric.run_dft(mol, bs, **kw)
    hooked = ferric.run_dft(mol, bs, fock_hook=_zero_hook, **kw)
    assert plain.converged and hooked.converged
    assert plain.guess_basis is not None
    assert hooked.guess_basis == plain.guess_basis


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


def test_ksdft_alias_forwards_dm0():
    mol, bs, _ = _water("cc-pvdz")
    first = ferric.run_ksdft(mol, bs, functional="PBE", **TIGHT)
    again = ferric.run_ksdft(mol, bs, functional="PBE", dm0=first.density(), **TIGHT)
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


# ── overflow-safe symmetry bound and symmetrization ──


def _const_hook(v):
    return lambda d: (0.0, v)


@pytest.mark.parametrize("scale", [1e200, 1e300])
def test_huge_asymmetric_hook_is_rejected(scale):
    # Squaring entries of 1e200 overflows a naive Frobenius norm to inf, and
    # an infinite bound accepts any asymmetry. The bound is formed on
    # max|V|-scaled entries, so a 100 percent asymmetry is still rejected.
    mol, bs, _ = _water("sto-3g")
    n = ferric.AoIntegrals(mol, bs).overlap().shape[0]
    v = np.eye(n) * scale
    v[0, 1] = scale
    with pytest.raises(ValueError, match="non-symmetric V"):
        ferric.run_rhf(mol, bs, fock_hook=_const_hook(v))


def test_huge_symmetric_hook_passes_the_symmetry_check():
    mol, bs, _ = _water("sto-3g")
    n = ferric.AoIntegrals(mol, bs).overlap().shape[0]
    v = np.eye(n) * 1e200
    v[0, 1] = v[1, 0] = 1e200
    try:
        ferric.run_rhf(mol, bs, fock_hook=_const_hook(v))
    except Exception as exc:  # the SCF on a 1e200 V need not converge
        assert "non-symmetric" not in str(exc)


def test_symmetrization_does_not_overflow_for_entries_near_float_max():
    # (V + Vᵀ) overflows to inf for 1.5e308 entries, and the SCF then fails in
    # the eigensolver; ½V + ½Vᵀ stays finite. The 1.5e308 shift is far above
    # the Fock entries, so the orbitals it leaves are not the physical ones
    # and the run's convergence and energy are not claimed; the test requires
    # only that the symmetrized hook reaches the SCF as a finite matrix.
    mol, bs, _ = _water("sto-3g")
    n = ferric.AoIntegrals(mol, bs).overlap().shape[0]
    v = np.eye(n) * 1.5e308
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ferric.run_rhf(mol, bs, fock_hook=_const_hook(v))
    assert np.isfinite(res.energy)


def test_dm0_with_overflowing_entries_is_rejected():
    # Σ|D_ij S_ij| overflows to inf, which makes the electron-count bound
    # infinite and any count acceptable.
    mol, bs, ints = _water("sto-3g")
    n = ints.overlap().shape[0]
    with pytest.raises(ValueError, match="not finite"):
        ferric.run_rhf(mol, bs, dm0=np.eye(n) * 1e308)


# ── hook provenance on the certificate ──


def _approx(c):
    return {a["name"]: a["detail"] for a in c["approximations"]}


@pytest.mark.parametrize(
    "run",
    [
        lambda mol, bs, **kw: ferric.run_rhf(mol, bs, **kw),
        lambda mol, bs, **kw: ferric.run_dft(mol, bs, functional="PBE", **kw),
    ],
    ids=["run_rhf", "run_dft"],
)
def test_hooked_certificate_records_the_hook_and_keeps_the_anchor(run):
    mol, bs, _ = _water("sto-3g")
    plain = run(mol, bs, **TIGHT).certificate
    hooked = run(mol, bs, fock_hook=_zero_hook, **TIGHT).certificate
    assert "fock-hook" not in _approx(plain)
    assert "fock-hook" in _approx(hooked)
    assert hooked["measured_error"] == plain["measured_error"]
    assert hooked["grade"] == plain["grade"]
    rest = dict(_approx(hooked))
    del rest["fock-hook"]
    assert rest == _approx(plain)


def test_hooked_run_uhf_certificate_records_the_hook():
    mol, bs, _ = _water("sto-3g", charge=1, multiplicity=2)

    def spin_zero(d_a, d_b):
        return 0.0, np.zeros_like(d_a), np.zeros_like(d_b)

    plain = ferric.run_uhf(mol, bs, **TIGHT).certificate
    hooked = ferric.run_uhf(mol, bs, fock_hook=spin_zero, **TIGHT).certificate
    assert "fock-hook" not in _approx(plain)
    assert "fock-hook" in _approx(hooked)
    assert hooked["measured_error"] == plain["measured_error"]


# ── dm0 does not reach the free-atom proatoms ──


def test_hirshfeld_scf_proatoms_ignore_dm0():
    # A molecular dm0 (molecule basis dimension) reaching a free-atom SCF is
    # rejected there and that atom falls back to the Slater proatom, which
    # moves water's charges by 0.2-0.7 e. The two densities agree to the
    # density_conv scale (1e-9), so the charges agree far below 1e-6.
    mol, bs, _ = _water("cc-pvdz")
    first = ferric.run_rhf(mol, bs, **TIGHT)
    again = ferric.run_rhf(mol, bs, dm0=first.density(), **TIGHT)
    q1 = np.array(ferric.hirshfeld_charges(mol, bs, first))
    q2 = np.array(ferric.hirshfeld_charges(mol, bs, again))
    assert np.max(np.abs(q1 - q2)) < 1e-6, (q1, q2)
