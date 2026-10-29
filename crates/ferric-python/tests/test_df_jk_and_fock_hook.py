"""`AoIntegrals.df_jk()` and `run_rhf(fock_hook=)` / `run_dft(fock_hook=)`.

Anchors are independent constructions:

* `DfJk` J/K equal a numpy fit assembled from `eri3()` + `eri2()`, and a
  numpy SCF on `DfJk` reproduces `run_rhf`'s own RI-JK SCF;
* a zero hook is bit-identical to no hook (the trivial limit);
* a constant-field hook reproduces `external_field=`, which folds the same
  term into hcore once before the loop instead of into F every iteration;
* a density-dependent hook matches a numpy SCF carrying the same term, and
  its converged energy obeys Hellmann-Feynman (dE/dλ = ∂E_hook/∂λ at fixed
  D), which fails if the hook energy and potential are wired inconsistently.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}


def _water(basis: str):
    mol = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _roothaan(ints, jk, term=None) -> tuple[float, np.ndarray]:
    """Plain Roothaan on `hcore() + J - K/2 (+ V_term)`, E with E_term."""
    s, h = ints.overlap(), ints.hcore()
    w, v = np.linalg.eigh(s)
    x = v / np.sqrt(w)
    nocc = ints.nelec // 2
    d = np.zeros_like(s)
    for it in range(300):
        j, k = jk(d)
        e_x, v_x = term(d) if term else (0.0, 0.0)
        _, c = np.linalg.eigh(x.T @ (h + j - 0.5 * k + v_x) @ x)
        c = x @ c
        d_new = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        if np.abs(d_new - d).max() < 1e-10:
            d = d_new
            break
        d = 0.5 * (d + d_new) if it > 100 else d_new
    else:
        pytest.fail("numpy Roothaan loop did not converge")
    j, k = jk(d)
    e_x = term(d)[0] if term else 0.0
    e = 0.5 * np.sum(d * (2.0 * h + j - 0.5 * k)) + e_x + ints.nuclear_repulsion()
    return e, d


# ── DfJk ──


@pytest.mark.parametrize(
    "operator, omega", [("coulomb", None), ("erfc", 0.4)], ids=["coulomb", "erfc"]
)
def test_df_jk_matches_a_numpy_fit_from_eri3_and_eri2(operator, omega):
    _, _, ints = _water("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    d = ferric.run_rhf(*_water("cc-pvdz")[:2], **TIGHT).density()
    b = ints.eri3(aux, operator, omega=omega)
    m = ints.eri2(aux, operator, omega=omega)
    gamma = np.linalg.solve(m, np.einsum("pkl,kl->p", b, d))
    j_ref = np.einsum("pij,p->ij", b, gamma)
    b_fit = np.linalg.solve(m, b.reshape(len(m), -1)).reshape(b.shape)
    k_ref = np.einsum("pik,pjl,kl->ij", b, b_fit, d)

    df = ints.df_jk(aux, operator, omega=omega)
    assert df.naux == len(m)
    j, k = df.jk(d)
    np.testing.assert_allclose(j, j_ref, atol=1e-9)
    np.testing.assert_allclose(k, k_ref, atol=1e-9)
    np.testing.assert_array_equal(df.j(d), j)
    np.testing.assert_array_equal(df.k(d), k)


def test_df_jk_carries_no_state_between_densities():
    mol, bs, ints = _water("sto-3g")
    df = ints.df_jk(ferric.BasisSet.bundled("cc-pvdz-ri"))
    d1 = ferric.run_rhf(mol, bs, **TIGHT).density()
    d2 = 0.5 * (d1 + np.eye(ints.nao))
    first = df.jk(d1)
    df.jk(d2)
    again = df.jk(d1)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)


def test_numpy_scf_on_df_jk_reproduces_run_rhf_ri_jk():
    mol, bs, ints = _water("cc-pvdz")
    df = ints.df_jk(ferric.BasisSet.bundled("cc-pvdz-ri"))
    e_np, _ = _roothaan(ints, df.jk)
    r = ferric.run_rhf(mol, bs, df_j_aux="cc-pvdz-ri", df_k_aux="cc-pvdz-ri", **TIGHT)
    assert abs(e_np - r.energy) < 1e-9


def test_df_jk_rejects_bad_densities():
    _, _, ints = _water("sto-3g")
    df = ints.df_jk(ferric.BasisSet.bundled("cc-pvdz-ri"))
    n = ints.nao
    with pytest.raises(ValueError, match="shape"):
        df.j(np.zeros((n, n + 1)))
    asym = np.zeros((n, n))
    asym[0, 1] = 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        df.k(asym)
    with pytest.raises(ValueError, match="NaN or inf"):
        df.jk(np.full((n, n), np.nan))


# ── fock_hook ──


def _zero_hook(d):
    return 0.0, np.zeros_like(d)


def test_zero_hook_is_bit_identical_to_no_hook():
    mol, bs, _ = _water("cc-pvdz")
    plain = ferric.run_rhf(mol, bs, **TIGHT)
    hooked = ferric.run_rhf(mol, bs, fock_hook=_zero_hook, **TIGHT)
    assert hooked.energy == plain.energy
    assert hooked.iterations == plain.iterations
    np.testing.assert_array_equal(hooked.density(), plain.density())


def test_zero_hook_is_bit_identical_in_run_dft():
    mol, bs, _ = _water("sto-3g")
    plain = ferric.run_dft(mol, bs, functional="PBE", **TIGHT)
    hooked = ferric.run_dft(mol, bs, functional="PBE", fock_hook=_zero_hook, **TIGHT)
    assert hooked.total_energy == plain.total_energy


def _field_hook(ints, mol, field_z):
    """H' = +E·r per electron (ferric's convention), as a hook."""
    v = field_z * ints.dipole()[2]
    zs = [{"O": 8, "H": 1}[sym] for sym in mol.symbols()]
    e_nuc = -field_z * sum(z * xyz[2] for z, xyz in zip(zs, mol.coords_bohr()))
    return lambda d: (float(np.sum(d * v)) + e_nuc, v)


def test_field_hook_reproduces_external_field():
    mol, bs, ints = _water("cc-pvdz")
    f = 0.01
    ref = ferric.run_rhf(mol, bs, external_field=(0.0, 0.0, f), **TIGHT)
    hooked = ferric.run_rhf(mol, bs, fock_hook=_field_hook(ints, mol, f), **TIGHT)
    assert abs(hooked.energy - ref.energy) < 1e-9
    assert abs(hooked.energy - ferric.run_rhf(mol, bs, **TIGHT).energy) > 1e-4


def test_field_hook_reproduces_external_field_in_run_dft():
    mol, bs, ints = _water("sto-3g")
    f = 0.01
    ref = ferric.run_dft(mol, bs, functional="PBE", external_field=(0, 0, f), **TIGHT)
    hooked = ferric.run_dft(
        mol, bs, functional="PBE", fock_hook=_field_hook(ints, mol, f), **TIGHT
    )
    assert abs(hooked.total_energy - ref.total_energy) < 1e-9


def _scaled_coulomb_hook(ints, lam):
    """E = (λ/2) Tr[D J(D)], V = λ J(D): a density-dependent term."""

    def hook(d):
        j, _ = ints.jk(d)
        return 0.5 * lam * float(np.sum(d * j)), lam * j

    return hook


def test_density_dependent_hook_matches_numpy_scf():
    mol, bs, ints = _water("sto-3g")
    hook = _scaled_coulomb_hook(ints, 0.1)
    e_np, _ = _roothaan(ints, ints.jk, term=hook)
    r = ferric.run_rhf(mol, bs, fock_hook=hook, **TIGHT)
    assert r.converged
    assert abs(e_np - r.energy) < 1e-9


def test_hooked_energy_obeys_hellmann_feynman():
    mol, bs, ints = _water("sto-3g")
    lam, h = 0.1, 1e-3

    def energy(x):
        return ferric.run_rhf(mol, bs, fock_hook=_scaled_coulomb_hook(ints, x), **TIGHT)

    r0 = energy(lam)
    j, _ = ints.jk(r0.density())
    analytic = 0.5 * float(np.sum(r0.density() * j))
    fd = (energy(lam + h).energy - energy(lam - h).energy) / (2 * h)
    assert abs(fd - analytic) < 1e-6 * abs(analytic)


def test_hook_sees_the_total_density():
    mol, bs, ints = _water("sto-3g")
    seen = []

    def hook(d):
        seen.append(float(np.trace(d @ ints.overlap())))
        return _zero_hook(d)

    ferric.run_rhf(mol, bs, fock_hook=hook, **TIGHT)
    assert seen and all(abs(n - ints.nelec) < 1e-8 for n in seen[1:])


def test_hook_exception_propagates_unchanged():
    mol, bs, _ = _water("sto-3g")

    def hook(d):
        raise KeyError("from the hook")

    with pytest.raises(KeyError, match="from the hook"):
        ferric.run_rhf(mol, bs, fock_hook=hook)


@pytest.mark.parametrize(
    "hook, exc, match",
    [
        (lambda d: (0.0, np.zeros((2, 2))), ValueError, "shape"),
        (lambda d: (0.0, np.triu(np.ones_like(d))), ValueError, "non-symmetric"),
        (lambda d: (float("nan"), np.zeros_like(d)), ValueError, "non-finite"),
        (lambda d: np.zeros_like(d), TypeError, "must return"),
    ],
    ids=["shape", "asymmetric", "nan-energy", "bad-return"],
)
def test_bad_hook_returns_are_rejected(hook, exc, match):
    mol, bs, _ = _water("sto-3g")
    with pytest.raises(exc, match=match):
        ferric.run_rhf(mol, bs, fock_hook=hook)


def test_non_callable_hook_is_a_type_error():
    mol, bs, _ = _water("sto-3g")
    with pytest.raises(TypeError, match="callable"):
        ferric.run_rhf(mol, bs, fock_hook=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"soscf": True}, "newton_trigger"),
        ({"stability_descent": True}, "check_stability"),
        ({"symmetry": True}, "symmetry"),
        ({"trah_trigger": 1e-2}, "trah_trigger"),
    ],
)
def test_run_rhf_refuses_settings_that_ignore_the_hook(kwargs, match):
    mol, bs, _ = _water("sto-3g")
    with pytest.raises(ValueError, match=match):
        ferric.run_rhf(mol, bs, fock_hook=_zero_hook, **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"with_gradient": True}, "with_gradient"),
        ({"symmetry": True}, "symmetry"),
        ({"dispersion": "d3bj"}, "dispersion"),
    ],
)
def test_run_dft_refuses_settings_that_ignore_the_hook(kwargs, match):
    mol, bs, _ = _water("sto-3g")
    with pytest.raises(ValueError, match=match):
        ferric.run_dft(mol, bs, functional="PBE", fock_hook=_zero_hook, **kwargs)


# ── open-shell fock_hook (run_uhf / run_rohf) ──
#
# hook(D_a, D_b) -> (E, V_a, V_b). The spin-dependent hooks below make a
# swapped V_a/V_b, a sign error or a missing E_hook visible: Hellmann-Feynman
# holds only when E and both V_s are wired consistently.

NH2 = "3\n\nN 0 0 0\nH 0 0.802 0.597\nH 0 -0.802 0.597\n"
# No symmetry: in C2v NH2 the singly occupied orbital is b1, so a totally
# symmetric hook cannot rotate it and ROHF only sees F_a + F_b, which is
# blind to a swapped V_a/V_b (measured: that mutation passed on C2v NH2).
NH2_ASYM = "3\n\nN 0 0 0\nH 0.12 0.80 0.60\nH -0.05 -0.74 0.52\n"
SOLVERS = {"uhf": ferric.run_uhf, "rohf": ferric.run_rohf}


def _nh2(basis: str, xyz: str = NH2):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=2)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _zero_spin_hook(d_a, d_b):
    return 0.0, np.zeros_like(d_a), np.zeros_like(d_b)


@pytest.mark.parametrize("kind", SOLVERS)
def test_zero_spin_hook_is_bit_identical_to_no_hook(kind):
    mol, bs, _ = _nh2("6-31g")
    plain = SOLVERS[kind](mol, bs, **TIGHT)
    hooked = SOLVERS[kind](mol, bs, fock_hook=_zero_spin_hook, **TIGHT)
    assert hooked.energy == plain.energy
    assert hooked.iterations == plain.iterations
    np.testing.assert_array_equal(hooked.density_alpha(), plain.density_alpha())


@pytest.mark.parametrize("kind", SOLVERS)
def test_spin_field_hook_reproduces_external_field(kind):
    mol, bs, ints = _nh2("6-31g")
    f = 0.01
    v = f * ints.dipole()[2]
    zs = [{"N": 7, "H": 1}[sym] for sym in mol.symbols()]
    e_nuc = -f * sum(z * xyz[2] for z, xyz in zip(zs, mol.coords_bohr()))

    def hook(d_a, d_b):
        return float(np.sum((d_a + d_b) * v)) + e_nuc, v, v

    ref = SOLVERS[kind](mol, bs, external_field=(0.0, 0.0, f), **TIGHT)
    hooked = SOLVERS[kind](mol, bs, fock_hook=hook, **TIGHT)
    assert abs(hooked.energy - ref.energy) < 1e-9
    assert abs(hooked.energy - SOLVERS[kind](mol, bs, **TIGHT).energy) > 1e-4


def _alpha_only_hook(ints, lam):
    """E = λ Tr[D_a μ_z], V_a = λ μ_z, V_b = 0: acts on one spin only."""
    m = ints.dipole()[2]
    return lambda d_a, d_b: (lam * float(np.sum(d_a * m)), lam * m, 0.0 * m)


def _spin_coulomb_hook(ints, lam):
    """E = (λ/2) Tr[Ds J(Ds)], Ds = D_a - D_b: V_a = λ J(Ds), V_b = -V_a."""

    def hook(d_a, d_b):
        j, _ = ints.jk(d_a - d_b)
        return 0.5 * lam * float(np.sum((d_a - d_b) * j)), lam * j, -lam * j

    return hook


def _alpha_only_derivative(ints, r):
    return float(np.sum(r.density_alpha() * ints.dipole()[2]))


def _spin_coulomb_derivative(ints, r):
    ds = r.density_alpha() - r.density_beta()
    return 0.5 * float(np.sum(ds * ints.jk(ds)[0]))


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize(
    "make_hook, derivative, lam",
    [
        (_alpha_only_hook, _alpha_only_derivative, 0.02),
        (_spin_coulomb_hook, _spin_coulomb_derivative, 0.1),
    ],
    ids=["alpha-only", "spin-coulomb"],
)
def test_spin_hook_energy_obeys_hellmann_feynman(kind, make_hook, derivative, lam):
    # Richardson-extrapolated central difference (error O(h^4)): the plain
    # one at h = 1e-3 carries an O(h^2) term of ~7e-7 here (measured: it
    # falls 4x per halving of h), too close to a 1e-6 bar to separate a
    # wiring error from truncation.
    mol, bs, ints = _nh2("sto-3g", NH2_ASYM)

    def energy(x):
        return SOLVERS[kind](mol, bs, fock_hook=make_hook(ints, x), **TIGHT).energy

    def central(h):
        return (energy(lam + h) - energy(lam - h)) / (2 * h)

    analytic = derivative(
        ints, SOLVERS[kind](mol, bs, fock_hook=make_hook(ints, lam), **TIGHT)
    )
    fd = (4 * central(5e-4) - central(1e-3)) / 3
    assert abs(fd - analytic) < 1e-8 * max(abs(analytic), 1.0)


@pytest.mark.parametrize("kind", SOLVERS)
def test_spin_hook_sees_the_per_spin_densities(kind):
    mol, bs, ints = _nh2("sto-3g")
    seen = []

    def hook(d_a, d_b):
        s = ints.overlap()
        seen.append((float(np.trace(d_a @ s)), float(np.trace(d_b @ s))))
        return _zero_spin_hook(d_a, d_b)

    SOLVERS[kind](mol, bs, fock_hook=hook, **TIGHT)
    assert seen and all(
        abs(na - 5) < 1e-8 and abs(nb - 4) < 1e-8 for na, nb in seen[1:]
    )


@pytest.mark.parametrize("kind", SOLVERS)
def test_spin_hook_exception_propagates_unchanged(kind):
    mol, bs, _ = _nh2("sto-3g")

    def hook(d_a, d_b):
        raise KeyError("from the spin hook")

    with pytest.raises(KeyError, match="from the spin hook"):
        SOLVERS[kind](mol, bs, fock_hook=hook)


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize(
    "hook, exc, match",
    [
        (lambda a, b: (0.0, np.zeros_like(a)), TypeError, "open-shell fock_hook must"),
        (
            lambda a, b: (0.0, np.zeros_like(a), np.zeros((2, 2))),
            ValueError,
            "V_b of shape",
        ),
        (
            lambda a, b: (0.0, np.triu(np.ones_like(a)), np.zeros_like(b)),
            ValueError,
            "non-symmetric V_a",
        ),
    ],
    ids=["two-tuple", "shape", "asymmetric"],
)
def test_bad_spin_hook_returns_are_rejected(kind, hook, exc, match):
    mol, bs, _ = _nh2("sto-3g")
    with pytest.raises(exc, match=match):
        SOLVERS[kind](mol, bs, fock_hook=hook)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"stability_descent": True}, "check_stability"),
        ({"trah_trigger": 1e-2}, "trah_trigger"),
    ],
)
def test_run_uhf_refuses_settings_that_ignore_the_hook(kwargs, match):
    mol, bs, _ = _nh2("sto-3g")
    with pytest.raises(ValueError, match=match):
        ferric.run_uhf(mol, bs, fock_hook=_zero_spin_hook, **kwargs)


# ── hooks under density fitting ──
#
# The hooked solvers take the same RhfConfig, so df_j_aux / df_k_aux apply
# to a hooked SCF exactly as to a plain one (run_dft's RI-J/RI-K defaults
# included). Same anchors as above, with the fits on.

DF = {"df_j_aux": "cc-pvdz-ri", "df_k_aux": "cc-pvdz-ri"}


def _field_spin_hook(ints, mol, f, zs):
    v = f * ints.dipole()[2]
    e_nuc = -f * sum(z * xyz[2] for z, xyz in zip(zs, mol.coords_bohr()))
    return lambda d_a, d_b: (float(np.sum((d_a + d_b) * v)) + e_nuc, v, v)


@pytest.mark.parametrize("driver", ["rhf", "dft"])
def test_closed_shell_hooks_under_density_fitting(driver):
    mol, bs, ints = _water("sto-3g")
    kw = dict(DF, **TIGHT)
    if driver == "dft":
        kw["functional"] = "B3LYP"
        run = ferric.run_dft
        energy = lambda r: r.total_energy  # noqa: E731
    else:
        run = ferric.run_rhf
        energy = lambda r: r.energy  # noqa: E731
    plain = run(mol, bs, **kw)
    assert energy(run(mol, bs, fock_hook=_zero_hook, **kw)) == energy(plain)
    f = 0.01
    ref = run(mol, bs, external_field=(0.0, 0.0, f), **kw)
    hooked = run(mol, bs, fock_hook=_field_hook(ints, mol, f), **kw)
    assert abs(energy(hooked) - energy(ref)) < 1e-9
    exact = dict(kw, df_j_aux="none", df_k_aux="none")
    assert abs(energy(run(mol, bs, **exact)) - energy(plain)) > 1e-6


@pytest.mark.parametrize("kind", SOLVERS)
def test_open_shell_hooks_under_density_fitting(kind):
    mol, bs, ints = _nh2("sto-3g")
    zs = [{"N": 7, "H": 1}[sym] for sym in mol.symbols()]
    run = SOLVERS[kind]
    kw = dict(DF, **TIGHT)
    plain = run(mol, bs, **kw)
    assert run(mol, bs, fock_hook=_zero_spin_hook, **kw).energy == plain.energy
    f = 0.01
    ref = run(mol, bs, external_field=(0.0, 0.0, f), **kw)
    hooked = run(mol, bs, fock_hook=_field_spin_hook(ints, mol, f, zs), **kw)
    assert abs(hooked.energy - ref.energy) < 1e-9
    assert abs(run(mol, bs, **TIGHT).energy - plain.energy) > 1e-6
