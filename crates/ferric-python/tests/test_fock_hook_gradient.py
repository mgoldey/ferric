"""Nuclear gradients of Fock-hooked SCFs: ``fock_hook_grad=``.

The gradient of a hooked SCF is the built-in analytic gradient of the hooked
result (its energy-weighted density comes from the hooked orbitals, so it
carries the hook's Pulay term) plus the hook's EXPLICIT term dE_hook/dR at
fixed AO density, which the caller supplies (derivation:
`book/src/07-hartree-fock/06-fock-hooks.md#fock-hooks-equations`).

Hooks used (each a function of the AO density and the geometry):

* ``trace_ds``: E = lam tr(D S), V = lam S, explicit term lam tr(D dS/dR).
  E = lam N is a constant, so the orbitals are those of the plain SCF and the
  gradient must equal the plain gradient: the explicit term and the hook's
  Pulay term -tr(1/2 lam D S D dS/dR) = -lam tr(D dS/dR) cancel. EXACTNESS
  ANCHOR: both halves are nonzero (lam = 0.1 gives ~1e-2 Ha/Bohr each) and
  their sum is known in closed form.
* ``scaled_h``: E = lam tr(D h), V = lam h, explicit term lam tr(D dh/dR)
  (``AoIntegrals.grad_hcore``). Both halves nonzero, and the orbitals change.
* ``d_squared``: E = lam sum D_uv^2 (per spin open-shell), V = 2 lam D; it
  depends on the density only, so the explicit term is ZERO and only the
  Pulay term carries the hook.

Bars.

* Anchor (hooked vs plain gradient): the two SCFs see Fock matrices that
  differ by lam S, which shifts every eigenvalue by lam and leaves the
  eigenvectors and the DIIS error FDS - SDF unchanged up to rounding, so the
  converged densities differ by rounding plus the stopping error. The
  gradient is linear in the density at fixed integrals; with density_conv
  1e-9 per element, at most 13^2 elements and derivative integrals below
  10 Ha/Bohr, ANCHOR_BAR = 2e-6 bounds it. Measured misses are printed.
* FD: as test_python_gradients.py (h = 1e-3 Bohr, FD_BAR = 1e-6, derivation
  there). The KS (LDA) FD bar adds the plain KS analytic-vs-FD miss measured
  in the same test (the grid is not translated with the atoms), so it tests
  only what the hook adds.

MEASURED (2026-11-14, 6-31G): anchor |hooked - plain| 2.8e-11 (RHF),
9.2e-12 (UHF), 4.8e-15 (ROHF) with each half ~3e-2 Ha/Bohr; FD misses
1.4e-7..1.8e-7 for scaled_h and d_squared on RHF/UHF/ROHF; LDA hooked miss
1.6e-7 against a plain miss of 1.5e-7; the constant-hook optimization ends
3e-9 Bohr from the plain one with the energy shifted by lam N to 9e-14.

Mutations (each made in the source, seen to fail on an assertion, restored):

* the explicit term dropped (``PyFockHookGrad::add_to`` returns ``grad``):
  8 tests fail (anchors by 2.7e-2..3.1e-2, ``scaled_h`` FD by 0.56..0.70,
  LDA, the constant-hook optimization); ``d_squared`` passes, as it must;
* the hook's V dropped from the energy-weighted density (the gradient gains
  ``+ tr[W_V dS/dR]``, W_V = 1/2 D V D closed-shell, sum_s D_s V_s D_s
  open-shell: the W of the un-hooked Fock): 11 tests fail, including
  ``d_squared`` (explicit term zero) by 1.9e-2.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

FD_H = 1e-3
FD_BAR = 1e-6
ANCHOR_BAR = 2e-6
LAM = 0.1
TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}

WATER = (
    ["O", "H", "H"],
    [(0.01, 0.02, 0.117), (0.03, 0.757, -0.467), (-0.02, -0.75, -0.45)],
)
NH2 = (["N", "H", "H"], [(0.0, 0.01, 0.0), (0.02, 0.80, 0.60), (-0.03, -0.81, 0.58)])
ANG = 1.0 / 0.52917721092  # ferric's ANGSTROM_TO_BOHR


@pytest.fixture(scope="module")
def basis():
    return ferric.BasisSet.bundled("6-31g")


def _mol(atoms, multiplicity=1, bohr=None):
    symbols, coords = atoms
    if bohr is None:
        return ferric.Molecule.from_arrays(symbols, coords, multiplicity=multiplicity)
    return ferric.Molecule.from_arrays(
        symbols, bohr, units="bohr", multiplicity=multiplicity
    )


# --- hooks ------------------------------------------------------------------
# Each factory returns (hook, grad) for one geometry: closed-shell hook(D) and
# grad(D, mol), or open-shell hook(Da, Db) and grad(Da, Db, mol).


def trace_ds(mol, bs, open_shell=False):
    ao = ferric.AoIntegrals(mol, bs)
    s = ao.overlap()
    if open_shell:
        return (
            lambda da, db: (LAM * np.sum((da + db) * s), LAM * s, LAM * s),
            lambda da, db, m: LAM * ao.grad_overlap(da + db),
        )
    return (
        lambda d: (LAM * np.sum(d * s), LAM * s),
        lambda d, m: LAM * ao.grad_overlap(d),
    )


def scaled_h(mol, bs, open_shell=False):
    ao = ferric.AoIntegrals(mol, bs)
    h = ao.hcore()
    if open_shell:
        return (
            lambda da, db: (LAM * np.sum((da + db) * h), LAM * h, LAM * h),
            lambda da, db, m: LAM * ao.grad_hcore(da + db),
        )
    return (
        lambda d: (LAM * np.sum(d * h), LAM * h),
        lambda d, m: LAM * ao.grad_hcore(d),
    )


def d_squared(mol, bs, open_shell=False):
    natoms = len(mol.symbols())
    if open_shell:
        return (
            lambda da, db: (
                LAM * (np.sum(da * da) + np.sum(db * db)),
                2 * LAM * da,
                2 * LAM * db,
            ),
            lambda da, db, m: np.zeros((natoms, 3)),
        )
    return (
        lambda d: (LAM * np.sum(d * d), 2 * LAM * d),
        lambda d, m: np.zeros((natoms, 3)),
    )


RUNNERS = {
    "rhf": (ferric.run_rhf, False, 1, WATER),
    "uhf": (ferric.run_uhf, True, 2, NH2),
    "rohf": (ferric.run_rohf, True, 2, NH2),
}


def _energy(name, hooks, bs):
    run, open_shell, mult, atoms = RUNNERS[name]

    def energy(mol):
        hook, _ = hooks(mol, bs, open_shell)
        return run(mol, bs, fock_hook=hook, **TIGHT).energy

    return energy, atoms, mult


def _hooked_gradient(name, hooks, bs, mol=None):
    run, open_shell, mult, atoms = RUNNERS[name]
    mol = mol or _mol(atoms, mult)
    hook, grad = hooks(mol, bs, open_shell)
    r = run(mol, bs, fock_hook=hook, fock_hook_grad=grad, with_gradient=True, **TIGHT)
    return np.asarray(r.gradient())


def _fd_gradient(energy, atoms, multiplicity):
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


def _miss(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


# --- the exactness anchor ---------------------------------------------------


@pytest.mark.parametrize("name", ["rhf", "uhf", "rohf"])
def test_constant_trace_hook_leaves_the_gradient_unchanged(name, basis):
    run, open_shell, mult, atoms = RUNNERS[name]
    mol = _mol(atoms, mult)
    r = run(mol, basis, with_gradient=True, **TIGHT)
    plain = np.asarray(r.gradient())
    hooked = _hooked_gradient(name, trace_ds, basis)
    # Each half is nonzero: the explicit term alone is lam tr(D dS/dR).
    d = r.density_alpha() + r.density_beta() if open_shell else r.density()
    half = LAM * np.abs(ferric.AoIntegrals(mol, basis).grad_overlap(d)).max()
    miss = _miss(hooked, plain)
    print(f"{name}: |hooked - plain| = {miss:.2e} (each half ~{half:.2e})")
    assert half > 1e-3
    assert miss < ANCHOR_BAR


# --- finite differences of the hooked energy ---------------------------------


@pytest.mark.parametrize("name", ["rhf", "uhf", "rohf"])
@pytest.mark.parametrize("hooks", [scaled_h, d_squared], ids=["scaled_h", "d_squared"])
def test_hooked_gradient_matches_finite_differences(name, hooks, basis):
    energy, atoms, mult = _energy(name, hooks, basis)
    analytic = _hooked_gradient(name, hooks, basis)
    fd = _fd_gradient(energy, atoms, mult)
    miss = _miss(analytic, fd)
    print(f"{name}/{hooks.__name__}: max |analytic - FD| = {miss:.2e}")
    assert np.abs(analytic).max() > 1e-3
    assert miss < FD_BAR


def test_hooked_ks_gradient_matches_finite_differences(basis):
    """run_dft (LDA, exact J): the hooked KS gradient against FD. The bar adds
    the plain KS analytic-vs-FD miss measured here (grid not translated)."""
    kw = {"functional": "LDA", "df_j_aux": "", "df_k_aux": "", **TIGHT}
    mol = _mol(WATER)

    def plain(m):
        return ferric.run_dft(m, basis, **kw).total_energy

    def hooked(m):
        hook, _ = scaled_h(m, basis)
        return ferric.run_dft(m, basis, fock_hook=hook, **kw).total_energy

    g_plain = np.asarray(
        ferric.run_dft(mol, basis, with_gradient=True, **kw).gradient()
    )
    hook, grad = scaled_h(mol, basis)
    g_hooked = np.asarray(
        ferric.run_dft(
            mol, basis, fock_hook=hook, fock_hook_grad=grad, with_gradient=True, **kw
        ).gradient()
    )
    plain_miss = _miss(g_plain, _fd_gradient(plain, WATER, 1))
    miss = _miss(g_hooked, _fd_gradient(hooked, WATER, 1))
    print(f"rks: plain miss {plain_miss:.2e}, hooked miss {miss:.2e}")
    assert miss < FD_BAR + 2 * plain_miss
    # The hook changes the gradient far beyond that bar.
    assert _miss(g_hooked, g_plain) > 100 * (FD_BAR + 2 * plain_miss)


# --- geometry optimization -----------------------------------------------------


def _geometry_hooks(factory, bs):
    """run_optimize's form: hook(D, mol) and grad(D, mol), rebuilt per geometry."""
    cache = {}

    def at(mol):
        key = tuple(map(tuple, np.round(np.asarray(mol.coords_bohr()), 14)))
        if key not in cache:
            cache.clear()
            cache[key] = factory(mol, bs)
        return cache[key]

    return (lambda d, mol: at(mol)[0](d)), (lambda d, mol: at(mol)[1](d, mol))


def test_optimize_with_a_constant_hook_reaches_the_plain_geometry():
    """lam tr(D S) = lam N: the hooked surface is the plain one shifted by lam N,
    so BFGS takes the same steps. Geometries agree to the optimizer's
    stopping noise, energies to it plus lam N."""
    bs = ferric.BasisSet.bundled("sto-3g")
    mol = _mol(WATER)
    plain = ferric.run_optimize(mol, "sto-3g", scf_warm_start=False)
    hook, grad = _geometry_hooks(trace_ds, bs)
    hooked = ferric.run_optimize(mol, "sto-3g", fock_hook=hook, fock_hook_grad=grad)
    assert plain.converged and hooked.converged
    shift = hooked.energy - plain.energy - LAM * 10
    dx = _miss(hooked.mol().coords_bohr(), plain.mol().coords_bohr())
    print(f"optimize: energy shift miss {shift:.2e}, max |dx| {dx:.2e} Bohr")
    assert abs(shift) < 1e-8
    assert dx < 1e-4
    cert = hooked.certificate
    assert any(a["name"] == "scf fock-hook" for a in cert["approximations"])


def test_optimize_with_a_geometry_dependent_hook_reaches_a_stationary_point():
    """scaled_h moves the minimum; at the returned geometry the hooked
    gradient (from run_rhf) is below the optimizer's gradient threshold."""
    bs = ferric.BasisSet.bundled("sto-3g")
    mol = _mol(WATER)
    hook, grad = _geometry_hooks(scaled_h, bs)
    plain = ferric.run_optimize(mol, "sto-3g", scf_warm_start=False)
    r = ferric.run_optimize(mol, "sto-3g", fock_hook=hook, fock_hook_grad=grad)
    assert r.converged
    h2, g2 = scaled_h(r.mol(), bs)
    g = np.asarray(
        ferric.run_rhf(
            r.mol(), bs, fock_hook=h2, fock_hook_grad=g2, with_gradient=True, **TIGHT
        ).gradient()
    )
    moved = _miss(r.mol().coords_bohr(), plain.mol().coords_bohr())
    print(f"scaled_h optimize: max|g| = {np.abs(g).max():.2e}, moved {moved:.2e} Bohr")
    assert np.abs(g).max() < 1e-3
    assert moved > 1e-3


# --- refusals -------------------------------------------------------------------


def _zero_hook(d):
    return 0.0, np.zeros_like(d)


def _zero_grad(d, mol):
    return np.zeros((len(mol.symbols()), 3))


def test_a_hooked_gradient_without_fock_hook_grad_names_the_kwarg(basis):
    mol = _mol(WATER)
    with pytest.raises(ValueError, match="fock_hook_grad"):
        ferric.run_rhf(mol, basis, fock_hook=_zero_hook, with_gradient=True)
    with pytest.raises(ValueError, match="fock_hook_grad"):
        ferric.run_dft(mol, basis, fock_hook=_zero_hook, with_gradient=True)

    def spin_hook(da, db):
        return 0.0, np.zeros_like(da), np.zeros_like(db)

    for fn in (ferric.run_uhf, ferric.run_rohf):
        with pytest.raises(ValueError, match="fock_hook_grad"):
            fn(_mol(NH2, 2), basis, fock_hook=spin_hook, with_gradient=True)


def test_fock_hook_grad_needs_a_hook_and_a_gradient(basis):
    mol = _mol(WATER)
    with pytest.raises(ValueError, match="needs a fock_hook"):
        ferric.run_rhf(mol, basis, fock_hook_grad=_zero_grad, with_gradient=True)
    with pytest.raises(ValueError, match="with_gradient=True"):
        ferric.run_rhf(mol, basis, fock_hook=_zero_hook, fock_hook_grad=_zero_grad)


def test_a_bad_explicit_term_raises(basis):
    mol = _mol(WATER)
    kw = {"fock_hook": _zero_hook, "with_gradient": True}
    with pytest.raises(ValueError, match="shape"):
        ferric.run_rhf(mol, basis, fock_hook_grad=lambda d, m: np.zeros((2, 3)), **kw)
    with pytest.raises(ValueError, match="non-finite"):
        ferric.run_rhf(
            mol, basis, fock_hook_grad=lambda d, m: np.full((3, 3), np.nan), **kw
        )
    with pytest.raises(TypeError, match="callable"):
        ferric.run_rhf(mol, basis, fock_hook_grad=3.0, **kw)

    def boom(d, m):
        raise KeyError("from the hook")

    with pytest.raises(KeyError, match="from the hook"):
        ferric.run_rhf(mol, basis, fock_hook_grad=boom, **kw)


def test_optimize_hook_refusals():
    mol = _mol(WATER)

    def hook(d, m):
        return _zero_hook(d)

    with pytest.raises(ValueError, match="go together"):
        ferric.run_optimize(mol, "sto-3g", fock_hook=hook)
    with pytest.raises(ValueError, match='method="scf"'):
        ferric.run_optimize(
            mol,
            "sto-3g",
            method="rimp2",
            auxbasis="cc-pvdz-ri",
            fock_hook=hook,
            fock_hook_grad=_zero_grad,
        )
    with pytest.raises(ValueError, match="scf_warm_start"):
        ferric.run_optimize(
            mol,
            "sto-3g",
            fock_hook=hook,
            fock_hook_grad=_zero_grad,
            scf_warm_start=True,
        )
