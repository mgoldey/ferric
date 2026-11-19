"""Roadmap 6b: the keyword drift between the six molecular SCF entry points.

``run_rhf`` / ``run_uhf`` / ``run_rohf`` / ``run_dft`` / ``run_uks`` /
``run_roks`` now take every keyword a sibling with the same solver already
took (appended at the end of each signature, default ``None`` = the old
behaviour); the knobs a solver cannot honour stay refused
(``ferric_scf::options::entry::ScfEntryPoint::refused``). Each newly accepted
keyword is shown to take effect, by one of:

* the sibling anchor: on closed-shell water the open-shell solvers reach the
  closed-shell state, so UHF/ROHF with the keyword equal RHF with it, and
  UKS/ROKS equal RKS (energy and gradient), while the keyword moves the energy;
* the knob's own signature: the same converged energy reached in a different
  number of iterations (convergence aids), or a refusal only the knob can
  trigger (``eri="incore"`` on an RI route, a COSX final pass on ROHF, a
  hooked gradient without ``fock_hook_grad``);
* the anchor of ``test_fock_hook_gradient``: a hook ``lam * tr(D S)`` adds the
  constant ``lam * N`` and leaves the gradient unchanged.
"""

from __future__ import annotations

import ferric
import numpy as np
import pytest

TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-8}
EXACT_JK = {"df_j_aux": "", "df_k_aux": ""}
WATER = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
# NH2 (2B1): an open shell without a degenerate frontier, so two SCF paths
# reach the same state (OH's 2Pi hole orientation is degenerate up to the XC
# grid's rotational variance, ~1e-6 Ha).
NH2 = "3\nNH2\nN 0 0 0.15\nH 0 0.80 -0.45\nH 0 -0.80 -0.45\n"
SMEARED = [(-0.5, 0.0, 0.0, 5.0, 1.0), (0.3, 4.0, 0.0, 0.0, 0.8)]
LAM = 0.1


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture(scope="module")
def nh2():
    return ferric.Molecule.from_xyz_string(NH2, charge=0, multiplicity=2)


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def b631g():
    return ferric.BasisSet.bundled("6-31g")


def hf(name):
    return {"rhf": ferric.run_rhf, "uhf": ferric.run_uhf, "rohf": ferric.run_rohf}[name]


def ks(name, functional="PBE"):
    if name == "dft":
        return lambda mol, bs, **kw: ferric.run_dft(
            mol, bs, functional=functional, **kw
        )
    run = {"uks": ferric.run_uks, "roks": ferric.run_roks}[name]
    return lambda mol, bs, **kw: run(mol, bs, functional, **kw)


# --- environment keywords: the open-shell solvers reproduce the closed shell ---

ENVIRONMENT = {
    "solvent": {"solvent": "water"},
    "pcm_lebedev_order": {"solvent": "water", "pcm_lebedev_order": 194},
    "smeared_charges": {"smeared_charges": SMEARED},
}


@pytest.mark.parametrize("kw", ENVIRONMENT, ids=list(ENVIRONMENT))
def test_open_shell_hf_takes_the_environment_of_run_rhf(kw, water, sto3g):
    kw = ENVIRONMENT[kw]
    # What the keyword changes: the solvent / smeared charges themselves, or
    # (pcm_lebedev_order) the cavity's tesserae against the default order.
    base = {"solvent": "water"} if "pcm_lebedev_order" in kw else {}
    e_rhf = ferric.run_rhf(water, sto3g, **kw, **TIGHT).energy
    e_off = ferric.run_rhf(water, sto3g, **base, **TIGHT).energy
    assert abs(e_rhf - e_off) > 1e-6
    for name in ["uhf", "rohf"]:
        e = hf(name)(water, sto3g, **kw, **TIGHT).energy
        assert abs(e - e_rhf) < 1e-8, (name, e - e_rhf)


@pytest.mark.parametrize("kw", ENVIRONMENT, ids=list(ENVIRONMENT))
def test_open_shell_ks_takes_the_environment_of_run_dft(kw, water, sto3g):
    kw = ENVIRONMENT[kw]
    base = {"solvent": "water"} if "pcm_lebedev_order" in kw else {}
    e_rks = ks("dft")(water, sto3g, **kw, **TIGHT).energy
    e_off = ks("dft")(water, sto3g, **base, **TIGHT).energy
    assert abs(e_rks - e_off) > 1e-6
    for name in ["uks", "roks"]:
        e = ks(name)(water, sto3g, **kw, **TIGHT).energy
        assert abs(e - e_rks) < 1e-8, (name, e - e_rks)


def test_environment_gradients_match_the_closed_shell(water, sto3g):
    """IEF-PCM and smeared charges in the open-shell gradients: equal to the
    closed-shell gradient of the same state (run_rhf's is the established
    one; run_dft's goes through the same dispatcher)."""
    kw = {
        "solvent": "water",
        "smeared_charges": SMEARED,
        "with_gradient": True,
        **TIGHT,
    }
    g_rhf = np.asarray(ferric.run_rhf(water, sto3g, **kw).gradient())
    g_rks = np.asarray(ks("dft")(water, sto3g, **kw).gradient())
    plain = np.asarray(
        ferric.run_rhf(water, sto3g, with_gradient=True, **TIGHT).gradient()
    )
    assert np.abs(g_rhf - plain).max() > 1e-3
    for name in ["uhf", "rohf"]:
        g = np.asarray(hf(name)(water, sto3g, **kw).gradient())
        assert np.abs(g - g_rhf).max() < 1e-6, name
    for name in ["uks", "roks"]:
        g = np.asarray(ks(name)(water, sto3g, **kw).gradient())
        assert np.abs(g - g_rks).max() < 1e-6, name


# --- convergence aids: same state, different path ---------------------------


def _same_state_new_path(run_default, run_knob, bar=1e-8):
    a, b = run_default(), run_knob()
    assert abs(a.energy - b.energy) < bar, (a.energy, b.energy)
    assert a.iterations != b.iterations, a.iterations
    return a, b


@pytest.mark.parametrize("name", ["uhf", "rohf", "uks", "roks"])
def test_soscf_on_the_open_shell_entry_points(name, nh2, b631g):
    run = hf(name) if name in ("uhf", "rohf") else ks(name)
    _same_state_new_path(
        lambda: run(nh2, b631g, **TIGHT), lambda: run(nh2, b631g, soscf=True, **TIGHT)
    )


@pytest.mark.parametrize("name", ["rohf", "dft", "roks"])
def test_guess_hcore(name, water, nh2, b631g):
    run = hf(name) if name == "rohf" else ks(name)
    mol = water if name == "dft" else nh2
    r = _same_state_new_path(
        lambda: run(mol, b631g, **TIGHT),
        lambda: run(mol, b631g, guess="hcore", **TIGHT),
    )[0]
    with pytest.raises(ValueError, match="unknown guess"):
        run(mol, b631g, guess="bogus")
    dm0 = r.density() if name == "dft" else (r.density_alpha(), r.density_beta())
    with pytest.raises(ValueError, match=f"run_{name}: dm0 and guess both set"):
        run(mol, b631g, guess="hcore", dm0=dm0)


@pytest.mark.parametrize("name", ["dft", "uks", "roks"])
def test_diis_size(name, water, nh2, b631g):
    run = ks(name)
    mol = water if name == "dft" else nh2
    n = 3 if name == "uks" else 2
    _same_state_new_path(
        lambda: run(mol, b631g, **TIGHT), lambda: run(mol, b631g, diis_size=n, **TIGHT)
    )


@pytest.mark.parametrize("name", ["dft", "uks"])
def test_trah_trigger(name, water, nh2, b631g):
    run = ks(name)
    mol = water if name == "dft" else nh2
    _same_state_new_path(
        lambda: run(mol, b631g, **TIGHT),
        lambda: run(mol, b631g, trah_trigger=0.5, **TIGHT),
    )
    with pytest.raises(ValueError, match="trah_trigger must be a positive number"):
        run(mol, b631g, trah_trigger=0.0)


def test_aurora_on_run_dft_for_a_hybrid(water, b631g):
    run = ks("dft", "PBE0")
    _same_state_new_path(
        lambda: run(water, b631g, **TIGHT),
        lambda: run(water, b631g, aurora=True, **TIGHT),
    )


def _ri_j_calls(r):
    return sum(k["calls_mixed"] for k in r.certificate["precision"]["kernels"])


def test_stability_descent_on_run_uks(nh2, sto3g):
    """The certificate's "stability not checked" warning goes away."""
    run = ks("uks")
    plain = run(nh2, sto3g, **TIGHT)
    checked = run(nh2, sto3g, stability_descent=True, **TIGHT)
    not_checked = "internal stability not checked"
    assert any(not_checked in w for w in plain.certificate["warnings"])
    assert not any(not_checked in w for w in checked.certificate["warnings"])
    assert abs(checked.energy - plain.energy) < 1e-8


def test_stability_descent_on_run_dft(water, sto3g):
    """The stability analysis runs (its orbital-Hessian products are extra
    RI-J builds on the certificate's counter) on a stable minimum, which it
    leaves where it is."""
    run = ks("dft")
    plain = run(water, sto3g, **TIGHT)
    checked = run(water, sto3g, stability_descent=True, **TIGHT)
    assert _ri_j_calls(checked) > _ri_j_calls(plain)
    assert checked.energy == plain.energy


# --- integral handling ----------------------------------------------------------


@pytest.mark.parametrize("name", ["dft", "uks", "roks"])
def test_integral_thresh(name, water, nh2, b631g):
    run = ks(name, "PBE0")
    mol = water if name == "dft" else nh2
    # The in-core store is unscreened (as for run_rhf), so screen the direct route.
    kw = {"eri": "direct", **EXACT_JK, **TIGHT}
    e = run(mol, b631g, **kw).energy
    e_loose = run(mol, b631g, integral_thresh=1e-3, **kw).energy
    assert abs(e - e_loose) > 1e-9


@pytest.mark.parametrize("name", ["dft", "uks", "roks"])
def test_eri(name, water, nh2, sto3g):
    run = ks(name, "PBE0")
    mol = water if name == "dft" else nh2
    with pytest.raises(RuntimeError, match="needs the exact four-centre J and K route"):
        run(mol, sto3g, eri="incore")
    with pytest.raises(ValueError, match="eri: unknown value"):
        run(mol, sto3g, eri="bogus")
    e = {
        m: run(mol, sto3g, eri=m, **EXACT_JK, **TIGHT).energy
        for m in ["direct", "incore"]
    }
    assert abs(e["direct"] - e["incore"]) < 1e-9


# --- COSX knobs on run_rohf / run_uks / run_roks --------------------------------


@pytest.mark.parametrize("name", ["rohf", "uks", "roks"])
def test_cosx_knobs(name, nh2, b631g):
    run = hf(name) if name == "rohf" else ks(name, "B3LYP")
    e = run(nh2, b631g, k_builder="cosx").energy
    coarse = run(
        nh2, b631g, k_builder="cosx", cosx_grid=(20, 50), cosx_final_pass=False
    )
    # A knob that did not reach the solver would give the identical energy.
    assert abs(coarse.energy - e) > 1e-8
    with pytest.raises(ValueError, match='only read with k_builder="cosx"'):
        run(nh2, b631g, cosx_grid=(20, 50))


def test_rohf_refuses_an_explicit_cosx_final_pass(nh2, b631g):
    """solve_rohf has no COSX final-grid pass: an explicit one is the
    solver's refusal, not a silently ignored knob."""
    for run in [ferric.run_rohf, ks("roks", "B3LYP")]:
        with pytest.raises(RuntimeError, match="not ROHF/ROKS"):
            run(nh2, b631g, k_builder="cosx", cosx_final_pass=True)


# --- run_uks / run_roks: guess_basis, mom_reference, fock_hook ------------------


@pytest.mark.parametrize("name", ["uks", "roks"])
def test_guess_basis(name, nh2, b631g):
    run = ks(name)
    assert run(nh2, b631g).guess_basis is None
    r = run(nh2, b631g, guess_basis="sto-3g", guess_projection="fock")
    assert r.guess_basis["basis"] == "sto-3g"
    assert r.guess_basis["used_as_guess"]
    with pytest.raises(
        ValueError, match="guess_projection is set but guess_basis is not"
    ):
        run(nh2, b631g, guess_projection="fock")
    n = len(np.asarray(r.density_alpha()))
    dm0 = (np.asarray(r.density_alpha()), np.asarray(r.density_beta()))
    assert n > 0
    with pytest.raises(ValueError, match=f"run_{name}: dm0 and guess_basis"):
        run(nh2, b631g, guess_basis="sto-3g", dm0=dm0)


def test_mom_reference_on_run_uks(nh2, sto3g):
    run = ks("uks")
    plain = run(nh2, sto3g, **TIGHT)
    assert run(nh2, sto3g, mom_reference="previous", **TIGHT).energy == plain.energy
    held = run(nh2, sto3g, mom_reference="initial", **TIGHT)
    assert held.converged
    with pytest.raises(ValueError, match='run_uks: mom_reference="initial"'):
        run(nh2, sto3g, mom_reference="initial", mom_after_iter=2)
    with pytest.raises(ValueError, match="mom_reference"):
        run(nh2, sto3g, mom_reference="bogus")


def _trace_hook(mol, bs):
    """E = lam tr((Da + Db) S) = lam N: a constant energy shift whose explicit
    nuclear derivative lam tr(D dS/dR) cancels the hooked Pulay term."""
    ao = ferric.AoIntegrals(mol, bs)
    s = np.asarray(ao.overlap())
    return (
        lambda da, db: (LAM * np.sum((da + db) * s), LAM * s, LAM * s),
        lambda da, db, m: LAM * np.asarray(ao.grad_overlap(da + db)),
    )


@pytest.mark.parametrize("name", ["uks", "roks"])
def test_fock_hook_on_open_shell_ks(name, nh2, sto3g):
    run = ks(name)
    hook, grad = _trace_hook(nh2, sto3g)
    plain = run(nh2, sto3g, with_gradient=True, **TIGHT)
    hooked = run(
        nh2, sto3g, fock_hook=hook, fock_hook_grad=grad, with_gradient=True, **TIGHT
    )
    nelec = 9  # NH2
    assert abs(hooked.energy - (plain.energy + LAM * nelec)) < 1e-8
    miss = np.abs(np.asarray(hooked.gradient()) - np.asarray(plain.gradient())).max()
    assert miss < 2e-6, miss
    with pytest.raises(
        ValueError, match=f"run_{name}: with_gradient=True with a fock_hook"
    ):
        run(nh2, sto3g, fock_hook=hook, with_gradient=True)
    with pytest.raises(
        ValueError, match=f"run_{name}: fock_hook_grad needs a fock_hook"
    ):
        run(nh2, sto3g, fock_hook_grad=grad, with_gradient=True)


def test_fock_hook_refuses_what_it_would_bypass_on_run_uks(nh2, sto3g):
    hook, _ = _trace_hook(nh2, sto3g)
    with pytest.raises(
        ValueError, match="trah_trigger is not supported with a Fock hook"
    ):
        ks("uks")(nh2, sto3g, fock_hook=hook, trah_trigger=0.5)


# --- what stays refused ---------------------------------------------------------

# The refusal table of `ScfEntryPoint::refused`, the one source (checked
# against every signature in test_scf_entry_point_surface.py).
REFUSED = {
    fn: [name for names, _ in knobs for name in names]
    for fn, knobs in ferric._scf_refused().items()
}
VALUES = {
    "diis": "adiis",
    "smearing_sigma": 0.01,
    "aurora": True,
    "symmetry": True,
    "irrep_nelec": {"A1": 2},
    "stability_descent": True,
    "trah_trigger": 0.1,
    "mom_reference": "previous",
    "soscf": True,
    "dispersion": "d3bj",
    "functional": "PBE",
    "grid_radial": 50,
    "grid_angular": 110,
    "grid_prune": "none",
    "grid_weight_cut": 1e-14,
}


@pytest.mark.parametrize(
    "fn,kwarg",
    [(f, k) for f, ks_ in REFUSED.items() for k in ks_],
    ids=lambda x: str(x),
)
def test_knobs_the_solver_cannot_honour_stay_refused(fn, kwarg, water, nh2, sto3g):
    run = getattr(ferric, fn)
    args = (water, sto3g) if fn in ("run_rhf", "run_dft") else (nh2, sto3g)
    if fn in ("run_uks", "run_roks"):
        args = (*args, "PBE")
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{kwarg}'"):
        run(*args, **{kwarg: VALUES[kwarg]})
