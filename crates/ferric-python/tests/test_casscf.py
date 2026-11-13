"""`ferric.run_casscf`: two-step CASSCF through the binding.

H2O / 6-31G CAS(4,4) above 3 core orbitals, the geometry of
`testdata/molecules/water.xyz`. The construction is anchored in
`crates/ferric-ci/tests/casscf.rs` (full-space limit, finite-difference
gradient per rotation class, converged energies); these checks cover the
binding: the energy is the Rust test's value, the returned arrays are
consistent with it, and non-convergence is reported.

Energy bar: `crates/ferric-ci/tests/casscf.rs` derives 1.1e-12 Ha against
the prototype's -76.0369690965018 from both gradient norms and the lowest
orbital Hessian eigenvalue (0.0195 Ha).
"""

import sys
import threading
import time
import warnings

import numpy as np
import pytest

import ferric

_H2O = (
    "3\n\nO 0.000000 0.000000 0.117790\n"
    "H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n"
)
E_PROTO = -76.0369690965018
E_BAR = 1.1e-12


@pytest.fixture(scope="module")
def res():
    mol = ferric.Molecule.from_xyz_string(_H2O)
    return ferric.run_casscf(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2))


def test_converged_energy(res):
    assert res.converged
    assert res.grad_norm < 1e-7
    assert abs(res.e_total - E_PROTO) < E_BAR
    assert res.e_total < res.rhf_energy


def test_arrays_are_consistent(res):
    assert res.n_core == 3 and res.n_determinants == 36
    c = res.mo_coeff
    ints = ferric.AoIntegrals(
        ferric.Molecule.from_xyz_string(_H2O), ferric.BasisSet.bundled("6-31g")
    )
    assert np.abs(c.T @ ints.overlap() @ c - np.eye(c.shape[1])).max() < 1e-10
    ci = res.ci_vector
    assert ci.shape == (6, 6) and abs(np.sum(ci * ci) - 1.0) < 1e-12
    assert abs(np.trace(res.dm1) - 4.0) < 1e-11
    assert abs(np.einsum("pprr", res.dm2) - 12.0) < 1e-11


def test_certificate(res):
    cert = res.certificate
    assert cert["capability"] == "casscf"
    assert cert["grade"] == "proven"
    assert [a["name"] for a in cert["approximations"]] == ["active-space"]
    assert cert["build"]["commit"] == ferric.build_info()["commit"]
    assert cert["warnings"] == []


def test_non_convergence_warns():
    mol = ferric.Molecule.from_xyz_string(_H2O)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = ferric.run_casscf(
            mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2), casscf_max_iter=3
        )
    assert not r.converged and r.iterations == 3
    assert any("CASSCF did not converge" in str(x.message) for x in w)
    assert any("CASSCF did not converge" in m for m in r.certificate["warnings"])
    assert r.certificate["grade"] == "ungraded"


def test_canonicalize_option(res):
    """Canonical orbitals: same energy to the Ritz bound of both CI solves
    (each residual below ci_conv_thresh, which CASSCF enforces), natural
    active orbitals (off-diagonal dm1 below 2 N (r0 + r1) / gap, 8.2e-10 for
    this system in the Rust test), ordered Fock diagonals. The bars are derived in
    crates/ferric-ci/tests/casscf_canonical.rs; this checks the binding passes
    the option and returns mo_energy."""
    assert res.mo_energy is None
    mol = ferric.Molecule.from_xyz_string(_H2O)
    r = ferric.run_casscf(
        mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2), canonicalize=True
    )
    assert r.converged
    assert abs(r.e_total - res.e_total) < 2.0 * 1e-10
    occ = np.diag(r.dm1)
    assert np.abs(r.dm1 - np.diag(occ)).max() < 8.2e-10
    assert np.all(np.diff(occ) <= 1e-12)
    eps = r.mo_energy
    assert eps.shape == (r.mo_coeff.shape[1],)
    assert np.all(np.diff(eps[:3]) >= 0.0) and np.all(np.diff(eps[7:]) >= 0.0)


def test_open_shell_active_space():
    """(n_alpha, n_beta) = (3, 1): the lowest M_s = 1 state, a triplet above
    the singlet CASSCF minimum of the same active space."""
    mol = ferric.Molecule.from_xyz_string(_H2O)
    r = ferric.run_casscf(mol, ferric.BasisSet.bundled("6-31g"), 4, (3, 1))
    assert r.converged and r.n_core == 3
    assert r.ci_vector.shape == (4, 4)
    assert abs(np.trace(r.dm1) - 4.0) < 1e-11
    assert r.e_total > E_PROTO


def test_unconverged_ci_raises():
    """CAS(8,8) has 4900 determinants, above the dense-diagonalization fallback
    (ferric_ci::DENSE_SOLVE_MAX_DETS = 1024) that rescues smaller spaces, so one
    Davidson iteration leaves the CI unconverged."""
    mol = ferric.Molecule.from_xyz_string(_H2O)
    with pytest.raises(Exception, match="ci_conv_thresh"):
        ferric.run_casscf(
            mol, ferric.BasisSet.bundled("6-31g"), 8, (4, 4), ci_max_iter=1
        )


def _longest_main_thread_stall(call):
    """Run `call` on a worker thread while this thread samples the clock in a
    busy loop; return (longest gap between samples, wall time of the call).
    A binding that holds the GIL through its Rust solve stalls this loop for
    the whole solve, so the gap is about the wall time; one that releases it
    stalls the loop only for GIL hand-offs (the 5 ms switch interval).

    The worker waits on `go`, which is set only after this thread has taken
    its first sample, so the sampler is running before the call can start.
    A measurement with no samples raises."""
    go = threading.Event()
    done = threading.Event()
    errors = []

    def work():
        go.wait()
        try:
            call()
        except BaseException as exc:  # re-raised on the main thread after join
            errors.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=work)
    t.start()
    start = last = time.perf_counter()
    samples = 1
    gap = 0.0
    go.set()
    while not done.is_set():
        now = time.perf_counter()
        gap = max(gap, now - last)
        last = now
        samples += 1
    t.join()
    end = time.perf_counter()
    if errors:
        raise errors[0]
    if samples < 2:  # the initial sample plus at least one in the loop
        raise RuntimeError("GIL measurement recorded no samples")
    return max(gap, end - last), end - start


@pytest.mark.parametrize("which", ["casci", "casscf"])
def test_solve_releases_the_gil(which):
    """The SCF and CI solves run with the GIL released: the main thread keeps
    running for all but a small part of the call. Bar: the longest stall is
    below half the call's wall time, which a GIL-holding solve (stall ~ the
    whole call minus Python argument handling) cannot meet; the call is
    required to last at least 20 switch intervals so the bar is not met by
    hand-off granularity alone."""
    mol = ferric.Molecule.from_xyz_string(_H2O)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    fn = ferric.run_casci if which == "casci" else ferric.run_casscf
    gap, wall = _longest_main_thread_stall(lambda: fn(mol, bs, 4, (2, 2)))
    assert wall > 20 * sys.getswitchinterval(), wall
    assert gap < 0.5 * wall, (gap, wall)
