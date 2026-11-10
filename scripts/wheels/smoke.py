#!/usr/bin/env python3
"""Smoke test of an INSTALLED ferric wheel (macOS / Windows release jobs).

    python smoke.py --all          # every case, each in its own process
    python smoke.py --case NAME    # one case
    python smoke.py --list

Run it from a directory that holds no ferric source tree, in a fresh venv
holding only the wheel and its declared dependencies, on a runner that never
saw the build prefix (wheels-macwin.yml's smoke job), so nothing outside the
wheel can satisfy an import or a library load.

The first cases are the Linux smoke test's (wheels.yml smoke-test: build
identity, libint2 variant, H2/STO-3G RHF + numpy return path, analytic water
Hessian). The rest compare against energies from a Linux build of the same
code (the 2026-11-09 spike matched every platform to every printed digit), to
the tolerance in CASES: the point is a platform-dependent numerics failure
(a wrong BLAS kernel, a miscompiled libint2), not a convention.

Each case runs in its own process because a native abort (e.g. a libint2
assert) kills the interpreter and would hide every later case.

Environment: EXPECT_SHA (the commit the wheel must report), GITHUB_REF (on
refs/tags/vX the wheel must report version X).
"""

from __future__ import annotations

import os
import subprocess
import sys

# The Linux smoke test's water geometry (wheels.yml smoke-test).
WATER_SMOKE = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
# testdata/molecules/water.xyz, the geometry of the reference energies below.
WATER = (
    "3\nwater\n"
    "O   0.000000   0.000000   0.117790\n"
    "H   0.000000   0.755453  -0.471161\n"
    "H   0.000000  -0.755453  -0.471161\n"
)
H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"


def mol(xyz):
    import ferric

    return ferric.Molecule.from_xyz_string(xyz, charge=0, multiplicity=1)


def basis(name):
    import ferric

    return ferric.BasisSet.bundled(name)


def case_identity():
    import ferric

    expect = {"git_sha": os.environ["EXPECT_SHA"], "dirty": False}
    assert ferric.__build__ == expect, f"build stamp {ferric.__build__!r} != {expect!r}"
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/v"):
        tag = ref[len("refs/tags/v") :]
        assert ferric.__version__ == tag, f"{ferric.__version__!r} != tag {tag!r}"
    return f"{ferric.__version__} {ferric.__build__}"


def case_variant():
    import importlib.metadata

    import ferric

    # The PyPI wheel's libint2: ferric's small 2.7.2 export, linked statically
    # (the values wheels.yml smoke-test asserts for its `pypi` variant).
    expect = {
        "libint_version": "2.7.2",
        "libint_linkage": "static",
        "libint_max_deriv_order": 2,
        "libint_max_am": {
            "eri": [6, 6, 3],
            "eri3": [6, 6],
            "eri2": [6, 6],
            "onebody": [6, 6, 6],
        },
        "libint_g12_max_am": None,
    }
    info = ferric.build_info()
    got = {k: info.get(k, "<missing>") for k in expect}
    assert got == expect, f"build_info libint report {got} != {expect}"
    wheel_meta = importlib.metadata.distribution("ferric").read_text("WHEEL") or ""
    builds = [ln for ln in wheel_meta.splitlines() if ln.startswith("Build:")]
    assert builds == [], f"WHEEL has Build lines {builds}"
    return str(got)


def case_h2():
    import numpy as np

    import ferric

    r = ferric.run_rhf(mol(H2), basis("sto-3g"))
    assert r.converged, "SCF did not converge"
    # Textbook RHF/STO-3G H2 at 0.74 A is about -1.1167 Ha.
    assert -1.30 < r.energy < -1.05, f"H2 RHF energy {r.energy} outside the basin"
    d, e = r.density(), r.orbital_energies()
    assert d.shape == (2, 2) and e.shape == (2,), (d.shape, e.shape)
    assert np.isfinite(d).all() and np.isfinite(e).all()
    return f"{r.energy:.12f}"


def case_freq_analytic():
    import ferric

    f = ferric.run_frequencies(mol(WATER_SMOKE), "sto-3g")
    assert f.hessian_source == "analytic", (
        f"expected an analytic Hessian, got {f.hessian_source!r}"
    )
    assert len(f.frequencies) == 3
    return f.hessian_source


def case_freq_fd():
    import ferric

    f = ferric.run_frequencies(mol(WATER), "sto-3g", hessian="fd")
    assert f.hessian_source == "finite-difference", f.hessian_source
    assert len(f.frequencies) == 3
    return f.hessian_source


def energy_case(run, ref, tol):
    def case():
        r = run()
        e = r.energy
        assert getattr(r, "converged", True), "not converged"
        assert abs(e - ref) < tol, (
            f"E = {e:.12f}, Linux {ref:.12f}, |dE| = {abs(e - ref):.2e} > {tol:g}"
        )
        return f"{e:.12f} (|dE| {abs(e - ref):.1e})"

    return case


def dft(**kw):
    import ferric

    return lambda: ferric.run_dft(mol(WATER), basis("sto-3g"), **kw)


def rimp2():
    import ferric

    return ferric.run_rimp2(mol(WATER), basis("cc-pvdz"), basis("cc-pvdz-ri"))


def rhf_water():
    import ferric

    return ferric.run_rhf(mol(WATER), basis("sto-3g"))


def case_pbe_gradient():
    import numpy as np

    import ferric

    g = np.asarray(
        ferric.run_dft(
            mol(WATER), basis("sto-3g"), functional="PBE", with_gradient=True
        ).gradient()
    )
    assert g.shape == (3, 3) and np.isfinite(g).all(), g
    assert abs(g.sum(axis=0)).max() < 1e-6, f"gradient not translation invariant: {g}"
    oz = g[0, 2]
    assert abs(oz - (-0.118224303)) < 1e-6, f"O z gradient {oz:.9f}, Linux -0.118224303"
    return f"O z {oz:.9f}"


def case_cli():
    """The pip-installed `ferric` console script (wheels.yml runs the same)."""
    import shutil
    import tempfile

    exe = shutil.which("ferric")
    assert exe, "no `ferric` console script on PATH"
    help_run = subprocess.run([exe, "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0, f"ferric --help exited {help_run.returncode}"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "smoke-cli.xyz"), "w") as fh:
            fh.write(H2)
        with open(os.path.join(d, "smoke-cli.toml"), "w") as fh:
            fh.write(
                '[molecule]\nxyz = "smoke-cli.xyz"\n\n[basis]\nname = "sto-3g"\n\n'
                '[method]\nkind = "rhf"\n'
            )
        run = subprocess.run(
            [exe, "smoke-cli.toml"], cwd=d, capture_output=True, text=True
        )
    print(run.stdout[-2000:], run.stderr[-2000:])
    assert run.returncode == 0, f"ferric smoke-cli.toml exited {run.returncode}"
    assert "converged  = true" in run.stdout, "CLI did not report converged = true"
    return "ok"


# name -> callable returning a short result string (raises on failure).
# Linux energies: the spike's same-code Linux build (wiki/notes/
# wheels-mac-win-2026-11-09), re-checked on main before this file landed.
CASES = {
    "identity": case_identity,
    "variant": case_variant,
    "h2": case_h2,
    "freq-analytic": case_freq_analytic,
    "cli": case_cli,
    "rhf-water": energy_case(rhf_water, -74.963146800039, 1e-8),
    "rimp2-water-ccpvdz": energy_case(rimp2, -76.2308014541, 1e-7),
    "pbe": energy_case(dft(functional="PBE"), -75.2261678018, 1e-6),
    "pbe-rij-f64": energy_case(
        dft(functional="PBE", ri_j_precision="f64"), -75.2261674870, 1e-6
    ),
    "pbe-flat-grid": energy_case(
        dft(functional="PBE", grid_prune="none"), -75.2261677983, 1e-6
    ),
    "lda": energy_case(dft(functional="LDA"), -74.7326093014, 1e-6),
    "pbe-gradient": case_pbe_gradient,
    "freq-fd": case_freq_fd,
}


def run_one(name):
    print(f"CASE {name} START", flush=True)
    print(f"CASE {name} OK {CASES[name]()}", flush=True)


def run_all():
    import platform

    import ferric

    print("python", sys.version.split()[0], platform.platform(), flush=True)
    print("ferric", ferric.__version__, ferric.__build__, flush=True)
    failed = []
    for name in CASES:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", name],
            capture_output=True,
            text=True,
            env=env,
        )
        # The MSVC CRT can write assert text as UTF-16; drop NULs so the
        # message stays readable in the log.
        out = (p.stdout + p.stderr).replace("\x00", "")
        ok = p.returncode == 0 and f"CASE {name} OK" in out
        line = next(
            (ln for ln in out.splitlines() if ln.startswith(f"CASE {name} OK")), ""
        )
        print(line if ok else f"CASE {name} FAILED (exit {p.returncode})", flush=True)
        if not ok:
            print(out[-4000:], flush=True)
            failed.append(name)
    if failed:
        print(f"SMOKE FAILED: {', '.join(failed)}", flush=True)
        return 1
    print(f"SMOKE PASSED ({len(CASES)} cases)", flush=True)
    return 0


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(CASES))
    elif "--case" in sys.argv:
        run_one(sys.argv[sys.argv.index("--case") + 1])
    elif "--all" in sys.argv:
        sys.exit(run_all())
    else:
        print(__doc__)
        sys.exit(2)
