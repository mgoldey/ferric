"""Spike smoke test for macOS/Windows wheels (spike/wheels-mac-win only).

    python wheel_spike_smoke.py [E_LINUX]        # core: water + H2 RHF/STO-3G
    python wheel_spike_smoke.py --case NAME      # one broad case, own process

Water/STO-3G RHF on the repo's testdata geometry, plus H2/STO-3G, printed to
12 decimals so the value can be compared with the Linux wheel's. Each broad
case runs in its own process (see the workflow) so a native abort in one
(e.g. a libint2 assert) cannot hide the others' results.
"""

import platform
import sys

import ferric

WATER = (
    "3\nwater\n"
    "O   0.000000   0.000000   0.117790\n"
    "H   0.000000   0.755453  -0.471161\n"
    "H   0.000000  -0.755453  -0.471161\n"
)


def water():
    return ferric.Molecule.from_xyz_string(WATER, charge=0, multiplicity=1)


def energy(r):
    for name in ("total_energy", "energy", "e_total"):
        if hasattr(r, name):
            return f"{getattr(r, name):.10f}"
    return repr(r)


CASES = {
    "dft-pbe/sto-3g": lambda: energy(
        ferric.run_dft(water(), ferric.BasisSet.bundled("sto-3g"), functional="PBE")
    ),
    "rimp2/cc-pvdz": lambda: energy(
        ferric.run_rimp2(
            water(), ferric.BasisSet.bundled("cc-pvdz"), ferric.BasisSet.bundled("cc-pvdz-ri")
        )
    ),
    # Discriminators for the Linux vs macOS/Windows PBE offset (3.2e-7 Ha):
    # RI-J storage precision, grid pruning, and a functional with no gradient.
    "pbe-rij64/sto-3g": lambda: energy(
        ferric.run_dft(
            water(), ferric.BasisSet.bundled("sto-3g"), functional="PBE", ri_j_precision="f64"
        )
    ),
    "pbe-flatgrid/sto-3g": lambda: energy(
        ferric.run_dft(
            water(), ferric.BasisSet.bundled("sto-3g"), functional="PBE", grid_prune="none"
        )
    ),
    "lda/sto-3g": lambda: energy(
        ferric.run_dft(water(), ferric.BasisSet.bundled("sto-3g"), functional="LDA")
    ),
    "pbe-grad/sto-3g": lambda: str(
        ferric.run_dft(
            water(), ferric.BasisSet.bundled("sto-3g"), functional="PBE", with_gradient=True
        ).gradient()
    ).replace("\n", " "),
    "freq-fd/sto-3g": lambda: ferric.run_frequencies(water(), "sto-3g", hessian="fd").hessian_source,
    "freq/sto-3g": lambda: ferric.run_frequencies(water(), "sto-3g").hessian_source,
}

if "--case" in sys.argv:
    name = sys.argv[sys.argv.index("--case") + 1]
    print(f"BROAD {name} START", flush=True)
    print(f"BROAD {name} OK {CASES[name]()}", flush=True)
    sys.exit(0)

if "--list" in sys.argv:
    print("\n".join(CASES))
    sys.exit(0)

print("python", sys.version.split()[0], platform.platform(), platform.machine(), flush=True)
print("ferric", getattr(ferric, "__version__", "?"), getattr(ferric, "__build__", "?"), flush=True)
res = ferric.run_rhf(water(), ferric.BasisSet.bundled("sto-3g"))
print(f"WATER_RHF_STO3G {res.energy:.12f} converged={res.converged} iters={res.iterations}", flush=True)
h2 = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n", charge=0, multiplicity=1)
r2 = ferric.run_rhf(h2, ferric.BasisSet.bundled("sto-3g"))
print(f"H2_RHF_STO3G {r2.energy:.12f} converged={r2.converged}", flush=True)
if len(sys.argv) > 1:
    ref = float(sys.argv[1])
    diff = abs(res.energy - ref)
    print(f"|E - E_linux| = {diff:.3e}", flush=True)
    assert res.converged and diff < 1e-7, "water energy disagrees with the Linux reference"
print("SPIKE SMOKE PASSED", flush=True)
