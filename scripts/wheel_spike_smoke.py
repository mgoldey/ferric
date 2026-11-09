"""Spike smoke test for macOS/Windows wheels (spike/wheels-mac-win only).

Water/STO-3G RHF on the repo's testdata geometry, plus H2/STO-3G, printed to
12 decimals so the value can be compared with the Linux wheel's.
"""

import platform
import sys

import ferric

print("python", sys.version.split()[0], platform.platform(), platform.machine())
print("ferric", getattr(ferric, "__version__", "?"), getattr(ferric, "__build__", "?"))

water = ferric.Molecule.from_xyz_string(
    "3\nwater\n"
    "O   0.000000   0.000000   0.117790\n"
    "H   0.000000   0.755453  -0.471161\n"
    "H   0.000000  -0.755453  -0.471161\n",
    charge=0,
    multiplicity=1,
)
res = ferric.run_rhf(water, ferric.BasisSet.bundled("sto-3g"))
print(f"WATER_RHF_STO3G {res.energy:.12f} converged={res.converged} iters={res.iterations}")

h2 = ferric.Molecule.from_xyz_string("2\nH2\nH 0 0 0\nH 0 0 0.74\n", charge=0, multiplicity=1)
r2 = ferric.run_rhf(h2, ferric.BasisSet.bundled("sto-3g"))
print(f"H2_RHF_STO3G {r2.energy:.12f} converged={r2.converged}")

if len(sys.argv) > 1:
    ref = float(sys.argv[1])
    diff = abs(res.energy - ref)
    print(f"|E - E_linux| = {diff:.3e}")
    assert res.converged and diff < 1e-7, "water energy disagrees with the Linux reference"
print("SPIKE SMOKE PASSED")

if "--broad" in sys.argv:
    import traceback

    def energy(r):
        for name in ("total_energy", "energy", "e_total"):
            if hasattr(r, name):
                return f"{getattr(r, name):.10f}"
        return repr(r)

    def attempt(name, fn):
        try:
            print(f"BROAD {name} OK {fn()}")
        except Exception:
            print(f"BROAD {name} FAILED")
            traceback.print_exc()

    bs = ferric.BasisSet.bundled
    attempt("dft-pbe/sto-3g", lambda: energy(ferric.run_dft(water, bs("sto-3g"), functional="PBE")))
    attempt("rimp2/cc-pvdz", lambda: energy(ferric.run_rimp2(water, bs("cc-pvdz"), bs("cc-pvdz-ri"))))
    attempt("freq/sto-3g", lambda: ferric.run_frequencies(water, "sto-3g").hessian_source)
