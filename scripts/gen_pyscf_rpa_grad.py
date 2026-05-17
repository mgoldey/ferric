"""
Generate PySCF RI-RPA nuclear-force reference values for ferric.

PySCF does NOT ship an analytical RI-RPA gradient (only the non-RI TDA/RPA
excited-state gradient and the canonical MP2 gradient).  We therefore fall
back to **central finite differences of the converged RI-RPA total energy**
in PySCF at h = 5e-4 Bohr.  This produces a reference gradient with
O(h^2) FD error (~1e-7 Ha/Bohr for smooth E(R)), well below the 5e-5
acceptance threshold for the ferric analytical gradient.

The reference is total-energy-derived: dE_tot/dR = dE_HF/dR + dE_corr/dR.
Ferric's `rpa_correlation_gradient` is also a total-energy gradient (it
re-runs RHF + RPA at displaced geometries), so the comparison is
apples-to-apples.

Critical pattern: use `RPA(mf).with_df = df.DF(mol, auxbasis=...)` (NOT
`density_fit(auxbasis=...)`), per `scripts/pyscf_rpa_ref.py` — the latter
computes a different quantity in PySCF.

Output: testdata/reference/h2o_cc-pvdz_rpa_grad.json (and cc-pvtz variant).

Run: OPENBLAS_NUM_THREADS=1 python3 scripts/gen_pyscf_rpa_grad.py
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/matt/qc/pyscf")  # local checkout

import numpy as np
from pyscf import df, gto, scf
from pyscf.gw.rpa import RPA

BOHR = 0.529177210903  # Å per Bohr

os.makedirs("testdata/reference", exist_ok=True)


def total_energy(atom_coords, atom_symbols, basis, aux):
    """Compute RHF + RI-RPA total energy at the given (Bohr) coordinates."""
    atom = [(sym, tuple(c)) for sym, c in zip(atom_symbols, atom_coords)]
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.run()
    rpa = RPA(mf)
    rpa.with_df = df.DF(mol, auxbasis=aux)
    rpa.kernel()
    return float(mf.e_tot + rpa.e_corr)


def fd_gradient(atom_coords_bohr, atom_symbols, basis, aux, h=5e-4):
    """Central finite-difference gradient at h Bohr."""
    coords = np.array(atom_coords_bohr, dtype=float)
    natoms = coords.shape[0]
    grad = np.zeros((natoms, 3))
    for a in range(natoms):
        for c in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, c] += h
            cm[a, c] -= h
            ep = total_energy(cp, atom_symbols, basis, aux)
            em = total_energy(cm, atom_symbols, basis, aux)
            grad[a, c] = (ep - em) / (2 * h)
            print(
                f"  atom {a} coord {c}: E+={ep:.10f} E-={em:.10f}  F={-grad[a,c]:+.6e}",
                flush=True,
            )
    return grad


# H2O geometry from testdata: oxygen at origin, hydrogens symmetric in YZ plane.
# Coordinates in the input XYZ file are in Angstroms; convert to Bohr.
xyz_ang = np.array(
    [
        [0.000000, 0.000000, 0.117790],
        [0.000000, 0.755453, -0.471161],
        [0.000000, -0.755453, -0.471161],
    ]
)
xyz_bohr = xyz_ang / BOHR
symbols = ["O", "H", "H"]

cases = [
    ("cc-pvdz", "cc-pvdz-ri", "h2o_cc-pvdz_rpa_grad.json"),
    ("cc-pvtz", "cc-pvtz-ri", "h2o_cc-pvtz_rpa_grad.json"),
]

for basis, aux, fname in cases:
    print(f"\n=== H2O / {basis} (aux={aux}) — central FD at h=5e-4 Bohr ===")
    t0 = time.time()
    try:
        grad = fd_gradient(xyz_bohr, symbols, basis, aux, h=5e-4)
    except Exception as e:
        print(f"FAILED: {e}")
        continue
    dt = time.time() - t0
    print(f"\nGradient (Ha/Bohr):")
    for a, (sym, g) in enumerate(zip(symbols, grad)):
        print(f"  {a:2d} {sym:2s} {g[0]:+.10f} {g[1]:+.10f} {g[2]:+.10f}")
    print(f"Elapsed: {dt:.1f} s")

    out = {
        "molecule": "H2O",
        "basis": basis,
        "aux": aux,
        "method": "ri-rpa",
        "gradient_method": "central-FD (h=5e-4 Bohr) of PySCF RHF+RI-RPA total energy",
        "fd_step_bohr": 5e-4,
        "geometry_unit": "Bohr",
        "atoms": symbols,
        "coords_bohr": xyz_bohr.tolist(),
        "gradient_ha_per_bohr": grad.tolist(),
    }
    path = f"testdata/reference/{fname}"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")

print("\nDone.")
