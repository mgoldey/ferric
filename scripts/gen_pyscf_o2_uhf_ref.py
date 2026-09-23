"""
Generate the PySCF UHF reference for O2 triplet / STO-3G.

Geometry matches `testdata/molecules/o2.xyz` and `o2_sto-3g_rohf.json`
(R = 1.208 A along z). Consumed by `crates/ferric-scf/tests/uhf_o2_sto3g.rs`.

O2/STO-3G UHF has (at least) THREE stationary points, and which one an SCF
returns depends on the guess. All three are recorded, because the test needs
to tell them apart:

* `energy_hcore_guess`   -- what PySCF reaches from init_guess = hcore / 1e /
  huckel. A saddle 0.255 Ha ABOVE the minimum (and above ROHF). This is the
  state ferric returned before it honoured the MINAO guess for UHF (#83).
* `energy_default_guess` -- what PySCF reaches from init_guess = minao / atom.
  Also a SADDLE of the UHF orbital Hessian (doubly degenerate lambda_min < 0):
  PySCF's own `stability()` reports it internally unstable.
* `energy`               -- the stable UHF minimum, reached by following the
  internal instability (`mf.stability()`) until PySCF reports stable. It
  breaks the x/y (pi) equivalence of the SPIN density on each atom (2px/2py
  spin populations 0.561/0.439, mirrored on the other atom) while keeping the
  total density centrosymmetric. 1.33 mHa below `energy_default_guess`.

At 6-31G and cc-pVDZ the default-guess solution is already stable (checked
here too), which is why this trap is STO-3G specific among the three bases.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf"))
)  # local checkout

from pyscf import gto, scf
from pyscf.soscf import newton_ah

os.makedirs("testdata/reference", exist_ok=True)

ATOM = "O 0 0 0; O 0 0 1.208"


def lowest_hessian_eig(mf) -> float:
    """Lowest eigenvalue of PySCF's UHF orbital Hessian (dense, small basis)."""
    _, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
    h = np.array([hop(e) for e in np.eye(hdiag.size)])
    return float(np.linalg.eigvalsh(0.5 * (h + h.T))[0])


def converge(mol, guess: str):
    mf = scf.UHF(mol)
    mf.init_guess = guess
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


def follow_to_stable(mf, max_rounds: int = 10):
    rounds = 0
    for _ in range(max_rounds):
        mo, _, stable, _ = mf.stability(return_status=True)
        if stable:
            return mf, rounds, True
        mf.kernel(mf.make_rdm1(mo, mf.mo_occ))
        rounds += 1
    return mf, rounds, False


mol = gto.M(atom=ATOM, basis="sto-3g", unit="angstrom", charge=0, spin=2, verbose=0)

hcore = converge(mol, "hcore")
e_hcore = float(hcore.e_tot)
lmin_hcore = lowest_hessian_eig(hcore)

default = converge(mol, "minao")
e_default = float(default.e_tot)
lmin_default = lowest_hessian_eig(default)

stable, rounds, is_stable = follow_to_stable(converge(mol, "minao"))
assert is_stable, "stability following did not terminate at a stable solution"
# The broken-symmetry minimum has a (near-)zero Hessian mode -- the x/y
# rotation of the spin polarization is a continuous degeneracy -- so plain
# DIIS may stop short of conv_tol_grad. Polish with second-order SCF so the
# recorded energy comes from a CONVERGED run.
polish = scf.UHF(mol).newton()
polish.conv_tol = 1e-12
polish.conv_tol_grad = 1e-8
polish.max_cycle = 200
polish.kernel(stable.mo_coeff, stable.mo_occ)
assert polish.converged, "Newton polish of the stable UHF minimum did not converge"
assert abs(polish.e_tot - stable.e_tot) < 1e-8, (polish.e_tot, stable.e_tot)
stable = polish
ss, mult = stable.spin_square()
lmin_stable = lowest_hessian_eig(stable)

# Same ROHF number as o2_sto-3g_rohf.json, recomputed so the variational
# ordering is visible in one file.
rohf = scf.ROHF(mol)
rohf.conv_tol = 1e-12
rohf.kernel()

d = {
    "energy": float(stable.e_tot),
    "converged": bool(stable.converged),
    "stability_rounds_from_default": rounds,
    "lowest_hessian_eigenvalue": lmin_stable,
    "energy_default_guess": e_default,
    "lowest_hessian_eigenvalue_default_guess": lmin_default,
    "energy_hcore_guess": e_hcore,
    "lowest_hessian_eigenvalue_hcore_guess": lmin_hcore,
    "energy_rohf": float(rohf.e_tot),
    "nelec_a": int(mol.nelec[0]),
    "nelec_b": int(mol.nelec[1]),
    "s_squared": float(ss),
    "multiplicity": float(mult),
    "nuclear_repulsion": float(mol.energy_nuc()),
    "basis": "sto-3g",
    "method": "uhf",
    "molecule": "O2",
    "note": "energy = stable UHF minimum; energy_default_guess and "
    "energy_hcore_guess are internally-unstable UHF saddles. "
    "Hessian eigenvalues in PySCF gen_g_hop_uhf units.",
}
for k in ("energy_hcore_guess", "energy_default_guess", "energy", "energy_rohf"):
    print(f"{k:22s} {d[k]:.10f}")
print(
    f"lambda_min hcore/default/stable = {lmin_hcore:+.4e} / "
    f"{lmin_default:+.4e} / {lmin_stable:+.4e}"
)

# Record (not assert) that the larger bases have no such trap.
for basis in ("6-31g", "cc-pvdz"):
    m = gto.M(atom=ATOM, basis=basis, spin=2, verbose=0)
    mf = converge(m, "minao")
    e0 = mf.e_tot
    mf, r, ok = follow_to_stable(mf)
    print(f"{basis:8s} default {e0:.10f}  after {r} follow rounds {mf.e_tot:.10f}")

with open("testdata/reference/o2_sto-3g_uhf.json", "w") as f:
    json.dump(d, f, indent=2)
print("Wrote testdata/reference/o2_sto-3g_uhf.json")
