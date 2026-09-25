"""Stage-4 box limit (independent of PySCF pbc): Gamma UHF in a cubic box a vs PySCF MOLECULAR UHF.

usage: python3 run_uhf_box_limit.py {H|O2} a1 a2 ...   (a=0 prints only the predictions)

PREDICTIONS (written before the sweep, molecular quantities only; pbc_uhf.uhf_c3_closed_form):
  physics, exxdiv='ewald': dE = E_pbc - E_mol = c3/a^3 + O(a^-5),
      c3 = -(2pi/3)(|d|^2 + Omega_a + Omega_b), Omega_sigma = Foster-Boys invariant spread of the
      sigma-occupied space (first order in k = 4pi/3a^3, so orbital relaxation does not enter c3).
      Cross-checked by uhf_r2_kernel_c3 (harmonic kernel added to the molecular UHF, relaxed, d/dk).
  physics, exxdiv=None: dE_none - dE_ewald = +v_M (N_a + N_b)/2 exactly at Gamma (D unchanged),
      i.e. 1/a convergence with coefficient +2.8373 (N_a+N_b)/2.
  artifact, Madelung factor 1/2 per spin (RHF bookkeeping applied to D_sigma): ewald residual gains
      +v_M N/4 -> exponent 1, coefficient +0.709 N.
  artifact, missing images / normalisation: plateau at nonzero dE (exponent -> 0).
  artifact, K[D_total]: O(1) error that does not vanish with a.
These are distinguishable by construction (exponent 3 with predicted c3 vs exponent 1 vs 0).
"""

import sys
import time

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_uhf import uhf, uhf_c3_closed_form, uhf_r2_kernel_c3  # noqa: E402

SYS = {
    "H": ([("H", (0.3, 0.2, 0.1))], 1, 0),
    "O2": ([("O", (0.3, 0.2, 0.1)), ("O", (0.3, 0.2, 0.1 + 2.282))], 9, 7),
}
name = sys.argv[1]
edges = [float(x) for x in sys.argv[2:]]
atoms, na, nb = SYS[name]
mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0, spin=na - nb)
mf = scf.UHF(mol)
mf.conv_tol = 1e-13
e_mol = mf.kernel()
Ca, Cb = mf.mo_coeff
c3, parts = uhf_c3_closed_form(mol, Ca, Cb, na, nb)
c3_r2 = uhf_r2_kernel_c3(mol, na, nb, guess=mf.make_rdm1())
print(f"{name}/STO-3G: E_mol(UHF) {e_mol:.12f}  <S2>_mol {mf.spin_square()[0]:.10f}")
print(
    f"  PREDICTED ewald c3 = {c3:.6f} (closed form; Omega_a {parts['omega_a']:.6f} Omega_b {parts['omega_b']:.6f}"
    f" |d| {np.linalg.norm(parts['dipole']):.1e});  r2-kernel relaxed FD {c3_r2:.6f}"
)
print(f"  PREDICTED none - ewald = +{2.8372974794806 * (na + nb) / 2:.6f}/a")
dm_mol = mf.make_rdm1()
for edge in edges:
    t = time.time()
    w = min(1.0, 8.0 / edge)
    ints = build_integrals(
        Cell(np.eye(3) * edge, atoms, "sto-3g"),
        w,
        rcut_bra=18.0,
        rcut_2e=18.0 + 6.0 / w,
        exxdiv="ewald",
        verbose=False,
    )
    vm = ints["madelung"]
    r = {}
    for ex in ("none", "ewald"):
        r[ex] = uhf(
            ints["S"],
            ints["h"],
            ints["I"],
            ints["enn"],
            na,
            nb,
            conv=1e-12,
            kshift=vm if ex == "ewald" else 0.0,
            guess=dm_mol,
        )
    de_e, de_n = r["ewald"]["e"] - e_mol, r["none"]["e"] - e_mol
    print(
        f"  a={edge:5.1f} w={w:.3f}  dE_ewald {de_e:+.10e}  dE_none {de_n:+.10e}  "
        f"(none-ewald) - vM N/2 {de_n - de_e - vm * (na + nb) / 2:+.1e}  dE_ewald*a^3 {de_e * edge**3:+.6f}  "
        f"<S2> {r['ewald']['s2']:.8f}  it {r['ewald']['it']}  ({time.time() - t:.0f}s)",
        flush=True,
    )
