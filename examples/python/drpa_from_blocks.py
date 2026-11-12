"""Direct RPA correlation energy by imaginary-frequency quadrature on ferric's
density-fitted MO tensors.

The PySCF ``gw.rpa.RPA`` pattern (Eshuis, Yarkony, Furche, JCP 132, 234114
(2010)). With fitted B^P_ia from ``AoIntegrals.df_mo(aux, C_occ, C_vir)`` and
Delta_ia = e_a - e_i on a closed-shell reference,

    Q_PQ(w) = 4 sum_ia B^P_ia B^Q_ia Delta_ia / (w^2 + Delta_ia^2),
    E_c = 1/(2 pi) int_0^inf dw ( ln det[1 + Q(w)] - tr Q(w) ).

The integral runs on Gauss-Legendre nodes t mapped to [0, inf) by
w = x0 (1 + t) / (1 - t) (PySCF's default nw = 40, x0 = 0.5). All electrons
are correlated. Hartree.

Run as a script: water/cc-pVDZ with cc-pVDZ-RI. E_c on 120 nodes equals the
exact dRPA of ``ferric.run_drpa`` (drCCD Riccati solve, no quadrature) to
1e-8 Ha; E_c on 40 nodes equals PySCF ``RPA`` on the same nodes to 1e-8 Ha
(reference numbers from scripts/gen_pyscf_block_ports_refs.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER, rhf  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_pyscf.json"


def legendre_frequencies(n: int, x0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the mapped Gauss-Legendre rule on [0, inf)."""
    t, w = np.polynomial.legendre.leggauss(n)
    return x0 * (1.0 + t) / (1.0 - t), w * 2.0 * x0 / (1.0 - t) ** 2


def drpa_corr(
    ints: ferric.AoIntegrals,
    aux: ferric.BasisSet,
    mf: dict,
    n_freq: int = 40,
    x0: float = 0.5,
) -> float:
    """dRPA correlation energy (Hartree) on the canonical RHF orbitals ``mf``."""
    nocc = int(np.count_nonzero(mf["mo_occ"]))
    c, e = mf["mo_coeff"], mf["mo_energy"]
    b = ints.df_mo(aux, c[:, :nocc], c[:, nocc:])
    b = b.reshape(b.shape[0], -1)
    delta = (e[None, nocc:] - e[:nocc, None]).ravel()
    e_corr = 0.0
    for w, wt in zip(*legendre_frequencies(n_freq, x0)):
        q = 4.0 * (b * (delta / (w * w + delta * delta))) @ b.T
        sign, logdet = np.linalg.slogdet(np.eye(len(q)) + q)
        assert sign > 0.0
        e_corr += wt / (2.0 * np.pi) * (logdet - np.trace(q))
    return float(e_corr)


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    ints = ferric.AoIntegrals(mol, bs)
    mf = rhf(ints)
    e_120 = drpa_corr(ints, aux, mf, n_freq=120)
    ref = ferric.run_drpa(mol, bs, aux, scf_energy_conv=1e-11, scf_density_conv=1e-9)[
        "e_corr"
    ]
    print(f"numpy dRPA (120 nodes) {e_120:.10f}")
    print(f"run_drpa (exact)       {ref:.10f}   diff {e_120 - ref:.1e}")
    assert abs(e_120 - ref) < 1e-8
    pyscf_ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["drpa"]
    e_40 = drpa_corr(ints, aux, mf, n_freq=pyscf_ref["nw"], x0=pyscf_ref["x0"])
    print(f"numpy dRPA (40 nodes)  {e_40:.10f}")
    print(f"PySCF RPA (40 nodes)   {pyscf_ref['e_corr']:.10f}")
    assert abs(e_40 - pyscf_ref["e_corr"]) < 1e-8


if __name__ == "__main__":
    main()
