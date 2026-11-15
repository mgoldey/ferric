"""PySCF side of the matched direct J/K timing (`direct_jk_profile` harness).

    OMP_NUM_THREADS=12 python scripts/direct_jk_pyscf.py <xyz> <basis> <reps> \
        <density_out.f64> [direct_scf_tol] [jk_out_prefix]

Builds the PySCF molecule from ferric's own basis JSON and geometry
(`validation/common.build_pyscf_mol`, so both codes integrate the same
segmented shells), takes PySCF's MINAO initial guess, writes it in ferric's AO
order as raw little-endian f64 (`density_out`) for the Rust harness, and times
`reps` full `get_jk` builds on it with `direct_scf_tol` (default 1e-12) and
`max_memory = 500` MB, so no ERI tensor is cached (`_eri` stays None).

Build 1 also builds the direct-SCF optimizer (`init_direct_scf`: Schwarz
`q_cond`); the script times that separately. Each rep's `get_jk` sets the
density-screen table (`dm_cond`) and runs the 8-fold direct driver. With
`jk_out_prefix`, J and K (ferric AO order) are written for a cross-code check.
Prints one `RESULT` line per rep.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402


def main() -> int:
    from pyscf import lib, scf

    xyz, basis_name, reps, dout = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
    )
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-12
    jk_prefix = sys.argv[6] if len(sys.argv) > 6 else None

    mol = common.build_pyscf_mol(Path(xyz), basis_name)
    symbols, _ = common.read_xyz(Path(xyz))
    perm = np.asarray(gp.ferric_ao_permutation(mol, basis_name, symbols))

    mf = scf.RHF(mol)
    mf.direct_scf_tol = tol
    mf.max_memory = 500
    dm = mf.get_init_guess(key="minao")
    dm_f = np.ascontiguousarray(dm[np.ix_(perm, perm)])
    dm_f.astype("<f8").tofile(dout)

    t = time.perf_counter()
    mf._opt[None] = mf.init_direct_scf(mol)
    t_opt = time.perf_counter() - t
    print(
        f"SETUP nbf={mol.nao_nr()} nsh={mol.nbas} threads={lib.num_threads()} "
        f"init_direct_scf_s={t_opt:.3f} pyscf={__import__('pyscf').__version__}"
    )
    for rep in range(1, reps + 1):
        assert mf._eri is None
        t = time.perf_counter()
        vj, vk = mf.get_jk(mol, dm)
        wall = time.perf_counter() - t
        e2 = float(np.sum(dm * (vj - 0.5 * vk)))
        print(
            f"RESULT rep={rep} wall_s={wall:.4f} tr_DJ={np.sum(dm * vj):.15e} "
            f"tr_DK={np.sum(dm * vk):.15e} e2={e2:.15e}"
        )
    if jk_prefix:
        np.ascontiguousarray(vj[np.ix_(perm, perm)]).astype("<f8").tofile(
            f"{jk_prefix}.J.f64"
        )
        np.ascontiguousarray(vk[np.ix_(perm, perm)]).astype("<f8").tofile(
            f"{jk_prefix}.K.f64"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
