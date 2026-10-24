import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.dirname(HERE)
)  # reference/pbc/bench: cell_facts, pyscf_bench
import numpy as np
from pyscf import gto, lib
from pyscf.df import incore
from pyscf_bench import ferric_basis_pyscf

lib.num_threads(1)
atoms = [("C", (0, 0, 0)), ("O", (0, 0, 2.19)), ("O", (0, 0, -2.19))]
syms = ["C", "O"]


def nzb(repr_):
    if repr_ == "rot":
        b = ferric_basis_pyscf("cc-pvdz", syms, "segmented")
        out = {}
        for k, v in b.items():
            new = []
            for sh in v:
                nzp = [p for p in sh[1:] if p[1] != 0.0]
                if len(nzp) > 1:
                    # drop the primitive that is a single-primitive column of the same l (the most diffuse one)
                    singles = {
                        tuple(p2[0] for p2 in s2[1:] if p2[1] != 0.0)
                        for s2 in v
                        if s2[0] == sh[0]
                        and sum(1 for p2 in s2[1:] if p2[1] != 0.0) == 1
                    }
                    drop = {x[0] for x in singles}
                    nzp = [p for p in nzp if p[0] not in drop]
                new.append([sh[0]] + nzp)
            out[k] = new
        return out
    if repr_ != "segnz":
        return ferric_basis_pyscf("cc-pvdz", syms, repr_)
    b = ferric_basis_pyscf("cc-pvdz", syms, "segmented")
    return {
        k: [[sh[0]] + [p for p in sh[1:] if p[1] != 0.0] for sh in v]
        for k, v in b.items()
    }


def mk(repr_):
    m = gto.M(atom=atoms, unit="Bohr", basis=nzb(repr_), cart=False)
    a = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=ferric_basis_pyscf("cc-pvdz-ri", syms, "segmented"),
        cart=False,
    )
    return m, a


for repr_ in ["general", "segmented", "segnz", "rot"]:
    m, a = mk(repr_)
    nctr = m._bas[:, gto.NCTR_OF]
    npr = m._bas[:, gto.NPRIM_OF]
    # drop zero primitives in segmented form for a fair comparison
    ntrip_calls = m.nbas**2 * a.nbas
    ncol = nctr.sum()
    with m.with_range_coulomb(-1.0), a.with_range_coulomb(-1.0):
        out = incore.aux_e2(m, a, "int3c2e", aosym="s1")
        best = 1e9
        for rep in range(5):
            t = time.process_time()
            for _ in range(20):
                incore.aux_e2(m, a, "int3c2e", aosym="s1")
            best = min(best, (time.process_time() - t) / 20)
    ncoltrip = ncol**2 * a.nbas
    print(
        f"{repr_:10s} nbas {m.nbas} ncol {ncol} nprim/shell {list(npr[:6])} nctr {list(nctr[:6])}: {best * 1e3:.2f} ms/eval, "
        f"{best / ntrip_calls * 1e6:.3f} us/libcint call, {best / ncoltrip * 1e6:.3f} us per column triplet"
    )
    np.save(f"cint_{repr_}.npy", out)
g, s = np.load("cint_general.npy"), np.load("cint_segnz.npy")
print("max|general-segmented|", np.abs(g - s).max(), "max|.|", np.abs(g).max())
