import sys

import numpy as np
from pyscf import gto, scf

heavy = sys.argv[1]
use_ecp = sys.argv[2] == "ecp"


def build(x):
    return gto.M(
        atom=[(heavy, (0, 0, 0)), ("H", (0.1, 0.2, 1.65 + x))],
        basis="def2-svp",
        ecp="def2-svp" if use_ecp else None,
        verbose=0,
        unit="Bohr",
    )


def run(x):
    m = build(x)
    mf = scf.RHF(m)
    mf.conv_tol = 1e-13
    mf.kernel()
    return mf


mf = run(0)
H = mf.Hessian().kernel()
h = 1e-3
fd = (run(h).nuc_grad_method().kernel() - run(-h).nuc_grad_method().kernel()) / (2 * h)
an = H[:, 1, :, 2]
print(
    heavy,
    "ecp" if use_ecp else "noecp",
    "max|an-fd|:",
    np.abs(an - fd).max(),
    "max|H|:",
    np.abs(H).max(),
)
print("an", an.ravel())
print("fd", fd.ravel())
