import numpy as np, ecp_clean as E, quad as Q

FULL = list(E.ECP_I)
E.ECP_I[:] = [
    (ll, ts) if ll == 0 else (ll, [(n, z, 0.0) for n, z, d in ts]) for ll, ts in FULL
]
pc = E.I0 + np.array([1.5, -1.2, 0.8])
args = (E.I_SH[2], E.I0, E.I_SH[0], E.I0 + E.LV, pc)
for na, nr in ((590, 120), (1202, 160), (2030, 200), (3074, 240)):
    print(na, nr, "%.15e" % Q.quad_value(*args, na, nr)[0, 0], flush=True)
# PySCF with s-projector-only custom ECP
from pyscf import gto


def py(sa, pa, sb, pb, pc):
    bas = lambda sh: [[sh[0]] + [[x, y] for x, y in zip(sh[1], sh[2])] + [[1e-3, 0.0]]]
    ecp = {
        "I": [
            46,
            [
                [-1, [[], [], [[1.0, 0.0]], [], [], [], []]],
                [
                    0,
                    [
                        [[127.9202670, 2.9380036]],
                        [[78.6211465, 41.2471267]],
                        [
                            [36.5146237, 287.8680095],
                            [9.9065681, 114.3758506],
                            [1.9420086, 37.6547714],
                        ],
                        [],
                        [],
                        [],
                        [],
                    ],
                ],
            ],
        ]
    }
    m = gto.M(
        atom=[("H@1", pa), ("H@2", pb), ("I", pc)],
        basis={"H@1": bas(sa), "H@2": bas(sb), "I": "lanl2dz"},
        ecp=ecp,
        unit="B",
        verbose=0,
        spin=1,
    )
    return m.intor("ECPscalar_sph", shls_slice=(0, 1, 1, 2))


print("pyscf %.15e" % py(*args)[0, 0], "lib %.15e" % E.lib_value(*args)[0, 0])
