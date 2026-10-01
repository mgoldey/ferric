import sys

sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import ferric_basis  # noqa: E402
from run_kpts_anchor import SP, TRI_A, TRI_ATOMS  # noqa: E402

cell = Cell(TRI_A, TRI_ATOMS, SP)
n = (1, 1, 2)
kb = PK.build_k(cell, n, exxdiv="ewald")
for auxname in ("cc-pvdz-ri", "def2-universal-jkfit"):
    aux = ferric_basis(auxname, [s for s, _ in TRI_ATOMS])
    kg = KG.build_kgdf(cell, n, aux, spherical=True)
    out = [
        PK.krhf(
            kb,
            4,
            conv=1e-12,
            kshift=(kb["madelung"] if ex == "ewald" else 0.0),
            jk=KG.jk_from_kB(kg),
        )[0]
        for ex in ("none", "ewald")
    ]
    print(f"{auxname}: none {out[0]!r} ewald {out[1]!r}", flush=True)
