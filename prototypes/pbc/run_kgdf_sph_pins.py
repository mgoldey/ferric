"""Pins for pbc_krsgdf.rs with SPHERICAL aux (ferric's bundled aux sets are pure; libint2's
2-centre ERIs refuse Cartesian shells with l > 1).  H2/STO-3G a=4, Gamma-centred 1x1x2."""

import sys

sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import ferric_basis  # noqa: E402
from run_kpts_anchor import H2_A, H2_ATOMS  # noqa: E402

cell = Cell(H2_A, H2_ATOMS, "sto-3g")
n = (1, 1, 2)
kb = PK.build_k(cell, n, exxdiv="ewald")
for auxname in sys.argv[2:] or ("cc-pvdz-ri", "def2-universal-jkfit"):
    aux = ferric_basis(auxname, [s for s, _ in H2_ATOMS])
    kg = KG.build_kgdf(cell, n, aux, spherical=True)
    kept = [kg["info"]["per_q"][i]["kept"] for i in range(kg["Nk"])]
    out = []
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        out.append(PK.krhf(kb, 2, conv=1e-12, kshift=vm, jk=KG.jk_from_kB(kg))[0])
    print(f"{auxname}: kept/q {kept} none {out[0]!r} ewald {out[1]!r}", flush=True)
