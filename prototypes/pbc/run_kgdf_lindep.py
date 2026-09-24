"""Per-q lindep: is it correctness or noise control?  (Predictions before the run.)
The duplicated-aux anchor (run_kgdf_anchor.py a) showed that an EXACTLY dependent aux set is harmless without a
cut: J3 lies in range(J2) to rounding, so the null direction carries 0/sqrt(1e-15).  The cut matters only when J2
and J3 carry INCONSISTENT noise (independently truncated SR/LR sums).  Test: H2/STO-3G, 1x1x3, diffuse
even-tempered l<=1 aux (the Iteration-2 set that dropped 3/72 at Gamma), at prec 1e-13 and a loose prec 1e-8.
PHYSICS (cut is noise control): with the per-q cut, E is stable across prec and lindep 1e-12..1e-8 to ~1e-8;
without it at q != 0 (mutant), E moves by >> 1e-6 or the SCF breaks, and the damage grows as prec loosens.
ARTIFACT: if the mutant does NOT move E, either the metric at q != 0 is well-conditioned for this aux (check
smin per q) or the noise is J2/J3-consistent; the smin print distinguishes the two."""

import sys


sys.path.insert(0, ".")
import pbc_kgdf as KG  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from pbc_gdf import even_tempered  # noqa: E402
from run_kgdf_anchor import H2_A, H2_ATOMS  # noqa: E402

cell = Cell(H2_A, H2_ATOMS, "sto-3g")
n = (1, 1, 3)
kb = PK.build_k(
    cell, n, gcut=PK.aft_gcut(cell, 1e-4), thresh=1e-8, verbose=False
)  # 1e only (h, S, enn); JK from B
aux = even_tempered(["H"], 1, 0.1, 2.2, 9)
for prec in (1e-13, 1e-8):
    for lindep, mut in (
        (1e-12, None),
        (1e-10, None),
        (1e-8, None),
        (1e-10, "no_lindep_q"),
        (1e-10, "no_herm_q0"),
    ):
        KG._MUTANT = mut
        kg = KG.build_kgdf(cell, n, aux, prec=prec, lindep=lindep)
        KG._MUTANT = None
        pq = kg["info"]["per_q"]
        try:
            e = PK.krhf(kb, 2, conv=1e-11, kshift=0.0, jk=KG.jk_from_kB(kg))[0]
            es = f"{e:.12f}"
        except Exception as exc:  # noqa: BLE001
            es = f"SCF failed ({type(exc).__name__})"
        print(
            f"prec {prec:.0e} lindep {lindep:.0e} {mut or 'per-q cut'}: E {es}; kept/q {[pq[i]['kept'] for i in range(3)]} "
            f"of {pq[0]['naux']}; smin/q {['%.1e' % pq[i]['smin'] for i in range(3)]}; "
            f"asym J3(q=0) {pq[0]['asym_J3_q0']:.1e}, asym J2/q {['%.0e' % pq[i]['asym_J2'] for i in range(3)]}",
            flush=True,
        )
