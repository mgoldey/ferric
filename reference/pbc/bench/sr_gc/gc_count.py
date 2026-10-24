"""Per-unit SR3 triplet counts for dry ice (ferric's exact rule, pbc_gdf_split.count_sr3_ferric),
then a cost model t = t0 + tp * N_prim(nonzero) per libint call (pbc_triplet_bench, cc-pVDZ, 1 thread).

Writes units_<cell>_<w><tag>.npz and masks_<cell>_<w><tag>.pkl to the CURRENT directory (gc_model.py and
gc_variants.py read and write there too). Run it from reference/pbc/bench/out/, which is gitignored:
  mkdir -p reference/pbc/bench/out && cd reference/pbc/bench/out && python ../sr_gc/gc_count.py dryice 1.0"""

import inspect
import os
import pickle
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.dirname(HERE)
)  # reference/pbc/bench: cell_facts, pyscf_bench
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "..", "..", "prototypes", "pbc")
)  # pbc_gdf_split
import pbc_gdf_split as g
import cell_facts as cf

src = inspect.getsource(g.count_sr3_ferric)


def patch(src, old, new):
    """Source patch that FAILS LOUDLY when pbc_gdf_split.count_sr3_ferric's text has moved on."""
    assert src.count(old) == 1, f"count_sr3_ferric changed; cannot patch {old[:60]!r}"
    return src.replace(old, new)


src = patch(src, "def count_sr3_ferric(", "def count_units(")
src = patch(
    src, "    n3 = 0\n", "    n3 = 0\n    CNT = np.zeros((len(units), len(aux)))\n"
)
src = patch(
    src,
    '                n3 += int(np.searchsorted(dist, rk[ok], side="right").sum())',
    '                c_ = np.searchsorted(dist, rk[ok], side="right")\n'
    "                n3 += int(c_.sum())\n"
    "                UU, KS = np.meshgrid(us, np.array(ks), indexing='ij')\n"
    "                np.add.at(CNT, (UU[ok], KS[ok]), c_)",
)
src = patch(
    src,
    "    out = dict(\n        n_sr3=n3,",
    "    out = dict(\n        CNT=CNT, UNITS=[(u[0], u[1]) for u in units], MASKS=UNIT_MASKS, n_sr3=n3,",
)
src = patch(
    src,
    "                units.append((i, j, kk, cc, p[pm].min(), p[pm].max()))",
    "                units.append((i, j, kk, cc, p[pm].min(), p[pm].max()))\n"
    "                UNIT_MASKS.append((masks.index(pm) if False else len([u for u in units if u[0]==i and u[1]==j])-1, pm.copy()))",
)
src = patch(
    src,
    "    units = []  # (i, j, kk, cc, pmin, pmax)",
    "    units = []  # (i, j, kk, cc, pmin, pmax)\n    UNIT_MASKS = []",
)
ns = dict(g.__dict__)
exec(src, ns)  # nosec B102 -- src is this repo's own pbc_gdf_split.count_sr3_ferric, patched above
count_units = ns["count_units"]

cell = sys.argv[1] if len(sys.argv) > 1 else "dryice"
w = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
sym, xyz, lat = cf.read_cell(cell)
tag = sys.argv[3] if len(sys.argv) > 3 else ""
obs = (
    pickle.load(open(f"obs_{cell}{tag}.pkl", "rb"))  # nosec B301 -- written by gc_variants.py in this CWD
    if tag
    else cf.ferric_shells("cc-pvdz", sym, xyz)
)
aux = cf.ferric_shells("cc-pvdz-ri", sym, xyz)
crit = g.Criterion(w, 1.0)
t = time.time()
r = count_units(
    obs, aux, lat, w=w, prec=1e-13, crit=crit, mode="twocall", return_sr2=False
)
print("n_sr3", r["n_sr3"], "elapsed", time.time() - t)
np.savez(
    f"units_{cell}_{w}{tag}.npz",
    CNT=r["CNT"],
    UNITS=np.array(r["UNITS"]),
    CALL=np.array([m[0] for m in r["MASKS"]]),
)
pickle.dump(
    dict(masks=[m[1] for m in r["MASKS"]]), open(f"masks_{cell}_{w}{tag}.pkl", "wb")
)
