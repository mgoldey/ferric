import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.dirname(HERE)
)  # reference/pbc/bench: cell_facts, pyscf_bench
import cell_facts as cf

DF = {0: 1.0, 1: 1.0, 2: 3.0}


def renorm(e, k, l):
    p = e[:, None] + e[None, :]
    sab = (math.pi / p) ** 1.5 * DF[l] / (2 * p) ** l
    return k / math.sqrt(np.einsum("i,j,ij->", k, k, sab))


def drop_zero(s):
    m = np.abs(s["k"]) > 0
    return dict(s, e=s["e"][m], k=s["k"][m], amin=s["e"][m].min(), amax=s["e"][m].max())


cell = sys.argv[1]
sym, xyz, lat = cf.read_cell(cell)
obs = cf.ferric_shells("cc-pvdz", sym, xyz)
pickle.dump([drop_zero(s) for s in obs], open(f"obs_{cell}_nz.pkl", "wb"))
# rotation: per (atom, l) group with a single-primitive column, remove that primitive from the other columns
rot = []
i = 0
while i < len(obs):
    grp = [obs[i]]
    j = i + 1
    while (
        j < len(obs)
        and obs[j]["l"] == obs[i]["l"]
        and np.allclose(obs[j]["c"], obs[i]["c"])
        and np.array_equal(obs[j]["e"], obs[i]["e"])
    ):
        grp.append(obs[j])
        j += 1
    singles = [g for g in grp if (np.abs(g["k"]) > 0).sum() == 1]
    out = []
    for g in grp:
        k = g["k"].copy()
        if (np.abs(k) > 0).sum() > 1:
            for s1 in singles:
                q = int(np.argmax(np.abs(s1["k"])))
                k = k - k[q] / s1["k"][q] * s1["k"]
                k[q] = 0.0
            k = renorm(g["e"], k, g["l"])
        out.append(drop_zero(dict(g, k=k)))
    rot += out
    i = j
for s in rot[:6]:
    print(s["l"], len(s["e"]))
pickle.dump(rot, open(f"obs_{cell}_rot.pkl", "wb"))
