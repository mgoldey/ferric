import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.dirname(HERE)
)  # reference/pbc/bench: cell_facts, pyscf_bench
import cell_facts as cf

cell, w = sys.argv[1], float(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else ""
d = np.load(f"units_{cell}_{w}{tag}.npz")
CNT = d["CNT"]
UN = d["UNITS"]
CALL = d["CALL"]
masks = pickle.load(open(f"masks_{cell}_{w}{tag}.pkl", "rb"))[  # nosec B301 -- written by gc_count.py in this CWD
    "masks"
]
sym, xyz, lat = cf.read_cell(cell)
obs = (
    pickle.load(open(f"obs_{cell}{tag}.pkl", "rb"))  # nosec B301 -- written by gc_variants.py in this CWD
    if tag
    else cf.ferric_shells("cc-pvdz", sym, xyz)
)
aux = cf.ferric_shells("cc-pvdz-ri", sym, xyz)
# per-class overhead from pbc_triplet_bench (diamond cc-pVDZ, 1 thread): t = t0[cls] + TP*N
TP = 0.07132
bench = {}
for ln in open(os.path.join(HERE, "bench_sr3.txt")):
    p = ln.split()
    if p and p[0].startswith("["):
        cls = tuple(int(x) for x in " ".join(p[:3]).strip("[]").split(","))
        us, nm = float(p[4]), float(p[5])
        bench[cls] = us - TP * nm


def t0(la, lb, lp):
    return bench[(la, lb, lp)]


nz = [np.abs(s["k"]) > 0 for s in obs]
atom_of = {}
for i, s in enumerate(obs):
    atom_of[i] = tuple(np.round(s["c"], 8))
auxl = np.array([p["l"] for p in aux])
Nu = np.array(
    [int((m & nz[i][:, None] & nz[j][None, :]).sum()) for (i, j), m in zip(UN, masks)]
)
la = np.array([obs[i]["l"] for i, _ in UN])
lb = np.array([obs[j]["l"] for _, j in UN])
T0 = np.array(
    [[t0(la[u], lb[u], auxl[k]) for k in range(len(aux))] for u in range(len(UN))]
)
seg_calls = CNT.sum()
seg_cost = (CNT * (T0 + TP * Nu[:, None])).sum() * 1e-6
seg_prim = (CNT * Nu[:, None]).sum()
print(
    f"{cell} w={w} {tag}: calls {seg_calls:.4g}  model CPU {seg_cost:.1f} s  (overhead {(CNT * T0).sum() * 1e-6:.1f} s, prim {(CNT * TP * Nu[:, None]).sum() * 1e-6:.1f} s)  prim-triples {seg_prim:.4g}  mean N {seg_prim / seg_calls:.2f}"
)


# grouping: key = (atom_i, l_i, atom_j, l_j, call) and support-grouping key adds support pattern
def grouped(keyf, label):
    groups = {}
    for u, (i, j) in enumerate(UN):
        groups.setdefault(keyf(u, i, j), []).append(u)
    cost = 0.0
    calls = 0.0
    prim = 0.0
    for key, us in groups.items():
        cmax = CNT[us].max(axis=0)  # a group triplet is visited if any member is (~max)
        i0, j0 = UN[us[0]]
        union = np.zeros_like(masks[us[0]], bool)
        for u in us:
            i, j = UN[u]
            union |= masks[u] & nz[i][:, None] & nz[j][None, :]
        N = union.sum()
        calls += cmax.sum()
        prim += (cmax * N).sum()
        cost += (
            cmax
            * (
                np.array(
                    [t0(obs[i0]["l"], obs[j0]["l"], auxl[k]) for k in range(len(aux))]
                )
                + TP * N
            )
        ).sum()
    print(
        f"  {label}: calls {calls:.4g} ({seg_calls / calls:.2f}x fewer), prim-triples {prim:.4g} ({seg_prim / prim:.2f}x fewer), model CPU {cost * 1e-6:.1f} s ({seg_cost / (cost * 1e-6):.2f}x)"
    )


if not tag:
    grouped(
        lambda u, i, j: (atom_of[i], obs[i]["l"], atom_of[j], obs[j]["l"], CALL[u]),
        "general (atom,l) groups",
    )
    grouped(
        lambda u, i, j: (
            atom_of[i],
            obs[i]["l"],
            tuple(nz[i]),
            atom_of[j],
            obs[j]["l"],
            tuple(nz[j]),
            CALL[u],
        ),
        "shared-support groups (1s+2s)",
    )
# primitive-level decontraction with today's per-call API: one call per nonzero prim pair of every visited unit (union per group)
prim_calls = seg_prim  # upper bound: no sharing
print(
    f"  s2 symmetry: calls {seg_calls / 2:.4g}, model CPU {seg_cost / 2:.1f} s (2.00x)"
)
# class mix
by = {}
for u in range(len(UN)):
    for k in range(len(aux)):
        if CNT[u, k]:
            key = (la[u], lb[u])
            by.setdefault(key, [0, 0.0])
            by[key][0] += CNT[u, k]
            by[key][1] += CNT[u, k] * (T0[u, k] + TP * Nu[u])
print("  (l_mu,l_nu): share of calls / share of model CPU")
for key in sorted(by):
    print(
        f"   {key}: {by[key][0] / seg_calls:.3f} / {by[key][1] * 1e-6 / seg_cost:.3f}"
    )
