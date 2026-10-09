"""Cost model + libecpint timing on 1,3,5-triiodobenzene / def2-SVP + def2-ECP (ferric's bundled JSON data):
75 shells, 3 ECP centres (I: 4 local + 25 semi-local terms).

  python cost.py [libshim.so]
Prints: shell/triple counts; windows, nodes and screened items of the prototype (type 2 and type 1); per-(pair, centre,
l) angular contractions; |ours - libecpint| on the molecular matrix; libecpint wall time for ferric_ecp_matrix
(1 thread); and writes windows.txt for radial_kernel.cc (measured per-window cost of the hot loop in C++)."""

import ctypes as C
import json
import math
import os
import sys
import time

import numpy as np

import ecpq as E
from fixtures import bare

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "../../../crates/ferric-core/src/basis/bundled")
ANG = 1.0 / 0.52917721092


def bse_shells(z, pos, basis="def2-svp.json"):
    el = json.load(open(os.path.join(BUNDLED, basis)))["elements"][str(z)]
    out = []
    for s in el["electron_shells"]:
        ex = [float(x) for x in s["exponents"]]
        ls = s["angular_momentum"]
        for k, co in enumerate(s["coefficients"]):
            l = ls[k] if len(ls) == len(s["coefficients"]) else ls[0]
            c = [float(x) for x in co]
            keep = [i for i, v in enumerate(c) if v != 0.0]
            e2, c2 = [ex[i] for i in keep], [c[i] for i in keep]
            out.append(
                dict(
                    l=l,
                    center=np.asarray(pos, float),
                    exponents=e2,
                    coefficients=bare(l, e2, c2),
                )
            )
    return out


def bse_ecp(z, pos):
    el = json.load(open(os.path.join(BUNDLED, "def2-ecp.json")))["elements"][str(z)]
    ams, ns, ex, co = [], [], [], []
    for p in el["ecp_potentials"]:
        l = p["angular_momentum"][0]
        for n, a, c in zip(
            p["r_exponents"], p["gaussian_exponents"], p["coefficients"][0]
        ):
            ams.append(l)
            ns.append(int(n))
            ex.append(float(a))
            co.append(float(c))
    return dict(
        center=np.asarray(pos, float), ams=ams, ns=ns, exponents=ex, coefficients=co
    )


def triiodobenzene():
    atoms = []
    rc, ri, rh = 1.39 * ANG, (1.39 + 2.10) * ANG, (1.39 + 1.08) * ANG
    for k in range(6):
        t = k * math.pi / 3
        u = np.array([math.cos(t), math.sin(t), 0.0])
        atoms.append((6, rc * u))
        atoms.append((53, ri * u) if k % 2 == 0 else (1, rh * u))
    return atoms


# ------------------------------------------------------------------ libecpint via the ferric shim (ctypes)
class GS(C.Structure):
    _fields_ = [
        ("l", C.c_int),
        ("nprim", C.c_int),
        ("x", C.c_double),
        ("y", C.c_double),
        ("z", C.c_double),
        ("exponents", C.POINTER(C.c_double)),
        ("coefficients", C.POINTER(C.c_double)),
    ]


class EC(C.Structure):
    _fields_ = [
        ("x", C.c_double),
        ("y", C.c_double),
        ("z", C.c_double),
        ("nterm", C.c_int),
        ("ams", C.POINTER(C.c_int)),
        ("ns", C.POINTER(C.c_int)),
        ("exponents", C.POINTER(C.c_double)),
        ("coefficients", C.POINTER(C.c_double)),
    ]


def libecpint_matrix(lib, shells, ecps):
    keep = []
    gs = (GS * len(shells))()
    for i, s in enumerate(shells):
        ea = (C.c_double * len(s["exponents"]))(*s["exponents"])
        ca = (C.c_double * len(s["exponents"]))(*s["coefficients"])
        keep += [ea, ca]
        gs[i] = GS(s["l"], len(s["exponents"]), *map(float, s["center"]), ea, ca)
    es = (EC * len(ecps))()
    for i, e in enumerate(ecps):
        arrs = [
            (C.c_int * len(e["ams"]))(*e["ams"]),
            (C.c_int * len(e["ns"]))(*e["ns"]),
            (C.c_double * len(e["exponents"]))(*e["exponents"]),
            (C.c_double * len(e["coefficients"]))(*e["coefficients"]),
        ]
        keep += arrs
        es[i] = EC(*map(float, e["center"]), len(e["ams"]), *arrs)
    nc = sum(E.ncart(s["l"]) for s in shells)
    out = (C.c_double * (nc * nc))()
    t = time.perf_counter()
    st = lib.ferric_ecp_matrix(gs, len(shells), es, len(ecps), out)
    dt = time.perf_counter() - t
    assert st == 0, st
    Vc = np.array(out[:]).reshape(nc, nc)
    ns = sum(2 * s["l"] + 1 for s in shells)
    V = np.zeros((ns, ns))
    oc = np.cumsum([0] + [E.ncart(s["l"]) for s in shells])
    osph = np.cumsum([0] + [2 * s["l"] + 1 for s in shells])
    for i, a in enumerate(shells):
        for j, b in enumerate(shells):
            V[osph[i] : osph[i + 1], osph[j] : osph[j + 1]] = E.to_sph(
                Vc[oc[i] : oc[i + 1], oc[j] : oc[j + 1]], a["l"], b["l"]
            )
    return V, dt


def main():
    atoms = triiodobenzene()
    shells = [s for z, p in atoms for s in bse_shells(z, p)]
    ecps = [bse_ecp(z, p) for z, p in atoms if z == 53]
    npair = len(shells) * (len(shells) + 1) // 2
    print(
        f"triiodobenzene def2-SVP: {len(shells)} shells, {sum(2 * s['l'] + 1 for s in shells)} AOs, {len(ecps)} ECPs, "
        f"{npair} shell pairs, {npair * len(ecps)} (pair, centre) triples; I ECP terms: local "
        f"{sum(1 for a in ecps[0]['ams'] if a == max(ecps[0]['ams']))}, semi-local "
        f"{sum(1 for a in ecps[0]['ams'] if a != max(ecps[0]['ams']))}"
    )
    E.STATS.reset()
    E.WINDOW_LOG = []
    t = time.perf_counter()
    V = E.ecp_matrix_spherical(shells, ecps)
    print(
        f"prototype (Python/numpy, NOT a cost measure): {time.perf_counter() - t:.1f} s"
    )
    s = E.STATS
    print(
        f"type 2: {s.windows} windows (after screening), {s.nodes} nodes, {s.ang_contractions} angular contractions;"
        f" type 1: {s.t1_windows} windows, {s.t1_nodes} nodes; screened (prim pair, term) items: {s.screened}"
    )
    print(
        f"angular work: type 2 {s.ang_flops / 1e6:.1f} Mflop over {s.ang_contractions} (pair, centre, l) contractions; "
        f"type 1 {s.t1_wflops / 1e6:.1f} Mflop over {s.t1_prim_pairs} primitive pairs (W build + contraction)"
    )
    with open(os.path.join(HERE, "windows.txt"), "w") as f:
        for w in E.WINDOW_LOG:
            f.write(" ".join(repr(x) for x in w) + "\n")
    so = sys.argv[1] if len(sys.argv) > 1 else None
    if so:
        lib = C.CDLL(so)
        lib.ferric_ecp_matrix.restype = C.c_int
        ts = []
        for _ in range(3):
            Vl, dt = libecpint_matrix(lib, shells, ecps)
            ts.append(dt)
        print(
            f"libecpint ferric_ecp_matrix: {min(ts):.3f} s (best of 3: {', '.join(f'{x:.3f}' for x in ts)})"
        )
        d = np.abs(V - Vl)
        print(
            f"|ours - libecpint|: max {d.max():.2e} (scale {np.abs(V).max():.2e}); elements > 1e-8: "
            f"{int((d > 1e-8).sum())} of {d.size}"
        )


if __name__ == "__main__":
    main()
