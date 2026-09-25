"""Iteration 21b anchors for k-point RS-GDF forces (pbc_kgrad_gdf.py).  Usage: python3 run_kgrad_gdf.py {gamma|<case>} [sc] [fd] [mut]

PREDICTIONS (written before the multi-cell runs; the smoke of the 1x1x1 energy build had not been run either):
 G0 energy: this module's per-q build (every q explicit) == pbc_kgdf.build_kgdf (time-reversal fill) at the same D:
    |dE| <= 1e-13 (Iteration 11: TR fill vs all-q 5.8e-15 in the kernels).
 G1 1x1x1 == Iteration 18's pbc_grad_gdf.gamma_gdf_grad at the same D (different builder, pbc_gdf vs pbc_kgdf,
    energies equal to 1.4e-14 in Iteration 11): <= 1e-11.
 G2 supercell: k RS-GDF is EXACTLY the supercell Gamma RS-GDF (Iteration 11, aux on every image, same ranges, same
    per-q cut == block-diagonal supercell metric), so the k force == Iteration 18's supercell force on every copy at the
    unfolded density to <= 1e-10 (the SR image sets of the two builds differ only below prec).
 G3 analytic - FD (h = 1e-4) of the k RS-GDF energy at the dense floor (~1e-9), both exxdiv; F(ewald) == F(none).
ARTIFACT HYPOTHESES:
 - aux_phase (e^{+iq.T} on the SR aux images of dJ3): blind at 1x1x1 and on 1x1x2 (q = -q), O(1e-2) at 1x1x3;
 - no_metric: O(1e-2) everywhere (Iteration 18: 1.8e-2 .. 2.5e-1);
 - no_g0: O(1e-3..1e-2) (Iteration 18: 3.5e-3 .. 3.2e-2);
 - no_herm: ZERO if J3(k,k) is Hermitian to rounding (then Z's anti-Hermitian part contracts to an imaginary number);
   a nonzero miss would size the prec-level non-Hermiticity (Iteration 11: 5.6e-12 at prec 1e-13) -> expected blind.
"""
import sys
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_gdf as PGG
import pbc_kgrad as KG
import pbc_kgrad_gdf as KGG
import pbc_kpts as PK
from pbc_gamma import Cell, madelung
from pbc_gdf import jk_from_B
from pbc_kgdf import build_kgdf, jk_from_kB
from pbc_kuhf import unfold_dm
from run_kgrad_anchor import H2, H3

H = 1e-4
W_SPLIT = 1.0
# small even-tempered s+p aux on H (fit quality is irrelevant to the anchors; cost is not)
ET_SP = {"H": gto.parse("""
H S
  4.7  1.0
H S
  1.9  1.0
H S
  0.75 1.0
H S
  0.3  1.0
H P
  1.25 1.0
H P
  0.5  1.0
""")}
CASES = {  # name: (system, mesh, na, nb, restricted, gcut, comps)
    "h2_113": (H2, (1, 1, 3), 1, 1, True, 10.0, [(0, 0), (0, 2), (1, 1)]),
    "h2_112": (H2, (1, 1, 2), 1, 1, True, 10.0, [(0, 2), (1, 1)]),
    "h3_113": (H3, (1, 1, 3), 2, 1, False, 10.0, [(0, 0), (2, 1)]),
}
MUTANTS = ["aux_phase", "no_metric", "no_g0", "no_herm"]


def setup(cell, n, gcut, lindep=1e-10):
    kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
    aux = KGG.make_auxmol(cell, ET_SP)
    g = KGG.build(cell, n, aux, w=W_SPLIT, lindep=lindep)
    return {"none": dict(kb, madelung=0.0), "ewald": kb}, g


def scf2(kbs, g, na, nb, rst, guess=None):
    jk = KGG.jk(g)
    r0 = KG.kscf(kbs["none"], na, nb, restricted=rst, guess=guess, jk=jk)
    r1 = KG.kscf(kbs["ewald"], na, nb, restricted=rst, guess=(r0["Da"], r0["Db"]), jk=jk)
    return {"none": r0, "ewald": r1}


def anchor_gamma():
    print("## G0/G1 at 1x1x1")
    for name, s, na, nb, rst in (("H2 RHF", H2, 1, 1, True), ("H3 UHF", H3, 2, 1, False)):
        c = Cell(s["a"], s["atoms"], s["basis"])
        t = time.time()
        kbs, g = setup(c, (1, 1, 1), 10.0)
        print(f"  {name}: k build {time.time() - t:.0f}s kept {[int(p['keep'].sum()) for p in g['per_q']]}/"
              f"{len(g['per_q'][0]['s'])} smin {g['per_q'][0]['s'].min():.1e}", flush=True)
        rs = scf2(kbs, g, na, nb, rst)
        # G0: energy with pbc_kgdf's own B at the same density
        kgref = build_kgdf(c, (1, 1, 1), auxmol=g["auxmol"], w=W_SPLIT)
        for ex in ("none", "ewald"):
            r = rs[ex]
            e2 = KG.kscf(kbs[ex], na, nb, restricted=rst, guess=(r["Da"], r["Db"]), jk=jk_from_kB(kgref))["e"]
            gk, pk, dg = KGG.grad(c, kbs[ex], g, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst)
            # G1: Iteration 18 at the same D
            ints = PGG.integrals(c, 10.0, ex)
            gd = PGG.gdf(c, g["auxmol"], W_SPLIT, lindep=1e-10)
            Da, Db = r["Da"][0].real, r["Db"][0].real
            J_a, K_a = jk_from_B(gd["B"])(Da)
            J_b, K_b = jk_from_B(gd["B"])(Db)
            S = ints["S"]
            Fa = ints["h"] + J_a + J_b - K_a - ints["madelung"] * S @ Da @ S
            Fb = ints["h"] + J_a + J_b - K_b - ints["madelung"] * S @ Db @ S
            gg, pg, _ = PGG.gamma_gdf_grad(c, ints, gd, g["auxmol"], Da, Db, Fa, Fb, 1.0, w=W_SPLIT)
            print(f"    {ex:5s}: E {r['e']:.12f}  E(pbc_kgdf B) - E {e2 - r['e']:+.1e}  |F_k - F_gamma18| "
                  f"{abs(gk - gg).max():.1e}  sumF {abs(gk.sum(0)).max():.1e}  |F|max {abs(gk).max():.3f}", flush=True)


def case(name, flags):
    s, n, na, nb, rst, gcut, comps = CASES[name]
    cell = Cell(s["a"], s["atoms"], s["basis"])
    t = time.time()
    kbs, g = setup(cell, n, gcut)
    print(f"## {name} RS-GDF: mesh {n}, ({na},{nb}), aux ET-sp {g['auxmol'].nao} cart, w {W_SPLIT}, build {g['wall']:.0f}s; "
          f"kept per q {[int(p['keep'].sum()) for p in g['per_q']]} of {len(g['per_q'][0]['s'])}, "
          f"smin per q {[float('%.1e' % p['s'].min()) for p in g['per_q']]}", flush=True)
    rs = scf2(kbs, g, na, nb, rst)
    kgref = build_kgdf(cell, n, auxmol=g["auxmol"], w=W_SPLIT)
    ref = {}
    for ex in ("none", "ewald"):
        r = rs[ex]
        e2 = KG.kscf(kbs[ex], na, nb, restricted=rst, guess=(r["Da"], r["Db"]), jk=jk_from_kB(kgref))["e"]
        gk, pk, dg = KGG.grad(cell, kbs[ex], g, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst)
        ref[ex] = dict(r=r, g=gk, parts=pk)
        print(f"  ref {ex:5s}: E {r['e']:.12f} (pbc_kgdf B: {e2 - r['e']:+.1e}) err {r['err']:.1e} nocc_a/k {r['nocc_a_k']} "
              f"gap_a {r['gap_a']:.3f} sumF {abs(gk.sum(0)).max():.1e} max|H-H^+| {max(d['Himag'] for d in dg):.1e}",
              flush=True)
    print("  F(none) =\n" + np.array2string(ref["none"]["g"], precision=12))
    print(f"  F(ewald) - F(none) {abs(ref['ewald']['g'] - ref['none']['g']).max():.1e}; E(ewald)-E(none) "
          f"{ref['ewald']['r']['e'] - ref['none']['r']['e']:+.12f} vs {-kbs['ewald']['madelung'] * (na + nb) / 2:+.12f}")
    muts = {}
    if "mut" in flags:
        for m in MUTANTS:
            KGG._MUTANT = m
            try:
                muts[m] = {ex: KGG.grad(cell, kbs[ex], g, ref[ex]["r"]["Da"], ref[ex]["r"]["Db"], ref[ex]["r"]["Fa"],
                                        ref[ex]["r"]["Fb"], restricted=rst)[0] for ex in ("none", "ewald")}
            finally:
                KGG._MUTANT = None
    if "sc" in flags:
        t = time.time()
        sc = PK.supercell_cell(cell, n)
        N = int(np.prod(n))
        aux_sc = KGG.make_auxmol(sc, ET_SP)
        gd = PGG.gdf(sc, aux_sc, W_SPLIT, lindep=1e-10)
        ints = PGG.integrals(sc, gcut, "ewald")
        vm_sc = madelung(sc)
        print(f"  supercell GDF build {time.time() - t:.0f}s (naux {gd['B'].shape[0]})", flush=True)
        cache = {}
        for ex in ("none", "ewald"):
            ig = dict(ints, madelung=vm_sc if ex == "ewald" else 0.0)
            r = ref[ex]["r"]
            Da = unfold_dm(cell, kbs[ex], r["Da"]).real
            Db = unfold_dm(cell, kbs[ex], r["Db"]).real
            J_a, K_a = jk_from_B(gd["B"])(Da)
            J_b, K_b = jk_from_B(gd["B"])(Db)
            S = ints["S"]
            Fa = ints["h"] + J_a + J_b - K_a - ig["madelung"] * S @ Da @ S
            Fb = ints["h"] + J_a + J_b - K_b - ig["madelung"] * S @ Db @ S
            e_sc = 0.5 * (np.sum((ints["h"] + Fa) * Da) + np.sum((ints["h"] + Fb) * Db)) + ints["enn"]
            comm = max(abs(Fa @ Da @ S - S @ Da @ Fa).max(), abs(Fb @ Db @ S - S @ Db @ Fb).max())
            gsc, _, _ = PGG.gamma_gdf_grad(sc, ig, gd, aux_sc, Da, Db, Fa, Fb, 1.0, w=W_SPLIT, cache=cache)
            gsc = gsc.reshape(N, cell.mol.natm, 3)
            print(f"  (G2) {ex:5s}: E_k - E_sc/N {r['e'] - e_sc / N:+.1e}, sc comm {comm:.0e}, max_c|F_k - F_sc(c)| "
                  f"{abs(gsc - ref[ex]['g'][None]).max():.1e}, spread {abs(gsc - gsc.mean(0)).max():.1e} "
                  f"({time.time() - t:.0f}s)", flush=True)
            for m, gm in muts.items():
                print(f"    mutant {m:9s} vs supercell: {abs(gm[ex] - gsc[0]).max():.1e}")
    if "fd" in flags:
        t = time.time()
        res = {}
        for A, x in comps:
            ev = {}
            for sgn in (1, -1):
                cd = PGd.displaced(cell, A, x, sgn * H)
                kd, gdd = setup(cd, n, gcut)
                jk = KGG.jk(gdd)
                for ex in ("none", "ewald"):
                    r0 = ref[ex]["r"]
                    ev[(ex, sgn)] = KG.kscf(kd[ex], na, nb, restricted=rst, guess=(r0["Da"], r0["Db"]), jk=jk)
                    ev[("kept", sgn)] = [int(p["keep"].sum()) for p in gdd["per_q"]]
                print(f"    FD ({A},{x}) {sgn:+d} {time.time() - t:.0f}s kept {ev[('kept', sgn)]}", flush=True)
            for ex in ("none", "ewald"):
                p, m = ev[(ex, 1)], ev[(ex, -1)]
                res[(ex, A, x)] = (p["e"] - m["e"]) / (2 * H)
                a = ref[ex]["g"][A, x]
                print(f"  | ({A},{x}) | {ex} | {a:+.12f} | {res[(ex, A, x)]:+.12f} | {a - res[(ex, A, x)]:+.2e} | "
                      f"occ {p['nocc_a_k'] == m['nocc_a_k'] == ref[ex]['r']['nocc_a_k']} |", flush=True)
        for m, gm in muts.items():
            miss = {ex: max(abs(gm[ex][A, x] - res[(ex, A, x)]) for (e2, A, x) in res if e2 == ex) for ex in ("none", "ewald")}
            print(f"  mutant {m:9s} vs FD: none {miss['none']:.1e} ewald {miss['ewald']:.1e} "
                  f"(sumF {max(abs(gm[e].sum(0)).max() for e in gm):.1e})")
    if muts and "fd" not in flags:
        for m, gm in muts.items():
            print(f"  mutant {m:9s} alg: none {abs(gm['none'] - ref['none']['g']).max():.1e} "
                  f"ewald {abs(gm['ewald'] - ref['ewald']['g']).max():.1e}")


if __name__ == "__main__":
    if sys.argv[1] == "gamma":
        anchor_gamma()
    else:
        case(sys.argv[1], set(sys.argv[2:]))
