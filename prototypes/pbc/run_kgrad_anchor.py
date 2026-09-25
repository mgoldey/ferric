"""Iteration 21 exactness anchors for k-point RHF / UHF forces (pbc_kgrad.py; dense AFT, no PySCF pbc).
Usage: python3 run_kgrad_anchor.py {gamma|<case>} [sc] [fd] [mut]
  gamma : 1x1x1 mesh == Gamma force (pbc_grad.gamma_rhf_grad for RHF, pbc_grad_open.gamma_grad for UHF / s+p)
  <case>: one of CASES below; flags: sc = supercell anchor, fd = central FD of the prototype's own k energy,
          mut = the five mutants against whichever anchors were run.

PREDICTIONS (written before any multi-cell run; only the H2 1x1x1 smoke had been seen, 5e-16 vs pbc_grad):
 P1 1x1x1 == Gamma force at the SAME density to rounding (<= 1e-12), RHF and UHF, s and s+p.
 P2 supercell: with the k-mesh density UNFOLDED to the diag(n) supercell (same K sphere, same v_M), the Gamma
    supercell force on EVERY copy c of atom A equals the k-mesh force on A to <= 1e-11 (the per-copy spread is itself
    a translation-symmetry check). Per-cell normalisation: E_cell = E_sc/N and moving A moves all N copies, so
    dE_cell/dR_A = (1/N) sum_c dE_sc/dR_{A,c}; by translation symmetry each term equals the mean, so the k force
    equals the force on ONE copy (NOT the sum over copies, which is N x). An independent tight Gamma SCF on the
    supercell agrees to the SCF residual (~1e-10).
 P3 analytic - central FD (h = 1e-4) of the k energy at the h^2 f'''/6 floor (~1e-9, as Iteration 16's 2.4e-9).
 P4 sum_A F_A ~ 1e-15; F(ewald) - F(none) <= 1e-10 (same state; the -v_M D_s S D_s M-term cancels W's -v_M D_s);
    E(ewald) - E(none) = -v_M (na + nb)/2 exactly with the MESH v_M.
 P5 closed shell: kUHF (converged from an asymmetric start) == kRHF force <= 1e-11 (two formula branches).
ARTIFACT HYPOTHESES (if broken, I expect):
 - no_phase: invisible at 1x1x1; at Nk > 1 O(1e-2..1e-1) everywhere (every residue gets the k-averaged density);
 - conj_D: D(k) -> D(k)^* == D(-k): BLIND on TRIM meshes (1x1x2, 2x2x2: D(k) is real), visible only when some
   n_i >= 3 (1x1x3), size ~ |Im D(k)|;
 - wrong_q (k = k' + q): BLIND when q == -q mod G for every q (all n_i <= 2), visible at 1x1x3;
 - gamma_vm: blind for exxdiv none and at 1x1x1 (the two v_M coincide); ewald at Nk > 1:
   miss = (v_M(prim) - v_M(mesh)) x |tr(D_s dS)|-sized, O(1e-2..1e-1);
 - no_invNk: blind at Nk = 1; else (Nk - 1) x (S + T parts), O(0.1..1);
 - a missing k-specific term would show as an FD residual far above 1e-9 with sum F still ~0 (sum F is blind to
   every mutant above: each acts on a translation-invariant contraction);
 - a displaced SCF landing on another state = an O(1e-3..1) jump; nocc per k and the global gaps are printed.
"""

import sys
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_open as O
import pbc_kgrad as KG
import pbc_kpts as PK
from pbc_dft import dense_jk
from pbc_gamma import Cell, build_integrals, madelung
from pbc_kuhf import unfold_dm

H = 1e-4
TRI_A = np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]])
SP_H = {"H": gto.parse("""
H S
  3.42525091  0.15432897
  0.62391373  0.53532814
  0.16885540  0.44463454
H P
  0.8         1.0
""")}
H2 = dict(a=np.eye(3) * 4.0, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))], basis="sto-3g")
H3 = dict(a=np.eye(3) * 4.5, atoms=[("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))],
          basis="sto-3g")
TRI2 = dict(a=TRI_A, atoms=[("H", (0.13, 0.25, 0.31)), ("H", (0.02, 0.27, 1.66))], basis=SP_H)
TRI4 = dict(a=TRI_A, atoms=[("H", (0.13, 0.25, 0.31)), ("H", (0.02, 0.27, 1.66)), ("H", (2.47, 2.41, 2.25)),
                            ("H", (3.52, 2.98, 2.71))], basis=SP_H)
# name: (system, mesh, na, nb, restricted, gcut, FD comps)
CASES = {
    "h2_112": (H2, (1, 1, 2), 1, 1, True, 10.0, [(0, 0), (0, 2), (1, 1)]),
    "h2_113": (H2, (1, 1, 3), 1, 1, True, 10.0, [(0, 0), (0, 2), (1, 1)]),
    "h2_222": (H2, (2, 2, 2), 1, 1, True, 10.0, [(0, 0), (0, 2), (1, 1)]),
    "h2_113_uhf": (H2, (1, 1, 3), 1, 1, False, 10.0, [(0, 2)]),
    "h3_113": (H3, (1, 1, 3), 2, 1, False, 10.0, [(0, 0), (2, 1), (1, 2)]),
    "tri2_113": (TRI2, (1, 1, 3), 1, 1, True, 10.0, [(1, 1), (0, 2)]),
    "tri2_113_trip": (TRI2, (1, 1, 3), 2, 0, False, 10.0, [(1, 1), (0, 2)]),
    "tri4_113": (TRI4, (1, 1, 3), 2, 2, True, 10.0, [(2, 1), (0, 2)]),
}
MUTANTS = ["no_phase", "conj_D", "wrong_q", "gamma_vm", "no_invNk"]


def cell_of(s, atoms=None):
    return Cell(s["a"], atoms or s["atoms"], s["basis"])


def kb_pair(cell, n, gcut):
    kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
    return {"none": dict(kb, madelung=0.0), "ewald": kb}


def grad_of(cell, kbe, r, rst):
    return KG.kgrad(cell, kbe, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=rst)


def refs(cell, kbs, na, nb, rst, guess=None):
    out = {}
    r0 = KG.kscf(kbs["none"], na, nb, restricted=rst, guess=guess)
    r1 = KG.kscf(kbs["ewald"], na, nb, restricted=rst, guess=(r0["Da"], r0["Db"]))  # staged ewald (Iteration 13)
    for ex, r in (("none", r0), ("ewald", r1)):
        g, parts = grad_of(cell, kbs[ex], r, rst)
        out[ex] = dict(r=r, g=g, parts=parts)
    return out


# ============================================================================ 1x1x1 == Gamma
def anchor_gamma():
    print("## (1) 1x1x1 mesh == Gamma force at the same density")
    # RHF H2: pbc_grad.gamma_rhf_grad (Iteration 16) supplies D and its own force
    c = cell_of(H2)
    for ex in ("none", "ewald"):
        r = PGd.gamma_rhf_grad(c, 2, exxdiv=ex, gcut=12.0)
        kb = PK.build_k(c, (1, 1, 1), exxdiv=ex, gcut=12.0, verbose=False)
        D = r["D"].astype(complex)[None]
        F = fock(kb, D / 2, D / 2)[0]
        g, _ = KG.kgrad(c, kb, D / 2, D / 2, F, F, restricted=True)
        rk = KG.kscf(kb, 1, 1, restricted=True)
        gk, _ = grad_of(c, kb, rk, True)
        print(f"  H2 RHF {ex:5s}: same-D |F_k - F_gamma| {abs(g - r['grad']).max():.1e}; own tight k-SCF "
              f"|F_k - F_gamma| {abs(gk - r['grad']).max():.1e} (E {rk['e'] - r['e']:+.1e})", flush=True)
    # UHF H3 and RHF tri2 (p shells) vs pbc_grad_open.gamma_grad at the same density
    for name, s, na, nb, rst in (("H3 UHF (2,1)", H3, 2, 1, False), ("tri2 s+p RHF", TRI2, 1, 1, True),
                                 ("tri2 s+p UHF triplet", TRI2, 2, 0, False)):
        c = cell_of(s)
        ints = build_integrals(c, None, exxdiv="ewald", gcut=10.0, verbose=False)
        kbs = kb_pair(c, (1, 1, 1), 10.0)
        for ex in ("none", "ewald"):
            rk = KG.kscf(kbs[ex], na, nb, restricted=rst)
            gk, _ = grad_of(c, kbs[ex], rk, rst)
            ig = dict(ints, madelung=ints["madelung"] if ex == "ewald" else 0.0)
            Da, Db, Fa, Fb = (x[0].real for x in (rk["Da"], rk["Db"], rk["Fa"], rk["Fb"]))
            gg, _ = O.gamma_grad(c, ig, Da, Db, Fa, Fb, 1.0)
            print(f"  {name:22s} {ex:5s}: |F_k - F_gamma(gamma_grad)| {abs(gk - gg).max():.1e}  "
                  f"max|Im D| {abs(rk['Da'].imag).max():.0e}  sumF {abs(gk.sum(0)).max():.1e}", flush=True)


def fock(kb, Da, Db):
    S, h = kb["S"], kb["h"]
    Ja, Ka = PK.jk_k(kb, Da)
    Jb, Kb = PK.jk_k(kb, Db)
    vm = kb["madelung"]
    Ka = Ka + vm * np.einsum("kab,kbc,kcd->kad", S, Da, S)
    Kb = Kb + vm * np.einsum("kab,kbc,kcd->kad", S, Db, S)
    return h + Ja + Jb - Ka, h + Ja + Jb - Kb


# ============================================================================ supercell
def supercell_grad(cell, n, kbs, ref, na, nb, gcut, independent=True):
    """Gamma supercell force per copy from the UNFOLDED k density (formula anchor) and, optionally, from an
    independent tight Gamma SCF on the supercell."""
    sc = PK.supercell_cell(cell, n)
    N = int(np.prod(n))
    t = time.time()
    ints = build_integrals(sc, None, exxdiv="ewald", gcut=gcut, verbose=False)
    vm_sc = madelung(sc)
    print(f"  supercell build {time.time() - t:.0f}s: nao {sc.mol.nao}, nG {len(ints['G'])}, v_M sc {vm_sc:.12f} "
          f"mesh {kbs['ewald']['madelung']:.12f}", flush=True)
    out = {}
    for ex in ("none", "ewald"):
        ig = dict(ints, madelung=vm_sc if ex == "ewald" else 0.0)
        r = ref[ex]["r"]
        Da = unfold_dm(cell, kbs[ex], r["Da"])
        Db = unfold_dm(cell, kbs[ex], r["Db"])
        imag = max(abs(Da.imag).max(), abs(Db.imag).max())
        Da, Db = Da.real, Db.real
        jk = dense_jk(ints["I"])
        Ja, Ka = jk(Da)
        Jb, Kb = jk(Db)
        S = ints["S"]
        Fa = ints["h"] + Ja + Jb - Ka - ig["madelung"] * S @ Da @ S
        Fb = ints["h"] + Ja + Jb - Kb - ig["madelung"] * S @ Db @ S
        e_sc = 0.5 * (np.sum((ints["h"] + Fa) * Da) + np.sum((ints["h"] + Fb) * Db)) + ints["enn"]
        comm = max(abs(Fa @ Da @ S - S @ Da @ Fa).max(), abs(Fb @ Db @ S - S @ Db @ Fb).max())
        g_sc, _ = O.gamma_grad(sc, ig, Da, Db, Fa, Fb, 1.0)
        d = dict(g=g_sc.reshape(N, cell.mol.natm, 3), e=e_sc / N, imag=imag, comm=comm)
        if independent:
            rs = O.scf(S, ints["h"], jk, ints["enn"], na * N, nb * N, None, "HF", ig["madelung"],
                       restricted=(na == nb), guess=(Da, Db))
            g2, _ = O.gamma_grad(sc, ig, rs["Da"], rs["Db"], rs["Fa"], rs["Fb"], 1.0)
            d.update(g_ind=g2.reshape(N, cell.mol.natm, 3), e_ind=rs["e"] / N)
        out[ex] = d
    return out


# ============================================================================ FD
def fd_all(cell, n, na, nb, rst, gcut, comps, ref):
    out = {}
    for A, x in comps:
        ev = {}
        for s in (1, -1):
            t = time.time()
            cd = PGd.displaced(cell, A, x, s * H)
            kbs = kb_pair(cd, n, gcut)
            for ex in ("none", "ewald"):
                r0 = ref[ex]["r"]
                ev[(ex, s)] = KG.kscf(kbs[ex], na, nb, restricted=rst, guess=(r0["Da"], r0["Db"]))
            print(f"    FD ({A},{x}) {s:+d} {time.time() - t:.0f}s", flush=True)
        for ex in ("none", "ewald"):
            p, m = ev[(ex, 1)], ev[(ex, -1)]
            same = (p["nocc_a_k"] == m["nocc_a_k"] == ref[ex]["r"]["nocc_a_k"]
                    and p["nocc_b_k"] == m["nocc_b_k"] == ref[ex]["r"]["nocc_b_k"])
            gap = min(p["gap_a"], m["gap_a"], *( (p["gap_b"], m["gap_b"]) if nb else ()))
            out[(ex, A, x)] = ((p["e"] - m["e"]) / (2 * H), same, gap)
    return out


# ============================================================================ main
def case(name, flags):
    s, n, na, nb, rst, gcut, comps = CASES[name]
    cell = cell_of(s)
    t = time.time()
    kbs = kb_pair(cell, n, gcut)
    print(f"## {name}: mesh {n}, (na, nb) = ({na}, {nb}) per cell, restricted {rst}, nao {cell.mol.nao}, gcut {gcut}, "
          f"v_M(mesh) {kbs['ewald']['madelung']:.12f}, v_M(prim) {madelung(cell):.12f}, build {time.time() - t:.0f}s",
          flush=True)
    guess = None
    if not rst and na == nb:  # closed-shell UHF from an ASYMMETRIC start (P5 must converge back, not start there)
        rr = refs(cell, kbs, na, nb, True)
        Da = rr["none"]["r"]["Da"]
        rng = np.random.default_rng(1)
        pert = rng.normal(size=Da.shape) * 0.02
        pert = pert + pert.transpose(0, 2, 1)
        guess = (Da + pert, Da - pert)
    t = time.time()
    ref = refs(cell, kbs, na, nb, rst, guess)
    for ex in ("none", "ewald"):
        r = ref[ex]["r"]
        print(f"  ref {ex:5s}: E/cell {r['e']:.12f} it {r['it']} err {r['err']:.1e} nocc_a/k {r['nocc_a_k']} "
              f"nocc_b/k {r['nocc_b_k']} gap_a {r['gap_a']:.4f} gap_b {r['gap_b']:.4f} max|Im D_a| "
              f"{abs(r['Da'].imag).max():.1e}  sumF {abs(ref[ex]['g'].sum(0)).max():.1e}", flush=True)
    print(f"  ref SCF + grads {time.time() - t:.0f}s")
    print("  F(none) =\n" + np.array2string(ref["none"]["g"], precision=12))
    print(f"  F(ewald) - F(none) {abs(ref['ewald']['g'] - ref['none']['g']).max():.1e}; E(ewald) - E(none) "
          f"{ref['ewald']['r']['e'] - ref['none']['r']['e']:+.12f} vs -v_M (na+nb)/2 "
          f"{-kbs['ewald']['madelung'] * (na + nb) / 2:+.12f}")
    if not rst and na == nb:
        print(f"  (P5) kUHF(asym start) - kRHF: |dF| none {abs(ref['none']['g'] - rr['none']['g']).max():.1e} "
              f"ewald {abs(ref['ewald']['g'] - rr['ewald']['g']).max():.1e}, dE "
              f"{ref['none']['r']['e'] - rr['none']['r']['e']:.1e}, |Da-Db| {abs(ref['none']['r']['Da'] - ref['none']['r']['Db']).max():.1e}")
    muts = {}
    if "mut" in flags:
        for m in MUTANTS:
            KG._MUTANT = m
            try:
                muts[m] = {ex: grad_of(cell, kbs[ex], ref[ex]["r"], rst)[0] for ex in ("none", "ewald")}
            finally:
                KG._MUTANT = None
    if "sc" in flags:
        scr = supercell_grad(cell, n, kbs, ref, na, nb, gcut)
        print("  (2) supercell anchor | exxdiv | E_k - E_sc(unfolded)/N | max_c |F_k - F_sc(copy c)| | "
              "spread over copies | |F_k - mean_c| | independent SCF: dE, max|dF| | unfold Im | sc comm |")
        for ex in ("none", "ewald"):
            d = scr[ex]
            gk = ref[ex]["g"]
            dmax = abs(d["g"] - gk[None]).max()
            spread = abs(d["g"] - d["g"].mean(0)[None]).max()
            print(f"  | {ex} | {ref[ex]['r']['e'] - d['e']:+.1e} | {dmax:.1e} | {spread:.1e} | "
                  f"{abs(d['g'].mean(0) - gk).max():.1e} | {ref[ex]['r']['e'] - d['e_ind']:+.1e}, "
                  f"{abs(d['g_ind'] - gk[None]).max():.1e} | {d['imag']:.0e} | {d['comm']:.0e} |", flush=True)
            print(f"      sum over copies / N == F_k: {abs(d['g'].sum(0) / len(d['g']) - gk).max():.1e}; "
                  f"sum over copies (N x) - F_k: {abs(d['g'].sum(0) - gk).max():.1e}")
        for m, gm in muts.items():
            print(f"  mutant {m:9s} vs supercell: none {abs(gm['none'] - scr['none']['g'][0]).max():.1e}  "
                  f"ewald {abs(gm['ewald'] - scr['ewald']['g'][0]).max():.1e}", flush=True)
    if "fd" in flags:
        t = time.time()
        fdv = fd_all(cell, n, na, nb, rst, gcut, comps, ref)
        print(f"  (3) FD h={H} ({time.time() - t:.0f}s): comp | exxdiv | analytic | FD | an - FD | same occ | min gap")
        for (ex, A, x), (f, same, gap) in fdv.items():
            a = ref[ex]["g"][A, x]
            print(f"  | ({A},{x}) | {ex} | {a:+.12f} | {f:+.12f} | {a - f:+.2e} | {same} | {gap:.3f} |", flush=True)
        for m, gm in muts.items():
            miss = {ex: max(abs(gm[ex][A, x] - fdv[(ex, A, x)][0]) for (e2, A, x) in fdv if e2 == ex)
                    for ex in ("none", "ewald")}
            print(f"  mutant {m:9s} vs FD: none {miss['none']:.1e}  ewald {miss['ewald']:.1e}  "
                  f"(alg |F_mut - F| {max(abs(gm[ex] - ref[ex]['g']).max() for ex in ('none', 'ewald')):.1e}, "
                  f"sumF {max(abs(gm[ex].sum(0)).max() for ex in ('none', 'ewald')):.1e})", flush=True)
    if muts and "fd" not in flags and "sc" not in flags:
        for m, gm in muts.items():
            print(f"  mutant {m:9s} alg |F_mut - F|: none {abs(gm['none'] - ref['none']['g']).max():.1e} "
                  f"ewald {abs(gm['ewald'] - ref['ewald']['g']).max():.1e}")


def hscan(name, A, x, hs=(2.5e-5, 5e-5, 1e-4, 2e-4)):
    """an - FD(h) for one component, exxdiv none: h^2 truncation (ratio 4 per doubling) vs a defect (h-independent)."""
    s, n, na, nb, rst, gcut, _ = CASES[name]
    cell = cell_of(s)
    kbs = kb_pair(cell, n, gcut)
    ref = refs(cell, kbs, na, nb, rst)
    a = ref["none"]["g"][A, x]
    out = []
    for h in hs:
        e = [KG.kscf(kb_pair(PGd.displaced(cell, A, x, sg * h), n, gcut)["none"], na, nb, restricted=rst,
                     guess=(ref["none"]["r"]["Da"], ref["none"]["r"]["Db"]))["e"] for sg in (1, -1)]
        out.append((h, (e[0] - e[1]) / (2 * h)))
        print(f"  hscan {name} ({A},{x}) h {h:.1e}: an - FD {a - out[-1][1]:+.3e}", flush=True)
    (h1, f1), (h2, f2) = out[1], out[2]
    rich = (4 * f1 - f2) / 3  # h1 = h2 / 2
    print(f"  Richardson ({h1:.0e}, {h2:.0e}): an - FD_R {a - rich:+.2e}")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gamma":
        anchor_gamma()
    elif which == "hscan":
        hscan(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        case(which, set(sys.argv[2:]))
