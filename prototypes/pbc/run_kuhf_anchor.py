"""Stage 9 (k-point UHF) exactness anchors + mutants for pbc_kuhf.

Predictions written BEFORE the run (2026-09-24):
(a) closed shell (na = nb, no mix) via kuhf == pbc_kpts.krhf.  Physics: identical Fock per spin (K[D/2] = K[D]/2,
    v_M S (D/2) S), so dE <= 1e-12, both exxdiv.  Artifacts: K from D_total -> O(0.1) Ha; Madelung v_M/2 per spin
    -> exactly +v_M N/4 per cell under ewald, none untouched; per-k aufbau -> INVISIBLE whenever global aufbau
    already puts the same count at every k (an insulator per spin), so this anchor cannot see it.
(b) 1x1x1 mesh == pbc_uhf.uhf on pbc_kpts.gamma_aft (pbc_gamma pair FT, same gcut), open-shell triplet
    tri 4H/STO-3G (na 3, nb 1): <= 1e-12 in E and <S^2>.  Blind to per-k aufbau (Nk = 1).
(c) 1x1x3 mesh E/cell == Gamma UHF of the explicit 1x1x3 supercell / 3, tri 4H/STO-3G triplet per cell
    (supercell na 9, nb 3).  Physics: exact at any gcut, IF the supercell state is translation-invariant: then its
    Fock commutes with the cell translations, is block-diagonal in k, and supercell aufbau over all 12 orbitals ==
    global per-spin aufbau over the mesh.  Two ways it can FAIL without a bug: (i) the supercell finds a lower
    translation-BROKEN UHF state (spin-density wave / localised hole) that no 1x1x3 k-UHF can represent -- a
    physics difference, recognisable because the k-UHF started from the same broken density cannot hold it;
    (ii) the aufbau cut splits a k/-k (time-reversal) degenerate pair: the supercell picks an arbitrary real
    combination (translation-broken, time-reversal-even), the k code picks one of +-k (translation-invariant,
    complex) -> different densities, same "state count".  Checked by gap_a/gap_b at the cut.
    Artifacts: K[D_total] or v_M/2 break (c) at O(0.1); per-k aufbau breaks (c) ONLY if the global count is
    non-uniform over k (printed: nocc per k).  Supercell from the unfolded k-density must reproduce the k energy
    to 1e-12; supercell from its own core guess and from a random symmetric-breaking perturbation tells whether
    a lower broken state exists.
(d) k RS-GDF (pbc_kgdf) with the Iteration-2/11 trivial aux == dense k-AFT, open shell: anchor H2 (one s per H),
    1x1x3, doublet per cell (na 1, nb 0; a +1 cell, 3 alpha electrons in 6 bands -> a non-trivial SCF) and the
    triplet (na 2, nb 0; full alpha band): <= 1e-11 (the Iteration-11 kernel error is 1.8e-12).

Usage: python3 run_kuhf_anchor.py [a|b|c|d|mut] ...
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kuhf as KU  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, madelung  # noqa: E402
from pbc_uhf import uhf  # noqa: E402
from test_prototype import H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, _anchor_cell_and_aux  # noqa: E402

PREC, TH = 1e-4, 1e-8  # loose gcut: the supercell anchor is exact at any gcut


def tri():
    return Cell(TRI_A, TRI_ATOMS, "sto-3g")


def zchain():
    """Found by a scan (after the predictions above): H2 (bond 2.0 along x) stacked every 2.0 Bohr along z, STO-3G.
    The z band is so wide that with one alpha electron per cell the GLOBAL aufbau puts both alpha electrons of a
    1x1x2 mesh at Gamma (nocc_a per k [2, 0], gap 0.51 / 1.11 Ha none / ewald): the per-k-aufbau mutant's target."""
    return Cell(
        np.diag([5.0, 5.0, 2.0]),
        [("H", (0.0, 0.0, 0.0)), ("H", (2.0, 0.0, 0.0))],
        "sto-3g",
    )


def anchor_a():
    cell = Cell(H2_A, H2_ATOMS, "sto-3g")
    kb = PK.build_k(
        cell, (1, 1, 3), gcut=PK.aft_gcut(cell, PREC), thresh=TH, verbose=False
    )
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        er, epr, _ = PK.krhf(kb, 2, conv=1e-12, kshift=vm)
        u = KU.kuhf(kb, 1, 1, conv=1e-12, kshift=vm)
        print(
            f"(a) H2 1x1x3 {ex}: kUHF {u['e']:.12f} kRHF {er:.12f} dE {u['e'] - er:.1e} "
            f"d eps_a {max(abs(u['eps_a'][k] - epr[k]).max() for k in range(3)):.1e} "
            f"|Da-Db| {abs(u['Da'] - u['Db']).max():.1e} <S2> {u['s2']:.1e} it {u['it']}"
        )
    return kb


def anchor_b():
    cell = tri()
    gcut = PK.aft_gcut(cell, PREC)
    kb = PK.build_k(cell, (1, 1, 1), gcut=gcut, thresh=TH, verbose=False)
    g = PK.gamma_aft(cell, gcut=gcut, thresh=TH)
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        st, r0 = KU.staged_kuhf(kb, 3, 1, conv=1e-12, kshift=vm)
        ug = uhf(
            g["S"],
            g["h"],
            g["I"],
            g["enn"],
            3,
            1,
            conv=1e-12,
            kshift=vm,
            guess=(r0["Da"][0].real, r0["Db"][0].real),
        )
        print(
            f"(b) tri triplet 1x1x1 {ex}: kUHF {st['e']:.12f} GammaUHF {ug['e']:.12f} dE {st['e'] - ug['e']:.1e} "
            f"d<S2> {st['s2'] - ug['s2']:.1e} (<S2> {st['s2']:.9f}) max|Im Da| {abs(st['Da'].imag).max():.1e}"
        )


def build_c(cell, n=(1, 1, 3)):
    gcut = PK.aft_gcut(cell, PREC)
    t = time.time()
    kb = PK.build_k(cell, n, gcut=gcut, thresh=TH, verbose=False)
    sc = PK.supercell_cell(cell, n)
    g = PK.gamma_aft(sc, gcut=gcut, thresh=TH)
    print(
        f"  build (k + supercell) {time.time() - t:.0f}s; v_M k {kb['madelung']:.12f} sc {madelung(sc):.12f}"
    )
    return kb, sc, g


def anchor_c(cell, na, nb, kb=None, sc=None, g=None, label="tri"):
    if kb is None:
        kb, sc, g = build_c(cell)
    Nk = kb["Nk"]
    out = {}
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        st, _ = KU.staged_kuhf(kb, na, nb, conv=1e-12, kshift=vm)
        Dsa, Dsb = KU.unfold_dm(cell, kb, st["Da"]), KU.unfold_dm(cell, kb, st["Db"])
        imu = max(abs(Dsa.imag).max(), abs(Dsb.imag).max())
        us = uhf(
            g["S"],
            g["h"],
            g["I"],
            g["enn"],
            na * Nk,
            nb * Nk,
            conv=1e-12,
            kshift=vm,
            guess=(Dsa.real, Dsb.real),
        )
        print(
            f"(c) {label} {Nk}k {ex}: kUHF/cell {st['e']:.12f} sc(from k dm)/N {us['e'] / Nk:.12f} "
            f"dE {st['e'] - us['e'] / Nk:.1e} <S2> k {st['s2']:.10f} sc {us['s2']:.10f} d {st['s2'] - us['s2']:.1e}; "
            f"nocc_a/k {st['nocc_a_k']} nocc_b/k {st['nocc_b_k']} gap_a {st['gap_a']:.4f} gap_b {st['gap_b']:.4f} "
            f"|Im unfolded D| {imu:.1e} it {st['it']}",
            flush=True,
        )
        # supercell from its own guesses: does a lower, translation-broken state exist?
        rng = np.random.default_rng(7)
        for gname in ("core", "core+mix0.5", "rand-perturb"):
            try:
                if gname == "rand-perturb":
                    nao = Dsa.shape[0]
                    A = rng.normal(size=(nao, nao)) * 0.2
                    U = np.linalg.qr(np.eye(nao) + A - A.T)[0]
                    # rotate the converged supercell orbitals, keep occupation counts
                    Ca, Cb = (
                        us["Ca"] @ U[: us["Ca"].shape[1], : us["Ca"].shape[1]],
                        us["Cb"] @ U[: us["Cb"].shape[1], : us["Cb"].shape[1]],
                    )
                    gs = (
                        Ca[:, : na * Nk] @ Ca[:, : na * Nk].T,
                        Cb[:, : nb * Nk] @ Cb[:, : nb * Nk].T,
                    )
                    r = uhf(
                        g["S"],
                        g["h"],
                        g["I"],
                        g["enn"],
                        na * Nk,
                        nb * Nk,
                        conv=1e-11,
                        kshift=0.0,
                        guess=gs,
                        maxiter=400,
                    )
                else:
                    r = uhf(
                        g["S"],
                        g["h"],
                        g["I"],
                        g["enn"],
                        na * Nk,
                        nb * Nk,
                        conv=1e-11,
                        kshift=0.0,
                        mix=0.5 if "mix" in gname else 0.0,
                        maxiter=400,
                    )
                if vm:
                    r = uhf(
                        g["S"],
                        g["h"],
                        g["I"],
                        g["enn"],
                        na * Nk,
                        nb * Nk,
                        conv=1e-11,
                        kshift=vm,
                        guess=(r["Da"], r["Db"]),
                        maxiter=400,
                    )
                # translation invariance of the density: D_sc[(T,m),(T',n)] depends only on T - T' (mod mesh)
                nao = cell.mol.nao
                Da4 = r["Da"].reshape(Nk, nao, Nk, nao)
                tb = max(
                    abs(Da4[(i + 1) % Nk, :, (j + 1) % Nk] - Da4[i, :, j]).max()
                    for i in range(Nk)
                    for j in range(Nk)
                )
                print(
                    f"     sc guess={gname:12s} E/N {r['e'] / Nk:.12f} (vs k {r['e'] / Nk - st['e']:+.1e}) "
                    f"<S2> {r['s2']:.6f} translation-breaking max|D(T+1,T'+1)-D(T,T')| {tb:.1e}",
                    flush=True,
                )
            except RuntimeError as exc:
                print(f"     sc guess={gname}: {exc}")
        out[ex] = st
    return out


def anchor_d():
    import pbc_kgdf as KG

    cell, aux = _anchor_cell_and_aux()
    kb = PK.build_k(cell, (1, 1, 3), exxdiv="ewald", verbose=False)
    kg = KG.build_kgdf(cell, (1, 1, 3), auxmol=aux, spherical=False)
    jkB = KG.jk_from_kB(kg)
    for na, nb, lab in ((1, 0, "doublet (+1 cell)"), (2, 0, "triplet")):
        for ex in ("none", "ewald"):
            vm = kb["madelung"] if ex == "ewald" else 0.0
            ud, _ = KU.staged_kuhf(kb, na, nb, conv=1e-12, kshift=vm)
            ub = KU.kuhf(
                kb, na, nb, conv=1e-12, kshift=vm, jk=jkB, guess=(ud["Da"], ud["Db"])
            )
            print(
                f"(d) anchor-H2 1x1x3 {lab} {ex}: dense {ud['e']:.12f} kgdf {ub['e']:.12f} dE {ub['e'] - ud['e']:.1e} "
                f"nocc_a/k {ud['nocc_a_k']} gap_a {ud['gap_a']:.4f}"
            )
    return kb, kg


def mutants(cell, kb, sc, g, na, nb):
    import pbc_kuhf as M

    ref = {}
    for ex in ("none", "ewald"):
        vm = kb["madelung"] if ex == "ewald" else 0.0
        ref[ex] = KU.staged_kuhf(kb, na, nb, conv=1e-12, kshift=vm)[0]
    orig_jk, orig_m = M._jk_spin, M._madelung_term

    def jk_total(jk, Da, Db):
        J, K = jk(Da + Db)
        return J, K, K

    def mad_half(S, Ds, vm):
        return 0.5 * orig_m(S, Ds, vm)

    for name, patch in (
        ("K_from_D_total", ("_jk_spin", jk_total)),
        ("madelung_half_per_spin", ("_madelung_term", mad_half)),
        ("per_k_aufbau", None),
    ):
        if patch:
            setattr(M, patch[0], patch[1])
        try:
            for ex in ("none", "ewald"):
                vm = kb["madelung"] if ex == "ewald" else 0.0
                kw = dict(aufbau=M._aufbau_per_k) if patch is None else {}
                try:
                    r = M.staged_kuhf(kb, na, nb, conv=1e-12, kshift=vm, **kw)[0]
                    print(
                        f"MUTANT {name} {ex}: dE vs correct k-UHF {r['e'] - ref[ex]['e']:+.3e} "
                        f"(nocc_a/k {r['nocc_a_k']})",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(f"MUTANT {name} {ex}: {exc}")
        finally:
            M._jk_spin, M._madelung_term = orig_jk, orig_m


if __name__ == "__main__":
    what = sys.argv[1:] or ["a", "b", "c", "d"]
    if "a" in what:
        anchor_a()
    if "b" in what:
        anchor_b()
    if "c" in what or "mut" in what:
        cell = tri()
        kb, sc, g = build_c(cell)
        if "c" in what:
            anchor_c(cell, 3, 1, kb, sc, g)
        if "mut" in what:
            mutants(cell, kb, sc, g, 3, 1)
    if "d" in what:
        anchor_d()
    for n in ((1, 1, 2), (1, 1, 3)):
        tag = "z" + "".join(map(str, n))
        if tag in what or tag + "mut" in what:
            cell = zchain()
            kb, sc, g = build_c(cell, n)
            if tag in what:
                anchor_c(cell, 1, 0, kb, sc, g, label=f"zchain doublet(+1) {n}")
            if tag + "mut" in what:
                mutants(cell, kb, sc, g, 1, 0)
