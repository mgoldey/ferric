"""Iteration 5b: Gamma LMP2 with the uniform (q=0 head) coupling handled explicitly (FINDINGS item 6).

System: the Iteration-5 needle sweep (1 x 1 x N, primitive cubic a0 = 7, one tilted H2, 6-31G, cc-pvdz-ri,
RS-GDF w = 0.5, shifted denominators).  Written, with the predictions below, BEFORE the sweep (2026-09-24).
Only a two-point smoke run (N = 4, 8: E_can - E_head = 2.955e-3 / 2.922e-3) preceded this docstring.

PHYSICS QUESTION: is the uniform coupling -(4pi/Omega_sc) mu_ia mu_jb an artifact or physical correlation?
  Derivation (pbc_lmp2.py, Iteration 5b block): the Gamma supercell is a 1 x 1 x N k-mesh; for every q != 0 on
  the mesh the G=0 head of (ia|jb) is (4pi/Omega_sc) mu_z mu_z, at q = 0 it is dropped.  In real space
  sum_{q!=0} e^{iqd} = N delta_d0 - 1: the uniform coupling IS minus the omitted q=0 head, a quadrature hole of
  measure 1/N.  It contributes O(1) in total, O(1/N) per molecule, 0 in the thermodynamic limit (TDL): a
  finite-size term like the Madelung one, not correlation that survives.  Iteration 3's c3/a^3 contains the
  same object: its d(ia|ia) = -(4pi/3 Omega)|d_ia|^2 is this head with the cubic depolarisation W = I/3 instead
  of the needle's zz; c3 ALSO contains the Fock-side heads (d eps_i, d eps_a), which J' = J - J_unif does not
  touch.  Consequence for candidate A: if it is a quadrature hole, adding it back reproduces the canonical
  Gamma number (the method's reference at fixed N), while REMOVING it (solving on J') gives a different,
  better-converged-to-TDL number.  Both are legitimate; they answer different questions.

PREDICTIONS
  H1 (artifact test, discriminating): E_head(N) = closed-form canonical MP2 on J' = J - J_unif.
     Quadrature-hole picture: E_can - E_head -> a positive N-independent constant (the head energy; the missing
     head underbinds), so b_head = b_can - const has SMALLER |b| than b_can = +3.96e-3, and the same e_inf
     (e_inf equality is guaranteed by 1/Omega scaling: a consistency check, NOT a discriminator).
     Physical-long-range picture would instead make E_head/N converge to the TDL SLOWER (|b_head| > |b_can|).
     Residual b_head expected O(1e-3), positive: the Fock-side heads (Iteration 3's d eps terms) remain.
  H2 (candidate A-add: gate on J - J_unif, solve on J, add back dropped pairs' uniform energy): partners per
     molecule saturate at the near-field count at EVERY N (no N* onset), eps=0 exact; dE vs E_can = a N + b
     with |b| <= 1e-5 (add-back residual; Iteration 5 measured 3e-6 at N=48 for R_c).  Mutation (no add-back):
     b -> +1.0e-3 (the far-pair uniform energy), so the add-back is needed iff the reference is E_can.
  H3 (candidate A-drop: gate and solve on J'): dE vs E_head = a N + b with |b| small (<= 1e-5): far pairs carry
     no energy once the head is restored, so there is nothing to add back.
  H1b (ADDED after the N <= 16 run had shown E_can - E_head -> ~2.9e-3, i.e. b_head ~ +1.07e-3): restoring the
     head ALSO in the Fock exchange (fock_head_correction: Fvv, diag Foo) should remove most of the remaining
     b: |b_headF| << 1.07e-3.  If b_headF stays ~1e-3 the residual finite-size term is something else
     (e.g. the SCF orbitals themselves, or Foo off-diagonals), and 'artifact' is only shown for the ERI part.
  H4 (candidate B, energy screen T on the pair estimate): from J, far uniform pair energies ~ 1e-3/N^2 each ->
     all pairs kept until N ~ sqrt(1e-3/T) (T = 1e-7: N ~ 100), i.e. an onset again (sqrt instead of linear);
     from J' no onset.  R_c 10.5 (min-image): partners 3 once N >= 4, dE b = +1.09e-3 without add-back,
     |b| <= 1e-5 with it.
ARTIFACT HYPOTHESES
  X1 inaccurate mu (Resta O(b^2)): J' far would not vanish -> the J' gate keeps far pairs at tight eps and the
     add-back misses by rel. error x 1e-3.  Measured each N: farthest-pair max|J'| / max|J|.  Richardson (b, 2b)
     used to push it to O(b^4).
  X2 an add-back that only matches because it is fitted: it is NOT fitted (analytic from mu, Fvv, Foo), and it
     is zero at eps = 0 (no pair dropped) -> anchor unchanged.
  X3 translation symmetry makes dE = N x per-molecule at every N (A2 of Iteration 5): only the a/b split and the
     partner saturation are evidence.
Usage: python3 run_lmp2_uniform.py N1 N2 ...   [--ragged N ... for the large sizes: skips dense-only rows]
"""

import sys
import time

import numpy as np

import pbc_lmp2 as L
from run_lmp2_anchor import system

EPS_A = (1e-3, 3e-4, 1e-4, 3e-5)
EPS_BASE = (1e-3, 1e-4)
RC = 10.5
TCUT = (1e-6, 1e-7)


def row(tag, r, ref, N):
    add = (
        f" addback {r['e_addback']:+.3e} (no-add dE {r['e_solve'] - ref:+.4e})"
        if r["e_addback"]
        else ""
    )
    print(
        f"   {tag:<28s} dE {r['e'] - ref:+.4e}  pairs {r['pairs_kept']:5d}/{N * N}  partners "
        f"{r['partners'].min()}/{r['partners'].max()}  keep {r['keep']:.5f}{add}",
        flush=True,
    )


def run(N, solver="dense"):
    t0 = time.time()
    sc, d, scf = system((1, 1, N))
    ec = L.canonical_mp2(scf, d["B"])
    p = L.prepare(sc, d, scf)
    Z2 = L.resta_z_at(sc, d, 2, 2)
    mu, mi = L.transition_dipoles_richardson(sc, d, p, Z2=Z2)
    Ju = L.uniform_coupling(p, mu)
    eu = L.uniform_pair_energies(p, Ju)
    eh = L.canonical_mp2_from_local(p, scf, d["S"], p["J"] - Ju)
    Foo2, Fvv2, _ = L.fock_head_correction(sc, d, p, mu, Z2=Z2)
    e_chk = L.mp2_local_closed_form(p["J"], p["Foo"], p["Fvv"])
    e_hf = L.mp2_local_closed_form(p["J"] - Ju, Foo2, Fvv2)
    e_f = L.mp2_local_closed_form(p["J"], Foo2, Fvv2)
    print(
        f"== N={N}  H1b: E_headF (J', Foo', Fvv') {e_hf:.10e}  E_Fonly (J, Foo', Fvv') {e_f:.10e}  "
        f"closed-form(J, Foo, Fvv) - E_can {e_chk - ec:+.1e}",
        flush=True,
    )
    dist = L.pair_distances(p)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    Jf = p["J"][i, :, j, :]
    res = abs(Jf - Ju[i, :, j, :]).max()
    res1 = abs(Jf + (4 * np.pi / sc.sc.vol) * np.outer(mi["f1"][i], mi["f1"][j])).max()
    Jp = p["J"] - Ju
    print(
        f"== N={N}  E_can {ec:.10e}  E_head {eh:.10e}  E_can-E_head {ec - eh:+.5e}  mu* {abs(mu).max():.6f}  "
        f"farthest pair max|J| {abs(Jf).max():.3e}  max|J'| {res:.2e} (Resta f1 {res1:.2e})  "
        f"sum eu(offdiag) {eu.sum() - np.trace(eu):+.4e}",
        flush=True,
    )
    kw = dict(solver=solver)
    if solver == "dense":
        r0 = L.solve(p, 0.0, gate="J-unif", J_unif=Ju, addback=eu)
        rh = L.solve(p, 0.0, J=Jp)
        print(
            f"   anchors eps=0: A-add vs E_can {r0['e'] - ec:+.1e} (addback {r0['e_addback']:.1e}); "
            f"A-drop CG vs closed-form E_head {rh['e'] - eh:+.1e}",
            flush=True,
        )
    for eps in EPS_BASE:
        row(f"base eps {eps:g} (gate J)", L.solve(p, eps, **kw), ec, N)
    for eps in EPS_A:
        row(
            f"A-add eps {eps:g}",
            L.solve(p, eps, gate="J-unif", J_unif=Ju, addback=eu, **kw),
            ec,
            N,
        )
    for eps in EPS_A:
        row(f"A-drop eps {eps:g} (vs E_head)", L.solve(p, eps, J=Jp, **kw), eh, N)
    row(f"B Rc {RC} + addback", L.solve(p, 0.0, pair_cut=RC, addback=eu, **kw), ec, N)
    for T in TCUT:
        row(
            f"B E-screen {T:g} on J",
            L.solve(p, 0.0, pair_ecut=T, addback=eu, **kw),
            ec,
            N,
        )
        row(
            f"B E-screen {T:g} on J'+add",
            L.solve(p, 0.0, gate="J-unif", J_unif=Ju, pair_ecut=T, addback=eu, **kw),
            ec,
            N,
        )
    print(f"   [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    solver = "dense"
    if args and args[0] == "--ragged":
        solver, args = "ragged", args[1:]
    for N in [int(x) for x in args] or [4, 6, 8, 12, 16]:
        run(N, solver)
