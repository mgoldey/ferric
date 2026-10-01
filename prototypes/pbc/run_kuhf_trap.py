"""Stage 9: does Iteration 6's ewald trap (and its remedies) carry over to k-point UHF?

System: tri 4H s+p triplet (na 3, nb 1 per cell), the Iteration-6 trap system, meshes 1x1x1 / 1x1x2 / 1x1x3,
dense k-AFT at gcut prec 1e-8 (the trap is a property of the model, not the cutoff).

Predictions (written before the run):
 P1 (algebra) v_M S(k) D_s(k) S(k) = v_M x (occupied projector) at EVERY k with the same v_M, so a None-stationary
    k-density is ewald-stationary, E_ewald - E_none = -v_M(n)(Na+Nb)/2 exactly, and the staged start (None, then
    ewald from that density) carries over unchanged: it converges in ~1 iteration to the None state.
 P2 (physics) the Madelung energy -v_M/2 sum_s (1/Nk) sum_k tr(S D_s S D_s) = -v_M N/2 for ANY idempotent occupation
    pattern over k, so moving an electron between k points costs no Madelung energy: the ewald and None energy
    landscapes are the same up to a constant, but ewald lowers every occupied level (at every k) by v_M, so the
    aufbau self-consistency window for a hole grows by v_M.  The trap therefore survives at k IF the trapped state's
    None-gap hole is shallower than v_M(n); v_M(n) ~ 1/n shrinks, so the trap should close at a large enough mesh
    (Iteration 6: hole 6 mHa vs v_M 0.62 at Gamma -> probably still trapped at n = 2, 3, where v_M ~ 0.3, 0.2).
 P3 criterion: per-spin GLOBAL gap (min virtual over all k minus max occupied over all k, from the actual
    occupations) >= v_M(n) is necessary for the HF minimum under ewald; equivalently the None-Fock gap of the same
    density >= 0.  Necessary, not sufficient (Iteration 6's O2 hcore state had gaps > v_M).
 Artifact: a wrong Madelung (1/Nk factor, primitive v_M) would show up as E_ewald - E_none != -v_M N/2.
Usage: python3 run_kuhf_trap.py [n3 ...]   (e.g. 111 112 113)"""

import sys
import time


sys.path.insert(0, ".")
import pbc_kpts as PK  # noqa: E402
import pbc_kuhf as KU  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from test_prototype import SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
for m in sys.argv[1:] or ["111", "112", "113"]:
    n = tuple(int(c) for c in m)
    t = time.time()
    kb = PK.build_k(cell, n, gcut=PK.aft_gcut(cell, 1e-8), thresh=1e-10, verbose=False)
    vm = kb["madelung"]
    print(f"{n}: build {time.time() - t:.0f}s v_M {vm:.6f}", flush=True)
    runs = {}
    runs["none core"] = KU.kuhf(kb, 3, 1, kshift=0.0, conv=1e-11)
    runs["ewald core"] = KU.kuhf(kb, 3, 1, conv=1e-11)
    runs["ewald core mix0.3"] = KU.kuhf(kb, 3, 1, conv=1e-11, mix=0.3)
    runs["ewald staged"], _ = KU.staged_kuhf(kb, 3, 1, conv=1e-11)
    for name, r in runs.items():
        ex = name.startswith("ewald")
        # None-Fock gap of the SAME (stationary) density: v_M S D S lowers exactly the occupied levels by v_M at
        # every k, so gap_none = gap_ewald - v_M (occupations from the actual D, not re-aufbau'd)
        g0 = (r["gap_a"] - vm, r["gap_b"] - vm) if ex else (r["gap_a"], r["gap_b"])
        print(
            f"  {name:18s} E {r['e']:.10f} <S2> {r['s2']:.6f} it {r['it']:3d} nocc_a/k {r['nocc_a_k']} "
            f"gap_a {r['gap_a']:+.4f} gap_b {r['gap_b']:+.4f}  (gap >= v_M? {min(r['gap_a'], r['gap_b']) >= vm if ex else '-'}) "
            f"None-Fock gaps of this D: {g0[0]:+.4f} {g0[1]:+.4f}",
            flush=True,
        )
    d = runs["ewald staged"]["e"] - runs["none core"]["e"]
    print(
        f"  staged - none = {d:+.12f} vs -v_M N/2 = {-vm * 2:+.12f} (d {d + 2 * vm:.1e}); "
        f"ewald core - staged = {runs['ewald core']['e'] - runs['ewald staged']['e']:+.3e}",
        flush=True,
    )
