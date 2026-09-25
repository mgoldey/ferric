"""Q2/Q3: lindep strategies in the k-point SCF, and the k-mesh == supercell anchor under filtering.

System: H2 (STO-3G + ONE diffuse s of exponent ad on each H), cubic a = 4, 1x1x3 mesh, exxdiv none, pure AFT
(loose gcut prec 1e-4, pair thresh 1e-8 on BOTH sides: the anchor is exact at any gcut, Iteration 9).
ad = 0.06: lam_min S(Gamma) 3.0e-8, S(+-k) 9.4e-4  (a clean unfiltered reference exists; tau = 1e-6 drops 1 at
           Gamma and 0 at +-k -> the kept count VARIES across k).
ad = 0.04: lam_min S(Gamma) 1.5e-12 (at the lattice-sum precision floor), S(+-k) 1.8e-5.

PREDICTIONS (before the run)
 canonical, absolute tau on the unnormalised S(k) (ferric today):
   PHYSICS: E(tau) - E_ref >= 0 and non-decreasing in tau (the kept space only shrinks); 0 to SCF precision while
   tau < lam_min everywhere; k-mesh == supercell to ~1e-12 for EVERY tau, although the count differs per k,
   because eig S_sc = union_k eig S(k) and the same cut therefore keeps the same space.
   ARTIFACT: an anchor failure here = a fold/phase bug; a DEcrease of E with tau = non-PSD integrals or a broken X.
 canonical on the diagonal-normalised S(k) (PySCF canonical_orth_): per-k diagonal != supercell diagonal, so the
   anchor can only hold when the cut falls in the same spectral gap; expect equality here (well separated gaps)
   and a mismatch at a tau inside the cluster.
 pivoted Cholesky (Lehtola, PySCF partial_cholesky_orth_): per k it drops AO BLOCH functions; on the supercell it
   drops real-space AOs in SOME cells -> translation symmetry broken -> anchor FAILS whenever something is dropped,
   holds when nothing is.
 exp_to_discard (emin between the diffuse s and STO-3G's 0.1689): a different, translation-invariant basis ->
   anchor exact; E - E_ref = the diffuse shell's whole contribution, >> canonical's error.
Usage: python3 run_lindep_scf.py [ad] [rcut_1e=45] [pair_thresh=1e-11] [asym=1]
"""

import sys
import time

import numpy as np
from pyscf import gto

sys.path.insert(0, ".")
import pbc_kpts as PK  # noqa: E402
import pbc_lindep as LD  # noqa: E402
from pbc_gamma import Cell  # noqa: E402

H2_A = np.eye(3) * 4.0
H2_ATOMS = [("H1", (0.3, 0.2, 0.1)), ("H2", (0.3, 0.2, 1.5))]
N = (1, 1, 3)


ASYM = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # != 1 breaks the cell's inversion symmetry (see below)


def model_basis(ad):
    """ASYM == 1: both H carry the same diffuse s -> the cell is inversion-symmetric and the Gamma near-null vector
    (the ODD combination of the two diffuse s) is orthogonal to the (even) occupied band BY SYMMETRY, so dropping it
    costs exactly 0: a symmetry fact, not evidence that small eigenvalues are harmless.  ASYM != 1 gives H2 the
    exponent ad*ASYM and removes that protection."""
    sto = gto.basis.load("sto-3g", "H")
    return {"H1": sto + [[0, [ad, 1.0]]], "H2": sto + [[0, [ad * ASYM, 1.0]]]}


RCUT_1E = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0  # 22 (the pbc_kpts default) truncates S at ~1e-6: ARTIFACT
PAIR_TH = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-11


def builds(basis):
    t0 = time.time()
    cell = Cell(H2_A, H2_ATOMS, basis)
    gcut, th = PK.aft_gcut(cell, 1e-4), PAIR_TH
    kb = PK.build_k(cell, N, gcut=gcut, thresh=th, rcut_1e=RCUT_1E, verbose=False)
    g = PK.gamma_aft(PK.supercell_cell(cell, N), gcut=gcut, thresh=th, rcut_1e=RCUT_1E)
    print(f"  builds {time.time() - t0:.0f}s nao={cell.mol.nao}", flush=True)
    return kb, LD.gamma_kb(g)


def run(kb, ksc, orth, label, ref=None):
    nk = len(N) and int(np.prod(N))
    try:
        e, eps, it = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, orth=orth, maxiter=300)
    except RuntimeError:
        e, it = np.nan, -1
    try:
        esc, _, itsc = PK.krhf(ksc, 2 * nk, conv=1e-12, kshift=0.0, orth=orth, maxiter=300)
        esc /= nk
    except RuntimeError:
        esc, itsc = np.nan, -1
    kc = LD.kept_counts(kb["S"], orth)
    ksc_c = LD.kept_counts(ksc["S"], orth)[0]
    r = f"{e - ref:+.3e}" if ref is not None else "   ref    "
    print(f"  {label:34s} E {e:.12f} (E-ref {r}) it {it:3d} kept/k {kc} (sum {sum(kc)}) | supercell kept {ksc_c} "
          f"it {itsc:3d} E_k - E_sc {e - esc:+.1e}", flush=True)
    return e


ad = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06
print(f"ad = {ad}, asym = {ASYM}, rcut_1e = {RCUT_1E}, pair thresh = {PAIR_TH:.0e}", flush=True)
kb, ksc = builds(model_basis(ad))
print("  lam_min S(k):", [f"{np.linalg.eigvalsh(s)[0]:.2e}" for s in kb["S"]],
      " S_sc:", f"{np.linalg.eigvalsh(ksc['S'][0])[0]:.2e}")
ref = run(kb, ksc, LD.orth_factory("lowdin"), "Lowdin S^-1/2 (nothing dropped)")
for thr in (1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
    run(kb, ksc, LD.orth_factory("canonical", thr=thr), f"canonical abs tau={thr:.0e}", ref)
for thr in (1e-7, 1e-6, 1e-4):
    run(kb, ksc, LD.orth_factory("canonical", thr=thr, normalize=True), f"canonical normalised tau={thr:.0e}", ref)
for tol in (1e-10, 1e-7, 1e-6, 1e-4):
    run(kb, ksc, LD.orth_factory("cholesky", tol=tol), f"pivoted Cholesky tol={tol:.0e}", ref)
bd, nd = LD.discard_basis(model_basis(ad), ["H1", "H2"], 0.1)
print(f"  exp_to_discard 0.1 drops {nd} primitive(s) per element")
kb2, ksc2 = builds(bd)
run(kb2, ksc2, LD.orth_factory("canonical", thr=1e-6), "exp_to_discard 0.1 + canonical 1e-6", ref)
bd0, nd0 = LD.discard_basis(model_basis(ad), ["H1", "H2"], 0.5 * ad)
print(f"  exp_to_discard {0.5 * ad} (below every exponent) drops {nd0}: identical basis -> anchor below")
