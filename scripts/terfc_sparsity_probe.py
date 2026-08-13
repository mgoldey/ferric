#!/usr/bin/env python3
"""Does decoupled-omega terfc unblock integral sparsity?

Background: Dutoi's curvature link (r0*omega = 1/sqrt2) made terfc a
deliberately WEAK screener (full Coulomb inside r0, transition width
~1/omega = r0*sqrt2 — huge), and the multi-width no-win proof
(multi-width-terf-no-sparsity-win memory) assumed the link. The link is now
measured non-binding for B-formulation rs correlation (ne2_seam_test:
r0*omega = 4 costs 1.2 uHa on a 35 uHa well) and the constructors are
decoupled (03214031). This probe measures what that buys in raw integral
sparsity: (P|w|mu nu) and (P|w|Q) dropped-fraction vs omega at FIXED r0.

Pre-registered expectations: sharpening omega at fixed r0 narrows the
transition (width ~1/omega) but keeps full Coulomb inside r0, so sparsity
gains should appear for shell pairs separated by > r0 + few/omega —
present in C12 (29 Bohr) with r0 = 6 Bohr. erfc(0.222 Bohr^-1) reference
shows what production attenuated-MP2 screening looks like. Anchor:
terf + terfc = Coulomb elementwise on the metric for every (r0, omega).

SERIAL: run with OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1.
Needs FERRIC_TERF_TABLE_DIR.

Usage: python scripts/terfc_sparsity_probe.py [xyz] [basis] [auxbasis]
"""
import sys

import numpy as np

import ferric

R0_BOHR = 6.01  # = 3.18 A, the production terf seam
R0W_LIST = [2.0**0.5 / 2.0, 2.0, 4.0, 8.0]  # first entry = linked 1/sqrt2
ERFC_W = 0.2224  # Bohr^-1 (= 0.42 A^-1 production attenuated-MP2)
TAUS = [1e-6, 1e-8, 1e-10]


def sparsity_stats(t, taus, gmax):
    return {tau: float(np.mean(np.abs(t) < tau * gmax)) for tau in taus}


def block_droppable(t3, offs, dims, tau, gmax):
    """Fraction of (mu-shell, nu-shell) pairs whose whole (P,*,*) block is
    below tau*gmax — the unit real screening drops."""
    nsh = len(dims)  # offs may carry a trailing sentinel
    dropped = total = 0
    for a in range(nsh):
        sa = slice(offs[a], offs[a] + dims[a])
        for b in range(a, nsh):
            sb = slice(offs[b], offs[b] + dims[b])
            total += 1
            if np.abs(t3[:, sa, sb]).max() < tau * gmax:
                dropped += 1
    return dropped / total


def main():
    xyz = sys.argv[1] if len(sys.argv) > 1 else "testdata/molecules/alkane_12.xyz"
    basis = sys.argv[2] if len(sys.argv) > 2 else "6-31g"
    auxbasis = sys.argv[3] if len(sys.argv) > 3 else "cc-pvdz-ri"
    mol = ferric.Molecule.from_xyz(xyz)
    obs = ferric.BasisSet.bundled(basis)
    aux = ferric.BasisSet.bundled(auxbasis)

    # Raw AO-side tensors via identity "MO" coefficients (blocked transform).
    rhf_dummy_dim = None
    v_c = ferric.compute_metric_2c(mol, obs, aux)
    naux = v_c.shape[0]
    # nbas from shell_info of the orbital basis.
    o_cent, o_offs, o_dims = ferric.shell_info(mol, obs)
    nbas = int(max(o + d for o, d in zip(o_offs, o_dims)))
    eye = np.eye(nbas)

    print(f"# terfc sparsity probe  {xyz}  {basis}/{auxbasis}  nbas={nbas} naux={naux}")
    print(f"# r0 = {R0_BOHR} Bohr fixed; thresholds relative to each tensor's own max\n")

    ops = [("coulomb", {}), (f"erfc w={ERFC_W}", dict(operator="erfc", omega=ERFC_W))]
    for r0w in R0W_LIST:
        w = r0w / R0_BOHR
        label = "terfc linked" if abs(r0w - 2**0.5 / 2) < 1e-12 else f"terfc r0w={r0w}"
        ops.append((label, dict(operator="terfc", omega=(None if abs(r0w - 2**0.5/2) < 1e-12 else w), r0=R0_BOHR)))

    print(f"{'operator':>16} {'metric<1e-8':>12} {'eri3<1e-6':>10} {'eri3<1e-8':>10} {'eri3<1e-10':>11} {'shpair@1e-8':>12}")
    for label, kw in ops:
        v = ferric.compute_metric_2c(mol, obs, aux, **kw) if kw else v_c
        t3 = ferric.compute_eri3_mo(mol, obs, aux, eye, eye, **kw)
        g3 = np.abs(t3).max()
        st = sparsity_stats(t3, TAUS, g3)
        mfrac = float(np.mean(np.abs(v) < 1e-8 * np.abs(v).max()))
        blk = block_droppable(t3, o_offs, o_dims, 1e-8, g3)
        print(f"{label:>16} {mfrac:>12.3f} {st[1e-6]:>10.3f} {st[1e-8]:>10.3f} {st[1e-10]:>11.3f} {blk:>12.3f}", flush=True)
        del t3, v

    # Anchor: split identity on the metric at the sharpest decoupled point.
    w = R0W_LIST[-1] / R0_BOHR
    v_sr = ferric.compute_metric_2c(mol, obs, aux, operator="terfc", omega=w, r0=R0_BOHR)
    v_lr = ferric.compute_metric_2c(mol, obs, aux, operator="terf", omega=w, r0=R0_BOHR)
    dev = np.abs(v_sr + v_lr - v_c).max() / np.abs(v_c).max()
    print(f"\n# anchor: max|terf+terfc-coulomb|/max|coulomb| at r0w={R0W_LIST[-1]} = {dev:.3e}")


if __name__ == "__main__":
    main()
