"""Iteration 22 drivers: Gamma periodic-ECP forces (pbc_grad_ecp.py) + the LANL2DZ pin-range check.

System: HI, H STO-3G, I LANL2DZ basis + LANL2DZ ECP (46 core e-, local + s/p/d projectors, Z_eff 7), nao 9 (cart),
the Iteration 14 system.  Anchor cell 6x6x7 with the atoms moved OFF the common axis (H (0.3, 0.2, 0.4),
I (0.9, -0.4, 3.3)) so no force component is zero by symmetry.  Pure AFT (w = None), gcut 12 for the FD anchors
(any G sphere is a consistent energy; the FD anchor needs consistency, not convergence).  ECP image sets frozen:
Lorb, Lecp from pbc_ecp.ecp_ranges(prec = 1e-10) at the REFERENCE geometry (r_ecp 15.50, r_orb 36.41 Bohr,
81 / 911 images), reused unchanged at +-h.

Predictions (written BEFORE running; FINDINGS Iteration 22 repeats them):
  term   analytic ECP term vs central FD (h = 1e-4) of E_ECP(R) = sum D V_ECP(R) at FIXED D (the SCF D):
         physics  -> agreement at the FD truncation level of a smooth Gaussian integral, <= 1e-10.
         artifact -> a wrong centre term / missing images leaves an O(1e-3 .. 1e-1) residual.
  full   analytic total force vs central FD (h = 1e-4) of the prototype's own SCF energy, exxdiv none AND ewald:
         physics  -> the Iteration 16 floor (~2-3e-9, h^2 f'''/6), |sum F| ~ 1e-14, F(ewald) == F(none) ~ 1e-12
                     (the ECP is one-electron, so the Madelung cancellation of Iteration 16 is untouched).
  mutants (ECP term only; each must exceed the full-anchor floor):
         no_centre   drop the ECP-centre derivative  -> O(1e-1) on I, AND breaks sum F (it is the only piece
                     of a translation-invariant triple that moves) -- the one ECP mutant sum F CAN see.
         centre_sign +(bra+ket) instead of -(bra+ket) -> twice no_centre's miss, breaks sum F.
         L0_only     ket images unshifted (orbital image L = 0 only in the derivative, all M) -> O(1e-2)
                     (the H 1s / I 5p tails reach neighbouring cells in a 6-Bohr box); sum F BLIND (each kept
                     triple is still translation invariant).
         M0_only     home ECP image only in the derivative -> O(1e-2 .. 1e-1); sum F BLIND, same reason.
         ket_transpose (NOT a mutant: the S/T shortcut ket := bra^T, valid for a symmetric full image set):
                     predicted equal to the direct ket term at the truncation level of the frozen sets (<= 1e-9).
  molterm big box (a = 30): our periodic ECP gradient term at the PySCF molecular D vs PySCF's molecular ECP gradient
         term (bra ECPscalar_ipnuc + centre ECPscalar_iprinv at the rinv origin: a different code path for the centre):
         physics -> <= 1e-12 (images at >= 30 Bohr are e^-40-small).  artifact -> an image/centre bookkeeping error
         shows as an O(1e-2) mismatch.
  box    total force in a cubic box a -> PySCF molecular RHF gradient (same basis + ECP, cart), exxdiv = ewald:
         g_box - g_mol = c3'/a^3 + c5'/a^5, c3' = d/dR [-(4pi/3) Omega_I - (2pi/3)|p(Z_eff)|^2] (FD of the
         molecular Iteration 14 prediction).  artifact -> an ECP-term error is a-INDEPENDENT (the ECP is short range),
         so a^3 * residual would diverge like a^3 instead of settling.
  pinrange the LANL2DZ 1x1x2 pin (-10.388991762015 none / -11.360192112699 ewald) vs larger lattice-sum ranges.
         Hypothesis under test (FINDINGS "Rust periodic ECP -- measured"): ferric - prototype = -2.9e-7 because the
         prototype's 1e range is too short.  If true: enlarging the responsible range moves E by ~ -2.9e-7.  If the
         ranges are converged: every variant moves E by < 1e-9 and the offset is on the Rust side (or elsewhere).
         NOTE the pin was made with rcut_1e = pbc_ecp.rcut_overlap = 26.74 Bohr (not 22), pair-FT image range
         sqrt(2 ln(1/thresh)/a_min) + 2 = 26.74 Bohr at thresh 1e-14, ECP ranges at prec 1e-14 (18.3 / 43.1).

Usage: python3 run_grad_ecp_anchor.py term | full | mutants | molterm | box a1 a2 ... | pinrange [variant ...]

STATUS 2026-09-25 (stopped early on request): pinrange, the term-level anchor (image subsets), the mutants (term
level) and molterm were run; `full` (SCF FD anchor) and `box` were NOT run.  See FINDINGS Iteration 22.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_ecp as PE  # noqa: E402
import pbc_kpts as PK  # noqa: E402

BASIS = {"H": "sto-3g", "I": "lanl2dz"}
ECP = {"I": "lanl2dz"}
MOL = [("H", (0.0, 0.0, 0.0)), ("I", (0.0, 0.0, 3.04))]
CELL_A = np.diag([6.0, 6.0, 7.0])
MOVED = [("H", (0.3, 0.2, 0.4)), ("I", (0.9, -0.4, 3.3))]
PIN_ATOMS = [("H", (0.3, 0.2, 0.4)), ("I", (0.3, 0.2, 3.44))]  # run_ecp_anchor.CELL_ATOMS
PIN = (-10.388991762015, -11.360192112699)
NELEC = 8
COMPS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
GCUT = 12.0


def anchor_cell():
    return PE.EcpCell(CELL_A, MOVED, BASIS, ECP)


# ------------------------------------------------------------------------------------------ anchors
def _setup():
    import pbc_grad_ecp as GE

    cell = anchor_cell()
    imgs = GE.frozen_images(cell, prec=1e-10)
    rcut_1e = PE.rcut_overlap(cell)
    print(f"frozen images: r_ecp {imgs['rcut_ecp']:.2f} r_orb {imgs['rcut_orb']:.2f} nM {len(imgs['Lecp'])} "
          f"nL {len(imgs['Lorb'])}; rcut_1e {rcut_1e:.2f}; gcut {GCUT}", flush=True)
    return GE, cell, imgs, rcut_1e


def term():
    GE, cell, imgs, rcut_1e = _setup()
    t = time.time()
    ref = GE.solve(cell, imgs, GCUT, rcut_1e, exxdivs=("none",))["none"]
    D = ref["D"]
    blocks = GE.ecp_grad_blocks(cell, imgs, D)
    g = GE.assemble_ecp_grad(blocks, cell)
    print(f"analytic ECP term (D fixed): {np.array2string(g, precision=10)} ({time.time() - t:.0f}s)", flush=True)
    for h in (1e-4, 5e-5):
        fd = GE.fd_ecp_term(cell, imgs, D, COMPS, h)
        res = max(abs(g[k] - v) for k, v in fd.items())
        print(f"  h {h:.0e}: max|analytic - FD(E_ECP at fixed D)| = {res:.2e}  "
              + " ".join(f"{k}:{g[k] - v:+.1e}" for k, v in fd.items()), flush=True)
    print(f"  V_ECP asymmetry at the frozen sets: {blocks['asym']:.1e}; sum F_ECP {abs(g.sum(0)).max():.1e}")
    for m in ("no_centre", "centre_sign", "L0_only", "M0_only", "ket_transpose"):
        gm = GE.assemble_ecp_grad(blocks, cell, mutant=m)
        print(f"  {m:13s}: max|g_m - g| = {abs(gm - g).max():.2e}  |sum F| = {abs(gm.sum(0)).max():.1e}")


def full():
    GE, cell, imgs, rcut_1e = _setup()
    t = time.time()
    ref = GE.solve(cell, imgs, GCUT, rcut_1e)
    out = {}
    for ex in ("none", "ewald"):
        r = ref[ex]
        g, parts, blocks = GE.gamma_ecp_grad(cell, imgs, r, GCUT, rcut_1e)
        out[ex] = (g, parts, blocks)
        print(f"{ex}: E {r['e']:.12f} scf err {r['err']:.1e} it {r['it']}; grad\n{np.array2string(g, precision=10)}"
              f"\n  ECP term {np.array2string(parts['ecp'], precision=8)} |sum F| {abs(g.sum(0)).max():.1e}", flush=True)
    print(f"F(ewald) - F(none) = {abs(out['ewald'][0] - out['none'][0]).max():.1e}; "
          f"E(ewald) - E(none) = {ref['ewald']['e'] - ref['none']['e']:.10f} (-v_M N/2 = "
          f"{-ref['ewald']['vM'] * NELEC / 2:.10f})  ({time.time() - t:.0f}s)", flush=True)
    # cross-check the chunked non-ECP assembly against pbc_grad_open.gamma_grad (dense P) on the same D
    x = GE.crosscheck_dense(cell, imgs, ref["none"], GCUT, rcut_1e)
    print(f"chunked non-ECP assembly vs pbc_grad_open.gamma_grad: {x:.1e}", flush=True)
    t = time.time()
    fd = GE.fd_total(cell, imgs, GCUT, rcut_1e, COMPS, 1e-4, guess=ref["none"]["D"])
    for ex in ("none", "ewald"):
        g = out[ex][0]
        res = max(abs(g[k] - v[ex]) for k, v in fd.items())
        print(f"{ex}: max|analytic - FD| = {res:.2e}  " + " ".join(f"{k}:{g[k] - v[ex]:+.1e}" for k, v in fd.items()),
              flush=True)
        for m in ("no_centre", "centre_sign", "L0_only", "M0_only", "ket_transpose"):
            gm = g - out[ex][1]["ecp"] + GE.assemble_ecp_grad(out[ex][2], cell, mutant=m)
            rm = max(abs(gm[k] - v[ex]) for k, v in fd.items())
            print(f"    {m:13s}: max|analytic - FD| = {rm:.2e}  |sum F| = {abs(gm.sum(0)).max():.1e}", flush=True)
    print(f"FD {time.time() - t:.0f}s")


def molterm(a=30.0):
    import pbc_grad_ecp as GE

    ref = PE.molecular_reference(MOL, BASIS, ECP, conv=1e-12)
    mol = ref["mf"].mol
    D = ref["mf"].make_rdm1()
    g_mol = GE.molecular_ecp_term(mol, D)
    fd = GE.molecular_ecp_term_fd(MOL, BASIS, ECP, D, 1e-4)
    print(f"PySCF molecular ECP term (ipnuc + iprinv):\n{np.array2string(g_mol, precision=10)}\n"
          f"  vs FD of sum D V_ECP(mol) (h 1e-4): {abs(g_mol - fd).max():.1e}", flush=True)
    shift = np.array([0.37, 0.21, 0.5 * a - 1.52])
    cell = PE.EcpCell(np.eye(3) * a, [(s, np.asarray(r) + shift) for s, r in MOL], BASIS, ECP)
    imgs = GE.frozen_images(cell, prec=1e-14)
    blocks = GE.ecp_grad_blocks(cell, imgs, D)
    g = GE.assemble_ecp_grad(blocks, cell)
    print(f"box a={a}: nM {len(imgs['Lecp'])} nL {len(imgs['Lorb'])}; periodic ECP term vs molecular: "
          f"{abs(g - g_mol).max():.1e}", flush=True)
    for m in ("no_centre", "centre_sign"):
        print(f"  {m}: {abs(GE.assemble_ecp_grad(blocks, cell, mutant=m) - g_mol).max():.1e}")


def box(alist, prec=1e-10):
    import pbc_grad_ecp as GE

    ref = PE.molecular_reference(MOL, BASIS, ECP, conv=1e-12)
    gm = GE.molecular_grad(ref["mf"])
    c3p = GE.c3_prime(MOL, BASIS, ECP, h=1e-3)
    print(f"molecular grad (PySCF) H z {gm[0, 2]:.10f} I z {gm[1, 2]:.10f}; predicted c3' (H z) {c3p[0, 2]:.6f} "
          f"(I z {c3p[1, 2]:.6f})", flush=True)
    for a in alist:
        t = time.time()
        shift = np.array([0.37, 0.21, 0.5 * a - 1.52])
        cell = PE.EcpCell(np.eye(3) * a, [(s, np.asarray(r) + shift) for s, r in MOL], BASIS, ECP)
        imgs = GE.frozen_images(cell, prec=1e-14)
        gcut = PK.aft_gcut(cell, prec)
        r = GE.solve(cell, imgs, gcut, PE.rcut_overlap(cell), exxdivs=("ewald",), guess=None)["ewald"]
        g, parts, _ = GE.gamma_ecp_grad(cell, imgs, r, gcut, PE.rcut_overlap(cell))
        d = g[0, 2] - gm[0, 2]
        print(f"a {a:5.1f}: E-E_mol {r['e'] - ref['e']:+.6e} g_box-g_mol (H z) {d:+.6e} a^3*d {d * a**3:+.6f} "
              f"(I z {g[1, 2] - gm[1, 2]:+.3e}; transverse max {abs(g[:, :2]).max():.1e}; |sum F| "
              f"{abs(g.sum(0)).max():.1e}; ECP term H z {parts['ecp'][0, 2]:+.3e}) ({time.time() - t:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------ LANL2DZ pin
def pinrange(variants):
    """Rebuild the Iteration 14 oracle system (1x1x2, pure AFT at prec 1e-14) with larger lattice-sum ranges."""
    cell = PE.EcpCell(CELL_A, PIN_ATOMS, BASIS, ECP)
    r_e, r_o, _, _ = PE.ecp_ranges(cell)
    print(f"pin ranges: rcut_1e {PE.rcut_overlap(cell):.2f}, ECP r_ecp {r_e:.2f} r_orb {r_o:.2f}, pair-FT "
          f"{np.sqrt(2 * np.log(1e14) / 0.1053) + 2:.2f} (thresh 1e-14)", flush=True)
    opts = {
        "base": dict(),
        "rcut1e_30": dict(rcut_1e=30.0),
        "rcut1e_40": dict(rcut_1e=40.0),
        "ecp_wide": dict(img=dict(rcut_ecp=24.0, rcut_orb=55.0)),
        "pair_1e-18": dict(thresh=1e-18),
    }
    for name in variants or ["base", "rcut1e_30", "rcut1e_40", "ecp_wide", "pair_1e-18"]:
        o = dict(opts[name])
        t = time.time()
        img = PE.ecp_images(cell, **o.pop("img")) if "img" in o else None
        kb = PE.build_k_ecp(cell, (1, 1, 2), img=img, verbose=False, **o)
        es = []
        for ex, pin in zip(("none", "ewald"), PIN):
            e = PK.krhf(kb, NELEC, conv=1e-12, kshift=kb["madelung"] if ex == "ewald" else 0.0)[0]
            es.append(e)
        print(f"{name:11s}: E none {es[0]:.12f} (- pin {es[0] - PIN[0]:+.2e}), ewald {es[1]:.12f} "
              f"(- pin {es[1] - PIN[1]:+.2e})  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "term":
        term()
    elif cmd == "full":
        full()
    elif cmd == "molterm":
        molterm()
    elif cmd == "box":
        box([float(x) for x in sys.argv[2:]])
    elif cmd == "pinrange":
        pinrange(sys.argv[2:])
