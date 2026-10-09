"""Stage-5b Rust pins (crates/ferric-pbc/tests/pbc_rohf.rs): Gamma ROHF / ROKS on the prototype's OWN
construction (pbc_dft.PeriodicGrid(cell, 75, 302, D=10, scheme="ssf"), dense pure-AFT I), the grid the
UKS pins of pbc_uks.rs use.  PySCF's pbc.dft.ROKS pins (FINDINGS It. 10) are on PySCF's A1 (50,146) grid,
which ferric does not build, so the tight Rust pins are these.

Each ROKS is started twice -- from the UKS density (the FINDINGS oracle's start) and from the core guess --
and with exxdiv none and ewald, so the pins record whether the state is start-independent and the identity
E_ewald - E_none = -a v_M N/2 at the same density.

Usage: python3 run_roks_ssf_pins.py [tri|hf]

Measured 2026-09-24 (PySCF 2.13.0, system python3):
  tri ROHF triplet  PySCF pbc.scf.ROHF  none -0.581222768976  ewald -1.826096526949 (both guesses; <S2> 2)
                    proto roks('HF')    none: DIIS stagnates (err 3.5e-2 / 5.5e-2) from both starts;
                                        ewald: -1.826096526950 (UHF density), -1.810787564982 (core: trapped, +1.53e-2)
  h2t ROHF triplet  none 0.322229103842, ewald -0.387095266028 (== UHF, N_b = 0; all starts, PySCF and proto)
  tri ROKS SSF (75,302) D=10, 70784 pts, start-independent:
                    LDA,VWN -1.694206925329   PBE -1.731150286924   PBE0 ewald -1.776676719941 / none -1.465458280448"""

import sys
import time

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402
from test_prototype import H2_A, H2_ATOMS, SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "tri"

if which == "hf":
    # ROHF: PySCF 2.13 pbc.scf.ROHF on AFTDF (mesh 61^3) is the oracle -- the same construction that matched
    # our pure-AFT UHF to 1e-12 (FINDINGS It. 6).  The prototype roks() (plain DIIS on the Roothaan Fock)
    # is tried too; for xc='HF' on tri it stagnates (err 4.8e-2 from the UHF density), recorded as such.
    from pyscf.pbc import df as pdf
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    for label, a, atoms, basis, na, nb in (
        ("tri", TRI_A, TRI_ATOMS, SP_BASIS, 3, 1),
        ("h2t", H2_A, H2_ATOMS, "sto-3g", 2, 0),
    ):
        cell = Cell(a, atoms, basis)
        I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        S, h, enn, vm = I["S"], I["h"], I["enn"], I["madelung"]
        jk = pd.dense_jk(I["I"])
        print(f"{label} v_M {vm:.12f}", flush=True)
        pc = pgto.Cell(
            a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=na - nb
        )
        pc.precision = 1e-12
        pc.build()
        for ks, ex in ((0.0, None), (vm, "ewald")):
            u = U.uks(
                S, h, jk, enn, na, nb, None, "HF", kshift=ks, conv=1e-12, staged=True
            )
            for start, g in (("uhf-density", (u["Da"], u["Db"])), ("core", None)):
                try:
                    r = U.roks(
                        S,
                        h,
                        jk,
                        enn,
                        na,
                        nb,
                        None,
                        "HF",
                        kshift=ks,
                        conv=1e-12,
                        guess=g,
                    )
                    print(
                        f"  proto ROHF {label} kshift {ks:.6f} start {start}: E {r['e']:.12f} "
                        f"<S2> {r['s2']:.12f} it {r['it']}",
                        flush=True,
                    )
                except RuntimeError as e:
                    print(
                        f"  proto ROHF {label} kshift {ks:.6f} start {start}: FAILED {e}",
                        flush=True,
                    )
            for gname in ("pyscf-default", "uhf-density"):
                mf = pscf.ROHF(pc, exxdiv=ex)
                mf.with_df = pdf.AFTDF(pc)
                mf.with_df.mesh = [61] * 3
                mf.conv_tol = 1e-12
                dm0 = None if gname == "pyscf-default" else np.array([u["Da"], u["Db"]])
                e = mf.kernel(dm0=dm0)
                print(
                    f"  PySCF ROHF {label} exxdiv={str(ex):5s} guess={gname:13s} E {e:.12f} "
                    f"<S2> {mf.spin_square()[0]:.12f} conv {mf.converged} | UHF E {u['e']:.12f} "
                    f"<S2> {u['s2']:.10f}",
                    flush=True,
                )
    sys.exit(0)

cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
na, nb = 3, 1
I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
S, h, enn, vm = I["S"], I["h"], I["enn"], I["madelung"]
jk = pd.dense_jk(I["I"])
t = time.time()
gr = pd.PeriodicGrid(cell, 75, 302, D=10.0, scheme="ssf")
print(
    f"tri ssf (75,302) npts {gr.size} v_M {vm:.12f} ({time.time() - t:.0f}s)",
    flush=True,
)
for xc in ("LDA,VWN", "PBE", "PBE0"):
    for ks in (0.0, vm):
        u = U.uks(S, h, jk, enn, na, nb, gr, xc, kshift=ks, conv=1e-12, staged=True)
        for start, g in (("uks-density", (u["Da"], u["Db"])), ("core", None)):
            t = time.time()
            try:
                r = U.roks(
                    S, h, jk, enn, na, nb, gr, xc, kshift=ks, conv=1e-12, guess=g
                )
            except RuntimeError as e:
                print(
                    f"  ROKS tri {xc} kshift {ks:.6f} start {start}: FAILED {e}",
                    flush=True,
                )
                continue
            print(
                f"  ROKS tri {xc} kshift {ks:.6f} start {start}: E {r['e']:.12f} <S2> {r['s2']:.12f} "
                f"it {r['it']} ({time.time() - t:.0f}s) | UKS E {u['e']:.12f}",
                flush=True,
            )
