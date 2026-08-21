"""
Prototype: B2PLYP and DSD-PBEP86 double-hybrid DFT.

Double hybrids combine a hybrid-DFT SCF with post-SCF MP2 correlation:
    E_DH = E_KS(modified XC) + scaling * E_MP2_corr

B2PLYP (Grimme, JCP 124, 034108, 2006):
    SCF: 0.53 HF + 0.47 B88 exchange, 0.73 LYP correlation
    Post-SCF: 0.27 * E_MP2(total)

DSD-PBEP86 (Kozuch & Martin, PCCP 13, 20104, 2011):
    SCF: 0.69 HF + 0.31 PBE exchange, 0.44 P86 correlation
    Post-SCF: SCS-MP2 with c_os=0.56, c_ss=0.29
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.environ.get("PYSCF_PATH", os.path.expanduser("~/qc/pyscf")))
from pyscf import gto, dft, mp as pyscf_mp, ao2mo


def water_pyscf(basis="cc-pvdz"):
    return gto.M(
        atom="O 0.0 0.0 0.117176; H 0.0 0.75695 -0.468706; H 0.0 -0.75695 -0.468706",
        basis=basis, unit="Angstrom", verbose=0,
    )


def mp2_spin_components(mf, mol):
    """Decompose MP2 correlation into OS and SS components (closed-shell spatial orbitals).

    Conventions (Grimme SCS-MP2):
      E_OS = sum_{ijab} (ia|jb)^2 / D            (opposite-spin, Coulomb)
      E_SS = sum_{ijab} (ia|jb)[(ia|jb)-(ib|ja)] / D  (same-spin, exchange-subtracted)
      E_MP2 = E_OS + E_SS
    """
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    mo_e = mf.mo_energy
    eo, ev = mo_e[:nocc], mo_e[nocc:]

    eri_mo = ao2mo.full(mol, mf.mo_coeff)
    eri_mo = ao2mo.restore(1, eri_mo, nmo)
    g = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)

    D = (eo[:, None, None, None] + eo[None, None, :, None]
         - ev[None, :, None, None] - ev[None, None, None, :])

    t2 = g / D

    e_os = np.einsum("iajb,iajb->", g, t2)
    g_exch = g.transpose(0, 3, 2, 1)  # (ib|ja) in (i,a,j,b) layout
    e_ss = np.einsum("iajb,iajb->", g, (g - g_exch) / D)

    return e_os, e_ss


# ═══════════════════════════════════════════════════════════════════════════
#  1. B2PLYP
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("B2PLYP / cc-pVDZ on water")
print("=" * 72)

mol = water_pyscf()

mf = dft.RKS(mol)
mf.xc = "0.53*HF + 0.47*B88, 0.73*LYP"
mf.conv_tol = 1e-11
e_ks = mf.kernel()
print(f"  E_KS (B2PLYP SCF)       = {e_ks:.10f} Ha")

pt = pyscf_mp.MP2(mf)
pt.kernel()
e_os, e_ss = mp2_spin_components(mf, mol)

print(f"  E_MP2_corr (PySCF)      = {pt.e_corr:.10f} Ha")
print(f"  E_OS                    = {e_os:.10f} Ha")
print(f"  E_SS                    = {e_ss:.10f} Ha")
print(f"  E_OS + E_SS             = {e_os + e_ss:.10f} Ha")
assert abs(e_os + e_ss - pt.e_corr) < 1e-8, "MP2 decomposition mismatch"

c_pt2 = 0.27
e_b2plyp = e_ks + c_pt2 * (e_os + e_ss)
print(f"\n  B2PLYP = E_KS + {c_pt2} * (E_OS + E_SS)")
print(f"  B2PLYP total            = {e_b2plyp:.10f} Ha")

print()

# ═══════════════════════════════════════════════════════════════════════════
#  2. DSD-PBEP86
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("DSD-PBEP86 / cc-pVDZ on water")
print("=" * 72)

mol2 = water_pyscf()

mf2 = dft.RKS(mol2)
mf2.xc = "0.69*HF + 0.31*PBE, 0.44*P86"
mf2.conv_tol = 1e-11
e_ks2 = mf2.kernel()
print(f"  E_KS (DSD-PBEP86 SCF)   = {e_ks2:.10f} Ha")

pt2 = pyscf_mp.MP2(mf2)
pt2.kernel()
e_os2, e_ss2 = mp2_spin_components(mf2, mol2)

print(f"  E_MP2_corr (PySCF)      = {pt2.e_corr:.10f} Ha")
print(f"  E_OS                    = {e_os2:.10f} Ha")
print(f"  E_SS                    = {e_ss2:.10f} Ha")
assert abs(e_os2 + e_ss2 - pt2.e_corr) < 1e-8

c_os_dsd, c_ss_dsd = 0.56, 0.29
e_corr_dsd = c_os_dsd * e_os2 + c_ss_dsd * e_ss2
e_dsd = e_ks2 + e_corr_dsd
print(f"\n  DSD-PBEP86 = E_KS + {c_os_dsd}*E_OS + {c_ss_dsd}*E_SS")
print(f"  DSD corr                = {e_corr_dsd:.10f} Ha")
print(f"  DSD total               = {e_dsd:.10f} Ha")

print()

# ═══════════════════════════════════════════════════════════════════════════
#  3. Rust implementation plan
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("Rust implementation plan")
print("=" * 72)
print(f"""
The double-hybrid pattern exists in ferric via wB97X-L-V
(crates/ferric-cc/src/double_hybrid.rs). Generalization needs:

1. FUNCTIONAL DEFINITION (libxc.rs friendly_to_libxc):
   - "B2PLYP" -> HYB_GGA_XC_B2PLYP or manual "0.53 HF + 0.47 B88, 0.73 LYP"
   - "DSD-PBEP86" -> manual GGA_X_PBE + GGA_C_P86 with b3lyp_mix = 0.69

2. DOUBLE-HYBRID DRIVER (ferric-mp2/src/double_hybrid.rs):
   - GenericDoubleHybridConfig {{ c_os: f64, c_ss: f64, xc: String }}
   - Takes a converged ScfResult from KS-DFT
   - Calls ri_mp2_spin_components for OS/SS decomposition
   - E_total = E_KS + c_os*E_OS + c_ss*E_SS

3. No new integrals or operators needed — everything exists.

Reference values (water/cc-pVDZ):
  B2PLYP:     E_KS = {e_ks:.10f}  E_corr = {c_pt2*(e_os+e_ss):.10f}  E_total = {e_b2plyp:.10f}
  DSD-PBEP86: E_KS = {e_ks2:.10f}  E_corr = {e_corr_dsd:.10f}  E_total = {e_dsd:.10f}
""")
