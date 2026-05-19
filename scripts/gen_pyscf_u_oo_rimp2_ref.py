"""
Validation harness for U-OO-RI-MP2 (task #49).

This script provides a **functional U-HF + U-RI-MP2 energy** evaluated at
arbitrary (C_α, C_β), and a **finite-difference orbital gradient** that
serves as the ground truth for validating an analytic gradient implemented
in Rust.

Why FD-only here: deriving the U-OO-MP2 analytic gradient from scratch in
Python and Rust simultaneously doubles the bug surface. The Python script
just needs to be *correct*; speed doesn't matter. FD is unambiguous.

Workflow:
  1. Match the Rust U-RI-MP2 energy to ~6e-6 vs PySCF UMP2 (already done).
  2. Implement analytic gradient in Rust.
  3. Run this script's FD gradient on OH/cc-pVDZ; compare to Rust analytic.
  4. When matched at 1e-5, ship.

Reference: Bozkaya, JCP 139, 154105 (2013).
"""
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, expm

sys.path.insert(0, "/home/matt/qc/pyscf")
from pyscf import df, gto, scf, mp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(ROOT, "testdata/reference"), exist_ok=True)


# ----------------------------------------------------------------------------
# Energy infrastructure
# ----------------------------------------------------------------------------

@dataclass
class MolCache:
    """Pre-computed AO-basis quantities that don't depend on MO rotations."""
    mol: object
    H: np.ndarray                 # core hamiltonian (nao, nao)
    eri4: np.ndarray              # 4-index ERIs (nao,)*4
    S: np.ndarray                 # overlap (nao, nao)
    auxmol: object
    pmunu: np.ndarray             # (naux, nao, nao) RI 3-index (P|μν)
    vinv: np.ndarray              # (naux, naux) V^{-1/2}
    nocc_a: int
    nocc_b: int
    e_nuc: float

    @classmethod
    def from_mol(cls, mol, aux_name):
        H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        eri4 = mol.intor("int2e", aosym="s1").reshape((mol.nao,)*4)
        S = mol.intor("int1e_ovlp")
        auxmol = df.addons.make_auxmol(mol, auxbasis=aux_name)
        nao = mol.nao
        naux = auxmol.nao
        pmunu = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        pmunu = pmunu.reshape(nao, nao, naux).transpose(2, 0, 1)
        v2c = auxmol.intor("int2c2e")
        w, u = eigh(v2c)
        vinv = u @ np.diag(1.0/np.sqrt(w)) @ u.T
        nelec = mol.nelectron
        spin = mol.spin
        nocc_a = (nelec + spin) // 2
        nocc_b = (nelec - spin) // 2
        return cls(mol=mol, H=H, eri4=eri4, S=S, auxmol=auxmol, pmunu=pmunu,
                   vinv=vinv, nocc_a=nocc_a, nocc_b=nocc_b,
                   e_nuc=mol.energy_nuc())


def hf_energy_and_fock(cache, C_a, C_b):
    """UHF energy + α/β Fock in AO basis at given MOs."""
    D_a = C_a[:, :cache.nocc_a] @ C_a[:, :cache.nocc_a].T
    D_b = C_b[:, :cache.nocc_b] @ C_b[:, :cache.nocc_b].T
    D_tot = D_a + D_b
    J = np.einsum("mnls,ls->mn", cache.eri4, D_tot)
    K_a = np.einsum("mlns,ls->mn", cache.eri4, D_a)
    K_b = np.einsum("mlns,ls->mn", cache.eri4, D_b)
    F_a = cache.H + J - K_a
    F_b = cache.H + J - K_b
    E = 0.5 * (np.einsum("mn,mn->", cache.H + F_a, D_a)
               + np.einsum("mn,mn->", cache.H + F_b, D_b))
    return E + cache.e_nuc, F_a, F_b


def build_b_full(cache, C):
    """Full-MO B tensor: B^P_{pq} = Σ_Q V^{-1/2}_{PQ} Σ_{μν} (Q|μν) C_μp C_νq."""
    pmo = np.einsum("Pmn,mp,nq->Ppq", cache.pmunu, C, C, optimize=True)
    return np.einsum("PQ,Qpq->Ppq", cache.vinv, pmo)


def umpt2_corr_and_amplitudes(b_full_a, b_full_b, eps_a, eps_b, nocc_a, nocc_b):
    """U-MP2 correlation energy from full-MO B tensors and canonical orbital energies.

    Returns dict with t_aa[i,a,j,b], t_bb[I,A,J,B], t_ab[i,a,J,B] and e_corr.

    Conventions:
      αα same-spin energy: E_aa = (1/4) Σ t_aa · ⟨ij||ab⟩
      ββ same-spin energy: E_bb = (1/4) Σ t_bb · ⟨IJ||AB⟩
      αβ opposite-spin:    E_ab = Σ t_ab · (ia|JB)   [no antisymmetrization]
      ⟨ij||ab⟩ = (ia|jb) - (ib|ja)
      t = (driving integral) / (ε_i + ε_j - ε_a - ε_b)
    """
    eo_a, ev_a = eps_a[:nocc_a], eps_a[nocc_a:]
    eo_b, ev_b = eps_b[:nocc_b], eps_b[nocc_b:]
    b_a_ov = b_full_a[:, :nocc_a, nocc_a:]
    b_b_ov = b_full_b[:, :nocc_b, nocc_b:]

    # αα block
    iajb_aa = np.einsum("Pia,Pjb->iajb", b_a_ov, b_a_ov, optimize=True)
    K_aa = iajb_aa - iajb_aa.transpose(0, 3, 2, 1)        # (ia|jb)-(ib|ja)
    delta_aa = (eo_a[:, None, None, None] + eo_a[None, None, :, None]
                - ev_a[None, :, None, None] - ev_a[None, None, None, :])
    t_aa = K_aa / delta_aa
    e_aa = 0.25 * np.einsum("iajb,iajb->", t_aa, K_aa)

    # ββ block
    iajb_bb = np.einsum("PIA,PJB->IAJB", b_b_ov, b_b_ov, optimize=True)
    K_bb = iajb_bb - iajb_bb.transpose(0, 3, 2, 1)
    delta_bb = (eo_b[:, None, None, None] + eo_b[None, None, :, None]
                - ev_b[None, :, None, None] - ev_b[None, None, None, :])
    t_bb = K_bb / delta_bb
    e_bb = 0.25 * np.einsum("iajb,iajb->", t_bb, K_bb)

    # αβ block (no antisymmetrization)
    iajb_ab = np.einsum("Pia,PJB->iaJB", b_a_ov, b_b_ov, optimize=True)
    delta_ab = (eo_a[:, None, None, None] + eo_b[None, None, :, None]
                - ev_a[None, :, None, None] - ev_b[None, None, None, :])
    t_ab = iajb_ab / delta_ab
    e_ab = np.einsum("iajb,iajb->", t_ab, iajb_ab)

    return {
        "e_corr": e_aa + e_bb + e_ab,
        "e_aa": e_aa, "e_bb": e_bb, "e_ab": e_ab,
        "t_aa": t_aa, "t_bb": t_bb, "t_ab": t_ab,
        "K_aa": K_aa, "K_bb": K_bb, "iajb_ab": iajb_ab,
    }


def total_energy_at_orbitals(cache, C_a, C_b):
    """E_total = E_HF(C_a, C_b) + E_MP2 using eigh(F,S) for canonical orbital energies.

    The MP2 expression uses orbital energies from the *current* Fock matrix
    (not the original HF Fock), which is what OO-MP2 requires — at each
    iteration the Fock changes and so do the denominators.
    """
    E_hf, F_a, F_b = hf_energy_and_fock(cache, C_a, C_b)
    # Canonical orbital energies via eigh(F_σ, S). Re-diagonalize to get
    # the canonical α/β orbitals; the OO loop's job is to drive the off-
    # diagonal F_ai block (after rotation) toward zero.
    eps_a_can, _ = eigh(F_a, cache.S)
    eps_b_can, _ = eigh(F_b, cache.S)
    # IMPORTANT: build B-tensor and amplitudes in the *given* (non-canonical)
    # MO basis with orbital energies = diagonal of F_mo (semi-canonical
    # OO-MP2 uses canonical denominators; we follow Bozkaya 2013 which uses
    # diagonal F_mo as denominators).
    b_a = build_b_full(cache, C_a)
    b_b = build_b_full(cache, C_b)
    F_mo_a = C_a.T @ F_a @ C_a
    F_mo_b = C_b.T @ F_b @ C_b
    eps_a = np.diag(F_mo_a)
    eps_b = np.diag(F_mo_b)
    mp2 = umpt2_corr_and_amplitudes(b_a, b_b, eps_a, eps_b, cache.nocc_a, cache.nocc_b)
    return E_hf + mp2["e_corr"], E_hf, mp2, F_mo_a, F_mo_b


# ----------------------------------------------------------------------------
# Finite-difference gradient
# ----------------------------------------------------------------------------

def cayley(kappa):
    """Cayley transform of antisymmetric κ: U = (I - κ/2)^-1 (I + κ/2). Exactly unitary."""
    n = kappa.shape[0]
    I = np.eye(n)
    return np.linalg.solve(I - 0.5*kappa, I + 0.5*kappa)


def rotate_mos(C, kappa):
    """Apply κ rotation: C → C · U(κ). Only occ-vir block of κ is non-zero."""
    return C @ cayley(kappa)


def make_kappa(nmo, nocc, ai, value):
    """Antisymmetric κ with κ[a+nocc, i] = value, κ[i, a+nocc] = -value.

    `ai` = (a, i) where a indexes virtuals (0..nvir), i indexes occupied (0..nocc).
    """
    a, i = ai
    K = np.zeros((nmo, nmo))
    K[nocc + a, i] = value
    K[i, nocc + a] = -value
    return K


def fd_gradient(cache, C_a, C_b, spin, h=1e-4):
    """Finite-difference gradient g_σ[a, i] = ∂E_total/∂κ_σ[a, i].

    Returns shape (nvir_σ, nocc_σ).
    """
    nmo = C_a.shape[1]
    if spin == 'a':
        nocc, nvir = cache.nocc_a, nmo - cache.nocc_a
    else:
        nocc, nvir = cache.nocc_b, nmo - cache.nocc_b
    g = np.zeros((nvir, nocc))
    E0, _, _, _, _ = total_energy_at_orbitals(cache, C_a, C_b)
    for a in range(nvir):
        for i in range(nocc):
            K = make_kappa(nmo, nocc, (a, i), h)
            if spin == 'a':
                C_a_p = rotate_mos(C_a, K)
                C_a_m = rotate_mos(C_a, -K)
                E_p, *_ = total_energy_at_orbitals(cache, C_a_p, C_b)
                E_m, *_ = total_energy_at_orbitals(cache, C_a_m, C_b)
            else:
                C_b_p = rotate_mos(C_b, K)
                C_b_m = rotate_mos(C_b, -K)
                E_p, *_ = total_energy_at_orbitals(cache, C_a, C_b_p)
                E_m, *_ = total_energy_at_orbitals(cache, C_a, C_b_m)
            g[a, i] = (E_p - E_m) / (2*h)
    return g


# ----------------------------------------------------------------------------
# Analytic gradient — DEFERRED. Initial derivation (Bozkaya 2013-style 4-term
# integral-response form) failed FD validation: max-elementwise diff ≈ 0.027
# vs ‖g_fd‖ ≈ 0.042 on OH/cc-pVDZ, with a best-scalar-fit factor that varies
# between spins (0.43 for α, 0.61 for β), indicating multiple wrong terms
# not a single missed prefactor.
#
# Plan: implement analytic gradient in Rust directly from Bozkaya 2013 with
# FD validation at each step. The Python FD output (above) is the ground
# truth — it doesn't need a Python analytic counterpart.
#
# Stashed but unused:
def _stashed_analytic_gradient_DO_NOT_USE(cache, C_a, C_b, mp2_data, F_mo_a, F_mo_b, spin):
    """Analytic orbital gradient g_σ[a, i] for U-OO-MP2.

    Bozkaya 2013: at a stationary point of the Hylleraas functional w.r.t.
    amplitudes (which our exact t solves give), the orbital gradient simplifies
    to the response of (ia|jb) integrals to orbital rotation, plus the HF
    Brillouin term −2·F_ai.

    For α-spin rotation κ_{ck}^α (c=virtual α, k=occupied α):
      ∂E/∂κ_{ck}^α = -2·F_{ck}^α        (Brillouin)
        + 2·(αα MP2 response)
        + 1·(αβ MP2 response, α-side only)

    αα response (closed-shell-style with antisymmetrized integral):
      Σ_{ijab} t^αα_{ij,ab} · ∂⟨ij||ab⟩/∂κ_{ck}^α
        where ∂(ia|jb)/∂κ_{ck} = δ_ik(ca|jb) + δ_jk(ia|cb) - δ_ac(ik|jb) - δ_bc(ia|jk)

    αβ response (only α-side of (ia|JB) feels κ^α):
      Σ_{iaJB} t^αβ_{i,a,J,B} · ∂(ia|JB)/∂κ_{ck}^α
        = Σ_{aJB} t^αβ_{k,a,J,B} (ca|JB) - Σ_{iJB} t^αβ_{i,c,J,B} (ik|JB)

    Returns shape (nvir_σ, nocc_σ).
    """
    if spin == 'a':
        no_p, F_p, C_p = cache.nocc_a, F_mo_a, C_a
        no_q = cache.nocc_b
        b_full_p = build_b_full(cache, C_a)
        b_full_q = build_b_full(cache, C_b)
        t_ss = mp2_data["t_aa"]
        t_os = mp2_data["t_ab"]               # (i_α, a_α, J_β, B_β)
    else:
        no_p, F_p, C_p = cache.nocc_b, F_mo_b, C_b
        no_q = cache.nocc_a
        b_full_p = build_b_full(cache, C_b)
        b_full_q = build_b_full(cache, C_a)
        t_ss = mp2_data["t_bb"]
        # For β-side of αβ amplitudes, transpose to (I_β, A_β, j_α, b_α) order.
        t_os = mp2_data["t_ab"].transpose(2, 3, 0, 1)
    nmo = b_full_p.shape[1]
    nv = nmo - no_p

    # Same-spin: B-blocks of σ.
    B_oo = b_full_p[:, :no_p, :no_p]
    B_ov = b_full_p[:, :no_p, no_p:]
    B_vv = b_full_p[:, no_p:, no_p:]

    # δ_ik term: Σ_{jab} t[k,a,j,b] · [(ca|jb) - (cb|ja)]
    eri_cajb = np.einsum("Pca,Pjb->cajb", B_vv, B_ov, optimize=True)
    g_t1 = (np.einsum("kajb,cajb->ck", t_ss, eri_cajb, optimize=True)
            - np.einsum("kbja,cajb->ck", t_ss, eri_cajb, optimize=True))

    # δ_jk term: Σ_{iab} t[i,a,k,b] · [(ia|cb) - (ib|ca)]
    eri_iacb = np.einsum("Pia,Pcb->iacb", B_ov, B_vv, optimize=True)
    g_t2 = (np.einsum("iakb,iacb->ck", t_ss, eri_iacb, optimize=True)
            - np.einsum("ibka,iacb->ck", t_ss, eri_iacb, optimize=True))

    # -δ_ac term: -Σ_{ijb} t[i,c,j,b] · [(ik|jb) - (ib|jk)]
    eri_ikjb = np.einsum("Pik,Pjb->ikjb", B_oo, B_ov, optimize=True)
    eri_ibjk = np.einsum("Pib,Pjk->ibjk", B_ov, B_oo, optimize=True)
    g_t3 = -(np.einsum("icjb,ikjb->ck", t_ss, eri_ikjb, optimize=True)
             - np.einsum("ibjc,ibjk->ck", t_ss, eri_ibjk, optimize=True))

    # -δ_bc term: -Σ_{ija} t[i,a,j,c] · [(ia|jk) - (ik|ja)]
    eri_iajk = np.einsum("Pia,Pjk->iajk", B_ov, B_oo, optimize=True)
    eri_ikja = np.einsum("Pik,Pja->ikja", B_oo, B_ov, optimize=True)
    g_t4 = -(np.einsum("iajc,iajk->ck", t_ss, eri_iajk, optimize=True)
             - np.einsum("icja,ikja->ck", t_ss, eri_ikja, optimize=True))

    g_ss = 0.5 * (g_t1 + g_t2 + g_t3 + g_t4)  # outer factor 1/2 from (1/4)·2 for t·∂K

    # Opposite-spin: only α-side (i,a) of (ia|JB) feels κ^α.
    # ∂(ia|JB)/∂κ_{ck} = δ_ik (ca|JB) - δ_ac (ik|JB)
    # → Σ_{aJB} t[k,a,J,B] (ca|JB) - Σ_{iJB} t[i,c,J,B] (ik|JB)
    B_q_ov = b_full_q[:, :no_q, no_q:]
    eri_caJB = np.einsum("Pca,PJB->caJB", B_vv, B_q_ov, optimize=True)
    eri_ikJB = np.einsum("Pik,PJB->ikJB", B_oo, B_q_ov, optimize=True)
    g_os = (np.einsum("kaJB,caJB->ck", t_os, eri_caJB, optimize=True)
            - np.einsum("icJB,ikJB->ck", t_os, eri_ikJB, optimize=True))

    # Brillouin (HF orbital gradient): -2·F^σ_ck. The factor 2 comes from
    # the AO→MO chain when differentiating the HF energy.
    g_hf = -2.0 * F_p[no_p:, :no_p]

    return g_hf + g_ss + g_os


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def emit_fd_reference(atom, basis, aux, charge, spin, stub):
    """Write E_total + FD gradient at HF orbitals to JSON. Used by ferric tests."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin,
                unit="angstrom", verbose=0)
    mf = scf.UHF(mol); mf.kernel()
    assert mf.converged
    cache = MolCache.from_mol(mol, aux)
    F_ao = mf.get_fock()
    _, C_a = eigh(F_ao[0], cache.S)
    _, C_b = eigh(F_ao[1], cache.S)
    E_tot, E_hf, mp2_data, F_mo_a, F_mo_b = total_energy_at_orbitals(cache, C_a, C_b)
    print(f"\n=== {stub}  atom={atom!r} basis={basis} ===")
    print(f"E_hf = {E_hf:.10f},  E_corr = {mp2_data['e_corr']:.10f},  E_total = {E_tot:.10f}")
    print("Computing FD gradient (h=1e-4) — slow but exact...")
    grad = {}
    for which in ['a', 'b']:
        nocc = cache.nocc_a if which == 'a' else cache.nocc_b
        if nocc == 0:
            print(f"[{which}] skipped (no occupied orbitals)")
            continue
        g_fd = fd_gradient(cache, C_a, C_b, which, h=1e-4)
        grad[which] = g_fd.tolist()
        print(f"[{which}] ‖g_fd‖ = {np.linalg.norm(g_fd):.6e}  (max|elem| {np.max(np.abs(g_fd)):.3e})")
    out = {
        "atom": atom, "basis": basis, "aux_basis": aux,
        "charge": charge, "spin_2s": spin,
        "nocc_a": cache.nocc_a, "nocc_b": cache.nocc_b,
        "e_hf": float(E_hf), "e_corr": float(mp2_data["e_corr"]),
        "e_total": float(E_tot),
        "C_a": C_a.tolist(), "C_b": C_b.tolist(),
        "grad_fd_a": grad.get('a'), "grad_fd_b": grad.get('b'),
        "method": "u-oo-mp2-reference",
        "note": ("E_total and grad at canonical UHF MOs. Use as ground truth "
                 "for the analytic U-OO-MP2 gradient. At HF reference, "
                 "‖grad_HF‖=0; grad here is dominated by the MP2 response.")
    }
    path = os.path.join(ROOT, f"testdata/reference/{stub}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    # Smoke test the U-MP2 energy convention against PySCF first.
    for atom, basis, aux, charge, spin in [
        ("O 0 0 0; H 0 0 0.97", "cc-pvdz", "cc-pvdz-ri", 0, 1),
    ]:
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin,
                    unit="angstrom", verbose=0)
        mf = scf.UHF(mol); mf.kernel()
        cache = MolCache.from_mol(mol, aux)
        F_ao = mf.get_fock()
        _, C_a = eigh(F_ao[0], cache.S)
        _, C_b = eigh(F_ao[1], cache.S)
        E_tot, E_hf, mp2_data, *_ = total_energy_at_orbitals(cache, C_a, C_b)
        pt = mp.UMP2(mf); pt.kernel()
        print(f"smoke: E_corr (our) = {mp2_data['e_corr']:.10f}  "
              f"PySCF UMP2 = {pt.e_corr:.10f}  diff = {abs(mp2_data['e_corr']-pt.e_corr):.2e}")

    # Write FD references to JSON for ferric tests to consume.
    emit_fd_reference("H 0 0 0",            "cc-pvdz", "cc-pvdz-ri", 0, 1, "h_cc-pvdz_u-oomp2-fd")
    emit_fd_reference("O 0 0 0; H 0 0 0.97","cc-pvdz", "cc-pvdz-ri", 0, 1, "oh_cc-pvdz_u-oomp2-fd")
