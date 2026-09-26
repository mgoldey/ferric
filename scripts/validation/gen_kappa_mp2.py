"""PySCF + numpy references for the VALIDATION.md "κ-MP2" row.

Consumer: crates/ferric-mp2/tests/validation_kappa_mp2.rs
Output:   testdata/reference/validation/kappa_mp2/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from `spin_components_from_b_ov_kappa` in
crates/ferric-mp2/src/rimp2.rs, not assumed): closed-shell RI-MP2 on an RHF
reference, all electrons correlated (`RiMp2Config::default().frozen_core == 0`),
Coulomb-metric density fitting, with every (i,a,j,b) term of BOTH spin
components multiplied by

    damp(Δ) = (1 − exp(−κ Δ))²,   Δ = ε_a + ε_b − ε_i − ε_j > 0,

so that

    E_OS(κ) = −Σ_ijab damp · (ia|jb)² / Δ
    E_SS(κ) = −Σ_ijab damp · (ia|jb)[(ia|jb) − (ib|ja)] / Δ

(Lee & Head-Gordon, JCTC 14, 5203 (2018) κ-regularizer; κ in 1/Hartree).

THE REFERENCE, built independently of ferric's code path:
  * RHF: PySCF `scf.RHF`, exact four-centre J/K (ferric's `RhfConfig::default()`
    has no SCF fitting either), conv_tol 1e-12 / conv_tol_grad 1e-10, ferric's
    bundled orbital basis and geometry in Bohr (common.py).
  * Integrals: PySCF `df.incore.cholesky_eri` with an auxmol built from
    ferric's bundled aux JSON. PySCF factorizes the Coulomb metric by Cholesky
    (V = L Lᵀ) where ferric uses V^{-1/2}; (ia|jb) = Σ_P B_ia^P B_jb^P is the
    same number either way (B Bᵀ = (ia|P) V^{-1} (P|jb)), so this is an
    independent construction of the SAME fitted integrals.
  * Energy: a dense numpy sum over the full (i,j,a,b) block — no i≤j symmetry
    folding, no i-blocked GEMMs — with the damping formula above.

Anchors asserted HERE before anything is written:
  * κ → ∞ (κ = 1e6) numpy == plain numpy to 1e-14 (the damping underflows to 1);
  * plain numpy == PySCF `mp.dfmp2.DFMP2(frozen=None)` e_corr to 1e-12 — the
    numpy integrals/denominators/energy expression reproduce an independent
    DF-MP2 implementation, so the only new ingredient per κ is the damping;
  * every κ in KAPPAS is INTERIOR: 1e-3 < E(κ)/E(∞) < 1 − 1e-3, i.e. neither
    trivial limit, and the damping factor spans a non-trivial range over the
    many (ij,ab) pairs of these systems.

Run (light; seconds per system):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_kappa_mp2.py   # the reference env (PySCF)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "kappa_mp2"
ROW_NAME = "κ-MP2"
# system -> (orbital basis, aux basis)
SYSTEMS = {
    "h2o": ("cc-pvdz", "cc-pvdz-ri"),
    "ch4": ("cc-pvdz", "cc-pvdz-ri"),
}
KAPPAS = (0.5, 1.1, 2.0)
KAPPA_INF = 1e6
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
TOL_ANCHOR = 1e-12


def df_factors(mol, aux_name, symbols):
    """(L[P, μ, ν], naux): Coulomb-metric DF factors with ferric's aux basis."""
    from pyscf import df, lib

    aux_bas, aux_cart = common.pyscf_basis(aux_name, symbols)
    assert not aux_cart, f"{aux_name}: Cartesian aux shells not supported here"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    cderi = df.incore.cholesky_eri(mol, auxmol=auxmol)
    naux = cderi.shape[0]
    return lib.unpack_tril(cderi), naux, aux_bas


def mp2_components(g, eo, ev, kappa):
    """Dense (E_OS, E_SS) with optional κ damping. g[i,a,j,b] = (ia|jb)."""
    delta = (
        ev[None, :, None, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, None, :, None]
    )  # [i,a,j,b], positive
    assert delta.min() > 0.0
    damp = 1.0 if kappa is None else (1.0 - np.exp(-kappa * delta)) ** 2
    g_x = g.transpose(0, 3, 2, 1)  # [i,a,j,b] -> (ib|ja)
    e_os = -np.sum(damp * g * g / delta)
    e_ss = -np.sum(damp * g * (g - g_x) / delta)
    return float(e_os), float(e_ss), delta


def pyscf_dfmp2(mf, aux_bas):
    from pyscf import df
    from pyscf.mp import dfmp2

    mp = dfmp2.DFMP2(mf, frozen=None)
    mp.with_df = df.DF(mf.mol)
    mp.with_df.auxmol = df.addons.make_auxmol(mf.mol, aux_bas)
    mp.with_df.auxmol.cart = False
    mp.with_df.auxmol.build()
    mp.with_df.auxbasis = aux_bas
    mp.kernel()
    return float(mp.e_corr)


def main() -> int:
    import pyscf
    from pyscf import scf

    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)}; known: {sorted(SYSTEMS)}"
        )
    written = []
    for system, (basis_name, aux_name) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)

        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.max_cycle = 200
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}: PySCF RHF did not converge")

        nocc = mol.nelectron // 2
        c = mf.mo_coeff
        eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
        lao, naux, aux_bas = df_factors(mol, aux_name, symbols)
        lov = np.einsum("Pmn,mi,na->Pia", lao, c[:, :nocc], c[:, nocc:], optimize=True)
        g = np.einsum("Pia,Pjb->iajb", lov, lov, optimize=True)

        e_os0, e_ss0, delta = mp2_components(g, eo, ev, None)
        e0 = e_os0 + e_ss0
        e_pyscf = pyscf_dfmp2(mf, aux_bas)
        d_anchor = abs(e0 - e_pyscf)
        if d_anchor > TOL_ANCHOR:
            raise RuntimeError(
                f"{system}: numpy plain DF-MP2 {e0:.14f} vs PySCF DFMP2 {e_pyscf:.14f} "
                f"(|d| {d_anchor:.2e}) — refusing to write"
            )
        e_os_inf, e_ss_inf, _ = mp2_components(g, eo, ev, KAPPA_INF)
        d_inf = abs(e_os_inf + e_ss_inf - e0)
        if d_inf > 1e-14:
            raise RuntimeError(
                f"{system}: kappa->inf limit broken in numpy ({d_inf:.2e})"
            )

        kappa_rows = []
        for kappa in KAPPAS:
            e_os, e_ss, _ = mp2_components(g, eo, ev, kappa)
            ratio = (e_os + e_ss) / e0
            if not (1e-3 < ratio < 1.0 - 1e-3):
                raise RuntimeError(
                    f"{system}: kappa={kappa} is not interior (ratio {ratio})"
                )
            damp = (1.0 - np.exp(-kappa * delta)) ** 2
            kappa_rows.append(
                {
                    "kappa": kappa,
                    "e_os": e_os,
                    "e_ss": e_ss,
                    "e_corr": e_os + e_ss,
                    "ratio_to_plain": ratio,
                    "damp_min": float(damp.min()),
                    "damp_max": float(damp.max()),
                }
            )
            print(
                f"{system:4s} kappa={kappa:4.1f} E_corr={e_os + e_ss:+.12f} "
                f"(OS {e_os:+.12f} SS {e_ss:+.12f}) ratio={ratio:.4f} "
                f"damp=[{damp.min():.3e},{damp.max():.6f}]"
            )

        n_terms = int(g.size)
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "aux_basis": aux_name,
            "nao": mol.nao_nr(),
            "naux": naux,
            "nocc": nocc,
            "nvir": len(ev),
            "n_ijab_terms": n_terms,
            "nuclear_repulsion": float(mol.energy_nuc()),
            "e_rhf": float(mf.e_tot),
            "plain": {
                "e_os": e_os0,
                "e_ss": e_ss0,
                "e_corr": e0,
                "pyscf_dfmp2_e_corr": e_pyscf,
                "numpy_vs_pyscf_abs_diff": d_anchor,
            },
            "kappa_inf_numpy_abs_diff": d_inf,
            "kappa": kappa_rows,
            "provenance": common.provenance(
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "scf": f"scf.RHF exact J/K, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
                    "integrals": "df.incore.cholesky_eri(mol, auxmol from ferric's aux JSON), "
                    "Coulomb metric, Cholesky-factorized",
                    "energy": "dense numpy sum over (i,a,j,b); damp=(1-exp(-kappa*Delta))^2 "
                    "on both OS and SS terms",
                    "anchor": "plain numpy == mp.dfmp2.DFMP2(frozen=None) to 1e-12",
                    "kappas": list(KAPPAS),
                    "numpy": np.__version__,
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=aux_name,
                frozen_core=None,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/validation/gen_kappa_mp2.py",
                extra={
                    "basis_self_check": basis_check,
                    "aux_basis_json_sha256": common.sha256_file(
                        common.basis_json_path(aux_name)
                    ),
                },
            ),
        }
        path = common.write_reference(ROW, system, basis_name, payload)
        written.append(path)
        print(
            f"{system:4s} E_RHF={mf.e_tot:.12f} plain E_corr={e0:+.12f} "
            f"numpy-vs-PySCF {d_anchor:.2e} naux={naux} terms={n_terms}"
        )
    print(f"GEN_KAPPA_MP2_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
