"""PySCF + numpy references for the VALIDATION.md "AO-Laplace MP2 (O(N))" row.

Consumer: crates/ferric-mp2/tests/validation_laplace_mp2.rs
Output:   testdata/reference/validation/laplace_mp2/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from `LaplaceMp2::compute_mo` / `compute_ao` in
crates/ferric-mp2/src/laplace.rs, not assumed): closed-shell RI-MP2 on an RHF
reference with the denominator `1/Δ` replaced by a minimax-Laplace sum

    1/Δ ≈ Σ_k w_k e^{−t_k Δ},    Δ = ε_a + ε_b − ε_i − ε_j > 0,

so that, with the τ-weighted DF amplitude B^P_ia(t) = B^P_ia e^{−t(ε_a−ε_i)/2},

    E_OS = −Σ_k w_k Σ_PQ J_PQ(t_k)²                        (the J / Coulomb term)
    E_SS = E_OS − (−Σ_k w_k · exchange Gram)               (`e_ss_k = e_os_k − e_exch_k`)
    E_corr = E_OS + E_SS.

The nodes come from `ferric_quadrature::LaplaceQuadrature::new(n_quad, ymin, ymax)`
for the range ferric itself derives from the RHF spectrum:

    ymin = 2(ε_LUMO − ε_HOMO),   ymax = 2(ε_max − ε_0),   R = ymax / ymin

with ε_0 the LOWEST orbital even when it is frozen (conservative: frozen core
does not shrink R today). `select_minimax_points` tabulates only
n_quad ∈ {3, 5, 7} and R ≤ 100 (k=3) / 1000 (k=5, 7); anything outside is a
hard `FerricError`, never a silent fallback.

THE REFERENCE, built independently of ferric's code path:
  * RHF: PySCF `scf.RHF`, exact four-centre J/K (ferric's `RhfConfig::default()`
    has no SCF fitting either), conv_tol 1e-12 / conv_tol_grad 1e-10, ferric's
    bundled orbital basis and geometry in Bohr (common.py).
  * Integrals: PySCF `df.incore.cholesky_eri` with an auxmol built from
    ferric's bundled aux JSON. PySCF factorizes the Coulomb metric by Cholesky
    (V = L Lᵀ) where ferric uses V^{-1/2}; (ia|jb) = Σ_P B_ia^P B_jb^P is the
    same number either way, so this is an independent construction of the SAME
    fitted integrals.
  * EXACT reference energy: a dense numpy sum over the full (i,a,j,b) block
    with the true 1/Δ denominator — i.e. exact DF-MP2, anchored against PySCF's
    own `mp.dfmp2.DFMP2` e_corr.
  * QUADRATURE-ISOLATED reference: the SAME dense numpy sum with 1/Δ replaced
    by Σ_k w_k e^{−t_k Δ}, using the IDENTICAL minimax nodes ferric would pick
    — parsed straight out of `crates/ferric-quadrature/src/minimax.rs` (its
    sha256 is recorded in provenance) and rescaled by the same
    `t/ymin, w/ymin` rule as `LaplaceQuadrature::new`.

WHY THE SECOND REFERENCE IS THE VALUABLE ONE. It splits ferric's error vs
exact DF-MP2 into two parts that a single comparison cannot separate:
  * ferric vs quadrature-isolated numpy  →  IMPLEMENTATION error (should be
    ~1e-12: same nodes, same integrals, different assembly);
  * quadrature-isolated numpy vs exact   →  QUADRATURE error, the physics of
    the method, which is a property of R and n_quad alone.

Anchors asserted HERE before anything is written:
  * exact numpy == PySCF `mp.dfmp2.DFMP2(frozen=None)` e_corr to 1e-10;
  * the node selection reproduces `select_minimax_points`' own rule (first
    tabulated R_tab ≥ 0.99 R), and R is recorded per system;
  * the quadrature error decreases monotonically over n_quad 3 → 5 → 7.

Run (light; minutes for octane):
    scripts/validation/run_slot.sh --light -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_laplace_mp2.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "laplace_mp2"
ROW_NAME = "AO-Laplace MP2 (O(N))"
MINIMAX_RS = common.ROOT / "crates" / "ferric-quadrature" / "src" / "minimax.rs"

# system -> (orbital basis, aux basis)
SYSTEMS = {
    "h2o": ("cc-pvdz", "cc-pvdz-ri"),
    "h2o_aug": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "ch4": ("cc-pvdz", "cc-pvdz-ri"),
    "butadiene": ("cc-pvdz", "cc-pvdz-ri"),
    "alkane_8": ("cc-pvdz", "cc-pvdz-ri"),
}
# system -> the geometry file stem in testdata/molecules/validation/
GEOMETRY = {
    "h2o": "h2o",
    "h2o_aug": "h2o",
    "ch4": "ch4",
    "butadiene": "butadiene",
    "alkane_8": "alkane_8",
}
N_QUADS = (3, 5, 7)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
# exact numpy DF-MP2 vs PySCF's own DFMP2: same integrals, different assembly.
TOL_ANCHOR = 1e-10


# ---------------------------------------------------------------------------
# ferric's minimax table, parsed from ferric's source
# ---------------------------------------------------------------------------


def parse_minimax_table(k: int) -> list[tuple[float, list[float], list[float]]]:
    """Parse `static MINIMAX_K<k>` out of ferric's minimax.rs.

    Returns [(R_tab, t[], w[]), ...] in tabulated order, i.e. exactly the slice
    `select_minimax_points` scans.
    """
    text = MINIMAX_RS.read_text()
    m = re.search(
        rf"static MINIMAX_K{k}: &\[MinimaxEntry\] = &\[(.*?)\n\];", text, re.S
    )
    if not m:
        raise RuntimeError(f"{MINIMAX_RS}: could not find static MINIMAX_K{k}")
    entries = re.findall(
        r"\(\s*([0-9.eE+-]+),\s*&\[(.*?)\],\s*&\[(.*?)\],\s*\)", m.group(1), re.S
    )
    out = []
    for r_tab, t_block, w_block in entries:
        t = [float(v) for v in re.findall(r"[-+0-9.eE]+", t_block)]
        w = [float(v) for v in re.findall(r"[-+0-9.eE]+", w_block)]
        if len(t) != k or len(w) != k:
            raise RuntimeError(
                f"MINIMAX_K{k} entry R={r_tab}: parsed {len(t)} t and {len(w)} w, expected {k}"
            )
        out.append((float(r_tab), t, w))
    if not out:
        raise RuntimeError(f"MINIMAX_K{k}: parsed no entries")
    return out


def laplace_quadrature(n_quad: int, ymin: float, ymax: float):
    """Reproduce `LaplaceQuadrature::new`: pick the first tabulated R_tab with
    R_tab >= 0.99 * (ymax/ymin), then rescale t/ymin, w/ymin.

    Raises RuntimeError with the same meaning as ferric's hard `FerricError`
    when R exceeds the table — never a silent fallback to a smaller n_quad or a
    clamped range.
    """
    r = ymax / ymin
    table = parse_minimax_table(n_quad)
    for r_tab, t, w in table:
        if r_tab >= r * 0.99:
            return (
                np.array(t) / ymin,
                np.array(w) / ymin,
                r_tab,
                r,
            )
    r_max = table[-1][0]
    raise RuntimeError(
        f"minimax quadrature: R={r:.4f} exceeds the largest tabulated range "
        f"R_max={r_max} for n_quad={n_quad} — ferric hard-errors here; this is a "
        f"FINDING to record, not a case to coerce with nearest_supported_n_quad"
    )


# ---------------------------------------------------------------------------
# DF integrals and the two MP2 evaluations
# ---------------------------------------------------------------------------


def df_factors(mol, aux_name, symbols):
    """(L[P, mu, nu], naux, aux_bas): Coulomb-metric DF factors, ferric's aux."""
    from pyscf import df, lib

    aux_bas, aux_cart = common.pyscf_basis(aux_name, symbols)
    assert not aux_cart, f"{aux_name}: Cartesian aux shells not supported here"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    cderi = df.incore.cholesky_eri(mol, auxmol=auxmol)
    naux = cderi.shape[0]
    return lib.unpack_tril(cderi), naux, aux_bas


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


def mp2_components(lov, eo, ev, nodes=None, block=None):
    """(E_OS, E_SS) by a dense sum over (i,a,j,b), i-blocked for memory only.

    `nodes=None` uses the exact 1/Delta denominator (DF-MP2). `nodes=(t, w)`
    substitutes 1/Delta -> sum_k w_k exp(-t_k Delta): the quadrature-isolated
    reference, identical in every other respect.

    Blocking over i only splits the OUTER sum, so it changes nothing but peak
    memory (each (i) slab is summed in full before accumulation).
    """
    nocc = len(eo)
    if block is None:
        block = max(1, nocc)
    e_os = 0.0
    e_ss = 0.0
    for i0 in range(0, nocc, block):
        i1 = min(nocc, i0 + block)
        # g[i,a,j,b] = (ia|jb) for i in [i0,i1)
        g = np.einsum("Pia,Pjb->iajb", lov[:, i0:i1], lov, optimize=True)
        delta = (
            ev[None, :, None, None]
            + ev[None, None, None, :]
            - eo[i0:i1, None, None, None]
            - eo[None, None, :, None]
        )
        assert delta.min() > 0.0, (
            "a negative MP2 denominator means a non-Aufbau reference"
        )
        if nodes is None:
            inv = 1.0 / delta
        else:
            t, w = nodes
            inv = np.zeros_like(delta)
            for tk, wk in zip(t, w):
                inv += wk * np.exp(-tk * delta)
        g_x = g.transpose(0, 3, 2, 1)  # [i,a,j,b] -> (ib|ja)
        e_os -= float(np.sum(inv * g * g))
        e_ss -= float(np.sum(inv * g * (g - g_x)))
        del g, g_x, delta, inv
    return e_os, e_ss


def main() -> int:
    import pyscf
    from pyscf import scf

    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)}; known: {sorted(SYSTEMS)}"
        )
    minimax_sha = common.sha256_file(MINIMAX_RS)
    written = []
    for system, (basis_name, aux_name) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{GEOMETRY[system]}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)

        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}: PySCF RHF did not converge")

        nocc = mol.nelectron // 2
        eps = mf.mo_energy
        nmo = len(eps)
        c = mf.mo_coeff
        eo, ev = eps[:nocc], eps[nocc:]

        # The Laplace range EXACTLY as LaplaceMp2::compute_mo derives it.
        ymin = 2.0 * (eps[nocc] - eps[nocc - 1])
        ymax = 2.0 * (eps[nmo - 1] - eps[0])
        r_range = ymax / ymin

        lao, naux, aux_bas = df_factors(mol, aux_name, symbols)
        lov = np.einsum("Pmn,mi,na->Pia", lao, c[:, :nocc], c[:, nocc:], optimize=True)
        del lao
        # Keep the dense (i,a,j,b) slab under ~2 GB: nvir^2 * nocc * 8 per i.
        per_i = len(ev) ** 2 * nocc * 8
        i_block = max(1, int(2.0e9 // max(per_i, 1)))

        e_os_x, e_ss_x = mp2_components(lov, eo, ev, None, i_block)
        e_exact = e_os_x + e_ss_x
        e_pyscf = pyscf_dfmp2(mf, aux_bas)
        d_anchor = abs(e_exact - e_pyscf)
        if d_anchor > TOL_ANCHOR:
            raise RuntimeError(
                f"{system}: numpy exact DF-MP2 {e_exact:.14f} vs PySCF DFMP2 "
                f"{e_pyscf:.14f} (|d| {d_anchor:.2e}) — refusing to write"
            )

        quad_rows = []
        for n_quad in N_QUADS:
            try:
                t, w, r_tab, r_req = laplace_quadrature(n_quad, ymin, ymax)
            except RuntimeError as exc:
                # A FINDING, recorded rather than coerced. Nothing is written
                # for this n_quad; the Rust test asserts ferric errors too.
                print(f"{system:10s} n_quad={n_quad}: RANGE EXCEEDED: {exc}")
                quad_rows.append(
                    {
                        "n_quad": n_quad,
                        "range_exceeded": True,
                        "message": str(exc),
                    }
                )
                continue
            e_os_q, e_ss_q = mp2_components(lov, eo, ev, (t, w), i_block)
            e_q = e_os_q + e_ss_q
            quad_rows.append(
                {
                    "n_quad": n_quad,
                    "range_exceeded": False,
                    "r_tab_selected": r_tab,
                    "points": [float(v) for v in t],
                    "weights": [float(v) for v in w],
                    "e_os": e_os_q,
                    "e_ss": e_ss_q,
                    "e_corr": e_q,
                    "quadrature_error_e_corr": e_q - e_exact,
                    "quadrature_error_e_os": e_os_q - e_os_x,
                    "quadrature_error_e_ss": e_ss_q - e_ss_x,
                }
            )
            print(
                f"{system:10s} n_quad={n_quad} R={r_req:8.3f} R_tab={r_tab:7.1f} "
                f"E_corr={e_q:+.12f} quad_err={e_q - e_exact:+.3e} "
                f"(OS {e_os_q - e_os_x:+.3e} SS {e_ss_q - e_ss_x:+.3e})"
            )

        # Monotone convergence of |quadrature error| over n_quad, in the
        # reference itself. A reference that does not converge is not a
        # reference for a convergence claim.
        errs = [
            abs(row["quadrature_error_e_corr"])
            for row in quad_rows
            if not row["range_exceeded"]
        ]
        monotone = all(a > b for a, b in zip(errs, errs[1:]))
        if len(errs) > 1 and not monotone:
            raise RuntimeError(
                f"{system}: |quadrature error| is NOT monotone over n_quad: {errs}"
            )

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "aux_basis": aux_name,
            "geometry_stem": GEOMETRY[system],
            "nao": mol.nao_nr(),
            "naux": naux,
            "nocc": nocc,
            "nvir": len(ev),
            "n_heavy_atoms": sum(1 for s in symbols if common.z_of(s) > 2),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "e_rhf": float(mf.e_tot),
            "laplace_range": {
                "eps_homo": float(eps[nocc - 1]),
                "eps_lumo": float(eps[nocc]),
                "eps_0": float(eps[0]),
                "eps_max": float(eps[nmo - 1]),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "r": float(r_range),
                "comment": "ymin=2(eps_LUMO-eps_HOMO), ymax=2(eps_max-eps_0); "
                "eps_0 is the lowest orbital even when frozen",
            },
            "exact": {
                "e_os": e_os_x,
                "e_ss": e_ss_x,
                "e_corr": e_exact,
                "pyscf_dfmp2_e_corr": e_pyscf,
                "numpy_vs_pyscf_abs_diff": d_anchor,
            },
            "quadrature": quad_rows,
            "quadrature_error_monotone": monotone,
            "provenance": common.provenance(
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "scf": f"scf.RHF exact J/K, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
                    "integrals": "df.incore.cholesky_eri(mol, auxmol from ferric's aux JSON), "
                    "Coulomb metric, Cholesky-factorized",
                    "exact_energy": "dense numpy sum over (i,a,j,b) with 1/Delta",
                    "quadrature_energy": "the SAME dense sum with 1/Delta -> sum_k w_k "
                    "exp(-t_k Delta), nodes parsed from ferric's MINIMAX_K{n} and "
                    "rescaled t/ymin, w/ymin as LaplaceQuadrature::new does",
                    "anchor": "exact numpy == mp.dfmp2.DFMP2(frozen=None) to 1e-10",
                    "n_quads": list(N_QUADS),
                    "numpy": np.__version__,
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=aux_name,
                frozen_core=0,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/validation/gen_laplace_mp2.py",
                extra={
                    "basis_self_check": basis_check,
                    "aux_basis_json_sha256": common.sha256_file(
                        common.basis_json_path(aux_name)
                    ),
                    "minimax_rs": str(MINIMAX_RS.relative_to(common.ROOT)),
                    "minimax_rs_sha256": minimax_sha,
                },
            ),
        }
        path = common.write_reference(ROW, system, basis_name, payload)
        written.append(path)
        print(
            f"{system:10s} E_RHF={mf.e_tot:.12f} exact E_corr={e_exact:+.12f} "
            f"numpy-vs-PySCF {d_anchor:.2e} nao={mol.nao_nr()} naux={naux} R={r_range:.3f}"
        )
    print(f"GEN_LAPLACE_MP2_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
