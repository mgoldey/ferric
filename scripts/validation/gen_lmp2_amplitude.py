"""PySCF DF-MP2 references (+ an ORCA DLPNO-MP2 ballpark) for the
VALIDATION.md "Amplitude-threshold LMP2" row.

Consumer: crates/ferric-mp2/tests/validation_lmp2_amplitude.rs
Output:   testdata/reference/validation/lmp2_amplitude/<system>_<basis>.json
          testdata/reference/validation/lmp2_amplitude/orca_<system>_cc-pvdz.json

WHAT FERRIC COMPUTES (read from `amplitude_lmp2_with_virtuals` in
crates/ferric-mp2/src/lmp2_amplitude.rs and `amplitude_lmp2_direct_with_virtuals`
in lmp2_direct.rs, not assumed): closed-shell RI-MP2 on an RHF reference in a
Boys-localized occupied / VV-HV localized virtual basis, Coulomb-metric density
fitting with the bundled aux basis, the lowest `frozen_core` orbitals frozen,
amplitudes masked by the single threshold ε and solved by a ragged
preconditioned CG, energy by the Hylleraas functional. At ε = 0 with every
locality knob at its trivial limit, the mask keeps everything and the energy
is EXACTLY the canonical DF-MP2 of the same fitted integrals — an orbital
rotation of the same quantity.

THE REFERENCE, built independently of ferric's code path:
  * RHF: PySCF `scf.RHF`, exact four-centre J/K (ferric's `RhfConfig::default()`
    has no SCF fitting either), conv_tol 1e-12 / conv_tol_grad 1e-10, ferric's
    bundled orbital basis and geometry in Bohr (common.py).
  * E_corr: PySCF `mp.dfmp2.DFMP2(mf, frozen=fc)` with an auxmol built from
    ferric's `cc-pvdz-ri` JSON (canonical orbitals, closed-form denominators —
    no localization, no CG), for fc = 0 and fc = number of heavy atoms.
  * Cross-check: a dense numpy sum over (i,a,j,b) on
    `df.incore.cholesky_eri` integrals with the same aux, which must equal
    PySCF's DFMP2 to 1e-12 before anything is written (OS / SS recorded too).

ORCA BALLPARK (`--orca`; c4h10 and c8h18 at cc-pVDZ, all electrons correlated):
ORCA 6.1.1 RHF (exact integrals, VeryTightSCF) then `RI-MP2` and `DLPNO-MP2`
at NormalPNO and TightPNO, ferric's orbital basis as `NewGTO` and ferric's
cc-pvdz-ri as the correlation aux (`NewAuxCGTO`). ORCA's DLPNO truncates by
PNO occupation (TCutPNO), PAO domains and pair prescreening on Foster-Boys
LMOs — a DIFFERENT local scheme from ferric's amplitude threshold, so its %
recovery is a ballpark and is NOT a bar for ferric. The one like-for-like
number is ORCA RI-MP2 vs PySCF DFMP2 (same basis, aux, frozen core = 0),
asserted here to `TOL_ORCA_RIMP2` before the file is written.

Run (PySCF part light; the ORCA part takes the slot):
    scripts/validation/run_slot.sh --light -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_lmp2_amplitude.py [system ...]
    scripts/validation/run_slot.sh -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_lmp2_amplitude.py --orca
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "lmp2_amplitude"
ROW_NAME = "Amplitude-threshold LMP2"
AUX = "cc-pvdz-ri"
# system -> orbital bases
SYSTEMS = {
    "h2o": ("cc-pvdz", "6-31g"),
    "c4h10": ("cc-pvdz", "6-31g"),
}
ORCA_SYSTEMS = ("c4h10", "c8h18")
ORCA_BASIS = "cc-pvdz"
ORCA_RUNS = {
    "ri_mp2": "RHF RI-MP2 VeryTightSCF NoFrozenCore",
    "dlpno_normalpno": "RHF DLPNO-MP2 NormalPNO VeryTightSCF NoFrozenCore",
    "dlpno_tightpno": "RHF DLPNO-MP2 TightPNO VeryTightSCF NoFrozenCore",
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
TOL_NUMPY_VS_PYSCF = 1e-12
# ORCA RI-MP2 vs PySCF DFMP2, same basis/aux/frozen core; both exact-integral
# RHF. ORCA prints the SCF energy to 14 decimals and converges to TolE 1e-9
# (VeryTightSCF), so the floor is ORCA's SCF/integral thresholds, not the fit.
TOL_ORCA_RIMP2 = 1e-6
TOL_ORCA_RHF = 1e-6


def heavy_atoms(symbols) -> int:
    return sum(1 for s in symbols if s.upper() != "H")


def aux_basis(mol, symbols):
    from pyscf import df

    aux_bas, aux_cart = common.pyscf_basis(AUX, symbols)
    assert not aux_cart, f"{AUX}: Cartesian aux shells not supported here"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    return aux_bas, auxmol


def run_rhf(mol, tag):
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{tag}: PySCF RHF did not converge")
    return mf


def pyscf_dfmp2(mf, aux_bas, fc):
    from pyscf import df
    from pyscf.mp import dfmp2

    mp = dfmp2.DFMP2(mf, frozen=fc if fc > 0 else None)
    mp.with_df = df.DF(mf.mol)
    mp.with_df.auxmol = df.addons.make_auxmol(mf.mol, aux_bas)
    mp.with_df.auxmol.cart = False
    mp.with_df.auxmol.build()
    mp.with_df.auxbasis = aux_bas
    mp.kernel()
    return float(mp.e_corr)


def numpy_dfmp2(mf, auxmol, fc):
    """Dense (E_OS, E_SS) over active occupieds fc..nocc."""
    from pyscf import df, lib

    nocc = mf.mol.nelectron // 2
    lao = lib.unpack_tril(df.incore.cholesky_eri(mf.mol, auxmol=auxmol))
    c = mf.mo_coeff
    eo, ev = mf.mo_energy[fc:nocc], mf.mo_energy[nocc:]
    lov = np.einsum("Pmn,mi,na->Pia", lao, c[:, fc:nocc], c[:, nocc:], optimize=True)
    g = np.einsum("Pia,Pjb->iajb", lov, lov, optimize=True)
    delta = (
        ev[None, :, None, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, None, :, None]
    )
    g_x = g.transpose(0, 3, 2, 1)
    e_os = -float(np.sum(g * g / delta))
    e_ss = -float(np.sum(g * (g - g_x) / delta))
    return e_os, e_ss, lao.shape[0]


def pyscf_block(system, basis_name):
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    mf = run_rhf(mol, f"{system}/{basis_name}")
    aux_bas, auxmol = aux_basis(mol, symbols)
    rows = []
    naux = None
    for fc in sorted({0, heavy_atoms(symbols)}):
        e_pyscf = pyscf_dfmp2(mf, aux_bas, fc)
        e_os, e_ss, naux = numpy_dfmp2(mf, auxmol, fc)
        d = abs(e_os + e_ss - e_pyscf)
        if d > TOL_NUMPY_VS_PYSCF:
            raise RuntimeError(
                f"{system}/{basis_name} fc={fc}: numpy {e_os + e_ss:.14f} vs PySCF "
                f"DFMP2 {e_pyscf:.14f} (|d| {d:.2e}) — refusing to write"
            )
        rows.append(
            {
                "frozen_core": fc,
                "e_corr": e_pyscf,
                "e_os": e_os,
                "e_ss": e_ss,
                "numpy_vs_pyscf_abs_diff": d,
            }
        )
        print(
            f"{system:6s} {basis_name:8s} fc={fc} E_corr={e_pyscf:+.12f} "
            f"(OS {e_os:+.12f} SS {e_ss:+.12f}) numpy-vs-PySCF {d:.2e}"
        )
    nocc = mol.nelectron // 2
    return {
        "symbols": symbols,
        "coords": coords,
        "xyz": xyz,
        "mol": mol,
        "mf": mf,
        "basis_check": basis_check,
        "payload": {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "aux_basis": AUX,
            "nao": mol.nao_nr(),
            "naux": naux,
            "nocc": nocc,
            "nvir": mol.nao_nr() - nocc,
            "n_heavy_atoms": heavy_atoms(symbols),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "e_rhf": float(mf.e_tot),
            "dfmp2": rows,
        },
    }


def provenance(blk, basis_name, code, version, keywords, frozen_core, extra=None):
    import pyscf

    ext = {
        "basis_self_check": blk["basis_check"],
        "aux_basis_json_sha256": common.sha256_file(common.basis_json_path(AUX)),
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
    }
    if extra:
        ext.update(extra)
    return common.provenance(
        code=code,
        version=version,
        keywords=keywords,
        basis_name=basis_name,
        xyz_path=blk["xyz"],
        coords_bohr=blk["coords"],
        symbols=blk["symbols"],
        grid=None,
        aux=AUX,
        frozen_core=frozen_core,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=None,
        generator="scripts/validation/gen_lmp2_amplitude.py",
        extra=ext,
    )


# ---------------------------------------------------------------------------
# ORCA
# ---------------------------------------------------------------------------


def orca_aux_block(symbols) -> str:
    """`NewAuxCGTO` entries carrying ferric's cc-pvdz-ri (same writer
    conventions as `common.orca_basis_block`)."""
    lines = []
    for s in dict.fromkeys(symbols):
        lines.append(f"  NewAuxCGTO {s}")
        for sh in common.ferric_shells(AUX, common.z_of(s)):
            if sh["l"] >= 2 and not sh["pure"]:
                raise ValueError(f"{AUX}: Cartesian l={sh['l']} shell")
            lines.append(f"    {common.ORCA_SHELL_LETTER[sh['l']]} {len(sh['exps'])}")
            for i, (e, c) in enumerate(zip(sh["exps"], sh["coefs"]), start=1):
                lines.append(f"      {i:3d} {e:.10E} {c:.10E}")
        lines.append("  end")
    return "\n".join(lines)


_ORCA_EXTRA = {
    "e_scf": re.compile(r"Total Energy\s+:\s+(-?\d+\.\d+)\s+Eh"),
    "ri_mp2_corr": re.compile(r"RI-MP2 CORRELATION ENERGY:\s+(-?\d+\.\d+)\s+Eh"),
    "dlpno_corr": re.compile(r"DLPNO-MP2 CORRELATION ENERGY:\s+(-?\d+\.\d+)\s+Eh"),
    "naux": re.compile(r"# of basis functions in Aux-C\s+\.+\s+(\d+)"),
    "tcut_pno": re.compile(r"TCutPNO\s+=\s+(\S+)"),
    "tcut_pairs": re.compile(r"TCutPre\s+=\s+(\S+)"),
    "pairs_included": re.compile(r"Number of orbital pairs included:\s+(\d+)"),
    "avg_pnos_per_pair": re.compile(r"Average number of PNOs per pair\s+(\S+)"),
}


def run_orca_job(system, keywords, symbols, workdir: Path) -> dict:
    xyz = common.MOL_DIR / f"{system}.xyz"
    inp = workdir / f"{system}.inp"
    common.write_orca_input(inp, keywords, xyz, ORCA_BASIS)
    std = common.orca_basis_block(ORCA_BASIS, symbols)
    with_aux = std[: -len("end")] + orca_aux_block(symbols) + "\nend"
    text = inp.read_text().replace(std, with_aux)
    inp.write_text(text)
    parsed = common.run_orca(inp)
    out = inp.with_suffix(".out").read_text()
    for key, pat in _ORCA_EXTRA.items():
        hits = pat.findall(out)
        if hits:
            parsed[key] = float(hits[-1])
    parsed["input"] = text
    return parsed


def orca_ballpark(system):
    blk = pyscf_block(system, ORCA_BASIS)
    payload = blk["payload"]
    e_df = next(r["e_corr"] for r in payload["dfmp2"] if r["frozen_core"] == 0)
    runs = {}
    for tag, kw in ORCA_RUNS.items():
        with tempfile.TemporaryDirectory(prefix="orca_lmp2_") as td:
            r = run_orca_job(system, kw, blk["symbols"], Path(td))
        d_rhf = abs(r["e_scf"] - payload["e_rhf"])
        if d_rhf > TOL_ORCA_RHF:
            raise RuntimeError(f"{system}/{tag}: ORCA E_RHF off PySCF by {d_rhf:.2e}")
        if int(r["naux"]) != payload["naux"]:
            raise RuntimeError(
                f"{system}/{tag}: ORCA naux {r['naux']} != {payload['naux']}"
            )
        e_corr = r["energy"] - r["e_scf"]
        row = {
            "keywords": kw,
            "e_scf": r["e_scf"],
            "e_total": r["energy"],
            "e_corr": e_corr,
            "e_rhf_vs_pyscf_abs_diff": d_rhf,
            "naux": int(r["naux"]),
            "version": r.get("version"),
            "input": r["input"],
        }
        for k in ("tcut_pno", "tcut_pairs", "pairs_included", "avg_pnos_per_pair"):
            if k in r:
                row[k] = r[k]
        runs[tag] = row
        print(f"{system:6s} ORCA {tag:16s} E_corr={e_corr:+.10f} dRHF={d_rhf:.1e}")
    d_ri = abs(runs["ri_mp2"]["e_corr"] - e_df)
    if d_ri > TOL_ORCA_RIMP2:
        raise RuntimeError(
            f"{system}: ORCA RI-MP2 {runs['ri_mp2']['e_corr']:.10f} vs PySCF DFMP2 "
            f"{e_df:.10f} (|d| {d_ri:.2e}) — refusing to write"
        )
    e_orca_ri = runs["ri_mp2"]["e_corr"]
    for tag in ("dlpno_normalpno", "dlpno_tightpno"):
        runs[tag]["pct_of_orca_ri_mp2"] = 100.0 * runs[tag]["e_corr"] / e_orca_ri
        runs[tag]["pct_of_pyscf_dfmp2"] = 100.0 * runs[tag]["e_corr"] / e_df
        print(
            f"{system:6s} ORCA {tag:16s} {runs[tag]['pct_of_orca_ri_mp2']:.4f}% of ORCA RI-MP2"
        )
    out = {
        "row": ROW_NAME + " (ORCA DLPNO-MP2 ballpark)",
        "system": system,
        "basis": ORCA_BASIS,
        "aux_basis": AUX,
        "nao": payload["nao"],
        "naux": payload["naux"],
        "nuclear_repulsion": payload["nuclear_repulsion"],
        "e_rhf": payload["e_rhf"],
        "frozen_core": 0,
        "pyscf_dfmp2_e_corr": e_df,
        "orca_ri_mp2_vs_pyscf_dfmp2_abs_diff": d_ri,
        "orca": runs,
        "note": (
            "BALLPARK ONLY: ORCA DLPNO-MP2 truncates by PNO occupation, PAO "
            "domains and pair prescreening on Foster-Boys LMOs; ferric's local "
            "MP2 truncates by an amplitude-space integral threshold eps. The "
            "two % recoveries are not like-for-like and do not rank the codes."
        ),
        "provenance": provenance(
            blk,
            ORCA_BASIS,
            code="ORCA + PySCF",
            version=f"ORCA {runs['ri_mp2'].get('version', '6.1.1')}",
            keywords={k: v for k, v in ORCA_RUNS.items()},
            frozen_core=0,
            extra={"orca_binary": str(common.ORCA_BINARY)},
        ),
    }
    path = common.REF_DIR / ROW / f"orca_{system}_{ORCA_BASIS}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def main() -> int:
    import pyscf

    args = sys.argv[1:]
    if "--orca" in args:
        only = {a for a in args if not a.startswith("--")}
        written = [orca_ballpark(s) for s in ORCA_SYSTEMS if not only or s in only]
        print(f"GEN_LMP2_AMPLITUDE_ORCA_DONE written={len(written)}")
        return 0
    only = set(args)
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)}; known: {sorted(SYSTEMS)}"
        )
    written = []
    for system, bases in SYSTEMS.items():
        if only and system not in only:
            continue
        for basis_name in bases:
            blk = pyscf_block(system, basis_name)
            payload = blk["payload"]
            payload["provenance"] = provenance(
                blk,
                basis_name,
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "scf": f"scf.RHF exact J/K, conv_tol {CONV_TOL}, "
                    f"conv_tol_grad {CONV_TOL_GRAD}",
                    "mp2": "mp.dfmp2.DFMP2(frozen=fc), auxmol from ferric's "
                    f"{AUX} JSON, Coulomb metric",
                    "cross_check": "dense numpy (i,a,j,b) sum on df.incore.cholesky_eri "
                    f"== DFMP2 to {TOL_NUMPY_VS_PYSCF}",
                },
                frozen_core=[r["frozen_core"] for r in payload["dfmp2"]],
            )
            written.append(common.write_reference(ROW, system, basis_name, payload))
            print(
                f"{system:6s} {basis_name:8s} E_RHF={payload['e_rhf']:.12f} "
                f"nao={payload['nao']} naux={payload['naux']}"
            )
    print(f"GEN_LMP2_AMPLITUDE_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
