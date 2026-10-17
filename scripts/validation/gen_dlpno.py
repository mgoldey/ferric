"""ORCA 6.1.1 + PySCF references for the VALIDATION.md "DLPNO" row (issue #269).

Consumer: crates/ferric-mp2/tests/validation_dlpno_mp2.rs
Output:   testdata/reference/validation/dlpno/<system>_cc-pvdz.json

SYSTEMS: n-butane, n-hexane, n-octane (testdata/molecules/alkane_{4,6,8}.xyz),
cc-pVDZ with ferric's cc-pvdz-ri as the correlation aux, in TWO frozen-core
variants: all electrons correlated (fc=0) and one frozen 1s per carbon.

PER SYSTEM x FROZEN-CORE VARIANT:
  * PySCF  mp.dfmp2.DFMP2(frozen=fc) with an auxmol from ferric's aux JSON
    (the anchor for ferric's canonical RI-MP2), cross-checked against a dense
    numpy (i,a,j,b) sum on the same integrals to 1e-12 before anything is
    written. RHF: exact four-centre J/K, conv_tol 1e-12.
  * ORCA   RHF RI-MP2 / DLPNO-MP2 NormalPNO / DLPNO-MP2 TightPNO, ferric's
    orbital basis as NewGTO, ferric's cc-pvdz-ri as NewAuxCGTO, geometry in
    Bohr with ferric's constant, VeryTightSCF, NoFrozenCore or FrozenCore.
    The ONE like-for-like ORCA number is RI-MP2 vs PySCF DFMP2; it must agree
    to TOL_ORCA_RIMP2 (and the RHF energy and aux count must agree) before the
    DLPNO numbers are trusted. ORCA DLPNO truncates by PNO occupation on
    Foster-Boys LOCALIZED occupieds with PAO domains; ferric's DLPNO-MP2 takes
    CANONICAL occupieds. The two % recoveries are therefore NOT like-for-like
    and are recorded side by side, not ranked.
  * (--ccsdt, C4 only) CONTEXT: ORCA canonical CCSD(T) and DLPNO-CCSD(T)
    Normal/TightPNO, all electrons correlated. ferric has no DLPNO-CCSD(T)
    driver, so these are not compared against anything in ferric.

Run (the ORCA part takes the slot):
    scripts/validation/run_slot.sh -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_dlpno.py [system ...] [--ccsdt]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_lmp2_amplitude as lm  # noqa: E402  (orca_aux_block, pyscf helpers)

ROW = "dlpno"
ROW_NAME = "DLPNO (MP2 / CCSD / CCSD(T) / RPA)"
BASIS = "cc-pvdz"
AUX = lm.AUX
SYSTEMS = {
    "alkane_4": "n-butane",
    "alkane_6": "n-hexane",
    "alkane_8": "n-octane",
}
MOL_DIR = common.ROOT / "testdata" / "molecules"
ORCA_NPROCS = 2
TOL_ORCA_RIMP2 = 1e-6
TOL_ORCA_RHF = 1e-6
TOL_NUMPY_VS_PYSCF = 1e-12

MP2_RUNS = {
    "ri_mp2": "RHF RI-MP2 VeryTightSCF {fc}",
    "dlpno_normalpno": "RHF DLPNO-MP2 NormalPNO VeryTightSCF {fc}",
    "dlpno_tightpno": "RHF DLPNO-MP2 TightPNO VeryTightSCF {fc}",
}
# Context only (C4, fc=0). Canonical CCSD(T) uses exact integrals; DLPNO-CC
# uses RI with the /C aux below, so the % includes that RI error (not removed).
CC_RUNS = {
    "ccsd_t": "RHF CCSD(T) VeryTightSCF NoFrozenCore",
    "dlpno_ccsd_t_normalpno": "RHF DLPNO-CCSD(T) NormalPNO VeryTightSCF NoFrozenCore",
    "dlpno_ccsd_t_tightpno": "RHF DLPNO-CCSD(T) TightPNO VeryTightSCF NoFrozenCore",
}

_EXTRA = dict(lm._ORCA_EXTRA)


def n_heavy(symbols) -> int:
    return sum(1 for s in symbols if s.upper() != "H")


def orca_job(xyz: Path, keywords: str, symbols, workdir: Path) -> dict:
    inp = workdir / f"{xyz.stem}.inp"
    common.write_orca_input(inp, keywords, xyz, BASIS, nprocs=ORCA_NPROCS)
    std = common.orca_basis_block(BASIS, symbols)
    with_aux = std[: -len("end")] + lm.orca_aux_block(symbols) + "\nend"
    text = inp.read_text().replace(std, with_aux)
    inp.write_text(text)
    parsed = common.run_orca(inp)
    out = inp.with_suffix(".out").read_text()
    for key, pat in _EXTRA.items():
        hits = pat.findall(out)
        if hits:
            try:
                parsed[key] = float(hits[-1])
            except ValueError:  # e.g. "..." in a table of another module
                continue
    parsed["input"] = text
    return parsed


def pyscf_reference(xyz: Path):
    """RHF + DF-MP2 for fc in {0, n_heavy}; numpy cross-check."""
    mol = common.build_pyscf_mol(xyz, BASIS)
    symbols, coords = common.read_xyz(xyz)
    basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
    mf = lm.run_rhf(mol, xyz.stem)
    aux_bas, auxmol = lm.aux_basis(mol, symbols)
    rows = []
    naux = None
    for fc in (0, n_heavy(symbols)):
        e_pyscf = lm.pyscf_dfmp2(mf, aux_bas, fc)
        e_os, e_ss, naux = lm.numpy_dfmp2(mf, auxmol, fc)
        d = abs(e_os + e_ss - e_pyscf)
        if d > TOL_NUMPY_VS_PYSCF:
            raise RuntimeError(f"{xyz.stem} fc={fc}: numpy vs DFMP2 {d:.2e}")
        rows.append(
            {
                "frozen_core": fc,
                "e_corr": e_pyscf,
                "e_os": e_os,
                "e_ss": e_ss,
                "numpy_vs_pyscf_abs_diff": d,
            }
        )
    return mol, mf, symbols, coords, basis_check, rows, naux


def run_mp2_block(xyz, symbols, mf, naux, py_row) -> dict:
    fc = py_row["frozen_core"]
    fc_kw = "FrozenCore" if fc > 0 else "NoFrozenCore"
    runs = {}
    for tag, tmpl in MP2_RUNS.items():
        kw = tmpl.format(fc=fc_kw)
        with tempfile.TemporaryDirectory(prefix="orca_dlpno_") as td:
            r = orca_job(xyz, kw, symbols, Path(td))
        d_rhf = abs(r["e_scf"] - float(mf.e_tot))
        if d_rhf > TOL_ORCA_RHF:
            raise RuntimeError(f"{xyz.stem}/{tag}: ORCA E_RHF off by {d_rhf:.2e}")
        if int(r["naux"]) != naux:
            raise RuntimeError(f"{xyz.stem}/{tag}: naux {r['naux']} != {naux}")
        row = {
            "keywords": kw,
            "e_scf": r["e_scf"],
            "e_corr": r["energy"] - r["e_scf"],
            "e_rhf_vs_pyscf_abs_diff": d_rhf,
            "version": r.get("version"),
            "input": r["input"],
        }
        for k in ("tcut_pno", "tcut_pairs", "pairs_included", "avg_pnos_per_pair"):
            if k in r:
                row[k] = r[k]
        runs[tag] = row
        print(
            f"{xyz.stem} fc={fc} ORCA {tag:16s} E_corr={row['e_corr']:+.10f}",
            flush=True,
        )
    d_ri = abs(runs["ri_mp2"]["e_corr"] - py_row["e_corr"])
    if d_ri > TOL_ORCA_RIMP2:
        raise RuntimeError(
            f"{xyz.stem} fc={fc}: ORCA RI-MP2 vs PySCF DFMP2 |d| {d_ri:.2e} — "
            "refusing to write"
        )
    e_ri = runs["ri_mp2"]["e_corr"]
    for tag in ("dlpno_normalpno", "dlpno_tightpno"):
        runs[tag]["pct_of_orca_ri_mp2"] = 100.0 * runs[tag]["e_corr"] / e_ri
    return {
        "frozen_core": fc,
        "pyscf_dfmp2": py_row,
        "orca_ri_mp2_vs_pyscf_dfmp2_abs_diff": d_ri,
        "orca": runs,
    }


def run_cc_block(xyz, symbols, e_rhf: float) -> dict:
    runs = {}
    for tag, kw in CC_RUNS.items():
        with tempfile.TemporaryDirectory(prefix="orca_dlpnocc_") as td:
            r = orca_job(xyz, kw, symbols, Path(td))
        runs[tag] = {
            "keywords": kw,
            "e_scf": r["e_scf"],
            "e_corr": r["energy"] - r["e_scf"],
            "e_rhf_vs_pyscf_abs_diff": abs(r["e_scf"] - e_rhf),
            "input": r["input"],
        }
        print(
            f"{xyz.stem} ORCA {tag:24s} E_corr={runs[tag]['e_corr']:+.10f}", flush=True
        )
    ref = runs["ccsd_t"]["e_corr"]
    for tag in ("dlpno_ccsd_t_normalpno", "dlpno_ccsd_t_tightpno"):
        runs[tag]["pct_of_orca_canonical_ccsd_t"] = 100.0 * runs[tag]["e_corr"] / ref
    return runs


def main() -> int:
    import pyscf

    args = sys.argv[1:]
    do_cc = "--ccsdt" in args
    cc_only = "--cc-only" in args  # add the CC context to an existing JSON
    only = {a for a in args if not a.startswith("--")}
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems {sorted(unknown)}; known {sorted(SYSTEMS)}")
    for system in SYSTEMS:
        if only and system not in only:
            continue
        xyz = MOL_DIR / f"{system}.xyz"
        if cc_only:
            path = common.reference_path(ROW, system, BASIS)
            payload = json.loads(path.read_text())
            payload["context_ccsd_t_fc0"] = run_cc_block(
                xyz, common.read_xyz(xyz)[0], payload["e_rhf"]
            )
            payload["provenance"]["keywords"]["orca_cc_context"] = CC_RUNS
            path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"updated {path}", flush=True)
            continue
        mol, mf, symbols, coords, basis_check, py_rows, naux = pyscf_reference(xyz)
        blocks = [run_mp2_block(xyz, symbols, mf, naux, r) for r in py_rows]
        nocc = mol.nelectron // 2
        payload = {
            "row": ROW_NAME,
            "system": system,
            "name": SYSTEMS[system],
            "basis": BASIS,
            "aux_basis": AUX,
            "nao": mol.nao_nr(),
            "naux": naux,
            "nocc": nocc,
            "nvir": mol.nao_nr() - nocc,
            "n_heavy_atoms": n_heavy(symbols),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "e_rhf": float(mf.e_tot),
            "variants": blocks,
            "note": (
                "NOT like-for-like for DLPNO: ORCA localizes the occupieds "
                "(Foster-Boys) and truncates PAO domains / PNOs / pairs by its own "
                "thresholds; ferric's DLPNO-MP2 consumes CANONICAL occupieds. The "
                "only like-for-like pair is ORCA RI-MP2 vs PySCF DFMP2."
            ),
        }
        if do_cc:
            payload["context_ccsd_t_fc0"] = run_cc_block(xyz, symbols, float(mf.e_tot))
        payload["provenance"] = common.provenance(
            code="ORCA + PySCF",
            version=f"ORCA {blocks[0]['orca']['ri_mp2'].get('version', '6.1.1')}; "
            f"PySCF {pyscf.__version__}",
            keywords={
                "orca": MP2_RUNS,
                "orca_cc_context": CC_RUNS if do_cc else "not run",
                "orca_nprocs": ORCA_NPROCS,
                "pyscf": "scf.RHF exact J/K conv_tol 1e-12; mp.dfmp2.DFMP2(frozen=fc), "
                f"auxmol from ferric's {AUX} JSON; numpy cross-check",
            },
            basis_name=BASIS,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            aux=AUX,
            frozen_core=[b["frozen_core"] for b in blocks],
            scf_conv={"conv_tol": 1e-12, "conv_tol_grad": 1e-10},
            generator="scripts/validation/gen_dlpno.py",
            extra={
                "basis_self_check": basis_check,
                "aux_basis_json_sha256": common.sha256_file(
                    common.basis_json_path(AUX)
                ),
                "orca_binary": str(common.ORCA_BINARY),
                "numpy": np.__version__,
            },
        )
        path = common.write_reference(ROW, system, BASIS, payload)
        print(f"wrote {path}", flush=True)
    print("GEN_DLPNO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
