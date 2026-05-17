"""Generate PySCF RI-RPA reference energies for the C9 benchmark (S66x8 / L7).

These are *apples-to-apples* references for ferric RI-RPA — same basis, same
auxiliary, same SCF convention — so MAE(ferric - PySCF) should be sub-1e-4 Ha.
This is distinct from the published CCSD(T)/CBS references in
`s66x8_ccsdt_cbs.json` (which are the "vs gold standard" comparison).

Convention (see memory `pyscf-ri-rpa-convention.md`):
    mol = gto.M(atom=..., basis='cc-pvdz')
    mf = scf.RHF(mol).run()                       # NON-DF SCF
    rpa = RPA(mf)
    rpa.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')   # manual DF override
    rpa.kernel()
    e_corr = rpa.e_corr

Output JSON: { name: { e_rhf, e_corr_rpa, e_total } }, energies in Hartree.

Run from repo root (full sweep, MANY hours):
    python scripts/gen_pyscf_c9_refs.py --tier s66x8

Smoke test (1 system only):
    python scripts/gen_pyscf_c9_refs.py --tier s66x8 --only s66_01_waterwater_1.00

NOTE: This script intentionally does NOT compute monomer energies (and hence
no interaction energies). The point is to validate ferric *total* RPA energies
against PySCF; CCSD(T)/CBS comparisons happen separately via the published refs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Prefer local PySCF checkout when present (see memory: local-pyscf-checkout).
LOCAL_PYSCF = Path.home() / "qc" / "pyscf"
if LOCAL_PYSCF.exists():
    sys.path.insert(0, str(LOCAL_PYSCF))

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_xyz(path: Path) -> str:
    """Return PySCF-style atom block (skip first two XYZ lines, return rest)."""
    lines = path.read_text().splitlines()
    return "\n".join(lines[2:])


def run_one(xyz_str: str, basis: str = "cc-pvdz", auxbasis: str = "cc-pvdz-ri") -> dict:
    from pyscf import gto, scf, df
    from pyscf.gw.rpa import RPA

    mol = gto.M(atom=xyz_str, basis=basis, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.run()
    rpa = RPA(mf)
    rpa.with_df = df.DF(mol, auxbasis=auxbasis)
    rpa.kernel()
    return {
        "e_rhf": float(mf.e_tot),
        "e_corr_rpa": float(rpa.e_corr),
        "e_total": float(mf.e_tot + rpa.e_corr),
        "n_ao": int(mol.nao_nr()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["s66x8", "l7"], required=True)
    ap.add_argument("--only", help="Comma-separated system names (basenames without .xyz)")
    ap.add_argument("--output", help="JSON output path (default tier-specific)")
    args = ap.parse_args()

    sysroot = REPO_ROOT / "testdata/molecules/c9_systems" / args.tier
    out_path = Path(args.output) if args.output else (
        REPO_ROOT / "testdata/reference/c9_refs" / f"{args.tier}_pyscf_rpa.json"
    )

    only: set[str] | None = None
    if args.only:
        only = {s.strip() for s in args.only.split(",")}

    # Load existing output (allow incremental population).
    out: dict = {}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text())
        except Exception:
            out = {}

    xyz_files = sorted(sysroot.glob("*.xyz"))
    print(f"[{args.tier}] {len(xyz_files)} xyz files in {sysroot}", file=sys.stderr)

    for xyz in xyz_files:
        name = xyz.stem
        if only is not None and name not in only:
            continue
        if name in out and "e_total" in out[name]:
            print(f"[skip] {name} already in {out_path.name}", file=sys.stderr)
            continue
        atoms = read_xyz(xyz)
        print(f"[run]  {name} (n_atoms={atoms.count(chr(10))+1})", file=sys.stderr)
        t0 = time.time()
        try:
            rec = run_one(atoms)
            rec["t_wall_s"] = time.time() - t0
            out[name] = rec
            print(f"       E_total = {rec['e_total']:.8f}  ({rec['t_wall_s']:.1f}s)", file=sys.stderr)
        except Exception as e:
            print(f"       FAIL: {e}", file=sys.stderr)
            out[name] = {"error": str(e)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"DONE: wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
