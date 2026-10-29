#!/usr/bin/env python3
"""PySCF references for every bundled basis set (orbital, RI and JK).

Writes ``testdata/reference/basis_coverage_pyscf.json``, consumed by
``crates/ferric-mp2/tests/basis_coverage_vs_pyscf.rs``. Every registered set
appears in at least one case:

* orbital sets: an RHF energy on a small molecule containing the set's
  elements (H2O for everything that has H and O, plus a heavier molecule
  where the set gained elements or carries an ECP);
* RI / JK auxiliary sets: an RI-MP2 correlation energy (all electrons
  correlated, exact-integral RHF reference) with the matching orbital set;
* JK sets additionally: a density-fitted RHF energy (J and K both fitted).

The basis data PySCF uses are fetched from the Basis Set Exchange BY NAME
(``basis_set_exchange`` 0.12, NWChem format, parsed by PySCF's own parser).
Nothing is read from ferric's bundled JSON, so the reference is an
independent construction of both the data and its normalization. Elements a
bundled set takes from another BSE set (the light elements of the -PP sets)
are fetched from that set, as recorded in
``crates/ferric-core/src/basis/provenance.json``.

PySCF's ``cart`` flag is global per molecule while BSE types every shell, so
each case asserts that all its l >= 2 shells share one type and sets ``cart``
from it. Run under the pinned environment, one PySCF job at a time::

    flock -o /tmp/ferric-compile.lock \\
      uvx --from basis_set_exchange==0.12 --with pyscf==2.13.1 \\
      python scripts/gen_pyscf_basis_coverage_refs.py [--update]

``--update`` keeps every case already in the output and computes only new
ones. The aug-cc-pVXZ-JKFIT sets are not in BSE; for them PySCF's own copy is
used (proved bit-identical to ferric's by scripts/import_psi4_jkfit.py).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import basis_set_exchange as bse
import pyscf
from pyscf import gto, scf
from pyscf.mp import dfmp2_native

ROOT = Path(__file__).resolve().parent.parent
PROV = json.loads((ROOT / "crates/ferric-core/src/basis/provenance.json").read_text())
OUT = ROOT / "testdata/reference/basis_coverage_pyscf.json"

MOLS = {
    "h2o": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0),
    "hcl": ("H 0 0 0; Cl 0 0 1.2746", 0),
    "hbr": ("H 0 0 0; Br 0 0 1.4145", 0),
    "agh": ("Ag 0 0 0; H 0 0 1.618", 0),
    "hi": ("H 0 0 0; I 0 0 1.609", 0),
    "ne": ("Ne 0 0 0", 0),
    "pd": ("Pd 0 0 0", 0),
}

# (orbital set, molecule) -- RHF energy.
RHF_CASES: list[tuple[str, str]] = [
    ("sto-3g", "h2o"),
    ("sto-3g", "hbr"),
    ("6-31g", "h2o"),
    ("6-31g", "hbr"),
    ("6-31g*", "h2o"),
    ("6-31g*", "hbr"),
    ("6-31g**", "h2o"),
    ("6-31+g*", "h2o"),
    ("6-31+g**", "h2o"),
    ("6-31++g**", "h2o"),
    ("6-311g**", "h2o"),
    ("6-311g**", "hcl"),
    ("6-311++g**", "h2o"),
    ("6-311++g**", "hcl"),
    ("cc-pvdz", "h2o"),
    ("cc-pvdz", "hbr"),
    ("cc-pvtz", "h2o"),
    ("cc-pvqz", "h2o"),
    ("aug-cc-pvdz", "h2o"),
    ("aug-cc-pvtz", "h2o"),
    ("aug-cc-pvqz", "h2o"),
    ("cc-pvdz-f12", "h2o"),
    ("cc-pvdz-f12", "ne"),
    ("cc-pvdz-pp", "hbr"),
    ("cc-pvtz-pp", "hbr"),
    ("cc-pvqz-pp", "hbr"),
    ("aug-cc-pvdz-pp", "hbr"),
    ("aug-cc-pvtz-pp", "hbr"),
    ("aug-cc-pvqz-pp", "hbr"),
    ("def2-svp", "h2o"),
    ("def2-svp", "agh"),
    ("def2-svpd", "h2o"),
    ("def2-tzvp", "h2o"),
    ("def2-tzvpp", "h2o"),
    ("def2-tzvpd", "h2o"),
    ("def2-tzvppd", "h2o"),
    ("def2-tzvppd", "hi"),
    ("def2-qzvp", "h2o"),
    ("def2-qzvpp", "h2o"),
    ("def2-qzvppd", "h2o"),
    ("pcseg-0", "h2o"),
    ("pcseg-1", "h2o"),
    ("pcseg-2", "h2o"),
    ("aug-pcseg-1", "h2o"),
    ("aug-pcseg-2", "h2o"),
]

# (orbital set, auxiliary set, molecule) -- RI-MP2 correlation energy.
RI_CASES: list[tuple[str, str, str]] = [
    ("cc-pvdz", "cc-pvdz-rifit", "h2o"),
    ("cc-pvtz", "cc-pvtz-rifit", "h2o"),
    ("cc-pvqz", "cc-pvqz-rifit", "h2o"),
    ("aug-cc-pvdz", "aug-cc-pvdz-rifit", "h2o"),
    ("aug-cc-pvtz", "aug-cc-pvtz-rifit", "h2o"),
    ("aug-cc-pvqz", "aug-cc-pvqz-rifit", "h2o"),
    ("cc-pvdz-pp", "cc-pvdz-pp-rifit", "pd"),
    ("cc-pvtz-pp", "cc-pvtz-pp-rifit", "pd"),
    # The QZ-PP ORBITAL sets put h (l=5) shells on Pd, beyond ferric's ECP
    # integrals (l <= 4, a clean setup error); the QZ PP-RIFIT data are
    # checked on the TZ-PP orbital set instead (aux shells never enter V_ECP).
    ("cc-pvtz-pp", "cc-pvqz-pp-rifit", "pd"),
    ("aug-cc-pvdz-pp", "aug-cc-pvdz-pp-rifit", "pd"),
    ("aug-cc-pvtz-pp", "aug-cc-pvtz-pp-rifit", "pd"),
    ("aug-cc-pvtz-pp", "aug-cc-pvqz-pp-rifit", "pd"),
    ("cc-pvtz", "cc-pvtz-jkfit", "h2o"),
    ("cc-pvqz", "cc-pvqz-jkfit", "h2o"),
    ("cc-pvdz-f12", "cc-pvdz-f12-optri", "h2o"),
    ("def2-svp", "def2-svp-rifit", "h2o"),
    ("def2-svpd", "def2-svpd-rifit", "h2o"),
    ("def2-tzvp", "def2-tzvp-rifit", "h2o"),
    ("def2-tzvpp", "def2-tzvpp-rifit", "h2o"),
    ("def2-tzvpd", "def2-tzvpd-rifit", "h2o"),
    ("def2-tzvppd", "def2-tzvppd-rifit", "h2o"),
    ("def2-qzvp", "def2-qzvp-rifit", "h2o"),
    ("def2-qzvpp", "def2-qzvpp-rifit", "h2o"),
    ("def2-qzvppd", "def2-qzvppd-rifit", "h2o"),
    ("def2-svp", "def2-universal-jkfit", "h2o"),
    ("aug-cc-pvtz", "aug-cc-pvtz-jkfit", "h2o"),
]


# (orbital set, J/K auxiliary set, molecule) -- density-fitted RHF energy,
# J and K both fitted with the aux (PySCF ``density_fit``).
DFHF_CASES: list[tuple[str, str, str]] = [
    ("aug-cc-pvdz", "aug-cc-pvdz-jkfit", "h2o"),
    ("aug-cc-pvtz", "aug-cc-pvtz-jkfit", "h2o"),
    ("aug-cc-pvtz", "aug-cc-pvtz-jkfit", "ne"),
    ("aug-cc-pvqz", "aug-cc-pvqz-jkfit", "h2o"),
    ("cc-pvtz", "cc-pvtz-jkfit", "h2o"),
    ("def2-svp", "def2-universal-jkfit", "h2o"),
]


def registry() -> dict[str, str]:
    """Primary bundled name -> file stem, read from basis.rs's REGISTRY."""
    src = (ROOT / "crates/ferric-core/src/basis.rs").read_text()
    body = src[src.index("const REGISTRY") : src.index("const UNREGISTERED_ALLOWLIST")]
    pairs = re.findall(r'\(\s*"([^"]+)",\s*&\[\s*"([^"]+)"', body)
    return {name: stem for stem, name in pairs}


REG = registry()
BY_FILE = {rec["file"]: rec for rec in PROV["sets"]}


def source_name(bundled: str, z: int) -> str:
    """The BSE set that element z of bundled set `bundled` comes from."""
    rec = BY_FILE[REG[bundled] + ".json"]
    if z in rec["bse_elements"]:
        return rec["bse_name"]
    fill = rec.get("filled_from")
    if fill and z in fill["elements"]:
        return fill["bse_name"]
    raise SystemExit(
        f"{bundled}: Z={z} is not from BSE (shipped-only); pick another molecule"
    )


def element_basis(bundled: str, symbols: set[str]) -> tuple[dict, dict, set[str]]:
    """(basis dict, ecp dict, set of l>=2 function types) for PySCF."""
    basis, ecp, ftypes = {}, {}, set()
    rec = BY_FILE[REG[bundled] + ".json"]
    for sym in sorted(symbols):
        z = gto.charge(sym)
        if rec.get("source") == "psi4":
            # Not in BSE. PySCF's own copy, which import_psi4_jkfit.py proved
            # bit-identical to the Psi4 file ferric ships: this case checks
            # ferric's parse/normalization, the import script checks digits.
            assert z in rec["elements"], f"{bundled}: no Z={z}"
            basis[sym] = gto.basis.load(REG[bundled], sym)
            ftypes.add("gto_spherical")
            continue
        name = source_name(bundled, z)
        data = json.loads(bse.get_basis(name, elements=[z], fmt="json", header=False))
        for sh in data["elements"][str(z)].get("electron_shells", []):
            if max(sh["angular_momentum"]) >= 2:
                ftypes.add(sh["function_type"])
        text = bse.get_basis(name, elements=[z], fmt="nwchem", header=False)
        # BSE's NWChem output is the basis block followed by an ECP block;
        # each PySCF parser wants only its own.
        orb_text, _, ecp_text = text.partition("\nECP\n")
        basis[sym] = gto.basis.parse(orb_text, sym)
        if "ecp_potentials" in data["elements"][str(z)]:
            ecp[sym] = gto.basis.parse_ecp("ECP\n" + ecp_text, sym)
    return basis, ecp, ftypes


def build(mol_key: str, orb: str, aux: str | None):
    atom, charge = MOLS[mol_key]
    symbols = {a.split()[0] for a in atom.split(";")}
    basis, ecp, ftypes = element_basis(orb, symbols)
    aux_basis = None
    if aux is not None:
        aux_basis, aux_ecp, aux_ftypes = element_basis(aux, symbols)
        assert not aux_ecp, f"{aux}: an auxiliary set must not carry an ECP"
        ftypes |= aux_ftypes
    assert len(ftypes) <= 1, f"{orb}/{aux}/{mol_key}: mixed l>=2 types {ftypes}"
    cart = ftypes == {"gto_cartesian"}
    mol = gto.M(
        atom=atom,
        basis=basis,
        ecp=ecp or None,
        charge=charge,
        spin=0,
        cart=cart,
        unit="Angstrom",
        verbose=0,
    )
    return mol, aux_basis, cart, bool(ecp)


def run_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    e = mf.kernel()
    assert mf.converged, "PySCF RHF did not converge"
    return mf, float(e)


def df_mp2_corr(mf, aux_basis) -> float:
    """All-electron closed-shell RI-MP2 correlation energy (PySCF dfmp2_native).

    Not ``mp.MP2(mf).density_fit()``: PySCF 2.13.1's DFMP2 direct path
    asserts on a non-F-contiguous metric under the numpy uvx resolves.
    """
    pt = dfmp2_native.DFRMP2(mf, auxbasis=aux_basis)
    pt.kernel()
    return float(pt.e_corr)


def case_record(kind, orb, aux, mol_key, mol, cart, has_ecp, e_rhf, e_corr, t0):
    return {
        "kind": kind,
        "basis": orb,
        "auxbasis": aux,
        "molecule": mol_key,
        "atoms_angstrom": MOLS[mol_key][0],
        "charge": MOLS[mol_key][1],
        "cart": cart,
        "ecp": has_ecp,
        "nao": int(mol.nao_nr()),
        "nelectron": int(mol.nelectron),
        "e_rhf": e_rhf,
        "e_mp2_corr": e_corr,
        "seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    # --update reuses every case already in OUT and computes only new ones.
    old = {}
    if "--update" in sys.argv and OUT.exists():
        for c in json.loads(OUT.read_text())["cases"]:
            old[(c["kind"], c["basis"], c["auxbasis"], c["molecule"])] = c
    cases = []
    for orb, aux, mol_key in DFHF_CASES:
        if (hit := old.get(("dfhf", orb, aux, mol_key))) is not None:
            cases.append(hit)
            continue
        t0 = time.time()
        mol, aux_basis, cart, has_ecp = build(mol_key, orb, aux)
        assert not cart, "DF cases must be spherical: PySCF's auxmol inherits cart"
        mf = scf.RHF(mol).density_fit(auxbasis=aux_basis)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 200
        e = float(mf.kernel())
        assert mf.converged, "PySCF DF-RHF did not converge"
        cases.append(
            case_record("dfhf", orb, aux, mol_key, mol, cart, has_ecp, e, None, t0)
        )
        print(f"dfhf   {orb:14s} {aux:22s} {mol_key:4s} E={e:.10f}", flush=True)
    for orb, mol_key in RHF_CASES:
        if (hit := old.get(("rhf", orb, None, mol_key))) is not None:
            cases.append(hit)
            continue
        t0 = time.time()
        mol, _, cart, has_ecp = build(mol_key, orb, None)
        _, e = run_rhf(mol)
        cases.append(
            case_record("rhf", orb, None, mol_key, mol, cart, has_ecp, e, None, t0)
        )
        print(
            f"rhf    {orb:22s} {mol_key:4s} nao={mol.nao_nr():4d} E={e:.10f}",
            flush=True,
        )
    for orb, aux, mol_key in RI_CASES:
        if (hit := old.get(("rimp2", orb, aux, mol_key))) is not None:
            cases.append(hit)
            continue
        t0 = time.time()
        mol, aux_basis, cart, has_ecp = build(mol_key, orb, aux)
        assert not cart, "RI cases must be spherical: PySCF's auxmol inherits cart"
        mf, e = run_rhf(mol)
        ec = df_mp2_corr(mf, aux_basis)
        cases.append(
            case_record("rimp2", orb, aux, mol_key, mol, cart, has_ecp, e, ec, t0)
        )
        print(f"rimp2  {orb:14s} {aux:22s} {mol_key:4s} Ec={ec:.10f}", flush=True)

    names = set(REG)
    covered = {c["basis"] for c in cases} | {
        c["auxbasis"] for c in cases if c["auxbasis"]
    }
    missing = sorted(names - covered)
    if missing:
        print(f"ERROR: registered sets with no case: {missing}")
        return 1
    OUT.write_text(
        json.dumps(
            {
                "generator": "scripts/gen_pyscf_basis_coverage_refs.py",
                "pyscf_version": pyscf.__version__,
                "bse_version": bse.version(),
                "scf": "RHF, exact integrals, conv_tol 1e-12; MP2 all-electron (no frozen core)",
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
