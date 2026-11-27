#!/usr/bin/env python3
"""PySCF spin-free X2C-1e SCF, DF-MP2 and DF-CCSD(T) references for
crates/ferric-scf/tests/x2c_scf.rs, crates/ferric-mp2/tests/x2c_rimp2.rs and
crates/ferric-python/tests/test_x2c_correlated.py.

Settings: ``scf.RHF(mol).sfx2c1e()`` with ``approx = '1e'``,
``xuncontract = True``, c = ``lib.param.LIGHT_SPEED`` (as
scripts/gen_pyscf_x2c_refs.py), spherical d, ``conv_tol = 1e-11``,
``conv_tol_grad = 1e-7``. Basis sets are built from ferric's bundled JSON:
``cc-pvdz`` (contracted) and ``unc-cc-pvdz`` (one primitive per unique
exponent and angular momentum, as ``ferric_scf::x2c::uncontracted_basis``).
Per system and basis the file stores the RHF energies without (``e_nr``) and
with (``e_x2c``) X2C. For HBr / cc-pVDZ it also stores the DF-MP2
correlation energy on both references with ferric's bundled cc-pVDZ-RI
auxiliary basis, all electrons correlated (``mp2``), and the DF-CCSD
correlation energy and its (T) correction with the same auxiliary basis for
every integral class (``ccsd``: ``pyscf.cc.dfccsd.RCCSD``, ``conv_tol`` 1e-10,
``conv_tol_normt`` 1e-8).

Usage: OPENBLAS_NUM_THREADS=1 python scripts/gen_pyscf_x2c_scf_refs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pyscf import gto, lib, scf
from pyscf.cc import dfccsd
from pyscf.mp import dfmp2

REPO = Path(__file__).resolve().parents[1]
BUNDLED = REPO / "crates/ferric-core/src/basis/bundled"
OUT = REPO / "testdata/reference/x2c_scf_sfx2c1e.json"
SYMBOLS = {1: "H", 34: "Se", 35: "Br"}
SYSTEMS = {
    "HBr": "H 0 0 0\nBr 0 0 1.414",
    "H2Se": "Se 0 0 0\nH 0 1.2 0.9\nH 0 -1.2 0.9",
}
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-7
CC_CONV_TOL = 1e-10
CC_CONV_TOL_NORMT = 1e-8
MAX_MEMORY_MB = 2000


def shells(name: str, z: int) -> list[tuple[int, list[float], list[float]]]:
    data = json.loads((BUNDLED / f"{name}.json").read_text())["elements"]
    out = []
    for sh in data[str(z)]["electron_shells"]:
        assert len(sh["angular_momentum"]) == 1, "sp shells not handled"
        ang = sh["angular_momentum"][0]
        exps = [float(e) for e in sh["exponents"]]
        for col in sh["coefficients"]:
            out.append((ang, exps, [float(c) for c in col]))
    return out


def contracted(name: str, zs: list[int]) -> dict:
    return {
        SYMBOLS[z]: [
            [ang] + [[e, c] for e, c in zip(exps, cs)]
            for ang, exps, cs in shells(name, z)
        ]
        for z in zs
    }


def uncontracted(name: str, zs: list[int]) -> dict:
    out = {}
    for z in zs:
        by_l: dict[int, list[float]] = {}
        for ang, exps, _ in shells(name, z):
            seen = by_l.setdefault(ang, [])
            seen.extend(e for e in exps if e not in seen)
        out[SYMBOLS[z]] = [[ang, [e, 1.0]] for ang, es in by_l.items() for e in es]
    return out


def converged(mf):
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.kernel()
    assert mf.converged
    return mf


def main() -> None:
    lib.num_threads(1)
    systems = {}
    for label, xyz in SYSTEMS.items():
        zs = sorted(
            {
                z
                for z, s in SYMBOLS.items()
                for ln in xyz.splitlines()
                if ln.split()[0] == s
            }
        )
        per_basis = {}
        for bname, basis in (
            ("cc-pvdz", contracted("cc-pvdz", zs)),
            ("unc-cc-pvdz", uncontracted("cc-pvdz", zs)),
        ):
            mol = gto.M(
                atom=xyz.replace("\n", "; "),
                basis=basis,
                unit="Angstrom",
                cart=False,
                verbose=0,
                max_memory=MAX_MEMORY_MB,
            )
            rel = scf.RHF(mol).sfx2c1e()
            x2c = rel.with_x2c
            assert x2c.approx.upper() == "1E" and x2c.xuncontract and x2c.basis is None
            rel = converged(rel)
            nr = converged(scf.RHF(mol))
            entry = {
                "nao": int(mol.nao),
                "e_nr": float(nr.e_tot),
                "e_x2c": float(rel.e_tot),
            }
            if label == "HBr" and bname == "cc-pvdz":
                aux = contracted("cc-pvdz-ri", zs)
                mp2 = {}
                for ref, mf in (("nr", nr), ("x2c", rel)):
                    pt = dfmp2.DFMP2(mf)
                    pt.with_df.auxbasis = aux
                    pt.kernel()
                    mp2[f"e_corr_{ref}"] = float(pt.e_corr)
                entry["mp2"] = mp2
                ccsd = {}
                for ref, mf in (("nr", nr), ("x2c", rel)):
                    cc = dfccsd.RCCSD(mf)
                    cc.with_df.auxbasis = aux
                    cc.conv_tol = CC_CONV_TOL
                    cc.conv_tol_normt = CC_CONV_TOL_NORMT
                    cc.kernel()
                    assert cc.converged
                    ccsd[f"e_corr_{ref}"] = float(cc.e_corr)
                    ccsd[f"e_t_{ref}"] = float(cc.ccsd_t())
                entry["ccsd"] = ccsd
            per_basis[bname] = entry
            print(label, bname, entry)
        systems[label] = {"xyz_angstrom": xyz, "bases": per_basis}
    OUT.write_text(
        json.dumps(
            {
                "generator": "scripts/gen_pyscf_x2c_scf_refs.py",
                "pyscf_version": __import__("pyscf").__version__,
                "light_speed": float(lib.param.LIGHT_SPEED),
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "settings": "scf.RHF(mol).sfx2c1e(), approx='1e', xuncontract=True, cart=False; "
                "mp2: pyscf.mp.dfmp2.DFMP2, all electrons, auxbasis cc-pvdz-ri; "
                "ccsd: pyscf.cc.dfccsd.RCCSD + ccsd_t(), all electrons, auxbasis cc-pvdz-ri",
                "cc_conv_tol": CC_CONV_TOL,
                "cc_conv_tol_normt": CC_CONV_TOL_NORMT,
                "systems": systems,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
