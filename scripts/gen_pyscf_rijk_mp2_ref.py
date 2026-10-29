"""PySCF reference for an RI-JK (density-fitted) RHF reference + DF-MP2.

Anchors ferric's `scf_jk="rijk"` MP2-family option
(`ferric_scf::scf_jk`, `run_rimp2(scf_jk="rijk")`): the reference SCF is
`scf.RHF(mol).density_fit(auxbasis=<JK aux>)` and the correlation is
`mp.dfmp2.DFMP2` on a SEPARATE RI aux basis, so the two fitting bases are
never shared.

Every basis (orbital, JK-fit, MP2-fit) is read from ferric's OWN bundled
BSE JSON and handed to PySCF, so the comparison measures the method and not
digit differences between two basis-set libraries.

Usage: python scripts/gen_pyscf_rijk_mp2_ref.py
Writes testdata/reference/h2o_cc-pvdz_rijk-def2-universal-jkfit_dfmp2.json
"""

import json
import os

from pyscf import df, gto, scf
from pyscf.mp import dfmp2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLED = os.path.join(ROOT, "crates", "ferric-core", "src", "basis", "bundled")
OUT = os.path.join(
    ROOT, "testdata", "reference", "h2o_cc-pvdz_rijk-def2-universal-jkfit_dfmp2.json"
)

BASIS, JK_AUX, RI_AUX = "cc-pvdz", "def2-universal-jkfit", "cc-pvdz-ri"
SYMBOLS = {1: "H", 8: "O"}


def ferric_basis(name: str) -> dict:
    """Ferric's bundled BSE JSON as a PySCF per-element basis dict."""
    with open(os.path.join(BUNDLED, f"{name}.json")) as f:
        data = json.load(f)
    out = {}
    for z, sym in SYMBOLS.items():
        shells = []
        for sh in data["elements"][str(z)]["electron_shells"]:
            (ang,) = sh["angular_momentum"]
            exps = [float(x) for x in sh["exponents"]]
            for coefs in sh["coefficients"]:
                c = [float(x) for x in coefs]
                prims = [(e, ci) for e, ci in zip(exps, c) if ci != 0.0]
                shells.append([ang, *prims])
        out[sym] = shells
    return out


def water() -> gto.Mole:
    with open(os.path.join(ROOT, "testdata", "molecules", "water.xyz")) as f:
        lines = f.read().splitlines()[2:]
    atom = "\n".join(line for line in lines if line.strip())
    return gto.M(atom=atom, basis=ferric_basis(BASIS), unit="Angstrom", verbose=0)


def main() -> None:
    mol = water()
    mf = scf.RHF(mol).density_fit(auxbasis=ferric_basis(JK_AUX))
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    e_hf = mf.kernel()
    assert mf.converged
    pt = dfmp2.DFMP2(mf)
    pt.with_df = df.DF(mol, auxbasis=ferric_basis(RI_AUX))
    e_corr = pt.kernel()[0]
    ref = {
        "molecule": "h2o",
        "basis": BASIS,
        "scf_jk": "rijk",
        "jk_auxbasis": JK_AUX,
        "auxbasis": RI_AUX,
        "frozen_core": 0,
        "rhf_energy": e_hf,
        "mp2_corr": e_corr,
        "total_energy": e_hf + e_corr,
        "generator": "scripts/gen_pyscf_rijk_mp2_ref.py",
        "pyscf_version": __import__("pyscf").__version__,
    }
    with open(OUT, "w") as f:
        json.dump(ref, f, indent=2)
        f.write("\n")
    print(json.dumps(ref, indent=2))


if __name__ == "__main__":
    main()
