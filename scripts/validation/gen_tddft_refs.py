#!/usr/bin/env python3
"""PySCF TDA / TDDFT (Casida) references for ferric's user-facing `run_tddft`.

Validation campaign row V-TDDFT (defect F1). Output:
testdata/reference/validation/tddft/<mol>_<basis>.json, read by
crates/ferric-tddft/tests/validation_tddft.rs.

WHAT IS MATCHED TO FERRIC, AND HOW
----------------------------------
The point is a like-for-like comparison, so every approximation ferric makes
is reproduced here rather than left as an unexplained residual:

* Basis functions: built from ferric's OWN bundled JSON
  (crates/ferric-core/src/basis/bundled/*.json), one segmented shell per
  coefficient column, exactly like `parse_bse_json` -- orbital basis, SCF JK
  aux, and response aux alike. (scripts/gen_pyscf_directjk_refs.py measured
  that PySCF's built-in cc-pVDZ layout differs AO-by-AO from ferric's.)
* Reference SCF: RI-J (+RI-K for HF/hybrids) with def2-universal-jkfit, the
  aux the CLI and Python `run_tddft` use for the reference.
* XC grid: (75, 110) Becke-Lebedev, unpruned, Becke radii adjustment -- the
  recipe of scripts/gen_pyscf_dft_refs.py (ferric's default main grid).
* Functionals: libxc identifiers identical to ferric's friendly-name mapping
  (crates/ferric-dft/src/libxc.rs::friendly_to_libxc), so no VWN3/VWN5 or
  B3LYP-variant ambiguity.
* Response Coulomb/exchange integrals: ferric builds (pq|rs) by Coulomb-metric
  RI over the response aux (`build_b_tensors`: Cholesky V^{-1/2}). Here
  `tdscf.rhf.get_ab` is run with `ao2mo.general` REPLACED by the same RI
  factorisation, so its A/B carry ferric's RI error exactly, while its XC
  kernel block is PySCF's own (the thing under test). The exact-4-index
  energies are stored too (`omega_ev_exact_eri`) to show the RI size, but the
  test compares against the RI-matched ones.
* Aux for the response: 6-31G -> cc-pvdz-ri (the CLI's TDDFT_DEFAULT_AUX
  alias), aug-cc-pVDZ -> aug-cc-pvdz-rifit.

Both A (TDA) and the Casida problem are solved DENSE, the way ferric does:
TDA: eigh(A); Casida: eigh((A-B)^1/2 (A+B) (A-B)^1/2) = Omega^2. No Davidson
tolerance enters either side.

NEGATIVE CONTROL: for every functional the kernel-LESS TDA energies
(`tda_nofxc`, i.e. A without PySCF's `iajb` term -- what ferric's run_tddft
returned before F1 was fixed) are stored, so the Rust test can assert that
the reference distinguishes the fixed code from the defect.

Usage (from the repo root):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_refs.py
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_refs.py water 6-31g
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import __version__ as PYSCF_VERSION
from pyscf import dft, gto, scf, tdscf

ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ROOT / "crates" / "ferric-core" / "src" / "basis" / "bundled"
MOLDIR = ROOT / "testdata" / "molecules"
OUTDIR = ROOT / "testdata" / "reference" / "validation" / "tddft"

HARTREE2EV = 27.211386245988
NSTATES = 10

MOLECULES = {
    # key: xyz file under testdata/molecules (Angstrom), charge 0, singlet
    "water": "water.xyz",
    "formaldehyde": "h2co.xyz",
    "nh3": "nh3.xyz",
}

# orbital basis -> response RI aux (ferric bundled names)
BASES = {
    "6-31g": "cc-pvdz-ri",
    "aug-cc-pvdz": "aug-cc-pvdz-rifit",
}
SCF_JK_AUX = "def2-universal-jkfit"

# ferric name -> libxc identifier string for PySCF (None = HF). Identical
# components to ferric's friendly_to_libxc.
FUNCTIONALS = {
    "HF": None,
    "LDA": "LDA_X,LDA_C_VWN",
    "PBE": "GGA_X_PBE,GGA_C_PBE",
    "B3LYP": "HYB_GGA_XC_B3LYP",
}

MAIN_GRID = (75, 110)
Z = {"H": "1", "C": "6", "N": "7", "O": "8"}


def ferric_basis(name, symbols):
    """PySCF basis dict from ferric's bundled BSE JSON, mirroring parse_bse_json.

    Every coefficient COLUMN becomes its own segmented shell, in file order;
    SP-style multi-l entries give column k to angular_momentum[k]. PySCF
    renormalizes contractions on input, as ferric does
    (`renormalize_contraction`).
    """
    fname = {"cc-pvdz-rifit": "cc-pvdz-ri"}.get(name, name)
    data = json.loads((BUNDLED / f"{fname}.json").read_text())
    out = {}
    for sym in sorted(set(symbols)):
        shells = []
        for sh in data["elements"][Z[sym]]["electron_shells"]:
            ang = sh["angular_momentum"]
            exps = [float(x) for x in sh["exponents"]]
            cols = [[float(x) for x in c] for c in sh["coefficients"]]
            if len(ang) == 1:
                for col in cols:
                    prims = [[e, c] for e, c in zip(exps, col) if c != 0.0]
                    shells.append([ang[0]] + prims)
            else:
                for k, ell in enumerate(ang):
                    prims = [[e, c] for e, c in zip(exps, cols[k]) if c != 0.0]
                    shells.append([ell] + prims)
        out[sym] = shells
    return out


def read_xyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def make_mol(atoms, basis_name):
    symbols = [a[0] for a in atoms]
    return gto.M(
        atom=[[a[0], a[1:]] for a in atoms],
        basis=ferric_basis(basis_name, symbols),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )


def ri_factor(mol, aux_name):
    """L^{-1}(P|mn) with V = L L^T: (mn|ls) ~= sum_P B[P,mn] B[P,ls].

    Same factorisation as ferric's `metric_inverse_sqrt` (Cholesky for the
    Coulomb operator).
    """
    auxmol = gto.M(
        atom=mol.atom,
        basis=ferric_basis(aux_name, [mol.atom_symbol(i) for i in range(mol.natm)]),
        unit=mol.unit,
        cart=False,
        verbose=0,
    )
    from pyscf.df import incore

    int3c = incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")  # (nao,nao,naux)
    v2c = auxmol.intor("int2c2e")
    low = scipy.linalg.cholesky(v2c, lower=True)
    nao = mol.nao
    b = scipy.linalg.solve_triangular(low, int3c.reshape(nao * nao, -1).T, lower=True)
    return b.reshape(-1, nao, nao)


class RiAo2mo:
    """Drop-in for the `ao2mo` module inside `pyscf.tdscf.rhf`: only
    `general(mol, (c1,c2,c3,c4), compact=False)` is used by `get_ab`."""

    def __init__(self, b_ao):
        self.b_ao = b_ao

    def general(self, mol, mos, compact=False, **_kw):
        c1, c2, c3, c4 = mos
        b12 = np.einsum("Pmn,mi,nj->Pij", self.b_ao, c1, c2, optimize=True)
        b34 = np.einsum("Pmn,mk,nl->Pkl", self.b_ao, c3, c4, optimize=True)
        eri = np.einsum("Pij,Pkl->ijkl", b12, b34, optimize=True)
        return eri.reshape(c1.shape[1] * c2.shape[1], c3.shape[1] * c4.shape[1])


def get_ab_ri(mf, b_ao):
    """PySCF get_ab with ferric's RI response integrals."""
    real = tdscf.rhf.ao2mo
    tdscf.rhf.ao2mo = RiAo2mo(b_ao)
    try:
        return tdscf.rhf.get_ab(mf)
    finally:
        tdscf.rhf.ao2mo = real


def hf_part_ab(mf, b_ao, hyb):
    """Diagonal + 2(ia|jb) - hyb (ij|ab) [A] and 2(ia|jb) - hyb (ib|ja) [B],
    RI integrals, NO XC kernel -- the pre-F1 ferric run_tddft matrices."""
    occ = mf.mo_occ > 0
    orbo, orbv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    e = mf.mo_energy
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    bov = np.einsum("Pmn,mi,na->Pia", b_ao, orbo, orbv, optimize=True)
    boo = np.einsum("Pmn,mi,nj->Pij", b_ao, orbo, orbo, optimize=True)
    bvv = np.einsum("Pmn,ma,nb->Pab", b_ao, orbv, orbv, optimize=True)
    iajb = np.einsum("Pia,Pjb->iajb", bov, bov, optimize=True)
    ijab = np.einsum("Pij,Pab->iajb", boo, bvv, optimize=True)
    a = np.diag((e[~occ][None, :] - e[occ][:, None]).ravel()).reshape(
        nocc, nvir, nocc, nvir
    )
    a = a + 2 * iajb - hyb * ijab
    b = 2 * iajb - hyb * np.einsum("ibja->iajb", iajb)
    return a, b


def dense_tda(a):
    n = a.shape[0] * a.shape[1]
    am = a.reshape(n, n)
    asym = np.abs(am - am.T).max() / max(np.abs(am).max(), 1e-30)
    assert asym < 1e-9, f"A not symmetric: {asym:e}"
    return np.linalg.eigh(0.5 * (am + am.T))


def dense_casida(a, b):
    n = a.shape[0] * a.shape[1]
    am, bm = a.reshape(n, n), b.reshape(n, n)
    apb, amb = am + bm, am - bm
    apb, amb = 0.5 * (apb + apb.T), 0.5 * (amb + amb.T)
    lam, u = np.linalg.eigh(amb)
    assert lam.min() > 0, f"(A-B) not positive definite: {lam.min():e}"
    amb_sqrt = (u * np.sqrt(lam)) @ u.T
    w2, z = np.linalg.eigh(amb_sqrt @ apb @ amb_sqrt)
    assert w2.min() > 0, f"Casida instability: Omega^2 = {w2.min():e}"
    w = np.sqrt(w2)
    xpy = (amb_sqrt @ z) / np.sqrt(w)[None, :]  # |X|^2 - |Y|^2 = 1
    return w, xpy


def states(mf, w, amps, nocc, nvir):
    """Energies, PySCF-convention oscillator strengths, dominant (i, a)."""
    occ = mf.mo_occ > 0
    orbo, orbv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    with mf.mol.with_common_orig((0, 0, 0)):
        mu_ao = mf.mol.intor("int1e_r", comp=3)
    dip_ia = np.einsum("pi,xpq,qa->xia", orbo, mu_ao, orbv)
    out = {"omega_ev": [], "omega_ha": [], "osc": [], "dominant_ia": []}
    for k in range(min(NSTATES, len(w))):
        x = amps[:, k].reshape(nocc, nvir)
        mu = np.sqrt(2.0) * np.einsum("ia,xia->x", x, dip_ia)
        idx = int(np.argmax(x * x))
        out["omega_ha"].append(float(w[k]))
        out["omega_ev"].append(float(w[k] * HARTREE2EV))
        out["osc"].append(float((2.0 / 3.0) * w[k] * (mu @ mu)))
        out["dominant_ia"].append(list(divmod(idx, nvir)))
    return out


def run_scf(mol, xc):
    if xc is None:
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.atom_grid = MAIN_GRID
        mf.grids.prune = None
        mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    jk_aux = ferric_basis(SCF_JK_AUX, [mol.atom_symbol(i) for i in range(mol.natm)])
    mf = mf.density_fit(auxbasis=jk_aux)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"SCF not converged ({xc})"
    return mf


def run_case(mol_key, basis):
    atoms = read_xyz(MOLDIR / MOLECULES[mol_key])
    mol = make_mol(atoms, basis)
    aux = BASES[basis]
    b_ao = ri_factor(mol, aux)
    rec = {
        "molecule": mol_key,
        "xyz": f"testdata/molecules/{MOLECULES[mol_key]}",
        "basis": basis,
        "response_aux": aux,
        "scf_jk_aux": SCF_JK_AUX,
        "grid": {"atom_grid": list(MAIN_GRID), "prune": None, "radii_adjust": "becke"},
        "nstates": NSTATES,
        "pyscf_version": PYSCF_VERSION,
        "generator": "scripts/validation/gen_tddft_refs.py",
        "note": "dense A / Casida solves; response (pq|rs) by Coulomb-metric RI over "
        "response_aux (ferric's factorisation); basis from ferric bundled JSON; "
        "osc = 2/3 w |sqrt(2) sum (X+Y) <i|r|a>|^2 with |X|^2-|Y|^2 = 1",
        "cases": {},
    }
    for name, xc in FUNCTIONALS.items():
        print(f"  {mol_key}/{basis}/{name}", file=sys.stderr)
        mf = run_scf(mol, xc)
        nocc = int((mf.mo_occ > 0).sum())
        nvir = mf.mo_occ.size - nocc
        if xc is None:
            hyb = 1.0
        else:
            omega, _alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            assert omega == 0, "RSH functionals are out of scope"

        a, b = get_ab_ri(mf, b_ao)
        w_tda, x_tda = dense_tda(a)
        w_cas, xpy = dense_casida(a, b)

        a0, _b0 = hf_part_ab(mf, b_ao, hyb)
        w_nofxc, _ = dense_tda(a0)
        if xc is None:
            # HF: get_ab's HF part must equal this independent RI assembly.
            assert np.allclose(w_nofxc, w_tda, atol=1e-10), "RI A assembly mismatch"

        a_ex, b_ex = tdscf.rhf.get_ab(mf)  # exact 4-index ERIs, for information
        w_tda_ex, _ = dense_tda(a_ex)
        w_cas_ex, _ = dense_casida(a_ex, b_ex)

        rec["cases"][name] = {
            "xc_pyscf": xc,
            "c_hf": float(hyb),
            "e_scf": float(mf.e_tot),
            "nocc": nocc,
            "nvir": nvir,
            "tda": {
                **states(mf, w_tda, x_tda, nocc, nvir),
                "omega_ev_exact_eri": [
                    float(v * HARTREE2EV) for v in w_tda_ex[:NSTATES]
                ],
            },
            "casida": {
                **states(mf, w_cas, xpy, nocc, nvir),
                "omega_ev_exact_eri": [
                    float(v * HARTREE2EV) for v in w_cas_ex[:NSTATES]
                ],
            },
            "tda_nofxc": {
                "omega_ev": [float(v * HARTREE2EV) for v in w_nofxc[:NSTATES]]
            },
        }
        t = rec["cases"][name]
        print(
            f"    TDA  {[round(v, 4) for v in t['tda']['omega_ev'][:5]]}\n"
            f"    CAS  {[round(v, 4) for v in t['casida']['omega_ev'][:5]]}\n"
            f"    noK  {[round(v, 4) for v in t['tda_nofxc']['omega_ev'][:5]]}\n"
            f"    RI-exact TDA S1: {t['tda']['omega_ev'][0] - t['tda']['omega_ev_exact_eri'][0]:+.2e} eV",
            file=sys.stderr,
        )
    return rec


def main(argv):
    mols = [argv[1]] if len(argv) > 1 else list(MOLECULES)
    bases = [argv[2]] if len(argv) > 2 else list(BASES)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for m in mols:
        for bname in bases:
            rec = run_case(m, bname)
            path = OUTDIR / f"{m}_{bname}.json"
            path.write_text(json.dumps(rec, indent=1) + "\n")
            print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
