#!/usr/bin/env python3
"""PySCF references for ferric's range-separated-hybrid and triplet TDA / TDDFT.

Output: testdata/reference/validation/tddft_rsh/<mol>_<basis>.json, read by
crates/ferric-tddft/tests/validation_tddft_rsh.rs.

WHAT IS MATCHED TO FERRIC, AND HOW
----------------------------------
Everything gen_tddft_refs.py matches (ferric's bundled basis JSON, RI-JK
def2-universal-jkfit reference SCF, identical libxc identifiers, the response
(pq|rs) by ferric's RI factorisation over the response aux), plus:

* Grid: flat (75, 110) Becke-Lebedev, no pruning, Becke radii adjustment, on
  BOTH sides (the Rust test runs ferric on the same flat grid), so grid error
  cancels instead of entering the bar.
* Range-separated SCF exchange: ferric fits c_sr K[erfc] + c_lr K[erf], each
  in its own metric, over the JK aux. PySCF's density-fitted RKS fits
  c_sr K[1/r] + (c_lr - c_sr) K[erf]. `match_rsh_scf_fit` makes PySCF's
  full-range K the sum of its erfc- and erf-fitted K, so both SCFs minimise
  the same fitted functional.
* Range-separated response exchange: ferric builds c_sr (pq|rs) over the
  Coulomb-metric tensors and (c_lr - c_sr) (pq|rs)_omega over erf-metric
  tensors B = M (Q|erf|pq), M = U diag(lambda^-1/2) U^T of the erf metric with
  eigenvalues below 1e-10 dropped (ferric_mp2::rimp2::eigh_inverse_sqrt).
  `RiAo2mo` returns exactly that factorisation when get_ab enters
  `mol.with_range_coulomb(omega)`, and the Coulomb one otherwise.
* Triplets: PySCF's RHF get_ab has no triplet path, so the spin blocks come
  from tdscf.uhf.get_ab on the SAME orbitals (the RKS converted to UKS):
  A_T = A_aa - A_ab, B_T = B_aa - B_ab, with PySCF's own spin-resolved
  kernel (f_aa, f_ab). The singlet combination A_aa + A_ab must equal the RHF
  get_ab A (asserted), which checks the conversion and the RI substitution.

Dense solves on both sides (TDA eigh(A); Casida eigh((A-B)^1/2 (A+B)
(A-B)^1/2)). A Casida problem with Omega^2 < 0 (a triplet instability) is
recorded as unstable with its lowest Omega^2; ferric must refuse it.

Negative controls stored per case: the roots with the long-range exchange
term dropped (`tda_no_lr`, range-separated functionals) and the triplet roots
with the SINGLET kernel sign (`tda_singlet_kernel`, KS triplets).

Usage (from the repo root):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_rsh_refs.py
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_rsh_refs.py water
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import __version__ as PYSCF_VERSION
from pyscf import dft, gto, scf, tdscf
from pyscf.df import incore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tddft_refs import (  # noqa: E402
    HARTREE2EV,
    MOLDIR,
    ROOT,
    SCF_JK_AUX,
    dense_tda,
    ferric_basis,
    make_mol,
    read_xyz,
    states,
)

OUTDIR = ROOT / "testdata" / "reference" / "validation" / "tddft_rsh"
NSTATES = 10
BASIS = "6-31g"
RESPONSE_AUX = "cc-pvdz-ri"
MAIN_GRID = (75, 110)
LINDEP = 1e-10  # ferric_mp2::rimp2::eigh_inverse_sqrt
MAX_MEMORY_MB = 1500
# Grid points per block in get_ab's kernel contraction (a multiple of PySCF's
# BLKSIZE = 56). get_ab floors its block memory at 2000 MB counted in AO
# values only, so for benzene one block is the whole grid and its
# (4, ngrid, nocc, nvir) intermediates reach ~10 GB.
GRID_BLOCK = 56 * 64

MOLECULES = {"water": "water.xyz", "c2h4": "c2h4.xyz", "benzene": "benzene.xyz"}

# ferric name -> PySCF xc (None = HF); identical libxc components.
FUNCTIONALS = {
    "HF": None,
    "LDA": "LDA_X,LDA_C_VWN",
    "PBE": "GGA_X_PBE,GGA_C_PBE",
    "B3LYP": "HYB_GGA_XC_B3LYP",
    "r2SCAN": "MGGA_X_R2SCAN,MGGA_C_R2SCAN",
    "HYB_GGA_XC_CAM_B3LYP": "HYB_GGA_XC_CAM_B3LYP",
    "HYB_GGA_XC_WB97X": "HYB_GGA_XC_WB97X",
}
RSH = ("HYB_GGA_XC_CAM_B3LYP", "HYB_GGA_XC_WB97X")


def auxmol_of(mol, aux_name):
    return gto.M(
        atom=mol.atom,
        basis=ferric_basis(aux_name, [mol.atom_symbol(i) for i in range(mol.natm)]),
        unit=mol.unit,
        cart=False,
        verbose=0,
    )


def ri_factor_op(mol, aux_name, omega):
    """B[P, mn] with (mn|ls) ~= sum_P B[P,mn] B[P,ls] for the Coulomb operator
    (omega = 0: Cholesky, ferric's `cholesky_inverse_sqrt`) or erf(omega r)/r
    (eigendecomposition of the erf metric, eigenvalues below LINDEP dropped,
    M = U diag(lambda^-1/2) U^T: ferric's `eigh_inverse_sqrt`)."""
    auxmol = auxmol_of(mol, aux_name)
    nao = mol.nao
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        int3c = incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        v2c = auxmol.intor("int2c2e")
    rhs = int3c.reshape(nao * nao, -1).T
    if omega == 0:
        low = scipy.linalg.cholesky(v2c, lower=True)
        b = scipy.linalg.solve_triangular(low, rhs, lower=True)
    else:
        lam, u = np.linalg.eigh(v2c)
        s = np.where(lam < LINDEP, 0.0, 1.0 / np.sqrt(np.abs(lam)))
        m = (u * s) @ u.T
        b = m @ rhs
    return b.reshape(-1, nao, nao)


class RiAo2mo:
    """Drop-in for `ao2mo` inside pyscf.tdscf.{rhf,uhf}: the Coulomb or the
    erf factorisation, by the `mol.omega` that get_ab has set."""

    def __init__(self, b_coul, b_erf):
        self.b_coul = b_coul
        self.b_erf = b_erf  # {omega: B}

    def general(self, mol, mos, compact=False, **_kw):
        b_ao = self.b_coul if mol.omega == 0 else self.b_erf[mol.omega]
        c1, c2, c3, c4 = mos
        b12 = np.einsum("Pmn,mi,nj->Pij", b_ao, c1, c2, optimize=True)
        b34 = np.einsum("Pmn,mk,nl->Pkl", b_ao, c3, c4, optimize=True)
        eri = np.einsum("Pij,Pkl->ijkl", b12, b34, optimize=True)
        return eri.reshape(c1.shape[1] * c2.shape[1], c3.shape[1] * c4.shape[1])


def small_grid_blocks(mf):
    """Pin the grid block size of `mf`'s NumInt to GRID_BLOCK (only the
    summation order of the kernel contraction changes)."""
    ni = getattr(mf, "_numint", None)
    if ni is not None:
        real = type(ni).block_loop

        def block_loop(
            mol,
            grids,
            nao=None,
            deriv=0,
            max_memory=2000,
            non0tab=None,
            blksize=None,
            buf=None,
        ):
            return real(
                ni, mol, grids, nao, deriv, max_memory, non0tab, GRID_BLOCK, buf
            )

        ni.block_loop = block_loop
    return mf


def with_ri(module, ri, fn):
    real = module.ao2mo
    module.ao2mo = ri
    try:
        return fn()
    finally:
        module.ao2mo = real


def match_rsh_scf_fit(mf, omega):
    """Make PySCF's full-range K the sum of its erfc- and erf-fitted K (see the
    module docstring), so hyb K + (alpha - hyb) K_lr = c_sr K_sr + c_lr K_lr."""
    real_get_jk = mf.get_jk

    def get_jk(
        mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega_k=None, **kw
    ):
        if "omega" in kw:
            omega_k = kw.pop("omega")
        if omega_k not in (None, 0) or not with_k:
            return real_get_jk(mol, dm, hermi, with_j, with_k, omega_k)
        vj = real_get_jk(mol, dm, hermi, with_j, False)[0] if with_j else None
        vk_sr = real_get_jk(mol, dm, hermi, False, True, -omega)[1]
        vk_lr = real_get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk_sr + vk_lr

    mf.get_jk = get_jk
    return mf


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
    omega = 0.0
    if xc is not None:
        omega, _alpha, _hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        if omega != 0:
            match_rsh_scf_fit(mf, omega)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"SCF not converged ({xc})"
    return mf, omega


def dense_casida_or_unstable(a, b):
    """Casida roots, or None and the lowest Omega^2 when A+B is indefinite."""
    n = a.shape[0] * a.shape[1]
    am, bm = a.reshape(n, n), b.reshape(n, n)
    apb, amb = am + bm, am - bm
    apb, amb = 0.5 * (apb + apb.T), 0.5 * (amb + amb.T)
    lam, u = np.linalg.eigh(amb)
    assert lam.min() > 0, f"(A-B) not positive definite: {lam.min():e}"
    amb_sqrt = (u * np.sqrt(lam)) @ u.T
    w2, z = np.linalg.eigh(amb_sqrt @ apb @ amb_sqrt)
    if w2.min() <= 0:
        return None, None, float(w2.min())
    w = np.sqrt(w2)
    return w, (amb_sqrt @ z) / np.sqrt(w)[None, :], float(w2.min())


def lr_part(mf, ri, omega, k_fac, nocc):
    """The long-range exchange blocks get_ab adds: -k_fac (ij|ab)_w (A) and
    -k_fac (ib|aj)_w (B), from the same RI ao2mo, for the negative control."""
    occ = mf.mo_occ > 0
    orbo, orbv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    mo = np.hstack((orbo, orbv))
    mol = mf.mol
    nmo = mo.shape[1]
    with mol.with_range_coulomb(omega):
        eri = ri.general(mol, (orbo, mo, mo, mo)).reshape(nocc, nmo, nmo, nmo)
    da = -np.einsum("ijba->iajb", eri[:nocc, :nocc, nocc:, nocc:]) * k_fac
    db = -np.einsum("jaib->iajb", eri[:nocc, nocc:, :nocc, nocc:]) * k_fac
    return da, db


def energies_ev(w):
    return [float(v * HARTREE2EV) for v in w[:NSTATES]]


def block(mf, a, b, nocc, nvir, triplet):
    """TDA and Casida records for one (A, B)."""
    w_tda, x_tda = dense_tda(a)
    tda = states(mf, w_tda, x_tda, nocc, nvir)
    w_cas, xpy, w2min = dense_casida_or_unstable(a, b)
    if w_cas is None:
        cas = {"unstable": True, "omega2_min": w2min}
    else:
        cas = {"unstable": False, **states(mf, w_cas, xpy, nocc, nvir)}
    if triplet:
        tda["osc"] = [0.0] * len(tda["osc"])
        if not cas["unstable"]:
            cas["osc"] = [0.0] * len(cas["osc"])
    return tda, cas


def run_case(mol_key, only=None):
    atoms = read_xyz(MOLDIR / MOLECULES[mol_key])
    mol = make_mol(atoms, BASIS)
    # PySCF sizes its grid blocks (and the get_ab kernel contraction) from
    # max_memory; the 4000 MB default overshoots it ~2.5x on benzene.
    mol.max_memory = MAX_MEMORY_MB
    b_coul = ri_factor_op(mol, RESPONSE_AUX, 0.0)
    rec = {
        "molecule": mol_key,
        "xyz": f"testdata/molecules/{MOLECULES[mol_key]}",
        "basis": BASIS,
        "response_aux": RESPONSE_AUX,
        "scf_jk_aux": SCF_JK_AUX,
        "grid": {"atom_grid": list(MAIN_GRID), "prune": None, "radii_adjust": "becke"},
        "nstates": NSTATES,
        "pyscf_version": PYSCF_VERSION,
        "generator": "scripts/validation/gen_tddft_rsh_refs.py",
        "note": "dense TDA / Casida; response (pq|rs) by ferric's Coulomb-metric RI and "
        "(pq|rs)_omega by ferric's erf-metric RI over response_aux; triplets from "
        "tdscf.uhf.get_ab spin blocks (A_aa - A_ab); flat (75,110) grid; RSH SCF fits "
        "c_sr K[erfc] + c_lr K[erf] as ferric's",
        "cases": {},
    }
    for name, xc in FUNCTIONALS.items():
        if only is not None and name not in only:
            continue
        print(f"  {mol_key}/{BASIS}/{name}", file=sys.stderr)
        mf, omega = run_scf(mol, xc)
        b_erf = {omega: ri_factor_op(mol, RESPONSE_AUX, omega)} if omega else {}
        ri = RiAo2mo(b_coul, b_erf)
        nocc = int((mf.mo_occ > 0).sum())
        nvir = mf.mo_occ.size - nocc
        if xc is None:
            c_sr = c_lr = 1.0
        else:
            omega, c_lr, c_sr = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            if omega == 0:
                c_lr = c_sr

        small_grid_blocks(mf)
        a_s, b_s = with_ri(tdscf.rhf, ri, lambda: tdscf.rhf.get_ab(mf))
        umf = small_grid_blocks(mf.to_uhf() if xc is None else mf.to_uks())
        (a_aa, a_ab, _), (b_aa, b_ab, _) = with_ri(
            tdscf.uhf, ri, lambda: tdscf.uhf.get_ab(umf)
        )
        dev_s = max(np.abs(a_aa + a_ab - a_s).max(), np.abs(b_aa + b_ab - b_s).max())
        assert dev_s < 1e-10, f"UKS singlet blocks differ from RHF get_ab by {dev_s:e}"
        a_t, b_t = a_aa - a_ab, b_aa - b_ab

        case = {
            "xc_pyscf": xc,
            "c_sr": float(c_sr),
            "c_lr": float(c_lr),
            "omega": float(omega),
            "e_scf": float(mf.e_tot),
            "nocc": nocc,
            "nvir": nvir,
            "singlet_from_uks_max_dev": float(dev_s),
        }
        for spin, (a, b) in (("singlet", (a_s, b_s)), ("triplet", (a_t, b_t))):
            tda, cas = block(mf, a, b, nocc, nvir, spin == "triplet")
            case[spin] = {"tda": tda, "casida": cas}
            if omega != 0:
                da, db = lr_part(mf, ri, omega, c_lr - c_sr, nocc)
                w, _ = dense_tda(a - da)
                case[spin]["tda_no_lr"] = {"omega_ev": energies_ev(w)}
            if spin == "triplet" and xc is not None:
                # The kernel with the singlet sign, f_aa + f_ab in place of
                # f_aa - f_ab: A_T + 2 f_ab, where A_ab = (ia|jb) + f_ab.
                f_ab = a_ab - coulomb_block(mf, ri, nocc)
                w, _ = dense_tda(a_t + 2 * f_ab)
                case[spin]["tda_singlet_kernel"] = {"omega_ev": energies_ev(w)}
        rec["cases"][name] = case
        t = case["triplet"]
        s = case["singlet"]
        print(
            f"    S TDA {[round(v, 4) for v in s['tda']['omega_ev'][:5]]}\n"
            f"    T TDA {[round(v, 4) for v in t['tda']['omega_ev'][:5]]}\n"
            f"    T CAS {'unstable' if t['casida']['unstable'] else [round(v, 4) for v in t['casida']['omega_ev'][:5]]}",
            file=sys.stderr,
        )
    return rec


def coulomb_block(mf, ri, nocc):
    """(ia|jb) over the RI Coulomb tensors, (nocc, nvir, nocc, nvir)."""
    occ = mf.mo_occ > 0
    orbo, orbv = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    eri = ri.general(mf.mol, (orbo, orbv, orbo, orbv))
    nvir = orbv.shape[1]
    return eri.reshape(nocc, nvir, nocc, nvir)


def main(argv):
    mols = [argv[1]] if len(argv) > 1 else list(MOLECULES)
    only = argv[2].split(",") if len(argv) > 2 else None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for m in mols:
        rec = run_case(m, only)
        path = OUTDIR / f"{m}_{BASIS}.json"
        if only is not None and path.exists():
            old = json.loads(path.read_text())
            old["cases"].update(rec["cases"])
            rec = old
        path.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
