#!/usr/bin/env python3
"""PySCF references for ferric's spin-conserving TDA / TDDFT on UHF and UKS
references, with <S^2> of each root.

Output: testdata/reference/validation/tddft_open_shell/<mol>_6-31g.json, read
by crates/ferric-tddft/tests/validation_tddft_open_shell.rs.

WHAT IS MATCHED TO FERRIC, AND HOW
----------------------------------
As gen_tddft_rsh_refs.py: ferric's bundled basis JSON, the RI-JK
def2-universal-jkfit reference SCF (for a range-separated functional the
exchange fitted as c_sr K[erfc] + c_lr K[erf]), identical libxc identifiers,
the flat (75, 110) Becke grid on both sides, and the response (pq|rs) and
(pq|rs)_omega by ferric's Coulomb- and erf-metric RI over the response aux,
substituted into tdscf.uhf.get_ab. The SCF is UHF / UKS (mol.spin = 2S).

The full two-spin A and B are assembled from get_ab's blocks
(A_aa, A_ab, A_bb; A_ba = A_ab^T) and solved dense: TDA eigh(A), Casida
eigh((A-B)^1/2 (A+B) (A-B)^1/2).

<S^2> OF EACH ROOT, BY TWO CONSTRUCTIONS
---------------------------------------
* `s2`: the closed formula of crates/ferric-tddft/theory.md section 11, in
  numpy (`s2_formula`), on the unit-normalized X (TDA) or the renormalized
  X = ((X+Y) + (X-Y))/2 (Casida).
* For OH and NH2 (FCI space small enough) the same X is written into a full
  determinant vector over the UHF orbitals (sign of a+_a a_i on the reference
  string from pyscf.fci.cistring) and <S^2> taken by
  pyscf.fci.spin_op.spin_square with the alpha and beta MO coefficients and
  the AO overlap: an independent construction (2-RDMs of the expanded
  vector). The generator asserts the two agree to 1e-10 on every root and
  records the largest difference.

Negative control stored per KS case: the TDA roots with the alpha-beta f_xc
block dropped (`tda_no_fxc_ab`: A_ab reduced to its Coulomb part).

Memory: mol.max_memory is capped and the get_ab grid block is pinned (as in
gen_tddft_rsh_refs.py); the molecules are small.

Usage (from the repo root):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_open_shell_refs.py
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_open_shell_refs.py oh PBE,B3LYP
"""

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import __version__ as PYSCF_VERSION
from pyscf import dft, gto, scf, tdscf
from pyscf.fci import cistring, spin_op

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tddft_refs import (  # noqa: E402
    HARTREE2EV,
    MOLDIR,
    ROOT,
    SCF_JK_AUX,
    ferric_basis,
    read_xyz,
)
from gen_tddft_rsh_refs import (  # noqa: E402
    MAIN_GRID,
    RESPONSE_AUX,
    RiAo2mo,
    dense_casida_or_unstable,
    match_rsh_scf_fit,
    ri_factor_op,
    small_grid_blocks,
    with_ri,
)

OUTDIR = ROOT / "testdata" / "reference" / "validation" / "tddft_open_shell"
NSTATES = 12
BASIS = "6-31g"
MAX_MEMORY_MB = 1500
FCI_CHECK_MAX_DETS = 2_000_000
S2_AGREE = 1e-10

# key -> (xyz, multiplicity, Casida compared, functionals). OH's 2Pi ground
# state is doubly degenerate: the beta pi -> pi rotation between its two
# components is a zero mode of A+B and A-B (a continuous symmetry of the UHF
# energy), so the Casida Omega^2 of that root sits at the noise floor with
# either sign and Omega = sqrt(Omega^2) is not comparable: OH is compared by
# TDA only. Its UKS (every functional here) does not converge in PySCF, DIIS
# or second-order, from the default or the UHF density: the occupied and the
# empty beta pi component are degenerate in KS. OH is therefore HF only.
MOLECULES = {
    "oh": ("oh.xyz", 2, False, ("HF",)),
    "nh2": ("nh2.xyz", 2, True, None),
    "o2": ("o2.xyz", 3, True, None),
}

FUNCTIONALS = {
    "HF": None,
    "LDA": "LDA_X,LDA_C_VWN",
    "PBE": "GGA_X_PBE,GGA_C_PBE",
    "B3LYP": "HYB_GGA_XC_B3LYP",
    "r2SCAN": "MGGA_X_R2SCAN,MGGA_C_R2SCAN",
    "TPSSh": "HYB_MGGA_XC_TPSSH",
    "HYB_GGA_XC_CAM_B3LYP": "HYB_GGA_XC_CAM_B3LYP",
}


def make_mol(atoms, mult):
    symbols = [a[0] for a in atoms]
    mol = gto.M(
        atom=[[a[0], a[1:]] for a in atoms],
        basis=ferric_basis(BASIS, symbols),
        unit="Angstrom",
        charge=0,
        spin=mult - 1,
        cart=False,
        verbose=0,
    )
    mol.max_memory = MAX_MEMORY_MB
    return mol


def exact_j(mf):
    """ferric's UHF/UKS builds J from exact four-index integrals when the
    functional is range-separated (its `df_j_aux` applies at omega = 0 only),
    so the PySCF reference does too; K stays fitted as ferric fits it."""
    fitted_get_jk = mf.get_jk

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, *args, **kw):
        mol = mf.mol if mol is None else mol
        dm = mf.make_rdm1() if dm is None else dm
        vk = (
            fitted_get_jk(mol, dm, hermi, False, with_k, *args, **kw)[1]
            if with_k
            else None
        )
        if (
            with_j
            and kw.get("omega") in (None, 0)
            and (not args or args[0] in (None, 0))
        ):
            vj = scf.hf.get_jk(mol, dm, hermi, with_j=True, with_k=False)[0]
        elif with_j:
            vj = fitted_get_jk(mol, dm, hermi, True, False, *args, **kw)[0]
        else:
            vj = None
        return vj, vk

    mf.get_jk = get_jk
    return mf


def run_uscf(mol, xc):
    if xc is None:
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
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
            exact_j(mf)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 300
    mf.kernel()
    assert mf.converged, f"UHF/UKS not converged ({xc})"
    return mf, omega


def full_ab(a_blocks, b_blocks):
    (a_aa, a_ab, a_bb), (b_aa, b_ab, b_bb) = a_blocks, b_blocks
    da = a_aa.shape[0] * a_aa.shape[1]
    db = a_bb.shape[0] * a_bb.shape[1]

    def assemble(aa, ab, bb):
        m = np.zeros((da + db, da + db))
        m[:da, :da] = aa.reshape(da, da)
        m[:da, da:] = ab.reshape(da, db)
        m[da:, :da] = ab.reshape(da, db).T
        m[da:, da:] = bb.reshape(db, db)
        return m

    return assemble(a_aa, a_ab, a_bb), assemble(b_aa, b_ab, b_bb), da


def spin_overlap(mf):
    s = mf.mol.intor("int1e_ovlp")
    ca, cb = mf.mo_coeff
    return ca.T @ s @ cb


def s2_formula(o, na, nb, x):
    """<S^2> of sum X_ia,s |Phi_i^a s> (theory.md section 11)."""
    nva, nvb = o.shape[0] - na, o.shape[1] - nb
    x = x / np.linalg.norm(x)
    xa = x[: na * nva].reshape(na, nva)
    xb = x[na * nva :].reshape(nb, nvb)
    m = 0.5 * (na - nb)
    p = o.T @ o
    q = o[:, :nb] @ o[:, :nb].T
    r = o[:na, :].T @ o[:na, :]
    wa, wb = (xa * xa).sum(), (xb * xb).sum()
    occ_w = (o[:, :nb] ** 2).sum()

    def ph(xs, mm, nocc):
        return np.trace(xs @ mm[nocc:, nocc:] @ xs.T) - np.trace(
            xs.T @ mm[:nocc, :nocc] @ xs
        )

    one = (wa + wb) * occ_w + ph(xb, p, nb)
    two_a = wa * np.trace(q[:na, :na]) + ph(xa, q, na)
    two_b = wb * np.trace(r[:nb, :nb]) + ph(xb, r, nb)
    cross = 2.0 * np.trace(xa.T @ o[:na, :nb] @ xb @ o[na:, nb:].T)
    return m * m + m + one - two_a - two_b - cross


def s2_reference_formula(o, na, nb):
    m = 0.5 * (na - nb)
    return m * m + m + (o[:, :nb] ** 2).sum() - (o[:na, :nb] ** 2).sum()


def s2_fci(mf, na, nb, x):
    """<S^2> of the same expansion, by pyscf.fci.spin_op on a determinant vector."""
    norb = mf.mo_coeff[0].shape[1]
    nva, nvb = norb - na, norb - nb
    x = x / np.linalg.norm(x)
    xa = x[: na * nva].reshape(na, nva)
    xb = x[na * nva :].reshape(nb, nvb)
    sa = cistring.make_strings(range(norb), na)
    sb = cistring.make_strings(range(norb), nb)
    ref_a, ref_b = (1 << na) - 1, (1 << nb) - 1
    ia_ref = cistring.str2addr(norb, na, ref_a)
    ib_ref = cistring.str2addr(norb, nb, ref_b)
    civec = np.zeros((len(sa), len(sb)))
    for i in range(na):
        for a in range(nva):
            p = na + a
            new = (ref_a ^ (1 << i)) | (1 << p)
            sign = cistring.cre_des_sign(p, i, ref_a)
            civec[cistring.str2addr(norb, na, new), ib_ref] += sign * xa[i, a]
    for i in range(nb):
        for a in range(nvb):
            p = nb + a
            new = (ref_b ^ (1 << i)) | (1 << p)
            sign = cistring.cre_des_sign(p, i, ref_b)
            civec[ia_ref, cistring.str2addr(norb, nb, new)] += sign * xb[i, a]
    s2, _ = spin_op.spin_square(
        civec, norb, (na, nb), mo_coeff=mf.mo_coeff, ovlp=mf.mol.intor("int1e_ovlp")
    )
    return float(s2)


def fci_dets(norb, na, nb):
    return cistring.num_strings(norb, na) * cistring.num_strings(norb, nb)


def x_of(w, amps, apb, casida):
    """Unit-free X per root: TDA X itself; Casida ((X+Y) + (X-Y))/2."""
    if not casida:
        return amps
    xmy = (apb @ amps) / w[None, :]
    return 0.5 * (amps + xmy)


def states(mf, w, amps, xs, na, nb, o, fci_ok):
    ca, cb = mf.mo_coeff
    with mf.mol.with_common_orig((0, 0, 0)):
        mu = mf.mol.intor("int1e_r", comp=3)
    da = na * (ca.shape[1] - na)
    dip_a = np.einsum("pi,xpq,qa->xia", ca[:, :na], mu, ca[:, na:]).reshape(3, -1)
    dip_b = np.einsum("pi,xpq,qa->xia", cb[:, :nb], mu, cb[:, nb:]).reshape(3, -1)
    out = {"omega_ev": [], "omega_ha": [], "osc": [], "s2": []}
    dev = 0.0
    for k in range(min(NSTATES, len(w))):
        t = dip_a @ amps[:da, k] + dip_b @ amps[da:, k]
        out["omega_ha"].append(float(w[k]))
        out["omega_ev"].append(float(w[k] * HARTREE2EV))
        out["osc"].append(float((2.0 / 3.0) * w[k] * (t @ t)))
        s2 = float(s2_formula(o, na, nb, xs[:, k]))
        if fci_ok:
            s2f = s2_fci(mf, na, nb, xs[:, k])
            dev = max(dev, abs(s2 - s2f))
            assert abs(s2 - s2f) < S2_AGREE, f"<S^2> formula {s2} != FCI {s2f}"
        out["s2"].append(s2)
    out["s2_fci_max_dev"] = float(dev) if fci_ok else None
    return out


def solve_both(mf, a, b, na, nb, o, fci_ok, casida):
    w, x = np.linalg.eigh(0.5 * (a + a.T))
    tda = states(mf, w, x, x, na, nb, o, fci_ok)
    if not casida:
        return tda, {
            "skipped": "zero mode of a degenerate ground state",
            "unstable": None,
        }
    # The helper reshapes 4-index (o, v, o, v) blocks; a 2-D matrix goes in as
    # (n, 1, n, 1).
    n = a.shape[0]
    w_c, xpy, w2min = dense_casida_or_unstable(
        a.reshape(n, 1, n, 1), b.reshape(n, 1, n, 1)
    )
    if w_c is None:
        cas = {"unstable": True, "omega2_min": w2min}
    else:
        xs = x_of(w_c, xpy, a + b, True)
        cas = {"unstable": False, **states(mf, w_c, xpy, xs, na, nb, o, fci_ok)}
    return tda, cas


def coulomb_cross(mf, ri, na, nb):
    """(ia alpha|jb beta) over the RI Coulomb tensors, (da, db)."""
    ca, cb = mf.mo_coeff
    eri = ri.general(mf.mol, (ca[:, :na], ca[:, na:], cb[:, :nb], cb[:, nb:]))
    return eri


def run_case(mol_key, only=None):
    fname, mult, casida, subset = MOLECULES[mol_key]
    if subset is not None:
        only = [f for f in (only or subset) if f in subset]
    atoms = read_xyz(MOLDIR / fname)
    mol = make_mol(atoms, mult)
    b_coul = ri_factor_op(mol, RESPONSE_AUX, 0.0)
    na, nb = mol.nelec
    norb = mol.nao
    fci_ok = fci_dets(norb, na, nb) <= FCI_CHECK_MAX_DETS
    rec = {
        "molecule": mol_key,
        "xyz": f"testdata/molecules/{fname}",
        "multiplicity": mult,
        "casida_compared": casida,
        "basis": BASIS,
        "response_aux": RESPONSE_AUX,
        "scf_jk_aux": SCF_JK_AUX,
        "grid": {"atom_grid": list(MAIN_GRID), "prune": None, "radii_adjust": "becke"},
        "nstates": NSTATES,
        "pyscf_version": PYSCF_VERSION,
        "generator": "scripts/validation/gen_tddft_open_shell_refs.py",
        "note": "dense TDA / Casida on the two-spin A, B of tdscf.uhf.get_ab with the "
        "response (pq|rs) by ferric's Coulomb-metric RI and (pq|rs)_omega by its "
        "erf-metric RI; flat (75,110) grid; s2 by the theory.md section 11 formula, "
        "checked against pyscf.fci.spin_op on the expanded determinant vector when "
        "s2_fci_max_dev is not null",
        "cases": {},
    }
    for name, xc in FUNCTIONALS.items():
        if only is not None and name not in only:
            continue
        print(f"  {mol_key}/{BASIS}/{name}", file=sys.stderr)
        mf, omega = run_uscf(mol, xc)
        b_erf = {omega: ri_factor_op(mol, RESPONSE_AUX, omega)} if omega else {}
        ri = RiAo2mo(b_coul, b_erf)
        small_grid_blocks(mf)
        a_blk, b_blk = with_ri(tdscf.uhf, ri, lambda: tdscf.uhf.get_ab(mf))
        a, b, da = full_ab(a_blk, b_blk)
        o = spin_overlap(mf)
        tda, cas = solve_both(mf, a, b, na, nb, o, fci_ok, casida)
        if xc is None:
            c_sr = c_lr = 1.0
        else:
            omega, c_lr, c_sr = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            if omega == 0:
                c_lr = c_sr
        s2_ref = float(s2_reference_formula(o, na, nb))
        s2_pyscf = float(mf.spin_square()[0])
        assert abs(s2_ref - s2_pyscf) < 1e-10, (s2_ref, s2_pyscf)
        case = {
            "xc_pyscf": xc,
            "c_sr": float(c_sr),
            "c_lr": float(c_lr),
            "omega": float(omega),
            "e_scf": float(mf.e_tot),
            "s2_reference": s2_ref,
            "n_alpha": int(na),
            "n_beta": int(nb),
            "dim_alpha": int(da),
            "dim": int(a.shape[0]),
            "tda": tda,
            "casida": cas,
        }
        if xc is not None:
            j_ab = coulomb_cross(mf, ri, na, nb)
            a_no = a.copy()
            a_no[:da, da:] = j_ab
            a_no[da:, :da] = j_ab.T
            w, _ = np.linalg.eigh(0.5 * (a_no + a_no.T))
            case["tda_no_fxc_ab"] = {
                "omega_ev": [float(v * HARTREE2EV) for v in w[:NSTATES]]
            }
        rec["cases"][name] = case
        print(
            f"    E {mf.e_tot:.8f} <S2>0 {s2_ref:.5f}\n"
            f"    TDA {[round(v, 4) for v in tda['omega_ev'][:6]]}\n"
            f"    S2  {[round(v, 3) for v in tda['s2'][:6]]} (fci dev {tda['s2_fci_max_dev']})\n"
            f"    CAS {cas.get('skipped') or ('unstable' if cas['unstable'] else [round(v, 4) for v in cas['omega_ev'][:6]])}",
            file=sys.stderr,
        )
    return rec


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
