"""PySCF-code references for the VALIDATION.md "RS-MP2-RPA at production omega" row.

Consumer: crates/ferric-rpa/tests/validation_rs_mp2_rpa_pyscf.rs.
Output:   testdata/reference/validation/rs_mp2_rpa_pyscf/<system>_<basis>.json

WHAT IS BUILT, AND WITH WHICH OPERATOR
--------------------------------------
ferric's `rs_mp2_lr_rpa` (crates/ferric-rpa/src/rs_mp2_rpa.rs) at the production
omega = 0.420 A^-1 = 0.222254 Bohr^-1 (`RsMp2RpaConfig::default().omega`), with
SC[op] = {E_OS, E_SS, E_MP2} the RI-MP2 spin components and E_dRPA[op] the
closed-shell direct RPA, op in the 3-centre integrals AND the metric:

  field               operator(s)                     definition
  e_mp2_full          Coulomb 1/r                     SC[C].E_MP2
  e_sr_mp2            erfc(w r)/r  (short range)      SC[erfc].E_MP2
  e_lr_mp2            erf(w r)/r   (long range)       SC[erf].E_MP2
  e_dmp2_lr           erf                             2 SC[erf].E_OS
  B e_drpa_lr         erf                             E_dRPA[erf]
  B e_corr_naive      erfc + erf                      SC[erfc].E_MP2 + E_dRPA[erf]
  B e_corr            Coulomb + erf                   SC[C].E_MP2 + E_dRPA[erf] - 2 SC[erf].E_OS
  T e_delta_drpa_full Coulomb                         E_dRPA[C] - 2 SC[C].E_OS
  T e_delta_drpa_sr   erfc                            E_dRPA[erfc] - 2 SC[erfc].E_OS
  T e_corr            Coulomb + erfc                  SC[C].E_MP2 + dfull - dsr

The RHF is the full-Coulomb RHF; the operator enters only the correlation.
No nuclear-attraction integral is attenuated anywhere (so libint's erf_nuclear
defect does not apply here).

THE REFERENCE: PySCF's own code on the attenuated fit
-----------------------------------------------------
For op in {Coulomb, erf, erfc} one `df.DF(mol, auxbasis)` is built inside
`mol.with_range_coulomb(w_pyscf)`; `make_auxmol` copies mol's `_env`, so the
auxiliary Mole carries the same w (asserted). PySCF's sign convention
(gto/mole.py set_range_coulomb): w > 0 is erf(w r)/r, w < 0 is erfc(|w| r)/r.
It is CHECKED here on a two-electron integral before anything else: one s
primitive (exponent a) on one centre has (ss|ss) = 2 sqrt(rho/pi) with
rho = a for 1/r and rho = a w^2/(a + w^2) for erf(w r)/r; PySCF `int2e` must give
the erf value at +w and Coulomb minus it at -w (`sign_convention`).

The fitted tensor is then handed to
  * `mp.dfmp2.DFMP2` (pt.with_df = that DF)  -> E_OS, E_SS, E_MP2;
  * `gw.rpa.RPA`     (rpa.with_df = that DF) -> E_dRPA on nw = 20 (ferric's
    default `QuadratureConfig`) and nw = 40, x0 = 0.5.
`RPA.get_e_hf` would build an EXX energy from the attenuated fit, which means
nothing; e_hf is preset to the RHF energy so that step is skipped (it does not
enter e_corr). Each field above is assembled from these PySCF numbers.

Orbitals: the PySCF RHF orbitals are written in ferric's AO order
(gen_properties.ferric_ao_permutation) so the Rust test can run ferric's
correlation code on THE SAME orbitals and orbital energies. That removes the
SCF from the comparison; what is left is the integrals, the metric
factorization and the frequency quadrature.

THE ONE KNOWN DIFFERENCE: THE RI METRIC FACTORIZATION
-----------------------------------------------------
ferric (`ferric_mp2::rimp2::metric_inverse_sqrt`): erf -> symmetric eigh,
eigenvalues below 1e-10 dropped; Coulomb and erfc -> Cholesky.
PySCF (`df.incore.cholesky_eri`): Cholesky; if that raises, eigh with
eigenvalues at or below lindep = 1e-7 dropped (`_eig_decompose`). The erf
metric has eigenvalues down to ~-1e-14, so PySCF falls back to eigh and keeps
FEWER modes than ferric. The decomposition PySCF used is recorded per kernel.

The size of that difference is computed here, not guessed: numpy evaluates the
same MP2 spin components and dRPA from two tensors on the SAME PySCF integrals,
B_pyscf (PySCF's own `_cderi`, unpacked) and B_ferric (the same int3c2e/int2c2e
with ferric's factorization, `gen_attenuated_rpa.metric_inv_sqrt`):

  metric_shift[field]   = field(B_ferric) - field(B_pyscf)        (numpy, same code)
  integral_shift[field] = field(B_ferric on ferric's integrals) - field(B_ferric)
  code_residual[field]  = |field(B_pyscf, numpy) - field(PySCF code)|

`integral_shift` takes ferric's own 3-centre and 2-centre integrals
(`ferric.AoIntegrals.eri3/eri2`, the libint2 engines at precision 1e-14 that
rs_mp2_lr_rpa runs) through the same numpy code. The erf fit's metric has
eigenvalues near ferric's 1e-10 cut, and V^{-1} amplifies a 1e-14 integral
difference by up to 1/lambda, so on the erf terms this shift is larger than
rounding. The largest elementwise integral difference is recorded per kernel.
So ferric's value is predicted to be PySCF's plus metric_shift plus
integral_shift. The test's bar
is built from these two numbers, `cut_flip` (the field change if the erf
metric's eigenvalue nearest ferric's 1e-10 cut lands on the other side of it,
counted only when that eigenvalue is within the eigensolver's backward-error
bound n eps lambda_max of the cut), `eigenvalue_error` (twice the summed field
change when each kept small metric eigenvalue alone moves by that bound,
EIG_WINDOW below), the quadrature node difference, a log-det rounding term the
test computes, and the floating-point floor `fp_floor` (the change of each
field when every integral is perturbed by its evaluation accuracy,
INT_REL_ERR below); see the test's header.

Run (light; under a minute per system):
    OPENBLAS_NUM_THREADS=1 box-lease exec -- \\
        uv run --no-sync python scripts/validation/gen_rs_mp2_rpa_pyscf.py [system_basis ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_attenuated_rpa as gar  # noqa: E402
import gen_properties  # noqa: E402
import gen_rs_mp2_rpa as grs  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "rs_mp2_rpa_pyscf"
ROW_NAME = "RS-MP2-RPA at production omega"
OMEGA = gar.OMEGA_DEFAULT_BOHR
X0 = 0.5
# ferric's default QuadratureConfig (20 points) and the 40-point grid the
# other RPA rows use.
NWS = (20, 40)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
FERRIC_LINDEP = 1e-10
PYSCF_LINDEP = 1e-7
# numpy on PySCF's own tensor must reproduce PySCF's code to this (both are
# one dot product / log-det away from each other); measured ~1e-15.
CODE_RESIDUAL_MAX = 1e-12
SIGN_CHECK_MAX = 1e-13
SIGN_EXPONENT = 1.3
# Floating-point floor: every 3-centre and 2-centre integral is perturbed
# elementwise by a relative +-INT_REL_ERR (random sign, symmetric for the
# metric) and the fields are recomputed with ferric's factorization. 64 eps is
# the relative accuracy of a contracted Gaussian integral from a few tens of
# rounding steps of a Rys / Obara-Saika recursion (libcint, libint2); the
# perturbed metric also carries any eigensolver backward error of that size
# into the 1e-10 cut. Two seeds; the larger change is kept.
INT_REL_ERR = 64 * 2.220446049250313e-16
FLOOR_SEEDS = (1, 2)
# Metric eigenvalue error: a symmetric eigensolver (or Cholesky) returns each
# eigenvalue of V to within n eps lambda_max, which is a large relative error
# on the erf metric's smallest kept modes (lambda ~ 1e-10). Each kept mode
# with err / lambda >= 1 / EIG_WINDOW is moved by +err alone and the field
# change recorded; the modes above the window change a field by less than
# 1e-9 of its size.
EIG_WINDOW = 1e9

CASES = {  # key -> (system, basis, aux)
    "h2o_cc-pvdz": ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    "h2o_aug-cc-pvdz": ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "c2h4_cc-pvdz": ("c2h4", "cc-pvdz", "cc-pvdz-ri"),
    "c2h4_aug-cc-pvdz": ("c2h4", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    "water_dimer_cc-pvdz": ("water_dimer", "cc-pvdz", "cc-pvdz-ri"),
    "water_dimer_aug-cc-pvdz": ("water_dimer", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
}
KINDS = ("coulomb", "erf", "erfc")


def sign_convention(omega):
    """PySCF int2e under with_range_coulomb(+-omega) vs the closed form for one
    s primitive on one centre. Returns the record; raises on a mismatch."""
    import numpy as np
    from pyscf import gto

    a = SIGN_EXPONENT
    m = gto.M(atom="H 0 0 0", basis={"H": [[0, [a, 1.0]]]}, spin=1, verbose=0)
    coul = 2.0 * np.sqrt(a / np.pi)
    erf = 2.0 * np.sqrt(a * omega**2 / (a + omega**2) / np.pi)
    rec = {"exponent": a, "omega": omega}
    for w, name, want in (
        (0.0, "coulomb", coul),
        (omega, "erf", erf),
        (-omega, "erfc", coul - erf),
    ):
        with m.with_range_coulomb(w):
            got = float(m.intor("int2e")[0, 0, 0, 0])
        rec[name] = {
            "pyscf_omega": w,
            "pyscf": got,
            "closed_form": want,
            "abs_diff": abs(got - want),
        }
        if abs(got - want) > SIGN_CHECK_MAX:
            raise RuntimeError(
                f"with_range_coulomb({w}) is not {name}: {got} vs {want}"
            )
    return rec


def unpack_cderi(dfo, nao):
    """PySCF's (naux_kept, nao*(nao+1)/2) _cderi -> (nao, nao, naux_kept)."""
    import numpy as np
    from pyscf import lib

    cd = np.asarray(dfo._cderi)
    return np.stack([lib.unpack_tril(row) for row in cd], axis=2)


def decomposition_pyscf_used(dfo):
    """'cd' if Cholesky of the auxmol's own int2c2e succeeds, else 'eig'."""
    import scipy.linalg

    j2c = dfo.auxmol.intor("int2c2e", hermi=1)
    try:
        scipy.linalg.cholesky(j2c, lower=True)
        return "cd"
    except scipy.linalg.LinAlgError:
        return "eig"


def numpy_energies(B, mf, d):
    sc = grs.spin_components(B, mf)
    for nw in NWS:
        sc[f"e_drpa_nw{nw}"] = gar.numpy_rpa(B, d, nw=nw)
    return sc


def metric_report(v2, nkept_pyscf):
    """Spectrum of the metric around both cuts. `flip_margin` is the distance
    of the nearest eigenvalue to ferric's 1e-10 cut over a backward-error bound
    of a symmetric eigensolver, n eps lambda_max: > 1 means no rounding can move
    a mode across the cut."""
    import numpy as np

    lam = np.linalg.eigvalsh(v2)
    n = lam.size
    eps = np.finfo(float).eps
    err = n * eps * float(lam[-1])
    near = float(np.min(np.abs(lam - FERRIC_LINDEP)))
    near_p = float(np.min(np.abs(lam - PYSCF_LINDEP)))
    return {
        "naux": int(n),
        "min_eigenvalue": float(lam[0]),
        "max_eigenvalue": float(lam[-1]),
        "kept_ferric": int(np.count_nonzero(lam >= FERRIC_LINDEP)),
        "kept_pyscf": int(nkept_pyscf),
        "eigenvalue_error_bound": err,
        "nearest_to_ferric_cut": near,
        "nearest_to_pyscf_cut": near_p,
        "flip_margin_ferric": near / err,
        "flip_margin_pyscf": near_p / err,
        "smallest_kept_ferric": float(lam[lam >= FERRIC_LINDEP][0]),
    }


def ferric_ints(xyz, basis_name, aux_name, perm, aux_perm, kind):
    """ferric's own (mu nu|P) and (P|Q) for `kind` (the libint2 engines
    rs_mp2_lr_rpa runs, through `AoIntegrals`), in PySCF's AO order with
    PySCF's (nao, nao, naux) layout."""
    import ferric
    import numpy as np

    ao = ferric.AoIntegrals(
        ferric.Molecule.from_xyz(str(xyz)), ferric.BasisSet.bundled(basis_name)
    )
    aux = ferric.BasisSet.bundled(aux_name)
    op = {"coulomb": ("coulomb", None), "erf": ("erf", OMEGA), "erfc": ("erfc", OMEGA)}[
        kind
    ]
    v3 = np.asarray(
        ao.eri3(aux, operator=op[0], omega=op[1])
    )  # (P, mu, nu) ferric order
    v2 = np.asarray(ao.eri2(aux, operator=op[0], omega=op[1]))
    inv = np.argsort(np.asarray(perm))
    ainv = np.argsort(np.asarray(aux_perm))
    v3 = v3.transpose(1, 2, 0)[np.ix_(inv, inv, ainv)]
    v2 = v2[np.ix_(ainv, ainv)]
    return v3, v2


def kernel_pieces(mf, mol, auxmol, aux, kind, d, fints):
    """PySCF DFMP2 + RPA on the `kind` fit, the numpy metric shift, and the
    numpy integral shift (ferric's integrals `fints` against PySCF's)."""
    import numpy as np
    from pyscf import df
    from pyscf.gw.rpa import RPA
    from pyscf.mp import dfmp2

    w = gar.pyscf_omega(kind, OMEGA)
    with mol.with_range_coulomb(w):
        dfo = df.DF(mol, auxbasis=aux)
        dfo.verbose = 0
        dfo.build()
    assert mol.omega == 0.0, "mol left attenuated"
    assert dfo.auxmol.omega == w, f"auxmol omega {dfo.auxmol.omega} != {w}"
    decomp = decomposition_pyscf_used(dfo)

    pt = dfmp2.DFMP2(mf)
    pt.with_df = dfo
    pt.verbose = 0
    pt.kernel()
    code = {
        "e_os": float(pt.e_corr_os),
        "e_ss": float(pt.e_corr_ss),
        "e_mp2": float(pt.e_corr),
    }
    for nw in NWS:
        r = RPA(mf)
        r.with_df = dfo
        r.verbose = 0
        r.e_hf = float(mf.e_tot)  # skip the (meaningless) attenuated EXX
        r.kernel(nw=nw, x0=X0)
        code[f"e_drpa_nw{nw}"] = float(r.e_corr)

    nao = mol.nao_nr()
    v3p = unpack_cderi(dfo, nao)
    b_pyscf = gar.b_ov(mf, v3p, np.eye(v3p.shape[2]))
    num_p = numpy_energies(b_pyscf, mf, d)
    v3, v2 = gar.ints(mol, auxmol, w)
    m, _ = gar.metric_inv_sqrt(v2, kind, FERRIC_LINDEP)
    num_f = numpy_energies(gar.b_ov(mf, v3, m), mf, d)
    floors = []
    for seed in FLOOR_SEEDS:
        rng = np.random.default_rng(seed)
        s3 = rng.choice((-1.0, 1.0), size=v3.shape)
        s3 = 0.5 * (s3 + s3.transpose(1, 0, 2))
        s2 = rng.choice((-1.0, 1.0), size=v2.shape)
        s2 = np.triu(s2) + np.triu(s2, 1).T
        m_p, _ = gar.metric_inv_sqrt(v2 * (1.0 + INT_REL_ERR * s2), kind, FERRIC_LINDEP)
        floors.append(
            numpy_energies(gar.b_ov(mf, v3 * (1.0 + INT_REL_ERR * s3), m_p), mf, d)
        )
    v3_fi, v2_fi = fints
    m_fi, _ = gar.metric_inv_sqrt(v2_fi, kind, FERRIC_LINDEP)
    num_fi = numpy_energies(gar.b_ov(mf, v3_fi, m_fi), mf, d)
    # One field evaluation per kept mode inside the window, that mode's
    # eigenvalue moved by the eigensolver error bound (ferric's metric).
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    iap = np.einsum("mnP,mi,na->iaP", v3_fi, c[:, :nocc], c[:, nocc:], optimize=True)
    iap = iap.reshape(-1, iap.shape[2])
    lam_f, u_f = np.linalg.eigh(v2_fi)
    err_f = lam_f.size * np.finfo(float).eps * float(lam_f[-1])
    kept_f = lam_f >= FERRIC_LINDEP if kind == "erf" else np.ones(lam_f.size, bool)
    window = np.flatnonzero(kept_f & (lam_f <= EIG_WINDOW * err_f))
    eig_moved = []
    for k in window:
        uk = u_f[:, k]
        m_k, _ = gar.metric_inv_sqrt(
            v2_fi + err_f * np.outer(uk, uk), kind, FERRIC_LINDEP
        )
        eig_moved.append(numpy_energies(iap @ m_k.T, mf, d))
    int_diff = {
        "max_abs_3c": float(np.max(np.abs(v3_fi - v3))),
        "max_abs_2c": float(np.max(np.abs(v2_fi - v2))),
    }
    flip = None
    if kind == "erf":
        # The mode of ferric's own metric nearest ferric's cut, moved to the
        # other side of it.
        lam = np.linalg.eigvalsh(v2_fi)
        near = float(lam[np.argmin(np.abs(lam - FERRIC_LINDEP))])
        cut = near * (1.0 + 1e-9) if near >= FERRIC_LINDEP else near * (1.0 - 1e-9)
        m_flip, _ = gar.metric_inv_sqrt(v2_fi, kind, cut)
        flip = numpy_energies(gar.b_ov(mf, v3_fi, m_flip), mf, d)
    resid = {k: abs(num_p[k] - code[k]) for k in code}
    worst = max(resid.values())
    if worst > CODE_RESIDUAL_MAX:
        raise RuntimeError(
            f"{kind}: numpy on PySCF's tensor misses PySCF's code by {worst:.2e}"
        )
    return {
        "pyscf_omega": w,
        "decomposition": decomp,
        "pyscf": code,
        "numpy_on_pyscf_tensor": num_p,
        "numpy_ferric_metric": num_f,
        "numpy_ferric_metric_perturbed": floors,
        "numpy_ferric_integrals": num_fi,
        "integral_difference": int_diff,
        "numpy_ferric_integrals_eigenvalue_moved": eig_moved,
        "eigenvalue_window": {
            "error_bound": err_f,
            "modes": int(window.size),
            "largest_eigenvalue_in_window": float(lam_f[window[-1]])
            if window.size
            else None,
        },
        "numpy_ferric_metric_nearest_mode_flipped": flip,
        "code_residual": resid,
        "metric": metric_report(v2_fi, v3p.shape[2]),
    }


def assemble(p, nw):
    """Every RsMp2RpaResult field from per-kernel pieces p[kind] =
    {e_os, e_ss, e_mp2, e_drpa_nw<nw>}."""
    c, lr, sr = p["coulomb"], p["erf"], p["erfc"]
    rpa = f"e_drpa_nw{nw}"
    d_full = c[rpa] - 2.0 * c["e_os"]
    d_sr = sr[rpa] - 2.0 * sr["e_os"]
    return {
        "e_mp2_full": c["e_mp2"],
        "e_sr_mp2": sr["e_mp2"],
        "e_lr_mp2": lr["e_mp2"],
        "e_dmp2_lr": 2.0 * lr["e_os"],
        "B_e_drpa_lr": lr[rpa],
        "B_e_corr_naive": sr["e_mp2"] + lr[rpa],
        "B_e_corr": c["e_mp2"] + lr[rpa] - 2.0 * lr["e_os"],
        "T_e_delta_drpa_full": d_full,
        "T_e_delta_drpa_sr": d_sr,
        "T_e_corr": c["e_mp2"] + d_full - d_sr,
        # Controls the test must miss: the ring subtraction with E_OS for
        # 2 E_OS (the ledger mutation, B), and the subtraction's sign flipped.
        "B_control_dmp2_coeff_1": c["e_mp2"] + lr[rpa] - lr["e_os"],
        "B_control_wrong_sign": c["e_mp2"] + lr[rpa] + 2.0 * lr["e_os"],
        "T_control_dmp2_coeff_1": c["e_mp2"]
        + (c[rpa] - c["e_os"])
        - (sr[rpa] - sr["e_os"]),
        "T_control_wrong_sign": c["e_mp2"] + d_full + d_sr,
    }


def gen_case(key):
    import ferric
    import numpy as np
    import pyscf
    import scipy
    from pyscf import gto, scf

    system, basis_name, aux_name = CASES[key]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = gen_urpa.aux_dict(aux_name, symbols)
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=aux,
        cart=False,
        verbose=0,
    )
    naux = int(auxmol.nao_nr())
    assert naux == common.ferric_nao(aux_name, symbols), "aux count differs from ferric"
    perm = gen_properties.ferric_ao_permutation(mol, basis_name, symbols)
    s_min = float(np.linalg.eigvalsh(mol.intor("int1e_ovlp"))[0])

    sign = sign_convention(OMEGA)
    ident = gar.kernel_identity(mol, auxmol, OMEGA)

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    g_norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    rhf_stable = common._stability_status(mf)[1]
    assert rhf_stable, "RHF not internally stable"
    d = gar.gaps(mf)

    aux_perm = gen_properties.ferric_ao_permutation(auxmol, aux_name, symbols)
    pieces = {
        k: kernel_pieces(
            mf,
            mol,
            auxmol,
            aux,
            k,
            d,
            ferric_ints(xyz, basis_name, aux_name, perm, aux_perm, k),
        )
        for k in KINDS
    }
    per_nw = {}
    for nw in NWS:
        src = {
            s: {k: pieces[k][s] for k in KINDS}
            for s in (
                "pyscf",
                "numpy_on_pyscf_tensor",
                "numpy_ferric_metric",
                "numpy_ferric_integrals",
            )
        }
        fields_pyscf = assemble(src["pyscf"], nw)
        fields_np = assemble(src["numpy_on_pyscf_tensor"], nw)
        fields_f = assemble(src["numpy_ferric_metric"], nw)
        fields_fi = assemble(src["numpy_ferric_integrals"], nw)
        # A mode closer to ferric's cut than the eigensolver error bound may
        # sit on either side of it in ferric's own eigh: the field change of
        # that flip enters the bar (0 when every metric has margin > 1).
        flipped = {
            k: (
                pieces[k]["numpy_ferric_metric_nearest_mode_flipped"]
                if pieces[k]["metric"]["flip_margin_ferric"] <= 1.0
                else pieces[k]["numpy_ferric_integrals"]
            )
            for k in KINDS
        }
        fields_flip = assemble(flipped, nw)
        pert = [
            assemble(
                {k: pieces[k]["numpy_ferric_metric_perturbed"][i] for k in KINDS}, nw
            )
            for i in range(len(FLOOR_SEEDS))
        ]
        # Each code's metric eigenvalues carry the error bound independently,
        # so the bound on their difference is twice the per-mode sum.
        eig_err = {k: 0.0 for k in fields_fi}
        for kind in KINDS:
            for moved in pieces[kind]["numpy_ferric_integrals_eigenvalue_moved"]:
                src_k = dict(src["numpy_ferric_integrals"])
                src_k[kind] = moved
                f_k = assemble(src_k, nw)
                for k in eig_err:
                    eig_err[k] += 2.0 * abs(f_k[k] - fields_fi[k])
        freqs, wts = gen_urpa.scaled_legendre(nw, X0)
        per_nw[str(nw)] = {
            "nw": nw,
            "pyscf_nodes": [float(x) for x in freqs],
            "pyscf_weights": [float(x) for x in wts],
            "fields": fields_pyscf,
            "metric_shift": {k: fields_f[k] - fields_np[k] for k in fields_pyscf},
            "code_residual": {
                k: abs(fields_np[k] - fields_pyscf[k]) for k in fields_pyscf
            },
            "integral_shift": {k: fields_fi[k] - fields_f[k] for k in fields_pyscf},
            "cut_flip": {k: abs(fields_flip[k] - fields_fi[k]) for k in fields_pyscf},
            "eigenvalue_error": eig_err,
            "fp_floor": {
                k: max(abs(p[k] - fields_f[k]) for p in pert) for k in fields_pyscf
            },
        }
        f = fields_pyscf
        print(
            f"{key} nw={nw}: B {f['B_e_corr']:.12f} T {f['T_e_corr']:.12f} "
            f"MP2 {f['e_mp2_full']:.12f}; metric shift B "
            f"{per_nw[str(nw)]['metric_shift']['B_e_corr']:+.2e} T "
            f"{per_nw[str(nw)]['metric_shift']['T_e_corr']:+.2e}; fp floor B "
            f"{per_nw[str(nw)]['fp_floor']['B_e_corr']:.1e} T "
            f"{per_nw[str(nw)]['fp_floor']['T_e_corr']:.1e}"
        )
    for k in KINDS:
        m = pieces[k]["metric"]
        print(
            f"{key} {k}: PySCF {pieces[k]['decomposition']} kept {m['kept_pyscf']}/{m['naux']}, "
            f"ferric kept {m['kept_ferric']}; flip margin {m['flip_margin_ferric']:.1e}; "
            f"code residual {max(pieces[k]['code_residual'].values()):.1e}; "
            f"ferric - PySCF integrals max 3c {pieces[k]['integral_difference']['max_abs_3c']:.1e} "
            f"2c {pieces[k]['integral_difference']['max_abs_2c']:.1e}"
        )

    c = np.asarray(mf.mo_coeff)[np.asarray(perm), :]
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "overlap_min_eigenvalue": s_min,
        "like_for_like": ll,
        "omega_bohr_inv": OMEGA,
        "omega_angstrom_inv": OMEGA * gar.BOHR_PER_ANG,
        "x0": X0,
        "frozen_core": 0,
        "rhf": {
            "energy": float(mf.e_tot),
            "orbital_gradient_norm": g_norm,
            "internal_stable": bool(rhf_stable),
            "mo_energy": [float(x) for x in mf.mo_energy],
            "mo_coeff_ferric_order": [[float(x) for x in row] for row in c],
        },
        "sign_convention": sign,
        "kernels": pieces,
        "quadratures": per_nw,
        "checks": {
            "erf_plus_erfc_minus_coulomb_max_abs": ident,
            "code_residual_max": max(
                max(pieces[k]["code_residual"].values()) for k in KINDS
            ),
        },
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (RHF; df.DF under Mole.with_range_coulomb; mp.dfmp2.DFMP2; gw.rpa.RPA) "
        "+ numpy (metric-factorization shift on the same integrals; integral shift "
        "on ferric's integrals)",
        version=pyscf.__version__,
        keywords={
            "rhf": f"scf.RHF, exact 4-index Coulomb J/K, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
            "operator": "df.DF built inside mol.with_range_coulomb(w): w=0 Coulomb, w>0 erf, w<0 erfc",
            "pyscf_metric": "cholesky_eri: Cholesky, else eigh dropping eigenvalues <= 1e-7",
            "ferric_metric": "erf: eigh dropping eigenvalues < 1e-10; Coulomb/erfc: Cholesky",
            "mp2": "DFMP2 e_corr_os, e_corr_ss, e_corr",
            "rpa": f"gw.rpa.RPA nw in {list(NWS)}, x0 {X0}; e_hf preset (EXX step skipped)",
            "B": "E_MP2[C] + E_dRPA[erf] - 2 E_OS[erf]",
            "T": "E_MP2[C] + (E_dRPA[C] - 2 E_OS[C]) - (E_dRPA[erfc] - 2 E_OS[erfc])",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "ferric_integrals": "ferric.AoIntegrals eri3/eri2 (libint2, precision 1e-14) "
            f"from build {ferric.__build__}",
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux_name,
        frozen_core=0,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"rhf_internal_stable": bool(rhf_stable)},
        generator="scripts/validation/gen_rs_mp2_rpa_pyscf.py",
        extra={
            "aux_basis_json": str(
                common.basis_json_path(aux_name).relative_to(common.ROOT)
            ),
            "aux_basis_sha256": common.sha256_file(common.basis_json_path(aux_name)),
        },
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    print(
        f"{key}: E_RHF {mf.e_tot:.10f} |g| {g_norm:.1e} sign check "
        f"{max(sign[k]['abs_diff'] for k in KINDS):.1e} identity {ident:.1e} "
        f"({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(CASES)
    unknown = [k for k in want if k not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown} (known: {sorted(CASES)})")
    for k in want:
        gen_case(k)
    print(f"GEN_RS_MP2_RPA_PYSCF_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
