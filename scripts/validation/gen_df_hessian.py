"""PySCF references for the VALIDATION.md "Analytic RHF Hessian, RI-J" and
"Analytic RHF Hessian, RI-J + RI-K" rows.

Consumers: crates/ferric-scf/tests/validation_df_hessian.rs (RI-J),
           crates/ferric-scf/tests/validation_df_hessian_rik.rs (RI-JK).
Output:    testdata/reference/validation/df_hessian/<system>_<basis>.json     (RI-J)
           testdata/reference/validation/df_hessian_jk/<system>_<basis>.json  (RI-JK)

RI-JK block (`--jk`, see _run_jk): PySCF's STANDARD density-fitted RHF
(`scf.RHF(mol).density_fit(auxbasis)`: J and K both fitted with the SAME
auxiliary basis, as ferric does with df_j_aux = df_k_aux). Its analytic
Hessian `mf.Hessian()` (`pyscf.df.hessian.rhf`, auxbasis_response = 2, the
full response) is therefore a DIRECT reference, no composition. Alongside:
`hessian_fd` (central FD of PySCF's analytic DF gradient) and
`hessian_exact` (the plain exact-integral analytic Hessian: the NEGATIVE
CONTROL). Systems: h2o, ch2o, nh3_distorted x cc-pVDZ / def2-universal-jkfit
and h2o_distorted x def2-SVP / def2-universal-jkfit.

WHAT FERRIC COMPUTES (crates/ferric-scf/src/hessian_df.rs, read from the code):
the analytic closed-shell RHF Hessian for RI-J with EXACT exchange: the
Coulomb energy is the fitted `1/2 d^T V^-1 d` (Cholesky-solved Coulomb
metric, full inverse), K stays the exact four-centre one. The skeleton, the
first-order Fock matrix and the CPHF Coulomb kernel are the fitted ones; the
orbital-response machinery is the exact-integral one.

PySCF has no analytic RI-J-with-exact-K Hessian (`pyscf.df.hessian.rhf`
always fits K too), so each block here carries two like-for-like references:

* `hessian_fd` - central FD (FD_STEP) of PySCF's ANALYTIC gradient of
  `scf.RHF(mol).density_fit(auxbasis, only_dfj=True)` (`pyscf.df.grad.rhf`
  honours `only_dfj`: fitted J, exact K), every displaced SCF started from the
  centre density, symmetrized. Independent of any Hessian formula.
* `hessian_analytic` - an analytic Hessian COMPOSED from PySCF's own pieces:
  the skeleton `e1 + ej` of `pyscf.df.hessian.rhf._partial_hess_ejk(with_k=False)`
  (full auxiliary-basis response, `auxbasis_response = 2`) minus the exact
  `ek` of `pyscf.hessian.rhf._partial_hess_ejk`; `make_h1` = hcore + DF `vj1`
  - 1/2 exact `vk1`; the CPHF response `gen_response` of the `only_dfj` mean
  field (fitted J, exact K). Stored symmetrized with its raw asymmetry. It is
  validated against `hessian_fd` here (`pyscf_fd_vs_analytic_*`); ferric is
  compared to BOTH.
* `hessian_skeleton` - the composed `partial_hess_elec + hess_nuc` (terms 1-4).
* `hessian_exact_j` - PySCF's plain exact-integral analytic Hessian of the
  same molecule/basis: the NEGATIVE CONTROL (the fitting error is the signal).
* Frequencies of all three via `thermo.harmonic_analysis(mass = ferric's
  masses, imaginary_freq = False)` (cm-1).

Basis and auxiliary basis like-for-like through common.py (ferric's own JSON
for both; `cc-pvdz-ri` is the BSE cc-pVDZ-RIFIT file). Geometry in Bohr with
ferric's constant.

Systems (geometries in testdata/molecules/validation/, used AS GIVEN):
    h2o, ch2o                 x cc-pVDZ / cc-pvdz-ri
    nh3_distorted             x cc-pVDZ / cc-pvdz-ri   (off-C3v: all blocks populated)
    h2o_distorted             x def2-SVP / def2-universal-jkfit

Run (light; under two minutes per block):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python \\
        scripts/validation/gen_df_hessian.py          # RI-J block
    ... gen_df_hessian.py --jk                       # RI-J + RI-K block
Restrict to systems by name:  ... gen_df_hessian.py nh3_distorted
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "df_hessian"
ROW_NAME = "Analytic RHF Hessian, RI-J with exact K"
ROW_JK = "df_hessian_jk"
ROW_JK_NAME = "Analytic RHF Hessian, RI-J + RI-K"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CPHF_TOL = 1e-12
FD_STEP = 1.0e-3

# ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 standard weights, u).
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

SYSTEMS = (
    ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    ("ch2o", "cc-pvdz", "cc-pvdz-ri"),
    ("nh3_distorted", "cc-pvdz", "cc-pvdz-ri"),
    ("h2o_distorted", "def2-svp", "def2-universal-jkfit"),
)


SYSTEMS_JK = (
    ("h2o", "cc-pvdz", "def2-universal-jkfit"),
    ("ch2o", "cc-pvdz", "def2-universal-jkfit"),
    ("nh3_distorted", "cc-pvdz", "def2-universal-jkfit"),
    ("h2o_distorted", "def2-svp", "def2-universal-jkfit"),
)


def _aux(symbols, aux_name):
    aux, cart = common.pyscf_basis(aux_name, symbols)
    if cart:
        raise ValueError(f"{aux_name}: Cartesian l>=2 aux shells; not like-for-like")
    return aux


def _check_aux(mol, aux, aux_name, symbols) -> dict:
    from pyscf.df import addons

    auxmol = addons.make_auxmol(mol, aux)
    want = common.ferric_nao(aux_name, symbols)
    assert auxmol.nao_nr() == want, (
        f"{aux_name}: PySCF aux nao {auxmol.nao_nr()} != ferric {want}"
    )
    return {"aux_nao": want}


def _rhf(mol, aux=None, dm0=None, full_df=False):
    """RHF; `aux` set = fitted J with exact K (`only_dfj`; `full_df` = J and K
    both fitted, PySCF's standard density_fit), None = exact."""
    from pyscf import scf

    mf = scf.RHF(mol)
    if aux is not None:
        mf = mf.density_fit(auxbasis=aux, only_dfj=not full_df)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.conv_tol_cpscf = CPHF_TOL
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def _as_3n(h4, natm):
    return h4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _composed_hessian_class():
    """Analytic RI-J + exact-K Hessian assembled from PySCF's own pieces."""
    import numpy as np
    from pyscf.df.hessian import rhf as df_hess
    from pyscf.hessian import rhf as ex_hess

    class RIJHessian(ex_hess.Hessian):
        auxbasis_response = 2

        def partial_hess_elec(
            self,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
            atmlst=None,
            max_memory=4000,
            verbose=None,
        ):
            e1, ej, _ = df_hess._partial_hess_ejk(
                self, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose, False
            )
            _, _, ek = ex_hess._partial_hess_ejk(
                self, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose, True
            )
            return e1 + ej - ek

        def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
            mol = self.mol
            if atmlst is None:
                atmlst = range(mol.natm)
            mocc = mo_coeff[:, mo_occ > 0]
            dm0 = np.dot(mocc, mocc.T) * 2
            aoslices = mol.aoslice_by_atom()
            h1ao = [None] * mol.natm
            for ia, h1, vj1, _ in df_hess._gen_jk(
                self, mo_coeff, mo_occ, chkfile, atmlst, verbose, False
            ):
                shl0, shl1, p0, p1 = aoslices[ia]
                shls_slice = (shl0, shl1) + (0, mol.nbas) * 3
                vk1, vk2 = ex_hess._get_jk(
                    mol,
                    "int2e_ip1",
                    3,
                    "s2kl",
                    ["li->s1kj", -dm0[:, p0:p1], "jk->s1il", -dm0],
                    shls_slice=shls_slice,
                )
                vhf = -vk1 * 0.5
                vhf[:, p0:p1] += -vk2 * 0.5
                h1ao[ia] = h1 + vj1 + vhf + vhf.transpose(0, 2, 1)
            return h1ao

    return RIJHessian


def _analytic_composed(mf):
    import numpy as np

    hobj = _composed_hessian_class()(mf)
    hobj.max_cycle = 100
    natm = mf.mol.natm
    raw = _as_3n(hobj.kernel(), natm)
    skel = _as_3n(
        hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ) + hobj.hess_nuc(),
        natm,
    )
    asym = float(np.max(np.abs(raw - raw.T)))
    return 0.5 * (raw + raw.T), asym, 0.5 * (skel + skel.T)


def _analytic_exact(mf):
    import numpy as np
    from pyscf import hessian  # noqa: F401  (registers mf.Hessian)

    hobj = mf.Hessian()
    hobj.max_cycle = 100
    raw = _as_3n(hobj.kernel(), mf.mol.natm)
    return 0.5 * (raw + raw.T), float(np.max(np.abs(raw - raw.T)))


def _fd(mol, mf0, aux, full_df=False):
    import numpy as np

    n = 3 * mol.natm
    x0 = mol.atom_coords(unit="Bohr")
    dm0 = mf0.make_rdm1()
    h = np.zeros((n, n))
    for b in range(n):
        grads = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * FD_STEP
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            grads.append(
                _rhf(m, aux, dm0=dm0, full_df=full_df)
                .nuc_grad_method()
                .kernel()
                .ravel()
            )
        h[:, b] = (grads[0] - grads[1]) / (2.0 * FD_STEP)
    asym = float(np.max(np.abs(h - h.T)))
    return 0.5 * (h + h.T), asym


def _frequencies(mol, h, masses):
    import numpy as np
    from pyscf.hessian import thermo

    natm = mol.natm
    h4 = h.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(
        mol, h4, mass=np.asarray(masses), imaginary_freq=False
    )
    return sorted(float(v) for v in np.asarray(res["freq_wavenumber"]).real)


def _write_compact(system: str, basis_name: str, payload: dict, row: str = ROW) -> Path:
    """common.write_reference's contract (provenance required), compact JSON."""
    if "provenance" not in payload:
        raise ValueError("refusing to write a reference with no provenance block")
    path = common.reference_path(row, system, basis_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def _run(system: str, basis_name: str, aux_name: str) -> Path:
    import numpy as np
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = _aux(symbols, aux_name)
    aux_check = _check_aux(mol, aux, aux_name, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    mf = _rhf(mol, aux)
    h_an, an_asym, h_skel = _analytic_composed(mf)
    h_fd, fd_asym = _fd(mol, mf, aux)
    mf_exact = _rhf(mol)
    h_ex, ex_asym = _analytic_exact(mf_exact)
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    f_ex = _frequencies(mol, h_ex, masses)
    gap_h = float(np.max(np.abs(h_fd - h_an)))
    gap_f = float(max(abs(a - b) for a, b in zip(f_an, f_fd)))
    fit_h = float(np.max(np.abs(h_ex - h_an)))
    fit_f = float(max(abs(a - b) for a, b in zip(f_ex, f_an)))
    print(
        f"{system:14s} {basis_name:9s} E {mf.e_tot:.10f} freqs "
        f"{['%.1f' % f for f in f_an]} | FD-vs-composed H {gap_h:.2e} freq {gap_f:.3f} cm-1 "
        f"| exact-vs-RIJ H {fit_h:.2e} freq {fit_f:.2f} cm-1 | an asym {an_asym:.1e}",
        flush=True,
    )
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux_basis": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "frequency_convention": (
            "harmonic_analysis(mass=masses_amu, imaginary_freq=False): ascending, "
            "imaginary reported as negative, translations+rotations projected out"
        ),
        "energy": float(mf.e_tot),
        "energy_exact_j": float(mf_exact.e_tot),
        "gradient": np.asarray(mf.nuc_grad_method().kernel()).tolist(),
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": an_asym,
        "hessian_skeleton": h_skel.tolist(),
        "freq_analytic_cm": f_an,
        "hessian_fd": h_fd.tolist(),
        "freq_fd_cm": f_fd,
        "hessian_exact_j": h_ex.tolist(),
        "hessian_exact_j_asymmetry": ex_asym,
        "freq_exact_j_cm": f_ex,
        "fd": {
            "fd_step_bohr": FD_STEP,
            "fd_asymmetry_max": fd_asym,
            "fd_n_gradients": 6 * mol.natm,
        },
        "pyscf_fd_vs_analytic_hessian_max_abs": gap_h,
        "pyscf_fd_vs_analytic_freq_max_abs_cm": gap_f,
        "exact_j_vs_ri_j_hessian_max_abs": fit_h,
        "exact_j_vs_ri_j_freq_max_abs_cm": fit_f,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "rhf",
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "conv_tol_cpscf": CPHF_TOL,
                "max_cycle": 500,
                "eri": "RI-J (density_fit(only_dfj=True), fitted J, exact K)",
                "analytic_hessian": (
                    "composed: df.hessian.rhf._partial_hess_ejk(with_k=False) "
                    "e1+ej (auxbasis_response=2) - hessian.rhf._partial_hess_ejk ek; "
                    "make_h1 = hcore + df vj1 - 1/2 exact vk1; response gen_response of "
                    "the only_dfj mean field; symmetrized"
                ),
                "skeleton": "composed partial_hess_elec + hess_nuc",
                "fd_hessian": (
                    f"central FD of the analytic only_dfj RHF gradient, step {FD_STEP} "
                    "Bohr, displaced SCFs from the centre density, symmetrized"
                ),
                "negative_control": "plain exact-integral mf.Hessian()",
                "masses": "ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 averages)",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=aux_name,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator="scripts/validation/gen_df_hessian.py",
            extra={"basis_self_check": basis_check, "aux_self_check": aux_check},
        ),
    }
    return _write_compact(system, basis_name, payload)


def _analytic_standard(mf):
    """PySCF's standard analytic Hessian of `mf` (full auxiliary response)."""
    import numpy as np

    hobj = mf.Hessian()
    hobj.max_cycle = 100
    hobj.conv_tol = CPHF_TOL
    raw = _as_3n(hobj.kernel(), mf.mol.natm)
    return 0.5 * (raw + raw.T), float(np.max(np.abs(raw - raw.T)))


def _run_jk(system: str, basis_name: str, aux_name: str) -> Path:
    import numpy as np
    import pyscf

    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = _aux(symbols, aux_name)
    aux_check = _check_aux(mol, aux, aux_name, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    mf = _rhf(mol, aux, full_df=True)
    h_an, an_asym = _analytic_standard(mf)
    h_fd, fd_asym = _fd(mol, mf, aux, full_df=True)
    mf_exact = _rhf(mol)
    h_ex, ex_asym = _analytic_exact(mf_exact)
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    f_ex = _frequencies(mol, h_ex, masses)
    gap_h = float(np.max(np.abs(h_fd - h_an)))
    gap_f = float(max(abs(a - b) for a, b in zip(f_an, f_fd)))
    fit_h = float(np.max(np.abs(h_ex - h_an)))
    fit_f = float(max(abs(a - b) for a, b in zip(f_ex, f_an)))
    print(
        f"JK {system:14s} {basis_name:9s} E {mf.e_tot:.10f} freqs "
        f"{['%.1f' % f for f in f_an]} | FD-vs-analytic H {gap_h:.2e} freq {gap_f:.3f} cm-1 "
        f"| exact-vs-RIJK H {fit_h:.2e} freq {fit_f:.2f} cm-1 | an asym {an_asym:.1e}",
        flush=True,
    )
    payload = {
        "row": ROW_JK_NAME,
        "system": system,
        "basis": basis_name,
        "aux_basis": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": mol.nao_nr(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "frequency_convention": (
            "harmonic_analysis(mass=masses_amu, imaginary_freq=False): ascending, "
            "imaginary reported as negative, translations+rotations projected out"
        ),
        "energy": float(mf.e_tot),
        "energy_exact": float(mf_exact.e_tot),
        "gradient": np.asarray(mf.nuc_grad_method().kernel()).tolist(),
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": an_asym,
        "freq_analytic_cm": f_an,
        "hessian_fd": h_fd.tolist(),
        "freq_fd_cm": f_fd,
        "hessian_exact": h_ex.tolist(),
        "hessian_exact_asymmetry": ex_asym,
        "freq_exact_cm": f_ex,
        "fd": {
            "fd_step_bohr": FD_STEP,
            "fd_asymmetry_max": fd_asym,
            "fd_n_gradients": 6 * mol.natm,
        },
        "pyscf_fd_vs_analytic_hessian_max_abs": gap_h,
        "pyscf_fd_vs_analytic_freq_max_abs_cm": gap_f,
        "exact_vs_ri_jk_hessian_max_abs": fit_h,
        "exact_vs_ri_jk_freq_max_abs_cm": fit_f,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "rhf",
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "conv_tol_cpscf": CPHF_TOL,
                "max_cycle": 500,
                "eri": "RI-J + RI-K (density_fit(auxbasis), same auxiliary basis for J and K)",
                "analytic_hessian": (
                    "mf.Hessian().kernel() of the density-fitted RHF "
                    "(pyscf.df.hessian.rhf, auxbasis_response = 2), symmetrized"
                ),
                "fd_hessian": (
                    f"central FD of the analytic DF RHF gradient, step {FD_STEP} "
                    "Bohr, displaced SCFs from the centre density, symmetrized"
                ),
                "negative_control": "plain exact-integral mf.Hessian()",
                "masses": "ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 averages)",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=aux_name,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator="scripts/validation/gen_df_hessian.py --jk",
            extra={"basis_self_check": basis_check, "aux_self_check": aux_check},
        ),
    }
    return _write_compact(system, basis_name, payload, ROW_JK)


def main() -> int:
    args = sys.argv[1:]
    if "--jk" in args:
        args = [a for a in args if a != "--jk"]
        only = set(args)
        written = [_run_jk(s, b, a) for s, b, a in SYSTEMS_JK if not only or s in only]
        for p in written:
            print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
        print(f"GEN_DF_HESSIAN_JK_DONE written={len(written)}")
        return 0
    only = set(args)
    written = [_run(s, b, a) for s, b, a in SYSTEMS if not only or s in only]
    for p in written:
        print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
    print(f"GEN_DF_HESSIAN_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
