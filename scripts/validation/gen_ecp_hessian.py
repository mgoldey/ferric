"""PySCF references for the VALIDATION.md row "Analytic Hessian with ECPs".

Consumer: crates/ferric-scf/tests/validation_ecp_hessian.rs.
Output:   testdata/reference/validation/ecp_hessian/<system>_<basis>.json

Same like-for-like machinery as gen_ecp_gradient.py: ferric's bundled def2 /
aug-cc-pVDZ-PP JSON is the basis AND the ECP, handed to PySCF through
`common.build_pyscf_mol(..., ecp_json=...)`, geometry in Bohr, exact four-index
J/K. `check_ecp_like_for_like` asserts the per-atom core-electron counts, the
electron count and the ECP term count before anything is written.

Systems (geometries in testdata/molecules/validation/ecp/, all OFF equilibrium
or non-symmetric so every Cartesian block is populated):

    hi     HI      RHF   def2-SVP    one ECP centre
    ch3i   CH3I    RHF   def2-SVP    distorted: three distinct atoms around the
                                     ECP centre, x/y/z all populated
    ag2    Ag2     RHF   aug-cc-pVDZ-PP   two ECP centres, f shells
    ch2i   CH2I*   UHF   def2-SVP    doublet radical, open-shell ECP Hessian

Per system the JSON carries (all 3N x 3N, Ha/Bohr^2, row/col = 3*atom + xyz):

* `hessian_analytic`  PySCF `mf.Hessian().kernel()` (CPHF tol 1e-12), symmetrised.
* `hessian_skeleton`  `partial_hess_elec + hess_nuc` (terms at frozen D and W).
* `hessian_ecp_skeleton`  the ECP-only part of that skeleton,
  sum_uv D_uv d2 V_ECP,uv / dR dR, rebuilt from the SAME integrals PySCF's
  `hessian.rhf.hcore_generator` uses (`ECPscalar_ipipnuc`, `ECPscalar_ipnucip`,
  `ECPscalar_ipiprinv`, `ECPscalar_iprinvip`) with the nuclear-attraction and
  kinetic pieces dropped, at PySCF's converged total density.
  SELF-CHECKED before writing against a 5-point central FD of PySCF's own
  analytic ECP gradient term at FIXED density (the AO basis moves with the
  atoms); a miss > ECP_FD_TOL refuses the file. The measured miss (recorded as
  `hessian_ecp_skeleton_fixed_density_fd_max_err`) is PySCF's own ECP-integral
  roughness (2.5e-7 CH3I, 3.7e-6 CH2I): ferric cannot agree with PySCF's ECP
  skeleton better than that, and the validation bars are set from it.
* `hessian_fd`  central FD of PySCF's ANALYTIC gradient (6N gradients).
* frequencies of both (`harmonic_analysis`, ferric's masses).

A reference whose ECP skeleton max element is < MIN_ECP_SKELETON is refused
(a vacuous comparison).

Usage:
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_ecp_hessian.py [system ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_ecp_gradient as gg  # noqa: E402

ROW = "ecp_hessian"
ROW_NAME = "Analytic Hessian with ECPs"
ECP_MOL_DIR = common.MOL_DIR / "ecp"
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
CPHF_TOL = 1e-12
FD_STEP = 1.0e-3
ECP_FD_STEP = 2.0e-3
ECP_FD_TOL = 1e-5
MIN_ECP_SKELETON = 1e-2
BASIN_TOL = 1e-3

# ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 standard weights, u); the
# Rust test asserts atom_masses(mol) equals the `masses_amu` written here.
FERRIC_MASSES = {"H": 1.008, "C": 12.011, "Ag": 107.8682, "I": 126.90447}

# system -> (charge, multiplicity, method, basis)
SYSTEMS = {
    "hi": (0, 1, "rhf", "def2-svp"),
    "ch3i": (0, 1, "rhf", "def2-svp"),
    "ag2": (0, 1, "rhf", "aug-cc-pvdz-pp"),
    "ch2i": (0, 2, "uhf", "def2-svp"),
}


def _as_3n(h4, natm):
    return h4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)


def _scf(mol, method, dm0=None):
    from pyscf import scf

    mf = (scf.RHF if method == "rhf" else scf.UHF)(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.conv_tol_cpscf = CPHF_TOL
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("displaced SCF did not converge")
    return mf


def pyscf_ecp_skeleton(mol, dm):
    """sum_uv D_uv d2 V_ECP,uv/dR_a dR_b as (3N, 3N): the ECP-only part of
    `hessian.rhf.hcore_generator` (zi -> 0, no kinetic/nuclear pieces)."""
    import numpy as np

    nao = mol.nao_nr()
    natm = mol.natm
    out = np.zeros((3 * natm, 3 * natm))
    if not mol.has_ecp():
        return out
    ecp_atoms = set(int(a) for a in mol._ecpbas[:, 0])
    aoslices = mol.aoslice_by_atom()
    h1aa = mol.intor("ECPscalar_ipipnuc", comp=9).reshape(3, 3, nao, nao)
    h1ab = mol.intor("ECPscalar_ipnucip", comp=9).reshape(3, 3, nao, nao)
    nbas = mol.nbas

    def rinv(ia, shls_slice=None):
        """-(ECPscalar_ipiprinv, ECPscalar_iprinvip) at nucleus ia (zero off-ECP)."""
        if ia not in ecp_atoms:
            return None, None
        with mol.with_rinv_at_nucleus(ia):
            kw = {} if shls_slice is None else {"shls_slice": shls_slice}
            aa = -mol.intor("ECPscalar_ipiprinv", comp=9, **kw)
            ab = -mol.intor("ECPscalar_iprinvip", comp=9, **kw)
        return aa, ab

    for ia in range(natm):
        ish0, ish1, i0, i1 = aoslices[ia]
        for ja in range(natm):
            jsh0, jsh1, j0, j1 = aoslices[ja]
            if ia == ja:
                aa, ab = rinv(ia)
                h = np.zeros((3, 3, nao, nao))
                if aa is not None:
                    aa = aa.reshape(3, 3, nao, nao)
                    ab = ab.reshape(3, 3, nao, nao)
                    h = -aa - ab
                    h[:, :, i0:i1] += aa[:, :, i0:i1]
                    h[:, :, i0:i1] += ab[:, :, i0:i1]
                    h[:, :, :, i0:i1] += aa[:, :, i0:i1].transpose(0, 1, 3, 2)
                    h[:, :, :, i0:i1] += ab[:, :, :, i0:i1]
                h[:, :, i0:i1] += h1aa[:, :, i0:i1]
                h[:, :, i0:i1, i0:i1] += h1ab[:, :, i0:i1, i0:i1]
            else:
                h = np.zeros((3, 3, nao, nao))
                h[:, :, i0:i1, j0:j1] += h1ab[:, :, i0:i1, j0:j1]
                aa, ab = rinv(ia, (jsh0, jsh1, 0, nbas))
                if aa is not None:
                    h[:, :, j0:j1] += aa.reshape(3, 3, j1 - j0, nao)
                    h[:, :, j0:j1] += ab.reshape(3, 3, j1 - j0, nao).transpose(
                        1, 0, 2, 3
                    )
                aa, ab = rinv(ja, (ish0, ish1, 0, nbas))
                if aa is not None:
                    h[:, :, i0:i1] += aa.reshape(3, 3, i1 - i0, nao)
                    h[:, :, i0:i1] += ab.reshape(3, 3, i1 - i0, nao)
            h = h + h.transpose(0, 1, 3, 2)
            out[3 * ia : 3 * ia + 3, 3 * ja : 3 * ja + 3] = np.einsum(
                "xypq,pq->xy", h, dm
            )
    return out


def ecp_skeleton_fd(mol, dm):
    """5-point FD of PySCF's analytic ECP gradient term at FIXED density:
    d/dR_{b,y} [ sum D dV_ECP/dR_{a,x} ], the AO basis moving with the atoms."""
    import numpy as np

    x0 = mol.atom_coords(unit="Bohr").copy()
    n3 = 3 * mol.natm
    out = np.zeros((n3, n3))
    for b in range(n3):
        acc = np.zeros(n3)
        for k, w in gg.STENCIL5:
            x = x0.copy()
            x[b // 3, b % 3] += k * ECP_FD_STEP
            m = mol.set_geom_(x, unit="Bohr", inplace=False)
            acc += w * gg.pyscf_ecp_gradient_term(m, dm).ravel()
        out[:, b] = acc / ECP_FD_STEP
    return out


def _analytic(mf):
    import numpy as np
    from pyscf import hessian  # noqa: F401  (registers mf.Hessian)

    hobj = mf.Hessian()
    hobj.max_cycle = 100
    natm = mf.mol.natm
    raw = _as_3n(hobj.kernel(), natm)
    skel = _as_3n(
        hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ) + hobj.hess_nuc(),
        natm,
    )
    asym = float(np.max(np.abs(raw - raw.T)))
    return 0.5 * (raw + raw.T), asym, 0.5 * (skel + skel.T)


def _fd(mol, mf0, method):
    import numpy as np

    n = 3 * mol.natm
    x0 = mol.atom_coords(unit="Bohr")
    dm0 = mf0.make_rdm1()
    e0 = mf0.e_tot
    h = np.zeros((n, n))
    for b in range(n):
        grads = []
        for sgn in (+1.0, -1.0):
            x = x0.copy()
            x[b // 3, b % 3] += sgn * FD_STEP
            m = mol.copy()
            m.set_geom_(x, unit="Bohr")
            mf = _scf(m, method, dm0=dm0)
            if abs(mf.e_tot - e0) > BASIN_TOL:
                raise RuntimeError(f"displaced SCF changed state: {mf.e_tot - e0:+.3e}")
            grads.append(mf.nuc_grad_method().kernel().ravel())
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


def _write_compact(system: str, basis_name: str, payload: dict) -> Path:
    if "provenance" not in payload:
        raise ValueError("refusing to write a reference with no provenance block")
    path = common.reference_path(ROW, system, basis_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def _run(system: str) -> Path:
    import numpy as np
    import pyscf

    charge, mult, method, basis_name = SYSTEMS[system]
    xyz = ECP_MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(basis_name)
    mol = common.build_pyscf_mol(
        xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
    masses = [FERRIC_MASSES[s.capitalize()] for s in symbols]

    if method == "rhf":
        mf, scf_block = gg.run_rhf(mol)
    else:
        scf_block, mf = common.run_open_shell(
            mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
        )
    mf.conv_tol_cpscf = CPHF_TOL
    dm = mf.make_rdm1()
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]

    print(f"{system}: SCF done, E {mf.e_tot:.10f}", flush=True)
    h_an, an_asym, h_skel = _analytic(mf)
    print(f"{system}: PySCF analytic Hessian done", flush=True)
    h_ecp = pyscf_ecp_skeleton(mol, dm_tot)
    h_ecp = 0.5 * (h_ecp + h_ecp.T)
    h_ecp_fd = ecp_skeleton_fd(mol, dm_tot)
    ecp_err = float(np.max(np.abs(h_ecp - 0.5 * (h_ecp_fd + h_ecp_fd.T))))
    ecp_max = float(np.max(np.abs(h_ecp)))
    if ecp_err > ECP_FD_TOL:
        raise RuntimeError(
            f"{system}: PySCF ECP skeleton misses its fixed-D FD by {ecp_err:.2e} "
            f"(> {ECP_FD_TOL:.0e}); the term decomposition is wrong, refusing"
        )
    if ecp_max < MIN_ECP_SKELETON:
        raise RuntimeError(f"{system}: ECP skeleton max {ecp_max:.2e}: vacuous")
    print(f"{system}: ECP skeleton checked ({ecp_err:.1e})", flush=True)
    h_fd, fd_asym = _fd(mol, mf, method)
    f_an = _frequencies(mol, h_an, masses)
    f_fd = _frequencies(mol, h_fd, masses)
    gap_h = float(np.max(np.abs(h_fd - h_an)))
    gap_f = float(max(abs(a - b) for a, b in zip(f_an, f_fd)))
    print(
        f"{system:6s} {basis_name:15s} E {mf.e_tot:.10f} freqs {['%.1f' % f for f in f_an]} "
        f"| FD-vs-an H {gap_h:.2e} freq {gap_f:.3f} cm-1 | ECP skeleton max {ecp_max:.3f} "
        f"fixed-D FD err {ecp_err:.1e} | an asym {an_asym:.1e}",
        flush=True,
    )
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "charge": charge,
        "multiplicity": mult,
        "method": method,
        "nao": mol.nao_nr(),
        "nelectron": int(mol.nelectron),
        "ecp_core_electrons": ecp_check["ecp_core_electrons"],
        "nuclear_repulsion": float(mol.energy_nuc()),
        "masses_amu": masses,
        "hessian_layout": "3N x 3N Cartesian, row/col = 3*atom + xyz, Hartree/Bohr^2",
        "frequency_convention": (
            "harmonic_analysis(mass=masses_amu, imaginary_freq=False): ascending, "
            "imaginary reported as negative, translations+rotations projected out"
        ),
        "energy": float(mf.e_tot),
        "scf": scf_block,
        "gradient": np.asarray(mf.nuc_grad_method().kernel()).tolist(),
        "hessian_analytic": h_an.tolist(),
        "hessian_analytic_asymmetry": an_asym,
        "hessian_skeleton": h_skel.tolist(),
        "hessian_ecp_skeleton": h_ecp.tolist(),
        "hessian_ecp_skeleton_fd": (0.5 * (h_ecp_fd + h_ecp_fd.T)).tolist(),
        "hessian_ecp_skeleton_fixed_density_fd_max_err": ecp_err,
        "hessian_ecp_skeleton_max_abs": ecp_max,
        "freq_analytic_cm": f_an,
        "hessian_fd": h_fd.tolist(),
        "freq_fd_cm": f_fd,
        "fd": {
            "fd_step_bohr": FD_STEP,
            "fd_asymmetry_max": fd_asym,
            "fd_n_gradients": 6 * mol.natm,
        },
        "pyscf_fd_vs_analytic_hessian_max_abs": gap_h,
        "pyscf_fd_vs_analytic_freq_max_abs_cm": gap_f,
        "provenance": common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": method,
                "conv_tol": CONV_TOL,
                "conv_tol_grad": CONV_TOL_GRAD,
                "conv_tol_cpscf": CPHF_TOL,
                "eri": "exact 4-index (no density fitting)",
                "analytic_hessian": "pyscf.hessian (mf.Hessian().kernel()), symmetrized",
                "skeleton": "Hessian.partial_hess_elec + Hessian.hess_nuc",
                "ecp_skeleton": (
                    "ECPscalar_ipipnuc/ipnucip/ipiprinv/iprinvip per hcore_generator, "
                    "ECP terms only, contracted with the total density"
                ),
                "fd_hessian": f"central FD of the analytic gradient, step {FD_STEP} Bohr",
                "masses": "ferric_core::elements::ATOMIC_MASSES (IUPAC 2013 averages)",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=None,
            frozen_core=None,
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=scf_block["stability"],
            ecp_json=ecp_json,
            generator="scripts/validation/gen_ecp_hessian.py",
            extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
        ),
    }
    return _write_compact(system, basis_name, payload)


def _scf_for(system: str):
    """(mol, mf, total density) of `system`, exactly as `_run` builds them."""
    charge, mult, method, basis_name = SYSTEMS[system]
    xyz = ECP_MOL_DIR / f"{system}.xyz"
    ecp_json = common.basis_json_path(basis_name)
    mol = common.build_pyscf_mol(
        xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    if method == "rhf":
        mf, _ = gg.run_rhf(mol)
    else:
        _, mf = common.run_open_shell(
            mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
        )
    dm = mf.make_rdm1()
    return mol, mf, (dm if dm.ndim == 2 else dm[0] + dm[1])


def _patch_ecp_fd(system: str) -> Path:
    """Add `hessian_ecp_skeleton_fd` to an existing reference (same SCF state:
    the stored ECP skeleton must be reproduced to 1e-8 first)."""
    import numpy as np

    path = common.reference_path(ROW, system, SYSTEMS[system][3])
    payload = json.loads(path.read_text())
    mol, _mf, dm_tot = _scf_for(system)
    h_ecp = pyscf_ecp_skeleton(mol, dm_tot)
    h_ecp = 0.5 * (h_ecp + h_ecp.T)
    drift = float(np.max(np.abs(h_ecp - np.array(payload["hessian_ecp_skeleton"]))))
    if drift > 1e-8:
        raise RuntimeError(
            f"{system}: re-run SCF drifted from the reference by {drift:.1e}"
        )
    fd = ecp_skeleton_fd(mol, dm_tot)
    payload["hessian_ecp_skeleton_fd"] = (0.5 * (fd + fd.T)).tolist()
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        f"{system}: patched hessian_ecp_skeleton_fd (SCF drift {drift:.1e})", flush=True
    )
    return path


def main() -> int:
    if sys.argv[1:2] == ["--patch-ecp-fd"]:
        for s in sys.argv[2:]:
            _patch_ecp_fd(s)
        return 0
    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        print(f"unknown systems: {sorted(unknown)}", file=sys.stderr)
        return 2
    written = [_run(s) for s in SYSTEMS if not only or s in only]
    for p in written:
        print(f"  {p.relative_to(common.ROOT)} ({p.stat().st_size} bytes)")
    print(f"GEN_ECP_HESSIAN_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
