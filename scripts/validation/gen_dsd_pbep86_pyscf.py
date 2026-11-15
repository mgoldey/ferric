"""PySCF references for the VALIDATION.md "DSD-PBEP86 double hybrid" row.

Consumer: crates/ferric-mp2/tests/validation_dsd_pbep86_pyscf.rs.
Output:   testdata/reference/validation/dsd_pbep86_pyscf/<system>_<basis>.json

THE DEFINITION BUILT HERE (Kozuch and Martin, J. Comput. Chem. 34, 2327
(2013), the set without a dispersion term, "DSD-PBEP86" in Psi4's
dh_functionals.py)

    E = E_KS[0.72 HF + 0.28 PBE exchange, 0.44 P86 correlation]
        + 0.51 E_OS + 0.36 E_SS

with E_OS / E_SS the opposite- and same-spin MP2 pair energies evaluated on
the KS orbitals and orbital energies of that functional. Nothing is copied
from ferric: the functional is PySCF's own libxc string
`0.72*HF + 0.28*PBE, 0.44*P86` (GGA_X_PBE, GGA_C_P86; asserted below), and the
PT2 term is `mp.dfmp2.DFMP2` on the converged KS object (it reads only
mo_coeff / mo_energy / mo_occ, which for a KS object are the KS ones), split
into its OS and SS parts and scaled separately.

The other two published sets (DSD-PBEP86-D3BJ: 0.69 / 0.31 / 0.44, 0.52 / 0.22,
D3(BJ); DSD-PBEP86-D2, the 2011 set: 0.68 / 0.32 / 0.45, 0.51 / 0.23, D2) carry
a dispersion term and are not what `dsd-pbep86` computes.

LIKE-FOR-LIKE SETTINGS (each is what ferric's CLI path
`run_mp2_double_hybrid_arm` + the library call `mp2_double_hybrid` do)

  * basis, geometry: ferric's JSON and Bohr coordinates through common.py;
  * KS grid: (75, 110) unpruned, Becke partition, Becke (1988) radii. ferric's
    KS default is the PRUNED `AtomicGridConfig::production()`, which PySCF
    cannot reproduce; the ferric test pins the flat grid exactly as
    validation_b2plyp_pyscf.rs does;
  * KS J/K: density_fit over ferric's def2-universal-jkfit (J AND the 0.72 K),
    the CLI default for the double-hybrid arm; the ferric test pins the RI-J
    tensor storage to f64 (the CLI default is f32);
  * PT2: df.DF over the aux basis named per case (ferric's cc-pVDZ-RI, or
    def2-SVP-RIFIT), Coulomb operator and metric; `frozen` = 0 and 1;
  * convergence: conv_tol 1e-12, conv_tol_grad 1e-9.

REPRODUCTION TERMS recorded for the consumer's bar: the same case is rerun
from a second initial guess (`atom`) and the spread of E_KS, E_OS, E_SS is
stored under `reproduction`.

CONTROLS the Rust side must MISS: the previous ferric parameter set
(0.69 HF + 0.31 PBE, 0.44 P86; c_os 0.56, c_ss 0.29), as a KS energy and as a
total on the same orbitals and pair energies.

Run:
    box-lease run --label dsd-gen -- env OPENBLAS_NUM_THREADS=1 \\
        uv run --no-sync python scripts/validation/gen_dsd_pbep86_pyscf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "dsd_pbep86_pyscf"
JKFIT = "def2-universal-jkfit"
XC = "0.72*HF + 0.28*PBE, 0.44*P86"
XC_OLD = "0.69*HF + 0.31*PBE, 0.44*P86"
C_OS = 0.51
C_SS = 0.36
C_OS_OLD = 0.56
C_SS_OLD = 0.29
MAIN_GRID = (75, 110)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9

# (system, basis, PT2 aux basis)
CASES = {
    "h2o_cc-pvdz": ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    "nh3_cc-pvdz": ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    "ch4_cc-pvdz": ("ch4", "cc-pvdz", "cc-pvdz-ri"),
    "h2o_def2-svp": ("h2o", "def2-svp", "def2-svp-rifit"),
    "nh3_def2-svp": ("nh3", "def2-svp", "def2-svp-rifit"),
    "ch4_def2-svp": ("ch4", "def2-svp", "def2-svp-rifit"),
}
FROZEN = (0, 1)


def ks(mol, jkfit, guess, xc=XC):
    from pyscf import dft

    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=jkfit)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.init_guess = guess
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RKS did not converge from {guess}")
    return mf


def check_functional():
    """The XC string parses to exactly 0.72 HF, 0.28 GGA_X_PBE, 0.44 GGA_C_P86."""
    from pyscf import dft

    (hyb, _alpha, _omega), funcs = dft.libxc.parse_xc(XC)
    got = {int(i): float(w) for i, w in funcs}
    pbe_x = dft.libxc.parse_xc("PBE,")[1][0][0]
    p86_c = dft.libxc.parse_xc(",P86")[1][0][0]
    want = {pbe_x: 0.28, p86_c: 0.44}
    assert abs(hyb - 0.72) < 1e-15, hyb
    assert got.keys() == want.keys(), got
    assert all(abs(got[k] - want[k]) < 1e-15 for k in want), got


def pt2(mf, mol, aux, frozen):
    from pyscf import df
    from pyscf.mp import dfmp2

    pt = dfmp2.DFMP2(mf)
    pt.with_df = df.DF(mol, auxbasis=aux)
    pt.frozen = frozen
    pt.verbose = 0
    pt.kernel()
    return {
        "e_os": float(pt.e_corr_os),
        "e_ss": float(pt.e_corr_ss),
        "e_corr": float(pt.e_corr),
    }


def gen_case(key):
    import numpy
    import pyscf

    check_functional()
    system, basis_name, aux_name = CASES[key]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    common.check_basis_like_for_like(mol, basis_name, symbols)
    jk, jk_cart = common.pyscf_basis(JKFIT, symbols)
    assert not jk_cart, "Cartesian jkfit shells cannot be matched"
    aux = gen_urpa.aux_dict(aux_name, symbols)
    mf = ks(mol, jk, "minao")
    mf2 = ks(mol, jk, "atom")
    nocc = mol.nelectron // 2
    out = {
        "row": "DSD-PBEP86",
        "system": system,
        "basis": basis_name,
        "pt2_aux": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "c_os": C_OS,
        "c_ss": C_SS,
        "ks": {
            "energy": float(mf.e_tot),
            "homo": float(mf.mo_energy[nocc - 1]),
            "lumo": float(mf.mo_energy[nocc]),
        },
    }
    repro = {"e_ks": abs(float(mf.e_tot) - float(mf2.e_tot))}
    for fc in FROZEN:
        p = pt2(mf, mol, aux, fc)
        p2 = pt2(mf2, mol, aux, fc)
        p["total"] = float(mf.e_tot) + C_OS * p["e_os"] + C_SS * p["e_ss"]
        p["total_old_scaling"] = (
            float(mf.e_tot) + C_OS_OLD * p["e_os"] + C_SS_OLD * p["e_ss"]
        )
        out[f"fc{fc}"] = p
        repro[f"fc{fc}_e_os"] = abs(p["e_os"] - p2["e_os"])
        repro[f"fc{fc}_e_ss"] = abs(p["e_ss"] - p2["e_ss"])
    out["reproduction"] = {
        "note": "|d| between the minao-guess and atom-guess PySCF runs of this case",
        **repro,
    }
    # Controls the Rust side must MISS: the previous ferric functional (KS
    # energy on the same grid / fit); the old PT2 scaling is stored per fc above.
    old = ks(mol, jk, "minao", xc=XC_OLD)
    out["old_parameter_control"] = {
        "xc": XC_OLD,
        "ks_energy": float(old.e_tot),
        "c_os": C_OS_OLD,
        "c_ss": C_SS_OLD,
    }
    out["provenance"] = common.provenance(
        code="PySCF (dft.RKS density_fit + mp.dfmp2.DFMP2)",
        version=pyscf.__version__,
        keywords={
            "xc": XC,
            "functional": "Kozuch-Martin 2013 DSD-PBEP86 (no dispersion): 0.72 HF + 0.28 PBE x, 0.44 P86 c, 0.51 OS + 0.36 SS PT2",
            "ks_jk": f"density_fit(auxbasis={JKFIT}) for J and K",
            "pt2": "mp.dfmp2.DFMP2 on the KS orbitals, with_df = df.DF(pt2 aux), frozen 0 and 1",
            "init_guess": "minao (reference), atom (reproduction)",
            "numpy": numpy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid={
            "main": {
                "atom_grid": list(MAIN_GRID),
                "prune": None,
                "partition": "Becke (original_becke)",
                "radii_adjust": "becke_atomic_radii_adjust",
            }
        },
        aux={
            "jk": JKFIT,
            "jk_json": str(common.basis_json_path(JKFIT).relative_to(common.ROOT)),
            "jk_sha256": common.sha256_file(common.basis_json_path(JKFIT)),
            "pt2": aux_name,
            "pt2_json": str(common.basis_json_path(aux_name).relative_to(common.ROOT)),
            "pt2_sha256": common.sha256_file(common.basis_json_path(aux_name)),
        },
        frozen_core=list(FROZEN),
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=None,
        generator="scripts/validation/gen_dsd_pbep86_pyscf.py",
    )
    path = common.write_reference(ROW, system, basis_name, out)
    print(
        f"{key}: E_KS {mf.e_tot:.10f} fc0 total {out['fc0']['total']:.10f} "
        f"repro E_KS {repro['e_ks']:.1e} ({time.perf_counter() - t0:.1f} s) "
        f"-> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(CASES)
    unknown = [k for k in want if k not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown} (known: {sorted(CASES)})")
    for k in want:
        gen_case(k)
    print(f"GEN_DSD_PBEP86_PYSCF_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
