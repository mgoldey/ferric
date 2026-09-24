"""PySCF references for the VALIDATION.md "IEF-PCM" row (validation tier W1).

Consumer: crates/ferric-scf/tests/validation_pcm.rs.
Output:   testdata/reference/validation/pcm/<system>_<basis>.json

Purpose: separate the IEF-PCM SOLVER from the CAVITY. Each file carries
PySCF's own cavity (the SWIG surface from `pyscf.solvent.pcm.gen_surface`:
points, outward normals, areas, sphere radii, Gaussian charge exponents and
switching-function values), which the Rust test injects into ferric's solver
through `PcmConfig::cavity`, plus PySCF's converged IEF-PCM results on that
cavity at two dielectrics:

    solver level: v_grids (the solute potential PySCF solved with), q_sym
                  (the symmetrized charges), e_pcm, and K.v / R.v / diag(S) /
                  diag(D) as matrix-level fingerprints
    SCF level:    the solvated total energy, the vacuum energy, and their
                  difference

Systems: water and NH3 (testdata/molecules/{water,nh3}.xyz) x STO-3G and
cc-pVDZ (ferric's bundled JSON, via common.py), eps = 78.4 and 4.7.

PySCF settings: method "IEF-PCM", lebedev_order 29 (302 points per sphere),
vdw_scale 1.2 on PySCF's `modified_Bondi` radii (H = 1.1 A), SWIG
discretization. These are PySCF's defaults except `method` (PySCF defaults to
C-PCM) and `eps`. ferric's OWN cavity differs (Bondi H = 1.2 A, 110 points
per sphere by default), which is what the ferric-native comparison in the test
measures.

Run (light step; seconds per system):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_pcm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "pcm"
ROW_NAME = "IEF-PCM"
SYSTEMS = ("water", "nh3")
BASES = ("sto-3g", "cc-pvdz")
EPSILONS = (78.4, 4.7)
LEBEDEV_ORDER = 29  # 302 points per sphere (PySCF default)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9


def _floats(a) -> list[float]:
    return [float(x) for x in a]


def main() -> int:
    import numpy
    import pyscf
    from pyscf import scf
    from pyscf.solvent import pcm

    written = []
    for system in SYSTEMS:
        xyz = common.ROOT / "testdata" / "molecules" / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in BASES:
            mol = common.build_pyscf_mol(xyz, basis_name)
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)

            vac = scf.RHF(mol)
            vac.conv_tol = CONV_TOL
            vac.conv_tol_grad = CONV_TOL_GRAD
            e_vac = vac.kernel()
            assert vac.converged, f"{system}/{basis_name}: vacuum RHF not converged"

            cavity = None
            solvents = {}
            for eps in EPSILONS:
                mf = scf.RHF(mol).PCM()
                ws = mf.with_solvent
                ws.method = "IEF-PCM"
                ws.eps = eps
                ws.lebedev_order = LEBEDEV_ORDER
                ws.vdw_scale = 1.2
                mf.conv_tol = CONV_TOL
                mf.conv_tol_grad = CONV_TOL_GRAD
                e_tot = mf.kernel()
                assert mf.converged, (
                    f"{system}/{basis_name}/eps={eps}: PCM RHF not converged"
                )

                surf = ws.surface
                im = ws._intermediates
                K, R, S, D = im["K"], im["R"], im["S"], im["D"]
                v = im["v_grids"]
                q_sym = im["q_sym"]
                e_pcm = 0.5 * float(q_sym @ v)

                # Self-checks that the arrays written are the ones PySCF used.
                q_raw = numpy.linalg.solve(K, R @ v)
                q_t = R.T @ numpy.linalg.solve(K.T, v)
                assert numpy.allclose(0.5 * (q_raw + q_t), q_sym, atol=1e-13, rtol=0)
                assert abs(e_pcm - float(ws.e)) < 1e-12, (e_pcm, ws.e)
                area = surf["weights"] * surf["R_vdw"] ** 2 * surf["switch_fun"]
                assert numpy.allclose(area, surf["area"], atol=0, rtol=1e-14)

                this_cavity = {
                    "ng": int(surf["ng"]),
                    "n_tesserae": int(len(surf["area"])),
                    "atom_slices": [
                        [int(a), int(b)] for a, b in surf["gslice_by_atom"]
                    ],
                    "position": [_floats(p) for p in surf["grid_coords"]],
                    "normal": [_floats(n) for n in surf["norm_vec"]],
                    "area": _floats(surf["area"]),
                    "sphere_radius": _floats(surf["R_vdw"]),
                    "charge_exp": _floats(surf["charge_exp"]),
                    "switch_fun": _floats(surf["switch_fun"]),
                }
                if cavity is None:
                    cavity = this_cavity
                else:
                    assert cavity == this_cavity, "cavity must not depend on eps"

                solvents[repr(eps)] = {
                    "epsilon": eps,
                    "f_epsilon": float(im["f_epsilon"]),
                    "energy": float(e_tot),
                    "e_solv": float(e_tot - e_vac),
                    "e_pcm": e_pcm,
                    "v_grids": _floats(v),
                    "q_sym": _floats(q_sym),
                    "q_raw": _floats(q_raw),
                    "k_dot_v": _floats(K @ v),
                    "r_dot_v": _floats(R @ v),
                    "s_diag": _floats(numpy.diag(S)),
                    "d_diag": _floats(numpy.diag(D)),
                    "scf_cycles": int(getattr(mf, "cycles", -1)),
                }

            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": 0,
                "multiplicity": 1,
                "nao": mol.nao_nr(),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "vacuum_energy": float(e_vac),
                "cavity": cavity,
                "solvents": solvents,
                "provenance": common.provenance(
                    code="PySCF",
                    version=pyscf.__version__,
                    keywords={
                        "methods": ["scf.RHF", "scf.RHF(mol).PCM()"],
                        "pcm_method": "IEF-PCM",
                        "lebedev_order": LEBEDEV_ORDER,
                        "points_per_sphere": int(cavity["ng"]),
                        "vdw_scale": 1.2,
                        "radii": "pyscf.solvent.pcm.modified_Bondi (H = 1.1 A)",
                        "surface_discretization_method": "SWIG",
                        "probe": "Gaussian (int3c2e/int2c2e vs fakemol_for_charges(expnt=charge_exp**2))",
                        "epsilons": list(EPSILONS),
                        "conv_tol": CONV_TOL,
                        "conv_tol_grad": CONV_TOL_GRAD,
                        "eri": "exact 4-index (no density fitting)",
                        "numpy": numpy.__version__,
                        "pcm_module": str(Path(pcm.__file__).name),
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid=f"PCM cavity: Lebedev {cavity['ng']} per sphere, SWIG",
                    aux=None,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability=None,
                    generator="scripts/validation/gen_pcm.py",
                    extra={"basis_self_check": basis_check},
                ),
            }
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
            summary = " ".join(
                f"eps={s['epsilon']}: E={s['energy']:.10f} dE={s['e_solv']:+.8f} "
                f"e_pcm={s['e_pcm']:+.8f}"
                for s in solvents.values()
            )
            print(
                f"{system:6s} {basis_name:8s} n_tess={cavity['n_tesserae']} "
                f"E_vac={e_vac:.10f} {summary}"
            )
    print(f"GEN_PCM_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
