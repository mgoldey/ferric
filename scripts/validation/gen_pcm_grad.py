"""PySCF references for the IEF-PCM analytic nuclear gradient.

Consumer: crates/ferric-scf/tests/pcm_gradient.rs
          (`rhf_pcm_gradient_matches_pyscf`).
Output:   testdata/reference/validation/pcm/grad_<system>_sto-3g.json

PySCF 2.13 `scf.RHF(mol).PCM()` with method "IEF-PCM", eps 78.4,
lebedev_order 29 (302 points per sphere), vdw_scale 1.2 on PySCF's
`modified_Bondi` radii, SWIG discretization, Gaussian probes; gradient from
`mf.nuc_grad_method()` (pyscf.solvent.grad.pcm). With
`PcmConfig { lebedev_order: 302, probe: GaussianSmeared, .. }` ferric builds
the same cavity from the molecule; the file records the cavity's tessera
count, total area and area-weighted centroid so the Rust test can assert that
before it compares gradients.

Systems: water and methanol (testdata/molecules/{water,ch3oh}.xyz), STO-3G.

Run (light step; seconds per system):
    flock -o /tmp/ferric-compile.lock \
        uv run --no-sync python scripts/validation/gen_pcm_grad.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "pcm"
SYSTEMS = ("water", "ch3oh")
BASIS = "sto-3g"
EPS = 78.4
LEBEDEV_ORDER = 29  # 302 points per sphere
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9


def main() -> int:
    import numpy
    import pyscf
    from pyscf import scf

    for system in SYSTEMS:
        xyz = common.ROOT / "testdata" / "molecules" / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, BASIS)
        basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)

        mf = scf.RHF(mol).PCM()
        ws = mf.with_solvent
        ws.method = "IEF-PCM"
        ws.eps = EPS
        ws.lebedev_order = LEBEDEV_ORDER
        ws.vdw_scale = 1.2
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        e_tot = mf.kernel()
        assert mf.converged, f"{system}: PCM RHF not converged"
        grad = mf.nuc_grad_method().kernel()

        surf = ws.surface
        area = numpy.asarray(surf["area"])
        pts = numpy.asarray(surf["grid_coords"])
        payload = {
            "row": "IEF-PCM gradient",
            "system": system,
            "basis": BASIS,
            "epsilon": EPS,
            "energy": float(e_tot),
            "e_pcm": float(ws.e),
            "gradient": [[float(x) for x in row] for row in grad],
            "cavity": {
                "ng": int(surf["ng"]),
                "n_tesserae": int(len(area)),
                "total_area": float(area.sum()),
                "area_centroid": [float(x) for x in (area @ pts) / area.sum()],
            },
            "provenance": common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "methods": ["scf.RHF(mol).PCM()", "nuc_grad_method()"],
                    "pcm_method": "IEF-PCM",
                    "lebedev_order": LEBEDEV_ORDER,
                    "vdw_scale": 1.2,
                    "radii": "pyscf.solvent.pcm.modified_Bondi (H = 1.1 A)",
                    "surface_discretization_method": "SWIG",
                    "conv_tol": CONV_TOL,
                    "conv_tol_grad": CONV_TOL_GRAD,
                    "eri": "exact 4-index (no density fitting)",
                    "numpy": numpy.__version__,
                },
                basis_name=BASIS,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=f"PCM cavity: Lebedev {int(surf['ng'])} per sphere, SWIG",
                aux=None,
                frozen_core=None,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/validation/gen_pcm_grad.py",
                extra={"basis_self_check": basis_check},
            ),
        }
        path = common.write_reference(ROW, f"grad_{system}", BASIS, payload)
        print(f"{system}: E={e_tot:.12f} n_tess={len(area)} -> {path}")
        print(numpy.array2string(grad, precision=10))
    print("GEN_PCM_GRAD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
