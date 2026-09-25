"""PySCF references for the VALIDATION.md "COSMO" row (validation tier W2).

Consumer: crates/ferric-scf/tests/validation_cosmo.rs.
Output:   testdata/reference/validation/cosmo/<system>_<basis>.json

Systems (geometries in testdata/molecules/validation/):

    h2o      H2O          RHF   closed shell
    nh3      NH3          RHF   closed shell
    ch3oh    CH3OH        RHF   closed shell
    acetate  CH3COO(-)    RHF   anion (charge -1)
    ho2      HO2 2A''     UHF   doublet (cc-pVDZ, eps 78.4 only)

The open-shell case is HO2, not OH. OH (2Pi) was tried first and is not a
well-posed reference: the pi hole's orientation is a symmetry-degenerate
mode in vacuum, and the Lebedev cavity is not axially symmetric, so solvated
UHF drifts along an almost flat mode (|g| stalls at 2e-6 with DIIS; the
energy is still moving by 1e-7 Ha after 60 DIIS + 100 Newton cycles, PySCF
2.13.1). HO2's 2A'' state is non-degenerate.

x bases STO-3G and cc-pVDZ (ferric's bundled JSON via common.py) x
eps in {4.7, 78.4}.

# What each reference block is

ferric's COSMO (crates/ferric-scf/src/cosmo.rs) is PySCF's `pcm.py`
`method="COSMO"` in every piece EXCEPT the solute<->segment potential:

    piece                      ferric                         PySCF pcm.py
    cavity                     gen_surface SWIG port          gen_surface SWIG
    radii                      Bondi (H 1.20 A) x 1.17        modified_Bondi (H 1.10 A) x vdw_scale
    keep criterion             w_norm*swf > 1e-16             w_4pi*swf > 1e-16
    S matrix                   Gaussian-smeared (get_D_S)     Gaussian-smeared (get_D_S)
    f(eps)                     (eps-1)/(eps+1/2)              (eps-1)/(eps+1/2)
    v_k, V_reaction            POINT charge at s_k (1/r)      Gaussian charge exp xi_k^2 (erf(xi r)/r)
    energy                     E_scf(h) + 1/2 q.v             E_scf(h) + 1/2 q_sym.v

So five blocks are written per (system, basis, eps):

* `ferric_model` — THE like-for-like reference. PySCF's own SCF + PCM driver
  with the cavity, S, K, R built by PySCF's `gen_surface`/`get_D_S`, ferric's
  radii (Bondi x 1.17, H = 1.20 A), 110-point Lebedev spheres, ferric's keep
  criterion (w_4pi*swf > 4*pi*1e-16), and the potential/reaction field
  replaced by POINT-charge integrals (`fakemol_for_charges` at its default
  exponent 1e16, cross-checked here against `int1e_rinv`). This is an
  independent construction of exactly ferric's model; agreement to the SCF
  floor tests ferric's cavity, S matrix, solve, reaction field and energy.
* `ferric_model_h_radius_1p10` — `ferric_model` with ONLY the H radius set
  to PySCF's modified-Bondi 1.10 A: the pure radii effect, like-for-like
  (ferric's radii are not changed by this row; this is a control).
* `pyscf_cosmo_matched_radii` — stock PySCF COSMO (Gaussian-smeared V),
  ferric's radii, 110 points. Its gap to `ferric_model` is the point-vs-
  smeared-potential FORMULATION difference; reported, not asserted tight.
* `pyscf_cosmo_modified_bondi` — stock PySCF COSMO with PySCF's default
  radii table (modified Bondi, H = 1.10 A) x 1.17, 110 points. Its gap to
  `pyscf_cosmo_matched_radii` is the RADII difference (H only).
* `pyscf_cosmo_default` — `PCM(method="COSMO")` with every PySCF default
  (modified Bondi x 1.2, 302 points). Information only.

Plus `vacuum` (plain SCF) so solvation energies can be formed.

Run (light step; seconds to a minute in total):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_cosmo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "cosmo"
ROW_NAME = "COSMO"
BASES = ("sto-3g", "cc-pvdz")
EPSILONS = (4.7, 78.4)
# system -> (charge, multiplicity, method, restrict to (basis, eps) or None)
SYSTEMS = {
    "h2o": (0, 1, "rhf", None),
    "nh3": (0, 1, "rhf", None),
    "ch3oh": (0, 1, "rhf", None),
    "acetate": (-1, 1, "rhf", None),
    "ho2": (0, 2, "uhf", [("cc-pvdz", 78.4)]),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-8
RADIUS_SCALE = 1.17
LEBEDEV_POINTS = 110
LEBEDEV_ORDER = 17  # PySCF gen_grid.LEBEDEV_ORDER[17] == 110
FOUR_PI = 12.566370614359172

# ferric's Bondi table (crates/ferric-scf/src/cosmo.rs::bondi_radius_angstrom),
# Angstrom. Only elements used here need to be present; the generator asserts
# every element of every system is in this table.
FERRIC_BONDI_ANGSTROM = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52}


def ferric_radii_table(scale: float):
    """Radii array indexed by Z (Bohr), ferric's values for H/C/N/O, PySCF's
    modified Bondi elsewhere (never read for these systems)."""
    import numpy as np
    from pyscf.solvent import pcm

    table = np.array(pcm.modified_Bondi, dtype=float).copy()
    for z, r in FERRIC_BONDI_ANGSTROM.items():
        table[z] = r / common.BOHR_IN_ANGSTROM
    return table * scale


def point_pcm_class():
    """PySCF COSMO with ferric's point-charge potential and keep criterion."""
    import numpy as np
    from pyscf import df, gto
    from pyscf.solvent import pcm

    class FerricModelCOSMO(pcm.PCM):
        def build(self, ng=None):
            mol = self.mol
            ng = LEBEDEV_POINTS
            surf = pcm.gen_surface(
                mol, rad=self.radii_table, ng=ng, surface_discretization_method="SWIG"
            )
            # ferric keeps a point iff w_norm*swf > 1e-16 with w_norm summing
            # to 1 over the sphere; PySCF's w sums to 4*pi.
            keep = surf["weights"] * surf["switch_fun"] > FOUR_PI * 1e-16
            self.n_dropped_by_ferric_criterion = int((~keep).sum())
            for key in (
                "grid_coords",
                "weights",
                "charge_exp",
                "switch_fun",
                "R_vdw",
                "norm_vec",
                "area",
            ):
                surf[key] = surf[key][keep]
            self.surface = surf
            _, S = pcm.get_D_S(surf, with_S=True, with_D=False)
            eps = self.eps
            f_eps = (eps - 1.0) / (eps + 0.5)
            self._intermediates = {
                "S": S,
                "K": S,
                "R": -f_eps * np.eye(S.shape[0]),
                "f_epsilon": f_eps,
            }
            coords = surf["grid_coords"]
            rij = np.linalg.norm(
                coords[:, None, :] - mol.atom_coords(unit="B")[None, :, :], axis=-1
            )
            self.v_grids_n = (mol.atom_charges()[None, :] / rij).sum(axis=1)

        def _point_v3c(self):
            fakemol = gto.fakemol_for_charges(self.surface["grid_coords"])
            fakemol.cart = self.mol.cart
            return df.incore.aux_e2(self.mol, fakemol, intor="int3c2e", aosym="s1")

        def _get_v(self, dms):
            v_nj = self._point_v3c()
            return np.einsum("ijL,xij->xL", v_nj, dms)

        def _get_vmat(self, q):
            v_nj = self._point_v3c()
            q = q.reshape(-1, v_nj.shape[-1])
            return -np.einsum("ijL,xL->xij", v_nj, q)

    return FerricModelCOSMO


def check_point_integrals(mol, coords) -> float:
    """max |fakemol(1e16) int3c2e - int1e_rinv| over a few surface points."""
    import numpy as np
    from pyscf import df, gto

    fakemol = gto.fakemol_for_charges(coords)
    fakemol.cart = mol.cart
    v = df.incore.aux_e2(mol, fakemol, intor="int3c2e", aosym="s1")
    worst = 0.0
    for k in range(0, coords.shape[0], max(1, coords.shape[0] // 7)):
        with mol.with_rinv_origin(coords[k]):
            ref = mol.intor("int1e_rinv")
        worst = max(worst, float(np.max(np.abs(v[:, :, k] - ref))))
    return worst


def run_scf(mol, method, solvent=None, dm0=None):
    from pyscf import scf

    mf = scf.RHF(mol) if method == "rhf" else scf.UHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    if solvent is not None:
        mf = mf.PCM(solvent)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError(
            f"SCF did not converge ({method}, solvent={solvent is not None})"
        )
    stab = None
    if method == "uhf":
        for _ in range(5):
            mo_new, _, stable, _ = mf.stability(return_status=True)
            if stable:
                break
            dm = mf.make_rdm1(mo_new, mf.mo_occ)
            mf.kernel(dm0=dm)
        else:
            raise RuntimeError("UHF did not reach an internally stable state")
        stab = {"internal_stable": True, "s_squared": float(mf.spin_square()[0])}
    return mf, stab


def solvent_block(mf, stab, extra=None) -> dict:
    import numpy as np

    ws = mf.with_solvent
    q = ws._intermediates["q"]
    out = {
        "energy": float(mf.e_tot),
        "e_solvent": float(mf.scf_summary["e_solvent"]),
        "n_segments": int(ws.surface["grid_coords"].shape[0]),
        "total_area_bohr2": float(np.sum(ws.surface["area"])),
        "sum_q": float(np.sum(q)),
        "f_epsilon": float(ws._intermediates["f_epsilon"]),
        "converged": bool(mf.converged),
        "stability": stab,
    }
    if extra:
        out.update(extra)
    return out


def main() -> int:
    import numpy as np
    import pyscf
    from pyscf.solvent import pcm

    only = set(sys.argv[1:])
    FerricModelCOSMO = point_pcm_class()
    written = []
    for system, (charge, mult, method, restrict) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for z in {common.z_of(s) for s in symbols}:
            assert z in FERRIC_BONDI_ANGSTROM, (
                f"{system}: Z={z} not in the ferric radius copy"
            )
        for basis_name in BASES:
            eps_here = [
                e for e in EPSILONS if restrict is None or (basis_name, e) in restrict
            ]
            if not eps_here:
                continue
            mol = common.build_pyscf_mol(
                xyz, basis_name, charge=charge, multiplicity=mult
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            vac, vac_stab = run_scf(mol, method)
            per_eps = {}
            point_check = None
            for eps in eps_here:
                # (1) ferric's model, like-for-like.
                fm = FerricModelCOSMO(mol)
                fm.method = "COSMO"
                fm.eps = eps
                fm.radii_table = ferric_radii_table(RADIUS_SCALE)
                fm.lebedev_order = LEBEDEV_ORDER
                mf_fm, st_fm = run_scf(mol, method, fm, vac.make_rdm1())
                if point_check is None:
                    point_check = check_point_integrals(mol, fm.surface["grid_coords"])
                ferric_model = solvent_block(
                    mf_fm,
                    st_fm,
                    {
                        "n_dropped_by_ferric_keep_criterion": fm.n_dropped_by_ferric_criterion
                    },
                )

                # (1b) ferric's model with ONLY the H radius changed to PySCF's
                # modified Bondi 1.10 A: the pure radii effect, like-for-like.
                fmb = FerricModelCOSMO(mol)
                fmb.method = "COSMO"
                fmb.eps = eps
                tbl = ferric_radii_table(RADIUS_SCALE)
                tbl[1] = pcm.modified_Bondi[1] * RADIUS_SCALE
                fmb.radii_table = tbl
                fmb.lebedev_order = LEBEDEV_ORDER
                mf_fmb, st_fmb = run_scf(mol, method, fmb, vac.make_rdm1())
                ferric_model_mb = solvent_block(mf_fmb, st_fmb)

                # (2) stock PySCF COSMO, ferric radii.
                sm = pcm.PCM(mol)
                sm.method = "COSMO"
                sm.eps = eps
                sm.radii_table = ferric_radii_table(RADIUS_SCALE)
                sm.lebedev_order = LEBEDEV_ORDER
                mf_sm, st_sm = run_scf(mol, method, sm, vac.make_rdm1())
                matched = solvent_block(mf_sm, st_sm)

                # (3) stock PySCF COSMO, PySCF modified-Bondi radii x 1.17.
                mb = pcm.PCM(mol)
                mb.method = "COSMO"
                mb.eps = eps
                mb.vdw_scale = RADIUS_SCALE
                mb.lebedev_order = LEBEDEV_ORDER
                mf_mb, st_mb = run_scf(mol, method, mb, vac.make_rdm1())
                modbondi = solvent_block(mf_mb, st_mb)

                # (4) all PySCF defaults (modified Bondi x 1.2, 302 points).
                de = pcm.PCM(mol)
                de.method = "COSMO"
                de.eps = eps
                mf_de, st_de = run_scf(mol, method, de, vac.make_rdm1())
                default = solvent_block(
                    mf_de,
                    st_de,
                    {"vdw_scale": de.vdw_scale, "lebedev_order": de.lebedev_order},
                )

                key = f"eps_{eps}"
                per_eps[key] = {
                    "epsilon": eps,
                    "ferric_model": ferric_model,
                    "ferric_model_h_radius_1p10": ferric_model_mb,
                    "pyscf_cosmo_matched_radii": matched,
                    "pyscf_cosmo_modified_bondi": modbondi,
                    "pyscf_cosmo_default": default,
                }
                h2k = 627.5094740631
                print(
                    f"{system:8s} {basis_name:8s} eps={eps:5.1f} "
                    f"nseg={ferric_model['n_segments']:4d} "
                    f"dG[kcal] ferric_model={(ferric_model['energy'] - vac.e_tot) * h2k:9.4f} "
                    f"pyscf_matched={(matched['energy'] - vac.e_tot) * h2k:9.4f} "
                    f"fm_H1.10={(ferric_model_mb['energy'] - vac.e_tot) * h2k:9.4f} "
                    f"modBondi={(modbondi['energy'] - vac.e_tot) * h2k:9.4f} "
                    f"default={(default['energy'] - vac.e_tot) * h2k:9.4f} "
                    f"sum_q={ferric_model['sum_q']:+.5f}"
                )

            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "charge": charge,
                "multiplicity": mult,
                "method": method,
                "nao": mol.nao_nr(),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "radius_scale": RADIUS_SCALE,
                "lebedev_points": LEBEDEV_POINTS,
                "radii_angstrom_ferric": {
                    str(z): r for z, r in FERRIC_BONDI_ANGSTROM.items()
                },
                "radii_angstrom_pyscf_modified_bondi": {
                    str(z): float(pcm.modified_Bondi[z] * common.BOHR_IN_ANGSTROM)
                    for z in FERRIC_BONDI_ANGSTROM
                },
                "point_charge_integral_check_max_abs": point_check,
                "vacuum": {"energy": float(vac.e_tot), "stability": vac_stab},
                "solvated": per_eps,
                "provenance": common.provenance(
                    code="PySCF",
                    version=pyscf.__version__,
                    keywords={
                        "method": f"scf.{method.upper()} + solvent.pcm.PCM(method='COSMO')",
                        "ferric_model": "FerricModelCOSMO: gen_surface(SWIG) with ferric radii, "
                        "keep w*swf > 4*pi*1e-16, get_D_S S, K=S, R=-f(eps) I, POINT-charge "
                        "v and V_reaction (fakemol_for_charges expnt 1e16)",
                        "f_epsilon": "(eps-1)/(eps+0.5)",
                        "eri": "exact 4-index (no density fitting)",
                        "numpy": np.__version__,
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid=f"Lebedev {LEBEDEV_POINTS} per sphere (cavity only)",
                    aux=None,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability={"method": method, "vacuum": vac_stab},
                    generator="scripts/validation/gen_cosmo.py",
                    extra={"basis_self_check": basis_check},
                ),
            }
            written.append(common.write_reference(ROW, system, basis_name, payload))
    print(f"GEN_COSMO_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
