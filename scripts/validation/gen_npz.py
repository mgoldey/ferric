"""PySCF / numpy references for the VALIDATION.md row "NPZ export" (design §5.3).

Consumer: crates/ferric-cli/tests/validation_npz.rs, which runs the real CLI
(`method.kind = "pdep-rpa"`, `[rpa] export_npz`) on water / cc-pVDZ with exact
J/K, reads the bundle back with `numpy.load` (the FOREIGN reader), and compares
every array against what this script writes to
`testdata/reference/validation/npz/h2o_cc-pvdz.json`.

TWO KINDS OF REFERENCE PER FIELD
--------------------------------
* INTEGRAL LEVEL: PySCF AO integrals in FERRIC'S AO order (rinv and ip+ip^T at
  every nucleus, <u|r|v>, <u|r r|v>, <u|r^2|v>, S, S^1/2, the AO->atom map),
  which the Rust test contracts with the density / MOs the NPZ itself carries.
  That isolates the export path (was this array computed from the exported
  density, and written in the right layout?) from SCF convergence, at ~1e-12.
* FULL CHAIN: the same property from PySCF's own converged RHF (exact 4-index
  ERIs, conv_tol 1e-12), compared at the SCF-convergence bar.

AO order is ferric's, via `gen_properties.ferric_ao_permutation`; the overlap
in ferric order is stored and the Rust test asserts it equals ferric's own
first, so a wrong permutation fails before any property is compared. The aux
basis (cc-pvdz-ri) gets the same treatment for the PDEP eigenpotentials.

WHAT EACH REFERENCE IS
----------------------
* density_matrix, mo_coeff, mo_energy: PySCF RHF.
* esp_atoms / electric_field: `gen_properties.esp_and_field` (no self term;
  E = -grad V; the analytic field is FD-checked there before being written).
* dipole: -Tr(D <r>) + sum_A Z_A R_A, origin 0.
* density_second_moment: sum D <r_i r_j>, electronic only, origin 0.
* orbital_centers / orbital_spreads: <p|r|p>, sqrt(<p|r^2|p> - |<p|r|p>|^2).
* lowdin / mulliken charges: Z_A - sum_{u in A} (S^1/2 D S^1/2)_uu / (D S)_uu.
* alpha_tensor: direct-RPA (no exchange) static alpha with an RI Coulomb
  kernel over cc-pvdz-ri (the definition validation_static_alpha.rs pins),
  dense ov-space solve. Also the same alpha with def2-universal-jkfit as a
  wrong-aux NEGATIVE CONTROL.
* pdep_eigenvectors: the columns E of the generalized symmetric problem
  (V + Pi) E = V E Lambda, E^T V E = I, with V = (P|Q) and
  Pi = 4 sum_ia (P|ia)(ia|Q) / (e_a - e_i), sorted by descending lambda and
  truncated at lambda - 1 > TRUNC_THRESH (ferric's default trunc_thresh). The
  derivation: ferric diagonalises eps~ = I + V^-1/2 Pi V^-1/2 (eigenvectors U)
  and exports E = V^-1/2 U, and substituting U = V^1/2 E gives the generalized
  problem above. The kept SUBSPACE is compared (projector E E^T), which is
  invariant to column signs and to rotations inside a degenerate block. The
  script REFUSES to write if an eigenvalue sits within TRUNC_MARGIN of the cut
  (the kept count would then be a coin toss, not a reference).
* esp_surface / esp_points: ferric's documented construction
  (`ferric_scf::properties::esp_on_surface`): a Lebedev-110 sphere at
  1.4 x the Bondi radius (H 1.20, O 1.52 A; ferric-pcm's 1.8897259886 Bohr/A)
  about each atom, dropping points strictly inside another atom's scaled
  sphere; ESP there = sum_B Z_B/|r-R_B| - Tr(D rinv(r)) with PySCF's D. The
  Lebedev set is PySCF's own table, so the POINT SET is itself a check.
* TS free-atom constants (alpha_free, C6_free) for H and O from Tkatchenko &
  Scheffler, PRL 102, 073005 (2009) (Chu & Dalgarno values): the London
  frequency omega_A = (4/3) C6_free / alpha_free^2 is independent of the
  volume ratio, so a single-pole fit of the exported alpha_atomic_dynamic must
  recover it.
* NEGATIVE CONTROL: water / def2-SVP density and orbital energies (the same
  nao = 24, so a wrong-basis reference can only miss on VALUES).

Run (light, seconds):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_npz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402

GENERATOR = "scripts/validation/gen_npz.py"
ROW = "npz"
SYSTEM = "h2o"
BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"  # ferric-cli config::DEFAULT_CORRELATION_AUX, also set in the TOML
SWAP_AUX = "def2-universal-jkfit"  # negative control only
WRONG_BASIS = "def2-svp"  # negative control only

TRUNC_THRESH = 1e-4  # PdepRpaConfig trunc_thresh default (CLI default too)
TRUNC_MARGIN = 1e-6

# esp_on_surface defaults in the CLI: esp_surface_vdw_scale 1.4, n_angular 110
SURFACE_SCALE = 1.4
SURFACE_N_ANG = 110
BONDI_ANGSTROM = {1: 1.20, 8: 1.52}  # Bondi, J. Phys. Chem. 68, 441 (1964)
FERRIC_PCM_BOHR_PER_ANGSTROM = 1.8897259886  # crates/ferric-pcm/src/radii.rs

# (alpha_free, C6_free), a.u. — Tkatchenko & Scheffler PRL 102, 073005 (2009).
TS_FREE_ATOM = {1: (4.5, 6.5), 8: (5.4, 15.6)}


def _alpha_ri(mol, mf, aux_name, symbols, coords, want_pdep=False):
    """dRPA-RI static alpha (and optionally V, Pi in ferric aux order)."""
    import numpy as np
    import scipy.linalg
    from pyscf import df

    auxmol = gp._aux_mol(mol, aux_name, symbols, coords)
    aux_check = common.check_basis_like_for_like(auxmol, aux_name, symbols)
    nocc = mol.nelectron // 2
    c, eps = mf.mo_coeff, mf.mo_energy
    co, cv = c[:, :nocc], c[:, nocc:]
    nov = nocc * cv.shape[1]
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    mu = np.einsum("xuv,ui,va->xia", mol.intor("int1e_r"), co, cv).reshape(3, nov)
    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    iap = np.einsum("uvP,ui,va->iaP", int3c, co, cv).reshape(nov, -1)
    v2c = auxmol.intor("int2c2e")
    low = np.linalg.cholesky(v2c)
    b = scipy.linalg.solve_triangular(low, iap.T, lower=True)
    alpha = gp._alpha(mu, delta, 4.0 * (b.T @ b))
    out = {"alpha": alpha, "aux_check": aux_check, "naux": auxmol.nao_nr()}
    if want_pdep:
        pi = 4.0 * (iap.T / delta[None, :]) @ iap  # (naux, naux), PySCF aux order
        aperm = gp.ferric_ao_permutation(auxmol, aux_name, symbols)
        v_f = gp.to_ferric_order(v2c, aperm)
        pi_f = gp.to_ferric_order(0.5 * (pi + pi.T), aperm)
        lam, e = scipy.linalg.eigh(v_f + pi_f, v_f)
        order = np.argsort(-lam)
        lam, e = lam[order], e[:, order]
        margin = float(np.min(np.abs(lam - 1.0 - TRUNC_THRESH)))
        if margin < TRUNC_MARGIN:
            raise RuntimeError(
                f"a dielectric eigenvalue sits {margin:.1e} from the truncation cut; "
                "the kept count is not a reference"
            )
        n_keep = max(int(np.sum(lam - 1.0 > TRUNC_THRESH)), 1)
        # Fix each column's sign (largest |component| positive) for readability;
        # the Rust test compares the projector, which does not see signs.
        e_keep = e[:, :n_keep].copy()
        for j in range(n_keep):
            if e_keep[np.argmax(np.abs(e_keep[:, j])), j] < 0:
                e_keep[:, j] *= -1.0
        out.update(
            {
                "aux_permutation": aperm,
                "metric_ferric_order": v_f,
                "pi_ferric_order": pi_f,
                "lambda_all": lam,
                "n_keep": n_keep,
                "trunc_margin": margin,
                "eigvecs_kept": e_keep,
            }
        )
    return out


def _surface(zs, xyz):
    """ferric's esp_on_surface point set (Bohr), in ferric's generation order."""
    import numpy as np
    from pyscf.dft.LebedevGrid import MakeAngularGrid

    unit = np.asarray(MakeAngularGrid(SURFACE_N_ANG))[:, :3]
    radii = [
        SURFACE_SCALE * BONDI_ANGSTROM[z] * FERRIC_PCM_BOHR_PER_ANGSTROM for z in zs
    ]
    pts = []
    for a in range(len(zs)):
        for u in unit:
            p = xyz[a] + radii[a] * u
            buried = any(
                b != a and float(np.sum((p - xyz[b]) ** 2)) < radii[b] ** 2
                for b in range(len(zs))
            )
            if not buried:
                pts.append(p)
    return np.asarray(pts), radii


def _esp_at_points(mol, dm, pts):
    import numpy as np

    zs = [float(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    out = []
    for p in pts:
        with mol.with_rinv_origin(p):
            t = mol.intor("int1e_rinv")
        v = -float(np.einsum("ij,ij->", dm, t))
        v += sum(zs[b] / np.linalg.norm(p - xyz[b]) for b in range(len(zs)))
        out.append(v)
    return out


def b64mat(a):
    """A large matrix as base64 of its little-endian float64 bytes (row-major)
    plus its shape: bit-exact, and keeps the JSON under the repo's 500 KB
    large-file limit."""
    import base64

    import numpy as np

    a = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return {
        "shape": list(a.shape),
        "f64le_b64": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def main() -> int:
    import numpy as np
    import pyscf
    import scipy

    xyz_path = common.MOL_DIR / f"{SYSTEM}.xyz"
    symbols, coords = common.read_xyz(xyz_path)
    mol = common.build_pyscf_mol(xyz_path, BASIS)
    basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
    scf_info, mf = gp.run_rhf(mol)
    perm = gp.ferric_ao_permutation(mol, BASIS, symbols)
    p = np.asarray(perm)
    f = lambda m: gp.to_ferric_order(m, perm)  # noqa: E731
    mj = gp.mat_json

    nao = mol.nao_nr()
    nocc = mol.nelectron // 2
    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    dm = mf.make_rdm1()
    c_f = mf.mo_coeff[p, :]
    s = mol.intor("int1e_ovlp")
    w, u = np.linalg.eigh(s)
    s_half = (u * np.sqrt(w)) @ u.T
    ao_atom_pyscf = [lab[0] for lab in mol.ao_labels(fmt=False)]
    ao_atom = [int(ao_atom_pyscf[i]) for i in perm]

    # One-electron integrals in ferric order.
    r_ao = mol.intor("int1e_r")  # (3, nao, nao), origin 0
    rr_ao = mol.intor("int1e_rr").reshape(3, 3, nao, nao)
    r2_ao = mol.intor("int1e_r2")
    rinv_f, ipsym_f, esp_nuc, field_nuc = [], [], [], []
    for a in range(mol.natm):
        with mol.with_rinv_origin(xyz[a]):
            rinv = mol.intor("int1e_rinv")
            ip = mol.intor("int1e_iprinv", comp=3)
        rinv_f.append(mj(f(rinv)))
        ipsym_f.append([mj(f(ip[d] + ip[d].T)) for d in range(3)])
        esp_nuc.append(gp._nuc_esp([float(z) for z in zs], xyz, a))
        field_nuc.append(gp._nuc_field([float(z) for z in zs], xyz, a).tolist())

    # Full-chain property references from PySCF's density / MOs.
    ef = gp.esp_and_field(mol, dm)
    dip = -np.einsum("xij,ji->x", r_ao, dm) + np.asarray(zs, float) @ xyz
    m2 = np.einsum("xyij,ji->xy", rr_ao, dm)
    cen = np.einsum("xuv,ui,vi->ix", r_ao, mf.mo_coeff, mf.mo_coeff)
    r2 = np.einsum("uv,ui,vi->i", r2_ao, mf.mo_coeff, mf.mo_coeff)
    spread = np.sqrt(np.maximum(r2 - np.sum(cen**2, axis=1), 0.0))
    ds = dm @ s
    sds = s_half @ dm @ s_half
    mull = [
        float(zs[a] - sum(ds[i, i] for i in range(nao) if ao_atom_pyscf[i] == a))
        for a in range(mol.natm)
    ]
    lowd = [
        float(zs[a] - sum(sds[i, i] for i in range(nao) if ao_atom_pyscf[i] == a))
        for a in range(mol.natm)
    ]

    ri = _alpha_ri(mol, mf, AUX, symbols, coords, want_pdep=True)
    ri_swap = _alpha_ri(mol, mf, SWAP_AUX, symbols, coords)
    pts, radii = _surface(zs, xyz)
    esp_surf = _esp_at_points(mol, dm, pts)

    # Wrong-basis negative control (same geometry, same nao).
    mol_w = common.build_pyscf_mol(xyz_path, WRONG_BASIS)
    if mol_w.nao_nr() != nao:
        raise RuntimeError(
            f"{WRONG_BASIS} nao {mol_w.nao_nr()} != {nao}; control is a shape miss"
        )
    _, mf_w = gp.run_rhf(mol_w)
    perm_w = gp.ferric_ao_permutation(mol_w, WRONG_BASIS, symbols)

    ts = {
        str(z): {
            "alpha_free": a,
            "c6_free": c6,
            "omega_london": 4.0 / 3.0 * c6 / a**2,
        }
        for z, (a, c6) in TS_FREE_ATOM.items()
    }

    payload = {
        "row": "NPZ export",
        "system": SYSTEM,
        "basis": BASIS,
        "aux_basis": AUX,
        "charge": 0,
        "multiplicity": 1,
        "nao": nao,
        "naux": ri["naux"],
        "nocc": nocc,
        "nelectron": int(mol.nelectron),
        "atomic_numbers": zs,
        "coords_bohr": xyz.tolist(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "scf": scf_info,
        "ao_permutation_ferric_to_pyscf": perm,
        "ao_atom_ferric_order": ao_atom,
        # integral level (ferric AO order)
        "overlap_ferric_order": mj(f(s)),
        "overlap_sqrt_ferric_order": mj(f(s_half)),
        "dipole_ints_ferric_order": [mj(f(r_ao[x])) for x in range(3)],
        "second_moment_ints_ferric_order": [
            [mj(f(rr_ao[x, y])) for y in range(3)] for x in range(3)
        ],
        "r2_ints_ferric_order": mj(f(r2_ao)),
        "rinv_at_nuclei_ferric_order": rinv_f,
        "iprinv_sym_at_nuclei_ferric_order": ipsym_f,
        "esp_nuclear_part": esp_nuc,
        "field_nuclear_part": field_nuc,
        # full chain (PySCF RHF)
        "density_total_ferric_order": mj(f(dm)),
        "mo_coeff_ferric_order": mj(c_f),
        "mo_energy": [float(x) for x in mf.mo_energy],
        "esp_at_nuclei": ef["esp_at_nuclei"],
        "electric_field_at_nuclei": ef["electric_field_at_nuclei"],
        "field_vs_fd_of_esp_max_abs": ef["field_vs_fd_of_esp_max_abs"],
        "dipole": dip.tolist(),
        "density_second_moment": mj(m2),
        "orbital_centers": mj(cen),
        "orbital_spreads": spread.tolist(),
        "mulliken_charges": mull,
        "lowdin_charges": lowd,
        "alpha_drpa_ri": mj(ri["alpha"]),
        "alpha_drpa_ri_swap_aux": mj(ri_swap["alpha"]),
        "swap_aux_negative_control": SWAP_AUX,
        "pdep": {
            "aux_permutation_ferric_to_pyscf": ri["aux_permutation"],
            "metric_ferric_order": b64mat(ri["metric_ferric_order"]),
            "pi_ferric_order": b64mat(ri["pi_ferric_order"]),
            "lambda_descending": [float(x) for x in ri["lambda_all"]],
            "trunc_thresh": TRUNC_THRESH,
            "trunc_margin": ri["trunc_margin"],
            "n_keep": ri["n_keep"],
            "eigvecs_kept_ferric_order": b64mat(ri["eigvecs_kept"]),
            "definition": "(V + Pi) E = V E Lambda, E^T V E = I, Pi = 4 sum_ia (P|ia)(ia|Q)/(e_a-e_i), "
            "descending lambda, kept lambda - 1 > trunc_thresh",
        },
        "esp_surface": {
            "vdw_scale": SURFACE_SCALE,
            "n_angular": SURFACE_N_ANG,
            "bondi_angstrom": {str(k): v for k, v in BONDI_ANGSTROM.items()},
            "bohr_per_angstrom": FERRIC_PCM_BOHR_PER_ANGSTROM,
            "radii_bohr": radii,
            "points_bohr": mj(pts),
            "esp": esp_surf,
        },
        "ts_free_atom": ts,
        "wrong_basis_negative_control": {
            "basis": WRONG_BASIS,
            "density_total_ferric_order": mj(
                gp.to_ferric_order(mf_w.make_rdm1(), perm_w)
            ),
            "mo_energy": [float(x) for x in mf_w.mo_energy],
        },
        "provenance": common.provenance(
            code="PySCF + numpy + scipy",
            version=pyscf.__version__,
            keywords={
                "scf": 'scf.RHF, exact 4-index ERIs (matches the TOML\'s df_j_aux = df_k_aux = "exact")',
                "esp_field": "int1e_rinv / int1e_iprinv (ip + ip^T) at each nucleus, no self term",
                "alpha": "dRPA-RI, dense ov-space solve (gen_properties._alpha)",
                "pdep": "scipy.linalg.eigh(V + Pi, V)",
                "surface": "PySCF LebedevGrid.MakeAngularGrid(110), Bondi radii x 1.4",
                "ts": "Tkatchenko & Scheffler PRL 102, 073005 (2009) free-atom alpha, C6",
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            basis_name=BASIS,
            xyz_path=xyz_path,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=AUX,
            frozen_core=0,
            scf_conv={"conv_tol": gp.CONV_TOL, "conv_tol_grad": gp.CONV_TOL_GRAD},
            stability=scf_info["stability"],
            generator=GENERATOR,
            extra={"basis_self_check": basis_check, "aux_self_check": ri["aux_check"]},
        ),
    }
    path = common.write_reference(ROW, SYSTEM, BASIS, payload)
    # Rewrite without indentation: indented, this reference exceeds the
    # repo's 500 KB large-file limit (the large matrices are already base64).
    path.write_text(
        json.dumps(json.loads(path.read_text()), separators=(",", ":")) + "\n"
    )
    iso = lambda a: float(np.trace(a)) / 3.0  # noqa: E731
    print(
        f"E={scf_info['energy']:.10f} alpha_iso={iso(ri['alpha']):.8f} "
        f"(swap aux {iso(ri_swap['alpha']):.8f}) naux={ri['naux']} n_keep={ri['n_keep']} "
        f"margin={ri['trunc_margin']:.1e} surface_pts={len(pts)}"
    )
    print(f"GEN_NPZ_DONE wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
