"""PySCF / numpy references for the VALIDATION.md "PDEP-RPA C6" row.

Consumer: crates/ferric-rpa/tests/validation_pdep_c6.rs
Output:   testdata/reference/validation/pdep_c6/<system>_<basis>.json
Systems:  H2O and N2, aug-cc-pVDZ, RI aux aug-cc-pvdz-rifit.

WHAT FERRIC COMPUTES (read from the loops, not the doc comments)
----------------------------------------------------------------
`ferric_rpa::dispersion::pdep_dynamic_polarizability(.., Becke, None)` returns

  freqs, weights   `quadrature::build_quadrature(&cfg.quadrature)`. Default
                   config = MiniMax, n = 20, which is the Gauss-Legendre map
                   w = u0 (1+x)/(1-x) with u0 = optimized_u0(20) = 0.5 and
                   weight w_GL * 2 u0 / (1-x)^2. Reproduced here with
                   numpy.polynomial.legendre.leggauss (an independent GL
                   routine); the Rust test asserts ferric's nodes equal these.
  molecular[k]     `properties::molecular_dynamic_polarizability`: closed-shell
                   DIRECT RPA (time-dependent Hartree, no exchange kernel),
                   RI Coulomb kernel, frozen_core = 0 (hard-coded):
                       alpha(iw) = 4 mu^T G - 16 (B G mu)^T eps~^-1 (B G mu),
                       G = diag(D/(w^2+D^2)), eps~ = I + 4 B G B^T,
                   which by Sherman-Morrison-Woodbury is
                       alpha(iw) = 4 mu^T [ diag((w^2+D^2)/D) + 4 K ]^-1 mu,
                       K = (ia|P) V^-1 (P|jb),  mu_ia = <i|r|a> (origin 0).
                   Full rank: this path never touches the PDEP eigensolver or
                   trunc_thresh (the name "PDEP" is historical here).
  per_atom[a][k]   `properties::pdep_polarizability_becke_dynamic`: on ferric's
                   flat (75, 110) TA-M4 x Lebedev Becke grid,
                       m^A_d = sum_{g: home=A} w_g (r_g - R_A)_d chi chi
                   (atom-centred, Becke weight of the home atom folded into
                   w_g, no renormalization), and
                       alpha^A_dj(iw) = 4 (m^A_d)^T R(iw) M_j,  symmetrised in (d,j),
                       M = sum_A m^A   (the SUM of atom-centred dipoles, NOT the
                                        analytic lab-frame dipole),
                       R(iw) = [diag((w^2+D^2)/D) + 4K]^-1.
                   So sum_A alpha^A = 4 M^T R M exactly (linearity), and it
                   differs from `molecular` by the charge-transfer pieces of
                   r = sum_A w_A (r - R_A) + sum_A w_A R_A.
  C6               `casimir_polder_c6`: c6_molecular_iso =
                   (3/pi) sum_k w_k abar(iw_k)^2 with abar = tr(alpha)/3 of
                   `molecular`; c6_iso_pair / c6_aniso_pair from `per_atom`
                   (aniso is ELEMENTWISE: (3/pi) sum_k w_k a^A_ij a^B_ij).

The STATIC per-atom path `pdep_polarizability_becke` uses the atom-centred m^A
on the left but the ANALYTIC lab-frame dipole on the right. Its doc claims it
matches the dynamic path's w = 0 limit exactly; by the two loops it cannot
(the right-hand operators differ by sum_A R_A q^A). This file stores BOTH
definitions at w = 0 so the test can measure the gap instead of assuming it.

THE REFERENCE (independent routes)
----------------------------------
PySCF RHF (exact integrals, ferric's own basis JSON, geometry in Bohr) gives
C, eps. The same orbitals are stored in ferric's AO order for injection.

  * molecular alpha(iw): SUM OVER STATES from one Casida diagonalisation,
        Omega^2 Z = D^1/2 (D + 4K) D^1/2 Z,
        alpha_xy(iw) = sum_n T_nx T_ny / (Omega_n^2 + w^2),  T_n = 2 mu^T D^1/2 Z_n,
    a different algebraic route from ferric's per-node SMW solve. The script
    ALSO does the dense per-node ov-space solve and REFUSES to write if the
    two disagree by more than SELF_CHECK_REL.
  * C6: the quadrature value on ferric's nodes (what ferric must reproduce)
    AND the exact Casimir-Polder integral of the SOS form,
        C6 = (3/2) sum_mn s_m s_n / (Omega_m Omega_n (Omega_m + Omega_n)),
        s_n = |T_n|^2 / 3,
    so the test can report ferric's quadrature error (measurement only).
  * per-atom: m^A rebuilt on ferric's grid from PySCF primitives
    (gen_properties.atom_centred_grid / becke_partition, the same replica the
    Becke-volume row matched to 7e-16), then the two per-atom definitions
    above. Also the grid lab-frame dipole M_lab = sum_A (m^A + R_A q^A),
    q^A = sum_{g: home=A} w_g chi chi, to show where sum_A alpha^A and the
    molecular tensor part ways (charge transfer) and that the grid lab-frame
    dipole reproduces the analytic one (grid error, recorded).

SCOPE values (not what ferric computes; recorded so the test can assert ferric
MISSES them or just report them):
  * exact-ERI dRPA molecular C6 (the RI error on C6),
  * TDHF (exchange kernel, exact ERI) molecular C6:
        alpha(iw) = 4 mu^T [ (A+B) + w^2 (A-B)^-1 ]^-1 mu,
        A+B = D + 4(ia|jb) - (ib|ja) - (ij|ab),  A-B = D + (ib|ja) - (ij|ab).

Run (light, < 1 minute):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_pdep_c6.py
    (a trailing system name restricts, e.g. `... gen_pdep_c6.py n2`)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402

GENERATOR = "scripts/validation/gen_pdep_c6.py"
ROW = "pdep_c6"

# (system, orbital basis, RI aux basis)
CASES = (
    ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
    ("n2", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
)
SWAP_AUX = "def2-universal-jkfit"  # negative control the Rust test swaps in

# ferric PdepRpaConfig::default().quadrature: MiniMax, n = 20 -> GL map, u0 = 0.5
N_QUAD = 20
U0 = 0.5

SELF_CHECK_REL = 1e-10  # SOS vs dense per-node solve; refuse to write above


def gl_nodes(n: int, u0: float):
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(n)  # ascending, like ferric
    freqs = u0 * (1.0 + x) / (1.0 - x)
    weights = w * 2.0 * u0 / (1.0 - x) ** 2
    return freqs, weights


def sym(a):
    return 0.5 * (a + a.T)


def rel(a, b) -> float:
    import numpy as np

    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def solve_alpha(left, right, delta, k4, omega):
    """4 left^T [diag((w^2+D^2)/D) + k4]^-1 right, (3, nov) operands."""
    import numpy as np

    m = np.diag((omega * omega + delta * delta) / delta) + k4
    return 4.0 * left @ np.linalg.solve(m, right.T)


def tdhf_alpha(mu, apb, amb, omega):
    import numpy as np

    m = apb + omega * omega * np.linalg.inv(amb)
    return sym(4.0 * mu @ np.linalg.solve(m, mu.T))


def c6_quad(tensors, weights):
    """(iso, elementwise 3x3) Casimir-Polder sums on a node list."""
    import numpy as np

    t = np.asarray(tensors)
    iso = np.trace(t, axis1=1, axis2=2) / 3.0
    c_iso = 3.0 / np.pi * float(np.sum(weights * iso * iso))
    c_ten = 3.0 / np.pi * np.einsum("k,kij,kij->ij", weights, t, t)
    return c_iso, c_ten


def c6_pair(per_atom, weights):
    """ferric's casimir_polder_c6 pair reduction, on (natm, nfreq, 3, 3)."""
    import numpy as np

    p = np.asarray(per_atom)
    iso = np.trace(p, axis1=2, axis2=3) / 3.0  # (natm, nfreq)
    c_iso = 3.0 / np.pi * np.einsum("k,ak,bk->ab", weights, iso, iso)
    c_ten = 3.0 / np.pi * np.einsum("k,akij,bkij->abij", weights, p, p)
    return c_iso, c_ten


def tensors_json(ts) -> list:
    return [gp.mat_json(t) for t in ts]


def grid_moments(mol, co, cv):
    """Per-atom (m^A_d, q^A) in the occ-vir MO basis on ferric's grid."""
    import numpy as np
    from pyscf.dft import numint

    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    xyz = np.asarray(mol.atom_coords(unit="Bohr"))
    home, pts, w_rl = gp.atom_centred_grid(zs, xyz, *gp.FERRIC_GRID)
    wb = gp.becke_partition(zs, xyz, pts)[home, np.arange(len(pts))]
    w = w_rl * wb
    ao = numint.eval_ao(mol, pts, deriv=0)  # (npts, nao)
    phi_o = ao @ co
    phi_v = ao @ cv
    natm = mol.natm
    nov = co.shape[1] * cv.shape[1]
    m_at = np.zeros((natm, 3, nov))
    q_at = np.zeros((natm, nov))
    for a in range(natm):
        sel = home == a
        po, pv, ww = phi_o[sel], phi_v[sel], w[sel]
        disp = pts[sel] - xyz[a]
        q_at[a] = np.einsum("g,gi,ga->ia", ww, po, pv).ravel()
        for d in range(3):
            m_at[a, d] = np.einsum("g,gi,ga->ia", ww * disp[:, d], po, pv).ravel()
    return m_at, q_at, xyz, len(pts)


def gen_case(system: str, basis_name: str, aux_name: str) -> Path:
    import numpy as np
    import pyscf
    import scipy.linalg
    from pyscf import ao2mo, df

    xyz_path = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz_path)
    mol = common.build_pyscf_mol(xyz_path, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    scf_info, mf = gp.run_rhf(mol)
    auxmol = gp._aux_mol(mol, aux_name, symbols, coords)
    aux_check = common.check_basis_like_for_like(auxmol, aux_name, symbols)

    nocc = mol.nelectron // 2
    c = mf.mo_coeff
    eps = mf.mo_energy
    co, cv = c[:, :nocc], c[:, nocc:]
    nvir = cv.shape[1]
    nov = nocc * nvir
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    mu_ao = mol.intor("int1e_r")
    mu = np.einsum("xuv,ui,va->xia", mu_ao, co, cv).reshape(3, nov)

    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    iap = np.einsum("uvP,ui,va->iaP", int3c, co, cv).reshape(nov, -1)
    low = np.linalg.cholesky(auxmol.intor("int2c2e"))
    b = scipy.linalg.solve_triangular(low, iap.T, lower=True)
    k4 = 4.0 * (b.T @ b)

    freqs, weights = gl_nodes(N_QUAD, U0)

    # --- molecular alpha(iw): SOS from one Casida diagonalisation.
    sd = np.sqrt(delta)
    casida = sd[:, None] * (np.diag(delta) + k4) * sd[None, :]
    om2, z = np.linalg.eigh(casida)
    if om2.min() <= 0.0:
        raise RuntimeError(
            f"{system}: non-positive RPA excitation energy^2 {om2.min()}"
        )
    omega_n = np.sqrt(om2)
    t_n = 2.0 * (mu * sd[None, :]) @ z  # (3, nstates)

    def alpha_sos(w):
        return sym((t_n / (om2 + w * w)[None, :]) @ t_n.T)

    alpha_mol = [alpha_sos(w) for w in freqs]
    alpha_static = alpha_sos(0.0)
    worst = 0.0
    for w, a_sos in zip([0.0, *freqs], [alpha_static, *alpha_mol]):
        worst = max(worst, rel(sym(solve_alpha(mu, mu, delta, k4, w)), a_sos))
    if worst > SELF_CHECK_REL:
        raise RuntimeError(
            f"{system}: SOS vs dense alpha {worst:.2e} > {SELF_CHECK_REL}"
        )

    c6_iso, c6_ten = c6_quad(alpha_mol, weights)
    s_n = np.sum(t_n * t_n, axis=0) / 3.0
    c6_iso_exact = 1.5 * float(
        np.einsum(
            "m,n,mn->",
            s_n / omega_n,
            s_n / omega_n,
            1.0 / (omega_n[:, None] + omega_n[None, :]),
        )
    )

    # --- per-atom (Becke grid replica).
    m_at, q_at, xyz, npts = grid_moments(mol, co, cv)
    natm = mol.natm
    m_sum = m_at.sum(axis=0)  # (3, nov): M = sum_A m^A
    m_lab_grid = m_sum + np.einsum("ad,ak->dk", xyz, q_at)
    lab_grid_vs_analytic = float(np.max(np.abs(m_lab_grid - mu)) / np.max(np.abs(mu)))

    def per_atom_at(w, right):
        return [sym(solve_alpha(m_at[a], right, delta, k4, w)) for a in range(natm)]

    per_atom = np.array([per_atom_at(w, m_sum) for w in freqs])  # (nfreq, natm, 3, 3)
    per_atom = per_atom.transpose(1, 0, 2, 3)  # (natm, nfreq, 3, 3), ferric layout
    per_atom_dyn_w0 = per_atom_at(0.0, m_sum)
    per_atom_static_lab = per_atom_at(0.0, mu)
    ac_sum = per_atom.sum(axis=0)  # (nfreq, 3, 3)
    ac_sum_direct = [sym(solve_alpha(m_sum, m_sum, delta, k4, w)) for w in freqs]
    lin = max(rel(a, b_) for a, b_ in zip(ac_sum, ac_sum_direct))
    if lin > SELF_CHECK_REL:
        raise RuntimeError(f"{system}: sum_A alpha^A != 4 M^T R M ({lin:.2e})")
    alpha_lab_grid = [
        sym(solve_alpha(m_lab_grid, m_lab_grid, delta, k4, w)) for w in freqs
    ]
    c6_pair_iso, c6_pair_ten = c6_pair(per_atom, weights)

    # --- scope: exact-ERI dRPA and TDHF C6.
    eri = ao2mo.kernel(mol, c, compact=False).reshape([c.shape[1]] * 4)
    ovov = eri[:nocc, nocc:, :nocc, nocc:].reshape(nov, nov)
    ibja = eri[:nocc, nocc:, :nocc, nocc:].transpose(0, 3, 2, 1).reshape(nov, nov)
    ijab = eri[:nocc, :nocc, nocc:, nocc:].transpose(0, 2, 1, 3).reshape(nov, nov)
    alpha_exact = [sym(solve_alpha(mu, mu, delta, 4.0 * ovov, w)) for w in freqs]
    apb = np.diag(delta) + 4.0 * ovov - ibja - ijab
    amb = np.diag(delta) + ibja - ijab
    alpha_tdhf = [tdhf_alpha(mu, apb, amb, w) for w in freqs]
    c6_iso_exact_eri, _ = c6_quad(alpha_exact, weights)
    c6_iso_tdhf, _ = c6_quad(alpha_tdhf, weights)

    perm = gp.ferric_ao_permutation(mol, basis_name, symbols)
    iso = lambda a: float(np.trace(a)) / 3.0  # noqa: E731
    payload = {
        "row": "PDEP-RPA C6",
        "system": system,
        "basis": basis_name,
        "aux_basis": aux_name,
        "swap_aux_negative_control": SWAP_AUX,
        "charge": 0,
        "multiplicity": 1,
        "nao": mol.nao_nr(),
        "naux": auxmol.nao_nr(),
        "nocc": nocc,
        "natoms": natm,
        "nuclear_repulsion": float(mol.energy_nuc()),
        "scf": scf_info,
        "ao_permutation_ferric_to_pyscf": perm,
        "overlap_ferric_order": gp.mat_json(
            gp.to_ferric_order(mol.intor("int1e_ovlp"), perm)
        ),
        "mo_coeff_ferric_order": gp.mat_json(c[np.asarray(perm), :]),
        "mo_energy": [float(x) for x in eps],
        "quadrature": {
            "scheme": "Gauss-Legendre (numpy leggauss), w = u0 (1+x)/(1-x), "
            "weight = w_GL 2 u0/(1-x)^2 == ferric MiniMax n=20 (optimized_u0 = 0.5)",
            "n_points": N_QUAD,
            "u0": U0,
            "freqs": [float(x) for x in freqs],
            "weights": [float(x) for x in weights],
        },
        "molecular": {
            "alpha_static": gp.mat_json(alpha_static),
            "alpha_iw": tensors_json(alpha_mol),
            "c6_iso": c6_iso,
            "c6_tensor": gp.mat_json(c6_ten),
            "c6_iso_exact_integral": c6_iso_exact,
            "quadrature_error_rel": (c6_iso - c6_iso_exact) / c6_iso_exact,
            "sos_vs_dense_max_rel": worst,
            "n_states": int(len(omega_n)),
            "lowest_excitation": float(omega_n[0]),
        },
        "per_atom_becke": {
            "grid": {
                "n_radial": gp.FERRIC_GRID[0],
                "n_angular": gp.FERRIC_GRID[1],
                "npts": int(npts),
                "radial": "Treutler-Ahlrichs M4",
                "partition": "Becke 1988 + ferric becke.rs Bragg-Slater size adjustment",
            },
            "alpha_iw": [tensors_json(per_atom[a]) for a in range(natm)],
            "alpha_dynamic_w0": tensors_json(per_atom_dyn_w0),
            "alpha_static_lab_right": tensors_json(per_atom_static_lab),
            "sum_over_atoms_iw": tensors_json(ac_sum),
            "alpha_lab_grid_iw": tensors_json(alpha_lab_grid),
            "lab_grid_dipole_vs_analytic_max_rel": lab_grid_vs_analytic,
            "c6_iso_pair": gp.mat_json(c6_pair_iso),
            "c6_aniso_pair": [
                [gp.mat_json(c6_pair_ten[a, b_]) for b_ in range(natm)]
                for a in range(natm)
            ],
            "definition": "alpha^A_dj(iw) = sym 4 (m^A_d)^T R(iw) M_j, "
            "m^A = sum_{home=A} w (r-R_A) chi chi, M = sum_A m^A; "
            "alpha_static_lab_right uses the analytic <i|r|a> as the right operand "
            "(pdep_polarizability_becke)",
        },
        "scope": {
            "c6_iso_drpa_exact_eri": c6_iso_exact_eri,
            "c6_iso_tdhf_exact_eri": c6_iso_tdhf,
            "alpha_iw_tdhf_exact_eri": tensors_json(alpha_tdhf),
        },
        "definition": "direct RPA (TDH, no exchange), RI Coulomb kernel K=(ia|P)V^-1(P|jb), "
        "frozen_core 0, alpha(iw) = 4 mu^T [diag((w^2+D^2)/D) + 4K]^-1 mu, mu = <i|r|a>; "
        "C6 = (3/pi) sum_k w_k abar(iw_k)^2",
        "provenance": common.provenance(
            code="PySCF + numpy",
            version=pyscf.__version__,
            keywords={
                "scf": "scf.RHF, exact 4-index",
                "alpha_molecular": "sum over states from one Casida eigh "
                "(D^1/2 (D+4K) D^1/2), cross-checked against a dense per-node solve",
                "ri": "df.incore.aux_e2 int3c2e + Cholesky of int2c2e (Coulomb metric)",
                "per_atom_grid": "gen_properties.atom_centred_grid + becke_partition "
                "(ferric (75,110) flat grid replica)",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz_path,
            coords_bohr=coords,
            symbols=symbols,
            grid={
                "per_atom": list(gp.FERRIC_GRID),
                "molecular": "none (analytic dipole)",
            },
            aux=aux_name,
            frozen_core=0,
            scf_conv={"conv_tol": gp.CONV_TOL, "conv_tol_grad": gp.CONV_TOL_GRAD},
            stability=scf_info["stability"],
            generator=GENERATOR,
            extra={"basis_self_check": basis_check, "aux_self_check": aux_check},
        ),
    }
    path = common.write_reference(ROW, system, basis_name, payload)
    gap_static = max(rel(a, b_) for a, b_ in zip(per_atom_static_lab, per_atom_dyn_w0))
    print(
        f"{system:4s} {basis_name} aux={aux_name} naux={auxmol.nao_nr()} nov={nov}\n"
        f"  alpha_static iso {iso(alpha_static):.8f}  c6_iso(quad) {c6_iso:.8f}  "
        f"exact {c6_iso_exact:.8f}  quad err {(c6_iso - c6_iso_exact) / c6_iso_exact:.2e}\n"
        f"  scope: c6 dRPA exact-ERI {c6_iso_exact_eri:.8f}  TDHF {c6_iso_tdhf:.8f}\n"
        f"  SOS vs dense {worst:.2e}; grid lab dipole vs analytic {lab_grid_vs_analytic:.2e}\n"
        f"  per-atom: sum_A alpha^A(w0) iso {iso(ac_sum[0]):.6f} vs molecular "
        f"{iso(alpha_mol[0]):.6f}; lab-grid {iso(alpha_lab_grid[0]):.6f}\n"
        f"  static-lab-right vs dynamic(w=0) per-atom max rel gap {gap_static:.2e}\n"
        f"  c6 pair sum {float(c6_pair_iso.sum()):.6f} vs molecular {c6_iso:.6f}"
    )
    return path


def main(argv: list[str]) -> int:
    only = set(argv[1:])
    written = [gen_case(*case) for case in CASES if not only or case[0] in only]
    for p in written:
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    print(f"GEN_PDEP_C6_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
