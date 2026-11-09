//! U-G0W0 on a ROHF/ROKS reference: it runs, and it reduces to known limits.
//!
//! # The defect (fixed in `ferric_scf::rohf`, `OccupationGuard::exit_labels`)
//!
//! Every Python `run_u_gw(reference="rohf", xc=...)` failed with "Full-rank
//! PDEP eigh failed: ... return_code = 69", at every basis. Python sets
//! `mom_after_iter = 5`, which disables the ROHF occupation guard; the ROHF
//! exit then re-labelled the converged orbitals by effective-Fock index
//! aufbau, which for OH ²Π/PBE (open π below closed π in that spectrum) hands
//! back the OTHER π determinant while `rohf_spin_focks` keep the converged
//! one. The semi-canonical β block then had its occupied π 0.026 Ha ABOVE the
//! empty π, χ₀'s `sqrt(2e/(ω²+e²))` went NaN and the eigh failed.
//!
//! # Tests
//!
//! * `u_g0w0_on_roks_with_mom_runs_and_equals_its_symmetry_image`: OH/STO-3G
//!   ROKS/PBE with `mom_after_iter = 5` (the Python recipe, exact J/K) runs
//!   and gives the same U-G0W0 window, both spins, as the guarded default run
//!   (π_x ↔ π_y symmetry images of one state). Before the fix: the error above.
//! * `u_g0w0_on_closed_shell_rohf_equals_rhf_g0w0`: exactness anchor — a
//!   closed-shell singlet (H2O/STO-3G) solved as ROHF and run through
//!   `run_u_gw` reproduces the UHF route to 1e-9 Ha and `run_gw` on the RHF
//!   reference to 1e-6 Ha (the residual is the closed- vs open-shell Σc
//!   frequency rule: 7.2e-7 at n_quad 16, 1.8e-8 at 60), both spins.
//!
//! Run: `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release --test rohf_ugw_runs`

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::unrestricted_reference;

/// Two SCF trajectories converged to 1e-9 in the density, and two independent
/// QP Newton solves stopping at |step| < 1e-7: 1e-6 Ha.
const TOL_QP: f64 = 1e-6;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        eigensolver_conv_thresh: 1e-9,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn g0w0() -> GwConfig {
    GwConfig {
        method: GwMethod::G0W0,
        ..Default::default()
    }
}

fn scf_cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    }
}

fn u_g0w0_roks_oh(mom_after_iter: usize) -> UGwResult {
    let mol = Molecule::parse_xyz("2\n\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        xc: Some("pbe".into()),
        mom_after_iter,
        ..scf_cfg()
    };
    let roks = solve_rohf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(roks.converged);
    // v_xc in the QP equation's own (semi-canonical) basis, as Python does.
    let view = unrestricted_reference(&mol, &roks).unwrap().into_owned();
    let (va, vb) = vxc_diagonal_mo(&mol, &bs, "pbe", &view).unwrap();
    run_u_gw(
        &mol,
        &obs,
        &dfbs,
        op,
        &roks,
        &pdep_cfg(),
        &g0w0(),
        Some((&va, &vb)),
    )
    .unwrap_or_else(|e| panic!("U-G0W0@ROKS (mom {mom_after_iter}): {e}"))
}

#[test]
fn u_g0w0_on_roks_with_mom_runs_and_equals_its_symmetry_image() {
    let mom = u_g0w0_roks_oh(5);
    let def = u_g0w0_roks_oh(0);
    assert_eq!(mom.mo_indices, def.mo_indices);
    let mut worst = 0.0f64;
    for (x, y) in mom
        .eps_qp_a
        .iter()
        .zip(&def.eps_qp_a)
        .chain(mom.eps_qp_b.iter().zip(&def.eps_qp_b))
    {
        assert!(x.is_finite() && y.is_finite());
        worst = worst.max((x - y).abs());
    }
    eprintln!(
        "U-G0W0@ROKS OH/STO-3G MOs {:?}\n  mom  qp_a {:?}\n  mom  qp_b {:?}\n  def  qp_b {:?}\n  max |d| {worst:.2e} Ha",
        mom.mo_indices, mom.eps_qp_a, mom.eps_qp_b, def.eps_qp_b
    );
    assert!(
        worst < TOL_QP,
        "MOM vs default U-G0W0 differ by {worst:.3e} Ha"
    );
}

#[test]
fn u_g0w0_on_closed_shell_rohf_equals_rhf_g0w0() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    // Tighter than scf_cfg(): the two SCF drivers' residual orbital errors are
    // the only difference between the arms.
    let tight = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-11,
        ..scf_cfg()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &tight).unwrap();
    let rohf = solve_rohf(&ctx, &mol, &obs, op, &bounds, &tight).unwrap();
    assert!(rhf.converged && rohf.converged);
    assert!((rhf.energy - rohf.energy).abs() < 1e-9);
    let uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &tight).unwrap();
    assert!(uhf.converged);
    let closed = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg(), &g0w0(), None).unwrap();
    let open = run_u_gw(&mol, &obs, &dfbs, op, &rohf, &pdep_cfg(), &g0w0(), None).unwrap();
    let via_uhf = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep_cfg(), &g0w0(), None).unwrap();
    assert_eq!(closed.mo_indices, open.mo_indices);
    let mut worst = 0.0f64;
    let mut worst_uhf = 0.0f64;
    for (i, &q) in closed.eps_qp.iter().enumerate() {
        worst = worst
            .max((q - open.eps_qp_a[i]).abs())
            .max((q - open.eps_qp_b[i]).abs());
        worst_uhf = worst_uhf
            .max((open.eps_qp_a[i] - via_uhf.eps_qp_a[i]).abs())
            .max((open.eps_qp_b[i] - via_uhf.eps_qp_b[i]).abs());
    }
    eprintln!(
        "H2O/STO-3G G0W0@RHF {:?}\n  U-G0W0@ROHF a {:?}\n  max |d| vs RHF {worst:.2e} Ha, vs UHF route {worst_uhf:.2e} Ha",
        closed.eps_qp, open.eps_qp_a
    );
    // ROHF route ≡ UHF route: same code after semi-canonicalization (2.1e-11
    // measured at n_quad 16).
    assert!(
        worst_uhf < 1e-9,
        "ROHF-route U-G0W0 misses the UHF route by {worst_uhf:.3e} Ha"
    );
    // vs RHF: the open-shell Σc uses the product-rule frequency integral, the
    // closed-shell one the plain Gauss–Legendre sum. MEASURED 7.2e-7 Ha at
    // n_quad 16 (independent of SCF tightness 1e-9 vs 1e-11) and 1.8e-8 at
    // n_quad 60, i.e. a quadrature difference that converges away.
    assert!(
        worst < TOL_QP,
        "ROHF-route U-G0W0 misses RHF G0W0 by {worst:.3e} Ha"
    );
}
