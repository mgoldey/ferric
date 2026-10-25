//! UKS analytic gradient under `fractional_occ`, anchored to a central finite
//! difference (1e-4 Bohr) of ferric's own SCF energy.
//!
//! System: O(³P)···He at 4.5 Bohr, UKS-PBE/6-31G, triplet. The O β 2p shell
//! holds one electron over three orbitals, so `fractional_occ` gives each
//! 1/3. The He splits p_z from p_x/p_y by 1.5 mHa, well inside the 0.05 Ha
//! grouping tolerance, so the ensemble stays fractional at convergence and at
//! both displaced geometries (asserted). The group is NOT degenerate, which is
//! the case that matters: the Pulay term is `−tr(W S^R)` and the displacement
//! along the axis treats p_z and p_x/p_y differently.
//!
//! Why not a linear radical such as OH (²Π, β π^1 → 1/2 + 1/2)? Measured: the
//! integer W (one π column) and the fractional W (half of each π) give the
//! SAME axial gradient to 8 digits, because tr(c cᵀ S^z) is the same for every
//! unit vector in the degenerate π pair. A displacement that splits the pair
//! has a zero gradient by symmetry. So OH cannot see a wrong W.
//!
//! Before 2026-10-24 `build_energy_weighted_density_uhf` built W from the
//! integer `nocc` columns while the gradient's densities were the fractional
//! ones: on this system the analytic gradient was −3.2636e-3 vs FD −2.5794e-3
//! Ha/Bohr (off by 6.8e-4, 27 % of the force). With W = C f ε Cᵀ the two agree
//! to 5e-10.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_uks;
use ferric_scf::result::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use ndarray::Array2;

/// FD step (Bohr) of the central difference.
const DELTA: f64 = 1e-4;
/// O–He distance (Bohr).
const R_OHE: f64 = 4.5;
const BASIS: &str = "6-31g";
const XC: &str = "PBE";

fn o_he(z_he: f64) -> Molecule {
    let bohr_to_ang = 0.529_177_210_903;
    let xyz = format!("2\nO-He\nO 0 0 0\nHe 0 0 {}\n", z_he * bohr_to_ang);
    Molecule::parse_xyz(&xyz, 0, 3).unwrap()
}

fn solve(mol: &Molecule) -> (ScfResult, PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = UhfConfig {
        xc: Some(XC.to_string()),
        fractional_occ: true,
        energy_conv: 1e-11,
        density_conv: 1e-8,
        max_iter: 300,
        ..Default::default()
    };
    let r = solve_uhf(&ParallelContext::default(), mol, &prep, &bounds, &cfg).unwrap();
    assert!(r.converged);
    (r, prep, bounds)
}

/// `max|D S D − D|` of the β density: zero for an integer occupation.
fn beta_idempotency_defect(r: &ScfResult, prep: &PreparedBasis) -> f64 {
    let s = oneelectron::overlap(prep);
    let d = r.density_beta.as_ref().unwrap();
    (&d.dot(&s).dot(d) - d)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// SCF energy at He position `z`, asserting the β ensemble is still fractional
/// there (otherwise the FD would cross onto the integer surface).
fn fractional_energy(z: f64) -> f64 {
    let (r, prep, _) = solve(&o_he(z));
    let defect = beta_idempotency_defect(&r, &prep);
    assert!(
        defect > 1e-2,
        "z = {z}: β ensemble collapsed to integer ({defect:.1e})"
    );
    r.energy
}

#[test]
fn uks_fractional_occ_gradient_matches_fd_of_energy() {
    let mol = o_he(R_OHE);
    let (r, prep, bounds) = solve(&mol);
    let defect = beta_idempotency_defect(&r, &prep);
    assert!(
        defect > 1e-2,
        "β density is integer ({defect:.1e}): the test cannot see W"
    );
    let eps_b = r.eps_beta.as_ref().unwrap();
    eprintln!("β eigenvalues around the frontier: {:?}", &eps_b[2..7]);

    let bs = basis::bundled(BASIS).unwrap();
    let g: Array2<f64> =
        ks_gradient_uks(&mol, &prep, &bs, Operator::coulomb(), &bounds, XC, &r, None).unwrap();
    let ana = g[(1, 2)];
    let fd = (fractional_energy(R_OHE + DELTA) - fractional_energy(R_OHE - DELTA)) / (2.0 * DELTA);
    let diff = (ana - fd).abs();
    eprintln!("dE/dz_He analytic {ana:+.8e} FD {fd:+.8e} |diff| {diff:.2e} β defect {defect:.2e}");
    assert!(fd.abs() > 1e-3, "force too small to test ({fd:.2e})");
    // Measured 4.9e-10 (grid response included); integer W misses by 6.8e-4.
    assert!(
        diff < 1e-6,
        "analytic {ana:.8e} vs FD {fd:.8e} ({diff:.2e})"
    );
    // Translational invariance: the O row is minus the He row.
    let drift = (g[(0, 2)] + g[(1, 2)]).abs();
    assert!(drift < 1e-8, "Σ_A dE/dz_A = {drift:.2e}");
}

#[test]
fn uhf_gradient_refuses_a_density_its_orbitals_cannot_build() {
    // A MOM-style mismatch: the stored β density is not built from the stored
    // orbitals' integer or fractional occupation, so no consistent W exists.
    let mol = o_he(R_OHE);
    let (mut r, prep, bounds) = solve(&mol);
    let c = r.mos_beta.as_ref().unwrap();
    let n = c.ncols();
    // Occupy the first three columns plus the HIGHEST, skipping the p shell.
    let mut d = Array2::<f64>::zeros((n, n));
    for p in [0, 1, 2, n - 1] {
        let col = c.column(p);
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] += col[i] * col[j];
            }
        }
    }
    r.density_beta = Some(d);
    let err =
        ferric_scf::gradient::uhf_gradient(&mol, &prep, Operator::coulomb(), &bounds, &r, None)
            .unwrap_err()
            .to_string();
    assert!(err.contains("energy-weighted density"), "{err}");
}
