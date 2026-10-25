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
use ferric_scf::gradient::build_energy_weighted_density_uhf;
use ferric_scf::ks_gradient::ks_gradient_uks;
use ferric_scf::result::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use ndarray::Array2;
use ndarray_linalg::{EigValsh, Eigh, UPLO};

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

/// The β occupation spectrum: eigenvalues of `S½ D_β S½`, descending.
fn beta_occupation_spectrum(r: &ScfResult, prep: &PreparedBasis) -> Vec<f64> {
    let s = oneelectron::overlap(prep);
    let (w, v) = s.eigh(UPLO::Lower).unwrap();
    let s_half = v.dot(&Array2::from_diag(&w.mapv(f64::sqrt))).dot(&v.t());
    let m = s_half.dot(r.density_beta.as_ref().unwrap()).dot(&s_half);
    let mut occ = m.eigvalsh(UPLO::Lower).unwrap().to_vec();
    occ.sort_by(|a, b| b.total_cmp(a));
    occ
}

/// Assert the β ensemble is the O(³P)···He one: O 1s, O 2s and He 1s full,
/// the O 2p shell 1/3 each, everything else empty. Holding the SAME spectrum
/// at all three FD points is what makes the FD differentiate one surface.
fn assert_fractional_spectrum(r: &ScfResult, prep: &PreparedBasis, at: &str) {
    let occ = beta_occupation_spectrum(r, prep);
    let third = 1.0 / 3.0;
    let expected = [1.0, 1.0, 1.0, third, third, third, 0.0];
    let dev = expected
        .iter()
        .zip(&occ)
        .fold(0.0_f64, |m, (e, o)| m.max((e - o).abs()));
    assert!(
        dev < 1e-8,
        "{at}: β occupation spectrum {:?} is not [1,1,1,1/3,1/3,1/3,0,…] ({dev:.1e})",
        &occ[..7]
    );
}

/// SCF energy at He position `z`, asserting the β ensemble there is the same
/// one as at the centre (otherwise the FD would cross onto another surface).
fn fractional_energy(z: f64) -> f64 {
    let (r, prep, _) = solve(&o_he(z));
    assert_fractional_spectrum(&r, &prep, &format!("z = {z}"));
    r.energy
}

#[test]
fn uks_fractional_occ_gradient_matches_fd_of_energy() {
    let mol = o_he(R_OHE);
    let (r, prep, bounds) = solve(&mol);
    assert_fractional_spectrum(&r, &prep, "centre");
    let eps_b = r.eps_beta.as_ref().unwrap();
    eprintln!("β eigenvalues around the frontier: {:?}", &eps_b[2..7]);

    let bs = basis::bundled(BASIS).unwrap();
    let g: Array2<f64> =
        ks_gradient_uks(&mol, &prep, &bs, Operator::coulomb(), &bounds, XC, &r, None).unwrap();
    let ana = g[(1, 2)];
    let fd = (fractional_energy(R_OHE + DELTA) - fractional_energy(R_OHE - DELTA)) / (2.0 * DELTA);
    let diff = (ana - fd).abs();
    eprintln!("dE/dz_He analytic {ana:+.8e} FD {fd:+.8e} |diff| {diff:.2e}");
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

/// `Σ_p f_p c_p c_pᵀ` over the listed columns with unit occupation.
fn density_of_columns(c: &Array2<f64>, cols: &[usize], weight: &[f64]) -> Array2<f64> {
    let n = c.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for (&p, &f) in cols.iter().zip(weight) {
        let col = c.column(p);
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] += f * col[i] * col[j];
            }
        }
    }
    d
}

fn uhf_gradient_error(
    r: &ScfResult,
    mol: &Molecule,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
) -> String {
    ferric_scf::gradient::uhf_gradient(mol, prep, Operator::coulomb(), b, r, None)
        .unwrap_err()
        .to_string()
}

#[test]
fn uhf_gradient_refuses_a_density_its_orbitals_cannot_build() {
    let mol = o_he(R_OHE);
    let (r, prep, bounds) = solve(&mol);
    let c = r.mos_beta.as_ref().unwrap();
    let n = c.ncols();
    // A MOM-style excited state: the first three columns plus the HIGHEST,
    // skipping the p shell. Integer, but not aufbau.
    let mut mom = r.clone();
    mom.density_beta = Some(density_of_columns(c, &[0, 1, 2, n - 1], &[1.0; 4]));
    let err = uhf_gradient_error(&mom, &mol, &prep, &bounds);
    assert!(err.contains("non-aufbau integer occupation"), "{err}");
    // Not an occupation pattern at all: a half-filled core orbital.
    let mut odd = r.clone();
    odd.density_beta = Some(density_of_columns(c, &[0, 1, 2, 3], &[0.5, 1.0, 1.0, 1.0]));
    let err = uhf_gradient_error(&odd, &mol, &prep, &bounds);
    assert!(err.contains("neither the integer"), "{err}");
}

#[test]
fn integer_w_after_a_loose_scf_is_the_old_expression_bit_for_bit() {
    // A loose SCF stores D from its last iteration and C from re-diagonalising
    // F(D), so D ≠ C_occ C_occᵀ by up to the convergence threshold. Occupation
    // recovery must still classify it as aufbau and return the old W exactly.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = UhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-5,
        density_conv: 1e-3,
        max_iter: 400,
        ..Default::default()
    };
    let r = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    let (na, nb) = (5, 4);
    let old_w = |c: &Array2<f64>, eps: &[f64], nocc: usize| {
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        let cw = &c_occ * &ndarray::ArrayView1::from(&eps[..nocc]);
        c_occ.dot(&cw.t())
    };
    let cb = r.mos_beta.as_ref().unwrap();
    let mismatch = |c: &Array2<f64>, nocc: usize, d: &Array2<f64>| {
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        (&c_occ.dot(&c_occ.t()) - d)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
    };
    let mis = mismatch(&r.mos_alpha, na, &r.density_alpha).max(mismatch(
        cb,
        nb,
        r.density_beta.as_ref().unwrap(),
    ));
    eprintln!(
        "loose OH/LDA: max |C_occ C_occᵀ − D| = {mis:.2e} after {} iterations",
        r.iterations
    );
    // The test only has teeth if the stored D really lags C.
    assert!(mis > 1e-5, "SCF too tight to exercise the lag ({mis:.1e})");
    let w = build_energy_weighted_density_uhf(&r, &prep, na, nb).unwrap();
    let expected =
        old_w(&r.mos_alpha, &r.eps_alpha, na) + old_w(cb, r.eps_beta.as_ref().unwrap(), nb);
    assert!(
        w == expected,
        "integer W is not the old expression bit for bit"
    );
    ks_gradient_uks(
        &mol,
        &prep,
        &bs,
        Operator::coulomb(),
        &bounds,
        "LDA",
        &r,
        None,
    )
    .unwrap();
}
