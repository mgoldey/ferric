//! Newton-accelerated RHF / RKS (closed-shell): correctness + engagement tests.
//!
//! The closed-shell analogue of `uhf_newton_smoke.rs` / `rohf_newton_smoke.rs`.
//! Covers:
//!   1. RHF (HF, no XC): Newton must reach the same energy as DIIS-only.
//!   2. RKS/PBE: the Newton path must engage a *GGA* f_xc kernel (not silently
//!      fall back to DIIS) and reach the same energy as DIIS-only.
//!   3. Hessian-matvec finite-difference check: at a converged RHF state, the
//!      analytic H·κ must match the central difference of the orbital gradient
//!      g(κ) = F_ai(κ) along a random rotation.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Deterministic xorshift64 PRNG (no external rand dep).
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

/// RHF (no XC): Newton-accelerated run must reach the same energy as DIIS-only
/// on closed-shell water/cc-pVDZ.
#[test]
fn rhf_newton_water_matches_diis_only() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RhfConfig { newton_trigger: 1e-2, ..cfg_diis.clone() };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();

    eprintln!("RHF/H2O  DIIS:   E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!("RHF/H2O  Newton: E = {:.10}, iters = {}", r_newton.energy, r_newton.iterations);

    assert!(r_diis.converged && r_newton.converged);
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-8,
        "RHF Newton must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// RKS/PBE: the Newton path must engage a GGA f_xc kernel (proved via the
/// `GGA_FXC_KERNEL_BUILDS` counter) and reach the same energy as DIIS-only.
#[test]
fn rks_pbe_newton_engages_gga_fxc_and_matches_diis() {
    use std::sync::atomic::Ordering;

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RhfConfig { newton_trigger: 1e-2, ..cfg_diis.clone() };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let before = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    let after = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let gga_builds = after.saturating_sub(before);

    eprintln!("RKS/H2O/PBE  DIIS:   E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!(
        "RKS/H2O/PBE  Newton: E = {:.10}, iters = {}, GGA-fxc builds = {}",
        r_newton.energy, r_newton.iterations, gga_builds
    );

    assert!(r_diis.converged, "DIIS-only RKS/PBE must converge");
    assert!(r_newton.converged, "Newton+GGA-fxc RKS/PBE must converge");
    assert!(
        gga_builds >= 1,
        "RKS Newton must engage the GGA f_xc kernel for PBE (got {gga_builds}); \
         it must not silently fall back to DIIS"
    );
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-7,
        "RKS Newton+GGA-fxc must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Hessian-matvec finite-difference check.
///
/// At a converged RHF state, rotate the MOs by ε·κ (random antisymmetric occ→virt
/// block), rebuild the Fock, and read the occ→virt gradient block g(ε). The
/// central difference [g(ε) − g(−ε)]/(2ε) must match the analytic Hessian-vector
/// product `rhf_newton::hessian_matvec`.
#[test]
fn rhf_hessian_matvec_matches_finite_difference() {
    use ferric_scf::rhf::build_jk;
    use ferric_integrals::oneelectron;
    use ndarray::Array2;
    use ndarray_linalg::Solve;

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);
    let c = res.mos_alpha.clone();
    let nocc = mol.nelec() as usize / 2;

    // Occ→virt gradient g_{ai} = F_{ai} at MOs c, via the restricted Fock
    // F = H + J[2·D_c] − ½·K[2·D_c], with D_c = C_occ·C_occᵀ.
    let grad_at = |cc: &Array2<f64>| -> Array2<f64> {
        let cocc = cc.slice(ndarray::s![.., ..nocc]);
        let dtot = 2.0 * cocc.dot(&cocc.t());
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
        let f = &h + &j - &(0.5 * &k);
        let f_mo = cc.t().dot(&f).dot(cc);
        let nv = n - nocc;
        let mut o = Array2::<f64>::zeros((nv, nocc));
        for (ir, a) in (nocc..n).enumerate() {
            for i in 0..nocc {
                o[(ir, i)] = f_mo[(a, i)];
            }
        }
        o
    };

    // Random antisymmetric rotation direction (occ→virt block).
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let nv = n - nocc;
    let k_dir = Array2::<f64>::from_shape_fn((nv, nocc), |_| 0.01 * rng.next_f64());

    // Apply Cayley rotation for the occ→virt block to C.
    let rotate = |c: &Array2<f64>, k_ov: &Array2<f64>, eps: f64| -> Array2<f64> {
        let mut kappa = Array2::<f64>::zeros((n, n));
        for (ir, a) in (nocc..n).enumerate() {
            for i in 0..nocc {
                let v = eps * k_ov[(ir, i)];
                kappa[(a, i)] = v;
                kappa[(i, a)] = -v;
            }
        }
        let half = 0.5 * &kappa;
        let eye = Array2::<f64>::eye(n);
        let am = &eye - &half;
        let bm = &eye + &half;
        let mut u = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let sol = am.solve(&bm.column(col).to_owned()).unwrap();
            for row in 0..n {
                u[(row, col)] = sol[row];
            }
        }
        c.dot(&u)
    };

    // Analytic H·κ from the solver's own matvec.
    let f_mo = {
        let cocc = c.slice(ndarray::s![.., ..nocc]);
        let dtot = 2.0 * cocc.dot(&cocc.t());
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
        let f = &h + &j - &(0.5 * &k);
        c.t().dot(&f).dot(&c)
    };

    let inputs = ferric_scf::rhf_newton::RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0, // pure HF
        fxc: None,
        thresh: 1e-12,
    };
    let hk = ferric_scf::rhf_newton::hessian_matvec(&ctx, &inputs, &k_dir).unwrap();

    // Central-difference the orbital gradient along κ:  [g(εκ) − g(−εκ)]/(2ε) → H·κ.
    let fd_hk = |eps: f64| -> Array2<f64> {
        let cp = rotate(&c, &k_dir, eps);
        let cm = rotate(&c, &k_dir, -eps);
        &(&grad_at(&cp) - &grad_at(&cm)) / (2.0 * eps)
    };

    let fro = |a: &Array2<f64>| -> f64 { a.iter().map(|&x| x * x).sum::<f64>().sqrt() };
    let fro_diff = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    };
    let scale = fro(&hk).max(1e-30);

    let eps_ladder = [1e-3, 5e-4, 2.5e-4, 1.25e-4];
    let mut rels = Vec::new();
    for &eps in &eps_ladder {
        let fa = fd_hk(eps);
        let rel = fro_diff(&hk, &fa) / scale;
        eprintln!("RHF H·κ analytic-vs-FD: eps={eps:.2e}  rel_err={rel:.3e}");
        rels.push(rel);
    }
    // The Hessian matvec is the EXACT analytic derivative of the orbital
    // gradient, so its central-difference residual sits at the FD round-off floor
    // for every ε. Require tight agreement at every ε on the ladder.
    let worst = rels.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        worst < 1e-6,
        "RHF analytic H·κ must match the gradient central difference at the FD \
         round-off floor for every ε: worst rel err {worst:e} (want < 1e-6). \
         ladder={rels:?}"
    );
}
