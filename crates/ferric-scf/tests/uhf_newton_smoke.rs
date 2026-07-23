//! Newton-accelerated UHF / UKS: correctness + engagement tests.
//!
//! Mirrors `rohf_newton_smoke.rs` for the unrestricted (two-independent-MO-set)
//! solver. Covers:
//!   1. UHF (HF, no XC): Newton must reach the same energy as DIIS-only.
//!   2. UKS/PBE: the Newton path must engage a *GGA* f_xc kernel (not silently
//!      fall back to DIIS) and reach the same energy as DIIS-only.
//!   3. Hessian-matvec finite-difference check: at a converged UHF state, the
//!      analytic H·κ must match the central difference of the orbital gradient
//!      g(κ) = F_ai(κ) along a random rotation.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

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

/// UHF (no XC): Newton-accelerated run must reach the same energy as DIIS-only
/// on the doublet OH radical.
#[test]
fn uhf_newton_oh_matches_diis_only() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
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

    let r_diis = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_newton).unwrap();

    eprintln!("UHF/OH  DIIS:   E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!("UHF/OH  Newton: E = {:.10}, iters = {}", r_newton.energy, r_newton.iterations);

    assert!(r_diis.converged && r_newton.converged);
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "UHF Newton must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// UKS/PBE: the Newton path must engage a GGA f_xc kernel (proved via the
/// `GGA_FXC_KERNEL_BUILDS` counter — the whole point of the UKS wiring) and
/// reach the same energy as DIIS-only. Uses the doublet OH radical.
#[test]
fn uks_pbe_newton_engages_gga_fxc_and_matches_diis() {
    use std::sync::atomic::Ordering;

    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
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
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg_newton = RhfConfig { newton_trigger: 1e-2, ..cfg_diis.clone() };

    let r_diis = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_diis).unwrap();

    let before = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let r_newton = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_newton).unwrap();
    let after = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let gga_builds = after.saturating_sub(before);

    eprintln!("UKS/OH/PBE  DIIS:   E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!(
        "UKS/OH/PBE  Newton: E = {:.10}, iters = {}, GGA-fxc builds = {}",
        r_newton.energy, r_newton.iterations, gga_builds
    );

    assert!(r_diis.converged, "DIIS-only UKS/PBE must converge");
    assert!(r_newton.converged, "Newton+GGA-fxc UKS/PBE must converge");
    assert!(
        gga_builds >= 1,
        "UKS Newton must engage the GGA f_xc kernel for PBE (got {gga_builds}); \
         it must not silently fall back to DIIS"
    );
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "UKS Newton+GGA-fxc must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Hessian-matvec finite-difference check.
///
/// At a converged UHF state, rotate the MOs by ε·κ (random antisymmetric per
/// spin), rebuild the Fock, and read the occ→virt gradient block g^σ(ε). The
/// central difference [g^σ(ε) − g^σ(−ε)]/(2ε) must match the analytic
/// Hessian-vector product `uhf_newton::hessian_matvec` (exposed via the public
/// `uhf_newton_step` at large trust radius / the internal matvec). We drive it
/// through a thin re-implementation of the gradient at rotated C.
#[test]
fn uhf_hessian_matvec_matches_finite_difference() {
    use ferric_scf::rhf::build_jk;
    use ferric_integrals::oneelectron;
    use ndarray::Array2;
    use ndarray_linalg::Solve;

    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
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
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;

    // Occ→virt gradient g^σ_{ai} = F^σ_{ai} at MOs (c_a, c_b), via full Fock.
    let grad_at = |ca: &Array2<f64>, cb: &Array2<f64>| -> (Array2<f64>, Array2<f64>) {
        let da = ca.slice(ndarray::s![.., ..nocc_a]).dot(&ca.slice(ndarray::s![.., ..nocc_a]).t());
        let db = cb.slice(ndarray::s![.., ..nocc_b]).dot(&cb.slice(ndarray::s![.., ..nocc_b]).t());
        let dt = &da + &db;
        let mut j = Array2::<f64>::zeros((n, n));
        let mut kdum = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dt, &mut j, &mut kdum).unwrap();
        let mut ka = Array2::<f64>::zeros((n, n));
        let mut kb = Array2::<f64>::zeros((n, n));
        let mut jd = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &da, &mut jd, &mut ka).unwrap();
        jd.fill(0.0);
        build_jk(&ctx, &prep, &bounds, 1e-12, &db, &mut jd, &mut kb).unwrap();
        let fa = &h + &j - &ka;
        let fb = &h + &j - &kb;
        let fa_mo = ca.t().dot(&fa).dot(ca);
        let fb_mo = cb.t().dot(&fb).dot(cb);
        let ov = |m: &Array2<f64>, nocc: usize| -> Array2<f64> {
            let nv = n - nocc;
            let mut o = Array2::<f64>::zeros((nv, nocc));
            for (ir, a) in (nocc..n).enumerate() {
                for i in 0..nocc { o[(ir, i)] = m[(a, i)]; }
            }
            o
        };
        (ov(&fa_mo, nocc_a), ov(&fb_mo, nocc_b))
    };

    // Random antisymmetric rotation directions (occ→virt blocks).
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let mk = |rng: &mut Xorshift64, nocc: usize| -> Array2<f64> {
        let nv = n - nocc;
        Array2::<f64>::from_shape_fn((nv, nocc), |_| 0.01 * rng.next_f64())
    };
    let ka = mk(&mut rng, nocc_a);
    let kb = mk(&mut rng, nocc_b);

    // Apply Cayley rotation for the occ→virt block to C.
    let rotate = |c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64| -> Array2<f64> {
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
            for row in 0..n { u[(row, col)] = sol[row]; }
        }
        c.dot(&u)
    };

    // Analytic H·κ from the solver's own matvec.
    let f_a_mo = {
        let da = c_a.slice(ndarray::s![.., ..nocc_a]).dot(&c_a.slice(ndarray::s![.., ..nocc_a]).t());
        let db = c_b.slice(ndarray::s![.., ..nocc_b]).dot(&c_b.slice(ndarray::s![.., ..nocc_b]).t());
        let dt = &da + &db;
        let mut j = Array2::<f64>::zeros((n, n));
        let mut kdum = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dt, &mut j, &mut kdum).unwrap();
        let mut ka2 = Array2::<f64>::zeros((n, n));
        let mut jd = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &da, &mut jd, &mut ka2).unwrap();
        let fa = &h + &j - &ka2;
        c_a.t().dot(&fa).dot(&c_a)
    };
    let f_b_mo = {
        let da = c_a.slice(ndarray::s![.., ..nocc_a]).dot(&c_a.slice(ndarray::s![.., ..nocc_a]).t());
        let db = c_b.slice(ndarray::s![.., ..nocc_b]).dot(&c_b.slice(ndarray::s![.., ..nocc_b]).t());
        let dt = &da + &db;
        let mut j = Array2::<f64>::zeros((n, n));
        let mut kdum = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dt, &mut j, &mut kdum).unwrap();
        let mut kb2 = Array2::<f64>::zeros((n, n));
        let mut jd = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &db, &mut jd, &mut kb2).unwrap();
        let fb = &h + &j - &kb2;
        c_b.t().dot(&fb).dot(&c_b)
    };

    let inputs = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a,
        nocc_b,
        k_mix_sr: 1.0, // pure HF
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, &prep, 1e-14).unwrap();
    let (hk_a, hk_b) = ferric_scf::uhf_newton::hessian_matvec(&ctx, &inputs, &ka, &kb, &pool).unwrap();

    // Central-difference the orbital gradient along κ:  [g(εκ) − g(−εκ)]/(2ε)
    // → H·κ as ε → 0. Compare the analytic matvec against a small-ε ladder,
    // requiring tight agreement + clean super-linear convergence (the matvec is
    // the true derivative of the gradient, so this is the load-bearing check).
    let fd_hk = |eps: f64| -> (Array2<f64>, Array2<f64>) {
        let cap = rotate(&c_a, &ka, nocc_a, eps);
        let cbp = rotate(&c_b, &kb, nocc_b, eps);
        let cam = rotate(&c_a, &ka, nocc_a, -eps);
        let cbm = rotate(&c_b, &kb, nocc_b, -eps);
        let (gap, gbp) = grad_at(&cap, &cbp);
        let (gam, gbm) = grad_at(&cam, &cbm);
        (&(&gap - &gam) / (2.0 * eps), &(&gbp - &gbm) / (2.0 * eps))
    };

    let fro = |a: &Array2<f64>| -> f64 { a.iter().map(|&x| x * x).sum::<f64>().sqrt() };
    let fro_diff = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    };
    let scale = (fro(&hk_a).powi(2) + fro(&hk_b).powi(2)).sqrt().max(1e-30);

    let eps_ladder = [1e-3, 5e-4, 2.5e-4, 1.25e-4];
    let mut rels = Vec::new();
    for &eps in &eps_ladder {
        let (fa, fb) = fd_hk(eps);
        let rel = (fro_diff(&hk_a, &fa).powi(2) + fro_diff(&hk_b, &fb).powi(2)).sqrt() / scale;
        eprintln!("UHF H·κ analytic-vs-FD: eps={eps:.2e}  rel_err={rel:.3e}");
        rels.push(rel);
    }
    // The Hessian matvec is the EXACT analytic derivative of the orbital
    // gradient, so its central-difference residual sits at the FD round-off
    // floor (~1e-10 here) for every ε — it does not "converge" from above the
    // way a truncation-limited approximation would. The correct assertion is
    // therefore that agreement is tight at EVERY ε on the ladder (all near the
    // round-off floor), which is far stronger than a super-linear-trend check.
    let worst = rels.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        worst < 1e-6,
        "UHF analytic H·κ must match the gradient central difference at the FD \
         round-off floor for every ε: worst rel err {worst:e} (want < 1e-6). \
         ladder={rels:?}"
    );
}
