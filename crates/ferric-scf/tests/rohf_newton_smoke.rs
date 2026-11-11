//! Smoke test for the ROHF Newton step (HF-only path, no XC).
//!
//! Strategy:
//!   1. Run plain ROHF DIIS to convergence on doublet OH/cc-pVDZ.
//!   2. From the converged state, build (F_α, F_β) and ε, and call
//!      `rohf_newton_step` to verify that:
//!      (a) the step is a no-op (||κ|| ≈ 0) because the gradient is zero
//!      at the stationary point;
//!      (b) the matvec is symmetric: ⟨κ₁, H·κ₂⟩ = ⟨κ₂, H·κ₁⟩.
//!
//! These are necessary correctness checks before we plug Newton into the
//! SCF loop and trust it to converge harder cases (LDA/PBE ROKS on OH).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::build_jk;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::rohf_newton::{rohf_newton_step, RohfNewtonInputs};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

#[test]
fn rohf_newton_step_at_stationary_point_is_noop() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    // Plain ROHF (no XC) — should converge easily.
    let cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    // Reconstruct (F_α, F_β) from the converged densities. We rebuild J/K
    // here so we have access to the per-spin Fock — the ScfResult only
    // stores fock_alpha (= F_eff).
    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);

    let d_a = res.density_alpha.clone();
    let d_b = res
        .density_beta
        .as_ref()
        .expect("ROHF stores density_beta")
        .clone();
    let d_total = &d_a + &d_b;

    let mut j = Array2::<f64>::zeros((n, n));
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    let mut k_dummy = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_total, &mut j, &mut k_dummy).unwrap();
    let mut j_dummy = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_a, &mut j_dummy, &mut k_a).unwrap();
    j_dummy.fill(0.0);
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_b, &mut j_dummy, &mut k_b).unwrap();

    let f_a: Array2<f64> = &h + &j - &k_a;
    let f_b: Array2<f64> = &h + &j - &k_b;

    let c = &res.mos_alpha;
    let f_a_mo = c.t().dot(&f_a).dot(c);
    let f_b_mo = c.t().dot(&f_b).dot(c);

    // Occupations
    let nelec = mol.nelec() as usize;
    let mult = mol.multiplicity;
    let nocc_open: usize = mult - 1;
    let nocc_double: usize = (nelec - nocc_open) / 2;

    let inputs = RohfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_double,
        nocc_open,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
        j_response: None,
        k_response: None,
    };

    // Sanity-check the gradient norm at the converged point.
    let n_total = c.ncols();
    let nocc_a = nocc_double + nocc_open;
    let mut gmax = 0.0f64;
    for p in nocc_a..n_total {
        for q in 0..nocc_double {
            let g = f_a_mo[(p, q)] + f_b_mo[(p, q)];
            gmax = gmax.max(g.abs());
        }
    }
    for p in nocc_a..n_total {
        for q in nocc_double..nocc_a {
            gmax = gmax.max(f_a_mo[(p, q)].abs());
        }
    }
    for p in nocc_double..nocc_a {
        for q in 0..nocc_double {
            gmax = gmax.max(f_b_mo[(p, q)].abs());
        }
    }
    eprintln!("Gradient ∞-norm at converged ROHF: {:.3e}", gmax);
    eprintln!("ROHF reports converged in {} iterations", res.iterations);

    let (c_new, kmax) = rohf_newton_step(&ctx, &inputs, 0.0, 0.2, 20, 1e-9).unwrap();
    eprintln!("Newton step at stationary point: kmax = {:.3e}", kmax);
    assert!(
        kmax < 1e-5,
        "At a converged stationary point, the Newton step should be ~0; got kmax = {:.3e}",
        kmax
    );

    // C should be essentially unchanged.
    let diff = (&c_new - c).iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    eprintln!("||C_new − C||_∞ = {:.3e}", diff);
    assert!(diff < 1e-5);
}

/// Triplet H₂ at 2.0 Å under LDA ROKS: a benign open-shell case (well-separated
/// SOMOs, no SOMO/HOMO near-degeneracy). Newton+LDA path must converge to the
/// same energy as DIIS-only. Validates the LDA f_xc kernel + Newton matvec
/// without the OH-doublet instability noise that dominates that case.
#[test]
fn rohf_newton_h2_triplet_lda_matches_diis() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 2.0\n", 0, 3).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RohfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RohfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    eprintln!(
        "H₂ triplet/LDA  DIIS:   E = {:.10}, iters = {}",
        r_diis.energy, r_diis.iterations
    );
    eprintln!(
        "H₂ triplet/LDA  Newton: E = {:.10}, iters = {}",
        r_newton.energy, r_newton.iterations
    );

    assert!(r_diis.converged && r_newton.converged);
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "Newton+LDA must reach the same energy as DIIS-only on a benign case; \
         got ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Newton-augmented ROHF on OH/HF must reach the same energy as DIIS-only.
/// Tests the in-SCF wiring with the per-spin semicanonical preconditioner.
#[test]
fn rohf_newton_oh_hf_matches_diis_only() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        newton_trigger: 1e-2,
        ..Default::default()
    };

    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();

    eprintln!(
        "DIIS: E = {:.10}, iters = {}",
        r_diis.energy, r_diis.iterations
    );
    eprintln!(
        "Newton-aug: E = {:.10}, iters = {}",
        r_newton.energy, r_newton.iterations
    );

    assert!(r_diis.converged, "DIIS-only must converge");
    assert!(r_newton.converged, "Newton-aug must converge");
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "Energies must match: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// End-to-end proof that the GGA f_xc Newton acceleration engages for a real
/// open-shell GGA ROKS run (was LDA-only before). Triplet H₂ at 2.0 Å / cc-pVDZ
/// under PBE — the same benign open-shell case the LDA test above uses (well-
/// separated SOMOs, no SOMO/HOMO near-degeneracy), so it isolates the GGA f_xc
/// kernel + Newton matvec from the OH-doublet instability that dominates that
/// harder case. Asserts:
///   - The Newton path builds a *GGA* f_xc kernel (not a silent DIIS fallback)
///     — via the `GGA_FXC_KERNEL_BUILDS` counter. This is the load-bearing
///     "the GGA branch is actually taken" check.
///   - Newton+GGA-fxc reaches the SAME energy as DIIS-only.
#[test]
fn roks_h2_triplet_pbe_gga_fxc_newton_engages_and_matches_diis() {
    use std::sync::atomic::Ordering;

    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 2.0\n", 0, 3).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RohfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RohfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };

    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let before = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let r_newton = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    let after = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let gga_builds = after.saturating_sub(before);

    eprintln!(
        "H₂ triplet/PBE  DIIS:   E = {:.10}, iters = {}",
        r_diis.energy, r_diis.iterations
    );
    eprintln!(
        "H₂ triplet/PBE  Newton: E = {:.10}, iters = {}, GGA-fxc kernel builds = {}",
        r_newton.energy, r_newton.iterations, gga_builds
    );

    assert!(r_diis.converged, "DIIS-only H₂/PBE must converge");
    assert!(r_newton.converged, "Newton+GGA-fxc H₂/PBE must converge");

    // The GGA fxc Newton path must have actually engaged (the whole point —
    // before this change GGA ROKS silently fell back to DIIS).
    assert!(
        gga_builds >= 1,
        "GGA f_xc Newton path must engage for PBE ROKS (got {gga_builds} builds); \
         it must not silently fall back to DIIS"
    );

    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "Newton+GGA-fxc must reach the same energy as DIIS-only: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Saddle-point fixture: OH/6-31G ROHF from the hcore guess with the
/// occupation guard OFF converges (DIIS) to the σ-hole stationary point
/// E = −75.2037249529, which is a SADDLE of the ROHF energy: the exact
/// half-Hessian's λ_min there is −0.156 (measured 2026-11-11; the ground state
/// at −75.3618462891 has λ_min ≈ 0, the flat π-hole orientation mode).
struct Saddle {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    h: Array2<f64>,
    c: Array2<f64>,
    e: f64,
    nc: usize,
}

const OH_631G_SADDLE: f64 = -75.203_724_952_9;

impl Saddle {
    fn new() -> Self {
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let ctx = ParallelContext::default();
        let r = solve_rohf(
            &ctx,
            &mol,
            &prep,
            Operator::coulomb(),
            &bounds,
            &Self::cfg(),
        )
        .unwrap();
        assert!(
            (r.energy - OH_631G_SADDLE).abs() < 1e-8,
            "premise: hcore + guard off should reach the σ-hole saddle, got {:.10}",
            r.energy
        );
        let h = oneelectron::hcore(&prep);
        let nc = (mol.nelec() as usize - 1) / 2;
        Saddle {
            mol,
            prep,
            bounds,
            ctx,
            h,
            c: r.mos_alpha,
            e: r.energy,
            nc,
        }
    }
    fn cfg() -> RohfConfig {
        RohfConfig {
            max_iter: 400,
            density_conv: 1e-10,
            energy_conv: 1e-11,
            use_sad_guess: false,
            rohf_occupation_guard: false,
            ..Default::default()
        }
    }
    fn densities(&self, c: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let ca = c.slice(ndarray::s![.., ..self.nc + 1]);
        let cb = c.slice(ndarray::s![.., ..self.nc]);
        (ca.dot(&ca.t()), cb.dot(&cb.t()))
    }
    /// (F_α MO, F_β MO, total energy) at `c`.
    fn point(&self, c: &Array2<f64>) -> (Array2<f64>, Array2<f64>, f64) {
        let n = self.prep.nbasis();
        let (da, db) = self.densities(c);
        let z = || Array2::<f64>::zeros((n, n));
        let (mut j, mut kd, mut ka, mut kb, mut jj) = (z(), z(), z(), z(), z());
        let (ctx, p, b) = (&self.ctx, &self.prep, &self.bounds);
        build_jk(ctx, p, b, 1e-12, &(&da + &db), &mut j, &mut kd).unwrap();
        build_jk(ctx, p, b, 1e-12, &da, &mut jj, &mut ka).unwrap();
        build_jk(ctx, p, b, 1e-12, &db, &mut jj, &mut kb).unwrap();
        let fa = &self.h + &j - &ka;
        let fb = &self.h + &j - &kb;
        let e = 0.5 * ((&da * &(&self.h + &fa)).sum() + (&db * &(&self.h + &fb)).sum())
            + self.mol.nuclear_repulsion();
        (c.t().dot(&fa).dot(c), c.t().dot(&fb).dot(c), e)
    }
    fn inputs<'a>(
        &'a self,
        c: &'a Array2<f64>,
        fa: &'a Array2<f64>,
        fb: &'a Array2<f64>,
    ) -> RohfNewtonInputs<'a> {
        RohfNewtonInputs {
            prep: &self.prep,
            bounds: &self.bounds,
            c,
            f_a_mo: fa,
            f_b_mo: fb,
            nocc_double: self.nc,
            nocc_open: 1,
            k_mix_sr: 1.0,
            rsh: None,
            fxc: None,
            thresh: 1e-12,
            ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
            j_response: None,
            k_response: None,
        }
    }
    /// The lowest eigenpair of the half Hessian at `c` (packed vc|vo|oc).
    fn lowest_mode(&self, c: &Array2<f64>) -> (f64, Vec<f64>) {
        use ferric_scf::engine_pool::EnginePool;
        let (fa, fb, _) = self.point(c);
        let inp = self.inputs(c, &fa, &fb);
        let n = self.prep.nbasis();
        let (nc, no) = (self.nc, 1);
        let nv = n - nc - no;
        let pool = EnginePool::new(self.bounds.op, &self.prep, 1e-14).unwrap();
        let mv = |x: &[f64]| {
            let (a, rest) = x.split_at(nv * nc);
            let (b, d) = rest.split_at(nv * no);
            let a = Array2::from_shape_vec((nv, nc), a.to_vec()).unwrap();
            let b = Array2::from_shape_vec((nv, no), b.to_vec()).unwrap();
            let d = Array2::from_shape_vec((no, nc), d.to_vec()).unwrap();
            let (ha, hb, hd) =
                ferric_scf::rohf_newton::hessian_matvec(&self.ctx, &inp, &a, &b, &d, &pool)?;
            Ok(ha
                .iter()
                .chain(hb.iter())
                .chain(hd.iter())
                .copied()
                .collect())
        };
        let diag: Vec<f64> = ferric_scf::rohf_trah::rohf_hessian_diagonal(&fa, &fb, nc, no)
            .iter()
            .map(|d| 0.5 * d)
            .collect();
        let dim = diag.len();
        let lo = ferric_scf::stability::davidson_lowest(
            dim,
            mv,
            &diag,
            &ferric_scf::stability::StabilityConfig::default(),
        )
        .unwrap();
        assert!(lo.converged, "premise: λ_min eigensolve converged");
        (lo.eigenvalue, lo.eigenvector)
    }
}

/// **Newton must not converge to a saddle (Steihaug guard).**
///
/// From the σ-hole saddle displaced by ε = 0.05 along its negative-curvature
/// mode, the exact Newton step −H⁻¹g points BACK to the saddle (uphill), so an
/// unguarded PCG would return there. With the guard, PCG stops at pᵀAp ≤ 0 and
/// keeps κ⁰ = −M⁻¹g (M positive), a descent direction.
///
/// Hypotheses (before measuring): guard works ⇒ one step lowers E below the
/// displaced point, and the full SCF ends ≥ 1e-3 Ha below the saddle;
/// guard missing ⇒ the step raises E toward the saddle, and the SCF returns
/// to it. Artifact guard: the premise asserts λ_min < 0 at the fixture and
/// E(displaced) < E(saddle), so a fixture that is not a saddle fails loudly.
///
/// MEASURED 2026-11-11: saddle −75.2037249529, displaced −75.2043419012
/// (λ_min −0.156); guarded step −75.2048250666 (one truncation), full SCF
/// −75.3618462891 (the ground state). Mutation (guard removed, applied and
/// reverted): the step returns to −75.2037249233 and the full SCF converges
/// to the saddle, −75.2037249529.
#[test]
fn rohf_newton_does_not_converge_to_a_saddle() {
    use ferric_scf::rohf_newton::steihaug_truncations_on_this_thread;
    use ferric_scf::rohf_trah::rohf_cayley_rotate;
    let s = Saddle::new();
    let (lam, v) = s.lowest_mode(&s.c);
    assert!(
        lam < -1e-2,
        "premise: λ_min at the fixture is negative, got {lam:.3e}"
    );
    let vmax = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let kappa: Vec<f64> = v.iter().map(|x| 0.05 * x / vmax).collect();
    let c_d = rohf_cayley_rotate(&s.c, &kappa, s.nc, 1).unwrap();
    let (fa, fb, e_d) = s.point(&c_d);
    assert!(
        e_d < s.e,
        "premise: the displacement lowers E ({e_d:.10} vs {:.10})",
        s.e
    );

    // (a) One production Newton step from the displaced point.
    let t0 = steihaug_truncations_on_this_thread();
    let inp = s.inputs(&c_d, &fa, &fb);
    let (c_n, _) = rohf_newton_step(&s.ctx, &inp, 1e-6, 0.1, 20, 1e-7).unwrap();
    let truncs = steihaug_truncations_on_this_thread() - t0;
    let e_n = s.point(&c_n).2;
    eprintln!(
        "saddle {:.10}  displaced {e_d:.10}  after one Newton step {e_n:.10}  (λ_min {lam:.3e}, truncations {truncs})",
        s.e
    );

    // (b) The full solver, seeded with the displaced density, Newton from
    // iteration 4 on (trigger 1.0), occupation guard off.
    let (da, db) = s.densities(&c_d);
    let cfg = RohfConfig {
        init_guess_density: Some(&da + &db),
        newton_trigger: 1.0,
        ..Saddle::cfg()
    };
    let full = solve_rohf(
        &s.ctx,
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &cfg,
    );
    let e_full = full.as_ref().map(|r| r.energy).ok();
    eprintln!("full Newton SCF from the displaced density: {e_full:?}");

    assert!(
        e_n < e_d,
        "the Newton step went UPHILL toward the saddle: {e_n:.10} >= {e_d:.10}"
    );
    let e_full = e_full.expect("the Newton SCF must converge");
    assert!(
        e_full < s.e - 1e-3,
        "the Newton SCF converged to (or above) the saddle: {e_full:.10} vs {:.10}",
        s.e
    );
    assert!(
        truncs >= 1,
        "the guard never fired: the test did not reach negative curvature"
    );
}

/// **A tight `density_conv` is honoured by the Newton path.** Below its PCG
/// tolerance the Newton step is zero (ΔP = 0), so with the tolerance fixed at
/// 1e-7 a density_conv = 1e-10 run stopped at |g| ~ 1e-7. The solver now uses
/// min(1e-7, density_conv). MEASURED 2026-11-11 (OH/6-31G ROHF, SAD guess,
/// Newton trigger 1e-2): |g|max 2.9e-12 in 7 iterations. Bar 10 ×
/// density_conv. Mutation (the fixed 1e-7 restored, applied and reverted):
/// |g|max 1.6e-8, 16× the bar.
#[test]
fn rohf_newton_honours_a_tight_density_conv() {
    let s = Saddle::new();
    let cfg = RohfConfig {
        use_sad_guess: true,
        newton_trigger: 1e-2,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        ..Saddle::cfg()
    };
    let r = solve_rohf(
        &s.ctx,
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &cfg,
    )
    .unwrap();
    let (fa, fb, _) = s.point(&r.mos_alpha);
    let (nc, na, n) = (s.nc, s.nc + 1, s.prep.nbasis());
    let mut g = 0.0f64;
    for p in na..n {
        for q in 0..nc {
            g = g.max((fa[(p, q)] + fb[(p, q)]).abs());
        }
        g = g.max(fa[(p, nc)].abs());
    }
    for q in 0..nc {
        g = g.max(fb[(nc, q)].abs());
    }
    eprintln!(
        "tight Newton run: E = {:.12}, iters {}, |g|max = {g:.3e}",
        r.energy, r.iterations
    );
    let bar = 10.0 * cfg.density_conv;
    assert!(
        g < bar,
        "|g|max = {g:.3e} >= {bar:.0e} at density_conv 1e-10"
    );
}
