//! The Hylleraas mask certificate of the amplitude-threshold local MP2
//! (`ferric_mp2::lmp2_mask_certificate`; book §lmp2-mask-certificate).
//!
//! Protocol order: the trivial-limit anchor first (ε = 0: nothing dropped,
//! so estimate, lower bound and far term are exactly 0 and the upper bound
//! is the PCG residual's alone), then the enclosure of the TRUE mask error,
//! measured against an independent construction: the unmasked solve (ε = 0)
//! on the same localized spaces and integrals.
//!
//! Artifact hypothesis: if the certificate mis-assembles the residual (drops
//! the Fock coupling of the kept amplitudes into the dropped set), uses a
//! node above λ_min (the local Fock diagonal gap), or loses the far
//! integrals (no Gram term), the certificate is wrong. Recorded mutations
//! (2026-11-19, each applied to the production code and run against this
//! file; book §lmp2-mask-certificate-mutations):
//!
//! - `A_QP t_P` dropped from the residual on the dropped set: the upper
//!   bound falls below the true error in all three enclosure tests.
//! - the local Fock diagonal gap as the node (about twice the true gap
//!   here): every enclosure still HOLDS on these systems, so only
//!   `assert_node_is_the_canonical_gap` catches it, in three tests.
//! - the Gram far term dropped (`g_far_sq = 0`): the τ = ∞ far-only test
//!   fails (ε = 1e-4 with the Fock screen, U/err 0.96; at ε = 1e-3 the
//!   h-only far term is 0.38-0.41 of the error); the default-τ tests do not
//!   see it, because there the near set carries the error.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    amplitude_lmp2_with_virtuals, build_vvhv, AmplitudeLmp2Config, AmplitudeLmp2Result, VvHv,
};
use ferric_mp2::lmp2_mask_certificate::MaskCertificate;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
    vvhv: VvHv,
    frozen_core: usize,
}

fn setup(xyz: &str, frozen_core: usize) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();
    Setup {
        mol,
        obs,
        dfbs,
        rhf,
        vvhv,
        frozen_core,
    }
}

fn run(su: &Setup, eps: f64, fock_screen: Option<f64>, reference: bool) -> AmplitudeLmp2Result {
    run_tau(su, eps, fock_screen, reference, None)
}

fn run_tau(
    su: &Setup,
    eps: f64,
    fock_screen: Option<f64>,
    reference: bool,
    tau: Option<f64>,
) -> AmplitudeLmp2Result {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: su.frozen_core,
        fock_screen,
        compute_reference: reference,
        mask_certificate: true,
        mask_certificate_tau: tau,
        ..Default::default()
    };
    amplitude_lmp2_with_virtuals(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &su.vvhv,
    )
    .unwrap()
}

fn cert(r: &AmplitudeLmp2Result) -> &MaskCertificate {
    r.mask_certificate.as_ref().expect("certificate requested")
}

/// The Radau node must be λ_min(A) = 2(ε_LUMO − ε_HOMO) of the CANONICAL
/// RHF orbitals (an independent construction: the localized Fock blocks
/// are rotations of the canonical ones). The local Fock diagonal gap
/// 2(min f_aa − max f_ii) is larger and not a valid node.
fn assert_node_is_the_canonical_gap(su: &Setup, c: &MaskCertificate) {
    let e = su.rhf.eps_r();
    let nocc = (su.mol.nelec() / 2) as usize;
    let gap = 2.0 * (e[nocc] - e[nocc - 1]);
    assert!(
        (c.lambda_min - gap).abs() <= 1e-8 * gap,
        "node {:.10} is not the canonical gap {gap:.10}",
        c.lambda_min
    );
}

/// ε = 0 keeps every amplitude: the dropped set is empty, so the estimate,
/// the lower bound and the far term are EXACTLY zero, the upper bound is the
/// PCG residual's (below the solver-error bound), and E_P is canonical
/// RI-MP2 to the existing anchor tolerance.
#[test]
fn eps_zero_certificate_is_exactly_zero() {
    let su = setup("water.xyz", 1);
    let r = run(&su, 0.0, None, true);
    let c = cert(&r);
    assert!(
        (r.e_corr - r.e_corr_canonical_ri).abs() < 1e-9,
        "eps=0 anchor: {} vs {}",
        r.e_corr,
        r.e_corr_canonical_ri
    );
    assert_eq!(c.error_estimate, 0.0);
    assert_eq!(c.lower_bound, 0.0);
    assert_eq!(c.far_upper, 0.0);
    assert_eq!(c.g_far_sq, 0.0);
    assert_eq!((c.n_near, c.n_far), (0, 0));
    assert_node_is_the_canonical_gap(&su, c);
    assert!(
        c.upper_bound <= r.cg_energy_error_bound * (1.0 + 1e-9),
        "upper {:.3e} above the solver-error bound {:.3e}",
        c.upper_bound,
        r.cg_energy_error_bound
    );
}

/// One measured case: the certified quantities against the true error.
struct Case {
    err: f64,
    c: MaskCertificate,
    cg_star: f64,
}

fn case(su: &Setup, eps: f64, fock_screen: Option<f64>, e_star: &AmplitudeLmp2Result) -> Case {
    let r = run(su, eps, fock_screen, false);
    Case {
        err: r.e_corr - e_star.e_corr,
        c: cert(&r).clone(),
        cg_star: e_star.cg_energy_error_bound,
    }
}

/// The rigorous bounds enclose the TRUE mask error, `L ≤ err ≤ U`, and the
/// point estimate sits inside its measured band (x/err 0.92-0.96 on the
/// tight plain-mask cases, 0.19-0.65 on the Fock-screen cases, measured
/// 2026-11-19). `E*` is the unmasked
/// solve; its own PCG error (≤ its solver bound) only lowers the measured
/// error, so it widens the lower check alone.
fn assert_encloses(tag: &str, k: &Case, x_band: (f64, f64)) {
    let c = &k.c;
    eprintln!(
        "{tag}: err {:.4e}  x/err {:.3}  L/err {:.3}  U/err {:.3}  near/far {:.2e}/{:.2e}  \
         |N| {} |P| {} t {:.3}s",
        k.err,
        c.error_estimate / k.err,
        c.lower_bound / k.err,
        c.upper_bound / k.err,
        c.near_upper,
        c.far_upper,
        c.n_near,
        c.n_kept,
        c.t_s
    );
    assert!(k.err > 1e-8, "{tag}: error {:.2e} too small to test", k.err);
    assert!(
        k.err <= c.upper_bound,
        "{tag}: upper bound {:.4e} below the true error {:.4e}",
        c.upper_bound,
        k.err
    );
    assert!(
        c.lower_bound <= k.err + k.cg_star,
        "{tag}: lower bound {:.4e} above the true error {:.4e}",
        c.lower_bound,
        k.err
    );
    // measured 1.09-8.0 on these cases: a bound far above that has lost
    // its near/far split
    assert!(
        c.upper_bound <= 10.0 * k.err,
        "{tag}: upper bound {:.4e} is more than 10x the error {:.4e}",
        c.upper_bound,
        k.err
    );
    let x = c.error_estimate / k.err;
    assert!(
        x >= x_band.0 && x <= x_band.1,
        "{tag}: x/err = {x:.3} outside its measured band {x_band:?}"
    );
    assert!(c.e_corr_interval[0] <= c.e_corr_interval[1]);
}

#[test]
fn bounds_enclose_the_true_mask_error_butane() {
    let su = setup("alkane_4.xyz", 4);
    let e_star = run(&su, 0.0, None, false);
    for (eps, fock, band) in [
        (1e-6, None, (0.85, 1.0)),
        (1e-5, None, (0.8, 1.0)),
        (1e-5, Some(1.0), (0.5, 0.8)),
        (1e-4, Some(1.0), (0.1, 0.3)),
    ] {
        let tag = format!("butane eps {eps:.0e} fock {fock:?}");
        let k = case(&su, eps, fock, &e_star);
        assert_node_is_the_canonical_gap(&su, &k.c);
        assert_encloses(&tag, &k, band);
    }
}

/// The far term alone: with τ = ∞ the near set is empty and the upper bound
/// is `(√3‖g_far‖ + ‖h_far‖_M)² / λ_min` over the whole dropped set, the
/// Gram-identity norm of the dropped integrals included. It must still
/// enclose the true error (and the estimate and lower bound are 0).
#[test]
fn far_term_alone_encloses_the_true_mask_error_butane() {
    let su = setup("alkane_4.xyz", 4);
    let e_star = run(&su, 0.0, None, false);
    for (eps, fock) in [
        (1e-5, None),
        (1e-4, None),
        (1e-4, Some(1.0)),
        (1e-3, None),
        (1e-3, Some(1.0)),
    ] {
        let r = run_tau(&su, eps, fock, false, Some(f64::INFINITY));
        let c = cert(&r);
        let err = r.e_corr - e_star.e_corr;
        eprintln!(
            "butane tau=inf eps {eps:.0e} fock {fock:?}: err {err:.4e} U/err {:.2} \
             g_far^2 {:.3e} h_far_M^2 {:.3e} |far| {} (h_far alone)/err {:.3}",
            c.upper_bound / err,
            c.g_far_sq,
            c.h_far_m_sq,
            c.n_far,
            c.h_far_m_sq / c.lambda_min / err
        );
        assert_eq!(c.n_near, 0);
        assert_eq!((c.error_estimate, c.lower_bound), (0.0, 0.0));
        assert!(
            err <= c.upper_bound,
            "far-only upper bound {:.4e} below the true error {err:.4e}",
            c.upper_bound
        );
    }
}

#[test]
fn bounds_enclose_the_true_mask_error_benzene() {
    let su = setup("benzene.xyz", 6);
    let e_star = run(&su, 0.0, None, false);
    for (eps, fock, band) in [(1e-6, None, (0.85, 1.0)), (1e-4, Some(1.0), (0.1, 0.35))] {
        let tag = format!("benzene eps {eps:.0e} fock {fock:?}");
        let k = case(&su, eps, fock, &e_star);
        assert_node_is_the_canonical_gap(&su, &k.c);
        assert_encloses(&tag, &k, band);
    }
}

/// The measurement behind the book's tables (§lmp2-mask-certificate-measured):
/// octane and naphthalene, 6-31G / cc-pVDZ-RI, frozen core, both mask arms
/// (plain Eq-8 and the Fock screen κ = 1), ε = 1e-3 … 1e-7, plain Boys
/// (λ = 0, the prototype's localization) and λ-Boys (λ = 10, the default).
/// Prints one markdown row per run: the true error against the unmasked
/// solve, the certificate ratios, the near-set size and the cost of the
/// certificate against the masked solve. Run with `--ignored --nocapture`
/// under `box-lease run`.
#[test]
#[ignore = "measurement: ~minutes; prints the book's tables"]
fn measure_mask_certificate_octane_naphthalene() {
    println!(
        "| system | lambda | arm | eps | keep | err Ha | x/err | L/err | U/err | |N|/|P| | \
         near up | far up | t_assembly s | t_solve s | t_cert s |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, xyz, fc) in [
        ("octane", "alkane_8.xyz", 8),
        ("naphthalene", "scaling/naphthalene.xyz", 10),
    ] {
        let su = setup(xyz, fc);
        for lambda in [0.0, ferric_mp2::boys::LAMBDA_BOYS_DEFAULT] {
            let run_l = |eps: f64, fock: Option<f64>| {
                let cfg = AmplitudeLmp2Config {
                    eps,
                    frozen_core: su.frozen_core,
                    fock_screen: fock,
                    lambda_boys: lambda,
                    mask_certificate: true,
                    compute_reference: eps == 0.0,
                    ..Default::default()
                };
                amplitude_lmp2_with_virtuals(
                    &su.mol,
                    &su.obs,
                    &su.dfbs,
                    Operator::coulomb(),
                    &su.rhf,
                    &cfg,
                    &su.vvhv,
                )
                .unwrap()
            };
            let star = run_l(0.0, None);
            println!(
                "anchor {name} lambda {lambda}: e_corr(eps=0) - canonical RI = {:.2e} Ha, \
                 upper {:.2e}, cg bound {:.2e}",
                star.e_corr - star.e_corr_canonical_ri,
                cert(&star).upper_bound,
                star.cg_energy_error_bound
            );
            for (arm, fock) in [("eq8", None), ("fock", Some(1.0))] {
                for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
                    let r = run_l(eps, fock);
                    let c = cert(&r);
                    let err = r.e_corr - star.e_corr;
                    assert!(
                        c.upper_bound >= err && c.lower_bound <= err + star.cg_energy_error_bound
                    );
                    println!(
                        "| {name} | {lambda} | {arm} | {eps:.0e} | {:.3} | {err:.3e} | {:.3} | \
                         {:.3} | {:.2} | {:.3} | {:.2e} | {:.2e} | {:.2} | {:.2} | {:.2} |",
                        r.keep_fraction,
                        c.error_estimate / err,
                        c.lower_bound / err,
                        c.upper_bound / err,
                        c.n_near as f64 / c.n_kept as f64,
                        c.near_upper,
                        c.far_upper,
                        r.timings.t_assembly_s,
                        r.timings.t_solve_s,
                        c.t_s
                    );
                }
            }
        }
    }
}
