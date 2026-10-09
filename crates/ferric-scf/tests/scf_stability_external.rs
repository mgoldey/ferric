//! RHF→UHF (external, triplet) stability: does ferric know a SPIN-BROKEN
//! saddle from a minimum?
//!
//! # The gap this closes
//!
//! `ferric_scf::stability::rhf_internal_stability` answers "is this RHF
//! solution an RHF minimum?" — the singlet channel, κ_α = +κ_β. It is
//! structurally BLIND to the classic stretched-bond instability, which lives
//! in the triplet channel κ_β = −κ_α where the Coulomb response cancels.
//! Measured (`testdata/reference/validation/scf_stability/
//! water_stretched_6-31g.json`): water at r(OH) = 2.0 Å / 6-31G has singlet
//! λ_min = +1.9710e-2 (STABLE) and triplet λ_min = −3.0724e-1 (UNSTABLE). A
//! user who set `check_stability` was told STABLE about a saddle point whose
//! broken-symmetry UHF state lies 0.222 Ha lower.
//!
//! # Test design (protocol order is deliberate)
//!
//! 1. **EXACTNESS ANCHOR FIRST** —
//!    [`anchor_triplet_matvec_equals_the_triplet_block_of_the_uhf_hessian_at_the_rhf_point`].
//!    The UHF Hessian evaluated at an RHF point (C_α = C_β = C_RHF,
//!    F_α = F_β = F_RHF) block-diagonalizes EXACTLY into the singlet (κ, +κ)/√2
//!    and triplet (κ, −κ)/√2 channels. Its triplet block is therefore the same
//!    operator `triplet_hessian_matvec` builds — reached through a COMPLETELY
//!    DIFFERENT code path (`uhf_newton::hessian_matvec`, two Fock matrices, two
//!    density blocks, δJ present and cancelling numerically rather than being
//!    absent algebraically). Two independent constructions of one operator is
//!    the only kind of agreement that separates signal from SYSTEMATIC error
//!    (Experimental Protocol: "CONSISTENCY IS NOT CORROBORATION"). This test
//!    was written and seen passing before any λ value below was believed.
//! 2. **BOTH VERDICTS REACHABLE** — a checker that always says one thing is
//!    arithmetic. [`water_equilibrium_is_externally_stable`] (positive) and
//!    [`stretched_h2_is_externally_unstable_while_internally_stable`] (negative)
//!    prove both, on real converged SCF states, through the library function.
//! 3. **THE DECISIVE TEST** — the negative control of the whole feature:
//!    internal says STABLE and external says UNSTABLE on the SAME state. If
//!    the external path silently computed the singlet operator, the two
//!    verdicts would agree and this test would fail.
//! 4. **CHANNEL IDENTITY** —
//!    [`the_singlet_operator_is_not_the_triplet_operator`]: the two matvecs,
//!    given the same κ, must produce DIFFERENT vectors, and the difference
//!    must be exactly δJ. This localizes a wrong-operator defect to the one
//!    term that distinguishes the channels.
//! 5. **WIRING** — [`check_stability_populates_both_verdicts`] and
//!    [`stability_external_off_is_bit_identical`]: the SCF path must carry
//!    BOTH verdicts out, and the flag off must cost nothing.
//!
//! # Artifact hypothesis (written before measuring)
//!
//! If the triplet operator is REAL I expect: λ_min > 0 on equilibrium water,
//! λ_min < 0 on stretched H₂ and stretched water, agreement with the
//! UHF-at-RHF triplet block at the integral-threshold floor, and the two
//! channels differing by exactly δJ.
//!
//! If it is BROKEN in the most likely ways:
//!
//! * **δJ not actually removed** (the copy-paste defect): the triplet λ equals
//!   the SINGLET λ, every verdict agrees with the internal one, and the anchor
//!   misses the UHF-at-RHF triplet block by the size of δJ. Caught by the
//!   anchor AND by test 4.
//! * **wrong factor on K** (½ dropped, or K doubled): every eigenvalue misses
//!   by tens of percent; the anchor fails on every system. A factor defect
//!   cannot hide, because the anchor compares against an independently
//!   normalized construction.
//! * **sign of the exchange term flipped**: equilibrium water comes out
//!   UNSTABLE, so test 2's positive case fails.
//! * **the analysis is run at a non-stationary point** (wrong Fock/MO pair):
//!   the λ values disagree with the in-tree PySCF references while keeping a
//!   plausible sign. That is why [`water_equilibrium_is_externally_stable`]
//!   asserts the NUMBER, not merely the sign.
//!
//! # Fast tier
//!
//! Everything here is H₂ or water in STO-3G / 6-31G and runs in seconds, so
//! none of it is `#[ignore]`d. The PySCF cross-validation at the full
//! reference tolerance lives in `validation_scf_stability.rs`
//! (`#[ignore = "validation: SCF stability"]`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::RhfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{
    rhf_external_stability, rhf_internal_stability, StabilityConfig, StabilityKind,
    StabilityVerdict,
};
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// Equilibrium water (the geometry `scf_stability.rs` and
/// `scf_stability_wiring.rs` both use, so λ values are comparable across the
/// three files).
const WATER: &str =
    "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";

/// H₂ at 2.0 Å — well past the Coulson–Fischer point, so RHF is internally
/// stable and externally a saddle. The smallest system that exhibits the
/// defect, which is what makes this file's negative control fast.
const H2_STRETCHED: &str = "2\nH2 stretched\nH 0.0 0.0 0.0\nH 0.0 0.0 2.0\n";

/// The validation row's geometries, read from the SAME xyz files the PySCF
/// references were generated from. They are read rather than inlined because
/// this file asserts λ against those references at 1e-8, and λ is
/// geometry-dependent: a transcribed geometry that differs in the 5th decimal
/// would fail the comparison for a reason that has nothing to do with the
/// operator under test.
const WATER_EQ_XYZ: &str = "testdata/molecules/validation/water_eq.xyz";
const WATER_STRETCHED_XYZ: &str = "testdata/molecules/validation/water_stretched.xyz";

/// The workspace root, found by walking up from the manifest dir (same rule as
/// `validation_scf_stability.rs`).
fn workspace_root() -> std::path::PathBuf {
    let looks_like_root = |p: &std::path::Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&std::path::Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

fn read_xyz(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — this test needs the validation geometry the \
             PySCF reference was generated from",
            path.display()
        )
    })
}

/// Bar on "two independent constructions of the same operator agree".
///
/// DERIVED, not guessed. Both constructions call `build_jk_with_pool` at
/// `integral_thresh = 1e-12` on the same density perturbation, so they differ
/// only by floating-point association in the UHF path's extra δJ
/// add-and-cancel. MEASURED max |H_triplet − H_UHF@RHF_triplet| element-wise
/// (2026-10-03, this box, OPENBLAS_NUM_THREADS=1): 3.553e-15 (water/STO-3G,
/// operator scale 2.01e1), 5.551e-17 (H₂ 2.0 Å/STO-3G, scale 4.00e-1),
/// 3.553e-15 (water/6-31G, scale 2.14e1) — the anchor's own eprintln prints
/// all three. That is ~1.7e-16 RELATIVE to the operator scale, i.e. machine
/// epsilon, which is what two valid constructions of one operator should give.
/// The bar is ~10x the largest measured absolute value, and still ~1e11 below
/// the smallest signal this file asserts (the H₂ triplet λ, −4.00e-1).
const TOL_ANCHOR: f64 = 4e-14;

/// Bar on the Davidson λ_min vs ferric's own dense λ_min for the SAME operator
/// (same matvec, different eigensolver). The Davidson converges on a 1e-6
/// residual, which gives a λ error of order r²/gap. MEASURED (same run as
/// `TOL_ANCHOR`): 3.16e-15 (water/STO-3G), 9.55e-14 (H₂ 2.0 Å/STO-3G, a
/// 1-dimensional rotation space), 8.33e-16 (water/6-31G). Kept at 1e-9 — the
/// same bar `validation_scf_stability.rs::TOL_DAVIDSON_VS_DENSE` uses for the
/// singlet and UHF channels — rather than tightened to the measured max,
/// because the quantity bounded here is the EIGENSOLVER's convergence, whose
/// 1e-6 residual threshold is a configured knob: tightening the bar to 1e-13
/// would make this test fail on a legitimate `StabilityConfig` change instead
/// of on a defect.
const TOL_DAVIDSON_VS_DENSE: f64 = 1e-9;

/// Bar on reproducing an in-tree PySCF reference λ. Same provenance and
/// reasoning as `validation_scf_stability.rs::TOL_LAMBDA` (measured max there
/// 4.5e-10): the references are converged to conv_tol_grad 1e-10, Hessian
/// eigenvalues are FIRST order in the orbital error, and this file converges
/// its own SCF to density_conv 1e-10. MEASURED (2026-10-03): stretched water
/// 6-31G singlet 5.62e-11, triplet 1.35e-10; equilibrium water 6-31G triplet
/// 2.1e-12. Bar ~10x above the largest, and ~2e6 below the smallest signal
/// asserted (the stretched-water singlet λ, +1.97e-2).
const TOL_LAMBDA_REF: f64 = 1e-8;

/// A quantity that must be MISSED is missed by at least this factor × its bar.
/// Same convention as `validation_scf_stability.rs`.
const MUST_MISS_FACTOR: f64 = 1000.0;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Rhf {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    res: ScfResult,
    c: Array2<f64>,
    f_mo: Array2<f64>,
    nocc: usize,
}

fn tight(check_stability: bool) -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        // The Hessian eigenvalues are FIRST order in the orbital error, and
        // this file asserts them against PySCF at 1e-8.
        density_conv: 1e-10,
        max_iter: 400,
        check_stability,
        ..Default::default()
    }
}

/// Converge RHF and rebuild the MO-basis Fock AT the converged orbitals (the
/// Hessian is evaluated at the stationary point, so the Fock must correspond
/// to exactly the MOs handed to the matvec).
fn converge_rhf(xyz: &str, basis_name: &str, check_stability: bool) -> Rhf {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_rhf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &tight(check_stability),
    )
    .unwrap();
    assert!(res.converged, "{xyz} / {basis_name}: SCF did not converge");
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let nocc = mol.nelec() as usize / 2;
    Rhf {
        mol,
        prep,
        bounds,
        ctx,
        res,
        c,
        f_mo,
        nocc,
    }
}

fn rhf_inputs<'a>(s: &'a Rhf) -> RhfNewtonInputs<'a> {
    RhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c: &s.c,
        f_mo: &s.f_mo,
        nocc: s.nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    }
}

/// The UHF Newton inputs for the SAME state, with C_α = C_β = C_RHF and
/// F_α = F_β = F_RHF. This is the independent construction the anchor rests on.
fn uhf_inputs_at_rhf_point<'a>(s: &'a Rhf) -> UhfNewtonInputs<'a> {
    UhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c_a: &s.c,
        c_b: &s.c,
        f_a_mo: &s.f_mo,
        f_b_mo: &s.f_mo,
        nocc_a: s.nocc,
        nocc_b: s.nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    }
}

fn sym_eigh(h: &Array2<f64>, what: &str) -> Vec<f64> {
    let asym = (h - &h.t()).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(asym < 1e-8, "{what}: Hessian not symmetric ({asym:.2e})");
    let (ev, _) = (0.5 * (h + &h.t())).eigh(UPLO::Lower).unwrap();
    ev.to_vec()
}

/// Dense triplet Hessian from ferric's triplet matvec (apply to every unit
/// vector). An independent EIGENSOLVER for the same operator the Davidson
/// drives.
fn dense_triplet(s: &Rhf, inp: &RhfNewtonInputs) -> Array2<f64> {
    let n = s.c.nrows();
    let (no, nv) = (s.nocc, n - s.nocc);
    let dim = nv * no;
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut v = vec![0.0; dim];
        v[col] = 1.0;
        let k = Array2::from_shape_vec((nv, no), v).unwrap();
        let hv = ferric_scf::rhf_newton::triplet_hessian_matvec(&s.ctx, inp, &k, &pool).unwrap();
        for (r, x) in hv.iter().enumerate() {
            h[(r, col)] = *x;
        }
    }
    h
}

/// Dense UHF Hessian at the RHF point, then its TRIPLET block in the
/// (κ, −κ)/√2 basis. Mirrors `validation_scf_stability.rs::singlet_triplet`.
/// Returns `(triplet_block, singlet_block, max_coupling)`.
fn dense_uhf_channels(s: &Rhf, inp: &UhfNewtonInputs) -> (Array2<f64>, Array2<f64>, f64) {
    let n = s.c.nrows();
    let (no, nv) = (s.nocc, n - s.nocc);
    let d = nv * no;
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((2 * d, 2 * d));
    for col in 0..2 * d {
        let mut v = vec![0.0; 2 * d];
        v[col] = 1.0;
        let ka = Array2::from_shape_vec((nv, no), v[..d].to_vec()).unwrap();
        let kb = Array2::from_shape_vec((nv, no), v[d..].to_vec()).unwrap();
        let (ha, hb) =
            ferric_scf::uhf_newton::hessian_matvec(&s.ctx, inp, &ka, &kb, &pool).unwrap();
        for (r, x) in ha.iter().chain(hb.iter()).enumerate() {
            h[(r, col)] = *x;
        }
    }
    let aa = h.slice(s![..d, ..d]);
    let ab = h.slice(s![..d, d..]);
    let ba = h.slice(s![d.., ..d]);
    let bb = h.slice(s![d.., d..]);
    let singlet = 0.5 * ((&aa + &ab) + (&ba + &bb));
    let triplet = 0.5 * ((&aa - &ab) - (&ba - &bb));
    let coupling = 0.5 * ((&aa - &ab) + (&ba - &bb));
    let cmax = coupling.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    (triplet, singlet, cmax)
}

// ===========================================================================
// 1. EXACTNESS ANCHOR — two independent constructions of one operator.
// ===========================================================================

/// THE ANCHOR. `triplet_hessian_matvec` must reproduce the TRIPLET BLOCK of
/// the UHF Hessian evaluated at the RHF point, element by element.
///
/// These are genuinely independent constructions, which is the whole point:
///
/// * ferric's triplet matvec OMITS δJ algebraically and applies `−½K(δD)` to
///   one block.
/// * the UHF matvec BUILDS δJ from `δD_α + δD_β` and lets it cancel
///   numerically when the test rotates into the (κ, −κ)/√2 basis, and it reads
///   its gaps from two separate Fock matrices.
///
/// A defect shared by both would have to be a defect of `build_jk_with_pool`
/// itself, which the J/K suite covers separately. A defect in EITHER triplet
/// construction shows up here as a mismatch of order δJ or of order the
/// dropped factor.
///
/// The test also asserts the premise the block decomposition rests on — that
/// the singlet/triplet COUPLING block vanishes — because if it did not, "the
/// triplet block" would not be a well-defined operator and the comparison
/// would be meaningless.
#[test]
fn anchor_triplet_matvec_equals_the_triplet_block_of_the_uhf_hessian_at_the_rhf_point() {
    for (label, xyz, basis_name) in [
        ("water/STO-3G", WATER, "sto-3g"),
        ("H2 2.0A/STO-3G", H2_STRETCHED, "sto-3g"),
        ("water/6-31G", WATER, "6-31g"),
    ] {
        let s = converge_rhf(xyz, basis_name, false);
        let rinp = rhf_inputs(&s);
        let uinp = uhf_inputs_at_rhf_point(&s);

        let h_trip = dense_triplet(&s, &rinp);
        let (h_uhf_trip, _singlet, coupling) = dense_uhf_channels(&s, &uinp);

        // PREMISE: the two channels must not couple, or "the triplet block" is
        // not a well-defined operator.
        assert!(
            coupling < 1e-10,
            "{label}: singlet/triplet coupling block max |.| = {coupling:.2e} — the UHF \
             Hessian at an RHF point must block-diagonalize, so the triplet block the \
             anchor compares against is not well defined"
        );

        let max_dev = (&h_trip - &h_uhf_trip)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        let scale = h_trip.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        eprintln!(
            "ANCHOR {label}: dim {}, max |H_triplet - H_UHF@RHF_triplet| = {max_dev:.3e} \
             (operator scale {scale:.3e}, bar {TOL_ANCHOR:.0e})",
            h_trip.nrows()
        );
        assert!(
            max_dev < TOL_ANCHOR,
            "{label}: ferric's triplet matvec and the triplet block of the UHF Hessian at \
             the RHF point are two constructions of ONE operator; they differ by \
             {max_dev:.3e} (bar {TOL_ANCHOR:.0e})"
        );

        // And the eigenvalues, which is what the verdict reads.
        let spec_t = sym_eigh(&h_trip, label);
        let spec_u = sym_eigh(&h_uhf_trip, label);
        for k in 0..spec_t.len().min(6) {
            assert!(
                (spec_t[k] - spec_u[k]).abs() < TOL_ANCHOR,
                "{label}: triplet eigenvalue[{k}] {:+.12e} vs UHF-block {:+.12e}",
                spec_t[k],
                spec_u[k]
            );
        }

        // The Davidson must find the same lowest eigenvalue the dense
        // construction does (independent eigensolver, same operator).
        let st = rhf_external_stability(&s.ctx, &rinp, &StabilityConfig::default()).unwrap();
        assert_eq!(st.kind, StabilityKind::RhfExternalTriplet);
        assert!(st.converged, "{label}: {}", st.summary());
        eprintln!(
            "ANCHOR {label}: Davidson {:+.12e} vs dense {:+.12e} (|d| {:.2e})",
            st.lowest_eigenvalue,
            spec_t[0],
            (st.lowest_eigenvalue - spec_t[0]).abs()
        );
        assert!(
            (st.lowest_eigenvalue - spec_t[0]).abs() < TOL_DAVIDSON_VS_DENSE,
            "{label}: Davidson lambda_min {:+.12e} vs dense {:+.12e}",
            st.lowest_eigenvalue,
            spec_t[0]
        );
        assert!(
            st.eigenvector_beta.is_none(),
            "{label}: the triplet beta block is -alpha by construction and must not be \
             stored separately"
        );
    }
}

// ===========================================================================
// 2+3. BOTH VERDICTS REACHABLE, and the DECISIVE negative control.
// ===========================================================================

/// POSITIVE case: equilibrium water is externally STABLE, with the number from
/// the in-tree PySCF reference (`water_eq_6-31g.json`
/// `/rhf/triplet_rhf_to_uhf/lowest[0]` = +2.8444621961e-1).
///
/// The NUMBER matters, not just the sign: a sign-only assertion cannot
/// distinguish "the triplet operator is right" from "the triplet operator is
/// the singlet one" (the singlet λ here is +3.6017e-1, also positive). The
/// `must_miss` below is that discrimination made explicit.
#[test]
fn water_equilibrium_is_externally_stable() {
    /// PySCF `hop_rhf2uhf`, water/6-31G equilibrium, from
    /// `testdata/reference/validation/scf_stability/water_eq_6-31g.json`
    /// `/rhf/triplet_rhf_to_uhf/lowest[0]`.
    const TRIPLET_REF: f64 = 0.28444621961166305;
    /// Same file, `/rhf/singlet_rhf_internal/lowest[0]`. The value the triplet
    /// λ must MISS.
    const SINGLET_REF: f64 = 0.3601655713576944;

    let s = converge_rhf(&read_xyz(WATER_EQ_XYZ), "6-31g", false);
    let inp = rhf_inputs(&s);
    let st = rhf_external_stability(&s.ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("water_eq/6-31G external: {}", st.summary());
    assert!(st.converged, "{}", st.summary());
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "equilibrium water must be externally stable: {}",
        st.summary()
    );
    let d = (st.lowest_eigenvalue - TRIPLET_REF).abs();
    eprintln!(
        "water_eq/6-31G triplet lambda_min ferric {:+.12e} PySCF {TRIPLET_REF:+.12e} \
         |d| {d:.2e} (bar {TOL_LAMBDA_REF:.0e})",
        st.lowest_eigenvalue
    );
    assert!(
        d < TOL_LAMBDA_REF,
        "triplet lambda_min {:+.12e} vs PySCF hop_rhf2uhf {TRIPLET_REF:+.12e} (|d| {d:.2e})",
        st.lowest_eigenvalue
    );
    // CHANNEL GUARD: the singlet value is also positive here, so a test that
    // only checked the sign would pass with the wrong operator.
    let d_wrong = (st.lowest_eigenvalue - SINGLET_REF).abs();
    assert!(
        d_wrong > MUST_MISS_FACTOR * TOL_LAMBDA_REF,
        "the triplet lambda {:+.12e} is within {:.0e} of the SINGLET reference \
         {SINGLET_REF:+.12e} — this comparison cannot tell the two operators apart",
        st.lowest_eigenvalue,
        MUST_MISS_FACTOR * TOL_LAMBDA_REF
    );
}

/// THE DECISIVE TEST and the negative control of the whole feature: stretched
/// H₂ is internally STABLE and externally UNSTABLE, from the SAME converged
/// state.
///
/// This is the bug in issue #289 reduced to its smallest instance. If the
/// external path computed the singlet operator (δJ left in), both verdicts
/// would read STABLE and this test would fail on the `Unstable` assertion.
#[test]
fn stretched_h2_is_externally_unstable_while_internally_stable() {
    let s = converge_rhf(H2_STRETCHED, "sto-3g", false);
    let inp = rhf_inputs(&s);

    let internal = rhf_internal_stability(&s.ctx, &inp, &StabilityConfig::default()).unwrap();
    let external = rhf_external_stability(&s.ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("H2 2.0A/STO-3G internal: {}", internal.summary());
    eprintln!("H2 2.0A/STO-3G external: {}", external.summary());
    assert!(internal.converged && external.converged);

    assert_eq!(
        internal.verdict(),
        StabilityVerdict::Stable,
        "stretched H2 RHF is internally (singlet) stable — that is exactly why an \
         internal-only check misses it: {}",
        internal.summary()
    );
    assert_eq!(
        external.verdict(),
        StabilityVerdict::Unstable,
        "stretched H2 RHF MUST be externally (RHF->UHF) unstable past the Coulson-Fischer \
         point: {}",
        external.summary()
    );
    assert!(
        external.lowest_eigenvalue < 0.0 && internal.lowest_eigenvalue > 0.0,
        "expected triplet < 0 < singlet, got triplet {:+.6e} singlet {:+.6e}",
        external.lowest_eigenvalue,
        internal.lowest_eigenvalue
    );
}

/// The same negative control on the system the issue reports, at the reference
/// basis, against the in-tree PySCF numbers for BOTH channels.
#[test]
fn stretched_water_reproduces_the_reference_internal_stable_external_unstable() {
    /// `water_stretched_6-31g.json` `/rhf/singlet_rhf_internal/lowest[0]`.
    const SINGLET_REF: f64 = 0.019710054818436733;
    /// `water_stretched_6-31g.json` `/rhf/triplet_rhf_to_uhf/lowest[0]`.
    const TRIPLET_REF: f64 = -0.30724190949739083;

    let s = converge_rhf(&read_xyz(WATER_STRETCHED_XYZ), "6-31g", false);
    let inp = rhf_inputs(&s);
    let internal = rhf_internal_stability(&s.ctx, &inp, &StabilityConfig::default()).unwrap();
    let external = rhf_external_stability(&s.ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("water 2.0A/6-31G internal: {}", internal.summary());
    eprintln!("water 2.0A/6-31G external: {}", external.summary());
    assert!(internal.converged && external.converged);

    let di = (internal.lowest_eigenvalue - SINGLET_REF).abs();
    let de = (external.lowest_eigenvalue - TRIPLET_REF).abs();
    eprintln!(
        "water 2.0A/6-31G singlet ferric {:+.12e} PySCF {SINGLET_REF:+.12e} |d| {di:.2e}; \
         triplet ferric {:+.12e} PySCF {TRIPLET_REF:+.12e} |d| {de:.2e} \
         (bar {TOL_LAMBDA_REF:.0e})",
        internal.lowest_eigenvalue, external.lowest_eigenvalue
    );
    assert!(di < TOL_LAMBDA_REF, "singlet lambda_min off by {di:.2e}");
    assert!(de < TOL_LAMBDA_REF, "triplet lambda_min off by {de:.2e}");

    // THE BUG, asserted: internal STABLE, external UNSTABLE, same state.
    assert_eq!(internal.verdict(), StabilityVerdict::Stable);
    assert_eq!(external.verdict(), StabilityVerdict::Unstable);
}

// ===========================================================================
// 4. CHANNEL IDENTITY — the two operators differ by exactly delta-J.
// ===========================================================================

/// The singlet and triplet matvecs, given the SAME κ, must differ — and the
/// difference must be exactly the δJ term the triplet channel drops, projected
/// to the occ→virt block.
///
/// This localizes a wrong-operator defect: the anchor proves the triplet
/// matvec equals an independent triplet construction, and this proves the
/// triplet operator is not the singlet one *and* names the one term that
/// separates them. Without it, "the external path returns the singlet number"
/// could only be caught indirectly, through a verdict that happened to differ.
#[test]
fn the_singlet_operator_is_not_the_triplet_operator() {
    use ferric_scf::rhf::build_jk;

    let s = converge_rhf(WATER, "sto-3g", false);
    let inp = rhf_inputs(&s);
    let n = s.c.nrows();
    let (no, nv) = (s.nocc, n - s.nocc);
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();

    // A generic (not unit) rotation, so no single matrix element can be
    // accidentally zero.
    let mut k = Array2::<f64>::zeros((nv, no));
    for (ir, row) in k.rows_mut().into_iter().enumerate() {
        for (ic, v) in row.into_iter().enumerate() {
            *v = 0.1 * ((ir + 1) as f64) - 0.03 * ((ic + 1) as f64);
        }
    }

    let h_s = ferric_scf::rhf_newton::hessian_matvec(&s.ctx, &inp, &k, &pool).unwrap();
    let h_t = ferric_scf::rhf_newton::triplet_hessian_matvec(&s.ctx, &inp, &k, &pool).unwrap();
    let diff_norm = (&h_s - &h_t).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let t_norm = h_t.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    eprintln!(
        "CHANNEL: max |H_singlet.k - H_triplet.k| = {diff_norm:.3e} (triplet scale \
         {t_norm:.3e})"
    );
    assert!(
        diff_norm > 1e-3 * t_norm,
        "the singlet and triplet matvecs produced nearly the same vector \
         ({diff_norm:.3e} vs scale {t_norm:.3e}) — the external path is computing the \
         internal operator"
    );

    // And the difference IS delta-J: build it independently.
    let mut dd_mo = Array2::<f64>::zeros((n, n));
    for (ir, a) in (no..n).enumerate() {
        for i in 0..no {
            dd_mo[(a, i)] = k[(ir, i)];
            dd_mo[(i, a)] = k[(ir, i)];
        }
    }
    let dd_ao = 2.0 * s.c.dot(&dd_mo).dot(&s.c.t());
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk = Array2::<f64>::zeros((n, n));
    build_jk(
        &s.ctx,
        &s.prep,
        &s.bounds,
        RhfConfig::default().integral_thresh,
        &dd_ao,
        &mut dj,
        &mut dk,
    )
    .unwrap();
    let dj_mo = s.c.t().dot(&dj).dot(&s.c);
    let mut dev = 0.0_f64;
    for (ir, a) in (no..n).enumerate() {
        for i in 0..no {
            dev = dev.max((h_s[(ir, i)] - h_t[(ir, i)] - dj_mo[(a, i)]).abs());
        }
    }
    eprintln!("CHANNEL: max |(H_s - H_t) - deltaJ_ov| = {dev:.3e}");
    assert!(
        dev < 1e-9,
        "the singlet-minus-triplet difference is not the delta-J term \
         (max deviation {dev:.3e}): the triplet channel must drop exactly the Coulomb \
         response and nothing else"
    );
}

/// A KS reference must be REFUSED, not analysed with the singlet XC kernel.
///
/// The gate is on `inp.fxc.is_some()`, so this test can exercise it with a
/// trivial (zero) response closure — what matters is that a kernel was
/// supplied at all, because supplying one means a semilocal response term
/// belongs in the Hessian and the triplet combination `f_αα − f_αβ` is not
/// what a closed-shell `FxcResponse` returns.
#[test]
fn a_ks_reference_is_refused_rather_than_analysed_with_the_singlet_kernel() {
    let s = converge_rhf(WATER, "sto-3g", false);
    let n = s.c.nrows();
    let zero_response = move |_da: &Array2<f64>, _db: &Array2<f64>| {
        (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)))
    };
    let response: &ferric_scf::rohf_newton::FxcResponse<'_> = &zero_response;
    let mut inp = rhf_inputs(&s);
    inp.fxc = Some(response);

    let err = rhf_external_stability(&s.ctx, &inp, &StabilityConfig::default())
        .expect_err("a KS reference must be refused, not analysed");
    let msg = format!("{err:?}");
    eprintln!("KS REFUSAL: {msg}");
    assert!(
        msg.contains("f_aa - f_ab"),
        "the refusal must name the missing triplet kernel; got: {msg}"
    );
    // The INTERNAL check on the same inputs still works — the refusal is
    // scoped to the external channel, it does not disable stability analysis.
    let internal = rhf_internal_stability(&s.ctx, &inp, &StabilityConfig::default())
        .expect("the internal (singlet) check accepts a KS reference");
    assert!(internal.converged, "{}", internal.summary());
}

// ===========================================================================
// 5. WIRING — the SCF path must carry BOTH verdicts, and off must be free.
// ===========================================================================

/// `check_stability` must populate BOTH `ScfResult::stability` (internal) and
/// `ScfResult::stability_external`, and on stretched water they must DISAGREE.
///
/// This is the test that fails if `check_stability` is wired to the internal
/// check only — mutation (c) in the issue's ledger.
#[test]
fn check_stability_populates_both_verdicts() {
    let s = converge_rhf(&read_xyz(WATER_STRETCHED_XYZ), "6-31g", true);
    let internal = s
        .res
        .stability
        .as_ref()
        .expect("check_stability set: the internal verdict must be present");
    let external = s.res.stability_external.as_ref().expect(
        "check_stability set on an RHF/HF run: the EXTERNAL (RHF->UHF) verdict must be \
         present too — an internal-only check reports this saddle as STABLE",
    );
    eprintln!("WIRING internal: {}", internal.summary());
    eprintln!("WIRING external: {}", external.summary());
    assert_eq!(internal.kind, StabilityKind::RhfInternal);
    assert_eq!(external.kind, StabilityKind::RhfExternalTriplet);
    assert_eq!(internal.verdict(), StabilityVerdict::Stable);
    assert_eq!(
        external.verdict(),
        StabilityVerdict::Unstable,
        "the SCF path must report this state UNSTABLE: {}",
        external.summary()
    );
    // The two verdicts must come from DIFFERENT operators, not the same one
    // stored twice.
    assert!(
        (internal.lowest_eigenvalue - external.lowest_eigenvalue).abs() > 1e-3,
        "the two verdicts carry the same lambda_min ({:+.6e} vs {:+.6e}) — one operator \
         is being stored under both fields",
        internal.lowest_eigenvalue,
        external.lowest_eigenvalue
    );
}

/// Both verdicts must be reachable THROUGH THE SCF PATH as STABLE too, or the
/// external check is a constant.
#[test]
fn the_external_verdict_is_not_constant_through_the_scf_path() {
    let stable = converge_rhf(&read_xyz(WATER_EQ_XYZ), "6-31g", true);
    let unstable = converge_rhf(&read_xyz(WATER_STRETCHED_XYZ), "6-31g", true);
    let a = stable.res.stability_external.as_ref().unwrap();
    let b = unstable.res.stability_external.as_ref().unwrap();
    eprintln!("NOT-CONSTANT eq: {}", a.summary());
    eprintln!("NOT-CONSTANT stretched: {}", b.summary());
    assert_eq!(a.verdict(), StabilityVerdict::Stable);
    assert_eq!(b.verdict(), StabilityVerdict::Unstable);
}

/// The trivial limit: `check_stability = false` must leave the converged
/// answer BIT-IDENTICAL and both verdict fields `None`.
///
/// Same discipline as `scf_stability_wiring.rs::stability_off_is_bit_identical_rhf`,
/// extended to the new field: asserted with `==` on `f64` bits, because
/// "close" would not catch an external check that runs and feeds its rotation
/// back into the SCF.
#[test]
fn stability_external_off_is_bit_identical() {
    let off = converge_rhf(WATER, "sto-3g", false);
    let on = converge_rhf(WATER, "sto-3g", true);
    assert_eq!(
        off.res.energy.to_bits(),
        on.res.energy.to_bits(),
        "turning check_stability ON must not change the converged energy by a single bit: \
         off = {:.16e} ({:#x}), on = {:.16e} ({:#x})",
        off.res.energy,
        off.res.energy.to_bits(),
        on.res.energy,
        on.res.energy.to_bits()
    );
    assert_eq!(off.res.iterations, on.res.iterations);
    for (x, y) in off
        .res
        .density_total
        .iter()
        .zip(on.res.density_total.iter())
    {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "converged density must be bit-identical with the flag on"
        );
    }
    assert!(
        off.res.stability.is_none() && off.res.stability_external.is_none(),
        "flag off must leave BOTH verdict fields None (not checked)"
    );
    assert!(
        on.res.stability.is_some() && on.res.stability_external.is_some(),
        "flag on must populate both"
    );
    eprintln!(
        "FLAG-OFF ANCHOR  RHF water/STO-3G  E = {:.16e} (bits {:#x}) off == on",
        off.res.energy,
        off.res.energy.to_bits()
    );
    // Keep the moved-out fields alive for the compiler's borrow checker.
    let _ = (&off.mol, &off.f_mo, off.nocc, &off.bounds, &off.prep);
}
