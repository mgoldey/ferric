//! ROHF reference into the unrestricted correlated methods: effective-Fock vs
//! semi-canonical orbitals.
//!
//! A ROHF result carries ONE spatial MO set and the eigenvalues of the Roothaan
//! EFFECTIVE Fock operator, which belongs to neither spin. Before the fix, U-PDEP-RPA,
//! U-GW and U-RI-MP2 on a ROHF reference used those orbitals and energies for BOTH
//! spins (`mo_b.rs` / `rimp2.rs` / `u_rimp2.rs` / `ferric-rpa lib.rs` fell back to
//! `mos_a()` and `eps_a()` for β). The standard ROHF → unrestricted bridge
//! semi-canonicalizes: diagonalize `F_α` in the α-occupied and α-virtual blocks, and
//! `F_β` in the β blocks, giving per-spin orbitals and energies.
//!
//! Three references per system (OH, CH3 doublets / cc-pVDZ, cc-pvdz-ri aux):
//!
//! * `legacy`: the pre-fix treatment, rebuilt explicitly (`legacy_effective_fock_view`:
//!   ROHF MOs + effective-Fock eigenvalues for both spins, labelled Unrestricted so the
//!   methods take them as given). Reproduces the old code path on any tree.
//! * `semi`: `ferric_scf::semicanonical::semicanonicalize` (spin Focks REBUILT from
//!   J/K on the final ROHF density) → `to_unrestricted_result`.
//! * `direct`: the raw ROHF result handed to the method, which now semi-canonicalizes
//!   it itself from the SCF's stored spin Focks (`unrestricted_reference`).
//!
//! Tests:
//!
//! * `rohf_effective_fock_vs_semicanonical_differ_*` (MEASURE): `legacy` and `semi`
//!   differ by far more than any numerical noise — the defect is real and large.
//! * `rohf_reference_is_semicanonicalized_*`: `direct` equals `semi` (two independent
//!   Fock constructions agree to the SCF convergence level). Fails on the pre-fix tree,
//!   where `direct` equalled `legacy`.
//!
//! Run: `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release --test
//! rohf_semicanonical_reference -- --include-ignored --nocapture`

use std::path::PathBuf;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_u_gw, GwConfig, GwMethod, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::run_u_pdep_rpa;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::semicanonicalize;
use ferric_scf::{ScfResult, Spin};

/// Frequency points for W (Gauss–Legendre, u0 = 0.5). Identical on every arm, so
/// the comparisons below contain no quadrature difference.
const N_QUAD: usize = 20;

/// `direct` vs `semi`: stored last-iteration spin Focks vs Focks rebuilt from the
/// final density — equal to the SCF density-convergence level (1e-9 here).
const TOL_DIRECT_VS_SEMI_E: f64 = 1e-7;
const TOL_DIRECT_VS_SEMI_QP: f64 = 1e-6;

/// MEASURE: proposed must-differ bars for `legacy` vs `semi`. The existing U-MP2
/// measurement (ferric-cc/tests/semicanonical_mp2.rs) is 4.7e-3 (OH) / 4.0e-3 (CH3)
/// Ha; these bars sit ~40x below it.
const MIN_LEGACY_SHIFT_E: f64 = 1e-4;
/// MEASURE: proposed; QP energies over the default window, max over both spins.
const MIN_LEGACY_SHIFT_QP: f64 = 1e-3;

struct Case {
    label: &'static str,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rohf: ScfResult,
    semi: ScfResult,
}

fn xyz_path(system: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules/validation")
        .join(format!("{system}.xyz"))
        .to_string_lossy()
        .into_owned()
}

fn setup(label: &'static str) -> Case {
    let mol = Molecule::load_xyz_with_charge(&xyz_path(label), 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rohf = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(rohf.converged, "{label}: ROHF did not converge");
    assert_eq!(rohf.spin, Spin::RestrictedOpen);
    let semi = semicanonicalize(&ctx, &mol, &obs, &bounds, &rohf, 1e-12, None)
        .unwrap()
        .to_unrestricted_result(&rohf);
    eprintln!("{label}: E_ROHF = {:.10}", rohf.energy);
    Case {
        label,
        mol,
        obs,
        dfbs,
        rohf,
        semi,
    }
}

/// The pre-fix treatment of a ROHF reference: ROHF MOs and the EFFECTIVE Fock's
/// eigenvalues for both spins, labelled Unrestricted so the methods use them as given.
fn legacy_effective_fock_view(rohf: &ScfResult) -> ScfResult {
    let mut v = rohf.clone();
    v.spin = Spin::Unrestricted;
    v.mos_beta = Some(rohf.mos_alpha.clone());
    v.eps_beta = Some(rohf.eps_alpha.clone());
    v
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: N_QUAD,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn rpa(c: &Case, scf: &ScfResult) -> f64 {
    run_u_pdep_rpa(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(),
    )
    .unwrap_or_else(|e| panic!("{}: run_u_pdep_rpa: {e:?}", c.label))
    .e_rpa
}

fn gw(c: &Case, scf: &ScfResult) -> UGwResult {
    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        ..Default::default()
    };
    run_u_gw(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(),
        &gw_cfg,
        None,
    )
    .unwrap_or_else(|e| panic!("{}: run_u_gw: {e:?}", c.label))
}

fn mp2(c: &Case, scf: &ScfResult) -> f64 {
    u_ri_mp2(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        scf,
        &RiMp2Config::default(),
    )
    .unwrap_or_else(|e| panic!("{}: u_ri_mp2: {e:?}", c.label))
    .mp2_corr
}

/// max |a − b| over both spins' QP energies (same window on both).
fn qp_dev(a: &UGwResult, b: &UGwResult) -> f64 {
    assert_eq!(a.mo_indices, b.mo_indices, "QP windows differ");
    a.eps_qp_a
        .iter()
        .zip(&b.eps_qp_a)
        .chain(a.eps_qp_b.iter().zip(&b.eps_qp_b))
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn print_gw(label: &str, arm: &str, r: &UGwResult) {
    let ha = 27.211_386_245_988;
    eprintln!(
        "{label} [{arm}] U-G0W0 MOs {:?}\n    eps_mf_a {:?}\n    eps_qp_a {:?}\n    eps_mf_b {:?}\n    eps_qp_b {:?}  (eV)",
        r.mo_indices,
        r.eps_mf_a.iter().map(|x| (x * ha * 1e4).round() / 1e4).collect::<Vec<_>>(),
        r.eps_qp_a.iter().map(|x| (x * ha * 1e4).round() / 1e4).collect::<Vec<_>>(),
        r.eps_mf_b.iter().map(|x| (x * ha * 1e4).round() / 1e4).collect::<Vec<_>>(),
        r.eps_qp_b.iter().map(|x| (x * ha * 1e4).round() / 1e4).collect::<Vec<_>>(),
    );
}

/// MEASURE: the effective-Fock (pre-fix) treatment and the semi-canonical one give
/// materially different U-RPA, U-G0W0 and U-RI-MP2 numbers.
fn defect_magnitude(system: &'static str) {
    let c = setup(system);
    let legacy = legacy_effective_fock_view(&c.rohf);

    let (rpa_l, rpa_s) = (rpa(&c, &legacy), rpa(&c, &c.semi));
    let (mp2_l, mp2_s) = (mp2(&c, &legacy), mp2(&c, &c.semi));
    let (gw_l, gw_s) = (gw(&c, &legacy), gw(&c, &c.semi));
    print_gw(c.label, "legacy", &gw_l);
    print_gw(c.label, "semi", &gw_s);
    let d_qp = qp_dev(&gw_l, &gw_s);
    eprintln!(
        "{}: U-RPA E_c   legacy {rpa_l:.10}  semi {rpa_s:.10}  shift {:+.3e}",
        c.label,
        rpa_s - rpa_l
    );
    eprintln!(
        "{}: U-MP2 E_c   legacy {mp2_l:.10}  semi {mp2_s:.10}  shift {:+.3e}",
        c.label,
        mp2_s - mp2_l
    );
    eprintln!("{}: U-G0W0 max |dQP| legacy vs semi {d_qp:.3e} Ha", c.label);

    assert!(
        (rpa_s - rpa_l).abs() > MIN_LEGACY_SHIFT_E,
        "{}: U-RPA semi vs legacy only {:.2e}",
        c.label,
        rpa_s - rpa_l
    );
    assert!(
        (mp2_s - mp2_l).abs() > MIN_LEGACY_SHIFT_E,
        "{}: U-MP2 semi vs legacy only {:.2e}",
        c.label,
        mp2_s - mp2_l
    );
    assert!(
        d_qp > MIN_LEGACY_SHIFT_QP,
        "{}: U-G0W0 semi vs legacy only {d_qp:.2e}",
        c.label
    );
}

/// The fix: a raw ROHF result handed to each method gives the semi-canonical answer.
fn direct_is_semicanonical(system: &'static str) {
    let c = setup(system);

    let (rpa_d, rpa_s) = (rpa(&c, &c.rohf), rpa(&c, &c.semi));
    let (mp2_d, mp2_s) = (mp2(&c, &c.rohf), mp2(&c, &c.semi));
    let (gw_d, gw_s) = (gw(&c, &c.rohf), gw(&c, &c.semi));
    let d_qp = qp_dev(&gw_d, &gw_s);
    eprintln!(
        "{}: direct-vs-semi |dE_c(RPA)| {:.2e}  |dE_c(MP2)| {:.2e}  max|dQP| {d_qp:.2e}",
        c.label,
        (rpa_d - rpa_s).abs(),
        (mp2_d - mp2_s).abs()
    );
    assert!(
        (rpa_d - rpa_s).abs() < TOL_DIRECT_VS_SEMI_E,
        "{}: U-RPA on raw ROHF {rpa_d:.10} is not the semi-canonical {rpa_s:.10}",
        c.label
    );
    assert!(
        (mp2_d - mp2_s).abs() < TOL_DIRECT_VS_SEMI_E,
        "{}: U-MP2 on raw ROHF {mp2_d:.10} is not the semi-canonical {mp2_s:.10}",
        c.label
    );
    assert!(
        d_qp < TOL_DIRECT_VS_SEMI_QP,
        "{}: U-G0W0 on raw ROHF differs from semi-canonical by {d_qp:.2e} Ha",
        c.label
    );
}

#[test]
fn rohf_reference_is_semicanonicalized_oh() {
    direct_is_semicanonical("oh");
}

#[test]
#[ignore = "slow: ROHF + 3x(U-RPA, U-G0W0, U-MP2) on CH3/cc-pVDZ; run with --release --ignored"]
fn rohf_reference_is_semicanonicalized_ch3() {
    direct_is_semicanonical("ch3");
}

#[test]
#[ignore = "measurement: effective-Fock vs semi-canonical ROHF reference; run with --release --ignored --nocapture"]
fn rohf_effective_fock_vs_semicanonical_differ_oh() {
    defect_magnitude("oh");
}

#[test]
#[ignore = "measurement: effective-Fock vs semi-canonical ROHF reference; run with --release --ignored --nocapture"]
fn rohf_effective_fock_vs_semicanonical_differ_ch3() {
    defect_magnitude("ch3");
}
