//! Exactness anchors for the integral-direct amplitude-threshold LMP2 under
//! the DECOUPLED terfc operator (`Operator::terfc_with_omega`), the operator
//! of the 2026 MP2-V modernization lane.
//!
//! ARTIFACT HYPOTHESIS (written before the first run, CLAUDE.md protocol):
//!  * physics/implementation OK  => at eps = 0 with inert maps the direct
//!    path equals an INDEPENDENT `ri_mp2` call with the same operator to
//!    <= 1e-9 Ha, at linked (r0*omega = 1/sqrt2) AND sharp (r0*omega = 4),
//!    on water, C4 and C8; energies sit between 0.5|E_coul| and 1.05|E_coul|
//!    and grow monotonically toward |E_coul| with r0.
//!  * broken table seam / metric => a far-field table bug (the ea7fee6 class:
//!    terf pass skipped out of table, terfc inflating to Coulomb at aux-pair
//!    separations >= 9 Bohr) would show ONLY on the alkane and only through
//!    one of the two paths if they used different integral code, i.e. as a
//!    size-dependent anchor miss or |E_terfc| > |E_coul|; a non-PD sharp
//!    metric would show as a Cholesky error or an energy off by orders.
//!    The direct path (batched 3c strips + per-pair domain V_DD) and ri_mp2
//!    (global 3c + global metric) share the engine, so a seam bug in the
//!    ENGINE is not caught by agreement alone: the monotonicity/bound
//!    invariants and the Coulomb-limit check (terfc at huge r0 -> Coulomb)
//!    carry that load.
//!
//! Needs the terf tables; skips (loudly) without FERRIC_TERF_TABLE_DIR:
//!   FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 \
//!     cargo test -p ferric-mp2 --release --test lmp2_direct_terfc_omega -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const ANG: f64 = ferric_core::units::ANGSTROM_TO_BOHR;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str, basis_name: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(basis_name).unwrap();
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
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
    }
}

fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        schwarz_skip: 0.0,
        ..Default::default()
    }
}

fn have_tables() -> bool {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIPPED: set FERRIC_TERF_TABLE_DIR (repo terf-tables/)");
        return false;
    }
    true
}

/// (label, r0 Bohr, omega Bohr^-1) for r0 = 1.0 A: linked and sharp.
fn settings(r0_ang: f64) -> [(&'static str, f64, f64); 2] {
    let r0 = r0_ang * ANG;
    [
        ("linked r0w=1/sqrt2", r0, (0.5f64).sqrt() / r0),
        ("sharp  r0w=4", r0, 4.0 / r0),
    ]
}

fn direct_e(su: &Setup, op: Operator, eps: f64, fc: usize, maps: &DirectConfig) -> f64 {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: fc,
        compute_reference: false,
        ..Default::default()
    };
    let (r, _) =
        amplitude_lmp2_direct(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, maps)
            .unwrap();
    assert!(r.cg_converged);
    r.e_corr
}

fn ri_e(su: &Setup, op: Operator, fc: usize) -> f64 {
    ri_mp2(
        &su.mol,
        &su.obs,
        &su.dfbs,
        op,
        &su.rhf,
        &RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        },
    )
    .unwrap()
    .mp2_corr
}

fn anchor(xyz: &str, fc: usize) {
    if !have_tables() {
        return;
    }
    let su = setup(xyz, "6-31g");
    let e_coul = ri_e(&su, Operator::coulomb(), fc);
    for (label, r0, w) in settings(1.0) {
        let op = Operator::terfc_with_omega(r0, w);
        let e_ri = ri_e(&su, op, fc);
        let e_dir = direct_e(&su, op, 0.0, fc, &trivial_maps());
        let d = (e_dir - e_ri).abs();
        eprintln!(
            "TERFC-OMEGA ANCHOR {xyz} {label}: direct {e_dir:.10} ri_mp2 {e_ri:.10} |d|={d:.3e} \
             (coulomb {e_coul:.10}, ratio {:.4})",
            e_ri / e_coul
        );
        assert!(
            d < 1e-9,
            "{xyz} {label}: eps=0 direct vs ri_mp2 {d:.3e} > 1e-9"
        );
        // Detection invariants (terfc-pq-metric-breakdown-gate0.md). The
        // |E(att)| <= |E(Coulomb)| invariant holds ONLY at the linked
        // r0*omega: it was derived from the kernel being pointwise <= 1/r,
        // but the MP2 energy is QUADRATIC in the kernel, and the sharp
        // terfc is a truncated Coulomb whose Fourier transform
        // 4pi/q^2 (1 - exp(-q^2/4w^2) cos(q r0)) reaches ~2x Coulomb at
        // q r0 ~ pi. MEASURED (water/C4/C8, 6-31G): E_sharp/E_coul = 1.98,
        // 2.00, 1.99, independent of the engine (M4b oracle 2e-13).
        // So sharp gets a wide garbage guard, linked the tight 5% one.
        let (lo, hi) = if label.starts_with("linked") {
            (0.5, 1.05)
        } else {
            (0.5, 2.5)
        };
        assert!(
            e_ri.abs() <= hi * e_coul.abs() && e_ri.abs() >= lo * e_coul.abs(),
            "{xyz} {label}: |E_terfc|={:.6} outside [{lo},{hi}]*|E_coul|={:.6}",
            e_ri.abs(),
            e_coul.abs()
        );
    }
}

#[test]
fn terfc_omega_direct_matches_ri_mp2_in_the_trivial_limit_water() {
    anchor("water.xyz", 0);
}

#[test]
fn terfc_omega_direct_matches_ri_mp2_in_the_trivial_limit_c4() {
    anchor("alkane_4.xyz", 4);
}

/// C8: aux-pair separations > 9 Bohr, the regime of the ea7fee6 far-field
/// table bug. Slow-ish (global ri_mp2 + metric at C8 for two settings).
#[test]
#[ignore]
fn terfc_omega_direct_matches_ri_mp2_in_the_trivial_limit_c8() {
    anchor("alkane_8.xyz", 8);
}

/// r0-monotonicity toward Coulomb at FIXED sharpness ratio, and the large-r0
/// Coulomb limit. A seam/metric defect that survives the agreement anchor
/// (both paths share the engine) breaks one of these.
#[test]
fn terfc_omega_linked_is_monotone_and_both_settings_reach_coulomb_at_large_r0() {
    if !have_tables() {
        return;
    }
    let su = setup("water.xyz", "6-31g");
    let e_coul = ri_e(&su, Operator::coulomb(), 0);
    for ratio in [(0.5f64).sqrt(), 4.0] {
        let mut prev = 0.0f64;
        let mut line = String::new();
        // 0.5 .. 4 A; omega = ratio / r0 keeps r0*omega fixed.
        for r0_ang in [0.5, 1.0, 2.0, 4.0] {
            let r0 = r0_ang * ANG;
            let e = ri_e(&su, Operator::terfc_with_omega(r0, ratio / r0), 0);
            line += &format!(" r0={r0_ang}A:{e:.6}");
            if ratio < 1.0 {
                // linked only: see anchor() for why sharp is not bounded by
                // Coulomb or monotone (truncated-Coulomb ripple).
                assert!(
                    e.abs() >= prev - 0.01 * e_coul.abs(),
                    "r0w={ratio:.3}: |E| not monotone in r0 ({line})"
                );
                assert!(e.abs() <= 1.05 * e_coul.abs());
            } else {
                assert!(e.abs() >= 0.5 * e_coul.abs() && e.abs() <= 2.5 * e_coul.abs());
            }
            prev = e.abs();
        }
        eprintln!("TERFC-OMEGA r0w={ratio:.3} water E(r0):{line}  coulomb {e_coul:.6}");
        // Large r0: the attenuated operator is Coulomb at all molecular
        // distances; the correlation energy approaches Coulomb's.
        assert!(
            (prev - e_coul.abs()).abs() < 0.05 * e_coul.abs(),
            "r0w={ratio:.3}: r0=4A not near Coulomb: {prev:.6} vs {:.6}",
            e_coul.abs()
        );
    }
}
