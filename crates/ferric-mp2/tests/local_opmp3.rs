//! Amplitude-threshold local OP-MP3 (`ferric_mp2::local_opmp3`).
//!
//! Exactness anchor first: at ε = 0 the pattern keeps everything, the
//! pattern solves are the full localized amplitude equations, and every
//! OP-MP3 model must equal the canonical driver's
//! (`opmp3_on_reference`, same RI basis) on the HF AND the PBE0 reference
//! (the KS one exercises the HF-Fock-of-the-KS-density localization, the
//! non-diagonal `F_oo`/`F_vv` of `A`, and the semicanonical frozen core).
//! W is compared by eigenvalues (`W_loc = Uᵀ W_can U`).
//!
//! Hypotheses. If the construction is right: ε = 0 agrees to the CG floor
//! (≲1e-10 Ha); at finite ε `E2` equals the amplitude LMP2 energy of the
//! same maps (an independent assembly: RI B with the symmetric metric root
//! and a Schwarz candidate prefilter, Hylleraas energy) and `E3 = c2·K =
//! c1·R(c1)` to the CG floor. If it is broken (a wrong Fock matrix, a wrong
//! frozen core, a residual read on the wrong indices, two masks) the ε = 0
//! models move by ≥1e-6 and the identity opens.
//!
//! Mutations seen to fail (2026-11-19, each a one-line source edit to
//! `src/local_opmp3.rs`, then reverted; every failure an assertion):
//!
//! * `F_oo` diagonal only in the c1 solve: every test (anchors at the
//!   1e-3 level, E2 vs LMP2 |d| 7.6e-3, E3 gap 4.5e-6);
//! * the KS Fock `scf.fock_r()` instead of the reference HF Fock: both PBE0
//!   anchors (models 1.6e-1, W eigenvalues 6.2e-3); HF unaffected, as it must be;
//! * `scf` instead of the semicanonical `scf_sc` for the localized spaces:
//!   the PBE0 frozen-core anchor only (models 2.0e-5);
//! * c2 solved on the ε·10 pattern: the E3 identity (gap 1.1e-4);
//! * the plain Eq-8 mask (no Fock screen): the E2-vs-LMP2 pattern check
//!   (keep 0.519 vs 0.563).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config, FOCK_SCREEN_KAPPA_DEFAULT};
use ferric_mp2::local_opmp3::{local_opmp3_on_reference, LocalOpmp3Result};
use ferric_mp2::opmp3::{Opmp3Models, WChannels};
use ferric_mp2::opmp3_driver::{
    opmp3_on_reference, opmp3_scf, Opmp3Config, Opmp3Integrals, Opmp3RefKind, Opmp3Result,
};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

/// ε = 0 bar: both pattern solves stop at relative residual 1e-11 and W is
/// linear in the amplitudes, so |dE| ~ 1e-11 · |E_c| ~ 1e-12; 1e-10 leaves
/// room for the metric-root and rotation round-off of the two routes.
const ANCHOR_BAR: f64 = 1e-10;

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    aux: PreparedBasis,
    bounds: SchwarzBounds,
    scf: ScfResult,
    config: Opmp3Config,
}

fn system(reference: &str, frozen_core: usize) -> Sys {
    let xyz = workspace_root().join("testdata/molecules/alkane_3.xyz");
    let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let mut config = Opmp3Config::new(Opmp3RefKind::parse_config_str(reference).unwrap());
    config.frozen_core = frozen_core;
    config.scf_energy_conv = Some(1e-12);
    config.scf_density_conv = Some(1e-10);
    let scf = opmp3_scf(&ParallelContext::default(), &mol, &obs, &bounds, &config).unwrap();
    Sys {
        mol,
        obs,
        obs_bs,
        aux,
        bounds,
        scf,
        config,
    }
}

fn local_cfg(sys: &Sys, eps: f64) -> AmplitudeLmp2Config {
    AmplitudeLmp2Config {
        eps,
        frozen_core: sys.config.frozen_core,
        fock_screen: Some(FOCK_SCREEN_KAPPA_DEFAULT),
        ..Default::default()
    }
}

fn local(sys: &Sys, eps: f64) -> LocalOpmp3Result {
    local_opmp3_on_reference(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        &sys.obs_bs,
        &sys.bounds,
        &sys.scf,
        &sys.aux,
        &sys.config,
        &local_cfg(sys, eps),
    )
    .expect("local OP-MP3")
}

fn canonical(sys: &Sys) -> Opmp3Result {
    opmp3_on_reference(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        &sys.bounds,
        &sys.scf,
        Opmp3Integrals::Ri(&sys.aux),
        &sys.config,
    )
    .expect("canonical OP-MP3")
}

fn models(m: &Opmp3Models) -> [(&'static str, f64); 6] {
    [
        ("mp2", m.mp2),
        ("mp3", m.mp3),
        ("p-mp3", m.p_mp3),
        ("op-mp3", m.op_mp3),
        ("sop-mp3", m.sop_mp3),
        ("ssd-opmp3", m.ssd_opmp3),
    ]
}

fn eigvals(m: &Array2<f64>) -> Vec<f64> {
    m.eigh(UPLO::Lower).unwrap().0.to_vec()
}

/// Largest eigenvalue deviation over the α SS / OS blocks of one order.
fn w_eig_dev(a: &WChannels, b: &WChannels) -> f64 {
    [(&a.ss[0], &b.ss[0]), (&a.os[0], &b.os[0])]
        .iter()
        .flat_map(|(x, y)| {
            eigvals(x)
                .into_iter()
                .zip(eigvals(y))
                .map(|(p, q)| (p - q).abs())
                .collect::<Vec<_>>()
        })
        .fold(0.0, f64::max)
}

/// The ε = 0 anchor on one reference: every model, `E_HF`, `E_NBS` and the
/// W spectra equal the canonical driver's.
fn anchor(reference: &str, frozen_core: usize) {
    let sys = system(reference, frozen_core);
    let can = canonical(&sys);
    let lr = local(&sys, 0.0);
    let loc = &lr.opmp3;
    assert!(
        (lr.keep_fraction - 1.0).abs() < 1e-15,
        "eps = 0 must keep everything, kept {}",
        lr.keep_fraction
    );
    let mut worst = 0.0f64;
    for ((name, l), (_, c)) in models(&loc.models).into_iter().zip(models(&can.models)) {
        let d = (l - c).abs();
        eprintln!(
            "{reference} fc={frozen_core} {name}: local {l:+.12} canonical {c:+.12} |d| {d:.2e}"
        );
        worst = worst.max(d);
    }
    let d_ref = (loc.e_hf - can.e_hf)
        .abs()
        .max((loc.e_nbs - can.e_nbs).abs());
    let d_w = w_eig_dev(&loc.w.w2, &can.w.w2).max(w_eig_dev(&loc.w.w3, &can.w.w3));
    eprintln!(
        "{reference} fc={frozen_core}: |dE_HF|,|dE_NBS| {d_ref:.2e}  W eig dev {d_w:.2e}  \
         E_NBS {:+.3e}  E3 gap {:.2e}",
        loc.e_nbs,
        lr.e3_identity_gap()
    );
    assert!(
        worst.max(d_ref).max(d_w) < ANCHOR_BAR,
        "{reference} fc={frozen_core}: eps = 0 local vs canonical: models {worst:.2e}, \
         reference {d_ref:.2e}, W eigenvalues {d_w:.2e} (bar {ANCHOR_BAR:.0e})"
    );
}

#[test]
fn local_opmp3_matches_exact_in_the_trivial_limit_hf() {
    anchor("hf", 0);
}

#[test]
fn local_opmp3_matches_exact_in_the_trivial_limit_pbe0() {
    anchor("pbe0", 0);
}

#[test]
fn local_opmp3_matches_exact_in_the_trivial_limit_pbe0_frozen_core() {
    anchor("pbe0", 3);
}

/// Finite ε: the KS reference's `E_NBS` is the canonical one exactly (it
/// does not see the mask), `E3` closes on one pattern, and the truncation
/// is real (the mask drops amplitudes and moves `E2`).
#[test]
fn finite_eps_keeps_e_nbs_and_the_2n1_identity() {
    let sys = system("pbe0", 0);
    let can = canonical(&sys);
    let lr = local(&sys, 1e-4);
    let loc = &lr.opmp3;
    eprintln!(
        "pbe0 eps 1e-4: keep {:.4} pairs {} E2 err {:.2e} E3 err {:.2e} OP err {:.2e} gap {:.2e}",
        lr.keep_fraction,
        lr.n_pairs,
        loc.models.mp2 - can.models.mp2,
        (loc.models.mp3 - loc.models.mp2) - (can.models.mp3 - can.models.mp2),
        loc.models.op_mp3 - can.models.op_mp3,
        lr.e3_identity_gap()
    );
    assert!(lr.keep_fraction < 0.9, "eps 1e-4 kept {}", lr.keep_fraction);
    assert!(
        (loc.models.mp2 - can.models.mp2).abs() > 1e-8,
        "the mask moved nothing"
    );
    assert_eq!(loc.e_nbs, can.e_nbs, "E_NBS must not depend on the mask");
    assert!(
        lr.e3_identity_gap() < 1e-10,
        "E3 identity gap {:.2e} on one pattern",
        lr.e3_identity_gap()
    );
}

/// Finite ε on the HF reference: `E2 = c1·K` equals the amplitude LMP2
/// energy of the same maps (λ-Boys, Fock screen, VV-HV), which is built by
/// an independent assembly (RI `B` with the symmetric metric root, Schwarz
/// candidates, Hylleraas energy).
#[test]
fn finite_eps_e2_matches_amplitude_lmp2_on_the_same_maps() {
    let sys = system("hf", 0);
    for eps in [1e-4, 1e-5] {
        let loc = local(&sys, eps);
        let lmp2 = amplitude_lmp2(
            &sys.mol,
            &sys.obs,
            &sys.obs_bs,
            &sys.aux,
            Operator::coulomb(),
            &sys.scf,
            &AmplitudeLmp2Config {
                cg_rtol: Some(1e-12),
                ..local_cfg(&sys, eps)
            },
        )
        .expect("amplitude LMP2");
        let d = (loc.opmp3.e2.total() - lmp2.e_corr).abs();
        eprintln!(
            "eps {eps:.0e}: local OP-MP3 E2 {:+.12} (keep {:.4}) LMP2 {:+.12} (keep {:.4}) |d| {d:.2e}",
            loc.opmp3.e2.total(),
            loc.keep_fraction,
            lmp2.e_corr,
            lmp2.keep_fraction
        );
        assert!(
            (loc.keep_fraction - lmp2.keep_fraction).abs() < 1e-12,
            "eps {eps:.0e}: patterns differ (keep {} vs {})",
            loc.keep_fraction,
            lmp2.keep_fraction
        );
        assert!(d < 1e-9, "eps {eps:.0e}: E2 vs LMP2 |d| {d:.2e}");
    }
}
