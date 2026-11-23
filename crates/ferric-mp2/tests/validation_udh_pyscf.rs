//! VALIDATION tier: the open-shell MP2-based double hybrids and the D3(BJ)
//! sets (`u_mp2_double_hybrid`, `mp2_double_hybrid` with
//! `DoubleHybridKind::{B2plyp, B2plypD3bj, DsdPbep86, DsdPbep86D3bj}`; CLI
//! kinds `b2plyp` / `dsd-pbep86`, `[dft] dispersion = "d3bj"`).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo test -p ferric-mp2 --release --test validation_udh_pyscf -- --ignored
//! ```
//!
//! # What is compared
//!
//! ferric's whole double hybrid on a UKS reference (its own UKS SCF with the
//! set's functional, then `c_os E_alpha-beta + c_ss (E_alpha-alpha +
//! E_beta-beta)` of the unrestricted RI-MP2 on the UKS orbitals) against PySCF
//! built from the DEFINITION: `dft.UKS` with the same libxc string, then
//! `mp.dfump2.DFUMP2` on those orbitals, its opposite-spin and same-spin tags
//! scaled here (`scripts/validation/gen_udh_pyscf.py`). NH2 and CH3 doublets at
//! cc-pVDZ (aux cc-pVDZ-RI) and def2-SVP (aux def2-SVP-RIFIT), frozen core 0
//! and 1, three SCF sets (B2PLYP; DSD-PBEP86 without dispersion; the
//! DSD-PBEP86-D3(BJ) refit set). The D3(BJ) energy is compared with the `dftd3`
//! package's (simple-dftd3) for the same geometry and the set's row, and the
//! B2PLYP-D3(BJ) total is the B2PLYP total plus its row.
//!
//! The singlet anchor (`singlet_uks_route_equals_rks_route`) needs no
//! reference: the UKS route on a closed-shell water equals the RKS route.
//!
//! # Settings
//!
//! Matched: ferric's basis JSON and Bohr geometry; the (75, 110) unpruned Becke
//! grid with Becke radii; RI-J and RI-K over def2-universal-jkfit (J tensor in
//! f64); the PT2 aux basis and Coulomb metric; frozen core 0 and 1; the same
//! stable UKS state (the reference refuses a case with two stable states more
//! than 1e-8 Ha apart, and the E_KS comparison below would expose a
//! state-selection difference). Not matched: the CLI's pruned production grid
//! and f32 RI-J storage, which the closed-shell evidence measures
//! (`validation_dsd_pbep86_pyscf.rs`); a default open-shell run carries no
//! anchor.
//!
//! # Bar derivation
//!
//! No free parameter: same grid, fits and state. What separates the codes is
//! (a) PySCF's own run-to-run spread (each reference's `reproduction` block,
//! minao/atom/huckel best against atom-only), (b) integral rounding and
//! grid-weight summation order, and (c) the SCF convergence ferric stops at
//! (energy 1e-10, density 1e-8). Each bar is `SAFETY (4) x (c + a + b)`,
//! with (c) the loose-vs-tight measurement of `ferric_scf_convergence_term`,
//! which asserts the constants below, and (b) `FLOOR`.
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! If the unrestricted assembly is right: E_KS, E_OS, E_SS and the total agree
//! at the noise level. If the beta channel is dropped from E_SS, E_SS misses by
//! E_beta-beta (mHa for a radical); if c_os and c_ss are swapped the total
//! misses by (c_os - c_ss)(E_SS - E_OS), up to mHa, while E_OS and E_SS (raw)
//! still pass; if a wrong D3(BJ) row is used the D3 energy misses by the
//! row difference, well above 1e-12. The three are separated because E_KS,
//! E_OS, E_SS, the D3 energy and the totals are compared individually.
//!
//! # Mutation
//!
//! Each of these production edits turns a named assertion red (restored after):
//! `u_mp2_double_hybrid` with the two scalings swapped; its `e_ss` without
//! `e_bb`; the `dsdpbep86` row of `ferric-d3/src/params.rs` with s6 0.418.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::double_hybrid::{
    mp2_double_hybrid, u_mp2_double_hybrid, DoubleHybridKind, Mp2DoubleHybridResult,
};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::ladder::{ksdft_ladder, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ferric_testkit::reference::num_ctx;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/udh_pyscf";
const MOL_DIR: &str = "testdata/molecules/validation";
const JKFIT: &str = "def2-universal-jkfit";
const TOL_ENUC: f64 = 1e-9;

/// (reference set name, kind of the SCF + scalings).
const SETS: [(&str, DoubleHybridKind); 3] = [
    ("b2plyp", DoubleHybridKind::B2plyp),
    ("dsd-pbep86", DoubleHybridKind::DsdPbep86),
    ("dsd-pbep86-d3bj", DoubleHybridKind::DsdPbep86D3bj),
];

const CASES: [(&str, &str, &str); 4] = [
    ("nh2", "cc-pvdz", "cc-pvdz-ri"),
    ("ch3", "cc-pvdz", "cc-pvdz-ri"),
    ("nh2", "def2-svp", "def2-svp-rifit"),
    ("ch3", "def2-svp", "def2-svp-rifit"),
];

const SAFETY: f64 = 4.0;
/// ferric loose-vs-tight SCF terms: maxima of `ferric_scf_convergence_term`
/// over the cases and sets, rounded up.
const CONV_E_KS: f64 = 1.0e-13;
const CONV_E_OS: f64 = 1.0e-10;
const CONV_E_SS: f64 = 1.0e-10;
const CONV_TOTAL: f64 = 1.0e-10;
/// Integral rounding and grid-weight summation order.
const FLOOR: f64 = 1.0e-12;
/// D3(BJ): the same closed form in f64 in two codes (the ferric-d3 test bar).
const TOL_D3: f64 = 1.0e-12;
/// A control is missed by at least this multiple of the total bar.
const MUST_MISS_FACTOR: f64 = 100.0;

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

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_udh_pyscf.py: a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

/// The KS config of `run_mp2_double_hybrid_arm`, pinned to the settings the
/// reference runs (flat grid, f64 RI-J storage).
fn ks_config(kind: DoubleHybridKind, tight: bool) -> RhfConfig {
    RhfConfig::default()
        .with_xc(kind.xc_name())
        .with_df_j_aux(JKFIT.to_string())
        .with_df_k_aux(JKFIT.to_string())
        .with_ri_j_precision(RiJPrecision::F64)
        .with_energy_conv(if tight { 1e-12 } else { 1e-10 })
        .with_density_conv(if tight { 1e-10 } else { 1e-8 })
        .with_max_iter(300)
        .with_dft_grid(AtomicGridConfig::default())
}

struct Run {
    ks: ScfResult,
    mol: Molecule,
    prep: PreparedBasis,
    dfbs: PreparedBasis,
}

fn load(
    system: &str,
    mult: usize,
    basis_name: &str,
    aux: &str,
) -> (Molecule, PreparedBasis, PreparedBasis) {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    (mol, prep, dfbs)
}

fn solve_uks(system: &str, basis_name: &str, aux: &str, cfg: RhfConfig) -> Run {
    let (mol, prep, dfbs) = load(system, 2, basis_name, aux);
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ks = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{system}/{basis_name}: {e}"));
    assert!(ks.converged, "{system}/{basis_name}: UKS not converged");
    Run {
        ks,
        mol,
        prep,
        dfbs,
    }
}

fn udh(run: &Run, kind: DoubleHybridKind, frozen_core: usize) -> Mp2DoubleHybridResult {
    let mut cfg = kind.mp2_config();
    cfg.frozen_core = frozen_core;
    u_mp2_double_hybrid(&run.mol, &run.prep, &run.dfbs, &run.ks, &cfg).expect("u_mp2_double_hybrid")
}

fn check(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
    d
}

fn run_case(system: &str, basis_name: &str, aux: &str) {
    let r = reference(system, basis_name);
    let ctx0 = format!("{system}/{basis_name}");
    for (set, kind) in SETS {
        let ctx = format!("{ctx0}/{set}");
        let s = format!("/sets/{set}");
        let c_os = num_ctx(&r, &format!("{s}/c_os"), &ctx);
        let c_ss = num_ctx(&r, &format!("{s}/c_ss"), &ctx);
        let published = kind.mp2_config();
        assert_eq!(
            (published.c_os, published.c_ss),
            (c_os, c_ss),
            "{ctx}: scalings"
        );
        let run = solve_uks(system, basis_name, aux, ks_config(kind, false));
        let enuc_ref = num_ctx(&r, "/nuclear_repulsion", &ctx);
        assert!(
            (run.mol.nuclear_repulsion() - enuc_ref).abs() < TOL_ENUC,
            "{ctx}: nuclear repulsion differs: geometry/unit mismatch"
        );
        let rep = |k: &str| num_ctx(&r, &format!("{s}/reproduction/{k}"), &ctx);
        let bar_ks = SAFETY * (CONV_E_KS + rep("e_ks") + FLOOR);
        check(
            &ctx,
            "E_KS",
            run.ks.energy,
            num_ctx(&r, &format!("{s}/ks/energy"), &ctx),
            bar_ks,
        );
        for fc in [0usize, 1] {
            let tag = format!("{ctx} fc{fc}");
            let out = udh(&run, kind, fc);
            let k = format!("{s}/fc{fc}");
            let (os, ss) = (rep(&format!("fc{fc}_e_os")), rep(&format!("fc{fc}_e_ss")));
            let bar_os = SAFETY * (CONV_E_OS + os + FLOOR);
            let bar_ss = SAFETY * (CONV_E_SS + ss + FLOOR);
            let bar_t = SAFETY * (CONV_TOTAL + rep("e_ks") + FLOOR + c_os * os + c_ss * ss);
            check(
                &tag,
                "E_OS",
                out.spin_components.e_os,
                num_ctx(&r, &format!("{k}/e_os"), &tag),
                bar_os,
            );
            check(
                &tag,
                "E_SS",
                out.spin_components.e_ss,
                num_ctx(&r, &format!("{k}/e_ss"), &tag),
                bar_ss,
            );
            check(
                &tag,
                "total",
                out.total_energy,
                num_ctx(&r, &format!("{k}/total"), &tag),
                bar_t,
            );
            // The PT2 term is not negligible; the total is the definition.
            assert!(out.e_corr_scaled.abs() > 1e-3, "{tag}: PT2 term vanished");
            // Both spin channels contribute: dropping E_beta-beta from E_SS, or
            // swapping the scalings, must be far outside the bar.
            assert!(
                (c_os - c_ss).abs() < 1e-12
                    || ((c_os - c_ss) * (out.spin_components.e_ss - out.spin_components.e_os))
                        .abs()
                        > MUST_MISS_FACTOR * bar_t,
                "{tag}: a c_os/c_ss swap would not be seen by the total bar"
            );
        }
        // D3(BJ): ferric's row and sum against dftd3, on the set's own name.
        if let Some(name) = kind.d3bj_functional() {
            let d3 = ferric_d3::d3bj_params_for_functional(name)
                .and_then(|p| ferric_d3::d3bj_energy_for_molecule(&run.mol, &p))
                .unwrap();
            check(
                &ctx,
                "E_D3(BJ)",
                d3,
                num_ctx(&r, &format!("{s}/d3bj/energy"), &ctx),
                TOL_D3,
            );
            let out = udh(&run, kind, 0);
            check(
                &ctx,
                "total+D3",
                out.total_energy + d3,
                num_ctx(&r, &format!("{s}/fc0/total_with_d3bj"), &ctx),
                SAFETY * (CONV_TOTAL + FLOOR) + TOL_D3,
            );
        }
    }
    // B2PLYP-D3(BJ): the B2PLYP SCF and PT2 plus its row.
    let ctx = format!("{ctx0}/b2plyp-d3bj");
    let kind = DoubleHybridKind::B2plypD3bj;
    assert_eq!(kind.xc_name(), DoubleHybridKind::B2plyp.xc_name());
    let run = solve_uks(system, basis_name, aux, ks_config(kind, false));
    let name = kind.d3bj_functional().expect("b2plyp-d3bj has a row");
    let d3 = ferric_d3::d3bj_params_for_functional(name)
        .and_then(|p| ferric_d3::d3bj_energy_for_molecule(&run.mol, &p))
        .unwrap();
    check(
        &ctx,
        "E_D3(BJ)",
        d3,
        num_ctx(&r, "/sets/b2plyp-d3bj/d3bj/energy", &ctx),
        TOL_D3,
    );
    for fc in [0usize, 1] {
        let out = udh(&run, kind, fc);
        let tag = format!("{ctx} fc{fc}");
        check(
            &tag,
            "total+D3",
            out.total_energy + d3,
            num_ctx(
                &r,
                &format!("/sets/b2plyp-d3bj/fc{fc}/total_with_d3bj"),
                &tag,
            ),
            SAFETY * (CONV_TOTAL + FLOOR) + TOL_D3,
        );
    }
}

/// ferric's own SCF-convergence term: the loose SCF (what the comparison runs)
/// against a two-orders-tighter one, per case and set. Each difference must
/// stay under the `CONV_*` constants the bars are built from.
#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn ferric_scf_convergence_term() {
    let mut worst = [0.0f64; 4];
    for (system, basis_name, aux) in CASES {
        for (set, kind) in SETS {
            let loose = solve_uks(system, basis_name, aux, ks_config(kind, false));
            let tight = solve_uks(system, basis_name, aux, ks_config(kind, true));
            let (a, b) = (udh(&loose, kind, 0), udh(&tight, kind, 0));
            let d = [
                (loose.ks.energy - tight.ks.energy).abs(),
                (a.spin_components.e_os - b.spin_components.e_os).abs(),
                (a.spin_components.e_ss - b.spin_components.e_ss).abs(),
                (a.total_energy - b.total_energy).abs(),
            ];
            eprintln!(
                "{system}/{basis_name}/{set}: loose-vs-tight E_KS {:.2e} E_OS {:.2e} E_SS {:.2e} total {:.2e}",
                d[0], d[1], d[2], d[3]
            );
            for i in 0..4 {
                worst[i] = worst[i].max(d[i]);
            }
        }
    }
    eprintln!("max over cases and sets: {worst:?}");
    let terms = [CONV_E_KS, CONV_E_OS, CONV_E_SS, CONV_TOTAL];
    for (i, what) in ["E_KS", "E_OS", "E_SS", "total"].iter().enumerate() {
        assert!(
            worst[i] <= terms[i],
            "{what} convergence term {:.2e} exceeds {:.1e}",
            worst[i],
            terms[i]
        );
    }
}

/// The singlet anchor: a closed-shell water through the UKS route equals the
/// RKS route, for every set. Both SCFs are converged tightly, so the bar is the
/// 1e-10 of the brief; the measured differences are printed.
#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn singlet_uks_route_equals_rks_route() {
    const BAR: f64 = 1.0e-10;
    for (set, kind) in SETS {
        let cfg = ks_config(kind, true);
        let (mol, prep, dfbs) = load("h2o", 1, "cc-pvdz", "cc-pvdz-ri");
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let ctx = ParallelContext::default();
        let ladder = ksdft_ladder(&cfg);
        let rks = solve_rhf_ladder(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &ladder)
            .expect("RKS")
            .result;
        let uks = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).expect("UKS");
        assert!(rks.converged && uks.converged, "{set}");
        for fc in [0usize, 1] {
            let mut mp2 = kind.mp2_config();
            mp2.frozen_core = fc;
            let r = mp2_double_hybrid(&mol, &prep, &dfbs, &rks, &mp2).unwrap();
            let u = u_mp2_double_hybrid(&mol, &prep, &dfbs, &uks, &mp2).unwrap();
            let d = [
                ("E_KS", (r.e_ks - u.e_ks).abs()),
                (
                    "E_OS",
                    (r.spin_components.e_os - u.spin_components.e_os).abs(),
                ),
                (
                    "E_SS",
                    (r.spin_components.e_ss - u.spin_components.e_ss).abs(),
                ),
                ("total", (r.total_energy - u.total_energy).abs()),
            ];
            eprintln!("h2o/{set} fc{fc}: UKS vs RKS {d:?}");
            for (what, v) in d {
                assert!(
                    v < BAR,
                    "{set} fc{fc}: singlet UKS vs RKS {what} {v:.2e} >= {BAR:.0e}"
                );
            }
            // Not vacuous: the PT2 term is large and both channels are used.
            assert!(u.spin_components.e_os.abs() > 1e-3 && u.spin_components.e_ss.abs() > 1e-4);
        }
    }
}

#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn udh_nh2_cc_pvdz_vs_pyscf() {
    run_case("nh2", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn udh_ch3_cc_pvdz_vs_pyscf() {
    run_case("ch3", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn udh_nh2_def2_svp_vs_pyscf() {
    run_case("nh2", "def2-svp", "def2-svp-rifit");
}

#[test]
#[ignore = "validation: UKS double hybrids vs PySCF"]
fn udh_ch3_def2_svp_vs_pyscf() {
    run_case("ch3", "def2-svp", "def2-svp-rifit");
}
