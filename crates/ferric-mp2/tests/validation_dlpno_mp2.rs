//! VALIDATION tier — VALIDATION.md row "DLPNO (MP2 / CCSD / CCSD(T) / RPA)",
//! MP2 part only (issue #269, stage 1).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 --test validation_dlpno_mp2 \
//!     -- --ignored --nocapture
//! ```
//!
//! # What is compared
//!
//! ferric's pair-driven DLPNO-MP2 (`dlpno_mp2_from_b_ov`) on n-butane,
//! n-hexane and n-octane (cc-pVDZ / cc-pvdz-ri, all electrons correlated and
//! one frozen 1s per carbon) against
//!
//! * PySCF `DFMP2` on identical geometry, basis, aux and frozen core — the
//!   independent anchor for ferric's canonical RI-MP2 and for the untruncated
//!   DLPNO (`scripts/validation/gen_dlpno.py` →
//!   `testdata/reference/validation/dlpno/<system>_cc-pvdz.json`);
//! * ORCA 6.1.1 RI-MP2 (a like-for-like cross-check of the PySCF number) and
//!   ORCA DLPNO-MP2 NormalPNO / TightPNO (recorded for context).
//!
//! # NOT like-for-like — read before quoting any number
//!
//! `dlpno_mp2_from_b_ov` consumes the CANONICAL `b_ov` and canonical orbital
//! energies. Nothing localizes the occupieds, so ferric's pairs are canonical
//! pairs and its PNOs are built per canonical pair. ORCA localizes the
//! occupieds (Foster-Boys), builds PAO domains and truncates by its own
//! TCutPNO / TCutPairs / TCutDO. The two "% of canonical correlation
//! recovered" columns therefore measure DIFFERENT approximations. They are
//! printed side by side and are never ranked, and no bar below compares a
//! ferric DLPNO number to an ORCA DLPNO number. The Boys-localized variant
//! (acceptance item (c)) is NOT done: `dlpno_mp2_from_b_ov` divides by a
//! diagonal `eps`, so a non-canonical (Fock-coupled) occupied block cannot be
//! passed to it, and faking it by rotating `b_ov` while keeping diagonal
//! denominators would not be MP2.
//!
//! Also NOT claimed: any speedup, any statement about DLPNO-CCSD / (T) in
//! ferric (no end-to-end driver exists; ORCA's CCSD(T) numbers in the JSON are
//! context only), any behaviour above C8 (C8 is below the locality onset, so
//! every "retains ~everything" result here is provisional).
//!
//! # Hypotheses (written before the sweep, per the Experimental Protocol)
//!
//! * PHYSICS: if the PNO truncation is right, the recovered fraction
//!   `E_dlpno / E_canonical` is monotone non-increasing in `t_cut_pno` and in
//!   `t_cut_pairs`, never exceeds 100 % (truncation drops only negative
//!   pair-energy contributions), and `max_discarded_weight` is small whenever
//!   the loss is small.
//! * ARTIFACT: if the canonical-occupied construction is the limiting factor,
//!   `pair_retention` stays ~1.0 at every useful `t_cut_pairs` on C4-C8
//!   because canonical pairs are delocalized, and any "recovery" gap against
//!   ORCA measures the localization difference, not the PNO code.
//! * OUTCOME FOR THE PAIR SCREEN: NOT DISTINGUISHABLE here. At C4-C8 (below
//!   the locality onset) BOTH hypotheses predict ~100% pair retention: the
//!   canonical pair energies are exact (`estimate_pair_energies_from_b_ov`)
//!   and average ~E_corr/npairs ~ 4e-3 Eh, far above 1e-4, and ORCA with
//!   Boys-localized occupieds also keeps (nearly) every pair (153/153, 91/91,
//!   323/325, 190/190, 551/561, 325/325). A flat pair dial is therefore
//!   uninformative at these sizes. Only the PNO (virtual) screen is
//!   exercised meaningfully.
//!
//! # All-electron columns are lopsided against ORCA
//!
//! With the cores correlated ORCA uses a separate, tighter `TCutPNO (core)`
//! (1e-10 against 1e-8 for valence at NormalPNO, observed in its output for
//! n-butane), while ferric applies one flat `t_cut_pno` to every pair, so
//! ferric's core pairs drag its all-electron recovery (n-butane at 1e-8:
//! 99.9957% frozen-core against 99.977% all-electron). Compare the
//! frozen-core rows for the cleaner side-by-side. The `tcut_pre` key in the
//! JSON is ORCA's printed `TCutPre` pair prescreen (ORCA prints no TCutPairs).
//!
//! # Exactness anchor (asserted first)
//!
//! E_nuc, nao/naux, E_RHF; ferric canonical RI-MP2 == PySCF DFMP2; then
//! `DlpnoConfig::exact()` through `dlpno_mp2_from_b_ov` == PySCF DFMP2.
//! Only after that is any truncated number examined.
//!
//! # Negative control (acceptance item)
//!
//! `t_cut_pno = 1e-6, t_cut_pairs = 0` must MISS canonical by more than the
//! anchor bar AND have `virtual_retention < 1`, proving the virtual screen is
//! live on these systems.
//!
//! # Mutation ledger
//!
//! See [`MUTATIONS`]: each mutant was applied, the test was run, and it
//! failed.
//!
//! A missing reference JSON is a HARD failure, never a skip.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::dlpno_mp2::{
    build_domains_for, dlpno_mp2_from_b_ov, estimate_pair_energies_from_b_ov, DlpnoConfig,
    DlpnoDiagnostics,
};
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/dlpno";
const MOL_DIR: &str = "testdata/molecules";
const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";

/// Geometry check.
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF.
const TOL_E_RHF: f64 = 1e-9;
/// ferric canonical RI-MP2 and untruncated pair-driven DLPNO vs PySCF DFMP2.
// Measured max 3.6e-11 (alkane_8 fc=0; 1.3e-11 C4, 2.4e-11 C6) over 12
// comparisons (canonical and exact() DLPNO give identical numbers).
const TOL_ANCHOR: f64 = 4e-10;
/// ORCA RI-MP2 vs PySCF DFMP2 (re-asserted from the JSON).
// Measured max 3.96e-8 (alkane_4 fc=0) over 6 runs; E_RHF agrees to 2.2e-10.
const TOL_ORCA_RIMP2: f64 = 4e-7;
/// Slack for "monotone" / "one-sided" comparisons (summation-order noise).
const MONO_SLACK: f64 = 1e-9;

const T_CUT_PNO: [f64; 5] = [0.0, 1e-9, 1e-8, 1e-7, 1e-6];
const T_CUT_PAIRS: [f64; 3] = [0.0, 1e-5, 1e-4];

/// Mutation ledger. Each mutant was applied, the butane test run, and the
/// source restored afterwards:
///
/// 1. skip semicanonicalization in `pair_energy_in_pno_basis` (diagonal
///    `f_aa` as PNO energies, `U = I`): `exact()` misses PySCF by 2.90e-2 Ha;
///    anchor FAILS.
/// 2. mirror weight `fac` swapped to `if i == j { 2.0 } else { 1.0 }`
///    (kernel and pair-energy estimator): `exact()` misses by 1.92e-1 Ha;
///    anchor FAILS.
/// 3. test-side: `t_cut_pno` ignored by `cfg_of` (screen silently dead): the
///    negative control FAILS (|d| 1.3e-11, virtual retention 1.0).
/// 4. `build_domains_for` ignoring `t_cut_pairs` (always the distance builder):
///    the all-electron pair-screen assertion FAILS (see the ledger result in
///    the commit message).
const MUTATIONS: &str = "(1) no semicanonicalization -> anchor fails (2.9e-2 Ha); \
     (2) fac 2/1 swapped -> anchor fails (1.9e-1 Ha); \
     (3) t_cut_pno ignored -> negative control fails; \
     (4) build_domains_for ignores t_cut_pairs -> pair-screen assertion fails";

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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_dlpno.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

/// One SCF + RI transform, reused for every threshold of one frozen-core row.
struct Row {
    ctx: String,
    b_ov: Array2<f64>,
    eps: Vec<f64>,
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    e_canonical: f64,
}

fn dlpno(row: &Row, cfg: &DlpnoConfig) -> (f64, DlpnoDiagnostics) {
    // Boys centers are only read by the distance screens, which are off here
    // (pair_cutoff = coupling_cutoff = inf); the energy screen reads none.
    let centers = Array2::<f64>::zeros((row.nocc, 3));
    let pe = estimate_pair_energies_from_b_ov(
        &row.b_ov,
        &row.eps,
        row.nocc,
        row.nvir,
        row.first_occ,
        row.nocc_total,
    )
    .unwrap_or_else(|e| panic!("{}: pair energies: {e:?}", row.ctx));
    let domains = build_domains_for(&centers, &pe, cfg)
        .unwrap_or_else(|e| panic!("{}: domains: {e:?}", row.ctx));
    let (sc, diag) = dlpno_mp2_from_b_ov(
        &row.b_ov,
        &row.eps,
        row.nocc,
        row.nvir,
        row.first_occ,
        row.nocc_total,
        &domains,
        cfg,
    )
    .unwrap_or_else(|e| panic!("{}: dlpno_mp2_from_b_ov: {e:?}", row.ctx));
    (sc.e_total, diag)
}

fn cfg_of(t_pno: f64, t_pairs: f64) -> DlpnoConfig {
    DlpnoConfig {
        t_cut_pno: t_pno,
        t_cut_pairs: t_pairs,
        ..DlpnoConfig::exact()
    }
}

/// `variant`: 0 = all electrons correlated, 1 = one frozen 1s per carbon.
fn run_system(system: &str, variants_wanted: &[usize]) {
    let r = reference(system);
    let ctx0 = system.to_string();
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx0,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx0),
        TOL_ENUC,
    );
    let obs_bs = basis::bundled(BASIS).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx0}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx0}: naux"
    );
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx0}: solve_rhf: {e:?}"));
    assert!(rhf.converged, "{ctx0}: RHF not converged");
    check_close(
        &ctx0,
        "E_RHF",
        rhf.energy,
        num(&r, "/e_rhf", &ctx0),
        TOL_E_RHF,
    );

    let nocc_total = mol.nelec() as usize / 2;
    let variants = r["variants"].as_array().expect("variants");
    assert_eq!(
        variants.len(),
        2,
        "{ctx0}: expected fc = 0 and fc = n_heavy"
    );
    for &vi in variants_wanted {
        let v = &variants[vi];
        let fc = v["frozen_core"].as_u64().unwrap() as usize;
        let ctx = format!("{system}/fc={fc}");
        let e_pyscf = num(v, "/pyscf_dfmp2/e_corr", &ctx);

        let rcfg = RiMp2Config {
            frozen_core: fc,
            ..RiMp2Config::default()
        };
        let (sc, b_ov) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &rcfg)
            .unwrap_or_else(|e| panic!("{ctx}: ri_mp2: {e:?}"));
        let nocc = nocc_total - fc;
        let nvir = obs.nbasis() - nocc_total;
        let row = Row {
            ctx: ctx.clone(),
            b_ov,
            eps: rhf.eps_r().to_vec(),
            nocc,
            nvir,
            first_occ: fc,
            nocc_total,
            e_canonical: sc.e_total,
        };

        // ---- ANCHORS (before any truncated number is looked at) ----
        check_close(
            &ctx,
            "canonical RI-MP2 vs PySCF",
            row.e_canonical,
            e_pyscf,
            TOL_ANCHOR,
        );
        let (e_exact, d_exact) = dlpno(&row, &DlpnoConfig::exact());
        check_close(&ctx, "DLPNO exact() vs PySCF", e_exact, e_pyscf, TOL_ANCHOR);
        assert_eq!(d_exact.pair_retention, 1.0, "{ctx}: exact() drops pairs");
        assert_eq!(
            d_exact.virtual_retention, 1.0,
            "{ctx}: exact() drops virtuals"
        );
        let d_orca = num(v, "/orca_ri_mp2_vs_pyscf_dfmp2_abs_diff", &ctx);
        assert!(
            d_orca < TOL_ORCA_RIMP2,
            "{ctx}: ORCA RI-MP2 vs PySCF {d_orca:.2e} exceeds {TOL_ORCA_RIMP2:.0e}"
        );

        // ---- NEGATIVE CONTROL: the virtual screen is live ----
        let (e_neg, d_neg) = dlpno(&row, &cfg_of(1e-6, 0.0));
        let miss = (e_neg - e_pyscf).abs();
        eprintln!(
            "{ctx}: NEG t_cut_pno=1e-6: |d| {miss:.2e} (must exceed {:.0e}), virt ret {:.4}",
            10.0 * TOL_ANCHOR,
            d_neg.virtual_retention
        );
        assert!(
            miss > 10.0 * TOL_ANCHOR,
            "{ctx}: truncated run did not miss"
        );
        assert!(
            d_neg.virtual_retention < 1.0,
            "{ctx}: virtual screen not live"
        );

        // ---- SWEEP ----
        let pct = |e: f64| 100.0 * e / row.e_canonical;
        let mut grid = [[0.0f64; 3]; 5];
        eprintln!(
            "{ctx}: canonical RI-MP2 {:+.10}  nocc={nocc} nvir={nvir}",
            row.e_canonical
        );
        eprintln!(
            "{ctx}: t_pno      t_pairs   pct_recovered  pair_ret  virt_ret  max_disc_w  n_pairs"
        );
        for (ip, &tp) in T_CUT_PNO.iter().enumerate() {
            for (iq, &tq) in T_CUT_PAIRS.iter().enumerate() {
                let (e, d) = dlpno(&row, &cfg_of(tp, tq));
                grid[ip][iq] = pct(e);
                eprintln!(
                    "{ctx}: {tp:<9.0e} {tq:<9.0e} {:>13.6} {:>9.4} {:>9.4} {:>11.3e} {:>8}",
                    pct(e),
                    d.pair_retention,
                    d.virtual_retention,
                    d.max_discarded_weight,
                    d.n_pairs
                );
                // one-sided: truncation never recovers MORE than canonical
                assert!(
                    pct(e) <= 100.0 + MONO_SLACK,
                    "{ctx}: ({tp:.0e},{tq:.0e}) recovers {:.9}% > 100%",
                    pct(e)
                );
            }
        }
        // monotone in both thresholds
        for ip in 0..5 {
            for iq in 0..3 {
                if ip + 1 < 5 {
                    assert!(
                        grid[ip + 1][iq] <= grid[ip][iq] + MONO_SLACK,
                        "{ctx}: not monotone in t_cut_pno at row {ip},{iq}"
                    );
                }
                if iq + 1 < 3 {
                    assert!(
                        grid[ip][iq + 1] <= grid[ip][iq] + MONO_SLACK,
                        "{ctx}: not monotone in t_cut_pairs at row {ip},{iq}"
                    );
                }
            }
        }

        // ---- PAIR SCREEN (t_cut_pairs): live, but INFORMATION-FREE about the
        // physics at C4-C8. All-electron, the screen removes the core pairs
        // (counts below, measured); frozen-core, canonical valence pairs are all
        // above 1e-4 (their mean estimated energy is ~E_corr/npairs ~ 6e-3 Eh),
        // so the dial is flat by arithmetic. ORCA with Boys-localized occupieds
        // also keeps nearly every pair here. Both hypotheses predict ~100% pair
        // retention, so this cannot tell them apart; it is below the locality
        // onset and is not evidence for either. ----
        if fc == 0 {
            let (n5, n4) = expected_pairs_all_electron(system);
            let (e4, d4) = dlpno(&row, &cfg_of(0.0, 1e-4));
            let (_, d5) = dlpno(&row, &cfg_of(0.0, 1e-5));
            assert!(
                d4.pair_retention < 1.0,
                "{ctx}: t_cut_pairs = 1e-4 removed no pair (screen dead?)"
            );
            assert!(
                d4.n_pairs.abs_diff(n4) <= 3 && d5.n_pairs.abs_diff(n5) <= 3,
                "{ctx}: pair counts {}/{} (1e-4/1e-5) outside +-3 of measured {n4}/{n5}",
                d4.n_pairs,
                d5.n_pairs
            );
            let p4 = pct(e4);
            assert!(
                (98.5..99.95).contains(&p4),
                "{ctx}: recovery {p4:.4}% at t_cut_pairs = 1e-4 outside the measured band"
            );
        } else {
            let (_, d) = dlpno(&row, &cfg_of(0.0, 1e-4));
            assert_eq!(
                d.pair_retention, 1.0,
                "{ctx}: measured: valence pairs all survive t_cut_pairs = 1e-4"
            );
        }

        // ---- ORCA side, recorded for context (never a bar) ----
        for tag in ["dlpno_normalpno", "dlpno_tightpno"] {
            let p = num(v, &format!("/orca/{tag}/pct_of_orca_ri_mp2"), &ctx);
            eprintln!("{ctx}: ORCA {tag:<16} {p:.6}% of ORCA RI-MP2 (different localization; not a ranking)");
        }
        eprintln!("{ctx}: mutation ledger: {MUTATIONS}");
    }
}

/// Measured pair counts with all electrons correlated, (t_cut_pairs = 1e-5,
/// 1e-4): C4 153/133 of 153, C6 325/256 of 325, C8 551/391 of 561.
fn expected_pairs_all_electron(system: &str) -> (usize, usize) {
    match system {
        "alkane_4" => (153, 133),
        "alkane_6" => (325, 256),
        "alkane_8" => (551, 391),
        _ => panic!("no measured pair counts for {system}"),
    }
}

// Octane is split by frozen-core variant and the others are not, because
// the reviewer-measured wall time under the validation profile was butane 51 s,
// hexane 293 s, octane (both variants) 809 s: octane alone is ~70% of the
// file. Two independent tests let `test-threads = 2` run the halves in
// parallel and keep each under the per-test slow-timeout (see the override in
// .config/nextest.toml for these two).

#[test]
#[ignore = "validation: DLPNO (MP2 / CCSD / CCSD(T) / RPA)"]
fn dlpno_mp2_butane() {
    run_system("alkane_4", &[0, 1]);
}

#[test]
#[ignore = "validation: DLPNO (MP2 / CCSD / CCSD(T) / RPA)"]
fn dlpno_mp2_hexane() {
    run_system("alkane_6", &[0, 1]);
}

#[test]
#[ignore = "validation: DLPNO (MP2 / CCSD / CCSD(T) / RPA)"]
fn dlpno_mp2_octane_all_electron() {
    run_system("alkane_8", &[0]);
}

#[test]
#[ignore = "validation: DLPNO (MP2 / CCSD / CCSD(T) / RPA)"]
fn dlpno_mp2_octane_frozen_core() {
    run_system("alkane_8", &[1]);
}
