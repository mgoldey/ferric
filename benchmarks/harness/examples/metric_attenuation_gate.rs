//! Error gate for the ATTENUATED RI FITTING METRIC spike.
//!
//! Premise under test (Ochsenfeld-style attenuated DF): replace the Coulomb
//! 2-center fitting metric `(P|1/r|Q)` with a short-ranged one `(P|w_met|Q)`,
//! keeping the PHYSICAL 3-index integrals `(P|1/r|mu nu)` untouched. A
//! short-ranged `V` is banded/sparse, so `V^{-1}` and the dressing GEMM could
//! become cheap. The cost is a fitting BIAS: this is no longer a variational
//! Coulomb fit.
//!
//! GATE (B', size-extensive form). An attenuated metric is "free" only if:
//!   1. MAGNITUDE  |dE_metric(N)| <= |dE_RI(N)| at EVERY N, where
//!      dE_RI = (Coulomb-metric RI-MP2) - (exact-ERI canonical MP2) is the
//!      approximation the user has ALREADY accepted.
//!   2. EXTENSIVITY  dE_metric(N)/N flat or decreasing in N. A per-monomer
//!      error that GROWS means the approximation degrades with size -- that
//!      kills the lane no matter how good water looks.
//!
//! Probe 0 (hard falsifier, run first, seconds): N non-interacting water
//! monomers at 50 Bohr must give E(N) = N*E(1). Any deviation is a pure
//! size-extensivity violation with no basis-set or geometry confound.
//!
//! A Cholesky failure at large omega_m is a REAL "this omega_m is unusable"
//! signal (attenuated metric has lost rank in a Coulomb-optimized aux basis),
//! NOT something to regularize away -- it is recorded as `unusable`.
//!
//! Run (small basis, serial -- see CLAUDE.md threading conventions):
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 \
//!     cargo run --release -p ferric-benchmarks --example metric_attenuation_gate

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ferric_mp2::canonical::canonical_mp2;
use ferric_mp2::rimp2::{
    ri_mp2, ri_mp2_robust_attenuated_metric, ri_mp2_robust_attenuated_metric_with,
    robust_fit_coulomb_parts, RiMp2Config,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};

/// Orbital + RI aux basis for the whole gate. Small by design: the gate asks
/// "does the metric perturbation match the RI error", a ferric-vs-ferric
/// question, so absolute basis quality is not the variable under study.
const OBS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";

/// Metric attenuation strengths (Bohr^-1). 0.0 is a sentinel for "Coulomb
/// metric" = the unattenuated reference.
const OMEGA_M: &[f64] = &[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 2.0];

struct Point {
    /// Exact-ERI MP2. `None` when not requested: Probe 0 compares RI-vs-RI only
    /// (the additivity residual needs no exact reference), and canonical_mp2 is
    /// the expensive no-RI O(N^5) path — computing it there would dominate the
    /// runtime of the cheapest, most decisive probe for nothing.
    e_canonical: Option<f64>,
    e_ri_coulomb: f64,
    /// Robust fit evaluated AT the Coulomb metric — must equal `e_ri_coulomb`.
    e_robust_coulomb: f64,
    /// (omega_m, energy or None if the metric was unusable)
    attenuated: Vec<(f64, Option<f64>)>,
}

fn run_system(mol: &Molecule, label: &str, want_canonical: bool) -> Point {
    let obs_bs = basis::bundled(OBS).unwrap();
    let aux_bs = basis::bundled(AUX).unwrap();
    let obs = PreparedBasis::new(mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let rhf = solve_rhf(&ParallelContext::default(), mol, &obs, op, &bounds, &RhfConfig::default())
        .unwrap_or_else(|e| panic!("{label}: RHF failed: {e}"));

    // Exact-ERI MP2: the ground truth both RI variants are measured against.
    let e_canonical = want_canonical.then(|| {
        canonical_mp2(mol, &obs, op, &rhf, 0)
            .unwrap_or_else(|e| panic!("{label}: canonical MP2 failed: {e}"))
    });

    let base_cfg = RiMp2Config { frozen_core: 0, memory_budget_bytes: None, ..Default::default() };
    let e_ri_coulomb = ri_mp2(mol, &obs, &dfbs, op, &rhf, &base_cfg)
        .unwrap_or_else(|e| panic!("{label}: Coulomb-metric RI-MP2 failed: {e}"))
        .mp2_corr;

    // SELF-CONSISTENCY: the robust fit with metric_op == the physical operator
    // must reproduce standard RI-MP2 (the fit residual is then the ordinary RI
    // residual and the three-term form collapses to it). Any disagreement here
    // is an IMPLEMENTATION bug and must be caught before any physics is read off
    // the attenuated numbers.
    let e_robust_coulomb =
        ri_mp2_robust_attenuated_metric(mol, &obs, &dfbs, op, op, &rhf, &base_cfg)
            .unwrap_or_else(|e| panic!("{label}: robust@Coulomb failed: {e}"))
            .e_total;

    // The Coulomb 3-index MO tensor and Coulomb 2-center metric do not depend on
    // omega_m — build them ONCE for the whole sweep rather than per point.
    let nbas = obs.nbasis();
    let nocc_total = mol.nelec() as usize / 2;
    let nocc = nocc_total;
    let nvir = nbas - nocc_total;
    let cmo = rhf.mos_r();
    let c_occ = cmo.slice(ndarray::s![.., 0..nocc]).to_owned();
    let c_vir = cmo.slice(ndarray::s![.., nocc_total..]).to_owned();
    let (j_ov, v_c) = robust_fit_coulomb_parts(&obs, &dfbs, op, &c_occ, &c_vir, &base_cfg)
        .unwrap_or_else(|e| panic!("{label}: Coulomb parts failed: {e}"));

    let mut attenuated = Vec::new();
    for &w in OMEGA_M.iter().filter(|w| **w > 0.0) {
        let met = Operator::erfc(w);
        // A failure here is DATA (unusable omega_m), not a crash: record None.
        let e = ri_mp2_robust_attenuated_metric_with(
            &obs, &dfbs, met, &j_ov, &v_c, &c_occ, &c_vir,
            rhf.eps_r(), nocc, nvir, 0, nocc_total, &base_cfg,
        )
        .ok()
        .map(|r| r.e_total);
        if e.is_none() {
            eprintln!("    omega_m={w:.2}: UNUSABLE (metric factorization failed)");
        }
        attenuated.push((w, e));
    }

    Point { e_canonical, e_ri_coulomb, e_robust_coulomb, attenuated }
}

/// Probe 0: N non-interacting waters at 50 Bohr separation along z.
fn noninteracting_cluster(n: usize) -> Molecule {
    // Single water, coordinates in Angstrom (parse_xyz converts to Bohr).
    let mono = [("O", 0.0, 0.0, 0.0), ("H", 0.757, 0.586, 0.0), ("H", -0.757, 0.586, 0.0)];
    const SEP_ANG: f64 = 26.46; // ~50 Bohr
    let mut lines = vec![format!("{}", 3 * n), String::from("non-interacting water cluster")];
    for i in 0..n {
        for (s, x, y, z) in mono.iter() {
            lines.push(format!("{s} {x} {y} {}", z + (i as f64) * SEP_ANG));
        }
    }
    Molecule::parse_xyz(&lines.join("\n"), 0, 1).unwrap()
}

fn main() {
    println!("# Attenuated RI fitting metric -- error gate (B', size-extensive)");
    println!("# basis {OBS} / aux {AUX}; physical operator = Coulomb throughout");
    println!();

    // ---- Probe 0: non-interacting additivity (hard falsifier) -------------
    println!("## Probe 0: non-interacting additivity, E(N) must equal N*E(1)");
    println!("# a nonzero residual here is a pure size-extensivity violation");
    let p1 = run_system(&noninteracting_cluster(1), "water x1", false);
    let p2 = run_system(&noninteracting_cluster(2), "water x2", false);

    println!(
        "# self-consistency robust@Coulomb vs standard RI: x1 {:.2e}, x2 {:.2e} Ha (must be ~0)",
        (p1.e_robust_coulomb - p1.e_ri_coulomb).abs(),
        (p2.e_robust_coulomb - p2.e_ri_coulomb).abs()
    );
    println!("{:>8}  {:>16}  {:>16}  {:>12}", "omega_m", "E(2)", "2*E(1)", "residual");
    let ri_resid = p2.e_ri_coulomb - 2.0 * p1.e_ri_coulomb;
    println!(
        "{:>8}  {:>16.10}  {:>16.10}  {:>12.2e}   <- Coulomb metric (baseline)",
        "coulomb", p2.e_ri_coulomb, 2.0 * p1.e_ri_coulomb, ri_resid
    );
    for ((w, e2), (_, e1)) in p2.attenuated.iter().zip(p1.attenuated.iter()) {
        match (e2, e1) {
            (Some(e2), Some(e1)) => {
                let resid = e2 - 2.0 * e1;
                let verdict = if resid.abs() <= ri_resid.abs().max(1e-9) { "ok" } else { "VIOLATION" };
                println!("{w:>8.2}  {:>16.10}  {:>16.10}  {resid:>12.2e}   {verdict}", e2, 2.0 * e1);
            }
            _ => println!("{w:>8.2}  {:>16}  {:>16}  {:>12}   unusable", "-", "-", "-"),
        }
    }
    println!();

    // ---- Probes 1+2: magnitude and extensivity on the alkane series -------
    println!("## Probes 1+2: magnitude vs RI error, and per-unit extensivity");
    println!("# dE_RI      = RI-MP2(Coulomb metric) - canonical MP2   [accepted error]");
    println!("# dE_metric  = RI-MP2(attenuated)     - RI-MP2(Coulomb) [new error]");
    println!("# GATE: |dE_metric| <= |dE_RI|  AND  dE_metric/N non-growing");
    println!();

    for n_c in [1usize, 2, 3, 4] {
        let path = format!("testdata/molecules/alkane_{n_c}.xyz");
        let mol = match Molecule::load_xyz(&path) {
            Ok(m) => m,
            Err(e) => {
                println!("alkane_{n_c}: SKIPPED ({e})");
                continue;
            }
        };
        eprintln!("  running alkane_{n_c} ...");
        let p = run_system(&mol, &format!("alkane_{n_c}"), true);
        let d_ri = p.e_ri_coulomb - p.e_canonical.expect("canonical requested for alkanes");

        println!("### alkane_{n_c}  (C{n_c})   dE_RI = {d_ri:+.3e} Ha  ({:+.4} kcal/mol)", d_ri * 627.509);
        println!(
            "{:>8}  {:>14}  {:>12}  {:>12}  {:>10}",
            "omega_m", "dE_metric/Ha", "|dE_m|/|dE_RI|", "dE_m/N", "verdict"
        );
        for (w, e) in p.attenuated.iter() {
            match e {
                Some(e) => {
                    let d_m = e - p.e_ri_coulomb;
                    let ratio = d_m.abs() / d_ri.abs().max(1e-14);
                    let per_unit = d_m / (n_c as f64);
                    let verdict = if ratio <= 1.0 { "PASS" } else { "fail" };
                    println!("{w:>8.2}  {d_m:>+14.3e}  {ratio:>12.2}  {per_unit:>+12.3e}  {verdict:>10}");
                }
                None => println!("{w:>8.2}  {:>14}  {:>12}  {:>12}  {:>10}", "-", "-", "-", "unusable"),
            }
        }
        println!();
    }

    println!("# Read the dE_m/N column DOWN the alkane series for a fixed omega_m:");
    println!("# flat/decreasing => size-extensive error (acceptable);");
    println!("# growing => approximation degrades with size => LANE DIES.");
}
