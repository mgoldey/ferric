//! GW100 subset — vertical ionization potentials via three methods.
//!
//! For each molecule in a 10-molecule subset of the GW100 set (van Setten,
//! Caruso, et al. JCTC 11, 5665, 2015), compute the vertical first IP by:
//!
//!   1. **Δ-SCF**: IP_SCF = E_UHF(cation) − E_RHF(neutral)
//!   2. **Δ-RPA**: IP_RPA = [E_UHF(cation) + E_U-RPA(cation)]
//!                       − [E_RHF(neutral) + E_RPA(neutral)]
//!   3. **Δ-MP2**: IP_MP2 = [E_UHF(cation) + E_U-MP2(cation)]
//!                       − [E_RHF(neutral) + E_MP2(neutral)]
//!
//! (Δ-OOMP2 would require open-shell OO-MP2 which ferric doesn't yet have;
//! Δ-MP2 substituted as the comparable correlated baseline. See task #42
//! for the open-shell OO-MP2 follow-up.)
//!
//! Output: prints a table to stdout with columns
//!   molecule | IP_exp(eV) | IP_ΔSCF | IP_ΔMP2 | IP_ΔRPA | MAE
//!
//! Reference IPs in eV from GW100 supplementary tables (or NIST
//! experimental where given).
//!
//! Usage:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release \
//!     --example gw100_ip_subset -p ferric-rpa

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa, PdepRpaConfig};
use ferric_integrals::oneelectron;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess, UhfConfig};
use ndarray::Array2;

const HARTREE_TO_EV: f64 = 27.211386245988_f64;

/// Geometries (Bohr) and reference experimental IPs (eV).
/// Source: NIST Chemistry WebBook + van Setten GW100 supplementary.
struct Case {
    name: &'static str,
    xyz: &'static str,
    /// Reference experimental vertical IP in eV (or G0W0@PBE if no exp).
    ip_ref: f64,
}

fn gw100_subset() -> Vec<Case> {
    vec![
        Case {
            name: "H2",
            xyz: "2\nH2\nH 0 0 0\nH 0 0 0.7414\n",
            ip_ref: 15.43,
        },
        Case {
            name: "He",
            xyz: "1\nHe\nHe 0 0 0\n",
            ip_ref: 24.59,
        },
        Case {
            name: "H2O",
            xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
            ip_ref: 12.62,
        },
        Case {
            name: "NH3",
            xyz: "4\nNH3\nN 0.0 0.0 0.116743\nH 0.94 0.0 -0.272400\nH -0.471 0.815 -0.272400\nH -0.471 -0.815 -0.272400\n",
            ip_ref: 10.82,
        },
        Case {
            name: "CH4",
            xyz: "5\nCH4\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n",
            ip_ref: 13.6,
        },
        Case {
            name: "N2",
            xyz: "2\nN2\nN 0 0 -0.5488\nN 0 0  0.5488\n",
            ip_ref: 15.58,
        },
        Case {
            name: "CO",
            xyz: "2\nCO\nC 0 0 0.000\nO 0 0 1.128\n",
            ip_ref: 14.01,
        },
        Case {
            name: "F2",
            xyz: "2\nF2\nF 0 0 -0.7095\nF 0 0  0.7095\n",
            ip_ref: 15.7,
        },
        Case {
            name: "HF",
            xyz: "2\nHF\nF 0 0 0\nH 0 0 0.917\n",
            ip_ref: 16.04,
        },
        Case {
            name: "C2H2",
            xyz: "4\nC2H2\nC 0 0 -0.6011\nC 0 0  0.6011\nH 0 0 -1.6612\nH 0 0  1.6612\n",
            ip_ref: 11.4,
        },
    ]
}

#[derive(Default, Clone, Copy)]
struct MethodResult {
    scf_neutral: f64,
    scf_cation: f64,
    corr_neutral: f64,
    corr_cation: f64,
}

#[allow(dead_code)]
impl MethodResult {
    fn ip_scf_ev(&self) -> f64 {
        (self.scf_cation - self.scf_neutral) * HARTREE_TO_EV
    }
    fn ip_total_ev(&self) -> f64 {
        ((self.scf_cation + self.corr_cation) - (self.scf_neutral + self.corr_neutral))
            * HARTREE_TO_EV
    }
}

/// Compute ⟨S²⟩ for a UHF/ROHF result; returns 0 for restricted.
fn s_squared(rhf: &ferric_scf::result::ScfResult, s_ao: &Array2<f64>, nocc_a: usize, nocc_b: usize) -> f64 {
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    if nocc_a == 0 || nocc_b == 0 { return s_ideal; }
    let c_a = &rhf.mos_alpha;
    let c_b = rhf.mos_beta.as_ref().unwrap_or(&rhf.mos_alpha);
    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]);
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]);
    let overlap_ab = c_a_occ.t().dot(s_ao).dot(&c_b_occ);
    let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
    s_ideal + (nocc_b as f64) - sum_sq
}

/// Diagnostic record from one cation SCF attempt.
struct CationDiag {
    method: &'static str,   // "UHF" or "ROHF"
    iters: usize,
    converged: bool,
    s2: f64,
    s2_ideal: f64,
    energy: f64,
}

fn run_case_diag(case: &Case) -> Option<(f64, f64, f64, f64, f64, CationDiag)> {
    let ctx = ParallelContext::default();
    // Upgraded from cc-pVDZ/cc-pVDZ-RI to aug-cc-pVTZ/aug-cc-pVTZ-RIFIT.
    // Diffuse functions matter for ionization potentials; TZ closes the
    // basis-set-incompleteness gap that dominated the cc-pVDZ error.
    let obs_bs = basis::bundled("aug-cc-pvtz").ok()?;
    let dfbs_bs = basis::bundled("aug-cc-pvtz-rifit").ok()?;
    let op = Operator::coulomb();

    // Parse neutral and cation as separate Molecule instances.
    let neutral = Molecule::parse_xyz(case.xyz, 0, 1).ok()?;
    let cation_mult: usize = {
        let n = neutral.nelec() as i32 - 1;
        if n % 2 == 1 { 2 } else { 3 }  // doublet if odd; triplet for even-1
    };
    // Actually cation has nelec - 1; if neutral was singlet (even), cation
    // has odd electrons → doublet. If neutral was open-shell (rare in GW100
    // subset), pick the natural cation multiplicity. Our subset is all
    // closed-shell singlets, so cation is always doublet.
    let cation = Molecule::parse_xyz(case.xyz, 1, 2).ok()?;
    let _ = cation_mult;

    let obs_n = PreparedBasis::new(&neutral, &obs_bs).ok()?;
    let dfbs_n = PreparedBasis::new(&neutral, &dfbs_bs).ok()?;
    let bounds_n = SchwarzBounds::compute(op, &obs_n).ok()?;
    let obs_c = PreparedBasis::new(&cation, &obs_bs).ok()?;
    let dfbs_c = PreparedBasis::new(&cation, &dfbs_bs).ok()?;
    let bounds_c = SchwarzBounds::compute(op, &obs_c).ok()?;

    let rhf_cfg = RhfConfig::default();
    let uhf_cfg = UhfConfig { max_iter: 200, ..Default::default() };
    let mp2_cfg = RiMp2Config { frozen_core: 0 };

    // Neutral RHF + MP2 + RPA
    let rhf_n = solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &rhf_cfg).ok()?;
    // Koopmans' theorem: IP ≈ -ε_HOMO from neutral RHF.
    let nocc_neutral = (neutral.nelec() as usize) / 2;
    let eps_n = rhf_n.eps_r();
    let ip_koopmans_ev = -eps_n[nocc_neutral - 1] * HARTREE_TO_EV;
    let mp2_n = ri_mp2(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &mp2_cfg).ok()?;
    let mut rpa_cfg = PdepRpaConfig::default();
    rpa_cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5,
    };
    rpa_cfg.trunc_thresh = 0.0;
    rpa_cfg.davidson_conv_thresh = 1e-9;
    let rpa_n = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &rpa_cfg).ok()?;

    // Cation UHF + MP2 (uses ri_mp2 which currently requires Restricted) ;
    // Actually ri_mp2 also uses rhf.mos_r() — closed-shell only. For U-MP2
    // we need a separate path. Skip Δ-MP2 for cations and report N/A.
    // UHF first; fall back to ROHF if it fails (common on symmetric cations
    // where UHF saddle-points without symmetry breaking — e.g. NH3⁺, CH4⁺).
    // Seed cation UHF from neutral RHF MOs. Removes the doublet-excited-
    // state trap that hcore guess + symmetric cations falls into (e.g.
    // H2O+ landed in ²A₁ excited doublet at -75.5465 instead of ²B₁
    // ground at -75.6318 when starting from hcore).
    let c_seed = rhf_n.mos_alpha.clone();
    let (uhf_c, diag_method): (ferric_scf::result::ScfResult, &'static str) =
        match solve_uhf_with_guess(&ctx, &cation, &obs_c, op, &bounds_c, &uhf_cfg, Some((&c_seed, &c_seed))) {
            Ok(r) => (r, "UHF(neutral-seed)"),
            Err(_) => {
                // Fall back to hcore-guess UHF, then ROHF if that also fails.
                match solve_uhf(&ctx, &cation, &obs_c, op, &bounds_c, &uhf_cfg) {
                    Ok(r) => (r, "UHF(hcore)"),
                    Err(_) => {
                        let rohf_cfg = RohfConfig { max_iter: 200, ..Default::default() };
                        let r = solve_rohf(&ctx, &cation, &obs_c, op, &bounds_c, &rohf_cfg).ok()?;
                        (r, "ROHF")
                    }
                }
            }
        };
    let s_ao = oneelectron::overlap(&obs_c);
    let nelec_c = cation.nelec() as i64;
    let mult_c = cation.multiplicity as i64;
    let two_s = mult_c - 1;
    let nocc_a = ((nelec_c + two_s) / 2) as usize;
    let nocc_b = ((nelec_c - two_s) / 2) as usize;
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let diag = CationDiag {
        method: diag_method,
        iters: uhf_c.iterations,
        converged: uhf_c.converged,
        s2: s_squared(&uhf_c, &s_ao, nocc_a, nocc_b),
        s2_ideal: s_true * (s_true + 1.0),
        energy: uhf_c.energy,
    };
    let rpa_c = run_u_pdep_rpa(&cation, &obs_c, &dfbs_c, op, &uhf_c, &rpa_cfg).ok()?;

    let ip_dscf_ev = (uhf_c.energy - rhf_n.energy) * HARTREE_TO_EV;
    let ip_dmp2_ev = {
        // No U-MP2 yet; report Δ-SCF + neutral correlation contribution
        // (a useful diagnostic but not Δ-MP2 proper).
        let e_n_mp2 = rhf_n.energy + mp2_n.mp2_corr;
        (uhf_c.energy - e_n_mp2) * HARTREE_TO_EV
    };
    let ip_drpa_ev = {
        let e_n = rhf_n.energy + rpa_n.e_rpa;
        let e_c = uhf_c.energy + rpa_c.e_rpa;
        (e_c - e_n) * HARTREE_TO_EV
    };

    // Δ-OOMP2: closed-shell OO-MP2 on neutral, U-OO-MP2 on cation.
    let oo_n = oo_ri_mp2(
        &neutral, &obs_n, &dfbs_n, op, &bounds_n, &rhf_n, &OoRiMp2Config::default(),
    ).ok();
    let oo_c = u_oo_ri_mp2(
        &cation, &obs_c, &dfbs_c, op, &bounds_c, &uhf_c, &UOoRiMp2Config::default(),
    ).ok();
    let ip_doomp2_ev = match (oo_n.as_ref(), oo_c.as_ref()) {
        (Some(n), Some(c)) => (c.total_energy - n.total_energy) * HARTREE_TO_EV,
        _ => f64::NAN,
    };

    Some((ip_koopmans_ev, ip_dscf_ev, ip_dmp2_ev, ip_doomp2_ev, ip_drpa_ev, diag))
}


fn main() {
    let cases = gw100_subset();
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "mol", "exp(eV)", "Koopmans", "ΔSCF", "ΔMP2*", "ΔOOMP2", "ΔRPA"
    );
    println!("{:-<74}", "");

    let mut mae_koop = 0.0_f64;
    let mut mae_dscf = 0.0_f64;
    let mut mae_dmp2 = 0.0_f64;
    let mut mae_doomp2 = 0.0_f64;
    let mut mae_drpa = 0.0_f64;
    let mut n_ok = 0_usize;
    let mut n_oomp2 = 0_usize;

    let mut diags: Vec<(&str, CationDiag)> = Vec::new();
    for case in &cases {
        match run_case_diag(case) {
            Some((koop, dscf, dmp2, doomp2, drpa, diag)) => {
                println!(
                    "{:<6} {:>10.2} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                    case.name, case.ip_ref, koop, dscf, dmp2, doomp2, drpa
                );
                mae_koop += (koop - case.ip_ref).abs();
                mae_dscf += (dscf - case.ip_ref).abs();
                mae_dmp2 += (dmp2 - case.ip_ref).abs();
                if doomp2.is_finite() {
                    mae_doomp2 += (doomp2 - case.ip_ref).abs();
                    n_oomp2 += 1;
                }
                mae_drpa += (drpa - case.ip_ref).abs();
                n_ok += 1;
                diags.push((case.name, diag));
            }
            None => {
                println!("{:<6} FAILED", case.name);
            }
        }
    }
    println!("\nCation SCF diagnostics:");
    println!("{:<6} {:>6} {:>5} {:>6} {:>9} {:>9} {:>14}",
        "mol", "method", "iter", "conv", "<S^2>", "ideal", "E_cation(Ha)");
    for (name, d) in &diags {
        println!("{:<6} {:>6} {:>5} {:>6} {:>9.4} {:>9.4} {:>14.6}",
            name, d.method, d.iters, d.converged, d.s2, d.s2_ideal, d.energy);
    }
    println!("{:-<74}", "");
    if n_ok > 0 {
        let n = n_ok as f64;
        let mae_oo = if n_oomp2 > 0 { mae_doomp2 / n_oomp2 as f64 } else { f64::NAN };
        println!(
            "{:<6} {:>10} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            "MAE", "", mae_koop / n, mae_dscf / n, mae_dmp2 / n, mae_oo, mae_drpa / n
        );
    }
    println!("\nKoopmans = -ε_HOMO from neutral RHF (no cation calc).");
    println!("ΔMP2* = Δ-SCF + neutral-MP2-correlation only (no U-MP2 for cation).");
    println!("ΔOOMP2 = E_U-OO-MP2(cation) − E_OO-MP2(neutral) [orbital-relaxed, both spins].");
}
