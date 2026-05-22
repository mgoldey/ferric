//! Full GW100 sweep — IPs by every available method.
//!
//! For each molecule in the 10-molecule subset of GW100, compute the
//! vertical first IP by:
//!
//!   1. **Koopmans**: IP_K = −ε_HOMO from neutral RHF
//!   2. **ΔSCF (UHF)**: E_UHF(cation) − E_RHF(neutral)
//!   3. **ΔOOMP2**: E_U-OO-MP2(cation) − E_OO-MP2(neutral)
//!   4. **ΔRPA**: [E_UHF(cation) + E_U-RPA(cation)] − [E_RHF(neutral) + E_RPA(neutral)]
//!   5. **G0W0@HF**: −ε_HOMO^QP from G0W0 on neutral
//!   6. **COHSEX@HF**: −ε_HOMO^QP from static-W COHSEX on neutral
//!   7. **evGW₀@HF**: eigenvalue-self-consistent (Σ updated, W frozen)
//!   8. **evGW@HF**:  full eigenvalue-self-consistent
//!
//! Output: per-molecule table with IPs and MAE versus experiment.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release \
//!     --example gw100_full -p ferric-gw 2>&1 | tee docs/gw100-full-results.txt

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess, UhfConfig};
use ndarray::Array2;

const HA_TO_EV: f64 = 27.211386245988_f64;

struct Case {
    name: &'static str,
    xyz: &'static str,
    ip_ref: f64,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "H2",   xyz: "2\nH2\nH 0 0 0\nH 0 0 0.7414\n", ip_ref: 15.43 },
        Case { name: "He",   xyz: "1\nHe\nHe 0 0 0\n", ip_ref: 24.59 },
        Case { name: "H2O",  xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", ip_ref: 12.62 },
        Case { name: "NH3",  xyz: "4\nNH3\nN 0.0 0.0 0.116743\nH 0.94 0.0 -0.272400\nH -0.471 0.815 -0.272400\nH -0.471 -0.815 -0.272400\n", ip_ref: 10.82 },
        Case { name: "CH4",  xyz: "5\nCH4\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n", ip_ref: 13.6 },
        Case { name: "N2",   xyz: "2\nN2\nN 0 0 -0.5488\nN 0 0  0.5488\n", ip_ref: 15.58 },
        Case { name: "CO",   xyz: "2\nCO\nC 0 0 -0.6442\nO 0 0  0.4828\n", ip_ref: 14.01 },
        Case { name: "F2",   xyz: "2\nF2\nF 0 0 -0.7080\nF 0 0  0.7080\n", ip_ref: 15.70 },
        Case { name: "HF",   xyz: "2\nHF\nF 0 0 0.0\nH 0 0 0.9168\n", ip_ref: 16.12 },
        Case { name: "C2H2", xyz: "4\nC2H2\nC 0 0 -0.6014\nC 0 0  0.6014\nH 0 0 -1.6605\nH 0 0  1.6605\n", ip_ref: 11.40 },
    ]
}

#[derive(Default, Clone, Copy)]
struct Ips {
    koop: f64,
    dscf: f64,
    doomp2: f64,
    drpa: f64,
    g0w0: f64,
    cohsex: f64,
    evgw0: f64,
    evgw: f64,
}

fn s_squared(uhf: &ferric_scf::result::ScfResult, s_ao: &Array2<f64>, nocc_a: usize, nocc_b: usize) -> f64 {
    let c_a = &uhf.mos_alpha;
    let c_b = uhf.mos_beta.as_ref().unwrap_or(&uhf.mos_alpha);
    let s_mo_ab = c_a.slice(ndarray::s![.., ..nocc_a]).t()
        .dot(s_ao)
        .dot(&c_b.slice(ndarray::s![.., ..nocc_b]));
    let ov2: f64 = s_mo_ab.iter().map(|x| x * x).sum();
    let s2_z = 0.25 * (nocc_a as f64 - nocc_b as f64).powi(2);
    let s2_xy = 0.5 * (nocc_a as f64 + nocc_b as f64) - ov2;
    s2_z + s2_xy
}

#[allow(dead_code)]
struct CationDiag {
    method: &'static str,
    iters: usize,
    converged: bool,
    s2: f64,
    s2_ideal: f64,
    energy: f64,
}

fn run_case(case: &Case) -> Option<(Ips, CationDiag)> {
    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled("aug-cc-pvtz").ok()?;
    let dfbs_bs = basis::bundled("aug-cc-pvtz-rifit").ok()?;
    let op = Operator::coulomb();

    let neutral = Molecule::parse_xyz(case.xyz, 0, 1).ok()?;
    let cation  = Molecule::parse_xyz(case.xyz, 1, 2).ok()?;

    let obs_n = PreparedBasis::new(&neutral, &obs_bs).ok()?;
    let dfbs_n = PreparedBasis::new(&neutral, &dfbs_bs).ok()?;
    let bounds_n = SchwarzBounds::compute(op, &obs_n).ok()?;
    let obs_c = PreparedBasis::new(&cation, &obs_bs).ok()?;
    let dfbs_c = PreparedBasis::new(&cation, &dfbs_bs).ok()?;
    let bounds_c = SchwarzBounds::compute(op, &obs_c).ok()?;

    let rhf_n = solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &RhfConfig::default()).ok()?;
    let nocc_n = (neutral.nelec() as usize) / 2;
    let homo_abs = nocc_n - 1;
    let ip_koop = -rhf_n.eps_r()[homo_abs] * HA_TO_EV;

    let mut rpa_cfg = PdepRpaConfig::default();
    rpa_cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5,
    };
    rpa_cfg.trunc_thresh = 0.0;
    rpa_cfg.davidson_conv_thresh = 1e-9;
    let rpa_n = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &rpa_cfg).ok()?;

    let uhf_cfg = UhfConfig { max_iter: 200, ..Default::default() };
    let c_seed = rhf_n.mos_alpha.clone();
    let (uhf_c, diag_method) = match solve_uhf_with_guess(&ctx, &cation, &obs_c, op, &bounds_c, &uhf_cfg, Some((&c_seed, &c_seed))) {
        Ok(r) => (r, "UHF(neutral-seed)"),
        Err(_) => match solve_uhf(&ctx, &cation, &obs_c, op, &bounds_c, &uhf_cfg) {
            Ok(r) => (r, "UHF(hcore)"),
            Err(_) => {
                let r = solve_rohf(&ctx, &cation, &obs_c, op, &bounds_c, &RohfConfig::default()).ok()?;
                (r, "ROHF")
            }
        },
    };
    let s_ao = oneelectron::overlap(&obs_c);
    let nelec_c = cation.nelec() as i64;
    let two_s = cation.multiplicity as i64 - 1;
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

    let ip_dscf = (uhf_c.energy - rhf_n.energy) * HA_TO_EV;
    let rpa_c = run_u_pdep_rpa(&cation, &obs_c, &dfbs_c, op, &uhf_c, &rpa_cfg).ok()?;
    let ip_drpa = {
        let e_n = rhf_n.energy + rpa_n.e_rpa;
        let e_c = uhf_c.energy + rpa_c.e_rpa;
        (e_c - e_n) * HA_TO_EV
    };

    let oo_n = oo_ri_mp2(&neutral, &obs_n, &dfbs_n, op, &bounds_n, &rhf_n, &OoRiMp2Config::default()).ok();
    let oo_c = u_oo_ri_mp2(&cation, &obs_c, &dfbs_c, op, &bounds_c, &uhf_c, &UOoRiMp2Config::default()).ok();
    let ip_doomp2 = match (oo_n.as_ref(), oo_c.as_ref()) {
        (Some(n), Some(c)) => (c.total_energy - n.total_energy) * HA_TO_EV,
        _ => f64::NAN,
    };

    let pdep_cfg_gw = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5,
        },
        davidson_conv_thresh: 1e-7,
        davidson_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
    };
    let mut ip_g0w0 = f64::NAN;
    let mut ip_cohsex = f64::NAN;
    let mut ip_evgw0 = f64::NAN;
    let mut ip_evgw = f64::NAN;
    for (method, slot) in [
        (GwMethod::G0W0,   &mut ip_g0w0),
        (GwMethod::Cohsex, &mut ip_cohsex),
        (GwMethod::EvGw0,  &mut ip_evgw0),
        (GwMethod::EvGw,   &mut ip_evgw),
    ] {
        let gcfg = GwConfig { method, max_ev_iter: 8, ev_conv_thresh: 1e-4, ..Default::default() };
        if let Ok(res) = run_gw(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &pdep_cfg_gw, &gcfg) {
            if let Some(local) = res.mo_indices.iter().position(|&i| i == homo_abs) {
                *slot = -res.eps_qp[local] * HA_TO_EV;
            }
        }
    }

    Some((
        Ips { koop: ip_koop, dscf: ip_dscf, doomp2: ip_doomp2, drpa: ip_drpa,
              g0w0: ip_g0w0, cohsex: ip_cohsex, evgw0: ip_evgw0, evgw: ip_evgw },
        diag,
    ))
}

fn main() {
    let cases = cases();
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "mol", "exp(eV)", "Koop", "ΔSCF", "ΔOOMP2", "ΔRPA", "G0W0", "COHSEX", "evGW0", "evGW"
    );
    println!("{:-<92}", "");

    let mut sum_abs = [0.0f64; 8];
    let mut n_ok = [0usize; 8];
    let mut diags: Vec<(&str, CationDiag)> = Vec::new();

    for case in &cases {
        match run_case(case) {
            Some((ips, diag)) => {
                println!(
                    "{:<6} {:>8.2} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                    case.name, case.ip_ref,
                    ips.koop, ips.dscf, ips.doomp2, ips.drpa,
                    ips.g0w0, ips.cohsex, ips.evgw0, ips.evgw
                );
                for (k, v) in [ips.koop, ips.dscf, ips.doomp2, ips.drpa,
                               ips.g0w0, ips.cohsex, ips.evgw0, ips.evgw].iter().enumerate() {
                    if v.is_finite() {
                        sum_abs[k] += (v - case.ip_ref).abs();
                        n_ok[k] += 1;
                    }
                }
                diags.push((case.name, diag));
            }
            None => println!("{:<6} FAILED", case.name),
        }
    }

    println!("{:-<92}", "");
    let mae: Vec<String> = sum_abs.iter().zip(n_ok.iter()).map(|(s, n)| {
        if *n > 0 { format!("{:>8.3}", s / *n as f64) } else { "     n/a".to_string() }
    }).collect();
    println!(
        "{:<6} {:>8} {} {} {} {} {} {} {} {}",
        "MAE", "",
        mae[0], mae[1], mae[2], mae[3], mae[4], mae[5], mae[6], mae[7],
    );

    println!("\nCation SCF diagnostics:");
    println!("{:<6} {:>18} {:>5} {:>5} {:>9} {:>9} {:>14}",
        "mol", "method", "iter", "conv", "<S^2>", "ideal", "E_cation(Ha)");
    for (name, d) in &diags {
        println!("{:<6} {:>18} {:>5} {:>5} {:>9.4} {:>9.4} {:>14.6}",
            name, d.method, d.iters, d.converged, d.s2, d.s2_ideal, d.energy);
    }

    println!("\nKoopmans, G0W0/COHSEX/evGW(0): direct QP energies on neutral RHF/HF.");
    println!("ΔSCF/ΔOOMP2/ΔRPA: cation − neutral total-energy differences.");
}
