//! AO-basis cubic-scaling RPA vs canonical RI-dRPA benchmark.
//!
//! Sweeps multiple ω-quadrature configurations on a single molecule and
//! reports, for each:
//!
//! - **Canonical RI-dRPA** (MO basis, O(N⁴) per ω) via
//!   [`ferric_rpa::diagnostics::ri_drpa_energy`].
//! - **AO RPA via Kaltak-Kresse τ** with **no MO fallback** (pure O(N³))
//!   via [`ferric_rpa::ao_rpa::ao_rpa_correlation_energy`] (b_ov=None).
//! - **AO RPA with fallback** — uses the MO-Π formula at ω·t_max > π/2;
//!   this is the safe/correct mode for production use and reports how
//!   many of the n_ω quadrature points hit the fallback.
//!
//! The sweep tests:
//!   - GaussLegendre with u₀ ∈ {0.5}        — unbounded; high ω → fallback
//!   - ChebyshevTan  with u₀ ∈ {0.2, 0.1}   — bounded max ω; targets n_fb=0
//!   - varying n_ω ∈ {10, 16, 20}
//!
//! The "win" condition for genuine cubic scaling: an (n_ω, u₀) where
//! n_fb=0 AND |Δ| < 50 µHa vs canonical on H2O.
//!
//! Usage:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-rpa --example ao_rpa_bench

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::compute_rpa_intermediates;
use ferric_rpa::ao_rpa::{ao_rpa_correlation_energy, build_tau_quadrature};
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::diagnostics::ri_drpa_energy;
use ferric_rpa::quadrature;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

struct CasePath {
    name: &'static str,
    path: Option<&'static str>,
    xyz: Option<&'static str>,
}

fn cases() -> Vec<CasePath> {
    vec![
        CasePath { name: "H2",  path: None, xyz: Some("2\nH2\nH 0 0 0\nH 0 0 0.7414\n") },
        CasePath { name: "H2O", path: None, xyz: Some("3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n") },
        CasePath { name: "NH3", path: None, xyz: Some("4\nNH3\nN 0.0 0.0 0.116743\nH 0.94 0.0 -0.272400\nH -0.471 0.815 -0.272400\nH -0.471 -0.815 -0.272400\n") },
        CasePath { name: "CH4", path: None, xyz: Some("5\nCH4\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n") },
    ]
}

#[derive(Clone)]
struct OmegaCfg {
    label: &'static str,
    scheme: QuadratureScheme,
    n_points: usize,
    u0: f64,
}

fn omega_cfgs() -> Vec<OmegaCfg> {
    vec![
        OmegaCfg { label: "GL    20 u0=0.5", scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5 },
        OmegaCfg { label: "GL    10 u0=0.5", scheme: QuadratureScheme::GaussLegendre, n_points: 10, u0: 0.5 },
        OmegaCfg { label: "Cheb  20 u0=0.2", scheme: QuadratureScheme::ChebyshevTan,  n_points: 20, u0: 0.2 },
        OmegaCfg { label: "Cheb  16 u0=0.2", scheme: QuadratureScheme::ChebyshevTan,  n_points: 16, u0: 0.2 },
        OmegaCfg { label: "Cheb  10 u0=0.2", scheme: QuadratureScheme::ChebyshevTan,  n_points: 10, u0: 0.2 },
        OmegaCfg { label: "Cheb  10 u0=0.1", scheme: QuadratureScheme::ChebyshevTan,  n_points: 10, u0: 0.1 },
    ]
}

fn main() {
    let ctx = ParallelContext::default();
    let obs_set = basis::bundled("cc-pvdz").unwrap();
    let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let n_tau = 12usize;

    println!("AO-RPA Kaltak-Kresse ω-quadrature spike  —  cc-pVDZ / cc-pVDZ-RI");
    println!("τ-grid: {n_tau}-pt minimax for AO path; canonical uses same ω-grid as AO");
    println!("Goal: find (scheme, n_ω, u₀) where n_fb=0 AND |Δ| vs canonical < 50 µHa");
    println!();
    println!("{:<6} {:<18} {:>14} {:>14} {:>10} {:>5} {:>8} {:>8}",
        "mol", "ω-config", "E_c canon", "E_c AO-τ", "Δ (µHa)", "n_fb", "t_canon", "t_AO");
    println!("{:-<92}", "");

    for case in &cases() {
        let mol = match (case.path, case.xyz) {
            (Some(p), _) => match Molecule::load_xyz(p) {
                Ok(m) => m,
                Err(_) => { println!("{:<6} (missing: {})", case.name, p); continue; }
            },
            (None, Some(x)) => Molecule::parse_xyz(x, 0, 1).unwrap(),
            _ => continue,
        };
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        let rhf_cfg = RhfConfig::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();

        let nbas = obs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc = nelec / 2;
        let nvir = nbas - nocc;

        let inter = compute_rpa_intermediates(
            &mol, &obs, &dfbs, op, &rhf, &ferric_mp2::rimp2::RiMp2Config { frozen_core: 0, memory_budget_bytes: None },
        ).unwrap();
        let eps = rhf.eps_r();
        let eps_occ: Vec<f64> = eps.iter().take(nocc).copied().collect();
        let eps_vir: Vec<f64> = eps.iter().skip(nocc).take(nvir).copied().collect();

        let eri3 = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

        // Report t_max so we know the safe-ω window for this molecule.
        let lap = build_tau_quadrature(&eps_occ, &eps_vir, n_tau);
        let t_max = lap.points.iter().cloned().fold(0.0_f64, f64::max);
        let omega_safe = std::f64::consts::FRAC_PI_2 / t_max;
        println!(">> {} nbas={} nocc={} t_max={:.3}  ω·t_max≤π/2 ⇒ ω≤{:.3} Ha",
            case.name, nbas, nocc, t_max, omega_safe);

        // Adaptive ChebyshevTan u₀ targeting ω_max · t_max = π/2 for each N.
        let mut adaptive: Vec<OmegaCfg> = vec![];
        for &n_om in &[10usize, 16, 20] {
            let nf = n_om as f64;
            let tan_max = (std::f64::consts::PI * nf / (2.0 * (nf + 1.0))).tan();
            let u0_adapt = std::f64::consts::FRAC_PI_2 / (t_max * tan_max);
            adaptive.push(OmegaCfg {
                label: Box::leak(format!("Cheb-adapt n={} u₀={:.4}", n_om, u0_adapt).into_boxed_str()),
                scheme: QuadratureScheme::ChebyshevTan,
                n_points: n_om,
                u0: u0_adapt,
            });
        }
        let mut configs = omega_cfgs();
        configs.extend(adaptive);

        for ocfg in &configs {
            let quad_cfg = QuadratureConfig {
                scheme: ocfg.scheme.clone(),
                n_points: ocfg.n_points,
                u0: ocfg.u0,
            };
            let (quad_freqs, quad_weights) = quadrature::build_quadrature(&quad_cfg);
            let omega_max = quad_freqs.iter().cloned().fold(0.0_f64, f64::max);

            // Canonical reference on this same ω-grid.
            let t0 = Instant::now();
            let e_canon = ri_drpa_energy(
                &inter.b_ov, &eps_occ, &eps_vir, &quad_freqs, &quad_weights,
            ).unwrap();
            let t_canon = t0.elapsed().as_secs_f64();

            // AO-τ path WITHOUT fallback — pure cubic candidate.
            let t0 = Instant::now();
            let (e_ao, _, _, n_fb) = ao_rpa_correlation_energy(
                &eri3, &inter.v_inv_sqrt, &c_occ, &c_vir,
                &eps_occ, &eps_vir,
                &quad_freqs, &quad_weights,
                n_tau,
                None,   // <-- no fallback; aliasing here will show up as a big |Δ|
            ).unwrap();
            let t_ao = t0.elapsed().as_secs_f64();

            let delta_uha = (e_ao - e_canon) * 1e6;
            let _ = nbas;
            let _ = nvir;
            let max_wt = omega_max * t_max;
            println!("{:<6} {:<18} ω_max={:>6.2} ω·t={:>5.2}  {:>10.6} {:>10.6} {:>10.1} {:>5} {:>7.3}s {:>7.3}s",
                case.name, ocfg.label, omega_max, max_wt, e_canon, e_ao, delta_uha, n_fb, t_canon, t_ao);
        }
        println!();
    }
}
