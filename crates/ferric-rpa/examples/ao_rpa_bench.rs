//! AO-basis cubic-scaling RPA vs canonical RI-dRPA benchmark.
//!
//! Compares two paths to the RI-dRPA correlation energy:
//!
//!   - **Canonical RI-dRPA** (MO basis, O(N⁴) per ω, current ferric default)
//!     via [`ferric_rpa::diagnostics::ri_drpa_energy`].
//!
//!   - **AO RPA via Kaltak-Kresse imaginary-time** (AO basis, O(N³) per τ)
//!     via [`ferric_rpa::ao_rpa::ao_rpa_correlation_energy`].
//!
//! For each system we report
//!   - E_c canonical (Ha)
//!   - E_c AO-imag-time (Ha)
//!   - Δ in milli-Ha (signed)
//!   - wall-clock for each path
//!
//! Usage:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-rpa --example ao_rpa_bench
//!
//! Defaults: cc-pVDZ + cc-pVDZ-RI, 20-pt Gauss-Legendre ω, 12-pt minimax τ.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::compute_rpa_intermediates;
use ferric_rpa::ao_rpa::ao_rpa_correlation_energy;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::diagnostics::ri_drpa_energy;
use ferric_rpa::quadrature;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

struct Case {
    name: &'static str,
    xyz: &'static str,
}

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
        CasePath { name: "n-hexane",    path: Some("testdata/molecules/scaling/n-hexane.xyz"),    xyz: None },
        CasePath { name: "naphthalene", path: Some("testdata/molecules/scaling/naphthalene.xyz"), xyz: None },
        CasePath { name: "n-decane",    path: Some("testdata/molecules/scaling/n-decane.xyz"),    xyz: None },
    ]
}

fn main() {
    let ctx = ParallelContext::default();
    let obs_set = basis::bundled("cc-pvdz").unwrap();
    let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let n_tau = 12usize;
    let n_omega = 20usize;

    println!("AO-RPA (Kaltak-Kresse) vs canonical RI-dRPA  —  cc-pVDZ / cc-pVDZ-RI");
    println!("τ-grid: {n_tau}-pt minimax,  ω-grid: {n_omega}-pt Gauss-Legendre");
    println!("{:<6} {:>4} {:>5} {:>5} {:>14} {:>14} {:>10} {:>5} {:>8} {:>8}",
        "mol", "nbas", "naux", "nocc", "E_c canon (Ha)", "E_c AO-τ (Ha)",
        "Δ (mHa)", "n_fb", "t_canon", "t_AO");
    println!("{:-<98}", "");

    let quad_cfg = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre,
        n_points: n_omega,
        u0: 0.5,
    };
    let (quad_freqs, quad_weights) = quadrature::build_quadrature(&quad_cfg);

    for case in &cases() {
        let mol = match (case.path, case.xyz) {
            (Some(p), _) => match Molecule::load_xyz(p) {
                Ok(m) => m,
                Err(_) => { println!("{:<12} (missing: {})", case.name, p); continue; }
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
        let naux = dfbs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc = nelec / 2;
        let nvir = nbas - nocc;

        // Canonical MO B_ov (b_ov is V^{-1/2}-dressed, ready for ri_drpa_energy).
        let inter = compute_rpa_intermediates(
            &mol, &obs, &dfbs, op, &rhf, &ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 },
        ).unwrap();
        let eps = rhf.eps_r();
        let eps_occ: Vec<f64> = eps.iter().take(nocc).copied().collect();
        let eps_vir: Vec<f64> = eps.iter().skip(nocc).take(nvir).copied().collect();

        // Canonical RI-dRPA timing.
        let t0 = Instant::now();
        let e_canon = ri_drpa_energy(
            &inter.b_ov, &eps_occ, &eps_vir, &quad_freqs, &quad_weights,
        ).unwrap();
        let t_canon = t0.elapsed().as_secs_f64();

        // AO-RPA setup: dressed eri3 happens inside the call.
        let eri3 = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

        let t0 = Instant::now();
        let (e_ao, _, _, n_fb) = ao_rpa_correlation_energy(
            &eri3, &inter.v_inv_sqrt, &c_occ, &c_vir,
            &eps_occ, &eps_vir,
            &quad_freqs, &quad_weights,
            n_tau,
            Some(&inter.b_ov),
        ).unwrap();
        let t_ao = t0.elapsed().as_secs_f64();


        let delta_mha = (e_ao - e_canon) * 1000.0;
        println!("{:<6} {:>4} {:>5} {:>5} {:>14.6} {:>14.6} {:>10.3} {:>5} {:>8.3}s {:>8.3}s",
            case.name, nbas, naux, nocc, e_canon, e_ao, delta_mha, n_fb, t_canon, t_ao);
    }
}
