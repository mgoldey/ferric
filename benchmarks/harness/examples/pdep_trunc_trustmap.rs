//! PDEP-truncation trust map: how does dropping PDEP eigenpotentials affect
//! each downstream observable — RPA correlation energy, IP, EA, static
//! polarizability α(0), and molecular C6?
//!
//! Truncation is by eigenvalue threshold: eigenpotentials with
//! |λ_α(0) − 1| ≤ trunc_thresh are dropped (weakly-screening modes). Sweeping
//! trunc_thresh upward keeps fewer, stronger PDEP vectors. trunc_thresh = 0 is
//! the untruncated reference; every observable is reported as a signed error
//! vs that reference.
//!
//! Closed-shell only. IP/EA here use the ΔRPA total-energy route
//! (cation/anion − neutral), which exercises the SAME PDEP basis as the
//! property channels, so all five columns probe one truncation knob.
//!
//! Run (memory-scoped, single-thread, yields to other load):
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo run --release \
//!     -p ferric-gw --example pdep_trunc_trustmap -- <molecule> <basis>
//!
//! molecule ∈ {water}, basis ∈ {aug-cc-pvdz, cc-pvdz}. The molecule's anion
//! must be (at least weakly) bound for the EA column to be meaningful; water's
//! is not, so EA is reported-but-flagged. A bound-anion ~10-atom system is the
//! deferred second target.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::properties::molecular_dynamic_polarizability_pdep;
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf_with_guess, UhfConfig};

const HA_TO_EV: f64 = 27.211_386_245_988;

/// (xyz string, neutral multiplicity 1, bound-anion? flag)
fn geometry(name: &str) -> (&'static str, bool) {
    match name {
        // Na4 — the molecule that hung the GW100 sweep (4+ CPU-hr). Large nov
        // (1892) + oversized cross-family def2 aux → big PDEP. Used to test
        // whether trunc @1e-4 rescues it (timing AND IP-unchanged).
        "na4" => (
            "4\nNa4\n\
             Na 0.0002445 -0.0998053  1.5471126\nNa -0.0002444 3.1776586 0.0486374\n\
             Na 0.0002444  0.0997722 -1.5472150\nNa -0.0002444 -3.1776254 -0.0485350\n",
            false, // Na4 anion UNBOUND (electropositive metal) — EA skipped, not hung
        ),
        "water" => (
            "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
            false, // water anion unbound at these bases — EA flagged
        ),
        "ethylene" => (
            // C2H4 D2h, experimental r_CC=1.339, r_CH=1.086 Å, HCH=117.4°.
            // Mid-size π system (6 atoms) between water and benzene. Anion unbound.
            "6\nC2H4\n\
             C  0.000000 0.000000  0.669500\nC  0.000000 0.000000 -0.669500\n\
             H  0.000000 0.922832  1.237695\nH  0.000000 -0.922832 1.237695\n\
             H  0.000000 0.922832 -1.237695\nH  0.000000 -0.922832 -1.237695\n",
            false,
        ),
        "benzene" => (
            // D6h, experimental r_CC=1.3915, r_CH=1.0800 Å. Anion is a shape
            // resonance (unbound) at these bases → EA flagged, not trusted.
            "12\nC6H6\n\
             C  1.3915  0.0000 0.0\nC  0.6957  1.2050 0.0\nC -0.6957  1.2050 0.0\n\
             C -1.3915  0.0000 0.0\nC -0.6957 -1.2050 0.0\nC  0.6957 -1.2050 0.0\n\
             H  2.4715  0.0000 0.0\nH  1.2357  2.1403 0.0\nH -1.2357  2.1403 0.0\n\
             H -2.4715  0.0000 0.0\nH -1.2357 -2.1403 0.0\nH  1.2357 -2.1403 0.0\n",
            false,
        ),
        other => panic!("unknown molecule '{other}' (water, ethylene, benzene)"),
    }
}

struct Row {
    thresh: f64,
    m_kept: usize,    // retained PDEP eigenpotentials at this threshold
    naux: usize,      // total aux functions (= M at thresh 0)
    e_rpa: f64,
    ip: f64,
    ea: f64,
    alpha: f64,
    c6: f64,
    gw_ip: [f64; 4],  // [G0W0, evGW0, evGW, COHSEX] HOMO IP (eV)
    gw_gap: [f64; 4], // QP gap = LUMO_qp − HOMO_qp (eV)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mol_name = args.get(1).map(String::as_str).unwrap_or("water");
    let basis_name = args.get(2).map(String::as_str).unwrap_or("aug-cc-pvdz");
    // --no-gw skips the GW family (4 methods × all thresholds) — expensive on
    // large systems and the least essential column for a truncation study, since
    // energy/IP/EA/α/C6 already show the scaling. GW truncation is on water/ethylene.
    let skip_gw = args.iter().any(|a| a == "--no-gw");
    let (xyz, anion_bound) = geometry(mol_name);

    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled(basis_name).expect("orbital basis");
    // RI-fit aux: aug-cc-pvtz-rifit is the bundled RPA aux; use the matching DZ.
    let dfbs_name = if basis_name.contains("aug") {
        "aug-cc-pvdz-rifit"
    } else {
        "cc-pvdz-ri"
    };
    let dfbs_bs = basis::bundled(dfbs_name).expect("aux basis");
    let op = Operator::coulomb();

    let neutral = Molecule::parse_xyz(xyz, 0, 1).expect("neutral");
    let cation = Molecule::parse_xyz(xyz, 1, 2).expect("cation");
    let anion = Molecule::parse_xyz(xyz, -1, 2).expect("anion");

    let obs_n = PreparedBasis::new(&neutral, &obs_bs).unwrap();
    let dfbs_n = PreparedBasis::new(&neutral, &dfbs_bs).unwrap();
    let bounds_n = SchwarzBounds::compute(op, &obs_n).unwrap();
    let rhf_n = solve_rhf(&ctx, &neutral, &obs_n, op, &bounds_n, &RhfConfig::default())
        .expect("neutral RHF");

    // Cation / anion UHF, neutral-MO seeded (per gw100 best practice).
    let uhf_cfg = UhfConfig { max_iter: 200, ..Default::default() };
    let seed = rhf_n.mos_alpha.clone();

    let obs_c = PreparedBasis::new(&cation, &obs_bs).unwrap();
    let dfbs_c = PreparedBasis::new(&cation, &dfbs_bs).unwrap();
    let bounds_c = SchwarzBounds::compute(op, &obs_c).unwrap();
    let uhf_c = solve_uhf_with_guess(&ctx, &cation, &obs_c, &bounds_c, &uhf_cfg, Some((&seed, &seed)))
        .expect("cation UHF");

    // Only solve the anion when it is (at least weakly) BOUND. For electropositive
    // species (Na4, K2, …) the extra electron is unbound: in a finite Gaussian
    // basis the UHF cannot converge (it spins all max_iter, ~hours on a 100+-bf
    // cluster) and the EA is physically meaningless anyway. Skip it → EA = NaN.
    let anion_ctx = if anion_bound {
        let obs_a = PreparedBasis::new(&anion, &obs_bs).unwrap();
        let dfbs_a = PreparedBasis::new(&anion, &dfbs_bs).unwrap();
        let bounds_a = SchwarzBounds::compute(op, &obs_a).unwrap();
        let uhf_a = solve_uhf_with_guess(&ctx, &anion, &obs_a, &bounds_a, &uhf_cfg, Some((&seed, &seed)))
            .expect("anion UHF");
        Some((obs_a, dfbs_a, uhf_a))
    } else {
        eprintln!("[spike] {mol_name}: anion unbound (electropositive) — EA skipped");
        None
    };

    eprintln!(
        "[spike] {mol_name}/{basis_name} aux={dfbs_name}  neutral E={:.6}  anion_bound={anion_bound}",
        rhf_n.energy
    );

    // Build a PdepRpaConfig at a given truncation threshold; everything else fixed.
    let make_cfg = |thresh: f64| PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        trunc_thresh: thresh,
        eigensolver_conv_thresh: 1e-7,
        ..Default::default()
    };

    // Truncation drops eigenpotentials with |λ(0)−1| ≤ thresh. Production uses
    // the 1e-4 default, so resolve THAT regime finely (1e-6 → 1e-2) to confirm
    // the default is safe with margin — reported alongside M_kept/naux so safety
    // is expressed as a mode fraction, not just a % error. (Past ~1e-2 we never
    // operate, so the lax 0.1+ end is dropped.)
    // Default grid includes thresh=0 (full-rank reference). For systems where
    // full-rank is intractable (large nov + oversized aux, e.g. Na4 — the whole
    // point of truncation), set TRUSTMAP_THRESHOLDS to a comma list to skip 0 and
    // confirm the TRUNCATED IPs agree with each other (convergence = the answer).
    let thresholds: Vec<f64> = std::env::var("TRUSTMAP_THRESHOLDS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]);
    let mut rows: Vec<Row> = Vec::new();

    for &thresh in &thresholds {
        let t_mol = std::time::Instant::now();
        let cfg = make_cfg(thresh);

        // RPA correlation energy (neutral).
        let rpa_n = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &cfg).expect("rpa neutral");

        // IP, EA via ΔRPA total energies (same PDEP basis).
        let rpa_c = run_u_pdep_rpa(&cation, &obs_c, &dfbs_c, op, &uhf_c, &cfg).expect("rpa cation");
        let e_n = rhf_n.energy + rpa_n.e_rpa;
        let ip = (uhf_c.energy + rpa_c.e_rpa - e_n) * HA_TO_EV;
        // EA only when the anion is bound (skipped for electropositive species).
        let ea = match anion_ctx.as_ref() {
            Some((obs_a, dfbs_a, uhf_a)) => {
                let rpa_a = run_u_pdep_rpa(&anion, obs_a, dfbs_a, op, uhf_a, &cfg).expect("rpa anion");
                (e_n - (uhf_a.energy + rpa_a.e_rpa)) * HA_TO_EV
            }
            None => f64::NAN,
        };

        // Truncation-aware dynamic polarizability — feed the ALREADY-TRUNCATED
        // rpa_n result to the _truncated dispersion path (the full
        // pdep_dynamic_polarizability and pdep_polarizability_static IGNORE
        // trunc_thresh — see pdep-trunc-noop memory). C6 is the global-origin
        // molecular Casimir-Polder integral (DOSD-comparable). Static α is the
        // lowest-frequency node of the same truncated dynamic tensor (the static
        // path has no truncation knob), iso = Tr/3, labelled as a node proxy.
        // Truncation-aware molecular α(iω) in the SAME retained PDEP basis as the
        // energy/GW columns (molecular_dynamic_polarizability_pdep). Static α = the
        // ω=0 (lowest node) iso = Tr/3. Molecular C6 = (3/π) Σ_k w_k ᾱ(iω_k)²
        // (the DOSD-comparable c6_molecular_iso definition).
        let mol_alpha = molecular_dynamic_polarizability_pdep(&rpa_n, &neutral, &obs_n, &dfbs_n, &rhf_n, op)
            .expect("molecular pdep alpha");
        let a0 = &mol_alpha[0];
        let alpha = (a0[0][0] + a0[1][1] + a0[2][2]) / 3.0;
        let c6 = {
            let w = &rpa_n.quad_weights;
            let mut s = 0.0;
            for k in 0..mol_alpha.len() {
                let a = &mol_alpha[k];
                let iso = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
                s += w[k] * iso * iso;
            }
            (3.0 / std::f64::consts::PI) * s
        };

        // GW family vs truncation — same neutral RHF + same trunc_thresh (cfg).
        let nocc_n = (neutral.nelec() as usize) / 2;
        let homo_abs = nocc_n - 1;
        let lumo_abs = nocc_n;
        let mut gw_ip = [f64::NAN; 4];
        let mut gw_gap = [f64::NAN; 4];
        let gw_methods: &[GwMethod] = if skip_gw {
            &[]
        } else {
            &[GwMethod::G0W0, GwMethod::EvGw0, GwMethod::EvGw, GwMethod::Cohsex]
        };
        for (mi, &method) in gw_methods.iter().enumerate() {
            let gcfg = GwConfig {
                method,
                qp_mos: Some(homo_abs..lumo_abs + 1),
                max_ev_iter: 8,
                ev_conv_thresh: 1e-4,
                ..Default::default()
            };
            if let Ok(res) = run_gw(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &cfg, &gcfg, None) {
                let homo_qp = res.mo_indices.iter().position(|&i| i == homo_abs)
                    .map(|loc| res.eps_qp[loc]);
                let lumo_qp = res.mo_indices.iter().position(|&i| i == lumo_abs)
                    .map(|loc| res.eps_qp[loc]);
                if let Some(h) = homo_qp {
                    gw_ip[mi] = -h * HA_TO_EV;
                    if let Some(l) = lumo_qp {
                        gw_gap[mi] = (l - h) * HA_TO_EV;
                    }
                }
            }
        }

        eprintln!(
            "[spike] thresh={thresh:.0e}  M_kept={}  IP={ip:.4}  G0W0={:.4}  evGW_IP={:.3}  wall={:.1}s",
            rpa_n.n_eigenpotentials, gw_ip[0], gw_ip[2], t_mol.elapsed().as_secs_f64()
        );
        rows.push(Row {
            thresh,
            m_kept: rpa_n.n_eigenpotentials,
            naux: dfbs_n.nbasis(),
            e_rpa: rpa_n.e_rpa,
            ip, ea, alpha, c6, gw_ip, gw_gap,
        });
    }

    // Reference = untruncated (thresh = 0), the first row.
    let r0 = &rows[0];
    println!("\n# PDEP-truncation trust map: {mol_name}/{basis_name}");
    println!("# anion_bound = {anion_bound}  (EA meaningful only if true)");
    println!("# signed error vs trunc_thresh=0 reference");
    println!(
        "{:>8} {:>13} | {:>12} {:>10} {:>10} {:>12} {:>10}",
        "thresh", "M_kept/naux", "dE_rpa(Ha)", "dIP(eV)", "dEA(eV)", "da(%)", "dC6(%)"
    );
    println!("{:-<90}", "");
    for r in &rows {
        let d_e = r.e_rpa - r0.e_rpa;
        let d_ip = r.ip - r0.ip;
        let d_ea = r.ea - r0.ea;
        let d_a = 100.0 * (r.alpha - r0.alpha) / r0.alpha;
        let d_c6 = 100.0 * (r.c6 - r0.c6) / r0.c6;
        let frac = format!("{}/{} ({:.0}%)", r.m_kept, r.naux,
                           100.0 * r.m_kept as f64 / r.naux as f64);
        println!(
            "{:>8.0e} {:>13} | {:>12.2e} {:>10.4} {:>10.4} {:>10.3} {:>10.3}",
            r.thresh, frac, d_e, d_ip, d_ea, d_a, d_c6
        );
    }
    println!(
        "\n# absolute reference (thresh=0): E_rpa={:.6} Ha  IP={:.4} eV  EA={:.4} eV  a={:.4} a.u.  C6={:.3} a.u.",
        r0.e_rpa, r0.ip, r0.ea, r0.alpha, r0.c6
    );

    // GW-family IP vs trunc_thresh (signed error vs thresh=0).
    let labels = ["G0W0", "evGW0", "evGW", "COHSEX"];
    println!("\n# GW-family HOMO IP (eV) vs trunc_thresh — signed error vs thresh=0");
    println!("{:>8} | {:>10} {:>10} {:>10} {:>10}", "thresh", labels[0], labels[1], labels[2], labels[3]);
    println!("{:-<58}", "");
    for r in &rows {
        print!("{:>8.0e} |", r.thresh);
        for mi in 0..4 {
            print!(" {:>10.4}", r.gw_ip[mi] - rows[0].gw_ip[mi]);
        }
        println!();
    }
    println!(
        "# absolute GW IP at thresh=0 (eV): G0W0={:.3} evGW0={:.3} evGW={:.3} COHSEX={:.3}",
        rows[0].gw_ip[0], rows[0].gw_ip[1], rows[0].gw_ip[2], rows[0].gw_ip[3]
    );
    println!(
        "# absolute QP gap at thresh=0 (eV): G0W0={:.3} evGW0={:.3} evGW={:.3} COHSEX={:.3}",
        rows[0].gw_gap[0], rows[0].gw_gap[1], rows[0].gw_gap[2], rows[0].gw_gap[3]
    );
}
