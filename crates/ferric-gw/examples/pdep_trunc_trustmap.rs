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
        "water" => (
            "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
            false, // water anion unbound at these bases — EA flagged
        ),
        other => panic!("unknown molecule '{other}' (water only in this spike)"),
    }
}

struct Row {
    thresh: f64,
    e_rpa: f64,
    ip: f64,
    ea: f64,
    alpha: f64,
    c6: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mol_name = args.get(1).map(String::as_str).unwrap_or("water");
    let basis_name = args.get(2).map(String::as_str).unwrap_or("aug-cc-pvdz");
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

    let obs_a = PreparedBasis::new(&anion, &obs_bs).unwrap();
    let dfbs_a = PreparedBasis::new(&anion, &dfbs_bs).unwrap();
    let bounds_a = SchwarzBounds::compute(op, &obs_a).unwrap();
    let uhf_a = solve_uhf_with_guess(&ctx, &anion, &obs_a, &bounds_a, &uhf_cfg, Some((&seed, &seed)))
        .expect("anion UHF");

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
        davidson_conv_thresh: 1e-7,
        ..Default::default()
    };

    let thresholds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1];
    let mut rows: Vec<Row> = Vec::new();

    for &thresh in &thresholds {
        let cfg = make_cfg(thresh);

        // RPA correlation energy (neutral).
        let rpa_n = run_pdep_rpa(&neutral, &obs_n, &dfbs_n, op, &rhf_n, &cfg).expect("rpa neutral");

        // IP, EA via ΔRPA total energies (same PDEP basis).
        let rpa_c = run_u_pdep_rpa(&cation, &obs_c, &dfbs_c, op, &uhf_c, &cfg).expect("rpa cation");
        let rpa_a = run_u_pdep_rpa(&anion, &obs_a, &dfbs_a, op, &uhf_a, &cfg).expect("rpa anion");
        let e_n = rhf_n.energy + rpa_n.e_rpa;
        let ip = (uhf_c.energy + rpa_c.e_rpa - e_n) * HA_TO_EV;
        let ea = (e_n - (uhf_a.energy + rpa_a.e_rpa)) * HA_TO_EV;

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

        eprintln!(
            "[spike] thresh={thresh:.0e}  E_rpa={:.6}  IP={ip:.4}  EA={ea:.4}  a={alpha:.4}  C6={c6:.3}",
            rpa_n.e_rpa
        );
        rows.push(Row { thresh, e_rpa: rpa_n.e_rpa, ip, ea, alpha, c6 });
    }

    // Reference = untruncated (thresh = 0), the first row.
    let r0 = &rows[0];
    println!("\n# PDEP-truncation trust map: {mol_name}/{basis_name}");
    println!("# anion_bound = {anion_bound}  (EA meaningful only if true)");
    println!("# signed error vs trunc_thresh=0 reference");
    println!(
        "{:>8} {:>10} | {:>12} {:>10} {:>10} {:>12} {:>10}",
        "thresh", "n_kept?", "dE_rpa(Ha)", "dIP(eV)", "dEA(eV)", "da(%)", "dC6(%)"
    );
    println!("{:-<86}", "");
    for r in &rows {
        let d_e = r.e_rpa - r0.e_rpa;
        let d_ip = r.ip - r0.ip;
        let d_ea = r.ea - r0.ea;
        let d_a = 100.0 * (r.alpha - r0.alpha) / r0.alpha;
        let d_c6 = 100.0 * (r.c6 - r0.c6) / r0.c6;
        println!(
            "{:>8.0e} {:>10} | {:>12.2e} {:>10.4} {:>10.4} {:>10.3} {:>10.3}",
            r.thresh, "-", d_e, d_ip, d_ea, d_a, d_c6
        );
    }
    println!(
        "\n# absolute reference (thresh=0): E_rpa={:.6} Ha  IP={:.4} eV  EA={:.4} eV  a={:.4} a.u.  C6={:.3} a.u.",
        r0.e_rpa, r0.ip, r0.ea, r0.alpha, r0.c6
    );
}
