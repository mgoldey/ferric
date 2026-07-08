//! PNO/OSV (DNV) compression sweep — measures ferric's ACTUAL prefactor win.
//!
//! `run_pdep_rpa_osv` compresses the virtual space to a shared reduced basis of
//! size n_vir_reduced (per t_osv threshold), then runs the SAME dense dielectric
//! solve in that smaller space. It is O(N⁴) like dense RPA — the win is the ratio
//! n_vir_reduced/nvir (a PREFACTOR cut, not a scaling-order change), because the
//! DNV transform re-canonicalizes and is virtual-space COMPRESSION, not spatial
//! localization.
//!
//! This sweep reports, per molecule × t_osv: the energy vs the dense reference,
//! the compression ratio, and wall-time speedup — the real accuracy/cost curve to
//! pick a production t_osv, replacing literature estimates with ferric numbers.
//!
//! Usage: OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!          cargo run --release -p ferric-rpa --example osv_scaling_sweep

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::pno::run_pdep_rpa_osv;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

const SYSTEMS: &[(&str, &str)] = &[
    ("water", "testdata/molecules/water.xyz"),
    ("methane", "testdata/molecules/methane.xyz"),
];
const T_OSV: &[f64] = &[1e-3, 1e-4, 1e-5, 1e-6];

fn main() {
    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();

    println!("# OSV (DNV) compression sweep — ferric PNO prefactor win");
    println!("# cc-pVDZ/cc-pVDZ-RI; e_dense via run_pdep_rpa; e_osv via run_pdep_rpa_osv");
    println!("{:>8} {:>6} {:>8} | {:>8} {:>12} {:>11} {:>9} {:>8}",
        "mol", "nvir", "t_osv", "n_vir_red", "compress%", "ΔE (Ha)", "t_dense", "speedup");

    for &(name, path) in SYSTEMS {
        let mol = match Molecule::load_xyz(path) {
            Ok(m) => m, Err(e) => { eprintln!("skip {name}: {e}"); continue; }
        };
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-9, ..Default::default() }).unwrap();

        let nocc = mol.nelec() as usize / 2;
        let nvir = obs.nbasis() - nocc;
        let cfg = PdepRpaConfig::default();

        // Dense reference.
        let t0 = Instant::now();
        let dense = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let t_dense = t0.elapsed().as_secs_f64();
        let e_dense = dense.e_rpa;

        for &t_osv in T_OSV {
            let t1 = Instant::now();
            let (e_osv, n_vir_red, _naux) =
                run_pdep_rpa_osv(&mol, &obs, &dfbs, op, &rhf, &cfg, t_osv).unwrap();
            let t_osv_s = t1.elapsed().as_secs_f64();
            let compress = 100.0 * n_vir_red as f64 / nvir as f64;
            let de = e_osv - e_dense;
            let speedup = t_dense / t_osv_s.max(1e-9);
            println!("{name:>8} {nvir:>6} {t_osv:>8.0e} | {n_vir_red:>8} {compress:>11.1} {de:>12.2e} {t_dense:>9.2} {speedup:>7.2}x");
        }
        println!();
    }
    println!("# compress% = n_vir_reduced/nvir (the prefactor ratio). ΔE = OSV − dense.");
    println!("# Read: production t_osv = smallest compress% whose |ΔE| is acceptable (e.g. <1e-4 Ha).");
    println!("# NOTE these molecules are tiny — speedup understates the win (overhead-dominated);");
    println!("#      the compress% ratio is the size-independent signal for the prefactor.");
}
