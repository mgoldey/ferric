//! Split the rpa_intermediates stage (1.19s on benzene/aTZ) into its parts so we
//! optimize the real cost centre, not a guess. The MO transform already uses the
//! fixed occ-first stream_dressed_mo_band, so the remainder is ERI3 generation,
//! the 2-centre metric, and its inverse square root.
use std::time::Instant;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let mol = Molecule::load_xyz("testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvtz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvtz-rifit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    eprintln!("RHF (not timed)...");
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    eprintln!("  converged={} iters={}", rhf.converged, rhf.iterations);
    println!("nbasis={} naux={}", obs.nbasis(), dfbs.nbasis());

    // Component: 2-centre metric
    let t = Instant::now();
    let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
    println!("coulomb_metric_2c        : {:8.1} ms  ({}x{})", t.elapsed().as_secs_f64()*1e3, v2c.nrows(), v2c.ncols());

    // Component: full stage
    let mp2_cfg = RiMp2Config { frozen_core: 6, memory_budget_bytes: None, ..Default::default() };
    let t = Instant::now();
    let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    let whole = t.elapsed().as_secs_f64()*1e3;
    println!("compute_rpa_intermediates: {whole:8.1} ms  (b_ov {}x{})", inter.b_ov.nrows(), inter.b_ov.ncols());

    // Component: raw 3-index AO integrals alone (no transform, no dressing)
    let t = Instant::now();
    let n3 = threeindex::eri3_tensor(op, &obs, &dfbs).map(|a| a.len()).unwrap_or(0);
    println!("eri3_tensor (AO, dense)  : {:8.1} ms  ({} elems)", t.elapsed().as_secs_f64()*1e3, n3);
}
