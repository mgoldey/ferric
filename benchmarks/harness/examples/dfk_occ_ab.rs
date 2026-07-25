//! A/B the DF-K density path vs the occ (C_occ half-transform) path on the case
//! that limit-cycled (benzene/def2-svp, RI-JK). Env FERRIC_DFK_OCC=1 switches the
//! SCF loop onto the occ path (see rhf.rs). Prints per-iteration dp_rms/dE via
//! FERRIC_SCF_TRACE so the failure mode is visible, not inferred.
use ferric_core::basis; use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn main(){
  let which = std::env::args().nth(1).unwrap_or_else(|| "benzene".into());
  let (path, bs_name) = match which.as_str() {
    "water"   => ("testdata/molecules/water.xyz", "cc-pvdz"),
    "methane" => ("testdata/molecules/methane.xyz", "cc-pvdz"),
    "benzene-atz" => ("testdata/molecules/benzene.xyz", "aug-cc-pvtz"),
    _         => ("testdata/molecules/benzene.xyz", "def2-svp"),
  };
  let mol=Molecule::load_xyz(path).unwrap();
  let obs=PreparedBasis::new(&mol,&basis::bundled(bs_name).unwrap()).unwrap();
  let op=Operator::coulomb();
  let bounds=SchwarzBounds::compute(op,&obs).unwrap();
  let cfg=RhfConfig{
    df_j_aux:Some("def2-universal-jkfit".into()),
    df_k_aux:Some("def2-universal-jkfit".into()),
    max_iter:200,
    energy_conv:std::env::var("EC").ok().and_then(|v|v.parse().ok()).unwrap_or(1e-3),
    density_conv:std::env::var("DC").ok().and_then(|v|v.parse().ok()).unwrap_or(1e-6),
    ..Default::default()};
  let t=Instant::now();
  let r=solve_rhf(&ParallelContext::default(),&mol,&obs,op,&bounds,&cfg).unwrap();
  println!("{which}/{bs_name} occ={}  converged={} iters={} E={:.10} wall={:.2}s",
    std::env::var("FERRIC_DFK_OCC").unwrap_or_else(|_|"0".into()),
    r.converged, r.iterations, r.energy, t.elapsed().as_secs_f64());
}
