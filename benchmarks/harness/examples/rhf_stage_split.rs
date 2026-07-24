//! Where does ferric's RI-JK RHF time go? PySCF does the same job in 5.29s vs
//! ferric's 21.85s (4.1x). Split setup (DF tensor build) from the SCF loop.
use std::time::Instant;
use ferric_core::basis; use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main(){
  let mol=Molecule::load_xyz("testdata/molecules/benzene.xyz").unwrap();
  let obs=PreparedBasis::new(&mol,&basis::bundled("aug-cc-pvtz").unwrap()).unwrap();
  let jk=PreparedBasis::new(&mol,&basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
  let op=Operator::coulomb();
  println!("nbasis={} n_jkaux={}", obs.nbasis(), jk.nbasis());

  let t=Instant::now();
  let bounds=SchwarzBounds::compute(op,&obs).unwrap();
  println!("SchwarzBounds        : {:8.0} ms", t.elapsed().as_secs_f64()*1e3);

  // The DF-JK 3-index tensor over the JK aux basis: built once, reused each iter.
  let t=Instant::now();
  let n=threeindex::eri3_tensor(op,&obs,&jk).map(|a|a.len()).unwrap_or(0);
  println!("eri3(JK aux) once    : {:8.0} ms  ({} elems)", t.elapsed().as_secs_f64()*1e3, n);

  let cfg=RhfConfig{
    df_j_aux:Some("def2-universal-jkfit".into()),
    df_k_aux:Some("def2-universal-jkfit".into()),
    ..Default::default()};
  let t=Instant::now();
  let r=solve_rhf(&ParallelContext::default(),&mol,&obs,op,&bounds,&cfg).unwrap();
  let tot=t.elapsed().as_secs_f64();
  println!("solve_rhf total      : {:8.0} ms  ({} iters, {:.1} ms/iter)",
           tot*1e3, r.iterations, tot*1e3/r.iterations as f64);
  println!("  converged={} E={:.8}", r.converged, r.energy);
}
