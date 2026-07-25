//! MWE: time ONE DF-K Fock build, nothing else. This is the operation that
//! dominates ferric's SCF loop (~1565 ms/iter at benzene/aTZ vs PySCF ~350 ms).
//! Reports occ-path vs density-path and the implied GFLOP/s so we can tell
//! whether we are GEMM-bound or losing time to repacking/allocation.
use ferric_core::basis; use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::fock::KBuilder;
use ndarray::Array2;
use std::time::Instant;

fn main(){
  let which=std::env::args().nth(1).unwrap_or_else(||"benzene-atz".into());
  let (path,bs,aux_name)=match which.as_str(){
    "water"=>("testdata/molecules/water.xyz","cc-pvdz","def2-universal-jkfit"),
    "benzene-svp"=>("testdata/molecules/benzene.xyz","def2-svp","def2-universal-jkfit"),
    _=>("testdata/molecules/benzene.xyz","aug-cc-pvtz","def2-universal-jkfit"),
  };
  let mol=Molecule::load_xyz(path).unwrap();
  let obs=PreparedBasis::new(&mol,&basis::bundled(bs).unwrap()).unwrap();
  let auxp=PreparedBasis::new(&mol,&basis::bundled(aux_name).unwrap()).unwrap();
  let op=Operator::coulomb();
  let n=obs.nbasis(); let naux=auxp.nbasis(); let nocc=(mol.nelec()/2) as usize;
  println!("{which}: n={n} naux={naux} nocc={nocc}");

  let t=Instant::now();
  let mut dfk=ferric_scf::df_k::DfK::new(op,&obs,&auxp,ferric_scf::rhf::resolve_three_index_budget(0)).unwrap();
  println!("  DfK::new (fit+dress, one-time): {:.0} ms", t.elapsed().as_secs_f64()*1e3);

  // Arbitrary but realistic C_occ: orthonormal columns via QR of a fixed matrix.
  let mut c=Array2::<f64>::zeros((n,nocc));
  for i in 0..n { for j in 0..nocc { c[[i,j]]=((i*7+j*13)%97) as f64/97.0; } }
  let d=2.0*c.dot(&c.t());

  let reps:usize=std::env::var("REPS").ok().and_then(|v|v.parse().ok()).unwrap_or(3);
  let mut k=Array2::<f64>::zeros((n,n));

  let t=Instant::now();
  for _ in 0..reps { dfk.build(&d,&mut k).unwrap(); }
  let t_den=t.elapsed().as_secs_f64()/reps as f64;
  // density path: two GEMMs, naux*n*n each pass -> 2*2*naux*n^2*n? see notes
  let fl_den=2.0*(naux as f64)*(n as f64)*(n as f64)*(n as f64)*2.0;
  println!("  build(D)        {:8.1} ms   ~{:6.1} GFLOP/s", t_den*1e3, fl_den/t_den/1e9);

  let t=Instant::now();
  for _ in 0..reps { dfk.build_from_occ(&c,&mut k).unwrap(); }
  let t_occ=t.elapsed().as_secs_f64()/reps as f64;
  let fl_occ=2.0*(naux as f64)*(n as f64)*(n as f64)*(nocc as f64)
            +2.0*(naux as f64)*(nocc as f64)*(n as f64)*(n as f64);
  println!("  build_from_occ  {:8.1} ms   ~{:6.1} GFLOP/s   speedup {:.2}x",
           t_occ*1e3, fl_occ/t_occ/1e9, t_den/t_occ);
}
