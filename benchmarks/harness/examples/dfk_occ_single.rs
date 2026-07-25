//! Single-Fock-build comparison of DF-K density path vs occ path on the SAME
//! C_occ, isolating the K discrepancy from any SCF trajectory feedback.
use ferric_core::basis; use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::fock::KBuilder;
use ndarray::Array2;

fn main(){
  let which = std::env::args().nth(1).unwrap_or_else(|| "benzene".into());
  let (path, bs_name) = match which.as_str() {
    "water"   => ("testdata/molecules/water.xyz","cc-pvdz"),
    "methane" => ("testdata/molecules/methane.xyz","cc-pvdz"),
    _         => ("testdata/molecules/benzene.xyz","def2-svp"),
  };
  let mol=Molecule::load_xyz(path).unwrap();
  let obs=PreparedBasis::new(&mol,&basis::bundled(bs_name).unwrap()).unwrap();
  let op=Operator::coulomb();
  let bounds=SchwarzBounds::compute(op,&obs).unwrap();
  let ctx=ParallelContext::default();
  // Converge first so we compare at a physically meaningful C_occ.
  let cfg=RhfConfig{ df_j_aux:Some("def2-universal-jkfit".into()),
                     df_k_aux:Some("def2-universal-jkfit".into()), ..Default::default()};
  let r=solve_rhf(&ctx,&mol,&obs,op,&bounds,&cfg).unwrap();
  let nocc=(mol.nelec()/2) as usize;
  let c_occ=r.mos_r().slice(ndarray::s![..,..nocc]).to_owned();
  let d = 2.0*c_occ.dot(&c_occ.t());
  let n=obs.nbasis();

  let aux=basis::bundled("def2-universal-jkfit").unwrap();
  let auxp=PreparedBasis::new(&mol,&aux).unwrap();
  let mut dfk=ferric_scf::df_k::DfK::new(op,&obs,&auxp,0).unwrap();
  let mut k_d=Array2::zeros((n,n)); dfk.build(&d,&mut k_d).unwrap();
  let mut k_o=Array2::zeros((n,n)); dfk.build_from_occ(&c_occ,&mut k_o).unwrap();
  k_o *= 2.0;
  let diff=(&k_d-&k_o).mapv(f64::abs);
  let maxd=diff.iter().cloned().fold(0.0f64,f64::max);
  let maxk=k_d.iter().cloned().fold(0.0f64,|a,b|a.max(b.abs()));
  println!("{which}/{bs_name} n={n} nocc={nocc}  max|K_d-K_occ|={maxd:.3e}  max|K_d|={maxk:.3e}  rel={:.3e}", maxd/maxk);
}
