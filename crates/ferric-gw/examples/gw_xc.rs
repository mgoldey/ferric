//! GW IP for one molecule with a selectable reference functional.
//!   cargo run --release -p ferric-gw --example gw_xc -- <xyz> [--xc pbe]
//! No --xc → HF reference (Σx−vxc is identically zero, skipped).
//! --xc <name> → RKS reference + Σx−vxc correction. G0W0 only (evGW@KS deferred).
//!
//! NOTE: the closed-shell Σx−vxc infrastructure is exact (ε_KS/v_xc/Σx match
//! PySCF), but Σc on a KS reference is currently off ~0.76 eV (H2O) — see
//! crates/ferric-gw/tests/g0w0_pbe_h2o.rs (ignored) and the gw-ks-sigma-c-offset
//! memory. @PBE numbers from this driver are NOT yet trustworthy; @HF is.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA: f64 = 27.211_386_245_988;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: gw_xc <xyz> [--xc pbe]");
    let xc = args.iter().position(|a| a == "--xc").map(|i| args[i + 1].clone());

    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(path).expect("xyz");
    let obs_bs = basis::bundled("aug-cc-pvtz").unwrap();
    let dfbs_bs = basis::bundled("aug-cc-pvtz-rifit").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let cfg = RhfConfig { xc: xc.clone(), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("scf");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;

    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        davidson_conv_thresh: 1e-7,
        trunc_thresh: 0.0,
        ..Default::default()
    };
    let gcfg = GwConfig { method: GwMethod::G0W0, qp_mos: Some(homo_abs..homo_abs + 1),
                          ..Default::default() };
    // KS reference: build v_xc and thread it into run_gw so Σx−vxc enters the QP
    // self-consistency. HF reference (xc=None): no shift.
    let vxc = xc.as_ref().map(|name| {
        let (v, _) = vxc_diagonal_mo(&mol, &obs_bs, name, &scf).expect("vxc");
        v
    });
    let res = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, vxc.as_ref()).expect("gw");
    let loc = res.mo_indices.iter().position(|&i| i == homo_abs).unwrap();
    let ref_label = xc.as_deref().unwrap_or("HF");
    println!("G0W0@{ref_label}  HOMO IP = {:.4} eV", -res.eps_qp[loc] * HA);
}
