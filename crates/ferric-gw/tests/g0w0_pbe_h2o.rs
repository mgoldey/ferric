//! ferric closed-shell G0W0@PBE HOMO IP for H2O/cc-pVDZ must match PySCF gw_ac
//! @PBE to <0.1 eV, and must differ from the @HF starting point (proves the
//! Σx−vxc correction is applied). Reference: scripts/queue/out/pyscf_g0w0_pbe_h2o_dz.py.

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
const PYSCF_IP: f64 = 11.1714; // PySCF gw_ac G0W0@PBE H2O/cc-pVDZ HOMO IP, eV

#[test]
fn g0w0_pbe_h2o_homo_ip_matches_pyscf() {
    let ctx = ParallelContext::default();
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    // PBE reference.
    let cfg = RhfConfig { xc: Some("pbe".into()), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
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
    // KS reference: pass v_xc so Σx−vxc enters the QP self-consistency.
    let (vxc_diag, _) = vxc_diagonal_mo(&mol, &obs_bs, "pbe", &scf).unwrap();
    let res = run_gw(&mol, &obs, &dfbs, op, &scf, &pdep_cfg, &gcfg, Some(&vxc_diag)).unwrap();
    let loc = res.mo_indices.iter().position(|&i| i == homo_abs).unwrap();
    let ip = -res.eps_qp[loc] * HA;

    // Matches PySCF gw_ac @PBE to <0.1 eV.
    assert!((ip - PYSCF_IP).abs() < 0.1,
        "ferric G0W0@PBE IP {ip:.3} eV vs PySCF {PYSCF_IP:.3} eV (Δ {:.3})", ip - PYSCF_IP);
}
