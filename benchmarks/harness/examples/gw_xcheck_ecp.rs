//! Same-geometry same-basis G0W0@HF cross-check harness — ECP variant.
//!
//! Identical to gw_xcheck.rs but calls `mol.apply_ecp(&obs_bs)` so the reduced
//! valence electron count and the V_ECP one-electron potential flow into the
//! RHF reference AND the GW intermediates. This is the validation gate for the
//! GW-through-ECP path (spec 2026-06-17-gw100-ecp-molecules.md): RHF@ECP ≡ PySCF
//! is already proven (crates/ferric-scf/tests/ecp_rhf.rs); this proves G0W0@ECP.
//!
//! Prints one parseable line:
//!   XCHECK <ip_g0w0_ev> <ip_koopmans_ev> <sigma_c_ha> <z_factor> <nelec> <e_rhf>
//!
//! Paired with scripts/gw100/pyscf_g0w0_ecp.py (PySCF gw_ac on the SAME xyz,
//! fed the SAME bundled aug-cc-pVDZ-PP JSON + ECP + def2-tzvp-rifit aux).
//!
//! Run:
//!   cargo run --release --example gw_xcheck_ecp -p ferric-gw -- \
//!     scripts/gw100/geom_ecp/7553-56-2.xyz aug-cc-pvdz-pp def2-tzvp-rifit

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA_TO_EV: f64 = 27.211386245988_f64;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gw_xcheck_ecp <file.xyz> <obs> <ri-aux>");
    let obs_name = args.next().unwrap_or_else(|| "aug-cc-pvdz-pp".to_string());
    let aux_name = args.next().unwrap_or_else(|| "def2-tzvp-rifit".to_string());

    let xyz = std::fs::read_to_string(&path).expect("read xyz");
    let mut mol = Molecule::parse_xyz(&xyz, 0, 1).expect("parse xyz (neutral singlet)");
    let obs_bs = basis::bundled(&obs_name).expect("obs basis");
    let aux_bs = basis::bundled(&aux_name).expect("ri aux basis");
    // The single point where the ECP enters: reduces nelec() and sets effective_z.
    mol.apply_ecp(&obs_bs);

    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("RHF");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;
    let ip_koop = -rhf.eps_r()[homo_abs] * HA_TO_EV;

    // Match gw100_full's production G0W0 knobs exactly.
    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        davidson_conv_thresh: 1e-7,
        davidson_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        // run_gw forces this on internally (gw::with_inv_dielectric); standalone
        // RPA uses here are energy-only, so stay lean per M9.
        need_inv_dielectric_freq: false,
    };
    let gcfg = GwConfig {
        method: GwMethod::G0W0,
        max_ev_iter: 8,
        ev_conv_thresh: 1e-4,
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg, &gcfg, None).expect("gw run");
    let homo_local = res
        .mo_indices
        .iter()
        .position(|&i| i == homo_abs)
        .expect("HOMO in qp range");
    let ip = -res.eps_qp[homo_local] * HA_TO_EV;
    println!(
        "XCHECK {:.4} {:.4} {:.5} {:.4} {} {:.6}",
        ip, ip_koop, res.sigma_c[homo_local], res.z_factor[homo_local],
        mol.nelec(), rhf.energy
    );
}
