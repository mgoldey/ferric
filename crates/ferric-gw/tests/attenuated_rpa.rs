//! Verdict test for "attenuated RPA": run `run_pdep_rpa` with
//! Operator::erfc(ω) and check (a) it runs to completion, (b) the
//! correlation energy is between 0 and the full-Coulomb RPA energy
//! (since erfc kills the long-range piece).
//!
//! See docs/superpowers/specs/2026-05-19-rpa-to-gw-spike-design.md §7.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // energy-only attenuated-RPA (M9 gate)
    }
}

#[test]
#[ignore = "slow: builds RHF + 2× PDEP-RPA; run with --release --ignored"]
fn attenuated_rpa_water_runs() {
    let xyz = "3
H2O
O  0.0   0.0       0.117790
H  0.0   0.755453 -0.471161
H  0.0  -0.755453 -0.471161
";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("xyz");
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs prep");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).expect("Schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(
        &ctx, &mol, &obs, Operator::coulomb(), &bounds,
        &RhfConfig::default(),
    )
    .expect("RHF");

    let pcfg = pdep_cfg();
    let res_full = run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg)
        .expect("full RPA");
    // omega = 0.222 Bohr⁻¹ = att-MP2 default
    let res_sr = run_pdep_rpa(&mol, &obs, &dfbs, Operator::erfc(0.222), &rhf, &pcfg)
        .expect("SR RPA");

    eprintln!("Full Coulomb RPA E_c     = {:.6} Ha", res_full.e_rpa);
    eprintln!("erfc(ω=0.222 Bohr⁻¹) RPA = {:.6} Ha", res_sr.e_rpa);
    eprintln!(
        "  ratio SR/full = {:.3} (expect 0 < r < 1; SR is a fraction of full)",
        res_sr.e_rpa / res_full.e_rpa
    );

    // Both correlation energies are negative.
    assert!(res_full.e_rpa < 0.0, "full RPA E_c should be negative");
    assert!(res_sr.e_rpa < 0.0, "SR RPA E_c should be negative");
    // SR captures only short-range correlation; magnitude is smaller than full.
    let ratio = res_sr.e_rpa / res_full.e_rpa;
    assert!(
        (0.0..1.5).contains(&ratio),
        "SR/full ratio out of band: {ratio:.3}"
    );
}
