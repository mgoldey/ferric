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
// RHF now goes through the level-shift ladder (see main); bare solve_rhf unused.
use ferric_scf::screening::SchwarzBounds;

const HA_TO_EV: f64 = 27.211386245988_f64;

/// Solve the molecule at aug-cc-pVDZ-PP (where ferric's guess reaches the
/// physical basin) and project the converged density into the current
/// (aug-cc-pVTZ-PP) AO space, to seed a same-geometry larger-basis RHF that
/// otherwise mis-converges. Projection: with S_c = ⟨big|small⟩ the cross-basis
/// overlap and S_b the big-basis self-overlap, the density transforms as
///   D_big = (S_b⁻¹ S_c) D_small (S_b⁻¹ S_c)ᵀ.
fn seed_from_smaller_basis(
    ctx: &ParallelContext,
    mol: &Molecule,
    big_obs: &PreparedBasis,
    op: Operator,
) -> Result<ndarray::Array2<f64>, ferric_core::error::FerricError> {
    use ndarray_linalg::Inverse;
    let small_bs = basis::bundled("aug-cc-pvdz-pp")?;
    let small_obs = PreparedBasis::new(mol, &small_bs)?;
    let small_bounds = SchwarzBounds::compute(op, &small_obs)?;
    // Physical aDZ-PP reference (the ladder gets Ag2 right here — no g-functions).
    let cfg = ferric_scf::rhf::RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        max_iter: 200,
        ..Default::default()
    };
    let small_rhf = ferric_scf::rhf::solve_rhf(ctx, mol, &small_obs, op, &small_bounds, &cfg)?;
    // Closed-shell total density = 2 · (occ Cᵀ C). density_alpha holds ½D_full.
    let d_small = 2.0 * &small_rhf.density_alpha;

    // Cross overlap ⟨big|small⟩ and big self-overlap.
    let big_bs = big_obs.basis_set();
    let cross = ferric_integrals::cabs::cross_overlap(mol, big_bs, &small_bs)?; // s_or: (nbig, nsmall)
    let s_big = ferric_integrals::oneelectron::overlap(big_obs); // (nbig, nbig)
    let s_big_inv = s_big
        .inv()
        .map_err(|e| ferric_core::error::FerricError::General(format!("S_big inverse: {e:?}")))?;
    let t = s_big_inv.dot(&cross.s_or); // (nbig, nsmall)
    let d_big = t.dot(&d_small).dot(&t.t()); // (nbig, nbig)
    Ok(d_big)
}

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
    // Level-shift RHF ladder (DIIS → ls=0.5 → ls=1.0), same as gw100_full.rs.
    let lr = ferric_scf::ladder::solve_rhf_ladder(
        &ctx, &mol, &obs, op, &bounds, &ferric_scf::ladder::default_ladder(),
    )
    .expect("RHF ladder");
    if !lr.converged {
        eprintln!("  [!] RHF ladder did not converge (best rung {})", lr.rung_reached);
    }
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;

    // PHYSICALITY RESCUE. The ladder stops at the first CONVERGED rung — but for a
    // homonuclear transition-metal dimer (Ag2 at aug-cc-pVTZ-PP) rung 0 (no shift)
    // converges to an UNPHYSICAL closed-shell state: HOMO ε > 0 (unbound), giving a
    // negative Koopmans and a garbage G0W0 (−2.8 eV). Because it "converged", the
    // ladder never escalates. PySCF finds the physical state (RHF −292.12 Ha,
    // Koopmans +6.34) on the SAME basis, so the bound solution exists — the default
    // guess just flows to the wrong basin. Rescue: if the HOMO is unbound, force a
    // strongly level-shifted solve (ls=1.0) from the hcore guess, which damps the
    // near-degenerate d-manifold rotations that seed the saddle and lands the
    // physical basin. Well-behaved molecules skip this entirely (HOMO ε < 0).
    let rhf = if lr.result.eps_r()[homo_abs] > 0.0 {
        eprintln!(
            "  [!] unphysical RHF (HOMO ε = {:.4} Ha > 0) — rescuing by seeding from a \
             converged aug-cc-pVDZ-PP density (Ag2/aug-cc-pVTZ-PP flows to a wrong basin: \
             ferric's SAD free-atom guess for the odd-Z metal Ag is poor with the g/f \
             functions, cf. PySCF's minao guess which converges to the physical state)",
            lr.result.eps_r()[homo_abs]
        );
        // ferric converges Ag2 PHYSICALLY at aug-cc-pVDZ-PP (no g-functions to
        // trip the free-atom guess). Solve there, project the converged density
        // into the aug-cc-pVTZ-PP AO space via the cross-basis overlap, and use it
        // as the aTZ init guess — starting the SCF in the correct basin.
        let init = seed_from_smaller_basis(&ctx, &mol, &obs, op)
            .map_err(|e| eprintln!("  [!] aDZ-PP seed failed: {e:?}"))
            .ok();
        let cfg = ferric_scf::rhf::RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            level_shift: 0.5,
            max_iter: 300,
            init_guess_density: init,
            use_sad_guess: false, // explicit init density overrides
            ..Default::default()
        };
        ferric_scf::rhf::solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("RHF rescue")
    } else {
        lr.result
    };
    let ip_koop = -rhf.eps_r()[homo_abs] * HA_TO_EV;

    // Match gw100_full's production G0W0 knobs exactly.
    let pdep_cfg = PdepRpaConfig {
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
        // Davidson by default (preserves the validated aDZ cross-check path
        // bit-for-bit). Set XCHECK_EIGENSOLVER=lanczos to switch — needed for the
        // harder aug-cc-pVTZ-PP dielectric eigenproblems (Ag2, whose dense d/f
        // manifold makes Davidson stall at its subspace cap; Lanczos ≡ Davidson
        // in result but converges it — see rpa-solver-and-pdep-tradeoffs).
        eigensolver: match std::env::var("XCHECK_EIGENSOLVER").as_deref() {
            Ok("lanczos") => Eigensolver::Lanczos,
            _ => Eigensolver::Davidson,
        },
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
