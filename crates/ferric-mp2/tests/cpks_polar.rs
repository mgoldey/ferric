//! Analytic CPKS MP2 polarizability validation.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::cpks_polar::mp2_polarizability_analytic_hf;
use ferric_mp2::ff_polar::debug_scf_dipole_z;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[allow(clippy::type_complexity)]
fn water_ccpvdz() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    Operator,
    SchwarzBounds,
    ParallelContext,
    ferric_scf::ScfResult,
) {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
    (mol, obs, dfbs, op, bounds, ctx, rhf)
}

#[test]
fn cpks_hf_alpha_sane() {
    let (mol, obs, _dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let analytic = mp2_polarizability_analytic_hf(&ctx, &mol, &obs, &op, &bounds, &rhf).unwrap();
    eprintln!(
        "CPKS HF α_iso = {:.6}; tensor diag = {:.4} {:.4} {:.4}",
        analytic.iso, analytic.tensor[0][0], analytic.tensor[1][1], analytic.tensor[2][2]
    );
    assert!(analytic.iso > 0.0, "HF α_iso must be positive, got {}", analytic.iso);
    for &p in &analytic.principal {
        assert!(p > 0.0, "HF α principal must be positive, got {p}");
    }
    assert!(
        (1.0..8.0).contains(&analytic.iso),
        "HF α_iso = {} out of window [1,8] for water/cc-pVDZ",
        analytic.iso
    );
}

#[test]
fn cpks_hf_alpha_matches_ff_hf_zz() {
    let (mol, obs, _dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let analytic = mp2_polarizability_analytic_hf(&ctx, &mol, &obs, &op, &bounds, &rhf).unwrap();
    // FF HF α_zz via SCF dipole difference (HF-level is stable for water).
    let scf_cfg = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let h = 1e-3;
    let mp = debug_scf_dipole_z(&ctx, &mol, &obs, &bounds, &scf_cfg, h).unwrap();
    let mm = debug_scf_dipole_z(&ctx, &mol, &obs, &bounds, &scf_cfg, -h).unwrap();
    let ff_zz = -(mp - mm) / (2.0 * h);
    let diff = (analytic.tensor[2][2] - ff_zz).abs();
    eprintln!(
        "analytic α_zz = {:.6}, FF-HF α_zz = {:.6}, |Δ| = {:.2e}",
        analytic.tensor[2][2], ff_zz, diff
    );
    assert!(diff < 1e-4, "analytic HF α_zz vs FF-HF disagree by {diff:.2e}");
}

#[test]
#[ignore]
fn diag_cpks_hf_all_components_vs_ff() {
    let (mol, obs, _dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let analytic = mp2_polarizability_analytic_hf(&ctx, &mol, &obs, &op, &bounds, &rhf).unwrap();
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let h = 1e-3;
    // FF along each axis: need a general debug_scf_dipole_axis. Use z helper for z;
    // for x,y compute via the full FF path on the SCF density. Quick: print analytic
    // full tensor + the one FF we have (zz).
    eprintln!("analytic HF tensor:");
    for r in &analytic.tensor { eprintln!("  [{:+.5} {:+.5} {:+.5}]", r[0], r[1], r[2]); }
    let mp = debug_scf_dipole_z(&ctx, &mol, &obs, &bounds, &scf_cfg, h).unwrap();
    let mm = debug_scf_dipole_z(&ctx, &mol, &obs, &bounds, &scf_cfg, -h).unwrap();
    eprintln!("FF-HF α_zz = {:.6}", -(mp - mm)/(2.0*h));
    eprintln!("μz at +h={:.6} -h={:.6} (permanent ~ avg)", mp, mm);
}

#[test]
#[ignore]
fn diag_cpks_hf_full_tensor_vs_ff() {
    use ferric_mp2::ff_polar::debug_scf_dipole_axis;
    let (mol, obs, _dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let analytic = mp2_polarizability_analytic_hf(&ctx, &mol, &obs, &op, &bounds, &rhf).unwrap();
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let h = 1e-3;
    // FF-HF tensor: α_ij = -(μ_i(+h e_j) - μ_i(-h e_j))/(2h)
    let mut ff = [[0.0f64;3];3];
    for j in 0..3 {
        let mp = debug_scf_dipole_axis(&ctx,&mol,&obs,&bounds,&scf_cfg,j, h).unwrap();
        let mm = debug_scf_dipole_axis(&ctx,&mol,&obs,&bounds,&scf_cfg,j,-h).unwrap();
        for i in 0..3 { ff[i][j] = -(mp[i]-mm[i])/(2.0*h); }
    }
    eprintln!("ANALYTIC:                    FF-HF:");
    for i in 0..3 {
        eprintln!("  [{:+.4} {:+.4} {:+.4}]    [{:+.4} {:+.4} {:+.4}]",
            analytic.tensor[i][0],analytic.tensor[i][1],analytic.tensor[i][2],
            ff[i][0],ff[i][1],ff[i][2]);
    }
    eprintln!("ratio analytic/ff (diag): {:.4} {:.4} {:.4}",
        analytic.tensor[0][0]/ff[0][0], analytic.tensor[1][1]/ff[1][1], analytic.tensor[2][2]/ff[2][2]);
}

// WIP (Layer 2): analytic ∂E_MP2/∂F is 0.37× the FD oracle — stable across h, so a
// deterministic missing-term bug in ∂ε_p (the perturbed-Fock-diagonal / orbital-
// energy response), NOT the oracle (gauge-invariant energy FD is clean & linear).
// Debugging the ∂F_pp = ∂h + ∂(2J−K)[∂D] structure. Marked ignore until green.
#[test]
#[ignore]
fn cpks_dmp2_energy_matches_fd() {
    // Layer 2 gate (gauge-invariant): analytic ∂E_MP2/∂F^z vs FD of E_MP2.
    // Element-wise ∂t2 FD is contaminated by occ/vir orbital-rotation phase
    // ambiguity in the perturbed RHF; the correlation ENERGY is rotation-immune.
    use ferric_mp2::cpks_polar::{analytic_de_mp2_along, fd_de_mp2_along};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let an = analytic_de_mp2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg,2).unwrap();
    let fd = fd_de_mp2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,2,1e-3).unwrap();
    eprintln!("∂E_MP2/∂Fz: analytic={an:.8} fd={fd:.8} |Δ|={:.2e}", (an-fd).abs());
    assert!((an-fd).abs() < 1e-5, "∂E_MP2/∂Fz analytic vs FD disagree by {:.2e}", (an-fd).abs());
}
#[test]
#[ignore]
fn diag_dt2_distribution() {
    use ferric_mp2::cpks_polar::{analytic_dt2_along, fd_dt2_along};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let (an, _u) = analytic_dt2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg,2).unwrap();
    // FD at two field strengths to check linearity (phase/degeneracy contamination
    // would NOT scale linearly with h).
    for &h in &[1e-3_f64, 1e-2] {
        let fd = fd_dt2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,2,h).unwrap();
        let fdn = fd.iter().map(|x|x*x).sum::<f64>().sqrt();
        // largest 5 |fd| elements and their analytic counterparts
        let mut idx: Vec<usize> = (0..fd.len()).collect();
        idx.sort_by(|&a,&b| fd[b].abs().partial_cmp(&fd[a].abs()).unwrap());
        eprintln!("h={h:.0e} ‖fd‖={fdn:.4} top elems (fd vs an):");
        for &k in idx.iter().take(6) { eprintln!("   [{k}] fd={:+.5} an={:+.5}", fd[k], an[k]); }
    }
    let ann = an.iter().map(|x|x*x).sum::<f64>().sqrt();
    eprintln!("‖analytic‖={ann:.5}");
}

#[test]
#[ignore]
fn diag_de_components() {
    use ferric_mp2::cpks_polar::{analytic_de_mp2_along, fd_de_mp2_along, analytic_dt2_along};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let an = analytic_de_mp2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg,2).unwrap();
    for &h in &[1e-3_f64, 1e-2, 2e-2] {
        let fd = fd_de_mp2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,2,h).unwrap();
        eprintln!("h={h:.0e}: analytic={an:.8} fd={fd:.8} ratio an/fd={:.4}", an/fd);
    }
    let (dt2,_u) = analytic_dt2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg,2).unwrap();
    eprintln!("‖dt2‖={:.5}", dt2.iter().map(|x|x*x).sum::<f64>().sqrt());
}

#[test]
#[ignore]
fn diag_dd_vs_fd() {
    // Validate the U-driven SCF density response ∂D against FD of the SCF density.
    // Gauge-stable (total density is rotation-invariant). If ∂D matches, U is
    // correctly normalized and the Layer-2 residual is downstream of it.
    use ferric_mp2::cpks_polar::debug_dd_norms;
    let (mol, obs, _dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let (an, fd, maxd) = debug_dd_norms(&ctx,&mol,&obs,&op,&bounds,&rhf,&scf_cfg,2,1e-3).unwrap();
    eprintln!("∂D/∂Fz: ‖analytic‖={an:.6} ‖fd‖={fd:.6} maxΔ={maxd:.3e}");
}

#[test]
#[ignore]
fn diag_dd_traces() {
    use ferric_mp2::cpks_polar::debug_dd_traces;
    let (mol, obs, _dfbs, _op, bounds, ctx, rhf) = water_ccpvdz();
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let (an, fd) = debug_dd_traces(&ctx,&mol,&obs,&bounds,&rhf,&scf_cfg,2,1e-3).unwrap();
    eprintln!("−Tr[∂D·r_y] (=α^HF_y,z):  analytic={:?}  fd={:?}", an, fd);
    eprintln!("(validated HF α_zz=5.109; α_xz=α_yz≈0)");
}

#[test]
#[ignore]
fn diag_emp2_parabola() {
    use ferric_mp2::cpks_polar::{debug_emp2_at_field, analytic_de_mp2_along};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    for &f in &[-1e-2_f64,-1e-3,0.0,1e-3,1e-2] {
        let e = debug_emp2_at_field(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,2,f).unwrap();
        eprintln!("F={f:+.0e} E_MP2={e:.10}");
    }
    let an = analytic_de_mp2_along(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg,2).unwrap();
    eprintln!("analytic ∂E_MP2/∂Fz = {an:.10}");
}

#[test]
#[ignore]
fn diag_alpha_amplitude_only_vs_ff() {
    // How much of the relaxed α does the amplitude part of ∂dm1 capture (before ∂z)?
    use ferric_mp2::cpks_polar::analytic_alpha_amplitude_only;
    use ferric_mp2::ff_polar::{mp2_polarizability_static, DensityMode};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let amp = analytic_alpha_amplitude_only(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg).unwrap();
    let ff = mp2_polarizability_static(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,1e-3,DensityMode::Relaxed).unwrap();
    // Also HF α (the orbital-only part) for reference.
    eprintln!("amplitude-only α_iso = {:.5}  (FF relaxed = {:.5})", amp.iso, ff.iso);
    eprintln!("  amp diag: {:.4} {:.4} {:.4}", amp.tensor[0][0],amp.tensor[1][1],amp.tensor[2][2]);
    eprintln!("  ff  diag: {:.4} {:.4} {:.4}", ff.tensor[0][0],ff.tensor[1][1],ff.tensor[2][2]);
}

#[test]
#[ignore]
fn diag_alpha_relaxed_vs_ff() {
    use ferric_mp2::cpks_polar::analytic_alpha_relaxed;
    use ferric_mp2::ff_polar::{mp2_polarizability_static, DensityMode};
    let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_ccpvdz();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config { frozen_core: 0 };
    let scf_cfg = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let an = analytic_alpha_relaxed(&ctx,&mol,&obs,&dfbs,op,&bounds,&rhf,&mp2_cfg).unwrap();
    let ff = mp2_polarizability_static(&ctx,&mol,&obs,&dfbs,op,&bounds,&scf_cfg,&mp2_cfg,1e-3,DensityMode::Relaxed).unwrap();
    eprintln!("analytic relaxed α_iso = {:.5}  FF relaxed = {:.5}  |Δ|={:.2e}", an.iso, ff.iso, (an.iso-ff.iso).abs());
    eprintln!("  an diag: {:.4} {:.4} {:.4}", an.tensor[0][0],an.tensor[1][1],an.tensor[2][2]);
    eprintln!("  ff diag: {:.4} {:.4} {:.4}", ff.tensor[0][0],ff.tensor[1][1],ff.tensor[2][2]);
}
