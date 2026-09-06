//! Exactness anchors + per-map mutation arms for the INTEGRAL-DIRECT
//! amplitude-threshold dRPA (`ferric_mp2::drpa_amplitude::amplitude_drpa_direct`).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): trivial-limit anchors
//! FIRST — (a) trivial maps + ε=0 must land on the canonical plasmon
//! reference (the independent construction the 2026-08-16 port anchored
//! at 1.1e-14-class on water) and the proof notebook's H2 value; (b) the
//! finite-ε trivial-map path must match the existing global-B path
//! closely (the two assemble the SAME fitted J through different
//! integral/transform code, so the gap is the reassociation floor plus
//! any borderline Eq-8 pattern flips — dRPA is FIRST order in integral
//! perturbations, so the bar is looser than LMP2's Hylleraas-protected
//! one); (c) production maps' extra error must stay sub-dominant to the
//! ε truncation. Each locality map then gets a gutting arm that must move
//! the energy loudly. No sweep runs until these pass.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{
    amplitude_drpa, amplitude_drpa_direct, AmplitudeDrpaConfig,
};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str, obs_name: &str, aux_name: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    Setup { mol, obs, obs_bs, dfbs, rhf }
}

/// Trivial maps: every locality knob at its no-op limit (aux_radius 1e6,
/// virt_radius None, ao_tail 0, schwarz_skip 0, batch_merge 1).
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

/// TRIVIAL-LIMIT ANCHOR (a1): H2/STO-3G single pair — the direct path
/// must land on the proof notebook's plasmon value -0.0126072623 AND on
/// its own canonical plasmon reference, exactly like the global-B port's
/// `h2_single_pair_matches_the_proof_notebook`.
#[test]
fn h2_direct_matches_the_proof_notebook() {
    let su = setup("h2.xyz", "sto-3g", "sto-3g");
    let cfg = AmplitudeDrpaConfig { eps: 0.0, ..Default::default() };
    let (r, stats) = amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let de = r.e_corr - r.e_corr_plasmon_canonical;
    eprintln!(
        "H2 direct: E_corr={:.10} plasmon={:.10} dE={de:+.3e} notebook=-0.0126072623 ({} iters, strips {}x{})",
        r.e_corr, r.e_corr_plasmon_canonical, r.iterations, stats.strip_rows_max, stats.strip_cols_max
    );
    assert!(r.converged);
    assert!(de.abs() < 1e-9, "H2 direct anchor FAILED: dE={de:+.3e}");
    assert!(
        (r.e_corr - (-0.012_607_262_3)).abs() < 1e-8,
        "H2 direct disagrees with the proof notebook: {:.10}",
        r.e_corr
    );
}

/// TRIVIAL-LIMIT ANCHOR (a2): water/6-31G ε=0 — direct-path Riccati vs
/// (i) its own canonical semicanonicalized plasmon reference (independent
/// construction) and (ii) the existing global-B path (same formulation,
/// different integral/transform code: batched masked strips + domain fit
/// at huge radius vs `eri3_mo_ov_blocked` + whitened Gram). dRPA is first
/// order in integral perturbations, so (ii)'s bar is the reassociation
/// floor of the two B constructions, not machine eps — measured and
/// printed here, asserted at 1e-9.
#[test]
fn eps_zero_trivial_maps_matches_plasmon_and_global_path() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r_glob = amplitude_drpa(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
    )
    .unwrap();
    let (r_dir, stats) = amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let de_plasmon = r_dir.e_corr - r_dir.e_corr_plasmon_canonical;
    let de_global = r_dir.e_corr - r_glob.e_corr;
    eprintln!(
        "ANCHOR water direct eps=0: E={:.12} |vs plasmon|={:.3e} |vs global-B path|={:.3e} \
         ({} iters, strips rows {}/{} cols {}/{})",
        r_dir.e_corr,
        de_plasmon.abs(),
        de_global.abs(),
        r_dir.iterations,
        stats.strip_rows_mean,
        stats.strip_rows_max,
        stats.strip_cols_mean,
        stats.strip_cols_max,
    );
    assert!(r_dir.converged);
    assert!(r_dir.keep_fraction == 1.0 && r_dir.pair_fraction == 1.0);
    assert!(de_plasmon.abs() < 1e-9, "eps=0 plasmon anchor FAILED: {de_plasmon:+.3e}");
    assert!(de_global.abs() < 1e-9, "eps=0 global-path anchor FAILED: {de_global:+.3e}");
    // trivial maps must actually be trivial: strips span everything
    assert_eq!(stats.strip_rows_max, su.dfbs.nbasis());
}

/// FINITE-ε ANCHOR (b): at ε = 1e-3 with trivial locality maps, direct and
/// global-B paths mask the same fitted B (up to transform round-off) and
/// run the identical Riccati solver, so the energies must agree to well
/// below the ε-truncation scale (~1e-8 class; the residual is the B
/// reassociation floor amplified first-order through the non-variational
/// energy, plus any borderline pattern flips — measured and printed).
#[test]
fn finite_eps_trivial_maps_matches_global_path() {
    let su = setup("alkane_4.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = AmplitudeDrpaConfig {
        eps: 1e-3,
        frozen_core: 4,
        diis: Some(6),
        compute_reference: false,
        ..Default::default()
    };
    let r_glob = amplitude_drpa(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
    )
    .unwrap();
    let (r_dir, _) = amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let dd = (r_dir.e_corr - r_glob.e_corr).abs();
    eprintln!(
        "DIRECT finite-eps C4: E={:.10} global={:.10} |diff|={dd:.3e} keep {:.4} vs {:.4}",
        r_dir.e_corr, r_glob.e_corr, r_dir.keep_fraction, r_glob.keep_fraction
    );
    assert!(r_dir.converged);
    assert!(dd < 1e-8, "finite-eps direct vs global FAILED: {dd:.3e}");
}

/// PRODUCTION-MAP SUB-DOMINANCE (c): with the LMP2-calibrated production
/// maps the extra error they introduce must stay sub-dominant to the ε
/// truncation itself. C8/Coulomb (dRPA's lane); truncation error is
/// measured against the canonical plasmon reference (the ε=0-class truth
/// the tight-rtol test on the global path uses), map error against the
/// trivial-map run at the same ε.
#[test]
fn production_maps_error_is_subdominant_to_eps_truncation() {
    let su = setup("alkane_8.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = AmplitudeDrpaConfig {
        eps: 1e-3,
        frozen_core: 8,
        pair_gate_cal: Some(0.7),
        diis: Some(6),
        ..Default::default()
    };
    // ε truncation alone (trivial maps) + the canonical plasmon reference
    let (r_eps, _) = amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    // ε + production locality maps (the LMP2-calibrated radii)
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    let (r_prod, stats) = amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf,
        &AmplitudeDrpaConfig { compute_reference: false, ..cfg },
        &prod,
    )
    .unwrap();
    let trunc_err = (r_eps.e_corr - r_eps.e_corr_plasmon_canonical).abs();
    let map_err = (r_prod.e_corr - r_eps.e_corr).abs();
    eprintln!(
        "C8/coul dRPA production maps: eps-trunc {trunc_err:.3e}, map-err {map_err:.3e} \
         (strips rows {:.0}/{} cols {:.0}/{}, keep {:.4})",
        stats.strip_rows_mean,
        stats.strip_rows_max,
        stats.strip_cols_mean,
        stats.strip_cols_max,
        r_prod.keep_fraction
    );
    assert!(map_err > 0.0, "maps changed nothing at production radii — vacuous?");
    assert!(
        map_err < trunc_err,
        "locality-map error ({map_err:.3e}) dominates the eps truncation ({trunc_err:.3e})"
    );
}

/// MUTATION ARMS — each locality map, gutted, must change the dRPA energy
/// loudly; otherwise the map is dead code on this path. ε=0 so the arms
/// judge the maps alone (mask inert), against the trivial-map base.
#[test]
fn each_map_gutted_is_loud() {
    let su = setup("alkane_4.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = AmplitudeDrpaConfig {
        eps: 0.0,
        frozen_core: 4,
        diis: Some(6),
        compute_reference: false,
        ..Default::default()
    };
    let run = |dcfg: &DirectConfig| {
        amplitude_drpa_direct(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, dcfg,
        )
    };
    let (base, _) = run(&trivial_maps()).unwrap();

    // (a) virtual domains gutted: 2 Bohr around each occupied centroid
    let (mut_v, _) = run(&DirectConfig { virt_radius_bohr: Some(2.0), ..trivial_maps() }).unwrap();
    let dv = (mut_v.e_corr - base.e_corr).abs();
    eprintln!("MUTATION virt_radius=2: |dE|={dv:.3e}");
    assert!(dv > 1e-3, "virt-domain map gutted silently: |dE|={dv:.3e}");

    // (b) AO support gutted: only shells with |C| >= 0.3 survive
    let (mut_a, _) = run(&DirectConfig { ao_tail: 0.3, ..trivial_maps() }).unwrap();
    let da = (mut_a.e_corr - base.e_corr).abs();
    eprintln!("MUTATION ao_tail=0.3: |dE|={da:.3e}");
    assert!(da > 1e-3, "AO-support map gutted silently: |dE|={da:.3e}");

    // (c) Schwarz triple cut gutted: an absurd threshold zeroes most of
    // the integral stream. First run measured the mutilated B driving the
    // Riccati fixed point to a NaN residual and a hard non-convergence
    // error — a LOUD failure (never a silent zero), accepted here; cap the
    // iterations so the diverging arm doesn't spin the full 500.
    let cfg_s = AmplitudeDrpaConfig { fp_max_iter: 60, ..cfg.clone() };
    match amplitude_drpa_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg_s,
        &DirectConfig { schwarz_skip: 10.0, ..trivial_maps() },
    ) {
        Err(e) => eprintln!("MUTATION schwarz_skip=10: hard error (loud, acceptable): {e}"),
        Ok((mut_s, st_s)) => {
            let ds = (mut_s.e_corr - base.e_corr).abs();
            eprintln!(
                "MUTATION schwarz_skip=10: |dE|={ds:.3e} ({} triples skipped)",
                st_s.n_eri3_skipped
            );
            assert!(
                ds > 1e-3 && st_s.n_eri3_skipped > 0,
                "Schwarz cut gutted silently: |dE|={ds:.3e}"
            );
        }
    }

    // (d) aux domains gutted: 4 Bohr fit domains (hard error acceptable —
    // an empty/singular local fit must be loud, never a silent zero)
    match run(&DirectConfig { aux_radius_bohr: 4.0, ..trivial_maps() }) {
        Err(e) => eprintln!("MUTATION aux_radius=4: hard error (acceptable): {e}"),
        Ok((mut_x, _)) => {
            let dx = (mut_x.e_corr - base.e_corr).abs();
            eprintln!("MUTATION aux_radius=4: |dE|={dx:.3e}");
            assert!(dx > 1e-5, "aux-domain map gutted silently: |dE|={dx:.3e}");
        }
    }
}

/// Wall-clock comparison, existing global-B dRPA path vs the direct path
/// at C8/C12, same solver settings (DIIS 6, gate 0.7, no reference).
/// #[ignore]d — release, capped box:
///   OPENBLAS_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2 \
///     scripts/ferric-limited --max=4G --high=3600M -- \
///     cargo test -p ferric-mp2 --release --test drpa_direct -- \
///     --ignored --nocapture bench_drpa_direct_vs_global
#[test]
#[ignore]
fn bench_drpa_direct_vs_global() {
    use std::time::Instant;
    // ..Default::default() so the NEXT DirectConfig field doesn't break this
    // bench again (the #27×#28 cross-branch E0063 — each PR was green alone)
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        scratch_budget_bytes: 1usize << 30,
        ..Default::default()
    };
    println!(
        "sys        eps    E(global)      E(direct)      |dE|      keep_d  iters g/d  t_global t_direct"
    );
    for nc in [8usize, 12] {
        let su = setup(&format!("alkane_{nc}.xyz"), "6-31g", "cc-pvdz-ri");
        for eps in [1e-3, 1e-4] {
            let cfg = AmplitudeDrpaConfig {
                eps,
                frozen_core: nc,
                pair_gate_cal: Some(0.7),
                diis: Some(6),
                compute_reference: false,
                eri3_budget_bytes: Some(2usize << 30),
                ..Default::default()
            };
            let t0 = Instant::now();
            let r_glob = amplitude_drpa(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
            )
            .unwrap();
            let t_glob = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let (r_dir, _) = amplitude_drpa_direct(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
                &prod,
            )
            .unwrap();
            let t_dir = t0.elapsed().as_secs_f64();
            println!(
                "alkane_{:<3} {:6.0e} {:.10} {:.10} {:.3e} {:.4} {:4}/{:<4} {:8.2} {:8.2}",
                nc,
                eps,
                r_glob.e_corr,
                r_dir.e_corr,
                (r_dir.e_corr - r_glob.e_corr).abs(),
                r_dir.keep_fraction,
                r_glob.iterations,
                r_dir.iterations,
                t_glob,
                t_dir,
            );
        }
    }
}
