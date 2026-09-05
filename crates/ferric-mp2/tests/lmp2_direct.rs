//! Exactness anchors + per-map mutation arms for the INTEGRAL-DIRECT
//! amplitude-LMP2 assembly (`ferric_mp2::lmp2_direct`).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): the trivial-limit
//! anchors come FIRST and every locality map gets a mutation arm that must
//! move the energy loudly — a map that can be gutted without consequence
//! is not a map, it's dead code. No sweep runs until these pass.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
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

/// Trivial maps: every locality knob at its no-op limit.
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

/// TRIVIAL-LIMIT ANCHOR: with every map inert and ε = 0, the integral-direct
/// path must reproduce (a) the existing global-B domain-fit path at the same
/// (huge) radius, and (b) the canonical closed-form ri_mp2. (a) shares the
/// fit formulation but NOT the integral/transform code path (batched masked
/// strips vs `eri3_mo_ov_blocked`), so agreement is a real check on the
/// direct evaluation; (b) is the fully independent construction.
#[test]
fn trivial_limit_matches_global_domain_fit_and_canonical() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let glob_cfg = AmplitudeLmp2Config { fit_radius_bohr: Some(1e6), ..cfg.clone() };
    let glob = amplitude_lmp2(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &glob_cfg,
    )
    .unwrap();
    let (dir, stats) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let dd = (dir.e_corr - glob.e_corr).abs();
    let de = (dir.e_corr - dir.e_corr_canonical_ri).abs();
    eprintln!(
        "DIRECT trivial limit water: E={:.10} |vs global-fit|={dd:.3e} |vs canonical|={de:.3e} \
         (strips {}x{} rows/cols max, {} eri3 triples)",
        dir.e_corr, stats.strip_rows_max, stats.strip_cols_max, stats.n_eri3_shell_triples
    );
    assert!(dir.cg_converged);
    assert!(dd < 1e-10, "direct vs global domain-fit FAILED: {dd:.3e}");
    assert!(de < 1e-9, "direct vs canonical ri_mp2 FAILED: {de:.3e}");
    // trivial maps must actually be trivial: strips span everything
    assert_eq!(stats.strip_rows_max, su.dfbs.nbasis());
}

/// Same anchor through the ATTENUATED kernel (erfc, same-kernel metric) —
/// op-threading on the direct path (3c engine, 2c metric blocks) must agree
/// with the canonical attenuated ri_mp2.
#[test]
fn trivial_limit_holds_for_erfc() {
    let su = setup("water.xyz");
    let op = Operator::erfc(1.0);
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 0, ..Default::default() };
    let (dir, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &trivial_maps(),
    )
    .unwrap();
    let de = (dir.e_corr - dir.e_corr_canonical_ri).abs();
    eprintln!("DIRECT erfc trivial limit water: E={:.10} |vs canonical|={de:.3e}", dir.e_corr);
    assert!(dir.e_corr < 0.0);
    assert!(de < 1e-9, "erfc trivial-limit anchor FAILED: {de:.3e}");
}

/// FINITE-ε path equality: at ε = 1e-3 with trivial locality maps, the
/// direct path and the global-B domain-fit path see the same fitted J
/// blocks (up to transform round-off), so the masked energies must agree
/// to well below the ε-truncation scale.
#[test]
fn finite_eps_direct_matches_global_domain_fit() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let glob_cfg = AmplitudeLmp2Config { fit_radius_bohr: Some(1e6), ..cfg.clone() };
    let glob = amplitude_lmp2(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &glob_cfg,
    )
    .unwrap();
    let (dir, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let dd = (dir.e_corr - glob.e_corr).abs();
    eprintln!(
        "DIRECT finite-eps C4: E={:.10} global={:.10} |diff|={dd:.3e} keep {:.4} vs {:.4}",
        dir.e_corr, glob.e_corr, dir.keep_fraction, glob.keep_fraction
    );
    assert!(dd < 1e-8, "finite-eps direct vs global FAILED: {dd:.3e}");
}

/// MUTATION ARMS — each locality map, gutted, must change the energy
/// loudly; otherwise the map is dead code and the sweep would measure
/// nothing. (Verified-to-fail construction: each arm's assertion is on a
/// strictly positive energy displacement.)
#[test]
fn each_map_gutted_is_loud() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 4, ..Default::default() };
    let (base, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &trivial_maps(),
    )
    .unwrap();

    // (a) virtual domains gutted: 2 Bohr around each occupied centroid
    let (mut_v, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &DirectConfig { virt_radius_bohr: Some(2.0), ..trivial_maps() },
    )
    .unwrap();
    let dv = (mut_v.e_corr - base.e_corr).abs();
    eprintln!("MUTATION virt_radius=2: |dE|={dv:.3e}");
    assert!(dv > 1e-3, "virt-domain map gutted silently: |dE|={dv:.3e}");

    // (b) AO support gutted: only shells with |C| >= 0.3 survive
    let (mut_a, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &DirectConfig { ao_tail: 0.3, ..trivial_maps() },
    )
    .unwrap();
    let da = (mut_a.e_corr - base.e_corr).abs();
    eprintln!("MUTATION ao_tail=0.3: |dE|={da:.3e}");
    assert!(da > 1e-3, "AO-support map gutted silently: |dE|={da:.3e}");

    // (d) Schwarz triple cut gutted: an absurd threshold must zero most of
    // the integral stream and wreck the energy
    let (mut_s, st_s) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &DirectConfig { schwarz_skip: 10.0, ..trivial_maps() },
    )
    .unwrap();
    let ds = (mut_s.e_corr - base.e_corr).abs();
    eprintln!("MUTATION schwarz_skip=10: |dE|={ds:.3e} ({} triples skipped)", st_s.n_eri3_skipped);
    assert!(ds > 1e-3 && st_s.n_eri3_skipped > 0, "Schwarz cut gutted silently: |dE|={ds:.3e}");

    // (c) aux domains gutted: 4 Bohr fit domains
    let r_mut = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        &DirectConfig { aux_radius_bohr: 4.0, ..trivial_maps() },
    );
    match r_mut {
        Err(e) => eprintln!("MUTATION aux_radius=4: hard error (acceptable): {e}"),
        Ok((mut_x, _)) => {
            let dx = (mut_x.e_corr - base.e_corr).abs();
            eprintln!("MUTATION aux_radius=4: |dE|={dx:.3e}");
            assert!(dx > 1e-5, "aux-domain map gutted silently: |dE|={dx:.3e}");
        }
    }
}

/// MAP-ERROR FLATNESS (the artifact discriminator, run before quoting any
/// scaling number): at FIXED radii/thresholds the locality-map error vs the
/// trivial-map (eps-only) path must stay FLAT as the alkane grows. A
/// mis-constructed map (mixed index sets) shows error GROWING with size.
/// #[ignore]d — release, quiet box:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
///     --test lmp2_direct -- --ignored --nocapture bench_map_error_flatness
#[test]
#[ignore]
fn bench_map_error_flatness() {
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    println!("sys      op     eps    E(trivial)      E(prod-maps)    map_err     eps_err(vs can)");
    for xyz in ["alkane_8.xyz", "alkane_12.xyz", "alkane_16.xyz"] {
        let su = setup(xyz);
        let nc = su.mol.atoms.iter().filter(|a| a.z == 6).count();
        for (opname, op, cal) in
            [("coul", Operator::coulomb(), 0.7), ("erfc1", Operator::erfc(1.0), 0.02)]
        {
            let cfg = AmplitudeLmp2Config {
                eps: 1e-3,
                frozen_core: nc,
                pair_gate_cal: Some(cal),
                ..Default::default()
            };
            let (triv, _) = amplitude_lmp2_direct(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &trivial_maps(),
            )
            .unwrap();
            let (pr, _) = amplitude_lmp2_direct(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf,
                &AmplitudeLmp2Config { compute_reference: false, ..cfg },
                &prod,
            )
            .unwrap();
            println!(
                "{:9} {:6} 1e-3  {:.10} {:.10} {:+.3e} {:+.3e}",
                xyz.trim_end_matches(".xyz"),
                opname,
                triv.e_corr,
                pr.e_corr,
                pr.e_corr - triv.e_corr,
                triv.e_corr - triv.e_corr_canonical_ri,
            );
        }
    }
}

/// Schwarz-skip calibration: error vs triples saved at C16 (both ops),
/// against the skip=0 baseline with otherwise-production maps.
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
///     --test lmp2_direct -- --ignored --nocapture bench_schwarz_skip_sweep
#[test]
#[ignore]
fn bench_schwarz_skip_sweep() {
    let su = setup("alkane_16.xyz");
    let nc = 16;
    println!("op     skip    E_corr         dE_vs_skip0  eval_Mtriples skipped_M t_eri3 t_asm");
    for (opname, op, cal) in
        [("coul", Operator::coulomb(), 0.7), ("erfc1", Operator::erfc(1.0), 0.02)]
    {
        let cfg = AmplitudeLmp2Config {
            eps: 1e-3,
            frozen_core: nc,
            pair_gate_cal: Some(cal),
            compute_reference: false,
            ..Default::default()
        };
        let mut e0 = f64::NAN;
        for skip in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] {
            let prod = DirectConfig {
                aux_radius_bohr: 10.0,
                virt_radius_bohr: Some(12.0),
                ao_tail: 1e-3,
                schwarz_skip: skip,
                ..Default::default()
            };
            let (r, st) = amplitude_lmp2_direct(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
            )
            .unwrap();
            if skip == 0.0 {
                e0 = r.e_corr;
            }
            println!(
                "{:6} {:7.0e} {:.10} {:+.3e} {:10.2} {:8.2} {:6.2} {:6.2}",
                opname,
                skip,
                r.e_corr,
                r.e_corr - e0,
                st.n_eri3_shell_triples as f64 / 1e6,
                st.n_eri3_skipped as f64 / 1e6,
                st.t_eri3_s,
                r.timings.t_assembly_s,
            );
        }
    }
}

/// Wall-clock + saturation sweep over the alkane series, integral-direct
/// path vs canonical ri_mp2 (the payoff/crossover measurement).
/// #[ignore]d — release, quiet box:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
///     --test lmp2_direct -- --ignored --nocapture bench_direct_alkane_series
/// FERRIC_LMP2_BENCH_MAX_C caps the series (default 20; set 32/48 for the
/// past-onset tail).
#[test]
#[ignore]
fn bench_direct_alkane_series() {
    let max_c: usize = std::env::var("FERRIC_LMP2_BENCH_MAX_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    println!(
        "sys        op     eps    E_corr        dE_vs_can  keep    npair/gated dom(mn/mx) \
         strips(r mn/mx | c mn/mx) eri3_Mtriples t_maps t_eri3 t_metric t_pairs | t_asm t_solve t_ref(ri_mp2)"
    );
    for nc in [4usize, 8, 12, 16, 20, 32, 48] {
        if nc > max_c {
            break;
        }
        let su = setup(&format!("alkane_{nc}.xyz"));
        for (opname, op, cal) in
            [("coul", Operator::coulomb(), 0.7), ("erfc1", Operator::erfc(1.0), 0.02)]
        {
            for eps in [1e-3, 1e-4] {
                let cfg = AmplitudeLmp2Config {
                    eps,
                    frozen_core: nc,
                    pair_gate_cal: Some(cal),
                    ..Default::default()
                };
                let (r, st) = amplitude_lmp2_direct(
                    &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
                )
                .unwrap();
                println!(
                    "alkane_{:<3} {:6} {:6.0e} {:.8} {:+.3e} {:.4} {:5}/{:<5} {:4.0}/{:<4} {:5.0}/{:<5} {:4.0}/{:<4} \
                     {:8.2} {:6.2} {:6.2} {:8.2} {:7.2} | {:6.2} {:7.2} {:7.2}",
                    nc,
                    opname,
                    eps,
                    r.e_corr,
                    r.e_corr - r.e_corr_canonical_ri,
                    r.keep_fraction,
                    // CnH2n+2 active occupieds (frozen_core = nc): no = 3nc+1
                    (r.pair_fraction * ((3 * nc + 1) * (3 * nc + 1)) as f64).round() as usize,
                    r.n_pairs_gated,
                    r.dom_mean,
                    r.dom_max,
                    st.strip_rows_mean,
                    st.strip_rows_max,
                    st.strip_cols_mean,
                    st.strip_cols_max,
                    st.n_eri3_shell_triples as f64 / 1e6,
                    st.t_maps_s,
                    st.t_eri3_s,
                    st.t_metric_s,
                    st.t_pairs_s,
                    r.timings.t_assembly_s,
                    r.timings.t_solve_s,
                    r.timings.t_reference_s,
                );
            }
        }
    }
}

/// PRODUCTION-MAP SUB-DOMINANCE: with working locality maps the extra error
/// they introduce must stay sub-dominant to the ε truncation itself — the
/// family's port-fidelity bar (same as the aux_tail_frac test on the global
/// path). C8/erfc(1): the operator the maps are supposed to pay off for.
#[test]
fn production_maps_error_is_subdominant_to_eps_truncation() {
    let su = setup("alkane_8.xyz");
    let op = Operator::erfc(1.0);
    let full_cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 8, ..Default::default() };
    let eps_cfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 8, pair_gate_cal: Some(0.02), ..Default::default() };
    // ε=0 with trivial maps = exact (anchored above)
    let (r_full, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &full_cfg, &trivial_maps(),
    )
    .unwrap();
    // ε truncation alone (trivial maps)
    let (r_eps, _) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &eps_cfg, &trivial_maps(),
    )
    .unwrap();
    // ε + production locality maps
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    let (r_prod, stats) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &eps_cfg, &prod,
    )
    .unwrap();
    let trunc_err = (r_eps.e_corr - r_full.e_corr).abs();
    let map_err = (r_prod.e_corr - r_eps.e_corr).abs();
    eprintln!(
        "C8/erfc production maps: eps-trunc {trunc_err:.3e}, map-err {map_err:.3e} \
         (strips rows {:.0}/{} cols {:.0}/{}, eri3 triples {})",
        stats.strip_rows_mean,
        stats.strip_rows_max,
        stats.strip_cols_mean,
        stats.strip_cols_max,
        stats.n_eri3_shell_triples
    );
    assert!(map_err > 0.0, "maps changed nothing at production radii — vacuous?");
    assert!(
        map_err < trunc_err,
        "locality-map error ({map_err:.3e}) dominates the eps truncation ({trunc_err:.3e})"
    );
    // NOT asserted here: strip shrinkage. C8 is BELOW the locality onset for
    // both strip axes (the 10/12 Bohr partner-unions span the whole ~20 Bohr
    // molecule — measured rows 700/700 = naux, cols max 75 = nv on first
    // run), so shrinkage/saturation is a SWEEP observable on the alkane
    // series (protocol: do not declare below the onset), while the energy
    // sub-dominance above is size-independent and is the assertion.
}
