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
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
    }
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
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 0.0,
        frozen_core: 1,
        ..Default::default()
    };
    let glob_cfg = AmplitudeLmp2Config {
        fit_radius_bohr: Some(1e6),
        ..cfg.clone()
    };
    let glob = amplitude_lmp2(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &glob_cfg,
    )
    .unwrap();
    let (dir, stats) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
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
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 0.0,
        frozen_core: 0,
        ..Default::default()
    };
    let (dir, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let de = (dir.e_corr - dir.e_corr_canonical_ri).abs();
    eprintln!(
        "DIRECT erfc trivial limit water: E={:.10} |vs canonical|={de:.3e}",
        dir.e_corr
    );
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
    let cfg = AmplitudeLmp2Config {
        eps: 1e-3,
        frozen_core: 4,
        ..Default::default()
    };
    let glob_cfg = AmplitudeLmp2Config {
        fit_radius_bohr: Some(1e6),
        ..cfg.clone()
    };
    let glob = amplitude_lmp2(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &glob_cfg,
    )
    .unwrap();
    let (dir, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
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
    let cfg = AmplitudeLmp2Config {
        eps: 0.0,
        frozen_core: 4,
        ..Default::default()
    };
    let (base, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();

    // (a) virtual domains gutted: 2 Bohr around each occupied centroid
    let (mut_v, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &DirectConfig {
            virt_radius_bohr: Some(2.0),
            ..trivial_maps()
        },
    )
    .unwrap();
    let dv = (mut_v.e_corr - base.e_corr).abs();
    eprintln!("MUTATION virt_radius=2: |dE|={dv:.3e}");
    assert!(dv > 1e-3, "virt-domain map gutted silently: |dE|={dv:.3e}");

    // (b) AO support gutted: only shells with |C| >= 0.3 survive
    let (mut_a, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &DirectConfig {
            ao_tail: 0.3,
            ..trivial_maps()
        },
    )
    .unwrap();
    let da = (mut_a.e_corr - base.e_corr).abs();
    eprintln!("MUTATION ao_tail=0.3: |dE|={da:.3e}");
    assert!(da > 1e-3, "AO-support map gutted silently: |dE|={da:.3e}");

    // (d) Schwarz triple cut gutted: an absurd threshold must zero most of
    // the integral stream and wreck the energy
    let (mut_s, st_s) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &DirectConfig {
            schwarz_skip: 10.0,
            ..trivial_maps()
        },
    )
    .unwrap();
    let ds = (mut_s.e_corr - base.e_corr).abs();
    eprintln!(
        "MUTATION schwarz_skip=10: |dE|={ds:.3e} ({} triples skipped)",
        st_s.n_eri3_skipped
    );
    assert!(
        ds > 1e-3 && st_s.n_eri3_skipped > 0,
        "Schwarz cut gutted silently: |dE|={ds:.3e}"
    );

    // (c) aux domains gutted: 4 Bohr fit domains
    let r_mut = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &DirectConfig {
            aux_radius_bohr: 4.0,
            ..trivial_maps()
        },
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
        for (opname, op, cal) in [
            ("coul", Operator::coulomb(), 0.7),
            ("erfc1", Operator::erfc(1.0), 0.02),
        ] {
            let cfg = AmplitudeLmp2Config {
                compute_reference: true,
                eps: 1e-3,
                frozen_core: nc,
                pair_gate_cal: Some(cal),
                ..Default::default()
            };
            let (triv, _) = amplitude_lmp2_direct(
                &su.mol,
                &su.obs,
                &su.obs_bs,
                &su.dfbs,
                op,
                &su.rhf,
                &cfg,
                &trivial_maps(),
            )
            .unwrap();
            let (pr, _) = amplitude_lmp2_direct(
                &su.mol,
                &su.obs,
                &su.obs_bs,
                &su.dfbs,
                op,
                &su.rhf,
                &AmplitudeLmp2Config {
                    compute_reference: false,
                    ..cfg
                },
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

/// BATCH-MERGE ANCHOR: merged batches (k=4) must reproduce per-atom
/// batches (k=1) to the float-reassociation floor — the transform runs
/// over the larger batch union with zero-masked coefficients, so exact
/// zeros enter the sums in different groupings — AND must evaluate
/// strictly fewer shell triples (prototype-predicted ~0.4×; the
/// amortization must be real, not a no-op).
#[test]
fn batch_merge_matches_per_atom_and_saves_triples() {
    let su = setup("alkane_8.xyz");
    let op = Operator::erfc(1.0);
    let cfg = AmplitudeLmp2Config {
        eps: 1e-3,
        frozen_core: 8,
        pair_gate_cal: Some(0.02),
        compute_reference: false,
        ..Default::default()
    };
    let prod = |k: usize| DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: k,
        ..Default::default()
    };
    let (r1, s1) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &cfg,
        &prod(1),
    )
    .unwrap();
    let (r4, s4) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &cfg,
        &prod(4),
    )
    .unwrap();
    let dd = (r4.e_corr - r1.e_corr).abs();
    eprintln!(
        "batch_merge=4 vs 1: |dE|={dd:.3e}, eval triples {} -> {} ({:.2}x)",
        s1.n_eri3_shell_triples,
        s4.n_eri3_shell_triples,
        s4.n_eri3_shell_triples as f64 / s1.n_eri3_shell_triples as f64
    );
    assert!(dd < 1e-10, "batch merge changed the energy: |dE|={dd:.3e}");
    assert!(
        s4.n_eri3_shell_triples < s1.n_eri3_shell_triples,
        "batch merge saved nothing ({} vs {})",
        s4.n_eri3_shell_triples,
        s1.n_eri3_shell_triples
    );
}

/// BOUND-CONSERVATIVENESS ANCHOR for the Schwarz triple cut (review
/// finding, fixed pre-merge: the first implementation compared only
/// Q(μν) against the threshold, omitting the √(P|P) aux factor its own
/// doc promised — non-conservative wherever √(P|P) > 1). For every shell
/// triple the corrected cut drops at a production-scale threshold, the
/// true max |(P|μν)| must not exceed the threshold; and the old
/// Q-only rule must disagree with the corrected √(P|P)·Q(μν) rule on at
/// least one triple, proving the aux factor genuinely changes decisions.
#[test]
fn schwarz_triple_cut_is_conservative() {
    use ferric_integrals::engine::Engine;
    let su = setup("alkane_4.xyz");
    let op = Operator::coulomb();
    let skip = 1e-5;
    let q = ferric_scf::screening::SchwarzBounds::compute(op, &su.obs)
        .unwrap()
        .q;
    let ddf = su.dfbs.shell_dims();
    let mut eng2 = Engine::new_2center(op, &su.dfbs, 1e-14).unwrap();
    let qp: Vec<f64> = (0..su.dfbs.nshells())
        .map(|sp| {
            let np = ddf[sp];
            let vals = eng2.compute_eri2(&su.dfbs, sp, sp);
            let mut m = 0.0f64;
            for p in 0..np {
                m = m.max(vals[p * np + p].abs());
            }
            m.sqrt()
        })
        .collect();
    let mut eng3 = Engine::new_3center(op, &su.obs, &su.dfbs, 1e-14).unwrap();
    let (mut n_dropped, mut n_disagree, mut worst) = (0usize, 0usize, 0.0f64);
    for sp in 0..su.dfbs.nshells() {
        for s1 in 0..su.obs.nshells() {
            for s2 in 0..=s1 {
                let old_drops = q[(s1, s2)] < skip;
                let new_drops = qp[sp] * q[(s1, s2)] < skip;
                if old_drops != new_drops {
                    n_disagree += 1;
                }
                if new_drops {
                    n_dropped += 1;
                    if let Some(block) = eng3.compute_eri3(&su.obs, &su.dfbs, sp, s1, s2) {
                        for v in block {
                            worst = worst.max(v.abs());
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "schwarz conservativeness: {n_dropped} triples dropped, worst dropped \
         |(P|uv)| = {worst:.3e}, old-vs-new disagreements = {n_disagree}"
    );
    assert!(
        n_dropped > 1000,
        "cut dropped only {n_dropped} triples — test too small"
    );
    assert!(
        worst <= skip,
        "cut dropped a triple with |(P|uv)| = {worst:.3e} > {skip:.0e}"
    );
    assert!(
        n_disagree > 0,
        "aux factor changed no decisions — fix vacuous on this system?"
    );
}

/// TERFC PROBE — the Gate-0 reopen measurement. The 2026-07-09 kill of the
/// terfc RI lane applies to the GLOBAL-inverse formulation only (the
/// integrals bug behind the metric garbage was fixed on main, ea7fee6+);
/// the lit-validated escape (Glasbrenner 2020) is "never form the global
/// inverse" — which is what this path's per-pair domain V_DD inverses do
/// by construction. This probe measures whether terfc survives the
/// integral-direct route: (a) trivial-limit anchor vs canonical terfc
/// ri_mp2 on water AND alkane_8 (the system that was −1.9e7 Ha garbage
/// pre-fix), (b) the recorded detection invariants — |E| monotone toward
/// Coulomb as r0 grows, |E(terfc)| ≤ |E(coulomb)| — and (c) production
/// maps assemble without a V_DD Cholesky failure at C8.
/// Needs the terf tables:
///   FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 \
///     cargo test -p ferric-mp2 --release --test lmp2_direct -- \
///     --ignored --nocapture terfc_direct_probe
#[test]
#[ignore]
fn terfc_direct_probe() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("terfc_direct_probe SKIPPED: set FERRIC_TERF_TABLE_DIR (repo terf-tables/)");
        return;
    }
    // r0 in Bohr; ≈ the 0.75/1.05/2.0 Å frontier rows of the 2026-07-09 table
    let r0s = [1.4, 2.0, 3.8];
    for (xyz, fc) in [("water.xyz", 0usize), ("alkane_8.xyz", 8)] {
        let su = setup(xyz);
        let cfg = AmplitudeLmp2Config {
            compute_reference: true,
            eps: 0.0,
            frozen_core: fc,
            ..Default::default()
        };
        let (coul, _) = amplitude_lmp2_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
            &trivial_maps(),
        )
        .unwrap();
        let mut prev = 0.0f64;
        for &r0 in &r0s {
            let op = Operator::terfc(r0);
            let (dir, _) = amplitude_lmp2_direct(
                &su.mol,
                &su.obs,
                &su.obs_bs,
                &su.dfbs,
                op,
                &su.rhf,
                &cfg,
                &trivial_maps(),
            )
            .unwrap();
            let de = (dir.e_corr - dir.e_corr_canonical_ri).abs();
            eprintln!(
                "TERFC {xyz} r0={r0}: E={:.10} canonical={:.10} |dE|={de:.3e} (coul {:.10})",
                dir.e_corr, dir.e_corr_canonical_ri, coul.e_corr
            );
            assert!(
                de < 1e-9,
                "terfc trivial-limit anchor FAILED at {xyz} r0={r0}: {de:.3e}"
            );
            // Detection invariants per the 2026-07-09 record, WITH its stated
            // ~5% tolerance: the pointwise-kernel argument does not bind
            // tightly (measured: water r0=2 overshoots |E_coul| by 0.05%).
            // The garbage these catch was off by SEVEN orders (−1.9e7 Ha).
            assert!(
                dir.e_corr.abs() > prev - 0.01 * coul.e_corr.abs()
                    && dir.e_corr.abs() >= 0.5 * coul.e_corr.abs()
                    && dir.e_corr.abs() <= 1.05 * coul.e_corr.abs(),
                "terfc invariant violated at {xyz} r0={r0}: |E|={:.6} prev={prev:.6} coul={:.6}",
                dir.e_corr.abs(),
                coul.e_corr.abs()
            );
            prev = dir.e_corr.abs();
        }
    }
    // (c) production maps at C8/eps=1e-3, r0=2.0 Bohr: assembly must succeed
    // (a V_DD Cholesky failure here would be the local echo of the old
    // global breakdown) and the map error must stay sub-dominant.
    let su = setup("alkane_8.xyz");
    let op = Operator::terfc(2.0);
    let eps_cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 1e-3,
        frozen_core: 8,
        ..Default::default()
    };
    let (triv, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &eps_cfg,
        &trivial_maps(),
    )
    .unwrap();
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 0.0, // pinned off: this test isolates the locality maps (the cut has its own test file)
        ..Default::default()
    };
    let (pr, st) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &AmplitudeLmp2Config {
            compute_reference: false,
            ..eps_cfg
        },
        &prod,
    )
    .unwrap();
    let eps_err = (triv.e_corr - triv.e_corr_canonical_ri).abs();
    let map_err = (pr.e_corr - triv.e_corr).abs();
    eprintln!(
        "TERFC C8 production maps: eps-err {eps_err:.3e} map-err {map_err:.3e} \
         (strips {:.0}/{} rows, keep {:.4})",
        st.strip_rows_mean, st.strip_rows_max, pr.keep_fraction
    );
    assert!(
        map_err < eps_err,
        "terfc locality-map error ({map_err:.3e}) dominates eps truncation ({eps_err:.3e})"
    );
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
    for (opname, op, cal) in [
        ("coul", Operator::coulomb(), 0.7),
        ("erfc1", Operator::erfc(1.0), 0.02),
    ] {
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

/// A bench knob's raw value: `None` when unset. A set but non-Unicode value
/// panics, like any other unusable value.
fn bench_env(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => panic!("{name}: {e}"),
    }
}

/// FERRIC_LMP2_BENCH_EPS: `None` is the default list; an unparsable, negative
/// or non-finite entry (including an empty one) panics.
fn bench_eps_list(raw: Option<&str>) -> Vec<f64> {
    let Some(raw) = raw else {
        return vec![1e-3, 1e-4];
    };
    raw.split(',')
        .map(|item| match item.trim().parse::<f64>() {
            Ok(eps) if eps.is_finite() && eps >= 0.0 => eps,
            _ => panic!("FERRIC_LMP2_BENCH_EPS: bad eps {item:?} in {raw:?}"),
        })
        .collect()
}

/// FERRIC_LMP2_BENCH_OPS: `None` is both operators; an unknown name (including
/// an empty one) panics. Each entry is (name, operator, pair-gate calibration).
fn bench_op_list(raw: Option<&str>) -> Vec<(&'static str, Operator, f64)> {
    raw.unwrap_or("coul,erfc1")
        .split(',')
        .map(|item| match item.trim() {
            "coul" => ("coul", Operator::coulomb(), 0.7),
            "erfc1" => ("erfc1", Operator::erfc(1.0), 0.02),
            _ => panic!("FERRIC_LMP2_BENCH_OPS: unknown op {item:?} in {raw:?}"),
        })
        .collect()
}

#[test]
fn bench_knobs_default_to_the_historical_sweep() {
    assert_eq!(bench_eps_list(None), vec![1e-3, 1e-4]);
    let names: Vec<_> = bench_op_list(None).iter().map(|o| o.0).collect();
    assert_eq!(names, ["coul", "erfc1"]);
}

#[test]
fn bench_knobs_parse_lists_in_order() {
    assert_eq!(bench_eps_list(Some("1e-5, 1e-6")), vec![1e-5, 1e-6]);
    let ops = bench_op_list(Some("erfc1"));
    assert_eq!((ops.len(), ops[0].0, ops[0].2), (1, "erfc1", 0.02));
}

#[test]
#[should_panic(expected = "FERRIC_LMP2_BENCH_EPS: bad eps")]
fn bench_eps_parse_failure_panics() {
    bench_eps_list(Some("1e-3,oops"));
}

#[test]
#[should_panic(expected = "FERRIC_LMP2_BENCH_EPS: bad eps")]
fn bench_eps_empty_value_panics() {
    bench_eps_list(Some(""));
}

#[test]
#[should_panic(expected = "FERRIC_LMP2_BENCH_OPS: unknown op")]
fn bench_ops_unknown_value_panics() {
    bench_op_list(Some("coul,erf"));
}

/// Wall-clock + saturation sweep over the alkane series, integral-direct
/// path vs canonical ri_mp2 (the payoff/crossover measurement).
/// #[ignore]d — release, quiet box:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
///     --test lmp2_direct -- --ignored --nocapture bench_direct_alkane_series
/// FERRIC_LMP2_BENCH_MAX_C caps the series (default 20; set 32/48 for the
/// past-onset tail). FERRIC_LMP2_BENCH_EPS (comma list, default "1e-3,1e-4";
/// the canonical reference runs on the FIRST entry) and FERRIC_LMP2_BENCH_OPS
/// (comma list of coul,erfc1, default both) select the sweep; a bad value
/// panics rather than silently measuring the default.
#[test]
#[ignore]
fn bench_direct_alkane_series() {
    let eps_list = bench_eps_list(bench_env("FERRIC_LMP2_BENCH_EPS").as_deref());
    let ops = bench_op_list(bench_env("FERRIC_LMP2_BENCH_OPS").as_deref());
    let max_c: usize = std::env::var("FERRIC_LMP2_BENCH_MAX_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let min_c: usize = std::env::var("FERRIC_LMP2_BENCH_MIN_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5, // calibrated: 66-69% triples cut for ~1e-8 Ha at C16
        // merged batches (2026-09-05b): rows recorded before this date used
        // batch_merge=1 — expect ~0.4× the eri3 evaluations vs those rows
        batch_merge: 4,
        // pinned: the auto budget (0.8×RAM) ignores the bench's cgroup cap
        // and memcg-OOMs the C32 reference — the budget-vs-enforcement trap
        scratch_budget_bytes: 1usize << 30,
        // OFF here: the sweep measures the distance-map configuration the
        // §28-§29 rows were recorded with (screen rows get their own sweep)
        virt_schwarz_kappa: None,
        split_eri3_timers: false,
    };
    println!(
        "sys        op     eps    E_corr        dE_vs_can  keep    npair/gated dom(mn/mx) \
         strips(r mn/mx | c mn/mx) eri3_Mtriples t_maps t_eri3 t_metric t_pairs | t_asm t_solve t_ref(ri_mp2)"
    );
    for nc in [4usize, 8, 12, 16, 20, 32, 48] {
        if nc < min_c {
            continue;
        }
        if nc > max_c {
            break;
        }
        let su = setup(&format!("alkane_{nc}.xyz"));
        for &(opname, op, cal) in &ops {
            let mut e_can = f64::NAN;
            for (i_eps, &eps) in eps_list.iter().enumerate() {
                let cfg = AmplitudeLmp2Config {
                    eps,
                    frozen_core: nc,
                    pair_gate_cal: Some(cal),
                    eri3_budget_bytes: Some(2usize << 30), // see prod note; C32 ref needs 1.40 GB
                    // ri_mp2 is eps-independent: one reference per (sys, op).
                    // No reference at C48: its ~2 GB B doesn't fit the
                    // contended box's headroom and its wall would be
                    // throttle-noise; the crossover is decided at C32.
                    compute_reference: i_eps == 0 && nc <= 32,
                    ..Default::default()
                };
                let (mut r, st) = amplitude_lmp2_direct(
                    &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
                )
                .unwrap();
                if cfg.compute_reference {
                    e_can = r.e_corr_canonical_ri;
                } else {
                    r.e_corr_canonical_ri = e_can;
                }
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
    let full_cfg = AmplitudeLmp2Config {
        eps: 0.0,
        frozen_core: 8,
        ..Default::default()
    };
    let eps_cfg = AmplitudeLmp2Config {
        eps: 1e-3,
        frozen_core: 8,
        pair_gate_cal: Some(0.02),
        ..Default::default()
    };
    // ε=0 with trivial maps = exact (anchored above)
    let (r_full, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &full_cfg,
        &trivial_maps(),
    )
    .unwrap();
    // ε truncation alone (trivial maps)
    let (r_eps, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &eps_cfg,
        &trivial_maps(),
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
    assert!(
        map_err > 0.0,
        "maps changed nothing at production radii — vacuous?"
    );
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

/// SCHWARZ VIRTUAL-CANDIDATE SCREEN (`virt_schwarz_kappa`, prototype GO in
/// WIKI-APPEND-eps-linked-maps.md) — TRIVIAL LIMIT: a kappa so small that
/// the threshold kappa*eps passes every candidate must reproduce the
/// kappa=None path exactly (same candidate sets => identical blocks =>
/// identical CG => identical energy), while the screen CODE actually runs
/// (q vectors computed, filter applied). Distinct from eps=0, which
/// disables the screen entirely.
#[test]
fn schwarz_virt_trivial_limit_is_a_noop() {
    let su = setup("alkane_4.xyz");
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    for op in [Operator::coulomb(), Operator::erfc(1.0)] {
        let cfg = AmplitudeLmp2Config {
            eps: 1e-3,
            frozen_core: 4,
            compute_reference: false,
            ..Default::default()
        };
        let (base, st0) = amplitude_lmp2_direct(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
        )
        .unwrap();
        let (scr, st1) = amplitude_lmp2_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            op,
            &su.rhf,
            &cfg,
            &DirectConfig {
                virt_schwarz_kappa: Some(1e-12),
                ..prod.clone()
            },
        )
        .unwrap();
        let dd = (scr.e_corr - base.e_corr).abs();
        eprintln!(
            "SCHWARZ trivial limit {:?}: |dE|={dd:.3e} cand mean/max {:.1}/{} vs {:.1}/{}",
            op.kind, st1.virt_cand_mean, st1.virt_cand_max, st0.virt_cand_mean, st0.virt_cand_max
        );
        assert!(
            dd < 1e-12,
            "screen at all-pass threshold changed the energy: {dd:.3e}"
        );
        assert_eq!(st0.virt_cand_max, st1.virt_cand_max);
        assert!(
            (st0.virt_cand_mean - st1.virt_cand_mean).abs() < 1e-12,
            "all-pass screen changed candidate sets"
        );
    }
}

/// SUB-DOMINANCE: at production maps and eps=1e-3, the screen's energy
/// displacement at kappa=1 and kappa=3 must stay >=100x below the eps
/// truncation error (prototype measured 0 at kappa=1 and >=1000x
/// sub-dominant at kappa=3 on both operators), and kappa=3 must actually
/// TRIM the candidate sets — a screen that never drops anything would make
/// the displacement assert vacuous.
#[test]
fn schwarz_virt_sub_dominant_and_trims() {
    let su = setup("alkane_4.xyz");
    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        ..Default::default()
    };
    for op in [Operator::coulomb(), Operator::erfc(1.0)] {
        let cfg = AmplitudeLmp2Config {
            compute_reference: true,
            eps: 1e-3,
            frozen_core: 4,
            ..Default::default()
        };
        let (base, st0) = amplitude_lmp2_direct(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
        )
        .unwrap();
        let d_eps = (base.e_corr - base.e_corr_canonical_ri).abs();
        let mut cand_k3 = f64::INFINITY;
        for kappa in [1.0, 3.0] {
            let (scr, st) = amplitude_lmp2_direct(
                &su.mol,
                &su.obs,
                &su.obs_bs,
                &su.dfbs,
                op,
                &su.rhf,
                &AmplitudeLmp2Config {
                    compute_reference: false,
                    ..cfg.clone()
                },
                &DirectConfig {
                    virt_schwarz_kappa: Some(kappa),
                    ..prod.clone()
                },
            )
            .unwrap();
            let d_scr = (scr.e_corr - base.e_corr).abs();
            eprintln!(
                "SCHWARZ {:?} kappa={kappa}: |dE|={d_scr:.3e} (eps err {d_eps:.3e}) \
                 cand mean/max {:.1}/{} (union {:.1}/{})",
                op.kind, st.virt_cand_mean, st.virt_cand_max, st0.virt_cand_mean, st0.virt_cand_max
            );
            // measured: 0 at kappa=1 (both ops), 1.55e-5 vs 1.89e-3 (122x)
            // at kappa=3/erfc — 50x leaves drift headroom while a broken
            // screen (OR->AND mutation) missed by >200x, seen to fail
            assert!(
                d_scr * 50.0 < d_eps,
                "screen error {d_scr:.3e} not sub-dominant to eps error {d_eps:.3e} \
                 at kappa={kappa}"
            );
            assert!(st.virt_cand_mean <= st0.virt_cand_mean + 1e-12);
            if kappa == 3.0 {
                cand_k3 = st.virt_cand_mean;
            }
        }
        assert!(
            cand_k3 < st0.virt_cand_mean,
            "kappa=3 trimmed nothing ({cand_k3} vs {}) — sub-dominance assert vacuous",
            st0.virt_cand_mean
        );
    }
}

/// MUTATION ARM: an absurd kappa must gut the candidate sets and move the
/// energy loudly (screen is live, not dead code). A hard error (empty
/// candidate structure) is also an acceptable loud outcome.
#[test]
fn schwarz_virt_gutted_is_loud() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config {
        eps: 1e-3,
        frozen_core: 4,
        compute_reference: false,
        ..Default::default()
    };
    let (base, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let r_mut = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &DirectConfig {
            virt_schwarz_kappa: Some(100.0),
            ..trivial_maps()
        },
    );
    match r_mut {
        Err(e) => eprintln!("MUTATION virt_schwarz_kappa=100: hard error (acceptable): {e}"),
        Ok((mut_k, st)) => {
            let dk = (mut_k.e_corr - base.e_corr).abs();
            eprintln!(
                "MUTATION virt_schwarz_kappa=100: |dE|={dk:.3e} cand mean/max {:.1}/{}",
                st.virt_cand_mean, st.virt_cand_max
            );
            assert!(dk > 1e-3, "Schwarz screen gutted silently: |dE|={dk:.3e}");
        }
    }
}

/// OPT-IN REFERENCE (2026-09-23), integral-direct path: the whole point of
/// `lmp2_direct` is never forming the global (naux, no·nv) tensor, and the
/// canonical `ri_mp2` reference forms exactly that. With the default config
/// it must not run: `e_corr_canonical_ri` NaN, `t_reference_s` exactly 0.
/// The opt-in arm must report a finite reference and positive reference
/// time without moving the method energy.
///
/// Fails if reverted: restoring `compute_reference: true` as the `Default`
/// makes the default run report a finite reference and `t_reference_s > 0`.
#[test]
fn default_config_does_not_compute_the_canonical_reference() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config {
        eps: 1e-3,
        frozen_core: 1,
        ..Default::default()
    };
    let (off, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    assert!(
        off.e_corr_canonical_ri.is_nan(),
        "default direct run computed a canonical reference: {}",
        off.e_corr_canonical_ri
    );
    assert_eq!(
        off.timings.t_reference_s, 0.0,
        "default direct run spent time in the canonical ri_mp2 reference"
    );

    let (on, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &AmplitudeLmp2Config {
            compute_reference: true,
            ..cfg
        },
        &trivial_maps(),
    )
    .unwrap();
    assert!(on.e_corr_canonical_ri.is_finite() && on.e_corr_canonical_ri < 0.0);
    assert!(on.timings.t_reference_s > 0.0);
    assert!(
        (on.e_corr - off.e_corr).abs() < 1e-12,
        "reference switch moved the method energy: {:+.3e}",
        on.e_corr - off.e_corr
    );
}

/// DECISIVE-SCALING bench (2026-09-16): repeats + free size list + cpu_s.
///
/// Motivation (wiki/amplitude-threshold-lmp2.md §34): the headline exponent
/// rests on THREE points and moved 0.02-0.09 when only box conditions
/// changed, and the doc itself records that an affine model t = a*N - b fits
/// those same points as well as or better than a power law. A 4th point at
/// the same spacing does not separate those models. Two things do:
///   (1) a per-size sigma, so an exponent can be compared to 1.0 at all;
///   (2) the DIFFERENCED observable dt/dN, which cancels the affine offset b
///       -- flat under the affine artifact hypothesis, rising as N^(p-1)
///       under the genuine-power-law hypothesis.
/// Hence: no ri_mp2 reference (the crossover is settled and the reference is
/// what makes large N unaffordable), REPEATS for sigma, and a caller-supplied
/// size list so the tail can be sampled densely rather than merely extended.
///
/// Release, quiet box, uncapped:
///   OPENBLAS_NUM_THREADS=1 FERRIC_LMP2_SIZES=24,32,40,48 \
///   FERRIC_LMP2_REPS=3 FERRIC_LMP2_OPS=erfc1 FERRIC_LMP2_EPS=1e-3 \
///   cargo test -p ferric-mp2 --release --test lmp2_direct -- \
///     --ignored --nocapture bench_scaling_decisive
///
/// Memory: this bench reads `FERRIC_MEM_BUDGET_GB` ITSELF (default 6) and
/// converts it as GiB (`x * 2^30` bytes), which it hands to both
/// `eri3_budget_bytes` and the direct `scratch_budget_bytes`. That matches
/// `ferric_core::memory`, which parses the same variable as GiB and prints
/// the resolved budget in GiB (`BudgetResolution::audit_line`); only its
/// over-budget refusal messages quote decimal GB (/1e9).
#[test]
#[ignore = "benchmark: release build, quiet box, OPENBLAS_NUM_THREADS=1; C32+ needs several GB"]
fn bench_scaling_decisive() {
    let sizes: Vec<usize> = std::env::var("FERRIC_LMP2_SIZES")
        .unwrap_or_else(|_| "20,32,48".into())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let reps: usize = std::env::var("FERRIC_LMP2_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let ops: Vec<String> = std::env::var("FERRIC_LMP2_OPS")
        .unwrap_or_else(|_| "coul,erfc1".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let epss: Vec<f64> = std::env::var("FERRIC_LMP2_EPS")
        .unwrap_or_else(|_| "1e-3,1e-4".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let budget_gb: f64 = std::env::var("FERRIC_MEM_BUDGET_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6.0);
    let budget = (budget_gb * (1u64 << 30) as f64) as usize;

    let prod = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        scratch_budget_bytes: budget,
        virt_schwarz_kappa: None,
        split_eri3_timers: false,
    };

    println!(
        "# DECISIVE rows: sys op eps rep E_corr t_method wall_s cpu_s \
         t_maps t_eri3 t_metric t_pairs t_asm t_solve Mtriples dom_mx strip_r_mx strip_c_mx pairs_per_occ"
    );
    for nc in &sizes {
        let nc = *nc;
        let su = setup(&format!("alkane_{nc}.xyz"));
        for opname in &ops {
            let (op, cal) = match opname.as_str() {
                "coul" => (Operator::coulomb(), 0.7),
                "erfc1" => (Operator::erfc(1.0), 0.02),
                other => panic!("unknown op {other}"),
            };
            for eps in &epss {
                for rep in 0..reps {
                    let cfg = AmplitudeLmp2Config {
                        eps: *eps,
                        frozen_core: nc,
                        pair_gate_cal: Some(cal),
                        eri3_budget_bytes: Some(budget),
                        compute_reference: false,
                        ..Default::default()
                    };
                    let w0 = std::time::Instant::now();
                    let c0 = cpu_time_s();
                    let (r, st) = amplitude_lmp2_direct(
                        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &prod,
                    )
                    .unwrap();
                    let wall = w0.elapsed().as_secs_f64();
                    let cpu = cpu_time_s() - c0;
                    let t_method = r.timings.t_assembly_s + r.timings.t_solve_s;
                    let no = 3 * nc + 1;
                    println!(
                        "DEC alkane_{:<3} {:6} {:6.0e} {} {:.10} {:8.3} {:8.3} {:9.3} \
                         {:6.2} {:7.2} {:7.2} {:7.2} {:7.2} {:7.2} {:8.2} {:4.0} {:5.0} {:4.0} {:6.2}",
                        nc, opname, eps, rep,
                        r.e_corr, t_method, wall, cpu,
                        st.t_maps_s, st.t_eri3_s, st.t_metric_s, st.t_pairs_s,
                        r.timings.t_assembly_s, r.timings.t_solve_s,
                        st.n_eri3_shell_triples as f64 / 1e6,
                        r.dom_max, st.strip_rows_max, st.strip_cols_max,
                        r.pair_fraction * (no * no) as f64 / no as f64,
                    );
                }
            }
        }
    }
}

/// Process CPU seconds (user+sys, all threads) via /proc/self/stat.
fn cpu_time_s() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    // fields 14 (utime) and 15 (stime), 1-indexed, after the comm field
    let rest = &s[s.rfind(')').unwrap() + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = f[11].parse().unwrap();
    let stime: u64 = f[12].parse().unwrap();
    (utime + stime) as f64 / 100.0
}

/// STAGE-TIMER LIVENESS (perf plan Stage A): every stage timer of both LMP2
/// drivers is actually measured on C4 — a timer hard-coded to 0.0 (the
/// pre-Stage-A `t_spaces_s`) fails here — and the new pattern/solver
/// counters are populated and in range.
#[test]
fn stage_timers_are_live() {
    let su = setup("alkane_4.xyz");
    let op = Operator::coulomb();
    let cfg = AmplitudeLmp2Config {
        eps: 1e-4,
        frozen_core: 4,
        pair_gate_cal: Some(0.7),
        compute_reference: true,
        ..Default::default()
    };
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 2,
        split_eri3_timers: true,
        ..Default::default()
    };
    let (r, st) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &dcfg,
    )
    .unwrap();
    // the default (split off): only the fill sum is timed, the split is NaN,
    // and the stats flag never changes the result
    let plain = DirectConfig {
        split_eri3_timers: false,
        ..dcfg.clone()
    };
    let (r0, st0) = amplitude_lmp2_direct(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &plain,
    )
    .unwrap();
    assert_eq!(r0.e_corr.to_bits(), r.e_corr.to_bits());
    assert!(st0.t_eri3_fill_thread_s > 0.0);
    assert!(st0.t_eri3_eval_thread_s.is_nan() && st0.t_eri3_scatter_thread_s.is_nan());
    let g = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let timers = [
        ("direct t_spaces_s", r.timings.t_spaces_s),
        ("direct t_assembly_s", r.timings.t_assembly_s),
        ("direct t_solve_s", r.timings.t_solve_s),
        ("direct t_reference_s", r.timings.t_reference_s),
        ("t_maps_s", st.t_maps_s),
        ("t_eri3_s", st.t_eri3_s),
        ("t_eri3_fill_thread_s", st.t_eri3_fill_thread_s),
        ("t_eri3_eval_thread_s", st.t_eri3_eval_thread_s),
        ("t_eri3_scatter_thread_s", st.t_eri3_scatter_thread_s),
        ("t_eri3_transform_thread_s", st.t_eri3_transform_thread_s),
        ("batch_work_max_s", st.batch_work_max_s),
        ("batch_work_mean_s", st.batch_work_mean_s),
        ("t_metric_s", st.t_metric_s),
        ("t_pairs_s", st.t_pairs_s),
        ("t_loc_s", st.t_loc_s),
        ("global t_spaces_s", g.timings.t_spaces_s),
        ("global t_assembly_s", g.timings.t_assembly_s),
        ("global t_solve_s", g.timings.t_solve_s),
        ("global t_reference_s", g.timings.t_reference_s),
    ];
    eprintln!("{timers:?}");
    for (name, t) in timers {
        assert!(t > 0.0 && t.is_finite(), "timer {name} not live: {t}");
    }
    let fill_split = st.t_eri3_eval_thread_s + st.t_eri3_scatter_thread_s;
    assert!(
        (fill_split - st.t_eri3_fill_thread_s).abs() <= 1e-6 * st.t_eri3_fill_thread_s.max(1.0),
        "fill {} != eval + scatter {fill_split}",
        st.t_eri3_fill_thread_s
    );
    let split = st.t_eri3_fill_thread_s + st.t_eri3_transform_thread_s;
    assert!(
        split <= st.batch_work_mean_s * st.n_batches as f64 * (1.0 + 1e-9),
        "split {split} exceeds the summed batch work"
    );
    assert!(st.n_batches > 0 && st.batch_work_max_s >= st.batch_work_mean_s);
    for res in [&r, &g] {
        assert!(res.gate_on);
        assert!(res.n_pairs_unique > 0);
        assert!(res.block_density_min > 0.0 && res.block_density_min <= res.block_density_mean);
        assert!(res.block_density_mean <= 1.0);
        assert!(
            res.cg_lanczos_kappa_est >= 1.0 && res.cg_lanczos_kappa_est.is_finite(),
            "kappa {}",
            res.cg_lanczos_kappa_est
        );
        assert!(res.solve_plan_bytes > 0);
        assert_eq!(res.matvec_path, ferric_mp2::ragged::MatvecPath::Planned);
    }
}

/// The pre-Stage-B PCG loop (fresh `z`/`p`/`ap` every iteration, indexed
/// matvec), frozen as the oracle of the planned solve (CG vectors preallocated;
/// the matvec still allocates one output block per pair per call).
fn pcg_oracle(
    rg: &ferric_mp2::ragged::Ragged,
    f_oo: &ndarray::Array2<f64>,
    rtol: f64,
    max_iter: usize,
) -> (Vec<ndarray::Array2<f64>>, usize, f64) {
    use ferric_mp2::ragged::{apply_pattern, dot_blocks, matvec_indexed};
    use ndarray::Array2;
    let mut rhs: Vec<Array2<f64>> = Vec::new();
    for pb in &rg.pairs {
        let mut b = -pb.j_blk.clone();
        apply_pattern(&mut b, &pb.pat, pb.db.len());
        rhs.push(b);
    }
    let bnorm = dot_blocks(&rhs, &rhs).sqrt();
    let precond = |r: &[Array2<f64>]| -> Vec<Array2<f64>> {
        r.iter()
            .zip(&rg.pairs)
            .map(|(rb, pb)| rb / &pb.denom)
            .collect()
    };
    let mut fl = 0u64;
    let mut t: Vec<Array2<f64>> = rhs.iter().map(|b| Array2::zeros(b.dim())).collect();
    let mut r = rhs.clone();
    let mut z = precond(&r);
    let mut p = z.clone();
    let mut rz = dot_blocks(&r, &z);
    let (mut it, mut relres) = (0, 1.0);
    while it < max_iter {
        it += 1;
        let ap = matvec_indexed(rg, f_oo, &p, &mut fl);
        let alpha = rz / dot_blocks(&p, &ap);
        for k in 0..p.len() {
            t[k].scaled_add(alpha, &p[k]);
            r[k].scaled_add(-alpha, &ap[k]);
        }
        relres = dot_blocks(&r, &r).sqrt() / bnorm;
        if relres < rtol {
            break;
        }
        z = precond(&r);
        let rz_new = dot_blocks(&r, &z);
        let beta = rz_new / rz;
        for k in 0..p.len() {
            let mut pk = z[k].clone();
            pk.scaled_add(beta, &p[k]);
            p[k] = pk;
        }
        rz = rz_new;
    }
    (t, it, relres)
}

fn bits_equal(a: &[ndarray::Array2<f64>], b: &[ndarray::Array2<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.dim() == y.dim()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// B3 anchor: on C8 at ε = 1e-4 (production maps, gate on) the planned
/// matvec is BITWISE the indexed one (and counts the same flops), and the
/// planned PCG (preallocated CG vectors, not allocation-free) reproduces
/// the frozen pre-change loop bit for bit: same iteration count, same
/// amplitudes, same Hylleraas energy.
#[test]
fn matvec_planned_is_bitwise_matvec_indexed() {
    use ferric_mp2::lmp2_amplitude::{checked_vvhv, linear_energy, localized_spaces};
    use ferric_mp2::lmp2_direct::assemble_ragged_direct_local;
    use ferric_mp2::ragged::{
        apply_pattern, matvec_indexed, matvec_planned, solve_ragged, SolvePlan,
    };
    let su = setup("alkane_8.xyz");
    let op = Operator::coulomb();
    let (vvhv, _) = checked_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf, "test").unwrap();
    let spaces = localized_spaces(&su.mol, &su.obs, &su.rhf, 8, &vvhv).unwrap();
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        ..Default::default()
    };
    let (rg, _, _) = assemble_ragged_direct_local(
        &su.mol,
        &su.obs,
        &su.dfbs,
        op,
        &spaces,
        1e-4,
        1.0,
        Some(0.7),
        &dcfg,
    )
    .unwrap();
    assert!(rg.pairs.len() > 20, "too few pairs to exercise the plan");
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let t: Vec<ndarray::Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| {
            let mut x = ndarray::Array2::from_shape_fn((pb.da.len(), pb.db.len()), |_| rnd());
            apply_pattern(&mut x, &pb.pat, pb.db.len());
            x
        })
        .collect();
    let (mut f1, mut f2) = (0u64, 0u64);
    let want = matvec_indexed(&rg, &spaces.f_oo, &t, &mut f1);
    let plan = SolvePlan::new(&rg, &spaces.f_oo);
    let mut got: Vec<ndarray::Array2<f64>> = rg
        .pairs
        .iter()
        .map(|_| ndarray::Array2::zeros((0, 0)))
        .collect();
    matvec_planned(&rg, &plan, &t, &mut got, &mut f2);
    assert!(
        bits_equal(&want, &got),
        "planned matvec differs from indexed"
    );
    assert_eq!(f1, f2, "flop counters differ");

    let (t_old, it_old, rel_old) = pcg_oracle(&rg, &spaces.f_oo, 1e-11, 400);
    let (t_new, it_new, rel_new, conv, _) = solve_ragged(&rg, &spaces.f_oo, 1e-11, 400);
    assert!(conv);
    assert_eq!(it_old, it_new, "CG iteration count changed");
    assert_eq!(rel_old.to_bits(), rel_new.to_bits(), "final relres changed");
    assert!(bits_equal(&t_old, &t_new), "amplitudes changed");
    let (e_old, e_new) = (linear_energy(&rg, &t_old), linear_energy(&rg, &t_new));
    eprintln!("planned PCG: {it_new} iters, E {e_new:.12} (oracle {e_old:.12})");
    assert_eq!(e_old.to_bits(), e_new.to_bits(), "energy changed");
}

/// The production ragged system of an n-alkane (6-31G / cc-pVDZ-RI,
/// carbon cores frozen, production maps, calibrated Coulomb gate) and its
/// occupied Fock block — the input of every solver-level test below.
fn production_ragged(
    xyz: &str,
    frozen_core: usize,
    eps: f64,
) -> (ferric_mp2::ragged::Ragged, ndarray::Array2<f64>) {
    let (rg, f_oo, _) = production_ragged_op(xyz, frozen_core, eps, Operator::coulomb(), 0.7);
    (rg, f_oo)
}

/// [`production_ragged`] for any molecule, operator and gate calibration;
/// also returns the virtual Fock block (for the λ_min bound).
fn production_ragged_op(
    xyz: &str,
    frozen_core: usize,
    eps: f64,
    op: Operator,
    gate_cal: f64,
) -> (
    ferric_mp2::ragged::Ragged,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    use ferric_mp2::lmp2_amplitude::{checked_vvhv, localized_spaces};
    use ferric_mp2::lmp2_direct::assemble_ragged_direct_local;
    let su = setup(xyz);
    let (vvhv, _) = checked_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf, "test").unwrap();
    let spaces = localized_spaces(&su.mol, &su.obs, &su.rhf, frozen_core, &vvhv).unwrap();
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        ..Default::default()
    };
    let (rg, _, _) = assemble_ragged_direct_local(
        &su.mol,
        &su.obs,
        &su.dfbs,
        op,
        &spaces,
        eps,
        1.0,
        Some(gate_cal),
        &dcfg,
    )
    .unwrap();
    (rg, spaces.f_oo, spaces.f_vv)
}

/// Least-squares slope of log10|y| against log10 x.
fn loglog_slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let (xs, ys): (Vec<f64>, Vec<f64>) = pts
        .iter()
        .map(|&(x, y)| (x.log10(), y.abs().log10()))
        .unzip();
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    sxy / sxx
}

/// D1 STATIONARITY (perf plan #3a). The Hylleraas functional
/// E_H[t] = Σ (2t − t̃)(2J + A t) on a swap-closed pattern is stationary at
/// the solution: E_H(t) − E* = ⟨(2 − S)δ, Aδ⟩ with δ = t − t*, i.e. second
/// order in the CG residual (and ≥ 0: an upper bound), while the linear form
/// Σ (2t − t̃) J errs at FIRST order. Artifact hypothesis: if the extra
/// matvec or the swap handling were wrong, E_H would err at first order too.
/// Measured at ε = 1e-4 (production maps) on C4 and C8 Coulomb, C4 and C8
/// erfc(ω = 1), and the 3-D S22 water dimer (Coulomb), by solving at a
/// ladder of cg_rtol values and fitting log|ΔE| against log(relres) over the
/// points above the roundoff floor.
///
/// THE PER-RUN BOUND: at every rung, ΔE_H ≤ 3·‖PJ‖²·relres²/λ_min with
/// λ_min ≥ 2(ε_LUMO − ε_HOMO) (`cg_energy_error_bound`, the certificate's
/// `cg_energy_error_bound`) is asserted. Mutations (seen to fail):
/// `hylleraas_energy` returning the linear form; the bound's factor 3
/// replaced by 0.01.
#[test]
fn hylleraas_is_stationary() {
    let cases = [
        ("alkane_4.xyz", 4usize, "coul", Operator::coulomb(), 0.7),
        ("alkane_8.xyz", 8, "coul", Operator::coulomb(), 0.7),
        ("alkane_4.xyz", 4, "erfc", Operator::erfc(1.0), 0.02),
        ("alkane_8.xyz", 8, "erfc", Operator::erfc(1.0), 0.02),
        ("s22/water_dimer.xyz", 2, "coul", Operator::coulomb(), 0.7),
    ];
    for (xyz, fc, opn, op, cal) in cases {
        stationarity_case(xyz, fc, opn, op, cal);
    }
}

/// One [`hylleraas_is_stationary`] system: slopes, the upper-bound sign,
/// and the per-run a-priori bound at every rtol rung.
fn stationarity_case(xyz: &str, fc: usize, opn: &str, op: Operator, cal: f64) {
    use ferric_mp2::lmp2_amplitude::{
        cg_energy_error_bound, hylleraas_energy, linear_energy, operator_lambda_min_lower_bound,
        pattern_rhs_norm,
    };
    use ferric_mp2::ragged::{check_swap_closed, solve_ragged_budgeted_with, MatvecPath};
    const FLOOR: f64 = 1e-12;
    let (rg, f_oo, f_vv) = production_ragged_op(xyz, fc, 1e-4, op, cal);
    check_swap_closed(&rg).unwrap();
    let lam = operator_lambda_min_lower_bound(&f_oo, &f_vv).unwrap();
    let bnorm = pattern_rhs_norm(&rg);
    assert!(lam > 0.0, "{xyz}: no positive gap");
    // the production (budgeted) solve, ample budget: planned, all dense
    let solve = |rtol: f64| {
        let b = solve_ragged_budgeted_with(&rg, &f_oo, rtol, 400, usize::MAX, 0.0);
        assert_eq!(b.decision.path, MatvecPath::Planned);
        let e_h = hylleraas_energy(&rg, &f_oo, b.plan(), &b.solve.t).unwrap();
        (b.solve, e_h)
    };
    let (reference, e_ref) = solve(1e-13);
    assert!(
        reference.converged,
        "{xyz}: reference solve did not converge"
    );
    let e_ref_lin = linear_energy(&rg, &reference.t);
    assert!(
        (e_ref - e_ref_lin).abs() < 1e-13,
        "{xyz}: forms disagree at the converged solution: {:.3e}",
        e_ref - e_ref_lin
    );
    let (mut hy, mut lin) = (Vec::new(), Vec::new());
    let mut seen_iters = Vec::new();
    let (mut c_h_max, mut use_max) = (0.0f64, 0.0f64);
    for rtol in [
        1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6,
    ] {
        let (s, e_h) = solve(rtol);
        if seen_iters.contains(&s.iterations) {
            continue;
        }
        seen_iters.push(s.iterations);
        let dh = e_h - e_ref;
        let dl = linear_energy(&rg, &s.t) - e_ref;
        let bound = cg_energy_error_bound(bnorm, s.relres, lam);
        eprintln!(
            "STAT {xyz} {opn} rtol {rtol:.0e} iters {:3} relres {:.3e}  dE_H {dh:+.3e}  \
             bound {bound:.3e} (dE_H/bound {:.3})  dE_lin {dl:+.3e}  dE_H/relres^2 {:.3e}",
            s.iterations,
            s.relres,
            dh / bound,
            dh / (s.relres * s.relres)
        );
        assert!(
            dh > -FLOOR,
            "{xyz}: E_H fell below the converged energy ({dh:.3e}): not an upper bound"
        );
        assert!(
            dh <= bound + FLOOR,
            "{xyz} {opn}: dE_H {dh:.3e} exceeds the a-priori bound {bound:.3e}"
        );
        if dh.abs() > FLOOR {
            hy.push((s.relres, dh));
            c_h_max = c_h_max.max(dh / (s.relres * s.relres));
            use_max = use_max.max(dh / bound);
        }
        if dl.abs() > FLOOR {
            lin.push((s.relres, dl));
        }
    }
    assert!(
        hy.len() >= 4 && lin.len() >= 4,
        "{xyz}: too few points to fit"
    );
    let (sh, sl) = (loglog_slope(&hy), loglog_slope(&lin));
    eprintln!(
        "STAT {xyz} {opn}: slope E_H {sh:.3} ({} pts), slope E_lin {sl:.3} ({} pts), \
         max dE_H/relres^2 {c_h_max:.3e} Ha, bound constant 3*|PJ|^2/lam {:.3e} Ha \
         (|PJ| {bnorm:.3e}, lam_lb {lam:.3e}), max dE_H/bound {use_max:.3}",
        hy.len(),
        lin.len(),
        3.0 * bnorm * bnorm / lam
    );
    assert!(
        (1.7..=2.6).contains(&sh),
        "{xyz}: Hylleraas error slope {sh:.3} is not quadratic"
    );
    assert!(
        (0.6..=1.4).contains(&sl),
        "{xyz}: linear-form error slope {sl:.3} is not first order"
    );
}

/// D2 (perf plan #3e): the adaptive Fvv product. On C8 at ε = 1e-4
/// (production maps, gate on): ρ* = 0 puts NO block on the sparse branch and
/// is bitwise the indexed matvec; a forced ρ* > 1 puts EVERY block on it and
/// agrees with the dense matvec to ≤ 1e-13·max|out| on a pattern-projected
/// random operand, and the two full PCG solves agree (same iteration count,
/// energies ≤ 1e-13); an intermediate ρ* (the median block density) splits
/// the blocks and still agrees. Mutation (seen to fail): the sparse branch
/// drops its t·Fvv_bb half.
#[test]
fn fvv_sparse_branch_matches_dense() {
    use ferric_mp2::lmp2_amplitude::hylleraas_energy;
    use ferric_mp2::ragged::{
        apply_pattern, matvec_indexed, matvec_planned, solve_ragged_budgeted_with, SolvePlan,
    };
    use ndarray::Array2;
    let (rg, f_oo) = production_ragged("alkane_8.xyz", 8, 1e-4);
    let mut seed = 11u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let t: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| {
            let mut x = Array2::from_shape_fn((pb.da.len(), pb.db.len()), |_| rnd());
            apply_pattern(&mut x, &pb.pat, pb.db.len());
            x
        })
        .collect();
    let empty = || -> Vec<Array2<f64>> { rg.pairs.iter().map(|_| Array2::zeros((0, 0))).collect() };
    let mut fl = 0u64;
    let want = matvec_indexed(&rg, &f_oo, &t, &mut fl);

    let dense = SolvePlan::with_fvv_threshold(&rg, &f_oo, 0.0);
    assert_eq!(
        dense.n_fvv_sparse(),
        0,
        "rho* = 0 must keep every block dense"
    );
    let mut got = empty();
    matvec_planned(&rg, &dense, &t, &mut got, &mut fl);
    assert!(
        bits_equal(&want, &got),
        "rho* = 0 is not bitwise the indexed matvec"
    );

    // the sparse branch reads Fvv_aa's column k as row k: Fvv must be
    // symmetric (to rounding — it is a block of Cᵀ F C)
    let (mut asym, mut fmax) = (0.0f64, 0.0f64);
    for pb in &rg.pairs {
        for m in [&pb.fvv_aa, &pb.fvv_bb] {
            asym = asym.max((m - &m.t()).iter().fold(0.0f64, |a, x| a.max(x.abs())));
            fmax = fmax.max(m.iter().fold(0.0f64, |a, x| a.max(x.abs())));
        }
    }
    eprintln!("Fvv block asymmetry {asym:.3e} (max |Fvv| {fmax:.3e})");
    assert!(
        asym <= 1e-13 * fmax,
        "Fvv blocks are not symmetric: {asym:.3e}"
    );
    let mut dens: Vec<f64> = rg
        .pairs
        .iter()
        .map(|pb| pb.pat.iter().filter(|&&x| x).count() as f64 / pb.pat.len() as f64)
        .collect();
    dens.sort_by(f64::total_cmp);
    let median = dens[dens.len() / 2];
    eprintln!(
        "block density: min {:.3} median {median:.3} max {:.3} ({} blocks)",
        dens[0],
        dens[dens.len() - 1],
        dens.len()
    );
    let scale = want
        .iter()
        .flat_map(|b| b.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    for (label, rho) in [("forced", 1.5), ("median", median)] {
        let plan = SolvePlan::with_fvv_threshold(&rg, &f_oo, rho);
        let ns = plan.n_fvv_sparse();
        if label == "forced" {
            assert_eq!(
                ns,
                rg.pairs.len(),
                "rho* > 1 must put every block on the sparse branch"
            );
        } else {
            assert!(
                ns > 0 && ns < rg.pairs.len(),
                "median rho* did not split the blocks ({ns})"
            );
        }
        let mut got = empty();
        matvec_planned(&rg, &plan, &t, &mut got, &mut fl);
        let dev = want
            .iter()
            .zip(&got)
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0f64, f64::max)
            / scale;
        eprintln!("{label} rho* {rho:.3}: {ns} sparse blocks, matvec dev {dev:.3e} rel");
        assert!(
            dev <= 1e-13,
            "{label}: sparse vs dense matvec {dev:.3e} > 1e-13"
        );

        // the production (budgeted) solve at rho* = 0 and at this rho*
        let bd = solve_ragged_budgeted_with(&rg, &f_oo, 1e-11, 400, usize::MAX, 0.0);
        let bs = solve_ragged_budgeted_with(&rg, &f_oo, 1e-11, 400, usize::MAX, rho);
        assert_eq!(bs.plan().map(SolvePlan::n_fvv_sparse), Some(ns));
        let (sd, ss) = (&bd.solve, &bs.solve);
        assert!(sd.converged && ss.converged);
        assert_eq!(
            sd.iterations, ss.iterations,
            "{label}: CG iteration count differs"
        );
        let (ed, es) = (
            hylleraas_energy(&rg, &f_oo, bd.plan(), &sd.t).unwrap(),
            hylleraas_energy(&rg, &f_oo, bs.plan(), &ss.t).unwrap(),
        );
        eprintln!(
            "{label}: E dense {ed:.15} sparse {es:.15} |dE| {:.3e}",
            (ed - es).abs()
        );
        assert!(
            (ed - es).abs() <= 1e-13,
            "{label}: energies differ by {:.3e}",
            (ed - es).abs()
        );
    }
}

/// `hylleraas_energy` checks swap closure ITSELF: a pattern made one-sided
/// in one block (C4 production system) is an error, not a non-stationary
/// number. Mutation (seen to fail): the check removed from
/// `hylleraas_energy`.
#[test]
fn hylleraas_energy_rejects_a_one_sided_pattern() {
    use ferric_mp2::lmp2_amplitude::hylleraas_energy;
    use ferric_mp2::ragged::solve_ragged_budgeted_with;
    let (mut rg, f_oo) = production_ragged("alkane_4.xyz", 4, 1e-4);
    let b = solve_ragged_budgeted_with(&rg, &f_oo, 1e-8, 400, usize::MAX, 0.0);
    assert!(hylleraas_energy(&rg, &f_oo, b.plan(), &b.solve.t).is_ok());
    let t = b.solve.t.clone();
    drop(b);
    let pb = rg
        .pairs
        .iter_mut()
        .find(|pb| {
            let n = pb.da.len();
            (0..n).any(|r| ((r + 1)..n).any(|c| pb.pat[r * n + c] && pb.pat[c * n + r]))
        })
        .expect("a block with an off-diagonal kept element");
    let n = pb.da.len();
    let (r, c) = (0..n)
        .flat_map(|r| ((r + 1)..n).map(move |c| (r, c)))
        .find(|&(r, c)| pb.pat[r * n + c] && pb.pat[c * n + r])
        .unwrap();
    pb.pat[c * n + r] = false;
    let err = hylleraas_energy(&rg, &f_oo, None, &t).unwrap_err();
    assert!(format!("{err}").contains("swap-closed"), "{err}");
}

/// `fvv_sparse_rho` must be a finite ρ* ≥ 0: NaN and negative values are a
/// configuration error from the driver (water, ε = 1e-4), and the validator
/// accepts 0 and any positive value. Mutation (seen to fail): the
/// validation call removed from `solve_and_count`.
#[test]
fn fvv_sparse_rho_rejects_nan_and_negative() {
    use ferric_mp2::lmp2_amplitude::validate_fvv_sparse_rho;
    assert!(validate_fvv_sparse_rho(0.0).is_ok() && validate_fvv_sparse_rho(1.5).is_ok());
    let su = setup("water.xyz");
    for bad in [f64::NAN, -0.1] {
        assert!(validate_fvv_sparse_rho(bad).is_err());
        let cfg = AmplitudeLmp2Config {
            eps: 1e-4,
            frozen_core: 1,
            fvv_sparse_rho: bad,
            ..Default::default()
        };
        let err = amplitude_lmp2_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
            &DirectConfig::default(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("fvv_sparse_rho"), "{err}");
    }
}

// ---- the adaptive PCG tolerance ----

/// One row of the tolerance claim: the default against an explicit 1e-11 from
/// the same code. `check` asserts |ΔE| <= reported bound <= 1e-10 Ha, no
/// tightening, and the certificate-facing outcome. Matvecs = iterations + 1
/// (the Hylleraas energy's own). Artifact hypothesis: a bound or energy that
/// did not track the residual would break |ΔE| <= bound on some row, and an
/// energy linear in the residual would miss it by orders of magnitude.
fn rtol_row(xyz: &str, fc: usize, opn: &str, op: Operator, cal: f64, in_core: bool, check: bool) {
    use ferric_mp2::lmp2_amplitude::{AmplitudeLmp2Result, CgRtolSource};
    let su = setup(xyz);
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        ..Default::default()
    };
    let cfg = |rtol: Option<f64>| AmplitudeLmp2Config {
        eps: 1e-4,
        frozen_core: fc,
        pair_gate_cal: Some(cal),
        cg_rtol: rtol,
        ..Default::default()
    };
    let run_direct = |rtol| {
        amplitude_lmp2_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            op,
            &su.rhf,
            &cfg(rtol),
            &dcfg,
        )
        .unwrap()
        .0
    };
    let run_core = |rtol| {
        amplitude_lmp2(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            op,
            &su.rhf,
            &cfg(rtol),
        )
        .unwrap()
    };
    let assess = |path: &str, new: AmplitudeLmp2Result, strict: AmplitudeLmp2Result| {
        let de = new.e_corr - strict.e_corr;
        let (o, bound) = (new.cg_rtol_outcome, new.cg_energy_error_bound);
        eprintln!(
            "RTOLROW {xyz} {opn} {path}: dE {de:+.3e} bound {bound:.3e} iters {} vs {} matvecs \
             {} vs {} relres {:.3e} true {:.3e} true/recursive-1 {:+.2e} tightened {} rtol {:.0e}",
            new.cg_iterations,
            strict.cg_iterations,
            new.cg_iterations + 1,
            strict.cg_iterations + 1,
            new.cg_relres,
            new.cg_true_relres,
            new.cg_true_relres / new.cg_relres - 1.0,
            o.tightened,
            o.rtol_used
        );
        assert!(new.cg_converged && strict.cg_converged);
        assert_eq!(strict.cg_rtol_outcome.source, CgRtolSource::Explicit);
        assert_eq!(strict.cg_rtol_outcome.rtol_used, 1e-11);
        assert_eq!(o.source, CgRtolSource::AdaptiveDefault);
        assert_eq!(o.target_met, Some(true), "{xyz} {opn} {path}");
        assert!(bound <= 1e-10, "{xyz} {opn} {path}: bound {bound:.3e}");
        assert!(
            de.abs() <= bound,
            "{xyz} {opn} {path}: |dE| {de:.3e} > bound {bound:.3e}"
        );
        if check {
            assert!(!o.tightened, "{xyz} {opn} {path}: tightening fired");
            assert_eq!(o.rtol_used, 1e-6);
            assert!(new.cg_iterations < strict.cg_iterations);
        }
    };
    assess("direct", run_direct(None), run_direct(Some(1e-11)));
    if in_core {
        assess("in-core", run_core(None), run_core(Some(1e-11)));
    }
}

#[test]
fn default_rtol_energy_is_within_its_bound() {
    let coul = Operator::coulomb();
    let erfc = Operator::erfc(1.0);
    rtol_row("alkane_4.xyz", 4, "coul", coul, 0.7, true, true);
    rtol_row("alkane_4.xyz", 4, "erfc", erfc, 0.02, true, true);
    rtol_row("alkane_8.xyz", 8, "coul", coul, 0.7, false, true);
    rtol_row("alkane_8.xyz", 8, "erfc", erfc, 0.02, false, true);
    rtol_row("s22/water_dimer.xyz", 2, "coul", coul, 0.7, true, true);
}

fn adaptive(start: f64, floor: f64, target_ha: f64) -> ferric_mp2::lmp2_amplitude::CgRtolPolicy {
    ferric_mp2::lmp2_amplitude::CgRtolPolicy::Adaptive {
        start,
        floor,
        target_ha,
    }
}

/// Continuing is not restarting: stopping at 1e-6 and carrying on to 1e-9 is
/// bitwise the straight 1e-9 solve, on both matvec paths, with no extra
/// iterations.
#[test]
fn continued_pcg_is_bitwise_a_straight_solve() {
    use ferric_mp2::ragged::{solve_ragged_budgeted_steered, solve_ragged_budgeted_with};
    let (rg, f_oo) = production_ragged("alkane_4.xyz", 4, 1e-4);
    for budget in [usize::MAX, 1] {
        let straight = solve_ragged_budgeted_with(&rg, &f_oo, 1e-9, 400, budget, 0.0);
        let mut stops = Vec::new();
        let cont = solve_ragged_budgeted_steered(&rg, &f_oo, 1e-6, 400, budget, 0.0, |relres| {
            stops.push(relres);
            (stops.len() == 1).then_some(1e-9)
        });
        assert_eq!(stops.len(), 2, "one stop per tolerance");
        assert!(stops[0] < 1e-6 && stops[0] > 1e-9);
        assert!(cont.solve.converged);
        assert_eq!(cont.solve.iterations, straight.solve.iterations);
        assert_eq!(cont.solve.relres.to_bits(), straight.solve.relres.to_bits());
        assert!(bits_equal(&cont.solve.t, &straight.solve.t));
    }
}

/// A target low enough to force tightening: it fires, lands at the derived
/// tolerance, meets the target, and the energy error obeys the final bound.
/// Mutants seen to fail: tightening never fires; the threshold check inverted.
#[test]
fn tightening_fires_and_meets_a_low_target() {
    use ferric_mp2::lmp2_amplitude::{
        hylleraas_energy, operator_lambda_min_lower_bound, pattern_rhs_norm,
        solve_under_rtol_policy, CgRtolSource, PolicySolve, LMP2_CG_ENERGY_ERROR_TARGET,
    };
    let (rg, f_oo, f_vv) = production_ragged_op("alkane_4.xyz", 4, 1e-4, Operator::coulomb(), 0.7);
    let run = |p| solve_under_rtol_policy(&rg, &f_oo, &f_vv, p, 400, usize::MAX, 0.0).unwrap();
    let reference = run(adaptive(1e-13, 1e-13, 1.0)).budgeted;
    let e_ref = hylleraas_energy(&rg, &f_oo, reference.plan(), &reference.solve.t).unwrap();

    // the default target does not fire on this system, from the same start
    let PolicySolve {
        budgeted: at_start,
        outcome: o0,
        bound: b0,
        ..
    } = run(adaptive(1e-3, 1e-11, 1.0));
    assert!(!o0.tightened && o0.target_met == Some(true));
    assert!(
        b0 > 1e-9,
        "start bound {b0:.3e} must exceed the targets below"
    );

    for target in [1e-8, LMP2_CG_ENERGY_ERROR_TARGET, 1e-13] {
        let PolicySolve {
            budgeted: b,
            outcome: o,
            bound,
            ..
        } = run(adaptive(1e-3, 1e-11, target));
        let e = hylleraas_energy(&rg, &f_oo, b.plan(), &b.solve.t).unwrap();
        let de = e - e_ref;
        eprintln!(
            "TIGHTEN target {target:.0e}: rtol {:.3e} iters {} (start {}) relres {:.3e} bound \
             {bound:.3e} dE {de:+.3e}",
            o.rtol_used, b.solve.iterations, at_start.solve.iterations, b.solve.relres
        );
        let lam = operator_lambda_min_lower_bound(&f_oo, &f_vv).unwrap();
        let derived = 0.5 * (target * lam / 3.0).sqrt() / pattern_rhs_norm(&rg);
        assert_eq!(o.source, CgRtolSource::AdaptiveDefault);
        assert!(o.tightened, "target {target:.0e} did not fire");
        assert_eq!(o.rtol_used, derived.max(1e-11));
        assert_eq!(o.target_met, Some(true));
        assert!(b.solve.converged && bound <= target);
        assert!(b.solve.iterations > at_start.solve.iterations);
        assert!(de.abs() <= bound + 1e-14, "dE {de:.3e} bound {bound:.3e}");
    }
}

/// An unreachable target is reported, at the floor and at the iteration limit,
/// and the floor is never passed.
#[test]
fn unreachable_target_is_reported() {
    use ferric_mp2::lmp2_amplitude::{solve_under_rtol_policy, PolicySolve};
    let (rg, f_oo, f_vv) = production_ragged_op("alkane_4.xyz", 4, 1e-4, Operator::coulomb(), 0.7);
    let PolicySolve {
        budgeted: b,
        outcome: o,
        bound,
        ..
    } = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &f_vv,
        adaptive(1e-6, 1e-9, 1e-40),
        400,
        usize::MAX,
        0.0,
    )
    .unwrap();
    assert!(b.solve.converged && o.tightened);
    assert_eq!(o.rtol_used, 1e-9);
    assert_eq!(o.target_met, Some(false));
    assert!(bound > 1e-40 && b.solve.relres > 1e-11);

    let PolicySolve { budgeted: s, .. } = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &f_vv,
        adaptive(1e-6, 1e-11, 1.0),
        400,
        usize::MAX,
        0.0,
    )
    .unwrap();
    let limit = s.solve.iterations + 1;
    let PolicySolve {
        budgeted: b,
        outcome: o,
        bound,
        ..
    } = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &f_vv,
        adaptive(1e-6, 1e-11, 1e-30),
        limit,
        usize::MAX,
        0.0,
    )
    .unwrap();
    assert!(!b.solve.converged && o.tightened);
    assert_eq!(b.solve.iterations, limit);
    assert_eq!(o.target_met, Some(false));
    assert!(bound > 1e-30);
}

/// With no positive gap the bound is undefined: the adaptive policy runs at
/// the strict tolerance from the start and never reports a met target.
#[test]
fn no_gap_uses_the_strict_tolerance() {
    use ferric_mp2::lmp2_amplitude::{
        solve_under_rtol_policy, CgRtolPolicy, CgRtolSource, PolicySolve,
    };
    use ferric_mp2::ragged::solve_ragged_budgeted_with;
    let (rg, f_oo, f_vv) = production_ragged_op("alkane_4.xyz", 4, 1e-4, Operator::coulomb(), 0.7);
    // only the bound reads f_vv: the operator lives in the pair blocks
    let closed = &f_vv - &(ndarray::Array2::<f64>::eye(f_vv.nrows()) * 100.0);
    let PolicySolve {
        budgeted: b,
        outcome: o,
        bound,
        ..
    } = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &closed,
        CgRtolPolicy::from_config(None),
        400,
        usize::MAX,
        0.0,
    )
    .unwrap();
    assert_eq!(o.source, CgRtolSource::StrictNoGap);
    assert_eq!(o.rtol_used, 1e-11);
    assert!(!o.tightened && o.target_met.is_none() && bound.is_nan());
    let straight = solve_ragged_budgeted_with(&rg, &f_oo, 1e-11, 400, usize::MAX, 0.0);
    assert!(bits_equal(&b.solve.t, &straight.solve.t));
}

/// An explicit tolerance is literal: the driver stops at it even with the
/// bound far above the default target, and the certificate says so.
#[test]
fn explicit_rtol_is_literal() {
    use ferric_mp2::lmp2_amplitude::CgRtolSource;
    use ferric_mp2::lmp2_certificate::{Lmp2Certificate, SystemShape};
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config {
        eps: 1e-4,
        frozen_core: 4,
        cg_rtol: Some(1e-3),
        ..Default::default()
    };
    let r = amplitude_lmp2(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap();
    let o = r.cg_rtol_outcome;
    assert_eq!(o.source, CgRtolSource::Explicit);
    assert!(!o.tightened && o.rtol_used == 1e-3 && o.target_met.is_none());
    assert!(r.cg_relres > 1e-6 && r.cg_energy_error_bound > 1e-10);
    let shape = SystemShape::new(&su.mol, &su.obs_bs);
    let cert = Lmp2Certificate::new(&cfg, None, &r, &shape, &Operator::coulomb());
    assert_eq!(cert.cg_rtol, 1e-3);
    assert_eq!(cert.cg_rtol_source, "explicit");
    assert!(
        cert.solver_warnings
            .iter()
            .any(|w| w.contains("not tightened")),
        "{:?}",
        cert.solver_warnings
    );
}

/// The reported bound is a posteriori: it sits on the TRUE residual, which an
/// independent matvec reproduces, and the true residual is what the helper
/// reads even when `t` is not the PCG's own iterate. Mutants seen to fail: the
/// bound fed a scaled-down residual; the bound on the recursive residual; the
/// helper's residual dropping `J`.
#[test]
fn reported_bound_uses_the_true_residual() {
    use ferric_mp2::lmp2_amplitude::{
        cg_energy_error_bound, hylleraas_energy_and_residual, operator_lambda_min_lower_bound,
        pattern_rhs_norm, solve_under_rtol_policy, CgRtolPolicy,
    };
    use ferric_mp2::ragged::matvec_indexed;
    let (rg, f_oo, f_vv) = production_ragged_op("alkane_4.xyz", 4, 1e-4, Operator::coulomb(), 0.7);
    let oracle = |t: &[ndarray::Array2<f64>]| {
        let mut fl = 0u64;
        let at = matvec_indexed(&rg, &f_oo, t, &mut fl);
        let mut r2 = 0.0;
        for (pb, a) in rg.pairs.iter().zip(&at) {
            let n = pb.db.len();
            for r in 0..pb.da.len() {
                for c in 0..n {
                    if pb.pat[r * n + c] {
                        r2 += (pb.j_blk[(r, c)] + a[(r, c)]).powi(2);
                    }
                }
            }
        }
        r2.sqrt()
    };
    let (bnorm, lam) = (
        pattern_rhs_norm(&rg),
        operator_lambda_min_lower_bound(&f_oo, &f_vv).unwrap(),
    );
    let p = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &f_vv,
        CgRtolPolicy::from_config(None),
        400,
        usize::MAX,
        0.0,
    )
    .unwrap();
    let true_r = oracle(&p.budgeted.solve.t) / bnorm;
    assert!(
        (p.true_relres / true_r - 1.0).abs() < 1e-13,
        "{} vs {true_r}",
        p.true_relres
    );
    let want = cg_energy_error_bound(bnorm, true_r, lam);
    // tight enough to tell it from the recursive residual, ~1e-12 away here
    assert!(
        (p.bound / want - 1.0).abs() < 1e-13,
        "{} vs {want}",
        p.bound
    );
    assert_ne!(p.true_relres.to_bits(), p.budgeted.solve.relres.to_bits());

    // a t the recursion knows nothing about: the helper must see its residual
    let t: Vec<_> = p.budgeted.solve.t.iter().map(|b| b * 1.01).collect();
    let (_, r) = hylleraas_energy_and_residual(&rg, &f_oo, None, &t).unwrap();
    assert!((r / oracle(&t) - 1.0).abs() < 1e-12);
    assert!(r / bnorm > 1e-3, "perturbation not visible: {}", r / bnorm);

    // a target set exactly at the recursive bound does not tighten, and the
    // verdict follows the true residual
    let target = cg_energy_error_bound(bnorm, p.budgeted.solve.relres, lam);
    let q = solve_under_rtol_policy(
        &rg,
        &f_oo,
        &f_vv,
        adaptive(1e-6, 1e-11, target),
        400,
        usize::MAX,
        0.0,
    )
    .unwrap();
    eprintln!(
        "TRUEVERDICT recursive-bound target {target:.6e} true bound {:.6e} met {:?}",
        q.bound, q.outcome.target_met
    );
    assert!(!q.outcome.tightened);
    assert_eq!(q.outcome.target_met, Some(q.bound <= target));
}

/// Pins both constants of `3·‖PJ‖²·relres²/λ_min` on a small system whose
/// operator is built densely. λ_min: the true smallest eigenvalue is never
/// below the lower bound, and on the full pattern equals it, so a wrong factor
/// of 2 fails either way. The 3: a perturbation along the lowest swap-ANTISYMMETRIC
/// eigenvector, where 2 − S = 3, gives an actual energy error above what a bound
/// without the 3 would allow. Mutants seen to fail: the factor 3 replaced by 1;
/// the gap's factor 2 replaced by 1 and by 4.
#[test]
fn bound_constants_are_pinned_on_a_dense_system() {
    use ferric_mp2::lmp2_amplitude::{
        cg_energy_error_bound, hylleraas_energy_and_residual, operator_lambda_min_lower_bound,
        pattern_rhs_norm,
    };
    use ferric_mp2::ragged::{matvec_indexed, solve_ragged_budgeted_with};
    use ndarray::{Array1, Array2};
    use ndarray_linalg::{Eigh, UPLO};
    for (eps, full) in [(0.0, true), (1e-4, false)] {
        let (rg, f_oo, f_vv) = production_ragged_op("water.xyz", 1, eps, Operator::coulomb(), 0.7);
        let idx: Vec<(usize, usize, usize)> = rg
            .pairs
            .iter()
            .enumerate()
            .flat_map(|(p, pb)| {
                let n = pb.db.len();
                (0..pb.da.len()).flat_map(move |r| (0..n).map(move |c| (p, r, c)))
            })
            .filter(|&(p, r, c)| rg.pairs[p].pat[r * rg.pairs[p].db.len() + c])
            .collect();
        let dim = idx.len();
        let zeros: Vec<Array2<f64>> = rg
            .pairs
            .iter()
            .map(|pb| Array2::zeros((pb.da.len(), pb.db.len())))
            .collect();
        let mut a = Array2::<f64>::zeros((dim, dim));
        for (col, &(p, r, c)) in idx.iter().enumerate() {
            let mut e = zeros.clone();
            e[p][(r, c)] = 1.0;
            let mut fl = 0u64;
            let ae = matvec_indexed(&rg, &f_oo, &e, &mut fl);
            for (row, &(q, s, d)) in idx.iter().enumerate() {
                a[(row, col)] = ae[q][(s, d)];
            }
        }
        let asym = (&a - &a.t()).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(asym < 1e-10, "A not symmetric: {asym:.2e}");
        let (w, _) = a.eigh(UPLO::Lower).unwrap();
        let lb = operator_lambda_min_lower_bound(&f_oo, &f_vv).unwrap();
        eprintln!(
            "DENSE eps {eps:e} dim {dim}: lambda_min {:.10} lower bound {lb:.10}",
            w[0]
        );
        assert!(w[0] >= lb - 1e-10, "lambda_min {} below {lb}", w[0]);
        if !full {
            continue;
        }
        assert_eq!(dim, rg.pairs.iter().map(|p| p.pat.len()).sum::<usize>());
        assert!((w[0] - lb).abs() < 1e-9, "full pattern: {} vs {lb}", w[0]);

        // lowest eigenvector with S = -1: lift the S = +1 sector out of the way
        let pos: std::collections::HashMap<_, _> =
            idx.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let swap: Vec<usize> = idx.iter().map(|&(p, r, c)| pos[&(p, c, r)]).collect();
        let mut lifted = a.clone();
        for (i, &si) in swap.iter().enumerate() {
            lifted[(i, i)] += 50.0;
            lifted[(i, si)] += 50.0;
        }
        let (wl, vl) = lifted.eigh(UPLO::Lower).unwrap();
        let lam_anti = wl[0];
        let v: Array1<f64> = vl.column(0).to_owned();
        let sv = swap.iter().map(|&s| v[s]).collect::<Array1<f64>>();
        assert!((&sv + &v).iter().all(|x| x.abs() < 1e-8), "not S = -1");
        assert!(
            lam_anti < 3.0 * lb,
            "a bound without the 3 would still hold"
        );

        let star = solve_ragged_budgeted_with(&rg, &f_oo, 1e-13, 400, usize::MAX, 0.0);
        let (e_star, _) =
            hylleraas_energy_and_residual(&rg, &f_oo, star.plan(), &star.solve.t).unwrap();
        let c = 1e-3;
        let mut t = star.solve.t.clone();
        for (i, &(p, r, d)) in idx.iter().enumerate() {
            t[p][(r, d)] += c * v[i];
        }
        let (e, r) = hylleraas_energy_and_residual(&rg, &f_oo, None, &t).unwrap();
        let de = e - e_star;
        let bnorm = pattern_rhs_norm(&rg);
        let bound = cg_energy_error_bound(bnorm, r / bnorm, lb);
        let exact = 3.0 * c * c * lam_anti;
        eprintln!(
            "DENSE lam_anti {lam_anti:.8} dE {de:.6e} exact {exact:.6e} bound {bound:.6e} ratio {:.3}",
            de / bound
        );
        assert!((de / exact - 1.0).abs() < 1e-6, "{de:.6e} vs {exact:.6e}");
        assert!(de <= bound, "dE {de:.3e} > bound {bound:.3e}");
    }
}
