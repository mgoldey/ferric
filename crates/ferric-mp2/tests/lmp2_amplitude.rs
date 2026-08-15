//! Exactness anchors for the amplitude-threshold local MP2 (Rust Phase 2).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): the trivial-limit
//! anchor and its mutation test come FIRST; the finite-ε behavior checks
//! ride behind them. The anchor pair is canonical-orbital closed-form RI-MP2
//! vs localized-orbital ragged CG — independent constructions sharing only
//! the RI integrals, so the bar is CG tolerance, not the RI floor.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    amplitude_lmp2, amplitude_lmp2_with_virtuals, build_vvhv, check_vvhv, AmplitudeLmp2Config,
    VvHv,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::s;

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

#[test]
fn vvhv_construction_is_orthonormal_and_spans_the_virtual_space() {
    let su = setup("water.xyz");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nocc = (su.mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(&su.obs, &su.rhf, nocc, &vvhv.c_vloc);
    assert!(dev_orth < 1e-8, "orthonormality dev {dev_orth:.2e}");
    assert!(dev_span < 1e-8, "span dev {dev_span:.2e}");
    assert_eq!(vvhv.n_valence + vvhv.n_hard, su.obs.nbasis() - nocc);
}

#[test]
fn eps_zero_matches_canonical_ri_mp2() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let de = r.e_corr - r.e_corr_canonical_ri;
    eprintln!(
        "ANCHOR water/6-31G: E_corr={:.10} canonical={:.10} dE={de:+.3e} (cg {} iters)",
        r.e_corr, r.e_corr_canonical_ri, r.cg_iterations
    );
    assert!(r.cg_converged);
    assert!(r.keep_fraction == 1.0 && r.pair_fraction == 1.0);
    assert!(de.abs() < 1e-9, "eps=0 anchor FAILED: dE={de:+.3e}");
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION: breaking the virtual
/// space (dropping one hard virtual) must break the anchor. Bar matches the
/// Python rig's measured mutation scale (span loss moves E by >1e-6).
#[test]
fn mutated_virtual_space_fails_the_anchor() {
    let su = setup("water.xyz");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nvir = vvhv.c_vloc.ncols();
    let broken = VvHv {
        c_vloc: vvhv.c_vloc.slice(s![.., ..nvir - 1]).to_owned(),
        n_valence: vvhv.n_valence,
        n_hard: vvhv.n_hard - 1,
    };
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_lmp2_with_virtuals(
        &su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &broken,
    )
    .unwrap();
    let de = (r.e_corr - r.e_corr_canonical_ri).abs();
    eprintln!("MUTATION water/6-31G: |dE|={de:.3e} (must exceed 1e-6)");
    assert!(de > 1e-6, "mutation NOT detected: |dE|={de:.3e}");
}

/// Finite-ε behavior on C4: error one-sided (under-correlation), counters
/// live, CG converged, and the keep fraction in the same band the Python
/// rig measured (0.0785 at ε=1e-3 with PySCF's auto aux; the aux here is
/// cc-pvdz-ri so J magnitudes — and hence the mask — differ slightly).
#[test]
fn eps_sweep_on_c4_is_one_sided_with_live_counters() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    eprintln!(
        "C4 eps=1e-3: dE={:+.3e} keep={:.4} pairs={:.3} dom(mean/max)={:.1}/{} cg={} raggedx={}",
        r.e_corr - r.e_corr_canonical_ri,
        r.keep_fraction,
        r.pair_fraction,
        r.dom_mean,
        r.dom_max,
        r.cg_iterations,
        r.dense_flops_per_matvec / r.ragged_flops_per_matvec.max(1),
    );
    assert!(r.cg_converged);
    let de = r.e_corr - r.e_corr_canonical_ri;
    assert!(de > 0.0, "threshold error must be one-sided (under-correlation), got {de:+.3e}");
    assert!(de < 5e-2, "eps=1e-3 error implausibly large: {de:+.3e}");
    assert!(
        r.keep_fraction > 0.03 && r.keep_fraction < 0.20,
        "keep fraction {:.4} outside the Python-measured band (0.0785 ±aux)",
        r.keep_fraction
    );
    // C4 is BELOW the locality onset, so dom_max may touch the full virtual
    // space (Python measured the same); the mask biting shows in the MEAN.
    let nv = su.obs.nbasis() - (su.mol.nelec() as usize) / 2;
    assert!(r.dom_mean < nv as f64, "dom mean {:.1} not below nv={nv}", r.dom_mean);
}

/// Operator threading + frozen_core=0 edge: the ε=0 anchor must also hold
/// for an ATTENUATED operator (erfc) with nothing frozen — same independent
/// construction, different kernel, so an op-threading bug on either side
/// breaks it.
#[test]
fn eps_zero_anchor_holds_for_erfc_with_no_frozen_core() {
    let su = setup("water.xyz");
    let op = Operator::erfc(1.0); // omega in Bohr^-1
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &su.obs).unwrap();
    let _ = &bounds; // RHF reference is Coulomb-SCF (attenuated MP2 convention)
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 0, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let de = r.e_corr - r.e_corr_canonical_ri;
    eprintln!(
        "ANCHOR water/erfc(1.0)/fc=0: E_corr={:.10} canonical={:.10} dE={de:+.3e}",
        r.e_corr, r.e_corr_canonical_ri
    );
    assert!(r.e_corr < 0.0, "SR correlation must be negative");
    assert!(de.abs() < 1e-9, "erfc eps=0 anchor FAILED: dE={de:+.3e}");
}

/// Independent-algebra cross-check of the FINITE-ε masked solve: a naive
/// dense masked PCG written here in the test (flat (no·nv)² matrices,
/// explicit loops, no shared solver code) must agree with the ragged
/// solver's energy to 1e-10 on the same assembled problem — and the
/// comparison is proven non-vacuous by perturbing the naive path's Fvv,
/// which must break it.
#[test]
fn ragged_masked_solve_matches_naive_dense_reference() {
    use ferric_mp2::lmp2_amplitude::assemble_localized;
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config { eps: 1e-4, frozen_core: 1, ..Default::default() };
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let lp = assemble_localized(&su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &vvhv)
        .unwrap();
    let r = amplitude_lmp2_with_virtuals(
        &su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &vvhv,
    )
    .unwrap();

    let e_naive = naive_masked_mp2(&lp.j_dense, &lp.f_oo, &lp.f_vv, lp.no, lp.nv, cfg.eps, 0.0);
    let dx = (r.e_corr - e_naive).abs();
    eprintln!("XCHECK ragged vs naive-dense (eps=1e-4): |dE|={dx:.3e}");
    assert!(dx < 1e-10, "ragged/naive cross-check FAILED: |dE|={dx:.3e}");

    // non-vacuousness: corrupt the naive path's Fvv; it must now disagree.
    // Sized per the measured quadratic insensitivity of the Hylleraas energy
    // (Python rig: 1e-3 moved E by only 1.8e-7).
    let e_broken = naive_masked_mp2(&lp.j_dense, &lp.f_oo, &lp.f_vv, lp.no, lp.nv, cfg.eps, 5e-2);
    let dxb = (r.e_corr - e_broken).abs();
    eprintln!("XCHECK mutation: |dE|={dxb:.3e} (must exceed 1e-10)");
    assert!(dxb > 1e-10, "xcheck comparison is vacuous: mutation not detected");
}

/// Naive dense masked preconditioned CG — deliberately simple flat-matrix
/// loops, sharing NOTHING with the ragged solver. `fvv_bump` perturbs
/// Fvv[0,1]/[1,0] for the non-vacuousness arm.
fn naive_masked_mp2(
    j: &ndarray::Array2<f64>,
    f_oo: &ndarray::Array2<f64>,
    fvv: &ndarray::Array2<f64>,
    no: usize,
    nv: usize,
    eps: f64,
    fvv_bump: f64,
) -> f64 {
    let n = no * nv;
    let mut fvv = fvv.clone();
    fvv[(0, 1)] += fvv_bump;
    fvv[(1, 0)] += fvv_bump;
    let idx = |i: usize, a: usize| i * nv + a;
    let mut mask = vec![false; n * n];
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    let jd = j[(idx(i, a), idx(jj, b))].abs();
                    let kd = j[(idx(i, b), idx(jj, a))].abs();
                    if eps == 0.0 || jd > eps || kd > eps {
                        mask[idx(i, a) * n + idx(jj, b)] = true;
                    }
                }
            }
        }
    }
    let aop = |t: &Vec<f64>| -> Vec<f64> {
        let mut r = vec![0.0; n * n];
        for i in 0..no {
            for a in 0..nv {
                for jj in 0..no {
                    for b in 0..nv {
                        let row = idx(i, a) * n + idx(jj, b);
                        if !mask[row] {
                            continue;
                        }
                        let mut v = 0.0;
                        for c in 0..nv {
                            v += fvv[(a, c)] * t[idx(i, c) * n + idx(jj, b)];
                            v += t[idx(i, a) * n + idx(jj, c)] * fvv[(c, b)];
                        }
                        for k in 0..no {
                            v -= f_oo[(i, k)] * t[idx(k, a) * n + idx(jj, b)];
                            v -= t[idx(i, a) * n + idx(k, b)] * f_oo[(k, jj)];
                        }
                        r[row] = v;
                    }
                }
            }
        }
        r
    };
    let mut d = vec![0.0; n * n];
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    d[idx(i, a) * n + idx(jj, b)] =
                        fvv[(a, a)] + fvv[(b, b)] - f_oo[(i, i)] - f_oo[(jj, jj)];
                }
            }
        }
    }
    let mut rhs = vec![0.0; n * n];
    for r_ in 0..n {
        for c_ in 0..n {
            if mask[r_ * n + c_] {
                rhs[r_ * n + c_] = -j[(r_, c_)];
            }
        }
    }
    let dot = |x: &Vec<f64>, y: &Vec<f64>| -> f64 { x.iter().zip(y).map(|(a, b)| a * b).sum() };
    let bnorm = dot(&rhs, &rhs).sqrt();
    let mut t = vec![0.0; n * n];
    let mut r = rhs.clone();
    let mut z: Vec<f64> = r.iter().zip(&d).map(|(a, b)| a / b).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..400 {
        let ap = aop(&p);
        let alpha = rz / dot(&p, &ap);
        for k in 0..n * n {
            t[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        if dot(&r, &r).sqrt() / bnorm < 1e-11 {
            break;
        }
        z = r.iter().zip(&d).map(|(a, b)| a / b).collect();
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for k in 0..n * n {
            p[k] = z[k] + beta * p[k];
        }
        rz = rz_new;
    }
    // E = sum (2 t_iajb - t_ibja) J_iajb
    let mut e = 0.0;
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    let jv = j[(idx(i, a), idx(jj, b))];
                    e += (2.0 * t[idx(i, a) * n + idx(jj, b)] - t[idx(i, b) * n + idx(jj, a)]) * jv;
                }
            }
        }
    }
    e
}

/// C8 counter cross-check vs the Python rig (loose band — different aux
/// basis, zero shared code): eps=1e-3 keep 0.0133, dom max 65, cg 17.
#[test]
fn c8_counters_land_in_the_python_measured_band() {
    let su = setup("alkane_8.xyz");
    let cfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 8, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    eprintln!(
        "C8 eps=1e-3: dE={:+.3e} keep={:.4} pairs={:.3} dom(mean/max)={:.1}/{} cg={} raggedx={}",
        r.e_corr - r.e_corr_canonical_ri,
        r.keep_fraction,
        r.pair_fraction,
        r.dom_mean,
        r.dom_max,
        r.cg_iterations,
        r.dense_flops_per_matvec / r.ragged_flops_per_matvec.max(1),
    );
    assert!(r.cg_converged);
    let de = r.e_corr - r.e_corr_canonical_ri;
    assert!(de > 0.0 && de < 6e-2, "C8 eps=1e-3 dE out of band: {de:+.3e}");
    assert!(
        r.keep_fraction > 0.008 && r.keep_fraction < 0.022,
        "keep {:.4} outside Python band (0.0133 ±aux)",
        r.keep_fraction
    );
    assert!(
        r.dom_max >= 55 && r.dom_max <= 75,
        "dom max {} far from Python's 65",
        r.dom_max
    );
    // the ragged work advantage must be substantial at this retention
    assert!(
        r.dense_flops_per_matvec / r.ragged_flops_per_matvec.max(1) > 20,
        "ragged advantage collapsed"
    );
}

/// TRIVIAL-LIMIT ANCHOR for the per-pair domain-local fit: an infinite
/// fit radius must reproduce the global-metric fit — algebraically
/// A^T V^-1 A vs (V^-1/2 A)^T (V^-1/2 A), two different factorizations,
/// so agreement is a real check, not an identity.
#[test]
fn domain_fit_trivial_limit_matches_global() {
    let su = setup("water.xyz");
    let base = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let glob = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &base)
        .unwrap();
    let cfg = AmplitudeLmp2Config { fit_radius_bohr: Some(1e6), ..base };
    let dom = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let dd = (dom.e_corr - glob.e_corr).abs();
    eprintln!("TRIVIAL LIMIT: |E(domain,inf) - E(global)| = {dd:.3e}");
    assert!(dd < 1e-10, "trivial-limit anchor FAILED: {dd:.3e}");
    // and the eps=0 canonical anchor must still hold through the domain path
    let de = (dom.e_corr - dom.e_corr_canonical_ri).abs();
    assert!(de < 1e-9, "canonical anchor through domain path FAILED: {de:.3e}");
}

/// Finite radius: µHa-class truncation, strictly nonzero (proves the domain
/// path actually truncates — the non-vacuousness arm of the trivial limit).
#[test]
fn domain_fit_finite_radius_truncation_is_microhartree_class() {
    let su = setup("alkane_4.xyz");
    let base = AmplitudeLmp2Config { eps: 0.0, frozen_core: 4, ..Default::default() };
    let glob = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &base)
        .unwrap();
    let cfg = AmplitudeLmp2Config { fit_radius_bohr: Some(8.0), ..base };
    let dom = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let dd = (dom.e_corr - glob.e_corr).abs();
    eprintln!("C4 r=8 Bohr domain truncation: |dE| = {dd:.3e} Ha");
    assert!(dd > 1e-12, "domain path identical to global at r=8 — truncation vacuous?");
    assert!(dd < 5e-5, "domain truncation not µHa-class: {dd:.3e}");
}

/// Integral-free pair gate on the attenuated operator: drops a substantial
/// pair fraction at bounded energy cost; a deliberately absurd calibration
/// (mutation arm) must gate nearly everything and wreck the energy —
/// proving the gate genuinely removes blocks from the solve.
#[test]
fn pair_gate_drops_pairs_with_bounded_cost_and_mutates_loudly() {
    let su = setup("alkane_8.xyz");
    let op = Operator::erfc(1.0);
    let base = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 8, ..Default::default() };
    let ungated = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &base).unwrap();
    let cfg = AmplitudeLmp2Config { pair_gate_cal: Some(0.02), ..base.clone() };
    let gated = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let cost = (gated.e_corr - ungated.e_corr).abs();
    eprintln!(
        "C8/erfc gate: dropped {} unique pairs (of {}), |dE(gate)| = {cost:.3e} Ha",
        gated.n_pairs_gated,
        25 * 24 / 2
    );
    assert!(gated.n_pairs_gated > 50, "gate dropped only {} pairs", gated.n_pairs_gated);
    assert!(cost < 1e-3, "gate cost too large: {cost:.3e}");
    // mutation: absurd calibration must gate ~everything and move E a lot
    let broken = AmplitudeLmp2Config { pair_gate_cal: Some(1e-12), ..base };
    let wrecked = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &broken).unwrap();
    let dwreck = (wrecked.e_corr - ungated.e_corr).abs();
    eprintln!(
        "gate MUTATION (cal=1e-12): dropped {} pairs, |dE| = {dwreck:.3e}",
        wrecked.n_pairs_gated
    );
    assert!(
        wrecked.n_pairs_gated > gated.n_pairs_gated && dwreck > 1e-3,
        "gate mutation not loud: {} pairs, |dE|={dwreck:.3e}",
        wrecked.n_pairs_gated
    );
}

/// Wall-clock stage table over the alkane series vs canonical ri_mp2.
/// #[ignore]d: run explicitly, RELEASE build, quiet box:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
///     --test lmp2_amplitude -- --ignored --nocapture bench_wall_clock
///
/// HONEST FRAMING (do not quote as a scaling win): J assembly here is
/// dense-from-RI, so total LMP2 cost is expected to EXCEED ri_mp2 at these
/// sizes — the meaningful numbers are the stage breakdown (does the solve
/// stage stay cheap as N grows, as the FLOP counters promised?) and the
/// counter parity with the Python rig. End-to-end wins require the
/// integral-direct assembly this module does not yet have.
#[test]
#[ignore]
fn bench_wall_clock_alkane_series() {
    use std::time::Instant;
    println!("mol      eps    op      E_corr        dE_vs_can  keep    dom(max)  t_asm(s) t_solve(s) t_ref(s) raggedx gated");
    for xyz in ["alkane_4.xyz", "alkane_8.xyz", "alkane_12.xyz"] {
        let su = setup(xyz);
        let nc = su.mol.atoms.iter().filter(|a| a.z == 6).count();
        for (opname, op, cal) in
            [("coul", Operator::coulomb(), 0.7), ("erfc1", Operator::erfc(1.0), 0.02)]
        {
            for eps in [1e-3, 1e-4] {
                let cfg = AmplitudeLmp2Config {
                    eps,
                    frozen_core: nc,
                    pair_gate_cal: Some(cal),
                    fit_radius_bohr: Some(10.0),
                    ..Default::default()
                };
                let t0 = Instant::now();
                let r = amplitude_lmp2(
                    &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg,
                )
                .unwrap();
                let _total = t0.elapsed().as_secs_f64();
                println!(
                    "{:8} {:6.0e} {:6} {:.8} {:+.3e} {:.4} {:4}/{:<4} {:8.2} {:9.2} {:7.2} {:6} {:5}",
                    xyz.trim_end_matches(".xyz"),
                    eps,
                    opname,
                    r.e_corr,
                    r.e_corr - r.e_corr_canonical_ri,
                    r.keep_fraction,
                    r.dom_max,
                    r.n_valence_virt + r.n_hard_virt,
                    r.timings.t_assembly_s,
                    r.timings.t_solve_s,
                    r.timings.t_reference_s,
                    r.dense_flops_per_matvec / r.ragged_flops_per_matvec.max(1),
                    r.n_pairs_gated,
                );
            }
        }
    }
}
