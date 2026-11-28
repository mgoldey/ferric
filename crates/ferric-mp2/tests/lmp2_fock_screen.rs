//! Exactness anchors for the Fock-propagated screen of the amplitude-threshold
//! LMP2 mask (`AmplitudeLmp2Config::fock_screen`, theory.md §12.6).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): the trivial limits come
//! first — screen ON at ε = 0 is canonical RI-MP2 on both assembly paths, and
//! a VACUOUS screen (κ → 0, keep every candidate) at finite ε is canonical
//! RI-MP2 on the integral-direct path with inert maps. The ragged-level
//! bitwise no-op of `fock_screen = None` is pinned in `ragged.rs`
//! (`screened_blocks_none_is_bitwise_plain_and_some_is_a_superset`).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config, CgEnergyTarget};
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
        &RhfConfig::default().with_energy_conv(1e-10),
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

/// Every map inert (the direct path's trivial limit, tests/lmp2_direct.rs).
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

#[test]
fn screen_on_at_eps_zero_matches_canonical_on_both_paths() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 0.0,
        frozen_core: 1,
        fock_screen: Some(1.0),
        // 1e-9 against canonical RI-MP2: below what the 1e-7 Ha default target guarantees
        cg_energy_target: CgEnergyTarget::Strict,
        ..Default::default()
    };
    let op = Operator::coulomb();
    let g = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let (d, _) = amplitude_lmp2_direct(
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
    for (name, r) in [("global", &g), ("direct", &d)] {
        let de = (r.e_corr - r.e_corr_canonical_ri).abs();
        eprintln!("ANCHOR screen-on eps=0 {name}: dE={de:.3e}");
        assert!(r.keep_fraction == 1.0, "{name}: eps=0 must keep everything");
        assert!(
            de < 1e-9,
            "{name}: eps=0 with the screen on FAILED: {de:.3e}"
        );
    }
}

/// A vacuous screen (κ = 1e-300: every dressed integral clears κε) keeps the
/// whole candidate square of every pair; with inert maps the candidates are
/// all virtuals, so a finite-ε run IS canonical RI-MP2. Mutations seen to
/// fail (2026-11-16/17): passing `None` instead of `ctx.fock_screen` in the
/// direct path's `mask_pair` (keep < 1, the assert below), and retaining the
/// gate values instead of `g` in `pair_blocks_screened` (the run errors).
/// The ε = 0 anchor above cannot see either: the gate is not evaluated at
/// ε = 0, so both are equivalent mutants there.
#[test]
fn vacuous_screen_at_finite_eps_matches_canonical() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 1e-3,
        frozen_core: 1,
        fock_screen: Some(1e-300),
        // 1e-9 against canonical RI-MP2: below what the 1e-7 Ha default target guarantees
        cg_energy_target: CgEnergyTarget::Strict,
        ..Default::default()
    };
    let (d, _) = amplitude_lmp2_direct(
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
    let de = (d.e_corr - d.e_corr_canonical_ri).abs();
    eprintln!(
        "ANCHOR vacuous screen eps=1e-3: keep={} dE={de:.3e}",
        d.keep_fraction
    );
    assert!(
        d.keep_fraction == 1.0,
        "vacuous screen must keep every element"
    );
    assert!(de < 1e-9, "vacuous screen FAILED: {de:.3e}");
}

/// The retained `(i, j, a, b)` elements of a ragged pattern, sorted.
fn retained(rg: &ferric_mp2::ragged::Ragged) -> Vec<(usize, usize, usize, usize)> {
    let mut v = Vec::new();
    for pb in &rg.pairs {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    v.push((pb.i, pb.j, a, b));
                }
            }
        }
    }
    v.sort_unstable();
    v
}

/// Converged Hylleraas energy of a fixed pattern.
fn pattern_energy(rg: &ferric_mp2::ragged::Ragged, f_oo: &ndarray::Array2<f64>) -> f64 {
    let (t, _, _, conv, _) = ferric_mp2::ragged::solve_ragged(rg, f_oo, 1e-11, 2000);
    assert!(conv, "reference ragged solve did not converge");
    ferric_mp2::lmp2_amplitude::hylleraas_energy(rg, f_oo, None, &t).unwrap()
}

/// CANDIDATE-COVERAGE ANCHOR for the Fock screen on the in-core path: the
/// Schwarz virtual pre-filter of `assemble_ragged_direct_aux` must not change
/// what the Fock-dressed gate keeps. Reference: every pair's FULL virtual
/// square (no pre-filter), gated by `pair_blocks_screened` with the same κ —
/// an independent construction from the public pieces. The kept pattern must
/// be identical and the energies equal to solver precision. Before the
/// Fock-aware bound (2026-11-19) the pre-filter was the plain Eq-8 Schwarz
/// bound, which drops virtuals whose bare integral is small but whose
/// dressed value clears κε, and restricted the dressed sum to the filtered
/// virtuals; this test failed there (n-octane/6-31G, κ = 1: 2 extra
/// elements at ε = 1e-3, 4 lost at 3e-3, 2 lost at 1e-2, ΔE up to 3.8e-5 Ha).
/// The fix has two parts, and on this molecule either alone is enough: the
/// plain candidate rows with full-space Fock sums pass, and so do the
/// Fock-aware candidates with candidate-only sums (both mutants seen to pass
/// here, 2026-11-19). Each part is pinned on its own in `ragged.rs`:
/// `fock_bound_dominates_the_dressed_gate` / `fock_bound_keeps_the_review_toy`
/// fail with the bound reduced to the plain filter, and
/// `rect_gate_is_the_full_square_gate_restricted` fails with the sums
/// restricted to the candidates.
#[test]
fn fock_screen_pattern_is_independent_of_the_schwarz_prefilter() {
    use ferric_mp2::lmp2_amplitude::{assemble_basis, assemble_ragged_direct_aux, build_vvhv};
    use ferric_mp2::ragged::{pair_blocks_screened, Ragged};
    use ferric_mp2::rimp2::metric_inverse_sqrt;
    let su = setup("alkane_8.xyz");
    let op = Operator::coulomb();
    let kappa = 1.0;
    for eps in [1e-3, 3e-3, 1e-2] {
        let cfg = AmplitudeLmp2Config {
            eps,
            frozen_core: 8,
            fock_screen: Some(kappa),
            ..Default::default()
        };
        let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
        let lb = assemble_basis(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, &cfg, &vvhv).unwrap();
        let (no, nv) = (lb.no, lb.nv);
        let fo: Vec<f64> = (0..no).map(|i| lb.f_oo[(i, i)]).collect();
        let fv: Vec<f64> = (0..nv).map(|a| lb.f_vv[(a, a)]).collect();
        let bt = metric_inverse_sqrt(&lb.v2c, op).unwrap().dot(&lb.b_flat);
        let all: Vec<usize> = (0..nv).collect();
        let mut blocks = Vec::new();
        for i in 0..no {
            for j in i..no {
                let bi = bt.slice(ndarray::s![.., i * nv..(i + 1) * nv]);
                let bj = bt.slice(ndarray::s![.., j * nv..(j + 1) * nv]);
                let g = bi.t().dot(&bj);
                blocks.extend(pair_blocks_screened(
                    i,
                    j,
                    &g,
                    &all,
                    nv,
                    &lb.f_vv,
                    &fo,
                    &fv,
                    eps,
                    Some(kappa),
                ));
            }
        }
        let full = Ragged::from_pairs(blocks);
        let (pre, _, _, _) = assemble_ragged_direct_aux(
            &su.mol,
            &su.dfbs,
            op,
            &lb,
            eps,
            1.0,
            None,
            None,
            None,
            Some(kappa),
        )
        .unwrap();
        let (rf, rp) = (retained(&full), retained(&pre));
        let lost = rf.iter().filter(|x| rp.binary_search(x).is_err()).count();
        let extra = rp.iter().filter(|x| rf.binary_search(x).is_err()).count();
        let (ef, ep) = (
            pattern_energy(&full, &lb.f_oo),
            pattern_energy(&pre, &lb.f_oo),
        );
        eprintln!(
            "eps={eps:.0e}: full keeps {}, pre-filtered keeps {}, lost {lost}, extra {extra}, \
             dE={:.3e}",
            rf.len(),
            rp.len(),
            (ef - ep).abs()
        );
        assert!(rf.len() > 1000, "test too small to be meaningful");
        assert_eq!(
            (lost, extra),
            (0, 0),
            "pre-filter changed the Fock-screen pattern"
        );
        assert!(
            (ef - ep).abs() < 1e-10,
            "energy moved: {:.3e}",
            (ef - ep).abs()
        );
    }
}

/// The integral-direct κ candidate screen bounds the bare integrals only, so
/// it is refused under the Fock screen at finite ε, and accepted with the
/// screen off.
#[test]
fn direct_virt_schwarz_kappa_is_refused_under_the_fock_screen() {
    let su = setup("water.xyz");
    let maps = DirectConfig {
        virt_schwarz_kappa: Some(1.0),
        ..trivial_maps()
    };
    let run = |fock: Option<f64>| {
        let cfg = AmplitudeLmp2Config {
            eps: 1e-4,
            frozen_core: 1,
            fock_screen: fock,
            ..Default::default()
        };
        amplitude_lmp2_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
            &maps,
        )
    };
    let e = run(Some(1.0)).unwrap_err().to_string();
    assert!(e.contains("fock_screen = false"), "{e}");
    assert!(run(None).is_ok());
}
