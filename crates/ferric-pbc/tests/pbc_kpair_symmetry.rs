//! k-point orbital-pair symmetry (s2) of the short-range integral stages
//! (`rsgdf::kpoint` module doc "Orbital-pair symmetry at k",
//! `hcore::kpoint` module doc "Orbital-pair symmetry"). The Gamma twin is
//! `pbc_pair_symmetry.rs`.
//!
//! # The relations under test (derived, not measured)
//!
//! * RS-GDF SR residue bins. `B[r_L, r_T][μν, P] = Σ_{L ≡ r_L} Σ_{T ≡ r_T}
//!   (μ_0 ν_L | P_T)`. With `(μ_0 ν_L | P_T) = (ν_0 μ_{−L} | P_{T−L})`:
//!   `B[b][μν] = B[M(b)][νμ]`, `M(r_L, r_T) = (−r_L mod mod_l,
//!   (r_T − r_L) mod mod_t)` (an involution; `mod_t | mod_l`). The s2 walk
//!   computes each unordered shell pair once per residue and writes the
//!   transposed half into bin `M(b)`. Diagonal pairs: canonical residues
//!   `r_L ≤ r_{−L}` only; at a self-conjugate residue the lexicographically
//!   smaller element of each `M`-orbit is written, directly and mirrored.
//!   `common::kpair_mirror` / `kpair_s2_expected` re-derive both rules here,
//!   independently of the library, and the library's own map
//!   (`kpair_mirror_bin`) is checked against them.
//! * hcore `S(k)`, `T(k)`, `V_SR(k)`: `X(k)[νμ] = conj X(k)[μν]` (the phase
//!   pairing `e^{±ik·L}` IS the conjugation; no bins).
//!
//! # What each stage must satisfy against the FROZEN ordered oracles
//!
//! (`sr3_kbins_s2_and_s1`'s ordered branch = `KRsGdf::build_pair_s1_oracle`'s
//! walk; `*_kpts_parallel_and_serial`'s serial loops =
//! `periodic_hcore_kpts_pair_s1_oracle`):
//!
//! 1. EXACT `M`-symmetry / off-diagonal Hermiticity, bitwise.
//! 2. Every element bitwise one of the oracle's two ordered evaluations
//!    `x = s1[e]`, `y = s1[M(e)]` (hcore: `y = conj s1[(ν, μ)]`); unsplit and
//!    hcore: EXACTLY the prescribed one.
//! 3. Bitwise across 1/2/6 threads.
//! 4. `|new − avg| ≤ ½|x − y| + ε|avg|`, `avg = fl(½ fl(x + y))`
//!    (derivation of `pbc_pair_symmetry.rs` item 4: `new ∈ {x, y}` and
//!    `avg = ½(x + y)(1 + δ)`, `|δ| ≤ ε/2` per real component, so
//!    `|new − avg| ≤ ½|x − y| + ½ε|x + y|`; the complex norm obeys the same
//!    bound). Scale: the ordered walk's own round-off asymmetry (printed).
//! 5. k-mesh RHF / UHF / range-split RHF energies within `E_BAR` = 1e-11 Ha of
//!    the frozen pre-s2 build (the task bar; the expected shift is
//!    first-order in a ~1e-16 relative change of J3 / h, i.e. ~1e-15 Ha),
//!    on a TRIM mesh (1×1×2) and a non-TRIM one (1×1×3).
//! 6. Counters: the ordered-equivalent count equals the ordered oracle's
//!    count (unsplit and hcore).
//! 7. The RELATION itself: the ordered walk's `x` and `y` agree to
//!    `MAP_BAR` (items 1, 2 and 4 hold for ANY map the library and this file
//!    share, right or wrong — `new ∈ {x, y}` makes item 4 automatic; only
//!    `x ≈ y` says the map pairs equal lattice sums).
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! A correct s2 walk: 0 symmetry violations, 0 elements outside `{x, y}`,
//! `max|new − avg|` ~1e-17 absolute. A broken one is O(|J3|) ~ 1e-2..1e-1
//! off somewhere: a dropped transposed write leaves rows `νμ` zero; a
//! skipped diagonal zeroes the diagonal blocks; a missing aux shift or a
//! wrong pair-image bin puts the right numbers in the wrong bin. Items 1, 2
//! and 4 fail by ~15 orders of magnitude, so the mutants below are asserted
//! against item 4 with a margin, not near the bound.
//!
//! # Mutants (in-library switches, `KPairSymMutant` / `KHcorePairMutant`)
//!
//! * `WrongPairBin` (transposed half into bin `r_L`, not `r_{−L}`) and hcore
//!   `NoConj` (transposed element unconjugated): IDENTITIES on a mesh of TRIM
//!   points only. At Gamma-centred 1×1×2, `mod_l = 2`, so `−r ≡ r` for every
//!   residue (`M` moves only `r_T`), and every phase is exactly ±1, so every
//!   `S(k)`, `V(k)` element is real (conj flips only the sign of a zero).
//!   Asserted blind there (bitwise bins, equal values, equal energy) and
//!   caught on the non-TRIM 1×1×3 (`mod_l = 3`, `−1 ≡ 2`, phases
//!   `e^{±2πi/3}`), at the bins, the matrices and the KRHF energy.
//! * `NoAuxShift`, `DropTransposed`, `SkipDiagonal`, `DiagonalNoMirror`
//!   (bins) and `DropTransposed`, `SkipDiagonal` (hcore): caught on BOTH
//!   meshes (`NoAuxShift` too: at 1×1×2 `r_L = 1` shifts `r_T` by 1 mod 2).
//!
//! Not covered: meshes beyond 1×1×3 / MP 1×2×2, d shells, the k force /
//! stress walks (their own ordered loops, unchanged).
//!
//! # Per-commit canaries vs the weekly anchors
//!
//! The unsplit bins anchor on the 4 H fixture over three meshes (~6 min on
//! CI) and the bins-mutant sweep on two meshes (~4 min) run weekly in
//! `validation_pbc_kpair_symmetry.rs` (harness: `kpair_symmetry_support/`).
//! Per-commit canaries on the 2 H twin of the fixture (`tri2_sp`):
//! `krsgdf_s2_bins_canary` (items 1-4, 6 and the exact prescription on the
//! non-TRIM 1×1×3 and on MP 1×2×2, at 1/2 threads) and `krsgdf_s2_bin_mutants_canary` (every
//! bins mutant caught on 1×1×3, `WrongPairBin`'s TRIM blindness on 1×1×2).

mod common;
#[path = "kpair_symmetry_support/mod.rs"]
mod support;

use common::*;
use ferric_core::basis;
use ferric_pbc::hcore::kpoint::{
    hcore_kpts_s2_mutant, overlap_kinetic_kpts_parallel_and_serial, periodic_hcore_kpts,
    periodic_hcore_kpts_pair_s1_oracle, sr_attraction_kpts_parallel_and_serial, KHcorePairMutant,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::solve_krhf_injected;
use ferric_pbc::rsgdf::kpoint::{KPairSymMutant, KRsGdf, KRsGdfMutation};
use ferric_pbc::rsgdf::RangeSplit;
use support::*;

#[test]
fn krsgdf_s2_split_bins_take_either_orientation() {
    let (cell, prep, aux) = tri_sp();
    let mesh = KPointMesh::gamma_centred(&cell, NON_TRIM).unwrap();
    check_kbins(
        &cell,
        &prep,
        &aux,
        &mesh,
        Some(RangeSplit::default()),
        "split Gamma 1x1x3",
        &THREADS,
    );
}

#[test]
fn hcore_kpts_s2_is_the_ordered_loop_mirrored_and_conjugated() {
    let (cell, prep, _) = tri_sp();
    let cfg = hcore_cfg();
    for dims in [TRIM, NON_TRIM] {
        let mesh = KPointMesh::gamma_centred(&cell, dims).unwrap();
        let runs = hcore_parts(&cell, &prep, &mesh);
        let (s, t, v, nc, no) = &runs[0];
        for (&th, (s_, t_, v_, nc_, no_)) in THREADS.iter().zip(&runs) {
            for (a, b, what) in [(s_, s, "S"), (t_, t, "T"), (v_, v, "V_SR")] {
                assert_eq!(
                    cmats_bit_diffs(a, b),
                    0,
                    "{dims:?} {what} at {th} vs 1 thread"
                );
            }
            assert_eq!((nc_, no_), (nc, no), "{dims:?} counts at {th}");
        }
        let ([_, (s_old, t_old)], _) =
            overlap_kinetic_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("S/T");
        let [_, (v_old, n_old)] =
            sr_attraction_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("V_SR");
        for (new, old, what) in [(s, &s_old, "S"), (t, &t_old, "T"), (v, &v_old, "V_SR")] {
            let x: Vec<_> = old.iter().map(mirror_upper_herm).collect();
            assert_eq!(
                cmats_bit_diffs(new, &x),
                0,
                "{dims:?} {what}: s2 vs the ordered μ ≤ ν elements and conjugates"
            );
            let (asym, va) = herm_vs_ordered(new, old);
            eprintln!("{dims:?} {what}: {asym} non-Hermitian off-diagonal; {va:?}");
            assert_eq!(
                asym, 0,
                "{dims:?} {what}: not exactly Hermitian off the diagonal"
            );
            assert!(
                va.hmax <= MAP_BAR,
                "{dims:?} {what}: ordered x and conj y differ by {:.2e}",
                va.hmax
            );
            assert_eq!(va.neither, 0, "{dims:?} {what}: outside {{x, conj y}}");
            assert_eq!(va.over, 0, "{dims:?} {what}: outside the item-4 bound");
        }
        if dims == NON_TRIM {
            assert!(
                s.iter().any(|m| m.iter().any(|z| z.im != 0.0)),
                "{dims:?}: S(k) real everywhere, so the conjugation is vacuous"
            );
        }
        assert_eq!(
            *no, n_old,
            "{dims:?}: ordered-equivalent vs the ordered count"
        );
        assert!(nc < no, "{dims:?}: s2 computed {nc} of {no}");
    }
}

#[test]
fn hcore_kpts_s2_mutants_fail_and_no_conj_needs_a_non_trim_mesh() {
    let (cell, prep, _) = tri_sp();
    let cfg = hcore_cfg();
    for (dims, trim) in [(NON_TRIM, false), (TRIM, true)] {
        let mesh = KPointMesh::gamma_centred(&cell, dims).unwrap();
        let ([_, (s_old, _)], _) =
            overlap_kinetic_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("S/T");
        let [_, (v_old, _)] =
            sr_attraction_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("V_SR");
        let (s_prod, _, v_prod, _, _) =
            hcore_kpts_s2_mutant(&cell, &prep, &mesh, &cfg, None).expect("production");
        for m in HCORE_MUTANTS {
            let (s, _, v, _, _) =
                hcore_kpts_s2_mutant(&cell, &prep, &mesh, &cfg, Some(m)).expect("mutant");
            let (_, vs) = herm_vs_ordered(&s, &s_old);
            let (_, vv) = herm_vs_ordered(&v, &v_old);
            eprintln!("{dims:?} hcore mutant {m:?}: S {vs:?}; V_SR {vv:?}");
            if trim && m == KHcorePairMutant::NoConj {
                // Real phases: conj changes at most the sign of a zero.
                for (a, b, what) in [(&s, &s_prod, "S"), (&v, &v_prod, "V_SR")] {
                    let differ: usize = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| x.iter().zip(y.iter()).filter(|(p, q)| p != q).count())
                        .sum();
                    assert_eq!(differ, 0, "{dims:?}: NoConj must be an identity ({what})");
                }
                continue;
            }
            for (va, what) in [(&vs, "S"), (&vv, "V_SR")] {
                assert!(
                    va.over > 0 && va.dmax > MUTANT_ELEMENT_BAR,
                    "{dims:?}: hcore mutant {m:?} not caught in {what} ({:.2e})",
                    va.dmax
                );
            }
        }
    }
}

#[test]
fn krhf_energy_matches_the_frozen_pre_s2_build_on_trim_and_non_trim_meshes() {
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    for dims in [TRIM, NON_TRIM] {
        let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &aux, dims);
        energy_vs_pre_s2(&ks, None, None, &format!("H2 KRHF {dims:?}"));
    }
}

#[test]
fn kuhf_energy_matches_the_frozen_pre_s2_build() {
    let ks = ksys(
        cell_at(&H3_ATOMS, cubic(H3_A), 2),
        &pyscf_sto3g_h(),
        &et_sp_aux(),
        NON_TRIM,
    );
    energy_vs_pre_s2(&ks, None, Some((2, 1)), "H3 KUHF 1x1x3");
}

#[test]
fn split_krhf_energy_matches_the_frozen_pre_s2_build() {
    let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &et_sp_aux(), NON_TRIM);
    energy_vs_pre_s2(&ks, Some(RangeSplit::new(1.0)), None, "H2 split KRHF 1x1x3");
}

/// The wrong-pair-bin mutant through the whole k-point RHF: the SAME energy
/// bit for bit at 1×1×2 (its bins are bitwise production's), off by far more
/// than `E_BAR` at 1×1×3.
#[test]
fn wrong_pair_bin_mutant_moves_the_krhf_energy_only_on_a_non_trim_mesh() {
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let mutant = Some(KRsGdfMutation::PairSym(KPairSymMutant::WrongPairBin));
    for (dims, trim) in [(TRIM, true), (NON_TRIM, false)] {
        let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &aux, dims);
        let (hk, good) = build(&ks, &kgdf_cfg(None, None), false);
        let bad = KRsGdf::build(
            &ks.cell,
            &ks.prep,
            &ks.aux,
            &ks.mesh,
            &hk.s,
            &kgdf_cfg(None, mutant),
        )
        .expect("mutant KRsGdf");
        let e_good = energy(&ks, &hk, &good, None);
        // The mutant's SCF is not required to converge (its J3 is wrong at
        // k' ≠ −k'); a failed or unconverged SCF also counts as caught.
        let bad_run = solve_krhf_injected(
            &ks.cell,
            &ks.mesh,
            &kscf_cfg(),
            inj(&hk, Box::new(bad.jk_builder())),
        )
        .map(|r| (r.energy, r.converged));
        eprintln!("H2 KRHF {dims:?}: WrongPairBin run {bad_run:?} vs E {e_good:.15}");
        match (trim, bad_run) {
            (true, Ok((e_bad, conv))) => assert!(
                conv && bits_eq(e_bad, e_good),
                "{dims:?}: WrongPairBin must be an identity on a TRIM mesh"
            ),
            (true, Err(e)) => panic!("{dims:?}: WrongPairBin broke a TRIM-mesh SCF: {e}"),
            (false, Ok((e_bad, conv))) => assert!(
                !conv || (e_bad - e_good).abs() > MUTANT_E_BAR,
                "{dims:?}: WrongPairBin not caught (ΔE {:.3e})",
                e_bad - e_good
            ),
            (false, Err(_)) => {}
        }
    }
}

/// The k-point ENERGY builds implement the SR column rotation: an explicit
/// request is accepted (this STO-3G basis has nothing to rotate, so the
/// build is bitwise the default one; the numerical anchors are
/// `pbc_sr_rotation_k.rs`). The frozen s1 oracles walk the parent shells and
/// refuse an explicit request by name.
#[test]
fn kpoint_energy_builds_accept_the_column_rotation_and_oracles_refuse() {
    use ferric_pbc::sr_rotation::SrColumnRotation;
    let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &et_sp_aux(), TRIM);
    let rot = SrColumnRotation::on();
    assert_eq!(hcore_cfg().sr_column_rotation, SrColumnRotation::Auto);
    let hk0 = periodic_hcore_kpts(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg()).expect("hcore(k)");
    let hk1 = periodic_hcore_kpts(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg().with_sr_column_rotation(rot),
    )
    .expect("k hcore with an explicit rotation");
    assert_eq!(hk1.sr_rotated_columns, 0, "nothing to rotate");
    assert_eq!(cmats_bit_diffs(&hk1.h, &hk0.h), 0);
    let mut cfg = kgdf_cfg(None, None);
    assert_eq!(cfg.gdf.sr_column_rotation, SrColumnRotation::Auto);
    let g0 = KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk0.s, &cfg).expect("default");
    cfg.gdf.sr_column_rotation = rot;
    let g1 = KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk0.s, &cfg)
        .expect("KRsGdf with an explicit rotation");
    assert_eq!(g1.stats().sr_rotated_columns, 0, "nothing to rotate");
    for k in 0..g0.nk() {
        for kp in 0..g0.nk() {
            let (a, b) = (g0.block(k, kp), g1.block(k, kp));
            assert!(
                a.iter().zip(b.iter()).all(|(x, y)| cbits_eq(*x, *y)),
                "B({k}, {kp})"
            );
        }
    }
    let msg = KRsGdf::build_pair_s1_oracle(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk0.s, &cfg)
        .expect_err("the s1 oracle with a column rotation must be refused")
        .to_string();
    assert!(
        msg.contains("sr_column_rotation") && msg.contains("explicitly"),
        "{msg}"
    );
    let msg = periodic_hcore_kpts_pair_s1_oracle(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg().with_sr_column_rotation(rot),
    )
    .expect_err("the hcore s1 oracle with a column rotation must be refused")
    .to_string();
    assert!(
        msg.contains("sr_column_rotation") && msg.contains("explicitly"),
        "{msg}"
    );
}

// ------------------------------------------- canaries of the weekly anchors

/// Canary of `krsgdf_s2_bins_are_the_ordered_walk_under_the_minus_rl_map`
/// (weekly): the 2 H fixture, 1/2 threads, on the non-TRIM Gamma 1×1×3 mesh
/// and on MP 1×2×2, whose pair-image moduli (2N = [1, 4, 4]) differ from the
/// aux-image moduli (N), so a bin mix-up between the two dimensions fails.
#[test]
fn krsgdf_s2_bins_canary() {
    let (cell, prep, aux) = tri2_sp();
    for (tag, mesh) in [
        (
            "2 H Gamma 1x1x3",
            KPointMesh::gamma_centred(&cell, NON_TRIM),
        ),
        ("2 H MP 1x2x2", KPointMesh::monkhorst_pack(&cell, [1, 2, 2])),
    ] {
        check_kbins(&cell, &prep, &aux, &mesh.unwrap(), None, tag, &[1, 2]);
    }
}

/// Canary of `krsgdf_s2_bin_mutants_fail_and_wrong_pair_bin_needs_a_non_trim_mesh`
/// (weekly): the same mutant sweep on the 2 H fixture.
#[test]
fn krsgdf_s2_bin_mutants_canary() {
    let (cell, prep, aux) = tri2_sp();
    kbins_mutants_case(&cell, &prep, &aux, &[(NON_TRIM, false), (TRIM, true)]);
}
