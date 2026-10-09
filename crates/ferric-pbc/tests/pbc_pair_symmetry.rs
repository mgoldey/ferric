//! Gamma orbital-pair symmetry (s2) of the short-range integral stages
//! (`rsgdf` and `hcore` module docs, "Orbital-pair symmetry").
//!
//! Construction under test: at Gamma `(ν_0|O|μ_L) = (μ_0|O|ν_{−L})`, so each
//! UNORDERED shell pair `i1 ≤ i2` is evaluated once over every pair image
//! (the ordered loop's own per-pair body and per-element addend sequence) and
//! ONE task writes the block into both `(μ, ν)` and `(ν, μ)`. Stages: RS-GDF
//! SR 3-centre `J3` (unsplit and range split) and hcore `S`, `T`, `V_SR`. The
//! SR metric `J2` and every k-point / derivative walk keep the ordered loops.
//!
//! What each stage must satisfy, against the FROZEN ordered oracles
//! (`RsGdf::build_pair_s1_oracle`, `periodic_hcore_pair_s1_oracle`, and the
//! serial ordered loops behind `*_parallel_and_serial` / the ordered branch
//! of `sr3_gamma_s2_and_s1`):
//!
//! 1. EXACT symmetry, bitwise (the ordered walks were symmetric only after
//!    the `½(x + y)` average).
//! 2. Bitwise identity with one of the oracle's two ordered evaluations:
//!    unsplit and hcore `new[μν] = old[min(μ,ν), max(μ,ν)]`; range split
//!    `new[μν] ∈ {old[μν], old[νμ]}`, and for compact × split pairs the
//!    row of the orientation with fewer kept calls.
//! 3. Bitwise across 1/2/6 threads.
//! 4. Against the AVERAGED oracle, per element,
//!    `|new − avg| ≤ ½|x − y| + ε|avg|` with `x, y` the two ordered
//!    evaluations. Derivation (measure-free): `new ∈ {x, y}` and
//!    `avg = fl(½ fl(x + y)) = ½(x + y)(1 + δ)`, `|δ| ≤ ε/2`, so
//!    `|new − avg| ≤ ½|x − y| + ½ε|x + y| = ½|x − y| + ε|avg|(1 + O(ε))`.
//!    The scale is the old walk's own round-off asymmetry (`asym_j3`,
//!    `sr_asymmetry`; printed below, ~1e-16 relative).
//! 5. The RHF energy matches the frozen pre-s2 build to `E_BAR` = 1e-11 Ha
//!    (the task bar; the expected first-order shift is ‖ΔJ3‖ ~ 1e-16
//!    relative, i.e. ~1e-15 Ha).
//! 6. Counters: the ordered-equivalent (off-diagonal × 2) equals the ordered
//!    oracle's count (unsplit and hcore; the screen is invariant under the
//!    map up to round-off exactly at a radius).
//!
//! Artifact hypothesis, stated before measuring: a correct s2 walk gives 0
//! symmetric-bit violations and max|new − avg| at the ~1e-17 level (half the
//! old asymmetry). A broken one is O(|J3|) ~ 1e-1 off: dropping the
//! transpose write leaves `(ν, μ)` zero; skipping the diagonal pairs zeroes
//! the diagonal blocks; a mirrored-index slip copies the wrong function
//! pair. Those fail items 1, 2 and 4 by ~15 orders of magnitude.
//!
//! Mutants the bound CANNOT see (recorded so nobody relies on it):
//! * negating the pair image `L` inside the s2 task: at Gamma
//!   `Σ_L B_{−L}[μν] = Σ_L B_L[νμ]`, the same number up to round-off (an
//!   identity at Γ; it is the k-point `−r_L` residue map that needs a
//!   non-TRIM mesh test);
//! * writing every `(i, j)` of a diagonal block (no `i ≤ j` rule): each
//!   element then holds the other ordered evaluation, still in `{x, y}` and
//!   still owned by one task. Only item 2's "which one" (unsplit
//!   `old[min, max]`) catches it, and only where the two differ in bits.

mod common;

use common::*;
use ferric_core::basis;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::{
    overlap_kinetic_parallel_and_serial, periodic_hcore, periodic_hcore_pair_s1_oracle,
    sr_attraction_parallel_and_serial, PeriodicHcore, PeriodicHcoreConfig, SrBound,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{
    sr3_gamma_s2_and_s1, sr_walk_counts, RangeSplit, RsGdf, RsGdfConfig, Sr3Parts,
};
use ndarray::Array2;

const THREADS: [usize; 3] = [1, 2, 6];
const HCORE_OMEGA: f64 = 0.8;
/// Item 5 of the module doc.
const E_BAR: f64 = 1e-11;

/// Runs `f` inside a fresh rayon pool of `n` threads and returns its result.
fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

fn gdf_cfg(split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(1 << 30),
        range_split: split,
        ..Default::default()
    }
}

/// The number of elements whose bit patterns differ; panics if the shapes
/// differ.
fn bit_diffs(a: &Array2<f64>, b: &Array2<f64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Asserts that `a` and `b` agree in every bit (see [`bit_diffs`]); `what`
/// labels a failure.
fn assert_bitwise(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    let d = bit_diffs(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} elements differ in bits", a.len());
}

/// Largest element-wise `|m|` (0 for an empty matrix).
fn max_abs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// Item 4: `(max |new − avg|, max ½|x − y|, elements over the bound)` with
/// `avg = 0.5 * (x + y)` — the expression `symmetrize_pairs` / `symmetrize`
/// evaluate.
fn vs_average(new: &Array2<f64>, x: &Array2<f64>, y: &Array2<f64>) -> (f64, f64, usize) {
    assert_eq!(new.dim(), x.dim());
    assert_eq!(new.dim(), y.dim());
    let (mut dmax, mut hmax, mut over) = (0.0_f64, 0.0_f64, 0usize);
    for ((&n, &a), &b) in new.iter().zip(x.iter()).zip(y.iter()) {
        let avg = 0.5 * (a + b);
        let d = (n - avg).abs();
        let half = 0.5 * (a - b).abs();
        dmax = dmax.max(d);
        hmax = hmax.max(half);
        if !(d <= half + f64::EPSILON * avg.abs()) {
            over += 1;
        }
    }
    (dmax, hmax, over)
}

/// The triclinic 4 H cell with STO-3G s + one Cartesian p (0.8): 8 shells,
/// 36 unordered pairs with 1×1, 1×3, 3×3 blocks. At ω = 1, λ = 1 the s shell
/// is SPLIT (its 0.169 primitive is smooth, `a ≤ λω²/2`) and the p shell is
/// compact only, so the range split has compact × split pairs in both index
/// orders (p of atom a before s of atom b, and s before p).
fn tri_sp() -> (Cell, PreparedBasis) {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    (cell, prep)
}

fn sr3_runs(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
) -> Vec<[Sr3Parts; 2]> {
    THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                sr3_gamma_s2_and_s1(cell, prep, aux, cfg).expect("SR3")
            })
        })
        .collect()
}

/// Items 1, 3 and 4 for one set of SR3 runs; returns the s2 / s1 pair of
/// the 1-thread run.
fn check_sr3_common<'a>(runs: &'a [[Sr3Parts; 2]], n: usize, tag: &str) -> &'a [Sr3Parts; 2] {
    let [(s2, c2, o2), (s1, c1, o1)] = &runs[0];
    assert!(s2.iter().any(|x| *x != 0.0), "{tag}: J3 vacuous");
    for (&t, [(a2, d2, p2), (a1, d1, p1)]) in THREADS.iter().zip(runs) {
        assert_bitwise(a2, s2, &format!("{tag}: s2 J3 at {t} vs 1 thread"));
        assert_bitwise(a1, s1, &format!("{tag}: s1 J3 at {t} vs 1 thread"));
        assert_eq!(
            (d2, p2, d1, p1),
            (c2, o2, c1, o1),
            "{tag}: counts at {t} threads"
        );
    }
    // 1. Exactly symmetric.
    assert_bitwise(
        s2,
        &transpose_pair_rows(s2, n),
        &format!("{tag}: s2 J3 vs its μ↔ν transpose"),
    );
    // 4. Against the averaged ordered oracle.
    let s1t = transpose_pair_rows(s1, n);
    let (dmax, hmax, over) = vs_average(s2, s1, &s1t);
    eprintln!(
        "{tag}: max|J3| {:.3e}; ordered walk asym max ½|x−y| {hmax:.2e}; \
         max|s2 − avg| {dmax:.2e}; {over} over the bound; bits x≠y in {} of {} elements; \
         triplets s2 {c2} (ordered-equivalent {o2}) vs s1 {c1}",
        max_abs(s2),
        bit_diffs(s1, &s1t),
        s1.len()
    );
    assert_eq!(
        over, 0,
        "{tag}: s2 J3 outside ½|x − y| + ε|avg| of the average"
    );
    assert_eq!(o1, c1, "{tag}: the ordered oracle's two counts");
    assert!(
        c2 < c1,
        "{tag}: s2 computed {c2} of the ordered {c1} triplets"
    );
    &runs[0]
}

#[test]
fn rsgdf_sr3_s2_is_the_ordered_walk_mirrored() {
    let (cell, prep) = tri_sp();
    let aux = prep_for(&cell, &basis::bundled("def2-universal-jkfit").unwrap());
    let n = prep.nbasis();
    let runs = sr3_runs(&cell, &prep, &aux, &gdf_cfg(None));
    let [(s2, c2, o2), (s1, c1, _)] = check_sr3_common(&runs, n, "unsplit");
    // 2. Bitwise the ordered walk's μ ≤ ν rows.
    assert_bitwise(
        s2,
        &mirror_upper_pair_rows(s1, n),
        "unsplit: s2 J3 vs the ordered walk's μ ≤ ν rows",
    );
    // 6. The ordered-equivalent is the ordered count.
    assert_eq!(o2, c1, "unsplit: ordered-equivalent vs the ordered count");
    let wc = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(None)).unwrap();
    assert_eq!(
        (
            wc.n_sr3_triplets,
            wc.n_sr3_triplets_ordered,
            wc.n_sr3_triplets_s1
        ),
        (*c2, *o2, *c1),
        "unsplit: the integral-free walk counts"
    );
}

#[test]
fn rsgdf_sr3_s2_split_takes_the_cheaper_orientation() {
    let (cell, prep) = tri_sp();
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let n = prep.nbasis();
    let cfg = gdf_cfg(Some(RangeSplit::default()));
    let runs = sr3_runs(&cell, &prep, &aux, &cfg);
    let [(s2, c2, o2), (s1, c1, _)] = check_sr3_common(&runs, n, "split");
    let s1t = transpose_pair_rows(s1, n);
    // 2. Every element is one of the two ordered evaluations, and for a
    // compact p × split s pair it is the (s, p) row: 1 kept call instead of
    // the (p, s) orientation's 2 (`SplitPlan::orient`).
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let (mut neither, mut wrong_orient, mut observable) = (0usize, 0usize, 0usize);
    for (i1, (&o1, &d1)) in offs.iter().zip(dims.iter()).enumerate() {
        for (i2, (&o2, &d2)) in offs.iter().zip(dims.iter()).enumerate() {
            for mu in o1..o1 + d1 {
                for nu in o2..o2 + d2 {
                    let r = mu * n + nu;
                    for p in 0..s2.ncols() {
                        let (v, x, y) = (s2[(r, p)], s1[(r, p)], s1t[(r, p)]);
                        if v.to_bits() != x.to_bits() && v.to_bits() != y.to_bits() {
                            neither += 1;
                        }
                        // p (dim 3, compact) before s (dim 1, split): the
                        // s2 walk evaluates the reversed row (ν, μ).
                        if i1 < i2 && d1 == 3 && d2 == 1 {
                            if x.to_bits() != y.to_bits() {
                                observable += 1;
                            }
                            if v.to_bits() != y.to_bits() {
                                wrong_orient += 1;
                            }
                        }
                        // s before p: the natural row, already 1 call.
                        if i1 < i2 && d1 == 1 && d2 == 3 && v.to_bits() != x.to_bits() {
                            wrong_orient += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "split: {neither} elements match neither ordered row; {wrong_orient} not in the cheaper \
         orientation; {observable} p×s elements where the two orientations differ in bits; \
         triplets s2 {c2} (ordered-equivalent {o2}) vs ordered {c1}"
    );
    assert_eq!(neither, 0, "split: s2 element outside {{x, y}}");
    assert!(
        observable > 0,
        "split: the two orientations agree in every bit on this fixture, so the orientation \
         check below is vacuous (redesign the fixture)"
    );
    assert_eq!(
        wrong_orient, 0,
        "split: s2 did not take the cheaper orientation"
    );
    let wc = sr_walk_counts(&cell, &prep, &aux, &cfg).unwrap();
    assert_eq!(
        (
            wc.n_sr3_triplets,
            wc.n_sr3_triplets_ordered,
            wc.n_sr3_triplets_s1
        ),
        (*c2, *o2, *c1),
        "split: the integral-free walk counts"
    );
}

#[test]
fn hcore_s2_is_the_ordered_loop_mirrored_and_exactly_symmetric() {
    let (cell, prep) = tri_sp();
    let cfg = PeriodicHcoreConfig::with_omega(HCORE_OMEGA);
    let hcs: Vec<PeriodicHcore> = THREADS
        .iter()
        .map(|&t| in_pool(t, || periodic_hcore(&cell, &prep, &cfg).expect("hcore")))
        .collect();
    for (&t, hc) in THREADS.iter().zip(&hcs) {
        for (a, b, what) in [
            (&hc.s, &hcs[0].s, "S"),
            (&hc.t, &hcs[0].t, "T"),
            (&hc.v_sr, &hcs[0].v_sr, "V_SR"),
            (&hc.h, &hcs[0].h, "h"),
        ] {
            assert_bitwise(a, b, &format!("{what} at {t} vs 1 thread"));
        }
        assert_eq!(hc.n_sr_triplets, hcs[0].n_sr_triplets);
        assert_eq!(hc.n_sr_triplets_ordered, hcs[0].n_sr_triplets_ordered);
    }
    let hc = &hcs[0];
    let old = periodic_hcore_pair_s1_oracle(&cell, &prep, &cfg).expect("s1 oracle hcore");
    // The raw ordered evaluations (x = m, y = mᵀ) behind the oracle.
    let ([_, (s_raw, t_raw)], _) =
        overlap_kinetic_parallel_and_serial(&cell, &prep, &cfg).expect("S/T");
    let [_, (v_raw, n_ord, _, _)] = sr_attraction_parallel_and_serial(
        &cell,
        &prep,
        cfg.omega,
        cfg.precision,
        cfg.precision,
        SrBound::Derived,
        false,
    )
    .expect("SR attraction");
    for (new, raw, avg_old, what) in [
        (&hc.s, &s_raw, &old.s, "S"),
        (&hc.t, &t_raw, &old.t, "T"),
        (&hc.v_sr, &v_raw, &old.v_sr, "V_SR"),
    ] {
        let rt = raw.t().to_owned();
        // The oracle is the pre-s2 build: the average of the raw loop.
        assert_bitwise(avg_old, &(0.5 * (raw + &rt)), &format!("{what}: oracle"));
        // 1 and 2.
        assert_bitwise(new, &new.t().to_owned(), &format!("{what}: exact symmetry"));
        assert_bitwise(
            new,
            &mirror_upper(raw),
            &format!("{what}: ordered μ ≤ ν mirrored"),
        );
        // 4.
        let (dmax, hmax, over) = vs_average(new, raw, &rt);
        eprintln!(
            "{what}: max|m| {:.3e}; ordered asym max ½|x−y| {hmax:.2e}; max|s2 − avg| {dmax:.2e}; \
             {over} over the bound; bits x≠y in {} of {}",
            max_abs(raw),
            bit_diffs(raw, &rt),
            raw.len()
        );
        assert_eq!(over, 0, "{what}: outside ½|x − y| + ε|avg| of the average");
    }
    assert_eq!(hc.sr_asymmetry, 0.0, "s2 V_SR must be exactly symmetric");
    // 6.
    assert_eq!(old.n_sr_triplets, n_ord);
    assert_eq!(old.n_sr_triplets_ordered, old.n_sr_triplets);
    assert_eq!(
        hc.n_sr_triplets_ordered, old.n_sr_triplets,
        "ordered-equivalent vs the ordered count"
    );
    assert!(hc.n_sr_triplets < old.n_sr_triplets);
}

#[test]
fn rhf_energy_matches_the_frozen_pre_s2_build() {
    let (tri, tri_prep) = tri_sp();
    let h2 = h2_cell(4.0);
    let h2_prep = prep_for(&h2, &pyscf_sto3g_h());
    let cases: [(&str, &Cell, &PreparedBasis, Option<RangeSplit>); 3] = [
        ("H2 STO-3G", &h2, &h2_prep, None),
        ("tri 4H s+p", &tri, &tri_prep, None),
        (
            "tri 4H s+p split",
            &tri,
            &tri_prep,
            Some(RangeSplit::default()),
        ),
    ];
    for (tag, cell, prep, split) in cases {
        let aux = prep_for(cell, &basis::bundled("cc-pvdz-ri").unwrap());
        let hcfg = PeriodicHcoreConfig::with_omega(HCORE_OMEGA);
        let cfg = gdf_cfg(split);
        let hc_new = periodic_hcore(cell, prep, &hcfg).expect("hcore");
        let hc_old = periodic_hcore_pair_s1_oracle(cell, prep, &hcfg).expect("hcore oracle");
        let new = RsGdf::build(cell, prep, &aux, &hc_new.s, &cfg).expect("rsgdf");
        let old = RsGdf::build_pair_s1_oracle(cell, prep, &aux, &hc_old.s, &cfg).expect("oracle");
        let energy = |hc: &PeriodicHcore, g: &RsGdf| {
            gamma_rhf_jk(
                cell,
                prep,
                hc,
                Box::new(g.j_builder()),
                Box::new(g.k_builder()),
            )
            .energy
        };
        let (e_new, e_old) = (energy(&hc_new, &new), energy(&hc_old, &old));
        let (sn, so) = (new.stats(), old.stats());
        eprintln!(
            "{tag}: E s2 {e_new:.15} vs pre-s2 {e_old:.15} (ΔE {:.2e}); asym_j3 {:.2e} → {:.2e}; \
             hcore SR asym {:.2e} → {:.2e}; SR3 {} computed / {} ordered-equivalent vs {}",
            e_new - e_old,
            so.asym_j3,
            sn.asym_j3,
            hc_old.sr_asymmetry,
            hc_new.sr_asymmetry,
            sn.n_sr3_triplets,
            sn.n_sr3_triplets_ordered,
            so.n_sr3_triplets
        );
        assert!(
            (e_new - e_old).abs() <= E_BAR,
            "{tag}: ΔE {:.3e}",
            e_new - e_old
        );
        assert_eq!(
            so.n_sr3_triplets_ordered, so.n_sr3_triplets,
            "{tag}: oracle counts"
        );
        if split.is_none() {
            assert_eq!(
                sn.n_sr3_triplets_ordered, so.n_sr3_triplets,
                "{tag}: ordered-equivalent vs the pre-s2 counter"
            );
        }
        assert_eq!(
            hc_new.n_sr_triplets_ordered, hc_old.n_sr_triplets,
            "{tag}: hcore counts"
        );
    }
}
