//! Second derivatives of the density-fitting integrals `(P|μν)` and `(P|Q)` by
//! a fourth-order stencil of the first-derivative blocks
//! (`ferric_scf::hessian_df_stencil`, issue #290).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --test df_hessian_stencil
//! ```
//!
//! # Oracle
//!
//! An INDEPENDENT construction: Richardson-extrapolated mixed second
//! differences of the integral VALUES, evaluated on really displaced
//! geometries (a new `Molecule`, new `PreparedBasis` for orbital AND aux basis,
//! plain `compute_eri3` / `compute_eri2`). It uses neither the first-derivative
//! engines nor the shifted engines, so it cannot share an error with the
//! stencil (a shared shift convention, say) the way a shifted-value oracle
//! could.
//!
//! # Artifact hypotheses (what a BROKEN stencil would look like)
//!
//! Every failure mode of this module is a clean small-looking number:
//!
//! * one shell displaced instead of the atom: a same-atom pair stops moving
//!   rigidly, the error is ~1e-3 on the blocks that have one (and exactly
//!   zero on blocks that do not), so a test using only cross-atom triples
//!   would pass it;
//! * a wrong shell-to-atom fold (bra and ket swapped): invisible in the
//!   translational ROW sums (every centre still gets summed once) and
//!   invisible whenever both shells sit on one atom; it needs the oracle or
//!   the symmetry `H[a,b] = H[b,a]`;
//! * a wrong sign on the aux displacement: leaves the symmetric part of the
//!   row sums intact in the 2-centre case (the sign flips the whole block);
//! * a too-coarse step or a second-order stencil: an O(h^2..h^4) truncation
//!   error that is smooth and systematic, ~1e-6 for the second-order stencil
//!   at h = 2e-3, which looks like "agrees to 6 digits".
//!
//! The oracle comparison is per shell block over EVERY (a,x,b,y), so each of
//! these shows up as a number far above the production floor. Bars are placed
//! between the measured production floor and the smallest mutant miss.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian_df_stencil::{
    eri2_deriv2_block, eri2_deriv2_contract, eri3_deriv2_block, eri3_deriv2_contract,
    weight_c_times_d, weight_matrix, StencilConfig, Variant,
};
use ndarray::Array2;
use std::sync::OnceLock;

/// Distorted water (Angstrom): no accidental symmetry zeros.
const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.1300 -0.7000 -0.4192
";

const H2: &str = "2
h2
H 0.0 0.0 0.0
H 0.1 0.2 0.8
";

struct Sys {
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
}

impl Sys {
    fn new(xyz: &str, obs: &str, aux: &str) -> Self {
        Sys {
            mol: Molecule::parse_xyz(xyz, 0, 1).expect("molecule"),
            obs: basis::bundled(obs).expect("orbital basis"),
            aux: basis::bundled(aux).expect("aux basis"),
        }
    }
    fn prep(&self, mol: &Molecule) -> (PreparedBasis, PreparedBasis) {
        (
            PreparedBasis::new(mol, &self.obs).expect("obs"),
            PreparedBasis::new(mol, &self.aux).expect("aux"),
        )
    }
    fn natoms(&self) -> usize {
        self.mol.atoms.len()
    }
}

fn displaced(mol: &Molecule, d: &[(usize, usize, f64)]) -> Molecule {
    let mut m = mol.clone();
    for &(a, ax, v) in d {
        let at = &mut m.atoms[a];
        match ax {
            0 => at.x += v,
            1 => at.y += v,
            _ => at.zpos += v,
        }
    }
    m
}

/// Full `(P, μ, ν)` value tensor on a (displaced) geometry.
fn values3(sys: &Sys, mol: &Molecule) -> Vec<f64> {
    let (obs, df) = sys.prep(mol);
    let mut eng = Engine::new_3center(Operator::coulomb(), &obs, &df, PREC).unwrap();
    let (nb, na) = (obs.nbasis(), df.nbasis());
    let mut t = vec![0.0; na * nb * nb];
    let (od, dd, oo, doff) = (
        obs.shell_dims().to_vec(),
        df.shell_dims().to_vec(),
        obs.shell_offsets().to_vec(),
        df.shell_offsets().to_vec(),
    );
    for sp in 0..df.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let Some(b) = eng.compute_eri3(&obs, &df, sp, s1, s2) else {
                    continue;
                };
                for p in 0..dd[sp] {
                    for i in 0..od[s1] {
                        for j in 0..od[s2] {
                            t[((doff[sp] + p) * nb + oo[s1] + i) * nb + oo[s2] + j] =
                                b[(p * od[s1] + i) * od[s2] + j];
                        }
                    }
                }
            }
        }
    }
    t
}

/// Full `(P|Q)` value matrix on a (displaced) geometry.
fn values2(sys: &Sys, mol: &Molecule) -> Vec<f64> {
    let (_, df) = sys.prep(mol);
    let mut eng = Engine::new_2center(Operator::coulomb(), &df, PREC).unwrap();
    let na = df.nbasis();
    let mut t = vec![0.0; na * na];
    let (dd, doff) = (df.shell_dims().to_vec(), df.shell_offsets().to_vec());
    for sp in 0..df.nshells() {
        for sq in 0..df.nshells() {
            let b = eng.compute_eri2(&df, sp, sq);
            for p in 0..dd[sp] {
                for q in 0..dd[sq] {
                    t[(doff[sp] + p) * na + doff[sq] + q] = b[p * dd[sq] + q];
                }
            }
        }
    }
    t
}

/// Second derivative table indexed `[3a+x][3b+y]`, each a flat tensor.
type Elems = Vec<Vec<Vec<f64>>>;

/// Sixth-order Richardson mixed second difference of VALUES on really displaced geometries.
fn oracle(sys: &Sys, h: f64, vals: &dyn Fn(&Sys, &Molecule) -> Vec<f64>) -> Elems {
    oracle_sel(sys, h, vals, None)
}

/// [`oracle`] restricted to the coordinate pairs `(i, j)`, `i ≤ j`, with
/// `i = 3a + x` (all pairs when `None`); unselected elements stay empty.
fn oracle_sel(
    sys: &Sys,
    h: f64,
    vals: &dyn Fn(&Sys, &Molecule) -> Vec<f64>,
    pairs: Option<&[(usize, usize)]>,
) -> Elems {
    let n3 = 3 * sys.natoms();
    let mut out: Elems = vec![vec![Vec::new(); n3]; n3];
    for i in 0..n3 {
        for j in i..n3 {
            if pairs.is_some_and(|p| !p.contains(&(i, j))) {
                continue;
            }
            let (a, x, b, y) = (i / 3, i % 3, j / 3, j % 3);
            let mixed = |h: f64| -> Vec<f64> {
                let mut m: Vec<f64> = Vec::new();
                for (sa, sb) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    let mol = displaced(&sys.mol, &[(a, x, sa * h), (b, y, sb * h)]);
                    let v = vals(sys, &mol);
                    if m.is_empty() {
                        m = vec![0.0; v.len()];
                    }
                    for (o, t) in m.iter_mut().zip(&v) {
                        *o += sa * sb * t / (4.0 * h * h);
                    }
                }
                m
            };
            // Three-level Richardson (h, 2h, 4h): O(h^6) truncation.
            let (m1, m2, m4) = (mixed(h), mixed(2.0 * h), mixed(4.0 * h));
            let r: Vec<f64> = (0..m1.len())
                .map(|k| {
                    let r1 = (4.0 * m1[k] - m2[k]) / 3.0;
                    let r2 = (4.0 * m2[k] - m4[k]) / 3.0;
                    (16.0 * r1 - r2) / 15.0
                })
                .collect();
            out[j][i] = r.clone();
            out[i][j] = r;
        }
    }
    out
}

/// The module's blocks scattered into full tensors, same indexing as the oracle.
fn stencil3(sys: &Sys, cfg: &StencilConfig) -> Elems {
    let (obs, df) = sys.prep(&sys.mol);
    let mut eng = Engine::new_3center_deriv(Operator::coulomb(), &obs, &df, PREC).unwrap();
    let (nb, na, n3) = (obs.nbasis(), df.nbasis(), 3 * sys.natoms());
    let mut out: Elems = vec![vec![vec![0.0; na * nb * nb]; n3]; n3];
    let (od, dd, oo, doff) = (
        obs.shell_dims().to_vec(),
        df.shell_dims().to_vec(),
        obs.shell_offsets().to_vec(),
        df.shell_offsets().to_vec(),
    );
    for sp in 0..df.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let Some(blk) = eri3_deriv2_block(&mut eng, &obs, &df, (sp, s1, s2), cfg).unwrap()
                else {
                    continue;
                };
                for (ia, &a) in blk.atoms.iter().enumerate() {
                    for (ib, &b) in blk.atoms.iter().enumerate() {
                        for x in 0..3 {
                            for y in 0..3 {
                                let src = blk.get(ia, x, ib, y);
                                let t = &mut out[3 * a + x][3 * b + y];
                                for p in 0..dd[sp] {
                                    for i in 0..od[s1] {
                                        for j in 0..od[s2] {
                                            t[((doff[sp] + p) * nb + oo[s1] + i) * nb
                                                + oo[s2]
                                                + j] = src[(p * od[s1] + i) * od[s2] + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn stencil2(sys: &Sys, cfg: &StencilConfig) -> Elems {
    let (_, df) = sys.prep(&sys.mol);
    let mut eng = Engine::new_2center_deriv(Operator::coulomb(), &df, PREC).unwrap();
    let (na, n3) = (df.nbasis(), 3 * sys.natoms());
    let mut out: Elems = vec![vec![vec![0.0; na * na]; n3]; n3];
    let (dd, doff) = (df.shell_dims().to_vec(), df.shell_offsets().to_vec());
    for sp in 0..df.nshells() {
        for sq in 0..df.nshells() {
            let Some(blk) = eri2_deriv2_block(&mut eng, &df, (sp, sq), cfg).unwrap() else {
                continue;
            };
            for (ia, &a) in blk.atoms.iter().enumerate() {
                for (ib, &b) in blk.atoms.iter().enumerate() {
                    for x in 0..3 {
                        for y in 0..3 {
                            let src = blk.get(ia, x, ib, y);
                            let t = &mut out[3 * a + x][3 * b + y];
                            for p in 0..dd[sp] {
                                for q in 0..dd[sq] {
                                    t[(doff[sp] + p) * na + doff[sq] + q] = src[p * dd[sq] + q];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// `(max |a - b|, max |b|)` over every element.
fn compare(a: &Elems, b: &Elems) -> (f64, f64) {
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for (ra, rb) in a.iter().zip(b) {
        for (ta, tb) in ra.iter().zip(rb) {
            for (u, v) in ta.iter().zip(tb) {
                worst = worst.max((u - v).abs());
                scale = scale.max(v.abs());
            }
        }
    }
    (worst, scale)
}

/// `(row, column, symmetry)` residuals: `Σ_a H[a,x;b,y]`, `Σ_b H[a,x;b,y]`,
/// `H[a,x;b,y] - H[b,y;a,x]`.
fn invariance(h: &Elems, natoms: usize) -> (f64, f64, f64) {
    let (mut row, mut col, mut sym) = (0.0f64, 0.0f64, 0.0f64);
    let n = h[0][0].len();
    for i in 0..n {
        for x in 0..3 {
            for b in 0..natoms {
                for y in 0..3 {
                    let r: f64 = (0..natoms).map(|a| h[3 * a + x][3 * b + y][i]).sum();
                    row = row.max(r.abs());
                    let c: f64 = (0..natoms).map(|a| h[3 * b + y][3 * a + x][i]).sum();
                    col = col.max(c.abs());
                }
            }
        }
        for p in 0..3 * natoms {
            for q in 0..3 * natoms {
                sym = sym.max((h[p][q][i] - h[q][p][i]).abs());
            }
        }
    }
    (row, col, sym)
}

fn cfg(variant: Variant, step: f64) -> StencilConfig {
    StencilConfig {
        step,
        variant,
        precision: PREC,
    }
}

/// libint2 precision of every engine in this file.
const PREC: f64 = 1e-14;

/// Step (Bohr) of the value mixed differences. Measured sweep
/// (`oracle_step_sweep`): the sixth-order oracle is truncation-limited above
/// ~1e-2 and double-roundoff-limited below ~3e-3 (error ~ eps·|f|/h^2); its
/// minimum is at 5e-3.
const H_ORACLE: f64 = 5e-3;

/// Bar on `max |stencil - oracle|` (the largest element is 2.4 for 3c and 3.5
/// for 2c). Derived from BOTH sides on the water/6-31G/cc-pVDZ-RI case at the
/// production step 2e-3: production misses the oracle by 9.7e-11 (3c) and
/// 1.8e-10 (2c), i.e. by the oracle's own floor; the closest mutant (step
/// x10) misses by 4.3e-7 (3c) and 8.3e-7 (2c). 1e-8 leaves a factor 55 above
/// the production floor and a factor 43 below the closest mutant divided by 10.
const BAR: f64 = 1e-8;
/// Translational-invariance / symmetry bar (row, column, `H[a,b] = H[b,a]`).
/// Production measures <= 8e-12 (rows, columns) and <= 4.1e-11 (symmetry).
const INV_BAR: f64 = 1e-10;
const STEP: f64 = 2e-3;

fn water() -> Sys {
    Sys::new(WATER, "6-31g", "cc-pvdz-ri")
}

fn oracle3_water() -> &'static Elems {
    static O: OnceLock<Elems> = OnceLock::new();
    O.get_or_init(|| oracle(&water(), H_ORACLE, &values3))
}

fn oracle2_water() -> &'static Elems {
    static O: OnceLock<Elems> = OnceLock::new();
    O.get_or_init(|| oracle(&water(), H_ORACLE, &values2))
}

// ---------------------------------------------------------------- anchors

/// Anchor (a), trivial limit: a ZERO shift of the shifted first-derivative
/// engines is bitwise the unshifted engines, so the stencil's zero-offset
/// evaluation IS the gradient integral the SCF uses.
#[test]
fn zero_shift_is_the_gradient_integral_bitwise() {
    let sys = water();
    let (obs, df) = sys.prep(&sys.mol);
    let mut e3 = Engine::new_3center_deriv(Operator::coulomb(), &obs, &df, PREC).unwrap();
    let mut e3s = Engine::new_3center_deriv(Operator::coulomb(), &obs, &df, PREC).unwrap();
    for sp in 0..df.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..=s1 {
                let a = e3
                    .compute_eri3_deriv(&obs, &df, sp, s1, s2)
                    .map(|b| b.to_vec());
                let b = e3s
                    .compute_eri3_deriv_shifted(&obs, &df, sp, s1, s2, [[0.0; 3]; 3])
                    .unwrap()
                    .map(|b| b.to_vec());
                assert_eq!(a, b, "3c ({sp}|{s1},{s2})");
            }
        }
    }
    let mut e2 = Engine::new_2center_deriv(Operator::coulomb(), &df, PREC).unwrap();
    let mut e2s = Engine::new_2center_deriv(Operator::coulomb(), &df, PREC).unwrap();
    for sp in 0..df.nshells() {
        for sq in 0..=sp {
            let a = e2.compute_eri2_deriv(&df, sp, sq).map(|b| b.to_vec());
            let b = e2s
                .compute_eri2_deriv_shifted(&df, sp, sq, [0.0; 3])
                .unwrap()
                .map(|b| b.to_vec());
            assert_eq!(a, b, "2c ({sp}|{sq})");
        }
    }
}

/// Anchor (a), translational invariance and symmetry of the stencil result:
/// `Σ_a H[a,x;b,y] = 0`, `Σ_b H[a,x;b,y] = 0` and `H[a,x;b,y] = H[b,y;a,x]`
/// for every integral. For two centres the ROW sum is zero by construction
/// (odd stencil weights, ket shift of opposite sign for the two atoms); only
/// the column sum and the symmetry carry information there.
#[test]
fn stencil_is_translationally_invariant_and_symmetric() {
    let sys = water();
    let p = cfg(Variant::Production, STEP);
    let (r3, c3, y3) = invariance(&stencil3(&sys, &p), sys.natoms());
    let (r2, c2, y2) = invariance(&stencil2(&sys, &p), sys.natoms());
    println!(
        "3c row {r3:.2e} col {c3:.2e} sym {y3:.2e}; 2c row {r2:.2e} col {c2:.2e} sym {y2:.2e}"
    );
    for (name, v) in [
        ("3c row", r3),
        ("3c col", c3),
        ("3c sym", y3),
        ("2c row", r2),
        ("2c col", c2),
        ("2c sym", y2),
    ] {
        assert!(v <= INV_BAR, "{name} = {v:.3e} > {INV_BAR:.0e}");
    }
}

// ----------------------------------------------------------- accuracy

#[test]
fn oracle_step_sweep() {
    let sys = water();
    let s3 = stencil3(&sys, &cfg(Variant::Production, STEP));
    let s2 = stencil2(&sys, &cfg(Variant::Production, STEP));
    for h in [1e-2, 5e-3, 3e-3, 2e-3, 1e-3] {
        let (d3, _) = compare(&s3, &oracle(&sys, h, &values3));
        let (d2, _) = compare(&s2, &oracle(&sys, h, &values2));
        println!("oracle h={h:.2e}: |stencil(2e-3) - oracle|: 3c {d3:.3e}  2c {d2:.3e}");
    }
}

#[test]
fn stencil_3c_matches_value_mixed_differences() {
    let sys = water();
    let (d, scale) = compare(
        &stencil3(&sys, &cfg(Variant::Production, STEP)),
        oracle3_water(),
    );
    println!("3c: max |stencil - oracle| = {d:.3e}, max |d2| = {scale:.3e}");
    assert!(scale > 1.0, "vacuous: max |d2| = {scale:.3e}");
    assert!(
        d < BAR,
        "3c stencil misses the value differences by {d:.3e}"
    );
}

#[test]
fn stencil_2c_matches_value_mixed_differences() {
    let sys = water();
    let (d, scale) = compare(
        &stencil2(&sys, &cfg(Variant::Production, STEP)),
        oracle2_water(),
    );
    println!("2c: max |stencil - oracle| = {d:.3e}, max |d2| = {scale:.3e}");
    assert!(scale > 1.0, "vacuous: max |d2| = {scale:.3e}");
    assert!(
        d < BAR,
        "2c stencil misses the value differences by {d:.3e}"
    );
}

/// Two atoms only (minimal case): every triple has at most two distinct
/// atoms, and triples with P and one orbital shell on the same atom exist.
#[test]
fn two_atom_system_matches_value_mixed_differences() {
    let sys = Sys::new(H2, "6-31g", "cc-pvdz-ri");
    let p = cfg(Variant::Production, STEP);
    let (d3, s3) = compare(&stencil3(&sys, &p), &oracle(&sys, H_ORACLE, &values3));
    let (d2, s2) = compare(&stencil2(&sys, &p), &oracle(&sys, H_ORACLE, &values2));
    println!("H2: 3c {d3:.3e} (scale {s3:.3e}), 2c {d2:.3e} (scale {s2:.3e})");
    assert!(s3 > 0.1 && s2 > 0.1);
    assert!(d3 < BAR && d2 < BAR);
}

/// A single atom: every triple and pair is rigidly translated, so the
/// (skipped) second derivative must be zero in the oracle too. This justifies
/// the mask: the module returns `None` for these blocks.
#[test]
fn single_atom_second_derivative_is_zero_in_the_oracle() {
    let sys = Sys::new("1\nne\nNe 0 0 0\n", "6-31g", "cc-pvdz-ri");
    let p = cfg(Variant::Production, STEP);
    let o3 = oracle(&sys, H_ORACLE, &values3);
    let o2 = oracle(&sys, H_ORACLE, &values2);
    let s3 = stencil3(&sys, &p);
    let s2 = stencil2(&sys, &p);
    // The stencil is exactly zero (all blocks masked); the oracle residual is
    // its own roundoff.
    assert!(s3.iter().flatten().flatten().all(|v| *v == 0.0));
    assert!(s2.iter().flatten().flatten().all(|v| *v == 0.0));
    let (d3, d2) = (compare(&s3, &o3).0, compare(&s2, &o2).0);
    println!("Ne: oracle residual 3c {d3:.3e} 2c {d2:.3e}");
    assert!(d3 < BAR && d2 < BAR);
}

/// The stencil error is O(h^4): consecutive step differences `S(h) - S(h/2)`
/// shrink by ~16 (the oracle floor is irrelevant here: both are stencils).
#[test]
fn step_scaling_is_fourth_order() {
    let sys = water();
    for (name, f) in [
        ("3c", stencil3 as fn(&Sys, &StencilConfig) -> Elems),
        ("2c", stencil2 as fn(&Sys, &StencilConfig) -> Elems),
    ] {
        let steps = [4.8e-2, 2.4e-2, 1.2e-2, 6e-3];
        let s: Vec<Elems> = steps
            .iter()
            .map(|&h| f(&sys, &cfg(Variant::Production, h)))
            .collect();
        let d: Vec<f64> = (0..3).map(|k| compare(&s[k], &s[k + 1]).0).collect();
        let ratios: Vec<f64> = d.windows(2).map(|w| w[0] / w[1]).collect();
        println!(
            "{name}: |S(h) - S(h/2)| for h = 4.8e-2, 2.4e-2, 1.2e-2: {:?}, ratios {ratios:.2?}",
            d.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
        );
        for r in ratios {
            assert!(
                (12.0..=20.0).contains(&r),
                "{name}: ratio {r:.2} is not the h^4 value 16"
            );
        }
    }
}

// ------------------------------------------------------------ mutants

fn production_and_mutants(
    name: &str,
    f: fn(&Sys, &StencilConfig) -> Elems,
    oracle: &Elems,
    mutants: &[Variant],
    identities: &[Variant],
) {
    let sys = water();
    let (prod, _) = compare(&f(&sys, &cfg(Variant::Production, STEP)), oracle);
    assert!(prod < BAR, "{name}: production {prod:.3e} >= bar {BAR:.0e}");
    for &v in mutants {
        let (d, _) = compare(&f(&sys, &cfg(v, STEP)), oracle);
        println!(
            "{name} mutant {v:?}: {d:.3e} = {:.1e} x bar (production {prod:.2e})",
            d / BAR
        );
        assert!(
            d > 10.0 * BAR,
            "{name}: mutant {v:?} misses the oracle by only {d:.3e} (< 10 x bar)"
        );
    }
    // Documented identities: the mutant does not change this construction.
    for &v in identities {
        let (d, _) = compare(&f(&sys, &cfg(v, STEP)), oracle);
        println!("{name} mutant {v:?}: {d:.3e} (identity here, by construction)");
        assert!(
            d < BAR,
            "{name}: {v:?} expected to be an identity, got {d:.3e}"
        );
    }
}

/// Every production-wired mutant of the 3c stencil misses the independent
/// oracle by more than 10 x the bar.
#[test]
fn mutants_are_caught_3c() {
    production_and_mutants(
        "3c",
        stencil3,
        oracle3_water(),
        &[
            Variant::SingleShell,
            Variant::AuxSignFlip,
            Variant::Step10,
            Variant::SecondOrder,
            Variant::DropCross,
            Variant::SwapBraKet,
        ],
        &[],
    );
}

/// The same for the 2c stencil. `SingleShell` is an IDENTITY for two centres
/// (a pair on two different atoms moves one shell per atom either way; a
/// same-atom pair is skipped), which is why the 3c case, where a same-atom
/// orbital pair exists, is the one that guards the rigid-atom requirement.
#[test]
fn mutants_are_caught_2c() {
    production_and_mutants(
        "2c",
        stencil2,
        oracle2_water(),
        &[
            Variant::AuxSignFlip,
            Variant::Step10,
            Variant::SecondOrder,
            Variant::DropCross,
            Variant::SwapBraKet,
        ],
        &[Variant::SingleShell],
    );
}

/// Which anchor sees which mutant (3c). The translational row sum is BLIND to
/// the bra/ket fold swap and to the aux sign flip (every centre is still
/// counted once); the symmetry `H[a,b] = H[b,a]` sees them. Only the oracle
/// sees a coarse or low-order stencil. This pins that the oracle is not
/// redundant with the cheap anchors.
#[test]
fn row_sum_anchor_is_blind_where_expected() {
    let sys = water();
    let n = sys.natoms();
    let inv = |v| invariance(&stencil3(&sys, &cfg(v, STEP)), n);
    assert!(inv(Variant::Production).0 < INV_BAR);
    assert!(
        inv(Variant::SwapBraKet).0 < INV_BAR,
        "swap should hide in row sums"
    );
    assert!(
        inv(Variant::AuxSignFlip).0 < INV_BAR,
        "sign flip should hide in row sums"
    );
    assert!(
        inv(Variant::SwapBraKet).2 > 1e-3,
        "symmetry must catch the swap"
    );
    assert!(
        inv(Variant::SingleShell).2 > 1e-3,
        "symmetry must catch single-shell"
    );
    assert!(
        inv(Variant::AuxSignFlip).2 > 1e-3,
        "symmetry must catch the sign flip"
    );
}

// ------------------------------------------------- contracted drivers

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

fn sym_matrix(n: usize, seed: &mut u64) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let v = lcg(seed);
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

fn contract3_with(sys: &Sys, cfg: &StencilConfig, c: &[f64], d: &Array2<f64>) -> Array2<f64> {
    let (obs, df) = sys.prep(&sys.mol);
    eri3_deriv2_contract(&obs, &df, Operator::coulomb(), cfg, |sp, s1, s2, w| {
        weight_c_times_d(&obs, &df, c, d, (sp, s1, s2), w)
    })
    .unwrap()
}

fn contract2_with(sys: &Sys, cfg: &StencilConfig, m: &Array2<f64>) -> Array2<f64> {
    let (_, df) = sys.prep(&sys.mol);
    eri2_deriv2_contract(&df, Operator::coulomb(), cfg, |sp, sq, w| {
        weight_matrix(&df, m, (sp, sq), w)
    })
    .unwrap()
}

/// Contractions of an element table with the same weights, as plain sums.
fn table_contract3(t: &Elems, c: &[f64], d: &Array2<f64>, nb: usize) -> Array2<f64> {
    let n3 = t.len();
    Array2::from_shape_fn((n3, n3), |(i, j)| {
        let mut acc = 0.0;
        for (p, cp) in c.iter().enumerate() {
            for mu in 0..nb {
                for nu in 0..nb {
                    acc += cp * d[(mu, nu)] * t[i][j][(p * nb + mu) * nb + nu];
                }
            }
        }
        acc
    })
}

fn table_contract2(t: &Elems, m: &Array2<f64>, na: usize) -> Array2<f64> {
    let n3 = t.len();
    Array2::from_shape_fn((n3, n3), |(i, j)| {
        let mut acc = 0.0;
        for p in 0..na {
            for q in 0..na {
                acc += m[(p, q)] * t[i][j][p * na + q];
            }
        }
        acc
    })
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Bar on the contracted drivers (weights O(1) over ~10^4..10^5 integrals, so
/// the oracle floor accumulates). Derived like [`BAR`] from both sides:
/// production misses by 3.9e-11 (3c) and 6.6e-10 (2c); the closest mutant
/// (second-order stencil) by 5.3e-6 (3c). 2e-8 is 30x above the production
/// floor and 26x below the closest mutant divided by 10.
const CONTRACT_BAR: f64 = 2e-8;

/// End to end: the contracted drivers (weights, block fold, scatter to
/// `3N x 3N`) against the contraction of the independent oracle tables with
/// the same random weights.
#[test]
fn contracted_drivers_match_the_oracle_contraction() {
    let sys = water();
    let (obs, df) = sys.prep(&sys.mol);
    let (nb, na) = (obs.nbasis(), df.nbasis());
    let mut seed = 12345u64;
    let c: Vec<f64> = (0..na).map(|_| lcg(&mut seed)).collect();
    let d = sym_matrix(nb, &mut seed);
    let m = sym_matrix(na, &mut seed);
    let p = cfg(Variant::Production, STEP);

    let h3 = contract3_with(&sys, &p, &c, &d);
    let r3 = table_contract3(oracle3_water(), &c, &d, nb);
    let h2 = contract2_with(&sys, &p, &m);
    let r2 = table_contract2(oracle2_water(), &m, na);
    let (e3, e2) = (max_abs(&(&h3 - &r3)), max_abs(&(&h2 - &r2)));
    let (s3, s2) = (max_abs(&r3), max_abs(&r2));
    println!("contracted: 3c {e3:.3e} (scale {s3:.2e}), 2c {e2:.3e} (scale {s2:.2e})");
    assert!(s3 > 1.0 && s2 > 1.0, "vacuous contraction");
    assert!(e3 < CONTRACT_BAR && e2 < CONTRACT_BAR);

    // The contracted result of every mutant misses too: the weights and the
    // scatter cannot hide a wrong stencil.
    for v in [
        Variant::SwapBraKet,
        Variant::DropCross,
        Variant::SecondOrder,
    ] {
        let q = cfg(v, STEP);
        let b3 = max_abs(&(&contract3_with(&sys, &q, &c, &d) - &r3));
        println!("contracted 3c mutant {v:?}: {b3:.3e}");
        assert!(b3 > 10.0 * CONTRACT_BAR, "{v:?}: {b3:.3e}");
        let b2 = max_abs(&(&contract2_with(&sys, &q, &m) - &r2));
        println!("contracted 2c mutant {v:?}: {b2:.3e}");
        assert!(b2 > 10.0 * CONTRACT_BAR, "{v:?}: {b2:.3e}");
    }
}

/// The contracted result is bit-identical for every rayon thread count.
#[test]
fn contracted_drivers_are_thread_count_bit_identical() {
    let sys = water();
    let (obs, df) = sys.prep(&sys.mol);
    let mut seed = 777u64;
    let c: Vec<f64> = (0..df.nbasis()).map(|_| lcg(&mut seed)).collect();
    let d = sym_matrix(obs.nbasis(), &mut seed);
    let m = sym_matrix(df.nbasis(), &mut seed);
    let p = cfg(Variant::Production, STEP);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    contract3_with(&sys, &p, &c, &d),
                    contract2_with(&sys, &p, &m),
                )
            })
    };
    let (a3, a2) = run(1);
    for t in [2, 3, 8] {
        let (b3, b2) = run(t);
        assert_eq!(a3, b3, "3c differs at {t} threads");
        assert_eq!(a2, b2, "2c differs at {t} threads");
    }
}

// ------------------------------------------- expensive (ignored) cases

/// Per angular-momentum class `(l_P, l_mu, l_nu)` (3c) or `(l_P, l_Q)` (2c):
/// `(max |stencil - oracle|, max |oracle|)`.
type Classes = std::collections::BTreeMap<Vec<i32>, (f64, f64)>;

fn func_shells(prep: &PreparedBasis) -> Vec<usize> {
    let mut v = Vec::new();
    for (s, &d) in prep.shell_dims().iter().enumerate() {
        v.resize(v.len() + d, s);
    }
    v
}

fn classes3(sys: &Sys, st: &Elems, or: &Elems) -> Classes {
    let (obs, df) = sys.prep(&sys.mol);
    let (fo, fd) = (func_shells(&obs), func_shells(&df));
    let (lo, ld) = (
        obs.located_shells().iter().map(|s| s.l).collect::<Vec<_>>(),
        df.located_shells().iter().map(|s| s.l).collect::<Vec<_>>(),
    );
    let nb = obs.nbasis();
    let mut out = Classes::new();
    for (rs, ro) in st.iter().zip(or) {
        for (ts, to) in rs.iter().zip(ro) {
            for (k, (u, v)) in ts.iter().zip(to).enumerate() {
                let (p, mu, nu) = (k / (nb * nb), (k / nb) % nb, k % nb);
                let key = vec![ld[fd[p]], lo[fo[mu]], lo[fo[nu]]];
                let e = out.entry(key).or_insert((0.0, 0.0));
                e.0 = e.0.max((u - v).abs());
                e.1 = e.1.max(v.abs());
            }
        }
    }
    out
}

fn classes2(sys: &Sys, st: &Elems, or: &Elems) -> Classes {
    let (_, df) = sys.prep(&sys.mol);
    let fd = func_shells(&df);
    let ld = df.located_shells().iter().map(|s| s.l).collect::<Vec<_>>();
    let na = df.nbasis();
    let mut out = Classes::new();
    for (rs, ro) in st.iter().zip(or) {
        for (ts, to) in rs.iter().zip(ro) {
            for (k, (u, v)) in ts.iter().zip(to).enumerate() {
                let key = vec![ld[fd[k / na]], ld[fd[k % na]]];
                let e = out.entry(key).or_insert((0.0, 0.0));
                e.0 = e.0.max((u - v).abs());
                e.1 = e.1.max(v.abs());
            }
        }
    }
    out
}

/// Coordinate pairs `(3a+x, 3b+y)` probed in the heavier cases: same-atom
/// diagonal, cross-atom, and an off-axis mixed pair.
const PAIRS: [(usize, usize); 6] = [(2, 2), (2, 4), (4, 4), (3, 6), (1, 7), (5, 8)];

fn report_case(name: &str, sys: &Sys, pairs: &[(usize, usize)], rel_bar: f64) {
    let sel = Some(pairs);
    let p = cfg(Variant::Production, STEP);
    let (s3, s2) = (stencil3(sys, &p), stencil2(sys, &p));
    let (o3, o2) = (
        oracle_sel(sys, H_ORACLE, &values3, sel),
        oracle_sel(sys, H_ORACLE, &values2, sel),
    );
    let (obs, df) = sys.prep(&sys.mol);
    println!(
        "{name}: orbital max l = {}, aux max l = {}, nbf {}, naux {}",
        obs.max_l(),
        df.max_l(),
        obs.nbasis(),
        df.nbasis()
    );
    let mut worst_rel = 0.0f64;
    for (tag, cl) in [
        ("3c (lP,l1,l2)", classes3(sys, &s3, &o3)),
        ("2c (lP,lQ)", classes2(sys, &s2, &o2)),
    ] {
        for (k, (e, sc)) in &cl {
            println!(
                "  {tag} {k:?}: max err {e:.3e}  max |d2| {sc:.3e}  rel {:.2e}",
                e / sc.max(1e-300)
            );
            if *sc > 1e-3 {
                worst_rel = worst_rel.max(e / sc);
            }
        }
    }
    println!("{name}: worst relative error over classes with max|d2| > 1e-3: {worst_rel:.2e}");
    assert!(
        worst_rel < rel_bar,
        "{name}: {worst_rel:.2e} >= {rel_bar:.0e}"
    );
}

/// All `(l_P, l_mu, l_nu)` and `(l_P, l_Q)` classes of water/cc-pVDZ with the
/// cc-pVDZ-RI aux (orbital d, aux f), every coordinate pair.
#[test]
fn accuracy_by_angular_momentum_class() {
    let sys = Sys::new(WATER, "cc-pvdz", "cc-pvdz-ri");
    report_case("water cc-pVDZ / cc-pVDZ-RI", &sys, &PAIRS, REL_BAR);
}

/// High angular momentum: orbital f (def2-TZVP on O) with aux g
/// (def2-universal-jkfit on O).
#[test]
fn high_angular_momentum_aux_g_orbital_f() {
    let sys = Sys::new(WATER, "def2-tzvp", "def2-universal-jkfit");
    report_case(
        "water def2-TZVP / def2-universal-jkfit",
        &sys,
        &PAIRS,
        REL_BAR,
    );
}

/// Two heavy atoms, so that (g | f f) and (f | f f)-type triples with their
/// centres on DIFFERENT atoms exist (on water the f and g shells all sit on
/// the one oxygen and are masked as a rigidly translated triple).
#[test]
#[ignore = "expensive: CO/def2-TZVP/def2-universal-jkfit class table; measured 4 s"]
fn high_angular_momentum_two_heavy_atoms() {
    let sys = Sys::new(
        "2\nco\nC 0.0 0.0 0.0\nO 0.05 0.1 1.13\n",
        "def2-tzvp",
        "def2-universal-jkfit",
    );
    report_case(
        "CO def2-TZVP / def2-universal-jkfit",
        &sys,
        &[(2, 2), (2, 5), (5, 5), (0, 3), (1, 4)],
        REL_BAR,
    );
}

/// Diffuse functions: aug-cc-pVDZ with the aug-cc-pVDZ-RIFIT aux.
#[test]
fn diffuse_functions() {
    let sys = Sys::new(WATER, "aug-cc-pvdz", "aug-cc-pvdz-rifit");
    report_case(
        "water aug-cc-pVDZ / aug-cc-pVDZ-RIFIT",
        &sys,
        &PAIRS,
        REL_BAR,
    );
}

/// Relative bar for the heavier cases (error / largest element of the class).
const REL_BAR: f64 = 1e-8;

fn alkane(n: usize, obs: &str, aux: &str) -> Sys {
    let path = format!(
        "{}/../../testdata/molecules/alkane_{n}.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    Sys {
        mol: Molecule::load_xyz(&path).unwrap(),
        obs: basis::bundled(obs).unwrap(),
        aux: basis::bundled(aux).unwrap(),
    }
}

/// Cost of the stencil against the first-derivative integral pass the DF
/// gradient does (same pool, one thread), on a small and a medium alkane.
#[test]
#[ignore = "timing: alkane_2 + alkane_4, 6-31G/cc-pVDZ-RI, one thread; measured 5.4 s"]
fn stencil_cost_against_the_gradient_integral_pass() {
    use std::time::Instant;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for n in [2usize, 4] {
        let sys = alkane(n, "6-31g", "cc-pvdz-ri");
        let (obs, df) = sys.prep(&sys.mol);
        let natoms = sys.natoms();
        let mut seed = 99u64;
        let c: Vec<f64> = (0..df.nbasis()).map(|_| lcg(&mut seed)).collect();
        let d = sym_matrix(obs.nbasis(), &mut seed);
        let m = sym_matrix(df.nbasis(), &mut seed);
        let p = cfg(Variant::Production, STEP);
        let (mut calls_g3, mut calls_s3) = (0u64, 0u64);
        for sp in 0..df.nshells() {
            for s1 in 0..obs.nshells() {
                for s2 in 0..=s1 {
                    let mut at = vec![
                        df.shell_to_atom()[sp],
                        obs.shell_to_atom()[s1],
                        obs.shell_to_atom()[s2],
                    ];
                    at.sort_unstable();
                    at.dedup();
                    calls_g3 += 1;
                    if at.len() > 1 {
                        calls_s3 += 12 * at.len() as u64;
                    }
                }
            }
        }
        let t_g3 = pool.install(|| {
            let mut eng = Engine::new_3center_deriv(Operator::coulomb(), &obs, &df, PREC).unwrap();
            let t0 = Instant::now();
            let mut acc = 0.0;
            for sp in 0..df.nshells() {
                for s1 in 0..obs.nshells() {
                    for s2 in 0..=s1 {
                        if let Some(b) = eng.compute_eri3_deriv(&obs, &df, sp, s1, s2) {
                            acc += b[0];
                        }
                    }
                }
            }
            std::hint::black_box(acc);
            t0.elapsed().as_secs_f64()
        });
        let t_s3 = pool.install(|| {
            let t0 = Instant::now();
            std::hint::black_box(contract3_with(&sys, &p, &c, &d));
            t0.elapsed().as_secs_f64()
        });
        let t_g2 = pool.install(|| {
            let mut eng = Engine::new_2center_deriv(Operator::coulomb(), &df, PREC).unwrap();
            let t0 = Instant::now();
            let mut acc = 0.0;
            for sp in 0..df.nshells() {
                for sq in 0..=sp {
                    if let Some(b) = eng.compute_eri2_deriv(&df, sp, sq) {
                        acc += b[0];
                    }
                }
            }
            std::hint::black_box(acc);
            t0.elapsed().as_secs_f64()
        });
        let t_s2 = pool.install(|| {
            let t0 = Instant::now();
            std::hint::black_box(contract2_with(&sys, &p, &m));
            t0.elapsed().as_secs_f64()
        });
        println!(
            "alkane_{n}: {natoms} atoms, nbf {}, naux {}\n  3c: gradient pass {t_g3:.3}s, stencil {t_s3:.3}s, ratio {:.1} (engine calls {calls_s3} vs {calls_g3} = {:.1}); per coordinate (3N = {}) {:.2}x\n  2c: gradient pass {t_g2:.3}s, stencil {t_s2:.3}s, ratio {:.1}",
            obs.nbasis(),
            df.nbasis(),
            t_s3 / t_g3,
            calls_s3 as f64 / calls_g3 as f64,
            3 * natoms,
            t_s3 / t_g3 / (3 * natoms) as f64,
            t_s2 / t_g2
        );
    }
}

/// Production, or the named mutant (`FERRIC_DERIV2_MUTANT`) to show the
/// libint comparison can fail.
fn dump_cfg() -> StencilConfig {
    let v = match std::env::var("FERRIC_DERIV2_MUTANT").as_deref() {
        Ok("swap") => Variant::SwapBraKet,
        Ok("order2") => Variant::SecondOrder,
        Ok("single") => Variant::SingleShell,
        _ => Variant::Production,
    };
    cfg(v, STEP)
}

/// Writes shell triples and their stencil blocks for
/// `scripts/probes/eri3_deriv2_vs_stencil.cc` (libint analytic second
/// derivatives, source build only). `FERRIC_DERIV2_DUMP=<file>`.
#[test]
#[ignore = "manual: needs FERRIC_DERIV2_DUMP and the libint 2.13.1 probe (scripts/probes/eri3_deriv2_vs_stencil.cc); measured 0.04 s"]
fn dump_for_libint_deriv2_probe() {
    use std::fmt::Write as _;
    let Ok(path) = std::env::var("FERRIC_DERIV2_DUMP") else {
        return;
    };
    let sys = Sys::new(WATER, "cc-pvdz", "cc-pvdz-ri");
    let (obs, df) = sys.prep(&sys.mol);
    let mut eng = Engine::new_3center_deriv(Operator::coulomb(), &obs, &df, PREC).unwrap();
    let (os, ds) = (obs.located_shells(), df.located_shells());
    let mut seen = std::collections::HashSet::new();
    let mut text = String::new();
    let mut ncase = 0;
    for sp in 0..df.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..=s1 {
                let (lp, l1, l2) = (ds[sp].l, os[s1].l, os[s2].l);
                if lp.max(l1).max(l2) > 3 {
                    continue;
                }
                let atoms = [
                    df.shell_to_atom()[sp],
                    obs.shell_to_atom()[s1],
                    obs.shell_to_atom()[s2],
                ];
                let nd = {
                    let mut d = atoms.to_vec();
                    d.sort_unstable();
                    d.dedup();
                    d.len()
                };
                // one case per (l, l, l, #distinct atoms, same-atom orbital pair?)
                if nd == 1 || !seen.insert((lp, l1, l2, nd, atoms[1] == atoms[2])) {
                    continue;
                }
                let Some(blk) =
                    eri3_deriv2_block(&mut eng, &obs, &df, (sp, s1, s2), &dump_cfg()).unwrap()
                else {
                    continue;
                };
                ncase += 1;
                writeln!(
                    text,
                    "case {lp} {l1} {l2} {} {} {} {}",
                    atoms[0], atoms[1], atoms[2], blk.n
                )
                .unwrap();
                for sh in [&ds[sp], &os[s1], &os[s2]] {
                    write!(
                        text,
                        "{} {} {} {:.17e} {:.17e} {:.17e}",
                        sh.l,
                        sh.pure as i32,
                        sh.exponents.len(),
                        sh.center[0],
                        sh.center[1],
                        sh.center[2]
                    )
                    .unwrap();
                    for v in sh.exponents.iter().chain(sh.coefficients) {
                        write!(text, " {v:.17e}").unwrap();
                    }
                    writeln!(text).unwrap();
                }
                writeln!(text, "{}", blk.atoms.len()).unwrap();
                let ids: Vec<String> = blk.atoms.iter().map(|a| a.to_string()).collect();
                writeln!(text, "{}", ids.join(" ")).unwrap();
                for ia in 0..blk.atoms.len() {
                    for x in 0..3 {
                        for ib in 0..blk.atoms.len() {
                            for y in 0..3 {
                                for v in blk.get(ia, x, ib, y) {
                                    write!(text, "{v:.17e} ").unwrap();
                                }
                                writeln!(text).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
    std::fs::write(&path, format!("{ncase}\n{text}")).unwrap();
}

/// Which side owns the (g|f f) floor? Sweeps the stencil step and the oracle
/// step on the CO case and prints the worst relative class error of each.
#[test]
#[ignore = "diagnostic: CO/def2-TZVP/def2-universal-jkfit step sweep; measured 16 s on a loaded box"]
fn high_angular_momentum_step_sweep() {
    let sys = Sys::new(
        "2\nco\nC 0.0 0.0 0.0\nO 0.05 0.1 1.13\n",
        "def2-tzvp",
        "def2-universal-jkfit",
    );
    let pairs = [(2usize, 2usize), (2, 5), (5, 5), (0, 3), (1, 4)];
    let worst = |cl: &Classes| {
        cl.values()
            .filter(|(_, sc)| *sc > 1e-3)
            .map(|(e, sc)| e / sc)
            .fold(0.0f64, f64::max)
    };
    for ho in [1e-2, 5e-3] {
        let o3 = oracle_sel(&sys, ho, &values3, Some(&pairs[..]));
        for step in [4e-3, 2e-3, 1e-3] {
            let s3 = stencil3(&sys, &cfg(Variant::Production, step));
            println!(
                "oracle h {ho:.0e}, stencil step {step:.0e}: worst 3c class rel err {:.2e}",
                worst(&classes3(&sys, &s3, &o3))
            );
        }
    }
}
