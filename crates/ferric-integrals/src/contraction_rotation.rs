//! Exact column rotations of generally contracted shells.
//!
//! Correlation-consistent (and ANO-type) bases store the contraction COLUMNS
//! of an atom's s (and p) set as separate shells over one shared exponent
//! list, and the most diffuse primitive is also a column of its own
//! (cc-pVDZ C: s 9 primitives in columns with 9, 9, 1 non-zero
//! coefficients; p 4 primitives in columns with 4, 1). libint evaluates
//! every column as an independent segmented shell, so the shared primitives
//! are recomputed per column, and the single diffuse primitive inside the
//! long columns keeps every tight × diffuse primitive pair alive at any
//! distance.
//!
//! This module replaces such a basis by an EXACTLY equivalent one (the same
//! span, function by function an invertible linear map `T`) whose columns
//! are more compact, for the integral walks that use it internally and
//! transform back. The AO basis every caller sees is never changed.
//!
//! # What is rotated
//!
//! A GROUP is the shells of one atom with the same `l`, the same `pure`
//! flag and bitwise the same exponent list. Two stages:
//!
//! 1. **Diffuse projection** ([`RotationLevel::Rot1`](crate::contraction_rotation::RotationLevel::Rot1)). Inside a group,
//!    every column `k` with more than one non-zero coefficient whose support
//!    contains the primitive `q` of a single-primitive column `s` of the
//!    group is replaced by
//!
//!    ```text
//!    k' = k − (c_kq / c_sq) s        (raw coefficients: c_kq set to exactly 0)
//!    ```
//!
//!    with its zero primitives DROPPED (the single-primitive columns used are
//!    zero-dropped too).
//! 2. **Optimized general contraction** ([`RotationLevel::Ogc`](crate::contraction_rotation::RotationLevel::Ogc), on top of
//!    stage 1; Hashimoto, Hirao and Tatewaki, Chem. Phys. Lett. 243, 190
//!    (1995)). The `n ≥ 2` remaining long columns of a group that share one
//!    exponent list (cc-pVDZ/cc-pVTZ C: the 1s and 2s columns) are
//!    recombined so that each loses `n − 1` primitives. With `P` the `n × n`
//!    matrix of their raw coefficients at `n` PIVOT primitives (one per
//!    column) and `D = diag(P)`, the new columns are `C'' = D P⁻¹ C'`: column
//!    `i` keeps its own pivot and is exactly zero at the other `n − 1`.
//!    Columns are ordered by their normalized weight on the tightest
//!    primitive; the first `n − 1` pivot at the `n − 1` tightest primitives,
//!    the last (the most valence-like) at the most diffuse one. For `n = 2`
//!    the core column loses the most diffuse primitive and the valence
//!    column the tightest (cc-pVDZ C s: 8, 8 → 7, 7 primitives).
//!
//!    The group's transform is refused (the group keeps stage 1) when the
//!    1-norm condition number of its `T` block exceeds [`OGC_MAX_COND`](crate::contraction_rotation::OGC_MAX_COND)
//!    (cc-pVDZ C s: 1.8, H s: 1.6).
//!
//! # The transform (exact)
//!
//! libint renormalises every contraction it is given (`N = s(c)^{-1/2}`,
//! `s` = self-overlap over unit primitives). With raw coefficient rows
//! `C = R C_rot` within a group, the normalized functions obey
//!
//! ```text
//! χ_k = Σ_m T_km χ'_m,      T_km = R_km √(s(c'_m) / s(c_k)),
//! ```
//!
//! identity outside the rotated groups. Stage 1 alone gives
//! `T_kk = √(s(c'_k) / s(c_k))`, `T_ks = (c_kq / c_sq) √(s(c_s) / s(c_k))`;
//! stage 2 composes `T = T₁ T₂`. Every integral is multilinear in the
//! orbitals and a rotated shell has the same `l`, centre and AO layout as
//! its parent, so a matrix built on the rotated shells transforms
//! EXACTLY: `X = T X' Tᵀ` ([`RotatedBasis::back_transform_rows`](crate::contraction_rotation::RotatedBasis::back_transform_rows)), and a
//! density contracts as `D' = Tᵀ D T` ([`AoTranspose::matrix`](crate::contraction_rotation::AoTranspose::matrix)). A screen
//! run on the rotated shells keeps a different quartet set than the parent
//! walk, so results agree with the unrotated build to the screening
//! precision, not bitwise.
//!
//! # Who uses it
//!
//! * `ferric-pbc`'s Gamma and k-point short-range 3-centre walks and the
//!   hcore SR attraction (stage 1 only; `ferric_pbc::sr_rotation`), through
//!   [`RotatedBasis::detect`](crate::contraction_rotation::RotatedBasis::detect).
//! * `ferric-scf`'s direct four-centre SCF J/K builder (`DirectJK`, stages 1
//!   and 2 by default; `FERRIC_GC_ROTATE`), through
//!   [`RotatedBasis::detect_prepared`](crate::contraction_rotation::RotatedBasis::detect_prepared).
//!
//! # Numerics
//!
//! * Identity: when nothing in the basis rotates (segmented bases, STO-3G,
//!   Pople sp shells — no group has a single-primitive column or two long
//!   columns over one exponent list), detection returns `None` and the
//!   caller runs its unrotated code path bit for bit.
//! * Guard: a stage-1 column whose `T_kk < `[`COLUMN_ROTATION_MIN_DIAG`](crate::contraction_rotation::COLUMN_ROTATION_MIN_DIAG) or
//!   with an `|T_ks| > `[`COLUMN_ROTATION_MAX_COEF`](crate::contraction_rotation::COLUMN_ROTATION_MAX_COEF) is left unrotated (its
//!   row of `T` stays the identity); cc-pVDZ C has `T_ks` 0.0032 and 0.58.
//! * The back-transforms are fixed serial loops, so they are bitwise
//!   identical across thread counts.

use crate::basis_bridge::PreparedBasis;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;
use std::collections::{BTreeMap, HashMap};

/// Smallest diagonal `T_kk` a rotated column may have (below it the rotated
/// column is nearly the removed one and the transform would amplify
/// round-off by `1/T_kk`).
pub const COLUMN_ROTATION_MIN_DIAG: f64 = 1e-3;
/// Largest `|T_ks|` a rotated column may have (the design's `cond(R) ≤ 1e3`).
pub const COLUMN_ROTATION_MAX_COEF: f64 = 1e3;
/// Largest 1-norm condition number of a group's `T` block that the
/// optimized-general-contraction stage accepts; above it the group keeps the
/// stage-1 rotation.
pub const OGC_MAX_COND: f64 = 1e2;

/// Production rotation or a deliberately BROKEN variant (negative controls
/// for `ferric-pbc`'s rotation tests; an anchor that cannot tell them from
/// [`ColumnRotationMutant::Production`] certifies nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRotationMutant {
    /// The rotation of the module doc.
    Production,
    /// MUTATION: every off-diagonal `T_ks` enters the back-transform with the
    /// wrong sign.
    FlipSign,
    /// MUTATION: libint's renormalisation is ignored (`T_kk = 1`,
    /// `T_ks = c_kq / c_sq`).
    NoRenormalization,
    /// MUTATION: the back-transform is applied to the first orbital index
    /// only (`J3[μν] = Σ_m T_μm J3'[mν]`).
    OneSidedBackTransform,
    /// MUTATION (RS-GDF only): the AUX basis is rotated as well in the SR
    /// 3-centre walk, and its columns are never transformed back. With a
    /// segmented aux basis (cc-pvdz-ri) this is a no-op; the test uses a
    /// generally contracted aux basis. `periodic_hcore` refuses it.
    RotateAux,
    /// MUTATION: the back-transform is skipped (`T` dropped; the rotated
    /// blocks are used as if they were the parent's).
    NoBackTransform,
    /// MUTATION: `Tᵀ` where `T` belongs (each row of `T` read as a column).
    TransposedBackTransform,
}

/// Which stages of the rotation run (module doc, "What is rotated").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationLevel {
    /// Stage 1 only: the single-primitive columns projected out of the long
    /// ones (what the periodic SR walks use).
    Rot1,
    /// Stage 1, then the optimized-general-contraction elimination among the
    /// remaining long columns of a group.
    Ogc,
}

/// `s(c)`: self-overlap of a contraction over unit-normalised primitives
/// (the formula of the range split's `contraction_norm2`, `rsgdf::split`).
fn contraction_norm2(l: i32, exps: &[f64], coefs: &[f64]) -> f64 {
    let e = l as f64 + 1.5;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(e);
        }
    }
    s
}

/// True when two shells have identical `l`, purity and bit-identical exponents.
fn same_primitives(a: &Shell, b: &Shell) -> bool {
    a.l == b.l
        && a.pure == b.pure
        && a.exponents.len() == b.exponents.len()
        && a.exponents
            .iter()
            .zip(&b.exponents)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// True when two shells are bitwise the same contraction.
fn same_shell(a: &Shell, b: &Shell) -> bool {
    same_primitives(a, b)
        && a.coefficients.len() == b.coefficients.len()
        && a.coefficients
            .iter()
            .zip(&b.coefficients)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Indices of the non-zero entries of `c`.
fn nonzero(c: &[f64]) -> Vec<usize> {
    (0..c.len()).filter(|&p| c[p] != 0.0).collect()
}

/// Copy of `sh` keeping only the primitives whose coefficient in `coefs` is non-zero.
fn drop_zeros(sh: &Shell, coefs: &[f64]) -> Shell {
    let (exponents, coefficients): (Vec<f64>, Vec<f64>) = sh
        .exponents
        .iter()
        .zip(coefs)
        .filter(|(_, &c)| c != 0.0)
        .map(|(&a, &c)| (a, c))
        .unzip();
    Shell {
        l: sh.l,
        pure: sh.pure,
        exponents,
        coefficients,
    }
}

/// One atom's (or element's) rotated templates: the libint shells, the rows
/// of `T` (template indices, diagonal first), each template shell's group,
/// and the number of rotated columns.
struct TemplateRotation {
    shells: Vec<Shell>,
    rows: Vec<Vec<(usize, f64)>>,
    group_of: Vec<usize>,
    n_groups: usize,
    n_rotated: usize,
}

/// The row of `T` for column `k` of `tmpls` against the single-primitive
/// columns `hits` (`(template shell, primitive)`), and the rotated raw
/// coefficients; `None` if the guard refuses it. Mutants alter the row only
/// (never the rotated shell).
fn rotated_row(
    tmpls: &[Shell],
    k: usize,
    hits: &[(usize, usize)],
    mutant: ColumnRotationMutant,
) -> Option<(Vec<(usize, f64)>, Vec<f64>)> {
    let sh = &tmpls[k];
    let (l, e, c) = (sh.l, &sh.exponents, &sh.coefficients);
    let s_k = contraction_norm2(l, e, c);
    let mut cp = c.clone();
    for &(_, q) in hits {
        cp[q] = 0.0;
    }
    let s_p = contraction_norm2(l, e, &cp);
    if !(s_k > 0.0) || !(s_p > 0.0) {
        return None;
    }
    let diag = (s_p / s_k).sqrt();
    let mut row = vec![(k, diag)];
    for &(js, q) in hits {
        let cs = &tmpls[js].coefficients;
        let t = c[q] / cs[q] * (contraction_norm2(l, e, cs) / s_k).sqrt();
        row.push((js, t));
    }
    if !(diag >= COLUMN_ROTATION_MIN_DIAG)
        || row
            .iter()
            .any(|&(_, t)| !t.is_finite() || t.abs() > COLUMN_ROTATION_MAX_COEF)
    {
        return None;
    }
    match mutant {
        ColumnRotationMutant::FlipSign => {
            for x in row.iter_mut().skip(1) {
                x.1 = -x.1;
            }
        }
        ColumnRotationMutant::NoRenormalization => {
            row[0].1 = 1.0;
            for (x, &(js, q)) in row.iter_mut().skip(1).zip(hits) {
                x.1 = c[q] / tmpls[js].coefficients[q];
            }
        }
        _ => {}
    }
    Some((row, cp))
}

/// Rotate one atom's (or element's) template shells (module doc, "What is
/// rotated"). Stage 2 runs only for [`RotationLevel::Ogc`] with the
/// production rotation.
fn rotate_templates(
    tmpls: &[Shell],
    mutant: ColumnRotationMutant,
    level: RotationLevel,
) -> TemplateRotation {
    let n = tmpls.len();
    let mut shells = tmpls.to_vec();
    let mut rows: Vec<Vec<(usize, f64)>> = (0..n).map(|k| vec![(k, 1.0)]).collect();
    let mut group_of = vec![usize::MAX; n];
    let (mut n_groups, mut n_rotated) = (0usize, 0usize);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for k0 in 0..n {
        if group_of[k0] != usize::MAX {
            continue;
        }
        let members: Vec<usize> = (k0..n)
            .filter(|&j| group_of[j] == usize::MAX && same_primitives(&tmpls[k0], &tmpls[j]))
            .collect();
        for &j in &members {
            group_of[j] = n_groups;
        }
        n_groups += 1;
        groups.push(members.clone());
        if members.len() < 2 {
            continue;
        }
        // Single-primitive columns (first one per primitive).
        let mut singles: Vec<(usize, usize)> = Vec::new();
        for &j in &members {
            let nz = nonzero(&tmpls[j].coefficients);
            if nz.len() == 1 && !singles.iter().any(|&(_, p)| p == nz[0]) {
                singles.push((j, nz[0]));
            }
        }
        if singles.is_empty() {
            continue;
        }
        let mut used = vec![false; singles.len()];
        for &k in &members {
            let c = &tmpls[k].coefficients;
            if nonzero(c).len() < 2 {
                continue;
            }
            let hit_idx: Vec<usize> = (0..singles.len())
                .filter(|&s| c[singles[s].1] != 0.0)
                .collect();
            if hit_idx.is_empty() {
                continue;
            }
            let hits: Vec<(usize, usize)> = hit_idx.iter().map(|&s| singles[s]).collect();
            let Some((row, cp)) = rotated_row(tmpls, k, &hits, mutant) else {
                continue;
            };
            rows[k] = row;
            shells[k] = drop_zeros(&tmpls[k], &cp);
            for &s in &hit_idx {
                used[s] = true;
            }
            n_rotated += 1;
        }
        for (s, &(js, _)) in singles.iter().enumerate() {
            if used[s] {
                shells[js] = drop_zeros(&tmpls[js], &tmpls[js].coefficients);
            }
        }
    }
    if level == RotationLevel::Ogc && mutant == ColumnRotationMutant::Production {
        for members in &groups {
            n_rotated += ogc_stage(members, &mut shells, &mut rows);
        }
    }
    TemplateRotation {
        shells,
        rows,
        group_of,
        n_groups,
        n_rotated,
    }
}

/// Stage 2 (module doc) on one stage-1 group `members`: every subset of at
/// least two long columns over one exponent list is recombined, its rows of
/// `T` composed with stage 1's. Returns the number of columns changed.
fn ogc_stage(members: &[usize], shells: &mut [Shell], rows: &mut [Vec<(usize, f64)>]) -> usize {
    let long: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&k| {
            shells[k].exponents.len() >= 2
                && nonzero(&shells[k].coefficients).len() == shells[k].exponents.len()
        })
        .collect();
    let mut taken = vec![false; long.len()];
    let mut changed = 0usize;
    for i0 in 0..long.len() {
        if taken[i0] {
            continue;
        }
        let set: Vec<usize> = (i0..long.len())
            .filter(|&i| !taken[i] && same_primitives(&shells[long[i0]], &shells[long[i]]))
            .collect();
        for &i in &set {
            taken[i] = true;
        }
        let cols: Vec<usize> = set.iter().map(|&i| long[i]).collect();
        if cols.len() < 2 || shells[cols[0]].exponents.len() < cols.len() {
            continue;
        }
        if let Some((new_shells, new_rows)) = ogc_subset(members, &cols, shells, rows) {
            for (&k, sh) in cols.iter().zip(new_shells) {
                shells[k] = sh;
            }
            for (k, row) in new_rows {
                rows[k] = row;
            }
            changed += cols.len();
        }
    }
    changed
}

/// The stage-2 recombination of the long columns `cols` (all over one
/// exponent list) of the group `members`: the new shells of `cols` and the
/// composed rows of every member whose row changes, or `None` when the
/// pivot block is singular or the group's `T` block is worse conditioned
/// than [`OGC_MAX_COND`].
#[allow(clippy::type_complexity)]
fn ogc_subset(
    members: &[usize],
    cols: &[usize],
    shells: &[Shell],
    rows: &[Vec<(usize, f64)>],
) -> Option<(Vec<Shell>, Vec<(usize, Vec<(usize, f64)>)>)> {
    let n = cols.len();
    let ref_sh = &shells[cols[0]];
    let (l, e) = (ref_sh.l, &ref_sh.exponents);
    let np = e.len();
    // Primitive order, tightest first (ties by index).
    let mut tight: Vec<usize> = (0..np).collect();
    tight.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
    let s1: Vec<f64> = cols
        .iter()
        .map(|&k| contraction_norm2(l, e, &shells[k].coefficients))
        .collect();
    // Columns by normalized weight on the tightest primitive, largest first.
    let mut order: Vec<usize> = (0..n).collect();
    let w = |i: usize| (shells[cols[i]].coefficients[tight[0]] / s1[i].sqrt()).abs();
    order.sort_by(|&a, &b| w(b).total_cmp(&w(a)).then(a.cmp(&b)));
    let mut pivot = vec![0usize; n];
    for (r, &i) in order.iter().enumerate() {
        pivot[i] = if r + 1 < n { tight[r] } else { tight[np - 1] };
    }
    // P[i][j] = c'_{cols[i]}[pivot[j]] (raw stage-1 coefficients).
    let p: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| shells[cols[i]].coefficients[pivot[j]])
                .collect()
        })
        .collect();
    let pinv = invert(&p)?;
    // C'' = D P⁻¹ C', exact zeros at the other columns' pivots.
    let mut new_coefs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut c = vec![0.0; np];
        for (q, cq) in c.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, &k) in cols.iter().enumerate() {
                acc += pinv[i][j] * shells[k].coefficients[q];
            }
            *cq = p[i][i] * acc;
        }
        for (j, &pj) in pivot.iter().enumerate() {
            if j != i {
                c[pj] = 0.0;
            }
        }
        c[pivot[i]] = p[i][i];
        if c.iter().any(|x| !x.is_finite()) {
            return None;
        }
        new_coefs.push(c);
    }
    let new_shells: Vec<Shell> = (0..n)
        .map(|i| drop_zeros(&shells[cols[i]], &new_coefs[i]))
        .collect();
    let s2: Vec<f64> = new_coefs
        .iter()
        .map(|c| contraction_norm2(l, e, c))
        .collect();
    if s2.iter().any(|&s| !(s > 0.0)) {
        return None;
    }
    // T₂[cols[i]][cols[j]] = R₂_ij √(s(c''_j) / s(c'_i)), R₂ = P D⁻¹.
    let t2 = |i: usize, j: usize| p[i][j] / p[j][j] * (s2[j] / s1[i]).sqrt();
    let col_pos = |m: usize| cols.iter().position(|&k| k == m);
    let mut new_rows: Vec<(usize, Vec<(usize, f64)>)> = Vec::new();
    for &k in members {
        if !rows[k].iter().any(|&(m, _)| col_pos(m).is_some()) {
            continue;
        }
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for &(m, t) in &rows[k] {
            match col_pos(m) {
                Some(i) => {
                    for (j, &kj) in cols.iter().enumerate() {
                        *acc.entry(kj).or_insert(0.0) += t * t2(i, j);
                    }
                }
                None => *acc.entry(m).or_insert(0.0) += t,
            }
        }
        let diag = acc.remove(&k).unwrap_or(0.0);
        let mut row = vec![(k, diag)];
        row.extend(acc.into_iter().filter(|&(_, t)| t != 0.0));
        if !(diag.abs() >= COLUMN_ROTATION_MIN_DIAG)
            || row
                .iter()
                .any(|&(_, t)| !t.is_finite() || t.abs() > COLUMN_ROTATION_MAX_COEF)
        {
            return None;
        }
        new_rows.push((k, row));
    }
    // Condition of the group's full T block.
    let pos = |m: usize| members.iter().position(|&k| k == m);
    let ng = members.len();
    let mut tg = vec![vec![0.0; ng]; ng];
    for (a, &k) in members.iter().enumerate() {
        let row = new_rows
            .iter()
            .find(|(kk, _)| *kk == k)
            .map_or(&rows[k], |(_, r)| r);
        for &(m, t) in row {
            tg[a][pos(m)?] = t;
        }
    }
    let tinv = invert(&tg)?;
    let norm1 = |m: &[Vec<f64>]| {
        (0..m.len())
            .map(|j| m.iter().map(|r| r[j].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let cond = norm1(&tg) * norm1(&tinv);
    if !(cond <= OGC_MAX_COND) {
        return None;
    }
    Some((new_shells, new_rows))
}

/// Inverse of a small dense matrix by Gauss–Jordan elimination with partial
/// pivoting; `None` if it is singular to working precision.
fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let scale = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if !(scale > 0.0) {
        return None;
    }
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for c in 0..n {
        let piv = (c..n).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if !(m[piv][c].abs() > 1e-13 * scale) {
            return None;
        }
        m.swap(c, piv);
        let d = m[c][c];
        for x in m[c].iter_mut() {
            *x /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r][c];
                if f != 0.0 {
                    let pivot_row = m[c].clone();
                    for (x, y) in m[r].iter_mut().zip(&pivot_row) {
                        *x -= f * y;
                    }
                }
            }
        }
    }
    Some(m.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// A column-rotated basis and the exact back-transform to its parent
/// (module doc).
pub struct RotatedBasis {
    /// The rotated libint basis: the parent's atoms, shell order, `l`,
    /// `pure` and AO layout (checked), rotated contractions.
    pub prep: PreparedBasis,
    /// Per parent shell: `(rotated shell, T entry)`, diagonal first; entries
    /// only within the shell's group.
    rows: Vec<Vec<(usize, f64)>>,
    /// Groups of shells (members ascending), a partition of all shells, in
    /// order of their first member.
    groups: Vec<Vec<usize>>,
    /// Whether every row of the group is the identity.
    identity: Vec<bool>,
    /// Shell → (group, first AO of the shell inside the group's AO list).
    slot: Vec<(usize, usize)>,
    offs: Vec<usize>,
    dims: Vec<usize>,
    nbasis: usize,
    /// Columns rotated (over all atoms).
    pub n_rotated_columns: usize,
    one_sided: bool,
    /// MUTATION ([`ColumnRotationMutant::NoBackTransform`]): the
    /// back-transforms are no-ops.
    skip_back: bool,
}

impl std::fmt::Debug for RotatedBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotatedBasis")
            .field("nbasis", &self.nbasis)
            .field("n_rotated_columns", &self.n_rotated_columns)
            .finish_non_exhaustive()
    }
}

/// TEST-ONLY defects of the DERIVATIVE paths' forward transform
/// (`Y' = Tᵀ Y T`, `D' = Tᵀ D T`); the energy was built with the production
/// rotation, so each of these makes force / stress differ from the unrotated
/// one by an O(|T_ks|) amount (`ferric-pbc`'s `tests/pbc_sr_rotation.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivRotMutant {
    /// Production.
    #[default]
    None,
    /// MUTATION: no transform at all (rotated walk, unrotated weights).
    DropT,
    /// MUTATION: `T Y Tᵀ` (T where Tᵀ belongs).
    UseT,
    /// MUTATION: only the first AO index transformed.
    OneSided,
    /// MUTATION: off-diagonal `T_ks` with the wrong sign.
    FlipSign,
}

/// The forward (transpose) transform of a weight over AO pairs onto the
/// rotated AO basis: `W'[m, n] = Σ_{μν} T_μm T_νn W[μ, ν]`, evaluated per
/// element from the parent entries (no `n²`-sized copy of `W`). An AO whose
/// column of `T` is the identity (nothing rotates there) reads its parent
/// entry through `1.0 · 1.0 · W` — bitwise `W`.
#[derive(Debug, Clone)]
pub struct AoTranspose {
    /// `start[m]..start[m+1]` indexes `idx` / `t` for rotated AO `m`.
    start: Vec<usize>,
    /// Parent AOs and their `T_μm`.
    idx: Vec<usize>,
    t: Vec<f64>,
    /// TEST mutant: the second index is not transformed.
    one_sided: bool,
}

impl AoTranspose {
    /// `Σ_{μν} T_μm T_νn f(μ, ν)` for rotated AOs `(m, n)`.
    #[inline]
    pub fn pair<F: Fn(usize, usize) -> f64>(&self, m: usize, n: usize, f: F) -> f64 {
        let (a0, a1) = (self.start[m], self.start[m + 1]);
        let mut acc = 0.0;
        if self.one_sided {
            for a in a0..a1 {
                acc += self.t[a] * f(self.idx[a], n);
            }
            return acc;
        }
        let (b0, b1) = (self.start[n], self.start[n + 1]);
        for a in a0..a1 {
            for b in b0..b1 {
                acc += self.t[a] * self.t[b] * f(self.idx[a], self.idx[b]);
            }
        }
        acc
    }

    /// `W'[mn, p] = Σ_{μν} T_μm T_νn W[μν, p]` of a PAIR-MAJOR real weight
    /// `(n², w)` (row `μ n + ν`; the k-point SR residue bins `Zbin`): the
    /// forward transform of every column. `swap` (TEST mutant) reads the
    /// pair transposed, `W[νμ, p]`. Elementwise independent, so bitwise
    /// reproducible at any thread count.
    pub fn pair_rows(&self, z: &Array2<f64>, n: usize, swap: bool) -> Array2<f64> {
        let w = z.ncols();
        let mut out = Array2::<f64>::zeros((n * n, w));
        for m in 0..n {
            for k in 0..n {
                for p in 0..w {
                    out[(m * n + k, p)] = self.pair(m, k, |a, b| {
                        if swap {
                            z[(b * n + a, p)]
                        } else {
                            z[(a * n + b, p)]
                        }
                    });
                }
            }
        }
        out
    }

    /// `Tᵀ D T` of an `(n, n)` matrix (any `D`, symmetric or not).
    pub fn matrix(&self, d: &Array2<f64>) -> Array2<f64> {
        let n = d.nrows();
        Array2::from_shape_fn((n, n), |(m, k)| self.pair(m, k, |a, b| d[(a, b)]))
    }
}

impl RotatedBasis {
    /// The rotation of `prep`, built from `mol` and its own `basis_set()`
    /// (one rotation per element), or `None` when nothing rotates (the
    /// identity: the caller then runs the unrotated walk, bit for bit). `who`
    /// names the caller in errors. The rotated basis is
    /// `PreparedBasis::new(mol, rotated templates)`.
    pub fn detect(
        mol: &Molecule,
        prep: &PreparedBasis,
        mutant: ColumnRotationMutant,
        level: RotationLevel,
        who: &str,
    ) -> Result<Option<Self>, FerricError> {
        let bs = prep.basis_set();
        let per_z: HashMap<i32, TemplateRotation> = bs
            .shells
            .iter()
            .map(|(z, t)| (*z, rotate_templates(t, mutant, level)))
            .collect();
        let located = prep.located_shells();
        let nsh = prep.nshells();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nsh);
        let mut group_id: Vec<usize> = Vec::with_capacity(nsh);
        let (mut base, mut gbase, mut n_rot) = (0usize, 0usize, 0usize);
        for atom in &mol.atoms {
            let tr = per_z.get(&atom.z).ok_or_else(|| {
                FerricError::Basis(format!(
                    "{who} column rotation: no orbital shells for Z = {} in {:?}",
                    atom.z, bs.name
                ))
            })?;
            let tmpls = bs.for_element(atom.z).unwrap_or(&[]);
            for (m, t) in tmpls.iter().enumerate() {
                let same = located.get(base + m).is_some_and(|p| {
                    p.l == t.l
                        && p.pure == t.pure
                        && p.exponents.len() == t.exponents.len()
                        && p.coefficients.len() == t.coefficients.len()
                        && p.exponents
                            .iter()
                            .zip(&t.exponents)
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                        && p.coefficients
                            .iter()
                            .zip(&t.coefficients)
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                });
                if !same {
                    return Err(FerricError::General(format!(
                        "{who} column rotation: shell {} does not match cell.mol() + its BasisSet \
                         (build the PreparedBasis from cell.mol())",
                        base + m
                    )));
                }
                rows.push(tr.rows[m].iter().map(|&(j, c)| (base + j, c)).collect());
                group_id.push(gbase + tr.group_of[m]);
            }
            base += tmpls.len();
            gbase += tr.n_groups;
            n_rot += tr.n_rotated;
        }
        if base != nsh {
            return Err(FerricError::General(format!(
                "{who} column rotation: cell.mol() + BasisSet give {base} shells, the basis has {nsh}"
            )));
        }
        if n_rot == 0 {
            return Ok(None);
        }
        let shells: HashMap<i32, Vec<Shell>> =
            per_z.into_iter().map(|(z, tr)| (z, tr.shells)).collect();
        let rbs = BasisSet {
            name: format!("{} (SR column rotation)", bs.name),
            shells,
            ecps: bs.ecps.clone(),
        };
        let rprep = PreparedBasis::new(mol, &rbs)?;
        Self::assemble(prep, rprep, rows, &group_id, (gbase, n_rot), mutant, who).map(Some)
    }

    /// The production rotation of `prep` at `level`, derived from the
    /// prepared shells alone (one rotation per run of shells on one atom; no
    /// molecule or element templates needed, so ghost atoms and per-atom
    /// bases are handled), or `None` when nothing rotates. The rotated
    /// basis is [`PreparedBasis::with_contractions`] of `prep`.
    pub fn detect_prepared(
        prep: &PreparedBasis,
        level: RotationLevel,
    ) -> Result<Option<Self>, FerricError> {
        let located = prep.located_shells();
        let s2a = prep.shell_to_atom();
        let nsh = prep.nshells();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nsh);
        let mut group_id: Vec<usize> = Vec::with_capacity(nsh);
        let mut rshells: Vec<Shell> = Vec::with_capacity(nsh);
        let (mut base, mut gbase, mut n_rot) = (0usize, 0usize, 0usize);
        // Atoms with the same shell list rotate the same way: compute each
        // distinct list once.
        let mut cache: Vec<(Vec<Shell>, TemplateRotation)> = Vec::new();
        while base < nsh {
            let atom = s2a[base];
            let end = (base..nsh).find(|&s| s2a[s] != atom).unwrap_or(nsh);
            let tmpls: Vec<Shell> = located[base..end]
                .iter()
                .map(|p| Shell {
                    l: p.l,
                    pure: p.pure,
                    exponents: p.exponents.to_vec(),
                    coefficients: p.coefficients.to_vec(),
                })
                .collect();
            let hit = cache.iter().position(|(t, _)| {
                t.len() == tmpls.len() && t.iter().zip(&tmpls).all(|(a, b)| same_shell(a, b))
            });
            let idx = match hit {
                Some(i) => i,
                None => {
                    let tr = rotate_templates(&tmpls, ColumnRotationMutant::Production, level);
                    cache.push((tmpls, tr));
                    cache.len() - 1
                }
            };
            let tr = &cache[idx].1;
            for m in 0..(end - base) {
                rows.push(tr.rows[m].iter().map(|&(j, c)| (base + j, c)).collect());
                group_id.push(gbase + tr.group_of[m]);
                rshells.push(tr.shells[m].clone());
            }
            gbase += tr.n_groups;
            n_rot += tr.n_rotated;
            base = end;
        }
        if n_rot == 0 {
            return Ok(None);
        }
        let name = format!("{} (column rotation)", prep.basis_set().name);
        let rprep = prep.with_contractions(&rshells, &name)?;
        Self::assemble(
            prep,
            rprep,
            rows,
            &group_id,
            (gbase, n_rot),
            ColumnRotationMutant::Production,
            "column rotation",
        )
        .map(Some)
    }

    /// The rotation record from the rows of `T` and each shell's group id
    /// (ids increase with the group's first member), after checking that the
    /// rotated basis keeps the parent's layout.
    fn assemble(
        prep: &PreparedBasis,
        rprep: PreparedBasis,
        mut rows: Vec<Vec<(usize, f64)>>,
        group_id: &[usize],
        (ngroups, n_rot): (usize, usize),
        mutant: ColumnRotationMutant,
        who: &str,
    ) -> Result<Self, FerricError> {
        let nsh = prep.nshells();
        let (offs_all, dims) = (prep.shell_offsets(), prep.shell_dims());
        if rprep.nshells() != nsh
            || rprep.nbasis() != prep.nbasis()
            || rprep.shell_dims() != dims
            || rprep.shell_offsets() != offs_all
        {
            return Err(FerricError::General(format!(
                "{who} column rotation: the rotated basis has another layout \
                 ({} shells / {} AOs vs {nsh} / {})",
                rprep.nshells(),
                rprep.nbasis(),
                prep.nbasis()
            )));
        }
        // Groups in order of first member (group ids increase with the
        // first member's shell index by construction).
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ngroups];
        for (k, &g) in group_id.iter().enumerate() {
            groups[g].push(k);
        }
        let mut slot = vec![(0usize, 0usize); nsh];
        let mut identity = vec![true; ngroups];
        for (g, members) in groups.iter().enumerate() {
            let mut start = 0usize;
            for &k in members {
                slot[k] = (g, start);
                start += dims[k];
                if !(rows[k].len() == 1 && rows[k][0] == (k, 1.0)) {
                    identity[g] = false;
                }
            }
        }
        if mutant == ColumnRotationMutant::TransposedBackTransform {
            // MUTATION: read every row of T as a column (T where Tᵀ belongs).
            let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nsh];
            for (k, row) in rows.iter().enumerate() {
                for &(m, t) in row {
                    cols[m].push((k, t));
                }
            }
            rows = cols;
        }
        Ok(Self {
            prep: rprep,
            rows,
            groups,
            identity,
            slot,
            offs: offs_all[..nsh].to_vec(),
            dims: dims.to_vec(),
            nbasis: prep.nbasis(),
            n_rotated_columns: n_rot,
            one_sided: mutant == ColumnRotationMutant::OneSidedBackTransform,
            skip_back: mutant == ColumnRotationMutant::NoBackTransform,
        })
    }

    /// Number of AO functions (parent and rotated alike).
    pub fn nbasis(&self) -> usize {
        self.nbasis
    }

    /// The row of `T` of parent shell `k`: `(rotated shell, T_km)`,
    /// diagonal first.
    pub fn row(&self, k: usize) -> &[(usize, f64)] {
        &self.rows[k]
    }

    /// The dense `nbasis × nbasis` matrix `T` (`χ_μ = Σ_m T_μm χ'_m`), for
    /// tests and diagnostics.
    pub fn dense_t(&self) -> Array2<f64> {
        let mut t = Array2::<f64>::zeros((self.nbasis, self.nbasis));
        for (k, row) in self.rows.iter().enumerate() {
            for &(m, tkm) in row {
                for i in 0..self.dims[k] {
                    t[(self.offs[k] + i, self.offs[m] + i)] = tkm;
                }
            }
        }
        t
    }

    /// AOs of group `g`, member by member.
    fn group_aos(&self, g: usize) -> Vec<usize> {
        self.groups[g]
            .iter()
            .flat_map(|&k| self.offs[k]..self.offs[k] + self.dims[k])
            .collect()
    }

    /// In-place back-transform of a pair-row tensor from the rotated to the
    /// parent AO basis: `data` is `(n², w)` row-major, row `μ n + ν`,
    /// EXACTLY symmetric under `μ ↔ ν` on input (the s2 walks), and on
    /// output holds `Σ_{m,n} T_μm T_νn data'[mn]`, exactly symmetric.
    ///
    /// Serial over (group, group) pairs `g1 ≤ g2` (the row set of a group
    /// pair, both orientations, is closed under the transform): the pair's
    /// rotated rows are gathered into a scratch first, then every unordered
    /// function pair is formed with the fixed addend order "row of μ, then
    /// row of ν" from `+0.0` and COPIED into both rows. Group pairs whose
    /// rows are all the identity are skipped (untouched, bit for bit).
    pub fn back_transform_pair_rows(&self, data: &mut [f64], w: usize) -> Result<(), FerricError> {
        self.back_transform_impl(data, w, true)
    }

    /// [`RotatedBasis::back_transform_pair_rows`] for a tensor that is NOT
    /// symmetric under `μ ↔ ν` (the k-point SR bins, `B[r_L, r_T][μν] =
    /// B[M(r_L, r_T)][νμ]` relates two DIFFERENT bins; Bloch-summed
    /// matrices; the SCF's J and K): `out[μν] = Σ_{m,n} T_μm T_νn data[mn]`,
    /// each element formed from its own two rows with the same fixed addend
    /// order (row of `μ`, then row of `ν`), serial and deterministic. The
    /// pair-orientation relation between bins survives to round-off only
    /// (the sums run in a different order for the two halves).
    pub fn back_transform_rows(&self, data: &mut [f64], w: usize) -> Result<(), FerricError> {
        self.back_transform_impl(data, w, false)
    }

    /// Shared body: `symmetric` = exactly symmetric input, unordered pairs
    /// written into both rows; otherwise every ordered pair is its own
    /// output element.
    fn back_transform_impl(
        &self,
        data: &mut [f64],
        w: usize,
        symmetric: bool,
    ) -> Result<(), FerricError> {
        let n = self.nbasis;
        if data.len() != n * n * w {
            return Err(FerricError::General(format!(
                "column rotation back-transform: {} values, expected {n}² × {w}",
                data.len()
            )));
        }
        if self.skip_back {
            return Ok(());
        }
        let ng = self.groups.len();
        let mut scratch: Vec<f64> = Vec::new();
        let mut out = vec![0.0_f64; w];
        for g1 in 0..ng {
            let f1 = self.group_aos(g1);
            for g2 in if symmetric { g1 } else { 0 }..ng {
                if self.identity[g1] && self.identity[g2] {
                    continue;
                }
                let f2 = self.group_aos(g2);
                let l2 = f2.len();
                scratch.clear();
                scratch.resize(f1.len() * l2 * w, 0.0);
                for (a, &mu) in f1.iter().enumerate() {
                    for (b, &nu) in f2.iter().enumerate() {
                        let (src, dst) = ((mu * n + nu) * w, (a * l2 + b) * w);
                        scratch[dst..dst + w].copy_from_slice(&data[src..src + w]);
                    }
                }
                for (i1, &k1) in self.groups[g1].iter().enumerate() {
                    for (i2, &k2) in self.groups[g2].iter().enumerate() {
                        if symmetric && g1 == g2 && i2 < i1 {
                            continue;
                        }
                        let ident = [(k2, 1.0)];
                        let row2: &[(usize, f64)] = if self.one_sided {
                            &ident
                        } else {
                            &self.rows[k2]
                        };
                        for i in 0..self.dims[k1] {
                            for j in 0..self.dims[k2] {
                                if symmetric && k1 == k2 && j < i {
                                    continue;
                                }
                                out.iter_mut().for_each(|x| *x = 0.0);
                                for &(m1, t1) in &self.rows[k1] {
                                    let a = self.slot[m1].1 + i;
                                    for &(m2, t2) in row2 {
                                        let b = self.slot[m2].1 + j;
                                        let t = t1 * t2;
                                        let src = (a * l2 + b) * w;
                                        for (o, &x) in out.iter_mut().zip(&scratch[src..src + w]) {
                                            *o += t * x;
                                        }
                                    }
                                }
                                let (mu, nu) = (self.offs[k1] + i, self.offs[k2] + j);
                                let ra = (mu * n + nu) * w;
                                data[ra..ra + w].copy_from_slice(&out);
                                if symmetric {
                                    let rb = (nu * n + mu) * w;
                                    data[rb..rb + w].copy_from_slice(&out);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The AO-level transpose map of the forward transforms (`D' = Tᵀ D T`;
    /// the periodic derivative paths' weights): for each rotated AO `m`, the
    /// parent AOs `μ` whose row of `T` holds `m`, with `T_μm`. `mutant` is a
    /// TEST-ONLY defect of the forward transform ([`DerivRotMutant`]).
    pub fn ao_transpose(&self, mutant: DerivRotMutant) -> AoTranspose {
        let nsh = self.rows.len();
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nsh];
        match mutant {
            DerivRotMutant::DropT => {
                for (m, c) in cols.iter_mut().enumerate() {
                    c.push((m, 1.0));
                }
            }
            DerivRotMutant::UseT => {
                // MUTATION: T applied where Tᵀ belongs (row read as column).
                for (m, c) in cols.iter_mut().enumerate() {
                    c.extend(self.rows[m].iter().copied());
                }
            }
            DerivRotMutant::None | DerivRotMutant::OneSided | DerivRotMutant::FlipSign => {
                for (k, row) in self.rows.iter().enumerate() {
                    for &(m, t) in row {
                        let t = if mutant == DerivRotMutant::FlipSign && m != k {
                            -t
                        } else {
                            t
                        };
                        cols[m].push((k, t));
                    }
                }
            }
        }
        let mut start = Vec::with_capacity(self.nbasis + 1);
        let (mut idx, mut t) = (Vec::new(), Vec::new());
        for (m, c) in cols.iter().enumerate() {
            for i in 0..self.dims[m] {
                start.push(idx.len());
                for &(k, tk) in c {
                    debug_assert_eq!(self.dims[k], self.dims[m]);
                    idx.push(self.offs[k] + i);
                    t.push(tk);
                }
            }
        }
        start.push(idx.len());
        debug_assert_eq!(start.len(), self.nbasis + 1);
        AoTranspose {
            start,
            idx,
            t,
            one_sided: mutant == DerivRotMutant::OneSided,
        }
    }

    /// [`RotatedBasis::back_transform_pair_rows`] of an exactly symmetric
    /// `(n, n)` matrix (`V_SR`): `T V' Tᵀ`, exactly symmetric.
    pub fn back_transform_matrix(&self, v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let mut out = v.as_standard_layout().into_owned();
        let data = out.as_slice_mut().ok_or_else(|| {
            FerricError::General("column rotation back-transform: matrix not contiguous".into())
        })?;
        self.back_transform_pair_rows(data, 1)?;
        Ok(out)
    }

    /// [`RotatedBasis::back_transform_rows`] of an `(n, n)` matrix that need
    /// not be symmetric: `T X' Tᵀ`.
    pub fn back_transform_general(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let mut out = x.as_standard_layout().into_owned();
        let data = out.as_slice_mut().ok_or_else(|| {
            FerricError::General("column rotation back-transform: matrix not contiguous".into())
        })?;
        self.back_transform_rows(data, 1)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
