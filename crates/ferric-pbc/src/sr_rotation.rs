//! Column rotation of generally contracted orbital shells for the Gamma
//! short-range integral walks (design: `reference/pbc/sr-general-contraction-design.md`
//! §3 option (d) and §4 step 2).
//!
//! # What is rotated
//!
//! Correlation-consistent bases store the contraction COLUMNS of an atom's
//! s (and p) set as separate shells over one shared exponent list; the most
//! diffuse primitive is also a column of its own (cc-pVDZ C: s 9 primitives
//! in columns with 9, 9, 1 non-zero coefficients; p 4 primitives in columns
//! with 4, 1). A GROUP is the shells of one element template with the same
//! `l`, the same `pure` flag and bitwise the same exponent list. Inside a
//! group, every column `k` with more than one non-zero coefficient whose
//! support contains the primitive `q` of a single-primitive column `s` of
//! the group is replaced by
//!
//! ```text
//! k' = k − (c_kq / c_sq) s        (raw coefficients: c_kq set to exactly 0)
//! ```
//!
//! with its zero primitives DROPPED (the single-primitive columns used are
//! zero-dropped too). The span of the group is unchanged, and every rotated
//! column is then either wholly compact or wholly smooth at the default
//! range split (ω = 1, λ = 1), so the split walk needs one kept call per
//! pair instead of two, and each call has fewer primitives.
//!
//! # The back-transform (exact)
//!
//! libint renormalises every contraction it is given (`N = s(c)^{-1/2}`,
//! `s` = self-overlap over unit primitives, the range split's convention).
//! With `χ_k = N_k Σ_p c_kp g_p` for the parent and `χ'_k = N'_k Σ_{p∉Q} c_kp g_p`
//! for the rotated column (`Q` the removed primitives),
//!
//! ```text
//! χ_k = T_kk χ'_k + Σ_{s} T_ks χ_s,
//! T_kk = √(s(c'_k) / s(c_k)),     T_ks = (c_kq / c_sq) √(s(c_s) / s(c_k)),
//! ```
//!
//! a unit-lower-triangular-like map (diagonal `T_kk ≈ 1`) inside each group,
//! identity elsewhere. Every SR block is bilinear in the two orbitals and the
//! range split's compact/smooth partition is linear in the coefficient
//! vector over a fixed primitive set, so the SR sums transform EXACTLY:
//! `J3[μν, P] = Σ_{m,n} T_μm T_νn J3'[mn, P]`, `V_SR = T V'_SR Tᵀ` (function
//! by function: rotated shell `m` has the same `l` and AO layout as parent
//! shell `m`). The screen runs on the rotated shells' own bounds
//! (coefficients `c'` times `prim_norm`, i.e. the size of `T_kk χ'_k` in
//! parent units), so the kept triplet set differs from the unrotated walk's:
//! the result agrees with the unrotated build to the screening precision,
//! not bitwise.
//!
//! # Where it applies
//!
//! Only inside the Gamma SR 3-centre walk of [`crate::rsgdf::RsGdf::build`]
//! (unsplit and range split) and the Gamma hcore SR attraction of
//! [`crate::hcore::periodic_hcore`]: the rotated blocks are back-transformed
//! into the parent AO basis right after the walk. The SCF, S/T, the LR and
//! G = 0 parts, the metric and every exposed matrix stay in the parent
//! basis. The k-point builds and the force / stress walks do not apply it.
//!
//! It is ON BY DEFAULT ([`SrColumnRotation::Auto`]) in those two Gamma
//! energy builds and silently off in every build that does not implement it
//! (k-point builds, the gradient build, the frozen s1 oracles); an explicit
//! [`SrColumnRotation::On`] is refused by name there. A run that computes
//! forces or stress builds its hcore with the rotation off
//! ([`SrColumnRotation::for_derivatives`]); the Gamma force and stress
//! builders refuse a rotated `PeriodicHcore`. Measured (FINDINGS, "Full
//! timing series on libint 2.13.1", Γ RHF cc-pVDZ / cc-pvdz-ri, range
//! split, gdf ω = 1, 6 threads): dry ice 25.9 → 22.4 s, diamond 7.84 →
//! 3.08 s; energies move ≤ 1.5e-11 Ha/cell at ω ≥ 0.7; bitwise across
//! thread counts.
//!
//! # Numerics
//!
//! * The back-transform is a fixed serial loop over (group, group) pairs on
//!   an exactly symmetric input; it writes each unordered function pair once
//!   and COPIES it into both `(μ, ν)` and `(ν, μ)`, so the output is exactly
//!   symmetric and bitwise identical across thread counts (the walks already
//!   are).
//! * Identity: when nothing in the basis rotates (segmented bases, STO-3G,
//!   Pople sp shells — no group has a single-primitive column), detection
//!   returns `None` and the build runs today's code path bit for bit.
//! * Guard: a column whose `T_kk < `[`COLUMN_ROTATION_MIN_DIAG`] or with an
//!   `|T_ks| > `[`COLUMN_ROTATION_MAX_COEF`] is left unrotated (its row of
//!   `T` stays the identity); cc-pVDZ C has `T_ks` 0.0032 and 0.58.

use crate::lattice::Cell;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use std::collections::HashMap;

/// Smallest diagonal `T_kk` a rotated column may have (below it the rotated
/// column is nearly the removed one and the transform would amplify
/// round-off by `1/T_kk`).
pub const COLUMN_ROTATION_MIN_DIAG: f64 = 1e-3;
/// Largest `|T_ks|` a rotated column may have (the design's `cond(R) ≤ 1e3`).
pub const COLUMN_ROTATION_MAX_COEF: f64 = 1e3;

/// Production rotation or a deliberately BROKEN variant (negative controls
/// for `tests/pbc_sr_rotation.rs`; an anchor that cannot tell them from
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
}

/// The column rotation a Gamma SR walk runs
/// ([`SrColumnRotation::On`]; production or a mutation-test variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRotation {
    /// [`ColumnRotationMutant::Production`] except in mutation tests.
    pub mutant: ColumnRotationMutant,
}

impl ColumnRotation {
    /// The production rotation.
    pub const fn new() -> Self {
        Self {
            mutant: ColumnRotationMutant::Production,
        }
    }
}

/// The column-rotation request of a build
/// ([`crate::rsgdf::RsGdfConfig::sr_column_rotation`],
/// [`crate::hcore::PeriodicHcoreConfig::sr_column_rotation`]).
///
/// * [`SrColumnRotation::Auto`] (the default): ON in the builds that
///   implement it — the Gamma energy builds [`crate::rsgdf::RsGdf::build`]
///   (and `build_with_fit_parts`, [`crate::rsgdf::sr_walk_counts`]) and
///   [`crate::hcore::periodic_hcore`] — and silently OFF in every build that
///   does not: the k-point builds, [`crate::rsgdf::RsGdf::build_for_gradient`]
///   and the frozen s1 oracles. A basis with nothing to rotate runs the
///   unrotated walk bit for bit either way.
/// * [`SrColumnRotation::Off`]: the unrotated walk everywhere, bit for bit.
/// * [`SrColumnRotation::On`]: an explicit request; the builds that cannot
///   honour it refuse it by name instead of ignoring it.
///
/// A run that computes forces or stress must build its Gamma hcore with the
/// rotation OFF ([`SrColumnRotation::for_derivatives`]): `periodic_hcore`
/// cannot know a gradient will follow, and the Gamma force and stress
/// builders refuse a `PeriodicHcore` whose SR attraction was rotated (they
/// differentiate the unrotated walk, whose truncated energy differs at the
/// screening precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SrColumnRotation {
    /// On where implemented, off elsewhere (see the type doc).
    #[default]
    Auto,
    /// The unrotated walk everywhere.
    Off,
    /// Explicit request (refused where it cannot be honoured).
    On(ColumnRotation),
}

impl SrColumnRotation {
    /// Explicit production rotation.
    pub const fn on() -> Self {
        Self::On(ColumnRotation::new())
    }

    /// Whether this is an explicit [`SrColumnRotation::On`].
    pub fn is_explicit_on(self) -> bool {
        matches!(self, Self::On(_))
    }

    /// The rotation a build that IMPLEMENTS it runs: `Auto` → the production
    /// rotation, `On(r)` → `r`, `Off` → `None`.
    pub fn resolve_supported(self) -> Option<ColumnRotation> {
        match self {
            Self::Auto => Some(ColumnRotation::new()),
            Self::Off => None,
            Self::On(r) => Some(r),
        }
    }

    /// For a build that does NOT implement it: `Auto` and `Off` resolve to
    /// off (`Ok(())`); an explicit `On` is a typed refusal naming `who` and
    /// `why`.
    pub fn refuse_explicit(self, who: &str, why: &str) -> Result<(), FerricError> {
        match self {
            Self::Auto | Self::Off => Ok(()),
            Self::On(r) => Err(FerricError::General(format!(
                "{who}: sr_column_rotation {:?} was requested explicitly, but {why} \
                 (use SrColumnRotation::Auto or Off)",
                r.mutant
            ))),
        }
    }

    /// The request for a run that will compute forces or stress: `Auto` and
    /// `Off` → `Off` (so the energy and the derivative walk the same
    /// unrotated shells); an explicit `On` is refused (`who` names the run).
    pub fn for_derivatives(self, who: &str) -> Result<Self, FerricError> {
        self.refuse_explicit(
            who,
            "the Gamma force and stress builders walk the unrotated shells, so they would \
             not differentiate the rotated energy",
        )?;
        Ok(Self::Off)
    }
}

impl Default for ColumnRotation {
    /// Same as [`ColumnRotation::new`].
    fn default() -> Self {
        Self::new()
    }
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

/// One element's rotated templates: the libint shells, the rows of `T`
/// (template indices, diagonal first), each template shell's group, and the
/// number of rotated columns.
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

/// Rotate one element's template shells (module doc, "What is rotated").
fn rotate_templates(tmpls: &[Shell], mutant: ColumnRotationMutant) -> TemplateRotation {
    let n = tmpls.len();
    let mut shells = tmpls.to_vec();
    let mut rows: Vec<Vec<(usize, f64)>> = (0..n).map(|k| vec![(k, 1.0)]).collect();
    let mut group_of = vec![usize::MAX; n];
    let (mut n_groups, mut n_rotated) = (0usize, 0usize);
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
    TemplateRotation {
        shells,
        rows,
        group_of,
        n_groups,
        n_rotated,
    }
}

/// A column-rotated orbital (or, for the aux mutant, auxiliary) basis and
/// the exact back-transform to its parent (module doc).
pub(crate) struct RotatedBasis {
    /// The rotated libint basis: the parent's atoms, shell order, `l`,
    /// `pure` and AO layout (checked), rotated contractions.
    pub(crate) prep: PreparedBasis,
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
    pub(crate) n_rotated_columns: usize,
    one_sided: bool,
}

impl RotatedBasis {
    /// The rotation of `prep` (built from `cell.mol()` and its own
    /// `basis_set()`), or `None` when nothing rotates (the identity: the
    /// caller then runs the unrotated walk, bit for bit). `who` names the
    /// caller in errors.
    pub(crate) fn detect(
        cell: &Cell,
        prep: &PreparedBasis,
        rot: ColumnRotation,
        who: &str,
    ) -> Result<Option<Self>, FerricError> {
        let bs = prep.basis_set();
        let per_z: HashMap<i32, TemplateRotation> = bs
            .shells
            .iter()
            .map(|(z, t)| (*z, rotate_templates(t, rot.mutant)))
            .collect();
        let located = prep.located_shells();
        let (offs_all, dims) = (prep.shell_offsets(), prep.shell_dims());
        let nsh = prep.nshells();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(nsh);
        let mut group_id: Vec<usize> = Vec::with_capacity(nsh);
        let (mut base, mut gbase, mut n_rot) = (0usize, 0usize, 0usize);
        for atom in &cell.mol().atoms {
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
        let rprep = PreparedBasis::new(cell.mol(), &rbs)?;
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
        let ngroups = gbase;
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
        Ok(Some(Self {
            prep: rprep,
            rows,
            groups,
            identity,
            slot,
            offs: offs_all[..nsh].to_vec(),
            dims: dims.to_vec(),
            nbasis: prep.nbasis(),
            n_rotated_columns: n_rot,
            one_sided: rot.mutant == ColumnRotationMutant::OneSidedBackTransform,
        }))
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
    pub(crate) fn back_transform_pair_rows(
        &self,
        data: &mut [f64],
        w: usize,
    ) -> Result<(), FerricError> {
        let n = self.nbasis;
        if data.len() != n * n * w {
            return Err(FerricError::General(format!(
                "column rotation back-transform: {} values, expected {n}² × {w}",
                data.len()
            )));
        }
        let ng = self.groups.len();
        let mut scratch: Vec<f64> = Vec::new();
        let mut out = vec![0.0_f64; w];
        for g1 in 0..ng {
            let f1 = self.group_aos(g1);
            for g2 in g1..ng {
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
                        if g1 == g2 && i2 < i1 {
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
                                if k1 == k2 && j < i {
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
                                let (ra, rb) = ((mu * n + nu) * w, (nu * n + mu) * w);
                                data[ra..ra + w].copy_from_slice(&out);
                                data[rb..rb + w].copy_from_slice(&out);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// [`RotatedBasis::back_transform_pair_rows`] of an exactly symmetric
    /// `(n, n)` matrix (`V_SR`): `T V' Tᵀ`, exactly symmetric.
    pub(crate) fn back_transform_matrix(
        &self,
        v: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        let mut out = v.as_standard_layout().into_owned();
        let data = out.as_slice_mut().ok_or_else(|| {
            FerricError::General("column rotation back-transform: matrix not contiguous".into())
        })?;
        self.back_transform_pair_rows(data, 1)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Cartesian (non-pure) shell of angular momentum `l` with the given
    /// exponents `e` and coefficients `c`.
    fn sh(l: i32, e: &[f64], c: &[f64]) -> Shell {
        Shell {
            l,
            pure: false,
            exponents: e.to_vec(),
            coefficients: c.to_vec(),
        }
    }

    /// The rotated column reproduces the parent as a function, primitive by
    /// primitive over the libint-normalised shells:
    /// `c_k / √s(c_k) = T_kk c'_k / √s(c'_k) + T_ks c_s / √s(c_s)`.
    #[test]
    fn rotated_row_reconstructs_the_parent_column() {
        let e = [30.0, 5.0, 1.0, 0.2];
        let big = sh(0, &e, &[0.1, 0.4, 0.5, 0.3]);
        let single = sh(0, &e, &[0.0, 0.0, 0.0, 1.3]);
        let t = rotate_templates(&[big.clone(), single], ColumnRotationMutant::Production);
        assert_eq!(t.n_rotated, 1);
        assert_eq!(t.shells[0].exponents, vec![30.0, 5.0, 1.0]);
        assert_eq!(t.shells[1].exponents, vec![0.2]);
        let row = &t.rows[0];
        assert_eq!(row[0].0, 0);
        assert_eq!(row[1].0, 1);
        let norm = |s: &Shell| contraction_norm2(s.l, &s.exponents, &s.coefficients).sqrt();
        let (nk, nr, ns) = (norm(&big), norm(&t.shells[0]), norm(&t.shells[1]));
        for p in 0..4 {
            let parent = big.coefficients[p] / nk;
            let rot = if p < 3 {
                t.shells[0].coefficients[p] / nr
            } else {
                0.0
            };
            let s = if p == 3 {
                t.shells[1].coefficients[0] / ns
            } else {
                0.0
            };
            let got = row[0].1 * rot + row[1].1 * s;
            assert!(
                (got - parent).abs() <= 1e-15,
                "primitive {p}: {got} vs {parent}"
            );
        }
        // Nothing to rotate: identity rows, templates untouched.
        let seg = rotate_templates(
            &[big, sh(1, &e, &[0.0, 0.0, 0.0, 1.0])],
            ColumnRotationMutant::Production,
        );
        assert_eq!(seg.n_rotated, 0);
        assert!(seg
            .rows
            .iter()
            .enumerate()
            .all(|(k, r)| r == &vec![(k, 1.0)]));
    }
}
