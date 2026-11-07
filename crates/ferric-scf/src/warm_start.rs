//! SCF warm start across nearby geometries (geometry optimization).
//!
//! A geometry driver solves one SCF per evaluated geometry, and consecutive
//! geometries differ by at most the optimizer's trust radius. Re-guessing each
//! one from MINAO throws away a solution that is already within a small
//! orbital rotation of the new one. [`ScfWarmStart`] keeps the previous
//! converged MOs and turns them into a guess at the next geometry.
//!
//! # Re-orthonormalizing the old MOs, not reusing the old density
//!
//! The AO basis moves with the atoms, so the old coefficients `C` are no
//! longer orthonormal in the new overlap `S'`: `M = C_occᵀ S' C_occ ≠ 1`.
//! Feeding the old density matrix in unchanged gives `Tr(D S') ≠ N` and a
//! non-idempotent `D S' D ≠ D`. Instead the occupied block is Löwdin
//! orthonormalized in the new metric, `C_occ' = C_occ M^{-1/2}` — the
//! orthonormal set CLOSEST to the old orbitals in the `S'` norm, so it changes
//! the occupied space as little as any orthonormalization can — and the guess
//! density built from `C_occ'` has the right electron count and is idempotent
//! by construction. Open-shell references are orthonormalized block by block
//! in occupation order (UHF: α-occupied, then the rest, separately per spin;
//! ROHF: doubly occupied, singly occupied, virtual), each block first
//! projected against the blocks before it, so the α/β (or docc/socc) partition
//! of the previous solution is carried over rather than mixed. Virtual columns
//! are carried the same way (projected, then Löwdin) because the UHF/ROHF
//! solvers take a full `(n, n)` MO matrix; only the occupied blocks decide the
//! guess density.
//!
//! # When the cold guess is used instead ([`ColdReason`])
//!
//! First geometry; a change of atoms, charge, multiplicity or basis size; a
//! reference other than the one recorded; a symmetry-blocked solve (which
//! refuses an explicit density); and a POOR OVERLAP: the occupied metric `M`
//! has an eigenvalue farther than [`MAX_METRIC_DEVIATION`] from 1. `M = 1`
//! exactly under a rigid translation or rotation (the AOs move with the atoms
//! and `S` is unchanged), and its eigenvalues leave 1 only as the molecule
//! DEFORMS, so `max |λ(M) − 1|` measures how far the old occupied orbitals are
//! from being a valid orbital set at the new geometry. A failed warm-started
//! SCF is retried from the cold guess by the caller (it never reaches a ladder
//! or a convergence error without the cold attempt), and only a CONVERGED SCF
//! is ever recorded.
//!
//! # What a warm start does NOT change
//!
//! The solver and its settings: MOM, level shifts, DIIS, Newton/TRAH
//! triggers and stability analysis all run exactly as configured, from a
//! different starting point. Two side effects of starting from MOs: the
//! secondary-basis guess (`guess_basis`) is skipped, and a forced UHF on a
//! closed shell (`nα = nβ`) still gets the solver's small β HOMO/LUMO mix,
//! applied to the carried orbitals. Continuity cuts both ways: the warm SCF
//! follows the previous geometry's state, which is what a geometry search
//! wants, but on a surface where the cold guess would land on a different
//! (lower) state the warm start keeps the old one, as any carried guess does.

use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// Largest `|λ − 1|` over the eigenvalues `λ` of the occupied metric
/// `C_occᵀ S' C_occ` for which the previous geometry's MOs are used.
///
/// Measured on water (unit test
/// `metric_deviation_separates_optimizer_steps_from_big_moves`, one H moved):
/// a 0.1 Bohr move (the default trust radius, all on one atom) gives
/// 1.9e-2 / 1.8e-2 / 2.1e-2 at STO-3G / 6-31G / cc-pVDZ, a 1 Bohr move
/// 0.18 / 0.16 / 0.19. 0.1 sits between the two with a factor of about 5
/// below and 1.6 above; beyond it the old orbitals have lost or gained 10% of
/// their norm along some occupied direction and MINAO is the more
/// trustworthy start. The threshold is a guard against a bad guess, not a
/// tuned cost model: a warm attempt that fails is retried cold anyway.
///
/// Also measured (`warm_matches_cold_along_a_path`): the saving is set by the
/// DIIS convergence RATE, not by how close the guess is. From a 0.003 Bohr
/// step the warm SCF starts at a commutator of 5e-4 and still needs about 6
/// iterations to `density_conv = 1e-6` (cold: 9-10), so the warm start saves
/// a few iterations per geometry rather than most of them.
pub const MAX_METRIC_DEVIATION: f64 = 0.1;

/// Smallest eigenvalue of a (projected) block metric that is still inverted.
/// Below it the carried columns are numerically dependent in the new basis
/// (e.g. the new geometry dropped a direction to the lindep filter) and the
/// cold guess is used.
const MIN_BLOCK_EIGENVALUE: f64 = 1e-6;

/// Why a geometry's SCF starts from the cold (default) guess.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColdReason {
    /// Warm start is switched off.
    Disabled,
    /// Nothing recorded yet (the first geometry).
    FirstGeometry,
    /// Atoms, charge, multiplicity, reference or basis size differ from the
    /// recorded solution.
    SystemChanged,
    /// `RhfConfig::symmetry` is set; the blocked solver refuses an explicit
    /// guess density.
    Symmetry,
    /// `max |λ(M) − 1|` of the occupied metric exceeds [`MAX_METRIC_DEVIATION`].
    PoorOverlap(f64),
    /// A carried block is numerically rank-deficient in the new metric.
    RankDeficient,
}

impl std::fmt::Display for ColdReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColdReason::Disabled => write!(f, "disabled"),
            ColdReason::FirstGeometry => write!(f, "first geometry"),
            ColdReason::SystemChanged => write!(f, "system changed"),
            ColdReason::Symmetry => write!(f, "symmetry-blocked SCF"),
            ColdReason::PoorOverlap(d) => {
                write!(
                    f,
                    "poor overlap (metric deviation {d:.2e} > {MAX_METRIC_DEVIATION})"
                )
            }
            ColdReason::RankDeficient => write!(f, "carried orbitals rank-deficient"),
        }
    }
}

/// The guess a warm start hands the solver, per reference.
#[derive(Debug, Clone)]
pub(crate) enum WarmGuess {
    /// Closed shell: the guess density `2 C_occ' C_occ'ᵀ`
    /// (`RhfConfig::init_guess_density`).
    Restricted(Array2<f64>),
    /// UHF/UKS: full `(n, n)` α and β MO matrices (`initial_mos`).
    Unrestricted(Array2<f64>, Array2<f64>),
    /// ROHF/ROKS: one full `(n, n)` MO matrix, columns docc | socc | virtual.
    RestrictedOpen(Array2<f64>),
}

/// Outcome of [`ScfWarmStart::prepare`].
#[derive(Debug, Clone)]
pub(crate) enum Prepared {
    Warm {
        guess: WarmGuess,
        metric_deviation: f64,
    },
    Cold(ColdReason),
}

/// One geometry evaluation's guess (see [`ScfWarmStart::records`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WarmStartRecord {
    /// `None` when the SCF was warm-started; the reason otherwise.
    pub cold: Option<ColdReason>,
    /// Occupied-metric deviation when a recorded solution was compared.
    pub metric_deviation: Option<f64>,
    /// The warm-started SCF failed and the cold guess was run after it.
    pub retried_cold: bool,
    /// SCF iterations of the solve that was used (the cold one after a retry).
    pub iterations: usize,
    /// Iterations spent in a failed warm attempt (0 when none).
    pub wasted_iterations: usize,
}

#[derive(Debug, Clone)]
struct Snapshot {
    spin: Spin,
    z: Vec<i32>,
    charge: i32,
    multiplicity: usize,
    nbasis: usize,
    mos_alpha: Array2<f64>,
    mos_beta: Option<Array2<f64>>,
}

/// Carries the last converged SCF of a geometry driver to the next geometry.
#[derive(Debug, Clone, Default)]
pub struct ScfWarmStart {
    enabled: bool,
    prev: Option<Snapshot>,
    records: Vec<WarmStartRecord>,
}

/// `(α-occupied, β-occupied)` counts for `mol` (the solvers' convention).
fn occupations(mol: &Molecule) -> Option<(usize, usize)> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return None;
    }
    Some((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// Symmetric `M^{-1/2}` of a small symmetric positive matrix, with its
/// eigenvalues, or `None` if any eigenvalue is below [`MIN_BLOCK_EIGENVALUE`].
fn inv_sqrt(m: &Array2<f64>) -> Result<Option<(Array2<f64>, Vec<f64>)>, FerricError> {
    let (lam, u) = m
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("warm start metric eigh: {e}")))?;
    if lam.iter().any(|&l| l.is_nan() || l <= MIN_BLOCK_EIGENVALUE) {
        return Ok(None);
    }
    let mut us = u.clone();
    for (j, &l) in lam.iter().enumerate() {
        us.column_mut(j).mapv_inplace(|v| v / l.sqrt());
    }
    Ok(Some((us.dot(&u.t()), lam.to_vec())))
}

/// `max |λ − 1|` over the eigenvalues of the occupied metric `C_occᵀ S C_occ`.
pub(crate) fn occupied_metric_deviation(
    c: &Array2<f64>,
    nocc: usize,
    s: &Array2<f64>,
) -> Result<f64, FerricError> {
    if nocc == 0 {
        return Ok(0.0);
    }
    let co = c.slice(s![.., ..nocc]);
    let m = co.t().dot(&s.dot(&co));
    let (lam, _) = m
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("warm start metric eigh: {e}")))?;
    Ok(lam.iter().fold(0.0f64, |a, &l| a.max((l - 1.0).abs())))
}

/// Orthonormalize the columns of `c` in the metric `s`, block by block:
/// `blocks` are the leading block sizes (occupation order); every remaining
/// non-zero column forms a last block. Each block is projected against the
/// blocks already placed, then Löwdin orthonormalized. Zero (padding) columns
/// stay zero. `None` if a block is rank-deficient.
pub(crate) fn orthonormalize_blocks(
    c: &Array2<f64>,
    blocks: &[usize],
    s: &Array2<f64>,
) -> Result<Option<Array2<f64>>, FerricError> {
    let (n, ncol) = c.dim();
    let lead: usize = blocks.iter().sum();
    if lead > ncol {
        return Ok(None);
    }
    // Remaining columns that carry anything (diagonalize() pads with zeros
    // when the lindep filter dropped directions).
    let rest: Vec<usize> = (lead..ncol)
        .filter(|&j| c.column(j).iter().any(|&v| v != 0.0))
        .collect();
    let mut out = Array2::<f64>::zeros((n, ncol));
    let mut placed = 0usize;
    let mut ranges: Vec<Vec<usize>> = Vec::new();
    let mut start = 0;
    for &b in blocks {
        ranges.push((start..start + b).collect());
        start += b;
    }
    ranges.push(rest);
    // Output column of each block's first entry: blocks are packed in order.
    for cols in ranges {
        if cols.is_empty() {
            continue;
        }
        let mut q = Array2::<f64>::zeros((n, cols.len()));
        for (k, &j) in cols.iter().enumerate() {
            q.column_mut(k).assign(&c.column(j));
        }
        if placed > 0 {
            let p = out.slice(s![.., ..placed]);
            let overlap = p.t().dot(&s.dot(&q));
            q = &q - &p.dot(&overlap);
        }
        let m = q.t().dot(&s.dot(&q));
        let Some((m_is, _)) = inv_sqrt(&m)? else {
            return Ok(None);
        };
        let qo = q.dot(&m_is);
        out.slice_mut(s![.., placed..placed + cols.len()])
            .assign(&qo);
        placed += cols.len();
    }
    Ok(Some(out))
}

impl ScfWarmStart {
    /// A warm-start carrier; `enabled = false` always answers
    /// [`ColdReason::Disabled`] (the cold path, byte-identical to no carrier).
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            prev: None,
            records: Vec::new(),
        }
    }

    /// Per-evaluation record, in order.
    pub fn records(&self) -> &[WarmStartRecord] {
        &self.records
    }

    /// The guess for `mol` on reference `spin` with the new basis `prep`.
    pub(crate) fn prepare(
        &self,
        mol: &Molecule,
        prep: &PreparedBasis,
        spin: Spin,
        symmetry: bool,
    ) -> Result<Prepared, FerricError> {
        if !self.enabled {
            return Ok(Prepared::Cold(ColdReason::Disabled));
        }
        if symmetry {
            return Ok(Prepared::Cold(ColdReason::Symmetry));
        }
        let Some(p) = self.prev.as_ref() else {
            return Ok(Prepared::Cold(ColdReason::FirstGeometry));
        };
        let z: Vec<i32> = mol.atoms.iter().map(|a| a.z).collect();
        let n = prep.nbasis();
        if p.spin != spin
            || p.z != z
            || p.charge != mol.charge
            || p.multiplicity != mol.multiplicity
            || p.nbasis != n
            || p.mos_alpha.nrows() != n
        {
            return Ok(Prepared::Cold(ColdReason::SystemChanged));
        }
        let Some((na, nb)) = occupations(mol) else {
            return Ok(Prepared::Cold(ColdReason::SystemChanged));
        };
        let s_new = ferric_integrals::oneelectron::overlap(prep);
        let mut dev = occupied_metric_deviation(&p.mos_alpha, na, &s_new)?;
        if let Some(cb) = p.mos_beta.as_ref() {
            dev = dev.max(occupied_metric_deviation(cb, nb, &s_new)?);
        }
        if dev.is_nan() || dev > MAX_METRIC_DEVIATION {
            return Ok(Prepared::Cold(ColdReason::PoorOverlap(dev)));
        }
        let guess = match spin {
            Spin::Restricted => {
                let Some(c) = orthonormalize_blocks(&p.mos_alpha, &[na], &s_new)? else {
                    return Ok(Prepared::Cold(ColdReason::RankDeficient));
                };
                let co = c.slice(s![.., ..na]);
                WarmGuess::Restricted(2.0 * co.dot(&co.t()))
            }
            Spin::Unrestricted => {
                let Some(cb_old) = p.mos_beta.as_ref() else {
                    return Ok(Prepared::Cold(ColdReason::SystemChanged));
                };
                let ca = orthonormalize_blocks(&p.mos_alpha, &[na], &s_new)?;
                let cb = orthonormalize_blocks(cb_old, &[nb], &s_new)?;
                match (ca, cb) {
                    (Some(ca), Some(cb)) => WarmGuess::Unrestricted(ca, cb),
                    _ => return Ok(Prepared::Cold(ColdReason::RankDeficient)),
                }
            }
            Spin::RestrictedOpen => {
                let Some(c) = orthonormalize_blocks(&p.mos_alpha, &[nb, na - nb], &s_new)? else {
                    return Ok(Prepared::Cold(ColdReason::RankDeficient));
                };
                WarmGuess::RestrictedOpen(c)
            }
        };
        Ok(Prepared::Warm {
            guess,
            metric_deviation: dev,
        })
    }

    /// Record the converged SCF at `mol` as the next geometry's starting point.
    /// An unconverged result is never recorded (the previous one is kept).
    pub(crate) fn remember(&mut self, mol: &Molecule, prep: &PreparedBasis, res: &ScfResult) {
        if !self.enabled || !res.converged {
            return;
        }
        self.prev = Some(Snapshot {
            spin: res.spin,
            z: mol.atoms.iter().map(|a| a.z).collect(),
            charge: mol.charge,
            multiplicity: mol.multiplicity,
            nbasis: prep.nbasis(),
            mos_alpha: res.mos_alpha.clone(),
            mos_beta: res.mos_beta.clone(),
        });
    }

    /// Append one evaluation's record.
    pub(crate) fn note(&mut self, record: WarmStartRecord) {
        self.records.push(record);
    }

    /// One-line summary for the end of an optimization (`None` when disabled
    /// or nothing was evaluated).
    pub fn summary(&self) -> Option<String> {
        if !self.enabled || self.records.is_empty() {
            return None;
        }
        let warm = self.records.iter().filter(|r| r.cold.is_none()).count();
        let retried = self.records.iter().filter(|r| r.retried_cold).count();
        let iters: Vec<String> = self
            .records
            .iter()
            .map(|r| {
                if r.retried_cold {
                    format!("{}+{}", r.wasted_iterations, r.iterations)
                } else {
                    r.iterations.to_string()
                }
            })
            .collect();
        let mut cold_why: Vec<String> = self
            .records
            .iter()
            .filter_map(|r| r.cold.map(|c| c.to_string()))
            .collect();
        cold_why.dedup();
        Some(format!(
            "SCF warm start: {warm} of {} evaluations warm-started ({} retried cold){}; \
             SCF iterations per evaluation: [{}]",
            self.records.len(),
            retried,
            if cold_why.is_empty() {
                String::new()
            } else {
                format!("; cold: {}", cold_why.join(", "))
            },
            iters.join(", ")
        ))
    }
}

/// Solve one geometry's SCF through `warm`: warm-started when
/// [`ScfWarmStart::prepare`] allows it, retried from the cold guess if the
/// warm attempt errors or does not converge, then recorded.
///
/// `cold()` is the driver's unchanged solve; `warm_solve(guess)` the same
/// solve started from `guess`. Both may return an unconverged `Ok` (the
/// caller applies its own convergence policy to the returned result, exactly
/// as before). With warm start disabled this is `cold()` plus a record.
pub(crate) fn solve_with_warm_start(
    warm: &mut ScfWarmStart,
    mol: &Molecule,
    prep: &PreparedBasis,
    spin: Spin,
    symmetry: bool,
    cold: impl FnOnce() -> Result<ScfResult, FerricError>,
    warm_solve: impl FnOnce(WarmGuess) -> Result<ScfResult, FerricError>,
) -> Result<ScfResult, FerricError> {
    let prepared = warm.prepare(mol, prep, spin, symmetry)?;
    let (res, record) = match prepared {
        Prepared::Cold(reason) => {
            let r = cold()?;
            let rec = WarmStartRecord {
                cold: Some(reason),
                metric_deviation: match reason {
                    ColdReason::PoorOverlap(d) => Some(d),
                    _ => None,
                },
                retried_cold: false,
                iterations: r.iterations,
                wasted_iterations: 0,
            };
            (r, rec)
        }
        Prepared::Warm {
            guess,
            metric_deviation,
        } => {
            let attempt = warm_solve(guess);
            match attempt {
                Ok(r) if r.converged => {
                    let rec = WarmStartRecord {
                        cold: None,
                        metric_deviation: Some(metric_deviation),
                        retried_cold: false,
                        iterations: r.iterations,
                        wasted_iterations: 0,
                    };
                    (r, rec)
                }
                failed => {
                    // A user interrupt is not a warm-guess failure: stop here
                    // instead of paying for a cold SCF the user asked to cancel.
                    if failed.is_err()
                        && ferric_core::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        return failed;
                    }
                    let wasted = match &failed {
                        Ok(r) => r.iterations,
                        Err(FerricError::ScfConvergence { iterations, .. }) => *iterations,
                        Err(_) => 0,
                    };
                    let why = match &failed {
                        Ok(_) => "did not converge".to_string(),
                        Err(e) => format!("failed ({e})"),
                    };
                    eprintln!(
                        "[ferric] warm-started SCF {why} after {wasted} iterations; \
                         retrying from the cold guess"
                    );
                    let r = cold()?;
                    let rec = WarmStartRecord {
                        cold: None,
                        metric_deviation: Some(metric_deviation),
                        retried_cold: true,
                        iterations: r.iterations,
                        wasted_iterations: wasted,
                    };
                    (r, rec)
                }
            }
        }
    };
    warm.remember(mol, prep, &res);
    warm.note(record);
    Ok(res)
}

fn wrong_guess(kind: &str) -> FerricError {
    FerricError::General(format!(
        "warm start: internal error, a non-{kind} guess reached the {kind} solver"
    ))
}

/// [`crate::rhf::solve_rhf`] (RHF/RKS) through [`solve_with_warm_start`]: the
/// warm attempt sets `init_guess_density` to the re-orthonormalized occupied
/// density. Returns an unconverged `Ok` exactly when `solve_rhf` would.
pub(crate) fn solve_rhf_warm(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<ScfResult, FerricError> {
    solve_with_warm_start(
        warm,
        mol,
        prep,
        Spin::Restricted,
        config.symmetry,
        || crate::rhf::solve_rhf(ctx, mol, prep, op, bounds, config),
        |g| {
            let WarmGuess::Restricted(d) = g else {
                return Err(wrong_guess("restricted"));
            };
            let mut cfg = config.clone();
            cfg.init_guess_density = Some(d);
            crate::rhf::solve_rhf(ctx, mol, prep, op, bounds, &cfg)
        },
    )
}

/// [`crate::uhf::solve_uhf`] (UHF/UKS) through [`solve_with_warm_start`]: the
/// warm attempt passes the carried α/β MOs as `initial_mos`. Errors on an
/// unconverged SCF exactly as `solve_uhf` does (after the cold retry).
pub(crate) fn solve_uhf_warm(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<ScfResult, FerricError> {
    solve_with_warm_start(
        warm,
        mol,
        prep,
        Spin::Unrestricted,
        config.symmetry,
        || crate::uhf::solve_uhf(ctx, mol, prep, bounds, config),
        |g| {
            let WarmGuess::Unrestricted(ca, cb) = g else {
                return Err(wrong_guess("unrestricted"));
            };
            crate::uhf::solve_uhf_best_effort(ctx, mol, prep, bounds, config, Some((&ca, &cb)))
        },
    )
}

/// [`crate::rohf::solve_rohf`] (ROHF/ROKS) through [`solve_with_warm_start`]:
/// the warm attempt starts from the carried docc | socc | virtual MOs. Errors
/// on an unconverged SCF exactly as `solve_rohf` does (after the cold retry).
pub(crate) fn solve_rohf_warm(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<ScfResult, FerricError> {
    solve_with_warm_start(
        warm,
        mol,
        prep,
        Spin::RestrictedOpen,
        config.symmetry,
        || crate::rohf::solve_rohf(ctx, mol, prep, op, bounds, config),
        |g| {
            let WarmGuess::RestrictedOpen(c) = g else {
                return Err(wrong_guess("restricted-open"));
            };
            crate::rohf::solve_rohf_best_effort_with_guess(ctx, mol, prep, bounds, config, &c)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::eps_span;

    const WATER: &str = "3\nH2O\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";
    const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

    /// `mol` with atom `atom` moved by `d` Bohr along `axis`.
    fn moved(mol: &Molecule, atom: usize, axis: usize, d: f64) -> Molecule {
        let mut m = mol.clone();
        match axis {
            0 => m.atoms[atom].x += d,
            1 => m.atoms[atom].y += d,
            _ => m.atoms[atom].zpos += d,
        }
        m
    }

    fn translated(mol: &Molecule, d: [f64; 3]) -> Molecule {
        let mut m = mol.clone();
        for a in &mut m.atoms {
            a.x += d[0];
            a.y += d[1];
            a.zpos += d[2];
        }
        m
    }

    fn setup(mol: &Molecule, basis: &str) -> (PreparedBasis, SchwarzBounds) {
        let bs = ferric_core::basis::bundled(basis).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        (prep, bounds)
    }

    /// One SCF at `mol` through `warm` on reference `spin`.
    fn solve(
        mol: &Molecule,
        basis: &str,
        spin: Spin,
        cfg: &RhfConfig,
        warm: &mut ScfWarmStart,
    ) -> ScfResult {
        let ctx = ParallelContext::default();
        let (prep, bounds) = setup(mol, basis);
        let op = Operator::coulomb();
        match spin {
            Spin::Restricted => solve_rhf_warm(&ctx, mol, &prep, op, &bounds, cfg, warm),
            Spin::Unrestricted => solve_uhf_warm(&ctx, mol, &prep, &bounds, cfg, warm),
            Spin::RestrictedOpen => solve_rohf_warm(&ctx, mol, &prep, op, &bounds, cfg, warm),
        }
        .unwrap()
    }

    /// The bound a warm and a cold SCF at ONE geometry must agree to, derived
    /// from the convergence test rather than guessed: each converged exit has
    /// an occupied-virtual residual rotation `|kappa|_F <= n * density_conv`,
    /// and the energy error of a residual rotation is second order,
    /// `<= span(eps) * |kappa|_F^2` (`crate::result::stationarity_tol`
    /// without its energy-gate term: the density gate is the binding one, and
    /// the 1e-3 default energy gate would make the bound vacuous). Two exits,
    /// so twice that.
    fn agreement_bound(cfg: &RhfConfig, nbasis: usize, r: &ScfResult) -> f64 {
        let mut span = eps_span(&r.eps_alpha);
        if let Some(eb) = r.eps_beta.as_ref() {
            span = span.max(eps_span(eb));
        }
        2.0 * span * (nbasis as f64 * cfg.density_conv).powi(2)
    }

    #[test]
    fn same_geometry_warm_guess_reproduces_the_converged_density() {
        // Trivial limit: at the SAME geometry the metric is the identity, the
        // Löwdin step is a no-op and the guess IS the converged density.
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        let mut warm = ScfWarmStart::new(true);
        let r0 = solve(&mol, "sto-3g", Spin::Restricted, &cfg, &mut warm);
        let (prep, _) = setup(&mol, "sto-3g");
        let Prepared::Warm {
            guess: WarmGuess::Restricted(d),
            metric_deviation,
        } = warm.prepare(&mol, &prep, Spin::Restricted, false).unwrap()
        else {
            panic!("same geometry must warm-start");
        };
        assert!(metric_deviation < 1e-10, "deviation {metric_deviation:e}");
        let max_abs = |a: &Array2<f64>| a.mapv(f64::abs).fold(0.0f64, |x, &y| x.max(y));
        // Exactly the density of the converged MOs ...
        let co = r0.mos_alpha.slice(s![.., ..5]);
        let d_mo = 2.0 * co.dot(&co.t());
        let diff = max_abs(&(&d - &d_mo));
        assert!(diff < 1e-12, "guess vs converged-MO density: {diff:e}");
        // ... which is the SCF's last input density to within its own
        // density_conv (the MOs come from the final Fock diagonalization).
        let diff = max_abs(&(&d - r0.density_total()));
        assert!(diff < cfg.density_conv, "guess vs density_total: {diff:e}");
        // And the warm-started SCF at the same geometry is (nearly) free.
        let r1 = solve(&mol, "sto-3g", Spin::Restricted, &cfg, &mut warm);
        eprintln!(
            "same-geometry warm SCF: {} iterations (cold {})",
            r1.iterations, r0.iterations
        );
        assert!(r1.iterations <= 3, "{} iterations", r1.iterations);
        assert!((r1.energy - r0.energy).abs() < 1e-10);
    }

    #[test]
    fn metric_is_the_identity_under_rigid_translation() {
        // The AOs move with the atoms, so a rigid move leaves S unchanged and
        // the old MOs remain exactly orthonormal: deviation at roundoff.
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        let mut warm = ScfWarmStart::new(true);
        let _ = solve(&mol, "6-31g", Spin::Restricted, &cfg, &mut warm);
        let far = translated(&mol, [3.0, -2.0, 1.5]);
        let (prep, _) = setup(&far, "6-31g");
        match warm.prepare(&far, &prep, Spin::Restricted, false).unwrap() {
            Prepared::Warm {
                metric_deviation, ..
            } => assert!(metric_deviation < 1e-10, "{metric_deviation:e}"),
            Prepared::Cold(c) => panic!("rigid translation went cold: {c}"),
        }
    }

    #[test]
    fn metric_deviation_separates_optimizer_steps_from_big_moves() {
        // Pins the MAX_METRIC_DEVIATION choice with the numbers it was
        // chosen from: an optimizer-sized move (0.1 Bohr, the default trust
        // radius, all on one H) stays well under the threshold; a 1 Bohr
        // move of the same H is well over it.
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        for basis in ["sto-3g", "6-31g", "cc-pvdz"] {
            let mut warm = ScfWarmStart::new(true);
            let _ = solve(&mol, basis, Spin::Restricted, &cfg, &mut warm);
            let dev = |d: f64| {
                let m = moved(&mol, 1, 1, d);
                let (prep, _) = setup(&m, basis);
                let s = ferric_integrals::oneelectron::overlap(&prep);
                occupied_metric_deviation(&warm.prev.as_ref().unwrap().mos_alpha, 5, &s).unwrap()
            };
            let (small, big) = (dev(0.1), dev(1.0));
            eprintln!("{basis}: deviation 0.1 Bohr = {small:.3e}, 1.0 Bohr = {big:.3e}");
            assert!(small < 0.3 * MAX_METRIC_DEVIATION, "{basis}: {small:e}");
            assert!(big > 1.5 * MAX_METRIC_DEVIATION, "{basis}: {big:e}");
        }
    }

    /// Walk `mol` through nearby geometries; at each, the warm-started SCF
    /// must land on the cold SCF's energy within [`agreement_bound`], and
    /// must actually be warm (and cheaper).
    fn warm_matches_cold_along_a_path(
        xyz: &str,
        mult: usize,
        basis: &str,
        spin: Spin,
        cfg: RhfConfig,
    ) {
        let mol0 = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        // An optimizer-shaped path: a trust-radius-sized first step, then
        // steps shrinking toward convergence (0.08, 0.03, ~0.014, ~0.003 Bohr).
        let path = [
            mol0.clone(),
            moved(&mol0, 1, 2, 0.08),
            moved(&mol0, 1, 2, 0.11),
            moved(&moved(&mol0, 1, 2, 0.12), 0, 1, -0.01),
            moved(&moved(&mol0, 1, 2, 0.123), 0, 1, -0.011),
        ];
        let mut warm = ScfWarmStart::new(true);
        let (mut it_warm, mut it_cold) = (0, 0);
        for (k, m) in path.iter().enumerate() {
            let rw = solve(m, basis, spin, &cfg, &mut warm);
            let rc = solve(m, basis, spin, &cfg, &mut ScfWarmStart::new(false));
            if k > 0 {
                it_warm += rw.iterations;
                it_cold += rc.iterations;
            }
            let rec = warm.records().last().unwrap().clone();
            let n = rc.mos_alpha.nrows();
            let bound = agreement_bound(&cfg, n, &rc);
            let de = (rw.energy - rc.energy).abs();
            eprintln!(
                "{basis} {spin:?} geom {k}: |dE| {de:.2e} (bound {bound:.2e}); iterations warm {} \
                 cold {}; record {rec:?}",
                rw.iterations, rc.iterations
            );
            assert!(de <= bound, "geom {k}: |dE| {de:e} > bound {bound:e}");
            if k == 0 {
                assert_eq!(rec.cold, Some(ColdReason::FirstGeometry));
            } else {
                assert_eq!(rec.cold, None, "geom {k} should warm-start");
                assert!(!rec.retried_cold);
            }
        }
        // Per geometry the count can tie or even lose by one (the DIIS
        // trajectory differs); over the path the warm start must save work.
        assert!(
            it_warm < it_cold,
            "warm {it_warm} >= cold {it_cold} iterations over the path"
        );
    }

    #[test]
    fn rhf_warm_energy_matches_cold_at_displaced_geometries() {
        warm_matches_cold_along_a_path(WATER, 1, "6-31g", Spin::Restricted, RhfConfig::default());
    }

    #[test]
    fn uhf_warm_energy_matches_cold_at_displaced_geometries() {
        warm_matches_cold_along_a_path(OH, 2, "6-31g", Spin::Unrestricted, RhfConfig::default());
    }

    #[test]
    fn rohf_warm_energy_matches_cold_at_displaced_geometries() {
        warm_matches_cold_along_a_path(OH, 2, "6-31g", Spin::RestrictedOpen, RhfConfig::default());
    }

    #[test]
    fn rks_pbe_warm_energy_matches_cold_at_displaced_geometries() {
        let cfg = RhfConfig {
            xc: Some("PBE".to_string()),
            ..Default::default()
        };
        warm_matches_cold_along_a_path(WATER, 1, "6-31g", Spin::Restricted, cfg);
    }

    #[test]
    fn a_big_step_takes_the_cold_guess() {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        let mut warm = ScfWarmStart::new(true);
        let _ = solve(&mol, "6-31g", Spin::Restricted, &cfg, &mut warm);
        let far = moved(&mol, 1, 1, 1.0);
        let r = solve(&far, "6-31g", Spin::Restricted, &cfg, &mut warm);
        assert!(r.converged);
        let rec = warm.records().last().unwrap();
        assert!(
            matches!(rec.cold, Some(ColdReason::PoorOverlap(d)) if d > MAX_METRIC_DEVIATION),
            "{rec:?}"
        );
        // And the cold path it took IS the plain solve: bit-identical energy.
        let rc = solve(
            &far,
            "6-31g",
            Spin::Restricted,
            &cfg,
            &mut ScfWarmStart::new(false),
        );
        assert_eq!(r.energy.to_bits(), rc.energy.to_bits());
    }

    #[test]
    fn system_changes_take_the_cold_guess() {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        let mut warm = ScfWarmStart::new(true);
        let _ = solve(&mol, "sto-3g", Spin::Restricted, &cfg, &mut warm);
        let cation = Molecule::parse_xyz(WATER, 1, 2).unwrap();
        let (prep, _) = setup(&cation, "sto-3g");
        assert!(matches!(
            warm.prepare(&cation, &prep, Spin::Unrestricted, false)
                .unwrap(),
            Prepared::Cold(ColdReason::SystemChanged)
        ));
        let (prep_big, _) = setup(&mol, "6-31g");
        assert!(matches!(
            warm.prepare(&mol, &prep_big, Spin::Restricted, false)
                .unwrap(),
            Prepared::Cold(ColdReason::SystemChanged)
        ));
        let (prep, _) = setup(&mol, "sto-3g");
        assert!(matches!(
            warm.prepare(&mol, &prep, Spin::Restricted, true).unwrap(),
            Prepared::Cold(ColdReason::Symmetry)
        ));
        assert!(matches!(
            ScfWarmStart::new(false)
                .prepare(&mol, &prep, Spin::Restricted, false)
                .unwrap(),
            Prepared::Cold(ColdReason::Disabled)
        ));
    }

    #[test]
    fn a_failed_warm_start_is_retried_cold_and_never_recorded() {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let cfg = RhfConfig::default();
        let mut warm = ScfWarmStart::new(true);
        let r0 = solve(&mol, "sto-3g", Spin::Restricted, &cfg, &mut warm);
        let next = moved(&mol, 1, 2, 0.05);
        let (prep, _) = setup(&next, "sto-3g");
        let cold_r = solve(
            &next,
            "sto-3g",
            Spin::Restricted,
            &cfg,
            &mut ScfWarmStart::new(false),
        );
        let mut failed = r0.clone();
        failed.converged = false;
        failed.iterations = 7;
        let mut cold_calls = 0;
        let got = solve_with_warm_start(
            &mut warm,
            &next,
            &prep,
            Spin::Restricted,
            false,
            || {
                cold_calls += 1;
                Ok(cold_r.clone())
            },
            |_| Ok(failed),
        )
        .unwrap();
        assert_eq!(cold_calls, 1);
        assert_eq!(got.energy.to_bits(), cold_r.energy.to_bits());
        let rec = warm.records().last().unwrap();
        assert!(rec.retried_cold && rec.wasted_iterations == 7, "{rec:?}");
        // The converged cold retry is what gets remembered.
        assert_eq!(warm.prev.as_ref().unwrap().mos_alpha, cold_r.mos_alpha);

        // Warm and cold both failing: the unconverged result is returned (the
        // caller's convergence policy applies) and is NOT remembered.
        let mut bad = cold_r.clone();
        bad.converged = false;
        bad.mos_alpha.fill(0.0);
        let got = solve_with_warm_start(
            &mut warm,
            &next,
            &prep,
            Spin::Restricted,
            false,
            || Ok(bad),
            |_| {
                Err(FerricError::ScfConvergence {
                    iterations: 3,
                    last_energy: 0.0,
                })
            },
        )
        .unwrap();
        assert!(!got.converged);
        assert_eq!(warm.prev.as_ref().unwrap().mos_alpha, cold_r.mos_alpha);
        assert_eq!(warm.records().last().unwrap().wasted_iterations, 3);
    }
}
