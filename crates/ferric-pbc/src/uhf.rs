//! Stage 4: Gamma-point open-shell UHF, ported from `reference/pbc/pbc_uhf.py`
//! (FINDINGS "Iteration 6 (Python, Gamma UHF)").
//!
//! At Gamma the periodic UHF is the molecular UHF on the lattice
//! `(S, h, E_nn)` plus lattice J/K, so the SCF itself is
//! [`ferric_scf::uhf::solve_uhf_injected`]. Per spin σ:
//!
//! ```text
//! F_σ = h + J[D_α + D_β] − K_σ,     K_σ = K[D_σ] + v_M S D_σ S   (exxdiv = ewald)
//! ```
//!
//! the SAME `v_M` as RHF on the single-spin density, coefficient 1 (PySCF
//! `df_jk._ewald_exxdiv_for_G0` adds `v_M S dm S` per dm). The K builders
//! carry that term and are linear in D, so no per-spin factor exists anywhere.
//! At convergence `E_ewald − E_none = −v_M (N_α + N_β)/2`.
//!
//! # The Ewald trap and the default ([`EwaldStart::Staged`])
//!
//! `v_M S D_σ S = v_M × (occupied projector)` at Gamma, so `none` and `ewald`
//! have IDENTICAL stationary densities, but ewald lowers every occupied level
//! by `v_M`. A state with a hole below the Fermi level under `none` (per-spin
//! gap < 0) can therefore be aufbau-self-consistent under ewald (gap
//! `< v_M`). Measured (prototype): triclinic 4H s+p triplet from the core
//! guess lands at −1.812958714837, 1.5e-2 above PySCF's −1.827999723359
//! (alpha gap 0.617 < v_M 0.622).
//!
//! [`gamma_uhf`] therefore defaults to [`EwaldStart::Staged`]: converge with
//! `exxdiv = none` first, then continue with ewald from those MOs (the second
//! stage starts at a stationary point, so it costs ~2 iterations). It ALWAYS
//! reports the per-spin gaps against `v_M` ([`SpinGapReport`]) and warns on
//! stderr when `min_σ gap_σ < v_M` — the necessary (single-swap,
//! positive-kernel) condition for the HF minimum; necessary, NOT sufficient
//! (O2 from the core guess lands 0.256 Ha high with gaps > v_M: guess quality
//! matters exactly as molecularly — pass `initial_mos`).
//! [`EwaldStart::Direct`] runs a single ewald SCF (kept for the trap test and
//! for callers who already hold a good guess).
//!
//! Units: Bohr and Hartree.

use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::ewald::madelung_constant;
use crate::hcore::PeriodicHcore;
use crate::lattice::Cell;
use crate::rsgdf::RsGdf;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf_injected;
use ndarray::Array2;

/// Where J/K come from. `exxdiv` is taken from [`GammaUhfConfig`], NOT from
/// the integral object (its own `madelung()` is ignored), so one tensor
/// serves both stages of [`EwaldStart::Staged`].
#[derive(Debug, Clone, Copy)]
pub enum GammaUhfIntegrals<'a> {
    /// Production: the fitted periodic B.
    RsGdf(&'a RsGdf),
    /// TEST/ORACLE ONLY: the dense pure-AFT tensor.
    DenseAft(&'a DenseAftEri),
}

impl GammaUhfIntegrals<'_> {
    /// The source's build timings plus its accumulated SCF J/K calls
    /// ([`RsGdf::timings`], [`DenseAftEri::timings`]).
    pub fn timings(&self) -> crate::timing::PbcTimings {
        match self {
            GammaUhfIntegrals::RsGdf(g) => g.timings(),
            GammaUhfIntegrals::DenseAft(e) => e.timings(),
        }
    }
}

/// How an `exxdiv = ewald` run is started (ignored for `exxdiv = none`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EwaldStart {
    /// Converge with `exxdiv = none`, then continue with ewald from those MOs
    /// (default; avoids the Gamma Ewald trap — module doc).
    #[default]
    Staged,
    /// One ewald SCF from the configured guess. The gap check still runs.
    Direct,
}

impl EwaldStart {
    /// Strict parse: `"staged"` or `"direct"` (case-insensitive); anything
    /// else is an error (config honesty).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "staged" => Ok(EwaldStart::Staged),
            "direct" => Ok(EwaldStart::Direct),
            other => Err(FerricError::General(format!(
                "ewald_start must be \"staged\" or \"direct\", got {other:?}"
            ))),
        }
    }
}

/// Configuration for [`gamma_uhf`].
#[derive(Debug, Clone)]
pub struct GammaUhfConfig {
    /// Exchange divergence treatment applied by the K builder.
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald`.
    pub ewald_start: EwaldStart,
    /// SCF knobs; validated by `ferric_scf::uhf::validate_injected_uhf`
    /// (e.g. `use_sad_guess` must be false unless `init_guess_density` is set).
    pub scf: RhfConfig,
    /// Optional per-spin starting MOs `(C_α, C_β)`, each `(nao, nao)` (e.g.
    /// the molecular UHF MOs of the cell contents: cell-0 AOs are the Gamma
    /// AOs). Used by the first stage only.
    pub initial_mos: Option<(Array2<f64>, Array2<f64>)>,
}

impl Default for GammaUhfConfig {
    fn default() -> Self {
        Self {
            exxdiv: ExxDiv::Ewald,
            ewald_start: EwaldStart::Staged,
            scf: RhfConfig {
                use_sad_guess: false,
                density_conv: 1e-10,
                max_iter: 200,
                ..Default::default()
            },
            initial_mos: None,
        }
    }
}

/// Per-spin HOMO–LUMO gaps of the converged Fock, against the Madelung shift
/// the run applied.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinGapReport {
    /// `ε_α[N_α] − ε_α[N_α − 1]`; `None` if the spin has no occupied or no
    /// virtual orbital.
    pub gap_alpha: Option<f64>,
    /// Same for β.
    pub gap_beta: Option<f64>,
    /// The occupied-level shift the gaps are tested against: `v_M` applied in
    /// K for UHF, `a·v_M` for a hybrid UKS ([`occupation_gaps`]); 0 for
    /// `exxdiv = none`.
    pub madelung_applied: f64,
    /// `min_σ gap_σ − v_M` (= the smallest gap in the `none` convention, since
    /// ewald lowers occupied levels by exactly `v_M` at Gamma); `None` if no
    /// spin has a gap.
    pub margin: Option<f64>,
}

impl SpinGapReport {
    /// `margin >= 0` (or no gap to test): the necessary condition for the HF
    /// minimum holds. Necessary, not sufficient.
    pub fn satisfied(&self) -> bool {
        self.margin.is_none_or(|m| m >= 0.0)
    }
}

/// Result of [`gamma_uhf`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaUhfResult {
    /// The final SCF (in the requested `exxdiv` convention).
    pub scf: ScfResult,
    /// The `exxdiv = none` first stage, when [`EwaldStart::Staged`] ran.
    pub none_stage: Option<ScfResult>,
    /// `v_M` of the cell (applied iff `exxdiv = ewald`).
    pub madelung: f64,
    /// `(N_α, N_β)` from the cell's charge and multiplicity.
    pub nocc: (usize, usize),
    /// `⟨S²⟩` with the lattice overlap.
    pub s2: f64,
    /// Per-spin gap check against `v_M`.
    pub gaps: SpinGapReport,
}

/// `(N_α, N_β)` from a molecule's electron count and multiplicity.
pub fn nocc_ab(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "gamma_uhf: incompatible nelec = {nelec} and multiplicity = {}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// `⟨S²⟩ = S_z(S_z+1) + N_β − Σ_ij |⟨i_α|S|j_β⟩|²` (occupied), with the
/// lattice overlap `s` at Gamma. Errors if `r` has no β MOs.
pub fn spin_square(
    r: &ScfResult,
    s: &Array2<f64>,
    na: usize,
    nb: usize,
) -> Result<f64, FerricError> {
    let cb = r
        .mos_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("spin_square: result has no beta MOs".into()))?;
    let sz = 0.5 * (na as f64 - nb as f64);
    if na == 0 || nb == 0 {
        // No α–β overlap term (and no zero-size GEMMs).
        return Ok(sz * (sz + 1.0) + nb as f64);
    }
    let ca_o = r.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb_o = cb.slice(ndarray::s![.., ..nb]);
    let ov = ca_o.t().dot(s).dot(&cb_o);
    let sum: f64 = ov.iter().map(|v| v * v).sum();
    Ok(sz * (sz + 1.0) + nb as f64 - sum)
}

/// Per-spin gaps of `r` against `madelung_applied` (see [`SpinGapReport`]).
/// Padding eigenvalues from the lindep filter (`1e6` sentinels) are not
/// counted as virtual levels.
pub fn spin_gaps(r: &ScfResult, na: usize, nb: usize, madelung_applied: f64) -> SpinGapReport {
    fn gap(eps: &[f64], nocc: usize) -> Option<f64> {
        if nocc == 0 || nocc >= eps.len() || eps[nocc] >= 1e5 {
            return None;
        }
        Some(eps[nocc] - eps[nocc - 1])
    }
    let gap_alpha = gap(&r.eps_alpha, na);
    let gap_beta = r.eps_beta.as_deref().and_then(|e| gap(e, nb));
    let min_gap = [gap_alpha, gap_beta].into_iter().flatten().reduce(f64::min);
    SpinGapReport {
        gap_alpha,
        gap_beta,
        madelung_applied,
        margin: min_gap.map(|g| g - madelung_applied),
    }
}

/// OCCUPATION-AWARE per-spin gaps of a converged `r` against `shift` (the
/// exchange-divergence shift of the occupied levels: `v_M` for UHF, `a·v_M`
/// for a hybrid UKS, 0 for `exxdiv = none`) — `pbc_uks.occ_gap` of the
/// prototype (FINDINGS "Iteration 10").
///
/// For each spin σ the eigenvectors `c_i` of the converged (undamped) Fock
/// `r.fock_σ` (`r.mos_σ`, eigenvalues `r.eps_σ`) are classified by their
/// ACTUAL occupation `n_i = c_iᵀ S D_σ S c_i` (> ½ occupied), and
/// `gap_σ = min ε(unoccupied) − max ε(occupied)`. Unlike [`spin_gaps`]
/// (sorted eigenvalues `ε[N_σ] − ε[N_σ − 1]`, which assumes aufbau), a state
/// with a hole below the Fermi level gives a NEGATIVE gap. Padding columns of
/// the lindep filter (`ε ≥ 1e5`, zero vectors) are skipped. `None` for a
/// spin with no occupied or no unoccupied level (or no β data).
///
/// `margin = min_σ gap_σ − shift`; [`SpinGapReport::satisfied`] is the
/// necessary condition `gap_σ ≥ shift` for every spin.
pub fn occupation_gaps(r: &ScfResult, s: &Array2<f64>, shift: f64) -> SpinGapReport {
    fn gap(c: &Array2<f64>, eps: &[f64], d: &Array2<f64>, s: &Array2<f64>) -> Option<f64> {
        if c.dim() != s.dim() || d.dim() != s.dim() || eps.len() != c.ncols() {
            return None;
        }
        let sds = s.dot(d).dot(s);
        let (mut occ_max, mut vir_min) = (f64::NEG_INFINITY, f64::INFINITY);
        for (i, &e) in eps.iter().enumerate() {
            if e >= 1e5 {
                continue;
            }
            let ci = c.column(i);
            let n_i = ci.dot(&sds.dot(&ci));
            if n_i > 0.5 {
                occ_max = occ_max.max(e);
            } else {
                vir_min = vir_min.min(e);
            }
        }
        (occ_max.is_finite() && vir_min.is_finite()).then_some(vir_min - occ_max)
    }
    let gap_alpha = gap(&r.mos_alpha, &r.eps_alpha, &r.density_alpha, s);
    let gap_beta = match (&r.mos_beta, &r.eps_beta, &r.density_beta) {
        (Some(c), Some(e), Some(d)) => gap(c, e, d, s),
        _ => None,
    };
    let min_gap = [gap_alpha, gap_beta].into_iter().flatten().reduce(f64::min);
    SpinGapReport {
        gap_alpha,
        gap_beta,
        madelung_applied: shift,
        margin: min_gap.map(|g| g - shift),
    }
}

pub(crate) fn builders(
    ints: GammaUhfIntegrals<'_>,
    madelung: f64,
) -> (Box<dyn JBuilder + '_>, Box<dyn KBuilder + '_>) {
    match ints {
        GammaUhfIntegrals::RsGdf(g) => (
            Box::new(g.j_builder()),
            Box::new(g.k_builder_with_madelung(madelung)),
        ),
        GammaUhfIntegrals::DenseAft(e) => (
            Box::new(e.j_builder()),
            Box::new(e.k_builder_with_madelung(madelung)),
        ),
    }
}

/// Gamma-point UHF of `cell` (spin state from `cell.mol()`'s charge and
/// multiplicity) on the lattice one-electron terms `hc` and the J/K of
/// `ints`. See the module doc for the Ewald-trap default.
///
/// `prep` must be the orbital basis `hc` and `ints` were built in (built from
/// `cell.mol()`). Errors if any SCF stage fails to converge, on config fields
/// the injected path refuses (by name), or on shape mismatches.
pub fn gamma_uhf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaUhfConfig,
) -> Result<GammaUhfResult, FerricError> {
    let mol = cell.mol();
    let (na, nb) = nocc_ab(mol)?;
    let v_m = madelung_constant(cell)?;
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never used for integrals on the
    // injected path (cell-0 molecular Schwarz table, O(nshell²) quartets).
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let run = |vm: f64, init: Option<(&Array2<f64>, &Array2<f64>)>| {
        let (j, k) = builders(ints, vm);
        let inj = PeriodicInjection {
            s: hc.s.clone(),
            h: hc.h.clone(),
            vnn: hc.enn,
            j,
            k,
            xc: None,
        };
        solve_uhf_injected(&ctx, mol, prep, &bounds, &cfg.scf, inj, init)
    };
    let init = cfg.initial_mos.as_ref().map(|(a, b)| (a, b));

    let staged = cfg.exxdiv == ExxDiv::Ewald && cfg.ewald_start == EwaldStart::Staged;
    let (scf, none_stage) = if staged {
        let first = run(0.0, init)?;
        let cb = first.mos_beta.as_ref().ok_or_else(|| {
            FerricError::General("gamma_uhf: the none stage returned no beta MOs".into())
        })?;
        let second = run(applied, Some((&first.mos_alpha, cb)))?;
        (second, Some(first))
    } else {
        (run(applied, init)?, None)
    };

    let s2 = spin_square(&scf, &hc.s, na, nb)?;
    let gaps = spin_gaps(&scf, na, nb, applied);
    if !gaps.satisfied() {
        eprintln!(
            "gamma_uhf WARNING: per-spin HOMO-LUMO gap below v_M (gap_alpha {:?}, gap_beta {:?}, \
             v_M applied {applied:.6}, margin {:?}). Under exxdiv=none this state has a hole \
             below the Fermi level: likely the Gamma Ewald trap (FINDINGS Iteration 6). Use \
             EwaldStart::Staged or a better initial_mos.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    Ok(GammaUhfResult {
        scf,
        none_stage,
        madelung: v_m,
        nocc: (na, nb),
        s2,
        gaps,
    })
}
