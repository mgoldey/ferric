//! Stage 9: k-point sampled open-shell UHF (`solve_kuhf`), ported from
//! `reference/pbc/pbc_kuhf.py` (FINDINGS "Iteration 13 (Python, k-point
//! UHF)"). A SIBLING of [`crate::kscf`] (not a mode flag in
//! `solve_krhf_injected`, whose contracts — even electron count,
//! `F = h + J − K/2`, occupation 2 — all change); the k helpers are shared
//! (`pub(crate)` in `kscf`).
//!
//! Per spin σ ∈ {α, β} and mesh point k (unit occupations):
//!
//! ```text
//! D_σ(k) = C_σ,occ(k) C_σ,occ(k)^H
//! F_σ(k) = h(k) + J[D_α + D_β](k) − K[D_σ](k)            (K includes v_M S D_σ S, exxdiv = ewald)
//! E/cell = (1/N_k) Σ_k Σ_σ ½ Re tr[(h(k) + F_σ(k)) D_σ(k)] + E_nn
//! ```
//!
//! * J/K: the injected [`KPointJk`] is called ONCE PER SPIN (it is linear in
//!   the density): `(J_α, K_α) = build(D_α)`, `(J_β, K_β) = build(D_β)`,
//!   `J = J_α + J_β`. The Madelung term lives inside the builder with the
//!   SAME `v_M` as k-RHF (the diag(n) supercell constant), coefficient 1 on
//!   D_σ, per k with no `1/N_k` — there is NO per-spin ½ (mutant
//!   [`KUhfMutation::MadelungHalfPerSpin`]: exactly `+v_M N/4` per cell).
//!   For `N_β = 0` the β build is skipped (`J_β = K_β = 0` exactly).
//! * Occupations: GLOBAL aufbau per spin over the whole mesh (the `N_σ N_k`
//!   lowest σ levels over all k; PySCF `kuhf.get_occ`). The per-k count may
//!   be non-uniform (measured: zchain 1×1×2 puts both α electrons at Gamma,
//!   `[2, 0]`); a molecule-style per-k aufbau is 0.24 Ha/cell wrong there and
//!   INVISIBLE wherever the global count is already uniform
//!   ([`KUhfMutation::PerKAufbau`]). `N_β = 0` is allowed.
//! * DIIS over the stacked `(spin, k)` commutators `[α(k0..), β(k0..)]`,
//!   Gram `Re Σ_{σ,k} ⟨e_i, e_j⟩`, real weights (time reversal survives).
//! * Final report: orbitals of the converged, UNEXTRAPOLATED Fock;
//!   occupations from the ACTUAL density (`n_i = c_i^H S D_σ S c_i > ½`),
//!   so a state with a hole below the Fermi level shows a NEGATIVE per-spin
//!   gap (`gap_σ = min ε(unocc, all k) − max ε(occ, all k)`).
//! * `⟨S²⟩` of the GIANT (supercell) determinant, PySCF `KUHF.spin_square`:
//!   `S_z(S_z+1) + N_β N_k − Σ_k Re tr[D_α(k) S(k) D_β(k) S(k)]`,
//!   `S_z = (N_α − N_β) N_k / 2`. NOT per cell (`S_z²` grows as `N_k²`).
//!
//! # Ewald trap ([`EwaldStart::Staged`], the default)
//!
//! `v_M S D_σ S = v_M × (occupied projector)` at EVERY k, so `none` and
//! `ewald` share their stationary densities while ewald lowers every
//! occupied level by `v_M`: a hole shallower than `v_M` is aufbau-consistent
//! under ewald. Measured (tri 4H s+p triplet): 1×1×1 trapped 15 mHa high,
//! 1×1×2 trapped 93 mHa high (DEEPER than Gamma — the trap does not fade
//! monotonically with the mesh), 1×1×3 not trapped. [`solve_kuhf`] converges
//! with `exxdiv = none` first and continues with ewald from that density
//! (1 iteration measured), and ALWAYS reports the per-spin gaps against the
//! applied `v_M` ([`SpinGapReport`], warning on stderr when violated —
//! necessary, not sufficient).
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::default_ewald_omega;
use crate::hcore::kpoint::{hermitize, periodic_hcore_kpts};
use crate::hcore::PeriodicHcoreConfig;
use crate::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use crate::kpts::KPointMesh;
use crate::kscf::{
    aufbau, diagonalize_all, herm_t, occupied_projector, orthogonalizers_with_report, KDiis,
    KJkKind, KPointInjection, KPointJk, KScfConfig,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use crate::uhf::{nocc_ab, EwaldStart, SpinGapReport};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use num_complex::Complex64;

/// TEST ONLY: deliberate defects the k-UHF tests must catch (FINDINGS
/// Iteration 13 mutation table).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KUhfMutation {
    /// Both K_σ from `D_α + D_β` (tri 1×1×3: −0.687 Ha/cell; closed shell:
    /// K doubled). Invisible when `N_β = 0`.
    KFromTotalDensity,
    /// `v_M / 2` in K per spin (RHF's "D/2" bookkeeping wrongly applied to a
    /// spin density): exactly `+v_M N/4` per cell under ewald, 0 under none.
    MadelungHalfPerSpin,
    /// `N_σ` lowest levels at EVERY k (molecule-style per-k aufbau):
    /// zchain 1×1×2 +0.2412 Ha/cell; 0 wherever global aufbau is uniform.
    PerKAufbau,
}

/// Result of a k-point UHF SCF (one stage).
#[derive(Debug, Clone)]
pub struct KUScfResult {
    /// Total energy PER CELL (Hartree) of `density_alpha`/`density_beta`.
    pub energy: f64,
    /// Nuclear repulsion per cell.
    pub e_nuc: f64,
    /// α electrons per cell (the mesh holds `nalpha · N_k`).
    pub nalpha: usize,
    /// β electrons per cell.
    pub nbeta: usize,
    /// α orbital energies per k (ascending), mesh order.
    pub eps_alpha: Vec<Vec<f64>>,
    /// β orbital energies per k.
    pub eps_beta: Vec<Vec<f64>>,
    /// α MO coefficients per k, `(nao, n_orth)`.
    pub mos_alpha: Vec<Array2<Complex64>>,
    /// β MO coefficients per k.
    pub mos_beta: Vec<Array2<Complex64>>,
    /// `D_α(k)` (unit occupation), the density `energy` belongs to.
    pub density_alpha: Vec<Array2<Complex64>>,
    /// `D_β(k)`.
    pub density_beta: Vec<Array2<Complex64>>,
    /// `F_α(k)` of the densities (not extrapolated).
    pub fock_alpha: Vec<Array2<Complex64>>,
    /// `F_β(k)`.
    pub fock_beta: Vec<Array2<Complex64>>,
    /// α occupations (1 or 0) per k from the ACTUAL density, aligned with
    /// `eps_alpha`.
    pub occ_alpha: Vec<Vec<f64>>,
    /// β occupations.
    pub occ_beta: Vec<Vec<f64>>,
    /// Occupied α levels per k (may be non-uniform over k).
    pub nocc_per_k_alpha: Vec<usize>,
    /// Occupied β levels per k.
    pub nocc_per_k_beta: Vec<usize>,
    /// `min ε(unocc) − max ε(occ)` over ALL k for α, from the actual
    /// occupations (negative = hole below the Fermi level); `None` if α has
    /// no occupied or no unoccupied level.
    pub gap_alpha: Option<f64>,
    /// Same for β.
    pub gap_beta: Option<f64>,
    /// `⟨S²⟩` of the giant (supercell) determinant — NOT per cell.
    pub s2: f64,
    /// Converged flag.
    pub converged: bool,
    /// SCF iterations run.
    pub iterations: usize,
    /// Final max orbital-gradient element over (spin, k).
    pub max_error: f64,
    /// Cartesian k-points (Bohr⁻¹), mesh order.
    pub kpts: Vec<[f64; 3]>,
    /// Per-k canonical-cut diagnostics of the shared (both-spin)
    /// orthogonalisers ([`crate::lindep`]).
    pub lindep: crate::lindep::LindepReport,
}

/// Per-spin gap report of `r` against the occupied-level shift `shift`
/// (`v_M` applied in K; 0 for `exxdiv = none`): `margin = min_σ gap_σ − shift`.
pub fn kuhf_gap_report(r: &KUScfResult, shift: f64) -> SpinGapReport {
    let min_gap = [r.gap_alpha, r.gap_beta]
        .into_iter()
        .flatten()
        .reduce(f64::min);
    SpinGapReport {
        gap_alpha: r.gap_alpha,
        gap_beta: r.gap_beta,
        madelung_applied: shift,
        margin: min_gap.map(|g| g - shift),
    }
}

/// Top-level settings for [`solve_kuhf`].
#[derive(Debug, Clone, Copy)]
pub struct KUhfConfig {
    /// SCF settings (`min_gap` applies per spin in the in-loop aufbau).
    pub scf: KScfConfig,
    /// Exchange-divergence treatment.
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald` (default staged; module doc).
    pub ewald_start: EwaldStart,
    /// One-electron lattice sums.
    pub hcore: PeriodicHcoreConfig,
    /// J/K builder.
    pub jk: KJkKind,
    /// Dense-AFT oracle settings (`jk = Dense`).
    pub dense: KDenseAftConfig,
    /// k-point RS-GDF settings (`jk = RsGdf`; its `gdf.exxdiv` is ignored —
    /// `exxdiv` above decides).
    pub rsgdf: KRsGdfConfig,
    /// Budget for the SCF work arrays (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    #[doc(hidden)]
    pub mutation: Option<KUhfMutation>,
}

impl KUhfConfig {
    /// Defaults (dense J/K, staged ewald start) with the balanced Ewald ω of
    /// `cell` for the nuclear split.
    pub fn for_cell(cell: &Cell, exxdiv: ExxDiv) -> Self {
        Self {
            scf: KScfConfig::default(),
            exxdiv,
            ewald_start: EwaldStart::Staged,
            hcore: PeriodicHcoreConfig::with_omega(default_ewald_omega(cell)),
            jk: KJkKind::Dense,
            dense: KDenseAftConfig::default(),
            rsgdf: KRsGdfConfig::default(),
            budget_bytes: None,
            mutation: None,
        }
    }
}

/// Result of [`solve_kuhf`].
#[derive(Debug, Clone)]
#[must_use]
pub struct KUhfResult {
    /// The final SCF (in the requested `exxdiv` convention).
    pub scf: KUScfResult,
    /// The `exxdiv = none` first stage, when [`EwaldStart::Staged`] ran.
    pub none_stage: Option<KUScfResult>,
    /// The mesh (supercell) `v_M`, applied iff `exxdiv = ewald`.
    pub madelung: f64,
    /// `(N_α, N_β)` per cell from the cell's charge and multiplicity.
    pub nocc: (usize, usize),
    /// Per-spin gaps (actual occupations) against the applied `v_M`.
    pub gaps: SpinGapReport,
}

enum KInts {
    Dense(KDenseAftEri),
    RsGdf(KRsGdf),
}

fn jk_of(ints: &KInts, madelung: f64) -> Box<dyn KPointJk + '_> {
    match ints {
        KInts::Dense(e) => Box::new(e.jk_builder_with_madelung(madelung)),
        KInts::RsGdf(g) => Box::new(g.jk_builder_with_madelung(madelung)),
    }
}

/// k-point UHF of `cell` (spin state from `cell.mol()`'s charge and
/// multiplicity, per cell): builds `S(k)`, `h(k)` and the J/K source of
/// `cfg.jk` (dense: `aux` must be `None`; rsgdf: `aux` REQUIRED) exactly as
/// [`crate::kscf::solve_krhf`], then runs [`solve_kuhf_injected`] — staged
/// (none → ewald) by default for `exxdiv = ewald`. Errors if a stage does
/// not converge.
pub fn solve_kuhf(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KUhfConfig,
) -> Result<KUhfResult, FerricError> {
    match (cfg.jk, aux) {
        (KJkKind::Dense, Some(_)) => {
            return Err(FerricError::General(
                "solve_kuhf: an aux basis was given but jk = dense does not use it; set jk = \
                 rsgdf or pass no aux basis"
                    .into(),
            ))
        }
        (KJkKind::RsGdf, None) => {
            return Err(FerricError::General(
                "solve_kuhf: jk = rsgdf requires an auxbasis (none given)".into(),
            ))
        }
        _ => {}
    }
    let (na, nb) = nocc_ab(cell.mol())?;
    let hk = periodic_hcore_kpts(cell, prep, mesh, &cfg.hcore)?;
    let v_m = mesh.madelung(cell)?;
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    // What the K builder applies (the mutant halves it per spin).
    let k_madelung = if cfg.mutation == Some(KUhfMutation::MadelungHalfPerSpin) {
        0.5 * applied
    } else {
        applied
    };
    let ints = match aux {
        None => KInts::Dense(KDenseAftEri::build(
            cell,
            prep,
            mesh,
            &hk.s,
            ExxDiv::None,
            &cfg.dense,
        )?),
        Some(aux) => KInts::RsGdf(KRsGdf::build(cell, prep, aux, mesh, &hk.s, &cfg.rsgdf)?),
    };
    let run = |vm: f64,
               guess: Option<(&[Array2<Complex64>], &[Array2<Complex64>])>|
     -> Result<KUScfResult, FerricError> {
        let inj = KPointInjection {
            s: hk.s.clone(),
            h: hk.h.clone(),
            vnn: hk.enn,
            jk: jk_of(&ints, vm),
        };
        let r = run_kuhf(
            cell,
            mesh,
            &cfg.scf,
            inj,
            na,
            nb,
            guess,
            cfg.budget_bytes,
            cfg.mutation,
        )?;
        if !r.converged {
            return Err(FerricError::General(format!(
                "solve_kuhf: SCF (v_M {vm:.6}) not converged in {} iterations (max error {:.2e})",
                r.iterations, r.max_error
            )));
        }
        Ok(r)
    };

    let staged = cfg.exxdiv == ExxDiv::Ewald && cfg.ewald_start == EwaldStart::Staged;
    let (scf, none_stage) = if staged {
        let first = run(0.0, None)?;
        let second = run(
            k_madelung,
            Some((&first.density_alpha[..], &first.density_beta[..])),
        )?;
        (second, Some(first))
    } else {
        (run(k_madelung, None)?, None)
    };
    let gaps = kuhf_gap_report(&scf, applied);
    if !gaps.satisfied() {
        eprintln!(
            "solve_kuhf WARNING: per-spin global gap below the applied v_M (gap_alpha {:?}, \
             gap_beta {:?}, v_M applied {applied:.6}, margin {:?}). Under exxdiv=none this state \
             has a hole below the Fermi level: likely the Ewald trap (FINDINGS Iterations 6, 13). \
             Use EwaldStart::Staged.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    Ok(KUhfResult {
        scf,
        none_stage,
        madelung: v_m,
        nocc: (na, nb),
        gaps,
    })
}

/// k-point UHF on injected `S(k)`, `h(k)`, `E_nn` and J/K (module doc), from
/// the core guess. `nalpha`/`nbeta` are PER CELL (`nbeta = 0` allowed) and
/// must sum to `cell.mol().nelec()`. The builder's Madelung term is whatever
/// it carries; `gap_*` in the result are plain gaps (see [`kuhf_gap_report`]).
/// Returns `converged = false` rather than an error on hitting `max_iter`.
pub fn solve_kuhf_injected(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    inj: KPointInjection<'_>,
    nalpha: usize,
    nbeta: usize,
) -> Result<KUScfResult, FerricError> {
    run_kuhf(cell, mesh, cfg, inj, nalpha, nbeta, None, None, None)
}

/// [`solve_kuhf_injected`] from starting densities `(D_α(k), D_β(k))`
/// (unit occupation, mesh order) — e.g. the none stage of a staged ewald
/// run, or another J/K source's converged state. `None` = core guess.
pub fn solve_kuhf_injected_with_guess(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    inj: KPointInjection<'_>,
    nalpha: usize,
    nbeta: usize,
    guess: Option<(&[Array2<Complex64>], &[Array2<Complex64>])>,
) -> Result<KUScfResult, FerricError> {
    run_kuhf(cell, mesh, cfg, inj, nalpha, nbeta, guess, None, None)
}

fn re_tr_prod(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    // Re tr[A B] = Re Σ_mn A_mn B_nm
    let n = a.nrows();
    let mut tr = 0.0;
    for m in 0..n {
        for nu in 0..n {
            tr += (a[(m, nu)] * b[(nu, m)]).re;
        }
    }
    tr
}

/// Per-spin occupations for one aufbau step: global (default) or the
/// per-k mutant. `n_cell` electrons of this spin per cell.
fn spin_occupations(
    eps: &[Vec<f64>],
    n_cell: usize,
    min_gap: f64,
    per_k: bool,
    spin: &str,
) -> Result<Vec<Vec<f64>>, FerricError> {
    let nk = eps.len();
    if n_cell == 0 {
        return Ok(eps.iter().map(|e| vec![0.0; e.len()]).collect());
    }
    if per_k {
        return Ok(eps
            .iter()
            .map(|e| {
                (0..e.len())
                    .map(|i| if i < n_cell { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect());
    }
    let (occ, _, _) = aufbau(eps, n_cell * nk, min_gap)
        .map_err(|e| FerricError::General(format!("solve_kuhf ({spin} spin aufbau): {e}")))?;
    Ok(occ
        .into_iter()
        .map(|o| {
            o.into_iter()
                .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect())
}

/// Occupations (1/0) of the eigenvectors `c` from the ACTUAL density:
/// `n_i = Re c_i^H S D S c_i > ½`.
fn actual_occupations(
    c: &Array2<Complex64>,
    s: &Array2<Complex64>,
    d: &Array2<Complex64>,
) -> Vec<f64> {
    let sds = s.dot(d).dot(s);
    let m = herm_t(c).dot(&sds).dot(c);
    (0..c.ncols())
        .map(|i| if m[(i, i)].re > 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn global_gap(eps: &[Vec<f64>], occ: &[Vec<f64>]) -> Option<f64> {
    let (mut omax, mut vmin) = (f64::NEG_INFINITY, f64::INFINITY);
    for (e, o) in eps.iter().zip(occ) {
        for (&x, &n) in e.iter().zip(o) {
            if n > 0.5 {
                omax = omax.max(x);
            } else {
                vmin = vmin.min(x);
            }
        }
    }
    (omax.is_finite() && vmin.is_finite()).then_some(vmin - omax)
}

#[allow(clippy::too_many_arguments)]
fn run_kuhf(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    mut inj: KPointInjection<'_>,
    na: usize,
    nb: usize,
    guess: Option<(&[Array2<Complex64>], &[Array2<Complex64>])>,
    budget_bytes: Option<usize>,
    mutation: Option<KUhfMutation>,
) -> Result<KUScfResult, FerricError> {
    let nk = mesh.nk();
    if inj.s.len() != nk || inj.h.len() != nk {
        return Err(FerricError::General(format!(
            "solve_kuhf: {} S / {} h matrices for {nk} k-points",
            inj.s.len(),
            inj.h.len()
        )));
    }
    let n = inj.s[0].nrows();
    if inj.s.iter().chain(inj.h.iter()).any(|m| m.dim() != (n, n)) {
        return Err(FerricError::General(
            "solve_kuhf: S(k)/h(k) must all be square of one size".into(),
        ));
    }
    if !inj.vnn.is_finite() {
        return Err(FerricError::General(format!(
            "solve_kuhf: non-finite E_nn {}",
            inj.vnn
        )));
    }
    let nelec = cell.mol().nelec();
    if na + nb == 0 || nelec < 0 || (na + nb) as i64 != nelec as i64 {
        return Err(FerricError::General(format!(
            "solve_kuhf: N_alpha {na} + N_beta {nb} per cell must equal the cell's (positive) \
             electron count {nelec}"
        )));
    }
    if !(cfg.lindep > 0.0) || !(cfg.min_gap >= 0.0) || cfg.max_iter == 0 {
        return Err(FerricError::General(format!(
            "solve_kuhf: invalid config (lindep {}, min_gap {}, max_iter {})",
            cfg.lindep, cfg.min_gap, cfg.max_iter
        )));
    }
    // Work arrays: per spin D, J/K, F, commutators, extrapolated F, MOs,
    // new D (~16 Nk n² in flight), X (1), DIIS history (2 spins × Fock+error
    // × cap).
    let per = (nk * n * n) as u64;
    let n_mats = 17 + 4 * cfg.diis_space as u64;
    let mut ledger = Ledger::new(crate::budget::resolve(budget_bytes));
    ledger.reserve(
        &format!(
            "k-UHF SCF work arrays ({n_mats} x N_k n^2 complex; nao = {n}, N_k = {nk}, DIIS {})",
            cfg.diis_space
        ),
        bytes_of(per.saturating_mul(n_mats), 16),
    )?;

    // Orthogonalisers once per k (shared by both spins); partners by
    // conjugation.
    let (x, lindep_report) = orthogonalizers_with_report(mesh, &inj.s, cfg.lindep, "solve_kuhf")?;
    let nmax = na.max(nb);
    for (k, xk) in x.iter().enumerate() {
        if xk.ncols() < nmax {
            return Err(FerricError::General(format!(
                "solve_kuhf: {nmax} occupied orbitals of one spin but S(k={k}) keeps only {} of {n} \
                 after the lindep cut",
                xk.ncols()
            )));
        }
    }
    // Global aufbau needs N_σ N_k levels over the mesh (always satisfied
    // when every k keeps >= N_σ, checked above).

    let zero = || Array2::<Complex64>::zeros((n, n));
    let zeros = || -> Vec<Array2<Complex64>> { (0..nk).map(|_| zero()).collect() };
    let (mut da, mut db) = match guess {
        None => (zeros(), zeros()),
        Some((ga, gb)) => {
            if ga.len() != nk
                || gb.len() != nk
                || ga.iter().chain(gb.iter()).any(|m| m.dim() != (n, n))
            {
                return Err(FerricError::General(format!(
                    "solve_kuhf: guess needs {nk} ({n}, {n}) densities per spin"
                )));
            }
            (ga.to_vec(), gb.to_vec())
        }
    };
    let mut ja = zeros();
    let mut jb = zeros();
    let mut ka = zeros();
    let mut kb = zeros();
    let mut diis = KDiis::new(cfg.diis_space);
    let mut e_old = f64::NAN;
    let inv_nk = 1.0 / nk as f64;
    let per_k = mutation == Some(KUhfMutation::PerKAufbau);
    // (energy, F_α, F_β, D_α, D_β, emax, iteration) of the latest evaluation.
    #[allow(clippy::type_complexity)]
    let mut last: Option<(
        f64,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        f64,
    )> = None;

    for it in 0..cfg.max_iter {
        // ---- J/K per spin (linear builder, called with D_σ).
        if mutation == Some(KUhfMutation::KFromTotalDensity) {
            let dt: Vec<Array2<Complex64>> = da.iter().zip(&db).map(|(a, b)| a + b).collect();
            inj.jk.build(&dt, &mut ja, &mut ka)?;
            for k in 0..nk {
                jb[k].fill(Complex64::new(0.0, 0.0));
                kb[k].assign(&ka[k]);
            }
        } else {
            inj.jk.build(&da, &mut ja, &mut ka)?;
            if nb > 0 {
                inj.jk.build(&db, &mut jb, &mut kb)?;
            } else {
                for k in 0..nk {
                    jb[k].fill(Complex64::new(0.0, 0.0));
                    kb[k].fill(Complex64::new(0.0, 0.0));
                }
            }
        }
        let jt: Vec<Array2<Complex64>> = (0..nk).map(|k| &ja[k] + &jb[k]).collect();
        let fock = |kx: &[Array2<Complex64>]| -> Vec<Array2<Complex64>> {
            (0..nk)
                .map(|k| hermitize(&(&(&inj.h[k] + &jt[k]) - &kx[k])))
                .collect()
        };
        let fa = fock(&ka);
        let fb = fock(&kb);
        let mut e_elec = 0.0;
        for k in 0..nk {
            e_elec += 0.5 * re_tr_prod(&(&inj.h[k] + &fa[k]), &da[k]);
            e_elec += 0.5 * re_tr_prod(&(&inj.h[k] + &fb[k]), &db[k]);
        }
        let energy = e_elec * inv_nk + inj.vnn;
        // ---- stacked (spin, k) commutators [α(k0..), β(k0..)].
        let mut errs: Vec<Array2<Complex64>> = Vec::with_capacity(2 * nk);
        for (f, d) in [(&fa, &da), (&fb, &db)] {
            for k in 0..nk {
                let fds = f[k].dot(&d[k]).dot(&inj.s[k]);
                let comm = &fds - &herm_t(&fds);
                errs.push(herm_t(&x[k]).dot(&comm).dot(&x[k]));
            }
        }
        let emax = errs
            .iter()
            .flat_map(|e| e.iter())
            .fold(0.0_f64, |a, z| a.max(z.norm()));
        if it > 0 && (energy - e_old).abs() < cfg.energy_conv && emax < cfg.grad_conv {
            return finish(
                mesh,
                &inj,
                &x,
                &lindep_report,
                na,
                nb,
                energy,
                fa,
                fb,
                da,
                db,
                true,
                it,
                emax,
            );
        }
        e_old = energy;
        let mut stacked: Vec<Array2<Complex64>> = Vec::with_capacity(2 * nk);
        stacked.extend(fa.iter().cloned());
        stacked.extend(fb.iter().cloned());
        let f_use = if it > 0 {
            diis.step(&stacked, &errs)
        } else {
            stacked
        };
        let (eps_a, ca) = diagonalize_all(mesh, &f_use[..nk], &x)?;
        let (eps_b, cb) = diagonalize_all(mesh, &f_use[nk..], &x)?;
        let occ_a = spin_occupations(&eps_a, na, cfg.min_gap, per_k, "alpha")?;
        let occ_b = spin_occupations(&eps_b, nb, cfg.min_gap, per_k, "beta")?;
        let new_da: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| occupied_projector(&ca[k], &occ_a[k]))
            .collect();
        let new_db: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| occupied_projector(&cb[k], &occ_b[k]))
            .collect();
        let old_da = std::mem::replace(&mut da, new_da);
        let old_db = std::mem::replace(&mut db, new_db);
        last = Some((energy, fa, fb, old_da, old_db, emax));
    }
    // Not converged: report the last evaluated energy with the Fock and
    // density that produced it (consistent triple).
    let (energy, fa, fb, lda, ldb, emax) = last.expect("max_iter >= 1");
    finish(
        mesh,
        &inj,
        &x,
        &lindep_report,
        na,
        nb,
        energy,
        fa,
        fb,
        lda,
        ldb,
        false,
        cfg.max_iter,
        emax,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish(
    mesh: &KPointMesh,
    inj: &KPointInjection<'_>,
    x: &[Array2<Complex64>],
    lindep: &crate::lindep::LindepReport,
    na: usize,
    nb: usize,
    energy: f64,
    fa: Vec<Array2<Complex64>>,
    fb: Vec<Array2<Complex64>>,
    da: Vec<Array2<Complex64>>,
    db: Vec<Array2<Complex64>>,
    converged: bool,
    iterations: usize,
    max_error: f64,
) -> Result<KUScfResult, FerricError> {
    let nk = mesh.nk();
    let (eps_a, ca) = diagonalize_all(mesh, &fa, x)?;
    let (eps_b, cb) = diagonalize_all(mesh, &fb, x)?;
    let occ_a: Vec<Vec<f64>> = (0..nk)
        .map(|k| actual_occupations(&ca[k], &inj.s[k], &da[k]))
        .collect();
    let occ_b: Vec<Vec<f64>> = (0..nk)
        .map(|k| actual_occupations(&cb[k], &inj.s[k], &db[k]))
        .collect();
    let count = |o: &Vec<Vec<f64>>| -> Vec<usize> {
        o.iter()
            .map(|v| v.iter().filter(|&&n| n > 0.5).count())
            .collect()
    };
    let nocc_per_k_alpha = count(&occ_a);
    let nocc_per_k_beta = count(&occ_b);
    let gap_alpha = global_gap(&eps_a, &occ_a);
    let gap_beta = global_gap(&eps_b, &occ_b);
    // Giant-determinant <S^2>: Σ_k ||C_α,occ^H S C_β,occ||² = Σ_k Re tr(D_α S D_β S).
    let sz = 0.5 * (na as f64 - nb as f64) * nk as f64;
    let mut ov = 0.0;
    if na > 0 && nb > 0 {
        for k in 0..nk {
            let das = da[k].dot(&inj.s[k]);
            let dbs = db[k].dot(&inj.s[k]);
            ov += re_tr_prod(&das, &dbs);
        }
    }
    let s2 = sz * (sz + 1.0) + (nb * nk) as f64 - ov;
    Ok(KUScfResult {
        energy,
        e_nuc: inj.vnn,
        nalpha: na,
        nbeta: nb,
        eps_alpha: eps_a,
        eps_beta: eps_b,
        mos_alpha: ca,
        mos_beta: cb,
        density_alpha: da,
        density_beta: db,
        fock_alpha: fa,
        fock_beta: fb,
        occ_alpha: occ_a,
        occ_beta: occ_b,
        nocc_per_k_alpha,
        nocc_per_k_beta,
        gap_alpha,
        gap_beta,
        s2,
        converged,
        iterations,
        max_error,
        kpts: mesh.kpts().to_vec(),
        lindep: lindep.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_occupations_allow_zero_and_per_k_mutant_differs_from_global() {
        // Band overlap: both lowest levels at k = 0 (the zchain pattern).
        let eps = vec![vec![-1.0, -0.8], vec![-0.2, 0.5]];
        let g = spin_occupations(&eps, 1, 1e-6, false, "alpha").unwrap();
        assert_eq!(g, vec![vec![1.0, 1.0], vec![0.0, 0.0]]);
        let p = spin_occupations(&eps, 1, 1e-6, true, "alpha").unwrap();
        assert_eq!(p, vec![vec![1.0, 0.0], vec![1.0, 0.0]]);
        let z = spin_occupations(&eps, 0, 1e-6, false, "beta").unwrap();
        assert_eq!(z, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(global_gap(&eps, &g), Some(-0.2 - -0.8));
        assert_eq!(global_gap(&eps, &z), None);
    }
}
