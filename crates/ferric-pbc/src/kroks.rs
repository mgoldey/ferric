//! k-point restricted open-shell HF / Kohn-Sham (`solve_krohf`,
//! `solve_kroks`): the k-mesh analogue of [`crate::rohf::gamma_roks`], with
//! LDA, GGA and global-hybrid functionals.
//!
//! One spatial orbital set per k with closed (2), open (1) and virtual (0)
//! levels; the per-spin Focks of the k-point UHF/UKS,
//!
//! ```text
//! F_σ(k) = h(k) + J[D_α + D_β](k) − a·K[D_σ](k) + V_σ^xc(k)      (K includes v_M S D_σ S)
//! ```
//!
//! are combined into the Guest-Saunders/Roothaan effective Fock (PySCF
//! `pbc.scf.krohf.get_roothaan_fock`, with `P† … P` complex adjoints)
//!
//! ```text
//! P_c = D_β S,  P_o = (D_α − D_β) S,  P_v = 1 − D_α S,   F_c = (F_α + F_β)/2
//! F_eff = ½ P_c† F_c P_c + ½ P_o† F_c P_o + ½ P_v† F_c P_v
//!         + P_o† F_β P_c + P_o† F_α P_v + P_v† F_c P_c  + h.c.
//! ```
//!
//! and diagonalised per k in the canonical orthogonaliser basis.
//!
//! * Occupations are PySCF `KROHF.get_occ`, GLOBAL over the mesh: the
//!   `N_β N_k` lowest Roothaan levels are closed; of the rest, the
//!   `(N_α − N_β) N_k` lowest by `ε^α = c† F_α c` are open.
//! * DIIS (real weights, so time reversal survives) on `F_eff` with the
//!   error `X†([F_eff, D_α] + [F_eff, D_β]) X` summed over the two spin
//!   commutators: it is nonzero exactly on the closed-open, closed-virtual
//!   and open-virtual blocks (twice on closed-virtual), i.e. it vanishes iff
//!   the ROHF orbital gradient does.
//! * `E/cell = (1/N_k) Σ_k Σ_σ ½ Re tr[(h + F_σ^{nox}) D_σ] + E_xc + E_nn`
//!   (`F_σ^{nox}` without `V_xc`); `a = 1` and `E_xc = V_xc = 0` for ROHF.
//! * `⟨S²⟩` of the giant (supercell) determinant is exactly
//!   `S_z(S_z+1)`, `S_z = (N_α − N_β) N_k / 2` (the β orbitals are a subset
//!   of the α ones), computed from the densities as a check.
//!
//! # Ewald trap
//!
//! As for k-UHF ([`crate::kuscf`]): ewald lowers every σ-occupied level by
//! `a·v_M`, so a hole shallower than that is aufbau-consistent under ewald.
//! [`EwaldStart::Staged`] (default when `a > 0`) converges with
//! `exxdiv = none` first and continues with ewald from that density; the
//! per-spin gaps (from the actual occupations) are reported against `a·v_M`.
//!
//! Semilocal XC uses the Bloch AO cache of [`crate::kdft::KPeriodicXc`] with
//! the spin-polarized libxc kernel (`V_σ = ∂E_xc/∂D_σ(k)`, no `1/N_k`).
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::dft::{resolve_periodic_functional, PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use crate::hcore::kpoint::{hermitize, periodic_hcore_kpts};
use crate::hcore::PeriodicHcoreConfig;
use crate::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use crate::kdft::{conj, KPeriodicXc};
use crate::kpts::KPointMesh;
use crate::kscf::{
    diagonalize_all, herm_t, occupied_projector, orthogonalizers_with_report, KDiis, KJkKind,
    KPointInjection, KPointJk, KScfConfig,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use crate::timing::{PbcTimings, StageClock};
use crate::uhf::{nocc_ab, EwaldStart, SpinGapReport};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::vxc::{polarized_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;

/// TEST ONLY: deliberate defects the k-ROKS tests must catch.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KRohfMutation {
    /// Diagonalise the bare `F_α(k)` instead of the Roothaan effective Fock
    /// (a UHF-like α Fock with ROHF occupations). Invisible in the closed-shell
    /// limit (`F_α = F_β`); seen by every open-shell anchor.
    BareAlphaFock,
    /// `N_β` closed then `N_α − N_β` open levels at EVERY k (molecule-style
    /// per-k aufbau instead of the global one).
    PerKAufbau,
    /// Exact exchange not scaled by `a` (full K in a hybrid).
    UnscaledExchange,
}

/// How the `(N_α − N_β) N_k` open orbitals are chosen among the levels above
/// the closed ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenOrbitalRule {
    /// The lowest Roothaan levels above the closed ones (aufbau on `F_eff`;
    /// what ferric's Gamma ROHF/ROKS does). Default.
    #[default]
    RoothaanOrder,
    /// The lowest by `ε^α = c† F_α c` among the levels above the closed ones
    /// (PySCF `KROHF.get_occ`). Can cycle between two states where the
    /// converged state is not an `F_eff`-aufbau fixed point.
    AlphaEnergy,
}

/// Spin-resolved semilocal exchange-correlation for the k-point ROKS
/// ([`solve_kroks_injected`]).
pub trait KSpinXc {
    /// `(E_xc per cell, V_α(k), V_β(k))` for the Hermitian unit-occupation
    /// per-spin densities `D_σ(k) = C_σ,occ C_σ,occ†`.
    #[allow(clippy::type_complexity)]
    fn build_spin(
        &mut self,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), FerricError>;
}

impl KPeriodicXc {
    /// Per-spin `ρ_σ`, `∇ρ_σ`, `σ_{σσ'}` of one chunk from the k densities.
    fn chunk_density_spin(
        &self,
        c: &crate::kdft::KAoChunk,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> UksDensityGrid {
        let np = c.points.len();
        let inv_nk = 1.0 / self.nk as f64;
        let mut rho = [Array1::<f64>::zeros(np), Array1::<f64>::zeros(np)];
        let mut grad = [
            ndarray::Array2::<f64>::zeros((3, np)),
            ndarray::Array2::<f64>::zeros((3, np)),
        ];
        for (s, dm) in [da, db].into_iter().enumerate() {
            for k in 0..self.nk {
                let p = dm[k].dot(&conj(&c.chi[k]));
                for g in 0..np {
                    let mut r = Complex64::new(0.0, 0.0);
                    for mu in 0..self.nbf {
                        r += c.chi[k][(mu, g)] * p[(mu, g)];
                    }
                    rho[s][g] += inv_nk * r.re;
                    if self.gga {
                        for ax in 0..3 {
                            let mut gr = Complex64::new(0.0, 0.0);
                            for mu in 0..self.nbf {
                                gr += c.dchi[k][(ax, mu, g)] * p[(mu, g)];
                            }
                            grad[s][(ax, g)] += inv_nk * 2.0 * gr.re;
                        }
                    }
                }
            }
        }
        let [rho_a, rho_b] = rho;
        let [grad_a, grad_b] = grad;
        let mut sigma = ndarray::Array2::<f64>::zeros((3, np));
        for g in 0..np {
            for ax in 0..3 {
                sigma[(0, g)] += grad_a[(ax, g)] * grad_a[(ax, g)];
                sigma[(1, g)] += grad_a[(ax, g)] * grad_b[(ax, g)];
                sigma[(2, g)] += grad_b[(ax, g)] * grad_b[(ax, g)];
            }
        }
        UksDensityGrid {
            rho_a,
            rho_b,
            grad_a,
            grad_b,
            sigma,
        }
    }
}

impl KSpinXc for KPeriodicXc {
    fn build_spin(
        &mut self,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), FerricError> {
        self.check(da)?;
        self.check(db)?;
        let n = self.nbf;
        let nk = self.nk;
        let zeros = || -> Vec<Array2<Complex64>> {
            (0..nk)
                .map(|_| Array2::<Complex64>::zeros((n, n)))
                .collect()
        };
        let mut e = 0.0;
        let mut va = zeros();
        let mut vb = zeros();
        for c in &self.chunks {
            let dens = self.chunk_density_spin(c, da, db);
            let kern = polarized_kernel(&dens, None, &self.xc_pol);
            let np = c.points.len();
            for g in 0..np {
                e += c.points[g].weight * (dens.rho_a[g] + dens.rho_b[g]) * kern.exc[g];
            }
            for spin in 0..2 {
                let (rho_s, vrho, vss, grad_self, grad_cross, v) = if spin == 0 {
                    (
                        &dens.rho_a,
                        &kern.vrho_a,
                        &kern.vsigma_aa,
                        &dens.grad_a,
                        &dens.grad_b,
                        &mut va,
                    )
                } else {
                    (
                        &dens.rho_b,
                        &kern.vrho_b,
                        &kern.vsigma_bb,
                        &dens.grad_b,
                        &dens.grad_a,
                        &mut vb,
                    )
                };
                let a_w: Vec<f64> = (0..np)
                    .map(|g| {
                        if rho_s[g] > DENSITY_FLOOR {
                            c.points[g].weight * vrho[g]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let f_ax: Vec<Vec<f64>> = if self.gga {
                    (0..3)
                        .map(|ax| {
                            (0..np)
                                .map(|g| {
                                    if rho_s[g] > DENSITY_FLOOR {
                                        c.points[g].weight
                                            * (2.0 * vss[g] * grad_self[(ax, g)]
                                                + kern.vsigma_ab[g] * grad_cross[(ax, g)])
                                    } else {
                                        0.0
                                    }
                                })
                                .collect()
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                for k in 0..nk {
                    let scale = |fac: &[f64]| {
                        let mut m = conj(&c.chi[k]);
                        for g in 0..np {
                            m.column_mut(g).mapv_inplace(|z| z * fac[g]);
                        }
                        m
                    };
                    v[k] += &scale(&a_w).dot(&c.chi[k].t());
                    if self.gga {
                        for ax in 0..3 {
                            let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                            let m = scale(&f_ax[ax]).dot(&d_ax.t());
                            let mh = conj(&m).t().to_owned();
                            v[k] += &m;
                            v[k] += &mh;
                        }
                    }
                }
            }
        }
        if !e.is_finite()
            || va
                .iter()
                .chain(vb.iter())
                .any(|m| m.iter().any(|z| !z.re.is_finite()))
        {
            return Err(FerricError::General(format!(
                "KPeriodicXc: non-finite spin-polarized E_xc/V_xc ({e}) for {}",
                self.name
            )));
        }
        for m in va.iter_mut().chain(vb.iter_mut()) {
            let h = conj(m).t().to_owned();
            *m = (&*m + &h).mapv(|z| z * 0.5);
        }
        Ok((e, va, vb))
    }
}

/// Result of a k-point ROHF/ROKS SCF (one stage).
#[derive(Debug, Clone)]
pub struct KRoScfResult {
    /// Total energy PER CELL (Hartree), of `density_alpha`/`density_beta`.
    pub energy: f64,
    /// Nuclear repulsion per cell.
    pub e_nuc: f64,
    /// `E_xc` per cell at the densities (0 for ROHF).
    pub e_xc: f64,
    /// α electrons per cell.
    pub nalpha: usize,
    /// β electrons per cell.
    pub nbeta: usize,
    /// Roothaan effective-Fock orbital energies per k (ascending), mesh order.
    pub eps: Vec<Vec<f64>>,
    /// `c† F_α c` per k and orbital.
    pub eps_alpha: Vec<Vec<f64>>,
    /// `c† F_β c` per k and orbital.
    pub eps_beta: Vec<Vec<f64>>,
    /// MO coefficients per k, `(nao, n_orth)` (one set for both spins).
    pub mos: Vec<Array2<Complex64>>,
    /// `D_α(k)` (unit occupation).
    pub density_alpha: Vec<Array2<Complex64>>,
    /// `D_β(k)`.
    pub density_beta: Vec<Array2<Complex64>>,
    /// `F_α(k)` of the densities (including `V_α^xc`).
    pub fock_alpha: Vec<Array2<Complex64>>,
    /// `F_β(k)`.
    pub fock_beta: Vec<Array2<Complex64>>,
    /// Roothaan `F_eff(k)` of the densities (not extrapolated or shifted).
    pub fock_eff: Vec<Array2<Complex64>>,
    /// Occupations 2 / 1 / 0 per k from the ACTUAL densities
    /// (`n_i^α + n_i^β`), aligned with `eps`.
    pub occupations: Vec<Vec<f64>>,
    /// Closed (doubly occupied) levels per k.
    pub nclosed_per_k: Vec<usize>,
    /// Open (α-only) levels per k.
    pub nopen_per_k: Vec<usize>,
    /// Per-spin gap `min ε^σ(unocc) − max ε^σ(occ)` over all k from the
    /// actual occupations (negative = hole below the Fermi level).
    pub gap_alpha: Option<f64>,
    /// See `gap_alpha`.
    pub gap_beta: Option<f64>,
    /// `⟨S²⟩` of the giant determinant (`S_z(S_z+1)` up to numerical noise).
    pub s2: f64,
    /// Converged flag.
    pub converged: bool,
    /// SCF iterations run.
    pub iterations: usize,
    /// Final max error element.
    pub max_error: f64,
    /// Cartesian k-points (Bohr⁻¹), mesh order.
    pub kpts: Vec<[f64; 3]>,
    /// Per-k canonical-cut diagnostics.
    pub lindep: crate::lindep::LindepReport,
}

/// Per-spin gap report of `r` against `shift` (`a·v_M` applied).
pub fn kroks_gap_report(r: &KRoScfResult, shift: f64) -> SpinGapReport {
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

/// Settings of [`solve_krohf`] (and the SCF/J-K part of [`KRoksConfig`]).
#[derive(Debug, Clone, Copy)]
pub struct KRohfConfig {
    /// SCF settings (`min_gap` applies to the closed and the open selection).
    pub scf: KScfConfig,
    /// Exchange-divergence treatment.
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald` with `a > 0`.
    pub ewald_start: EwaldStart,
    /// One-electron lattice sums.
    pub hcore: PeriodicHcoreConfig,
    /// J/K builder.
    pub jk: KJkKind,
    /// Dense-AFT settings (`jk = Dense`).
    pub dense: KDenseAftConfig,
    /// k-point RS-GDF settings (`jk = RsGdf`; `exxdiv` above decides).
    pub rsgdf: KRsGdfConfig,
    /// Budget for the SCF work arrays (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    /// Open-orbital selection rule ([`OpenOrbitalRule`]).
    pub open_rule: OpenOrbitalRule,
    /// Virtual-space level shift (Hartree), ramped as `ls·err/(err + 1e-3)`
    /// so it vanishes at convergence (the Gamma ROKS convention).
    pub level_shift: f64,
    #[doc(hidden)]
    pub mutation: Option<KRohfMutation>,
}

impl KRohfConfig {
    /// Defaults (dense J/K, staged ewald start, no level shift).
    pub fn for_cell(cell: &Cell, exxdiv: ExxDiv) -> Self {
        Self {
            scf: KScfConfig::default(),
            exxdiv,
            ewald_start: EwaldStart::Staged,
            hcore: PeriodicHcoreConfig::for_cell(cell),
            jk: KJkKind::Dense,
            dense: KDenseAftConfig::default(),
            rsgdf: KRsGdfConfig::default(),
            budget_bytes: None,
            open_rule: OpenOrbitalRule::default(),
            level_shift: 0.0,
            mutation: None,
        }
    }
}

/// Settings of [`solve_kroks`].
#[derive(Debug, Clone)]
pub struct KRoksConfig {
    /// SCF, J/K source, exxdiv and one-electron settings.
    pub rohf: KRohfConfig,
    /// LDA / GGA / global-hybrid name.
    pub functional: String,
    /// The periodic grid.
    pub grid: PeriodicGridConfig,
    /// AO truncation / budget of the Bloch AO cache.
    pub xc: PeriodicXcConfig,
}

impl KRoksConfig {
    /// Defaults (dense J/K, grid (75, 302) SSF) for `functional`.
    pub fn new(cell: &Cell, exxdiv: ExxDiv, functional: &str) -> Self {
        Self {
            rohf: KRohfConfig::for_cell(cell, exxdiv),
            functional: functional.to_string(),
            grid: PeriodicGridConfig::default(),
            xc: PeriodicXcConfig::default(),
        }
    }
}

/// Result of [`solve_krohf`] / [`solve_kroks`].
#[derive(Debug, Clone)]
#[must_use]
pub struct KRoksResult {
    /// The final SCF (requested `exxdiv`).
    pub scf: KRoScfResult,
    /// The `exxdiv = none` first stage, when the staged start ran.
    pub none_stage: Option<KRoScfResult>,
    /// The mesh (supercell) `v_M`, applied (times `a`) iff `exxdiv = ewald`.
    pub madelung: f64,
    /// The functional's global exact-exchange fraction (1 for ROHF).
    pub exact_exchange_fraction: f64,
    /// `(N_α, N_β)` per cell.
    pub nocc: (usize, usize),
    /// Per-spin gaps (actual occupations) against `a·v_M` applied.
    pub gaps: SpinGapReport,
    /// `E_xc` at the converged densities (0 for ROHF).
    pub e_xc: f64,
    /// Grid points used (0 for ROHF).
    pub n_grid_points: usize,
    /// `∫_cell ρ` at the converged densities (0 for ROHF).
    pub electrons_on_grid: f64,
    /// Coarse stages and J/K counters.
    pub timings: PbcTimings,
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

/// k-point ROHF of `cell` (spin state from `cell.mol()`'s charge and
/// multiplicity, per cell). `aux` is required iff `cfg.jk = RsGdf`.
pub fn solve_krohf(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRohfConfig,
) -> Result<KRoksResult, FerricError> {
    drive("solve_krohf", cell, prep, aux, mesh, cfg, None)
}

/// k-point ROKS on the atom-centred periodic grid built from `cfg.grid`.
pub fn solve_kroks(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRoksConfig,
) -> Result<KRoksResult, FerricError> {
    resolve_periodic_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    solve_kroks_on_grid(cell, prep, aux, mesh, &grid, cfg)
}

/// [`solve_kroks`] on a caller-supplied grid (`cfg.grid` is ignored).
pub fn solve_kroks_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KRoksConfig,
) -> Result<KRoksResult, FerricError> {
    // KPeriodicXc accepts meta-GGA for the closed-shell solver, but the
    // spin-resolved XC here has no tau term: refuse it on this entry point
    // too, not only in `solve_kroks`.
    resolve_periodic_functional(&cfg.functional)?;
    let mut pxc = KPeriodicXc::new(cell, prep.basis_set(), &cfg.functional, grid, mesh, &cfg.xc)?;
    let a = pxc.exact_exchange_fraction();
    let mut r = drive(
        "solve_kroks",
        cell,
        prep,
        aux,
        mesh,
        &cfg.rohf,
        Some((&mut pxc, a)),
    )?;
    let da = &r.scf.density_alpha;
    let tot: Vec<Array2<Complex64>> = da
        .iter()
        .zip(&r.scf.density_beta)
        .map(|(x, y)| x + y)
        .collect();
    r.electrons_on_grid = pxc.integrate_density(&tot)?;
    r.n_grid_points = grid.len();
    Ok(r)
}

fn drive(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRohfConfig,
    mut xc: Option<(&mut KPeriodicXc, f64)>,
) -> Result<KRoksResult, FerricError> {
    match (cfg.jk, aux) {
        (KJkKind::Dense, Some(_)) => {
            return Err(FerricError::General(format!(
                "{who}: an aux basis was given but jk = dense does not use it"
            )))
        }
        (KJkKind::RsGdf, None) => {
            return Err(FerricError::General(format!(
                "{who}: jk = rsgdf requires an auxbasis (none given)"
            )))
        }
        _ => {}
    }
    let a = xc.as_ref().map_or(1.0, |(_, a)| *a);
    if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction must be in [0, 1], got {a}"
        )));
    }
    let (na, nb) = nocc_ab(cell.mol())?;
    let total = StageClock::start();
    let mut timings = PbcTimings::default();
    let clock = StageClock::start();
    let hk = periodic_hcore_kpts(cell, prep, mesh, &cfg.hcore)?;
    timings.stop("k hcore", &clock);
    let v_m = mesh.madelung(cell)?;
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let clock = StageClock::start();
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
    timings.stop("k J/K build", &clock);
    if let KInts::RsGdf(g) = &ints {
        crate::rsgdf::kpoint::record_stats(&mut timings, g.stats());
    }
    let clock = StageClock::start();
    let mut run = |vm: f64,
                   guess: Option<(&[Array2<Complex64>], &[Array2<Complex64>])>|
     -> Result<KRoScfResult, FerricError> {
        let inj = KPointInjection {
            s: hk.s.clone(),
            h: hk.h.clone(),
            vnn: hk.enn,
            jk: jk_of(&ints, vm),
        };
        let xc_dyn: Option<&mut dyn KSpinXc> = match xc.as_mut() {
            Some((x, _)) => Some(&mut **x),
            None => None,
        };
        let r = run_kroks(
            cell,
            mesh,
            &cfg.scf,
            inj,
            na,
            nb,
            a,
            xc_dyn,
            guess,
            cfg.open_rule,
            cfg.level_shift,
            cfg.budget_bytes,
            cfg.mutation,
        )?;
        if !r.converged {
            return Err(FerricError::General(format!(
                "{who}: SCF (v_M {vm:.6}) not converged in {} iterations (max error {:.2e})",
                r.iterations, r.max_error
            )));
        }
        Ok(r)
    };
    let staged = cfg.exxdiv == ExxDiv::Ewald && cfg.ewald_start == EwaldStart::Staged && a > 0.0;
    let (scf, none_stage) = if staged {
        let first = run(0.0, None)?;
        let second = run(
            applied,
            Some((&first.density_alpha[..], &first.density_beta[..])),
        )?;
        (second, Some(first))
    } else {
        (run(applied, None)?, None)
    };
    timings.stop("k SCF", &clock);
    let shift = a * applied;
    let gaps = kroks_gap_report(&scf, shift);
    if !gaps.satisfied() {
        eprintln!(
            "{who} WARNING: per-spin global gap below a*v_M (gap_alpha {:?}, gap_beta {:?}, \
             a*v_M {shift:.6}, margin {:?}). Under exxdiv=none this state has a hole below the \
             Fermi level: likely the Ewald trap. Use EwaldStart::Staged.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    timings.finish(&total);
    Ok(KRoksResult {
        e_xc: scf.e_xc,
        scf,
        none_stage,
        madelung: v_m,
        exact_exchange_fraction: a,
        nocc: (na, nb),
        gaps,
        n_grid_points: 0,
        electrons_on_grid: 0.0,
        timings,
    })
}

/// k-point ROKS on injected `S(k)`, `h(k)`, `E_nn`, J/K and spin-resolved
/// semilocal XC, from the core guess. `nalpha`/`nbeta` are PER CELL and must
/// sum to the cell's electron count. The builder's Madelung term is whatever
/// it carries (scaled by `exact_exchange` as a whole). Returns
/// `converged = false` rather than an error on hitting `max_iter`.
#[allow(clippy::too_many_arguments)]
pub fn solve_kroks_injected(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    inj: KPointInjection<'_>,
    nalpha: usize,
    nbeta: usize,
    xc: Option<&mut dyn KSpinXc>,
    exact_exchange: f64,
) -> Result<KRoScfResult, FerricError> {
    run_kroks(
        cell,
        mesh,
        cfg,
        inj,
        nalpha,
        nbeta,
        exact_exchange,
        xc,
        None,
        OpenOrbitalRule::default(),
        0.0,
        None,
        None,
    )
}

fn re_tr_prod(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    let n = a.nrows();
    let mut tr = 0.0;
    for m in 0..n {
        for nu in 0..n {
            tr += (a[(m, nu)] * b[(nu, m)]).re;
        }
    }
    tr
}

/// Roothaan effective Fock of one k (module doc), complex adjoints.
pub(crate) fn roothaan_fock_k(
    fa: &Array2<Complex64>,
    fb: &Array2<Complex64>,
    da: &Array2<Complex64>,
    db: &Array2<Complex64>,
    s: &Array2<Complex64>,
) -> Array2<Complex64> {
    let n = s.nrows();
    let fc = (fa + fb).mapv(|z| z * 0.5);
    let pc = db.dot(s);
    let po = (da - db).dot(s);
    let pv = Array2::<Complex64>::eye(n) - da.dot(s);
    let (pc_h, po_h, pv_h) = (herm_t(&pc), herm_t(&po), herm_t(&pv));
    let mut f = pc_h.dot(&fc).dot(&pc).mapv(|z| z * 0.5);
    f = f + po_h.dot(&fc).dot(&po).mapv(|z| z * 0.5);
    f = f + pv_h.dot(&fc).dot(&pv).mapv(|z| z * 0.5);
    f = f + po_h.dot(fb).dot(&pc);
    f = f + po_h.dot(fa).dot(&pv);
    f = f + pv_h.dot(&fc).dot(&pc);
    let fh = herm_t(&f);
    f + fh
}

/// Closed (2) / open (1) / virtual (0) labels per `(k, level)` (module doc).
/// `min_gap` is enforced on both selections (0 disables).
fn roothaan_occupations(
    eps: &[Vec<f64>],
    ea: &[Vec<f64>],
    na: usize,
    nb: usize,
    min_gap: f64,
    per_k: bool,
    rule: OpenOrbitalRule,
) -> Result<Vec<Vec<f64>>, FerricError> {
    let nk = eps.len();
    let mut occ: Vec<Vec<f64>> = eps.iter().map(|e| vec![0.0; e.len()]).collect();
    if per_k {
        for (k, o) in occ.iter_mut().enumerate() {
            for (i, v) in o.iter_mut().enumerate() {
                *v = if i < nb {
                    2.0
                } else if i < na {
                    1.0
                } else {
                    0.0
                };
            }
            let _ = k;
        }
        return Ok(occ);
    }
    let mut all: Vec<(f64, usize, usize)> = eps
        .iter()
        .enumerate()
        .flat_map(|(k, e)| e.iter().enumerate().map(move |(i, &v)| (v, k, i)))
        .collect();
    let (n_closed, n_open) = (nb * nk, (na - nb) * nk);
    if all.len() < n_closed + n_open {
        return Err(FerricError::General(format!(
            "k-ROKS: {} levels requested but {} orbitals exist",
            n_closed + n_open,
            all.len()
        )));
    }
    all.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    if n_closed > 0 {
        if let Some(next) = all.get(n_closed) {
            let gap = next.0 - all[n_closed - 1].0;
            if gap < min_gap {
                return Err(FerricError::General(format!(
                    "k-ROKS: closed-shell gap {gap:.3e} Ha < min_gap {min_gap:.1e} \
                     (metallic / fractional occupation is not supported)"
                )));
            }
        }
        for &(_, k, i) in &all[..n_closed] {
            occ[k][i] = 2.0;
        }
    }
    if n_open > 0 {
        let mut rest: Vec<(f64, usize, usize)> = all[n_closed..]
            .iter()
            .map(|&(e0, k, i)| match rule {
                OpenOrbitalRule::RoothaanOrder => (e0, k, i),
                OpenOrbitalRule::AlphaEnergy => (ea[k][i], k, i),
            })
            .collect();
        rest.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        if let Some(next) = rest.get(n_open) {
            let gap = next.0 - rest[n_open - 1].0;
            if gap < min_gap {
                return Err(FerricError::General(format!(
                    "k-ROKS: open-shell alpha gap {gap:.3e} Ha < min_gap {min_gap:.1e} \
                     (metallic / fractional occupation is not supported)"
                )));
            }
        }
        for &(_, k, i) in &rest[..n_open] {
            occ[k][i] = 1.0;
        }
    }
    Ok(occ)
}

/// `Re c_i† M c_i` per column.
fn diag_expect(c: &Array2<Complex64>, m: &Array2<Complex64>) -> Vec<f64> {
    let mc = m.dot(c);
    (0..c.ncols())
        .map(|i| {
            (0..c.nrows())
                .map(|r| (c[(r, i)].conj() * mc[(r, i)]).re)
                .sum()
        })
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
fn run_kroks(
    cell: &Cell,
    mesh: &KPointMesh,
    cfg: &KScfConfig,
    mut inj: KPointInjection<'_>,
    na: usize,
    nb: usize,
    a: f64,
    mut xc: Option<&mut dyn KSpinXc>,
    guess: Option<(&[Array2<Complex64>], &[Array2<Complex64>])>,
    open_rule: OpenOrbitalRule,
    level_shift: f64,
    budget_bytes: Option<usize>,
    mutation: Option<KRohfMutation>,
) -> Result<KRoScfResult, FerricError> {
    let nk = mesh.nk();
    if inj.s.len() != nk || inj.h.len() != nk {
        return Err(FerricError::General(format!(
            "solve_kroks: {} S / {} h matrices for {nk} k-points",
            inj.s.len(),
            inj.h.len()
        )));
    }
    let n = inj.s[0].nrows();
    if inj.s.iter().chain(inj.h.iter()).any(|m| m.dim() != (n, n)) {
        return Err(FerricError::General(
            "solve_kroks: S(k)/h(k) must all be square of one size".into(),
        ));
    }
    if !inj.vnn.is_finite() {
        return Err(FerricError::General(format!(
            "solve_kroks: non-finite E_nn {}",
            inj.vnn
        )));
    }
    let nelec = cell.mol().nelec();
    if na < nb || na + nb == 0 || nelec < 0 || (na + nb) as i64 != nelec as i64 {
        return Err(FerricError::General(format!(
            "solve_kroks: N_alpha {na} >= N_beta {nb} per cell must sum to the cell's (positive) \
             electron count {nelec}"
        )));
    }
    if !(cfg.lindep > 0.0) || !(cfg.min_gap >= 0.0) || cfg.max_iter == 0 {
        return Err(FerricError::General(format!(
            "solve_kroks: invalid config (lindep {}, min_gap {}, max_iter {})",
            cfg.lindep, cfg.min_gap, cfg.max_iter
        )));
    }
    if !(level_shift >= 0.0 && level_shift.is_finite()) {
        return Err(FerricError::General(format!(
            "solve_kroks: level_shift must be finite and >= 0, got {level_shift}"
        )));
    }
    // Work arrays: D, J, K, V per spin, F, F_eff, commutators, shifted copy,
    // MOs, new D (~24 N_k n² in flight), X, DIIS history (Fock + error).
    let per = (nk * n * n) as u64;
    let n_mats = 26 + 2 * cfg.diis_space as u64;
    let mut ledger = Ledger::new(crate::budget::resolve(budget_bytes));
    ledger.reserve(
        &format!(
            "k-ROKS SCF work arrays ({n_mats} x N_k n^2 complex; nao = {n}, N_k = {nk}, DIIS {})",
            cfg.diis_space
        ),
        bytes_of(per.saturating_mul(n_mats), 16),
    )?;
    let (x, lindep_report) = orthogonalizers_with_report(mesh, &inj.s, cfg.lindep, "solve_kroks")?;
    for (k, xk) in x.iter().enumerate() {
        if xk.ncols() < na {
            return Err(FerricError::General(format!(
                "solve_kroks: {na} occupied orbitals but S(k={k}) keeps only {} of {n} after the \
                 lindep cut",
                xk.ncols()
            )));
        }
    }
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
                    "solve_kroks: guess needs {nk} ({n}, {n}) densities per spin"
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
    let per_k = mutation == Some(KRohfMutation::PerKAufbau);
    let k_scale = if mutation == Some(KRohfMutation::UnscaledExchange) {
        1.0
    } else {
        a
    };
    #[allow(clippy::type_complexity)]
    let mut last: Option<(
        f64,
        f64,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        Vec<Array2<Complex64>>,
        f64,
    )> = None;

    for it in 0..cfg.max_iter {
        inj.jk.build(&da, &mut ja, &mut ka)?;
        if nb > 0 {
            inj.jk.build(&db, &mut jb, &mut kb)?;
        } else {
            for k in 0..nk {
                jb[k].fill(Complex64::new(0.0, 0.0));
                kb[k].fill(Complex64::new(0.0, 0.0));
            }
        }
        let fock_nox = |kx: &[Array2<Complex64>]| -> Vec<Array2<Complex64>> {
            (0..nk)
                .map(|k| {
                    let j = &ja[k] + &jb[k];
                    hermitize(&(&(&inj.h[k] + &j) - &kx[k].mapv(|z| z * k_scale)))
                })
                .collect()
        };
        let fa_nox = fock_nox(&ka);
        let fb_nox = fock_nox(&kb);
        let (e_xc, fa, fb) = match xc.as_mut() {
            None => (0.0, fa_nox.clone(), fb_nox.clone()),
            Some(b) => {
                let (e, va, vb) = b.build_spin(&da, &db)?;
                if va.len() != nk || vb.len() != nk {
                    return Err(FerricError::General(format!(
                        "solve_kroks: XC builder returned {}/{} V_xc(k) for {nk} k-points",
                        va.len(),
                        vb.len()
                    )));
                }
                let fa = (0..nk).map(|k| hermitize(&(&fa_nox[k] + &va[k]))).collect();
                let fb = (0..nk).map(|k| hermitize(&(&fb_nox[k] + &vb[k]))).collect();
                (e, fa, fb)
            }
        };
        let mut e_elec = 0.0;
        for k in 0..nk {
            e_elec += 0.5 * re_tr_prod(&(&inj.h[k] + &fa_nox[k]), &da[k]);
            e_elec += 0.5 * re_tr_prod(&(&inj.h[k] + &fb_nox[k]), &db[k]);
        }
        let energy = e_elec * inv_nk + e_xc + inj.vnn;
        let feff: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| {
                if mutation == Some(KRohfMutation::BareAlphaFock) {
                    fa[k].clone()
                } else {
                    roothaan_fock_k(&fa[k], &fb[k], &da[k], &db[k], &inj.s[k])
                }
            })
            .collect();
        let errs: Vec<Array2<Complex64>> = (0..nk)
            .map(|k| {
                let fs = feff[k].dot(&(&da[k] + &db[k])).dot(&inj.s[k]);
                let comm = &fs - &herm_t(&fs);
                herm_t(&x[k]).dot(&comm).dot(&x[k])
            })
            .collect();
        let emax = errs
            .iter()
            .flat_map(|e| e.iter())
            .fold(0.0_f64, |m, z| m.max(z.norm()));
        if it > 0 && (energy - e_old).abs() < cfg.energy_conv && emax < cfg.grad_conv {
            return finish(
                mesh,
                &inj,
                &x,
                &lindep_report,
                na,
                nb,
                energy,
                e_xc,
                fa,
                fb,
                feff,
                da,
                db,
                true,
                it,
                emax,
                cfg.min_gap,
                per_k,
                open_rule,
            );
        }
        e_old = energy;
        let mut f_use = if it > 0 {
            diis.step(&feff, &errs)
        } else {
            feff.clone()
        };
        if level_shift > 0.0 {
            let ls = level_shift * emax / (emax + 1e-3);
            for k in 0..nk {
                let sds = inj.s[k].dot(&da[k]).dot(&inj.s[k]);
                f_use[k] = &f_use[k] + &(&inj.s[k] - &sds).mapv(|z| z * ls);
            }
        }
        let (eps, cs) = diagonalize_all(mesh, &f_use, &x)?;
        let ea: Vec<Vec<f64>> = (0..nk).map(|k| diag_expect(&cs[k], &fa[k])).collect();
        let occ = roothaan_occupations(&eps, &ea, na, nb, 0.0, per_k, open_rule)?;
        let occ_a: Vec<Vec<f64>> = occ
            .iter()
            .map(|o| o.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect())
            .collect();
        let occ_b: Vec<Vec<f64>> = occ
            .iter()
            .map(|o| o.iter().map(|&v| if v > 1.5 { 1.0 } else { 0.0 }).collect())
            .collect();
        let new_da: Vec<_> = (0..nk)
            .map(|k| occupied_projector(&cs[k], &occ_a[k]))
            .collect();
        let new_db: Vec<_> = (0..nk)
            .map(|k| occupied_projector(&cs[k], &occ_b[k]))
            .collect();
        let old_da = std::mem::replace(&mut da, new_da);
        let old_db = std::mem::replace(&mut db, new_db);
        last = Some((energy, e_xc, fa, fb, feff, old_da, old_db, emax));
    }
    let (energy, e_xc, fa, fb, feff, lda, ldb, emax) = last.expect("max_iter >= 1");
    finish(
        mesh,
        &inj,
        &x,
        &lindep_report,
        na,
        nb,
        energy,
        e_xc,
        fa,
        fb,
        feff,
        lda,
        ldb,
        false,
        cfg.max_iter,
        emax,
        0.0,
        per_k,
        open_rule,
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
    e_xc: f64,
    fa: Vec<Array2<Complex64>>,
    fb: Vec<Array2<Complex64>>,
    feff: Vec<Array2<Complex64>>,
    da: Vec<Array2<Complex64>>,
    db: Vec<Array2<Complex64>>,
    converged: bool,
    iterations: usize,
    max_error: f64,
    min_gap: f64,
    per_k: bool,
    open_rule: OpenOrbitalRule,
) -> Result<KRoScfResult, FerricError> {
    let nk = mesh.nk();
    let (eps, cs) = diagonalize_all(mesh, &feff, x)?;
    let eps_a: Vec<Vec<f64>> = (0..nk).map(|k| diag_expect(&cs[k], &fa[k])).collect();
    let eps_b: Vec<Vec<f64>> = (0..nk).map(|k| diag_expect(&cs[k], &fb[k])).collect();
    if converged && min_gap > 0.0 {
        // The converged density must itself satisfy the gap requirement.
        roothaan_occupations(&eps, &eps_a, na, nb, min_gap, per_k, open_rule)?;
    }
    let actual = |d: &Array2<Complex64>, s: &Array2<Complex64>, c: &Array2<Complex64>| {
        diag_expect(c, &s.dot(d).dot(s))
    };
    let occ_a: Vec<Vec<f64>> = (0..nk)
        .map(|k| {
            actual(&da[k], &inj.s[k], &cs[k])
                .into_iter()
                .map(|v| if v > 0.5 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let occ_b: Vec<Vec<f64>> = (0..nk)
        .map(|k| {
            actual(&db[k], &inj.s[k], &cs[k])
                .into_iter()
                .map(|v| if v > 0.5 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let occupations: Vec<Vec<f64>> = occ_a
        .iter()
        .zip(&occ_b)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
        .collect();
    let nclosed_per_k = occupations
        .iter()
        .map(|o| o.iter().filter(|&&v| v > 1.5).count())
        .collect();
    let nopen_per_k = occupations
        .iter()
        .map(|o| o.iter().filter(|&&v| v > 0.5 && v < 1.5).count())
        .collect();
    let gap_alpha = global_gap(&eps_a, &occ_a);
    let gap_beta = global_gap(&eps_b, &occ_b);
    let sz = 0.5 * (na as f64 - nb as f64) * nk as f64;
    let mut ov = 0.0;
    if na > 0 && nb > 0 {
        for k in 0..nk {
            ov += re_tr_prod(&da[k].dot(&inj.s[k]), &db[k].dot(&inj.s[k]));
        }
    }
    let s2 = sz * (sz + 1.0) + (nb * nk) as f64 - ov;
    Ok(KRoScfResult {
        energy,
        e_nuc: inj.vnn,
        e_xc,
        nalpha: na,
        nbeta: nb,
        eps,
        eps_alpha: eps_a,
        eps_beta: eps_b,
        mos: cs,
        density_alpha: da,
        density_beta: db,
        fock_alpha: fa,
        fock_beta: fb,
        fock_eff: feff,
        occupations,
        nclosed_per_k,
        nopen_per_k,
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
