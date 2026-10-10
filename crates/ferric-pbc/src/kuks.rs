//! k-point open-shell Kohn-Sham DFT (`solve_kuks`: LDA, GGA, global hybrids):
//! the spin-polarized generalization of [`crate::kdft`] on the SCF skeleton
//! of [`crate::kuscf`] (k-point UHF).
//!
//! Per spin `σ` and mesh point `k` (unit occupations, `a` the global
//! exact-exchange fraction):
//!
//! ```text
//! F_σ(k) = h(k) + J[D_α + D_β](k) − a K[D_σ](k) + V_xc,σ(k)
//! E/cell = (1/N_k) Σ_k Σ_σ ½ Re tr[(h + F_σ^HF) D_σ] + E_xc + E_nn
//! ```
//!
//! (the electronic part is `½ tr[(h + F_σ^HF) D_σ]` with
//! `F_σ^HF = h + J − aK_σ`). The injected K carries the Madelung term and is
//! scaled as a whole, linear in `D_σ` (no per-spin ½). The XC part is
//! [`KPeriodicXcPolarized`]: `ρ_σ`, `∇ρ_σ` and the three `σ` channels from
//! the Bloch AO cache of [`KPeriodicXc`], the libxc polarized kernel of
//! ferric-dft, and per spin
//! `V_σ(k) = ∫ φ* [w v_ρσ] φ + ∫ (2 v_σσσ ∇ρ_σ + v_σαβ ∇ρ_other) ·
//! (∇φ* φ + φ* ∇φ)`.
//!
//! The Ewald trap of the k-UHF (see [`crate::kuscf`]) applies to hybrids with
//! `exxdiv = ewald`: for `a > 0` the default start is staged (none, then
//! ewald from that density) and the per-spin gaps are checked against
//! `a·v_M`. Range-separated hybrids, meta-GGA, VV10 and double hybrids are
//! refused exactly as at Gamma.

use crate::dense_aft::ExxDiv;
use crate::dft::{
    resolve_periodic_functional, PeriodicDftError, PeriodicGrid, PeriodicGridConfig,
    PeriodicXcConfig,
};
use crate::hcore::kpoint::periodic_hcore_kpts;
use crate::kdense_aft::KDenseAftEri;
use crate::kdft::{conj, KAoChunk, KPeriodicXc};
use crate::kpts::KPointMesh;
use crate::kscf::{KJkKind, KPointInjection, KPointJk};
use crate::kuscf::{
    kuhf_gap_report, run_kuhf, KPointXcPolarized, KUScfResult, KUhfConfig, KUhfMutation,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::KRsGdf;
use crate::timing::{PbcTimings, StageClock};
use crate::uhf::{nocc_ab, EwaldStart, SpinGapReport};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::libxc::{xc_def_from_name_nspin, XcDef};
use ferric_dft::vxc::{polarized_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Spin-polarized semilocal XC on the Bloch AO cache of a [`KPeriodicXc`].
pub struct KPeriodicXcPolarized {
    inner: KPeriodicXc,
    /// The functional on `nspin = 2` libxc handles.
    xc_pol: XcDef,
}

impl std::fmt::Debug for KPeriodicXcPolarized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KPeriodicXcPolarized")
            .field("inner", &self.inner)
            .finish()
    }
}

impl KPeriodicXcPolarized {
    /// Wrap an already built Bloch AO cache.
    pub fn from_closed(inner: KPeriodicXc) -> Result<Self, FerricError> {
        // KPeriodicXc may accept meta-GGA for the closed-shell solver; the
        // spin-resolved XC here has no tau term, so refuse it for every
        // construction path (including `solve_kuks_on_grid`).
        resolve_periodic_functional(&inner.name)?;
        let xc_pol = xc_def_from_name_nspin(&inner.name, 2).map_err(|e| {
            FerricError::General(format!(
                "periodic DFT: functional {:?} (spin-polarized): {e:?}",
                inner.name
            ))
        })?;
        Ok(Self { inner, xc_pol })
    }

    /// Resolve `functional` and build the Bloch AO cache (see
    /// [`KPeriodicXc::new`]).
    pub fn new(
        cell: &Cell,
        bs: &ferric_core::basis::BasisSet,
        functional: &str,
        grid: &PeriodicGrid,
        mesh: &KPointMesh,
        cfg: &PeriodicXcConfig,
    ) -> Result<Self, FerricError> {
        Self::from_closed(KPeriodicXc::new(cell, bs, functional, grid, mesh, cfg)?)
    }

    /// The functional's global exact-exchange fraction.
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.inner.exact_exchange_fraction()
    }

    fn check(&self, d: &[Array2<Complex64>]) -> Result<(), FerricError> {
        let x = &self.inner;
        if d.len() != x.nk || d.iter().any(|m| m.dim() != (x.nbf, x.nbf)) {
            return Err(FerricError::General(format!(
                "KPeriodicXcPolarized: expected {} density matrices of shape ({n}, {n})",
                x.nk,
                n = x.nbf
            )));
        }
        Ok(())
    }

    /// `(ρ, ∇ρ)` of one spin on one chunk, plus `P_σ[k]` is not kept.
    fn spin_density(&self, c: &KAoChunk, dm: &[Array2<Complex64>]) -> (Array1<f64>, Array2<f64>) {
        let x = &self.inner;
        let np = c.points.len();
        let inv_nk = 1.0 / x.nk as f64;
        let mut rho = Array1::<f64>::zeros(np);
        let mut grad = Array2::<f64>::zeros((3, np));
        for k in 0..x.nk {
            let p = dm[k].dot(&conj(&c.chi[k]));
            for g in 0..np {
                let mut r = Complex64::new(0.0, 0.0);
                for mu in 0..x.nbf {
                    r += c.chi[k][(mu, g)] * p[(mu, g)];
                }
                rho[g] += inv_nk * r.re;
                if x.gga {
                    for ax in 0..3 {
                        let mut gr = Complex64::new(0.0, 0.0);
                        for mu in 0..x.nbf {
                            gr += c.dchi[k][(ax, mu, g)] * p[(mu, g)];
                        }
                        grad[(ax, g)] += inv_nk * 2.0 * gr.re;
                    }
                }
            }
        }
        (rho, grad)
    }

    fn chunk_density(
        &self,
        c: &KAoChunk,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> UksDensityGrid {
        let np = c.points.len();
        let (rho_a, grad_a) = self.spin_density(c, da);
        let (rho_b, grad_b) = self.spin_density(c, db);
        let mut sigma = Array2::<f64>::zeros((3, np));
        for g in 0..np {
            let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
                (0..3).map(|ax| x[(ax, g)] * y[(ax, g)]).sum()
            };
            sigma[(0, g)] = dot(&grad_a, &grad_a);
            sigma[(1, g)] = dot(&grad_a, &grad_b);
            sigma[(2, g)] = dot(&grad_b, &grad_b);
        }
        UksDensityGrid {
            rho_a,
            rho_b,
            grad_a,
            grad_b,
            sigma,
        }
    }

    /// `∫_cell (ρ_α + ρ_β)` of the k densities.
    pub fn integrate_density(
        &self,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> Result<f64, FerricError> {
        self.check(da)?;
        self.check(db)?;
        Ok(self
            .inner
            .chunks
            .iter()
            .map(|c| {
                let d = self.chunk_density(c, da, db);
                c.points
                    .iter()
                    .enumerate()
                    .map(|(g, p)| p.weight * (d.rho_a[g] + d.rho_b[g]))
                    .sum::<f64>()
            })
            .sum())
    }
}

impl KPointXcPolarized for KPeriodicXcPolarized {
    fn build_polarized(
        &mut self,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), FerricError> {
        self.check(da)?;
        self.check(db)?;
        let x = &self.inner;
        let n = x.nbf;
        let zeros = || -> Vec<Array2<Complex64>> {
            (0..x.nk)
                .map(|_| Array2::<Complex64>::zeros((n, n)))
                .collect()
        };
        let mut e = 0.0;
        let mut va = zeros();
        let mut vb = zeros();
        for c in &x.chunks {
            let dens = self.chunk_density(c, da, db);
            let kern = polarized_kernel(&dens, None, &self.xc_pol);
            let np = c.points.len();
            for g in 0..np {
                e += c.points[g].weight * (dens.rho_a[g] + dens.rho_b[g]) * kern.exc[g];
            }
            // Per-spin weights (LDA piece) and axis functions (GGA piece),
            // each floored on its own spin density (as the Gamma builder).
            for (spin, v) in [(0usize, &mut va), (1usize, &mut vb)] {
                let (rho, vrho, vs_self, grad_self, grad_cross) = if spin == 0 {
                    (
                        &dens.rho_a,
                        &kern.vrho_a,
                        &kern.vsigma_aa,
                        &dens.grad_a,
                        &dens.grad_b,
                    )
                } else {
                    (
                        &dens.rho_b,
                        &kern.vrho_b,
                        &kern.vsigma_bb,
                        &dens.grad_b,
                        &dens.grad_a,
                    )
                };
                let a: Vec<f64> = (0..np)
                    .map(|g| {
                        if rho[g] > DENSITY_FLOOR {
                            c.points[g].weight * vrho[g]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let f_ax: Vec<Vec<f64>> = if x.gga {
                    (0..3)
                        .map(|ax| {
                            (0..np)
                                .map(|g| {
                                    if rho[g] > DENSITY_FLOOR {
                                        c.points[g].weight
                                            * (2.0 * vs_self[g] * grad_self[(ax, g)]
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
                for k in 0..x.nk {
                    let scale = |fac: &[f64]| {
                        let mut m = conj(&c.chi[k]);
                        for g in 0..np {
                            m.column_mut(g).mapv_inplace(|z| z * fac[g]);
                        }
                        m
                    };
                    v[k] += &scale(&a).dot(&c.chi[k].t());
                    if x.gga {
                        for ax in 0..3 {
                            let d_ax = c.dchi[k].index_axis(ndarray::Axis(0), ax);
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
                "KPeriodicXcPolarized: non-finite E_xc/V_xc ({e}) for {}",
                x.name
            )));
        }
        for m in va.iter_mut().chain(vb.iter_mut()) {
            let h = conj(m).t().to_owned();
            *m = (&*m + &h).mapv(|z| z * 0.5);
        }
        Ok((e, va, vb))
    }
}

/// Settings of [`solve_kuks`].
#[derive(Debug, Clone)]
pub struct KUksConfig {
    /// SCF, J/K source, exxdiv, staged start and one-electron settings.
    pub kuhf: KUhfConfig,
    /// LDA / GGA / global-hybrid name (as [`crate::kdft::KRksConfig`]).
    pub functional: String,
    /// The periodic grid.
    pub grid: PeriodicGridConfig,
    /// AO truncation / budget of the Bloch AO cache.
    pub xc: PeriodicXcConfig,
}

impl KUksConfig {
    /// Defaults (dense J/K, grid (75, 302) SSF, staged ewald start).
    pub fn new(cell: &Cell, exxdiv: ExxDiv, functional: &str) -> Self {
        Self {
            kuhf: KUhfConfig::for_cell(cell, exxdiv),
            functional: functional.to_string(),
            grid: PeriodicGridConfig::default(),
            xc: PeriodicXcConfig::default(),
        }
    }
}

/// Result of [`solve_kuks`].
#[derive(Debug, Clone)]
#[must_use]
pub struct KUksResult {
    /// The final SCF (energy per cell includes `E_xc` and `E_nn`).
    pub scf: KUScfResult,
    /// The `exxdiv = none` first stage, when the staged start ran.
    pub none_stage: Option<KUScfResult>,
    /// The mesh (supercell) `v_M`.
    pub madelung: f64,
    /// `(N_α, N_β)` per cell.
    pub nocc: (usize, usize),
    /// Per-spin gaps against `a·v_M` applied.
    pub gaps: SpinGapReport,
    /// `E_xc` at the converged densities.
    pub e_xc: f64,
    /// The functional's global exact-exchange fraction.
    pub exact_exchange_fraction: f64,
    /// Grid points used.
    pub n_grid_points: usize,
    /// `∫_cell ρ_total` at the converged densities (vs `N_e`).
    pub electrons_on_grid: f64,
    /// Coarse stages.
    pub timings: PbcTimings,
}

/// k-point UKS on the atom-centred periodic grid built from `cfg.grid`.
pub fn solve_kuks(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KUksConfig,
) -> Result<KUksResult, FerricError> {
    resolve_periodic_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    solve_kuks_on_grid(cell, prep, aux, mesh, &grid, cfg)
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

/// [`solve_kuks`] on a caller-supplied grid (`cfg.grid` is ignored).
pub fn solve_kuks_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KUksConfig,
) -> Result<KUksResult, FerricError> {
    let kc = &cfg.kuhf;
    match (kc.jk, aux) {
        (KJkKind::Dense, Some(_)) => {
            return Err(FerricError::General(
                "solve_kuks: an aux basis was given but jk = dense does not use it".into(),
            ))
        }
        (KJkKind::RsGdf, None) => {
            return Err(FerricError::General(
                "solve_kuks: jk = rsgdf requires an auxbasis (none given)".into(),
            ))
        }
        _ => {}
    }
    let (na, nb) = nocc_ab(cell.mol())?;
    let total = StageClock::start();
    let mut timings = PbcTimings::default();
    let mut pxc =
        KPeriodicXcPolarized::new(cell, prep.basis_set(), &cfg.functional, grid, mesh, &cfg.xc)?;
    let a = pxc.exact_exchange_fraction();
    if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
        return Err(PeriodicDftError::Unsupported {
            feature: "exact-exchange fraction",
            reason: format!("{a} outside [0, 1]"),
        }
        .into());
    }
    let clock = StageClock::start();
    let hk = periodic_hcore_kpts(cell, prep, mesh, &kc.hcore)?;
    timings.stop("k hcore", &clock);
    hk.record_stats(&mut timings);
    let v_m = mesh.madelung(cell)?;
    let applied = match kc.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let k_madelung = if kc.mutation == Some(KUhfMutation::MadelungHalfPerSpin) {
        0.5 * applied
    } else {
        applied
    };
    let clock = StageClock::start();
    let ints = match aux {
        None => KInts::Dense(KDenseAftEri::build(
            cell,
            prep,
            mesh,
            &hk.s,
            ExxDiv::None,
            &kc.dense,
        )?),
        Some(aux) => KInts::RsGdf(KRsGdf::build(cell, prep, aux, mesh, &hk.s, &kc.rsgdf)?),
    };
    timings.stop("k J/K build", &clock);
    if let KInts::RsGdf(g) = &ints {
        crate::rsgdf::kpoint::record_stats(&mut timings, g.stats());
    }
    let clock = StageClock::start();
    let mut run = |vm: f64,
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
            &kc.scf,
            inj,
            na,
            nb,
            guess,
            kc.budget_bytes,
            kc.mutation,
            Some((&mut pxc as &mut dyn KPointXcPolarized, a)),
        )?;
        if !r.converged {
            return Err(FerricError::General(format!(
                "solve_kuks: {} SCF (v_M {vm:.6}) not converged in {} iterations (max error \
                 {:.2e})",
                cfg.functional, r.iterations, r.max_error
            )));
        }
        Ok(r)
    };
    let staged = kc.exxdiv == ExxDiv::Ewald && kc.ewald_start == EwaldStart::Staged && a > 0.0;
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
    timings.stop("k SCF", &clock);
    let shift = a * applied;
    let gaps = kuhf_gap_report(&scf, shift);
    if !gaps.satisfied() {
        eprintln!(
            "solve_kuks WARNING: per-spin global gap below a*v_M (gap_alpha {:?}, gap_beta {:?}, \
             a*v_M {shift:.6}, margin {:?}): possible Ewald trap. Use EwaldStart::Staged.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    let electrons_on_grid = pxc.integrate_density(&scf.density_alpha, &scf.density_beta)?;
    let (e_xc, _, _) = pxc.build_polarized(&scf.density_alpha, &scf.density_beta)?;
    timings.finish(&total);
    Ok(KUksResult {
        e_xc,
        exact_exchange_fraction: a,
        n_grid_points: grid.len(),
        electrons_on_grid,
        madelung: v_m,
        nocc: (na, nb),
        gaps,
        none_stage,
        scf,
        timings,
    })
}
