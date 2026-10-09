//! k-point closed-shell Kohn-Sham DFT (LDA, GGA, global hybrids): the
//! k-mesh analogue of [`crate::dft::gamma_rks`].
//!
//! # What is new relative to Gamma
//!
//! * Bloch AOs on the periodic grid, `φ^k_μ(r) = Σ_L e^{ik·L} χ_μ(r − L)`
//!   (and `∇φ^k_μ`), the convention of the k-point overlap / hcore
//!   (`S(k)_{μν} = Σ_L e^{ik·L} ⟨μ0|νL⟩`, [`crate::hcore::kpoint`]). The grid
//!   is the Gamma [`PeriodicGrid`] (atom-centred, infinite-crystal SSF
//!   partition, integrates lattice-periodic functions over ONE cell).
//! * Density from all k: with `D(k)_{μν} = 2 Σ_i C_μi C*_νi`,
//!
//!   ```text
//!   ρ(r)  = (1/N_k) Σ_k Σ_{μν} D(k)_{μν} φ^k_μ(r) φ^k_ν(r)*
//!   ∇ρ(r) = (1/N_k) Σ_k 2 Re Σ_{μν} D(k)_{μν} ∇φ^k_μ φ^k_ν*
//!   ```
//!
//!   (real by construction; summed over a full `{k, −k}`-closed mesh it is
//!   lattice periodic).
//! * `V_xc(k)_{μν} = ∫_cell φ^k_μ* [v_ρ φ^k_ν] + ∫_cell 2 v_σ ∇ρ ·
//!   (∇φ^k_μ* φ^k_ν + φ^k_μ* ∇φ^k_ν)`, the derivative of
//!   `E_xc = ∫_cell ρ ε_xc` with respect to `D(k)` (no `1/N_k`: the
//!   k-point Fock convention of [`crate::kscf`]). The libxc kernel is
//!   ferric-dft's unchanged [`closed_kernel`].
//! * Hybrids: `F(k) = h + J − (a/2) K + V_xc`; the injected K carries the
//!   Madelung term and is scaled as a whole (see
//!   [`crate::kscf::solve_krks_injected`]).
//!
//! Open shell (KUKS), range-separated hybrids, meta-GGA, VV10 and double
//! hybrids are refused/absent, exactly as at Gamma. Time reversal is NOT used
//! to halve the XC work (every k gets its own Bloch AO cache).

use crate::budget::{bytes_of, Ledger};
use crate::dft::{dist3, PeriodicDftError};
use crate::dft::{
    image_translations, resolve_periodic_functional, shell_extent, PeriodicGrid,
    PeriodicGridConfig, PeriodicXcConfig, CHUNK, SORT_BOX,
};
use crate::hcore::kpoint::periodic_hcore_kpts;
use crate::kdense_aft::KDenseAftEri;
use crate::kpts::KPointMesh;
use crate::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointXc, KRhfConfig, KScfResult,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::KRsGdf;
use crate::timing::{PbcTimings, StageClock};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::DensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{FunctionalFamily, XcDef};
use ferric_dft::vxc::{closed_kernel, DENSITY_FLOOR};
use ferric_integrals::ao_grid::{collect_shells, eval_shell_and_grad};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rayon::prelude::*;

/// Bloch AO values on one spatial chunk, one entry per k.
pub(crate) struct KAoChunk {
    pub(crate) points: Vec<GridPoint>,
    /// `chi[k]`: `(nbf, npts)`.
    pub(crate) chi: Vec<Array2<Complex64>>,
    /// `dchi[k]`: `(3, nbf, npts)`; empty for LDA.
    pub(crate) dchi: Vec<Array3<Complex64>>,
}

/// Semilocal XC on a [`PeriodicGrid`] with Bloch AOs: the [`KPointXc`] of
/// [`solve_krks`].
pub struct KPeriodicXc {
    pub(crate) name: String,
    pub(crate) xc: XcDef,
    /// The same functional on `nspin = 2` libxc handles (spin-resolved k-point KS).
    pub(crate) xc_pol: XcDef,
    exx: f64,
    pub(crate) chunks: Vec<KAoChunk>,
    pub(crate) nbf: usize,
    pub(crate) nk: usize,
    pub(crate) gga: bool,
}

impl std::fmt::Debug for KPeriodicXc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KPeriodicXc")
            .field("functional", &self.name)
            .field("exact_exchange_fraction", &self.exx)
            .field("nbf", &self.nbf)
            .field("nk", &self.nk)
            .field("npoints", &self.npoints())
            .finish()
    }
}

fn build_k_chunks(
    cell: &Cell,
    bs: &BasisSet,
    grid: &PeriodicGrid,
    kpts: &[[f64; 3]],
    thresh: f64,
    gga: bool,
    ledger: &mut Ledger,
) -> Result<(Vec<KAoChunk>, usize), FerricError> {
    if !(thresh.is_finite() && thresh > 0.0 && thresh < 1.0) {
        return Err(PeriodicDftError::InvalidGrid {
            field: "ao_threshold",
            reason: format!("must be in (0, 1), got {thresh}"),
        }
        .into());
    }
    let shells = collect_shells(cell.mol(), bs)?;
    let mut offsets = Vec::with_capacity(shells.len());
    let mut nbf = 0usize;
    for sh in &shells {
        if sh.l > 4 {
            return Err(PeriodicDftError::Unsupported {
                feature: "basis angular momentum",
                reason: format!("l = {} (the AO evaluator supports s..g)", sh.l),
            }
            .into());
        }
        offsets.push(nbf);
        nbf += num_functions(sh.l, sh.pure);
    }
    let nk = kpts.len();
    let npts = grid.len();
    let per_pt = if gga { 4 } else { 1 };
    ledger.reserve(
        "k-point periodic XC Bloch AO cache",
        bytes_of(
            (npts as u64)
                .saturating_mul(nbf as u64)
                .saturating_mul(nk as u64),
            per_pt * std::mem::size_of::<Complex64>(),
        ),
    )?;
    let ext: Vec<f64> = shells.iter().map(|s| shell_extent(s, thresh)).collect();
    let r_ao = ext.iter().copied().fold(0.0, f64::max);

    let pts = grid.points();
    let mut order: Vec<usize> = (0..npts).collect();
    let key = |p: &[f64; 3]| {
        [
            (p[0] / SORT_BOX).floor() as i64,
            (p[1] / SORT_BOX).floor() as i64,
            (p[2] / SORT_BOX).floor() as i64,
        ]
    };
    order.sort_by_key(|&i| key(&pts[i].xyz));
    let trans = image_translations(cell, pts, r_ao, ledger)?;
    // e^{ik·L} per (translation, k).
    let phases: Vec<Vec<Complex64>> = trans
        .iter()
        .map(|l| {
            kpts.iter()
                .map(|k| {
                    let a = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
                    Complex64::new(a.cos(), a.sin())
                })
                .collect()
        })
        .collect();

    let chunk_idx: Vec<Vec<usize>> = order.chunks(CHUNK).map(|c| c.to_vec()).collect();
    let chunks: Result<Vec<KAoChunk>, FerricError> = chunk_idx
        .par_iter()
        .map(|idx| -> Result<KAoChunk, FerricError> {
            let cp: Vec<GridPoint> = idx.iter().map(|&i| pts[i]).collect();
            let m = cp.len() as f64;
            let cen = [
                cp.iter().map(|g| g.xyz[0]).sum::<f64>() / m,
                cp.iter().map(|g| g.xyz[1]).sum::<f64>() / m,
                cp.iter().map(|g| g.xyz[2]).sum::<f64>() / m,
            ];
            let rad = cp.iter().map(|g| dist3(&g.xyz, &cen)).fold(0.0, f64::max);
            // Live (shell, shifted centre, translation index).
            let mut live: Vec<(usize, [f64; 3], usize)> = Vec::new();
            for (il, l) in trans.iter().enumerate() {
                for (s, sh) in shells.iter().enumerate() {
                    let c = [
                        sh.center[0] + l[0],
                        sh.center[1] + l[1],
                        sh.center[2] + l[2],
                    ];
                    if dist3(&c, &cen) <= rad + ext[s] {
                        live.push((s, c, il));
                    }
                }
            }
            let np = cp.len();
            let mut chi = vec![Array2::<Complex64>::zeros((nbf, np)); nk];
            let mut dchi = if gga {
                vec![Array3::<Complex64>::zeros((3, nbf, np)); nk]
            } else {
                Vec::new()
            };
            let mut buf = [0.0_f64; 15];
            let mut gbuf = [[0.0_f64; 15]; 3];
            for (g, pt) in cp.iter().enumerate() {
                for &(s, c, il) in &live {
                    let (dx, dy, dz) = (pt.xyz[0] - c[0], pt.xyz[1] - c[1], pt.xyz[2] - c[2]);
                    if dx * dx + dy * dy + dz * dz > ext[s] * ext[s] {
                        continue;
                    }
                    let sh = &shells[s];
                    let n = num_functions(sh.l, sh.pure);
                    buf.fill(0.0);
                    for row in gbuf.iter_mut() {
                        row.fill(0.0);
                    }
                    eval_shell_and_grad(sh, dx, dy, dz, &mut buf[..n], &mut gbuf)?;
                    let o = offsets[s];
                    for (k, ph) in phases[il].iter().enumerate() {
                        for i in 0..n {
                            chi[k][(o + i, g)] += *ph * buf[i];
                            if gga {
                                for ax in 0..3 {
                                    dchi[k][(ax, o + i, g)] += *ph * gbuf[ax][i];
                                }
                            }
                        }
                    }
                }
            }
            Ok(KAoChunk {
                points: cp,
                chi,
                dchi,
            })
        })
        .collect();
    Ok((chunks?, nbf))
}

pub(crate) fn conj(m: &Array2<Complex64>) -> Array2<Complex64> {
    m.mapv(|z| z.conj())
}

impl KPeriodicXc {
    /// Resolve `functional` and evaluate the Bloch AO cache of `bs` (the
    /// basis the cell's `PreparedBasis` was built from) on `grid` for every
    /// point of `mesh`.
    pub fn new(
        cell: &Cell,
        bs: &BasisSet,
        functional: &str,
        grid: &PeriodicGrid,
        mesh: &KPointMesh,
        cfg: &PeriodicXcConfig,
    ) -> Result<Self, FerricError> {
        let (xc, exx) = resolve_periodic_functional(functional)?;
        let xc_pol = ferric_dft::libxc::xc_def_from_name_nspin(functional, 2).map_err(|e| {
            FerricError::General(format!(
                "periodic DFT: functional {functional:?} (spin-polarized): {e:?}"
            ))
        })?;
        let gga = xc
            .funcs
            .iter()
            .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        let (chunks, nbf) = build_k_chunks(
            cell,
            bs,
            grid,
            mesh.kpts(),
            cfg.ao_threshold,
            gga,
            &mut ledger,
        )?;
        Ok(Self {
            name: functional.to_string(),
            xc,
            xc_pol,
            exx,
            chunks,
            nbf,
            nk: mesh.nk(),
            gga,
        })
    }

    /// The functional's global exact-exchange fraction.
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.exx
    }

    /// Grid points cached.
    pub fn npoints(&self) -> usize {
        self.chunks.iter().map(|c| c.points.len()).sum()
    }

    /// AO count.
    pub fn nbasis(&self) -> usize {
        self.nbf
    }

    pub(crate) fn check(&self, dm: &[Array2<Complex64>]) -> Result<(), FerricError> {
        if dm.len() != self.nk || dm.iter().any(|d| d.dim() != (self.nbf, self.nbf)) {
            return Err(FerricError::General(format!(
                "KPeriodicXc: expected {} density matrices of shape ({n}, {n})",
                self.nk,
                n = self.nbf
            )));
        }
        Ok(())
    }

    /// `ρ`, `∇ρ`, `σ` of one chunk from the k densities (module doc).
    fn chunk_density(&self, c: &KAoChunk, dm: &[Array2<Complex64>]) -> DensityGrid {
        let np = c.points.len();
        let inv_nk = 1.0 / self.nk as f64;
        let mut rho = Array1::<f64>::zeros(np);
        let mut grad = Array2::<f64>::zeros((3, np));
        for k in 0..self.nk {
            // P_{μg} = Σ_ν D_{μν} φ*_{νg}
            let p = dm[k].dot(&conj(&c.chi[k]));
            for g in 0..np {
                let mut r = Complex64::new(0.0, 0.0);
                for mu in 0..self.nbf {
                    r += c.chi[k][(mu, g)] * p[(mu, g)];
                }
                rho[g] += inv_nk * r.re;
                if self.gga {
                    for ax in 0..3 {
                        let mut gr = Complex64::new(0.0, 0.0);
                        for mu in 0..self.nbf {
                            gr += c.dchi[k][(ax, mu, g)] * p[(mu, g)];
                        }
                        grad[(ax, g)] += inv_nk * 2.0 * gr.re;
                    }
                }
            }
        }
        let sigma = (0..np)
            .map(|g| grad[(0, g)].powi(2) + grad[(1, g)].powi(2) + grad[(2, g)].powi(2))
            .collect();
        DensityGrid { rho, grad, sigma }
    }

    /// `∫_cell ρ` of the k densities (`= N_e` up to the grid error).
    pub fn integrate_density(&self, dm: &[Array2<Complex64>]) -> Result<f64, FerricError> {
        self.check(dm)?;
        Ok(self
            .chunks
            .iter()
            .map(|c| {
                let d = self.chunk_density(c, dm);
                c.points
                    .iter()
                    .zip(d.rho.iter())
                    .map(|(g, r)| g.weight * r)
                    .sum::<f64>()
            })
            .sum())
    }

    /// `S^grid(k)_{μν} = Σ_g w_g φ^k_μ* φ^k_ν` (≈ the lattice `S(k)`), for
    /// the Bloch-convention check.
    pub fn overlap_on_grid(&self, k: usize) -> Array2<Complex64> {
        let mut s = Array2::<Complex64>::zeros((self.nbf, self.nbf));
        for c in &self.chunks {
            let mut wchi = conj(&c.chi[k]);
            for (g, p) in c.points.iter().enumerate() {
                wchi.column_mut(g).mapv_inplace(|v| v * p.weight);
            }
            s += &wchi.dot(&c.chi[k].t());
        }
        s
    }
}

impl KPointXc for KPeriodicXc {
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), FerricError> {
        self.check(dm)?;
        let n = self.nbf;
        let mut e = 0.0;
        let mut v: Vec<Array2<Complex64>> = (0..self.nk)
            .map(|_| Array2::<Complex64>::zeros((n, n)))
            .collect();
        for c in &self.chunks {
            let dens = self.chunk_density(c, dm);
            let kern = closed_kernel(&dens, None, &self.xc);
            let np = c.points.len();
            for g in 0..np {
                e += c.points[g].weight * dens.rho[g] * kern.exc[g];
            }
            let a: Vec<f64> = (0..np)
                .map(|g| {
                    if dens.rho[g] > DENSITY_FLOOR {
                        c.points[g].weight * kern.vrho[g]
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
                                if dens.rho[g] > DENSITY_FLOOR {
                                    2.0 * c.points[g].weight * kern.vsigma[g] * dens.grad[(ax, g)]
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
            for k in 0..self.nk {
                let scale = |fac: &[f64]| {
                    let mut m = conj(&c.chi[k]);
                    for g in 0..np {
                        m.column_mut(g).mapv_inplace(|z| z * fac[g]);
                    }
                    m
                };
                v[k] += &scale(&a).dot(&c.chi[k].t());
                if self.gga {
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
        if !e.is_finite() || v.iter().any(|m| m.iter().any(|z| !z.re.is_finite())) {
            return Err(FerricError::General(format!(
                "KPeriodicXc: non-finite E_xc/V_xc ({e}) for {}",
                self.name
            )));
        }
        // Hermitise (the kernel is Hermitian up to rounding).
        for m in v.iter_mut() {
            let h = conj(m).t().to_owned();
            *m = (&*m + &h).mapv(|z| z * 0.5);
        }
        Ok((e, v))
    }
}

/// Settings of [`solve_krks`].
#[derive(Debug, Clone)]
pub struct KRksConfig {
    /// SCF, J/K source, exxdiv and one-electron settings.
    pub krhf: KRhfConfig,
    /// LDA / GGA / global-hybrid name (as [`crate::dft::GammaRksConfig`]).
    pub functional: String,
    /// The periodic grid.
    pub grid: PeriodicGridConfig,
    /// AO truncation / budget of the Bloch AO cache.
    pub xc: PeriodicXcConfig,
}

impl KRksConfig {
    /// Defaults (dense J/K, grid (75, 302) SSF) for `functional`.
    pub fn new(cell: &Cell, exxdiv: crate::dense_aft::ExxDiv, functional: &str) -> Self {
        Self {
            krhf: KRhfConfig::for_cell(cell, exxdiv),
            functional: functional.to_string(),
            grid: PeriodicGridConfig::default(),
            xc: PeriodicXcConfig::default(),
        }
    }
}

/// Result of [`solve_krks`].
#[derive(Debug, Clone)]
#[must_use]
pub struct KRksResult {
    /// The converged SCF (energy per cell includes `E_xc` and `E_nn`).
    pub scf: KScfResult,
    /// `E_xc` at the converged densities.
    pub e_xc: f64,
    /// The functional's global exact-exchange fraction.
    pub exact_exchange_fraction: f64,
    /// Grid points used.
    pub n_grid_points: usize,
    /// `∫_cell ρ` at the converged densities (vs `N_e`: the grid error).
    pub electrons_on_grid: f64,
}

/// k-point closed-shell KS-DFT on the atom-centred periodic grid built from
/// `cfg.grid`. See [`solve_krks_on_grid`] to supply the grid.
pub fn solve_krks(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRksConfig,
) -> Result<KRksResult, FerricError> {
    resolve_periodic_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    solve_krks_on_grid(cell, prep, aux, mesh, &grid, cfg)
}

/// [`solve_krks`] on a caller-supplied grid (e.g. [`PeriodicGrid::uniform`],
/// the independent-reference construction). `cfg.grid` is ignored.
pub fn solve_krks_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KRksConfig,
) -> Result<KRksResult, FerricError> {
    let mol = cell.mol();
    if mol.nelec() % 2 != 0 || mol.multiplicity != 1 {
        return Err(PeriodicDftError::Unsupported {
            feature: "open shell",
            reason: "solve_krks is closed-shell RKS only (no k-point UKS yet)".to_string(),
        }
        .into());
    }
    let kc = &cfg.krhf;
    match (kc.jk, aux) {
        (KJkKind::Dense, Some(_)) => {
            return Err(FerricError::General(
                "solve_krks: an aux basis was given but jk = dense does not use it".into(),
            ))
        }
        (KJkKind::RsGdf, None) => {
            return Err(FerricError::General(
                "solve_krks: jk = rsgdf requires an auxbasis (none given)".into(),
            ))
        }
        _ => {}
    }
    let mut pxc = KPeriodicXc::new(cell, prep.basis_set(), &cfg.functional, grid, mesh, &cfg.xc)?;
    let a = pxc.exact_exchange_fraction();
    let hk = periodic_hcore_kpts(cell, prep, mesh, &kc.hcore)?;
    let mut timings = PbcTimings::default();
    let gdf;
    let eri;
    let jk: Box<dyn crate::kscf::KPointJk + '_>;
    match aux {
        None => {
            eri = KDenseAftEri::build(cell, prep, mesh, &hk.s, kc.exxdiv, &kc.dense)?;
            jk = Box::new(eri.jk_builder());
        }
        Some(aux) => {
            gdf = KRsGdf::build(cell, prep, aux, mesh, &hk.s, &kc.rsgdf)?.with_exxdiv(kc.exxdiv);
            crate::rsgdf::kpoint::record_stats(&mut timings, gdf.stats());
            jk = Box::new(gdf.jk_builder());
        }
    }
    let inj = KPointInjection {
        s: hk.s,
        h: hk.h,
        vnn: hk.enn,
        jk,
    };
    let clock = StageClock::start();
    let mut scf = solve_krks_injected(cell, mesh, &kc.scf, inj, &mut pxc, a)?;
    timings.stop("k SCF", &clock);
    scf.timings = timings;
    if !scf.converged {
        return Err(FerricError::General(format!(
            "solve_krks: {} SCF did not converge in {} iterations (last E = {})",
            cfg.functional, scf.iterations, scf.energy
        )));
    }
    let electrons_on_grid = pxc.integrate_density(&scf.densities)?;
    let (e_xc, _) = KPointXc::build(&mut pxc, &scf.densities)?;
    Ok(KRksResult {
        e_xc,
        exact_exchange_fraction: a,
        n_grid_points: grid.len(),
        electrons_on_grid,
        scf,
    })
}
