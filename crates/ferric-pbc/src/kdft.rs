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
//! Meta-GGA (SCAN / r2SCAN / TPSS): `τ = (1/N_k) Σ_k ½ Σ_ax Re Σ D(k) ∂φ^k_μ (∂φ^k_ν)*`
//! and `V^τ(k)_{μν} = Σ_g ½ w v_τ Σ_ax (∂_ax φ^k_μ)* ∂_ax φ^k_ν`
//! (the molecular `semilocal_vxc` is mirrored). The energy is validated with
//! the dense-AFT and RS-GDF J/K; the closed-shell forces and stress cover
//! SCAN / r2SCAN / TPSS ([`crate::kxcgrad::is_validated_mgga_deriv`]), the
//! open-shell ones SCAN / r2SCAN ([`crate::kxcgrad::is_validated_mgga`]).
//!
//! Range-separated hybrids (HSE06, CAM-B3LYP, wB97X, ...) are supported with
//! the dense-AFT J/K ([`crate::rsh`]) and the RS-GDF J/K (`KRsGdf::build_rsh`:
//! full-Coulomb fit plus a G-space fit of the erf-attenuated kernel). The
//! closed-shell nuclear force and stress cover both J/K
//! ([`crate::kxcgrad`], [`crate::kxcstress`]), the open-shell UKS and ROKS
//! forces and stresses too ([`crate::kuxcgrad`], [`crate::kuxcstress`]). The Gamma-point [`crate::gamma_rsh`]
//! reuses this exchange at k = 0, and the open-shell [`crate::kuks`] (KUKS)
//! and [`crate::kroks`] (KROKS) take it per spin (with SCAN / r2SCAN
//! open-shell energies in KUKS too, spin-resolved tau). Range-separated meta-GGAs, VV10 and double hybrids are
//! refused. Time reversal is NOT used
//! to halve the XC work (every k gets its own Bloch AO cache).
//!
//! # Parallelism and reproducibility
//!
//! The build runs the grid chunks (`CHUNK` points each, fixed by the grid
//! alone) in parallel. Each chunk's `E_xc` and `V_xc(k)` are summed from zero
//! and the chunk results are added in ascending chunk order
//! (`KPeriodicXc::fold_chunks_in_order`), so the result is bit-identical at
//! any thread count (`tests/pbc_kxc_parallel.rs`). The per-point density
//! contractions sweep the AO rows contiguously with the per-point sum order
//! unchanged. Per-phase timings: [`KPeriodicXc::xc_timings`] and the KS
//! drivers' `timings` (`examples/kxc_build_bench.rs`).

use crate::budget::{bytes_of, Ledger};
use crate::dft::{dist3, PeriodicDftError};
use crate::dft::{
    image_translations, shell_extent, PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig, CHUNK,
    SORT_BOX,
};
use crate::hcore::kpoint::periodic_hcore_kpts;
use crate::kdense_aft::KDenseAftEri;
use crate::kpts::KPointMesh;
use crate::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointXc, KRetained, KRetainedJk, KRhfConfig,
    KScfResult,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::KRsGdf;
use crate::rsh::{resolve_periodic_functional_rsh, RshParams};
use crate::timing::{CallClock, PbcTimings, PhaseClock, StageClock, StageTiming};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::DensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{FunctionalFamily, XcDef};
use ferric_dft::vxc::{closed_kernel, DENSITY_FLOOR};
use ferric_integrals::ao_grid::{collect_shells, eval_shell_and_grad};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
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
    exx: f64,
    pub(crate) chunks: Vec<KAoChunk>,
    pub(crate) nbf: usize,
    pub(crate) nk: usize,
    pub(crate) gga: bool,
    rsh: Option<RshParams>,
    /// Meta-GGA: tau (and its Vxc term) is evaluated from `dchi`.
    pub(crate) mgga: bool,
    /// Wall/CPU of the Bloch AO cache build ([`KPeriodicXc::new`]).
    pub(crate) ao_cache: StageTiming,
    /// Per-phase clocks of every XC build ([`KPeriodicXc::xc_timings`]).
    pub(crate) clocks: KXcClocks,
    /// Test hook ([`KPeriodicXc::set_serial_reduction`]).
    serial_reduction: bool,
}

/// Accumulated per-phase time of the k-point XC builds (closed and open
/// shell), for [`KPeriodicXc::xc_timings`]. Observation only: the clocks never
/// touch a number. The phase clocks are summed over every timed call, so
/// when phases run on several threads their wall time is THREAD-SUMMED:
/// compare phases by wall share only. They run per chunk, so they are
/// wall-only ([`PhaseClock`]); the process CPU clock is read only around a
/// whole build.
#[derive(Debug, Default, Clone)]
pub(crate) struct KXcClocks {
    /// One whole `build` / `build_polarized` call.
    pub(crate) build: CallClock,
    /// `P(k) = D(k) φ^k*` (complex GEMM, per spin and k).
    pub(crate) density_gemm: PhaseClock,
    /// `ρ`, `∇ρ` from `P` and `φ` (the per-point contraction loops).
    pub(crate) density_contract: PhaseClock,
    /// `τ` (meta-GGA): its own GEMMs and contraction.
    pub(crate) tau: PhaseClock,
    /// The libxc kernel.
    pub(crate) kernel: PhaseClock,
    /// The weighted conjugate AO copies (and `m^H`) of the potential build.
    pub(crate) vxc_scale: PhaseClock,
    /// The potential's complex GEMMs.
    pub(crate) vxc_gemm: PhaseClock,
    /// The potential's accumulation into `V(k)`.
    pub(crate) vxc_add: PhaseClock,
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

/// Most bytes of per-chunk results [`KPeriodicXc::fold_chunks_in_order`]
/// holds at once (it never limits the window below one chunk).
const FOLD_WINDOW_BYTES: usize = 256 << 20;

/// Upper bound on what [`KPeriodicXc::fold_chunks_in_order`] holds at once,
/// independent of the thread count: the window cap (or one chunk's result
/// if larger), but never more than every chunk's. One result is the
/// open-shell `(V_α(k), V_β(k))`, which also bounds the closed shell's.
fn fold_window_bound(nk: usize, nbf: usize, nchunks: usize) -> usize {
    let one = bytes_of(
        (2 * nk as u64)
            .saturating_mul(nbf as u64)
            .saturating_mul(nbf as u64),
        std::mem::size_of::<Complex64>(),
    );
    one.saturating_mul(nchunks).min(FOLD_WINDOW_BYTES.max(one))
}

/// `out[g] += s · Re Σ_μ a[μ,g] b[μ,g]`. The `μ` sum of each point runs
/// ascending from zero, the order of the former per-point loop, so the bits
/// are unchanged; the sweep goes row by row so both operands stream
/// contiguously, and only the real part is formed. `acc` is scratch of
/// length `out.len()`.
pub(crate) fn add_re_column_dots(
    out: &mut [f64],
    s: f64,
    a: ArrayView2<'_, Complex64>,
    b: ArrayView2<'_, Complex64>,
    acc: &mut [f64],
) {
    acc.fill(0.0);
    for (ra, rb) in a.rows().into_iter().zip(b.rows()) {
        for ((t, x), y) in acc.iter_mut().zip(ra.iter()).zip(rb.iter()) {
            *t += x.re * y.re - x.im * y.im;
        }
    }
    for (o, t) in out.iter_mut().zip(acc.iter()) {
        *o += s * t;
    }
}

/// `conj(x) · diag(fac)` (column `g` scaled by `fac[g]`), built row by row.
pub(crate) fn weighted_conj(x: ArrayView2<'_, Complex64>, fac: &[f64]) -> Array2<Complex64> {
    let mut m = Array2::<Complex64>::zeros(x.raw_dim());
    for (mut rm, rx) in m.rows_mut().into_iter().zip(x.rows()) {
        for ((o, z), f) in rm.iter_mut().zip(rx.iter()).zip(fac) {
            *o = z.conj() * *f;
        }
    }
    m
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
        let (xc, exx, rsh) = resolve_periodic_functional_rsh(functional)?;
        let mgga = xc
            .funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
        let gga = xc
            .funcs
            .iter()
            .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        let clock = StageClock::start();
        let (chunks, nbf) = build_k_chunks(
            cell,
            bs,
            grid,
            mesh.kpts(),
            cfg.ao_threshold,
            gga,
            &mut ledger,
        )?;
        ledger.reserve(
            "k-point periodic XC chunk fold window",
            fold_window_bound(mesh.nk(), nbf, chunks.len()),
        )?;
        let (wall_s, cpu_s) = clock.elapsed();
        Ok(Self {
            ao_cache: StageTiming {
                name: "k XC AO cache",
                wall_s,
                cpu_s,
                calls: 1,
            },
            clocks: KXcClocks::default(),
            serial_reduction: false,
            name: functional.to_string(),
            xc,
            exx,
            chunks,
            nbf,
            nk: mesh.nk(),
            gga,
            rsh,
            mgga,
        })
    }

    /// The functional's global exact-exchange fraction (`0` for a
    /// range-separated hybrid: see [`Self::rsh`]).
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.exx
    }

    /// Range-separation parameters if the functional is a CAM hybrid.
    pub fn rsh(&self) -> Option<RshParams> {
        self.rsh
    }

    /// Grid points cached.
    pub fn npoints(&self) -> usize {
        self.chunks.iter().map(|c| c.points.len()).sum()
    }

    /// AO count.
    pub fn nbasis(&self) -> usize {
        self.nbf
    }

    /// Stage `"k XC AO cache"` (the one-off Bloch AO evaluation) and stage
    /// `"k XC build"` (every `build` / `build_polarized` call so far), with
    /// the build's phases as sub-stages `"k XC build/<phase>"` (density GEMM,
    /// density contraction, tau, libxc kernel, Vxc scale, Vxc GEMM, Vxc add).
    /// Phase wall times are summed over threads when the chunks run in parallel.
    pub fn xc_timings(&self) -> PbcTimings {
        let mut t = PbcTimings::default();
        t.add_stage(&self.ao_cache);
        t.add_stage(&self.clocks.build.timing("k XC build"));
        let c = &self.clocks;
        for (clock, name) in [
            (&c.density_gemm, "k XC build/density GEMM"),
            (&c.density_contract, "k XC build/density contraction"),
            (&c.tau, "k XC build/tau"),
            (&c.kernel, "k XC build/libxc kernel"),
            (&c.vxc_scale, "k XC build/Vxc scale"),
            (&c.vxc_gemm, "k XC build/Vxc GEMM"),
            (&c.vxc_add, "k XC build/Vxc add"),
        ] {
            if clock.calls() > 0 {
                let st = clock.timing(name);
                t.substages.push(st);
            }
        }
        t.set_counter("k XC grid points", self.npoints() as u64);
        t.set_counter("k XC chunks", self.chunks.len() as u64);
        t
    }

    /// Add the XC cost to a KS driver's record: the AO cache as the leaf stage
    /// `"k XC AO cache"` (built before the SCF), and every build so far as the
    /// sub-stage `"k XC build"` (inside `"k SCF"`) with its phases.
    pub(crate) fn record_scf_timings(&self, t: &mut PbcTimings) {
        let x = self.xc_timings();
        t.add_stage(&self.ao_cache);
        if let Some(b) = x.stage("k XC build") {
            t.substages.push(b.clone());
        }
        t.substages.extend(x.substages);
        for &(n, v) in &x.counters {
            t.set_counter(n, v);
        }
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
    fn chunk_density(
        &self,
        c: &KAoChunk,
        dm: &[Array2<Complex64>],
        clk: &KXcClocks,
    ) -> DensityGrid {
        let np = c.points.len();
        let inv_nk = 1.0 / self.nk as f64;
        let mut rho = Array1::<f64>::zeros(np);
        let mut grad = Array2::<f64>::zeros((3, np));
        let mut acc = vec![0.0; np];
        for k in 0..self.nk {
            // P_{μg} = Σ_ν D_{μν} φ*_{νg}
            let p = clk.density_gemm.time(|| dm[k].dot(&conj(&c.chi[k])));
            let t0 = std::time::Instant::now();
            let rho_s = rho.as_slice_mut().expect("standard layout");
            add_re_column_dots(rho_s, inv_nk, c.chi[k].view(), p.view(), &mut acc);
            if self.gga {
                for ax in 0..3 {
                    let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                    let mut row = grad.row_mut(ax);
                    let out = row.as_slice_mut().expect("standard layout");
                    add_re_column_dots(out, inv_nk * 2.0, d_ax, p.view(), &mut acc);
                }
            }
            clk.density_contract.record(t0);
        }
        let sigma = (0..np)
            .map(|g| grad[(0, g)].powi(2) + grad[(1, g)].powi(2) + grad[(2, g)].powi(2))
            .collect();
        DensityGrid { rho, grad, sigma }
    }

    /// Total kinetic-energy density of one chunk,
    /// `τ = (1/N_k) Σ_k ½ Σ_ax Re Σ_{μν} D(k)_{μν} ∂_ax φ^k_μ (∂_ax φ^k_ν)*`.
    pub(crate) fn chunk_tau(&self, c: &KAoChunk, dm: &[Array2<Complex64>]) -> Array1<f64> {
        self.clocks.tau.time(|| self.chunk_tau_untimed(c, dm))
    }

    fn chunk_tau_untimed(&self, c: &KAoChunk, dm: &[Array2<Complex64>]) -> Array1<f64> {
        let np = c.points.len();
        let w = 0.5 / self.nk as f64;
        let mut tau = Array1::<f64>::zeros(np);
        let mut acc = vec![0.0; np];
        for k in 0..self.nk {
            for ax in 0..3 {
                let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                let p = dm[k].dot(&conj(&d_ax.to_owned()));
                let out = tau.as_slice_mut().expect("standard layout");
                add_re_column_dots(out, w, d_ax, p.view(), &mut acc);
            }
        }
        tau
    }

    /// `∫_cell ρ` of the k densities (`= N_e` up to the grid error).
    pub fn integrate_density(&self, dm: &[Array2<Complex64>]) -> Result<f64, FerricError> {
        self.check(dm)?;
        Ok(self
            .chunks
            .iter()
            .map(|c| {
                let d = self.chunk_density(c, dm, &KXcClocks::default());
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
        let clock = StageClock::start();
        let out = self.build_untimed(dm);
        self.clocks.build.record(&clock);
        out
    }
}

impl KPeriodicXc {
    /// `V += (conj(x) · diag(fac)) · yᵀ` (plus its adjoint if `with_adjoint`):
    /// one potential term of one chunk, timed per phase.
    pub(crate) fn add_weighted_product(
        &self,
        v: &mut Array2<Complex64>,
        x: ArrayView2<'_, Complex64>,
        y: ArrayView2<'_, Complex64>,
        fac: &[f64],
        with_adjoint: bool,
    ) {
        let c = &self.clocks;
        let m = c.vxc_scale.time(|| weighted_conj(x, fac));
        let m = c.vxc_gemm.time(|| m.dot(&y.t()));
        if with_adjoint {
            let mh = c.vxc_scale.time(|| conj(&m).t().to_owned());
            c.vxc_add.time(|| {
                *v += &m;
                *v += &mh;
            });
        } else {
            c.vxc_add.time(|| *v += &m);
        }
    }

    /// Run `f` on every chunk (in parallel, at most `window` chunks in flight,
    /// unless the serial reduction is set) and hand the results to `fold` in
    /// ASCENDING chunk order. The reduction order is therefore fixed by the
    /// chunking alone: the result is bit-identical at any thread count.
    /// `out_bytes` (one result's size) caps the window at [`FOLD_WINDOW_BYTES`].
    pub(crate) fn fold_chunks_in_order<T: Send>(
        &self,
        out_bytes: usize,
        f: impl Fn(&KAoChunk) -> T + Sync,
        mut fold: impl FnMut(T),
    ) {
        let window = if self.serial_reduction {
            1
        } else {
            let by_bytes = (FOLD_WINDOW_BYTES / out_bytes.max(1)).max(1);
            rayon::current_num_threads().clamp(1, by_bytes)
        };
        for w in self.chunks.chunks(window) {
            let outs: Vec<T> = if self.serial_reduction {
                w.iter().map(&f).collect()
            } else {
                w.par_iter().map(&f).collect()
            };
            outs.into_iter().for_each(&mut fold);
        }
    }

    /// Test hook: evaluate the chunks one by one on the calling thread (no
    /// rayon), the serial reference of the ordered reduction.
    #[doc(hidden)]
    pub fn set_serial_reduction(&mut self, serial: bool) {
        self.serial_reduction = serial;
    }

    /// One chunk's `(E_xc, V_xc(k))`, each summed from zero.
    fn chunk_build(&self, c: &KAoChunk, dm: &[Array2<Complex64>]) -> (f64, Vec<Array2<Complex64>>) {
        let n = self.nbf;
        let dens = self.chunk_density(c, dm, &self.clocks);
        let tau = self.mgga.then(|| self.chunk_tau(c, dm));
        let kern = self
            .clocks
            .kernel
            .time(|| closed_kernel(&dens, tau.as_ref(), &self.xc));
        let np = c.points.len();
        let mut e = 0.0;
        for g in 0..np {
            e += c.points[g].weight * dens.rho[g] * kern.exc[g];
        }
        let gated = |g: usize, x: f64| if dens.rho[g] > DENSITY_FLOOR { x } else { 0.0 };
        let a: Vec<f64> = (0..np)
            .map(|g| gated(g, c.points[g].weight * kern.vrho[g]))
            .collect();
        let f_ax: Vec<Vec<f64>> = if self.gga {
            (0..3)
                .map(|ax| {
                    (0..np)
                        .map(|g| {
                            let w = c.points[g].weight;
                            gated(g, 2.0 * w * kern.vsigma[g] * dens.grad[(ax, g)])
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        let f_tau: Vec<f64> = if self.mgga {
            (0..np)
                .map(|g| gated(g, 0.5 * c.points[g].weight * kern.vtau[g]))
                .collect()
        } else {
            Vec::new()
        };
        let mut v: Vec<Array2<Complex64>> = (0..self.nk).map(|_| Array2::zeros((n, n))).collect();
        for (k, vk) in v.iter_mut().enumerate() {
            let chi = c.chi[k].view();
            self.add_weighted_product(vk, chi, chi, &a, false);
            if self.gga {
                for (ax, f) in f_ax.iter().enumerate() {
                    let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                    self.add_weighted_product(vk, chi, d_ax, f, true);
                }
            }
            if self.mgga {
                // V^tau(k)_{μν} = Σ_g ½ w v_tau Σ_ax (∂φ_μ)* ∂φ_ν
                for ax in 0..3 {
                    let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                    self.add_weighted_product(vk, d_ax, d_ax, &f_tau, false);
                }
            }
        }
        (e, v)
    }

    fn build_untimed(
        &self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), FerricError> {
        self.check(dm)?;
        let n = self.nbf;
        let mut e = 0.0;
        let mut v: Vec<Array2<Complex64>> = (0..self.nk)
            .map(|_| Array2::<Complex64>::zeros((n, n)))
            .collect();
        let out_bytes = bytes_of((self.nk * n * n) as u64, std::mem::size_of::<Complex64>());
        self.fold_chunks_in_order(
            out_bytes,
            |c| self.chunk_build(c, dm),
            |(ec, vc)| {
                e += ec;
                for (acc, x) in v.iter_mut().zip(&vc) {
                    *acc += x;
                }
            },
        );
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
    /// The functional's global exact-exchange fraction (the short-range
    /// fraction `c_sr` for a range-separated hybrid).
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
    resolve_periodic_functional_rsh(&cfg.functional)?;
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
    Ok(solve_krks_keep_on_grid(cell, prep, aux, mesh, grid, cfg)?.0)
}

/// [`solve_krks`] that also returns the hcore output and J/K object the SCF
/// ran on ([`KRetained`]), for [`crate::kxcgrad::kpoint_rks_gradient`]. The SCF
/// is the same code path as [`solve_krks`] (which delegates here), so the
/// energy is bit-identical.
pub fn solve_krks_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KRksConfig,
) -> Result<(KRksResult, KRetained), FerricError> {
    resolve_periodic_functional_rsh(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    solve_krks_keep_on_grid(cell, prep, aux, mesh, &grid, cfg)
}

/// [`solve_krks_keep`] on a caller-supplied grid.
pub fn solve_krks_keep_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KRksConfig,
) -> Result<(KRksResult, KRetained), FerricError> {
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
    let total = StageClock::start();
    let mut pxc = KPeriodicXc::new(cell, prep.basis_set(), &cfg.functional, grid, mesh, &cfg.xc)?;
    let mut a = pxc.exact_exchange_fraction();
    // Range-separated hybrid: the fractions live inside the K kernel (dense
    // AFT: the attenuated kernel weights; RS-GDF: the full-Coulomb plus the
    // erf-attenuated fit, `rsgdf::kpoint::klr`), so the SCF scales K by 1.
    let mut kc = *kc;
    if let Some(r) = pxc.rsh() {
        if kc.jk == KJkKind::Dense {
            kc.dense.rsh = Some(r);
        }
        a = 1.0;
    }
    let kc = &kc;
    let hk = periodic_hcore_kpts(cell, prep, mesh, &kc.hcore)?;
    let mut timings = PbcTimings::default();
    hk.record_stats(&mut timings);
    let retained_jk = match aux {
        None => KRetainedJk::Dense(KDenseAftEri::build(
            cell, prep, mesh, &hk.s, kc.exxdiv, &kc.dense,
        )?),
        Some(aux) => {
            let gdf = match pxc.rsh() {
                Some(r) => KRsGdf::build_rsh(cell, prep, aux, mesh, &hk.s, &kc.rsgdf, r)?,
                None => KRsGdf::build(cell, prep, aux, mesh, &hk.s, &kc.rsgdf)?,
            }
            .with_exxdiv(kc.exxdiv);
            crate::rsgdf::kpoint::record_stats(&mut timings, gdf.stats());
            KRetainedJk::RsGdf(gdf)
        }
    };
    let jk: Box<dyn crate::kscf::KPointJk + '_> = match &retained_jk {
        KRetainedJk::Dense(e) => Box::new(e.jk_builder()),
        KRetainedJk::RsGdf(g) => Box::new(g.jk_builder()),
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    };
    let clock = StageClock::start();
    let mut scf = solve_krks_injected(cell, mesh, &kc.scf, inj, &mut pxc, a)?;
    timings.stop("k SCF", &clock);
    pxc.record_scf_timings(&mut timings);
    timings.finish(&total);
    scf.timings = timings;
    if !scf.converged {
        return Err(FerricError::Convergence(format!(
            "solve_krks: {} SCF did not converge in {} iterations (last E = {})",
            cfg.functional, scf.iterations, scf.energy
        )));
    }
    let electrons_on_grid = pxc.integrate_density(&scf.densities)?;
    let (e_xc, _) = KPointXc::build(&mut pxc, &scf.densities)?;
    let result = KRksResult {
        e_xc,
        exact_exchange_fraction: pxc.rsh().map_or(a, |r| r.c_sr),
        n_grid_points: grid.len(),
        electrons_on_grid,
        scf,
    };
    Ok((
        result,
        KRetained {
            hk,
            jk: retained_jk,
        },
    ))
}
