//! k-point open-shell Kohn-Sham DFT (`solve_kuks`: LDA, GGA, global hybrids,
//! and the pure SCAN / r2SCAN meta-GGAs, energy only):
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
//! `a·v_M`.
//!
//! Meta-GGA (issue #355): ONLY the pure SCAN / r2SCAN, on the dense-AFT or
//! the RS-GDF J/K (any other meta-GGA and a meta-GGA hybrid are refused by
//! name; a pure meta-GGA has no exact exchange, so the J/K route enters only
//! through `J`, and the tau path is the same object on both routes,
//! `tests/pbc_kuks_mgga_rsgdf.rs`, issue #492). The
//! per-spin kinetic-energy density is
//! `τ_σ = (1/N_k) Σ_k ½ Σ_ax Re Σ D_σ(k)_{μν} ∂φ_μ (∂φ_ν)*` (libxc's `½`
//! convention, `KPeriodicXc::chunk_tau` applied to `D_σ`), fed to the
//! polarized libxc kernel as `(τ_α, τ_β)`, and each spin gets its own
//! `V^τ_σ(k) = Σ_g ½ w v_τσ Σ_ax (∂φ_μ)* ∂φ_ν`. The library force and stress
//! are [`crate::kuxcgrad::kpoint_uks_gradient`] and
//! [`crate::kuxcstress::kpoint_uks_stress`] (dense-AFT or RS-GDF J/K). The
//! Python / CLI derivative gate ([`crate::kxcgrad::check_deriv_functional`])
//! accepts them on the dense-AFT J/K only (the RS-GDF route is library-only
//! until the bindings are wired).
//!
//! Range-separated hybrids (HSE06, CAM-B3LYP validated against PySCF, wB97X unpinned; mgoldey/ferric#354) take
//! the closed-shell attenuated K unchanged ([`crate::rsh`], dense AFT or
//! `KRsGdf::build_rsh`): the K builder is linear in the density and the
//! fractions and the MIXED Madelung constant are inside it, so the open-shell
//! step is `K_σ = K[D_σ]` per spin scaled by 1 (`a = 1`; the reported fraction
//! is `c_sr`). The nuclear force and stress on the dense-AFT and RS-GDF J/K
//! are [`crate::kuxcgrad::kpoint_uks_gradient`] and
//! [`crate::kuxcstress::kpoint_uks_stress`] (at Gamma
//! [`crate::gamma_rsh::gamma_uks_rsh_gradient`] /
//! [`crate::gamma_rsh::gamma_uks_rsh_stress`]), wired to Python / CLI on
//! either J/K for HSE06 / CAM-B3LYP. Range-separated
//! meta-GGAs, VV10 and double hybrids are refused as at Gamma.

use crate::budget::bytes_of;
use crate::dense_aft::ExxDiv;
use crate::dft::{
    resolve_periodic_functional, resolve_periodic_functional_mgga, PeriodicDftError, PeriodicGrid,
    PeriodicGridConfig, PeriodicXcConfig,
};
use crate::hcore::kpoint::periodic_hcore_kpts;
use crate::kdense_aft::KDenseAftEri;
use crate::kdft::{add_re_column_dots, conj, KAoChunk, KPeriodicXc, KXcClocks};
use crate::kpts::KPointMesh;
use crate::kscf::{KJkKind, KPointInjection, KPointJk, KRetained, KRetainedJk};
use crate::kuscf::{
    kuhf_gap_report, run_kuhf, KPointXcPolarized, KUScfResult, KUhfConfig, KUhfMutation,
};
use crate::lattice::Cell;
use crate::rsgdf::kpoint::KRsGdf;
use crate::timing::{PbcTimings, StageClock, StageTiming};
use crate::uhf::{nocc_ab, EwaldStart, SpinGapReport};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::libxc::{xc_def_from_name_nspin, XcDef};
use ferric_dft::vxc::{polarized_kernel, PolarizedKernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;

/// Spin-polarized semilocal XC on the Bloch AO cache of a [`KPeriodicXc`].
pub struct KPeriodicXcPolarized {
    inner: KPeriodicXc,
    /// The functional on `nspin = 2` libxc handles.
    xc_pol: XcDef,
    mutation: Option<KUksMggaMutation>,
}

/// Deliberate spin-resolved tau defects, for the mutation tests of
/// `tests/pbc_kuks_mgga.rs`. Each must be caught by a genuinely POLARIZED pin.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KUksMggaMutation {
    /// `(τ_α, τ_β)` handed to libxc swapped.
    SwapTau,
    /// `V^τ_β` dropped (the β Fock gets no tau term).
    DropBetaVtau,
    /// The TOTAL `τ_α + τ_β` handed to libxc for both channels.
    TotalTauBothChannels,
    /// The spin-summed evaluator: `ρ/2`, `∇ρ/2`, `τ/2` in both channels.
    UnpolarizedEvaluator,
    /// `τ_β` built from `D_α` (both channels get `τ_α`).
    TauBetaFromAlpha,
    /// Zero `τ` handed to libxc in both channels (the tau term dropped from
    /// the kernel input; `V^τ` still assembled from what libxc returns).
    DropTau,
}

/// The functional gate of [`solve_kuks`]: everything
/// [`resolve_periodic_functional`] accepts, plus ONLY the pure SCAN / r2SCAN
/// meta-GGAs (the spin-resolved tau term is validated for those alone).
pub fn resolve_kuks_functional(name: &str) -> Result<(), FerricError> {
    resolve_open_shell_k_functional(name, "UKS")
}

/// The functional gate shared by the k-point UKS and ROKS SCFs (both run on
/// [`KPeriodicXcPolarized`]); `route` names the SCF in the refusal.
pub(crate) fn resolve_open_shell_k_functional(name: &str, route: &str) -> Result<(), FerricError> {
    // A range-separated hybrid goes through the RSH resolver (it refuses
    // range-separated meta-GGA, VV10 and double hybrids); the meta-GGA gate
    // below is for the global functionals only.
    let range_separated = ferric_dft::libxc::xc_def_from_name(name)
        .map(|d| d.cam.is_some())
        .unwrap_or(false);
    if range_separated {
        crate::rsh::resolve_periodic_functional_rsh(name)?;
    } else if crate::dft::is_meta_gga_name(name) {
        if !crate::kxcgrad::is_validated_mgga(name) {
            return Err(PeriodicDftError::Unsupported {
                feature: "meta-GGA",
                reason: format!("{name}: k-point {route} supports only the pure SCAN / r2SCAN"),
            }
            .into());
        }
        resolve_periodic_functional_mgga(name, true)?;
    } else {
        resolve_periodic_functional(name)?;
    }
    Ok(())
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
        resolve_kuks_functional(&inner.name)?;
        let xc_pol = xc_def_from_name_nspin(&inner.name, 2).map_err(|e| {
            FerricError::General(format!(
                "periodic DFT: functional {:?} (spin-polarized): {e:?}",
                inner.name
            ))
        })?;
        Ok(Self {
            inner,
            xc_pol,
            mutation: None,
        })
    }

    /// Whether the functional is a meta-GGA (the tau term is active).
    pub fn is_meta_gga(&self) -> bool {
        self.inner.mgga
    }

    /// Test hook: inject a deliberate tau defect (see [`KUksMggaMutation`]).
    #[doc(hidden)]
    pub fn with_mgga_mutation(mut self, m: KUksMggaMutation) -> Self {
        self.mutation = Some(m);
        self
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

    /// The functional's global exact-exchange fraction (`0` for a
    /// range-separated hybrid: see [`Self::rsh`]).
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.inner.exact_exchange_fraction()
    }

    /// Range-separation parameters if the functional is a CAM hybrid.
    pub fn rsh(&self) -> Option<crate::rsh::RshParams> {
        self.inner.rsh()
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
    fn spin_density(
        &self,
        c: &KAoChunk,
        dm: &[Array2<Complex64>],
        clk: &KXcClocks,
    ) -> (Array1<f64>, Array2<f64>) {
        let x = &self.inner;
        let np = c.points.len();
        let inv_nk = 1.0 / x.nk as f64;
        let mut rho = Array1::<f64>::zeros(np);
        let mut grad = Array2::<f64>::zeros((3, np));
        let mut acc = vec![0.0; np];
        for k in 0..x.nk {
            let p = clk.density_gemm.time(|| dm[k].dot(&conj(&c.chi[k])));
            let t0 = std::time::Instant::now();
            let rho_s = rho.as_slice_mut().expect("standard layout");
            add_re_column_dots(rho_s, inv_nk, c.chi[k].view(), p.view(), &mut acc);
            if x.gga {
                for ax in 0..3 {
                    let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                    let mut row = grad.row_mut(ax);
                    let out = row.as_slice_mut().expect("standard layout");
                    add_re_column_dots(out, inv_nk * 2.0, d_ax, p.view(), &mut acc);
                }
            }
            clk.density_contract.record(t0);
        }
        (rho, grad)
    }

    fn chunk_density(
        &self,
        c: &KAoChunk,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
        clk: &KXcClocks,
    ) -> UksDensityGrid {
        let np = c.points.len();
        let (rho_a, grad_a) = self.spin_density(c, da, clk);
        let (rho_b, grad_b) = self.spin_density(c, db, clk);
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
                let d = self.chunk_density(c, da, db, &KXcClocks::default());
                c.points
                    .iter()
                    .enumerate()
                    .map(|(g, p)| p.weight * (d.rho_a[g] + d.rho_b[g]))
                    .sum::<f64>()
            })
            .sum())
    }
}

impl KPeriodicXcPolarized {
    /// The density and per-spin tau the libxc kernel is evaluated on: the
    /// physical ones, except under a [`KUksMggaMutation`].
    #[allow(clippy::type_complexity)]
    fn kernel_inputs(
        &self,
        dens: UksDensityGrid,
        tau: Option<(Array1<f64>, Array1<f64>)>,
    ) -> (UksDensityGrid, Option<(Array1<f64>, Array1<f64>)>) {
        match (self.mutation, tau) {
            (Some(KUksMggaMutation::SwapTau), Some((ta, tb))) => (dens, Some((tb, ta))),
            (Some(KUksMggaMutation::TauBetaFromAlpha), Some((ta, _))) => {
                (dens, Some((ta.clone(), ta)))
            }
            (Some(KUksMggaMutation::DropTau), Some((ta, _))) => {
                let z = Array1::<f64>::zeros(ta.len());
                (dens, Some((z.clone(), z)))
            }
            (Some(KUksMggaMutation::TotalTauBothChannels), Some((ta, tb))) => {
                let t = &ta + &tb;
                (dens, Some((t.clone(), t)))
            }
            (Some(KUksMggaMutation::UnpolarizedEvaluator), tau) => {
                let rho = (&dens.rho_a + &dens.rho_b) * 0.5;
                let grad = (&dens.grad_a + &dens.grad_b) * 0.5;
                let np = rho.len();
                let mut sigma = Array2::<f64>::zeros((3, np));
                for g in 0..np {
                    let s: f64 = (0..3).map(|ax| grad[(ax, g)] * grad[(ax, g)]).sum();
                    sigma[(0, g)] = s;
                    sigma[(1, g)] = s;
                    sigma[(2, g)] = s;
                }
                let tau = tau.map(|(ta, tb)| {
                    let t = (&ta + &tb) * 0.5;
                    (t.clone(), t)
                });
                (
                    UksDensityGrid {
                        rho_a: rho.clone(),
                        rho_b: rho,
                        grad_a: grad.clone(),
                        grad_b: grad,
                        sigma,
                    },
                    tau,
                )
            }
            (_, tau) => (dens, tau),
        }
    }
}

impl KPointXcPolarized for KPeriodicXcPolarized {
    fn build_polarized(
        &mut self,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), FerricError> {
        let clock = StageClock::start();
        let out = self.build_polarized_untimed(da, db);
        self.inner.clocks.build.record(&clock);
        out
    }
}

impl KPeriodicXcPolarized {
    /// Every `build_polarized` call's stage and phase timings (see
    /// [`KPeriodicXc::xc_timings`]).
    pub fn xc_timings(&self) -> PbcTimings {
        self.inner.xc_timings()
    }

    /// See [`KPeriodicXc::record_scf_timings`].
    pub(crate) fn record_scf_timings(&self, t: &mut PbcTimings) {
        self.inner.record_scf_timings(t);
    }

    /// The one-off Bloch AO cache build (stage `"k XC AO cache"`).
    pub(crate) fn ao_cache_timing(&self) -> &StageTiming {
        &self.inner.ao_cache
    }

    /// One chunk's `(E_xc, V_α(k), V_β(k))`, each summed from zero.
    #[allow(clippy::type_complexity)]
    fn chunk_build_polarized(
        &self,
        c: &KAoChunk,
        da: &[Array2<Complex64>],
        db: &[Array2<Complex64>],
    ) -> (f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
        let x = &self.inner;
        let dens = self.chunk_density(c, da, db, &x.clocks);
        // Per-spin tau of THIS chunk (libxc's 1/2 convention), and the
        // density the libxc kernel sees (only the unpolarized-evaluator
        // mutant alters it).
        let tau = x.mgga.then(|| (x.chunk_tau(c, da), x.chunk_tau(c, db)));
        let (dens, tau_in) = self.kernel_inputs(dens, tau);
        let kern = x
            .clocks
            .kernel
            .time(|| polarized_kernel(&dens, tau_in.as_ref().map(|(a, b)| (a, b)), &self.xc_pol));
        let mut e = 0.0;
        for (g, p) in c.points.iter().enumerate() {
            e += p.weight * (dens.rho_a[g] + dens.rho_b[g]) * kern.exc[g];
        }
        let va = self.spin_potential(c, &dens, &kern, 0);
        let vb = self.spin_potential(c, &dens, &kern, 1);
        (e, va, vb)
    }

    /// Spin `spin`'s `V_σ(k)` of one chunk from zero: the LDA weight, the GGA
    /// axis functions and the tau weight, each floored on its own spin
    /// density (as the Gamma builder); `V^τ_β` is dropped under
    /// [`KUksMggaMutation::DropBetaVtau`].
    fn spin_potential(
        &self,
        c: &KAoChunk,
        dens: &UksDensityGrid,
        kern: &PolarizedKernel,
        spin: usize,
    ) -> Vec<Array2<Complex64>> {
        let x = &self.inner;
        let np = c.points.len();
        let (rho, vrho, vs_self, vtau, grad_self, grad_cross) = if spin == 0 {
            let k = kern;
            (
                &dens.rho_a,
                &k.vrho_a,
                &k.vsigma_aa,
                &k.vtau_a,
                &dens.grad_a,
                &dens.grad_b,
            )
        } else {
            let k = kern;
            (
                &dens.rho_b,
                &k.vrho_b,
                &k.vsigma_bb,
                &k.vtau_b,
                &dens.grad_b,
                &dens.grad_a,
            )
        };
        let gated = |g: usize, v: f64| if rho[g] > DENSITY_FLOOR { v } else { 0.0 };
        let w = |g: usize| c.points[g].weight;
        let a: Vec<f64> = (0..np).map(|g| gated(g, w(g) * vrho[g])).collect();
        let f_ax: Vec<Vec<f64>> = if x.gga {
            (0..3)
                .map(|ax| {
                    (0..np)
                        .map(|g| {
                            let cross = kern.vsigma_ab[g] * grad_cross[(ax, g)];
                            gated(g, w(g) * (2.0 * vs_self[g] * grad_self[(ax, g)] + cross))
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        let drop_tau = spin == 1 && self.mutation == Some(KUksMggaMutation::DropBetaVtau);
        let f_tau: Vec<f64> = if x.mgga && !drop_tau {
            (0..np).map(|g| gated(g, 0.5 * w(g) * vtau[g])).collect()
        } else {
            Vec::new()
        };
        let n = x.nbf;
        let mut v: Vec<Array2<Complex64>> = (0..x.nk).map(|_| Array2::zeros((n, n))).collect();
        for (k, vk) in v.iter_mut().enumerate() {
            let chi = c.chi[k].view();
            x.add_weighted_product(vk, chi, chi, &a, false);
            if !f_tau.is_empty() {
                // V^tau_s(k)_{mn} = sum_g 1/2 w v_tau_s sum_ax (d phi_m)* d phi_n
                for ax in 0..3 {
                    let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                    x.add_weighted_product(vk, d_ax, d_ax, &f_tau, false);
                }
            }
            for (ax, f) in f_ax.iter().enumerate() {
                let d_ax = c.dchi[k].index_axis(Axis(0), ax);
                x.add_weighted_product(vk, chi, d_ax, f, true);
            }
        }
        v
    }

    #[allow(clippy::type_complexity)]
    fn build_polarized_untimed(
        &self,
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
        let out_bytes = bytes_of((2 * x.nk * n * n) as u64, std::mem::size_of::<Complex64>());
        x.fold_chunks_in_order(
            out_bytes,
            |c| self.chunk_build_polarized(c, da, db),
            |(ec, ca, cb)| {
                e += ec;
                for (acc, m) in va.iter_mut().zip(&ca).chain(vb.iter_mut().zip(&cb)) {
                    *acc += m;
                }
            },
        );
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
    /// Test hook: a deliberate spin-resolved tau defect (meta-GGA only).
    #[doc(hidden)]
    pub mgga_mutation: Option<KUksMggaMutation>,
}

impl KUksConfig {
    /// Defaults (dense J/K, grid (75, 302) SSF, staged ewald start).
    pub fn new(cell: &Cell, exxdiv: ExxDiv, functional: &str) -> Self {
        Self {
            kuhf: KUhfConfig::for_cell(cell, exxdiv),
            functional: functional.to_string(),
            grid: PeriodicGridConfig::default(),
            xc: PeriodicXcConfig::default(),
            mgga_mutation: None,
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
    Ok(solve_kuks_keep(cell, prep, aux, mesh, cfg)?.0)
}

/// [`solve_kuks`] that also returns the hcore output and J/K object the SCF ran
/// on ([`KRetained`]) for the analytic forces and stress. `solve_kuks`
/// delegates here, so the energy is bit-identical. The dense object is built
/// with `ExxDiv::None` (the Madelung term is applied per stage by the builder,
/// and by the force/stress entry points from their `exxdiv` argument).
pub fn solve_kuks_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    cfg: &KUksConfig,
) -> Result<(KUksResult, KRetained), FerricError> {
    resolve_kuks_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    solve_kuks_on_grid_keep(cell, prep, aux, mesh, &grid, cfg)
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
    Ok(solve_kuks_on_grid_keep(cell, prep, aux, mesh, grid, cfg)?.0)
}

/// [`solve_kuks_on_grid`] that also returns the retained hcore and J/K.
pub fn solve_kuks_on_grid_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KUksConfig,
) -> Result<(KUksResult, KRetained), FerricError> {
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
    pxc.mutation = cfg.mgga_mutation;
    // Range-separated hybrid: fractions and the mixed Madelung constant live
    // INSIDE the attenuated K, which the SCF scales by 1 (see `kdft`).
    let rsh = pxc.rsh();
    let a = if rsh.is_some() {
        1.0
    } else {
        pxc.exact_exchange_fraction()
    };
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
    let clock = StageClock::start();
    let ints = match aux {
        None => {
            let mut dense = kc.dense;
            dense.rsh = rsh;
            KInts::Dense(KDenseAftEri::build(
                cell,
                prep,
                mesh,
                &hk.s,
                ExxDiv::None,
                &dense,
            )?)
        }
        Some(aux) => KInts::RsGdf(match rsh {
            Some(r) => KRsGdf::build_rsh(cell, prep, aux, mesh, &hk.s, &kc.rsgdf, r)?,
            None => KRsGdf::build(cell, prep, aux, mesh, &hk.s, &kc.rsgdf)?,
        }),
    };
    // The exxdiv constant the K applies: the mesh Madelung constant, or for
    // a range-separated hybrid the mixed one the attenuated K was built with.
    let v_m = match (&ints, rsh) {
        (KInts::Dense(e), Some(_)) => e.madelung_ewald(),
        (KInts::RsGdf(g), Some(_)) => g.madelung_ewald(),
        _ => mesh.madelung(cell)?,
    };
    let v_m = if rsh.is_some() && kc.mutation == Some(KUhfMutation::PlainMadelungForRsh) {
        mesh.madelung(cell)?
    } else {
        v_m
    };
    let applied = match kc.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let k_madelung = if kc.mutation == Some(KUhfMutation::MadelungHalfPerSpin) {
        0.5 * applied
    } else {
        applied
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
            return Err(FerricError::Convergence(format!(
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
    pxc.record_scf_timings(&mut timings);
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
    let kept = KRetained {
        hk,
        jk: match ints {
            KInts::Dense(e) => KRetainedJk::Dense(e),
            KInts::RsGdf(g) => KRetainedJk::RsGdf(g),
        },
    };
    let res = KUksResult {
        e_xc,
        exact_exchange_fraction: rsh.map_or(a, |r| r.c_sr),
        n_grid_points: grid.len(),
        electrons_on_grid,
        madelung: v_m,
        nocc: (na, nb),
        gaps,
        none_stage,
        scf,
        timings,
    };
    Ok((res, kept))
}
