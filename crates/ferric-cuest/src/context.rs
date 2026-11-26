//! A persistent cuEST session for an SCF: one-electron matrices once, then
//! exact (integral-direct, no density fitting) Coulomb and exchange matrices
//! every iteration.
//!
//! Everything here is in cuEST's AO order. The caller (ferric-scf's
//! `cuest_route`) owns the map to ferric's order.
//!
//! Conventions, from the cuEST 0.3 headers:
//!   J_pq = Σ_rs (pq|rs) D_rs        (cuestDirectCoulombCompute)
//!   K_pr = Σ_qs (pq|rs) D_qs        (cuestDirectSymmetricExchangeCompute)
//! which are ferric's `J[D]` and `K[D]`. The plan keeps cuEST's default exchange
//! operator (full-range 1/r, fraction 1.0); hybrids scale K on the caller's side.

use crate::cuda::DeviceBuffer;
use crate::ffi;
use crate::handle::{check, CuestHandle, Params, Workspace};
use crate::layout::AoLayout;
use crate::oe::{build_session, compute_stv, download, reject_unsupported, Session};
use crate::Result;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array2;
use std::time::Instant;

/// Settings for [`CuestContext::new`].
#[derive(Debug, Clone, Copy)]
pub struct ContextOptions {
    /// Pair-list screening threshold; 0 keeps every shell pair.
    pub pair_threshold: f64,
    /// Density-weighted Schwarz threshold for J and K (cuEST default 1e-13).
    pub density_schwarz: f64,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            pair_threshold: 0.0,
            density_schwarz: 1e-13,
        }
    }
}

/// Cumulative wall time spent in cuEST, seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextTimings {
    pub setup: f64,
    pub one_electron: f64,
    pub coulomb: f64,
    pub n_coulomb: usize,
    pub exchange: f64,
    pub n_exchange: usize,
}

/// Owned cuEST objects for one molecule and basis. Fields drop in declaration
/// order: the session (basis, pair list, plans) goes before the handle.
pub struct CuestContext {
    sess: Session,
    j_params: Params,
    k_params: Params,
    j_tmp: Workspace,
    k_tmp: Workspace,
    d_buf: DeviceBuffer,
    out_buf: DeviceBuffer,
    mol: Molecule,
    pub layout: AoLayout,
    pub timings: ContextTimings,
    h: CuestHandle,
}

impl CuestContext {
    /// Build the basis, pair list, one-electron plan and direct J/K plan.
    pub fn new(mol: &Molecule, bs: &BasisSet, opts: ContextOptions) -> Result<Self> {
        reject_unsupported(mol, bs)?;
        let layout = AoLayout::new(mol, bs)?;
        let nao = layout.nao;
        let t0 = Instant::now();
        let h = CuestHandle::new()?;
        let mut sess = build_session(&h, mol, bs, &layout, opts.pair_threshold)?;

        let plan_params = Params::new(ffi::CUEST_DIRECTINTPLAN_PARAMETERS)?;
        let (mut pd, mut td) = Default::default();
        // SAFETY: live basis and pair list; descriptors are valid out-pointers.
        check("cuestDirectIntPlanCreateWorkspaceQuery", unsafe {
            ffi::cuestDirectIntPlanCreateWorkspaceQuery(
                h.raw(),
                sess.basis,
                sess.pairs,
                plan_params.raw,
                &mut pd,
                &mut td,
                std::ptr::null_mut(),
            )
        })?;
        let mut ws = Workspace::allocate(&pd)?;
        let mut tmp = Workspace::allocate(&td)?;
        // SAFETY: workspaces sized by the query; the plan views basis and pairs,
        // which the session keeps alive until after the plan is destroyed.
        check("cuestDirectIntPlanCreate", unsafe {
            ffi::cuestDirectIntPlanCreate(
                h.raw(),
                sess.basis,
                sess.pairs,
                plan_params.raw,
                &mut ws.raw,
                &mut tmp.raw,
                &mut sess.direct,
            )
        })?;
        sess._direct_ws = Some(ws);

        let bytes = nao * nao * std::mem::size_of::<f64>();
        let d_buf = DeviceBuffer::new(bytes)?;
        let out_buf = DeviceBuffer::new(bytes)?;
        let (dp, op) = (d_buf.ptr().cast::<f64>(), out_buf.ptr().cast::<f64>());

        let mut j_params = Params::new(ffi::CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS)?;
        j_params.set_f64(
            ffi::CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
            opts.density_schwarz,
        )?;
        let mut jd = Default::default();
        // SAFETY: live plan; buffers hold nao*nao doubles.
        check("cuestDirectCoulombComputeWorkspaceQuery", unsafe {
            ffi::cuestDirectCoulombComputeWorkspaceQuery(
                h.raw(),
                sess.direct,
                j_params.raw,
                &mut jd,
                1,
                dp,
                op,
            )
        })?;
        let mut k_params = Params::new(ffi::CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS)?;
        k_params.set_f64(
            ffi::CUEST_DIRECTSYMMETRICEXCHANGECOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD,
            opts.density_schwarz,
        )?;
        let mut kd = Default::default();
        // SAFETY: as above.
        check(
            "cuestDirectSymmetricExchangeComputeWorkspaceQuery",
            unsafe {
                ffi::cuestDirectSymmetricExchangeComputeWorkspaceQuery(
                    h.raw(),
                    sess.direct,
                    k_params.raw,
                    &mut kd,
                    1,
                    dp,
                    op,
                )
            },
        )?;
        let j_tmp = Workspace::allocate(&jd)?;
        let k_tmp = Workspace::allocate(&kd)?;
        let timings = ContextTimings {
            setup: t0.elapsed().as_secs_f64(),
            ..Default::default()
        };
        Ok(Self {
            sess,
            j_params,
            k_params,
            j_tmp,
            k_tmp,
            d_buf,
            out_buf,
            mol: mol.clone(),
            layout,
            timings,
            h,
        })
    }

    pub fn nao(&self) -> usize {
        self.layout.nao
    }

    /// S, T, V in cuEST order.
    pub fn one_electron(&mut self) -> Result<[Array2<f64>; 3]> {
        let t0 = Instant::now();
        let (m, _) = compute_stv(&self.h, &self.sess, &self.mol, self.nao())?;
        self.timings.one_electron += t0.elapsed().as_secs_f64();
        Ok(m)
    }

    /// Upload ½(D + Dᵀ): both cuEST builds require a symmetric density.
    fn upload(&mut self, d: &Array2<f64>) -> Result<()> {
        let n = self.nao();
        assert_eq!(d.dim(), (n, n), "density is not nao x nao");
        let sym = (d + &d.t()) * 0.5;
        let sym = sym.as_standard_layout();
        self.d_buf
            .copy_from_host(sym.as_slice().expect("standard layout"))
    }

    /// `J[D]` in cuEST order (exact four-centre, integral-direct).
    pub fn coulomb(&mut self, d: &Array2<f64>) -> Result<Array2<f64>> {
        let t0 = Instant::now();
        self.upload(d)?;
        let (dp, op) = (
            self.d_buf.ptr().cast::<f64>(),
            self.out_buf.ptr().cast::<f64>(),
        );
        // SAFETY: live plan; workspace sized by the query for one matrix.
        check("cuestDirectCoulombCompute", unsafe {
            ffi::cuestDirectCoulombCompute(
                self.h.raw(),
                self.sess.direct,
                self.j_params.raw,
                &mut self.j_tmp.raw,
                1,
                dp,
                op,
            )
        })?;
        let j = download(&self.out_buf, self.nao())?;
        self.timings.coulomb += t0.elapsed().as_secs_f64();
        self.timings.n_coulomb += 1;
        Ok(j)
    }

    /// `K[D]` in cuEST order (exact four-centre, integral-direct, full-range 1/r).
    pub fn exchange(&mut self, d: &Array2<f64>) -> Result<Array2<f64>> {
        let t0 = Instant::now();
        self.upload(d)?;
        let (dp, op) = (
            self.d_buf.ptr().cast::<f64>(),
            self.out_buf.ptr().cast::<f64>(),
        );
        // SAFETY: as in `coulomb`.
        check("cuestDirectSymmetricExchangeCompute", unsafe {
            ffi::cuestDirectSymmetricExchangeCompute(
                self.h.raw(),
                self.sess.direct,
                self.k_params.raw,
                &mut self.k_tmp.raw,
                1,
                dp,
                op,
            )
        })?;
        let k = download(&self.out_buf, self.nao())?;
        self.timings.exchange += t0.elapsed().as_secs_f64();
        self.timings.n_exchange += 1;
        Ok(k)
    }
}
