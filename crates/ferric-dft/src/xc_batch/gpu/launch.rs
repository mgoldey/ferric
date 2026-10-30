//! The device launches of one wave: one strided-batched GEMM per (group, item)
//! over the zero-padded layout (see `layout`), or, under the `XC_GEMM_LOOP`
//! test seam, the per-batch loop it replaces (each batch's products at its true
//! `nact`, same staging, same AO copy), kept as the reference the batched form
//! is tested against.
use ndarray::s;

use ferric_core::gpu::batched::{
    dev_batched_left, dev_batched_right, gemm_f64_strided_batched_dev_into, BatchedDims,
};
use ferric_core::gpu::device::GpuError;
use ferric_core::gpu::gemm::{
    dev_left, dev_left_padded, dev_right, dev_right_padded, gemm_f64_dev_into,
};
use ferric_core::gpu::mixed::{
    effective_k_panel, gemm_f32_f64acc_dev_into, gemm_f32_f64acc_strided_batched_dev,
};

use super::layout::{Group, WaveLayout};
use super::{twin, DensityInput, DeviceXc, ScreenedGrid};

/// Test seam (`test-seams` feature): every wave launches its products batch by
/// batch instead of as strided batches.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static XC_GEMM_LOOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn per_batch_loop() -> bool {
    #[cfg(feature = "test-seams")]
    {
        XC_GEMM_LOOP.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

/// The geometry one group shares: padded `nact`, its batch count, and where
/// its first AO block starts (consecutive blocks are `ao_stride` apart).
#[derive(Clone, Copy)]
struct GroupGeo {
    np: usize,
    cnt: usize,
    ao_at: usize,
    ao_stride: usize,
    nb_pad: usize,
}

impl GroupGeo {
    /// The group's AO blocks (all `cnt` of them) in the AO copy.
    fn ao_range(&self) -> std::ops::Range<usize> {
        self.ao_at..self.ao_at + self.cnt * self.ao_stride
    }
}

impl DeviceXc {
    fn geo(&self, b0: usize, g: &Group) -> Result<GroupGeo, GpuError> {
        let first = b0 + g.batches[0];
        let ao_at = self.layout.offset(first).ok_or_else(|| {
            GpuError::Layout(format!(
                "XC batch {first} has no AO block but is in a group"
            ))
        })?;
        Ok(GroupGeo {
            np: self.layout.na_pad[g.bucket],
            cnt: g.batches.len(),
            ao_at,
            ao_stride: self.layout.block_len(g.bucket),
            nb_pad: self.layout.nb_pad,
        })
    }

    /// Pass 1 of a wave: `Φ` for every density, group by group.
    pub(super) fn launch_density_wave(
        &mut self,
        sg: &ScreenedGrid,
        b0: usize,
        wl: &WaveLayout,
        ds: &[DensityInput],
        nc_p: usize,
    ) -> Result<(), GpuError> {
        for g in &wl.groups {
            let geo = self.geo(b0, g)?;
            let nas: Vec<usize> = g
                .batches
                .iter()
                .map(|&j| sg.batches[b0 + j].funcs.len())
                .collect();
            for (d, dd) in ds.iter().enumerate() {
                let at = [g.item_base(0, d), g.item_base(1, d), g.item_base(2, d)];
                match (dd, per_batch_loop()) {
                    (DensityInput::Dense(_), false) => self.dense_batched(geo, at, nc_p)?,
                    (DensityInput::Dense(_), true) => self.dense_loop(geo, at, nc_p, &nas)?,
                    (DensityInput::Factored { c_occ, .. }, false) => {
                        self.factored_batched(geo, at, nc_p, c_occ.ncols())?
                    }
                    (DensityInput::Factored { c_occ, .. }, true) => {
                        self.factored_loop(geo, at, nc_p, c_occ.ncols(), &nas)?
                    }
                }
            }
        }
        Ok(())
    }

    /// `Φ_b = D_b · X_b` for the whole group (`np × np · np × nc_p`).
    fn dense_batched(
        &mut self,
        geo: GroupGeo,
        [ii, _, oo]: [usize; 3],
        nc_p: usize,
    ) -> Result<(), GpuError> {
        let GroupGeo { np, cnt, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_out,
            ..
        } = self;
        let dt = twin(np, np);
        let left = dev_batched_left(
            stage_in.buf().slice(ii..ii + cnt * np * np),
            &dt.view(),
            cnt,
            np * np,
        )?;
        let xt = twin(np, 4 * geo.nb_pad);
        let right = dev_batched_right(
            ao.buf().slice(geo.ao_range()),
            &xt.slice(s![.., ..nc_p]),
            geo.cnt,
            geo.ao_stride,
        )?;
        let mut phi = stage_out.buf_mut().slice_mut(oo..oo + cnt * np * nc_p);
        let dims = BatchedDims {
            m: np,
            k: np,
            n: nc_p,
            batch: cnt,
        };
        gemm_f64_strided_batched_dev_into(dev, dims, &left, &right, &mut phi, np * nc_p)
    }

    /// `Ψ_b = C_bᵀ · X_b`, then `Φ_b = C_b · Ψ_b`, for the whole group.
    fn factored_batched(
        &mut self,
        geo: GroupGeo,
        [ii, mm, oo]: [usize; 3],
        nc_p: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let GroupGeo { np, cnt, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_mid,
            stage_out,
            ..
        } = self;
        let ct = twin(np, k);
        let csub = || stage_in.buf().slice(ii..ii + cnt * np * k);
        {
            let left = dev_batched_left(csub(), &ct.t(), cnt, np * k)?;
            let xt = twin(np, 4 * geo.nb_pad);
            let right = dev_batched_right(
                ao.buf().slice(geo.ao_range()),
                &xt.slice(s![.., ..nc_p]),
                geo.cnt,
                geo.ao_stride,
            )?;
            let mut psi = stage_mid.buf_mut().slice_mut(mm..mm + cnt * k * nc_p);
            let dims = BatchedDims {
                m: k,
                k: np,
                n: nc_p,
                batch: cnt,
            };
            gemm_f64_strided_batched_dev_into(dev, dims, &left, &right, &mut psi, k * nc_p)?;
        }
        let pt = twin(k, nc_p);
        let psi = dev_batched_right(
            stage_mid.buf().slice(mm..mm + cnt * k * nc_p),
            &pt.view(),
            cnt,
            k * nc_p,
        )?;
        let left = dev_batched_left(csub(), &ct.view(), cnt, np * k)?;
        let mut phi = stage_out.buf_mut().slice_mut(oo..oo + cnt * np * nc_p);
        let dims = BatchedDims {
            m: np,
            k,
            n: nc_p,
            batch: cnt,
        };
        gemm_f64_strided_batched_dev_into(dev, dims, &left, &psi, &mut phi, np * nc_p)
    }

    /// [`Self::dense_batched`] one batch at a time, at each batch's true `nact`.
    fn dense_loop(
        &mut self,
        geo: GroupGeo,
        [ii, _, oo]: [usize; 3],
        nc_p: usize,
        nas: &[usize],
    ) -> Result<(), GpuError> {
        let GroupGeo { np, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_out,
            ..
        } = self;
        let (dt, xt) = (twin(np, np), twin(np, 4 * geo.nb_pad));
        for (jg, &na) in nas.iter().enumerate() {
            let i0 = ii + jg * np * np;
            let a0 = geo.ao_at + jg * geo.ao_stride;
            let o0 = oo + jg * np * nc_p;
            let d = dev_left_padded(
                stage_in.buf().slice(i0..i0 + np * np),
                &dt.slice(s![..na, ..na]),
            )?;
            let x = dev_right_padded(
                ao.buf().slice(a0..a0 + geo.ao_stride),
                &xt.slice(s![..na, ..nc_p]),
            )?;
            let mut phi = stage_out.buf_mut().slice_mut(o0..o0 + na * nc_p);
            gemm_f64_dev_into(dev, na, na, nc_p, &d, &x, &mut phi, na.max(1))?;
        }
        Ok(())
    }

    /// [`Self::factored_batched`] one batch at a time, at each batch's true `nact`.
    fn factored_loop(
        &mut self,
        geo: GroupGeo,
        [ii, mm, oo]: [usize; 3],
        nc_p: usize,
        k: usize,
        nas: &[usize],
    ) -> Result<(), GpuError> {
        let GroupGeo { np, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_mid,
            stage_out,
            ..
        } = self;
        let xt = twin(np, 4 * geo.nb_pad);
        for (jg, &na) in nas.iter().enumerate() {
            let i0 = ii + jg * np * k;
            let m0 = mm + jg * k * nc_p;
            let a0 = geo.ao_at + jg * geo.ao_stride;
            let o0 = oo + jg * np * nc_p;
            let ct = twin(na, k);
            let csub = || stage_in.buf().slice(i0..i0 + na * k);
            {
                let ctt = dev_left(csub(), &ct.t())?;
                let x = dev_right_padded(
                    ao.buf().slice(a0..a0 + geo.ao_stride),
                    &xt.slice(s![..na, ..nc_p]),
                )?;
                let mut psi = stage_mid.buf_mut().slice_mut(m0..m0 + k * nc_p);
                gemm_f64_dev_into(dev, k, na, nc_p, &ctt, &x, &mut psi, na.max(1))?;
            }
            let pt = twin(k, nc_p);
            let psi = dev_right(stage_mid.buf().slice(m0..m0 + k * nc_p), &pt.view())?;
            let c = dev_left(csub(), &ct.view())?;
            let mut phi = stage_out.buf_mut().slice_mut(o0..o0 + na * nc_p);
            gemm_f64_dev_into(dev, na, k, nc_p, &c, &psi, &mut phi, k.max(1))?;
        }
        Ok(())
    }

    /// Pass 2 of a wave: `V_b = A_b · X_bᵀ` for every matrix, group by group;
    /// f64, or with `mixed` the f32-operand / f64-accumulate kernel on `ao32`
    /// and `stage_in32`. Returns the mixed panels flushed.
    pub(super) fn launch_potential_wave(
        &mut self,
        sg: &ScreenedGrid,
        b0: usize,
        wl: &WaveLayout,
        (nmats, mixed): (usize, bool),
        nc_p: usize,
    ) -> Result<usize, GpuError> {
        if mixed && self.mixed.is_none() {
            return Err(GpuError::Layout(
                "mixed XC launch without the f32 state".into(),
            ));
        }
        let mut panels = 0usize;
        for g in &wl.groups {
            let geo = self.geo(b0, g)?;
            let nas: Vec<usize> = g
                .batches
                .iter()
                .map(|&j| sg.batches[b0 + j].funcs.len())
                .collect();
            for mat in 0..nmats {
                let at = (g.item_base(0, mat), g.item_base(2, mat));
                panels += if per_batch_loop() {
                    self.vxc_loop(geo, at, nc_p, mixed, &nas)?
                } else {
                    self.vxc_batched(geo, at, nc_p, mixed)?
                };
            }
        }
        Ok(panels)
    }

    /// `V_b = A_b · X_bᵀ` for the whole group (`np × nc_p · nc_p × np`).
    fn vxc_batched(
        &mut self,
        geo: GroupGeo,
        (ii, oo): (usize, usize),
        nc_p: usize,
        mixed: bool,
    ) -> Result<usize, GpuError> {
        let GroupGeo { np, cnt, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_out,
            mixed: mixed_state,
            ..
        } = self;
        let (at, xt) = (twin(np, nc_p), twin(np, 4 * geo.nb_pad));
        let x_t = xt.slice(s![.., ..nc_p]);
        let x_t = x_t.t();
        let mut vb = stage_out.buf_mut().slice_mut(oo..oo + cnt * np * np);
        let dims = BatchedDims {
            m: np,
            k: nc_p,
            n: np,
            batch: cnt,
        };
        match (mixed, mixed_state.as_mut()) {
            (true, Some(m)) => {
                let a = dev_batched_left(
                    m.stage_in32.buf().slice(ii..ii + cnt * np * nc_p),
                    &at.view(),
                    cnt,
                    np * nc_p,
                )?;
                let x = dev_batched_right(
                    m.ao32.buf().slice(geo.ao_range()),
                    &x_t,
                    geo.cnt,
                    geo.ao_stride,
                )?;
                let mut c32 = m.c32.buf_mut().slice_mut(..cnt * np * np);
                gemm_f32_f64acc_strided_batched_dev(
                    dev,
                    dims,
                    &a,
                    &x,
                    &mut c32,
                    &mut vb,
                    effective_k_panel(),
                )
            }
            _ => {
                let a = dev_batched_left(
                    stage_in.buf().slice(ii..ii + cnt * np * nc_p),
                    &at.view(),
                    cnt,
                    np * nc_p,
                )?;
                let x = dev_batched_right(
                    ao.buf().slice(geo.ao_range()),
                    &x_t,
                    geo.cnt,
                    geo.ao_stride,
                )?;
                gemm_f64_strided_batched_dev_into(dev, dims, &a, &x, &mut vb, np * np)?;
                Ok(0)
            }
        }
    }

    /// [`Self::vxc_batched`] one batch at a time, at each batch's true `nact`
    /// (rows of `V_b`; its columns stay padded, so the rows keep their place).
    fn vxc_loop(
        &mut self,
        geo: GroupGeo,
        (ii, oo): (usize, usize),
        nc_p: usize,
        mixed: bool,
        nas: &[usize],
    ) -> Result<usize, GpuError> {
        let GroupGeo { np, .. } = geo;
        let Self {
            dev,
            ao,
            stage_in,
            stage_out,
            mixed: mixed_state,
            ..
        } = self;
        let xt = twin(np, 4 * geo.nb_pad);
        let x_t = xt.slice(s![.., ..nc_p]);
        let x_t = x_t.t();
        let mut panels = 0usize;
        for (jg, &na) in nas.iter().enumerate() {
            let i0 = ii + jg * np * nc_p;
            let a0 = geo.ao_at + jg * geo.ao_stride;
            let o0 = oo + jg * np * np;
            let at = twin(na, nc_p);
            let mut vb = stage_out.buf_mut().slice_mut(o0..o0 + na * np);
            match (mixed, mixed_state.as_mut()) {
                (true, Some(m)) => {
                    let a = dev_left(m.stage_in32.buf().slice(i0..i0 + na * nc_p), &at.view())?;
                    let x = dev_right_padded(m.ao32.buf().slice(a0..a0 + geo.ao_stride), &x_t)?;
                    let mut c32 = m.c32.buf_mut().slice_mut(..na * np);
                    panels += gemm_f32_f64acc_dev_into(
                        dev,
                        (na, nc_p, np),
                        &a,
                        &x,
                        &mut c32,
                        &mut vb,
                        effective_k_panel(),
                    )?;
                }
                _ => {
                    let a = dev_left(stage_in.buf().slice(i0..i0 + na * nc_p), &at.view())?;
                    let x = dev_right_padded(ao.buf().slice(a0..a0 + geo.ao_stride), &x_t)?;
                    gemm_f64_dev_into(dev, na, nc_p, np, &a, &x, &mut vb, nc_p.max(1))?;
                }
            }
        }
        Ok(panels)
    }
}
