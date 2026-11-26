//! One-electron matrices (S, T, V) on the GPU through cuEST.
//!
//! The matrices come back in cuEST's AO order; [`crate::layout::AoMap`]
//! reorders them into ferric's. V uses charges q_A = -Z_A (cuEST evaluates
//! the potential of a positive unit test charge), so it carries libint's sign.

use crate::basis::convert_shell;
use crate::cuda::DeviceBuffer;
use crate::ffi;
use crate::handle::{check, CuestHandle, Params, Workspace};
use crate::layout::AoLayout;
use crate::{CuestError, Result};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array2;
use std::os::raw::c_void;
use std::time::Instant;

/// S, T, V in cuEST AO order, plus the layout they were built with.
#[derive(Debug)]
pub struct OneElectronMatrices {
    pub s: Array2<f64>,
    pub t: Array2<f64>,
    pub v: Array2<f64>,
    pub layout: AoLayout,
    /// Wall time of each stage, seconds: (setup, S, T, V).
    pub timings: [f64; 4],
}

/// Owned cuEST objects; fields drop in declaration order, so each object is
/// destroyed before the persistent workspace it lives in. The direct (J/K)
/// plan is optional: [`compute_one_electron`] does not build it,
/// [`crate::context::CuestContext`] does.
pub(crate) struct Session {
    pub(crate) direct: ffi::cuestDirectIntPlan_t,
    pub(crate) _direct_ws: Option<Workspace>,
    pub(crate) plan: ffi::cuestOEIntPlan_t,
    _plan_ws: Workspace,
    pub(crate) pairs: ffi::cuestAOPairList_t,
    _pairs_ws: Workspace,
    pub(crate) basis: ffi::cuestAOBasis_t,
    _basis_ws: Workspace,
    shells: Vec<ffi::cuestAOShell_t>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: each pointer came from its Create call and is destroyed once,
        // dependents first.
        unsafe {
            if !self.direct.is_null() {
                let _ = ffi::cuestDirectIntPlanDestroy(self.direct);
            }
            if !self.plan.is_null() {
                let _ = ffi::cuestOEIntPlanDestroy(self.plan);
            }
            if !self.pairs.is_null() {
                let _ = ffi::cuestAOPairListDestroy(self.pairs);
            }
            if !self.basis.is_null() {
                let _ = ffi::cuestAOBasisDestroy(self.basis);
            }
            for &s in &self.shells {
                let _ = ffi::cuestAOShellDestroy(s);
            }
        }
    }
}

struct ShellGuard(Vec<ffi::cuestAOShell_t>);
impl Drop for ShellGuard {
    fn drop(&mut self) {
        for &s in &self.0 {
            // SAFETY: created by cuestAOShellCreate, destroyed once.
            unsafe {
                let _ = ffi::cuestAOShellDestroy(s);
            }
        }
    }
}

pub(crate) fn reject_unsupported(mol: &Molecule, bs: &BasisSet) -> Result<()> {
    if mol.atoms.iter().any(|a| a.ghost) {
        return Err(CuestError::Unsupported(
            "ghost atoms (not wired yet)".into(),
        ));
    }
    if mol.atoms.iter().any(|a| bs.ecp_for_element(a.z).is_some()) {
        return Err(CuestError::Unsupported(
            "ECP atoms (cuEST ECP plan not wired yet)".into(),
        ));
    }
    Ok(())
}

/// Shells, basis, pair list and one-electron plan. `pair_threshold` is the
/// pair-list screening threshold (0 keeps every pair).
pub(crate) fn build_session(
    h: &CuestHandle,
    mol: &Molecule,
    bs: &BasisSet,
    layout: &AoLayout,
    pair_threshold: f64,
) -> Result<Session> {
    let natoms = mol.atoms.len();
    let shell_params = Params::new(ffi::CUEST_AOSHELL_PARAMETERS)?;
    let mut guard = ShellGuard(Vec::with_capacity(layout.shells.len()));
    for atom in &mol.atoms {
        for sh in bs.for_element(atom.z).unwrap_or_default() {
            let c = convert_shell(sh)
                .map_err(|e| CuestError::Unsupported(format!("shell conversion: {e}")))?;
            let mut out = std::ptr::null_mut();
            // SAFETY: exponent/coefficient slices outlive the call (cuEST copies them).
            check("cuestAOShellCreate", unsafe {
                ffi::cuestAOShellCreate(
                    h.raw(),
                    i32::from(layout.cuest_pure),
                    c.l as u64,
                    c.exponents.len() as u64,
                    c.exponents.as_ptr(),
                    c.coefficients.as_ptr(),
                    shell_params.raw,
                    &mut out,
                )
            })?;
            guard.0.push(out);
        }
    }
    let per_atom = layout.shells_per_atom(natoms);

    // AO basis.
    let basis_params = Params::new(ffi::CUEST_AOBASIS_PARAMETERS)?;
    let (mut pd, mut td) = Default::default();
    // SAFETY: arrays sized natoms / nshells; descriptors are valid out-pointers.
    check("cuestAOBasisCreateWorkspaceQuery", unsafe {
        ffi::cuestAOBasisCreateWorkspaceQuery(
            h.raw(),
            natoms as u64,
            per_atom.as_ptr(),
            guard.0.as_ptr(),
            basis_params.raw,
            &mut pd,
            &mut td,
            std::ptr::null_mut(),
        )
    })?;
    let mut basis_ws = Workspace::allocate(&pd)?;
    let mut basis_tmp = Workspace::allocate(&td)?;
    let mut basis = std::ptr::null_mut();
    // SAFETY: as above, workspaces sized by the query.
    check("cuestAOBasisCreate", unsafe {
        ffi::cuestAOBasisCreate(
            h.raw(),
            natoms as u64,
            per_atom.as_ptr(),
            guard.0.as_ptr(),
            basis_params.raw,
            &mut basis_ws.raw,
            &mut basis_tmp.raw,
            &mut basis,
        )
    })?;
    // From here the Session owns the shells and the basis.
    let mut sess = Session {
        direct: std::ptr::null_mut(),
        _direct_ws: None,
        plan: std::ptr::null_mut(),
        _plan_ws: Workspace::allocate(&Default::default())?,
        pairs: std::ptr::null_mut(),
        _pairs_ws: Workspace::allocate(&Default::default())?,
        basis,
        _basis_ws: basis_ws,
        shells: std::mem::take(&mut guard.0),
    };

    let mut nao_c: u64 = 0;
    // SAFETY: live basis object; out-value sized u64 (documented attribute type).
    check("cuestQuery(AOBASIS_NUM_AO)", unsafe {
        ffi::cuestQuery(
            h.raw(),
            ffi::CUEST_AOBASIS,
            sess.basis as *const c_void,
            ffi::CUEST_AOBASIS_NUM_AO,
            (&mut nao_c as *mut u64).cast(),
            std::mem::size_of::<u64>(),
        )
    })?;
    if nao_c as usize != layout.nao {
        return Err(CuestError::Unsupported(format!(
            "cuEST reports {nao_c} AOs, ferric's layout has {}",
            layout.nao
        )));
    }

    // Pair list.
    let xyz: Vec<f64> = mol.atoms.iter().flat_map(|a| [a.x, a.y, a.zpos]).collect();
    let pair_params = Params::new(ffi::CUEST_AOPAIRLIST_PARAMETERS)?;
    let (mut pd, mut td) = Default::default();
    // SAFETY: xyz has 3*natoms entries.
    check("cuestAOPairListCreateWorkspaceQuery", unsafe {
        ffi::cuestAOPairListCreateWorkspaceQuery(
            h.raw(),
            sess.basis,
            natoms as u64,
            xyz.as_ptr(),
            pair_threshold,
            pair_params.raw,
            &mut pd,
            &mut td,
            std::ptr::null_mut(),
        )
    })?;
    sess._pairs_ws = Workspace::allocate(&pd)?;
    let mut tmp = Workspace::allocate(&td)?;
    // SAFETY: as above.
    check("cuestAOPairListCreate", unsafe {
        ffi::cuestAOPairListCreate(
            h.raw(),
            sess.basis,
            natoms as u64,
            xyz.as_ptr(),
            pair_threshold,
            pair_params.raw,
            &mut sess._pairs_ws.raw,
            &mut tmp.raw,
            &mut sess.pairs,
        )
    })?;

    // One-electron plan.
    let plan_params = Params::new(ffi::CUEST_OEINTPLAN_PARAMETERS)?;
    let (mut pd, mut td) = Default::default();
    // SAFETY: live basis and pair list.
    check("cuestOEIntPlanCreateWorkspaceQuery", unsafe {
        ffi::cuestOEIntPlanCreateWorkspaceQuery(
            h.raw(),
            sess.basis,
            sess.pairs,
            plan_params.raw,
            &mut pd,
            &mut td,
            std::ptr::null_mut(),
        )
    })?;
    sess._plan_ws = Workspace::allocate(&pd)?;
    let mut tmp = Workspace::allocate(&td)?;
    // SAFETY: as above.
    check("cuestOEIntPlanCreate", unsafe {
        ffi::cuestOEIntPlanCreate(
            h.raw(),
            sess.basis,
            sess.pairs,
            plan_params.raw,
            &mut sess._plan_ws.raw,
            &mut tmp.raw,
            &mut sess.plan,
        )
    })?;
    Ok(sess)
}

pub(crate) fn download(buf: &DeviceBuffer, nao: usize) -> Result<Array2<f64>> {
    let mut host = vec![0.0; nao * nao];
    buf.copy_to_host(&mut host)?;
    Ok(Array2::from_shape_vec((nao, nao), host).expect("nao*nao"))
}

/// S, T, V (cuEST order) on a live session, and the seconds each took.
pub(crate) fn compute_stv(
    h: &CuestHandle,
    sess: &Session,
    mol: &Molecule,
    nao: usize,
) -> Result<([Array2<f64>; 3], [f64; 3])> {
    let out = DeviceBuffer::new(nao * nao * std::mem::size_of::<f64>())?;
    let out_ptr = out.ptr().cast::<f64>();

    // S
    let t1 = Instant::now();
    let p = Params::new(ffi::CUEST_OVERLAPCOMPUTE_PARAMETERS)?;
    let mut td = Default::default();
    // SAFETY: live plan; out buffer holds nao*nao doubles on the device.
    check("cuestOverlapComputeWorkspaceQuery", unsafe {
        ffi::cuestOverlapComputeWorkspaceQuery(h.raw(), sess.plan, p.raw, &mut td, out_ptr)
    })?;
    let mut tmp = Workspace::allocate(&td)?;
    check("cuestOverlapCompute", unsafe {
        ffi::cuestOverlapCompute(h.raw(), sess.plan, p.raw, &mut tmp.raw, out_ptr)
    })?;
    let s = download(&out, nao)?;
    let t_s = t1.elapsed().as_secs_f64();

    // T
    let t1 = Instant::now();
    let p = Params::new(ffi::CUEST_KINETICCOMPUTE_PARAMETERS)?;
    let mut td = Default::default();
    // SAFETY: as above.
    check("cuestKineticComputeWorkspaceQuery", unsafe {
        ffi::cuestKineticComputeWorkspaceQuery(h.raw(), sess.plan, p.raw, &mut td, out_ptr)
    })?;
    let mut tmp = Workspace::allocate(&td)?;
    check("cuestKineticCompute", unsafe {
        ffi::cuestKineticCompute(h.raw(), sess.plan, p.raw, &mut tmp.raw, out_ptr)
    })?;
    let t = download(&out, nao)?;
    let t_t = t1.elapsed().as_secs_f64();

    // V: charges -Z at the nuclei. cuestPotentialCompute documents xyz and q
    // as GPU arrays (unlike the pair-list xyz, which is on the CPU).
    let t1 = Instant::now();
    let xyz: Vec<f64> = mol.atoms.iter().flat_map(|a| [a.x, a.y, a.zpos]).collect();
    let q: Vec<f64> = mol.atoms.iter().map(|a| -f64::from(a.z)).collect();
    let mut xyz_d = DeviceBuffer::new(std::mem::size_of_val(xyz.as_slice()))?;
    xyz_d.copy_from_host(&xyz)?;
    let mut q_d = DeviceBuffer::new(std::mem::size_of_val(q.as_slice()))?;
    q_d.copy_from_host(&q)?;
    let mut p = Params::new(ffi::CUEST_POTENTIALCOMPUTE_PARAMETERS)?;
    p.set_enum(
        ffi::CUEST_POTENTIALCOMPUTE_PARAMETERS_FFLOAT_USAGE_MODE,
        ffi::CUEST_FFLOAT_USAGE_MODE_OFF,
    )?;
    let n = q.len() as u64;
    let (xp, qp) = (xyz_d.ptr().cast::<f64>(), q_d.ptr().cast::<f64>());
    let mut td = Default::default();
    // SAFETY: device arrays hold 3n and n doubles; out holds nao*nao.
    check("cuestPotentialComputeWorkspaceQuery", unsafe {
        ffi::cuestPotentialComputeWorkspaceQuery(
            h.raw(),
            sess.plan,
            p.raw,
            &mut td,
            n,
            xp,
            qp,
            out_ptr,
        )
    })?;
    let mut tmp = Workspace::allocate(&td)?;
    check("cuestPotentialCompute", unsafe {
        ffi::cuestPotentialCompute(h.raw(), sess.plan, p.raw, &mut tmp.raw, n, xp, qp, out_ptr)
    })?;
    let v = download(&out, nao)?;
    let t_v = t1.elapsed().as_secs_f64();
    Ok(([s, t, v], [t_s, t_t, t_v]))
}

/// Compute S, T and V for `mol` in `bs` on GPU 0, in cuEST AO order.
///
/// V is computed in fp64 (ffloat explicitly off) so it can be compared with
/// libint at 1e-10; S and T have no precision knob. The pair list keeps every
/// pair (threshold 0), to compare against libint exactly.
pub fn compute_one_electron(mol: &Molecule, bs: &BasisSet) -> Result<OneElectronMatrices> {
    reject_unsupported(mol, bs)?;
    let layout = AoLayout::new(mol, bs)?;
    let t0 = Instant::now();
    let h = CuestHandle::new()?;
    let sess = build_session(&h, mol, bs, &layout, 0.0)?;
    let setup = t0.elapsed().as_secs_f64();
    let ([s, t, v], [t_s, t_t, t_v]) = compute_stv(&h, &sess, mol, layout.nao)?;
    drop(sess);
    Ok(OneElectronMatrices {
        s,
        t,
        v,
        layout,
        timings: [setup, t_s, t_t, t_v],
    })
}
