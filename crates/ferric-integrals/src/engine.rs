//! Integral evaluation engines wrapping libint2 compute calls.
//!
//! Each [`Engine`] owns a libint2 engine handle and a scratch buffer.
//! Engines are created for specific integral types (1e, 2e, 3-center, derivatives)
//! and reused across shell loops.

use crate::basis_bridge::PreparedBasis;
use crate::ffi;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use std::os::raw::{c_int, c_void};

/// An integral evaluation engine backed by a libint2 engine handle.
///
/// Not `Send` or `Sync` -- create one engine per thread for parallel evaluation.
pub struct Engine {
    handles: Vec<(f64, *mut c_void)>,
    buf: Vec<f64>,
    scratch: Vec<f64>,
}

unsafe impl Send for Engine {}

impl Engine {
    /// Create a 4-center two-electron integral engine.
    pub fn new_2e(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite {
                (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i])
            } else {
                (1.0, op.kind, op.omega)
            };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented in v1", kind))),
            };
            let h = unsafe { ffi::scf_engine_create(op_kind, omega, prep.max_nprim(), prep.max_l(), precision) };
            if h.is_null() { return Err(FerricError::Libint("engine_create returned null".into())); }
            handles.push((coeff, h));
        }
        
        Ok(Engine { handles, buf: vec![0.0; max_fn * max_fn * max_fn * max_fn], scratch: vec![0.0; max_fn * max_fn * max_fn * max_fn] })
    }

    /// Create a geminal (F12) two-electron engine from an STG `Operator`
    /// (built via `Operator::stg`). The Gaussian fit carried in the operator's
    /// composite arrays is passed as one `ContractedGaussianGeminal` to libint —
    /// a SINGLE engine, unlike `new_2e`'s sum-of-engines for linear composites.
    ///
    /// Returns an error if libint2 lacks the G12 integral class (the FFI returns
    /// null) or the operator is not a geminal kind.
    pub fn new_2e_geminal(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let op_kind = match op.kind {
            OperatorKind::Cgtg => ffi::OP_CGTG,
            OperatorKind::CgtgCoulomb => ffi::OP_CGTG_X_COULOMB,
            OperatorKind::Delcgtg2 => ffi::OP_DELCGTG2,
            other => return Err(FerricError::Libint(format!("not a geminal operator: {other:?}"))),
        };
        if !op.is_composite || op.num_components == 0 {
            return Err(FerricError::Libint("geminal operator has no Gaussian fit".into()));
        }
        let ng = op.num_components;
        let exps: Vec<f64> = op.c_omegas[..ng].to_vec();
        let coefs: Vec<f64> = op.c_coeffs[..ng].to_vec();
        let handle = unsafe {
            ffi::scf_engine_create_geminal(
                op_kind, ng as c_int, exps.as_ptr(), coefs.as_ptr(),
                prep.max_nprim(), prep.max_l(), precision,
            )
        };
        if handle.is_null() {
            return Err(FerricError::Libint(
                "geminal engine_create returned null (libint2 built without G12?)".into(),
            ));
        }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine {
            handles: vec![(1.0, handle)],
            buf: vec![0.0; max_fn * max_fn * max_fn * max_fn],
            scratch: vec![0.0; max_fn * max_fn * max_fn * max_fn],
        })
    }

    /// Create a one-electron integral engine (overlap, kinetic, or nuclear).
    pub fn new_1e(op_kind: c_int, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let handle = unsafe { ffi::scf_engine_create(op_kind, 0.0, prep.max_nprim(), prep.max_l(), precision) };
        if handle.is_null() { return Err(FerricError::Libint("engine_create returned null".into())); }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handles: vec![(1.0, handle)], buf: vec![0.0; max_fn * max_fn], scratch: Vec::new() })
    }

    /// Mutable pointer to the underlying libint2 engine handle. (Returns the first component).
    pub fn handle_mut(&mut self) -> *mut c_void { self.handles[0].1 }

    /// Set nuclear point charges for the nuclear attraction operator.
    pub fn set_point_charges(&mut self, prep: &PreparedBasis) {
        for &(_, h) in &self.handles {
            unsafe { ffi::scf_engine_set_point_charges(h, prep.atoms().as_ptr(), prep.atoms().len() as c_int); }
        }
    }

    /// Compute a shell quartet of 4-center ERIs. Returns `None` if screened to zero.
    pub fn compute_quartet(
        &mut self, prep: &PreparedBasis, sh1: usize, sh2: usize, sh3: usize, sh4: usize,
    ) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2] * prep.shell_dims()[sh3] * prep.shell_dims()[sh4];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        if self.scratch.len() < n { self.scratch.resize(n, 0.0); }
        
        let mut max_written = 0;
        self.buf[..n].fill(0.0);

        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri_quartet(h, prep.handle(), sh1 as c_int, sh2 as c_int, sh3 as c_int, sh4 as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri quartet ({sh1},{sh2},{sh3},{sh4}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        
        if max_written == 0 { None } else { Some(&self.buf[..max_written]) }
    }

    /// Compute a shell pair block of one-electron integrals.
    pub fn compute_1e_block(&mut self, prep: &PreparedBasis, sh1: usize, sh2: usize) -> &[f64] {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        let written = unsafe { ffi::scf_compute_1e_block(self.handles[0].1, prep.handle(), sh1 as c_int, sh2 as c_int, self.buf.as_mut_ptr()) };
        assert!(written >= 0, "libint2 internal error in 1e block ({sh1},{sh2}): status {written}");
        &self.buf[..written as usize]
    }

    /// Create a first-derivative one-electron integral engine.
    pub fn new_1e_deriv(op_kind: c_int, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let handle = unsafe { ffi::scf_engine_create_deriv(op_kind, 0.0, prep.max_nprim(), prep.max_l(), precision) };
        if handle.is_null() { return Err(FerricError::Libint("derivative engine not available".into())); }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handles: vec![(1.0, handle)], buf: vec![0.0; 6 * max_fn * max_fn], scratch: Vec::new() })
    }

    /// Create a first-derivative 4-center two-electron integral engine.
    pub fn new_2e_deriv(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite { (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i]) } else { (1.0, op.kind, op.omega) };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", kind))),
            };
            let h = unsafe { ffi::scf_engine_create_deriv(op_kind, omega, prep.max_nprim(), prep.max_l(), precision) };
            if h.is_null() { return Err(FerricError::Libint("derivative engine not available".into())); }
            handles.push((coeff, h));
        }
        Ok(Engine { handles, buf: vec![0.0; 12 * max_fn * max_fn * max_fn * max_fn], scratch: vec![0.0; 12 * max_fn * max_fn * max_fn * max_fn] })
    }

    /// Returns the derivative blocks of n1*n2 doubles each: 6 blocks
    /// [dx1, dy1, dz1, dx2, dy2, dz2] for overlap/kinetic engines, and
    /// 3*(2 + natoms) blocks for a nuclear engine with point charges set
    /// (the extra blocks are the operator-center derivatives).
    /// Returns None if all derivatives were screened to zero.
    pub fn compute_1e_deriv_block(&mut self, prep: &PreparedBasis, sh1: usize, sh2: usize) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2];
        // Worst case is the nuclear operator: 3*(2 + natoms) blocks. Sizing for
        // it unconditionally keeps the shim's write (nderiv * n doubles) in
        // bounds for every 1e engine kind.
        let total = 3 * (2 + prep.atoms().len()) * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        let written = unsafe { ffi::scf_compute_1e_deriv_block(self.handles[0].1, prep.handle(), sh1 as c_int, sh2 as c_int, self.buf.as_mut_ptr()) };
        assert!(written >= 0, "libint2 internal error in 1e deriv block ({sh1},{sh2}): status {written}");
        if written == 0 { None } else { Some(&self.buf[..written as usize]) }
    }

    /// Create a 3-center integral engine for density fitting: (P|mu nu).
    pub fn new_3center(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_nprim = obs.max_nprim().max(dfbs.max_nprim());
        let max_l = obs.max_l().max(dfbs.max_l());
        let max_fn_obs = obs.shell_dims().iter().copied().max().unwrap_or(1);
        let max_fn_df = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite { (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i]) } else { (1.0, op.kind, op.omega) };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", kind))),
            };
            let h = unsafe { ffi::scf_engine_create_3center(op_kind, omega, max_nprim, max_l, precision) };
            if h.is_null() { return Err(FerricError::Libint("3-center engine not available".into())); }
            handles.push((coeff, h));
        }
        Ok(Engine { handles, buf: vec![0.0; max_fn_df * max_fn_obs * max_fn_obs], scratch: vec![0.0; max_fn_df * max_fn_obs * max_fn_obs] })
    }

    /// Create a 2-center integral engine for the Coulomb metric: (P|Q).
    pub fn new_2center(op: Operator, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_fn = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite { (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i]) } else { (1.0, op.kind, op.omega) };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", kind))),
            };
            let h = unsafe { ffi::scf_engine_create_2center(op_kind, omega, dfbs.max_nprim(), dfbs.max_l(), precision) };
            if h.is_null() { return Err(FerricError::Libint("2-center engine not available".into())); }
            handles.push((coeff, h));
        }
        Ok(Engine { handles, buf: vec![0.0; max_fn * max_fn], scratch: vec![0.0; max_fn * max_fn] })
    }

    /// Compute a 3-center ERI shell triplet (P|mu nu). Returns `None` if screened.
    pub fn compute_eri3(&mut self, obs: &PreparedBasis, dfbs: &PreparedBasis, sh_p: usize, sh1: usize, sh2: usize) -> Option<&[f64]> {
        let n = dfbs.shell_dims()[sh_p] * obs.shell_dims()[sh1] * obs.shell_dims()[sh2];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        if self.scratch.len() < n { self.scratch.resize(n, 0.0); }
        
        let mut max_written = 0;
        self.buf[..n].fill(0.0);
        
        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri3(h, obs.handle(), dfbs.handle(), sh_p as c_int, sh1 as c_int, sh2 as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri3 ({sh_p}|{sh1},{sh2}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        if max_written == 0 { None } else { Some(&self.buf[..max_written]) }
    }

    /// Compute a 2-center ERI shell pair (P|Q).
    pub fn compute_eri2(&mut self, dfbs: &PreparedBasis, sh_p: usize, sh_q: usize) -> &[f64] {
        let n = dfbs.shell_dims()[sh_p] * dfbs.shell_dims()[sh_q];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        if self.scratch.len() < n { self.scratch.resize(n, 0.0); }
        
        let mut max_written = 0;
        self.buf[..n].fill(0.0);
        
        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri2(h, dfbs.handle(), sh_p as c_int, sh_q as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri2 ({sh_p}|{sh_q}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        &self.buf[..max_written]
    }

    /// Returns 12 blocks of n1*n2*n3*n4 doubles: [dx1..dz1, dx2..dz2, dx3..dz3, dx4..dz4].
    /// Returns None if all derivatives were screened to zero.
    pub fn compute_eri_deriv_quartet(
        &mut self, prep: &PreparedBasis, sh1: usize, sh2: usize, sh3: usize, sh4: usize,
    ) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2] * prep.shell_dims()[sh3] * prep.shell_dims()[sh4];
        let total = 12 * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        if self.scratch.len() < total { self.scratch.resize(total, 0.0); }
        
        let mut max_written = 0;
        self.buf[..total].fill(0.0);
        
        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri_deriv_quartet(h, prep.handle(), sh1 as c_int, sh2 as c_int, sh3 as c_int, sh4 as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri deriv quartet ({sh1},{sh2},{sh3},{sh4}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        if max_written == 0 { None } else { Some(&self.buf[..max_written]) }
    }

    /// Create a 3-center derivative engine: d(P|mu nu)/dR.
    pub fn new_3center_deriv(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_nprim = obs.max_nprim().max(dfbs.max_nprim());
        let max_l = obs.max_l().max(dfbs.max_l());
        let max_fn_obs = obs.shell_dims().iter().copied().max().unwrap_or(1);
        let max_fn_df = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite { (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i]) } else { (1.0, op.kind, op.omega) };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", kind))),
            };
            let h = unsafe { ffi::scf_engine_create_3center_deriv(op_kind, omega, max_nprim, max_l, precision) };
            if h.is_null() { return Err(FerricError::Libint("3-center derivative engine not available".into())); }
            handles.push((coeff, h));
        }
        Ok(Engine { handles, buf: vec![0.0; 9 * max_fn_df * max_fn_obs * max_fn_obs], scratch: vec![0.0; 9 * max_fn_df * max_fn_obs * max_fn_obs] })
    }

    /// Create a 2-center derivative engine: d(P|Q)/dR.
    pub fn new_2center_deriv(op: Operator, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let max_fn = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        let mut handles = Vec::new();
        
        let n_comp = if op.is_composite { op.num_components } else { 1 };
        for i in 0..n_comp {
            let (coeff, kind, omega) = if op.is_composite { (op.c_coeffs[i], op.c_kinds[i], op.c_omegas[i]) } else { (1.0, op.kind, op.omega) };
            let op_kind = match kind {
                OperatorKind::Coulomb => ffi::OP_COULOMB,
                OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
                OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
                _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", kind))),
            };
            let h = unsafe { ffi::scf_engine_create_2center_deriv(op_kind, omega, dfbs.max_nprim(), dfbs.max_l(), precision) };
            if h.is_null() { return Err(FerricError::Libint("2-center derivative engine not available".into())); }
            handles.push((coeff, h));
        }
        Ok(Engine { handles, buf: vec![0.0; 6 * max_fn * max_fn], scratch: vec![0.0; 6 * max_fn * max_fn] })
    }

    /// Compute 3-center ERI derivatives: 9 blocks (3 centers × 3 coords) of nP*n1*n2.
    pub fn compute_eri3_deriv(&mut self, obs: &PreparedBasis, dfbs: &PreparedBasis, sh_p: usize, sh1: usize, sh2: usize) -> Option<&[f64]> {
        let n = dfbs.shell_dims()[sh_p] * obs.shell_dims()[sh1] * obs.shell_dims()[sh2];
        let total = 9 * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        if self.scratch.len() < total { self.scratch.resize(total, 0.0); }
        
        let mut max_written = 0;
        self.buf[..total].fill(0.0);
        
        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri3_deriv(h, obs.handle(), dfbs.handle(), sh_p as c_int, sh1 as c_int, sh2 as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri3 deriv ({sh_p}|{sh1},{sh2}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        if max_written == 0 { None } else { Some(&self.buf[..max_written]) }
    }

    /// Compute 2-center ERI derivatives: 6 blocks (2 centers × 3 coords) of nP*nQ.
    pub fn compute_eri2_deriv(&mut self, dfbs: &PreparedBasis, sh_p: usize, sh_q: usize) -> Option<&[f64]> {
        let n = dfbs.shell_dims()[sh_p] * dfbs.shell_dims()[sh_q];
        let total = 6 * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        if self.scratch.len() < total { self.scratch.resize(total, 0.0); }
        
        let mut max_written = 0;
        self.buf[..total].fill(0.0);
        
        for &(coeff, h) in &self.handles {
            let written = unsafe { ffi::scf_compute_eri2_deriv(h, dfbs.handle(), sh_p as c_int, sh_q as c_int, self.scratch.as_mut_ptr()) };
            assert!(written >= 0, "libint2 internal error in eri2 deriv ({sh_p}|{sh_q}): status {written}");
            if written > 0 {
                let w = written as usize;
                max_written = max_written.max(w);
                for i in 0..w { self.buf[i] += coeff * self.scratch[i]; }
            }
        }
        if max_written == 0 { None } else { Some(&self.buf[..max_written]) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        for &(_, h) in &self.handles {
            if !h.is_null() {
                unsafe { ffi::scf_engine_destroy(h) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn h2_sto3g() -> (Molecule, PreparedBasis) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        (mol, prep)
    }

    #[test]
    fn test_overlap_diagonal() {
        let (_, prep) = h2_sto3g();
        let mut eng = Engine::new_1e(ffi::OP_OVERLAP, &prep, 1e-14).unwrap();
        let block = eng.compute_1e_block(&prep, 0, 0);
        assert_eq!(block.len(), 1);
        assert!((block[0] - 1.0).abs() < 1e-10, "S[0,0] = {}", block[0]);
    }

    #[test]
    fn test_overlap_offdiag() {
        let (_, prep) = h2_sto3g();
        let mut eng = Engine::new_1e(ffi::OP_OVERLAP, &prep, 1e-14).unwrap();
        let block = eng.compute_1e_block(&prep, 0, 1);
        assert_eq!(block.len(), 1);
        assert!(
            (block[0] - 0.6594).abs() < 1e-2,
            "S[0,1] = {}, expected ~0.6594",
            block[0]
        );
    }

    #[test]
    fn test_eri_quartet() {
        let (_, prep) = h2_sto3g();
        let mut eng = Engine::new_2e(Operator::coulomb(), &prep, 1e-14).unwrap();
        let q = eng.compute_quartet(&prep, 0, 0, 0, 0);
        assert!(q.is_some());
        let v = q.unwrap()[0];
        assert!((v - 0.7746).abs() < 1e-3, "(00|00) = {v}, expected ~0.7746");
    }

    #[test]
    fn test_erfc_coulomb_quartet() {
        let (_, prep) = h2_sto3g();
        let op = Operator::erfc(0.5);
        let mut eng = Engine::new_2e(op, &prep, 1e-14).unwrap();
        let q = eng.compute_quartet(&prep, 0, 0, 0, 0);
        assert!(q.is_some(), "ErfcCoulomb should produce non-zero integrals");
        let v = q.unwrap()[0];
        assert!(v > 0.0 && v < 0.7746, "erfc (00|00) = {v}, should be between 0 and full Coulomb 0.7746");
    }

    #[test]
    fn test_erf_plus_erfc_equals_coulomb() {
        let (_, prep) = h2_sto3g();
        let omega = 0.5;
        let mut eng_full = Engine::new_2e(Operator::coulomb(), &prep, 1e-14).unwrap();
        let mut eng_erf = Engine::new_2e(Operator::erf(omega), &prep, 1e-14).unwrap();
        let op_erfc = Operator::erfc(omega);
        let mut eng_erfc = Engine::new_2e(op_erfc, &prep, 1e-14).unwrap();

        let v_full = eng_full.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        let v_erf = eng_erf.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        let v_erfc = eng_erfc.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];

        assert!((v_full - v_erf - v_erfc).abs() < 1e-10,
            "erf + erfc should equal Coulomb: {} + {} = {} vs {}", v_erf, v_erfc, v_erf + v_erfc, v_full);
    }

    #[test]
    fn test_geminal_cgtg_quartet_runs() {
        // The honest proof that libint2 was built WITH the G12 integral class:
        // construct each geminal engine and compute a quartet. The Operator::cgtg
        // enum compiles even when G12 is stripped, so only a successful runtime
        // call (non-null engine + integral computed) proves the kernel is in.
        let (_, prep) = h2_sto3g();
        let gamma = 1.0; // STG exponent (a.u.); cc-pVDZ-F12 uses ~0.9–1.0.

        for kind in [
            OperatorKind::Cgtg,
            OperatorKind::CgtgCoulomb,
            OperatorKind::Delcgtg2,
        ] {
            let op = Operator::stg(gamma, kind);
            let eng = Engine::new_2e_geminal(op, &prep, 1e-14);
            assert!(
                eng.is_ok(),
                "geminal engine {kind:?} failed — libint2 built without G12? ({:?})",
                eng.err()
            );
            let mut eng = eng.unwrap();
            let q = eng.compute_quartet(&prep, 0, 0, 0, 0);
            assert!(q.is_some(), "geminal {kind:?} (00|00) screened/empty");
            let v = q.unwrap()[0];
            assert!(v.is_finite(), "geminal {kind:?} (00|00) = {v} not finite");
        }
    }

    #[test]
    fn test_geminal_cgtg_signs() {
        // f12 = -(1/gamma) exp(-gamma r12) is negative everywhere, so ⟨f12⟩ < 0.
        // f12/r12 is also negative. delcgtg2 = |∇f12|^2 ≥ 0.
        let (_, prep) = h2_sto3g();
        let gamma = 1.0;

        let mut e_f12 = Engine::new_2e_geminal(Operator::stg(gamma, OperatorKind::Cgtg), &prep, 1e-14).unwrap();
        let v_f12 = e_f12.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        assert!(v_f12 < 0.0, "⟨f12⟩ should be < 0 (geminal is -exp/gamma), got {v_f12}");

        let mut e_fc = Engine::new_2e_geminal(Operator::stg(gamma, OperatorKind::CgtgCoulomb), &prep, 1e-14).unwrap();
        let v_fc = e_fc.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        assert!(v_fc < 0.0, "⟨f12/r12⟩ should be < 0, got {v_fc}");

        let mut e_dc = Engine::new_2e_geminal(Operator::stg(gamma, OperatorKind::Delcgtg2), &prep, 1e-14).unwrap();
        let v_dc = e_dc.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        assert!(v_dc >= 0.0, "⟨|∇f12|^2⟩ should be ≥ 0, got {v_dc}");
    }

    #[test]
    fn test_composite_operator() {
        let (_, prep) = h2_sto3g();
        let omega = 0.5;
        // Construct composite: 1.0 * erf(omega) + 1.0 * erfc(omega)
        let op_composite = Operator::composite(&[
            (1.0, OperatorKind::ErfCoulomb, omega),
            (1.0, OperatorKind::ErfcCoulomb, omega),
        ]);
        let mut eng_comp = Engine::new_2e(op_composite, &prep, 1e-14).unwrap();
        let mut eng_full = Engine::new_2e(Operator::coulomb(), &prep, 1e-14).unwrap();

        let v_comp = eng_comp.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        let v_full = eng_full.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];

        assert!(
            (v_comp - v_full).abs() < 1e-10,
            "composite (erf + erfc) = {v_comp}, should equal full coulomb = {v_full}"
        );
    }

    #[test]
    fn test_terfc_spike() {
        let (_, prep) = h2_sto3g();
        let r0 = 1.05; // Standard variant I
        let op_terfc = Operator::terfc_fit(r0);
        let mut eng_terfc = Engine::new_2e(op_terfc, &prep, 1e-14).unwrap();
        let q = eng_terfc.compute_quartet(&prep, 0, 0, 0, 0);
        assert!(q.is_some(), "terfc_fit should produce non-zero integrals");
        
        let v = q.unwrap()[0];
        // The value should be bounded by the standard coulomb operator
        let op_coulomb = Operator::coulomb();
        let mut eng_coulomb = Engine::new_2e(op_coulomb, &prep, 1e-14).unwrap();
        let v_coulomb = eng_coulomb.compute_quartet(&prep, 0, 0, 0, 0).unwrap()[0];
        
        assert!(v > 0.0 && v < v_coulomb, "terfc integral {v} should be attenuated (0 < v < {v_coulomb})");
    }

    #[test]
    fn test_nuclear_deriv_block_safe_wrapper_sized_for_operator_centers() {
        // A nuclear deriv engine with point charges returns 3*(2+natoms) blocks,
        // not 6 — the safe wrapper must size its buffer for the operator-center
        // derivatives too (regression: it allocated 6*n and the shim wrote
        // 3*(2+natoms)*n, past the end of the Vec).
        let (_, prep) = h2_sto3g();
        let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, &prep, 1e-14).unwrap();
        eng.set_point_charges(&prep);
        let n = prep.shell_dims()[0] * prep.shell_dims()[1];
        let natoms = prep.atoms().len();
        let deriv = eng
            .compute_1e_deriv_block(&prep, 0, 1)
            .expect("nuclear deriv block screened unexpectedly");
        assert_eq!(deriv.len(), 3 * (2 + natoms) * n);
        // Translational invariance: shell-center + charge-center derivatives
        // sum to zero per coordinate.
        for coord in 0..3 {
            for idx in 0..n {
                let sum: f64 = (0..2 + natoms).map(|c| deriv[(3 * c + coord) * n + idx]).sum();
                assert!(sum.abs() < 1e-10, "coord={coord} idx={idx} sum={sum:.2e}");
            }
        }
    }

    #[test]
    fn test_eri3_deriv_translational_invariance() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let mut eng = Engine::new_3center_deriv(op, &obs, &dfbs, 1e-14).unwrap();

        let nsh_obs = obs.nshells();
        let nsh_aux = dfbs.nshells();
        for sh_p in 0..nsh_aux.min(3) {
            for sh1 in 0..nsh_obs {
                for sh2 in 0..nsh_obs {
                    if let Some(deriv) = eng.compute_eri3_deriv(&obs, &dfbs, sh_p, sh1, sh2) {
                        let np = dfbs.shell_dims()[sh_p];
                        let n1 = obs.shell_dims()[sh1];
                        let n2 = obs.shell_dims()[sh2];
                        let block_sz = np * n1 * n2;
                        for idx in 0..block_sz {
                            for coord in 0..3 {
                                let d0 = deriv[coord * block_sz + idx];
                                let d1 = deriv[(3 + coord) * block_sz + idx];
                                let d2 = deriv[(6 + coord) * block_sz + idx];
                                let sum = d0 + d1 + d2;
                                assert!(sum.abs() < 1e-10,
                                    "3c translational invariance: sh({},{},{}) coord={} idx={} sum={:.2e}",
                                    sh_p, sh1, sh2, coord, idx, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_eri2_deriv_translational_invariance() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let mut eng = Engine::new_2center_deriv(op, &dfbs, 1e-14).unwrap();

        let nsh = dfbs.nshells();
        for sh_p in 0..nsh.min(4) {
            for sh_q in 0..nsh.min(4) {
                if let Some(deriv) = eng.compute_eri2_deriv(&dfbs, sh_p, sh_q) {
                    let np = dfbs.shell_dims()[sh_p];
                    let nq = dfbs.shell_dims()[sh_q];
                    let block_sz = np * nq;
                    for idx in 0..block_sz {
                        for coord in 0..3 {
                            let d0 = deriv[coord * block_sz + idx];
                            let d1 = deriv[(3 + coord) * block_sz + idx];
                            let sum = d0 + d1;
                            assert!(sum.abs() < 1e-10,
                                "2c translational invariance: sh({},{}) coord={} idx={} sum={:.2e}",
                                sh_p, sh_q, coord, idx, sum);
                        }
                    }
                }
            }
        }
    }
}
