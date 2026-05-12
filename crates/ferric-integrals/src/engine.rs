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
    handle: *mut c_void,
    buf: Vec<f64>,
}

impl Engine {
    /// Create a 4-center two-electron integral engine.
    pub fn new_2e(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let op_kind = match op.kind {
            OperatorKind::Coulomb => ffi::OP_COULOMB,
            OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
            _ => return Err(FerricError::Libint(format!(
                "operator {:?} not implemented in v1", op.kind
            ))),
        };
        let handle = unsafe {
            ffi::goscf_engine_create(op_kind, op.omega, prep.max_nprim(), prep.max_l(), precision)
        };
        if handle.is_null() {
            return Err(FerricError::Libint("engine_create returned null".into()));
        }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; max_fn * max_fn * max_fn * max_fn] })
    }

    /// Create a one-electron integral engine (overlap, kinetic, or nuclear).
    pub fn new_1e(op_kind: c_int, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let handle = unsafe {
            ffi::goscf_engine_create(op_kind, 0.0, prep.max_nprim(), prep.max_l(), precision)
        };
        if handle.is_null() {
            return Err(FerricError::Libint("engine_create returned null".into()));
        }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; max_fn * max_fn] })
    }

    /// Mutable pointer to the underlying libint2 engine handle.
    pub fn handle_mut(&mut self) -> *mut c_void { self.handle }

    /// Set nuclear point charges for the nuclear attraction operator.
    pub fn set_point_charges(&mut self, prep: &PreparedBasis) {
        unsafe {
            ffi::goscf_engine_set_point_charges(
                self.handle,
                prep.atoms().as_ptr(),
                prep.atoms().len() as c_int,
            );
        }
    }

    /// Compute a shell quartet of 4-center ERIs. Returns `None` if screened to zero.
    pub fn compute_quartet(
        &mut self,
        prep: &PreparedBasis,
        sh1: usize,
        sh2: usize,
        sh3: usize,
        sh4: usize,
    ) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1]
            * prep.shell_dims()[sh2]
            * prep.shell_dims()[sh3]
            * prep.shell_dims()[sh4];
        if self.buf.len() < n {
            self.buf.resize(n, 0.0);
        }
        let written = unsafe {
            ffi::goscf_compute_eri_quartet(
                self.handle,
                prep.handle(),
                sh1 as c_int,
                sh2 as c_int,
                sh3 as c_int,
                sh4 as c_int,
                self.buf.as_mut_ptr(),
            )
        };
        if written == 0 {
            None
        } else {
            Some(&self.buf[..written as usize])
        }
    }

    /// Compute a shell pair block of one-electron integrals.
    pub fn compute_1e_block(
        &mut self,
        prep: &PreparedBasis,
        sh1: usize,
        sh2: usize,
    ) -> &[f64] {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2];
        if self.buf.len() < n {
            self.buf.resize(n, 0.0);
        }
        let written = unsafe {
            ffi::goscf_compute_1e_block(
                self.handle,
                prep.handle(),
                sh1 as c_int,
                sh2 as c_int,
                self.buf.as_mut_ptr(),
            )
        };
        &self.buf[..written as usize]
    }

    /// Create a first-derivative one-electron integral engine.
    pub fn new_1e_deriv(op_kind: c_int, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let handle = unsafe {
            ffi::goscf_engine_create_deriv(op_kind, 0.0, prep.max_nprim(), prep.max_l(), precision)
        };
        if handle.is_null() {
            return Err(FerricError::Libint("derivative engine not available (libint2 built without derivative support?)".into()));
        }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; 6 * max_fn * max_fn] })
    }

    /// Create a first-derivative 4-center two-electron integral engine.
    pub fn new_2e_deriv(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let op_kind = match op.kind {
            OperatorKind::Coulomb => ffi::OP_COULOMB,
            OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
            _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", op.kind))),
        };
        let handle = unsafe {
            ffi::goscf_engine_create_deriv(op_kind, op.omega, prep.max_nprim(), prep.max_l(), precision)
        };
        if handle.is_null() {
            return Err(FerricError::Libint("derivative engine not available".into()));
        }
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; 12 * max_fn * max_fn * max_fn * max_fn] })
    }

    /// Returns 6 blocks of n1*n2 doubles: [dx1, dy1, dz1, dx2, dy2, dz2].
    /// Returns None if all derivatives were screened to zero.
    pub fn compute_1e_deriv_block(
        &mut self, prep: &PreparedBasis, sh1: usize, sh2: usize,
    ) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2];
        let total = 6 * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        let written = unsafe {
            ffi::goscf_compute_1e_deriv_block(
                self.handle, prep.handle(), sh1 as c_int, sh2 as c_int, self.buf.as_mut_ptr(),
            )
        };
        if written == 0 { None } else { Some(&self.buf[..written as usize]) }
    }

    /// Create a 3-center integral engine for density fitting: (P|mu nu).
    pub fn new_3center(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let op_kind = match op.kind {
            OperatorKind::Coulomb => ffi::OP_COULOMB,
            OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
            _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", op.kind))),
        };
        let max_nprim = obs.max_nprim().max(dfbs.max_nprim());
        let max_l = obs.max_l().max(dfbs.max_l());
        let handle = unsafe { ffi::goscf_engine_create_3center(op_kind, op.omega, max_nprim, max_l, precision) };
        if handle.is_null() { return Err(FerricError::Libint("3-center engine not available".into())); }
        let max_fn_obs = obs.shell_dims().iter().copied().max().unwrap_or(1);
        let max_fn_df = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; max_fn_df * max_fn_obs * max_fn_obs] })
    }

    /// Create a 2-center integral engine for the Coulomb metric: (P|Q).
    pub fn new_2center(op: Operator, dfbs: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let op_kind = match op.kind {
            OperatorKind::Coulomb => ffi::OP_COULOMB,
            OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
            _ => return Err(FerricError::Libint(format!("operator {:?} not implemented", op.kind))),
        };
        let handle = unsafe { ffi::goscf_engine_create_2center(op_kind, op.omega, dfbs.max_nprim(), dfbs.max_l(), precision) };
        if handle.is_null() { return Err(FerricError::Libint("2-center engine not available".into())); }
        let max_fn = dfbs.shell_dims().iter().copied().max().unwrap_or(1);
        Ok(Engine { handle, buf: vec![0.0; max_fn * max_fn] })
    }

    /// Compute a 3-center ERI shell triplet (P|mu nu). Returns `None` if screened.
    pub fn compute_eri3(&mut self, obs: &PreparedBasis, dfbs: &PreparedBasis,
                        sh_p: usize, sh1: usize, sh2: usize) -> Option<&[f64]> {
        let n = dfbs.shell_dims()[sh_p] * obs.shell_dims()[sh1] * obs.shell_dims()[sh2];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        let written = unsafe {
            ffi::goscf_compute_eri3(self.handle, obs.handle(), dfbs.handle(),
                                    sh_p as c_int, sh1 as c_int, sh2 as c_int, self.buf.as_mut_ptr())
        };
        if written == 0 { None } else { Some(&self.buf[..written as usize]) }
    }

    /// Compute a 2-center ERI shell pair (P|Q).
    pub fn compute_eri2(&mut self, dfbs: &PreparedBasis, sh_p: usize, sh_q: usize) -> &[f64] {
        let n = dfbs.shell_dims()[sh_p] * dfbs.shell_dims()[sh_q];
        if self.buf.len() < n { self.buf.resize(n, 0.0); }
        let written = unsafe {
            ffi::goscf_compute_eri2(self.handle, dfbs.handle(), sh_p as c_int, sh_q as c_int, self.buf.as_mut_ptr())
        };
        &self.buf[..written as usize]
    }

    /// Returns 12 blocks of n1*n2*n3*n4 doubles: [dx1..dz1, dx2..dz2, dx3..dz3, dx4..dz4].
    /// Returns None if all derivatives were screened to zero.
    pub fn compute_eri_deriv_quartet(
        &mut self, prep: &PreparedBasis,
        sh1: usize, sh2: usize, sh3: usize, sh4: usize,
    ) -> Option<&[f64]> {
        let n = prep.shell_dims()[sh1] * prep.shell_dims()[sh2]
            * prep.shell_dims()[sh3] * prep.shell_dims()[sh4];
        let total = 12 * n;
        if self.buf.len() < total { self.buf.resize(total, 0.0); }
        let written = unsafe {
            ffi::goscf_compute_eri_deriv_quartet(
                self.handle, prep.handle(),
                sh1 as c_int, sh2 as c_int, sh3 as c_int, sh4 as c_int,
                self.buf.as_mut_ptr(),
            )
        };
        if written == 0 { None } else { Some(&self.buf[..written as usize]) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::goscf_engine_destroy(self.handle) };
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
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
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
}
