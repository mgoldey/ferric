use crate::basis_bridge::PreparedBasis;
use crate::ffi;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use std::os::raw::{c_int, c_void};

pub struct Engine {
    handle: *mut c_void,
    buf: Vec<f64>,
}

impl Engine {
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

    pub fn handle_mut(&mut self) -> *mut c_void { self.handle }

    pub fn set_point_charges(&mut self, prep: &PreparedBasis) {
        unsafe {
            ffi::goscf_engine_set_point_charges(
                self.handle,
                prep.atoms().as_ptr(),
                prep.atoms().len() as c_int,
            );
        }
    }

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
