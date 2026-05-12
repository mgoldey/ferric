use crate::basis_bridge::PreparedBasis;
use crate::ffi;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use ndarray::Array2;

pub fn schwarz(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let op_kind = match op.kind {
        OperatorKind::Coulomb => ffi::OP_COULOMB,
        OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
        _ => return Err(FerricError::Libint(format!(
            "operator {:?} not implemented", op.kind
        ))),
    };
    let handle = unsafe {
        ffi::goscf_engine_create(op_kind, op.omega, prep.max_nprim(), prep.max_l(), 1e-14)
    };
    if handle.is_null() {
        return Err(FerricError::Libint("schwarz engine_create null".into()));
    }
    let nsh = prep.nshells();
    let mut qmat = Array2::zeros((nsh, nsh));
    unsafe {
        ffi::goscf_compute_schwarz(handle, prep.handle(), qmat.as_mut_ptr());
        ffi::goscf_engine_destroy(handle);
    }
    Ok(qmat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_schwarz_symmetric() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q = schwarz(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(
                    (q[(i, j)] - q[(j, i)]).abs() < 1e-12,
                    "Q not symmetric at ({i},{j})"
                );
                assert!(q[(i, j)] >= 0.0, "Q[{i},{j}] < 0");
            }
        }
    }
}
