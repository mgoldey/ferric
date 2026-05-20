//! Schwarz upper-bound screening matrix for shell-pair integrals.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::ffi;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use ndarray::Array2;

/// Compute the Schwarz screening matrix Q(i,j) = sqrt(|(ij|ij)|) for all shell pairs.
///
/// Q(i,j) * Q(k,l) provides an upper bound on |(ij|kl)|, enabling integral screening.
pub fn schwarz(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let op_kind = match op.kind {
        OperatorKind::Coulomb => ffi::OP_COULOMB,
        OperatorKind::ErfCoulomb => ffi::OP_ERF_COULOMB,
        OperatorKind::ErfcCoulomb => ffi::OP_ERFC_COULOMB,
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

/// Per-shell Schwarz bound for an auxiliary (density-fitting) basis:
/// Q3[P] = sqrt(max_a |(P_a | P_a)|) over the functions a in aux shell P.
///
/// Combined with the orbital-pair matrix Q(μ,ν) = sqrt(|(μν|μν)|), this gives
/// the rigorous 3-index Cauchy–Schwarz bound
///   |(P | μν)|  ≤  Q3[P] · Q(μ,ν)
/// which lets `eri3_tensor_screened` skip shell triples whose contribution
/// is below threshold without computing them.
pub fn schwarz3_aux(op: Operator, dfbs: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
    let mut q3 = vec![0.0f64; nsh];
    for p in 0..nsh {
        let block = eng.compute_eri2(dfbs, p, p);
        let np = dims[p];
        // (P_a | P_a) lives on the diagonal of the np×np block.
        let mut maxv = 0.0f64;
        for a in 0..np {
            let v = block[a * np + a].abs();
            if v > maxv {
                maxv = v;
            }
        }
        q3[p] = maxv.sqrt();
    }
    Ok(q3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_schwarz_erfc_bounded_by_coulomb() {
        // erfc(ωr)/r ≤ 1/r pointwise, so |(ij|ij)_erfc| ≤ |(ij|ij)_Coulomb|
        // and therefore Q_erfc(i,j) ≤ Q_Coulomb(i,j) for every shell pair.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q_c = schwarz(Operator::coulomb(), &prep).unwrap();
        let q_e = schwarz(Operator::erfc(0.222), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(q_e[(i, j)] >= 0.0, "Q_erfc[{i},{j}] < 0");
                assert!(
                    q_e[(i, j)] <= q_c[(i, j)] + 1e-12,
                    "Q_erfc[{i},{j}]={} exceeds Q_Coulomb={}",
                    q_e[(i, j)], q_c[(i, j)]
                );
            }
        }
    }

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
