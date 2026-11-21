//! omega-RI: density fitting in the erfc-attenuated Coulomb metric.
//!
//! The one fit module shared by every omega-RI method in ferric: canonical
//! omega-RI-MP2 ([`crate::rimp2::ri_mp2_fitted`]), omega-RI-CDD-MP2 and its
//! SOS variant ([`crate::omega_ri_cdd`]), and omega-RI RPA
//! (`ferric_rpa::omega_ri`). Each of them only changes how the dressed
//! 3-index tensor `B` is built; the energy code downstream is unchanged.
//!
//! # The fit
//!
//! Glasbrenner, Graf, Ochsenfeld, JCTC 16, 6856 (2020), eqs 15-18, and
//! Luenser, Schurkus, Ochsenfeld, JCTC 13, 1647 (2017), eq 29:
//!
//! ```text
//! (mn|ls) ~ sum_PQ (mn|P)_w  C~_PQ  (Q|ls)_w,     C~ = V~^-1 V V~^-1
//! (mn|P)_w = (mn| erfc(w r12)/r12 |P),  V~_PQ = (P|erfc(w r12)/r12|Q),  V_PQ = (P|1/r12|Q)
//! ```
//!
//! Each pair density is fitted in the erfc metric and the two fitted densities
//! then interact through the full Coulomb operator. With `V = L Lᵀ`,
//! `C~ = W Wᵀ` for `W = V~^-1 L`, so the dressing `D = Wᵀ = Lᵀ V~^-1` gives
//! `B = D (P|ia)_w` and `(ia|jb) ~ B_iaᵀ B_jb`, a Gram product: positive
//! semidefinite, so MP2 and RPA stay bounded. At `w = 0`, `V~ = V` and
//! `D = Lᵀ V^-1 = L^-1`, the Coulomb-metric RI factor ferric already uses.
//!
//! The form is NOT Dunlap-robust: the erfc fit makes `(d|P)_w = 0` for the fit
//! residual `d`, not `(d|P) = 0`, so the energy error is FIRST order in the
//! residual (book section 06-density-fitting/04-omega-ri-fit, proved in
//! `scripts/omega_ri_sympy.py`). `w` is therefore an accuracy threshold, not a
//! free parameter: lowering it converges smoothly to the Coulomb-metric result.
//!
//! # Units
//!
//! `omega` is in Bohr⁻¹ everywhere in this module and on every surface that
//! reaches it (CLI `[ri] omega_bohr`, Python `fit_omega_bohr=`). The papers
//! work in atomic units; their recommended value is 0.1.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_integrals::threeindex;
use ndarray::Array2;
use ndarray_linalg::{Cholesky, UPLO};

/// Which metric the RI fit uses.
///
/// `Coulomb` is the standard fit: the fitting metric is the physical operator
/// itself, and every existing ferric path is this variant.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RiFit {
    /// Fit in the metric of the physical operator (standard RI).
    #[default]
    Coulomb,
    /// Fit in the erfc-attenuated Coulomb metric, `omega` in Bohr⁻¹ (`>= 0`).
    /// Valid only when the physical operator is the bare Coulomb operator.
    OmegaRi {
        /// Attenuation parameter of erfc(ω r)/r, Bohr⁻¹.
        omega: f64,
    },
}

impl RiFit {
    /// The omega-RI fit, validated: `omega` must be finite and `>= 0`.
    /// `omega = 0` is the Coulomb metric (the exactness anchor).
    pub fn omega_ri(omega: f64) -> Result<Self, FerricError> {
        if !(omega.is_finite() && omega >= 0.0) {
            return Err(FerricError::General(format!(
                "omega-RI fit: omega must be finite and >= 0 Bohr^-1 (got {omega})"
            )));
        }
        Ok(Self::OmegaRi { omega })
    }

    /// Parse the user-facing pair (`metric`, `omega_bohr`), STRICTLY.
    ///
    /// `metric`: `None`/`"coulomb"` or `"erfc"`. `"erfc"` requires
    /// `omega_bohr > 0` (the papers' recommendation is 0.1; there is no
    /// silent default, because the right value is an accuracy choice). An
    /// `omega_bohr` with the Coulomb metric is an error, not an ignored knob.
    pub fn parse_config(metric: Option<&str>, omega_bohr: Option<f64>) -> Result<Self, FerricError> {
        match metric {
            None | Some("coulomb") => {
                if let Some(w) = omega_bohr {
                    return Err(FerricError::General(format!(
                        "fit omega_bohr = {w} is only meaningful with metric = \"erfc\"; \
                         set metric = \"erfc\" or drop omega_bohr"
                    )));
                }
                Ok(Self::Coulomb)
            }
            Some("erfc") => {
                let w = omega_bohr.ok_or_else(|| {
                    FerricError::General(
                        "fit metric = \"erfc\" requires omega_bohr (Bohr^-1, > 0; the \
                         omega-RI papers recommend 0.1). There is no default: omega is \
                         an accuracy threshold."
                            .into(),
                    )
                })?;
                if !(w.is_finite() && w > 0.0) {
                    return Err(FerricError::General(format!(
                        "fit omega_bohr must be finite and > 0 (got {w}); omega = 0 is \
                         the Coulomb metric, spell it metric = \"coulomb\""
                    )));
                }
                Ok(Self::OmegaRi { omega: w })
            }
            Some(other) => Err(FerricError::General(format!(
                "unknown fit metric \"{other}\"; expected \"coulomb\" (default) or \"erfc\" \
                 (omega-RI, needs omega_bohr)"
            ))),
        }
    }

    /// `true` for the standard fit.
    pub fn is_coulomb(&self) -> bool {
        matches!(self, Self::Coulomb)
    }

    /// The attenuation parameter, `None` for the Coulomb fit.
    pub fn omega(&self) -> Option<f64> {
        match self {
            Self::Coulomb => None,
            Self::OmegaRi { omega } => Some(*omega),
        }
    }

    /// One-line provenance label, e.g. `"coulomb"` or `"erfc(omega=0.1 Bohr^-1)"`.
    pub fn label(&self) -> String {
        match self {
            Self::Coulomb => "coulomb".into(),
            Self::OmegaRi { omega } => format!("erfc(omega={omega} Bohr^-1)"),
        }
    }

    /// The operator of the 3-index integrals `(P|mn)` this fit consumes.
    ///
    /// For the omega-RI fit the physical operator must be the bare Coulomb
    /// operator (the papers' setting; an attenuated PHYSICAL operator would
    /// need its own derivation), and the 3-index operator is erfc(ω r)/r. At
    /// `ω = 0` that operator IS 1/r, and the Coulomb operator is returned so
    /// the anchor runs through the plain Coulomb integrals.
    pub fn three_index_operator(&self, phys: Operator) -> Result<Operator, FerricError> {
        match self {
            Self::Coulomb => Ok(phys),
            Self::OmegaRi { omega } => {
                if phys.kind != OperatorKind::Coulomb || phys.is_composite() {
                    return Err(FerricError::General(format!(
                        "omega-RI fit needs the bare Coulomb physical operator (got {:?}); \
                         an attenuated or composite physical operator is not covered by the \
                         derivation",
                        phys.kind
                    )));
                }
                Ok(if *omega == 0.0 {
                    Operator::coulomb()
                } else {
                    Operator::erfc(*omega)
                })
            }
        }
    }
}

/// Bytes [`fit_dressing`] holds at its peak for an omega-RI fit: `V`, `V~`,
/// the Cholesky factor of `V~`, and the solve's output, all `naux x naux`.
pub fn omega_ri_dressing_peak_bytes(naux: usize) -> usize {
    naux.saturating_mul(naux).saturating_mul(8).saturating_mul(5)
}

/// The 3-index operator and the dressing `D` for a fit, so that the dressed
/// tensor `B = D (P|pq)_op` satisfies `(pq|rs) ~ B_pqᵀ B_rs`.
///
/// * [`RiFit::Coulomb`]: `(phys, metric_inverse_sqrt(V_phys))`, exactly the
///   factor every existing RI path builds.
/// * [`RiFit::OmegaRi`]: `(erfc(ω), Lᵀ V~^-1)` with `V = L Lᵀ` the Coulomb
///   metric. Built by Cholesky solves, never an explicit inverse (the metric is
///   ill-conditioned; see `rimp2::solve_metric_system`). A Cholesky failure of
///   `V~` is reported as an error: the aux basis has lost rank in that metric.
///
/// `budget_bytes` gates the omega-RI path's `naux²` planes before they are
/// allocated.
pub fn fit_dressing(
    fit: &RiFit,
    phys: Operator,
    dfbs: &PreparedBasis,
    budget_bytes: usize,
) -> Result<(Operator, Array2<f64>), FerricError> {
    let op3 = fit.three_index_operator(phys)?;
    match fit {
        RiFit::Coulomb => {
            let v = threeindex::coulomb_metric_2c(phys, dfbs)?;
            Ok((op3, crate::rimp2::metric_inverse_sqrt(&v, phys)?))
        }
        RiFit::OmegaRi { .. } => {
            let naux = dfbs.nbasis();
            ferric_core::memory::check_alloc(
                &format!("omega-RI fit dressing (naux={naux}, five naux^2 planes)"),
                omega_ri_dressing_peak_bytes(naux),
                budget_bytes,
            )?;
            let v = threeindex::coulomb_metric_2c(Operator::coulomb(), dfbs)?;
            let v_w = threeindex::coulomb_metric_2c(op3, dfbs)?;
            Ok((op3, omega_ri_dressing_from_metrics(&v, &v_w)?))
        }
    }
}

/// `D = Lᵀ V~^-1` from the two metrics (`V = L Lᵀ`), so `Dᵀ D = V~^-1 V V~^-1`.
///
/// Split out so the derivation script's numerical cross-check and the unit
/// tests can exercise the algebra on matrices they built themselves.
pub fn omega_ri_dressing_from_metrics(
    v: &Array2<f64>,
    v_w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let l = with_blas_threads(opt_in_blas_threads(), || v.cholesky(UPLO::Lower)).map_err(|e| {
        FerricError::Lapack(format!(
            "omega-RI fit: Cholesky of the Coulomb metric V failed (aux basis is \
             numerically rank deficient): {e}"
        ))
    })?;
    // W = V~^-1 L  (naux x naux), then D = Wᵀ.
    let w = crate::rimp2::solve_metric_system(v_w, &l)?;
    Ok(w.t().as_standard_layout().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn spd(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let mut a = Array2::<f64>::zeros((n, n));
        for v in a.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        let mut m = a.t().dot(&a);
        for i in 0..n {
            m[(i, i)] += n as f64;
        }
        m
    }

    /// `Dᵀ D = V~^-1 V V~^-1` for an arbitrary SPD pair (the identity the fit
    /// rests on), and `V~ = V` gives `Dᵀ D = V^-1` (the omega = 0 limit).
    #[test]
    fn dressing_gram_is_the_omega_ri_metric_and_reduces_to_v_inverse() {
        use ndarray_linalg::Inverse;
        let v = spd(7, 1);
        let vw = spd(7, 2);
        let d = omega_ri_dressing_from_metrics(&v, &vw).unwrap();
        let vwi = vw.inv().unwrap();
        let want = vwi.dot(&v).dot(&vwi);
        let got = d.t().dot(&d);
        let err = (&got - &want).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(err < 1e-12, "D^T D != V~^-1 V V~^-1: {err:e}");

        let d0 = omega_ri_dressing_from_metrics(&v, &v).unwrap();
        let vi = v.inv().unwrap();
        let err0 = (&d0.t().dot(&d0) - &vi)
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(err0 < 1e-12, "omega=0 limit: D^T D != V^-1: {err0:e}");
    }

    #[test]
    fn parse_config_is_strict() {
        assert_eq!(RiFit::parse_config(None, None).unwrap(), RiFit::Coulomb);
        assert_eq!(
            RiFit::parse_config(Some("erfc"), Some(0.1)).unwrap(),
            RiFit::OmegaRi { omega: 0.1 }
        );
        assert!(RiFit::parse_config(Some("erfc"), None).is_err());
        assert!(RiFit::parse_config(Some("erfc"), Some(0.0)).is_err());
        assert!(RiFit::parse_config(Some("erfc"), Some(f64::NAN)).is_err());
        assert!(RiFit::parse_config(Some("coulomb"), Some(0.1)).is_err());
        assert!(RiFit::parse_config(Some("overlap"), None).is_err());
        assert!(RiFit::omega_ri(-1.0).is_err());
        assert!(RiFit::omega_ri(0.0).is_ok());
    }

    #[test]
    fn omega_ri_refuses_a_non_coulomb_physical_operator() {
        let fit = RiFit::OmegaRi { omega: 0.1 };
        assert!(fit.three_index_operator(Operator::erfc(0.3)).is_err());
        assert_eq!(
            fit.three_index_operator(Operator::coulomb()).unwrap().kind,
            OperatorKind::ErfcCoulomb
        );
    }
}
