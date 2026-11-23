//! Coulomb and exchange matrices of arbitrary symmetric densities, and the
//! explicit derivative of the two-electron energy, on the route a
//! Hartree-Fock reference (RHF, or UHF for the open-shell derivative) took.
//!
//! A correlated gradient on top of an SCF (the RI-MP2 Z-vector and
//! relaxed-density Lagrangian) needs `J[D]` and `K[D]` of densities that are
//! not the SCF's own (correlation, response and Z-vector densities, all
//! indefinite), plus `∂_R B(D1, D2)` of the bilinear two-electron energy.
//! Both must be built with the SAME integrals the SCF used, or the gradient
//! is that of a different energy. [`ResponseJk::for_scf`] reads the route
//! the SCF recorded in [`ScfResult::df_jk`] and serves:
//!
//! * no route (four-centre J and K), from the in-core ERI store when the
//!   SCF attached one to `bounds` ([`crate::incore_jk`], `[scf] eri`);
//! * no route, no store: [`crate::rhf::build_jk`] /
//!   [`crate::rhf::build_jk_with_pool`] and
//!   [`crate::gradient::twoelectron_gradient_bilinear`], called exactly as a
//!   caller of those functions would, so the exact route is bit for bit the
//!   direct calls;
//! * RI-J + RI-K on one auxiliary basis (`scf_jk = "rijk"`): the SCF's
//!   [`crate::df_j::DfJ`] / [`crate::df_k::DfK`] (same basis, operator,
//!   budget and storage precision; `DfK`'s density path takes indefinite
//!   matrices), and the fitted bilinear derivative of
//!   [`fitted_bilinear_gradient`].
//!
//! The open-shell bilinear derivative
//! [`ResponseJk::two_electron_gradient_bilinear_open`] (the unrestricted
//! RI-MP2 gradient) takes a density pair per argument and uses
//! [`crate::gradient::twoelectron_gradient_bilinear_uhf`] or
//! [`fitted_bilinear_gradient_open`] on the same two routes.
//!
//! Any other route (RI-J with four-centre or COSX exchange, range-separated
//! exchange, J and K on different bases) is refused: nothing here builds its
//! exchange response.

use crate::df_gradient::{df_jk_gradient_at_precision, DfKChannel};
use crate::fock::{JBuilder, KBuilder};
use crate::result::DfJkRoute;
use crate::screening::SchwarzBounds;
use crate::ScfResult;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use std::cell::RefCell;

/// The RI-J / RI-K builders of a fitted route.
struct FittedJk {
    j: RefCell<crate::df_j::DfJ<'static>>,
    k: RefCell<crate::df_k::DfK<'static>>,
    route: DfJkRoute,
}

/// J/K builds and the separable two-electron derivative on an SCF's route.
pub struct ResponseJk<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    fitted: Option<FittedJk>,
    /// The in-core arm of the four-centre route: the store the SCF attached
    /// to `bounds` when it built one.
    incore: Option<std::sync::Arc<crate::incore_jk::IncoreEri>>,
}

/// The RI-J and RI-K builders of `route` on `dfbs`. With an f64 RI-J
/// tensor the raw `(P|μν)` tensor is built ONCE: RI-K dresses it (its own
/// constructor builds the same tensor with the same call and dresses it the
/// same way) and RI-J then keeps it with the same metric `DfJ::new` builds,
/// for one three-index pass instead of two. The builders are bitwise those of
/// two independent builds when both take the same storage tier (always under
/// an ample budget; under a tight pool the single raw tensor can leave room
/// for a different RI-K layout than the second raw tensor did) (the SCF's `fock_assembly::build_df_jk`
/// shares it the same way). An f32 RI-J tensor is built by its own f32 pass,
/// so that route keeps the two independent builds.
fn fitted_pair(
    route: &DfJkRoute,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
) -> Result<(crate::df_j::DfJ<'static>, crate::df_k::DfK<'static>), FerricError> {
    use crate::df_j::{DfJ, RiJPrecision};
    use crate::df_k::DfK;
    use ferric_integrals::three_index_source::ThreeIndexSource;
    let (op, budget) = (route.op, route.budget_bytes);
    if route.ri_j_precision != RiJPrecision::F64 {
        let j = DfJ::new_with_precision(op, prep, dfbs, budget, route.ri_j_precision)?;
        let k = DfK::new_with_precision(op, prep, dfbs, budget, route.ri_k_precision)?;
        return Ok((j, k));
    }
    let mut raw = ThreeIndexSource::build(op, prep, dfbs, budget)?;
    let k = DfK::from_full_raw_with_precision(
        &mut raw,
        prep,
        dfbs,
        op,
        budget,
        None,
        route.ri_k_precision,
    )?;
    let j = DfJ::from_source(raw, op, dfbs, budget, None, RiJPrecision::F64)?;
    Ok((j, k))
}

impl<'a> ResponseJk<'a> {
    /// The four-centre route.
    pub fn exact(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
    ) -> Self {
        ResponseJk {
            ctx,
            prep,
            bounds,
            fitted: None,
            incore: bounds.incore.get(),
        }
    }

    /// The route `scf` recorded: four-centre when it has none, RI-JK when J
    /// and K were fitted on one auxiliary basis with no range separation.
    /// Builds the fitted builders up front (one shared three-index pass for an
    /// f64 RI-J tensor, see `fitted_pair`).
    pub fn for_scf(
        ctx: &'a ParallelContext,
        mol: &Molecule,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        scf: &ScfResult,
    ) -> Result<Self, FerricError> {
        if scf.cosx_final.is_some() || scf.cosx_schedule.is_some() {
            return Err(FerricError::General(
                "the reference SCF built exchange with COSX; only four-centre or RI-JK \
                 references have a J/K response here"
                    .into(),
            ));
        }
        Self::for_route(ctx, mol, prep, bounds, scf.df_jk.as_ref())
    }

    /// [`Self::for_scf`] from a recorded route alone (`None`: four-centre),
    /// for a caller that kept the route but not the SCF result. The caller
    /// vouches that the route's exchange was not COSX.
    pub fn for_route(
        ctx: &'a ParallelContext,
        mol: &Molecule,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        route: Option<&DfJkRoute>,
    ) -> Result<Self, FerricError> {
        let Some(route) = route else {
            return Ok(Self::exact(ctx, prep, bounds));
        };
        let aux = match (&route.j_aux, &route.k_aux, &route.rsh_k) {
            (Some(j), Some(k), None) if j == k => j.clone(),
            _ => {
                return Err(FerricError::General(format!(
                    "the reference SCF's J/K route (RI-J {:?}, RI-K {:?}, range-separated K \
                     {:?}) has no J/K response here: only four-centre J/K or RI-J + RI-K on \
                     one auxiliary basis (scf_jk = \"rijk\") are supported",
                    route.j_aux, route.k_aux, route.rsh_k
                )))
            }
        };
        let dfbs = crate::df_gradient::aux_basis(mol, &aux)?;
        let (j, k) = fitted_pair(route, prep, &dfbs)?;
        Ok(ResponseJk {
            ctx,
            prep,
            bounds,
            fitted: Some(FittedJk {
                j: RefCell::new(j),
                k: RefCell::new(k),
                route: route.clone(),
            }),
            incore: None,
        })
    }

    /// The fitted route, `None` for four-centre J/K.
    pub fn route(&self) -> Option<&DfJkRoute> {
        self.fitted.as_ref().map(|f| &f.route)
    }

    /// Does the four-centre route serve J/K from the in-core ERI store?
    pub fn is_incore(&self) -> bool {
        self.incore.is_some()
    }

    /// The in-core store, checked against the basis it is about to serve.
    fn store(&self) -> Result<Option<&crate::incore_jk::IncoreEri>, FerricError> {
        match &self.incore {
            Some(s) if !s.serves(self.prep, self.bounds.op) => Err(FerricError::General(
                "ResponseJk: the in-core ERI store was built for another geometry, basis or \
                 operator"
                    .into(),
            )),
            Some(s) => Ok(Some(s)),
            None => Ok(None),
        }
    }

    /// `(J[d], K[d])` of a symmetric `d`. Four-centre: [`crate::rhf::build_jk`]
    /// with screening threshold `thresh`; fitted: the SCF's builders
    /// (`thresh` unused).
    pub fn jk(
        &self,
        d: &Array2<f64>,
        thresh: f64,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let n = d.nrows();
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        match (&self.fitted, self.store()?) {
            (Some(f), _) => fitted_jk(f, d, &mut j, &mut k)?,
            (None, Some(s)) => s.add_jk(d, &mut j, &mut k),
            (None, None) => {
                crate::rhf::build_jk(self.ctx, self.prep, self.bounds, thresh, d, &mut j, &mut k)?;
            }
        }
        Ok((j, k))
    }

    /// [`Self::jk`] with a caller-held engine pool and reduction band for the
    /// four-centre route ([`crate::rhf::build_jk_with_pool`]); `pool` may be
    /// `None` only on the fitted route.
    pub fn jk_with_pool(
        &self,
        d: &Array2<f64>,
        thresh: f64,
        pool: Option<&crate::engine_pool::EnginePool>,
        band_bytes: usize,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let n = d.nrows();
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        if let (None, Some(s)) = (&self.fitted, self.store()?) {
            s.add_jk(d, &mut j, &mut k);
            return Ok((j, k));
        }
        match (&self.fitted, pool) {
            (None, Some(pool)) => {
                crate::rhf::build_jk_with_pool(
                    self.ctx,
                    self.prep,
                    self.bounds,
                    thresh,
                    d,
                    &mut j,
                    &mut k,
                    pool,
                    band_bytes,
                )?;
            }
            (None, None) => {
                return Err(FerricError::General(
                    "ResponseJk::jk_with_pool: the four-centre route needs an engine pool".into(),
                ))
            }
            (Some(f), _) => fitted_jk(f, d, &mut j, &mut k)?,
        }
        Ok((j, k))
    }

    /// The engine pool a four-centre [`Self::jk_with_pool`] loop reuses, at
    /// `precision`; `None` on the fitted route, which needs none.
    pub fn engine_pool(
        &self,
        precision: f64,
    ) -> Result<Option<crate::engine_pool::EnginePool>, FerricError> {
        match self.fitted {
            None => Ok(Some(crate::engine_pool::EnginePool::new(
                self.bounds.op,
                self.prep,
                precision,
            )?)),
            Some(_) => Ok(None),
        }
    }

    /// `∂_R B(d1, d2)` at fixed densities, `(natoms, 3)`: the explicit
    /// nuclear derivative of the symmetric bilinear two-electron energy whose
    /// diagonal `B(D, D)` is the closed-shell `½ Σ D D (μν|λσ) − ¼ Σ D D
    /// (μλ|νσ)` of total densities. Four-centre:
    /// [`crate::gradient::twoelectron_gradient_bilinear`] with operator `op`;
    /// fitted: [`fitted_bilinear_gradient`] with the route's operator.
    pub fn two_electron_gradient_bilinear(
        &self,
        mol: &Molecule,
        op: Operator,
        d1: &Array2<f64>,
        d2: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        match &self.fitted {
            None => {
                crate::gradient::twoelectron_gradient_bilinear(self.prep, op, self.bounds, d1, d2)
            }
            Some(f) => fitted_bilinear_gradient(mol, self.prep, &f.route, d1, d2),
        }
    }

    /// `∂_R U(x, y)` at fixed spin densities `x = (X_α, X_β)`, `y = (Y_α,
    /// Y_β)`, `(natoms, 3)`: the open-shell sibling of
    /// [`Self::two_electron_gradient_bilinear`], the polarization of the
    /// UHF two-electron energy `U(D, D) = ½ Σ D_T D_T (μν|λσ) − ½ Σ_σ Σ D_σ
    /// D_σ (μλ|νσ)`. Four-centre:
    /// [`crate::gradient::twoelectron_gradient_bilinear_uhf`] with operator
    /// `op`; fitted: [`fitted_bilinear_gradient_open`] with the route's
    /// operator. For `X_α = X_β = X/2` (and the same for `Y`) it is
    /// [`Self::two_electron_gradient_bilinear`] of the totals.
    pub fn two_electron_gradient_bilinear_open(
        &self,
        mol: &Molecule,
        op: Operator,
        x: (&Array2<f64>, &Array2<f64>),
        y: (&Array2<f64>, &Array2<f64>),
    ) -> Result<Array2<f64>, FerricError> {
        match &self.fitted {
            None => {
                crate::gradient::twoelectron_gradient_bilinear_uhf(self.prep, op, self.bounds, x, y)
            }
            Some(f) => fitted_bilinear_gradient_open(mol, self.prep, &f.route, x, y),
        }
    }
}

fn fitted_jk(
    f: &FittedJk,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<(), FerricError> {
    j.fill(0.0);
    k.fill(0.0);
    f.j.borrow_mut().build(d, j)?;
    f.k.borrow_mut().build(d, k)?;
    Ok(())
}

/// Exchange channels of `weight · Q_K(d)`, `Q_K(D) = Σ D_μν D_λσ (μλ|νσ)_fit`,
/// for a symmetric `d` of either sign: a positive-semidefinite `d` (no
/// eigenvalue below `−1e-8·max|λ|`, the RI-K density factorization's own
/// tolerance) is the single channel `(d, weight)`; otherwise `d = P − N`
/// with `P, N ⪰ 0` and `Q_K(P − N) = 2Q_K(P) + 2Q_K(N) − Q_K(P + N)`.
fn signed_channels(d: &Array2<f64>, weight: f64) -> Result<Vec<(Array2<f64>, f64)>, FerricError> {
    let (evals, evecs) = d.eigh(UPLO::Upper).map_err(|e| {
        FerricError::Lapack(format!("fitted 2e gradient: density eigh failed: {e}"))
    })?;
    let scale = evals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if evals.iter().all(|&v| v >= -1e-8 * scale) {
        return Ok(vec![(d.clone(), weight)]);
    }
    let part = |keep: &dyn Fn(f64) -> bool| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros(d.dim());
        for (k, &lam) in evals.iter().enumerate() {
            if keep(lam) {
                let v = evecs.column(k).insert_axis(ndarray::Axis(1));
                m.scaled_add(lam.abs(), &v.dot(&v.t()));
            }
        }
        m
    };
    let (p, n) = (part(&|l| l > 0.0), part(&|l| l < 0.0));
    let abs = &p + &n;
    Ok(vec![(p, 2.0 * weight), (n, 2.0 * weight), (abs, -weight)])
}

/// `∂_R B(d1, d2)` at fixed densities under an RI-JK route (`route.j_aux ==
/// route.k_aux`, no range separation), with `B` the polarization of the
/// fitted closed-shell two-electron energy
/// `Q(D) = ½ dᵀV⁻¹d − ¼ Q_K(D)` (`d_P = (P|μν) D_μν`).
///
/// `B(d1, d2) = Q(u) − Q(v)` with `u = (d1 + d2)/2`, `v = (d1 − d2)/2`, and
/// each `∂Q` is the RI-J / RI-K gradient of [`crate::df_gradient`]
/// ([`df_jk_gradient_at_precision`], the route's RI-J precision). The
/// exchange channels of `u` and `v` share one three-centre pass; the Coulomb
/// part of `v` is a second, J-only pass. Derivation:
/// `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-fitted-ri-jk-reference`.
pub fn fitted_bilinear_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    route: &DfJkRoute,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let aux = match (&route.j_aux, &route.k_aux, &route.rsh_k) {
        (Some(j), Some(k), None) if j == k => j,
        _ => {
            return Err(FerricError::General(
                "fitted_bilinear_gradient: the route is not RI-J + RI-K on one basis".into(),
            ))
        }
    };
    let dfbs = crate::df_gradient::aux_basis(mol, aux)?;
    let u = 0.5 * &(d1 + d2);
    let v = 0.5 * &(d1 - d2);
    let mut owned = signed_channels(&u, -0.25)?;
    owned.extend(signed_channels(&v, 0.25)?);
    let channels: Vec<DfKChannel<'_>> = owned
        .iter()
        .map(|(density, alpha)| DfKChannel {
            density,
            alpha: *alpha,
        })
        .collect();
    let natoms = mol.atoms.len();
    let mut grad = df_jk_gradient_at_precision(
        natoms,
        prep,
        &dfbs,
        route.op,
        Some(&u),
        &channels,
        route.budget_bytes,
        route.ri_j_precision,
    )?;
    grad -= &df_jk_gradient_at_precision(
        natoms,
        prep,
        &dfbs,
        route.op,
        Some(&v),
        &[],
        route.budget_bytes,
        route.ri_j_precision,
    )?;
    Ok(grad)
}

/// The aux basis of an RI-JK route (`j_aux == k_aux`, no range separation),
/// or an error naming `who`.
fn rijk_aux<'r>(route: &'r DfJkRoute, who: &str) -> Result<&'r String, FerricError> {
    match (&route.j_aux, &route.k_aux, &route.rsh_k) {
        (Some(j), Some(k), None) if j == k => Ok(j),
        _ => Err(FerricError::General(format!(
            "{who}: the route is not RI-J + RI-K on one basis"
        ))),
    }
}

/// `∂_R U(x, y)` at fixed spin densities under an RI-JK route, with `U` the
/// polarization of the fitted UHF two-electron energy
/// `Q(D) = ½ d_Tᵀ V⁻¹ d_T − ½ Σ_σ Q_K(D_σ)` (`d_T,P = (P|μν) (D_α + D_β)_μν`).
///
/// `U(x, y) = Q(u) − Q(v)` with `u_σ = (X_σ + Y_σ)/2`, `v_σ = (X_σ − Y_σ)/2`:
/// the exchange channels of `u_α, u_β` (weight `−½`) and `v_α, v_β` (weight
/// `+½`) share one three-centre pass with the Coulomb part of `u_T`; the
/// Coulomb part of `v_T` is a second, J-only pass. The closed-shell
/// [`fitted_bilinear_gradient`] is the case `X_α = X_β`, `Y_α = Y_β`
/// (`2 · Q_K(D/2) · ½ = ¼ Q_K(D)`). Derivation:
/// `book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-energy-weights`.
pub fn fitted_bilinear_gradient_open(
    mol: &Molecule,
    prep: &PreparedBasis,
    route: &DfJkRoute,
    x: (&Array2<f64>, &Array2<f64>),
    y: (&Array2<f64>, &Array2<f64>),
) -> Result<Array2<f64>, FerricError> {
    let aux = rijk_aux(route, "fitted_bilinear_gradient_open")?;
    let dfbs = crate::df_gradient::aux_basis(mol, aux)?;
    let half = |a: &Array2<f64>, b: &Array2<f64>, s: f64| 0.5 * &(a + &(s * b));
    let (ua, ub) = (half(x.0, y.0, 1.0), half(x.1, y.1, 1.0));
    let (va, vb) = (half(x.0, y.0, -1.0), half(x.1, y.1, -1.0));
    let mut owned = signed_channels(&ua, -0.5)?;
    owned.extend(signed_channels(&ub, -0.5)?);
    owned.extend(signed_channels(&va, 0.5)?);
    owned.extend(signed_channels(&vb, 0.5)?);
    let channels: Vec<DfKChannel<'_>> = owned
        .iter()
        .map(|(density, alpha)| DfKChannel {
            density,
            alpha: *alpha,
        })
        .collect();
    let natoms = mol.atoms.len();
    let (ut, vt) = (&ua + &ub, &va + &vb);
    let mut grad = df_jk_gradient_at_precision(
        natoms,
        prep,
        &dfbs,
        route.op,
        Some(&ut),
        &channels,
        route.budget_bytes,
        route.ri_j_precision,
    )?;
    grad -= &df_jk_gradient_at_precision(
        natoms,
        prep,
        &dfbs,
        route.op,
        Some(&vt),
        &[],
        route.budget_bytes,
        route.ri_j_precision,
    )?;
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df_gradient::{routed_two_electron_gradient, ExchangeMix, TwoElectronDensity};
    use crate::rhf::{solve_rhf, RhfConfig};

    const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";

    fn setup(fit: bool) -> (Molecule, PreparedBasis, SchwarzBounds, ScfResult) {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut cfg = RhfConfig {
            density_conv: 1e-9,
            ..Default::default()
        }
        .for_geometry_task();
        if fit {
            cfg.df_j_aux = Some("def2-universal-jkfit".into());
            cfg.df_k_aux = Some("def2-universal-jkfit".into());
        }
        let ctx = ParallelContext::default();
        let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        (mol, prep, bounds, scf)
    }

    /// A symmetric, indefinite test density: the SCF density minus a
    /// rescaled copy of its own square, so both eigenvalue signs occur.
    fn indefinite(d: &Array2<f64>) -> Array2<f64> {
        let dd = d.dot(d);
        d - &(0.37 * &dd)
    }

    fn bits(a: &Array2<f64>) -> Vec<u64> {
        a.iter().map(|v| v.to_bits()).collect()
    }

    /// The four-centre route is the direct calls, bit for bit: J/K with and
    /// without a caller-held pool, and the bilinear derivative.
    #[test]
    fn exact_route_is_the_direct_calls_bit_for_bit() {
        let (mol, prep, bounds, scf) = setup(false);
        let ctx = ParallelContext::default();
        let jk = ResponseJk::for_scf(&ctx, &mol, &prep, &bounds, &scf).unwrap();
        assert!(jk.route().is_none());
        let d = indefinite(scf.density_r());
        let n = d.nrows();
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        crate::rhf::build_jk(&ctx, &prep, &bounds, 1e-12, &d, &mut j, &mut k).unwrap();
        let (j2, k2) = jk.jk(&d, 1e-12).unwrap();
        assert_eq!(bits(&j), bits(&j2));
        assert_eq!(bits(&k), bits(&k2));

        let pool = jk
            .engine_pool(1e-14)
            .unwrap()
            .expect("four-centre route has a pool");
        let band = crate::reduce::default_band_bytes();
        let (mut jp, mut kp) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        crate::rhf::build_jk_with_pool(
            &ctx, &prep, &bounds, 1e-12, &d, &mut jp, &mut kp, &pool, band,
        )
        .unwrap();
        let (jp2, kp2) = jk.jk_with_pool(&d, 1e-12, Some(&pool), band).unwrap();
        assert_eq!(bits(&jp), bits(&jp2));
        assert_eq!(bits(&kp), bits(&kp2));

        let op = Operator::coulomb();
        let hf = scf.density_r();
        let g = crate::gradient::twoelectron_gradient_bilinear(&prep, op, &bounds, hf, &d).unwrap();
        let g2 = jk.two_electron_gradient_bilinear(&mol, op, hf, &d).unwrap();
        assert_eq!(bits(&g), bits(&g2));
    }

    /// Fitted route. At `d1 = d2 = D` the polarization is `Q(D) − Q(0)`, the
    /// same RI-J/RI-K channels the fitted SCF gradient differentiates, so it is
    /// that gradient's two-electron part bit for bit. `B` is symmetric; the
    /// swapped call reorders the exchange channels of `v → −v` (its `P` and
    /// `N` trade places), so it agrees to rounding: bar `64 ε max|g|`, the
    /// association error of a sum of a few dozen terms of size `max|g|`.
    #[test]
    fn fitted_bilinear_reduces_to_the_scf_gradient_and_is_symmetric() {
        let (mol, prep, bounds, scf) = setup(true);
        let ctx = ParallelContext::default();
        let jk = ResponseJk::for_scf(&ctx, &mol, &prep, &bounds, &scf).unwrap();
        let route = jk.route().expect("fitted route").clone();
        let op = Operator::coulomb();
        let hf = scf.density_r();
        let b_dd = jk.two_electron_gradient_bilinear(&mol, op, hf, hf).unwrap();
        let scf_2e = routed_two_electron_gradient(
            &mol,
            &prep,
            op,
            &bounds,
            TwoElectronDensity::Closed(hf),
            &route,
            ExchangeMix::hartree_fock(),
        )
        .unwrap();
        assert_eq!(bits(&b_dd), bits(&scf_2e));

        let d = indefinite(hf);
        let b12 = jk.two_electron_gradient_bilinear(&mol, op, hf, &d).unwrap();
        let b21 = jk.two_electron_gradient_bilinear(&mol, op, &d, hf).unwrap();
        let scale = b12.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diff = (&b12 - &b21).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            diff <= 64.0 * f64::EPSILON * scale,
            "|B12 − B21| = {diff:.3e}, max|B| {scale:.3e}"
        );
        assert!(scale > 1e-3, "the test density must give a non-trivial B");
    }

    /// The fitted J/K of an indefinite density are linear in it and differ
    /// from the four-centre ones by the fit error (nonzero, small), so the
    /// fitted route really builds fitted matrices.
    #[test]
    fn fitted_jk_is_linear_and_not_four_centre() {
        let (mol, prep, bounds, scf) = setup(true);
        let ctx = ParallelContext::default();
        let jk = ResponseJk::for_scf(&ctx, &mol, &prep, &bounds, &scf).unwrap();
        let d = indefinite(scf.density_r());
        let (j1, k1) = jk.jk(&d, 1e-12).unwrap();
        let (j2, k2) = jk.jk(&(2.0 * &d), 1e-12).unwrap();
        let max = |a: &Array2<f64>| a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(max(&(&j2 - &(2.0 * &j1))) <= 64.0 * f64::EPSILON * max(&j2));
        assert!(max(&(&k2 - &(2.0 * &k1))) <= 64.0 * f64::EPSILON * max(&k2));
        let ex = ResponseJk::exact(&ctx, &prep, &bounds);
        let (je, ke) = ex.jk(&d, 1e-12).unwrap();
        let (dj, dk) = (max(&(&j1 - &je)), max(&(&k1 - &ke)));
        eprintln!("fit error: max|ΔJ| {dj:.3e}, max|ΔK| {dk:.3e}");
        assert!(
            dj > 1e-8 && dk > 1e-8,
            "fitted J/K equal the four-centre ones"
        );
        assert!(
            dj < 1e-2 && dk < 1e-2,
            "fit error too large: {dj:.3e} {dk:.3e}"
        );
    }

    /// The response builders built from one shared raw tensor
    /// ([`fitted_pair`]) give the J and K of two independent builds (the
    /// constructors `ResponseJk` called before), bit for bit.
    #[test]
    fn shared_raw_pair_is_the_independent_builds_bitwise() {
        let (mol, prep, _, scf) = setup(true);
        let route = scf.df_jk.clone().expect("RI-JK route");
        assert_eq!(route.ri_j_precision, crate::df_j::RiJPrecision::F64);
        let dfbs = crate::df_gradient::aux_basis(&mol, route.j_aux.as_deref().unwrap()).unwrap();
        let (mut j_shared, mut k_shared) = fitted_pair(&route, &prep, &dfbs).unwrap();
        let mut j_alone = crate::df_j::DfJ::new_with_precision(
            route.op,
            &prep,
            &dfbs,
            route.budget_bytes,
            route.ri_j_precision,
        )
        .unwrap();
        let mut k_alone = crate::df_k::DfK::new_with_precision(
            route.op,
            &prep,
            &dfbs,
            route.budget_bytes,
            route.ri_k_precision,
        )
        .unwrap();
        for d in [scf.density_r().clone(), indefinite(scf.density_r())] {
            let n = d.nrows();
            let mut out = [(); 4].map(|_| Array2::<f64>::zeros((n, n)));
            j_shared.build(&d, &mut out[0]).unwrap();
            j_alone.build(&d, &mut out[1]).unwrap();
            k_shared.build(&d, &mut out[2]).unwrap();
            k_alone.build(&d, &mut out[3]).unwrap();
            assert_eq!(bits(&out[0]), bits(&out[1]), "J");
            assert_eq!(bits(&out[2]), bits(&out[3]), "K");
        }
    }

    const OH: &str = "2\nOH doublet\nO 0.0 0.0 0.0\nH 0.05 0.0 0.9697\n";

    /// A converged UHF on OH/cc-pVDZ, four-centre or RI-JK.
    fn setup_uhf(fit: bool) -> (Molecule, PreparedBasis, SchwarzBounds, ScfResult) {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let mut cfg = RhfConfig {
            density_conv: 1e-9,
            ..Default::default()
        }
        .for_geometry_task();
        if fit {
            cfg.df_j_aux = Some("def2-universal-jkfit".into());
            cfg.df_k_aux = Some("def2-universal-jkfit".into());
        }
        let ctx = ParallelContext::default();
        let scf = crate::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(scf.converged);
        (mol, prep, bounds, scf)
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// Rounding bar of two evaluations of one gradient that associate their
    /// sums differently: `2⁸ ε max|g|`. Each element is a sum over
    /// `O(nao⁴)` (four-centre) or `O(naux nao²)` (fitted) terms of mixed sign;
    /// the measured spread is at most `13 ε max|g|` (fitted route, spin-
    /// symmetric operands; exact route 0.7 ε).
    fn rounding_bar(g: &Array2<f64>) -> f64 {
        256.0 * f64::EPSILON * max_abs(g)
    }

    /// Open-shell bilinear at spin-symmetric operands `(X/2, X/2)`, `(Y/2,
    /// Y/2)` is the closed-shell bilinear of the totals, on both routes; the
    /// test density is indefinite so the fitted route's signed channels are
    /// exercised.
    #[test]
    fn open_bilinear_at_symmetric_spins_is_the_closed_bilinear() {
        for fit in [false, true] {
            let (mol, prep, bounds, scf) = setup(fit);
            let ctx = ParallelContext::default();
            let jk = ResponseJk::for_scf(&ctx, &mol, &prep, &bounds, &scf).unwrap();
            let op = Operator::coulomb();
            let hf = scf.density_r();
            let d = indefinite(hf);
            let closed = jk.two_electron_gradient_bilinear(&mol, op, hf, &d).unwrap();
            let (h2, d2) = (0.5 * hf, 0.5 * &d);
            let open = jk
                .two_electron_gradient_bilinear_open(&mol, op, (&h2, &h2), (&d2, &d2))
                .unwrap();
            let diff = max_abs(&(&open - &closed));
            eprintln!(
                "symmetric spins fit={fit}: spread {:.1} eps max|g|",
                diff / (f64::EPSILON * max_abs(&closed))
            );
            assert!(max_abs(&closed) > 1e-3, "non-trivial bilinear");
            assert!(
                diff <= rounding_bar(&closed),
                "fit={fit}: |open - closed| = {diff:.3e}, bar {:.3e}",
                rounding_bar(&closed)
            );
        }
    }

    /// At `x = y = (D_α, D_β)` of a converged UHF the open bilinear is the
    /// UHF two-electron gradient: `twoelectron_gradient_uhf` (four-centre),
    /// the routed RI-JK gradient of the SCF (fitted).
    #[test]
    fn open_bilinear_at_the_uhf_densities_is_the_uhf_two_electron_gradient() {
        for fit in [false, true] {
            let (mol, prep, bounds, scf) = setup_uhf(fit);
            let ctx = ParallelContext::default();
            let jk = ResponseJk::for_scf(&ctx, &mol, &prep, &bounds, &scf).unwrap();
            let op = Operator::coulomb();
            let da = &scf.density_alpha;
            let db = scf.density_beta.as_ref().unwrap();
            let open = jk
                .two_electron_gradient_bilinear_open(&mol, op, (da, db), (da, db))
                .unwrap();
            let reference = match jk.route() {
                None => {
                    let dt = da + db;
                    crate::gradient::twoelectron_gradient_uhf(&prep, op, &bounds, &dt, da, db)
                        .unwrap()
                }
                Some(route) => routed_two_electron_gradient(
                    &mol,
                    &prep,
                    op,
                    &bounds,
                    TwoElectronDensity::Open {
                        alpha: da,
                        beta: db,
                    },
                    route,
                    ExchangeMix::hartree_fock(),
                )
                .unwrap(),
            };
            let diff = max_abs(&(&open - &reference));
            eprintln!(
                "uhf densities fit={fit}: spread {:.1} eps max|g|",
                diff / (f64::EPSILON * max_abs(&reference))
            );
            assert!(
                diff <= rounding_bar(&reference),
                "fit={fit}: |open - uhf| = {diff:.3e}, bar {:.3e}",
                rounding_bar(&reference)
            );
        }
    }
}
