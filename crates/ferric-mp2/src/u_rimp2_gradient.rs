//! Unrestricted RI-MP2 nuclear gradient on a UHF reference.
//!
//! The open-shell sibling of [`crate::gradient::rimp2_gradient_analytical`]:
//! every term of the closed-shell Lagrangian gradient is a sum over the two
//! spins of the same term built from that spin's orbitals, orbital energies,
//! unrelaxed density, Lagrangian matrix `Imat^σ` and Z-vector `z^σ`, with the
//! two spins coupled only through the Coulomb part of the orbital Hessian and
//! of the two-electron derivative. The derivation, and the closed-shell
//! reduction (`P^α = P^β = P/2`, `Imat^α = Imat^β = Imat/2`, `z^α = z^β =
//! z/2`), is
//! `book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-energy-weights`; the
//! scalar identities are checked by
//! `scripts/u_rimp2_gradient_sympy.py`.
//!
//! Per spin σ (`i, j` occupied, `a, b` virtual, `p, q` any MO, all of spin σ):
//!
//! ```text
//!   x^σ_{P,ia}     = Σ_jb t^σσ_{ij,ab} B^σ_{P,jb} + Σ_{JB} t^{σσ̄}_{iJ,aB} B^σ̄_{P,JB}   (∂E/∂B^σ = 2 x^σ)
//!   Imat^σ[q, i]   = −Σ_{P,a} x^σ_{P,ia} b^σ[P, q, a],   Imat^σ[q, a] = −Σ_{P,i} x^σ_{P,ia} b^σ[P, q, i]
//!   P^σ            = P^σ_oo ⊕ P^σ_vv (unrelaxed spin-orbital MP2 density) + frozen core block
//!   veff^σ[X]      = J[X_α + X_β] − K[X_σ]
//!   Xvo^σ_ai       = (C_σᵀ veff^σ[P_AO] C_σ)_ai + Imat^σ[i, a] − Imat^σ[a, i]
//!   (ε_a − ε_i) z^σ_ai + (C_σᵀ veff^σ[D^z] C_σ)_ai = −Xvo^σ_ai,   D^z_σ = C_v z^σ C_oᵀ + h.c.
//! ```
//!
//! and the gradient is
//!
//! ```text
//!   g = Σ (D_HF + P_rel)_T · ∂h  +  Σ_σ ∂S · [im1^σ − ζ^σ − W^σ_HF − P^σ_occ veff^σ[P_rel] P^σ_occ]
//!     + ∂U((D^α_HF, D^β_HF), (D^α_HF + 2P^α_rel, D^β_HF + 2P^β_rel))  +  RI 3c/2c response,
//! ```
//!
//! with `U` the polarization of the UHF two-electron energy
//! ([`ferric_scf::response_jk::ResponseJk::two_electron_gradient_bilinear_open`])
//! and the RI response one channel per spin of weight 1
//! (`gradient::ri_response_gradient_channels`).
//!
//! Both J/K routes of the closed-shell gradient are supported: four-centre and
//! RI-JK on one auxiliary basis, read from the reference's recorded route.
//! A ROHF reference is refused (see [`u_rimp2_gradient_analytical`](crate::u_rimp2_gradient::u_rimp2_gradient_analytical)).

use crate::gradient::{ri_response_gradient_channels, RiResponseChannel};
use crate::rimp2::{RiMp2Config, SpinScaling};
use crate::u_rimp2::{
    build_u_mp2_density, build_u_mp2_density_scaled, compute_u_mp2_amplitudes, u_ri_mp2,
    UMp2Amplitudes,
};
use crate::zvector::ZvectorControls;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_scf::diis::Diis;
use ferric_scf::response_jk::ResponseJk;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::{solve_uhf_warm, ScfWarmStart};
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2, Axis};

/// The reason a ROHF/ROKS reference has no unrestricted RI-MP2 gradient here.
pub const ROHF_REFUSAL: &str = "the unrestricted RI-MP2 gradient needs a UHF reference: a ROHF \
     reference is not stationary under independent alpha and beta orbital rotations, and its \
     MP2 energy is evaluated on semicanonical orbitals, so the gradient needs the ROHF Z-vector \
     (closed/open/virtual rotations) and the response of the semicanonicalization, neither of \
     which is implemented. Use a UHF reference ([scf] reference = \"uhf\", the open-shell \
     default; Python reference=\"uhf\")";

/// One spin's orbitals and partition.
struct SpinOrbitals<'a> {
    c: &'a Array2<f64>,
    eps: &'a [f64],
    orb: OrbitalSpace,
}

impl SpinOrbitals<'_> {
    fn nmo(&self) -> usize {
        self.c.ncols()
    }
    /// Every occupied orbital, frozen core included.
    fn c_occ_all(&self) -> Array2<f64> {
        self.c.slice(s![.., ..self.orb.nocc_total]).to_owned()
    }
    fn c_occ_active(&self) -> Array2<f64> {
        let o = self.orb;
        self.c
            .slice(s![.., o.first_occ..o.first_occ + o.nocc])
            .to_owned()
    }
    fn c_vir(&self) -> Array2<f64> {
        self.c.slice(s![.., self.orb.nocc_total..]).to_owned()
    }
    fn to_ao(&self, m: &Array2<f64>) -> Array2<f64> {
        with_blas_threads(opt_in_blas_threads(), || self.c.dot(m).dot(&self.c.t()))
    }
    fn to_mo(&self, m: &Array2<f64>) -> Array2<f64> {
        with_blas_threads(opt_in_blas_threads(), || self.c.t().dot(m).dot(self.c))
    }
}

/// One correlation term an unrestricted gradient differentiates,
/// `c_os · E_αβ[op] + c_ss · (E_αα + E_ββ)[op]` (`scaling`): `(Coulomb,
/// (1, 1))` for U-RI-MP2, `(erfc(ωr)/r, (s, s))` for attenuated U-RI-MP2
/// and `(terfc, (1, 1))` for terfc U-RI-MP2 ([`crate::att_rimp2_gradient`]);
/// SCS-MP2(2terfc) is a sum of two terms ([`crate::scs_2terfc_gradient`]).
/// `op` is the operator of the `(P|ia)` integrals, of the RI metric and of
/// their derivatives; the reference's own two-electron terms run on the
/// operator of its Schwarz bounds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UCorrelation {
    pub op: Operator,
    pub scaling: SpinScaling,
}

impl UCorrelation {
    /// `s · E_c[op]`, both spin components scaled by `s`.
    pub(crate) fn uniform(op: Operator, s: f64) -> Self {
        Self {
            op,
            scaling: SpinScaling { c_os: s, c_ss: s },
        }
    }
}

/// Refuse a reference the unrestricted RI-MP2 gradient does not
/// differentiate: anything but a converged, aufbau UHF with at least one
/// electron of each spin, an X2C reference, a non-Coulomb reference, or a
/// correlation operator other than Coulomb, erfc or terfc.
fn check_reference(
    mol: &Molecule,
    uhf: &ScfResult,
    corr: &[UCorrelation],
    ref_op: Operator,
    obs: &PreparedBasis,
) -> Result<(usize, usize), FerricError> {
    ferric_scf::x2c::refuse_relativistic(uhf.relativistic, "U-RI-MP2 analytic gradient")?;
    match uhf.spin {
        Spin::Unrestricted => {}
        Spin::RestrictedOpen => return Err(FerricError::General(ROHF_REFUSAL.into())),
        Spin::Restricted => {
            return Err(FerricError::General(
                "u_rimp2_gradient_analytical: the reference is restricted (RHF); use \
                 ferric_mp2::gradient::rimp2_gradient_analytical"
                    .into(),
            ))
        }
    }
    if !uhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: uhf.iterations,
            last_energy: uhf.energy,
        });
    }
    if ref_op.is_composite() || ref_op.kind != OperatorKind::Coulomb {
        return Err(FerricError::General(format!(
            "U-RI-MP2 gradient: the reference UHF must be Coulomb (its Schwarz bounds are for \
             {:?})",
            ref_op.kind
        )));
    }
    if corr.is_empty() {
        return Err(FerricError::General(
            "U-RI-MP2 gradient: no correlation term".into(),
        ));
    }
    for c in corr {
        if c.op.is_composite()
            || !matches!(
                c.op.kind,
                OperatorKind::Coulomb | OperatorKind::ErfcCoulomb | OperatorKind::Terfc
            )
        {
            return Err(FerricError::General(format!(
                "U-RI-MP2 gradient: the correlation operator must be Coulomb, erfc or terfc \
                 (got {:?})",
                c.op.kind
            )));
        }
        if !(c.scaling.c_os.is_finite() && c.scaling.c_ss.is_finite()) {
            return Err(FerricError::General(format!(
                "U-RI-MP2 gradient: the correlation scaling must be finite (got c_os = {}, \
                 c_ss = {})",
                c.scaling.c_os, c.scaling.c_ss
            )));
        }
    }
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "U-RI-MP2 gradient: {nelec} electrons with multiplicity {} is not a spin state",
            mol.multiplicity
        )));
    }
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    if nocc_b == 0 {
        return Err(FerricError::General(
            "U-RI-MP2 gradient: no beta electron, so there is no correlation energy to \
             differentiate; use the UHF gradient"
                .into(),
        ));
    }
    if nocc_a >= obs.nbasis() {
        return Err(FerricError::General(
            "U-RI-MP2 gradient: no alpha virtual orbital".into(),
        ));
    }
    // The Lagrangian assumes D_σ = C_occ C_occᵀ of the first nocc_σ MOs (an
    // aufbau UHF); a MOM / non-aufbau reference has another occupied set.
    let db = uhf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("U-RI-MP2 gradient: no beta density".into()))?;
    for (label, c, d, n) in [
        ("alpha", uhf.mos_a(), &uhf.density_alpha, nocc_a),
        ("beta", uhf.mos_b(), db, nocc_b),
    ] {
        let co = c.slice(s![.., ..n]);
        let proj = co.dot(&co.t());
        let dev = max_abs(&(&proj - d));
        if dev > 1e-6 * max_abs(d).max(1.0) {
            return Err(FerricError::General(format!(
                "U-RI-MP2 gradient: the {label} density is not the projector onto the first \
                 {n} {label} MOs (max|D - C_occ C_occ^T| = {dev:.3e}); a non-aufbau (MOM) \
                 reference is not supported"
            )));
        }
    }
    Ok((nocc_a, nocc_b))
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// The per-spin energy weights `x^α`, `x^β` (`(naux, nocc_σ·nvir_σ)`):
///
/// ```text
///   x^α_{P,ia} = Σ_jb t^αα_{ij,ab} B^α_{P,jb} + Σ_JB t^αβ_{iJ,aB} B^β_{P,JB}
///   x^β_{P,JB} = Σ_KC t^ββ_{JK,BC} B^β_{P,KC} + Σ_ia t^αβ_{iJ,aB} B^α_{P,ia}
/// ```
///
/// so `∂E_MP2/∂B^σ = 2 x^σ` at fixed orbital energies. One GEMM pair per
/// occupied orbital; each writes its own column band, so the result does not
/// depend on the number of workers.
pub(crate) fn u_energy_weights(amps: &UMp2Amplitudes) -> (Array2<f64>, Array2<f64>) {
    u_energy_weights_impl(amps, None)
}

/// [`u_energy_weights`] of `c_ss · (E_αα + E_ββ) + c_os · E_αβ`: the
/// same-spin products carry `c_ss`, the opposite-spin ones `c_os`.
fn u_energy_weights_scaled(
    amps: &UMp2Amplitudes,
    c_ss: f64,
    c_os: f64,
) -> (Array2<f64>, Array2<f64>) {
    u_energy_weights_impl(amps, Some((c_ss, c_os)))
}

fn u_energy_weights_impl(
    amps: &UMp2Amplitudes,
    weights: Option<(f64, f64)>,
) -> (Array2<f64>, Array2<f64>) {
    use rayon::prelude::*;
    // `x = w_ss · B M_ssᵀ + w_os · B̄ N_osᵀ`; unweighted, the plain sum.
    let combine = |ss: Array2<f64>, os: Array2<f64>| match weights {
        None => ss + &os,
        Some((w_ss, w_os)) => {
            let mut x = ss * w_ss;
            x.scaled_add(w_os, &os);
            x
        }
    };
    let (ba, bb) = (&amps.inter_a.b_ov, &amps.inter_b.b_ov);
    let (oa, _, va, _) = amps.t_aa.dim();
    let (ob, _, vb, _) = amps.t_bb.dim();
    let naux = ba.nrows();
    let t_aa = &amps.t_aa;
    let t_bb = &amps.t_bb;
    let t_ab = &amps.t_ab;
    let mut x_a = Array2::<f64>::zeros((naux, oa * va));
    let mut x_b = Array2::<f64>::zeros((naux, ob * vb));
    if va > 0 {
        x_a.axis_chunks_iter_mut(Axis(1), va)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut slot)| {
                // M[a, (j, b)] = t_aa[i, j, a, b]; N[a, (J, B)] = t_ab[i, J, a, B].
                let m = t_aa
                    .slice(s![i, .., .., ..])
                    .permuted_axes([1, 0, 2])
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order((va, oa * va))
                    .expect("standard layout");
                let n = t_ab
                    .slice(s![i, .., .., ..])
                    .permuted_axes([1, 0, 2])
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order((va, ob * vb))
                    .expect("standard layout");
                let x = combine(ba.dot(&m.t()), bb.dot(&n.t()));
                slot.assign(&x);
            });
    }
    if vb > 0 {
        x_b.axis_chunks_iter_mut(Axis(1), vb)
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut slot)| {
                // M[B, (K, C)] = t_bb[J, K, B, C]; N[B, (i, a)] = t_ab[i, J, a, B].
                let m = t_bb
                    .slice(s![j, .., .., ..])
                    .permuted_axes([1, 0, 2])
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order((vb, ob * vb))
                    .expect("standard layout");
                let n = t_ab
                    .slice(s![.., j, .., ..])
                    .permuted_axes([2, 0, 1])
                    .as_standard_layout()
                    .into_owned()
                    .into_shape_with_order((vb, oa * va))
                    .expect("standard layout");
                let x = combine(bb.dot(&m.t()), ba.dot(&n.t()));
                slot.assign(&x);
            });
    }
    (x_a, x_b)
}

/// The unrelaxed MO correlation density of one spin, `(nmo, nmo)`: the
/// symmetrized `P^σ_oo` on the active occupied block, `P^σ_vv` on the virtual
/// block and the frozen core–active block
/// [`crate::zvector::frozen_core_block`] of `imat`.
fn unrelaxed_density_mo(
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    imat: &Array2<f64>,
    so: &SpinOrbitals<'_>,
) -> Result<Array2<f64>, FerricError> {
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = so.orb;
    let mut d = Array2::<f64>::zeros((so.nmo(), so.nmo()));
    for i in 0..nocc {
        for j in 0..nocc {
            d[(first_occ + i, first_occ + j)] = 0.5 * (p_oo[(i, j)] + p_oo[(j, i)]);
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            d[(nocc_total + a, nocc_total + b)] = 0.5 * (p_vv[(a, b)] + p_vv[(b, a)]);
        }
    }
    let dco = crate::zvector::frozen_core_block(imat, so.eps, &so.orb)?;
    for k in 0..first_occ {
        for i in 0..nocc {
            d[(k, first_occ + i)] = dco[(k, i)];
            d[(first_occ + i, k)] = dco[(k, i)];
        }
    }
    Ok(d)
}

/// The converged coupled Z-vector and the J/K of the relaxed correlation
/// densities it closes over.
struct UZvector {
    /// `z^α`, `z^β`, `(nvir_σ, nocc_total_σ)`.
    z: [Array2<f64>; 2],
    /// `(J, K)` of `P^α_rel`, `P^β_rel` in AO.
    corr_jk: [(Array2<f64>, Array2<f64>); 2],
}

/// `(C_σᵀ veff^σ[X] C_σ)` virtual–occupied block over every occupied orbital,
/// with `veff^σ[X] = J[X_α + X_β] − K[X_σ]` from the J/K of each spin's `X`.
fn veff_vo(jk: &[(Array2<f64>, Array2<f64>); 2], so: [&SpinOrbitals<'_>; 2]) -> [Array2<f64>; 2] {
    let j_t = &jk[0].0 + &jk[1].0;
    [0, 1].map(|s| {
        let v = &j_t - &jk[s].1;
        let o = so[s].orb;
        so[s]
            .to_mo(&v)
            .slice(s![o.nocc_total.., ..o.nocc_total])
            .to_owned()
    })
}

/// `D^z_σ = C_v z^σ C_oᵀ + h.c.` over every occupied orbital.
fn response_density(so: &SpinOrbitals<'_>, z: &Array2<f64>) -> Array2<f64> {
    let t = with_blas_threads(opt_in_blas_threads(), || {
        so.c_vir().dot(z).dot(&so.c_occ_all().t())
    });
    &t + &t.t()
}

/// Solve the coupled UHF Z-vector
/// `(ε_a − ε_i) z^σ + (C_σᵀ veff^σ[D^z] C_σ)_vo = rhs^σ` with DIIS on the
/// pair, every J/K on `jk`. An unconverged solve is an error.
#[allow(clippy::too_many_arguments)]
fn solve_u_zvector(
    jk: &ResponseJk<'_>,
    so: [&SpinOrbitals<'_>; 2],
    rhs: [Array2<f64>; 2],
    corr_jk_unrelaxed: [(Array2<f64>, Array2<f64>); 2],
    budget_bytes: usize,
    controls: ZvectorControls,
    acc: &mut PhaseAccumulator,
) -> Result<UZvector, FerricError> {
    let gap = |s: usize| {
        let o = so[s].orb;
        let nv = so[s].nmo() - o.nocc_total;
        Array2::from_shape_fn((nv, o.nocc_total), |(a, i)| {
            so[s].eps[o.nocc_total + a] - so[s].eps[i]
        })
    };
    let gaps = [gap(0), gap(1)];
    if gaps
        .iter()
        .any(|g| g.iter().any(|&x| x.is_nan() || x <= 1e-12))
    {
        return Err(FerricError::General(
            "U-RI-MP2 Z-vector: a non-positive orbital-energy gap ε_a − ε_i; the reference \
             is not an aufbau UHF minimum"
                .into(),
        ));
    }
    let pool = jk.engine_pool(1e-14)?;
    let band = ferric_scf::reduce::resolve_band_bytes(budget_bytes);
    // `H z` coupling term and the J/K of the unscaled D^z. The pair is first
    // scaled by the power of two that brings max|z| into (½, 1], so the
    // four-centre screening sees a unit-scale density (the reason
    // `zvector::compute_az_product` does the same); the scale is exact.
    let coupling = |z: &[Array2<f64>; 2]| -> Result<
        ([Array2<f64>; 2], [(Array2<f64>, Array2<f64>); 2]),
        FerricError,
    > {
        let zmax = max_abs(&z[0]).max(max_abs(&z[1]));
        let n = so[0].c.nrows();
        let Some((scale, inv)) = ferric_scf::rhf::pow2_normalizer(zmax) else {
            let zero = || (Array2::zeros((n, n)), Array2::zeros((n, n)));
            return Ok(([z[0].clone(), z[1].clone()], [zero(), zero()]));
        };
        let mut jks = Vec::with_capacity(2);
        for s in 0..2 {
            let dz = response_density(so[s], &(inv * &z[s]));
            let (j, k) = jk.jk_with_pool(&dz, 1e-12, pool.as_ref(), band)?;
            jks.push((scale * &j, scale * &k));
        }
        let jks: [(Array2<f64>, Array2<f64>); 2] = [jks.remove(0), jks.remove(0)];
        Ok((veff_vo(&jks, so), jks))
    };

    let mut z = [&rhs[0] / &gaps[0], &rhs[1] / &gaps[1]];
    let mut diis = Diis::new(8);
    let ZvectorControls { max_iter, tol } = controls;
    let tok = acc.begin("u-zvec: CPHF iterations (J/K)");
    let mut max_resid = f64::INFINITY;
    for _ in 0..max_iter {
        let (az, jkz) = coupling(&z)?;
        let r = [0, 1].map(|s| &rhs[s] - &(&gaps[s] * &z[s]) - &az[s]);
        max_resid = max_abs(&r[0]).max(max_abs(&r[1]));
        if crate::zvector::zvec_trace() {
            eprintln!("  [u-zvec] max_resid={max_resid:.3e}");
        }
        if max_resid < tol {
            acc.stop(tok);
            let [(ja, ka), (jb, kb)] = corr_jk_unrelaxed;
            let [(jza, kza), (jzb, kzb)] = jkz;
            return Ok(UZvector {
                z,
                corr_jk: [(ja + &jza, ka + &kza), (jb + &jzb, kb + &kzb)],
            });
        }
        let z_new = [0, 1].map(|s| &(&rhs[s] - &az[s]) / &gaps[s]);
        let (za, zb) = diis.step_pair(&z_new[0], &z_new[1], &r[0], &r[1]);
        z = [za, zb];
    }
    acc.stop(tok);
    Err(FerricError::Convergence(format!(
        "U-RI-MP2 Z-vector did not converge: max |r| = {max_resid:.3e} after {max_iter} \
         iterations (tolerance {tol:.1e}); the analytic gradient needs a converged orbital \
         response"
    )))
}

/// The analytical unrestricted RI-MP2 nuclear gradient `d(E_UHF + E_UMP2)/dR`
/// on the converged UHF `uhf`, `(natoms, 3)` in Hartree/Bohr.
///
/// The reference's J/K route is read from `uhf` (four-centre, or RI-J + RI-K
/// on one auxiliary basis, `scf_jk = "rijk"`): the Z-vector, the
/// relaxed-density Lagrangian and the two-electron derivative are all built on
/// it, so the gradient is that of the energy the SCF and
/// [`crate::u_rimp2::u_ri_mp2`] compute. Frozen core (`config.frozen_core`,
/// the same count of each spin) is supported as in the closed-shell gradient.
/// `ext`, when `Some`, adds the external-potential Hellmann-Feynman term with
/// the relaxed density and the classical charge–nuclear term; the reference
/// must have been solved with the same potential.
///
/// # Errors
///
/// A ROHF/ROKS reference ([`ROHF_REFUSAL`]); a restricted, unconverged,
/// non-aufbau or X2C reference; a non-Coulomb operator; no beta electron;
/// a COSX or otherwise unsupported J/K route; a Z-vector that misses
/// `max |r| < 1e-8` in 50 iterations; any integral error.
#[allow(clippy::too_many_arguments)]
pub fn u_rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    let mut acc = PhaseAccumulator::new().report_on_drop("U-RI-MP2 gradient");
    u_rimp2_gradient_timed(
        mol,
        obs,
        dfbs,
        op,
        bounds,
        uhf,
        config,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

#[allow(clippy::too_many_arguments)]
fn u_rimp2_gradient_timed(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    controls: ZvectorControls,
    acc: &mut PhaseAccumulator,
) -> Result<Array2<f64>, FerricError> {
    if op.kind != OperatorKind::Coulomb {
        return Err(FerricError::General(format!(
            "U-RI-MP2 gradient: only the Coulomb operator is differentiated (got {:?}); the \
             erfc-attenuated gradient is ferric_mp2::att_rimp2_gradient",
            op.kind
        )));
    }
    let corr = UCorrelation::uniform(op, 1.0);
    u_correlated_gradient_timed(
        mol, obs, dfbs, corr, bounds, uhf, config, ext, controls, acc,
    )
}

/// `d(E_UHF + E_c[corr])/dR` on the converged Coulomb UHF `uhf`: the
/// U-RI-MP2 Lagrangian with the operators split as in the closed-shell
/// attenuated gradient
/// (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-attenuated-erfc-variant`).
/// `corr.op` builds the amplitudes,
/// `b_full` of `Imat^σ` and the 3c/2c RI response; `bounds.op` (Coulomb) the
/// Z-vector Hessian, the `veff` of the correlation density and the
/// two-electron derivative.
#[allow(clippy::too_many_arguments)]
pub(crate) fn u_correlated_gradient_timed(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    corr: UCorrelation,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    controls: ZvectorControls,
    acc: &mut PhaseAccumulator,
) -> Result<Array2<f64>, FerricError> {
    u_terms_gradient_timed(
        mol,
        obs,
        dfbs,
        &[corr],
        bounds,
        uhf,
        config,
        ext,
        controls,
        acc,
    )
}

/// One spin-resolved correlation term's contribution to the Lagrangian:
/// `Imat^σ`, the unrelaxed `P^σ_oo`, `P^σ_vv` (index `[α, β]`) and the 3c/2c
/// RI response.
struct UTermParts {
    imat: [Array2<f64>; 2],
    p_oo: [Array2<f64>; 2],
    p_vv: [Array2<f64>; 2],
    g_ri: Array2<f64>,
}

impl UTermParts {
    fn add(self, o: UTermParts) -> UTermParts {
        let [ia, ib] = self.imat;
        let [oa, ob] = self.p_oo;
        let [va, vb] = self.p_vv;
        UTermParts {
            imat: [ia + &o.imat[0], ib + &o.imat[1]],
            p_oo: [oa + &o.p_oo[0], ob + &o.p_oo[1]],
            p_vv: [va + &o.p_vv[0], vb + &o.p_vv[1]],
            g_ri: self.g_ri + &o.g_ri,
        }
    }
}

/// `d(E_UHF + Σ_k E_c[corr_k])/dR`: each term builds its own amplitudes,
/// `Imat^σ`, unrelaxed densities and RI response on its operator and with
/// its spin scaling, one term resident at a time; the sums close ONE coupled
/// Z-vector and one relaxed Lagrangian
/// (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-terfc-scs-mp2-2terfc`).
/// Every first
/// variation of `E_c` is linear in `x^σ` and the unrelaxed densities, so
/// the sums are exact.
#[allow(clippy::too_many_arguments)]
pub(crate) fn u_terms_gradient_timed(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    corr: &[UCorrelation],
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    controls: ZvectorControls,
    acc: &mut PhaseAccumulator,
) -> Result<Array2<f64>, FerricError> {
    use ferric_scf::gradient::{
        build_energy_weighted_density_uhf, ecp_gradient, oneelectron_gradient,
        overlap_deriv_contract,
    };
    let (nocc_a, nocc_b) = check_reference(mol, uhf, corr, bounds.op, obs)?;
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);

    let mut sum: Option<UTermParts> = None;
    let mut orbs: Option<[OrbitalSpace; 2]> = None;
    for &term in corr {
        let (parts, o) = u_term_parts(mol, obs, dfbs, term, uhf, config, acc)?;
        debug_assert!(orbs.is_none_or(|p| p == o), "one reference, one partition");
        orbs = Some(o);
        sum = Some(match sum {
            None => parts,
            Some(acc_parts) => acc_parts.add(parts),
        });
    }
    let (Some(parts), Some(orbs)) = (sum, orbs) else {
        return Err(FerricError::General(
            "U-RI-MP2 gradient: no correlation term".into(),
        ));
    };
    let UTermParts {
        imat,
        p_oo,
        p_vv,
        g_ri,
    } = parts;
    let spins = [
        SpinOrbitals {
            c: uhf.mos_a(),
            eps: uhf.eps_a(),
            orb: orbs[0],
        },
        SpinOrbitals {
            c: uhf.mos_b(),
            eps: uhf.eps_b(),
            orb: orbs[1],
        },
    ];
    debug_assert_eq!(spins[0].orb.nocc_total, nocc_a);
    debug_assert_eq!(spins[1].orb.nocc_total, nocc_b);
    let so = [&spins[0], &spins[1]];
    let ctx = ferric_core::parallel::ParallelContext::default();
    let jk = acc.time("J/K route setup", || {
        ResponseJk::for_scf(&ctx, mol, obs, bounds, uhf)
    })?;

    // Unrelaxed MO correlation densities and their J/K.
    let p_mo = [
        unrelaxed_density_mo(&p_oo[0], &p_vv[0], &imat[0], &spins[0])?,
        unrelaxed_density_mo(&p_oo[1], &p_vv[1], &imat[1], &spins[1])?,
    ];
    let rhs_tok = acc.begin("u-zvec: rhs (J/K of P^σ)");
    let p_ao = [spins[0].to_ao(&p_mo[0]), spins[1].to_ao(&p_mo[1])];
    let corr_jk = [jk.jk(&p_ao[0], 1e-12)?, jk.jk(&p_ao[1], 1e-12)?];
    let v = veff_vo(&corr_jk, so);
    let rhs = [0, 1].map(|s| {
        let o = spins[s].orb;
        let nv = spins[s].nmo() - o.nocc_total;
        Array2::from_shape_fn((nv, o.nocc_total), |(a, i)| {
            let a_mo = o.nocc_total + a;
            -(v[s][(a, i)] + imat[s][(i, a_mo)] - imat[s][(a_mo, i)])
        })
    });
    acc.stop(rhs_tok);
    let zv = solve_u_zvector(&jk, so, rhs, corr_jk, budget_bytes, controls, acc)?;
    let lag_tok = acc.begin("lagr: densities, im1, zeta, vhf_s1occ");
    let mut p_rel_ao = Vec::with_capacity(2);
    let mut w_overlap = Array2::<f64>::zeros((obs.nbasis(), obs.nbasis()));
    let j_t = &zv.corr_jk[0].0 + &zv.corr_jk[1].0;
    for s in 0..2 {
        let sp = &spins[s];
        let o = sp.orb;
        let nmo = sp.nmo();
        // Relaxed MO density: z on both occupied–virtual blocks.
        let mut p = p_mo[s].clone();
        for a in 0..nmo - o.nocc_total {
            for i in 0..o.nocc_total {
                p[(o.nocc_total + a, i)] = zv.z[s][(a, i)];
                p[(i, o.nocc_total + a)] = zv.z[s][(a, i)];
            }
        }
        // im1: Imat with the virtual–occupied block := occupied–virtual ᵀ.
        let mut ip = imat[s].clone();
        for a in o.nocc_total..nmo {
            for i in 0..o.nocc_total {
                ip[(a, i)] = imat[s][(i, a)];
            }
        }
        // ζ ⊙ P: ½(ε_p + ε_q) on oo/vv, ε_i on the occupied–virtual blocks.
        let eps = sp.eps;
        let mut zeta =
            Array2::from_shape_fn((nmo, nmo), |(p_, q)| 0.5 * (eps[p_] + eps[q]) * p[(p_, q)]);
        for a in o.nocc_total..nmo {
            for i in 0..o.nocc_total {
                zeta[(a, i)] = eps[i] * p[(a, i)];
                zeta[(i, a)] = eps[i] * p[(i, a)];
            }
        }
        // P_occ veff^σ[P_rel] P_occ.
        let veff = &j_t - &zv.corr_jk[s].1;
        let co = sp.c_occ_all();
        let p_occ = co.dot(&co.t());
        let vhf_s1occ = p_occ.dot(&veff).dot(&p_occ);
        w_overlap += &(&sp.to_ao(&(&ip - &zeta)) - &vhf_s1occ);
        p_rel_ao.push(sp.to_ao(&p));
    }
    // − W_HF = − Σ_σ C_occ ε C_occᵀ.
    w_overlap -= &build_energy_weighted_density_uhf(uhf, obs, nocc_a, nocc_b)?;
    let d_a = &uhf.density_alpha;
    let d_b = uhf
        .density_beta
        .as_ref()
        .ok_or_else(|| FerricError::General("U-RI-MP2 gradient: no beta density".into()))?;
    let dm1_total = d_a + d_b + &p_rel_ao[0] + &p_rel_ao[1];
    acc.stop(lag_tok);

    let one_tok = acc.begin("lagr: 1e + ECP + overlap derivs");
    let zero_w = Array2::<f64>::zeros((obs.nbasis(), obs.nbasis()));
    let mut grad = oneelectron_gradient(mol, obs, &dm1_total, &zero_w, ext)?;
    grad += &ecp_gradient(mol, obs, &dm1_total)?;
    grad += &overlap_deriv_contract(obs, &w_overlap)?;
    acc.stop(one_tok);

    let dpa = d_a + &(2.0 * &p_rel_ao[0]);
    let dpb = d_b + &(2.0 * &p_rel_ao[1]);
    grad += &acc.time("lagr: 2e derivs", || {
        jk.two_electron_gradient_bilinear_open(mol, bounds.op, (d_a, d_b), (&dpa, &dpb))
    })?;

    grad += &g_ri;
    Ok(grad)
}

/// One term's [`UTermParts`] and the per-spin orbital partition: the
/// amplitudes on `term.op`, the energy weights `x^σ` and unrelaxed densities
/// with the term's spin scaling, `Imat^σ` and the 3c/2c RI response.
fn u_term_parts(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    term: UCorrelation,
    uhf: &ScfResult,
    config: &RiMp2Config,
    acc: &mut PhaseAccumulator,
) -> Result<(UTermParts, [OrbitalSpace; 2]), FerricError> {
    let op = term.op;
    let amps = acc.time("amplitudes (b_ov, t_aa/t_bb/t_ab)", || {
        compute_u_mp2_amplitudes(mol, obs, dfbs, op, uhf, config)
    })?;
    let spins = [
        SpinOrbitals {
            c: uhf.mos_a(),
            eps: uhf.eps_a(),
            orb: amps.inter_a.orbital_space(),
        },
        SpinOrbitals {
            c: uhf.mos_b(),
            eps: uhf.eps_b(),
            orb: amps.inter_b.orbital_space(),
        },
    ];

    // The per-spin energy weights, held to the RI response at the end.
    let naux = amps.inter_a.naux;
    let ov = |o: &OrbitalSpace| o.nocc * o.nvir;
    let _x_charge = crate::rimp2::charge_mo_side(
        "U-RI-MP2 gradient x^α/x^β",
        naux.saturating_mul(ov(&spins[0].orb) + ov(&spins[1].orb))
            .saturating_mul(8),
    )?;
    let SpinScaling { c_os, c_ss } = term.scaling;
    let (x_a, x_b, dens) = if c_os == c_ss {
        // `s · E_c`: every first variation of E_c is linear in x^σ and the
        // unrelaxed densities (Imat^σ, the frozen-core block and the RI
        // response are built from them), so scaling those two scales the
        // whole response.
        let mut dens = acc.time("unrelaxed densities", || build_u_mp2_density(&amps));
        let (mut x_a, mut x_b) = acc.time("energy weights x^σ", || u_energy_weights(&amps));
        let s = c_os;
        x_a *= s;
        x_b *= s;
        for p in [
            &mut dens.p_oo_a,
            &mut dens.p_vv_a,
            &mut dens.p_oo_b,
            &mut dens.p_vv_b,
        ] {
            *p *= s;
        }
        (x_a, x_b, dens)
    } else {
        // `c_ss (E_αα + E_ββ) + c_os E_αβ`: x^σ and the densities split into
        // same-spin and opposite-spin products with no cross term, each
        // carrying its own coefficient
        // (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-terfc-scs-mp2-2terfc`).
        let dens = acc.time("unrelaxed densities", || {
            build_u_mp2_density_scaled(&amps, c_ss, c_os)
        });
        let (x_a, x_b) = acc.time("energy weights x^σ", || {
            u_energy_weights_scaled(&amps, c_ss, c_os)
        });
        (x_a, x_b, dens)
    };

    // Imat^σ = ½ build_imat_ri(x^σ, b^σ_full): the closed-shell builder's
    // factor −2 is the 2-PDM weight 2·(2t − t̄) of both spins; one spin with
    // its energy weight carries half of it
    // (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-energy-weights`).
    let imat = {
        let tok = acc.begin("Imat^σ (b_full per spin)");
        let ao = crate::oo_rimp2::OoRiMp2AoTensors::build_with_budget(
            obs,
            dfbs,
            op,
            crate::rimp2::eri3_budget_bytes(config.memory_budget_bytes),
        )?;
        let mut out = Vec::with_capacity(2);
        for (s, x) in [(0usize, &x_a), (1, &x_b)] {
            let b_full = crate::oo_rimp2::compute_b_full_mo_with(&ao, spins[s].c)?;
            out.push(0.5 * crate::zvector::build_imat_ri(x, &b_full, &spins[s].orb));
        }
        acc.stop(tok);
        [out.remove(0), out.remove(0)]
    };

    // RI 3c/2c response: one channel per spin, weight 1.
    let f = &amps.inter_a.v_inv_sqrt;
    let y = [&x_a, &x_b].map(|x| with_blas_threads(opt_in_blas_threads(), || f.t().dot(x)));
    let channels = [
        RiResponseChannel {
            y_ov: &y[0],
            b_ov: &amps.inter_a.b_ov,
            c_occ: spins[0].c_occ_active(),
            c_vir: spins[0].c_vir(),
            weight: 1.0,
        },
        RiResponseChannel {
            y_ov: &y[1],
            b_ov: &amps.inter_b.b_ov,
            c_occ: spins[1].c_occ_active(),
            c_vir: spins[1].c_vir(),
            weight: 1.0,
        },
    ];
    let g_ri = ri_response_gradient_channels(mol, obs, dfbs, op, f, &channels, acc)?;
    let orbs = [spins[0].orb, spins[1].orb];
    let crate::u_rimp2::UMp2Density {
        p_oo_a,
        p_vv_a,
        p_oo_b,
        p_vv_b,
    } = dens;
    Ok((
        UTermParts {
            imat,
            p_oo: [p_oo_a, p_oo_b],
            p_vv: [p_vv_a, p_vv_b],
            g_ri,
        },
        orbs,
    ))
}

/// The total U-RI-MP2 energy `E_UHF + E_UMP2` ([`u_ri_mp2`]) and its analytic
/// gradient ([`u_rimp2_gradient_analytical`]) at one geometry, the UHF on the
/// J/K route `scf_jk` (`"exact"` or `"rijk"`; `"rijcosx"` is an error) and
/// warm-started through `warm`. The UHF is solved at `density_conv = 1e-9`
/// with the molecule's charge and multiplicity. On the fitted route the
/// declared peak memory ([`crate::gradient_memory::preflight_fitted_gradient_unrestricted`])
/// is checked against `mp2_config.memory_budget_bytes` before any integral.
#[allow(clippy::too_many_arguments)]
pub fn total_u_rimp2_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let rhf_cfg = RhfConfig::gradient_reference(ext.cloned());
    total_u_rimp2_gradient_with(
        mol, obs_basis, aux_basis, op, mp2_config, ext, scf_jk, warm, rhf_cfg,
    )
}

/// [`total_u_rimp2_gradient_on`] with the UHF solved on `scf_base` (before the
/// geometry-task and `scf_jk` adjustments): the finite-difference tests use it
/// to converge every displaced reference far below the default.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn total_u_rimp2_gradient_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
    scf_base: RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    let (uhf, obs, dfbs, bounds) = solve_reference(
        mol, obs_basis, aux_basis, op, mp2_config, scf_jk, warm, scf_base,
    )?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-RI-MP2 energy + gradient");
    let energy = acc.time("u_ri_mp2 energy", || {
        u_ri_mp2(mol, &obs, &dfbs, op, &uhf, mp2_config).map(|r| r.total_energy)
    })?;
    let grad = u_rimp2_gradient_timed(
        mol,
        &obs,
        &dfbs,
        op,
        &bounds,
        &uhf,
        mp2_config,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )?;
    Ok((energy, grad))
}

/// The total U-RI-MP2 energy at one geometry on the reference of
/// [`total_u_rimp2_gradient_with`] (same SCF settings and route), without
/// the gradient: the finite-difference reference of the tests.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn total_u_rimp2_energy_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    scf_jk: &ScfJk,
    scf_base: RhfConfig,
) -> Result<(f64, ScfResult), FerricError> {
    let (uhf, obs, dfbs, _) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        op,
        mp2_config,
        scf_jk,
        &mut ScfWarmStart::new(false),
        scf_base,
    )?;
    let e = u_ri_mp2(mol, &obs, &dfbs, op, &uhf, mp2_config)?.total_energy;
    Ok((e, uhf))
}

/// The UHF reference of the total-gradient drivers, on the operator `op`
/// (Coulomb) of its Schwarz bounds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_reference(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
    scf_base: RhfConfig,
) -> Result<(ScfResult, PreparedBasis, PreparedBasis, SchwarzBounds), FerricError> {
    if scf_jk.mode == ScfJkMode::RiJCosx {
        return Err(FerricError::General(
            "U-RI-MP2 gradient: scf_jk = \"rijcosx\" is not supported (no COSX exchange \
             response); use \"exact\" or \"rijk\""
                .into(),
        ));
    }
    let ctx = ferric_core::parallel::ParallelContext::default();
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let mut cfg = scf_base.for_geometry_task();
    scf_jk.apply(&mut cfg);
    // The route's J/K digest: the SCF records it on `bounds`, which the
    // response builds reuse.
    cfg.jk_digest = scf_jk.jk_digest;
    if let (ScfJkMode::RiJk, Some(aux)) = (scf_jk.mode, scf_jk.aux.as_deref()) {
        let budget = ferric_core::memory::resolve_budget_bytes(mp2_config.memory_budget_bytes);
        let shape = crate::gradient_memory::UnrestrictedFittedGradientShape::new(
            mol,
            &obs,
            &dfbs,
            aux,
            mp2_config.frozen_core,
        )?;
        crate::gradient_memory::preflight_fitted_gradient_unrestricted(shape, budget)?;
        cfg.three_index_budget_bytes = mp2_config.memory_budget_bytes.unwrap_or(0);
    }
    let bounds = SchwarzBounds::compute(op, &obs)?;
    let uhf = solve_uhf_warm(&ctx, mol, &obs, &bounds, &cfg, warm)?;
    if !uhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: uhf.iterations,
            last_energy: uhf.energy,
        });
    }
    Ok((uhf, obs, dfbs, bounds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rimp2::SpinScaling;
    use ferric_core::basis;
    use ferric_scf::rhf::solve_rhf;

    const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";

    /// Tight Z-vector controls: the anchor compares two solves, so each must
    /// sit far below the comparison bar.
    const TIGHT: ZvectorControls = ZvectorControls {
        max_iter: 200,
        tol: 1e-12,
    };

    /// The RHF `rhf` written as the unrestricted result with identical α and β
    /// orbitals, densities and Fock matrices: a UHF stationary point.
    fn lift_to_unrestricted(rhf: &ScfResult) -> ScfResult {
        let mut u = rhf.clone();
        u.spin = Spin::Unrestricted;
        u.density_beta = Some(rhf.density_alpha.clone());
        u.mos_beta = Some(rhf.mos_alpha.clone());
        u.eps_beta = Some(rhf.eps_alpha.clone());
        u.fock_beta = Some(rhf.fock_alpha.clone());
        u
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        max_abs(&(a - b))
    }

    /// EXACTNESS ANCHOR. A closed-shell reference written as a UHF (α ≡ β)
    /// has `P^α = P^β = P/2`, `Imat^σ = Imat/2`, `z^σ = z/2`, and every term
    /// of the unrestricted Lagrangian sums to the restricted one
    /// (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-closed-shell-limit`).
    /// Same orbitals on both sides, so the only
    /// differences are (i) operation order, and (ii) the two Z-vector
    /// residuals, each `< 1e-12`. The gradient error from a residual `r` is
    /// `≤ max|∂_R Xvo| · ‖A⁻¹‖ · r ≲ 10 · (1/Δε_min ≈ 2) · 1e-12 = 2e-11`;
    /// operation-order rounding is `O(ε · Σ|terms|)`, with the terms of order
    /// 10 Ha/Bohr at water/6-31G, `≈ 1e-14`. Bar 1e-10 Ha/Bohr.
    fn closed_limit(route: Option<&str>, frozen_core: usize) -> f64 {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let mut cfg = RhfConfig::default()
            .with_density_conv(1e-11)
            .for_geometry_task();
        if let Some(aux) = route {
            ScfJk::rijk(aux).apply(&mut cfg);
        }
        let ctx = ferric_core::parallel::ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
        assert!(rhf.converged);
        assert_eq!(rhf.df_jk.is_some(), route.is_some(), "route not recorded");
        // density_alpha of a restricted result is half the total density.
        assert!(max_abs_diff(&(2.0 * &rhf.density_alpha), &rhf.density_total) < 1e-14);
        let uhf = lift_to_unrestricted(&rhf);
        let config = RiMp2Config {
            frozen_core,
            ..Default::default()
        };
        let g_r = crate::gradient::rimp2_gradient_analytical_timed(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &config,
            SpinScaling::MP2,
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let g_u = u_rimp2_gradient_timed(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &uhf,
            &config,
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let d = max_abs_diff(&g_r, &g_u);
        eprintln!("closed limit route={route:?} fc={frozen_core}: max|g_U - g_R| = {d:.3e}, max|g| = {:.3e}", max_abs(&g_r));
        d
    }

    const CLOSED_LIMIT_BAR: f64 = 1e-10;

    #[test]
    fn closed_shell_limit_matches_restricted_four_centre() {
        let d = closed_limit(None, 0);
        assert!(d < CLOSED_LIMIT_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    #[test]
    fn closed_shell_limit_matches_restricted_four_centre_frozen_core() {
        let d = closed_limit(None, 1);
        assert!(d < CLOSED_LIMIT_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    #[test]
    fn closed_shell_limit_matches_restricted_rijk() {
        let d = closed_limit(Some("def2-universal-jkfit"), 0);
        assert!(d < CLOSED_LIMIT_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    /// The energy weights reduce to the closed-shell `x_ov` in the α ≡ β
    /// limit, and the unrestricted energy to the restricted one: guards the
    /// t_ab index convention independently of the gradient.
    #[test]
    fn energy_weights_at_alpha_equals_beta_are_the_closed_weights() {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default().with_density_conv(1e-10),
        )
        .unwrap();
        let uhf = lift_to_unrestricted(&rhf);
        let config = RiMp2Config::default();
        let amps = compute_u_mp2_amplitudes(&mol, &obs, &dfbs, op, &uhf, &config).unwrap();
        let (xa, xb) = u_energy_weights(&amps);
        assert!(max_abs_diff(&xa, &xb) < 1e-13);
        let e_u = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &config).unwrap();
        let e_corr = e_u.total_energy - uhf.energy;
        let sum: f64 = (&amps.inter_a.b_ov * &xa).sum() + (&amps.inter_b.b_ov * &xb).sum();
        // Σ_σ B^σ · ∂E/∂B^σ = Σ_σ B^σ · 2x^σ = 4 E (E homogeneous of degree
        // 4 in B at fixed ε, since t is degree 2).
        let rel = (2.0 * sum - 4.0 * e_corr).abs() / e_corr.abs();
        assert!(
            rel < 1e-12,
            "Euler identity: 2Σ B·x = {:.12e}, 4E = {:.12e}",
            2.0 * sum,
            4.0 * e_corr
        );
    }

    #[test]
    fn rohf_reference_is_refused_with_the_reason() {
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let rohf =
            ferric_scf::rohf::solve_rohf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default())
                .unwrap();
        assert_eq!(rohf.spin, Spin::RestrictedOpen);
        let err = u_rimp2_gradient_analytical(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rohf,
            &RiMp2Config::default(),
            None,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("needs a UHF reference"), "{err}");
    }
}
