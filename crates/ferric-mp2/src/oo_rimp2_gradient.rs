//! Analytical nuclear gradients for OO-RI-MP2.
//!
//! ## Why this is SIMPLER than plain RI-MP2's gradient, not harder
//!
//! Plain (non-orbital-optimized) RI-MP2's analytical gradient
//! ([`crate::gradient::rimp2_gradient_analytical`]) needs a genuine CPHF/
//! Z-vector solve ([`crate::zvector::solve_zvector`]) because plain MP2's
//! orbitals are HF-stationary, not MP2-stationary: `dE_MP2/dkappa_ai != 0` in
//! general, so a first-order orbital-response correction `z_ai` is required
//! (the standard Lagrangian-multiplier trick — Pulay 1969 / Handy-Schaefer
//! 1984).
//!
//! OO-RI-MP2 converges orbitals to a stationary point of its energy in the
//! occ-vir directions, and the energy (the textbook full-Fock OMP2 functional,
//! see `oo_rimp2`'s module doc) is INVARIANT to occ-occ and vir-vir rotations.
//! Together these make the point stationary in EVERY orbital-rotation
//! direction, so the envelope theorem applies and there is no Z-vector:
//! the occ-vir block of the relaxed density is zero. (With the former
//! diag(CᵀFC) functional this argument FAILED: it was not invariant, its
//! converged point carried a ~1e-2 occ-occ/vir-vir gradient, and FD of its
//! re-converged energy disagreed with FD of the same energy along any fixed
//! orbital connection by 5.8e-4..8.2e-4. That was the leading part of the
//! 2.2e-3 / 8.7e-4 analytic-vs-FD residual this file used to carry.)
//!
//! ## What survives at z = 0 (all verified, defect F2 2026-09-24)
//!
//! `scripts/oo_mp2_stationarity_proto.py nuc` checks this term by term. It
//! perturbs S (resp. h) by `e·R`, re-converges OO-MP2, and compares the
//! central FD with `Σ R∘w` (resp. `Σ R∘γ`) for the densities assembled below:
//!
//! * 1-PDM `γ = D_HF + C·W·Cᵀ`, `W = P_oo+P_ooᵀ ⊕ P_vv+P_vvᵀ`: FD agreement
//!   1.3e-9 (H2) / 9.2e-10 (H2O).
//! * energy-weighted density `w = im1 − zeta − vhf_s1occ`: FD agreement
//!   6.4e-10 / 4.0e-9. The version shipped before F2 missed by 2.2e-4 / 3.0e-4
//!   on the textbook functional itself, for two reasons:
//!   - it DROPPED `vhf_s1occ = P_occ·G[2·C W Cᵀ]·P_occ` on the claim that it
//!     only cancels a CPHF piece. That was wrong. The MP2 energy depends on
//!     `F(D_HF)`, and `D_HF` moves with the occupied orbitals under the
//!     overlap connection, z-vector or not. This was 2.2e-4 / 3.2e-4 of the
//!     miss.
//!   - its occ-vir zeta block was `½(F·W + W·F)_ia`. At a stationary point
//!     the occ-vir Lagrangian must be taken in ONE representation, the one
//!     Imat uses (derivative w.r.t. the virtual orbital), which gives
//!     `(F·W)_ia`. The two differ where `F_ov != 0`, i.e. always after OO.
//!     This was 6e-6 / 3e-5 of the miss.
//!
//! The 2e-derivative (bilinear `Γ(D_HF, D_HF + 2·C W Cᵀ)`) and RI 3c/2c
//! pieces are the plain RI-MP2 ones: the OO energy depends on the ERIs and
//! on `B` exactly as plain RI-MP2 does.
//!
//! Frozen core: the core-active and core-virtual rotations are not optimised,
//! so with `frozen_core > 0` the point is NOT stationary in those directions
//! and this z = 0 gradient is not exact. That is unchanged by F2 and untested.

use crate::gradient::integral_response_gradient_3c2c;
use crate::oo_rimp2::{
    build_mp2_density, compute_b_full_mo, compute_t2_and_integrals, DensityJk, OoRiMp2Result,
    JK_THRESH,
};
use crate::rimp2::{active_occ, cholesky_inverse_sqrt, Mp2Intermediates};
use crate::zvector::{build_imat_ri, build_x_ov, x_ov_worker_bytes};
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::gradient::{ecp_gradient, oneelectron_gradient, overlap_deriv_contract};
use ferric_scf::response_jk::ResponseJk;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the analytical nuclear gradient for OO-RI-MP2.
///
/// See the module doc comment for the no-Z-vector derivation and its
/// numerical verification.
///
/// J/K route: the one `result` was optimised on (`result.jk_route`):
/// four-centre, or the reference SCF's RI-JK fit, whose J/K and bilinear
/// two-electron derivative come from [`ResponseJk`]. The two J/K builds here
/// (of the HF density and of `2·C W Cᵀ`) reuse `result.hf_jk` /
/// `result.response_jk` when their densities agree with the ones formed
/// here to [`REUSE_TOL`]; otherwise they are built.
///
/// With `FERRIC_TIMING` set, prints a phase table ("OO-RI-MP2 gradient").
pub fn oo_ri_mp2_gradient(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &OoRiMp2Result,
    frozen_core: usize,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() / 2) as usize;
    let nocc = active_occ(nocc_total, frozen_core)?;
    let first_occ = frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let nmo = nbas;
    let mut acc = PhaseAccumulator::new().report_on_drop("OO-RI-MP2 gradient");

    let ctx = ferric_core::parallel::ParallelContext::default();
    let jk = acc.time("J/K route setup", || {
        ResponseJk::for_route(&ctx, mol, obs, bounds, result.jk_route.as_ref())
    })?;
    // The z = 0 formula below needs the SEMICANONICAL frame; see
    // `semicanonical_frame`.
    let (f_ao, c_sc, eps_sc) = acc.time("semicanonical frame (J/K of D_HF)", || {
        semicanonical_frame(
            &jk,
            mol,
            obs,
            &result.mos,
            &result.hf_jk,
            nocc_total,
            first_occ,
            ext,
        )
    })?;
    let c = &c_sc;
    let eps = &eps_sc;

    let tok = acc.begin("b_full, t2, MP2 density");
    let b_full = compute_b_full_mo(obs, dfbs, op, c)?;
    let b_ov_3d = b_full.slice(ndarray::s![.., first_occ..nocc_total, nocc_total..]);
    let b_ov = b_ov_3d
        .to_owned()
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();

    let (t2, _) = compute_t2_and_integrals(&b_ov, eps, nocc, nvir, nocc_total, first_occ, naux);

    let (p_oo, p_vv) = build_mp2_density(&t2, nocc, nvir);
    acc.stop(tok);

    let orb = OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    };
    let dm1mo = relaxed_dm1_mo(&p_oo, &p_vv, &orb, nmo);

    let dm1_corr_ao = {
        let cp = c.dot(&dm1mo);
        cp.dot(&c.t())
    };
    // hf_dm1: the 2*C_occ*C_occ^T occupied-block piece, in AO, at the
    // (optimized) OO-MP2 orbitals — there is no separate "HF density" left
    // post-optimization; this is the occupied-identity contribution to the
    // total 1-PDM, same role plain RHF's density plays in plain MP2's
    // gradient (dm1_total = dm1_corr + hf_dm1).
    let c_occ_hf = c.slice(ndarray::s![.., ..nocc_total]);
    let hf_dm1 = c_occ_hf.dot(&c_occ_hf.t()) * 2.0;
    let dm1_total_ao = &dm1_corr_ao + &hf_dm1;

    // Fock matrix in the (semicanonical) MO frame. F_oo and F_vv are
    // diagonal there, but F_ov is NOT zero at an OO stationary point (only the
    // TOTAL occ-vir gradient vanishes), and the energy-weighted density below
    // needs F_ov explicitly.
    let f_mo = c.t().dot(&f_ao).dot(c);

    // --- Imat (RI-MP2 Lagrangian matrix); see `pulay_imat_ao`.
    let im1 = acc.time("x_ov and Imat", || {
        pulay_imat_ao(&t2, &b_ov, &b_full, &orb, naux, c)
    });

    let zeta_ao = energy_weighted_density(&f_mo, &dm1mo, c, nocc_total);
    let vhf_s1occ = acc.time("vhf_s1occ (J/K of 2·dm1_corr)", || {
        vhf_s1occ(
            &jk,
            &dm1_corr_ao,
            result.response_jk.as_ref(),
            c,
            nocc_total,
        )
    })?;

    // --- Assemble (same signs as plain RI-MP2's mp2_relaxed_lagrangian_gradient).
    let tok = acc.begin("1e, ECP and overlap derivatives");
    let zero_w = Array2::<f64>::zeros((nmo, nmo));
    let mut grad = oneelectron_gradient(mol, obs, &dm1_total_ao, &zero_w, ext)?;
    // ECP: V_ECP enters the energy as a plain one-electron operator (through
    // `hcore_ecp_with_external` in `oo_ri_mp2`), so its derivative is
    // Σ γ dV_ECP/dR with the same total (spin-summed, relaxed — z = 0 here)
    // 1-PDM as the T/V_nuc terms; `oneelectron_gradient` does not include it.
    // Returns zeros (zero work) for an all-electron basis.
    grad += &ecp_gradient(mol, obs, &dm1_total_ao)?;

    let w_overlap = &im1 - &zeta_ao - &vhf_s1occ;
    grad += &overlap_deriv_contract(obs, &w_overlap)?;
    acc.stop(tok);

    // 2e-integral-derivative: bilinear Gamma(hf_dm1, hf_dm1 + 2*dm1_corr),
    // same structure as plain RI-MP2's bilinear term, on the J/K route.
    let two_dm_corr = 2.0 * &dm1_corr_ao;
    let dm1p = &hf_dm1 + &two_dm_corr;
    grad += &acc.time("2e derivative (bilinear)", || {
        jk.two_electron_gradient_bilinear(mol, op, &hf_dm1, &dm1p)
    })?;

    // RI 3c/2c integral-response term: reuse plain RI-MP2's implementation by
    // packaging OO-MP2's own t2/b_ov/v_inv_sqrt into an Mp2Intermediates.
    let tok = acc.begin("RI 3c/2c response");
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    // `uncharged`, not a struct literal: these buffers were built by the
    // OO-MP2 driver above and (where charged at all) are charged by it, so
    // taking a second reservation here would double-debit the shared pool for
    // one allocation. `b_oo`/`b_vv` -- the only planes large enough for the
    // distinction to matter -- are `None` on this path.
    let inter = Mp2Intermediates::uncharged(
        t2,
        b_ov,
        None,
        None,
        v_inv_sqrt,
        p_oo,
        p_vv,
        ferric_core::orbitals::OrbitalSpace::new(nocc, nvir, nocc_total, first_occ),
        naux,
        result.mp2_corr,
    );
    grad += &integral_response_gradient_3c2c(mol, obs, dfbs, op, &inter, c)?;
    acc.stop(tok);

    Ok(grad)
}

/// Elementwise agreement (relative to `max(1, max|D|)`) at which a density
/// carried in [`OoRiMp2Result`] counts as the one formed here, so its J/K are
/// reused. The two differ only by the rounding of re-forming the density
/// from the re-semicanonicalised orbitals (~1e-14 at these sizes); any
/// change of the orbitals' occupied space is many orders larger.
pub const REUSE_TOL: f64 = 1e-10;

/// J/K of `d`: `cached`'s when it holds the same density (scaled by
/// `scale`, which multiplies the cached matrices), else a build on `jk`.
fn jk_or_reuse(
    jk: &ResponseJk<'_>,
    d: &Array2<f64>,
    cached: Option<&DensityJk>,
    scale: f64,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    if let Some(c) = cached {
        if c.is_for(&(d / scale), REUSE_TOL) {
            return Ok((scale * &c.j, scale * &c.k));
        }
    }
    jk.jk(d, JK_THRESH)
}

/// Relaxed 1-PDM in the MO basis: NO Z-vector (occ-vir/vir-occ blocks are
/// exactly zero, see the module doc). Same `doo + dooᵀ` / `dvv + dvvᵀ`
/// structure as plain RI-MP2's unrelaxed correlation density, but here this
/// full object already IS the relaxed density (nothing left to add).
fn relaxed_dm1_mo(
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    orb: &OrbitalSpace,
    nmo: usize,
) -> Array2<f64> {
    let (o0, v0) = (orb.first_occ, orb.nocc_total);
    let mut dm1mo = Array2::<f64>::zeros((nmo, nmo));
    let oo = &p_oo.view() + &p_oo.t();
    let vv = &p_vv.view() + &p_vv.t();
    dm1mo
        .slice_mut(ndarray::s![o0..o0 + orb.nocc, o0..o0 + orb.nocc])
        .assign(&oo);
    dm1mo
        .slice_mut(ndarray::s![v0..v0 + orb.nvir, v0..v0 + orb.nvir])
        .assign(&vv);
    dm1mo
}

/// Imat (RI-MP2 Lagrangian matrix) with its occ-vir block taken in the
/// virtual-derivative representation, in AO: `C · Imat_pulay · Cᵀ`. Same
/// object plain-MP2's Z-vector pipeline builds, at the OO orbitals/t2 (no z
/// contribution enters `build_imat_ri`: it is an x_ov/b_full contraction).
///
/// `build_x_ov` holds one `tt_i` block per worker. SOFT, for the reason the
/// RI-MP2 response path's per-worker charge is soft (`gradient.rs`): the size
/// reads the pool width, so a refusal would make admission depend on
/// `RAYON_NUM_THREADS`. Dropped once `x_ov` is built.
fn pulay_imat_ao(
    t2: &[f64],
    b_ov: &Array2<f64>,
    b_full: &ndarray::Array3<f64>,
    orb: &OrbitalSpace,
    naux: usize,
    c: &Array2<f64>,
) -> Array2<f64> {
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let x_ov = {
        let _worker_charge = crate::rimp2::charge_mo_side_soft(
            "OO-RI-MP2 gradient x_ov per-worker transients (tt_i)",
            rayon::current_num_threads()
                .max(1)
                .saturating_mul(x_ov_worker_bytes(nocc, nvir)),
        );
        build_x_ov(t2, b_ov, nocc, nvir, naux)
    };
    let imat = build_imat_ri(&x_ov, b_full, orb);
    let mut imat_pulay = imat.clone();
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            imat_pulay[(a_mo, i_mo)] = imat[(i_mo, a_mo)];
        }
    }
    c.dot(&imat_pulay).dot(&c.t())
}

/// Re-establish the SEMICANONICAL frame of `mos` and return
/// `(F_AO, C_sc, eps_sc)`.
///
/// The z = 0 formula needs diagonal denominators in
/// `compute_t2_and_integrals`. `oo_ri_mp2` already returns that frame; doing
/// it again means a caller that rotated the orbitals cannot silently get a
/// wrong gradient. The energy is invariant to this rotation, so for a
/// well-formed input it is a no-op up to eigenvector phase.
#[allow(clippy::too_many_arguments)]
fn semicanonical_frame(
    jk: &ResponseJk<'_>,
    mol: &Molecule,
    obs: &PreparedBasis,
    mos: &Array2<f64>,
    hf_jk: &DensityJk,
    nocc_total: usize,
    first_occ: usize,
    ext: Option<&ExternalPotential>,
) -> Result<(Array2<f64>, Array2<f64>, Vec<f64>), FerricError> {
    // Same hcore as `oo_ri_mp2` (V_ECP included for an ECP basis), so the
    // Fock matrix here is the one the OO energy was stationary for.
    let h = oneelectron::hcore_ecp_with_external(obs, mol, obs.basis_set(), ext)?;
    let c_occ = mos.slice(ndarray::s![.., ..nocc_total]);
    let d = c_occ.dot(&c_occ.t()) * 2.0;
    let (jv, kv) = jk_or_reuse(jk, &d, Some(hf_jk), 1.0)?;
    let f_ao = &h + &jv - 0.5 * &kv;
    let (u_sc, eps_sc) = crate::oo_rimp2::semicanonical_rotation(
        &mos.t().dot(&f_ao).dot(mos),
        first_occ,
        nocc_total,
    )?;
    let c_sc = mos.dot(&u_sc);
    Ok((f_ao, c_sc, eps_sc))
}

/// Energy-weighted density (AO) of the OO-MP2 1-PDM, `zeta` in PySCF's
/// grad/mp2.py naming, WITHOUT the `vhf_s1occ` piece (see [`vhf_s1occ`]).
///
/// * occ-occ / vir-vir of the correlation part: ½(F·W + W·F), the symmetric
///   part of the Lagrangian. In the semicanonical frame this reduces to
///   ½(eps_p+eps_q)·W_pq.
/// * occ-vir: (F·W)_ia, the SAME virtual-orbital-derivative representation
///   that Imat's occ-vir block (`imat_pulay`) uses. At a stationary point the
///   occ-vir Lagrangian is symmetric and may be taken in either
///   representation, but only CONSISTENTLY: mixing ½(F·W + W·F)_ia with
///   Imat_ia cost 6e-6..3e-5 (see the module doc).
/// * plus the HF energy-weighted occupied density 2·C_occ F_oo C_occᵀ. This
///   is the full occ block, which is not diagonal between core and active
///   orbitals when frozen_core > 0.
fn energy_weighted_density(
    f_mo: &Array2<f64>,
    dm1mo: &Array2<f64>,
    c: &Array2<f64>,
    nocc_total: usize,
) -> Array2<f64> {
    let nmo = c.ncols();
    let f_dot_p = f_mo.dot(dm1mo);
    let mut zeta_mo = 0.5 * (&f_dot_p + &f_dot_p.t());
    for i in 0..nocc_total {
        for a in nocc_total..nmo {
            zeta_mo[(i, a)] = f_dot_p[(i, a)];
            zeta_mo[(a, i)] = f_dot_p[(i, a)];
        }
    }
    let mut zeta_ao = c.dot(&zeta_mo).dot(&c.t());
    let c_occ = c.slice(ndarray::s![.., ..nocc_total]);
    let f_oo = f_mo.slice(ndarray::s![..nocc_total, ..nocc_total]);
    zeta_ao += &(c_occ.dot(&f_oo).dot(&c_occ.t()) * 2.0);
    zeta_ao
}

/// `vhf_s1occ = P_occ · G[2·dm1_corr] · P_occ`, with G[X] = J[X] − ½K[X]
/// (PySCF grad/mp2.py lines 161-163; the same object as plain RI-MP2's
/// `mp2_relaxed_lagrangian_gradient`).
///
/// This is NOT a CPHF artifact. It is the overlap-connection response of
/// F(D_HF) inside Σ W_pq F_pq, and it is present at z = 0. Dropping it was
/// 2.2e-4 / 3.2e-4 of the pre-F2 residual.
fn vhf_s1occ(
    jk: &ResponseJk<'_>,
    dm1_corr_ao: &Array2<f64>,
    response_jk: Option<&DensityJk>,
    c: &Array2<f64>,
    nocc_total: usize,
) -> Result<Array2<f64>, FerricError> {
    let two_dm_corr = 2.0 * dm1_corr_ao;
    let (jv, kv) = jk_or_reuse(jk, &two_dm_corr, response_jk, 2.0)?;
    let veff = &jv - &(0.5 * &kv);
    let c_occ = c.slice(ndarray::s![.., ..nocc_total]);
    let p_occ = c_occ.dot(&c_occ.t());
    Ok(p_occ.dot(&veff).dot(&p_occ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
    use ferric_core::basis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};

    /// Finite-difference the TRUE re-converged OO-RI-MP2 energy: perturb one
    /// nuclear coordinate, re-run RHF + full orbital optimization from
    /// scratch at the perturbed geometry (not a fixed-orbital probe), central
    /// difference. This is the only honest FD reference for an
    /// orbital-optimized method -- probing at frozen orbitals would silently
    /// validate only the integral-response terms, not the (absent, by
    /// construction) orbital-response ones.
    fn oo_rimp2_total_energy(
        mol: &Molecule,
        obs_bs: &ferric_core::basis::BasisSet,
        aux_bs: &ferric_core::basis::BasisSet,
        op: Operator,
        oo_config: &OoRiMp2Config,
        jk_aux: Option<&str>,
    ) -> f64 {
        let obs = PreparedBasis::new(mol, obs_bs).unwrap();
        let dfbs = PreparedBasis::new(mol, aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = reference_rhf(mol, &obs, op, &bounds, jk_aux);
        let oo = oo_ri_mp2(mol, &obs, &dfbs, op, &bounds, &rhf, oo_config, None).unwrap();
        assert!(
            oo.converged,
            "OO-RI-MP2 did not converge: |g|={:.2e}",
            oo.grad_norm
        );
        oo.total_energy
    }

    /// Tightly converged RHF reference: four-centre J/K, or RI-JK on
    /// `jk_aux` (J and K fitted on one basis, the route `oo_ri_mp2` follows).
    fn reference_rhf(
        mol: &Molecule,
        obs: &PreparedBasis,
        op: Operator,
        bounds: &SchwarzBounds,
        jk_aux: Option<&str>,
    ) -> ferric_scf::ScfResult {
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            mol,
            obs,
            op,
            bounds,
            &{
                let mut c = RhfConfig::default()
                    .with_energy_conv(1e-11)
                    .with_density_conv(1e-10);
                c.df_j_aux = jk_aux.map(str::to_string);
                c.df_k_aux = jk_aux.map(str::to_string);
                c
            }
            .for_geometry_task(),
        )
        .unwrap();
        assert!(rhf.converged);
        assert_eq!(rhf.df_jk.is_some(), jk_aux.is_some());
        rhf
    }

    fn oo_rimp2_gradient_fd(
        mol: &Molecule,
        obs_bs: &ferric_core::basis::BasisSet,
        aux_bs: &ferric_core::basis::BasisSet,
        op: Operator,
        oo_config: &OoRiMp2Config,
        delta: f64,
        jk_aux: Option<&str>,
    ) -> Array2<f64> {
        let natoms = mol.atoms.len();
        let mut grad = Array2::zeros((natoms, 3));
        for atom in 0..natoms {
            for coord in 0..3 {
                let mut mol_p = mol.clone();
                let mut mol_m = mol.clone();
                match coord {
                    0 => {
                        mol_p.atoms[atom].x += delta;
                        mol_m.atoms[atom].x -= delta;
                    }
                    1 => {
                        mol_p.atoms[atom].y += delta;
                        mol_m.atoms[atom].y -= delta;
                    }
                    _ => {
                        mol_p.atoms[atom].zpos += delta;
                        mol_m.atoms[atom].zpos -= delta;
                    }
                }
                let e_p = oo_rimp2_total_energy(&mol_p, obs_bs, aux_bs, op, oo_config, jk_aux);
                let e_m = oo_rimp2_total_energy(&mol_m, obs_bs, aux_bs, op, oo_config, jk_aux);
                grad[(atom, coord)] = (e_p - e_m) / (2.0 * delta);
            }
        }
        grad
    }

    /// Analytic-vs-FD bar for the nuclear gradient (Ha/Bohr).
    ///
    /// Derivation: the OO-specific ingredients (energy-weighted density and
    /// 1-PDM) agree with re-converged FD to <= 4e-9 in
    /// `scripts/oo_mp2_stationarity_proto.py nuc`. The Rust FD reference adds
    /// central-difference truncation (h = 1e-4: h²/6·E''' ~ 2e-9) plus J/K
    /// screening noise (~1e-12 Ha / 1e-4 ~ 1e-8), and re-convergence at
    /// grad_conv 1e-8 contributes O(|g|²) ~ 1e-16. 1e-6 is >= 100x that floor and
    /// 870x below the smallest pre-F2 residual (8.71e-4, H2O/STO-3G).
    const NUC_GRAD_BAR: f64 = 1e-6;

    fn tight_oo_config() -> OoRiMp2Config {
        // Tighter than the library default (grad_conv 1e-4) so the FD
        // reference geometry's re-converged energy is not itself
        // orbital-optimization-noise-limited at the 1e-4 to 1e-3 magnitude
        // the analytic-vs-FD comparison is trying to resolve.
        OoRiMp2Config {
            grad_conv: 1e-8,
            energy_conv: 1e-11,
            max_iter: 200,
            ..Default::default()
        }
    }

    #[test]
    fn test_oo_rimp2_gradient_analytic_vs_fd_h2_ccpvdz() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let oo_config = tight_oo_config();

        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default()
                .with_energy_conv(1e-11)
                .with_density_conv(1e-10),
        )
        .unwrap();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, None).unwrap();
        assert!(oo.converged);

        let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let fd = oo_rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &oo_config, 1e-4, None);

        eprintln!("=== H2/cc-pVDZ OO-RI-MP2 analytic vs FD gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..2 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!(
                    "  atom={} coord={}: analytic={:+.8} fd={:+.8} diff={:.2e}",
                    atom,
                    c,
                    analytic[(atom, c)],
                    fd[(atom, c)],
                    diff
                );
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // BAR (defect F2, 2026-09-24): NUC_GRAD_BAR, derived in its doc.
        // MUTATION NOTE: the pre-F2 solver + gradient measured 2.22e-3 here
        // (stable across FD delta 5e-5..2e-4) and passed the former 3e-3 bar.
        assert!(
            max_diff < NUC_GRAD_BAR,
            "H2 OO-RI-MP2 analytic vs FD max diff = {:.2e}",
            max_diff
        );
    }

    #[test]
    fn test_oo_rimp2_gradient_analytic_vs_fd_h2o_sto3g() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let oo_config = tight_oo_config();

        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default()
                .with_energy_conv(1e-11)
                .with_density_conv(1e-10),
        )
        .unwrap();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, None).unwrap();
        assert!(oo.converged);

        let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let fd = oo_rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &oo_config, 1e-4, None);

        eprintln!("=== H2O/STO-3G OO-RI-MP2 analytic vs FD gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..3 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!(
                    "  atom={} coord={}: analytic={:+.8} fd={:+.8} diff={:.2e}",
                    atom,
                    c,
                    analytic[(atom, c)],
                    fd[(atom, c)],
                    diff
                );
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // MUTATION NOTE: the pre-F2 solver + gradient measured 8.71e-4 here and
        // passed the former 1.5e-3 bar.
        assert!(
            max_diff < NUC_GRAD_BAR,
            "H2O OO-RI-MP2 analytic vs FD max diff = {:.2e}",
            max_diff
        );
    }

    const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    const JKFIT: &str = "def2-universal-jkfit";

    /// RI-JK reference: the nuclear gradient of the FITTED functional against
    /// FD of its re-converged energy, H2O/STO-3G. Same bar as the four-centre
    /// tests (`NUC_GRAD_BAR`): the fitted J/K have no screening, so the floor
    /// is the FD truncation and re-convergence terms of that derivation.
    ///
    /// TEETH: the fitted and four-centre analytic gradients at this geometry
    /// differ by more than the bar, so a gradient that ignored the route
    /// (four-centre 2e derivative or J/K under a fitted functional) fails.
    #[test]
    fn oo_rimp2_gradient_on_an_rijk_reference_matches_fd_h2o_sto3g() {
        let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let cfg = tight_oo_config();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = reference_rhf(&mol, &obs, op, &bounds, Some(JKFIT));
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &cfg, None).unwrap();
        assert!(oo.converged);
        assert!(
            oo.jk_route.is_some(),
            "an RI-JK reference must give the fitted route"
        );
        let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let fd = oo_rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &cfg, 1e-4, Some(JKFIT));
        let err = (&analytic - &fd).iter().fold(0.0f64, |m, v| m.max(v.abs()));

        let rhf_ex = reference_rhf(&mol, &obs, op, &bounds, None);
        let oo_ex = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf_ex, &cfg, None).unwrap();
        let g_ex = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo_ex, 0, None).unwrap();
        let fit_effect = (&analytic - &g_ex)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!(
            "H2O/STO-3G RI-JK OO-RI-MP2: max|analytic − FD| = {err:.3e}, \
             max|fitted − four-centre gradient| = {fit_effect:.3e}"
        );
        assert!(
            fit_effect > 10.0 * NUC_GRAD_BAR,
            "fit effect {fit_effect:.3e}"
        );
        assert!(err < NUC_GRAD_BAR, "max|analytic − FD| = {err:.3e}");
    }

    /// The J/K the result carries are reused only for the densities they
    /// belong to. A result whose cached densities are wrong (scaled by 1.01)
    /// gives the same gradient as the correct cache, to the rebuild's
    /// rounding (bar 1e-10 Ha/Bohr: the cached J/K and a fresh build of the
    /// same density differ only by the summation order of identical
    /// integrals and the re-formed density's rounding, ~1e-14 relative).
    ///
    /// TEETH: with the density check removed, the wrong cache moves the
    /// gradient by ~1e-2 relative of the J/K terms.
    #[test]
    fn oo_rimp2_gradient_reuses_only_matching_jk() {
        let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = reference_rhf(&mol, &obs, op, &bounds, None);
        let oo = oo_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &tight_oo_config(),
            None,
        )
        .unwrap();
        assert!(oo.converged);
        assert!(
            oo.response_jk.is_some(),
            "a converged run carries the response J/K"
        );
        let g = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let mut stale = oo.clone();
        fn wrong(x: &DensityJk) -> DensityJk {
            DensityJk {
                d: 1.01 * &x.d,
                j: 1.01 * &x.j + 1e-3,
                k: 1.01 * &x.k + 1e-3,
            }
        }
        stale.hf_jk = wrong(&stale.hf_jk);
        stale.response_jk = stale.response_jk.as_ref().map(wrong);
        let g_stale = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &stale, 0, None).unwrap();
        let diff = (&g - &g_stale).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("cached vs rebuilt J/K: max|Δg| = {diff:.3e}");
        assert!(diff < 1e-10, "max|Δg| = {diff:.3e}");
    }
}
