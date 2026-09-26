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
    build_mp2_density, compute_b_full_mo, compute_t2_and_integrals, OoRiMp2Result,
};
use crate::rimp2::{active_occ, cholesky_inverse_sqrt, Mp2Intermediates};
use crate::zvector::{build_imat_ri, build_x_ov};
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::gradient::{
    ecp_gradient, oneelectron_gradient, overlap_deriv_contract, twoelectron_gradient_bilinear,
};
use ferric_scf::rhf::build_jk;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the analytical nuclear gradient for OO-RI-MP2.
///
/// See the module doc comment for the no-Z-vector derivation and its
/// numerical verification.
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

    // The z = 0 formula below needs the SEMICANONICAL frame; see
    // `semicanonical_frame`.
    let ctx = ferric_core::parallel::ParallelContext::default();
    let (f_ao, c_sc, eps_sc) = semicanonical_frame(
        &ctx,
        mol,
        obs,
        bounds,
        &result.mos,
        nocc_total,
        first_occ,
        ext,
    )?;
    let c = &c_sc;
    let eps = &eps_sc;

    let b_full = compute_b_full_mo(obs, dfbs, op, c)?;
    let b_ov_3d = b_full.slice(ndarray::s![.., first_occ..nocc_total, nocc_total..]);
    let b_ov = b_ov_3d
        .to_owned()
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();

    let (t2, _) = compute_t2_and_integrals(&b_ov, eps, nocc, nvir, nocc_total, first_occ, naux);

    let (p_oo, p_vv) = build_mp2_density(&t2, nocc, nvir);

    // --- Relaxed 1-PDM in MO basis: NO Z-vector (occ-vir/vir-occ blocks are
    // exactly zero — see module doc comment). Same doo+doo^T / dvv+dvv^T
    // structure as plain RI-MP2's unrelaxed correlation density, but here
    // this full object already IS the relaxed density (nothing left to add).
    let mut dm1mo = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc {
        let i_mo = first_occ + i;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            dm1mo[(i_mo, j_mo)] = p_oo[(i, j)] + p_oo[(j, i)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            dm1mo[(a_mo, b_mo)] = p_vv[(a, b)] + p_vv[(b, a)];
        }
    }
    // occ-vir / vir-occ: z = 0, left at zero.

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

    // --- Imat (RI-MP2 Lagrangian matrix), same object plain-MP2's Z-vector
    // pipeline builds, evaluated at the OO orbitals/t2 (no z contribution
    // enters build_imat_ri itself -- it is purely an x_ov/b_full contraction).
    let x_ov = build_x_ov(&t2, &b_ov, nocc, nvir, naux);
    let orb = OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    };
    let imat = build_imat_ri(&x_ov, &b_full, &orb);
    let mut imat_pulay = imat.clone();
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            imat_pulay[(a_mo, i_mo)] = imat[(i_mo, a_mo)];
        }
    }
    let im1 = {
        let cp = c.dot(&imat_pulay);
        cp.dot(&c.t())
    };

    let zeta_ao = energy_weighted_density(&f_mo, &dm1mo, c, nocc_total);
    let vhf_s1occ = vhf_s1occ(&ctx, obs, bounds, &dm1_corr_ao, c, nocc_total)?;

    // --- Assemble (same signs as plain RI-MP2's mp2_relaxed_lagrangian_gradient).
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

    // 2e-integral-derivative: bilinear Gamma(hf_dm1, hf_dm1 + 2*dm1_corr),
    // same structure as plain RI-MP2's bilinear term.
    let two_dm_corr = 2.0 * &dm1_corr_ao;
    let dm1p = &hf_dm1 + &two_dm_corr;
    grad += &twoelectron_gradient_bilinear(obs, op, bounds, &hf_dm1, &dm1p)?;

    // RI 3c/2c integral-response term: reuse plain RI-MP2's implementation by
    // packaging OO-MP2's own t2/b_ov/v_inv_sqrt into an Mp2Intermediates.
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

    Ok(grad)
}

/// Re-establish the SEMICANONICAL frame of `mos` and return
/// `(F_AO, C_sc, eps_sc)`.
///
/// The z = 0 formula needs diagonal denominators in
/// `compute_t2_and_integrals`. `oo_ri_mp2` already returns that frame; doing
/// it again means a caller that rotated the orbitals cannot silently get a
/// wrong gradient. The energy is invariant to this rotation, so for a
/// well-formed input it is a no-op up to eigenvector phase.
fn semicanonical_frame(
    ctx: &ferric_core::parallel::ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    mos: &Array2<f64>,
    nocc_total: usize,
    first_occ: usize,
    ext: Option<&ExternalPotential>,
) -> Result<(Array2<f64>, Array2<f64>, Vec<f64>), FerricError> {
    let nmo = mos.nrows();
    // Same hcore as `oo_ri_mp2` (V_ECP included for an ECP basis), so the
    // Fock matrix here is the one the OO energy was stationary for.
    let h = oneelectron::hcore_ecp_with_external(obs, mol, obs.basis_set(), ext)?;
    let c_occ = mos.slice(ndarray::s![.., ..nocc_total]);
    let d = c_occ.dot(&c_occ.t()) * 2.0;
    let (mut jv, mut kv) = (Array2::zeros((nmo, nmo)), Array2::zeros((nmo, nmo)));
    build_jk(ctx, obs, bounds, 1e-12, &d, &mut jv, &mut kv)?;
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
    ctx: &ferric_core::parallel::ParallelContext,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    dm1_corr_ao: &Array2<f64>,
    c: &Array2<f64>,
    nocc_total: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = c.nrows();
    let two_dm_corr = 2.0 * dm1_corr_ao;
    let (mut jv, mut kv) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    build_jk(ctx, obs, bounds, 1e-12, &two_dm_corr, &mut jv, &mut kv)?;
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
    ) -> f64 {
        let obs = PreparedBasis::new(mol, obs_bs).unwrap();
        let dfbs = PreparedBasis::new(mol, aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-11,
                density_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged);
        let oo = oo_ri_mp2(mol, &obs, &dfbs, op, &bounds, &rhf, oo_config, None).unwrap();
        assert!(
            oo.converged,
            "OO-RI-MP2 did not converge: |g|={:.2e}",
            oo.grad_norm
        );
        oo.total_energy
    }

    fn oo_rimp2_gradient_fd(
        mol: &Molecule,
        obs_bs: &ferric_core::basis::BasisSet,
        aux_bs: &ferric_core::basis::BasisSet,
        op: Operator,
        oo_config: &OoRiMp2Config,
        delta: f64,
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
                let e_p = oo_rimp2_total_energy(&mol_p, obs_bs, aux_bs, op, oo_config);
                let e_m = oo_rimp2_total_energy(&mol_m, obs_bs, aux_bs, op, oo_config);
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
            &RhfConfig {
                energy_conv: 1e-11,
                density_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, None).unwrap();
        assert!(oo.converged);

        let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let fd = oo_rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &oo_config, 1e-4);

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
            &RhfConfig {
                energy_conv: 1e-11,
                density_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, None).unwrap();
        assert!(oo.converged);

        let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        let fd = oo_rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &oo_config, 1e-4);

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
}
