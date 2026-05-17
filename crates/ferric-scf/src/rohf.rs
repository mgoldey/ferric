//! Restricted Open-Shell Hartree-Fock (ROHF) solver.
//!
//! Spin-pure open-shell HF: a single set of MOs partitioned into doubly
//! occupied (closed), singly occupied (open, α-only), and virtual blocks.
//! ⟨S²⟩ is exact S(S+1) by construction.
//!
//! Coupling: **Guest-Saunders** (PySCF default, hard-coded — no knob).
//! Implementation mirrors PySCF's `get_roothaan_fock` (`pyscf/scf/rohf.py`):
//! the effective Fock is built via density-based projectors
//!   P_c = D_β · S,   P_o = (D_α − D_β) · S,   P_v = I − D_α · S
//! and then assembled from F_c = (F_α + F_β)/2, F_α, F_β according to the
//! Roothaan block table:
//! ```text
//! ========  ======== ====== =========
//! space      closed   open   virtual
//! ========  ======== ====== =========
//! closed       Fc      Fb     Fc
//! open         Fb      Fc     Fa
//! virtual      Fc      Fa     Fc
//! ========  ======== ====== =========
//! ```
//! Per Guest-Saunders (a_cc = a_oo = a_vv = 1/2, b = -1/2, c = 3/2 in the
//! a/b/c parametrisation), the diagonal blocks reduce to F_c — exactly what
//! the projector form above produces.

use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// ROHF configuration mirrors RHF.
pub type RohfConfig = RhfConfig;

/// Solve restricted open-shell Hartree-Fock equations.
///
/// Uses `mol.charge` and `mol.multiplicity` to determine doubly/singly
/// occupied orbital counts:
///   - nocc_open   = mult − 1                (singly α-occupied)
///   - nocc_double = (nelec − nocc_open) / 2 (doubly occupied)
pub fn solve_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
) -> Result<ScfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore(prep);
    let n = prep.nbasis();
    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "ROHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1; // 2S = number of singly-occupied (α) orbitals
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "ROHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;
    let nocc_a = nocc_double + nocc_open;
    let nocc_b = nocc_double;
    if nocc_a + nocc_b != nelec as usize {
        return Err(FerricError::General(
            "ROHF: nocc_a + nocc_b != nelec".into(),
        ));
    }
    let vnn = mol.nuclear_repulsion();

    // S^{-1/2}
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());

    // hcore guess MOs
    let _ = hcore_guess(&s, &h, nocc_a.max(1))?;
    let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    let mut c = s_inv_sqrt.dot(&c_prime);

    // ROHF densities (AO):
    //   D_c (closed/doubly-occupied) = 2 Σ_i C_i C_i^T  (i = 0..nocc_double)
    //   D_o (open/singly-α-occupied) = Σ_j C_j C_j^T    (j = nocc_double..nocc_a)
    // D_α = D_c/2 + D_o,  D_β = D_c/2  → (D_α + D_β) = D_c + D_o (total).
    // We track D_α and D_β internally to feed J/K builders (matching UHF JK).
    let (mut d_a, mut d_b) = build_rohf_densities(&c, nocc_double, nocc_open);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;
    let mut total_quartets = 0usize;

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        j_buf.fill(0.0);
        k_a_buf.fill(0.0);
        k_b_buf.fill(0.0);
        let d_total = &d_a + &d_b;

        // J from D_total
        {
            let mut dj = DirectJ::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += dj.build(&d_total, &mut j_buf)?;
        }
        // K_α from D_α, K_β from D_β
        {
            let mut dk = DirectK::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += <DirectK as KBuilder>::build(&mut dk, &d_a, &mut k_a_buf)?;
        }
        {
            let mut dk = DirectK::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += <DirectK as KBuilder>::build(&mut dk, &d_b, &mut k_b_buf)?;
        }

        // F_σ = H + J − K_σ
        let f_a: Array2<f64> = &h + &j_buf - &k_a_buf;
        let f_b: Array2<f64> = &h + &j_buf - &k_b_buf;

        // Energy: 0.5 tr((H+F_α) D_α + (H+F_β) D_β) — identical to UHF formula
        let e_elec: f64 = 0.5
            * ((0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| {
                    (h[(i, j)] + f_a[(i, j)]) * d_a[(i, j)]
                        + (h[(i, j)] + f_b[(i, j)]) * d_b[(i, j)]
                })
                .sum::<f64>());
        let energy = e_elec + vnn;

        // Build Roothaan effective Fock (Guest-Saunders, via PySCF projector form).
        let f_eff = roothaan_fock(&f_a, &f_b, &d_a, &d_b, &s);

        // DIIS error from the effective Fock (single stream).
        let d_tot_diis = &d_a + &d_b;
        let err = f_eff.dot(&d_tot_diis).dot(&s) - s.dot(&d_tot_diis).dot(&f_eff);

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let converged = de < config.energy_conv && err_max < config.density_conv;

        if iter > 1 && converged {
            let (eps, c_f) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let (d_a_f, d_b_f) = build_rohf_densities(&c_f, nocc_double, nocc_open);
            let density_total = &d_a_f + &d_b_f;
            return Ok(ScfResult {
                spin: Spin::RestrictedOpen,
                energy,
                density_total,
                density_alpha: d_a_f,
                density_beta: Some(d_b_f),
                mos_alpha: c_f,
                mos_beta: None,
                eps_alpha: eps,
                eps_beta: None,
                fock_alpha: f_eff,
                fock_beta: None,
                converged: true,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        // DIIS extrapolate effective Fock, diagonalize, rebuild densities.
        let f_new = diis.step(&f_eff, &err);
        let (_, c_new) = diagonalize(&f_new, &s_inv_sqrt)?;
        c = c_new;
        let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
        d_a = da_n;
        d_b = db_n;
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
}

/// Build ROHF α/β densities from MO coefficients:
///   D_β = Σ_{i<nocc_double} C_i C_i^T
///   D_α = D_β + Σ_{j∈open} C_j C_j^T
fn build_rohf_densities(
    c: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = c.nrows();
    let mut d_b = Array2::<f64>::zeros((n, n));
    if nocc_double > 0 {
        let cd = c.slice(ndarray::s![.., ..nocc_double]);
        d_b = cd.dot(&cd.t());
    }
    let mut d_a = d_b.clone();
    if nocc_open > 0 {
        let co = c.slice(ndarray::s![.., nocc_double..nocc_double + nocc_open]);
        d_a = &d_a + &co.dot(&co.t());
    }
    (d_a, d_b)
}

/// Roothaan effective Fock (Guest-Saunders coupling).
/// Mirrors `pyscf.scf.rohf.get_roothaan_fock`.
fn roothaan_fock(
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Array2<f64> {
    let n = s.shape()[0];
    let f_c = 0.5 * (f_a + f_b);
    // Projectors: P_c = D_β S, P_o = (D_α − D_β) S, P_v = I − D_α S
    let p_c = d_b.dot(s);
    let do_diff: Array2<f64> = d_a - d_b;
    let p_o = do_diff.dot(s);
    let mut p_v = Array2::<f64>::eye(n);
    p_v = &p_v - &d_a.dot(s);

    // Upper-triangle pieces (PySCF builds half then symmetrises by F + F^T).
    let p_c_t = p_c.t();
    let p_o_t = p_o.t();
    let p_v_t = p_v.t();

    let mut f = 0.5 * p_c_t.dot(&f_c).dot(&p_c);
    f = &f + &(0.5 * p_o_t.dot(&f_c).dot(&p_o));
    f = &f + &(0.5 * p_v_t.dot(&f_c).dot(&p_v));
    f = &f + &p_o_t.dot(f_b).dot(&p_c);
    f = &f + &p_o_t.dot(f_a).dot(&p_v);
    f = &f + &p_v_t.dot(&f_c).dot(&p_c);
    let f_sym = &f + &f.t();
    f_sym
}

fn diagonalize(
    f: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let f_prime = s_inv_sqrt.dot(f).dot(s_inv_sqrt);
    let (evals, evecs) = f_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    let c = s_inv_sqrt.dot(&evecs);
    Ok((evals.to_vec(), c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_rohf_h_atom_sto3g() {
        // Single H atom — trivial 1-electron case.
        let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RohfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(res.converged);
        assert!(
            (res.energy + 0.466581850).abs() < 1e-5,
            "H atom energy = {}",
            res.energy
        );
    }
}
