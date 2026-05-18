//! Unrestricted Hartree-Fock (UHF) solver.
//!
//! Parallels `rhf.rs` but tracks independent α/β densities, Fock matrices, and
//! DIIS streams. Uses J built from D_total = D_α + D_β and K built per spin.

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

/// UHF configuration mirrors RHF (separate type for forward extensibility).
pub type UhfConfig = RhfConfig;

/// Solve unrestricted Hartree-Fock equations for a molecule.
///
/// Uses `mol.charge` and `mol.multiplicity` to determine α/β electron counts.
/// The initial guess is built from a single hcore diagonalization; symmetry is
/// broken by occupying fewer β orbitals than α (or by a small HOMO/LUMO mixing
/// when nocc_a == nocc_b).
pub fn solve_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
) -> Result<ScfResult, FerricError> {
    solve_uhf_with_guess(ctx, mol, prep, op, bounds, config, None)
}

/// UHF with optional caller-supplied initial MOs.
///
/// `initial_mos` lets the caller provide a directed starting point (e.g.
/// neutral RHF MOs for a cation calculation, to avoid landing in a
/// doublet-excited basin from the symmetric hcore guess). Pass `None`
/// for the default hcore guess.
///
/// The provided `c_a`/`c_b` must have shape (nbasis, nbasis) and span
/// the AO basis; only the first `nocc_α`/`nocc_β` columns are used as
/// the occupied set.
pub fn solve_uhf_with_guess(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<ScfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore(prep);
    let n = prep.nbasis();
    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "UHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1; // 2S
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "UHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    if nocc_a + nocc_b != nelec as usize {
        return Err(FerricError::General(
            "UHF: nocc_a + nocc_b != nelec".into(),
        ));
    }
    if nocc_b > nocc_a {
        return Err(FerricError::General("UHF: nocc_b > nocc_a".into()));
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

    // Initial guess: caller-supplied MOs if provided, else hcore.
    let (mut c_a, mut c_b) = if let Some((ca0, cb0)) = initial_mos {
        if ca0.dim() != (n, n) || cb0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "solve_uhf_with_guess: initial MO shape mismatch (got {:?}/{:?}, want ({n},{n}))",
                ca0.dim(), cb0.dim()
            )));
        }
        (ca0.clone(), cb0.clone())
    } else {
        // hcore guess: get MO coefficients from H' = S^{-1/2} H S^{-1/2}.
        let _ = hcore_guess(&s, &h, nocc_a.max(1))?; // sanity check it succeeds
        let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
        let (_, c_prime) = h_prime
            .eigh(ndarray_linalg::UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
        let c = s_inv_sqrt.dot(&c_prime);
        (c.clone(), c)
    };

    // For genuine open shell (nocc_a > nocc_b), occupying the lowest nocc_σ
    // orbitals per spin is already symmetry-broken. For "forced" UHF on a
    // closed-shell, mix HOMO/LUMO in β with a small angle to break symmetry.
    if nocc_a == nocc_b && nocc_a > 0 && nocc_a < n {
        let theta = 0.1f64;
        let (cs, sn) = (theta.cos(), theta.sin());
        let homo = nocc_b - 1;
        let lumo = nocc_b;
        for mu in 0..n {
            let h_val = c_b[(mu, homo)];
            let l_val = c_b[(mu, lumo)];
            c_b[(mu, homo)] = cs * h_val + sn * l_val;
            c_b[(mu, lumo)] = -sn * h_val + cs * l_val;
        }
    }

    let mut d_a = density(&c_a, nocc_a);
    let mut d_b = density(&c_b, nocc_b);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    // Coupled α/β DIIS — single subspace, joint error norm. PySCF-style.
    // Independent per-spin DIIS desyncs α and β on cations (e.g. H2O+ took
    // 421 iterations to converge with independent DIIS; coupled converges
    // in ~15-25 cycles).
    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;
    let mut total_quartets = 0usize;

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        j_buf.fill(0.0);
        k_a_buf.fill(0.0);
        k_b_buf.fill(0.0);
        let d_total = &d_a + &d_b;

        // J built from total density (one call).
        {
            let mut dj = DirectJ::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += dj.build(&d_total, &mut j_buf)?;
        }
        // K built per spin.
        {
            let mut dk = DirectK::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += <DirectK as KBuilder>::build(&mut dk, &d_a, &mut k_a_buf)?;
        }
        {
            let mut dk = DirectK::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += <DirectK as KBuilder>::build(&mut dk, &d_b, &mut k_b_buf)?;
        }

        // F_σ = H + J - K_σ
        let f_a: Array2<f64> = &h + &j_buf - &k_a_buf;
        let f_b: Array2<f64> = &h + &j_buf - &k_b_buf;

        // Electronic energy: 0.5 tr((H+F_α) D_α + (H+F_β) D_β)
        let e_elec: f64 = 0.5
            * ((0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| {
                    (h[(i, j)] + f_a[(i, j)]) * d_a[(i, j)]
                        + (h[(i, j)] + f_b[(i, j)]) * d_b[(i, j)]
                })
                .sum::<f64>());
        let energy = e_elec + vnn;

        // DIIS errors per spin: F_σ D_σ S − S D_σ F_σ
        let err_a = f_a.dot(&d_a).dot(&s) - s.dot(&d_a).dot(&f_a);
        let err_b = f_b.dot(&d_b).dot(&s) - s.dot(&d_b).dot(&f_b);

        let de = (energy - prev_e).abs();
        let err_max_a = err_a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max_b = err_b.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max = err_max_a.max(err_max_b);

        let converged = de < config.energy_conv && err_max < config.density_conv;

        if iter > 1 && converged {
            let (eps_a, c_a_f) = diagonalize(&f_a, &s_inv_sqrt)?;
            let (eps_b, c_b_f) = diagonalize(&f_b, &s_inv_sqrt)?;
            // ⟨S²⟩ diagnostic
            let s2 = expectation_s_squared(&c_a_f, &c_b_f, &s, nocc_a, nocc_b);
            let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
            let s_ideal = s_true * (s_true + 1.0);
            if s2 > s_ideal + 0.1 {
                eprintln!(
                    "UHF warning: spin contamination ⟨S²⟩ = {:.4} (ideal {:.4})",
                    s2, s_ideal
                );
            }
            let density_total = &d_a + &d_b;
            return Ok(ScfResult {
                spin: Spin::Unrestricted,
                energy,
                density_total,
                density_alpha: d_a,
                density_beta: Some(d_b),
                mos_alpha: c_a_f,
                mos_beta: Some(c_b_f),
                eps_alpha: eps_a,
                eps_beta: Some(eps_b),
                fock_alpha: f_a,
                fock_beta: Some(f_b),
                converged: true,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        // Coupled DIIS extrapolation: one set of coefficients applied to
        // both spin Fock histories.
        let (f_a_new, f_b_new) = diis.step_pair(&f_a, &f_b, &err_a, &err_b);
        let (_, c_a_new) = diagonalize(&f_a_new, &s_inv_sqrt)?;
        let (_, c_b_new) = diagonalize(&f_b_new, &s_inv_sqrt)?;
        c_a = c_a_new;
        c_b = c_b_new;
        d_a = density(&c_a, nocc_a);
        d_b = density(&c_b, nocc_b);
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
}

fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let n = c.nrows();
    if nocc == 0 {
        return Array2::zeros((n, n));
    }
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    c_occ.dot(&c_occ.t())
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

/// ⟨S²⟩ for a UHF determinant:
/// ⟨S²⟩ = S(S+1) + N_β − Σ_{i∈α-occ, j∈β-occ} |⟨α_i|β_j⟩|²
fn expectation_s_squared(
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    s: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> f64 {
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    if nocc_a == 0 || nocc_b == 0 {
        return s_ideal;
    }
    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]);
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]);
    // overlap_ab[i,j] = (C_α^T S C_β)[i,j]
    let overlap_ab = c_a_occ.t().dot(s).dot(&c_b_occ);
    let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
    s_ideal + (nocc_b as f64) - sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn test_uhf_h_atom_sto3g() {
        // Single H atom, doublet. Energy = -0.466581 in STO-3G (one electron, no e-e).
        let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_uhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(res.converged);
        // STO-3G H atom: H = -0.46658185 (one electron, -ζ_1s).
        assert!(
            (res.energy + 0.466581850).abs() < 1e-5,
            "H atom energy = {}",
            res.energy
        );
        // ⟨S²⟩ exact = 0.75 for doublet, single electron.
        let s2 = expectation_s_squared(
            &res.mos_alpha,
            res.mos_beta.as_ref().unwrap(),
            &oneelectron::overlap(&prep),
            1,
            0,
        );
        assert!((s2 - 0.75).abs() < 1e-10, "⟨S²⟩ = {}", s2);
    }
}
