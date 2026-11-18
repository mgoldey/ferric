//! Boys orbital localization and LMP2 domain construction.
//!
//! Maximizes the Boys functional F = Σ_i (⟨i|x|i⟩² + ⟨i|y|i⟩² + ⟨i|z|i⟩²)
//! via 2×2 Jacobi rotations (Foster & Boys 1960).
//!
//! Domain construction assigns each Boys-localized orbital i a set of AO basis
//! functions whose shells lie within a distance cutoff of its Boys center.  The
//! union of domains for a given μ index gives the sparsity pattern of the
//! Laplace pseudo-densities P(t) and Q(t).

use ndarray::{s, Array2};

/// Result of Boys localization.
#[derive(Debug, Clone)]
#[must_use]
pub struct BoysResult {
    /// Localized MO coefficients, shape (nbas, nocc).
    pub c_loc: Array2<f64>,
    /// Boys centers [nocc × 3]: ⟨i|r|i⟩ for each localized orbital.
    pub centers: Array2<f64>,
    /// Number of sweeps performed.
    pub iterations: usize,
    /// Whether the localization converged.
    pub converged: bool,
}

/// Boys localization of occupied orbitals.
///
/// `c_occ`: (nbas, nocc) — canonical occupied MO coefficient matrix.
/// `dip`: [x_mat, y_mat, z_mat] — dipole integral matrices (nbas × nbas), AO basis.
/// `max_iter`: maximum number of sweeps over all pairs.
///
/// Returns localized orbitals and Boys centers.
pub fn boys_localize(c_occ: &Array2<f64>, dip: &[Array2<f64>; 3], max_iter: usize) -> BoysResult {
    let nocc = c_occ.ncols();

    // Output buffer, same layout as the input copy (filled at the end).
    let mut c = c_occ.to_owned();

    // Compute MO dipole matrices in the occupied subspace: d[alpha]_ij = C^T @ dip_alpha @ C
    // We keep these as nocc×nocc matrices and update them in-place upon each rotation.
    let mut d: [Array2<f64>; 3] = std::array::from_fn(|alpha| {
        // d_ij = sum_mu sum_nu C_mu_i * dip_alpha[mu,nu] * C_nu_j
        let dip_c: Array2<f64> = dip[alpha].dot(&c); // (nbas, nocc)
        c.t().dot(&dip_c) // (nocc, nocc)
    });

    // Working copy TRANSPOSED, (nocc, nbas) row-major: a Jacobi rotation of
    // orbitals i, j then sweeps two contiguous rows instead of two
    // stride-nocc columns. Every element sees the same two-term update as
    // before, so the result is bitwise the column-major sweep's.
    let mut ct: Array2<f64> = c.t().as_standard_layout().into_owned();

    let mut converged = false;
    let mut iterations = 0;

    'outer: for _sweep in 0..max_iter {
        iterations += 1;
        let mut max_theta: f64 = 0.0;

        for i in 0..nocc {
            for j in (i + 1)..nocc {
                // Jacobi rotation of the (i, j) pair that MAXIMIZES the Boys
                // functional Σ_i |⟨i|r|i⟩|² (equivalently, minimizes the orbital
                // spread). With
                //   A = Σ_α [ (q_ii − q_jj)²/4 − q_ij² ]
                //   B = Σ_α q_ij (q_ii − q_jj)
                // the stationary angles are θ = atan2(∓B, ±A)/4, and the two
                // branches are the MAXIMUM and the MINIMUM of the same functional.
                //
                // The sign below is `atan2(-B, A)`. Getting it backwards does not
                // fail loudly — it converges happily to the *minimum*, i.e. the
                // maximally DELOCALIZED orbitals, with every Boys center collapsed
                // onto the molecular centroid.
                //
                // MEASURED with the wrong branch (`atan2(B, -A)`, fixed 2026-07-26):
                // water/cc-pVDZ reported `converged = true` in 11 sweeps while the
                // functional FELL 0.2277 → 0.0620, all five centers sat at y = 0.000
                // despite the H atoms being at y = ±1.428 Bohr, and three orbitals
                // shared one center. Benzene/STO-3G collapsed all 21 centers onto
                // (0,0,0) — max center-center separation 0.000 Bohr. A 2×2 brute-force
                // scan confirms this branch: it reproduces the scan optimum exactly,
                // while the old one scored below even the unrotated F(θ=0).
                let mut big_a = 0.0_f64;
                let mut big_b = 0.0_f64;
                for alpha in 0..3 {
                    let da = d[alpha][(i, i)] - d[alpha][(j, j)]; // q_ii - q_jj
                    let oa = d[alpha][(i, j)]; // q_ij
                    big_a += da * da / 4.0 - oa * oa;
                    big_b += oa * da;
                }

                let theta = f64::atan2(-big_b, big_a) / 4.0;
                if theta.abs() > max_theta {
                    max_theta = theta.abs();
                }

                if theta.abs() < 1e-12 {
                    continue;
                }

                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Rotate coefficient columns i and j (rows of `ct`).
                {
                    let (mut ri, mut rj) = ct.multi_slice_mut((s![i, ..], s![j, ..]));
                    for (x, y) in ri.iter_mut().zip(rj.iter_mut()) {
                        let ci = *x;
                        let cj = *y;
                        *x = cos_t * ci - sin_t * cj;
                        *y = sin_t * ci + cos_t * cj;
                    }
                }

                // Update MO dipole matrices for all alpha.
                // For each alpha, update row/col i and j of d[alpha].
                // New d[alpha]_pi = cos(t)*d_pi_old - sin(t)*d_pj_old  for all p
                // New d[alpha]_pj = sin(t)*d_pi_old + cos(t)*d_pj_old  for all p
                // Then similarly for rows.
                for alpha in 0..3 {
                    // Update columns i and j.
                    for p in 0..nocc {
                        let dpi = d[alpha][(p, i)];
                        let dpj = d[alpha][(p, j)];
                        d[alpha][(p, i)] = cos_t * dpi - sin_t * dpj;
                        d[alpha][(p, j)] = sin_t * dpi + cos_t * dpj;
                    }
                    // Update rows i and j.
                    for p in 0..nocc {
                        let dip_val = d[alpha][(i, p)];
                        let djp = d[alpha][(j, p)];
                        d[alpha][(i, p)] = cos_t * dip_val - sin_t * djp;
                        d[alpha][(j, p)] = sin_t * dip_val + cos_t * djp;
                    }
                }
            }
        }

        if max_theta < 1e-8 {
            converged = true;
            break 'outer;
        }
    }

    c.assign(&ct.t());

    // Extract Boys centers: ⟨i|r_alpha|i⟩ = d[alpha][i,i]
    let mut centers = Array2::zeros((nocc, 3));
    for i in 0..nocc {
        for alpha in 0..3 {
            centers[(i, alpha)] = d[alpha][(i, i)];
        }
    }

    BoysResult {
        c_loc: c,
        centers,
        iterations,
        converged,
    }
}

/// [`boys_localize`] that refuses an unconverged localization.
///
/// The sparse Laplace formulations mask the pseudo-densities by the localized
/// orbitals' domains; orbitals from a localization that stopped short of its
/// tolerance still span the occupied space but give domains that were not
/// meant to be used, so the run errors instead of continuing.
pub fn boys_localize_checked(
    c_occ: &Array2<f64>,
    dip: &[Array2<f64>; 3],
    max_iter: usize,
) -> Result<BoysResult, ferric_core::FerricError> {
    let res = boys_localize(c_occ, dip, max_iter);
    if res.converged {
        Ok(res)
    } else {
        Err(ferric_core::FerricError::Convergence(format!(
            "Boys localization did not converge in {} sweeps (max {max_iter}); the \
             domain-truncated Laplace formulation needs converged localized orbitals",
            res.iterations
        )))
    }
}

// ---------------------------------------------------------------------------
// λ-Boys: Boys plus a penalty on the off-diagonal occupied Fock coupling
// ---------------------------------------------------------------------------

/// Default λ (Bohr²/Ha²) of the λ-Boys occupied localization in
/// amplitude-threshold local MP2. `0` is plain Boys.
pub const LAMBDA_BOYS_DEFAULT: f64 = 10.0;

/// Convergence target of [`lambda_boys_rotation`]: the largest pair
/// gradient `|∂G/∂κ_pq|` (Bohr², the units of the Boys term) after a sweep.
pub const LAMBDA_BOYS_GRAD_TOL: f64 = 1e-8;

/// Jacobi-sweep cap of the λ-Boys optimiser in the LMP2 drivers.
pub const LAMBDA_BOYS_MAX_SWEEPS: usize = 2000;

/// What a λ-Boys localization did, recorded on every LMP2 result.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LambdaBoysReport {
    /// The weight λ of `Σ_i F_ii²` (Bohr²/Ha²). `0` = plain Boys (the
    /// optimiser does not run and the Boys orbitals are used bit for bit).
    pub lambda: f64,
    /// Jacobi sweeps the λ-Boys optimiser ran (0 at λ = 0).
    pub sweeps: usize,
    /// Largest pair gradient at the returned orbitals (0 at λ = 0).
    pub max_grad: f64,
    /// `G` at the Boys start and at the returned orbitals.
    pub objective_start: f64,
    /// `G` at the returned orbitals.
    pub objective_end: f64,
    /// Wall seconds of the λ-Boys optimiser alone (the Boys start excluded).
    pub t_s: f64,
}

/// The λ-Boys objective `G = Σ_x Σ_i (r^x_ii)² + λ Σ_i F_ii²` of
/// symmetric matrices `x = [r^x, r^y, r^z]`, `f = F_oo`, all in the same
/// orbital basis (WLLH25 Eq 9; book section
/// `amplitude-lmp2-lambda-boys`).
pub fn lambda_boys_objective(x: &[Array2<f64>; 3], f: &Array2<f64>, lambda: f64) -> f64 {
    let diag2 = |m: &Array2<f64>| (0..m.nrows()).map(|i| m[(i, i)] * m[(i, i)]).sum::<f64>();
    x.iter().map(diag2).sum::<f64>() + lambda * diag2(f)
}

/// Analytic `∂G/∂θ` at `θ = 0` for the plane rotation of orbitals `p < q`
/// with the convention of [`lambda_boys_rotation`] (`p' = c p + s q`,
/// `q' = −s p + c q`): `4 Σ_X w_X (X_pp − X_qq) X_pq`. This is
/// `−∂G/∂κ_pq` of the book's `U = 1 + κ` parametrization.
pub fn lambda_boys_pair_gradient(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
    p: usize,
    q: usize,
) -> f64 {
    let term = |m: &Array2<f64>| (m[(p, p)] - m[(q, q)]) * m[(p, q)];
    4.0 * (x.iter().map(term).sum::<f64>() + lambda * term(f))
}

/// Largest `|∂G/∂θ_pq|` over all pairs.
fn lambda_boys_max_grad(x: &[Array2<f64>; 3], f: &Array2<f64>, lambda: f64) -> f64 {
    let n = f.nrows();
    let mut g = 0.0f64;
    for p in 0..n {
        for q in p + 1..n {
            g = g.max(lambda_boys_pair_gradient(x, f, lambda, p, q).abs());
        }
    }
    g
}

/// The Jacobi angle maximizing `G` in the `(p, q)` plane:
/// `θ = atan2(b, a)/4` with `d_X = (X_pp − X_qq)/2`, `c_X = X_pq`,
/// `a = Σ w (d² − c²)`, `b = 2 Σ w d c` (book eq `lmp2-combined-lambda-boys`).
fn lambda_boys_pair_angle(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
    p: usize,
    q: usize,
) -> f64 {
    let (mut a, mut b) = (0.0, 0.0);
    let mut add = |m: &Array2<f64>, w: f64| {
        let d = 0.5 * (m[(p, p)] - m[(q, q)]);
        let c = m[(p, q)];
        a += w * (d * d - c * c);
        b += w * 2.0 * d * c;
    };
    for m in x {
        add(m, 1.0);
    }
    add(f, lambda);
    0.25 * b.atan2(a)
}

/// `M ← Rᵀ M R` for the plane rotation `p' = c p + s q`, `q' = −s p + c q`.
fn rotate_sym_pair(m: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
    let n = m.nrows();
    for k in 0..n {
        let (a, b) = (m[(k, p)], m[(k, q)]);
        m[(k, p)] = c * a + s * b;
        m[(k, q)] = -s * a + c * b;
    }
    for k in 0..n {
        let (a, b) = (m[(p, k)], m[(q, k)]);
        m[(p, k)] = c * a + s * b;
        m[(q, k)] = -s * a + c * b;
    }
}

/// Result of [`lambda_boys_rotation`].
#[derive(Debug, Clone)]
#[must_use]
pub struct LambdaBoysRotation {
    /// Orthogonal (n, n) rotation: column `i` is new orbital `i` in the
    /// input orbital basis.
    pub u: Array2<f64>,
    /// Sweeps run.
    pub sweeps: usize,
    /// Largest pair gradient at the returned rotation.
    pub max_grad: f64,
    /// Whether `max_grad < grad_tol` was reached within `max_sweeps`.
    pub converged: bool,
    /// `G` before and after.
    pub objective_start: f64,
    /// `G` at the returned rotation.
    pub objective_end: f64,
}

/// Jacobi maximization of the λ-Boys objective over orthogonal rotations of
/// the orbitals in which `x` (dipole matrices) and `f` (occupied Fock block)
/// are expressed.
///
/// Each pair step is the exact maximum of `G` in its plane, so `G` never
/// decreases. Pairs are visited in the fixed order `p < q`, row by row, in
/// one thread: the result does not depend on the thread count. Converged
/// when the largest pair gradient after a full sweep is below `grad_tol`.
/// Errors on a negative or non-finite λ.
pub fn lambda_boys_rotation(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
    grad_tol: f64,
    max_sweeps: usize,
) -> Result<LambdaBoysRotation, ferric_core::FerricError> {
    check_lambda_boys(lambda)?;
    let n = f.nrows();
    let mut x = x.clone();
    let mut f = f.clone();
    let mut u = Array2::<f64>::eye(n);
    let objective_start = lambda_boys_objective(&x, &f, lambda);
    let mut max_grad = lambda_boys_max_grad(&x, &f, lambda);
    let mut sweeps = 0;
    while max_grad >= grad_tol && sweeps < max_sweeps {
        sweeps += 1;
        for p in 0..n {
            for q in p + 1..n {
                // θ* = atan2(b, a)/4: the maximum of G in the (p, q) plane
                let th = lambda_boys_pair_angle(&x, &f, lambda, p, q);
                if th == 0.0 {
                    continue; // b = 0 and a > 0: already the in-plane maximum
                }
                let (c, s) = (th.cos(), th.sin());
                // r^x ← Rᵀ r^x R and f ← Rᵀ f R
                for m in x.iter_mut() {
                    rotate_sym_pair(m, p, q, c, s);
                }
                rotate_sym_pair(&mut f, p, q, c, s);
                // U ← U R: p' = c p + s q, q' = −s p + c q
                for k in 0..n {
                    let (a, b) = (u[(k, p)], u[(k, q)]);
                    u[(k, p)] = c * a + s * b;
                    u[(k, q)] = -s * a + c * b;
                }
            }
        }
        // stop on max over p < q of |∂G/∂θ_pq| after the whole sweep
        max_grad = lambda_boys_max_grad(&x, &f, lambda);
    }
    Ok(LambdaBoysRotation {
        u,
        sweeps,
        max_grad,
        converged: max_grad < grad_tol,
        objective_start,
        objective_end: lambda_boys_objective(&x, &f, lambda),
    })
}

/// λ must be finite and ≥ 0 (0 = plain Boys).
pub fn check_lambda_boys(lambda: f64) -> Result<(), ferric_core::FerricError> {
    if lambda.is_finite() && lambda >= 0.0 {
        Ok(())
    } else {
        Err(ferric_core::FerricError::General(format!(
            "lambda_boys must be finite and >= 0 (0 = plain Boys), got {lambda}"
        )))
    }
}

/// λ-Boys localization of occupied orbitals: converged Boys first
/// ([`boys_localize_checked`]), then [`lambda_boys_rotation`] from the Boys
/// orbitals with `f_ao` the AO Fock matrix.
///
/// `lambda = 0` returns the Boys result unchanged, bit for bit (the
/// optimiser is not run). A λ-Boys optimiser that misses
/// [`LAMBDA_BOYS_GRAD_TOL`] in `max_sweeps` is a `Convergence` error, the
/// same rule as the Boys refusal: the orbitals would still span the
/// occupied space, but every locality map built from them would differ.
pub fn lambda_boys_localize_checked(
    c_occ: &Array2<f64>,
    dip: &[Array2<f64>; 3],
    f_ao: &Array2<f64>,
    lambda: f64,
    boys_max_sweeps: usize,
    max_sweeps: usize,
) -> Result<(BoysResult, LambdaBoysReport), ferric_core::FerricError> {
    check_lambda_boys(lambda)?;
    let boys = boys_localize_checked(c_occ, dip, boys_max_sweeps)?;
    if lambda == 0.0 {
        return Ok((
            boys,
            LambdaBoysReport {
                lambda,
                ..Default::default()
            },
        ));
    }
    let t0 = std::time::Instant::now();
    let c = &boys.c_loc;
    let x: [Array2<f64>; 3] = std::array::from_fn(|k| c.t().dot(&dip[k].dot(c)));
    let f = c.t().dot(&f_ao.dot(c));
    let rot = lambda_boys_rotation(&x, &f, lambda, LAMBDA_BOYS_GRAD_TOL, max_sweeps)?;
    if !rot.converged {
        return Err(ferric_core::FerricError::Convergence(format!(
            "lambda-Boys localization (lambda = {lambda}) did not converge in {} sweeps \
             (max |gradient| {:.2e} >= {LAMBDA_BOYS_GRAD_TOL:.0e}); the local truncation \
             needs converged localized orbitals",
            rot.sweeps, rot.max_grad
        )));
    }
    let c_loc = c.dot(&rot.u).as_standard_layout().into_owned();
    let n = c_loc.ncols();
    let mut centers = Array2::zeros((n, 3));
    for (k, d) in dip.iter().enumerate() {
        let dc = d.dot(&c_loc);
        for i in 0..n {
            centers[(i, k)] = c_loc.column(i).dot(&dc.column(i));
        }
    }
    let report = LambdaBoysReport {
        lambda,
        sweeps: rot.sweeps,
        max_grad: rot.max_grad,
        objective_start: rot.objective_start,
        objective_end: rot.objective_end,
        t_s: t0.elapsed().as_secs_f64(),
    };
    Ok((
        BoysResult {
            c_loc,
            centers,
            iterations: boys.iterations,
            converged: true,
        },
        report,
    ))
}

/// LMP2 domain for a set of Boys-localized orbitals.
///
/// `ao_mask[μ]` is true if basis function μ belongs to at least one orbital's domain.
/// `orbital_domains[i]` is the sorted list of AO indices in orbital i's domain.
#[derive(Debug, Clone)]
pub struct LmpDomains {
    /// For each orbital i: sorted AO indices within cutoff_bohr of its Boys center.
    pub orbital_domains: Vec<Vec<usize>>,
    /// Union mask over all orbitals: `ao_mask[μ] = true` if μ is in any domain.
    pub ao_mask: Vec<bool>,
    /// Sorted list of AO indices that appear in any domain.
    pub active_aos: Vec<usize>,
}

/// Build LMP2 domains from Boys centers and shell spatial information.
///
/// Each shell is assigned to an orbital's domain if the shell center (atom position)
/// lies within `cutoff_bohr` of the orbital's Boys center.  All AO functions of a
/// qualifying shell are included (shells are not split).
///
/// `centers`: (nocc × 3) Boys centers in Bohr.
/// `shell_centers`: per-shell Cartesian centers (Bohr), length nshells.
/// `shell_offsets`: cumulative AO index for each shell, length nshells+1.
pub fn build_domains(
    centers: &Array2<f64>,
    shell_centers: &[[f64; 3]],
    shell_offsets: &[usize],
    cutoff_bohr: f64,
) -> LmpDomains {
    let nocc = centers.nrows();
    let nshells = shell_centers.len();
    let nbas = *shell_offsets.last().unwrap();
    let cutoff_sq = cutoff_bohr * cutoff_bohr;

    let mut orbital_domains: Vec<Vec<usize>> = vec![Vec::new(); nocc];

    for i in 0..nocc {
        let cx = centers[(i, 0)];
        let cy = centers[(i, 1)];
        let cz = centers[(i, 2)];
        for s in 0..nshells {
            let sc = shell_centers[s];
            let dx = sc[0] - cx;
            let dy = sc[1] - cy;
            let dz = sc[2] - cz;
            if dx * dx + dy * dy + dz * dz <= cutoff_sq {
                let start = shell_offsets[s];
                let end = shell_offsets[s + 1];
                for mu in start..end {
                    orbital_domains[i].push(mu);
                }
            }
        }
        orbital_domains[i].sort_unstable();
        orbital_domains[i].dedup();
    }

    // Union mask
    let mut ao_mask = vec![false; nbas];
    for dom in &orbital_domains {
        for &mu in dom {
            ao_mask[mu] = true;
        }
    }
    let active_aos: Vec<usize> = (0..nbas).filter(|&mu| ao_mask[mu]).collect();

    LmpDomains {
        orbital_domains,
        ao_mask,
        active_aos,
    }
}

/// Build a sparse occupied pseudo-density restricted to the LMP2 domains.
///
/// `P(t)_{μν} = Σ_{ij} C^loc_{μi} [exp(+t F_loc)]_{ij} C^loc_{νj}`, masked to
/// the domain-allowed (μ,ν) elements.
///
/// # The index-mismatch bug this replaces (FIXED 2026-07-27)
///
/// The previous implementation took **canonical** coefficients `C^can` with the
/// canonical scalar `exp(t·ε_i)`, but restricted orbital `i`'s contribution to
/// the domain of **Boys-localized** orbital `i`. After a Jacobi localization
/// those two index sets have no correspondence — orbital `i` in one set is not
/// orbital `i` in the other — so the routine did not truncate `P(t)`, it built
/// a *different, wrong* matrix. Retained elements were incorrect, not merely
/// incomplete.
///
/// It also forced exactness to require every domain ball to cover every AO,
/// which **predicts `r(exact) ≈ molecular diameter` a priori, with no physics
/// involved**. That artifact was previously mistaken for a locality finding
/// (see the `ao-laplace-domain-radius-tracks-diameter` write-up).
///
/// # Why the matrix exponential is required
///
/// Localized orbitals are NOT Fock eigenvectors, so the scalar `exp(t·ε_i)`
/// does not apply to them: the occupied-block Fock `F_loc = C_locᵀ F C_loc` is
/// no longer diagonal and the correct weight is the matrix exponential
/// `exp(+t F_loc)`. Pairing localized coefficients with the canonical scalar is
/// exactly the inconsistency above.
///
/// # Correctness anchor
///
/// With domains covering every AO this reproduces the canonical
/// [`crate::laplace::build_pseudo_density_occ`] to round-off, because
/// `C_loc exp(t F_loc) C_locᵀ = C_can exp(t ε) C_canᵀ` — the same operator in
/// two bases. Asserted by `localized_pseudo_density_matches_canonical_when_full`.
///
/// `c_loc`: **localized** occupied MOs (nbas × nocc), from [`boys_localize`].
/// `f_loc`: occupied-block Fock in that SAME localized basis (nocc × nocc).
/// `domains`: orbital domains, indexed consistently with `c_loc`'s columns.
///
/// Errors if `f_loc` cannot be diagonalized (not square, or non-finite).
pub fn build_pseudo_density_occ_sparse(
    c_loc: &Array2<f64>,
    f_loc: &Array2<f64>,
    t: f64,
    domains: &LmpDomains,
) -> Result<Array2<f64>, ferric_core::FerricError> {
    let nbas = c_loc.nrows();
    let nocc = c_loc.ncols();
    debug_assert_eq!(f_loc.dim(), (nocc, nocc));
    debug_assert_eq!(domains.orbital_domains.len(), nocc);

    // exp(+t F_loc): the localized-basis generalization of the canonical
    // scalar exp(t·ε_i). Done once per quadrature point on the small
    // (nocc × nocc) block.
    let expf = sym_matrix_exp(f_loc, t)?;

    // Element mask, as the doc has always promised: keep (μ,ν) iff μ and ν lie
    // in a COMMON orbital domain. This masks the FINAL matrix, not per-orbital
    // contributions — the distinction the old code got wrong. Every retained
    // element is therefore exact.
    let mut keep = vec![false; nbas * nbas];
    for dom in &domains.orbital_domains {
        for &mu in dom {
            for &nu in dom {
                keep[mu * nbas + nu] = true;
            }
        }
    }

    let cw = c_loc.dot(&expf); // (nbas, nocc)
    let mut p = Array2::zeros((nbas, nbas));
    for mu in 0..nbas {
        for nu in 0..nbas {
            if keep[mu * nbas + nu] {
                p[(mu, nu)] = cw.row(mu).dot(&c_loc.row(nu));
            }
        }
    }
    Ok(p)
}

/// `exp(scale · A)` for symmetric `A`, via eigendecomposition.
///
/// Returns an error (never panics) if the eigensolve fails, e.g. for a
/// non-square or non-finite block: that must not feed garbage into an energy.
fn sym_matrix_exp(a: &Array2<f64>, scale: f64) -> Result<Array2<f64>, ferric_core::FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let n = a.nrows();
    let (vals, vecs) = a.eigh(UPLO::Lower).map_err(|e| {
        ferric_core::FerricError::Lapack(format!(
            "sym_matrix_exp: occupied-block Fock eigensolve failed: {e}"
        ))
    })?;
    let mut scaled = Array2::zeros((n, n));
    for k in 0..n {
        let w = (scale * vals[k]).exp();
        for i in 0..n {
            scaled[(i, k)] = vecs[(i, k)] * w;
        }
    }
    Ok(scaled.dot(&vecs.t()))
}

/// Build a sparse virtual pseudo-density restricted to the active AO union.
///
/// `Q(t)_{μν} = Σ_a C_{μa} exp(-t ε_a) C_{νa}` for μ,ν ∈ active_aos only.
/// (Virtuals are not localized — use full virtual space but restrict AO indices.)
pub fn build_pseudo_density_vir_sparse(
    c_vir: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nocc_total: usize,
    domains: &LmpDomains,
) -> Array2<f64> {
    let nbas = c_vir.nrows();
    let nvir = c_vir.ncols();
    let mut q = Array2::zeros((nbas, nbas));
    for a in 0..nvir {
        let factor = (-t * eps[nocc_total + a]).exp();
        for &mu in &domains.active_aos {
            let c_mu_a = c_vir[(mu, a)] * factor;
            for &nu in &domains.active_aos {
                q[(mu, nu)] += c_mu_a * c_vir[(nu, a)];
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron::dipole;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ndarray::s;

    #[test]
    fn sym_matrix_exp_errors_instead_of_panicking() {
        let bad = Array2::<f64>::zeros((2, 3));
        assert!(sym_matrix_exp(&bad, 1.0).is_err());
        let ok = Array2::<f64>::from_diag(&ndarray::arr1(&[0.0, 1.0]));
        let e = sym_matrix_exp(&ok, 1.0).unwrap();
        assert!((e[(1, 1)] - 1.0_f64.exp()).abs() < 1e-14);
    }

    /// Synthetic symmetric matrices for the λ-Boys optimiser: three
    /// "dipole" components and an "F_oo" with a spread diagonal.
    fn lambda_boys_model(n: usize) -> ([Array2<f64>; 3], Array2<f64>) {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut sym = |scale: f64, diag: f64| {
            let mut m = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in i..n {
                    let v = scale * rnd() + if i == j { diag * i as f64 } else { 0.0 };
                    m[(i, j)] = v;
                    m[(j, i)] = v;
                }
            }
            m
        };
        let x = [sym(2.0, 0.7), sym(2.0, -0.4), sym(2.0, 0.2)];
        let f = sym(0.3, -0.5);
        (x, f)
    }

    fn rotated(m: &Array2<f64>, u: &Array2<f64>) -> Array2<f64> {
        u.t().dot(m).dot(u)
    }

    fn offdiag_frob(f: &Array2<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..f.nrows() {
            for j in 0..f.ncols() {
                if i != j {
                    s += f[(i, j)] * f[(i, j)];
                }
            }
        }
        s.sqrt()
    }

    /// The analytic pair gradient matches a central difference of `G` along
    /// the same plane rotation, at λ = 50 (so the F term is not negligible).
    /// Mutations seen to fail (2026-11-18): factor 4 dropped from the
    /// gradient; the F term's weight set to 0 in the gradient; the rotation
    /// sign flipped in `rotate_sym_pair` (the FD then differentiates a
    /// different curve).
    #[test]
    fn lambda_boys_pair_gradient_matches_central_difference() {
        let (x, f) = lambda_boys_model(6);
        let lam = 50.0;
        for (p, q) in [(0usize, 1usize), (2, 5), (3, 4)] {
            let g = |th: f64| {
                let mut xs = x.clone();
                let mut fs = f.clone();
                for m in xs.iter_mut() {
                    rotate_sym_pair(m, p, q, th.cos(), th.sin());
                }
                rotate_sym_pair(&mut fs, p, q, th.cos(), th.sin());
                lambda_boys_objective(&xs, &fs, lam)
            };
            let h = 1e-6;
            let fd = (g(h) - g(-h)) / (2.0 * h);
            let an = lambda_boys_pair_gradient(&x, &f, lam, p, q);
            // central difference: O(h²·G''') truncation + O(ε·G/h) rounding,
            // both ≲ 1e-8 here; 1e-6 relative leaves two decades.
            assert!(
                (fd - an).abs() < 1e-6 * an.abs().max(1.0),
                "pair ({p},{q}): fd {fd} vs analytic {an}"
            );
        }
    }

    /// The Jacobi angle is the in-plane MAXIMUM: G at θ* is ≥ G on a fine
    /// grid of the same plane, and the optimiser's G never falls. Mutation
    /// seen to fail (2026-11-18): `b.atan2(a)` → `(-b).atan2(-a)` (the
    /// in-plane minimum) fails the grid comparison.
    #[test]
    fn lambda_boys_pair_angle_is_the_in_plane_maximum() {
        let (x, f) = lambda_boys_model(5);
        let lam = 10.0;
        for (p, q) in [(0usize, 1usize), (1, 3), (2, 4)] {
            let g = |th: f64| {
                let mut xs = x.clone();
                let mut fs = f.clone();
                for m in xs.iter_mut() {
                    rotate_sym_pair(m, p, q, th.cos(), th.sin());
                }
                rotate_sym_pair(&mut fs, p, q, th.cos(), th.sin());
                lambda_boys_objective(&xs, &fs, lam)
            };
            let th = lambda_boys_pair_angle(&x, &f, lam, p, q);
            let best = (0..2000)
                .map(|k| {
                    g(-std::f64::consts::FRAC_PI_4
                        + k as f64 * std::f64::consts::FRAC_PI_2 / 2000.0)
                })
                .fold(f64::MIN, f64::max);
            assert!(
                g(th) >= best - 1e-10,
                "pair ({p},{q}): G(θ*) {} < grid max {best}",
                g(th)
            );
        }
    }

    /// λ = 0 restarted from its own optimum stays put; λ > 0 from the λ = 0
    /// optimum raises G and makes F_oo more diagonal; the optimiser output
    /// satisfies the stationarity condition (max gradient < tol). Mutations
    /// seen to fail (2026-11-18): the F weight in the angle set to −λ (F
    /// becomes LESS diagonal); the F term dropped from the angle (F does not
    /// move, and the gradient including it never reaches tol).
    #[test]
    fn lambda_boys_penalty_direction_and_lambda_zero_fixed_point() {
        let (x, f) = lambda_boys_model(6);
        let r0 = lambda_boys_rotation(&x, &f, 0.0, 1e-10, 10_000).unwrap();
        assert!(r0.converged);
        let x0: [Array2<f64>; 3] = std::array::from_fn(|k| rotated(&x[k], &r0.u));
        let f0 = rotated(&f, &r0.u);
        // λ = 0 from its own optimum: zero sweeps, U = 1 exactly.
        let again = lambda_boys_rotation(&x0, &f0, 0.0, 1e-10, 10_000).unwrap();
        assert_eq!(again.sweeps, 0);
        assert_eq!(again.u, Array2::<f64>::eye(6));
        // λ = 10 from the Boys optimum.
        let lam = 10.0;
        let r1 = lambda_boys_rotation(&x0, &f0, lam, 1e-10, 10_000).unwrap();
        assert!(r1.converged, "max grad {}", r1.max_grad);
        assert!(r1.objective_end >= r1.objective_start);
        let f1 = rotated(&f0, &r1.u);
        let x1: [Array2<f64>; 3] = std::array::from_fn(|k| rotated(&x0[k], &r1.u));
        assert!(
            offdiag_frob(&f1) < 0.9 * offdiag_frob(&f0),
            "λ > 0 must make F more diagonal: {} -> {}",
            offdiag_frob(&f0),
            offdiag_frob(&f1)
        );
        assert!(lambda_boys_max_grad(&x1, &f1, lam) < 1e-10);
        // U orthogonal to rounding: n·ε-scale per element after O(sweeps·n)
        // plane rotations; 1e-12 is far above that and far below any real
        // non-orthogonality.
        let utu = r1.u.t().dot(&r1.u);
        let dev = (&utu - &Array2::<f64>::eye(6))
            .mapv(f64::abs)
            .fold(0.0f64, |a, &b| a.max(b));
        assert!(dev < 1e-12, "UᵀU − 1 = {dev}");
    }

    /// An optimiser that misses the gradient target is refused, and a bad λ
    /// is an error, not a silent default. λ = 0 is the Boys result bit for
    /// bit: running the optimiser at λ = 0 instead of returning the Boys
    /// orbitals was seen to fail the bitwise comparison (2026-11-18; the
    /// Boys stop leaves a ~1e-10 residual rotation the optimiser removes).
    #[test]
    fn lambda_boys_refuses_nonconvergence_and_bad_lambda() {
        let (x, f) = lambda_boys_model(6);
        let r = lambda_boys_rotation(&x, &f, 10.0, 1e-10, 1).unwrap();
        assert!(!r.converged && r.sweeps == 1);
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(lambda_boys_rotation(&x, &f, bad, 1e-10, 10).is_err());
            assert!(check_lambda_boys(bad).is_err());
        }
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let c_occ = rhf.mos_r().slice(s![.., 1..5]).to_owned();
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
        let f_ao = rhf.fock_r().clone();
        // λ = 0: the Boys orbitals bit for bit, optimiser not run.
        let boys = boys_localize_checked(&c_occ, &dip, 200).unwrap();
        let (b0, rep0) = lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 0.0, 200, 1).unwrap();
        assert_eq!(b0.c_loc, boys.c_loc);
        assert_eq!(b0.centers, boys.centers);
        assert_eq!(rep0.sweeps, 0);
        // λ = 10 with a one-sweep cap: refused; with the production cap: ok.
        let e = lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 10.0, 200, 1);
        assert!(
            matches!(e, Err(ferric_core::FerricError::Convergence(_))),
            "{e:?}"
        );
        let (b10, rep10) =
            lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 10.0, 200, LAMBDA_BOYS_MAX_SWEEPS)
                .unwrap();
        assert!(rep10.max_grad < LAMBDA_BOYS_GRAD_TOL && rep10.sweeps >= 1);
        // still orthonormal occupied orbitals spanning the same space
        let s_ao = ferric_integrals::oneelectron::overlap(&prep);
        let o = b10.c_loc.t().dot(&s_ao).dot(&b10.c_loc);
        let dev = (&o - &Array2::<f64>::eye(4))
            .mapv(f64::abs)
            .fold(0.0f64, |a, &b| a.max(b));
        assert!(dev < 1e-12, "CᵀSC − 1 = {dev}");
        let proj = boys.c_loc.t().dot(&s_ao).dot(&b10.c_loc); // 4×4 orthogonal
        let pp = proj.t().dot(&proj);
        let dev = (&pp - &Array2::<f64>::eye(4))
            .mapv(f64::abs)
            .fold(0.0f64, |a, &b| a.max(b));
        assert!(dev < 1e-12, "span changed: {dev}");
    }

    #[test]
    fn checked_boys_refuses_an_unconverged_localization() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let nocc = mol.nelec() as usize / 2;
        let c_occ = rhf.mos_r().slice(s![.., ..nocc]).to_owned();
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
        // One sweep cannot reach the 1e-8 rotation-angle tolerance.
        assert!(!boys_localize(&c_occ, &dip, 1).converged);
        assert!(boys_localize_checked(&c_occ, &dip, 1).is_err());
        assert!(boys_localize_checked(&c_occ, &dip, 200).is_ok());
    }

    #[test]
    fn test_boys_water_sto3g() {
        // Run RHF on water, then localize occupied orbitals.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();

        let ctx = ParallelContext::default();
        let config = RhfConfig::default();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        let nocc = mol.nelec() as usize / 2;
        // c_occ: (nbas, nocc)
        let c_occ = rhf.mos_r().slice(s![.., ..nocc]).to_owned();

        // Compute dipole integrals at origin (0,0,0)
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();

        let result = boys_localize(&c_occ, &dip, 200);

        println!(
            "Boys localization: {} iterations, converged={}",
            result.iterations, result.converged
        );
        println!("Centers:\n{:.4}", result.centers);

        assert!(
            result.converged,
            "Boys localization did not converge in {} iterations",
            result.iterations
        );
        assert!(
            result.iterations <= 200,
            "Too many iterations: {}",
            result.iterations
        );

        // The localized orbital centers should lie near water molecule atoms.
        // Water has 5 occupied orbitals in STO-3G: 1 O core, 2 O lone pairs, 2 O-H bonds.
        assert_eq!(result.centers.nrows(), nocc);
        assert_eq!(result.centers.ncols(), 3);

        // THE LOAD-BEARING CHECK: the Boys functional Σ_i |⟨i|r|i⟩|² must INCREASE.
        //
        // Everything above (convergence, iteration count, shape, orthonormality) was
        // ALSO satisfied by a version of this routine that rotated the wrong way and
        // converged happily to the *minimum* — i.e. maximally delocalized orbitals
        // with every center collapsed onto the molecular centroid. Nothing caught it
        // for as long as those were the only assertions, so this is the one that
        // matters. See the sign discussion at the θ computation.
        let boys_functional = |c: &Array2<f64>| -> f64 {
            let mut acc = 0.0;
            for alpha in 0..3 {
                let dc = dip[alpha].dot(c);
                for i in 0..c.ncols() {
                    let v = c.column(i).dot(&dc.column(i));
                    acc += v * v;
                }
            }
            acc
        };
        let f_before = boys_functional(&c_occ);
        let f_after = boys_functional(&result.c_loc);
        println!("Boys functional: {f_before:.8} -> {f_after:.8}");
        assert!(
            f_after > f_before,
            "Boys localization must MAXIMIZE Σ|⟨i|r|i⟩|², but it fell {f_before:.8} \
             -> {f_after:.8}; the rotation angle's sign branch is inverted"
        );

        // Physical placement: water's H atoms sit at y = ±1.428 Bohr, so the two O–H
        // bond orbitals must be genuinely displaced along ±y. Collapsed (delocalized)
        // centers all sit at y ≈ 0, which is precisely the failure the sign bug
        // produced — so this pins the geometry, not just the functional value.
        let max_abs_y = (0..nocc)
            .map(|i| result.centers[(i, 1)].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_y > 0.3,
            "no orbital is displaced along y (max |y| = {max_abs_y:.4} Bohr); the O–H \
             bond orbitals have collapsed onto the molecular axis"
        );

        // Distinct orbitals must have distinct centers. The broken version put three
        // orbitals on one center and two on another.
        let mut max_sep = 0.0_f64;
        for i in 0..nocc {
            for j in 0..nocc {
                let d: f64 = (0..3)
                    .map(|a| (result.centers[(i, a)] - result.centers[(j, a)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_sep = max_sep.max(d);
            }
        }
        assert!(
            max_sep > 1.0,
            "all Boys centers are within {max_sep:.4} Bohr of each other — the \
             orbitals are not localized"
        );

        // Check that the localized orbitals are still orthonormal via C^T S C = I
        use ferric_integrals::oneelectron::overlap;
        let s = overlap(&prep);
        let sc = s.dot(&result.c_loc);
        let cts_c = result.c_loc.t().dot(&sc);
        for i in 0..nocc {
            for j in 0..nocc {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (cts_c[(i, j)] - expected).abs();
                assert!(
                    diff < 1e-8,
                    "Orthonormality violated at ({i},{j}): got {:.2e}",
                    diff
                );
            }
        }
    }

    /// CORRECTNESS ANCHOR for the 2026-07-27 index-mismatch fix.
    ///
    /// With every domain covering every AO, the domain mask is vacuous, so the
    /// localized construction `C_loc exp(t F_loc) C_locᵀ` must reproduce the
    /// canonical `C_can exp(t ε) C_canᵀ` to round-off — they are the same
    /// operator expressed in two bases.
    ///
    /// This test FAILS against the old code, which is the point: the old path
    /// paired canonical coefficients with localized-orbital domains, so even
    /// with a full mask it only agreed by accident of the mask, and its whole
    /// premise (scalar exp(t·ε_i) on non-eigenvectors) was wrong.
    #[test]
    fn localized_pseudo_density_matches_canonical_when_full() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                density_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();

        let nocc = mol.nelec() as usize / 2;
        let c = rhf.mos_r();
        let c_occ = c.slice(s![.., ..nocc]).to_owned();
        let eps = rhf.eps_r();
        let nbas = prep.nbasis();

        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();
        let boys = boys_localize(&c_occ, &dip, 200);
        let f_loc = boys.c_loc.t().dot(rhf.fock_r()).dot(&boys.c_loc);

        // TEETH: the localizer must actually have rotated, or "agreement" is
        // trivially true and this test proves nothing.
        let dcoef = (&boys.c_loc - &c_occ)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            dcoef > 0.05,
            "localizer barely moved the orbitals (max d = {dcoef:.3e})"
        );
        // TEETH: F_loc must be genuinely NON-diagonal, else exp(tF) degenerates
        // to the canonical scalar and the matrix exponential is untested.
        let offdiag = (0..nocc)
            .flat_map(|i| (0..nocc).filter(move |&j| i != j).map(move |j| (i, j)))
            .map(|(i, j)| f_loc[(i, j)].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            offdiag > 1e-3,
            "F_loc is essentially diagonal (max offdiag {offdiag:.3e}); the matrix \
             exponential path is not actually exercised"
        );

        // A domain set covering every AO for every orbital => vacuous mask.
        let full = LmpDomains {
            orbital_domains: vec![(0..nbas).collect(); nocc],
            ao_mask: vec![true; nbas],
            active_aos: (0..nbas).collect(),
        };

        for &t in &[0.05_f64, 0.3, 1.0, 3.0] {
            let p_loc = build_pseudo_density_occ_sparse(&boys.c_loc, &f_loc, t, &full).unwrap();
            // Canonical reference, built inline so this test does not depend on
            // the laplace module's blocking/frozen-core plumbing.
            let mut p_can: Array2<f64> = Array2::zeros((nbas, nbas));
            for i in 0..nocc {
                let w = (t * eps[i]).exp();
                for mu in 0..nbas {
                    let cmu = c_occ[(mu, i)] * w;
                    for nu in 0..nbas {
                        p_can[(mu, nu)] += cmu * c_occ[(nu, i)];
                    }
                }
            }
            let maxerr = (&p_loc - &p_can)
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            let scale = p_can.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            eprintln!(
                "t={t:.3}: |dP|max = {maxerr:.3e} (rel {:.3e})",
                maxerr / scale
            );
            assert!(
                maxerr / scale < 1e-10,
                "localized exp(tF) form must reproduce the canonical pseudo-density \
                 when the mask is vacuous: t={t}, rel err {:.3e}",
                maxerr / scale
            );
        }
    }
}
