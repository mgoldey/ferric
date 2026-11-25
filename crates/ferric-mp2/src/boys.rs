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

/// λ of PLAIN Boys (the λ-Boys optimiser does not run): the occupied
/// localization of the dRPA / LinLCCD / rCCD members of the amplitude family.
pub const LAMBDA_BOYS_PLAIN: f64 = 0.0;

/// Convergence target of [`lambda_boys_rotation`]: the largest pair
/// gradient `|∂G/∂κ_pq|` (Bohr², the units of the Boys term) after a sweep.
pub const LAMBDA_BOYS_GRAD_TOL: f64 = 1e-8;

/// Convergence target of the predicted distance to the λ-Boys maximum: the
/// largest entry `|κ_pq|` (radians) of the Newton step at the returned
/// orbitals ([`LambdaBoysSolver::step_tol`]).
pub const LAMBDA_BOYS_STEP_TOL: f64 = 1e-8;

/// How the λ-Boys optimiser runs: Jacobi sweeps first, then Newton steps
/// on the antisymmetric generator ([`lambda_boys_newton`]), which take no
/// iteration when the sweeps already meet both targets.
///
/// Jacobi sweeps are a coordinate ascent: their per-sweep contraction is
/// set by the ratio of the softest curvature of `G` to the pair
/// curvatures. Near-D6h benzene rings have one near-flat direction (an
/// exact zero mode in an isolated D6h ring, curvature 1e-7 to 1e-2 Bohr² in
/// a distorted ring, a ghost basis or a dimer, against 3.6 to 1e3 for the
/// other modes), and the sweeps then need 1e4 to beyond 5e4 sweeps or
/// stall. The Newton step resolves that mode in two to six iterations
/// (measured 2026-11-25; book section `amplitude-lmp2-lambda-boys-newton`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaBoysSolver {
    /// The target `max_{p<q} |∂G/∂θ_pq|` (Bohr²), for both stages.
    pub grad_tol: f64,
    /// The target `max_{p<q} |κ_pq|` (radians) of the Newton step at the
    /// result: the predicted distance to the maximum, which bounds how far
    /// the orbitals (and every map built from them) are from the optimum.
    pub step_tol: f64,
    /// Jacobi sweeps before the Newton stage takes over.
    pub jacobi_sweeps: usize,
    /// Newton iterations; missing the target after them is an error.
    pub newton_iters: usize,
}

/// The production λ-Boys solver: [`LAMBDA_BOYS_GRAD_TOL`] and
/// [`LAMBDA_BOYS_STEP_TOL`], 100 Jacobi
/// sweeps, 30 Newton iterations. Measured 2026-11-25 (6-31G, λ = 10): the
/// n-alkanes C12, C20 meet both targets in 7 sweeps, take no Newton
/// iteration and keep the Jacobi result bit for bit; sulfamethoxazole
/// meets the gradient target in 39 sweeps and takes one Newton iteration;
/// the S22 benzene pieces that stall take three to six.
pub const LAMBDA_BOYS_SOLVER: LambdaBoysSolver = LambdaBoysSolver {
    grad_tol: LAMBDA_BOYS_GRAD_TOL,
    step_tol: LAMBDA_BOYS_STEP_TOL,
    jacobi_sweeps: 100,
    newton_iters: 30,
};

impl Default for LambdaBoysSolver {
    fn default() -> Self {
        LAMBDA_BOYS_SOLVER
    }
}

/// The occupied localization of the LMP2 drivers: λ (`0` = plain Boys) and
/// the λ-Boys solver. A bare `f64` converts with [`LAMBDA_BOYS_SOLVER`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaBoysSpec {
    /// λ (Bohr²/Ha²).
    pub lambda: f64,
    /// The optimiser settings.
    pub solver: LambdaBoysSolver,
}

impl From<f64> for LambdaBoysSpec {
    fn from(lambda: f64) -> Self {
        Self {
            lambda,
            solver: LAMBDA_BOYS_SOLVER,
        }
    }
}

/// What a λ-Boys localization did, recorded on every LMP2 result.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LambdaBoysReport {
    /// The weight λ of `Σ_i F_ii²` (Bohr²/Ha²). `0` = plain Boys (the
    /// optimiser does not run and the Boys orbitals are used bit for bit).
    pub lambda: f64,
    /// Jacobi sweeps the λ-Boys optimiser ran (0 at λ = 0).
    pub sweeps: usize,
    /// Newton iterations after the sweeps (0 when the sweeps converged).
    pub newton_iters: usize,
    /// `max |κ_pq|` of the Newton step at the returned orbitals (radians;
    /// 0 at λ = 0): the predicted distance to the maximum.
    pub newton_step: f64,
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
    /// Newton iterations after the sweeps ([`lambda_boys_solve`]; 0 from
    /// [`lambda_boys_rotation`] itself).
    pub newton_iters: usize,
    /// `max |κ_pq|` of the Newton step at the result ([`lambda_boys_solve`];
    /// NaN from [`lambda_boys_rotation`], which does not compute it).
    pub newton_step: f64,
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
        newton_iters: 0,
        newton_step: f64::NAN,
        objective_start,
        objective_end: lambda_boys_objective(&x, &f, lambda),
    })
}

// ---------------------------------------------------------------------------
// Second-order stage: Newton on the antisymmetric generator κ, U = U₀ exp(κ)
// (book section `amplitude-lmp2-lambda-boys-newton`)
// ---------------------------------------------------------------------------

/// `(X, w_X)`: the three dipole matrices with weight 1 and `f` with λ.
fn lambda_boys_terms<'a>(
    x: &'a [Array2<f64>; 3],
    f: &'a Array2<f64>,
    lambda: f64,
) -> impl Iterator<Item = (&'a Array2<f64>, f64)> {
    x.iter()
        .map(|m| (m, 1.0))
        .chain(std::iter::once((f, lambda)))
}

/// `[D, Y]` with `D` diagonal (given by its entries `d`): `(d_p − d_q) Y_pq`.
fn diag_comm(d: &[f64], y: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(y.dim(), |(p, q)| (d[p] - d[q]) * y[(p, q)])
}

/// `Mᵀ − M`: the antisymmetric matrix whose `(p, q)` entry is
/// `∂ tr(Mκ)/∂κ_pq` for `κ` antisymmetric with independent `κ_pq`, `p < q`.
fn antisym_part(m: &Array2<f64>) -> Array2<f64> {
    &m.t() - m
}

/// Gradient of `G(U exp(κ))` at `κ = 0` with respect to the independent
/// entries `κ_pq` (`p < q`, `κ_qp = −κ_pq`), as the antisymmetric matrix
/// `γ = Γᵀ − Γ`, `Γ = 2 Σ_X w_X [D_X, X]`, `D_X = diag(X)`. Entry `(p, q)` is
/// `−4 Σ_X w_X (X_pp − X_qq) X_pq`, minus [`lambda_boys_pair_gradient`]
/// (book eq `lmp2-lambda-boys-newton`).
pub fn lambda_boys_generator_gradient(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
) -> Array2<f64> {
    let n = f.nrows();
    let mut gam = Array2::<f64>::zeros((n, n));
    for (m, w) in lambda_boys_terms(x, f, lambda) {
        let d: Vec<f64> = m.diag().to_vec();
        gam.scaled_add(2.0 * w, &diag_comm(&d, m));
    }
    antisym_part(&gam)
}

/// Hessian of `G(U exp(κ))` at `κ = 0` applied to the antisymmetric `eta`,
/// in the coordinates of [`lambda_boys_generator_gradient`]:
/// `Hη = Mᵀ − M`, `M = Σ_X w_X (2[V_X, X] + [E_X, X] + [D_X, A_X])` with
/// `A_X = [X, η]`, `V_X = diag(A_X)`, `E_X = [η, D_X]` (book eq
/// `lmp2-lambda-boys-newton`).
pub fn lambda_boys_hessian_vec(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
    eta: &Array2<f64>,
) -> Array2<f64> {
    let n = f.nrows();
    let mut m_acc = Array2::<f64>::zeros((n, n));
    for (m, w) in lambda_boys_terms(x, f, lambda) {
        let d: Vec<f64> = m.diag().to_vec();
        let a = m.dot(eta) - eta.dot(m);
        let v: Vec<f64> = a.diag().to_vec();
        // E = [η, D] = −[D, η]
        let e = -diag_comm(&d, eta);
        let ex = e.dot(m) - m.dot(&e);
        let term = diag_comm(&v, m) * 2.0 + ex + diag_comm(&d, &a);
        m_acc.scaled_add(w, &term);
    }
    antisym_part(&m_acc)
}

/// Diagonal of `−H` in the same coordinates: `16 a_pq`,
/// `a_pq = Σ_X w_X (d_X² − c_X²)` (the in-plane `−G''(0)`, eq
/// `lmp2-lambda-boys-stationarity`), floored at `1e-3 · max` so it is a
/// positive preconditioner away from the maximum too.
fn lambda_boys_neg_hessian_diag(x: &[Array2<f64>; 3], f: &Array2<f64>, lambda: f64) -> Array2<f64> {
    let n = f.nrows();
    let mut h = Array2::<f64>::zeros((n, n));
    for (m, w) in lambda_boys_terms(x, f, lambda) {
        for p in 0..n {
            for q in 0..n {
                let d = 0.5 * (m[(p, p)] - m[(q, q)]);
                let c = m[(p, q)];
                h[(p, q)] += 16.0 * w * (d * d - c * c);
            }
        }
    }
    let top = h.iter().fold(0.0f64, |a, &b| a.max(b));
    let floor = (1e-3 * top).max(f64::MIN_POSITIVE);
    h.mapv_inplace(|v| v.max(floor));
    h
}

/// `Σ_{p<q} A_pq B_pq` for antisymmetric `A`, `B`.
fn antisym_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    0.5 * (a * b).sum()
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}

/// Rounding noise of the generator gradient at the rotation `u`: the
/// difference between `γ` from `Uᵀ(XU)` and from `(UᵀX)U`, as a 2-norm
/// over `p < q`, times 10. Below it, a gradient component, and the step
/// it would drive along a near-flat mode, is not determined.
fn lambda_boys_grad_noise(
    x0: &[Array2<f64>; 3],
    f0: &Array2<f64>,
    lambda: f64,
    u: &Array2<f64>,
    gam: &Array2<f64>,
) -> f64 {
    let x: [Array2<f64>; 3] = std::array::from_fn(|k| u.t().dot(&x0[k]).dot(u));
    let f = u.t().dot(f0).dot(u);
    let d = lambda_boys_generator_gradient(&x, &f, lambda) - gam;
    10.0 * antisym_dot(&d, &d).sqrt()
}

/// Largest step entry `|κ_pq|` (radians) a Newton iteration may take.
const NEWTON_MAX_STEP: f64 = 0.3;
/// A search direction whose curvature `|pᵀ(−H)p| / pᵀp` is below this
/// fraction of the largest pair curvature `16 a_pq` is flat: an exact
/// symmetry of `G` (measured 2026-11-25: 2e-15 relative for the zero mode of
/// a D6h benzene ring), not a soft mode to resolve (the softest measured,
/// S22 benzene B with ghost A in STO-3G, is 1.4e-9 relative).
const NEWTON_FLAT_REL: f64 = 1e-11;
/// Most flat directions [`NewtonSystem::step`] deflates.
const NEWTON_MAX_FLAT: usize = 8;
/// Relative residual of the inner PCG solve `(−H) κ = γ` for a Newton move.
const NEWTON_CG_RTOL: f64 = 1e-3;
/// Relative residual of the PCG solve that measures the distance to the
/// maximum for the convergence test. A loose solve cannot: along a mode of
/// curvature `h` it leaves a step error `‖r‖/h`, and with `h` ~ 1e-7 (the
/// S22 benzene B + ghost A in STO-3G) `1e-3 ‖γ‖` is a 1e-6 rad error.
const NEWTON_CG_RTOL_FINAL: f64 = 1e-12;

/// The Newton system `(−H) κ = γ` at one point, with its preconditioner
/// ([`lambda_boys_neg_hessian_diag`]) and its flat-curvature threshold.
struct NewtonSystem<'a> {
    x: &'a [Array2<f64>; 3],
    f: &'a Array2<f64>,
    lambda: f64,
    prec: Array2<f64>,
    h_flat: f64,
}

impl<'a> NewtonSystem<'a> {
    fn new(x: &'a [Array2<f64>; 3], f: &'a Array2<f64>, lambda: f64) -> Self {
        let prec = lambda_boys_neg_hessian_diag(x, f, lambda);
        let h_flat = NEWTON_FLAT_REL * prec.iter().fold(0.0f64, |m, &v| m.max(v));
        Self {
            x,
            f,
            lambda,
            prec,
            h_flat,
        }
    }

    /// `(−H) p`
    fn apply(&self, p: &Array2<f64>) -> Array2<f64> {
        -lambda_boys_hessian_vec(self.x, self.f, self.lambda, p)
    }

    /// `p` with its components along the orthonormal `flat` removed.
    fn deflate(p: &Array2<f64>, flat: &[Array2<f64>]) -> Array2<f64> {
        let mut out = p.clone();
        for v in flat {
            out.scaled_add(-antisym_dot(v, &out), v);
        }
        out
    }

    /// PCG on `(−H) κ = γ` in the complement of `flat`, to the residual
    /// 2-norm `r_target`. Negative curvature along a search direction (a
    /// saddle region) ends it with a step to the cap along that direction
    /// (Steihaug). Returns `(κ, cg iterations)`.
    fn pcg(&self, gam: &Array2<f64>, r_target: f64, flat: &[Array2<f64>]) -> (Array2<f64>, usize) {
        let n = self.f.nrows();
        let max_cg = (n * n.saturating_sub(1) / 2).max(1);
        let mut kap = Array2::<f64>::zeros((n, n));
        let mut r = Self::deflate(gam, flat);
        if antisym_dot(&r, &r).sqrt() <= r_target {
            return (kap, 0);
        }
        let mut z = Self::deflate(&(&r / &self.prec), flat);
        let mut p = z.clone();
        let mut rz = antisym_dot(&r, &z);
        let mut it = 0;
        while it < max_cg {
            let ap = Self::deflate(&self.apply(&p), flat);
            let curv = antisym_dot(&p, &ap);
            it += 1;
            if curv <= 0.0 {
                // G is convex along p: both signs raise its quadratic part,
                // the sign of ⟨r, p⟩ raises the linear one. Go to the step
                // cap (Steihaug); the caller's backtracking bounds the move.
                let sgn = if antisym_dot(&r, &p) < 0.0 { -1.0 } else { 1.0 };
                let tau = sgn * NEWTON_MAX_STEP / max_abs(&p).max(f64::MIN_POSITIVE);
                kap.scaled_add(tau, &p);
                break;
            }
            let alpha = rz / curv;
            kap.scaled_add(alpha, &p);
            r.scaled_add(-alpha, &ap);
            if antisym_dot(&r, &r).sqrt() <= r_target {
                break;
            }
            z = Self::deflate(&(&r / &self.prec), flat);
            let rz_new = antisym_dot(&r, &z);
            p = &z + &(&p * (rz_new / rz));
            rz = rz_new;
        }
        (kap, it)
    }

    /// Inexact Newton step for the maximum, [`Self::pcg`] with flat
    /// directions removed: a step whose curvature `κᵀ(−H)κ / κᵀκ` is below
    /// `h_flat` ([`NEWTON_FLAT_REL`]) lies along an exact symmetry of `G`,
    /// where every point is equally a maximum. Its direction is deflated
    /// and the solve repeated (at most [`NEWTON_MAX_FLAT`] times). Returns
    /// `(κ, cg iterations)`.
    fn step(&self, gam: &Array2<f64>, r_target: f64) -> (Array2<f64>, usize) {
        let mut flat: Vec<Array2<f64>> = Vec::new();
        let mut cg_total = 0;
        loop {
            let (kap, cg) = self.pcg(gam, r_target, &flat);
            cg_total += cg;
            let kk = antisym_dot(&kap, &kap);
            if kk == 0.0 || flat.len() == NEWTON_MAX_FLAT {
                return (kap, cg_total);
            }
            let rayleigh = antisym_dot(&kap, &self.apply(&kap)) / kk;
            if rayleigh.abs() > self.h_flat {
                return (kap, cg_total);
            }
            let v = Self::deflate(&kap, &flat);
            let vv = antisym_dot(&v, &v).sqrt();
            if vv == 0.0 {
                return (kap, cg_total);
            }
            flat.push(v / vv);
        }
    }
}

/// Cayley retraction `(1 − κ/2)⁻¹ (1 + κ/2)`: orthogonal for antisymmetric
/// `κ`, equal to `exp(κ)` through second order.
fn cayley(k: &Array2<f64>) -> Result<Array2<f64>, ferric_core::FerricError> {
    use ndarray_linalg::Solve;
    let n = k.nrows();
    let eye = Array2::<f64>::eye(n);
    let lhs = &eye - &(k * 0.5);
    let rhs = &eye + &(k * 0.5);
    let mut out = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let col = lhs.solve(&rhs.column(j).to_owned()).map_err(|e| {
            ferric_core::FerricError::Lapack(format!("lambda-Boys Cayley retraction: {e}"))
        })?;
        out.column_mut(j).assign(&col);
    }
    Ok(out)
}

/// `Uᵀ M U` for each matrix.
fn rotate_all(
    x0: &[Array2<f64>; 3],
    f0: &Array2<f64>,
    u: &Array2<f64>,
) -> ([Array2<f64>; 3], Array2<f64>) {
    let x: [Array2<f64>; 3] = std::array::from_fn(|k| u.t().dot(&x0[k].dot(u)));
    (x, u.t().dot(&f0.dot(u)))
}

/// Result of [`lambda_boys_newton`].
#[derive(Debug, Clone)]
#[must_use]
pub struct LambdaBoysNewton {
    /// The rotation, in the basis of the input matrices.
    pub u: Array2<f64>,
    /// Newton iterations run.
    pub iters: usize,
    /// PCG iterations over all Newton iterations.
    pub cg_iters: usize,
    /// Largest `|γ_pq|` at the returned rotation (the same quantity as
    /// [`LambdaBoysRotation::max_grad`]).
    pub max_grad: f64,
    /// `max |κ_pq|` of the Newton step at the returned rotation (radians):
    /// the predicted distance to the maximum.
    pub step: f64,
    /// Whether `max_grad < grad_tol` and `step < step_tol`.
    pub converged: bool,
}

/// Newton maximization of the λ-Boys objective from the rotation `u0` of
/// the orbitals in which `x0` and `f0` are expressed. At each point it
/// solves `(−H) κ = γ` inexactly (preconditioned CG); it stops,
/// converged, when `max |γ_pq| < solver.grad_tol` and `max |κ_pq| <
/// solver.step_tol`, and otherwise (up to `solver.newton_iters` times) caps
/// `max |κ_pq|` at 0.3 rad and backtracks (halving, at most 30 times) until
/// `G` rises or, within rounding of `G`, the gradient falls. A step that does
/// neither is a `Convergence` error. Serial and in a fixed order:
/// thread-count independent.
pub fn lambda_boys_newton(
    x0: &[Array2<f64>; 3],
    f0: &Array2<f64>,
    lambda: f64,
    u0: &Array2<f64>,
    solver: &LambdaBoysSolver,
) -> Result<LambdaBoysNewton, ferric_core::FerricError> {
    check_lambda_boys(lambda)?;
    let mut u = u0.clone();
    let (mut x, mut f) = rotate_all(x0, f0, &u);
    let mut gam = lambda_boys_generator_gradient(&x, &f, lambda);
    let mut g_max = max_abs(&gam);
    let mut obj = lambda_boys_objective(&x, &f, lambda);
    let (mut iters, mut cg_iters) = (0, 0);
    let mut converged = false;
    let step = loop {
        // The Newton step at the current point: its size is the predicted
        // distance to the maximum, which the gradient alone does not bound
        // along a soft mode. PCG never resolves the step below the
        // gradient's rounding noise (a flat direction stays put).
        let noise = lambda_boys_grad_noise(x0, f0, lambda, &u, &gam);
        let g2 = antisym_dot(&gam, &gam).sqrt();
        let system = NewtonSystem::new(&x, &f, lambda);
        let (mut kap, cg) = system.step(&gam, (NEWTON_CG_RTOL * g2).max(noise));
        cg_iters += cg;
        let mut step = max_abs(&kap);
        if g_max < solver.grad_tol && step < solver.step_tol {
            // converged if an accurate solve confirms the distance
            let (k_fine, cg) = system.step(&gam, (NEWTON_CG_RTOL_FINAL * g2).max(noise));
            cg_iters += cg;
            step = max_abs(&k_fine);
            if step < solver.step_tol {
                converged = true;
                break step;
            }
            kap = k_fine;
        }
        if iters == solver.newton_iters {
            break step;
        }
        iters += 1;
        let big = max_abs(&kap);
        if big > NEWTON_MAX_STEP {
            kap *= NEWTON_MAX_STEP / big;
        }
        let obj_round = 64.0 * f64::EPSILON * obj.abs().max(1.0);
        let mut accepted = false;
        for _ in 0..=30 {
            let u_try = u.dot(&cayley(&kap)?);
            let (x_t, f_t) = rotate_all(x0, f0, &u_try);
            let obj_t = lambda_boys_objective(&x_t, &f_t, lambda);
            let gam_t = lambda_boys_generator_gradient(&x_t, &f_t, lambda);
            let g_t = max_abs(&gam_t);
            if obj_t > obj + obj_round || (obj_t >= obj - obj_round && g_t < g_max) {
                (u, x, f, gam, g_max, obj) = (u_try, x_t, f_t, gam_t, g_t, obj_t);
                accepted = true;
                break;
            }
            kap *= 0.5;
        }
        if !accepted {
            return Err(ferric_core::FerricError::Convergence(format!(
                "lambda-Boys Newton iteration {iters} found no step that raises the \
                 objective (max |gradient| {g_max:.2e}); lambda_boys = 0 (plain Boys) \
                 avoids this optimiser"
            )));
        }
    };
    Ok(LambdaBoysNewton {
        u,
        iters,
        cg_iters,
        max_grad: g_max,
        step,
        converged,
    })
}

/// The λ-Boys optimiser of the LMP2 drivers: [`lambda_boys_rotation`] for
/// at most `solver.jacobi_sweeps` sweeps, then [`lambda_boys_newton`] from
/// the Jacobi rotation. When the sweeps leave both the gradient and the
/// Newton step below their targets, the Newton stage takes no iteration and
/// the result is the Jacobi rotation bit for bit.
pub fn lambda_boys_solve(
    x: &[Array2<f64>; 3],
    f: &Array2<f64>,
    lambda: f64,
    solver: &LambdaBoysSolver,
) -> Result<LambdaBoysRotation, ferric_core::FerricError> {
    let mut rot = lambda_boys_rotation(x, f, lambda, solver.grad_tol, solver.jacobi_sweeps)?;
    let nt = lambda_boys_newton(x, f, lambda, &rot.u, solver)?;
    rot.newton_iters = nt.iters;
    rot.newton_step = nt.step;
    rot.converged = nt.converged;
    if nt.iters > 0 {
        let (xr, fr) = rotate_all(x, f, &nt.u);
        rot.objective_end = lambda_boys_objective(&xr, &fr, lambda);
        rot.u = nt.u;
        rot.max_grad = nt.max_grad;
    }
    Ok(rot)
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
/// optimiser is not run). Otherwise the optimiser is [`lambda_boys_solve`]
/// with `solver`; one that misses `solver.grad_tol` is a `Convergence`
/// error, the same rule as the Boys refusal: the orbitals would still span
/// the occupied space, but every locality map built from them would differ.
pub fn lambda_boys_localize_checked(
    c_occ: &Array2<f64>,
    dip: &[Array2<f64>; 3],
    f_ao: &Array2<f64>,
    lambda: f64,
    boys_max_sweeps: usize,
    solver: &LambdaBoysSolver,
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
    let rot = lambda_boys_solve(&x, &f, lambda, solver)?;
    if !rot.converged {
        return Err(ferric_core::FerricError::Convergence(format!(
            "lambda-Boys localization (lambda = {lambda}) did not converge in {} sweeps \
             and {} Newton iterations (max |gradient| {:.2e}, target {:.0e}; Newton step \
             {:.2e} rad, target {:.0e}); the local truncation needs converged localized \
             orbitals. lambda_boys = 0 (plain Boys) avoids this optimiser",
            rot.sweeps,
            rot.newton_iters,
            rot.max_grad,
            solver.grad_tol,
            rot.newton_step,
            solver.step_tol
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
        newton_iters: rot.newton_iters,
        newton_step: rot.newton_step,
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
        // a solver that stops after one sweep, with no Newton stage
        let one_sweep = LambdaBoysSolver {
            jacobi_sweeps: 1,
            newton_iters: 0,
            ..LAMBDA_BOYS_SOLVER
        };
        // λ = 0: the Boys orbitals bit for bit, optimiser not run.
        let boys = boys_localize_checked(&c_occ, &dip, 200).unwrap();
        let (b0, rep0) =
            lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 0.0, 200, &one_sweep).unwrap();
        assert_eq!(b0.c_loc, boys.c_loc);
        assert_eq!(b0.centers, boys.centers);
        assert_eq!(rep0.sweeps, 0);
        // λ = 10 with a one-sweep cap: refused; with the production cap: ok.
        let e = lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 10.0, 200, &one_sweep);
        assert!(
            matches!(e, Err(ferric_core::FerricError::Convergence(_))),
            "{e:?}"
        );
        let (b10, rep10) =
            lambda_boys_localize_checked(&c_occ, &dip, &f_ao, 10.0, 200, &LAMBDA_BOYS_SOLVER)
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
            &RhfConfig::default().with_density_conv(1e-10),
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

    /// Antisymmetric matrix from deterministic pseudo-random upper entries.
    fn antisym_model(n: usize, seed: u64) -> Array2<f64> {
        let mut st = seed;
        let mut k = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in p + 1..n {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let v = (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                k[(p, q)] = v;
                k[(q, p)] = -v;
            }
        }
        k
    }

    /// `G(U₀ exp(κ))` through the Cayley retraction, which equals `exp`
    /// through second order, so its first and second derivatives at 0 are
    /// those of the book's parametrization.
    fn g_along(x: &[Array2<f64>; 3], f: &Array2<f64>, lam: f64, k: &Array2<f64>) -> f64 {
        let (xr, fr) = rotate_all(x, f, &cayley(k).unwrap());
        lambda_boys_objective(&xr, &fr, lam)
    }

    /// The generator gradient and the Hessian-vector product against
    /// central differences of `G` itself (no pair formula), at a point that
    /// is NOT stationary, at λ = 50: `⟨γ, η⟩ = dG/ds`, and
    /// `⟨η, Hζ⟩ = ∂²G/∂s∂t` by the polarization identity; the diagonal of
    /// `−H` is `16 a_pq`, and `γ_pq` is minus the pair gradient.
    #[test]
    fn lambda_boys_generator_gradient_and_hessian_match_finite_differences() {
        let (x, f) = lambda_boys_model(6);
        let lam = 50.0;
        let gam = lambda_boys_generator_gradient(&x, &f, lam);
        for p in 0..6 {
            for q in p + 1..6 {
                let pg = lambda_boys_pair_gradient(&x, &f, lam, p, q);
                assert!((gam[(p, q)] + pg).abs() < 1e-12 * pg.abs().max(1.0));
            }
        }
        let (eta, zeta) = (antisym_model(6, 7), antisym_model(6, 99));
        let h = 1e-4;
        let fd_g =
            (g_along(&x, &f, lam, &(&eta * h)) - g_along(&x, &f, lam, &(&eta * -h))) / (2.0 * h);
        let an_g = antisym_dot(&gam, &eta);
        assert!((fd_g - an_g).abs() < 1e-6 * an_g.abs(), "{fd_g} vs {an_g}");
        let hz = lambda_boys_hessian_vec(&x, &f, lam, &zeta);
        let an_h = antisym_dot(&eta, &hz);
        let hh = 1e-3;
        let gk = |a: f64, b: f64| g_along(&x, &f, lam, &(&(&eta * a) + &(&zeta * b)));
        let fd_h = (gk(hh, hh) - gk(hh, -hh) - gk(-hh, hh) + gk(-hh, -hh)) / (4.0 * hh * hh);
        assert!(
            (fd_h - an_h).abs() < 1e-5 * an_h.abs().max(1.0),
            "{fd_h} vs {an_h}"
        );
        // symmetric
        let he = lambda_boys_hessian_vec(&x, &f, lam, &eta);
        assert!((antisym_dot(&zeta, &he) - an_h).abs() < 1e-10 * an_h.abs().max(1.0));
        // the diagonal of −H is 16 a_pq (before the preconditioner's floor)
        for (p, q) in [(0, 1), (2, 5), (3, 4)] {
            let mut e = Array2::<f64>::zeros((6, 6));
            e[(p, q)] = 1.0;
            e[(q, p)] = -1.0;
            let hpp = -lambda_boys_hessian_vec(&x, &f, lam, &e)[(p, q)];
            let a: f64 = lambda_boys_terms(&x, &f, lam)
                .map(|(m, w)| {
                    let d = 0.5 * (m[(p, p)] - m[(q, q)]);
                    w * (d * d - m[(p, q)] * m[(p, q)])
                })
                .sum();
            assert!((hpp - 16.0 * a).abs() < 1e-10 * hpp.abs().max(1.0));
        }
    }

    /// RHF/STO-3G, converged Boys on the occupieds above `n_core`, and the
    /// dipole matrices and occupied Fock block in the Boys orbitals.
    fn boys_dipoles_and_fock(mol: &Molecule, n_core: usize) -> ([Array2<f64>; 3], Array2<f64>) {
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::for_rank(0, 1);
        let rhf = solve_rhf(&ctx, mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let nocc = mol.nelec() as usize / 2;
        let c = rhf.mos_r().slice(s![.., n_core..nocc]).to_owned();
        let dip = dipole(&prep, [0.0; 3]).unwrap();
        let boys = boys_localize_checked(&c, &dip, 10_000).unwrap();
        let cl = &boys.c_loc;
        let x: [Array2<f64>; 3] = std::array::from_fn(|k| cl.t().dot(&dip[k].dot(cl)));
        let f = cl.t().dot(&rhf.fock_r().dot(cl));
        (x, f)
    }

    /// An exactly D6h benzene (testdata/molecules/benzene.xyz), STO-3G,
    /// λ = 10: the λ-Boys maximum lies on a flat curve (Hessian eigenvalue
    /// ~1e-13, 2e-15 of the largest; measured 2026-11-25), where every
    /// point is a maximum. The sweeps converge, the Newton stage deflates
    /// the flat direction instead of wandering along it, and the result is
    /// the Jacobi rotation bit for bit. Without the deflation the solver
    /// took 7 Newton iterations along the flat curve (seen, 2026-11-25).
    #[test]
    fn lambda_boys_leaves_the_flat_mode_of_a_d6h_benzene_alone() {
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let (x, f) = boys_dipoles_and_fock(&mol, 6);
        let lam = LAMBDA_BOYS_DEFAULT;
        let jac = lambda_boys_rotation(&x, &f, lam, LAMBDA_BOYS_GRAD_TOL, 100).unwrap();
        assert!(jac.converged);
        let r = lambda_boys_solve(&x, &f, lam, &LAMBDA_BOYS_SOLVER).unwrap();
        assert!(r.converged && r.newton_iters == 0, "{r:?}");
        assert!(r.newton_step < LAMBDA_BOYS_STEP_TOL);
        assert_eq!(r.u, jac.u);
    }

    /// S22 benzene B with ghost A, STO-3G, λ = 10: the λ-Boys maximum has a
    /// soft mode (Hessian eigenvalue ≈ −4e-7 against ≈ −3.6 for the next
    /// one; an isolated D6h ring has an exact zero mode here, measured
    /// 2026-11-25), on which Jacobi sweeps stall. Checks:
    /// * the pre-Newton optimiser (2000 sweeps to the 1e-8 gradient, no
    ///   step test) fails, and the production solver converges through the
    ///   Newton stage;
    /// * the result is a maximum: the dense Hessian is negative definite;
    /// * the result is within the step target of the maximum by the dense
    ///   Newton step (an eigensolve, not the PCG). Without the step test the
    ///   solver stops at the 1e-8 gradient 3.6e-4 rad from it (seen to fail,
    ///   2026-11-25);
    /// * invariance: entering the Newton stage after 100 sweeps or straight
    ///   from the Boys orbitals gives the same orbitals to 1e-7 and the same
    ///   `G`.
    #[test]
    fn lambda_boys_newton_converges_and_is_entry_point_invariant_on_a_soft_benzene() {
        let dimer =
            std::fs::read_to_string("../../testdata/molecules/s22/benzene_dimer_T.xyz").unwrap();
        let atoms: Vec<&str> = dimer
            .lines()
            .skip(2)
            .filter(|l| !l.trim().is_empty())
            .collect();
        let mut xyz = format!("{}\nS22 benzene B + ghost A\n", atoms.len());
        for (i, l) in atoms.iter().enumerate() {
            xyz.push_str(&if i < 12 {
                format!("@{}\n", l.trim())
            } else {
                format!("{l}\n")
            });
        }
        let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
        let (x, f) = boys_dipoles_and_fock(&mol, 6);
        let lam = LAMBDA_BOYS_DEFAULT;
        let old = LambdaBoysSolver {
            step_tol: f64::INFINITY,
            jacobi_sweeps: 2000,
            newton_iters: 0,
            ..LAMBDA_BOYS_SOLVER
        };
        assert!(
            !lambda_boys_rotation(&x, &f, lam, old.grad_tol, old.jacobi_sweeps)
                .unwrap()
                .converged
        );
        let a = lambda_boys_solve(&x, &f, lam, &LAMBDA_BOYS_SOLVER).unwrap();
        assert!(a.converged && a.newton_iters >= 1, "{a:?}");
        // the other entry point: Newton straight from the Boys orbitals
        let late = LambdaBoysSolver {
            jacobi_sweeps: 0,
            ..LAMBDA_BOYS_SOLVER
        };
        let b = lambda_boys_solve(&x, &f, lam, &late).unwrap();
        assert!(b.converged, "{b:?}");
        assert!((a.objective_end - b.objective_end).abs() < 1e-10);
        // same orbitals up to order and sign: match each column of Ua to the
        // column of Ub it overlaps most, and compare the coefficients (an
        // error linear in the rotation angle, unlike 1 − |overlap|)
        let o = a.u.t().dot(&b.u);
        let n = o.nrows();
        let mut worst = 0.0f64;
        for i in 0..n {
            let j = (0..n)
                .max_by(|&j1, &j2| o[(i, j1)].abs().total_cmp(&o[(i, j2)].abs()))
                .unwrap();
            let sg = o[(i, j)].signum();
            for k in 0..n {
                worst = worst.max((a.u[(k, i)] - sg * b.u[(k, j)]).abs());
            }
        }
        assert!(worst < 1e-7, "entry points disagree by {worst:.2e} in U");
        // a maximum: the dense Hessian at the result is negative definite
        let (xa, fa) = rotate_all(&x, &f, &a.u);
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .collect();
        let mut hm = Array2::<f64>::zeros((pairs.len(), pairs.len()));
        for (j, &(p, q)) in pairs.iter().enumerate() {
            let mut e = Array2::<f64>::zeros((n, n));
            e[(p, q)] = 1.0;
            e[(q, p)] = -1.0;
            let col = lambda_boys_hessian_vec(&xa, &fa, lam, &e);
            for (i, &(r, t)) in pairs.iter().enumerate() {
                hm[(i, j)] = col[(r, t)];
            }
        }
        use ndarray_linalg::{Eigh, UPLO};
        let (ev, evec) = hm.eigh(UPLO::Lower).unwrap();
        let top = ev[ev.len() - 1];
        assert!(top < 0.0 && top > -1e-5, "softest curvature {top:.3e}");
        // distance to the maximum by the dense Newton step −H⁻¹γ (an
        // eigensolve, independent of the PCG): within the step target
        let ga = lambda_boys_generator_gradient(&xa, &fa, lam);
        let gv: ndarray::Array1<f64> = pairs.iter().map(|&(p, q)| ga[(p, q)]).collect();
        let dense_step = evec.dot(&(&evec.t().dot(&gv) / &ev));
        let dist = dense_step.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            dist < LAMBDA_BOYS_STEP_TOL,
            "dense Newton step {dist:.2e} rad"
        );
    }
}
