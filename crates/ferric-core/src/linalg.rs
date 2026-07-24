//! Small, dependency-light LAPACK helpers shared across the workspace.
//!
//! # Why this module exists
//!
//! The symmetric/Hermitian eigensolver reached through `ndarray-linalg`'s
//! [`Eigh`](ndarray_linalg::Eigh) trait resolves (via the `lax` crate) to the
//! QR-algorithm LAPACK drivers `dsyev_`/`zheev_`. `lax` hardcodes those and
//! exposes **no** divide-and-conquer variant — there is no feature flag or
//! parameter to switch. For the large symmetric Fock / dielectric / metric
//! diagonalizations ferric performs every SCF iteration and every RPA
//! quadrature point, the divide-and-conquer driver `dsyevd_` is materially
//! faster (~5x measured at n=414 single-threaded: 99 ms → 21 ms) while
//! computing the *same* decomposition.
//!
//! [`eigh_dc`] wraps `dsyevd_` directly (through the already-linked
//! `lapack-sys` bindings and the OpenBLAS backend the rest of the workspace
//! uses) with the standard LAPACK workspace-query calling convention, and
//! returns results in the **same convention as `ndarray_linalg::Eigh::eigh`**:
//! eigenvalues ascending, eigenvectors as the **columns** of a
//! standard-layout `(n, n)` array (`a · vecs[:, i] = evals[i] · vecs[:, i]`).
//!
//! # Layout note (row-major ndarray vs column-major LAPACK)
//!
//! LAPACK is column-major; `ndarray` is row-major. For a **symmetric** input
//! `A = Aᵀ`, the row-major byte buffer of `A` is bit-identical to the
//! column-major byte buffer of `A`, so we can hand the buffer to LAPACK
//! without transposing the input (this is why we require a symmetric matrix).
//! On output `dsyevd_` overwrites that buffer with the eigenvectors as
//! **columns in column-major order**. Read back as a row-major `ndarray`,
//! those columns land in the **rows**, so [`eigh_dc`] transposes once at the
//! end to restore the "eigenvectors in columns" convention that every ferric
//! call site (and `ndarray_linalg`) expects.
//!
//! # When it is (and is NOT) safe to swap `.eigh()` → these helpers
//!
//! `dsyev_` (QR) and `dsyevd_` (divide-and-conquer) return **identical
//! eigenvalues** but may return **different eigenvectors inside a degenerate
//! eigenspace** — any orthonormal basis of that subspace is a valid answer,
//! and the two algorithms make different (equally correct) choices.
//!
//! - **Safe:** consumers that use only the eigen*values*
//!   ([`eigvalsh_dc`] — e.g. RPA per-frequency dielectric trace-log sweeps),
//!   and consumers that build a **symmetric matrix function** `U f(Λ) Uᵀ`
//!   (S^{±1/2}, V^{-1/2}) whose result is invariant to degenerate-subspace
//!   rotation.
//! - **NOT safe:** consumers that feed the raw eigenvectors forward in a way
//!   that must reproduce a specific basis — most importantly the
//!   every-iteration SCF **Fock diagonalization** (`ferric-scf`
//!   `driver::diagonalize_rect`). There, the dsyevd eigenvector choice on a
//!   symmetric molecule (methane/Td degenerate MOs) shifts the SCF fixed point
//!   just enough that the density-change ΔP limit-cycles above the 1e-8
//!   convergence gate (the *energy* stays correct to ~1e-8 Ha, but the run is
//!   reported non-converged). That call site is deliberately left on `.eigh()`.

use std::os::raw::{c_char, c_int};

use ndarray::Array2;
use crate::error::FerricError;

/// Which triangle of the (symmetric) input holds the data. Mirrors
/// `ndarray_linalg::UPLO` so call sites read the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    Upper,
    Lower,
}

/// Divide-and-conquer symmetric eigensolver (`dsyevd_`).
///
/// Computes all eigenvalues (ascending) and eigenvectors of the real
/// symmetric matrix `a`. Returns `(evals, evecs)` where `evecs.column(i)` is
/// the normalized eigenvector for `evals[i]` — identical convention to
/// `ndarray_linalg::Eigh::eigh(UPLO::…)`.
///
/// `a` MUST be square and symmetric (only the `uplo` triangle is read, exactly
/// like `dsyev`/`eigh`). A non-zero LAPACK `info` return surfaces as
/// [`FerricError::Lapack`] rather than a panic.
pub fn eigh_dc(a: &Array2<f64>, uplo: Uplo) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err(FerricError::Lapack(format!(
            "eigh_dc: matrix must be square, got {nrows}x{ncols}"
        )));
    }
    let n = nrows;
    if n == 0 {
        return Ok((Vec::new(), Array2::<f64>::zeros((0, 0))));
    }

    // Column-major work buffer. We copy `a` in column-major order (buf[i + j*n]
    // = a[[i, j]]). Because LAPACK reads only the `uplo` triangle and the input
    // is symmetric, this is exact. On output `dsyevd_` overwrites `buf` with the
    // eigenvectors as columns in column-major order, i.e. eigenvector `j`
    // occupies buf[0 + j*n .. n + j*n]. We reconstruct the ndarray from that
    // below so eigenvectors end up in the COLUMNS of a standard-layout array,
    // matching `ndarray_linalg::Eigh::eigh`.
    let mut buf: Vec<f64> = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            buf[i + j * n] = a[[i, j]];
        }
    }

    // `uplo` names a triangle of `a` in ndarray's row-major sense. Our `buf` is
    // column-major, so the same bytes present the *opposite* triangle to
    // LAPACK. Flip the flag so LAPACK reads the triangle the caller intended.
    // (For a symmetric matrix the result is identical either way, but flip for
    // honesty / correctness under a non-symmetric input passed by mistake.)
    let uplo_lapack: c_char = match uplo {
        Uplo::Upper => b'L' as c_char,
        Uplo::Lower => b'U' as c_char,
    };
    let jobz: c_char = b'V' as c_char; // eigenvalues + eigenvectors
    let n_i = n as c_int;

    let mut evals: Vec<f64> = vec![0.0; n];
    let mut info: c_int = 0;

    // --- Workspace query (lwork = -1, liwork = -1) -------------------------
    let mut work_query = [0.0f64; 1];
    let mut iwork_query = [0 as c_int; 1];
    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo_lapack,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            evals.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &(-1 as c_int),
            iwork_query.as_mut_ptr(),
            &(-1 as c_int),
            &mut info,
        );
    }
    if info != 0 {
        return Err(FerricError::Lapack(format!(
            "eigh_dc: dsyevd_ workspace query failed (info={info})"
        )));
    }
    let lwork = work_query[0] as usize;
    let liwork = iwork_query[0] as usize;
    // LAPACK guarantees lwork ≥ 1+6n+2n², liwork ≥ 3+5n for jobz='V'; use the
    // returned optimum but never allocate below the documented floor.
    let lwork = lwork.max(1 + 6 * n + 2 * n * n);
    let liwork = liwork.max(3 + 5 * n);

    let mut work: Vec<f64> = vec![0.0; lwork];
    let mut iwork: Vec<c_int> = vec![0; liwork];
    let lwork_i = lwork as c_int;
    let liwork_i = liwork as c_int;

    // --- Real solve --------------------------------------------------------
    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo_lapack,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            evals.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork_i,
            iwork.as_mut_ptr(),
            &liwork_i,
            &mut info,
        );
    }
    if info != 0 {
        return Err(FerricError::Lapack(format!(
            "eigh_dc: dsyevd_ failed to converge or bad argument (info={info})"
        )));
    }

    // `buf` now holds eigenvectors as column-major columns: eigenvector j lives
    // in buf[i + j*n]. Place it into evecs[[i, j]] so eigenvectors are the
    // COLUMNS of a standard-layout (row-major) array — the ndarray_linalg
    // convention.
    let mut evecs = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        for i in 0..n {
            evecs[[i, j]] = buf[i + j * n];
        }
    }

    Ok((evals, evecs))
}

/// Divide-and-conquer symmetric eigen*value* solver (`dsyevd_`, `jobz='N'`).
///
/// Returns only the ascending eigenvalues of the real symmetric matrix `a`,
/// skipping eigenvector accumulation (cheaper than [`eigh_dc`] when the vectors
/// are unused — e.g. RPA per-frequency dielectric eigenvalue sweeps). Same
/// eigenvalues as `eigh_dc(a, uplo).0` and as `ndarray_linalg::Eigh::eigh`.
pub fn eigvalsh_dc(a: &Array2<f64>, uplo: Uplo) -> Result<Vec<f64>, FerricError> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err(FerricError::Lapack(format!(
            "eigvalsh_dc: matrix must be square, got {nrows}x{ncols}"
        )));
    }
    let n = nrows;
    if n == 0 {
        return Ok(Vec::new());
    }

    // Column-major copy (symmetric → row/col-major buffers coincide). With
    // jobz='N' the input triangle is still read but `buf` is not overwritten
    // with eigenvectors.
    let mut buf: Vec<f64> = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            buf[i + j * n] = a[[i, j]];
        }
    }

    let uplo_lapack: c_char = match uplo {
        Uplo::Upper => b'L' as c_char,
        Uplo::Lower => b'U' as c_char,
    };
    let jobz: c_char = b'N' as c_char; // eigenvalues only
    let n_i = n as c_int;

    let mut evals: Vec<f64> = vec![0.0; n];
    let mut info: c_int = 0;

    let mut work_query = [0.0f64; 1];
    let mut iwork_query = [0 as c_int; 1];
    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo_lapack,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            evals.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &(-1 as c_int),
            iwork_query.as_mut_ptr(),
            &(-1 as c_int),
            &mut info,
        );
    }
    if info != 0 {
        return Err(FerricError::Lapack(format!(
            "eigvalsh_dc: dsyevd_ workspace query failed (info={info})"
        )));
    }
    // jobz='N' floors: lwork ≥ 2n+1, liwork ≥ 1.
    let lwork = (work_query[0] as usize).max(2 * n + 1);
    let liwork = (iwork_query[0] as usize).max(1);

    let mut work: Vec<f64> = vec![0.0; lwork];
    let mut iwork: Vec<c_int> = vec![0; liwork];
    let lwork_i = lwork as c_int;
    let liwork_i = liwork as c_int;

    unsafe {
        lapack_sys::dsyevd_(
            &jobz,
            &uplo_lapack,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            evals.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork_i,
            iwork.as_mut_ptr(),
            &liwork_i,
            &mut info,
        );
    }
    if info != 0 {
        return Err(FerricError::Lapack(format!(
            "eigvalsh_dc: dsyevd_ failed to converge or bad argument (info={info})"
        )));
    }

    Ok(evals)
}

/// `ln det(A)` for a real matrix with a **positive** determinant, via LU
/// factorization (`dgetrf_`).
///
/// # Why this exists
///
/// The RPA correlation energy is a *trace* functional of the dielectric
/// matrix: `Σ_α [ln λ_α + (1 − λ_α)]`. Both terms are basis-invariant, so the
/// individual eigenvalues are never needed — `Σ_α ln λ_α ≡ ln det(ε)` and
/// `Σ_α (1 − λ_α) ≡ tr(I − ε)` (the trace is available for free from the
/// diagonal). LU is ~2/3·n³ flops versus ~4/3·n³ for the eigenvalue-only
/// divide-and-conquer `dsyevd_`, so routing the energy through this function
/// roughly halves the diagonalization slice of the quadrature loop. This is
/// the same formulation PySCF uses (`pyscf/gw/rpa.py`: `np.log(np.linalg.det(
/// ...))`).
///
/// # Layout note
///
/// Unlike [`eigh_dc`]/[`eigvalsh_dc`], `dgetrf_` reads the **whole** matrix,
/// not one triangle — so the row-major/column-major distinction matters in
/// general. It does not matter for the intended caller: `det(Aᵀ) = det(A)`,
/// and reinterpreting a row-major buffer as column-major *is* transposition.
/// So `ln det` is identical either way, for symmetric AND non-symmetric input,
/// and this function copies the buffer straight across without transposing.
/// (Pivoting differs between A and Aᵀ, but the signed determinant does not.)
///
/// # Sign handling
///
/// `det(A) = (−1)^(#row swaps) · Π_i U_ii`. A physically-sensible dielectric
/// matrix at imaginary frequency is positive-definite, so every `U_ii > 0` and
/// the determinant is positive. If the accumulated sign comes out negative, or
/// any `U_ii` is exactly zero (`info > 0`, exactly singular), `ln det` is not a
/// real number: this returns [`FerricError::Lapack`] rather than silently
/// producing `NaN`/`-inf`. A non-finite result likewise errors — a NaN/Inf
/// dielectric (e.g. from a near-degenerate reference poisoning χ₀) must surface
/// as `Err`, per the repo's dielectric-solve reliability convention.
pub fn logdet_lu(a: &Array2<f64>) -> Result<f64, FerricError> {
    let (nrows, ncols) = a.dim();
    if nrows != ncols {
        return Err(FerricError::Lapack(format!(
            "logdet_lu: matrix must be square, got {nrows}x{ncols}"
        )));
    }
    let n = nrows;
    if n == 0 {
        // det(empty) = 1 by convention → ln det = 0.
        return Ok(0.0);
    }

    // Straight buffer copy (no transpose): see the layout note above — LAPACK
    // sees Aᵀ, and det(Aᵀ) = det(A).
    let mut buf: Vec<f64> = Vec::with_capacity(n * n);
    for row in a.rows() {
        buf.extend(row.iter().copied());
    }

    let n_i = n as c_int;
    let mut ipiv: Vec<c_int> = vec![0; n];
    let mut info: c_int = 0;

    unsafe {
        lapack_sys::dgetrf_(
            &n_i,
            &n_i,
            buf.as_mut_ptr(),
            &n_i,
            ipiv.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        return Err(FerricError::Lapack(format!(
            "logdet_lu: dgetrf_ bad argument (info={info})"
        )));
    }
    if info > 0 {
        return Err(FerricError::Lapack(format!(
            "logdet_lu: matrix is exactly singular (U[{i},{i}] = 0, dgetrf_ info={info}); \
             ln det is undefined",
            i = info - 1
        )));
    }

    // det = (−1)^(#swaps) · Π U_ii. `ipiv` is 1-based; entry j records a swap
    // of row j with row ipiv[j], and only ipiv[j] != j+1 counts as a swap.
    let mut negative = false;
    for (j, &p) in ipiv.iter().enumerate() {
        if p != (j as c_int) + 1 {
            negative = !negative;
        }
    }

    let mut log_abs_det = 0.0f64;
    for i in 0..n {
        // Column-major diagonal: U_ii lives at buf[i + i*n].
        let u_ii = buf[i + i * n];
        if u_ii < 0.0 {
            negative = !negative;
        }
        log_abs_det += u_ii.abs().ln();
    }

    if negative {
        return Err(FerricError::Lapack(
            "logdet_lu: determinant is negative — ln det is undefined over the reals \
             (a dielectric matrix at imaginary frequency must be positive-definite)"
                .to_string(),
        ));
    }
    if !log_abs_det.is_finite() {
        return Err(FerricError::Lapack(format!(
            "logdet_lu: non-finite ln det ({log_abs_det}) — NaN/Inf or numerically \
             singular input matrix"
        )));
    }

    Ok(log_abs_det)
}
