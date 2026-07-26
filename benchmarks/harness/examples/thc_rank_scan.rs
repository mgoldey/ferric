//! THC rank measurement — the MVP viability gate for tensor hypercontraction.
//!
//! # The single question
//!
//! THC replaces the RI factorization `(ia|jb) ~ B^P_ia B^P_jb` with a
//! grid-collocation factorization
//! ```text
//!     (ia|jb) ~ sum_{PQ} X_i^P X_a^P  Z_PQ  X_j^Q X_b^Q
//! ```
//! over `N_thc` interpolation points. The per-frequency dRPA cost then goes
//! from `O(naux^2 * nov)` to `O(N_thc^2 * (no+nv))` — a scaling win ONLY if the
//! rank `N_thc` stays a small multiple of `nbf`. Writing
//! ```text
//!     c = N_thc / nbf
//! ```
//! the FLOP crossover (computed for cc-pVDZ alkane shapes) sits near c ~ 6-10:
//! at c=3 THC wins 5-35x and the win GROWS with size; at c=10 it LOSES on small
//! systems and only breaks even around nbf ~ 200.
//!
//! So `c` decides the entire lane, and nothing downstream can rescue a bad one.
//! This harness measures it BEFORE any THC-RPA machinery is built.
//!
//! # Method
//!
//! THC's interpolation points are chosen by a pivoted factorization of the AO
//! product ("collocation") matrix. For each grid point g we form the vector of
//! AO products `chi_mu(r_g) chi_nu(r_g)`; the pivoted QR of that matrix ranks
//! grid points by how much NEW product-space direction each one adds. The
//! decay of the QR diagonal `|R_kk|` IS the rank spectrum — no least-squares
//! fit is needed to read it off, which is what makes this cheap.
//!
//! We report, per system, the `N_thc` needed for `|R_kk|/|R_11|` to fall below
//! a set of thresholds, and the implied `c = N_thc/nbf`.
//!
//! # Pre-registered falsifier (written before running)
//!
//!   c <~ 4, flat or falling with system size  -> REAL lane, proceed
//!   c ~ 5-8                                   -> marginal, decide by workload
//!   c >~ 10, or GROWING with system size      -> DEAD, stop, record it
//!
//! The size trend matters more than any single value: THC's promise is
//! asymptotic, so a `c` that grows with N forecloses it regardless of magnitude.
//!
//! Run (small systems, serial):
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2 \
//!     cargo run --release -p ferric-benchmarks --example thc_rank_scan

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ndarray::Array2;
use std::os::raw::c_int;

const OBS: &str = "sto-3g";
/// Second basis to test the c-vs-basis-quality question (cc-pVDZ is the
/// production small basis; STO-3G is the floor).
const OBS2: &str = "cc-pvdz";

/// Relative thresholds on the pivoted-QR diagonal at which we call the product
/// space "captured". 1e-4..1e-6 brackets the accuracy THC papers target.
const THRESHOLDS: &[f64] = &[1e-3, 1e-4, 1e-5, 1e-6];

/// Column-pivoted QR via LAPACK `dgeqp3`, returning |R_kk| (the rank spectrum).
///
/// Follows the direct-LAPACK conventions in `ferric_core::linalg` (column-major
/// staging buffer, workspace query, `info` checked). Kept local to this
/// harness rather than added to the core library: it exists to answer one
/// exploratory question, and should only be promoted if the lane survives.
fn pivoted_qr_diagonal(a: &Array2<f64>) -> Result<Vec<f64>, String> {
    let (m, n) = (a.nrows(), a.ncols());
    if m == 0 || n == 0 {
        return Err("empty matrix".into());
    }
    // Row-major ndarray -> column-major LAPACK buffer.
    let mut buf = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            buf[i + j * m] = a[[i, j]];
        }
    }
    let m_i = m as c_int;
    let n_i = n as c_int;
    let mut jpvt = vec![0 as c_int; n]; // all free columns
    let mut tau = vec![0.0f64; m.min(n)];
    let mut info: c_int = 0;

    // Workspace query.
    let mut wq = [0.0f64; 1];
    unsafe {
        lapack_sys::dgeqp3_(
            &m_i, &n_i, buf.as_mut_ptr(), &m_i,
            jpvt.as_mut_ptr(), tau.as_mut_ptr(),
            wq.as_mut_ptr(), &(-1 as c_int), &mut info,
        );
    }
    if info != 0 {
        return Err(format!("dgeqp3 workspace query failed: info={info}"));
    }
    let lwork = wq[0] as usize;
    let mut work = vec![0.0f64; lwork.max(1)];
    let lwork_i = work.len() as c_int;
    unsafe {
        lapack_sys::dgeqp3_(
            &m_i, &n_i, buf.as_mut_ptr(), &m_i,
            jpvt.as_mut_ptr(), tau.as_mut_ptr(),
            work.as_mut_ptr(), &lwork_i, &mut info,
        );
    }
    if info != 0 {
        return Err(format!("dgeqp3 failed: info={info}"));
    }
    // |R_kk| lives on the diagonal of the (column-major) factored buffer.
    let k = m.min(n);
    Ok((0..k).map(|i| buf[i + i * m].abs()).collect())
}

/// Rank at which the relative QR diagonal first falls below `thresh`.
fn rank_at(diag: &[f64], thresh: f64) -> Option<usize> {
    let d0 = *diag.first()?;
    if d0 <= 0.0 {
        return None;
    }
    diag.iter().position(|&d| d / d0 < thresh).map(|p| p + 1)
}

/// Build the AO-product collocation matrix `P[(mu,nu), g] = chi_mu(g) chi_nu(g)`,
/// restricted to the unique mu<=nu pairs (the product space is symmetric).
///
/// Rows are product pairs, columns are grid points — so a column-pivoted QR
/// selects GRID POINTS, which is exactly the THC interpolation-point choice.
fn product_collocation(chi: &Array2<f64>) -> Array2<f64> {
    let (nbf, npts) = (chi.nrows(), chi.ncols());
    let npair = nbf * (nbf + 1) / 2;
    let mut p = Array2::<f64>::zeros((npair, npts));
    let mut row = 0usize;
    for mu in 0..nbf {
        for nu in 0..=mu {
            for g in 0..npts {
                p[(row, g)] = chi[(mu, g)] * chi[(nu, g)];
            }
            row += 1;
        }
    }
    p
}

fn scan_system(label: &str, mol: &Molecule, obs_name: &str, n_rad: usize, n_ang: usize) {
    let bs = match basis::bundled(obs_name) {
        Ok(b) => b,
        Err(e) => {
            println!("{label}/{obs_name}: basis unavailable ({e})");
            return;
        }
    };
    let cfg = AtomicGridConfig { n_radial: n_rad, n_angular: n_ang, prune: None };
    let grid = build_atomic_grid(mol, &cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|p| p.xyz).collect();

    let chi = match eval_basis_on_points(mol, &bs, &points) {
        Ok(c) => c,
        Err(e) => {
            println!("{label}/{obs_name}: AO eval failed ({e:?})");
            return;
        }
    };
    let nbf = chi.nrows();
    let npts = chi.ncols();

    let prod = product_collocation(&chi);
    let diag = match pivoted_qr_diagonal(&prod) {
        Ok(d) => d,
        Err(e) => {
            println!("{label}/{obs_name}: pivoted QR failed ({e})");
            return;
        }
    };

    let npair = prod.nrows();
    let max_rank = npair.min(npts);
    print!("{label:>10} {obs_name:>9} {nbf:>5} {npts:>7}");
    let mut grid_limited = false;
    for &t in THRESHOLDS {
        match rank_at(&diag, t) {
            // A rank within 2% of the factorization's own ceiling means the
            // spectrum never actually decayed — the measurement is limited by
            // npair/npts, not by the physics. Report it as invalid, never as a
            // large-but-real c.
            Some(r) if r as f64 > 0.98 * max_rank as f64 => {
                grid_limited = true;
                print!("  {:>5} ({:>5})", "LIM", "-");
            }
            Some(r) => print!("  {:>5} ({:>5.2})", r, r as f64 / nbf as f64),
            None => {
                grid_limited = true;
                print!("  {:>5} ({:>5})", ">max", "-");
            }
        }
    }
    if grid_limited {
        print!("   <- rank hit the npair/npts ceiling ({max_rank}); INVALID, enlarge grid");
    }
    println!();
}

fn main() {
    println!("# THC rank scan — measuring c = N_thc / nbf");
    println!("# Rows: rank at which the pivoted-QR diagonal of the AO-product");
    println!("# collocation matrix falls below each relative threshold.");
    println!("# Each cell: N_thc (c = N_thc/nbf).  c is the number that decides the lane.");
    println!("#");
    println!("# FALSIFIER: c<~4 flat/falling => real; c~5-8 marginal; c>~10 or GROWING => dead.");
    println!();
    print!("{:>10} {:>9} {:>5} {:>7}", "system", "basis", "nbf", "npts");
    for &t in THRESHOLDS {
        print!("  {:>13}", format!("{t:.0e}"));
    }
    println!();

    // Grid size is a COST knob here, not an accuracy knob in the usual sense.
    // We are measuring the rank of the AO product space, which is a property of
    // the BASIS; the grid only has to carry enough points to span that space
    // (rank <= min(npair, npts), so npts must comfortably exceed the rank we
    // expect to find, and no more).
    //
    // A 50x110 integration grid gives 27k-60k points, which makes the pivoted
    // QR absurd: alkane_3/cc-pVDZ would be a 3403 x 60500 matrix (1.65 GB) at
    // ~1.4 TFLOP for dgeqp3. A 20x50 grid gives ~1000 points/atom — still
    // several times any plausible THC rank for these systems — at ~1/30th the
    // QR cost. If a reported rank ever approaches npts, the grid IS the
    // limiting factor and the number must be discarded (flagged below).
    let (n_rad, n_ang) = (20usize, 50usize);

    for n_c in [1usize, 2, 3, 4] {
        let path = format!("testdata/molecules/alkane_{n_c}.xyz");
        let Ok(mol) = Molecule::load_xyz(&path) else {
            println!("alkane_{n_c}: SKIPPED (missing)");
            continue;
        };
        scan_system(&format!("alkane_{n_c}"), &mol, OBS, n_rad, n_ang);
    }
    println!();
    for n_c in [1usize, 2, 3] {
        let path = format!("testdata/molecules/alkane_{n_c}.xyz");
        let Ok(mol) = Molecule::load_xyz(&path) else { continue };
        scan_system(&format!("alkane_{n_c}"), &mol, OBS2, n_rad, n_ang);
    }

    println!();
    println!("# Read c DOWN each basis block: flat/falling => THC's asymptotic");
    println!("# promise holds; growing => the rank tracks system size and the");
    println!("# O(N^3) claim never materializes at reachable sizes.");
}
