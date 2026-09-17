//! Does the INTERPOLATED terfc 4-center engine actually support a Schwarz bound?
//!
//! `schwarz.rs` proves the terfc kernel is positive-definite for the EXACT
//! operator, and its own note flags the gap this file measures: ferric's engine
//! is interpolated from tables, so a `(PQ|PQ)` entry computed slightly LOW
//! breaks the Cauchy-Schwarz bound by exactly that much. Positive-definiteness
//! of the exact kernel is necessary but NOT sufficient to license screening on
//! the implemented one.
//!
//! This file measures two things and asserts only what it measures:
//!
//!   1. The diagonal `(PQ|PQ)` blocks are non-negative. A negative diagonal
//!      would make `sqrt()` meaningless and kill Schwarz outright.
//!   2. The bound actually HOLDS on real off-diagonal quartets:
//!      `|(ab|cd)| <= Q_ab * Q_cd`. This is the property screening relies on,
//!      and it is the one interpolation error can break.
//!
//! It deliberately does NOT enable screening anywhere. It is evidence about
//! whether that would be safe.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::os::raw::{c_int, c_void};

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

fn water() -> PreparedBasis {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap()
}

fn terfc_eri4(eng: &mut Engine, prep: &PreparedBasis, q: [usize; 4]) -> Option<Vec<f64>> {
    let dims = prep.shell_dims();
    let n: usize = q.iter().map(|&s| dims[s]).product();
    let mut out = vec![0.0f64; n];
    let h = eng.handle_mut();
    // SAFETY: live terfc engine handle, valid scf_basis, in-bounds shell
    // indices, buffer sized n1*n2*n3*n4. Status checked before reading.
    let written = unsafe {
        ffi::scf_compute_terfc_eri4(
            h,
            prep.handle() as *const c_void,
            q[0] as c_int,
            q[1] as c_int,
            q[2] as c_int,
            q[3] as c_int,
            out.as_mut_ptr(),
        )
    };
    if written < 0 {
        return None;
    }
    Some(out)
}

/// `Q_ij = sqrt(max |(ij|ij)|)` from the diagonal quartet, as schwarz.rs does.
fn q_pair(eng: &mut Engine, prep: &PreparedBasis, i: usize, j: usize) -> Option<(f64, f64)> {
    let dims = prep.shell_dims();
    let (n1, n2) = (dims[i], dims[j]);
    let block = terfc_eri4(eng, prep, [i, j, i, j])?;
    let mut max_diag = 0.0f64;
    let mut min_diag = f64::INFINITY;
    for a in 0..n1 {
        for b in 0..n2 {
            let v = block[((a * n2 + b) * n1 + a) * n2 + b];
            max_diag = max_diag.max(v.abs());
            min_diag = min_diag.min(v);
        }
    }
    Some((max_diag.sqrt(), min_diag))
}

#[test]
fn terfc_4center_diagonals_are_nonnegative() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let prep = water();
    let mut eng = Engine::new_2center(Operator::terfc(2.0), &prep, 0.0).unwrap();
    let nsh = prep.shell_dims().len();

    let mut worst = f64::INFINITY;
    let mut checked = 0usize;
    for i in 0..nsh {
        for j in i..nsh {
            if let Some((_, min_diag)) = q_pair(&mut eng, &prep, i, j) {
                worst = worst.min(min_diag);
                checked += 1;
            }
        }
    }
    assert!(
        checked > 10,
        "only {checked} pairs -- too few to mean anything"
    );
    eprintln!("terfc 4-center: {checked} pairs, min diagonal {worst:.6e}");
    assert!(
        worst >= 0.0,
        "terfc (PQ|PQ) diagonal went NEGATIVE ({worst:.6e}) -- the interpolated \
         engine does not preserve positive-definiteness, so sqrt() in a Schwarz \
         table is meaningless and screening must stay refused"
    );
}

/// The property screening actually depends on: |(ab|cd)| <= Q_ab * Q_cd.
///
/// This is where interpolation error bites. If it holds with margin, a Schwarz
/// table for terfc is safe; if it is violated even slightly, the refusal in
/// schwarz.rs is correct as written and the bound needs an inflation factor.
#[test]
fn terfc_4center_schwarz_bound_holds_on_real_quartets() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let prep = water();
    let mut eng = Engine::new_2center(Operator::terfc(2.0), &prep, 0.0).unwrap();
    let dims = prep.shell_dims();
    let nsh = dims.len();

    // Q table over all pairs.
    let mut qtab = vec![vec![0.0f64; nsh]; nsh];
    for i in 0..nsh {
        for j in 0..nsh {
            let (i0, j0) = if i <= j { (i, j) } else { (j, i) };
            if let Some((q, _)) = q_pair(&mut eng, &prep, i0, j0) {
                qtab[i][j] = q;
            }
        }
    }

    // Worst ratio |(ab|cd)| / (Q_ab Q_cd) over a sweep of off-diagonal quartets.
    // A ratio > 1 is a BOUND VIOLATION.
    let mut worst_ratio = 0.0f64;
    let mut worst_at = [0usize; 4];
    let mut checked = 0usize;
    let step = (nsh / 4).max(1);

    for a in (0..nsh).step_by(step) {
        for b in (0..nsh).step_by(step) {
            for c in (0..nsh).step_by(step) {
                for d in (0..nsh).step_by(step) {
                    let Some(block) = terfc_eri4(&mut eng, &prep, [a, b, c, d]) else {
                        continue;
                    };
                    let bound = qtab[a][b] * qtab[c][d];
                    if bound <= 0.0 {
                        continue;
                    }
                    let maxv = block.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    // Ignore numerically-zero blocks: a ratio of two noise-level
                    // numbers is not evidence about the bound.
                    if maxv < 1e-14 {
                        continue;
                    }
                    let ratio = maxv / bound;
                    if ratio > worst_ratio {
                        worst_ratio = ratio;
                        worst_at = [a, b, c, d];
                    }
                    checked += 1;
                }
            }
        }
    }

    assert!(
        checked > 20,
        "only {checked} quartets -- too few to mean anything"
    );
    eprintln!(
        "terfc Schwarz: {checked} quartets, worst |(ab|cd)|/(Q_ab Q_cd) = \
         {worst_ratio:.9} at {worst_at:?}"
    );
    // MEASURED (water/cc-pVDZ, r0=2, 248 quartets): worst ratio is 1 + 8.88e-16
    // -- ONE ULP -- and it occurs at [6,0,0,6], a diagonal-type quartet
    // (a==d, b==c) where Cauchy-Schwarz is an EQUALITY by construction. So the
    // bound is saturated exactly where theory says it must be, and the overshoot
    // is float rounding on that equality, not interpolation error leaking
    // through. That is the signature of a working bound, not a broken one.
    //
    // The 1e-9 slack is for rounding only. If this ever fires by more than a few
    // ulp, the refusal in schwarz.rs is RIGHT and Q needs an explicit inflation
    // factor covering table error rather than a straight sqrt of the diagonal.
    assert!(
        worst_ratio <= 1.0 + 1e-9,
        "Schwarz bound VIOLATED for interpolated terfc: ratio {worst_ratio:.9} at \
         {worst_at:?}. Table error makes the bound non-rigorous; screening must \
         stay refused or Q must be inflated to cover it."
    );
}
