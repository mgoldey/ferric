//! Sparsity + FLOP accounting for the attenuated RI fitting metric.
//!
//! Timing on a loaded box is unreliable, and the question this lane actually
//! turns on is not "how fast is it today" but "does `V_w` become sparse enough
//! that a banded/sparse solve beats the dense one". That is a STRUCTURAL
//! property of the matrix — measurable exactly, with no wall-clock noise.
//!
//! Measures, per system and per omega_m:
//!   - fraction of |V_w| entries above a drop tolerance (density)
//!   - the same for V_w^{-1} (an attenuated V can be sparse while its INVERSE
//!     fills in — that would defeat the whole idea, so it must be checked)
//!   - implied FLOPs for the dense vs sparse-exploiting inverse/solve
//!
//! FLOP model (leading order, naux = M, nov = number of (occ,vir) pairs):
//!   dense inverse/solve         : (2/3) M^3
//!   banded (bandwidth b)        : ~2 M b^2
//!   dressing GEMM  V^-1 . A     : 2 M^2 nov   dense
//!                               : 2 M nnz_row nov  sparse (nnz_row = avg row nnz)
//! The dressing GEMM dominates at production sizes (nov >> M), so that is the
//! number that decides the lane.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example metric_sparsity_flops

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ndarray::Array2;
use ndarray_linalg::Inverse;

const OBS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";
const OMEGA_M: &[f64] = &[0.2, 0.4, 0.8, 1.2, 2.0];

/// Relative drop tolerance: |v_ij| < tol * max|V| counts as zero. 1e-6 is far
/// below the metric-bias level the error gate measured, so anything dropped
/// here is genuinely negligible to the energy.
const DROP_TOL: f64 = 1e-6;

/// Density (fraction nonzero) and mean row-nnz of `m` under `DROP_TOL`.
fn density(m: &Array2<f64>) -> (f64, f64) {
    let maxabs = m.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let thresh = DROP_TOL * maxabs.max(1e-300);
    let n = m.nrows();
    let nnz = m.iter().filter(|v| v.abs() >= thresh).count();
    (nnz as f64 / (n * n) as f64, nnz as f64 / n as f64)
}

/// Half-bandwidth: max |i-j| over entries above tolerance.
fn bandwidth(m: &Array2<f64>) -> usize {
    let maxabs = m.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let thresh = DROP_TOL * maxabs.max(1e-300);
    let n = m.nrows();
    let mut b = 0usize;
    for i in 0..n {
        for j in 0..n {
            if m[(i, j)].abs() >= thresh {
                b = b.max(i.abs_diff(j));
            }
        }
    }
    b
}

fn main() {
    println!("# Attenuated RI metric: sparsity + FLOP accounting (drop tol {DROP_TOL:.0e})");
    println!("# basis {OBS} / aux {AUX}");
    println!("#");
    println!("# KEY QUESTION: is V_w sparse AND does V_w^-1 stay sparse?");
    println!("# An attenuated V that inverts to a DENSE matrix wins nothing:");
    println!("# the dressing GEMM uses V^-1, not V.");
    println!();

    for n_c in [1usize, 2, 3, 4, 6, 8] {
        let path = format!("testdata/molecules/alkane_{n_c}.xyz");
        let Ok(mol) = Molecule::load_xyz(&path) else {
            println!("alkane_{n_c}: SKIPPED (missing)");
            continue;
        };
        let obs_bs = basis::bundled(OBS).unwrap();
        let aux_bs = basis::bundled(AUX).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        let nocc = mol.nelec() as usize / 2;
        let nov = nocc * (nbas - nocc);

        println!("### alkane_{n_c}   naux={naux}  nbas={nbas}  nov={nov}");
        println!(
            "{:>8}  {:>10}  {:>10}  {:>8}  {:>14}  {:>10}",
            "omega_m", "dens(V_w)", "dens(V^-1)", "bandwid", "GEMM flops", "vs dense"
        );

        // Coulomb reference row.
        let v_c = threeindex::coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
        let vc_inv = v_c.inv().unwrap();
        let (d_vc, _) = density(&v_c);
        let (d_vci, rownnz_c) = density(&vc_inv);
        let dense_gemm = 2.0 * (naux as f64) * (naux as f64) * (nov as f64);
        println!(
            "{:>8}  {:>10.3}  {:>10.3}  {:>8}  {:>14.3e}  {:>10}",
            "coulomb", d_vc, d_vci, bandwidth(&v_c), 2.0 * naux as f64 * rownnz_c * nov as f64, "1.00x"
        );

        for &w in OMEGA_M {
            let v_w = match threeindex::coulomb_metric_2c(Operator::erfc(w), &dfbs) {
                Ok(v) => v,
                Err(e) => {
                    println!("{w:>8.2}  metric build failed: {e}");
                    continue;
                }
            };
            let Ok(v_inv) = v_w.inv() else {
                println!("{w:>8.2}  {:>10}  {:>10}  {:>8}  {:>14}  {:>10}", "-", "-", "-", "-", "unusable");
                continue;
            };
            let (d_v, _) = density(&v_w);
            let (d_vi, rownnz) = density(&v_inv);
            let gemm = 2.0 * naux as f64 * rownnz * nov as f64;
            println!(
                "{w:>8.2}  {d_v:>10.3}  {d_vi:>10.3}  {:>8}  {gemm:>14.3e}  {:>9.2}x",
                bandwidth(&v_w),
                dense_gemm / gemm.max(1.0)
            );
        }
        println!();
    }

    println!("# Read dens(V^-1) as the verdict: if it stays ~1.0 while dens(V_w) falls,");
    println!("# the metric is sparse but its inverse is not, and the dressing GEMM");
    println!("# -- the dominant cost at production size -- sees no speedup at all.");
}
