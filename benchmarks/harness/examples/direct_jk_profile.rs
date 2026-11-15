//! One closed-shell direct J/K build (`DirectJK::build`) on a fixed density,
//! timed per build, for matched comparisons with PySCF's `get_jk`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 FERRIC_TIMING=1 cargo run --release -p ferric-benchmarks \
//!     --example direct_jk_profile -- <xyz> <basis> <density.f64> <reps> \
//!     [integral_thresh] [out_prefix]
//! ```
//!
//! `density.f64` is the `nbf × nbf` AO density in ferric's AO order as raw
//! little-endian f64, row-major (`scripts/direct_jk_pyscf.py` writes PySCF's
//! initial guess in that form, so both codes contract the same density).
//! Every rep is a FULL build from zero J/K with the same builder; rep 1 also
//! pays the engine-pool construction, which the `setup` line reports apart.
//! With `out_prefix`, the last rep's J and K are written as `<prefix>.J.f64`
//! and `<prefix>.K.f64` (same raw layout) for a bitwise or rounding-bar
//! comparison between builds. Prints one `RESULT` line per rep.
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::DirectJK;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::time::Instant;

fn read_matrix(path: &str, n: usize) -> Array2<f64> {
    let bytes = std::fs::read(path).expect("read density");
    assert_eq!(bytes.len(), n * n * 8, "{path}: expected {n}x{n} f64");
    let v: Vec<f64> = (0..n * n)
        .map(|i| f64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap()))
        .collect();
    Array2::from_shape_vec((n, n), v).unwrap()
}

fn write_matrix(path: &str, m: &Array2<f64>) {
    let bytes: Vec<u8> = m.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write matrix");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: direct_jk_profile <xyz> <basis> <density.f64> <reps> \
             [integral_thresh] [out_prefix]"
        );
        std::process::exit(2);
    }
    let (xyz, basis_name, dpath) = (&args[1], &args[2], &args[3]);
    let reps: usize = args[4].parse().expect("reps");
    let thresh: f64 = args.get(5).map_or(1e-12, |t| t.parse().expect("thresh"));
    let out_prefix = args.get(6);

    ferric_core::blas_threads::init_threading();
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let bs = ferric_core::basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let n = prep.nbasis();
    let d = read_matrix(dpath, n);

    let t = Instant::now();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let schwarz_s = t.elapsed().as_secs_f64();
    println!(
        "SETUP nbf={n} nsh={} threads={} schwarz_s={schwarz_s:.3}",
        prep.nshells(),
        rayon::current_num_threads()
    );

    let mut djk = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX / 4);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    for rep in 1..=reps {
        j.fill(0.0);
        k.fill(0.0);
        let t = Instant::now();
        let quartets = djk.build(&d, &mut j, &mut k).unwrap();
        let wall = t.elapsed().as_secs_f64();
        let e2 = (&d * &(&j - &(0.5 * &k))).sum();
        println!(
            "RESULT rep={rep} wall_s={wall:.4} quartets={quartets} \
             tr_DJ={:.15e} tr_DK={:.15e} e2={e2:.15e}",
            (&d * &j).sum(),
            (&d * &k).sum(),
        );
    }
    if let Some(p) = out_prefix {
        write_matrix(&format!("{p}.J.f64"), &j);
        write_matrix(&format!("{p}.K.f64"), &k);
    }
}
