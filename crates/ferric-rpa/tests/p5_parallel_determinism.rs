//! P5: bit-identical NPZ-feeding values across thread counts for the newly
//! parallelized per-frequency dynamic-polarizability loops and per-atom
//! esp/field loops (properties.rs, dispersion.rs).
//!
//! Mirrors the freq_quad_parallel.rs pattern: build a dedicated rayon pool
//! at a fixed thread count, run the routine inside `pool.install`, and
//! compare raw `f64::to_bits` between a 1-thread and a 4-thread run. Any
//! reordering of the parallel reduction (order-preserving collect broken,
//! or shared scratch aliased across workers) would show up as a bit
//! mismatch here even where the physics is trivially small enough that an
//! epsilon-based comparison would hide it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{esp_at_atoms, pdep_polarizability_becke_dynamic};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run_pool<T>(n_threads: usize, f: impl FnOnce() -> T + Send) -> T
where
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    pool.install(f)
}

/// `pdep_polarizability_becke_dynamic` (properties.rs :1160, CS frequency
/// loop at the site formerly cited as :1528) must give bit-identical
/// per-atom α^A(iω_k) tensors whether the frequency axis is processed by a
/// 1-thread or a 4-thread rayon pool. Small H2O/cc-pVDZ system, a handful
/// of imaginary frequencies — cheap enough to run at full precision in CI.
#[test]
fn becke_dynamic_polarizability_bit_identical_across_thread_counts() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let cfg = PdepRpaConfig::default();
    let freqs = vec![0.0, 0.25, 0.5, 1.0, 2.0];

    let run = |n: usize| {
        let (mol, obs, obs_bs, dfbs, op, rhf, cfg, freqs) =
            (&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &freqs);
        run_pool(n, move || {
            pdep_polarizability_becke_dynamic(mol, obs, obs_bs, dfbs, rhf, op, cfg, freqs).unwrap()
        })
    };

    let serial = run(1);
    let parallel = run(4);

    assert_eq!(serial.len(), parallel.len(), "natoms mismatch");
    let mut mismatches = 0usize;
    for a in 0..serial.len() {
        assert_eq!(serial[a].len(), parallel[a].len(), "nfreq mismatch");
        for k in 0..serial[a].len() {
            for i in 0..3 {
                for j in 0..3 {
                    let sv = serial[a][k][i][j];
                    let pv = parallel[a][k][i][j];
                    if sv.to_bits() != pv.to_bits() {
                        mismatches += 1;
                        eprintln!(
                            "mismatch atom={a} freq_idx={k} [{i}][{j}]: serial={sv:.17e} \
                             (bits {:x}) parallel={pv:.17e} (bits {:x})",
                            sv.to_bits(),
                            pv.to_bits()
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "becke_dynamic_polarizability: {mismatches} non-bit-identical tensor entries \
         between 1-thread and 4-thread rayon pools"
    );
}

/// `esp_at_atoms` (properties.rs) must give bit-identical V(R_A) per atom
/// across thread counts — each atom probe is an independent engine.compute
/// contraction with the shared density; nothing should leak across workers.
#[test]
fn esp_at_atoms_bit_identical_across_thread_counts() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let density = rhf.density_r().clone();

    let run = |n: usize| {
        let (mol, obs, density) = (&mol, &obs, &density);
        run_pool(n, move || esp_at_atoms(mol, obs, density).unwrap())
    };

    let serial = run(1);
    let parallel = run(4);

    assert_eq!(serial.len(), parallel.len());
    let mut mismatches = 0usize;
    for a in 0..serial.len() {
        if serial[a].to_bits() != parallel[a].to_bits() {
            mismatches += 1;
            eprintln!(
                "esp mismatch atom={a}: serial={:.17e} (bits {:x}) parallel={:.17e} (bits {:x})",
                serial[a],
                serial[a].to_bits(),
                parallel[a],
                parallel[a].to_bits()
            );
        }
    }
    assert_eq!(
        mismatches, 0,
        "esp_at_atoms: {mismatches} non-bit-identical V(R_A) values between \
         1-thread and 4-thread rayon pools"
    );
}
