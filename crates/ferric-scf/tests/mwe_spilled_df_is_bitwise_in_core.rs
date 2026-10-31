//! At one BLAS thread a spilled DF J/K path must be bitwise the in-core one, so
//! the in-core-or-spill choice cannot reach a result, whatever the pool or the
//! thread count made it choose.
//!
//! These live in `ferric-scf` for a reason: `ferric-integrals`' own test binary
//! has no OpenBLAS (ndarray's `blas` feature is off there), so a bit test in that
//! crate exercises matrixmultiply and says nothing about production. The
//! `cblas_dgemm` reference below makes this file fail to LINK without OpenBLAS.
//!
//! Own binary: these install the process-global pool.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3};
use std::sync::{Mutex, MutexGuard};

extern "C" {
    fn cblas_dgemm();
}

static SLOT: Mutex<()> = Mutex::new(());

fn slot() -> MutexGuard<'static, ()> {
    let g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    clear_global();
    assert_ne!(cblas_dgemm as *const () as usize, 0);
    assert_eq!(
        std::env::var("OPENBLAS_NUM_THREADS").as_deref(),
        Ok("1"),
        "bit-identity holds at one BLAS thread; run with OPENBLAS_NUM_THREADS=1"
    );
    g
}

fn with_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(f)
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    aux: PreparedBasis,
}

/// k = 420 and nao = 114: wide enough to cross the DRESS_K_BLOCK grid and to
/// split GEMMs that OpenBLAS really does treat differently at an odd width.
fn benzene() -> Sys {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    Sys { mol, obs, aux }
}

fn metric(naux: usize) -> Array2<f64> {
    Array2::from_shape_fn((naux, naux), |(i, j)| {
        if i == j {
            1.5
        } else {
            0.01 / ((i as f64 - j as f64).abs() + 1.0)
        }
    })
}

fn read(s: &mut ThreeIndexSource) -> Array3<f64> {
    let (naux, nao) = (s.naux(), s.nao());
    let mut out = Array3::<f64>::zeros((naux, nao, nao));
    s.for_each_block(|blk| {
        let b = blk.data.shape()[0];
        out.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..])
            .assign(&blk.data);
        Ok(())
    })
    .unwrap();
    out
}

fn moved(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// The dressed tensor itself, across budgets that give different accumulator
/// pieces, read-back widths and raw streams.
#[test]
fn spilled_dressed_tensor_is_bitwise_the_in_core_one() {
    let _s = slot();
    let s = benzene();
    let op = Operator::coulomb();
    let (naux, nao) = (s.aux.nbasis(), s.obs.nbasis());
    let m = metric(naux);
    let needed = naux * nao * nao * 8;
    let ample = usize::MAX / 4;
    let mut raw = ThreeIndexSource::build(op, &s.obs, &s.aux, ample).unwrap();
    let reference = read(&mut ThreeIndexSource::build_dressed(&mut raw, &m, ample).unwrap());

    let mut widths = Vec::new();
    for frac in [0.45, 0.6, 0.8] {
        let budget = (needed as f64 * frac) as usize;
        let mut raw = ThreeIndexSource::build(op, &s.obs, &s.aux, budget).unwrap();
        let mut d = ThreeIndexSource::build_dressed(&mut raw, &m, budget).unwrap();
        assert!(d.is_spilled_for_test(), "frac {frac} must spill the output");
        widths.push(d.block_naux());
        let got = read(&mut d);
        let n = moved(reference.as_slice().unwrap(), got.as_slice().unwrap());
        assert_eq!(n, 0, "frac {frac}: {n} elements moved");
    }
    assert!(
        widths.iter().all(|&w| w < naux),
        "the read-back width must bind: {widths:?}"
    );
}

/// `jk_storage = "disk"` with an output that fits in core takes the blocked path.
/// It must match the fast path and the spill bitwise, or a disk-raw run could
/// depend on the live pool.
#[test]
fn blocked_in_core_dressing_is_bitwise_the_fast_path() {
    let _s = slot();
    let s = benzene();
    let op = Operator::coulomb();
    let (naux, nao) = (s.aux.nbasis(), s.obs.nbasis());
    let m = metric(naux);
    let ample = usize::MAX / 4;
    let mut raw = ThreeIndexSource::build(op, &s.obs, &s.aux, ample).unwrap();
    let reference = read(&mut ThreeIndexSource::build_dressed(&mut raw, &m, ample).unwrap());

    let budget = 3 * naux * nao * nao * 8;
    use ferric_integrals::three_index_source::{set_jk_storage, JkStorage};
    set_jk_storage(Some(JkStorage::Disk));
    let raw = ThreeIndexSource::build_for_jk(op, &s.mol, &s.obs, &s.aux, budget, (0, naux), true);
    set_jk_storage(None);
    let mut raw = raw.unwrap();
    assert!(raw.is_spilled_for_test(), "raw must be on disk");
    let mut d = ThreeIndexSource::build_dressed(&mut raw, &m, budget).unwrap();
    assert!(d.is_incore(), "the output must fit in core");
    let got = read(&mut d);
    let n = moved(reference.as_slice().unwrap(), got.as_slice().unwrap());
    assert_eq!(n, 0, "{n} elements moved");
}

/// An odd nao and a different aux: the fast-path grid must hold for any row width,
/// and packing an odd triangle must not disturb it. One kernel's evidence only.
#[test]
fn spilled_dressed_tensor_is_bitwise_in_core_at_an_odd_nao() {
    let _s = slot();
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_3.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let (naux, nao) = (aux.nbasis(), obs.nbasis());
    assert_eq!(nao % 2, 1, "fixture must have an odd nao");
    assert!(naux > 128, "fixture must cross the Q grid");
    let m = metric(naux);
    let ample = usize::MAX / 4;
    let mut raw = ThreeIndexSource::build(op, &obs, &aux, ample).unwrap();
    let reference = read(&mut ThreeIndexSource::build_dressed(&mut raw, &m, ample).unwrap());
    let budget = (naux * nao * nao * 8) as f64 * 0.45;
    let mut raw = ThreeIndexSource::build(op, &obs, &aux, budget as usize).unwrap();
    let mut d = ThreeIndexSource::build_dressed(&mut raw, &m, budget as usize).unwrap();
    assert!(d.is_spilled_for_test());
    let got = read(&mut d);
    let n = moved(reference.as_slice().unwrap(), got.as_slice().unwrap());
    assert_eq!(n, 0, "{n} elements moved at odd nao {nao}");
}

/// K through the real DfK, at a binding width and two thread counts.
#[test]
fn spilled_k_is_bitwise_in_core_and_thread_independent() {
    let _s = slot();
    let s = benzene();
    let op = Operator::coulomb();
    let (naux, nao) = (s.aux.nbasis(), s.obs.nbasis());
    let needed = naux * nao * nao * 8;
    let d = Array2::from_shape_fn((nao, nao), |(i, j)| {
        0.5 / ((i as f64 - j as f64).abs() + 1.0)
    });
    let k_at = |budget: usize, threads: usize| -> Array2<f64> {
        with_threads(threads, || {
            let mut dfk = DfK::new(op, &s.obs, &s.aux, budget).unwrap();
            let mut k = Array2::<f64>::zeros((nao, nao));
            dfk.build(&d, &mut k).unwrap();
            k
        })
    };
    let reference = k_at(usize::MAX / 4, 1);
    for threads in [1, 4] {
        let got = k_at((needed as f64 * 0.45) as usize, threads);
        let n = moved(reference.as_slice().unwrap(), got.as_slice().unwrap());
        assert_eq!(n, 0, "{threads} threads: {n} K elements moved");
    }
}

/// End to end, in the refused band: raw has the same backend either way, so only
/// the dressed tensor's route differs, and the SCF must be bitwise unchanged.
#[test]
fn scf_in_the_refused_band_is_bitwise_the_in_core_run() {
    let _s = slot();
    let s = benzene();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &s.obs).unwrap();
    let ctx = ParallelContext::default();
    let aux = "def2-universal-jkfit";
    let a = PreparedBasis::new(&s.mol, &basis::bundled(aux).unwrap()).unwrap();
    let needed = a.nbasis() * s.obs.nbasis() * s.obs.nbasis() * 8;
    let budget = needed + needed / 2;
    let cfg = RhfConfig {
        df_j_aux: Some(aux.into()),
        df_k_aux: Some(aux.into()),
        three_index_budget_bytes: budget,
        ..Default::default()
    };
    // No pool: the budget alone admits it, so this is the in-core route.
    let in_core = solve_rhf(&ctx, &s.mol, &s.obs, op, &bounds, &cfg).unwrap();
    let pool = MemoryPool::with_capacity_bytes(budget);
    install_global(pool.clone());
    let refused = solve_rhf(&ctx, &s.mol, &s.obs, op, &bounds, &cfg);
    clear_global();
    let refused = refused.unwrap();
    assert!(
        pool.peak_bytes() < 2 * needed,
        "the pool must really have refused the second copy"
    );
    assert_eq!(in_core.iterations, refused.iterations);
    assert_eq!(in_core.energy.to_bits(), refused.energy.to_bits());
}

/// Raw changing backend moves J by a few ulp, independently of this path: DF-J's
/// unpacked and packed contractions sum differently. Bounded, not bitwise.
#[test]
fn a_fully_spilled_scf_stays_within_the_df_j_backend_gap() {
    let _s = slot();
    let s = benzene();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &s.obs).unwrap();
    let ctx = ParallelContext::default();
    let aux = "def2-universal-jkfit";
    let a = PreparedBasis::new(&s.mol, &basis::bundled(aux).unwrap()).unwrap();
    let needed = a.nbasis() * s.obs.nbasis() * s.obs.nbasis() * 8;
    let run = |budget: usize| {
        let cfg = RhfConfig {
            df_j_aux: Some(aux.into()),
            df_k_aux: Some(aux.into()),
            three_index_budget_bytes: budget,
            ..Default::default()
        };
        solve_rhf(&ctx, &s.mol, &s.obs, op, &bounds, &cfg).unwrap()
    };
    let in_core = run(needed * 4);
    let spilled = run((needed as f64 * 0.4) as usize);
    assert!(in_core.converged && spilled.converged);
    assert_eq!(in_core.iterations, spilled.iterations);
    let de = (in_core.energy - spilled.energy).abs();
    eprintln!("in-core vs spilled |dE| = {de:.3e}");
    assert!(de < 1e-10, "|dE| = {de:.3e}");
}

/// REVIEW F1: a hybrid's XC evaluator charges the pool `threads × worker` before
/// DF is built, so the same pool can admit the dressed tensor at 1 thread and
/// refuse it at 4. The bits must not care.
#[test]
fn hybrid_ks_energy_ignores_threads_across_the_refused_band() {
    let _s = slot();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = "def2-universal-jkfit";
    let a = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let needed = a.nbasis() * obs.nbasis() * obs.nbasis() * 8;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let run = |cap: usize, threads: usize| {
        with_threads(threads, || {
            clear_global();
            let pool = MemoryPool::with_capacity_bytes(cap);
            install_global(pool.clone());
            let cfg = RhfConfig {
                xc: Some("B3LYP".into()),
                df_j_aux: Some(aux.into()),
                df_k_aux: Some(aux.into()),
                three_index_budget_bytes: cap,
                ..Default::default()
            };
            let r = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg);
            clear_global();
            (
                r.map(|r| r.energy.to_bits()).map_err(|e| e.to_string()),
                pool.peak_bytes(),
            )
        })
    };
    // The evaluator's working set is several MB and thread-scaled, so the scan
    // has to sit where it is charged and leaves about two tensors of room.
    // Measure each thread count's XC charge at a cap with room for everything,
    // then a peak of XC + 2 tensors means the dressed tensor went in core.
    let xc_charge = |threads: usize| run(needed * 22, threads).1 - 2 * needed;
    let (xc1, xc4) = (xc_charge(1), xc_charge(4));
    assert!(
        xc4 > xc1,
        "the XC charge must scale with threads: {xc1} vs {xc4}"
    );
    let mut routes_differed = 0;
    let mut both_ran = 0;
    for tenths in (120..=170).step_by(5) {
        let cap = needed * tenths / 10;
        let (one, peak1) = run(cap, 1);
        let (four, peak4) = run(cap, 4);
        // A thread-scaled charge can also refuse the RAW tensor outright, which
        // is an error, not a route: see the skip here.
        if let (Ok(e1), Ok(e4)) = (&one, &four) {
            both_ran += 1;
            let split = (peak1 == xc1 + 2 * needed) != (peak4 == xc4 + 2 * needed);
            routes_differed += usize::from(split);
            eprintln!("cap={tenths}/10 split={split}");
            assert_eq!(e1, e4, "energy moved with threads at cap {tenths}/10");
        }
    }
    eprintln!("both_ran={both_ran} routes_differed={routes_differed}");
    assert!(both_ran >= 3, "too few capacities ran to prove anything");
    assert!(
        routes_differed >= 1,
        "no capacity split the routes by thread count, so this proved nothing"
    );
}
