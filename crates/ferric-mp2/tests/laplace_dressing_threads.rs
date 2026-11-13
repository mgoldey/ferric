//! The four Laplace-MP2 metric dressings (`compute_mo`, `compute_ao`,
//! `compute_sos_mo`, `compute_sos_ao`) run on the rayon pool through
//! `ferric_core::linalg::par_gemm`, and each path's energy is bit-identical
//! across pool sizes.
//!
//! Own binary: it reads the process-global `par_gemm` probe.

use ferric_core::basis;
use ferric_core::linalg::{par_gemm_probe, PAR_GEMM_COL_BLOCK};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::LaplaceMp2;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;

const N_QUAD: usize = 5;

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    rhf: ScfResult,
}

/// One dressing site: the path's name, its energy, and the `par_gemm` block
/// count it must produce (`ceil(columns / PAR_GEMM_COL_BLOCK)` for its one
/// dressing GEMM; no other `par_gemm` call is reached on these paths).
struct Slot {
    name: &'static str,
    run: fn(&Fixture) -> f64,
    dressed_columns: fn(&Fixture) -> usize,
}

fn nov(f: &Fixture) -> usize {
    let nocc = f.mol.nelec() as usize / 2;
    nocc * (f.obs.nbasis() - nocc)
}

fn nbas2(f: &Fixture) -> usize {
    f.obs.nbasis() * f.obs.nbasis()
}

const SLOTS: [Slot; 4] = [
    Slot {
        name: "compute_mo",
        run: |f| {
            LaplaceMp2::new(N_QUAD)
                .compute_mo(&f.mol, &f.obs, &f.dfbs, f.op, &f.rhf, 0)
                .unwrap()
        },
        dressed_columns: nov,
    },
    Slot {
        name: "compute_ao",
        run: |f| {
            LaplaceMp2::new(N_QUAD)
                .compute_ao(&f.mol, &f.obs, &f.dfbs, f.op, &f.rhf, 0, None)
                .unwrap()
                .0
        },
        dressed_columns: nbas2,
    },
    Slot {
        name: "compute_sos_mo",
        run: |f| {
            LaplaceMp2::new(N_QUAD)
                .compute_sos_mo(&f.mol, &f.obs, &f.dfbs, f.op, &f.rhf, 0)
                .unwrap()
        },
        dressed_columns: nov,
    },
    Slot {
        name: "compute_sos_ao",
        run: |f| {
            LaplaceMp2::new(N_QUAD)
                .compute_sos_ao(&f.mol, &f.obs, &f.dfbs, f.op, &f.rhf, 0, None)
                .unwrap()
        },
        dressed_columns: nbas2,
    },
];

/// Water/cc-pVDZ: nocc 5, nbas 24, naux 84, so the MO dressings are 95 columns
/// wide (1 block) and the AO dressings 576 (5 blocks). For each of the four
/// paths, pools of 1, 2 and 6 threads must record exactly that block count and
/// give the same energy bits.
///
/// FAILS, naming the path, if any one dressing goes back to a direct `.dot`
/// (zero blocks recorded for that path) or if a path's energy starts to depend
/// on the thread count.
#[test]
fn laplace_dressings_use_the_pool_and_are_thread_count_independent() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    let f = Fixture {
        mol,
        obs,
        dfbs,
        op,
        rhf,
    };

    for slot in &SLOTS {
        let expected_blocks = (slot.dressed_columns)(&f).div_ceil(PAR_GEMM_COL_BLOCK);
        let mut reference: Option<f64> = None;
        for threads in [1, 2, 6] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            par_gemm_probe::reset();
            let e = pool.install(|| (slot.run)(&f));
            let blocks = par_gemm_probe::BLOCKS.load(Ordering::Relaxed);
            assert_eq!(
                blocks, expected_blocks,
                "{}: {threads} threads ran {blocks} par_gemm blocks, expected \
                 {expected_blocks}; the metric dressing is not going through par_gemm",
                slot.name
            );
            assert!(
                e.is_finite() && e < 0.0,
                "{}: energy {e} is not a finite negative correlation energy",
                slot.name
            );
            match reference {
                None => reference = Some(e),
                Some(r) => assert_eq!(
                    e.to_bits(),
                    r.to_bits(),
                    "{}: {threads} threads gave {e:.17e}, 1 thread {r:.17e}",
                    slot.name
                ),
            }
        }
    }
}
