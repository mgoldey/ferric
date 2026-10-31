//! A MINAO block lost to memory must not pass silently.
//!
//! With VV10 the NLC cache is charged at construction, so the free-atom solves
//! behind MINAO can be refused. That element's block of the guess is then left zero:
//! more iterations, and different energy bits in UKS and ROKS. The numbers here are
//! the band that does it: the cache is about 1.06 `T`, and 1.15 `T` leaves too little
//! for the oxygen atom's tensor.
//!
//! If reporting works, the loss appears on stderr and in the run log, and an ample
//! pool reports nothing. If the pool stopped refusing, the band would be empty and
//! the first assertion would fail rather than pass vacuously.
//!
//! Own binary: the pool and run log are process-global.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use std::process::Command;

const AUX: &str = "def2-universal-jkfit";
const CHILD: &str = "FERRIC_GUESS_FALLBACK_CHILD";
const WATER: &str = "../../testdata/molecules/water.xyz";

fn run(solver: &str, ratio: Option<f64>) {
    let (charge, mult) = if solver == "RHF" { (0, 1) } else { (1, 2) };
    let mol = Molecule::load_xyz_with_charge(WATER, charge, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let tensor = aux.nbasis() * prep.nbasis() * prep.nbasis() * 8;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let p = ratio.map_or(8_000_000_000, |r| (r * tensor as f64) as usize);
    install_global(MemoryPool::with_capacity_bytes(p));
    let grid = |n_radial, n_angular| {
        Some(AtomicGridConfig {
            n_radial,
            n_angular,
            prune: None,
        })
    };
    let c = RhfConfig {
        xc: Some("wB97X-V".into()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        three_index_budget_bytes: p,
        max_iter: 1,
        dft_grid: grid(30, 50),
        nlc_grid: grid(10, 14),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    // One iteration is enough: the guess comes first, and not converging is not
    // what this is about.
    let _ = match solver {
        "RHF" => solve_rhf(&ctx, &mol, &prep, op, &bounds, &c),
        "UHF" => solve_uhf(&ctx, &mol, &prep, &bounds, &c),
        _ => solve_rohf(&ctx, &mol, &prep, op, &bounds, &c),
    };
    clear_global();
}

/// The body each child process runs. A no-op in a normal test run.
#[test]
fn child() {
    let Ok(spec) = std::env::var(CHILD) else {
        return;
    };
    let (solver, rest) = spec.split_once(':').unwrap();
    let (ratio, log) = rest.split_once(':').unwrap();
    ferric_scf::runlog::init(std::path::Path::new(log));
    run(solver, ratio.parse().ok());
}

/// `(stderr, run log)` of `solver` in its own process.
fn in_child(solver: &str, ratio: &str) -> (String, String) {
    let log = std::env::temp_dir().join(format!(
        "ferric_guess_fallback_{}_{solver}_{ratio}.jsonl",
        std::process::id()
    ));
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child", "--nocapture"])
        .env(CHILD, format!("{solver}:{ratio}:{}", log.display()))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{stderr}");
    let records = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    (stderr, records)
}

#[test]
fn a_block_lost_to_memory_is_reported_by_every_solver() {
    for solver in ["RHF", "UHF", "ROHF"] {
        let (stderr, records) = in_child(solver, "1.15");
        assert!(
            stderr.contains("MINAO: free-atom SCF failed for O (Z=8)"),
            "{solver}:\n{stderr}"
        );
        assert!(stderr.contains("left zero"), "{solver}:\n{stderr}");
        assert!(
            stderr.contains("partial MINAO guess, not from hcore"),
            "{solver}:\n{stderr}"
        );
        assert!(stderr.contains("DF 3-index"), "names the plane:\n{stderr}");
        let noted = records
            .lines()
            .find(|l| l.contains("guess_block_lost"))
            .unwrap_or_else(|| panic!("{solver}: no record:\n{records}"));
        assert!(noted.contains("DF 3-index"), "{noted}");
    }
}

#[test]
fn an_ample_pool_reports_no_fallback() {
    for solver in ["RHF", "UHF", "ROHF"] {
        let (stderr, records) = in_child(solver, "ample");
        assert!(!stderr.contains("left zero"), "{solver}:\n{stderr}");
        assert!(
            !records.contains("guess_block_lost"),
            "{solver}:\n{records}"
        );
    }
}
