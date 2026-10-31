//! Prints one line per run, to compare two trees. Asserts nothing: see
//! `scripts/compare_ks_bits.sh`, which runs this on both and counts the lines.

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

const AUX: &str = "def2-universal-jkfit";

/// `(solver, molecule, functional, P/T, jk_storage)`. `P/T` 0 is an ample budget.
/// Butane uses a small grid, so sub-`T` budgets leave room for raw's soft tiers.
const CASES: &[(&str, &str, &str, f64, &str)] = &[
    // Ample.
    ("rhf", "water", "PBE", 0.0, "auto"),
    ("rhf", "water", "B3LYP", 0.0, "auto"),
    ("uhf", "water", "PBE", 0.0, "auto"),
    ("uhf", "water", "B3LYP", 0.0, "auto"),
    ("rohf", "water", "PBE", 0.0, "auto"),
    ("rhf", "butane", "PBE", 0.0, "auto"),
    ("rhf", "butane", "B3LYP", 0.0, "auto"),
    // Tight.
    ("rhf", "water", "PBE", 5.0, "auto"),
    ("rhf", "water", "PBE", 15.0, "auto"),
    ("rhf", "water", "PBE", 30.0, "auto"),
    ("rhf", "water", "B3LYP", 5.0, "auto"),
    ("rhf", "water", "B3LYP", 30.0, "auto"),
    ("uhf", "water", "PBE", 30.0, "auto"),
    ("rohf", "water", "PBE", 30.0, "auto"),
    ("rhf", "butane", "PBE", 0.95, "auto"),
    ("rhf", "butane", "PBE", 0.70, "auto"),
    ("rhf", "butane", "PBE", 1.5, "auto"),
    // Below the tensor, each raw tier by force, hybrid included.
    ("rhf", "butane", "PBE", 0.70, "memory"),
    ("rhf", "butane", "PBE", 0.70, "disk"),
    ("rhf", "butane", "PBE", 0.70, "direct"),
    ("rhf", "butane", "PBE", 0.95, "direct"),
    ("rhf", "butane", "B3LYP", 0.70, "auto"),
    ("rhf", "butane", "B3LYP", 0.70, "direct"),
    // Range-separated, ample and below the tensor.
    ("rhf", "water", "HSE06", 0.0, "auto"),
    ("rhf", "butane", "HSE06", 0.0, "auto"),
    ("rhf", "butane", "HSE06", 0.70, "auto"),
    ("rhf", "butane", "HSE06", 0.95, "auto"),
    // VV10, with its cache in the ledger.
    ("rhf", "butane", "wB97X-V", 0.0, "auto"),
    ("rhf", "butane", "wB97X-V", 0.70, "auto"),
    ("rhf", "butane", "wB97X-V", 0.95, "auto"),
];

#[test]
#[ignore = "prints a table for scripts/compare_ks_bits.sh"]
fn print_ks_bits_table() {
    for &(solver, molecule, xc, ratio, storage) in CASES {
        let (xyz, basis) = match molecule {
            "water" => ("../../testdata/molecules/water.xyz", "cc-pvdz"),
            _ => ("../../testdata/molecules/alkane_4.xyz", "def2-svp"),
        };
        let small = molecule == "butane";
        let mol = Molecule::load_xyz(xyz).unwrap();
        let prep = PreparedBasis::new(&mol, &ferric_core::basis::bundled(basis).unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &ferric_core::basis::bundled(AUX).unwrap()).unwrap();
        let tensor = aux.nbasis() * prep.nbasis() * prep.nbasis() * 8;
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        std::env::set_var("FERRIC_JK_STORAGE", storage);
        for t in [1usize, 4, 12] {
            let p = if ratio == 0.0 {
                8_000_000_000
            } else {
                (ratio * tensor as f64) as usize
            };
            clear_global();
            install_global(MemoryPool::with_capacity_bytes(p));
            let c = RhfConfig {
                xc: Some(xc.into()),
                df_j_aux: Some(AUX.into()),
                df_k_aux: (xc != "PBE").then(|| AUX.into()),
                three_index_budget_bytes: p,
                dft_grid: small.then_some(AtomicGridConfig {
                    n_radial: 30,
                    n_angular: 50,
                    prune: None,
                }),
                nlc_grid: small.then_some(AtomicGridConfig {
                    n_radial: 10,
                    n_angular: 14,
                    prune: None,
                }),
                ..Default::default()
            };
            let ctx = ParallelContext::default();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            let out = pool.install(|| match solver {
                "rhf" => solve_rhf(&ctx, &mol, &prep, op, &bounds, &c),
                "uhf" => solve_uhf(&ctx, &mol, &prep, &bounds, &c),
                _ => solve_rohf(&ctx, &mol, &prep, op, &bounds, &c),
            });
            let tag = format!("{solver} {molecule} {xc} r={ratio} {storage} t={t}");
            match out {
                Ok(o) => println!(
                    "BITS {tag} {:016x} {:.15} it={}",
                    o.energy.to_bits(),
                    o.energy,
                    o.iterations
                ),
                Err(e) => println!("BITS {tag} ERR {e}"),
            }
            clear_global();
        }
    }
    std::env::remove_var("FERRIC_JK_STORAGE");
}
