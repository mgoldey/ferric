//! The libint2-native terf engine must evaluate with the table set IT was
//! built with, whatever table directory is loaded afterwards.
//!
//! libint2's terf hook carries no engine context. It used to read the
//! process-global table cache, which `get_terfc_tables` REASSIGNS when a
//! different directory is loaded. That had two consequences: the hook copied a
//! `shared_ptr` that another thread could be replacing (the unlocked
//! copy-vs-replace use-after-free of `core_eval_singleton_race.rs`), and an
//! engine silently evaluated with the LAST-loaded tables instead of its own.
//! Now each `scf_compute_*` entry point publishes its engine's table set to the
//! hook through a thread-local.
//!
//! Test: engine A on the shipped tables, engine B on a copy with every table
//! value doubled. B must differ from A, which shows the copy is really used.
//! A's block computed after B was created must be bit-identical to A's block
//! before. Before the fix, A's second block was B's block.
//!
//! Only builds under the `libint2_terf` feature against a terf-patched libint2
//! with FERRIC_LIBINT2_TERF set, like the other terf_libint2_* tests.
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{Engine, TerfLibint2Braket};

const TABLES: [&str; 4] = ["16_4_2.bin", "8_10_5.bin", "4_20_20.bin", "2_20_80.bin"];

fn shipped_dir() -> String {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables").into())
}

/// A copy of the shipped tables with every value doubled (the 16-byte header,
/// four i32 dimensions, untouched).
fn doubled_tables(src: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ferric-terf-doubled-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in TABLES {
        let mut bytes = std::fs::read(format!("{src}/{name}")).unwrap();
        assert_eq!((bytes.len() - 16) % 8, 0, "{name}: payload is not f64");
        for at in (16..bytes.len()).step_by(8) {
            let v = f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) * 2.0;
            bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join(name), bytes).unwrap();
    }
    dir
}

fn block(eng: &mut Engine, obs: &PreparedBasis) -> Vec<f64> {
    let nsh = obs.shell_dims().len();
    let mut out = Vec::new();
    for a in 0..nsh {
        for b in 0..=a {
            if let Some(x) = eng.compute_quartet(obs, a, b, 0, 0) {
                out.extend_from_slice(x);
            }
        }
    }
    out
}

#[test]
fn terf_engine_uses_its_own_tables_after_another_directory_loads() {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let (r0, omega) = (2.0_f64, 0.35_f64);
    let bk = TerfLibint2Braket::XxXx;
    let np = obs.max_nprim() as usize;
    let shipped = shipped_dir();
    let doubled = doubled_tables(&shipped);

    let mut a = Engine::new_terf_libint2(r0, omega, bk, np, 1, 0.0, Some(&shipped)).unwrap();
    let a_before = block(&mut a, &obs);
    let mut b =
        Engine::new_terf_libint2(r0, omega, bk, np, 1, 0.0, Some(doubled.to_str().unwrap()))
            .unwrap();
    let b_vals = block(&mut b, &obs);
    let a_after = block(&mut a, &obs);
    std::fs::remove_dir_all(&doubled).ok();

    assert!(a_before.len() > 10, "only {} integrals", a_before.len());
    assert!(
        a_before.iter().zip(&b_vals).any(|(x, y)| x != y),
        "doubled tables gave identical integrals: the test cannot see which tables were used"
    );
    let same = a_before
        .iter()
        .zip(&a_after)
        .all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        same && a_before.len() == a_after.len(),
        "engine A changed after engine B loaded other tables: it is not using its own table set"
    );
}
