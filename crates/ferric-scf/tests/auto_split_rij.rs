//! `jk_storage = "auto"` builds J by Split-RI-J once the stored RI-J tensor
//! would not be held in memory, and keeps the stored tensor while it fits.
//!
//! Which builder an SCF took is read from `split_rij::constructed_count()`, not
//! from energy bits: an explicit `jk_storage = "split"` also applies to the
//! free-atom guess SCFs, whose small tensors `auto` keeps in memory, so the two
//! runs start from guesses that differ in the last bits. Both RI-J paths are
//! covered: J-only (PBE: no DF-K), and the J/K raw tensor shared with DF-K
//! (B3LYP, one aux basis), spilled or recomputed and freed after DF-K is
//! dressed.
//!
//! Own binary: these set the process-global `jk_storage` override and read a
//! process-wide counter.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{set_jk_storage, JkStorage};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::split_rij::constructed_count;
use std::sync::Mutex;

static SLOT: Mutex<()> = Mutex::new(());

/// Benzene / def2-SVP with def2-universal-jkfit: the stored J tensor is tens of
/// MB, so a 1 MB budget misses every in-memory tier; the free atoms' tensors
/// are far smaller.
const TIGHT: usize = 1 << 20;
const AMPLE: usize = 1 << 34;

/// The SCF energy and how many Split-RI-J builders it constructed.
fn run(xc: &str, storage: JkStorage, budget: usize) -> (f64, usize) {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        three_index_budget_bytes: budget,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    set_jk_storage(Some(storage));
    let before = constructed_count();
    let r = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg);
    let built = constructed_count() - before;
    set_jk_storage(None);
    let r = r.unwrap();
    assert!(
        r.converged,
        "{xc} {storage} budget {budget}: SCF did not converge"
    );
    (r.energy, built)
}

fn check(xc: &str) {
    let _g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let (e_tight, n_tight) = run(xc, JkStorage::Auto, TIGHT);
    let (e_ample, n_ample) = run(xc, JkStorage::Auto, AMPLE);
    let (e_split, n_split) = run(xc, JkStorage::Split, TIGHT);
    let (e_memory, n_memory) = run(xc, JkStorage::Memory, AMPLE);
    eprintln!(
        "{xc}: auto(tight) {e_tight:.12} [{n_tight} split builders], auto(ample) {e_ample:.12} \
         [{n_ample}], split {e_split:.12} [{n_split}], memory {e_memory:.12} [{n_memory}]"
    );
    assert_eq!(
        n_tight, 1,
        "{xc}: auto under a tight budget must build J by Split-RI-J once"
    );
    assert_eq!(
        n_ample, 0,
        "{xc}: auto with room must keep the stored tensor"
    );
    assert_eq!(n_memory, 0, "{xc}: memory never builds Split-RI-J");
    assert!(
        n_split > n_tight,
        "{xc}: explicit split must also reach the guess SCFs ({n_split} builders)"
    );
    // With room, auto is exactly the stored tensor.
    assert_eq!(
        e_ample.to_bits(),
        e_memory.to_bits(),
        "{xc}: auto(ample) != memory"
    );
    // Same J to integral precision: Split-RI-J agrees with the stored tensor.
    for (what, e) in [("auto(tight)", e_tight), ("split", e_split)] {
        assert!(
            (e - e_memory).abs() < 1e-8,
            "{xc}: {what} - memory = {:.3e}",
            e - e_memory
        );
    }
}

/// J-only path: a pure functional builds no DF-K.
#[test]
fn auto_takes_split_rij_for_j_only_when_the_tensor_misses_memory() {
    check("PBE");
}

/// Shared J/K raw tensor: the raw tensor is spilled or recomputed, so J is
/// Split-RI-J and the raw tensor is only used to dress DF-K.
#[test]
fn auto_takes_split_rij_for_the_shared_raw_when_it_is_not_resident() {
    check("B3LYP");
}
