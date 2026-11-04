//! Danuglipron PBE/def2-SVP with RI-J (def2-universal-jkfit), flat 75x110 grid,
//! `budget_gb = 16`: the grid AO blocks stay resident beside the RI-J tensor.
//!
//! # The defect
//!
//! A binary without the RI-J pair screen held the raw tensor as the full
//! `naux·nbf²` square (3635·724²·8 = 15.24 GB): it fits the 16 GiB budget alone,
//! so `memory_tier` takes `InCore`. That left 1.9 GB of the 17.18 GB pool, the
//! 3.58 GB AO cache (plus its working set) was refused, and XC recomputed its AO
//! blocks every Fock build (peak RSS 15.9 GB). With the Schwarz pair screen at
//! the default `integral_thresh = 1e-12` the stored tensor is the kept pairs only
//! (~52% of the packed triangle), and the AO cache fits beside it.
//!
//! # What is pinned, without an SCF
//!
//! The ledger as `commit_storage` sees it: a 16 GiB pool with the RI-J plane
//! reserved at the size `DfJ::new_for_jk` stores, then the deferred evaluator's
//! plan. `plan_memory` makes the same `resident + working set <= available`
//! comparison `decide_storage` makes at commit, so `resident_if_alone > 0` is the
//! admission. Nothing is allocated but the grid and its batch lists.
//!
//! Negative control: the unpacked square reserved instead starves the cache, so
//! the plan is not vacuous.
//!
//! Own binary: it installs the process-global pool.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::xc_trait::XcStorage;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;

const XYZ: &str = "../../testdata/molecules/c9_systems/danuglipron/conf_00_cryo_em.xyz";
const AUX: &str = "def2-universal-jkfit";
/// `[memory] budget_gb = 16`, as the CLI installs it.
const BUDGET: usize = 16 << 30;
/// `RhfConfig::integral_thresh`'s default, which `solve_rhf` passes as the
/// RI-J pair threshold.
const PAIR_THRESH: f64 = 1e-12;

/// The plan of a deferred PBE evaluator on the flat 75x110 grid, beside a pool
/// that already holds `raw` bytes for the RI-J plane.
fn resident_beside(mol: &Molecule, raw: usize) -> usize {
    let bs = basis::bundled("def2-svp").unwrap();
    let main = AtomicGridConfig {
        n_radial: 75,
        n_angular: 110,
        prune: None,
        // Whatever else the grid carries (e.g. a weight cut) at its default.
        ..AtomicGridConfig::default()
    };
    let nlc = AtomicGridConfig::default_nlc();
    let pool = MemoryPool::with_capacity_bytes(BUDGET);
    install_global(pool.clone());
    let _raw = pool
        .reserve("DF 3-index (P|mn) in-core", raw)
        .expect("the RI-J plane fits the pool alone");
    let ks = ferric_dft::ks::KsXc::new_deferred(mol, &bs, "PBE", &main, &nlc, None, Some(BUDGET))
        .unwrap();
    let plan = ks.memory_plan();
    drop(ks);
    clear_global();
    eprintln!("raw {:.3} GB -> {plan:?}", raw as f64 / 1e9);
    plan.resident_if_alone
}

#[test]
#[cfg_attr(
    feature = "gpu",
    ignore = "PRECONDITION: [gpu] mode off (the input's); a device that may take J skips the pair screen"
)]
fn the_screened_ri_j_tensor_leaves_room_for_resident_ao_blocks() {
    let mol = Molecule::load_xyz(XYZ).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let (n, naux) = (obs.nbasis(), dfbs.nbasis());
    assert_eq!((n, naux), (724, 3635), "not the measured shape");

    let screen =
        ferric_scf::df_j::ri_j_pair_screen_for_test(Operator::coulomb(), &obs, &dfbs, PAIR_THRESH)
            .unwrap()
            .expect("the screen drops pairs on a 71-atom molecule");
    let screened = ThreeIndexSource::pair_screened_bytes(naux, &screen);
    let packed = naux * n * (n + 1) / 2 * 8;
    let unpacked = naux * n * n * 8;
    eprintln!(
        "RI-J stored {:.3} GB (packed {:.3}, unpacked {:.3})",
        screened as f64 / 1e9,
        packed as f64 / 1e9,
        unpacked as f64 / 1e9
    );
    assert!(screened < packed, "the screen must store fewer pairs");
    // The pair-screened tier is taken only when it fits the budget and the pool.
    assert!(screened <= BUDGET);

    let resident = resident_beside(&mol, screened);
    assert!(
        resident > 0,
        "AO blocks refused beside the {:.3} GB screened RI-J tensor",
        screened as f64 / 1e9
    );
    // Negative control: what the binary without the screen held.
    assert!(unpacked <= BUDGET, "unpacked must be the tier it took");
    assert_eq!(
        resident_beside(&mol, unpacked),
        0,
        "the unpacked square must starve the cache, or this test pins nothing"
    );
}
