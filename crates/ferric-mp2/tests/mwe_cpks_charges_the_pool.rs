//! The dense nmo⁴ MO-ERI paths in `cpks_polar` must charge the shared pool.
//!
//! # The defect
//!
//! `check_full_mo_eri_alloc` compared ~3 nmo⁴ f64 against the whole budget
//! ceiling and charged nothing, so the nmo⁴ tensors held by
//! `dynamic_cphf_alpha_iw`, `bse_gate0_residuals` and `analytic_alpha_full`
//! were invisible to every other plane's gate. It now also takes a pool
//! reservation of the caller's TRUE peak (2 nmo⁴ + b_full for the TDHF/BSE
//! pair, 5 nmo⁴ + b_full for the analytic axis loop), held to function end.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the charge is REAL, a pool whose free room is just under that peak must
//! refuse, naming the CPHF plane. If the charge is DECORATION (inert, or a
//! token amount), the same pool admits, because the only other plane on the
//! path (the AO 3-index tensor, charged by its own source) fits in the room
//! left. The predictions differ, so the test can tell them apart.
//! `the_same_pool_admits_once_the_incumbent_is_gone` is the reachability guard
//! that the refusal is not just a pool too small for anything.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, reserve_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::cpks_polar::{analytic_alpha_full, dynamic_cphf_alpha_iw};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const AMPLE: usize = 64_000_000_000;
const OMEGA: f64 = 0.5;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Clear the process-global pool on entry AND exit, so one test's pool can
/// never refuse a sibling (see `mwe_mp2_planes_compose_in_one_pool.rs`).
struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ferric_scf::ScfResult,
}

/// water/STO-3G + cc-pVDZ-RI (nmo = 7).
fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("sto-3g").expect("sto-3g");
    let dfbs_set = ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("rhf");
    Fixture {
        mol,
        obs,
        dfbs,
        bounds,
        rhf,
    }
}

impl Fixture {
    fn nmo(&self) -> usize {
        self.obs.nbasis()
    }
    fn nmo4_bytes(&self) -> usize {
        self.nmo().pow(4) * 8
    }
    fn b_full_bytes(&self) -> usize {
        self.dfbs.nbasis() * self.nmo() * self.nmo() * 8
    }
    /// MEASURED pool high-water of `compute_b_full_mo` alone (the AO 3-index
    /// tensor, charged by its own source). Every CPHF path calls it while its
    /// own reservation is outstanding, so a held charge stacks on top of this.
    fn ao_path_peak(&self) -> usize {
        let pool = MemoryPool::with_capacity_bytes(AMPLE);
        install_global(pool.clone());
        let b = ferric_mp2::oo_rimp2::compute_b_full_mo(
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            self.rhf.mos_r(),
        );
        clear_global();
        b.expect("b_full");
        pool.peak_bytes()
    }
    /// The TDHF path's true peak: Gram + tensor + b_full.
    fn dynamic_peak(&self) -> usize {
        2 * self.nmo4_bytes() + self.b_full_bytes()
    }
    fn run_dynamic(&self) -> Result<[[f64; 3]; 3], ferric_core::FerricError> {
        dynamic_cphf_alpha_iw(
            &ParallelContext::default(),
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
            OMEGA,
        )
    }
    fn run_analytic(&self) -> Result<[[f64; 3]; 3], ferric_core::FerricError> {
        analytic_alpha_full(
            &ParallelContext::default(),
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.bounds,
            &self.rhf,
            &RiMp2Config::default(),
        )
        .map(|a| a.tensor)
    }
}

fn bits(t: &[[f64; 3]; 3]) -> [[u64; 3]; 3] {
    t.map(|row| row.map(f64::to_bits))
}

/// Inertness anchor: with no pool installed the reservation is inert, and an
/// ample pool changes no bit of α(iω).
#[test]
fn an_ample_pool_does_not_move_dynamic_alpha_one_bit() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let no_pool = f.run_dynamic().expect("no pool must run");
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let with_pool = f.run_dynamic().expect("an ample pool must admit");
    clear_global();
    assert_eq!(bits(&no_pool), bits(&with_pool));
    assert!(no_pool[0][0] > 0.0, "sanity: α_xx(iω) > 0, got {no_pool:?}");
}

/// Composes: free room just under the peak must refuse, naming the CPHF plane.
#[test]
fn a_pool_with_room_just_under_the_peak_refuses() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let room = f.dynamic_peak() - 1;
    assert!(
        room > f.ao_path_peak(),
        "the room must still fit the AO tensor, or an inert charge would refuse too"
    );
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let incumbent = reserve_global("incumbent", AMPLE - room).expect("incumbent");
    let out = f.run_dynamic();
    drop(incumbent);
    clear_global();
    let err = match out {
        Ok(t) => panic!("pool with {room} B free admitted the TDHF plane: {t:?}"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("dynamic CPHF/TDHF"), "{err}");
}

/// Reachability guard for the refusal above.
#[test]
fn the_same_pool_admits_once_the_incumbent_is_gone() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let incumbent = reserve_global("incumbent", AMPLE - (f.dynamic_peak() - 1)).unwrap();
    drop(incumbent);
    let out = f.run_dynamic();
    clear_global();
    out.expect("with the incumbent released, the pool must admit");
}

/// Released, charged at the true size (not a token), and HELD across the AO
/// build: the high-water is the CPHF charge PLUS the AO path's own peak. A
/// charge released early (before `compute_b_full_mo`) peaks at the max of the
/// two instead.
#[test]
fn the_charge_is_the_true_peak_and_is_released_on_return() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    f.run_dynamic().expect("ample pool");
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    assert_eq!(outstanding, 0, "a reservation outlived the call");
    let want = f.dynamic_peak() + f.ao_path_peak();
    assert!(
        peak >= want,
        "pool peak {peak} < {want} = TDHF peak (2·nmo⁴ + b_full) + AO path peak"
    );
}

/// The analytic axis loop holds five nmo⁴ buffers at once; the charge must
/// cover them (the old ceiling figure, 3 nmo⁴, undercounts).
#[test]
fn the_analytic_path_charges_five_nmo4_buffers() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let no_pool = f.run_analytic().expect("no pool must run");
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let with_pool = f.run_analytic().expect("ample pool");
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    assert_eq!(bits(&no_pool), bits(&with_pool));
    assert_eq!(outstanding, 0, "a reservation outlived the call");
    let want = 5 * f.nmo4_bytes() + f.b_full_bytes() + f.ao_path_peak();
    assert!(
        peak >= want,
        "pool peak {peak} < {want} = analytic peak + AO path peak"
    );
}
