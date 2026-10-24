//! The dense nmo⁴ MO-ERI paths in `cpks_polar` must charge the shared pool.
//!
//! # The defect
//!
//! `check_full_mo_eri_alloc` compared ~3 nmo⁴ f64 against the whole budget
//! ceiling and charged nothing, so the nmo⁴ tensors held by
//! `dynamic_cphf_alpha_iw`, `bse_gate0_residuals` and `analytic_alpha_full`
//! were invisible to every other plane's gate. `guard_full_mo_eri` now checks
//! the ceiling against AND charges the pool with the caller's true nmo⁴-phase
//! peak (2 nmo⁴ + b_full for the TDHF/BSE pair, 5 nmo⁴ + b_full for the
//! analytic axis loop), held to function end and so, conservatively, across
//! the AO build too.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the charge is REAL, a pool whose free room is just under that peak must
//! refuse, naming the CPHF plane. If the charge is DECORATION (inert, or a
//! token amount), the same pool admits, because the only other plane on the
//! path (the AO 3-index tensor, charged by its own source) fits in the room
//! left. The predictions differ, so the test can tell them apart.
//! `the_dynamic_path_admits_at_exactly_its_peak` is the reachability guard
//! that the refusal is not just a pool too small for anything.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, reserve_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::cpks_polar::{analytic_alpha_full, bse_gate0_residuals, dynamic_cphf_alpha_iw};
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
    /// The analytic path's true peak: five nmo⁴ buffers + b_full.
    fn analytic_peak(&self) -> usize {
        5 * self.nmo4_bytes() + self.b_full_bytes()
    }
    fn run_analytic(&self) -> Result<[[f64; 3]; 3], ferric_core::FerricError> {
        self.run_analytic_with(RiMp2Config::default())
    }
    fn run_analytic_with(
        &self,
        cfg: RiMp2Config,
    ) -> Result<[[f64; 3]; 3], ferric_core::FerricError> {
        analytic_alpha_full(
            &ParallelContext::default(),
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.bounds,
            &self.rhf,
            &cfg,
        )
        .map(|a| a.tensor)
    }
    fn run_bse_gate0(&self) -> Result<(f64, f64), ferric_core::FerricError> {
        bse_gate0_residuals(
            &ParallelContext::default(),
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
        )
    }
}

/// Run `f` under a 64 GB pool whose incumbent leaves `room` bytes free.
fn with_room<T>(room: usize, f: impl FnOnce() -> T) -> T {
    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let incumbent = reserve_global("incumbent", AMPLE - room).expect("incumbent");
    let out = f();
    drop(incumbent);
    clear_global();
    out
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

/// Reachability boundary: room for exactly the CPHF charge plus the AO
/// path's peak (they are outstanding together) admits; one byte less refuses.
#[test]
fn the_dynamic_path_admits_at_exactly_its_peak() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let room = f.dynamic_peak() + f.ao_path_peak();
    with_room(room, || f.run_dynamic()).expect("room for the full peak must admit");
    let refused = with_room(room - 1, || f.run_dynamic());
    assert!(
        refused.is_err(),
        "{} B free admitted: {refused:?}",
        room - 1
    );
}

/// The BSE gate-0 path is guarded too: room just under its peak refuses.
#[test]
fn the_bse_gate0_path_refuses_just_under_its_peak() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let room = f.dynamic_peak() - 1;
    assert!(room > f.ao_path_peak());
    let err = match with_room(room, || f.run_bse_gate0()) {
        Ok(r) => panic!("{room} B free admitted the BSE gate-0 plane: {r:?}"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("BSE gate-0 residuals"), "{err}");
}

/// No pool: the analytic path's CEILING now uses the true 5 nmo⁴ + b_full,
/// so a budget between the old 3 nmo⁴ + b_full and 5 nmo⁴ + b_full refuses.
#[test]
fn the_analytic_ceiling_refuses_between_three_and_five_nmo4() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let budget = 4 * f.nmo4_bytes() + f.b_full_bytes();
    assert!(3 * f.nmo4_bytes() + f.b_full_bytes() < budget && budget < f.analytic_peak());
    let cfg = RiMp2Config {
        memory_budget_bytes: Some(budget),
        ..Default::default()
    };
    let err = match f.run_analytic_with(cfg) {
        Ok(t) => panic!("ceiling {budget} B admitted the analytic path: {t:?}"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("relaxed-MP2 α") && err.contains("budget is"),
        "{err}"
    );
    let ample = RiMp2Config {
        memory_budget_bytes: Some(f.analytic_peak()),
        ..Default::default()
    };
    f.run_analytic_with(ample)
        .expect("a ceiling of exactly the peak must admit");
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
    // Deterministic: the CPHF charge is outstanding across the AO build.
    assert_eq!(
        peak,
        f.dynamic_peak() + f.ao_path_peak(),
        "pool high-water != TDHF peak (2·nmo⁴ + b_full) + AO path peak"
    );
}

/// The analytic axis loop holds five nmo⁴ buffers at once; the charge is
/// exactly that (the old ceiling figure, 3 nmo⁴, undercounted).
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
    assert_eq!(
        peak,
        f.analytic_peak() + f.ao_path_peak(),
        "pool high-water != analytic peak (5·nmo⁴ + b_full) + AO path peak"
    );
}
