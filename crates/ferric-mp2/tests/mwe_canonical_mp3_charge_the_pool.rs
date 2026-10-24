//! Canonical MP2 and spin-orbital MP3 charge ONE true peak to both the budget
//! ceiling and the shared pool.
//!
//! # The defect
//!
//! Both functions carried two different peaks. `canonical_mp2`'s ceiling
//! summed `ao + t1 + mo` (but `mo` never coexists with `ao`), while its pool
//! charge took a max that missed `t1` being alive while `t2` is zeroed.
//! `mp3_energy`'s ceiling was 2·(2nv)⁴ (the spatial `g_abcd` counted at
//! spin-orbital size), its pool charge was max(2·(2nv)⁴, 5·(2no)²(2nv)²), and
//! neither counted the RI tensors held for the whole call, the VVVV phase's
//! co-resident OOOO/OVVO/amplitude blocks, or the two interleaved-axis copies
//! `einsum!` makes in the particle-hole term (6 OOVV-sized blocks, not 5).
//! Following #537, each now computes one peak (`canonical_mp2_peak_bytes`,
//! `mp3_peak_bytes`) that feeds both `check_alloc` and the pool.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the charge is the stated peak, the pool high-water after a call is
//! EXACTLY the formula re-derived below (nothing else on these paths charges
//! the pool), room of `peak` admits and `peak - 1` refuses. A token, inert or
//! stale charge moves the high-water off the formula and moves the boundary.
//! The formulas here are written as the DOMINANT stage for each fixture, so a
//! `max` in the crate that drops that stage fails too. Two MP3 fixtures put
//! the peak in different stages (amplitudes for water/STO-3G, the VVVV build
//! for H2/cc-pVDZ).

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, reserve_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_mp2::mp3::mp3_energy;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const AMPLE: usize = 64_000_000_000;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Serialize, clear the process-global pool on entry AND exit, and pin the
/// budget env so the caller's environment cannot move a ceiling.
struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        pin_env_budget("64");
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
        pin_env_budget("64");
    }
}

fn pin_env_budget(gib: &str) {
    std::env::set_var("FERRIC_MEM_BUDGET_GB", gib);
    std::env::remove_var("FERRIC_OOC_BUDGET_GB");
    std::env::remove_var("FERRIC_ERI3_BUDGET_GB");
}

/// `FERRIC_MEM_BUDGET_GB` (GiB) that resolves to exactly `bytes`: a byte count
/// over 2^30 is an exact binary fraction, `{}` prints the round-trip repr, and
/// `gib_to_bytes` multiplies back by 2^30.
fn exact_gib(bytes: usize) -> String {
    format!("{}", bytes as f64 / 1_073_741_824.0)
}

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::ScfResult,
}

fn fixture(xyz_path: Option<&str>, obs_name: &str) -> Fixture {
    let mol = match xyz_path {
        Some(p) => Molecule::load_xyz(p).expect("xyz"),
        None => Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).expect("H2"),
    };
    let obs = PreparedBasis::new(&mol, &ferric_core::basis::bundled(obs_name).expect("obs"))
        .expect("prepared obs");
    let dfbs = PreparedBasis::new(
        &mol,
        &ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri"),
    )
    .expect("prepared dfbs");
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
        rhf,
    }
}

/// water/STO-3G (+ cc-pVDZ-RI for MP3): nbas 7, nocc 5, nvir 2.
fn water() -> Fixture {
    fixture(Some("../../testdata/molecules/water.xyz"), "sto-3g")
}

/// H2/cc-pVDZ (+ cc-pVDZ-RI): nbas 10, nocc 1, nvir 9.
fn h2() -> Fixture {
    fixture(None, "cc-pvdz")
}

impl Fixture {
    fn nbas(&self) -> usize {
        self.obs.nbasis()
    }
    fn nocc(&self) -> usize {
        self.mol.nelec() as usize / 2
    }
    fn nvir(&self) -> usize {
        self.nbas() - self.nocc()
    }
    fn naux(&self) -> usize {
        self.dfbs.nbasis()
    }

    /// Canonical MP2: c_occ + c_vir, plus the `ao + t1` stage (dominant).
    fn canonical_peak(&self) -> usize {
        let (n, o, v) = (self.nbas(), self.nocc(), self.nvir());
        (n * (o + v) + n.pow(4) + o * n.pow(3)) * 8
    }
    fn run_canonical(&self) -> Result<f64, FerricError> {
        canonical_mp2(&self.mol, &self.obs, Operator::coulomb(), &self.rhf, 0)
    }

    /// Everything MP3 holds for the whole 4-index phase: coefficients, the
    /// metric pair, the AO 3-index tensor and the four dressed RI blocks.
    fn mp3_held(&self) -> usize {
        let (a, n, o, v) = (self.naux(), self.nbas(), self.nocc(), self.nvir());
        n * (o + v) + 2 * a * a + a * (n * n + 2 * o * v + o * o + v * v)
    }
    /// MP3 with the amplitude stage dominant: V⁴ + O⁴ + 4·O²V² spin-orbital.
    fn mp3_amplitude_peak(&self) -> usize {
        let (o2, v2) = (2 * self.nocc(), 2 * self.nvir());
        (self.mp3_held() + v2.pow(4) + o2.pow(4) + 4 * o2 * o2 * v2 * v2) * 8
    }
    /// MP3 with the VVVV build dominant: OOVV + spatial g_abcd + V⁴.
    fn mp3_vvvv_peak(&self) -> usize {
        let (o, v) = (self.nocc(), self.nvir());
        let (o2, v2) = (2 * o, 2 * v);
        (self.mp3_held() + o2 * o2 * v2 * v2 + v.pow(4) + v2.pow(4)) * 8
    }
    fn run_mp3(&self, budget: Option<usize>) -> Result<f64, FerricError> {
        mp3_energy(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
            0,
            budget,
        )
        .map(|r| r.e_total)
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

/// Run `f` under an ample pool; return its result, outstanding bytes after the
/// call, and the pool high-water.
fn high_water<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let out = f();
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    (out, outstanding, peak)
}

// ---------------------------------------------------------------- canonical

#[test]
fn canonical_ample_pool_does_not_move_the_energy_one_bit() {
    let _s = CleanSlot::acquire();
    let f = water();
    let no_pool = f.run_canonical().expect("no pool must run");
    let (with_pool, _, _) = high_water(|| f.run_canonical());
    assert_eq!(
        no_pool.to_bits(),
        with_pool.expect("ample pool must admit").to_bits()
    );
    assert!(no_pool < 0.0, "sanity: E_corr < 0, got {no_pool}");
}

#[test]
fn canonical_high_water_is_exactly_the_true_peak() {
    let _s = CleanSlot::acquire();
    let f = water();
    let (out, outstanding, peak) = high_water(|| f.run_canonical());
    out.expect("ample pool");
    assert_eq!(outstanding, 0, "a reservation outlived the call");
    assert_eq!(
        peak,
        f.canonical_peak(),
        "pool high-water != c_occ + c_vir + nbas⁴ + nocc·nbas³"
    );
}

#[test]
fn canonical_pool_admits_at_its_peak_and_refuses_one_byte_under() {
    let _s = CleanSlot::acquire();
    let f = water();
    let peak = f.canonical_peak();
    with_room(peak, || f.run_canonical()).expect("room for exactly the peak must admit");
    let err = match with_room(peak - 1, || f.run_canonical()) {
        Ok(e) => panic!("{} B free admitted canonical MP2: {e}", peak - 1),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("canonical MP2"), "{err}");
}

#[test]
fn canonical_ceiling_admits_at_its_peak_and_refuses_one_byte_under() {
    let _s = CleanSlot::acquire();
    let f = water();
    let peak = f.canonical_peak();
    pin_env_budget(&exact_gib(peak));
    let at_peak = f.run_canonical();
    pin_env_budget(&exact_gib(peak - 1));
    let under = f.run_canonical();
    at_peak.expect("a ceiling of exactly the peak must admit");
    let err = match under {
        Ok(e) => panic!("ceiling {} B admitted canonical MP2: {e}", peak - 1),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("canonical MP2") && err.contains("budget is"),
        "{err}"
    );
}

// ---------------------------------------------------------------------- MP3

#[test]
fn mp3_ample_pool_does_not_move_the_energy_one_bit() {
    let _s = CleanSlot::acquire();
    let f = water();
    let no_pool = f.run_mp3(None).expect("no pool must run");
    let (with_pool, _, _) = high_water(|| f.run_mp3(None));
    assert_eq!(
        no_pool.to_bits(),
        with_pool.expect("ample pool must admit").to_bits()
    );
}

/// water/STO-3G: the amplitude stage (V⁴ + O⁴ + 4·O²V²) is the peak. The old
/// pool charge (5·O²V²) and ceiling (2·V⁴) both sat far below it.
#[test]
fn mp3_high_water_is_exactly_the_amplitude_stage_peak() {
    let _s = CleanSlot::acquire();
    let f = water();
    let (out, outstanding, peak) = high_water(|| f.run_mp3(None));
    out.expect("ample pool");
    assert_eq!(outstanding, 0, "a reservation outlived the call");
    assert_eq!(
        peak,
        f.mp3_amplitude_peak(),
        "pool high-water != held RI + V⁴ + O⁴ + 4·O²V²"
    );
}

/// H2/cc-pVDZ: nv ≫ no puts the peak in the VVVV build instead.
#[test]
fn mp3_high_water_is_exactly_the_vvvv_stage_peak() {
    let _s = CleanSlot::acquire();
    let f = h2();
    let (out, outstanding, peak) = high_water(|| f.run_mp3(None));
    out.expect("ample pool");
    assert_eq!(outstanding, 0, "a reservation outlived the call");
    assert_eq!(
        peak,
        f.mp3_vvvv_peak(),
        "pool high-water != held RI + O²V² + nv⁴ + V⁴"
    );
}

#[test]
fn mp3_pool_admits_at_its_peak_and_refuses_one_byte_under() {
    let _s = CleanSlot::acquire();
    let f = water();
    let peak = f.mp3_amplitude_peak();
    with_room(peak, || f.run_mp3(None)).expect("room for exactly the peak must admit");
    let err = match with_room(peak - 1, || f.run_mp3(None)) {
        Ok(e) => panic!("{} B free admitted MP3: {e}", peak - 1),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("MP3"), "{err}");
}

#[test]
fn mp3_ceiling_admits_at_its_peak_and_refuses_one_byte_under() {
    let _s = CleanSlot::acquire();
    let f = water();
    let peak = f.mp3_amplitude_peak();
    f.run_mp3(Some(peak))
        .expect("a ceiling of exactly the peak must admit");
    let err = match f.run_mp3(Some(peak - 1)) {
        Ok(e) => panic!("ceiling {} B admitted MP3: {e}", peak - 1),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("MP3") && err.contains("budget is"), "{err}");
}
