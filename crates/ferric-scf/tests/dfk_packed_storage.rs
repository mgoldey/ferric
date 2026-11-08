//! DF-K's dressed tensor held as the packed lower μν triangle (`[scf]
//! ri_k_precision`, `ferric_integrals::three_index_source::dressed_packed`).
//!
//! WHAT IS PINNED (u32 = 2⁻²⁴, u64 = 2⁻⁵³, γ_k = k·u64/(1 − k·u64)):
//!  * f64: the packed band holds exactly the lower triangle of the square band
//!    the pre-2026-11-08 default held, and K from it (occupied path and density
//!    path) is bit-for-bit the square tensor's; so is a whole RI-JK SCF
//!    (`ri_k_precision = "f64"` reproduces the old default bitwise).
//!  * f32 storage: each stored value is the nearest f32 of the f64 packed
//!    value, bit for bit.
//!  * f32 TWIN (sharp): K from the f32 band equals the test's own f64 sums over
//!    the widened f32 values to the f64 summation floor
//!    `2γ_{n + naux·nocc}·Σ_P Â_P Â_Pᵀ`, `Â_P = |B̂_P||C|`.
//!  * f32 CONTRACT against the f64 band: `M̂_P = B̂_P C` with
//!    `|B̂ − B| ≤ u32|B| + η` gives `|M̂_P − M_P| ≤ ε A_P`, `A_P = |B_P||C|`,
//!    `ε = u32 + γ_n(1 + u32)` (plus η terms, below 1e-40 here), so elementwise
//!    `|K̂ − K| ≤ ((2ε + ε²) + 2γ_{n+naux·nocc}(1 + ε)²)·Σ_P A_P A_Pᵀ`.
//!    Not vacuous: the worst element uses more than 1e-3 of it.
//!  * SCF sandwich (RHF, RI-JK, water / def2-SVP): with D* the f64 SCF density,
//!    D' the f32 one and `E'` the rounded-B functional, both runs at their
//!    minima give `L = E'_K(D') − E_K(D') ≤ E_f32 − E_f64 ≤ E'_K(D*) − E_K(D*) =
//!    U`, `E_K(D) = −¼ tr(D K(D))`, up to the two runs' termination (`2|dE|`
//!    each, as `dfj_f32_storage.rs`) and the floor of the four K evaluations.
//!  * Mutants (`test-seams`): f32 rounding toward zero breaks the storage
//!    check; a skewed unpack of the packed row breaks the bitwise f64 check
//!    and the f32 twin.
use ferric_core::basis;
use ferric_core::gpu::mixed_host::{gamma, U32};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::DressedRows;
use ferric_scf::df_k::{set_dfk_f32_for_test, set_dfk_square_for_test, DfK, RiKPrecision};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());
const AUX: &str = "def2-universal-jkfit";
const BUDGET: usize = 4 << 30;

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Restores the process-wide hooks and seams on drop.
struct Hooks;
impl Drop for Hooks {
    fn drop(&mut self) {
        set_dfk_f32_for_test(None);
        set_dfk_square_for_test(false);
        #[cfg(feature = "test-seams")]
        {
            use std::sync::atomic::Ordering;
            ferric_scf::df_k::TRUNCATE_DFK_F32.store(false, Ordering::SeqCst);
            ferric_scf::df_k::SKEW_DFK_PACKED_UNPACK.store(false, Ordering::SeqCst);
        }
    }
}

fn testdata(rel: &str) -> String {
    format!(
        "{}/../../testdata/molecules/{rel}",
        env!("CARGO_MANIFEST_DIR")
    )
}

struct System {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
}

impl System {
    fn new(xyz: &str, obs: &str) -> Self {
        let mol = Molecule::load_xyz(&testdata(xyz)).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        Self { mol, obs, dfbs }
    }
    fn dfk(&self, precision: RiKPrecision) -> DfK<'static> {
        DfK::new_with_precision(
            Operator::coulomb(),
            &self.obs,
            &self.dfbs,
            BUDGET,
            precision,
        )
        .unwrap()
    }
    fn dfk_square(&self) -> DfK<'static> {
        set_dfk_square_for_test(true);
        let k = self.dfk(RiKPrecision::F64);
        set_dfk_square_for_test(false);
        k
    }
}

/// A deterministic, non-orthogonal "occupied" factor with mixed signs.
fn factor(n: usize, nocc: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, nocc), |(i, j)| {
        let x = ((i * 7 + j * 13 + 3) % 17) as f64 / 17.0 - 0.45;
        x * (1.0 + 0.1 * j as f64)
    })
}

fn k_occ(k: &mut DfK<'_>, c: &Array2<f64>) -> Array2<f64> {
    let n = c.nrows();
    let mut out = Array2::zeros((n, n));
    k.build_from_occ(c, &mut out).unwrap();
    out
}

fn k_density(k: &mut DfK<'_>, d: &Array2<f64>) -> Array2<f64> {
    let n = d.nrows();
    let mut out = Array2::zeros((n, n));
    k.build(d, &mut out).unwrap();
    out
}

/// Row `r` of a packed band as the full symmetric f64 square.
fn square_of(rows: &DressedRows<'_>, r: usize, n: usize) -> Array2<f64> {
    let mut sq = vec![0.0; n * n];
    rows.unpack_square_into(r, &mut sq, n);
    Array2::from_shape_vec((n, n), sq).unwrap()
}

/// `Σ_P (|B_P||C|)(|B_P||C|)ᵀ` and `Σ_P (B_P C)(B_P C)ᵀ` from a packed band,
/// both by the test's own ndarray GEMMs.
fn reference_k(rows: &DressedRows<'_>, c: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = c.nrows();
    let ca = c.mapv(f64::abs);
    let mut s = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    for p in 0..rows.nrows() {
        let b = square_of(rows, p, n);
        let a = b.mapv(f64::abs).dot(&ca);
        s += &a.dot(&a.t());
        let m = b.dot(c);
        k += &m.dot(&m.t());
    }
    (k, s)
}

#[test]
fn packed_f64_band_and_k_are_bitwise_the_square_ones() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    let n = sys.obs.nbasis();
    let mut sq = sys.dfk_square();
    let mut pk = sys.dfk(RiKPrecision::F64);
    assert!(!sq.is_packed() && pk.is_packed(), "layouts not as forced");
    assert_eq!(pk.precision(), RiKPrecision::F64);
    let pair = n * (n + 1) / 2;
    assert_eq!(
        pk.resident_tensor_bytes() * n * n,
        sq.resident_tensor_bytes() * pair,
        "packed band is not n(n+1)/2 of the square"
    );
    let c = factor(n, 5);
    let (a, b) = (k_occ(&mut sq, &c), k_occ(&mut pk, &c));
    assert!(a == b, "occupied-path K differs from the square tier");
    let d = c.dot(&c.t());
    let (a, b) = (k_density(&mut sq, &d), k_density(&mut pk, &d));
    assert!(a == b, "density-path K differs from the square tier");
}

fn rhf(sys: &System, precision: Option<RiKPrecision>) -> ferric_scf::ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &sys.obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-3,
        density_conv: 1e-8,
        max_iter: 100,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_k_precision: precision,
        ..Default::default()
    };
    solve_rhf(&ctx, &sys.mol, &sys.obs, op, &bounds, &cfg).unwrap()
}

#[test]
fn explicit_f64_scf_is_bitwise_the_square_default() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    set_dfk_square_for_test(true);
    let old = rhf(&sys, Some(RiKPrecision::F64));
    set_dfk_square_for_test(false);
    let new = rhf(&sys, Some(RiKPrecision::F64));
    assert_eq!(
        old.energy.to_bits(),
        new.energy.to_bits(),
        "{} vs {}",
        old.energy,
        new.energy
    );
    assert_eq!(old.iterations, new.iterations);
    assert_eq!(
        new.df_jk.as_ref().unwrap().ri_k_precision,
        RiKPrecision::F64,
        "route does not record the f64 tensor"
    );
}

#[test]
fn f32_stores_the_nearest_rounding_of_the_f64_band() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    let f64k = sys.dfk(RiKPrecision::F64);
    let f32k = sys.dfk(RiKPrecision::F32);
    assert_eq!(f32k.precision(), RiKPrecision::F32);
    assert_eq!(
        2 * f32k.resident_tensor_bytes(),
        f64k.resident_tensor_bytes()
    );
    let (DressedRows::F64(b), DressedRows::F32(bh)) = (
        f64k.dressed_rows_for_test().unwrap(),
        f32k.dressed_rows_for_test().unwrap(),
    ) else {
        panic!("tiers not as requested")
    };
    let bad = b
        .iter()
        .zip(bh.iter())
        .filter(|(x, y)| (**x as f32).to_bits() != y.to_bits())
        .count();
    assert_eq!(bad, 0, "{bad} stored values are not the nearest f32");
}

#[test]
fn f32_k_meets_its_twin_and_its_contract() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    let n = sys.obs.nbasis();
    let naux = sys.dfbs.nbasis();
    let nocc = 5;
    let c = factor(n, nocc);
    let mut f64k = sys.dfk(RiKPrecision::F64);
    let mut f32k = sys.dfk(RiKPrecision::F32);
    let k64 = k_occ(&mut f64k, &c);
    let k32 = k_occ(&mut f32k, &c);
    let floor = 2.0 * gamma(n + naux * nocc, ferric_core::gpu::mixed_host::U64);

    // Twin: the kernel against the test's f64 sums over the stored f32 values.
    let (k_twin, s_hat) = reference_k(&f32k.dressed_rows_for_test().unwrap(), &c);
    let twin = worst_ratio(&k32, &k_twin, &s_hat, floor);
    println!("f32 twin worst ratio {twin:.3e}");
    assert!(
        twin <= 1.0,
        "f32 K is not its twin to the f64 floor ({twin:.3e})"
    );

    // Contract against the f64 band.
    let (k_ref, s) = reference_k(&f64k.dressed_rows_for_test().unwrap(), &c);
    let ok64 = worst_ratio(&k64, &k_ref, &s, floor);
    assert!(
        ok64 <= 1.0,
        "f64 K is not the reference to the floor ({ok64:.3e})"
    );
    let eps = U32 + gamma(n, ferric_core::gpu::mixed_host::U64) * (1.0 + U32);
    let factor_k = (2.0 * eps + eps * eps) + floor * (1.0 + eps).powi(2);
    let ratio = worst_ratio(&k32, &k64, &s, factor_k);
    println!("f32 contract worst ratio {ratio:.3e}");
    assert!(ratio <= 1.0, "f32 K breaks its bound ({ratio:.3e})");
    assert!(ratio > 1e-3, "bound vacuous ({ratio:.3e})");
}

/// `max |a − b| / (f·s)` over the elements.
fn worst_ratio(a: &Array2<f64>, b: &Array2<f64>, s: &Array2<f64>, f: f64) -> f64 {
    a.iter()
        .zip(b.iter())
        .zip(s.iter())
        .map(|((x, y), s)| (x - y).abs() / (f * s).max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

#[test]
fn f32_scf_energy_sits_in_its_variational_sandwich() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    let e64 = rhf(&sys, Some(RiKPrecision::F64));
    let e32 = rhf(&sys, Some(RiKPrecision::F32));
    assert_eq!(
        e32.df_jk.as_ref().unwrap().ri_k_precision,
        RiKPrecision::F32
    );
    let mut f64k = sys.dfk(RiKPrecision::F64);
    let mut f32k = sys.dfk(RiKPrecision::F32);
    // E'_K(D) − E_K(D) = −¼ tr(D (K̂(D) − K(D))).
    let mut delta = |d: &Array2<f64>| {
        let dk = k_density(&mut f32k, d) - k_density(&mut f64k, d);
        -0.25 * (d * &dk).sum()
    };
    let upper = delta(&e64.density_total);
    let lower = delta(&e32.density_total);
    let de = e32.energy - e64.energy;
    let term = |r: &ferric_scf::ScfResult| r.convergence.as_ref().map_or(0.0, |c| c.de.abs());
    let tol = 2.0 * (term(&e64) + term(&e32)) + 1e-12 * e64.energy.abs();
    println!(
        "E64 {:.12} E32 {:.12} dE {de:.3e} L {lower:.3e} U {upper:.3e} tol {tol:.3e} it {}/{}",
        e64.energy, e32.energy, e64.iterations, e32.iterations
    );
    assert!(de >= lower - tol && de <= upper + tol, "dE outside [L, U]");
    assert!(
        de.abs() > tol,
        "sandwich not resolved: |dE| below termination"
    );
}

#[cfg(feature = "test-seams")]
#[test]
fn mutants_are_caught() {
    use std::sync::atomic::Ordering;
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("water.xyz", "def2-svp");
    let n = sys.obs.nbasis();
    // Truncation: stored values are no longer the nearest rounding.
    ferric_scf::df_k::TRUNCATE_DFK_F32.store(true, Ordering::SeqCst);
    let f32k = sys.dfk(RiKPrecision::F32);
    ferric_scf::df_k::TRUNCATE_DFK_F32.store(false, Ordering::SeqCst);
    let f64k = sys.dfk(RiKPrecision::F64);
    let (DressedRows::F64(b), DressedRows::F32(bh)) = (
        f64k.dressed_rows_for_test().unwrap(),
        f32k.dressed_rows_for_test().unwrap(),
    ) else {
        panic!()
    };
    let bad = b
        .iter()
        .zip(bh.iter())
        .filter(|(x, y)| (**x as f32).to_bits() != y.to_bits())
        .count();
    assert!(bad > 0, "truncation mutant survived the storage check");
    // Skewed unpack: packed f64 K no longer equals the square one.
    let c = factor(n, 5);
    let mut sq = sys.dfk_square();
    let mut pk = sys.dfk(RiKPrecision::F64);
    ferric_scf::df_k::SKEW_DFK_PACKED_UNPACK.store(true, Ordering::SeqCst);
    let skewed = k_occ(&mut pk, &c);
    ferric_scf::df_k::SKEW_DFK_PACKED_UNPACK.store(false, Ordering::SeqCst);
    assert!(
        skewed != k_occ(&mut sq, &c),
        "skew mutant survived the bitwise check"
    );
}

/// Timing probe (not a gate): occupied-path K from the square, packed f64 and
/// packed f32 tiers on n-hexadecane / def2-SVP, the same orbitals, best of 5
/// builds each. Run on a quiet box:
/// `cargo test --release -p ferric-scf --test dfk_packed_storage -- --ignored --nocapture k_build_time`.
#[test]
#[ignore = "timing probe; run by hand on a quiet box"]
fn k_build_time_square_vs_packed() {
    let _g = lock();
    let _h = Hooks;
    let sys = System::new("alkane_16.xyz", "def2-svp");
    let n = sys.obs.nbasis();
    let nocc = sys.mol.nelec() as usize / 2;
    let c = factor(n, nocc);
    let mut tiers = [
        ("square f64", sys.dfk_square()),
        ("packed f64", sys.dfk(RiKPrecision::F64)),
        ("packed f32", sys.dfk(RiKPrecision::F32)),
    ];
    for (label, k) in tiers.iter_mut() {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            std::hint::black_box(k_occ(k, &c));
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{label}: best of 5 occupied-path K builds {best:.3} s (n {n}, nocc {nocc})");
    }
}
