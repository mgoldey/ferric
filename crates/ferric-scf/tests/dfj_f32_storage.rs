//! `dfj-f32`: the CPU RI-J raw tensor STORED as f32 (one nearest rounding per
//! integral), both passes accumulated in f64 against f64 vectors. CPU only, no
//! device. The kernel is opt-in (`[precision] mode = "mixed"`,
//! `mixed_kernels = "dfj-f32"`); the tests switch it per builder with the
//! `set_dfj_f32_for_test` hook and hold `LOCK` (the hook is process-wide).
//!
//! WHAT IS DERIVED (module docs of `ferric_scf::df_j`, "f32 tensor storage";
//! u32 = 2⁻²⁴, u64 = 2⁻⁵³, γ_k = ku64/(1−ku64), η = 2⁻¹⁵⁰):
//!  * Storage: every stored value is `x as f32` of the f64 tier's value, bit for
//!    bit (`stored_values_are_the_nearest_rounding`).
//!  * Per pass, elementwise, against the f64 stored tensor `B` with the test's
//!    sequential f64 sums (which add their own γ):
//!    `|d̂_P − d_P| ≤ (ε_d + γ_cols)(|B||w|)_P + 2ηΣ|w|`,  ε_d = u32 + γ_cols(1+u32)
//!    `|Ĵ_k − J_k| ≤ (ε_J + γ_naux)(|B|ᵀ|c|)_k + 2ηΣ|c|`, ε_J = u32 + γ_naux(1+u32)
//!    with the SAME `c` given to both (`df_j::f32_passes_for_test`). Not vacuous:
//!    the worst ratio must exceed 1e-3 (a nearest rounding uses a visible part of
//!    its u32 budget somewhere in hundreds of rows).
//!  * TWIN (sharp): against f64 sums over `B` rounded to nearest by the TEST, the
//!    kernel differs only by f64 summation order: bound `2γ_cols Σ|B̂||w|`
//!    (`2γ_naux Σ|B̂|ᵀ|c|`). The mutants (`test-seams`): rounding toward zero
//!    and f32 accumulation each break the twin check (ratio > 1) and the first
//!    also the storage check.
//!  * End to end through `JBuilder::build`: J equals the hook passes on
//!    `ĉ = V⁻¹d̂` bit for bit, and `|Ĵ_k − (B c)_k| ≤ (ε_J+γ_naux)(|B|ᵀ|ĉ|)_k +
//!    (1+2γ_naux)(|B|ᵀ|ĉ−c|)_k + 2ηΣ|ĉ|` with `c = V⁻¹(Bw)` (the δc term
//!    carries ‖V⁻¹‖ and is MEASURED, not bounded).
//!  * Memory: the f32 tier charges exactly 4 B/element, half the f64 tier with
//!    the same layout (packed, pair-screened, and the converted shared tensor).
//!  * SCF (RHF, RI-JK, water/cc-pVDZ and benzene/def2-SVP): with D* the f64
//!    minimum, D' the f32 one and `E'` the rounded-B functional,
//!    `L = E'_J(D') − E_J(D') ≤ E_f32 − E_f64 ≤ E'_J(D*) − E_J(D*) = U`,
//!    within `bar` (floors and termination, below), and the f64 energy exit
//!    `E_f32 + (E_J(D') − E'_J(D'))` lands in `[0, U − L]` of the f64 energy.
//!    `bar = 2(|dE_exit,f64| + |dE_exit,f32|) + T(D*) + T(D') + 2F(D*) + 2F(D')`:
//!    `F = (γ_pair+γ_naux)S_J` is the summation floor of ONE `½dᵀV⁻¹d`
//!    evaluation (`S_J = |c|ᵀ|B||w|`), `T = F + g_K·½Σ|D|∘S_K +
//!    γ_{n²}·½Σ|D||H+F|` the `gpu_jk_ladder.rs` floor of a whole energy (`S_K`
//!    from the V^{-1/2}-dressed tensor, `g_K = 2γ_n + γ_{naux·nocc+naux}`). As
//!    there, the Cholesky solve is NOT in the floor: its worst-case backward
//!    error `½γ_{3naux+1}‖|Lᵀ||c|‖²` (Higham Thm 10.4) is printed, not added (it
//!    is not sharp for an ill-conditioned jkfit metric; the measured smoothness
//!    of `E_J` is ~1e-13, `df_j_energy_is_smooth_in_the_density`). Not vacuous:
//!    `|ΔE| > bar`. The 2·|dE_exit| termination terms follow `gpu_jk_ladder.rs`.
//!  * Off by default: an unset request builds the f64 tiers, bit-identical to an
//!    explicit off, and moves no `dfj-f32` counter; the default resolved settings
//!    do not allow `dfj-f32`; `mode = mixed` + `dfj-f32` resolves with the GPU off.
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit, PrecisionExplicit};
use ferric_core::gpu::mixed_host::{gamma, U32, U64};
use ferric_core::gpu::provenance::{KernelDevice, PrecisionProbe};
use ferric_core::gpu::stats::kernel_counters;
use ferric_core::gpu::{GpuMode, MixedKernel, MixedKernelSet, Precision};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{ThreeIndexSource, F32_UNDERFLOW_ABS};
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::df_j::{
    dfj_f32_d_error_factor, dfj_f32_energy_error_factor, dfj_f32_j_error_factor, dfj_f32_requested,
    ri_j_pair_screen_for_test, set_dfj_f32_for_test, DfJ,
};
use ferric_scf::fock::JBuilder;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_linalg::cholesky::Cholesky;
use ndarray_linalg::{Eigh, UPLO};
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

const AUX: &str = "def2-universal-jkfit";

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Restores the request hook (and the mutation seams) on drop.
struct Request;
impl Request {
    fn set(on: Option<bool>) -> Self {
        set_dfj_f32_for_test(on);
        Request
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        set_dfj_f32_for_test(None);
        #[cfg(feature = "test-seams")]
        {
            use std::sync::atomic::Ordering;
            ferric_scf::df_j::TRUNCATE_DFJ_F32.store(false, Ordering::SeqCst);
            ferric_scf::df_j::ACCUMULATE_DFJ_F32_IN_F32.store(false, Ordering::SeqCst);
        }
    }
}

struct System {
    label: &'static str,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
}

impl System {
    fn new(label: &'static str, xyz: &str, obs: &str) -> Self {
        let mol = Molecule::load_xyz(xyz).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        Self {
            label,
            mol,
            obs,
            dfbs,
        }
    }
    fn n(&self) -> usize {
        self.obs.nbasis()
    }
    fn naux(&self) -> usize {
        self.dfbs.nbasis()
    }
    /// `DfJ::new_for_jk` (the SCF RI-J constructor), with `dfj-f32` on or off.
    fn dfj(&self, f32: bool, budget: usize, pair_thresh: f64) -> DfJ<'static> {
        let _r = Request::set(Some(f32));
        let j = DfJ::new_for_jk(
            Operator::coulomb(),
            &self.mol,
            &self.obs,
            &self.dfbs,
            budget,
            None,
            pair_thresh,
        )
        .unwrap();
        assert_eq!(j.is_f32_storage(), f32, "{}: wrong storage", self.label);
        j
    }
}

fn water() -> System {
    System::new(
        "H2O/cc-pVDZ",
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
    )
}

fn benzene() -> System {
    System::new(
        "C6H6/def2-SVP",
        "../../testdata/molecules/benzene.xyz",
        "def2-svp",
    )
}

/// A symmetric test density with both signs and a spread of sizes.
fn density(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        0.02 * ((((i + j) * 7 + i * j) % 11) as f64 - 4.0) / (1.0 + (i as f64 - j as f64).abs())
    })
}

fn aux_vector(naux: usize) -> Array1<f64> {
    Array1::from_shape_fn(naux, |p| ((p * 13) % 17) as f64 / 17.0 - 0.4)
}

/// Weights `w_k` of the stored columns: `D_μμ` on the diagonal, `D_μν + D_νμ`
/// off it (so `Σ_k B_k w_k = Σ_μν B_μν D_μν`).
fn weights(d: &Array2<f64>, cols: &[(u32, u32)]) -> Vec<f64> {
    cols.iter()
        .map(|&(m, n)| {
            let (m, n) = (m as usize, n as usize);
            if m == n {
                d[(m, m)]
            } else {
                d[(m, n)] + d[(n, m)]
            }
        })
        .collect()
}

/// `y = B w` and `S = |B||w|` (sequential f64).
fn host_d(b: &ArrayView2<f64>, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let rows = b.nrows();
    let (mut y, mut s) = (vec![0.0; rows], vec![0.0; rows]);
    for ((row, yp), sp) in b.rows().into_iter().zip(&mut y).zip(&mut s) {
        for (&bk, &wk) in row.iter().zip(w) {
            let t = bk * wk;
            *yp += t;
            *sp += t.abs();
        }
    }
    (y, s)
}

/// `y = Bᵀ c` and `S = |B|ᵀ|c|` (sequential f64).
fn host_j(b: &ArrayView2<f64>, c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let cols = b.ncols();
    let (mut y, mut s) = (vec![0.0; cols], vec![0.0; cols]);
    for (row, &cp) in b.rows().into_iter().zip(c) {
        for ((yk, sk), &bk) in y.iter_mut().zip(s.iter_mut()).zip(row.iter()) {
            let t = bk * cp;
            *yk += t;
            *sk += t.abs();
        }
    }
    (y, s)
}

fn nearest(b: &Array2<f64>) -> Array2<f64> {
    b.mapv(|x| f64::from(x as f32))
}

fn abs_sum(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// Worst `|got − want| / (eps·S + extra)`; a zero bound admits only equality.
fn worst(got: &[f64], want: &[f64], s: &[f64], eps: f64, extra: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .zip(s)
        .map(|((g, w), s)| {
            let bound = eps * s + extra;
            let diff = (g - w).abs();
            if bound > 0.0 {
                diff / bound
            } else if diff == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

fn at_cols(j: &Array2<f64>, cols: &[(u32, u32)]) -> Vec<f64> {
    cols.iter()
        .map(|&(m, n)| j[(m as usize, n as usize)])
        .collect()
}

/// The f64 stored rows and their columns of an f64 builder with the same layout.
fn f64_rows(sys: &System, pair_thresh: f64) -> (Array2<f64>, Vec<(u32, u32)>) {
    let mut f64j = sys.dfj(false, usize::MAX, pair_thresh);
    f64j.source_mut_for_test()
        .stored_rows_f64_for_test()
        .unwrap()
}

#[derive(Debug, Clone, Copy)]
struct Ratios {
    exact_d: f64,
    exact_j: f64,
    twin_d: f64,
    twin_j: f64,
    stored_exact: bool,
}

impl Ratios {
    /// The per-build check the test asserts (and a mutant must fail).
    fn passes(&self) -> bool {
        self.stored_exact
            && self.exact_d <= 1.0
            && self.exact_j <= 1.0
            && self.twin_d <= 1.0
            && self.twin_j <= 1.0
    }
}

/// The per-pass statements for one f32 builder against the f64 rows `b64`.
fn ratios(
    dfj32: &mut DfJ<'_>,
    b64: &Array2<f64>,
    cols: &[(u32, u32)],
    d: &Array2<f64>,
    c: &Array1<f64>,
) -> Ratios {
    let (b32, cols32) = dfj32
        .source_mut_for_test()
        .stored_rows_f64_for_test()
        .unwrap();
    assert_eq!(cols32, cols, "f32 and f64 tiers store different columns");
    let twin = nearest(b64);
    let stored_exact = b32.dim() == twin.dim()
        && b32
            .iter()
            .zip(twin.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let w = weights(d, cols);
    let (d32, j32) = dfj32.f32_passes_for_test(d, c).unwrap();
    let j32 = at_cols(&j32, cols);
    let c = c.to_vec();
    let (ncols, naux) = (b64.ncols(), b64.nrows());
    let (d_ref, s_d) = host_d(&b64.view(), &w);
    let (j_ref, s_j) = host_j(&b64.view(), &c);
    let (d_tw, s_dt) = host_d(&twin.view(), &w);
    let (j_tw, s_jt) = host_j(&twin.view(), &c);
    let (eta_w, eta_c) = (
        2.0 * F32_UNDERFLOW_ABS * abs_sum(&w),
        2.0 * F32_UNDERFLOW_ABS * abs_sum(&c),
    );
    let eps_d = dfj_f32_d_error_factor(ncols) + gamma(ncols, U64);
    let eps_j = dfj_f32_j_error_factor(naux) + gamma(naux, U64);
    Ratios {
        exact_d: worst(d32.as_slice().unwrap(), &d_ref, &s_d, eps_d, eta_w),
        exact_j: worst(&j32, &j_ref, &s_j, eps_j, eta_c),
        twin_d: worst(
            d32.as_slice().unwrap(),
            &d_tw,
            &s_dt,
            2.0 * gamma(ncols, U64),
            0.0,
        ),
        twin_j: worst(&j32, &j_tw, &s_jt, 2.0 * gamma(naux, U64), 0.0),
        stored_exact,
    }
}

/// The smallest Schwarz threshold (of a short ladder) whose screen drops pairs.
fn dropping_thresh(sys: &System) -> f64 {
    for t in [1e-10, 1e-8, 1e-6, 1e-4] {
        let s = ri_j_pair_screen_for_test(Operator::coulomb(), &sys.obs, &sys.dfbs, t).unwrap();
        if s.is_some_and(|s| s.ncols() < s.total_pairs()) {
            return t;
        }
    }
    panic!("{}: no threshold up to 1e-4 drops a pair", sys.label);
}

/// Each layout the f32 tier takes: packed (threshold 0) and, for benzene,
/// pair-screened.
fn layouts(sys: &System) -> Vec<(&'static str, f64)> {
    let mut v = vec![("packed", 0.0)];
    if sys.label.starts_with("C6H6") {
        v.push(("pair-screened", dropping_thresh(sys)));
    }
    v
}

#[test]
fn per_build_errors_are_inside_the_derived_bounds_and_not_vacuous() {
    let _g = lock();
    for sys in [water(), benzene()] {
        for (layout, thresh) in layouts(&sys) {
            let (b64, cols) = f64_rows(&sys, thresh);
            let mut dfj32 = sys.dfj(true, usize::MAX, thresh);
            if layout == "pair-screened" {
                assert!(cols.len() < sys.n() * (sys.n() + 1) / 2, "nothing dropped");
            }
            let d = density(sys.n());
            let c = aux_vector(sys.naux());
            let r = ratios(&mut dfj32, &b64, &cols, &d, &c);
            eprintln!(
                "{} {layout}: cols {} naux {}  ratio to bound: exact d {:.3e} J {:.3e} | twin d {:.3e} J {:.3e} | storage bit-exact {}",
                sys.label,
                cols.len(),
                sys.naux(),
                r.exact_d,
                r.exact_j,
                r.twin_d,
                r.twin_j,
                r.stored_exact
            );
            assert!(
                r.stored_exact,
                "{} {layout}: stored values are not x as f32",
                sys.label
            );
            assert!(
                r.passes(),
                "{} {layout}: a derived bound is violated: {r:?}",
                sys.label
            );
            assert!(
                r.exact_d > 1e-3 && r.exact_j > 1e-3,
                "{} {layout}: the exact bound is vacuous (no rounding visible): {r:?}",
                sys.label
            );
        }
    }
}

/// End to end through `JBuilder::build`: the build is the bounded passes on
/// `ĉ = V⁻¹d̂` bit for bit, inside the derived elementwise bound against the
/// exact-B contraction, deterministic across rayon worker counts, and it counts
/// one `dfj-f32` call per build.
#[test]
fn end_to_end_j_is_the_bounded_passes_and_deterministic() {
    let _g = lock();
    for sys in [water(), benzene()] {
        for (layout, thresh) in layouts(&sys) {
            let n = sys.n();
            let (b64, cols) = f64_rows(&sys, thresh);
            let mut dfj32 = sys.dfj(true, usize::MAX, thresh);
            let d = density(n);
            let k0 = kernel_counters()[MixedKernel::DfjF32.index()];
            let mut j = Array2::<f64>::zeros((n, n));
            dfj32.build(&d, &mut j).unwrap();
            let k1 = kernel_counters()[MixedKernel::DfjF32.index()];
            assert_eq!(
                k1.since(&k0).mixed,
                1,
                "{}: one build, one dfj-f32 call",
                sys.label
            );
            assert_eq!(
                j,
                j.t(),
                "{} {layout}: J must be exactly symmetric",
                sys.label
            );

            // The build is the hook's passes on ĉ = V⁻¹ d̂.
            let (d_hat, _) = dfj32
                .f32_passes_for_test(&d, &aux_vector(sys.naux()))
                .unwrap();
            let c_hat = dfj32.solve_metric_for_test(&d_hat).unwrap();
            let (_, j_hook) = dfj32.f32_passes_for_test(&d, &c_hat).unwrap();
            let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert!(
                bits(&j) == bits(&j_hook),
                "{} {layout}: build differs from its passes",
                sys.label
            );

            // Against the exact-B contraction with c = V⁻¹(B w).
            let w = weights(&d, &cols);
            let (d_ref, _) = host_d(&b64.view(), &w);
            let c_ref = dfj32.solve_metric_for_test(&Array1::from(d_ref)).unwrap();
            let (j_ref, _) = host_j(&b64.view(), &c_ref.to_vec());
            let (_, s_hat) = host_j(&b64.view(), &c_hat.to_vec());
            let dc: Vec<f64> = (&c_hat - &c_ref).to_vec();
            let (_, s_dc) = host_j(&b64.view(), &dc);
            let naux = sys.naux();
            let eps_j = dfj_f32_j_error_factor(naux) + gamma(naux, U64);
            let eta = 2.0 * F32_UNDERFLOW_ABS * abs_sum(&c_hat.to_vec());
            let got = at_cols(&j, &cols);
            let mut worst_ratio = 0.0f64;
            let mut dc_share = 0.0f64;
            let terms = got.iter().zip(&j_ref).zip(s_hat.iter().zip(&s_dc));
            for (k, ((g, r), (sh, sdc))) in terms.enumerate() {
                let bound = eps_j * sh + (1.0 + 2.0 * gamma(naux, U64)) * sdc + eta;
                let diff = (g - r).abs();
                assert!(
                    diff <= bound,
                    "{} {layout}: J[{k}] off by {diff:e} > bound {bound:e}",
                    sys.label
                );
                if bound > 0.0 {
                    worst_ratio = worst_ratio.max(diff / bound);
                    dc_share = dc_share.max(sdc / bound);
                }
            }
            let rel = (&j
                - &{
                    let mut f = sys.dfj(false, usize::MAX, thresh);
                    let mut jf = Array2::<f64>::zeros((n, n));
                    f.build(&d, &mut jf).unwrap();
                    jf
                })
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()))
                / j.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            eprintln!(
                "{} {layout}: end-to-end worst ratio {worst_ratio:.3e} (largest share of the measured δc term {dc_share:.3e}); max|J_f32 − J_f64 build|/max|J| {rel:.3e}",
                sys.label
            );
            assert!(
                rel > 0.0,
                "{} {layout}: f32 J equals f64 J: f32 tensor unused",
                sys.label
            );

            // Deterministic across worker counts.
            let mut first: Option<Vec<u64>> = None;
            for threads in [1usize, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let jt = pool.install(|| {
                    let mut f = sys.dfj(true, usize::MAX, thresh);
                    let mut jt = Array2::<f64>::zeros((n, n));
                    f.build(&d, &mut jt).unwrap();
                    jt
                });
                let b = bits(&jt);
                match &first {
                    None => first = Some(b),
                    Some(f) => assert!(*f == b, "{} {layout}: thread-count dependent", sys.label),
                }
            }
        }
    }
}

/// The record names the CPU kernel honestly: `dfj-f32`, mixed, on the CPU,
/// one call per build, no ladder.
#[test]
fn provenance_names_dfj_f32_on_the_cpu() {
    let _g = lock();
    let sys = water();
    let mut dfj32 = sys.dfj(true, usize::MAX, 0.0);
    let d = density(sys.n());
    let probe = PrecisionProbe::start();
    let mut j = Array2::<f64>::zeros((sys.n(), sys.n()));
    for _ in 0..3 {
        dfj32.build(&d, &mut j).unwrap();
    }
    let r = probe.finish(&[MixedKernel::DfjF32]);
    assert_eq!(r.precision, Precision::Mixed);
    let k = &r.kernels[0];
    assert_eq!(k.kernel, MixedKernel::DfjF32);
    assert_eq!(k.mode, Precision::Mixed);
    assert_eq!(k.device, Some(KernelDevice::Cpu));
    assert_eq!(k.calls_mixed, 3);
    assert_eq!(k.ladder, None, "dfj-f32 is not behind the SCF ladder");
    assert_eq!(r.to_json()["kernels"][0]["kernel"], "dfj-f32");
}

/// The f32 tier charges 4 B/element: exactly half the f64 tier of the same
/// layout (packed, pair-screened), and the shared-path conversion (unpacked
/// in-core and spilled f64 sources) charges the f32 packed bytes.
#[test]
fn the_f32_tier_charges_half_the_f64_bytes() {
    use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
    let _g = lock();
    let sys = benzene();
    let (n, naux) = (sys.n(), sys.naux());
    let pair = n * (n + 1) / 2;
    let _prev = install_global(MemoryPool::with_capacity_bytes(64 << 30));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Packed: an f64 budget where only the packed tier fits.
        let head = (64 * n * n * 8).min(512 << 20);
        let budget = naux * pair * 8 + head + (1 << 20);
        assert!(
            budget < naux * n * n * 8,
            "budget admits the unpacked tensor"
        );
        let mut f64p = sys.dfj(false, budget, 0.0);
        assert!(f64p.source_mut_for_test().is_packed_incore_for_test());
        let mut f32p = sys.dfj(true, budget, 0.0);
        let (c64, c32) = (
            f64p.source_mut_for_test().charged_bytes(),
            f32p.source_mut_for_test().charged_bytes(),
        );
        eprintln!("packed: f64 {c64} B, f32 {c32} B");
        assert_eq!(c64, 8 * naux * pair);
        assert_eq!(c32, 4 * naux * pair);
        assert_eq!(f32p.source_mut_for_test().resident_tensor_bytes(), c32);

        // Pair-screened.
        let t = dropping_thresh(&sys);
        let mut f64s = sys.dfj(false, usize::MAX, t);
        let ncols = f64s
            .source_mut_for_test()
            .pair_screened()
            .unwrap()
            .1
            .ncols();
        let mut f32s = sys.dfj(true, usize::MAX, t);
        let (s64, s32) = (
            f64s.source_mut_for_test().charged_bytes(),
            f32s.source_mut_for_test().charged_bytes(),
        );
        eprintln!("pair-screened ({ncols} of {pair} pairs): f64 {s64} B, f32 {s32} B");
        assert_eq!(s64, 8 * naux * ncols);
        assert_eq!(s32, 4 * naux * ncols);

        // Shared path: an adopted f64 source is converted and dropped.
        let op = Operator::coulomb();
        for (what, raw_budget) in [
            ("unpacked in-core", usize::MAX),
            ("spilled", n * n * 8 * 40),
        ] {
            let raw =
                ThreeIndexSource::build_band(op, &sys.obs, &sys.dfbs, raw_budget, 0, naux).unwrap();
            let _r = Request::set(Some(true));
            let mut conv = DfJ::from_source(raw, op, &sys.dfbs, usize::MAX, None).unwrap();
            assert!(conv.is_f32_storage(), "{what}: not converted");
            assert_eq!(
                conv.source_mut_for_test().charged_bytes(),
                4 * naux * pair,
                "{what}"
            );
            let (b32, _) = conv
                .source_mut_for_test()
                .stored_rows_f64_for_test()
                .unwrap();
            let (b64, _) = f64_rows(&sys, 0.0);
            assert!(
                b32.iter()
                    .zip(nearest(&b64).iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{what}: converted values are not x as f32"
            );
        }
    }));
    clear_global();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Off by default: the default request is off, builds the f64 tier, is
/// bit-identical to an explicit off and moves no `dfj-f32` counter; the
/// default settings do not allow it; it resolves (opt-in, CPU) with the GPU off.
#[test]
fn off_by_default_and_opt_in_resolves_on_the_cpu() {
    let _g = lock();
    let none = |_: &str| None::<String>;
    let (s, _) = GpuSettings::resolve(GpuSettingsExplicit::default(), none).unwrap();
    assert!(!s.mixed_allows(MixedKernel::DfjF32));
    assert!(!MixedKernelSet::SHIPPED.contains(MixedKernel::DfjF32));
    let only = MixedKernelSet::EMPTY.with(MixedKernel::DfjF32);
    for mode in [GpuMode::Off, GpuMode::Auto] {
        let (s, _) = GpuSettings::resolve(
            GpuSettingsExplicit {
                mode: Some(mode),
                precision_section: PrecisionExplicit {
                    mode: Some(Precision::Mixed),
                    mixed_kernels: Some(only),
                },
                ..Default::default()
            },
            none,
        )
        .unwrap_or_else(|e| panic!("mode {mode}: {e}"));
        assert!(s.mixed_allows(MixedKernel::DfjF32), "mode {mode}");
    }

    set_dfj_f32_for_test(None);
    assert!(!dfj_f32_requested(), "dfj-f32 is on without a request");
    let sys = benzene();
    let n = sys.n();
    let d = density(n);
    let k0 = kernel_counters()[MixedKernel::DfjF32.index()];
    let mut by_default = DfJ::new_for_jk(
        Operator::coulomb(),
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        usize::MAX,
        None,
        0.0,
    )
    .unwrap();
    assert!(!by_default.is_f32_storage());
    let mut off = sys.dfj(false, usize::MAX, 0.0);
    let (mut j1, mut j2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    by_default.build(&d, &mut j1).unwrap();
    off.build(&d, &mut j2).unwrap();
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert!(
        bits(&j1) == bits(&j2),
        "the default differs from an explicit off"
    );
    let k1 = kernel_counters()[MixedKernel::DfjF32.index()];
    assert_eq!(k1.since(&k0).mixed, 0, "a default build counted dfj-f32");
}

// ------------------------------------------------------------------ mutants

/// The per-build check FAILS under each mutant: rounding toward zero (storage
/// and twin) and f32 accumulation (twin).
#[cfg(feature = "test-seams")]
#[test]
fn mutants_fail_the_per_build_check() {
    use ferric_scf::df_j::{ACCUMULATE_DFJ_F32_IN_F32, TRUNCATE_DFJ_F32};
    use std::sync::atomic::Ordering;
    let _g = lock();
    for sys in [water(), benzene()] {
        let (b64, cols) = f64_rows(&sys, 0.0);
        let d = density(sys.n());
        let c = aux_vector(sys.naux());

        TRUNCATE_DFJ_F32.store(true, Ordering::SeqCst);
        let mut trunc = sys.dfj(true, usize::MAX, 0.0);
        TRUNCATE_DFJ_F32.store(false, Ordering::SeqCst);
        let r = ratios(&mut trunc, &b64, &cols, &d, &c);
        eprintln!("{} truncation mutant: {r:?}", sys.label);
        assert!(
            !r.stored_exact,
            "{}: truncation not seen in storage",
            sys.label
        );
        assert!(
            r.twin_d > 1.0 && r.twin_j > 1.0,
            "{}: truncation passes the twin",
            sys.label
        );
        assert!(!r.passes());

        let mut clean = sys.dfj(true, usize::MAX, 0.0);
        let _r = Request::set(Some(true));
        ACCUMULATE_DFJ_F32_IN_F32.store(true, Ordering::SeqCst);
        let r = ratios(&mut clean, &b64, &cols, &d, &c);
        ACCUMULATE_DFJ_F32_IN_F32.store(false, Ordering::SeqCst);
        eprintln!("{} f32-accumulation mutant: {r:?}", sys.label);
        assert!(r.stored_exact);
        assert!(
            r.twin_d > 1.0 && r.twin_j > 1.0,
            "{}: f32 accumulation passes the twin",
            sys.label
        );
        assert!(!r.passes());
    }
}

// ---------------------------------------------------------------- the SCF

fn scf(sys: &System, f32: bool) -> ScfResult {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &sys.obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-6,
        density_conv: 1e-9,
        max_iter: 200,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        three_index_budget_bytes: 3 << 30,
        ..Default::default()
    };
    let _r = Request::set(Some(f32));
    let k0 = kernel_counters()[MixedKernel::DfjF32.index()];
    let res = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        op,
        &bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{} f32={f32}: {e:?}", sys.label));
    let ran = kernel_counters()[MixedKernel::DfjF32.index()]
        .since(&k0)
        .mixed;
    assert!(res.converged, "{} f32={f32}: not converged", sys.label);
    assert_eq!(
        ran > 0,
        f32,
        "{} f32={f32}: {ran} dfj-f32 builds",
        sys.label
    );
    res
}

fn last_de(r: &ScfResult) -> f64 {
    r.convergence.as_ref().map(|c| c.de.abs()).unwrap_or(0.0)
}

/// What the energy checks need of the tensor and the metric.
struct Model {
    b: Array2<f64>,
    b_hat: Array2<f64>,
    cols: Vec<(u32, u32)>,
    /// V^{-1/2}-dressed tensor `(naux, n²)` for the K floor.
    dressed: Array2<f64>,
    /// `|Lᵀ|` of the metric's Cholesky factor `V = L Lᵀ` (solve floor).
    abs_lt: Array2<f64>,
    solver: DfJ<'static>,
    h: Array2<f64>,
}

impl Model {
    fn new(sys: &System) -> Self {
        let op = Operator::coulomb();
        let naux = sys.naux();
        let mut raw =
            ThreeIndexSource::build_band(op, &sys.obs, &sys.dfbs, usize::MAX, 0, naux).unwrap();
        let flat = raw.incore_flat().expect("in-core f64 tensor").to_owned();
        let (b, cols) = raw.stored_rows_f64_for_test().unwrap();
        let v = coulomb_metric_2c(op, &sys.dfbs).unwrap();
        let (lam, u) = v.eigh(UPLO::Lower).unwrap();
        assert!(lam.iter().all(|&l| l > 0.0));
        let vmh = u
            .dot(&Array2::from_diag(&lam.mapv(|l| 1.0 / l.sqrt())))
            .dot(&u.t());
        let l = v.cholesky(UPLO::Lower).unwrap();
        let solver = DfJ::new(op, &sys.obs, &sys.dfbs, usize::MAX).unwrap();
        Self {
            b_hat: nearest(&b),
            b,
            cols,
            dressed: vmh.dot(&flat),
            abs_lt: l.t().mapv(f64::abs),
            solver,
            h: oneelectron::hcore(&sys.obs),
        }
    }

    /// `(E_J, E'_J, S_J, F, Q, solve)` at `d`: the exact-B and rounded-B
    /// Coulomb energies `½dᵀV⁻¹d`, `S_J = |c|ᵀ|B||w|`, the one-evaluation
    /// summation floor `F`, `Q = ½δdᵀV⁻¹δd` (the second-order term) and the
    /// worst-case Cholesky backward-error term (printed only).
    fn coulomb(&self, d: &Array2<f64>) -> (f64, f64, f64, f64, f64, f64) {
        let w = weights(d, &self.cols);
        let (dv, sd) = host_d(&self.b.view(), &w);
        let (dh, _) = host_d(&self.b_hat.view(), &w);
        let (dv, dh) = (Array1::from(dv), Array1::from(dh));
        let c = self.solver.solve_metric_for_test(&dv).unwrap();
        let ch = self.solver.solve_metric_for_test(&dh).unwrap();
        let dd = &dh - &dv;
        let q = 0.5 * dd.dot(&self.solver.solve_metric_for_test(&dd).unwrap());
        let s_j: f64 = c.iter().zip(&sd).map(|(c, s)| c.abs() * s).sum();
        let (pair, naux) = (self.b.ncols(), self.b.nrows());
        let lc = self.abs_lt.dot(&c.mapv(f64::abs));
        let floor = (gamma(pair, U64) + gamma(naux, U64)) * s_j;
        let solve = 0.5 * gamma(3 * naux + 1, U64) * lc.dot(&lc);
        (0.5 * dv.dot(&c), 0.5 * dh.dot(&ch), s_j, floor, q, solve)
    }

    /// `T` of one converged energy: `F + g_K ½Σ|D|∘S_K + γ_{n²} ½Σ|D||H+F|`.
    fn energy_floor(&self, sys: &System, r: &ScfResult, f_j: f64) -> f64 {
        let n = sys.n();
        let nocc = sys.mol.nelec() as usize / 2;
        let c = r.mos_alpha.slice(ndarray::s![.., ..nocc]).mapv(f64::abs);
        let all = self.dressed.as_slice().expect("standard layout");
        let mut s = Array2::<f64>::zeros((n, n));
        for p in 0..self.dressed.nrows() {
            let bp = ArrayView2::from_shape((n, n), &all[p * n * n..(p + 1) * n * n]).unwrap();
            let a = bp.mapv(f64::abs).dot(&c);
            s += &a.dot(&a.t());
        }
        let abs_dot = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum()
        };
        let naux = sys.naux();
        let g_k = 2.0 * gamma(n, U64) + gamma(naux * nocc + naux, U64);
        let s_k = 0.5 * abs_dot(&r.density_total, &s);
        let s_tr = 0.5 * abs_dot(&r.density_total, &(&self.h + &r.fock_alpha));
        f_j + g_k * s_k + gamma(n * n, U64) * s_tr
    }
}

/// The f32 SCF's energy sits in the derived first-order window `[L, U]`; the
/// f64 energy exit brings it into the second-order window `[0, U − L]`.
#[test]
fn scf_energy_is_inside_the_derived_window_and_the_f64_exit_is_second_order() {
    let _g = lock();
    for sys in [water(), benzene()] {
        let r64 = scf(&sys, false);
        let r32 = scf(&sys, true);
        let m = Model::new(&sys);
        let (ej_s, ejh_s, sj_s, f_s, q_s, solve_s) = m.coulomb(&r64.density_total);
        let (ej_p, ejh_p, sj_p, f_p, _, _) = m.coulomb(&r32.density_total);
        let (u, l) = (ejh_s - ej_s, ejh_p - ej_p);
        let de = r32.energy - r64.energy;
        let bar = 2.0 * (last_de(&r64) + last_de(&r32))
            + m.energy_floor(&sys, &r64, f_s)
            + m.energy_floor(&sys, &r32, f_p)
            + 2.0 * (f_s + f_p);

        // The f64 exit through the production pass (a fresh f32 builder).
        let exit = sys
            .dfj(true, usize::MAX, 0.0)
            .f64_exit_coulomb_energy(&r32.density_total)
            .unwrap()
            .expect("new_for_jk enables the f64 exit");
        let corr = exit.correction();
        let corrected = de + corr;
        let (pair, naux) = (m.b.ncols(), sys.naux());
        let m_first = dfj_f32_energy_error_factor(pair, naux) * sj_s.max(sj_p);
        eprintln!(
            "{}: E_f32 − E_f64 {de:.3e}  window [L {l:.3e}, U {u:.3e}] (U − L {:.3e})  bar {bar:.3e}  \
             first-order estimate {m_first:.3e} (|dE|/est {:.3e})  exit correction {corr:.3e} (vs −L: {:.3e})  \
             corrected error {corrected:.3e}  [worst-case solve term, not in bar: {solve_s:.3e}]  \
             iterations f64 {} f32 {}",
            sys.label,
            u - l,
            de.abs() / m_first,
            corr + l,
            r64.iterations,
            r32.iterations
        );
        assert!(
            de.abs() > bar,
            "{}: the f32 effect is not above the floors",
            sys.label
        );
        assert!(
            l - bar <= de && de <= u + bar,
            "{}: ΔE outside [L, U]",
            sys.label
        );
        assert!(u - l >= -bar, "{}: the window is inverted", sys.label);
        // The first-order model: −u32·S_J' ≤ L and U ≤ u32·S_J* + Q* (|δB| ≤ u32|B| + η).
        assert!(
            -(U32 * sj_p) - bar <= de && de <= U32 * sj_s + q_s + bar,
            "{}: ΔE outside the first-order model",
            sys.label
        );
        // The production exit is −L to the floors of four evaluations.
        assert!((corr + l).abs() <= 4.0 * f_p, "{}: exit ≠ −L", sys.label);
        assert!(
            -bar - 4.0 * f_p <= corrected && corrected <= (u - l) + bar + 4.0 * f_p,
            "{}: the f64 exit is not second order",
            sys.label
        );
    }
}
