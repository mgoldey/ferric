//! Stage 3: k-point RHF (`ferric_pbc::kscf::solve_krhf_injected`) on the
//! per-k one-electron matrices (`hcore::kpoint::periodic_hcore_kpts`) and the
//! dense pure-AFT k-point J/K oracle (`kdense_aft::KDenseAftEri`).
//!
//! Port of the prototype's Iteration-9 tests (`reference/pbc/pbc_kpts.py`,
//! `run_kpts_anchor.py`, `run_kpts_oracle.py`, the end of
//! `test_prototype.py`; FINDINGS "Iteration 9").
//!
//! (a) 1×1×1 mesh ≡ the Gamma `solve_rhf_injected` path (dense AFT), both
//!     exxdiv, ≤ 1e-12 — an INDEPENDENT construction (real half-sphere Gamma
//!     code vs complex full-sphere residue code).
//! (b) Gamma-centred N-mesh E/cell ≡ Gamma RHF of the explicit diag(N)
//!     supercell / N, both exxdiv, ≤ 1e-11. `{G + q}` over the mesh IS the
//!     supercell reciprocal lattice and both sides cut the same |K| sphere
//!     with the same pair screen, so the anchor is exact at ANY precision —
//!     a loose one (1e-6) keeps it cheap. Needs N >= 3 on some axis to see a
//!     phase SIGN (at N = 2, e^{ik·L} = e^{−ik·L}); 2×2×2 checks the 3-D
//!     residue bookkeeping. Also: none − ewald = nocc·v_M exactly, and the
//!     k-mesh v_M IS the supercell's.
//! (c) S(k) vs PySCF `pbc_intor("int1e_ovlp", kpts=)` (embedded; the only
//!     test that sees a GLOBAL phase-sign flip k → −k, which every energy is
//!     blind to by time reversal), S(−k) = S(k)* exactly, V(−k) = V(k)*,
//!     J/K Hermitian for a Hermitian test density, k-points vs PySCF
//!     `make_kpts` (Gamma-centred and Monkhorst-Pack), mesh v_M vs PySCF
//!     `tools.pbc.madelung(cell, kpts)`.
//! (d) PySCF 2.13 KRHF + AFTDF (mesh 61³, cell.precision 1e-12) pins from
//!     `run_kpts_oracle.py` (H2 1×1×2 and 2×2×2, triclinic s+p 1×1×2), and a
//!     shifted Monkhorst-Pack 1×1×2 H2 pin (PySCF own guess, conv 1e-11,
//!     generated 2026-09-24 for this port).
//! (e) Mutations (`KMutation`, the prototype's `_MUTANT`; prototype
//!     magnitudes on H2 1×1×3: phase −0.219, kernel +0.377, Madelung −0.520
//!     Ha; the Rust phase mutant flips the ERI pair FT only, V keeps its
//!     phase, so its magnitude differs and is unmeasured here — the
//!     assertion is a loose 1e-3):
//!     * `PhaseSignInPairFt` (e^{−ik'·L} in the ERI pair FT only) — caught
//!       by (b); flipping the phase EVERYWHERE is an exact relabelling
//!       k → −k and only (c)'s PySCF S(k) pin sees it.
//!     * `KernelAtG` (v(G) instead of v(G + q)) — caught by (b), both exxdiv.
//!     * `PrimitiveMadelung` (primitive-cell v_M) — caught by (b) ewald only;
//!       the none row must stay exact (asserted).
//!
//! Cost: (a), (b)-H2 and (d)-H2 are small. The triclinic cases run the SR
//! nuclear sum on a 12-atom supercell / a full-sphere 1e-14 K set and are
//! UNMEASURED in Rust: if either exceeds ~60 s in release, mark it
//! `#[ignore = "slow: ..."]`.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri, KMutation};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::lattice::Cell;
use ndarray::Array2;
use num_complex::Complex64;

/// Nuclear-attraction Ewald split (as `pbc_dense_aft_scf.rs`).
const OMEGA: f64 = 0.8;
/// Loose AFT precision for the supercell anchors (exact at any precision).
const ANCHOR_PRECISION: f64 = 1e-6;

// ---- (d) PySCF pins (run_kpts_oracle.py, FINDINGS "Iteration 9") ----------
const H2_112: (f64, f64) = (-0.902683427348, -1.354143879961); // (none, ewald)
const H2_222: (f64, f64) = (-0.700885391756, -1.055547576692);
const TRI_112: (f64, f64) = (-1.587649533398, -2.327120141714);
/// PySCF KRHF/AFTDF (mesh 61³, precision 1e-12, conv 1e-11, own guess),
/// H2/STO-3G a = 4, `make_kpts([1,1,2], with_gamma_point=False)`
/// (k = ±b₃/4): (none, ewald). none − ewald − v_M = −9e-15 in PySCF too.
const H2_MP112: (f64, f64) = (-0.9745084309202314, -1.425968883533299);
/// PySCF `tools.pbc.madelung(cell, make_kpts([1,1,2], with_gamma_point=False))`.
const H2_MADELUNG_112: f64 = 0.45146045261307666;

// ---- (c) PySCF pbc_intor("int1e_ovlp", hermi=1, kpts=make_kpts([1,1,3])) --
/// H2/STO-3G a = 4, k = 0, b₃/3, 2b₃/3; row-major (re, im).
const H2_SK_113: [[(f64, f64); 4]; 3] = [
    [
        (1.8564558917189458, 0.0),
        (1.6425187685488494, 0.0),
        (1.6425187685488494, 0.0),
        (1.8564558917189455, 0.0),
    ],
    [
        (1.2803501252635219, -9.196372639647709e-17),
        (0.7406892634839631, -0.4091261219362238),
        (0.7406892634839631, 0.4091261219362238),
        (1.2803501252635219, 4.3134265202357096e-18),
    ],
    [
        (1.2803501252635219, 9.610871559935468e-17),
        (0.7406892634839627, 0.4091261219362236),
        (0.7406892634839627, -0.4091261219362236),
        (1.2803501252635219, 4.1683713725839186e-18),
    ],
];
/// PySCF `make_kpts([1,1,3])` for H2 a = 4 (z components; x = y = 0).
const H2_KZ_113: [f64; 3] = [0.0, 0.5235987755982988, 1.0471975511965976];

/// Triclinic 4H s+p, `make_kpts([1,1,3])` point 1, rows 0..4 (atom 0: s,
/// px, py, pz) × all 16 columns, row-major (re, im).
const TRI_SK1_ROWS: [(f64, f64); 64] = [
    // row 0
    (1.2181600092936196, 1.3552527156068805e-20),
    (4.777265822514254e-19, -0.0012962282958369118),
    (7.087336427913545e-18, -0.0015057872130684632),
    (2.489059255065965e-18, -0.03846355145349451),
    (0.7640974971230913, -0.20099749919961668),
    (-0.0030532012890440996, -0.005426580705673024),
    (-0.003788780218894206, -0.006723183027946358),
    (-0.5238054743298768, -0.08562950611958638),
    (0.3854375551925698, -0.27354378325708373),
    (-0.0225458939437533, 0.010827793043852806),
    (0.026390923590158676, 0.004142250871883179),
    (-0.20073187041300178, -0.08922556719146933),
    (0.25252703026716433, -0.3441515648428281),
    (0.015278477774750417, -0.008164566755527706),
    (0.014059701950439667, 0.007132409125700862),
    (-0.19425957176375963, -0.10228648162290126),
    // row 1
    (4.777265822514254e-19, 0.0012962282958369118),
    (0.9934835635419377, -2.0679020303607666e-24),
    (-0.002662020403363754, 6.897546081460776e-21),
    (0.00013971763981676096, 6.939051610152912e-21),
    (0.0030532012890441135, 0.005426580705673015),
    (0.4507484038579238, -0.004986421241347604),
    (-0.00018395183743906974, 0.0018031531771927492),
    (0.004883780485412591, 0.008458295676782046),
    (0.022545893943753254, -0.010827793043852806),
    (-0.0362812263370561, 0.006807651567952303),
    (0.04633170135425258, 0.002881018942272409),
    (-0.04916105387191695, -0.011650128101769863),
    (-0.015278477774750434, 0.008164566755527685),
    (-0.024006017111493718, 0.01435766847128698),
    (-0.011479271035797019, -0.0137514154708318),
    (0.03276060452548788, 0.011492632504703443),
    // row 2
    (7.087336427913545e-18, 0.0015057872130684632),
    (-0.002662020403363754, -6.897546081460776e-21),
    (0.9880673604338991, 8.134834385877917e-21),
    (0.00019187876907945052, -4.988488380669397e-20),
    (0.0037887802188942025, 0.006723183027946356),
    (-0.00018395183743907025, 0.0018031531771927482),
    (0.4491681231099818, -0.003425972730483548),
    (0.00670705423558376, 0.011616051954734568),
    (-0.026390923590158696, -0.004142250871883179),
    (0.04633170135425259, 0.0028810189422724054),
    (-0.06639837347133976, 0.009278822549174661),
    (0.041532106781233175, -0.0052727979664538484),
    (-0.014059701950439504, -0.007132409125700866),
    (-0.011479271035797005, -0.0137514154708318),
    (-0.020766970251886983, 0.01922087159463541),
    (0.0069904830571701155, -0.012199936444957028),
    // row 3
    (2.489059255065965e-18, 0.03846355145349451),
    (0.00013971763981676096, -6.939051610152912e-21),
    (0.00019187876907945052, 4.988488380669397e-20),
    (1.0026148877868069, -2.4805926417017874e-20),
    (0.523805474329877, 0.08562950611958638),
    (0.004883780485412593, 0.00845829567678205),
    (0.006707054235583765, 0.011616051954734578),
    (-0.2293173319782581, 0.052574895307900535),
    (0.20073187041300186, 0.08922556719146932),
    (-0.04916105387191694, -0.011650128101769859),
    (0.041532106781233195, -0.0052727979664538484),
    (-0.04032091762473197, 0.026304462746990154),
    (0.19425957176375977, 0.10228648162290134),
    (0.032760604525487846, 0.011492632504703438),
    (0.006990483057170126, -0.012199936444957035),
    (-0.032938888188530926, 0.03087253175649998),
];

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        ..Default::default()
    }
}

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA)
}

fn kdense_cfg(precision: f64, mutation: Option<KMutation>) -> KDenseAftConfig {
    KDenseAftConfig {
        precision,
        mutation,
        ..Default::default()
    }
}

fn cmax(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
}

/// k-point RHF on prebuilt integrals with an explicit v_M.
fn krhf_try(
    cell: &Cell,
    mesh: &KPointMesh,
    hk: &PeriodicHcoreK,
    eri: &KDenseAftEri,
    madelung: f64,
) -> Result<KScfResult, ferric_core::FerricError> {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(eri.jk_builder_with_madelung(madelung)),
    };
    solve_krhf_injected(cell, mesh, &kscf_cfg(), inj)
}

fn krhf(
    cell: &Cell,
    mesh: &KPointMesh,
    hk: &PeriodicHcoreK,
    eri: &KDenseAftEri,
    madelung: f64,
) -> KScfResult {
    let r = krhf_try(cell, mesh, hk, eri, madelung).expect("k-point RHF");
    assert!(
        r.converged,
        "k-point RHF did not converge ({} it)",
        r.iterations
    );
    r
}

/// Mesh, per-k hcore and dense k-point kernels (exxdiv none; v_M passed
/// separately to the builder).
fn k_integrals(
    cell: &Cell,
    bs: &BasisSet,
    mesh: KPointMesh,
    precision: f64,
    mutation: Option<KMutation>,
) -> (KPointMesh, PeriodicHcoreK, KDenseAftEri) {
    let prep = prep_for(cell, bs);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcore_cfg()).expect("hcore(k)");
    let eri = KDenseAftEri::build(
        cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &kdense_cfg(precision, mutation),
    )
    .expect("dense k-point kernels");
    (mesh, hk, eri)
}

/// Explicit diag(n) supercell (atoms of cell m = (m0, m1, m2) at
/// `R + Σ m_i a_i`, m0 outer — the prototype's `supercell_cell`).
fn supercell(cell: &Cell, n: [usize; 3]) -> Cell {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    Cell::new(hydrogens(&atoms), lat).expect("supercell")
}

struct GammaRef {
    /// (E_none, E_ewald) per primitive cell.
    e: (f64, f64),
    /// Sorted supercell orbital energies (none, ewald).
    eps: (Vec<f64>, Vec<f64>),
    vm: f64,
}

fn gamma_supercell_reference(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    precision: f64,
) -> GammaRef {
    let nk = (n[0] * n[1] * n[2]) as f64;
    let sc = supercell(cell, n);
    let prep = prep_for(&sc, bs);
    let hc = periodic_hcore(&sc, &prep, &hcore_cfg()).expect("supercell hcore");
    let base = DenseAftEri::build(
        &sc,
        &prep,
        &hc.s,
        ExxDiv::None,
        precision,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("supercell dense AFT");
    let mut e = [0.0; 2];
    let mut eps: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (i, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let eri = base.clone().with_exxdiv(&sc, exx).unwrap();
        let r = gamma_rhf(&sc, &prep, &hc, &eri);
        e[i] = r.energy / nk;
        let mut v = r.eps_r().to_vec();
        v.sort_by(f64::total_cmp);
        eps[i] = v;
    }
    let [e0, e1] = e;
    let [p0, p1] = eps;
    GammaRef {
        e: (e0, e1),
        eps: (p0, p1),
        vm: madelung_constant(&sc).unwrap(),
    }
}

fn sorted_union(r: &KScfResult) -> Vec<f64> {
    let mut v: Vec<f64> = r.eps.iter().flatten().copied().collect();
    v.sort_by(f64::total_cmp);
    v
}

fn max_vec_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

// ============================================================== (a)

#[test]
fn one_point_mesh_is_the_gamma_driver() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let base = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let (mesh, hk, eri) = k_integrals(
        &cell,
        &bs,
        mesh,
        ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION,
        None,
    );

    // One-electron: Gamma phases are exactly 1, so S, T are bit-for-bit
    // the real sums; V differs only in the LR summation order.
    let n = prep.nbasis();
    let im_s = hk.s[0].iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert_eq!(im_s, 0.0, "S(Gamma) has an imaginary part");
    let ds = (0..n * n).fold(0.0_f64, |m, i| {
        m.max((hk.s[0][(i / n, i % n)].re - hc.s[(i / n, i % n)]).abs())
    });
    let dh = (0..n * n).fold(0.0_f64, |m, i| {
        let z = hk.h[0][(i / n, i % n)];
        m.max((z.re - hc.h[(i / n, i % n)]).abs()).max(z.im.abs())
    });
    eprintln!("1x1x1: |S - S_gamma| {ds:.2e}, |h - h_gamma| {dh:.2e}");
    assert!(ds <= 1e-15, "{ds:e}");
    assert!(dh < 1e-13, "{dh:e}");

    // Coulomb kernel at q = 0 == the Gamma dense tensor.
    let jk = eri.jker(0, 0);
    let dj = jk
        .iter()
        .zip(base.eri().iter())
        .fold(0.0_f64, |m, (z, x)| m.max((z.re - x).abs()).max(z.im.abs()));
    eprintln!("1x1x1: |Jker - I_gamma| {dj:.2e}");
    assert!(dj < 1e-13, "{dj:e}");
    // 1x1x1 supercell == the cell: same Madelung constant.
    let vm = madelung_constant(&cell).unwrap();
    assert!((eri.madelung_ewald() - vm).abs() < 1e-15);

    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let g = gamma_rhf(
            &cell,
            &prep,
            &hc,
            &base.clone().with_exxdiv(&cell, exx).unwrap(),
        );
        let vmk = if exx == ExxDiv::Ewald { vm } else { 0.0 };
        let k = krhf(&cell, &mesh, &hk, &eri, vmk);
        let de = k.energy - g.energy;
        let deps = max_vec_diff(&k.eps[0], g.eps_r());
        eprintln!(
            "1x1x1 {exx:?}: E_k {:.13} E_gamma {:.13} dE {de:.1e} d eps {deps:.1e}",
            k.energy, g.energy
        );
        assert!(de.abs() <= 1e-12, "{exx:?}: dE {de:e}");
        assert!(deps <= 1e-10, "{exx:?}: d eps {deps:e}");
    }
}

// ============================================================== (b)

/// k-mesh E/cell vs supercell Gamma / N, both exxdiv; mesh v_M vs supercell
/// v_M; none − ewald = nocc·v_M; sorted eigenvalue union.
fn supercell_anchor(name: &str, cell: &Cell, bs: &BasisSet, n: [usize; 3], tol: f64) {
    let reference = gamma_supercell_reference(cell, bs, n, ANCHOR_PRECISION);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let (mesh, hk, eri) = k_integrals(cell, bs, mesh, ANCHOR_PRECISION, None);
    let vm = eri.madelung_ewald();
    eprintln!(
        "{name} {n:?}: v_M mesh {vm:.14} supercell {:.14} ({} q passes, {} K)",
        reference.vm,
        eri.n_q_passes(),
        eri.n_k_total()
    );
    assert!(
        (vm - reference.vm).abs() < 1e-13,
        "v_M {vm} vs {}",
        reference.vm
    );
    let nocc = (cell.mol().nelec() / 2) as f64;
    let mut ek = [0.0; 2];
    for (i, (exx, e_ref, eps_ref)) in [
        (ExxDiv::None, reference.e.0, &reference.eps.0),
        (ExxDiv::Ewald, reference.e.1, &reference.eps.1),
    ]
    .into_iter()
    .enumerate()
    {
        let r = krhf(
            cell,
            &mesh,
            &hk,
            &eri,
            if exx == ExxDiv::Ewald { vm } else { 0.0 },
        );
        let de = r.energy - e_ref;
        let deps = max_vec_diff(&sorted_union(&r), eps_ref);
        eprintln!(
            "{name} {n:?} {exx:?}: E_k/cell {:.13} E_sc/N {e_ref:.13} dE {de:.1e} d(all eps) {deps:.1e} \
             ({} it, nocc/k {:?})",
            r.energy, r.iterations, r.nocc_per_k
        );
        assert!(de.abs() <= tol, "{name} {exx:?}: dE {de:e}");
        assert!(deps <= 1e-9, "{name} {exx:?}: d eps {deps:e}");
        ek[i] = r.energy;
    }
    // Identity: v_M S D S shifts every occupied level by −v_M at fixed D.
    let id = ek[0] - ek[1] - nocc * vm;
    assert!(id.abs() < 1e-11, "none − ewald − nocc v_M = {id:e}");
}

#[test]
fn h2_1x1x3_mesh_is_the_gamma_supercell() {
    supercell_anchor("H2", &h2_cell(4.0), &pyscf_sto3g_h(), [1, 1, 3], 1e-11);
}

#[test]
fn h2_2x2x2_mesh_is_the_gamma_supercell() {
    supercell_anchor("H2", &h2_cell(4.0), &pyscf_sto3g_h(), [2, 2, 2], 1e-11);
}

/// Triclinic s+p: the p functions reach the (−iK)^t branch of the residue
/// pair FT with a complex phase. Time unmeasured in Rust (module doc).
#[test]
#[ignore = "slow: triclinic s+p 1x1x3 k-mesh plus its 3-cell Gamma supercell (~70 s release); run with --release -- --ignored, serially; the fast triclinic_sp_1x1x2 PySCF pin covers the cell"]
fn triclinic_sp_1x1x3_mesh_is_the_gamma_supercell() {
    supercell_anchor(
        "tri s+p",
        &triclinic_cell(),
        &sp_basis_h(),
        [1, 1, 3],
        1e-11,
    );
}

// ============================================================== (c)

#[test]
fn bloch_convention_time_reversal_and_hermiticity() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    for (k, kz) in H2_KZ_113.iter().enumerate() {
        let p = mesh.kpts()[k];
        assert!(
            p[0].abs() < 1e-15 && p[1].abs() < 1e-15 && (p[2] - kz).abs() < 1e-14,
            "{p:?}"
        );
    }
    assert_eq!(mesh.minus(1), 2);
    assert!(mesh.is_trim(0) && !mesh.is_trim(1));
    assert_eq!(mesh.tr_representatives(), vec![0, 1]);

    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    // PySCF pin: the one test that sees a global k -> -k flip.
    let mut worst = 0.0_f64;
    let mut worst_conj = 0.0_f64;
    for k in 0..3 {
        for i in 0..4 {
            let (re, im) = H2_SK_113[k][i];
            let py = Complex64::new(re, im);
            let z = hk.s[k][(i / 2, i % 2)];
            worst = worst.max((z - py).norm());
            worst_conj = worst_conj.max((z - py.conj()).norm());
        }
    }
    eprintln!("H2 1x1x3 S(k) vs PySCF {worst:.1e} (vs conj PySCF {worst_conj:.2})");
    assert!(worst < 1e-10, "{worst:e}");
    assert!(
        worst_conj > 0.5,
        "the pin must be able to see the phase sign"
    );

    // Time reversal, exact for S (exact phases), roundoff for V (LR sum order).
    assert_eq!(
        hk.s[2],
        hk.s[1].mapv(|z| z.conj()),
        "S(-k) != S(k)* bitwise"
    );
    let dv = cmax(&hk.v[2], &hk.v[1].mapv(|z| z.conj()));
    assert!(dv < 1e-12, "V(-k) vs V(k)*: {dv:e}");

    // J, K Hermitian for a Hermitian, time-reversal-symmetric test density.
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &kdense_cfg(ANCHOR_PRECISION, None),
    )
    .unwrap();
    let n = prep.nbasis();
    let d1 = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex64::new(0.3, 0.0)
        } else if i < j {
            Complex64::new(0.1, 0.05)
        } else {
            Complex64::new(0.1, -0.05)
        }
    });
    let dm = vec![
        d1.mapv(|z| Complex64::new(z.re, 0.0)),
        d1.clone(),
        d1.mapv(|z| z.conj()),
    ];
    let mut j = vec![Array2::zeros((n, n)); 3];
    let mut kx = vec![Array2::zeros((n, n)); 3];
    eri.jk_builder().build(&dm, &mut j, &mut kx).unwrap();
    for k in 0..3 {
        let hj = cmax(&j[k], &j[k].t().mapv(|z| z.conj()));
        let hkx = cmax(&kx[k], &kx[k].t().mapv(|z| z.conj()));
        assert!(
            hj < 1e-13 && hkx < 1e-13,
            "k={k}: |J-J^H| {hj:e} |K-K^H| {hkx:e}"
        );
    }
    let tj = cmax(&j[2], &j[1].mapv(|z| z.conj()));
    let tk = cmax(&kx[2], &kx[1].mapv(|z| z.conj()));
    assert!(tj < 1e-13 && tk < 1e-13, "J/K time reversal {tj:e} {tk:e}");
}

#[test]
fn triclinic_sp_overlap_matches_pyscf_bloch_sums() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let mut worst = 0.0_f64;
    for r in 0..4 {
        for c in 0..16 {
            let (re, im) = TRI_SK1_ROWS[r * 16 + c];
            worst = worst.max((hk.s[1][(r, c)] - Complex64::new(re, im)).norm());
        }
    }
    eprintln!("tri s+p 1x1x3 S(k1) rows 0-3 vs PySCF {worst:.1e}");
    assert!(worst < 1e-10, "{worst:e}");
    assert_eq!(hk.s[2], hk.s[1].mapv(|z| z.conj()));
}

#[test]
// k-points are PySCF make_kpts output pasted verbatim, not approximations of pi/8.
#[allow(clippy::approx_constant)]
fn monkhorst_pack_points_and_mesh_madelung_match_pyscf() {
    let cell = h2_cell(4.0);
    // make_kpts([1,1,2], with_gamma_point=False): z = ∓π/8.
    let mp = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    let want = [-std::f64::consts::FRAC_PI_8, std::f64::consts::FRAC_PI_8];
    for (k, w) in want.iter().enumerate() {
        assert!((mp.kpts()[k][2] - w).abs() < 1e-14, "{:?}", mp.kpts()[k]);
    }
    assert_eq!(mp.minus(0), 1);
    assert_eq!(mp.residue_moduli(), [1, 1, 4]);
    // make_kpts([2,1,3], with_gamma_point=False): (x, z) order x outer.
    let mp = KPointMesh::monkhorst_pack(&cell, [2, 1, 3]).unwrap();
    let want: [[f64; 2]; 6] = [
        [-0.39269908169872414, -0.5235987755982988],
        [-0.39269908169872414, 0.0],
        [-0.39269908169872414, 0.5235987755982988],
        [0.39269908169872414, -0.5235987755982988],
        [0.39269908169872414, 0.0],
        [0.39269908169872414, 0.5235987755982988],
    ];
    for (k, w) in want.iter().enumerate() {
        let p = mp.kpts()[k];
        assert!(
            (p[0] - w[0]).abs() < 1e-14 && (p[2] - w[1]).abs() < 1e-14,
            "{k}: {p:?}"
        );
    }
    for k in 0..6 {
        let m = mp.minus(k);
        let (p, q) = (mp.kpts()[k], mp.kpts()[m]);
        assert!(m == k || ((p[0] + q[0]).abs() < 1e-14 && (p[2] + q[2]).abs() < 1e-14));
    }
    // Both centrings share the supercell Madelung constant.
    for mesh in [
        KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap(),
        KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap(),
    ] {
        let vm = mesh.madelung(&cell).unwrap();
        assert!(
            (vm - H2_MADELUNG_112).abs() < 1e-13,
            "{vm} vs {H2_MADELUNG_112}"
        );
    }
}

// ============================================================== (d)

fn pinned(
    name: &str,
    cell: &Cell,
    bs: &BasisSet,
    mesh: KPointMesh,
    pins: &[(ExxDiv, f64)],
    tol: f64,
) {
    let (mesh, hk, eri) = k_integrals(
        cell,
        bs,
        mesh,
        ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_PRECISION,
        None,
    );
    for &(exx, e_ref) in pins {
        let vm = if exx == ExxDiv::Ewald {
            eri.madelung_ewald()
        } else {
            0.0
        };
        let r = krhf(cell, &mesh, &hk, &eri, vm);
        eprintln!(
            "{name} {exx:?}: E {:.12} PySCF {e_ref:.12} dE {:.1e} ({} it, nocc/k {:?}, gap {:.4})",
            r.energy,
            r.energy - e_ref,
            r.iterations,
            r.nocc_per_k,
            r.lumo - r.homo
        );
        assert!(
            (r.energy - e_ref).abs() < tol,
            "{name} {exx:?}: {}",
            r.energy
        );
    }
}

#[test]
fn h2_1x1x2_matches_pinned_pyscf_krhf_aftdf() {
    let cell = h2_cell(4.0);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    pinned(
        "H2 1x1x2",
        &cell,
        &pyscf_sto3g_h(),
        mesh,
        &[(ExxDiv::None, H2_112.0), (ExxDiv::Ewald, H2_112.1)],
        1e-9,
    );
}

#[test]
fn h2_2x2x2_matches_pinned_pyscf_krhf_aftdf() {
    let cell = h2_cell(4.0);
    let mesh = KPointMesh::gamma_centred(&cell, [2, 2, 2]).unwrap();
    pinned(
        "H2 2x2x2",
        &cell,
        &pyscf_sto3g_h(),
        mesh,
        &[(ExxDiv::None, H2_222.0), (ExxDiv::Ewald, H2_222.1)],
        1e-9,
    );
}

/// Shifted mesh: no supercell anchor exists (twisted boundary), so this
/// PySCF pin is its only energy check. Exercises the 2N residue modulus.
#[test]
fn h2_shifted_mp_1x1x2_matches_pinned_pyscf_krhf_aftdf() {
    let cell = h2_cell(4.0);
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    pinned(
        "H2 MP 1x1x2",
        &cell,
        &pyscf_sto3g_h(),
        mesh,
        &[(ExxDiv::None, H2_MP112.0), (ExxDiv::Ewald, H2_MP112.1)],
        1e-9,
    );
}

/// Time unmeasured in Rust (module doc).
#[test]
fn triclinic_sp_1x1x2_matches_pinned_pyscf_krhf_aftdf() {
    let cell = triclinic_cell();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    pinned(
        "tri s+p 1x1x2",
        &cell,
        &sp_basis_h(),
        mesh,
        &[(ExxDiv::None, TRI_112.0), (ExxDiv::Ewald, TRI_112.1)],
        1e-9,
    );
}

// ============================================================== (e)

/// A mutant must move the H2 1×1×3 energy off the supercell reference by
/// more than `min_err` (or make the SCF refuse, which also counts as caught).
fn mutant_is_caught(mutation: KMutation, exx: ExxDiv, min_err: f64) -> f64 {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let reference = gamma_supercell_reference(&cell, &bs, [1, 1, 3], ANCHOR_PRECISION);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let (mesh, hk, eri) = k_integrals(&cell, &bs, mesh, ANCHOR_PRECISION, Some(mutation));
    let (vm, e_ref) = match exx {
        ExxDiv::None => (0.0, reference.e.0),
        ExxDiv::Ewald => (eri.madelung_ewald(), reference.e.1),
    };
    match krhf_try(&cell, &mesh, &hk, &eri, vm) {
        Ok(r) => {
            let de = r.energy - e_ref;
            eprintln!(
                "mutant {mutation:?} {exx:?}: dE {de:+.4} (converged {})",
                r.converged
            );
            assert!(
                de.abs() > min_err,
                "mutant {mutation:?} survived: dE {de:e}"
            );
            de
        }
        Err(e) => {
            eprintln!("mutant {mutation:?} {exx:?}: SCF refused ({e}) — caught");
            f64::INFINITY
        }
    }
}

#[test]
fn mutant_phase_sign_in_pair_ft_is_caught() {
    mutant_is_caught(KMutation::PhaseSignInPairFt, ExxDiv::None, 1e-3);
}

#[test]
fn mutant_kernel_without_q_shift_is_caught() {
    mutant_is_caught(KMutation::KernelAtG, ExxDiv::None, 1e-3);
    mutant_is_caught(KMutation::KernelAtG, ExxDiv::Ewald, 1e-3);
}

#[test]
fn mutant_primitive_madelung_is_caught_and_leaves_exxdiv_none_exact() {
    mutant_is_caught(KMutation::PrimitiveMadelung, ExxDiv::Ewald, 1e-3);
    // exxdiv = none never reads v_M: the mutant must be invisible there.
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let reference = gamma_supercell_reference(&cell, &bs, [1, 1, 3], ANCHOR_PRECISION);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let (mesh, hk, eri) = k_integrals(
        &cell,
        &bs,
        mesh,
        ANCHOR_PRECISION,
        Some(KMutation::PrimitiveMadelung),
    );
    let r = krhf(&cell, &mesh, &hk, &eri, 0.0);
    assert!(
        (r.energy - reference.e.0).abs() < 1e-11,
        "{}",
        r.energy - reference.e.0
    );
}

#[test]
fn dense_k_kernels_refuse_an_oversize_mesh() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let s: Vec<Array2<Complex64>> = (0..2).map(|_| Array2::eye(2)).collect();
    let need = 2 * 4 * 16 * 16; // 2 N_k² nao⁴ × 16
    let cfg = KDenseAftConfig {
        max_bytes: need - 1,
        ..kdense_cfg(1e-4, None)
    };
    let msg = KDenseAftEri::build(&cell, &prep, &mesh, &s, ExxDiv::None, &cfg)
        .expect_err("cap must be enforced")
        .to_string();
    assert!(
        msg.contains(&format!("{need} bytes")) && msg.contains("cap"),
        "{msg}"
    );
}
