//! Stage 9: k-point UHF (`ferric_pbc::kuscf::solve_kuhf[_injected]`), ported
//! from `reference/pbc/pbc_kuhf.py` and its tests at the end of
//! `test_prototype.py` (FINDINGS "Iteration 13 (Python, k-point UHF)").
//!
//! (a) Closed shell kUHF ≡ kRHF (H2/STO-3G a = 4, 1×1×3), both exxdiv,
//!     ≤ 1e-12 (prototype 4.4e-16), `⟨S²⟩` 0. Mutants caught here:
//!     `KFromTotalDensity` (K doubled, |ΔE| > 1e-2) and
//!     `MadelungHalfPerSpin` (exactly `+v_M N/4` under ewald, 0 under none).
//! (b) 1×1×1 ≡ Gamma `gamma_uhf` (tri 4H/STO-3G triplet), both exxdiv,
//!     ≤ 1e-12 (prototype 5.9e-15), `⟨S²⟩` ≤ 1e-9 — an independent
//!     construction (real Gamma path through `ferric_scf::uhf`).
//! (c) 1×1×3 E/cell ≡ Gamma UHF of the explicit supercell / 3 (tri triplet
//!     per cell, supercell (9, 3)), both exxdiv, ≤ 1e-11 (prototype
//!     −9.8e-13), giant-determinant `⟨S²⟩` ≤ 1e-9, per-k counts [3,3,3] /
//!     [1,1,1]. The supercell starts from its own core guess (staged): the
//!     prototype found no lower translation-broken state from core / beta mix
//!     / random-rotation guesses. `#[ignore]`d as slow (12-atom supercell SR
//!     nuclear sum; the analogous k-RHF triclinic 1×1×3 anchor is ~70 s).
//! (d) Trivial-aux k RS-GDF ≡ dense, open shell (anchor H2 one-s/H, 1×1×3;
//!     doublet per (+1) cell and triplet per cell), both exxdiv, ≤ 1e-11
//!     (prototype −9.4e-14 / −7.6e-13).
//! (e) PySCF 2.13 KUHF + AFTDF pins (mesh 61³, precision 1e-12): H atom
//!     1×1×2 and 1×1×3, H2 triplet 1×1×2, energies (1e-9) and `⟨S²⟩`
//!     (giant determinant = S_z(S_z+1) for N_β = 0: 2, 3.75, 6). PySCF was
//!     told `mf.nelec = (N_α N_k, N_β N_k)` because its `cell.spin` counts
//!     the WHOLE mesh; ferric's `N_α`, `N_β` are per cell. Plus the staged
//!     identity `E_ewald − E_none = −v_M (N_α + N_β)/2`. These have N_β = 0
//!     so they cannot see `KFromTotalDensity` ((a) does); they DO see the
//!     per-spin Madelung factor.
//! (f) zchain (H2 bond 2.0 along x, stacked every 2.0 Bohr along z, +1
//!     cell, doublet), 1×1×2: GLOBAL aufbau puts both α electrons at Gamma
//!     (per-k [2, 0]); ≡ supercell Gamma UHF / 2 (prototype 1.3e-14,
//!     asserted 1e-11) and the `PerKAufbau` mutant ([1, 1]) lands > 0.2 Ha
//!     high (prototype +0.2412) — the ONLY test that sees per-k aufbau
//!     (uniform-occupation systems are blind to it).
//! (g) The Ewald trap at 1×1×2 (tri 4H s+p triplet, dense AFT precision
//!     1e-8): a single-stage ewald SCF from the core guess lands ~93 mHa
//!     high (prototype −1.7207518511) and the per-spin gap check flags it;
//!     the staged default reaches −1.8138633926 with the check satisfied.
//!
//! # Mutation plan (each must fail ≥ 1 test)
//!
//! * K from `D_α + D_β` (`KUhfMutation::KFromTotalDensity`, in-test) → (a).
//! * Madelung `v_M/2` per spin (`MadelungHalfPerSpin`, in-test) → (a) ewald,
//!   (e) ewald pins.
//! * Per-k aufbau (`PerKAufbau`, in-test) → (f).
//! * Production edits for the main agent: in `run_kuhf`, pass `0.5 *` the
//!   energy of one spin, or build `J = J_α` only → (a)/(b)/(e); in
//!   `solve_kuhf`, pass `k_madelung` to the first stage → (g) staged fails
//!   (it becomes direct); `kuhf_gap_report` margin `g − shift` → `g` → (g)
//!   flag fails; `s2` without the overlap term → (a) (⟨S²⟩ ≠ 0).

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf, KPointInjection, KPointJk, KRhfConfig, KScfConfig};
use ferric_pbc::kuscf::{
    solve_kuhf, solve_kuhf_injected_with_guess, KUScfResult, KUhfConfig, KUhfMutation, KUhfResult,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::{gamma_uhf, EwaldStart, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ndarray::Array2;
use num_complex::Complex64;

/// Nuclear-attraction Ewald split (as `pbc_krhf.rs`).
const OMEGA: f64 = 0.8;
/// Loose AFT precision for the supercell anchors (exact at any precision).
const ANCHOR_PRECISION: f64 = 1e-6;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

// ---- (e) PySCF KUHF + AFTDF pins (run_kuhf_oracle.py): (none, ewald) ----
const H_ATOM: [[f64; 3]; 1] = [[0.3, 0.2, 0.1]];
const H1X2: (f64, f64) = (-0.399399818915, -0.625130045222);
const H1X3: (f64, f64) = (-0.528717460735, -0.623551487522);
const H2T_1X2: (f64, f64) = (-0.200418842427, -0.651879295040);
const PIN_TOL: f64 = 1e-9;

// ---- (g) run_kuhf_trap.py, tri 4H s+p triplet 1x1x2, gcut prec 1e-8 -----
const TRAP_112_DIRECT: f64 = -1.7207518511;
const TRAP_112_STAGED: f64 = -1.8138633926;

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

fn cell_cm(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], charge: i32, mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.charge = charge;
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn kucfg(
    cell: &Cell,
    exx: ExxDiv,
    start: EwaldStart,
    precision: f64,
    mutation: Option<KUhfMutation>,
) -> KUhfConfig {
    let mut c = KUhfConfig::for_cell(cell, exx);
    c.scf = kscf_cfg();
    c.hcore = hcore_cfg();
    c.ewald_start = start;
    c.dense = KDenseAftConfig {
        precision,
        ..Default::default()
    };
    c.mutation = mutation;
    c
}

fn kuhf(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    exx: ExxDiv,
    start: EwaldStart,
    precision: f64,
    mutation: Option<KUhfMutation>,
) -> KUhfResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let r = solve_kuhf(
        cell,
        &prep,
        None,
        &mesh,
        &kucfg(cell, exx, start, precision, mutation),
    )
    .expect("k-point UHF");
    let s = &r.scf;
    eprintln!(
        "  kUHF {n:?} {exx:?} {start:?} mut {mutation:?}: E/cell {:.13} <S2> {:.10} ({} it) \
         nocc a/b per k {:?} {:?} gaps {:?} v_M {:.10}",
        s.energy, s.s2, s.iterations, s.nocc_per_k_alpha, s.nocc_per_k_beta, r.gaps, r.madelung
    );
    assert!(s.converged);
    r
}

/// Gamma UHF (staged) of `cell` on the dense Gamma AFT tensor.
fn gamma(cell: &Cell, bs: &BasisSet, exx: ExxDiv, precision: f64) -> GammaUhfResult {
    let prep = prep_for(cell, bs);
    let hc = periodic_hcore(cell, &prep, &hcore_cfg()).expect("gamma hcore");
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        precision,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("gamma dense AFT");
    let cfg = GammaUhfConfig {
        exxdiv: exx,
        ewald_start: EwaldStart::Staged,
        ..Default::default()
    };
    let r =
        gamma_uhf(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg).expect("gamma UHF");
    assert!(r.scf.converged);
    r
}

/// Explicit diag(n) supercell with charge / multiplicity for the whole
/// supercell (atoms of cell m at `R + Σ m_i a_i`, m0 outer).
fn supercell(cell: &Cell, n: [usize; 3], charge: i32, mult: usize) -> Cell {
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
    cell_cm(&atoms, lat, charge, mult)
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.13} (want {want:.13}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.13e} vs {want:.13e} (tol {tol:.1e})"
    );
}

// ============================================================== (a)

#[test]
fn closed_shell_kuhf_equals_krhf_and_catches_spin_mutants() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let n = [1, 1, 3];
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    for exx in EXX {
        let mut rc = KRhfConfig::for_cell(&cell, exx);
        rc.scf = kscf_cfg();
        rc.hcore = hcore_cfg();
        let r = solve_krhf(&cell, &prep, None, &mesh, &rc).expect("kRHF");
        assert!(r.converged);
        let u = kuhf(
            &cell,
            &bs,
            n,
            exx,
            EwaldStart::Staged,
            DEFAULT_DENSE_AFT_PRECISION,
            None,
        );
        close(
            u.scf.energy,
            r.energy,
            1e-12,
            &format!("{exx:?} kUHF vs kRHF"),
        );
        assert!(u.scf.s2.abs() < 1e-10, "closed-shell <S2> {}", u.scf.s2);
        let dd = u
            .scf
            .density_alpha
            .iter()
            .zip(&u.scf.density_beta)
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).norm()))
            .fold(0.0_f64, f64::max);
        assert!(dd < 1e-8, "{exx:?}: |D_a - D_b| {dd:e}");

        // Mutant: K from D_a + D_b (K doubled for a closed shell). A failed
        // or unconverged mutant SCF also counts as caught.
        let m = solve_kuhf(
            &cell,
            &prep,
            None,
            &mesh,
            &kucfg(
                &cell,
                exx,
                EwaldStart::Staged,
                DEFAULT_DENSE_AFT_PRECISION,
                Some(KUhfMutation::KFromTotalDensity),
            ),
        );
        match m {
            Ok(m) => {
                let d = m.scf.energy - r.energy;
                eprintln!("  MUTANT K[D_total] {exx:?}: dE {d:+.3e}");
                assert!(d.abs() > 1e-2, "K[D_total] mutant not caught: {d:e}");
            }
            Err(e) => eprintln!("  MUTANT K[D_total] {exx:?}: SCF failed ({e}) -- caught"),
        }

        // Mutant: v_M/2 per spin — exactly +v_M N/4 (N = 2) under ewald.
        let m = kuhf(
            &cell,
            &bs,
            n,
            exx,
            EwaldStart::Staged,
            DEFAULT_DENSE_AFT_PRECISION,
            Some(KUhfMutation::MadelungHalfPerSpin),
        );
        let d = m.scf.energy - r.energy;
        let want = if exx == ExxDiv::Ewald {
            u.madelung * 2.0 / 4.0
        } else {
            0.0
        };
        eprintln!("  MUTANT v_M/2 per spin {exx:?}: dE {d:+.3e} (predicted {want:+.3e})");
        close(d, want, 1e-9, &format!("{exx:?} v_M/2 mutant shift"));
        if exx == ExxDiv::Ewald {
            // v_M*N/4 = 0.0948 here (N = 2); the exact prediction is checked above.
            assert!(d > 0.05, "the Madelung mutant must be visible under ewald");
        }
    }
}

// ============================================================== (b)

#[test]
fn one_point_mesh_is_the_gamma_uhf() {
    let cell = cell_cm(&TRI_ATOMS, TRI_A, 0, 3);
    let bs = pyscf_sto3g_h();
    for exx in EXX {
        let g = gamma(&cell, &bs, exx, DEFAULT_DENSE_AFT_PRECISION);
        let k = kuhf(
            &cell,
            &bs,
            [1, 1, 1],
            exx,
            EwaldStart::Staged,
            DEFAULT_DENSE_AFT_PRECISION,
            None,
        );
        close(
            k.scf.energy,
            g.scf.energy,
            1e-12,
            &format!("{exx:?} 1x1x1 E"),
        );
        close(k.scf.s2, g.s2, 1e-9, &format!("{exx:?} 1x1x1 <S2>"));
        assert!((k.madelung - g.madelung).abs() < 1e-14);
    }
}

// ============================================================== (c)

fn supercell_anchor(
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    sc_charge: i32,
    sc_mult: usize,
    tol: f64,
) -> [KUhfResult; 2] {
    let nk = (n[0] * n[1] * n[2]) as f64;
    let sc = supercell(cell, n, sc_charge, sc_mult);
    let mut out = Vec::new();
    for exx in EXX {
        let k = kuhf(cell, bs, n, exx, EwaldStart::Staged, ANCHOR_PRECISION, None);
        let g = gamma(&sc, bs, exx, ANCHOR_PRECISION);
        assert!(
            (k.madelung - g.madelung).abs() < 1e-13,
            "mesh v_M {} vs supercell {}",
            k.madelung,
            g.madelung
        );
        close(
            k.scf.energy,
            g.scf.energy / nk,
            tol,
            &format!("{exx:?} {n:?} kUHF/cell vs supercell/N"),
        );
        close(k.scf.s2, g.s2, 1e-9, &format!("{exx:?} {n:?} giant <S2>"));
        out.push(k);
    }
    let [a, b]: [KUhfResult; 2] = out.try_into().ok().unwrap();
    // Staged ewald − none = −v_M (N_α + N_β)/2 per cell.
    let ne = (b.nocc.0 + b.nocc.1) as f64;
    close(
        b.scf.energy - a.scf.energy,
        -b.madelung * ne / 2.0,
        1e-10,
        "ewald - none",
    );
    [a, b]
}

#[test]
#[ignore = "slow: tri 1x1x3 k-mesh plus the 12-atom Gamma supercell UHF (unmeasured in Rust; the k-RHF triclinic 1x1x3 analogue is ~70 s release); run with --release -- --ignored, serially"]
fn tri_triplet_1x1x3_is_the_gamma_supercell_uhf() {
    let cell = cell_cm(&TRI_ATOMS, TRI_A, 0, 3);
    // Supercell: N_α 9, N_β 3 -> 2S = 6.
    let [none, ewald] = supercell_anchor(&cell, &pyscf_sto3g_h(), [1, 1, 3], 0, 7, 1e-11);
    for r in [&none, &ewald] {
        assert_eq!(r.scf.nocc_per_k_alpha, vec![3, 3, 3]);
        assert_eq!(r.scf.nocc_per_k_beta, vec![1, 1, 1]);
    }
}

// ============================================================== (f)

fn zchain() -> Cell {
    cell_cm(
        &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]],
        1,
        2,
    )
}

#[test]
fn zchain_nonuniform_occupation_is_the_supercell_and_per_k_aufbau_is_caught() {
    let cell = zchain();
    let bs = pyscf_sto3g_h();
    // Supercell (1x1x2): charge +2, N_α 2, N_β 0 -> triplet.
    let [none, ewald] = supercell_anchor(&cell, &bs, [1, 1, 2], 2, 3, 1e-11);
    for (exx, r) in EXX.into_iter().zip([&none, &ewald]) {
        assert_eq!(
            r.scf.nocc_per_k_alpha,
            vec![2, 0],
            "{exx:?}: global aufbau must put both alpha electrons at Gamma"
        );
        assert_eq!(r.scf.nocc_per_k_beta, vec![0, 0]);
        let m = kuhf(
            &cell,
            &bs,
            [1, 1, 2],
            exx,
            EwaldStart::Staged,
            ANCHOR_PRECISION,
            Some(KUhfMutation::PerKAufbau),
        );
        let d = m.scf.energy - r.scf.energy;
        eprintln!(
            "  MUTANT per-k aufbau {exx:?}: dE {d:+.4} (prototype +0.2412), nocc_a/k {:?}",
            m.scf.nocc_per_k_alpha
        );
        assert_eq!(m.scf.nocc_per_k_alpha, vec![1, 1]);
        assert!(d > 0.2, "per-k aufbau mutant not caught: {d:e}");
    }
}

// ============================================================== (d)

const ANCHOR_ALPHA: f64 = 0.5;

/// The 24 anchor aux centres (copy of `pbc_krsgdf.rs::anchor_sites`).
fn anchor_sites(cell: &Cell) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
        }
    }
    out
}

type Guess<'a> = Option<(&'a [Array2<Complex64>], &'a [Array2<Complex64>])>;

#[allow(clippy::too_many_arguments)]
fn run_injected(
    cell: &Cell,
    mesh: &KPointMesh,
    hk: &PeriodicHcoreK,
    jk: Box<dyn KPointJk + '_>,
    na: usize,
    nb: usize,
    guess: Guess<'_>,
    label: &str,
) -> KUScfResult {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    };
    let r = solve_kuhf_injected_with_guess(cell, mesh, &kscf_cfg(), inj, na, nb, guess)
        .expect("injected kUHF");
    assert!(r.converged, "{label}: not converged");
    r
}

#[test]
fn trivial_aux_krsgdf_equals_dense_open_shell() {
    for (charge, mult, na, nb, label) in [
        (1, 2, 1usize, 0usize, "doublet (+1)"),
        (0, 3, 2, 0, "triplet"),
    ] {
        let cell = cell_cm(&H2_ATOMS, cubic(4.0), charge, mult);
        let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
        let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
        let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
        let eri = KDenseAftEri::build(
            &cell,
            &prep,
            &mesh,
            &hk.s,
            ExxDiv::None,
            &KDenseAftConfig::default(),
        )
        .expect("dense k kernels");
        let site = SiteBasis::new(&anchor_sites(&cell), 0).unwrap();
        let gdf = KRsGdf::build(
            &cell,
            &prep,
            &site.prep,
            &mesh,
            &hk.s,
            &KRsGdfConfig {
                gdf: RsGdfConfig {
                    omega: OMEGA,
                    exxdiv: ExxDiv::None,
                    budget_bytes: Some(1 << 30),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("k RS-GDF");
        let vm = eri.madelung_ewald();
        // Dense staged (none, then ewald from it); fit from the dense state.
        let d0 = run_injected(
            &cell,
            &mesh,
            &hk,
            Box::new(eri.jk_builder_with_madelung(0.0)),
            na,
            nb,
            None,
            label,
        );
        let de = run_injected(
            &cell,
            &mesh,
            &hk,
            Box::new(eri.jk_builder_with_madelung(vm)),
            na,
            nb,
            Some((&d0.density_alpha[..], &d0.density_beta[..])),
            label,
        );
        for (v, dense) in [(0.0, &d0), (vm, &de)] {
            let fit = run_injected(
                &cell,
                &mesh,
                &hk,
                Box::new(gdf.jk_builder_with_madelung(v)),
                na,
                nb,
                Some((&dense.density_alpha[..], &dense.density_beta[..])),
                label,
            );
            eprintln!(
                "  anchor H2 1x1x3 {label} v_M {v:.6}: nocc_a/k {:?} gap_a {:?}",
                dense.nocc_per_k_alpha, dense.gap_alpha
            );
            close(
                fit.energy,
                dense.energy,
                1e-11,
                &format!("{label} v_M {v:.4} k RS-GDF vs dense"),
            );
        }
    }
}

// ============================================================== (e)

fn pinned(atoms: &[[f64; 3]], mult: usize, n: [usize; 3], pins: (f64, f64), s2: f64, name: &str) {
    let cell = cell_cm(atoms, cubic(4.0), 0, mult);
    let bs = pyscf_sto3g_h();
    let mut e = [0.0; 2];
    for (i, (exx, want)) in [(ExxDiv::None, pins.0), (ExxDiv::Ewald, pins.1)]
        .into_iter()
        .enumerate()
    {
        let r = kuhf(
            &cell,
            &bs,
            n,
            exx,
            EwaldStart::Staged,
            DEFAULT_DENSE_AFT_PRECISION,
            None,
        );
        close(
            r.scf.energy,
            want,
            PIN_TOL,
            &format!("{name} {exx:?} vs PySCF KUHF"),
        );
        close(r.scf.s2, s2, 1e-12, &format!("{name} {exx:?} giant <S2>"));
        e[i] = r.scf.energy;
        if exx == ExxDiv::Ewald {
            let ne = (r.nocc.0 + r.nocc.1) as f64;
            close(
                e[1] - e[0],
                -r.madelung * ne / 2.0,
                1e-10,
                &format!("{name} ewald - none"),
            );
        }
    }
}

#[test]
fn h_atom_1x1x2_matches_pinned_pyscf_kuhf_aftdf() {
    pinned(&H_ATOM, 2, [1, 1, 2], H1X2, 2.0, "H atom 1x1x2");
}

#[test]
fn h_atom_1x1x3_matches_pinned_pyscf_kuhf_aftdf() {
    pinned(&H_ATOM, 2, [1, 1, 3], H1X3, 3.75, "H atom 1x1x3");
}

#[test]
fn h2_triplet_1x1x2_matches_pinned_pyscf_kuhf_aftdf() {
    pinned(&H2_ATOMS, 3, [1, 1, 2], H2T_1X2, 6.0, "H2 triplet 1x1x2");
}

// ============================================================== (g)

#[test]
fn ewald_trap_at_1x1x2_direct_lands_high_and_is_flagged_staged_reaches_ground_state() {
    let cell = cell_cm(&TRI_ATOMS, TRI_A, 0, 3);
    let bs = sp_basis_h();
    let n = [1, 1, 2];
    let trapped = kuhf(&cell, &bs, n, ExxDiv::Ewald, EwaldStart::Direct, 1e-8, None);
    let staged = kuhf(&cell, &bs, n, ExxDiv::Ewald, EwaldStart::Staged, 1e-8, None);
    let vm = staged.madelung;
    eprintln!(
        "  trap 1x1x2 (v_M {vm:.6}, prototype 0.369735): direct E {:.10} (prototype {TRAP_112_DIRECT}) \
         gaps {:?}; staged E {:.10} gaps {:?}; direct - staged {:+.4e} (prototype +9.311e-2)",
        trapped.scf.energy,
        trapped.gaps,
        staged.scf.energy,
        staged.gaps,
        trapped.scf.energy - staged.scf.energy
    );
    close(staged.scf.energy, TRAP_112_STAGED, 1e-6, "staged ewald E");
    assert!(staged.gaps.satisfied(), "staged: {:?}", staged.gaps);
    assert!(
        trapped.scf.energy - staged.scf.energy > 5e-2,
        "the single-stage run must land ~93 mHa high (prototype), got {:+e}",
        trapped.scf.energy - staged.scf.energy
    );
    assert!(
        !trapped.gaps.satisfied(),
        "the trapped state must be flagged: {:?}",
        trapped.gaps
    );
    // The none stage and the staged ewald share the density: −v_M N/2.
    let first = staged
        .none_stage
        .as_ref()
        .expect("staged runs a none stage");
    close(
        staged.scf.energy - first.energy,
        -vm * 4.0 / 2.0,
        1e-10,
        "staged ewald - none",
    );
}
