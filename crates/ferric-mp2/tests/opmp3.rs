//! OP-MP3 (`ferric_mp2::opmp3`, `opmp3_reference`, `opmp3_driver`) anchors.
//!
//! # What is compared
//!
//! 1. EXACTNESS FIRST, the series limit: on an RHF H2O reference the traces
//!    `Tr W^(2) + Tr W^(3)` (every model with the resummation switched off)
//!    equal MP2 + MP3 of the closed-shell spin-orbital `mp3.rs` (RI), an
//!    independent contraction code; and on H2O RHF and OH UHF the traces of
//!    every channel equal the `u_mp3` channel energies (`Tr W^SS = E_αα +
//!    E_ββ`, `Tr W^OS = E_αβ`, Eq. 11b).
//! 2. The Padé's Taylor coefficients: `f(x) = OP-MP3(x)` has `f(0) = E2`,
//!    `f'(0) = E3`, `½ f''(0) = Σ_σ Tr W3 W2⁻¹ W3` (central differences).
//! 3. Eq. 18: on OH UHF, the c and ⟨‖⟩ blocks expanded to full spin-orbital
//!    tensors give Eq. 11a by brute force; its k-spin split equals the
//!    spin-blocked SS / OS channels, and its αβ block is zero.
//! 4. Size consistency: N2 ⋯ CH4 at 100 Å (cc-pVDZ, exact): every model
//!    is additive, the scalar Padé is not.
//! 5. The reference path: for an RHF, UHF and ROHF reference, `E_HF[ρ]`
//!    rebuilt from the density equals the SCF energy, `E_NBS` vanishes for
//!    RHF and UHF and not for ROHF, and the ROHF E2/E3 channels equal PySCF's
//!    (`testdata/reference/ump3/oh_rohf_cc-pvdz.json`, numpy
//!    semicanonicalization on the UHF Fock of the ROHF density).
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! * Right: items 1 and 3 at the rounding floor, item 2 at its truncation
//!   bar, item 4 at the SCF floor, the scalar Padé off by ~1e-3 Ha.
//! * A wrong OS prefactor (⅛ for ¼): item 1 OS traces miss by a factor 2,
//!   item 3 OS blocks miss.
//! * αβ amplitude block read with the wrong index order for W^OS_β: OH
//!   item 3 β block misses (closed shell cannot see it).
//! * Padé numerator and denominator swapped: item 2 `f'(0)` misses.
//! * Scalar Padé per block: item 4 fails.
//! * A KS orbital energy leaking into the denominators, or a wrong
//!   semicanonical rotation: the ROHF rows of item 5 miss in E2.
//!
//! # TOLERANCES (derived)
//!
//! * [`TOL_ROUNDING`] 1e-12 Ha: the same products summed in another order,
//!   ~n_terms·ε·|E| ≈ 1e5·1.1e-16·0.3 = 3e-12 at worst for these sizes.
//!   Item 3 uses 1e3·ε·max|W| per element.
//! * [`TOL_MP3_RS`] 1e-10 Ha: `u_mp3` vs `mp3.rs` on the same RI integrals
//!   (the Lane A bar, three decades above the 2e-13 floor).
//! * Item 2: central difference at h = 1e-3, truncation `h²|c3|` for f'
//!   and `2h²|c4|·½` for ½f'', times 10, plus rounding `100 ε |E2| / h^k`.
//! * Item 4 [`TOL_SIZE`] 1e-9 Ha: the SCF density threshold 1e-10 per
//!   fragment moves E_c to first order (|E_c| ~ 0.6 Ha → 6e-11, ×10), and the
//!   N2 quadrupole × CH4 octupole at 189 Bohr is ~1e-13 Ha. The scalar Padé
//!   must miss by more than 1e-6.
//! * Item 5: E_HF[ρ] vs the SCF energy is stationary, so 10·energy_conv
//!   = 1e-11; E_NBS of RHF/UHF is quadratic in the orbital gradient
//!   (≲ g²/gap ≈ 1e-16), bar 1e-12. vs PySCF: the Lane A derivation,
//!   10·|E2 + E3|·(1e-9 + 1e-10) (PySCF conv_tol_grad and ferric
//!   density_conv).
//!
//! # MUTATIONS (run 2026-11-15; each failed on an assertion)
//!
//! `crates/ferric-mp2/src/opmp3.rs`:
//! * `w_channels`: OS prefactor `0.25` → `0.125` (both spins): the series
//!   limit and the Eq. 18 split fail.
//! * `w_channels`: β OS from `beta_first(..)` → the unpermuted αβ blocks: the
//!   Eq. 18 split fails.
//! * `pade`: numerator `ak.dot(&v)` → `(A − B)·v`: the Taylor-coefficient
//!   test fails.
//! * `sop_mp3` → per-block scalar Padé: the size-consistency test fails.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mp3::mp3_energy;
use ferric_mp2::opmp3::{op_mp3, w_matrices, Opmp3Models, WMatrices};
use ferric_mp2::opmp3_driver::{
    opmp3_on_reference, run_opmp3, Opmp3Config, Opmp3Integrals, Opmp3RefKind,
};
use ferric_mp2::opmp3_reference::opmp3_reference;
use ferric_mp2::u_mp3::{u_mp3, UMp3Config, UMp3Eri, UMp3Result};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{PreparedSystem, ScfResult};
use ndarray::{Array2, Array4};
use ndarray_linalg::Solve;
use serde_json::Value;

const TOL_ROUNDING: f64 = 1e-12;
const TOL_MP3_RS: f64 = 1e-10;
const TOL_SIZE: f64 = 1e-9;
const EPS: f64 = f64::EPSILON;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn check(ctx: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.1e})");
    assert!(
        d < tol,
        "{ctx}: {got:.12} vs {want:.12}: |d| {d:.2e} >= {tol:.1e}"
    );
}

fn tight() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    }
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    aux: PreparedBasis,
    bounds: SchwarzBounds,
    scf: ScfResult,
}

fn system(name: &str, scf_kind: &str) -> Sys {
    let (charge, mult) = match name {
        "h2o" => (0, 1),
        "oh" => (0, 2),
        _ => panic!("unknown system {name}"),
    };
    let xyz = workspace_root().join(format!("testdata/molecules/validation/{name}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf = match scf_kind {
        "rhf" => solve_rhf(&ctx, &mol, &obs, op, &bounds, &tight()),
        "uhf" => solve_uhf(
            &ctx,
            &mol,
            &obs,
            &bounds,
            &RhfConfig {
                check_stability: true,
                scf_stability_descent: true,
                ..tight()
            },
        ),
        "rohf" => solve_rohf(&ctx, &mol, &obs, op, &bounds, &tight()),
        _ => panic!("unknown reference {scf_kind}"),
    }
    .unwrap_or_else(|e| panic!("{name} {scf_kind}: {e:?}"));
    assert!(scf.converged, "{name} {scf_kind}: SCF not converged");
    Sys {
        mol,
        obs,
        aux,
        bounds,
        scf,
    }
}

fn ri_u_mp3(sys: &Sys) -> UMp3Result {
    u_mp3(
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.scf,
        UMp3Eri::Ri { aux: &sys.aux },
        &UMp3Config::default(),
    )
    .expect("u_mp3")
}

fn traces_equal_channel_energies(label: &str, r: &UMp3Result, w: &WMatrices) {
    for (n, wn, e) in [(2, &w.w2, &r.e2), (3, &w.w3, &r.e3)] {
        let tr_ss: f64 = (0..2).map(|s| wn.ss[s].diag().sum()).sum();
        let tr_os: f64 = (0..2).map(|s| wn.os[s].diag().sum()).sum();
        check(
            &format!("{label} Tr W{n}^SS - (E_aa + E_bb)"),
            tr_ss,
            e.same_spin(),
            TOL_ROUNDING,
        );
        check(
            &format!("{label} Tr W{n}^OS - E_ab"),
            tr_os,
            e.opposite_spin(),
            TOL_ROUNDING,
        );
    }
}

/// Item 1: the series limit equals MP3 (mp3.rs) and every trace equals its
/// u_mp3 channel energy.
#[test]
fn series_limit_equals_mp3_and_traces_equal_channel_energies() {
    let h2o = system("h2o", "rhf");
    let r = ri_u_mp3(&h2o);
    let w = w_matrices(&r);
    traces_equal_channel_energies("H2O RHF", &r, &w);
    let rs = mp3_energy(
        &h2o.mol,
        &h2o.obs,
        &h2o.aux,
        Operator::coulomb(),
        &h2o.scf,
        0,
        None,
    )
    .expect("mp3.rs");
    check(
        "H2O series limit Tr W2 - mp3.rs E2",
        w.w2.trace(),
        rs.e_mp2,
        TOL_MP3_RS,
    );
    check(
        "H2O series limit Tr(W2 + W3) - mp3.rs (E2 + E3)",
        w.w2.trace() + w.w3.trace(),
        rs.e_mp2 + rs.e_mp3,
        TOL_MP3_RS,
    );
    let oh = system("oh", "uhf");
    let r = ri_u_mp3(&oh);
    traces_equal_channel_energies("OH UHF", &r, &w_matrices(&r));
}

/// `Σ_σ Tr (W3 W2⁻¹)^(n−1) W3` for n = 1..=4, per spin block.
fn taylor_coefficients(w: &WMatrices) -> [f64; 4] {
    let mut c = [0.0; 4];
    for s in 0..2 {
        let (a, b) = (w.w2.full(s), w.w3.full(s));
        if a.nrows() == 0 {
            continue;
        }
        let ainv_b = Array2::from_shape_fn(a.raw_dim(), |_| 0.0);
        let mut ainv_b = ainv_b;
        for (j, col) in b.columns().into_iter().enumerate() {
            ainv_b
                .column_mut(j)
                .assign(&a.solve(&col.to_owned()).unwrap());
        }
        let mut m = b.clone();
        for cn in c.iter_mut() {
            *cn += m.diag().sum();
            m = m.dot(&ainv_b);
        }
    }
    c
}

/// Item 2: the Padé's own Taylor coefficients at x = 0.
#[test]
fn pade_taylor_coefficients_are_e2_e3_and_the_second_order_term() {
    for (name, kind) in [("h2o", "rhf"), ("oh", "uhf")] {
        let sys = system(name, kind);
        let w = w_matrices(&ri_u_mp3(&sys));
        let (e2, e3) = (w.w2.trace(), w.w3.trace());
        let c = taylor_coefficients(&w);
        let h = 1e-3;
        let f = |x: f64| op_mp3(&w, x).unwrap();
        let (fm, f0, fp) = (f(-h), f(0.0), f(h));
        let rnd = 100.0 * EPS * e2.abs();
        check(&format!("{name} f(0) - E2"), f0, e2, rnd);
        let d1 = (fp - fm) / (2.0 * h);
        check(
            &format!("{name} f'(0) - E3"),
            d1,
            e3,
            10.0 * h * h * c[2].abs() + rnd / h,
        );
        let d2 = 0.5 * (fp + fm - 2.0 * f0) / (h * h);
        check(
            &format!("{name} f''(0)/2 - Tr W3 W2^-1 W3"),
            d2,
            c[1],
            0.5 * (10.0 * 2.0 * h * h * c[3].abs() + 4.0 * rnd / (h * h)),
        );
    }
}

/// Occupied spin orbitals `[α.., β..]` and the same for virtuals; `t` and
/// `⟨ij‖ab⟩` expanded from the three stored blocks.
fn spin_orbital(blocks: &ferric_mp2::u_mp3::SpinBlockedDoubles, d: [usize; 4]) -> Array4<f64> {
    let [oa, ob, va, vb] = d;
    let (no, nv) = (oa + ob, va + vb);
    let mut t = Array4::zeros((no, no, nv, nv));
    for ((i, j, a, b), &x) in blocks.aa.indexed_iter() {
        t[[i, j, a, b]] = x;
    }
    for ((i, j, a, b), &x) in blocks.bb.indexed_iter() {
        t[[oa + i, oa + j, va + a, va + b]] = x;
    }
    for ((i, j, a, b), &x) in blocks.ab.indexed_iter() {
        let (jj, bb) = (oa + j, va + b);
        t[[i, jj, a, bb]] = x;
        t[[jj, i, a, bb]] = -x;
        t[[i, jj, bb, a]] = -x;
        t[[jj, i, bb, a]] = x;
    }
    t
}

/// Item 3: Eq. 11a by brute force on spin-orbital tensors vs the channels.
#[test]
fn eq18_channels_equal_the_k_spin_split_of_spin_orbital_eq11a() {
    let oh = system("oh", "uhf");
    let r = ri_u_mp3(&oh);
    let w = w_matrices(&r);
    let dm = r.dims;
    let d = [dm.no_a, dm.no_b, dm.nv_a, dm.nv_b];
    let g = spin_orbital(&r.ints, d);
    let no = dm.no_a + dm.no_b;
    let spin = |p: usize| usize::from(p >= dm.no_a);
    let offset = [0, dm.no_a];
    for (n, c, wn) in [(2, &r.c1, &w.w2), (3, &r.c2, &w.w3)] {
        let t = spin_orbital(c, d);
        // y[ks][i][j] = Σ_{k of spin ks, a, b} t_ikab g_jkab
        let mut full = [Array2::<f64>::zeros((no, no)), Array2::zeros((no, no))];
        for (i, j, k) in itertools_iter(no) {
            let s: f64 = t
                .slice(ndarray::s![i, k, .., ..])
                .iter()
                .zip(g.slice(ndarray::s![j, k, .., ..]).iter())
                .map(|(x, y)| x * y)
                .sum();
            full[spin(k)][[i, j]] += 0.125 * s;
            full[spin(k)][[j, i]] += 0.125 * s;
        }
        let scale = full
            .iter()
            .flat_map(|m| m.iter())
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let tol = 1e3 * EPS * scale;
        let mut cross = 0.0f64;
        for i in 0..no {
            for j in 0..no {
                if spin(i) != spin(j) {
                    cross = cross.max((full[0][[i, j]] + full[1][[i, j]]).abs());
                }
            }
        }
        check(
            &format!("W{n} alpha-beta block of Eq. 11a"),
            cross,
            0.0,
            tol,
        );
        for s in 0..2 {
            let ns = [dm.no_a, dm.no_b][s];
            let blk = |m: &Array2<f64>| {
                m.slice(ndarray::s![
                    offset[s]..offset[s] + ns,
                    offset[s]..offset[s] + ns
                ])
                .to_owned()
            };
            let dss = (&blk(&full[s]) - &wn.ss[s])
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            let dos = (&blk(&full[1 - s]) - &wn.os[s])
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            check(
                &format!("W{n} spin {s} SS: max |Eq.11a k-split - channel|"),
                dss,
                0.0,
                tol,
            );
            check(
                &format!("W{n} spin {s} OS: max |Eq.11a k-split - Eq. 18|"),
                dos,
                0.0,
                tol,
            );
        }
    }
}

fn itertools_iter(no: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    (0..no).flat_map(move |i| (0..no).flat_map(move |j| (0..no).map(move |k| (i, j, k))))
}

const N2: &str = "N 0.0 0.0 0.0\nN 0.0 0.0 1.0977";
const CH4: &str = "C 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n\
                   H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276";

fn molecule(atoms: &str, shift: f64) -> Molecule {
    let lines: Vec<String> = atoms
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let z: f64 = f[3].parse().unwrap();
            format!("{} {} {} {}", f[0], f[1], f[2], z + shift)
        })
        .collect();
    Molecule::parse_xyz(&format!("{}\n\n{}\n", lines.len(), lines.join("\n")), 0, 1).unwrap()
}

fn hf_models(mol: &Molecule) -> (f64, Opmp3Models) {
    let sys = PreparedSystem::new(mol.clone(), &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let cfg = Opmp3Config {
        scf_energy_conv: Some(1e-12),
        scf_density_conv: Some(1e-10),
        scf_max_iter: Some(400),
        ..Opmp3Config::new(Opmp3RefKind::Hf)
    };
    let r = run_opmp3(&sys, Opmp3Integrals::Exact, &cfg).expect("opmp3");
    (r.e_hf + r.e_nbs, r.models)
}

/// Item 4: every model is size-consistent; the scalar Padé is not.
#[test]
fn models_are_size_consistent_and_the_scalar_pade_is_not() {
    let (ea, ma) = hf_models(&molecule(N2, 0.0));
    let (eb, mb) = hf_models(&molecule(CH4, 0.0));
    let dimer = format!("{N2}\n{}", CH4);
    let shifted: Vec<String> = dimer
        .lines()
        .enumerate()
        .map(|(n, l)| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let z: f64 = f[3].parse::<f64>().unwrap() + if n >= 2 { 100.0 } else { 0.0 };
            format!("{} {} {} {z}", f[0], f[1], f[2])
        })
        .collect();
    let (eab, mab) = hf_models(&molecule(&shifted.join("\n"), 0.0));
    check("size E_HF", eab - ea - eb, 0.0, TOL_SIZE);
    let fields = |m: &Opmp3Models| {
        [
            ("MP2", m.mp2),
            ("MP3", m.mp3),
            ("OP-MP3", m.op_mp3),
            ("SOP-MP3", m.sop_mp3),
            ("SSD-OPMP3(0)", m.ssd_opmp3),
        ]
    };
    for ((k, a), ((_, b), (_, ab))) in fields(&ma)
        .into_iter()
        .zip(fields(&mb).into_iter().zip(fields(&mab)))
    {
        check(&format!("size {k}"), ab - a - b, 0.0, TOL_SIZE);
    }
    let scalar = mab.p_mp3 - ma.p_mp3 - mb.p_mp3;
    eprintln!("size scalar Pade (must exceed 1e-6): {scalar:+.3e}");
    assert!(
        scalar.abs() > 1e-6,
        "the scalar Pade must fail size consistency, |d| = {scalar:.2e}"
    );
}

fn pyscf_rohf_reference() -> Value {
    let p = workspace_root().join("testdata/reference/ump3/oh_rohf_cc-pvdz.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("oh_rohf reference")).unwrap()
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
}

/// Item 5: E_HF[ρ], E_NBS and the semicanonical ROHF reference.
#[test]
fn reference_path_rebuilds_e_hf_and_e_nbs_and_matches_pyscf_rohf() {
    let ctx = ParallelContext::default();
    for (name, kind) in [("h2o", "rhf"), ("oh", "uhf"), ("oh", "rohf")] {
        let sys = system(name, kind);
        let r = opmp3_reference(&ctx, &sys.mol, &sys.obs, &sys.bounds, 1e-12, &sys.scf).unwrap();
        check(
            &format!("{name} {kind} E_HF[rho] - E_SCF"),
            r.e_hf,
            sys.scf.energy,
            1e-11,
        );
        let nbs = r.e_nbs(0).unwrap();
        if kind == "rohf" {
            eprintln!(
                "{name} ROHF E_NBS {nbs:+.3e}, max |F_ia| {:.2e}",
                r.spins[0].max_ov()
            );
            assert!(
                nbs < -1e-4,
                "ROHF E_NBS must be clearly non-zero, got {nbs:.2e}"
            );
        } else {
            check(&format!("{name} {kind} E_NBS"), nbs, 0.0, 1e-12);
        }
    }
    let oh = system("oh", "rohf");
    // The same molecule and basis as `oh` (no ECP), so the same bounds.
    let oh_sys = PreparedSystem::new(oh.mol.clone(), &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let res = opmp3_on_reference(
        &oh_sys,
        &oh.scf,
        Opmp3Integrals::Exact,
        &Opmp3Config::new(Opmp3RefKind::Hf),
    )
    .unwrap();
    let refv = pyscf_rohf_reference();
    let tol = 10.0 * (res.e2.total() + res.e3.total()).abs() * (1e-9 + 1e-10);
    check(
        "OH ROHF E2 aa vs PySCF",
        res.e2.aa,
        num(&refv, "/exact/e2_aa"),
        tol,
    );
    check(
        "OH ROHF E2 bb vs PySCF",
        res.e2.bb,
        num(&refv, "/exact/e2_bb"),
        tol,
    );
    check(
        "OH ROHF E2 ab vs PySCF",
        res.e2.ab,
        num(&refv, "/exact/e2_ab"),
        tol,
    );
    check(
        "OH ROHF E3 vs PySCF",
        res.e3.total(),
        num(&refv, "/exact/e3"),
        tol,
    );
}
