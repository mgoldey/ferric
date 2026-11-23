//! Spin-blocked unrestricted MP3 (`ferric_mp2::u_mp3`) anchors.
//!
//! # What is compared
//!
//! 1. EXACTNESS FIRST, closed shell: an RHF reference through the U path
//!    equals the spin-orbital `mp3.rs` contraction, RI (`mp3_energy` itself)
//!    and exact (`mp3_energy_from_spin_orbital_blocks` fed exact spatial
//!    blocks). Same integrals, different contraction code, so the bar is the
//!    floating-point floor: [`TOL_SAME_INTEGRALS`].
//! 2. Spin adaptation of the amplitudes on that RHF input:
//!    `c^αα_ijab = c^αβ_ijab − c^αβ_ijba` for BOTH c^(1) and c^(2), and
//!    `c^ββ = c^αα`. This sees a wrong αα/ββ residual term that an energy
//!    could hide.
//! 3. Spin channels: the E2 split αα / ββ / αβ equals `u_ri_mp2`'s
//!    components on the same RI integrals, UHF and ROHF, and UHF with one
//!    frozen core orbital per spin ([`TOL_SAME_INTEGRALS`]).
//! 4. Independent code: E_SCF, E2 per channel and E3 against PySCF
//!    (UMP2 + ADC(3) ground state; `scripts/gen_pyscf_ump3_refs.py` →
//!    `testdata/reference/ump3/`), exact and RI, for H2O RHF, OH ²Π and
//!    CH2 ³B1 with UHF and semicanonicalized-ROHF references.
//! 5. The memory pre-flight: a budget one byte under the stated peak is
//!    refused with an actionable message, the peak itself admits.
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! * Right: the RHF U path sits at the 1e-13 floor of item 1, and the
//!   open-shell energies at the SCF-convergence floor of item 4.
//! * An αβ-vs-ββ (or αα-vs-ββ) block mix-up in a residual term: closed shell
//!   CANNOT see a pure α↔β relabelling (the blocks are equal there), but it does
//!   see a mix of αβ with a same-spin block (antisymmetric vs not), and the
//!   open-shell rows see both by 1e-4..1e-3 Ha in E3.
//! * A sign or factor error in one ring term: E3 misses by 1e-3..1e-2 Ha in
//!   every row, E2 untouched (c^(1) is unaffected).
//! * A wrong ROHF semicanonicalization: the ROHF rows miss in E2 already,
//!   the UHF rows pass.
//!
//! # TOLERANCES (derived)
//!
//! * [`TOL_SAME_INTEGRALS`] 1e-10 Ha: the same integrals summed in another
//!   order differ by ~n_terms·ε·|E| ≈ 1e4·1e-16·0.2 = 2e-13; 1e-10 is the
//!   brief's bar, three decades above that floor.
//! * vs PySCF, correlation energies: E_c is not stationary in the orbitals, so
//!   an orbital error δκ moves it linearly, |δE_c| ≲ |E_c| δκ. δκ is bounded by
//!   the two SCF convergence criteria, PySCF `conv_tol_grad` 1e-9 and ferric
//!   `density_conv` 1e-10; with a factor 10 for the gradient-norm-to-δκ
//!   conversion, bar = 10·|E2 + E3|·(1e-9 + 1e-10) (1.8e-9 Ha for OH).
//! * vs PySCF, E_SCF: stationary, so δE ∝ δκ²; the floor is the two energy
//!   criteria, bar = 10·(1e-12 + 1e-12) = 2e-11 Ha.
//!
//! # MUTATIONS (run 2026-11-14; each failed on an assertion unless noted)
//!
//! `crates/ferric-mp2/src/u_mp3_residual.rs` `mp3_residual`: swap the αβ and
//! ββ amplitude blocks in the particle-particle ladder,
//! `particle_ladder(src, ALPHA, BETA, &t.ab, ..)` → `&t.bb` and
//! `particle_ladder(src, BETA, BETA, &t.bb, ..)` → `&t.ab`. Shapes agree only
//! for a closed shell, so the closed-shell energy anchor is the one that must
//! fail on an assertion (an open-shell row panics on the shape mismatch,
//! which does not count). Outcome: RI E3 −0.031018 vs mp3.rs −0.006894
//! (|d| 2.4e-2), and the H2O-vs-PySCF E3 row by 2.4e-2; OH UHF panicked in
//! the GEMM shape check, as predicted.
//!
//! Also run: swapping the αα and ββ labels in the channel energies fails the
//! `u_ri_mp2` channel test and the PySCF channel rows (|d| 1.3e-2) while the
//! closed-shell anchor passes (αα = ββ there); and halving the pre-flighted
//! peak fails the pre-flight test (one byte under the peak is admitted).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mp3::{mp3_energy, mp3_energy_from_spin_orbital_blocks};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::spinorbital::{asym_oovv, asym_ovvo, asym_same};
use ferric_mp2::u_mp3::{u_mp3, u_mp3_peak_bytes, UMp3Config, UMp3Eri, UMp3Result};
use ferric_mp2::u_mp3_eri::exact_mo_block;
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::{s, Array4};
use serde_json::Value;

const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";
const TOL_SAME_INTEGRALS: f64 = 1e-10;
const PYSCF_CONV_TOL_GRAD: f64 = 1e-9;
const FERRIC_DENSITY_CONV: f64 = 1e-10;
const TOL_E_SCF: f64 = 10.0 * (1e-12 + 1e-12);

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

fn reference(system: &str, scf_kind: &str) -> Value {
    let path = workspace_root()
        .join("testdata/reference/ump3")
        .join(format!("{system}_{scf_kind}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/gen_pyscf_ump3_refs.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
}

fn check(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.1e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs {want:.12}: |d| {d:.2e} >= {tol:.1e}"
    );
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    aux: PreparedBasis,
    scf: ScfResult,
}

fn tight() -> RhfConfig {
    RhfConfig::default()
        .with_max_iter(400)
        .with_energy_conv(1e-12)
        .with_density_conv(FERRIC_DENSITY_CONV)
}

/// Molecule, bases and a converged reference of kind `rhf`/`uhf`/`rohf`.
fn system(name: &str, scf_kind: &str) -> Sys {
    let (charge, mult) = match name {
        "h2o" => (0, 1),
        "oh" => (0, 2),
        "ch2_triplet" => (0, 3),
        _ => panic!("unknown system {name}"),
    };
    let xyz = workspace_root().join(format!("testdata/molecules/validation/{name}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf = match scf_kind {
        "rhf" => solve_rhf(&ctx, &mol, &obs, op, &bounds, &tight()),
        "uhf" => solve_uhf(&ctx, &mol, &obs, &bounds, &{
            let mut c = tight().with_check_stability(true);
            c.scf_stability_descent = true;
            c
        }),
        "rohf" => solve_rohf(&ctx, &mol, &obs, op, &bounds, &tight()),
        _ => panic!("unknown reference {scf_kind}"),
    }
    .unwrap_or_else(|e| panic!("{name} {scf_kind}: {e:?}"));
    assert!(scf.converged, "{name} {scf_kind}: SCF not converged");
    Sys { mol, obs, aux, scf }
}

fn run(sys: &Sys, exact: bool) -> UMp3Result {
    run_fc(sys, exact, 0)
}

fn run_fc(sys: &Sys, exact: bool, frozen_core: usize) -> UMp3Result {
    let eri = if exact {
        UMp3Eri::Exact
    } else {
        UMp3Eri::Ri { aux: &sys.aux }
    };
    u_mp3(
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.scf,
        eri,
        &UMp3Config {
            frozen_core,
            ..Default::default()
        },
    )
    .expect("u_mp3")
}

/// Spin-orbital `mp3.rs` contraction on EXACT spatial blocks of an RHF.
fn spin_orbital_exact_mp3(sys: &Sys) -> (f64, f64) {
    let n = sys.obs.nbasis();
    let nocc = sys.mol.nelec() as usize / 2;
    let c = sys.scf.mos_r();
    let co = c.slice(s![.., ..nocc]).to_owned();
    let cv = c.slice(s![.., nocc..]).to_owned();
    let (no, nv) = (co.ncols(), cv.ncols());
    let ao = ferric_mp2::canonical::dense_ao_eri(&sys.obs, Operator::coulomb()).unwrap();
    let blk = |c4: [&ndarray::Array2<f64>; 4]| exact_mo_block(&ao, n, c4).into_dyn();
    let oovv = asym_oovv(&blk([&co, &cv, &co, &cv]), no, nv);
    let vvvv = asym_same(&blk([&cv, &cv, &cv, &cv]), nv);
    let oooo = asym_same(&blk([&co, &co, &co, &co]), no);
    let ovvo = asym_ovvo(
        &blk([&co, &cv, &cv, &co]),
        &blk([&co, &co, &cv, &cv]),
        no,
        nv,
    );
    let eps = sys.scf.eps_r();
    let eo: Vec<f64> = (0..2 * no).map(|p| eps[p / 2]).collect();
    let ev: Vec<f64> = (0..2 * nv).map(|p| eps[nocc + p / 2]).collect();
    mp3_energy_from_spin_orbital_blocks(oovv, vvvv, oooo, ovvo, &eo, &ev)
}

/// max |c_aa[i,j,a,b] − (c_ab[i,j,a,b] − c_ab[i,j,b,a])| and max |c_bb − c_aa|.
fn spin_adaptation_residual(aa: &Array4<f64>, bb: &Array4<f64>, ab: &Array4<f64>) -> (f64, f64) {
    let swapped = ab.view().permuted_axes([0, 1, 3, 2]);
    let mut d_adapt = 0.0f64;
    ndarray::Zip::from(aa)
        .and(ab)
        .and(&swapped)
        .for_each(|&x, &y, &z| d_adapt = d_adapt.max((x - (y - z)).abs()));
    let mut d_ab = 0.0f64;
    ndarray::Zip::from(aa)
        .and(bb)
        .for_each(|&x, &y| d_ab = d_ab.max((x - y).abs()));
    (d_adapt, d_ab)
}

#[test]
fn closed_shell_u_path_equals_mp3_rs_and_is_spin_adapted() {
    let sys = system("h2o", "rhf");
    let ctx = "h2o/cc-pvdz RHF";

    let r_ri = run(&sys, false);
    let m = mp3_energy(
        &sys.mol,
        &sys.obs,
        &sys.aux,
        Operator::coulomb(),
        &sys.scf,
        0,
        None,
    )
    .unwrap();
    check(ctx, "RI E2", r_ri.e2.total(), m.e_mp2, TOL_SAME_INTEGRALS);
    check(ctx, "RI E3", r_ri.e3.total(), m.e_mp3, TOL_SAME_INTEGRALS);

    let r_ex = run(&sys, true);
    let (e2, e3) = spin_orbital_exact_mp3(&sys);
    check(ctx, "exact E2", r_ex.e2.total(), e2, TOL_SAME_INTEGRALS);
    check(ctx, "exact E3", r_ex.e3.total(), e3, TOL_SAME_INTEGRALS);

    // Amplitude scale ~|c| ≤ 0.1; the identity holds to the same summation floor.
    for (label, r) in [("RI", &r_ri), ("exact", &r_ex)] {
        for (order, c) in [("c1", &r.c1), ("c2", &r.c2)] {
            let (d_adapt, d_ab) = spin_adaptation_residual(&c.aa, &c.bb, &c.ab);
            eprintln!(
                "{ctx} {label} {order}: max|aa-(ab-abᵀ)| {d_adapt:.2e} max|bb-aa| {d_ab:.2e}"
            );
            assert!(
                d_adapt < 1e-12,
                "{ctx} {label} {order}: not spin adapted ({d_adapt:.2e})"
            );
            assert!(
                d_ab < 1e-12,
                "{ctx} {label} {order}: c_bb != c_aa ({d_ab:.2e})"
            );
        }
    }
}

#[test]
fn e2_spin_channels_equal_u_rimp2_uhf_and_rohf() {
    for (scf_kind, frozen_core) in [("uhf", 0), ("rohf", 0), ("uhf", 1)] {
        let sys = system("oh", scf_kind);
        let ctx = format!("oh/cc-pvdz {scf_kind} frozen_core {frozen_core}");
        let r = run_fc(&sys, false, frozen_core);
        let u = u_ri_mp2(
            &sys.mol,
            &sys.obs,
            &sys.aux,
            Operator::coulomb(),
            &sys.scf,
            &RiMp2Config {
                frozen_core,
                ..Default::default()
            },
        )
        .unwrap();
        check(
            &ctx,
            "E2 aa",
            r.e2.aa,
            u.components.e_aa,
            TOL_SAME_INTEGRALS,
        );
        check(
            &ctx,
            "E2 bb",
            r.e2.bb,
            u.components.e_bb,
            TOL_SAME_INTEGRALS,
        );
        check(
            &ctx,
            "E2 ab",
            r.e2.ab,
            u.components.e_ab,
            TOL_SAME_INTEGRALS,
        );
        check(
            &ctx,
            "E2 SS",
            r.e2.same_spin(),
            u.components.e_aa + u.components.e_bb,
            TOL_SAME_INTEGRALS,
        );
    }
}

/// Every system/reference/integral row against PySCF.
fn check_against_pyscf(name: &str, scf_kind: &str) {
    let refj = reference(name, scf_kind);
    let sys = system(name, scf_kind);
    check(
        &format!("{name} {scf_kind}"),
        "E_SCF",
        sys.scf.energy,
        num(&refj, "/scf/energy"),
        TOL_E_SCF,
    );
    for (block, exact) in [("exact", true), ("ri", false)] {
        let ctx = format!("{name}/cc-pvdz {scf_kind} {block}");
        let r = run(&sys, exact);
        let p = |k: &str| num(&refj, &format!("/{block}/{k}"));
        let bar = 10.0 * (p("e2") + p("e3")).abs() * (PYSCF_CONV_TOL_GRAD + FERRIC_DENSITY_CONV);
        check(&ctx, "E2", r.e2.total(), p("e2"), bar);
        check(&ctx, "E2 aa", r.e2.aa, p("e2_aa"), bar);
        check(&ctx, "E2 bb", r.e2.bb, p("e2_bb"), bar);
        check(&ctx, "E2 ab", r.e2.ab, p("e2_ab"), bar);
        check(&ctx, "E3", r.e3.total(), p("e3"), bar);
    }
}

#[test]
fn h2o_rhf_matches_pyscf() {
    check_against_pyscf("h2o", "rhf");
}

#[test]
fn oh_uhf_matches_pyscf() {
    check_against_pyscf("oh", "uhf");
}

#[test]
fn oh_rohf_matches_pyscf() {
    check_against_pyscf("oh", "rohf");
}

#[test]
fn ch2_uhf_matches_pyscf() {
    check_against_pyscf("ch2_triplet", "uhf");
}

#[test]
fn ch2_rohf_matches_pyscf() {
    check_against_pyscf("ch2_triplet", "rohf");
}

#[test]
fn preflight_refuses_one_byte_under_the_peak_and_admits_the_peak() {
    let sys = system("oh", "uhf");
    for (exact, eri) in [
        (false, UMp3Eri::Ri { aux: &sys.aux }),
        (true, UMp3Eri::Exact),
    ] {
        let r = run(&sys, exact);
        assert_eq!(r.peak_bytes, u_mp3_peak_bytes(&r.dims, exact));
        let go = |budget: usize| {
            u_mp3(
                &sys.mol,
                &sys.obs,
                Operator::coulomb(),
                &sys.scf,
                eri,
                &UMp3Config {
                    memory_budget_bytes: Some(budget),
                    ..Default::default()
                },
            )
        };
        let refused = go(r.peak_bytes - 1);
        assert!(refused.is_err(), "one byte under the peak was admitted");
        let msg = refused.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("U-MP3") && msg.contains("budget is"),
            "unexpected refusal text: {msg}"
        );
        let ok = go(r.peak_bytes).expect("the peak itself must admit");
        assert_eq!(ok.e3.total().to_bits(), r.e3.total().to_bits());
    }
}
