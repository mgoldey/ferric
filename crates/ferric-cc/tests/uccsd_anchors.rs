//! Exactness anchors for spin-blocked UCCSD and UCCSD(T).
//!
//! Two independent routes compute the same numbers:
//!
//! * the spin-blocked solver (`u_ccsd::u_ccsd`, `u_ccsd_t::u_ccsd_t`), which
//!   carries t1a, t1b, t2aa, t2ab, t2bb and integral blocks per spin;
//! * the spin-orbital solver of `ccsd.rs` / `ccsd_t.rs` fed the UHF spin
//!   orbitals (`ccsd_uhf_spin_orbital`, `ccsd_t_uhf_spin_orbital`), which
//!   antisymmetrizes dense `<pq||rs>` over the full spin-orbital space and has
//!   no spin bookkeeping at all.
//!
//! They share the per-spin RI factors `B^P_pq` and orbital energies
//! (`URiBlocks`). A fault in that shared input is invisible here: giving the
//! alpha virtual block the beta orbital energies passes every test in this
//! file. `validation_uccsd.rs` (PySCF DF-UCCSD) catches it (2.3e-3 Ha on
//! NH2). If the spin-blocked
//! residuals or (T) class weights are right, the two agree to round-off on
//! every amplitude block and every (T) class. A dropped term, a wrong spin
//! label or a wrong weight changes the result by 6e-6 Ha or more on these
//! systems (measured with a dropped t2ab ring term, an aab weight of 1/6, an
//! alpha orbital energy on a beta (T) index and a dropped beta t1).
//!
//! The closed-shell anchor feeds the RHF orbitals to the UHF route as two
//! identical spin sets, so the spin-blocked solver must reproduce the
//! spin-adapted `ccsd_closed_shell` / `ccsd_t_closed_shell` pair.

use ferric_cc::ccsd::ccsd_uhf_spin_orbital;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t::ccsd_t_uhf_spin_orbital;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell;
use ferric_cc::u_ccsd::{u_ccsd, UCcsdResult};
use ferric_cc::u_ccsd_t::u_ccsd_t;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2, Array4};

/// Both routes converge |ΔE| < 1e-12 and share the integral factors, so the
/// energies agree to the CC convergence floor. Measured ≤ 1.1e-11 Ha
/// (CH3 E_CCSD); (T) classes ≤ 2.3e-13; closed shell ≤ 1.3e-13.
const TOL_E: f64 = 1e-10;
/// Amplitudes: the residual norm at |ΔE| < 1e-12 is ~1e-7; measured
/// ≤ 2.3e-10 (CH3 t1b).
const TOL_T: f64 = 1e-6;

fn cc_cfg(frozen_core: usize) -> CcConfig {
    CcConfig {
        frozen_core,
        max_iter: 300,
        energy_conv: 1e-12,
        ..Default::default()
    }
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
}

fn system(xyz: &str, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        bounds,
    }
}

fn scf_cfg() -> RhfConfig {
    RhfConfig {
        density_conv: 1e-10,
        energy_conv: 1e-11,
        max_iter: 300,
        ..Default::default()
    }
}

fn uhf(sys: &Sys) -> ScfResult {
    let r = solve_uhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        &sys.bounds,
        &scf_cfg(),
    )
    .unwrap();
    assert!(r.converged && matches!(r.spin, Spin::Unrestricted));
    r
}

const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9697\n";
const NH2: &str = "3\n\nN 0.0 0.0 0.1449\nH 0.0 0.8029 -0.5072\nH 0.0 -0.8029 -0.5072\n";
const CH3: &str =
    "4\n\nC 0.0 0.0 0.0\nH 0.0 1.0790 0.0\nH 0.9344 -0.5395 0.0\nH -0.9344 -0.5395 0.0\n";
const WATER: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

fn max_abs_diff2(a: &Array2<f64>, b: ndarray::ArrayView2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_abs_diff4(a: &Array4<f64>, b: ndarray::ArrayView4<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} spin-blocked {got:+.13} oracle {want:+.13} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

/// Spin-blocked UCCSD and UCCSD(T) against the spin-orbital route on the same
/// UHF orbitals: total energy, every amplitude block, every (T) class.
fn spin_blocked_matches_spin_orbital(label: &str, xyz: &str, mult: usize, frozen_core: usize) {
    let sys = system(xyz, mult, "6-31g");
    let scf = uhf(&sys);
    let op = Operator::coulomb();
    let cfg = cc_cfg(frozen_core);
    let ctx = format!("{label}/6-31G fc{frozen_core}");

    let blk: UCcsdResult = u_ccsd(&sys.mol, &sys.obs, &sys.dfbs, op, &scf, &cfg).unwrap();
    let so = ccsd_uhf_spin_orbital(&sys.mol, &sys.obs, &sys.dfbs, op, &scf, &cfg).unwrap();
    check(
        &ctx,
        "E_CCSD",
        blk.correlation_energy,
        so.correlation_energy,
        TOL_E,
    );
    check(
        &ctx,
        "aa+ab+bb",
        blk.e_aa + blk.e_ab + blk.e_bb,
        blk.correlation_energy,
        1e-13,
    );

    // Spin-orbital layout: α block then β block, for occupied and virtual.
    let t = &blk.amplitudes;
    let (na, va) = t.t1a.dim();
    let so_t1 = so.t1.as_ref().unwrap();
    let so_t2 = &so.t2;
    let d1a = max_abs_diff2(&t.t1a, so_t1.slice(s![..na, ..va]));
    let d1b = max_abs_diff2(&t.t1b, so_t1.slice(s![na.., va..]));
    let d2aa = max_abs_diff4(&t.t2aa, so_t2.slice(s![..na, ..na, ..va, ..va]));
    let d2ab = max_abs_diff4(&t.t2ab, so_t2.slice(s![..na, na.., ..va, va..]));
    let d2bb = max_abs_diff4(&t.t2bb, so_t2.slice(s![na.., na.., va.., va..]));
    // A spin-forbidden block must stay zero in the oracle (layout check).
    let forbidden = so_t1
        .slice(s![..na, va..])
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    eprintln!(
        "{ctx}: max|Δt| t1a {d1a:.1e} t1b {d1b:.1e} t2aa {d2aa:.1e} t2ab {d2ab:.1e} \
         t2bb {d2bb:.1e}; oracle α→β t1 {forbidden:.1e}"
    );
    assert!(
        forbidden < 1e-12,
        "{ctx}: oracle spin-orbital layout is not α-then-β"
    );
    for (name, d) in [
        ("t1a", d1a),
        ("t1b", d1b),
        ("t2aa", d2aa),
        ("t2ab", d2ab),
        ("t2bb", d2bb),
    ] {
        assert!(
            d < TOL_T,
            "{ctx}: {name} differs from the oracle by {d:.2e}"
        );
    }
    // The β singles are not small: a solver that drops them cannot pass.
    let t1b_norm = t.t1b.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        t1b_norm > 1e-3,
        "{ctx}: premise: |t1b| = {t1b_norm:.1e} is too small to test"
    );

    let tri = u_ccsd_t(&sys.mol, &sys.obs, &sys.dfbs, op, &scf, &blk, &cfg).unwrap();
    let so_t = ccsd_t_uhf_spin_orbital(&sys.mol, &sys.obs, &sys.dfbs, op, &scf, &so, &cfg).unwrap();
    for (name, got, want) in [
        ("(T) aaa", tri.aaa, so_t[0]),
        ("(T) aab", tri.aab, so_t[1]),
        ("(T) abb", tri.abb, so_t[2]),
        ("(T) bbb", tri.bbb, so_t[3]),
        ("E_(T)", tri.e_t, so_t.iter().sum()),
    ] {
        check(&ctx, name, got, want, TOL_E);
    }
    // Each class is resolvable: a wrong class weight must move it by more
    // than the bar.
    for (name, v) in [("aab", tri.aab), ("abb", tri.abb)] {
        assert!(
            v.abs() > 1e3 * TOL_E,
            "{ctx}: premise: (T) {name} = {v:.1e} too small"
        );
    }
}

#[test]
fn oh_spin_blocked_matches_spin_orbital() {
    spin_blocked_matches_spin_orbital("OH", OH, 2, 0);
}

#[test]
fn oh_frozen_core_spin_blocked_matches_spin_orbital() {
    spin_blocked_matches_spin_orbital("OH", OH, 2, 1);
}

#[test]
fn nh2_spin_blocked_matches_spin_orbital() {
    spin_blocked_matches_spin_orbital("NH2", NH2, 2, 0);
}

#[test]
fn ch3_spin_blocked_matches_spin_orbital() {
    spin_blocked_matches_spin_orbital("CH3", CH3, 2, 0);
}

/// RHF orbitals presented as a UHF reference with identical spin sets.
fn as_unrestricted(rhf: &ScfResult) -> ScfResult {
    let mut u = rhf.clone();
    u.spin = Spin::Unrestricted;
    u.mos_beta = Some(rhf.mos_alpha.clone());
    u.eps_beta = Some(rhf.eps_alpha.clone());
    u.density_beta = Some(rhf.density_alpha.clone());
    u.fock_beta = Some(rhf.fock_alpha.clone());
    u
}

/// The trivial limit: a closed-shell molecule through the UHF route equals the
/// spin-adapted closed-shell CCSD and (T).
fn closed_shell_through_uhf(frozen_core: usize) {
    let sys = system(WATER, 1, "6-31g");
    let op = Operator::coulomb();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        op,
        &sys.bounds,
        &scf_cfg(),
    )
    .unwrap();
    assert!(rhf.converged);
    let u = as_unrestricted(&rhf);
    let cfg = cc_cfg(frozen_core);
    let ctx = format!("H2O/6-31G closed shell fc{frozen_core}");

    let cs = ccsd_closed_shell(&sys.mol, &sys.obs, &sys.dfbs, op, &rhf, &cfg).unwrap();
    let blk = u_ccsd(&sys.mol, &sys.obs, &sys.dfbs, op, &u, &cfg).unwrap();
    check(
        &ctx,
        "E_CCSD",
        blk.correlation_energy,
        cs.correlation_energy,
        TOL_E,
    );
    check(&ctx, "e_aa-e_bb", blk.e_aa - blk.e_bb, 0.0, 1e-12);
    let t = &blk.amplitudes;
    let dt1 = max_abs_diff2(&t.t1a, t.t1b.view());
    let dt2 = max_abs_diff4(&t.t2aa, t.t2bb.view());
    assert!(
        dt1 < 1e-9 && dt2 < 1e-9,
        "{ctx}: α and β amplitudes differ ({dt1:.1e}, {dt2:.1e})"
    );
    // Closed-shell spatial t2 is the αβ block.
    let dab = max_abs_diff4(&t.t2ab, cs.t2.view());
    assert!(dab < TOL_T, "{ctx}: t2ab vs closed-shell t2 {dab:.2e}");

    let e_t_cs = ccsd_t_closed_shell(&sys.mol, &sys.obs, &sys.dfbs, op, &rhf, &cs, &cfg).unwrap();
    let tri = u_ccsd_t(&sys.mol, &sys.obs, &sys.dfbs, op, &u, &blk, &cfg).unwrap();
    check(&ctx, "E_(T)", tri.e_t, e_t_cs, TOL_E);
    check(&ctx, "aaa-bbb", tri.aaa - tri.bbb, 0.0, 1e-12);
    check(&ctx, "aab-abb", tri.aab - tri.abb, 0.0, 1e-12);
}

#[test]
fn closed_shell_through_uhf_matches_closed_shell() {
    closed_shell_through_uhf(0);
}

#[test]
fn closed_shell_through_uhf_matches_closed_shell_frozen_core() {
    closed_shell_through_uhf(1);
}

/// A restricted reference is refused with a pointer to the closed-shell solver.
#[test]
fn u_ccsd_refuses_a_restricted_reference() {
    let sys = system(WATER, 1, "sto-3g");
    let op = Operator::coulomb();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        op,
        &sys.bounds,
        &scf_cfg(),
    )
    .unwrap();
    let err = u_ccsd(&sys.mol, &sys.obs, &sys.dfbs, op, &rhf, &cc_cfg(0)).unwrap_err();
    assert!(format!("{err}").contains("ccsd_closed_shell"), "{err}");
}
