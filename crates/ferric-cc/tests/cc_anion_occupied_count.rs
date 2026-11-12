//! The CCSD and (T) drivers take the occupied count from the electron count,
//! and the virtual count from the MOs the RHF retained.
//!
//! # Systems
//!
//! CH3⁻ and Te²⁻ in def2-SVP (no diffuse functions), closed shell. Their RHF
//! HOMO energies are positive (+0.016 and +0.193 Ha; asserted here and in the
//! generator), so the number of negative orbital energies (4 and 10) is
//! smaller than `nelec / 2` (5 and 13); a driver that counts `eps < 0`
//! correlates the wrong electron count. Te²⁻ also carries the def2
//! 28-electron ECP, so its occupied count is the valence count
//! (`Molecule::nelec` after `apply_ecp`): 26 electrons, 13 orbitals.
//!
//! References: `scripts/gen_pyscf_cc_anion_refs.py`, which runs the
//! like-for-like DF construction of `scripts/validation/gen_cc.py` (exact
//! RHF, the same auxiliary basis, diagonal canonical Fock, PySCF CCSD and
//! (T) on those integrals). Output: `testdata/reference/<system>_def2-svp_anion_ccsd.json`.
//!
//! # Bars
//!
//! The bars of `validation_cc.rs` (set there from measured maxima over the
//! neutral validation systems): RHF 1e-9, CCSD 2e-10, (T) 1e-10. Getting the
//! occupied count wrong moves the correlation energy by the pair energies of
//! a whole orbital, 1e-2 Ha here, or fails on a shape check.
//!
//! # Lindep
//!
//! `nv_counts_only_the_retained_mos` puts a ghost He 0.001 Å from a He atom,
//! so the SCF drops near-null overlap modes and pads `mos_r()` with zero
//! columns. The CC amplitudes must span only the retained virtuals.

use std::path::{Path, PathBuf};

use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t::ccsd_t;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell;
use ferric_cc::helpers::closed_shell_counts;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

/// RHF total energy (validation_cc.rs `TOL_RHF`).
const TOL_RHF: f64 = 1e-9;
/// CCSD correlation energy (validation_cc.rs `TOL_CC`).
const TOL_CC: f64 = 2e-10;
/// (T) energy (validation_cc.rs `TOL_T`).
const TOL_T: f64 = 1e-10;

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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn uint(v: &Value, ptr: &str, ctx: &str) -> usize {
    v.pointer(ptr)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an integer"))
        as usize
}

/// Convergence is tighter than validation_cc.rs's 1e-11: at 1e-11 the
/// spin-orbital CCSD on CH3⁻ stops with amplitudes whose error still moves
/// E_CCSD by 1.2e-10 and E_(T) by 3.1e-10 (measured), while the spin-adapted
/// driver on the same system is within 1e-13 of PySCF.
fn cc_cfg(frozen_core: usize) -> CcConfig {
    CcConfig {
        frozen_core,
        max_iter: 400,
        energy_conv: 1e-13,
        ..Default::default()
    }
}

fn close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<24} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

struct Case {
    ctx: String,
    reference: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn load_case(system: &str) -> Case {
    let path = workspace_root()
        .join("testdata/reference")
        .join(format!("{system}_def2-svp_anion_ccsd.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/gen_pyscf_cc_anion_refs.py",
            path.display()
        )
    });
    let reference: Value = serde_json::from_str(&text).unwrap();
    let ctx = format!("{}/def2-svp", reference["system"].as_str().unwrap());
    let xyz = reference["xyz_angstrom"].as_str().unwrap();
    let charge = reference["charge"].as_i64().unwrap() as i32;
    let bs = basis::bundled(reference["basis"].as_str().unwrap()).unwrap();
    let mut mol = Molecule::parse_xyz(xyz, charge, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(
        (mol.nuclear_repulsion() - num(&reference, "/nuclear_repulsion", &ctx)).abs() < 1e-9,
        "{ctx}: geometry mismatch"
    );
    assert_eq!(
        mol.nelec() as usize,
        uint(&reference, "/nelectron", &ctx),
        "{ctx}: electron count (valence, after apply_ecp) differs from PySCF's"
    );
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let aux = basis::bundled(reference["aux_basis"].as_str().unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
    assert_eq!(obs.nbasis(), uint(&reference, "/nao", &ctx));
    assert_eq!(dfbs.nbasis(), uint(&reference, "/naux", &ctx));

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: RHF failed: {e:?}"));
    assert!(rhf.converged && rhf.df_jk.is_none(), "{ctx}: exact RHF");
    close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&reference, "/rhf/energy", &ctx),
        TOL_RHF,
    );

    // The point of the system: the HOMO lies above zero, so counting negative
    // orbital energies gives fewer than nelec / 2 occupied orbitals.
    let nocc = uint(&reference, "/nocc", &ctx);
    let homo = rhf.eps_r()[nocc - 1];
    let n_negative = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    eprintln!("{ctx}: HOMO {homo:+.6} Ha, {n_negative} negative MO energies, nocc {nocc}");
    assert!(
        homo > 0.0,
        "{ctx}: HOMO {homo} is not positive; the test is vacuous"
    );
    assert!(n_negative < nocc);
    assert_eq!(closed_shell_counts(&mol, &rhf).unwrap().0, nocc);

    Case {
        ctx,
        reference,
        mol,
        obs,
        dfbs,
        rhf,
    }
}

/// Spin-orbital and spin-adapted CCSD and (T), all-electron, against PySCF.
fn check_all_electron(c: &Case) {
    let (ctx, r) = (c.ctx.as_str(), &c.reference);
    let op = Operator::coulomb();
    let cfg = cc_cfg(0);
    let e_ccsd = num(r, "/all_electron/ccsd/e_corr", ctx);
    let e_t = num(r, "/all_electron/ccsd_t/e_t", ctx);

    let sa = ccsd_closed_shell(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &cfg).unwrap();
    close(
        ctx,
        "E_CCSD spin-adapted",
        sa.correlation_energy,
        e_ccsd,
        TOL_CC,
    );
    let t_sa = ccsd_t_closed_shell(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &sa, &cfg).unwrap();
    close(ctx, "E_(T) spin-adapted", t_sa, e_t, TOL_T);

    let so = ccsd(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &cfg).unwrap();
    close(
        ctx,
        "E_CCSD spin-orbital",
        so.correlation_energy,
        e_ccsd,
        TOL_CC,
    );
    let t_so = ccsd_t(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &so, &cfg).unwrap();
    close(ctx, "E_(T) spin-orbital", t_so, e_t, TOL_T);
}

/// Frozen core: the frozen orbitals are the lowest ones, counted inside the
/// electron-count occupied space.
fn check_frozen_core(c: &Case) {
    let (ctx, r) = (c.ctx.as_str(), &c.reference);
    let op = Operator::coulomb();
    let cfg = cc_cfg(uint(r, "/n_frozen_core_control", ctx));
    let sa = ccsd_closed_shell(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &cfg).unwrap();
    close(
        ctx,
        "E_CCSD sa frozen core",
        sa.correlation_energy,
        num(r, "/frozen_core/ccsd/e_corr", ctx),
        TOL_CC,
    );
    let t_sa = ccsd_t_closed_shell(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &sa, &cfg).unwrap();
    close(
        ctx,
        "E_(T) sa frozen core",
        t_sa,
        num(r, "/frozen_core/ccsd_t/e_t", ctx),
        TOL_T,
    );
    let so = ccsd(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &cfg).unwrap();
    close(
        ctx,
        "E_CCSD so frozen core",
        so.correlation_energy,
        num(r, "/frozen_core/ccsd/e_corr", ctx),
        TOL_CC,
    );
    let t_so = ccsd_t(&c.mol, &c.obs, &c.dfbs, op, &c.rhf, &so, &cfg).unwrap();
    close(
        ctx,
        "E_(T) so frozen core",
        t_so,
        num(r, "/frozen_core/ccsd_t/e_t", ctx),
        TOL_T,
    );
}

#[test]
fn methyl_anion_ccsd_and_t_match_pyscf() {
    let c = load_case("ch3");
    check_all_electron(&c);
    check_frozen_core(&c);
}

/// Te²⁻ with the def2 ECP: 26 valence electrons, 13 occupied orbitals.
#[test]
fn telluride_with_ecp_counts_valence_electrons() {
    let c = load_case("te");
    assert_eq!(
        c.mol.atoms[0].n_core_ecp, 28,
        "def2-SVP carries a 28-electron ECP on Te"
    );
    assert_eq!(c.mol.nelec(), 26);
    assert_eq!(closed_shell_counts(&c.mol, &c.rhf).unwrap().0, 13);
    check_all_electron(&c);
    check_frozen_core(&c);
}

/// With near-linear-dependent AOs the virtual space is the retained MOs only.
#[test]
fn nv_counts_only_the_retained_mos() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = Molecule::parse_xyz("2\n\nHe 0.0 0.0 0.0\n@He 0.0 0.0 0.001\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-svp-rifit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    assert!(rhf.converged);
    let nbas = obs.nbasis();
    let (nocc, nmo) = closed_shell_counts(&mol, &rhf).unwrap();
    eprintln!("He + ghost He: nbas {nbas}, retained {nmo}, nocc {nocc}");
    assert_eq!(nocc, 1);
    assert!(
        nmo < nbas,
        "the overlap must have dropped modes ({nmo} of {nbas})"
    );
    let nv = nmo - nocc;

    let cfg = cc_cfg(0);
    let sa = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    assert_eq!(sa.t1.as_ref().unwrap().shape(), [nocc, nv]);
    let so = ccsd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    assert_eq!(so.t1.as_ref().unwrap().shape(), [2 * nocc, 2 * nv]);
    // Two independent ferric CCSD pipelines on the same integrals
    // (validation_cc.rs `TOL_INTERNAL`).
    close(
        "He+ghost",
        "CCSD sa vs so",
        sa.correlation_energy,
        so.correlation_energy,
        1e-10,
    );
    let t_sa = ccsd_t_closed_shell(&mol, &obs, &dfbs, op, &rhf, &sa, &cfg).unwrap();
    let t_so = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &so, &cfg).unwrap();
    // One occupied spatial orbital: (T) is zero. The spin-orbital driver
    // returns 0 before forming any triple; the spin-adapted one sums terms
    // that cancel, measured -4.2e-21 Ha, so it is held at 1e-16.
    assert_eq!(t_so, 0.0);
    assert!(t_sa.abs() < 1e-16, "(T) for two electrons is {t_sa:e}");
}
