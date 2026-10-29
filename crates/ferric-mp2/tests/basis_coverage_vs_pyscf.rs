//! Every bundled basis set against PySCF fed the same basis from the Basis
//! Set Exchange: the exactness anchor that a set must pass before it counts.
//!
//! References: `testdata/reference/basis_coverage_pyscf.json`, written by
//! `scripts/gen_pyscf_basis_coverage_refs.py`. That script fetches each set
//! from BSE BY NAME and lets PySCF parse and normalize it, so it shares
//! neither ferric's bundled JSON nor its loader: agreement checks the shipped
//! digits, the BSE-JSON parse, the contraction renormalization, the
//! spherical/Cartesian choice and (for the -PP sets) the inline ECP.
//!
//! * orbital sets: RHF energy, exact integrals, `TOL_E` Ha;
//! * RI / JK auxiliary sets: all-electron RI-MP2 correlation energy with the
//!   matching orbital set, `TOL_E` Ha (and the RHF energy again);
//! * JK sets also in their own role: RHF with J AND K density-fitted
//!   (`df_j_aux` = `df_k_aux` = the set) vs PySCF `density_fit`, `TOL_E` Ha.
//!   aug-cc-pVXZ-JKFIT is not in BSE: its PySCF side is PySCF's own copy,
//!   proved bit-identical to ferric's by `scripts/import_psi4_jkfit.py`, so
//!   for those three sets this file checks parsing/normalization and the
//!   import script checks the digits;
//! * every case: basis-function count equal to PySCF's (catches a 5D/6D or
//!   7F/10F mix-up independently of the energy), and the libint overlap
//!   diagonal equal to 1 (Cartesian l >= 2 sets: the axial components only,
//!   since libint and PySCF normalize Cartesian shells on x^l).
//!
//! `every_registered_set_has_a_case` makes the parametrization exhaustive: a
//! set added to `basis::REGISTRY` without a reference case fails here.
//!
//! MUTATIONS (run 2026-10-29 with BASIS_COVERAGE_ONLY; each failed as shown):
//! * `6-31g_d.json`, first O coefficient x1.01 -> `orbital_sets_match_pyscf_rhf`
//!   6-31g*/h2o off by 4.3e-6 Ha;
//! * `def2-svp-rifit.json`, a mid O p-shell exponent x1.01 ->
//!   `auxiliary_sets_match_pyscf_rimp2` off by 1.6e-6 Ha;
//! * `aug-cc-pvtz-jkfit.json`, a mid O p-shell exponent x1.01 ->
//!   `jk_sets_match_pyscf_density_fitted_rhf` off by 1.1e-7 Ha;
//! * the same x1.01 on the TIGHTEST O s exponent of either aux set PASSED
//!   (fitting error < 1e-8 Ha): this bar catches errors in the functions that
//!   carry weight, not every digit. Digit-level fidelity is the job of
//!   `scripts/fetch_bse_bases.py --check` (numerical equality with BSE) and,
//!   for the aug-JKFIT sets, `scripts/import_psi4_jkfit.py`;
//! * `renormalize_contraction` made a no-op -> ferric-core's
//!   `every_bundled_contraction_has_unit_self_overlap` (libint renormalizes
//!   internally, so the energies here cannot see it).

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;

/// Energy agreement with PySCF, Ha.
const TOL_E: f64 = 1e-8;
/// Overlap-diagonal deviation from 1.
const TOL_S: f64 = 1e-10;

#[derive(Deserialize)]
struct Refs {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    kind: String,
    basis: String,
    auxbasis: Option<String>,
    molecule: String,
    atoms_angstrom: String,
    charge: i32,
    cart: bool,
    ecp: bool,
    nao: usize,
    nelectron: i32,
    e_rhf: f64,
    e_mp2_corr: Option<f64>,
}

impl Case {
    fn label(&self) -> String {
        match &self.auxbasis {
            Some(a) => format!("{}+{}/{}", self.basis, a, self.molecule),
            None => format!("{}/{}", self.basis, self.molecule),
        }
    }
}

fn refs() -> Refs {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/basis_coverage_pyscf.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn molecule(case: &Case) -> Molecule {
    let lines: Vec<&str> = case.atoms_angstrom.split(';').map(str::trim).collect();
    let xyz = format!("{}\n{}\n{}\n", lines.len(), case.molecule, lines.join("\n"));
    Molecule::parse_xyz(&xyz, case.charge, 1).unwrap()
}

/// Is component `k` of a Cartesian shell of angular momentum `l` (libint's
/// CCA order) an axial one (x^l, y^l or z^l)?
fn cartesian_axial(l: i32, k: usize) -> bool {
    let mut idx = 0;
    for i in 0..=l {
        for j in 0..=i {
            let (lx, ly, lz) = (l - i, i - j, j);
            if idx == k {
                return lx == l || ly == l || lz == l;
            }
            idx += 1;
        }
    }
    unreachable!("component {k} out of range for l={l}")
}

/// Max |S_ii - 1| over every function whose diagonal is 1 by convention.
fn overlap_diag_error(mol: &Molecule, bs: &BasisSet, prep: &PreparedBasis) -> f64 {
    let s = oneelectron::overlap(prep);
    let mut i = 0;
    let mut worst = 0.0_f64;
    for atom in &mol.atoms {
        for sh in bs.for_element(atom.z).unwrap() {
            let n = basis::num_functions(sh.l, sh.pure);
            for k in 0..n {
                if sh.pure || sh.l < 2 || cartesian_axial(sh.l, k) {
                    worst = worst.max((s[[i + k, i + k]] - 1.0).abs());
                }
            }
            i += n;
        }
    }
    assert_eq!(i, s.nrows(), "shell walk does not cover the AO space");
    worst
}

/// SCF settings. For a `dfhf` case J and K are both fitted with the case's
/// auxiliary set, as PySCF's `density_fit(auxbasis=...)` does.
fn scf_config(case: &Case) -> RhfConfig {
    let df = (case.kind == "dfhf").then(|| case.auxbasis.clone().unwrap());
    RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 400,
        df_j_aux: df.clone(),
        df_k_aux: df,
        ..Default::default()
    }
}

/// Structural checks against PySCF: ECP, electron count, basis-function count
/// and the overlap diagonal. Returns `false` when the AO spaces differ (the
/// energies would then be meaningless).
fn check_setup(
    case: &Case,
    mol: &Molecule,
    bs: &BasisSet,
    obs: &PreparedBasis,
    fails: &mut Vec<String>,
) -> bool {
    let label = case.label();
    let has_ecp = mol.atoms.iter().any(|a| a.n_core_ecp > 0);
    if has_ecp != case.ecp {
        fails.push(format!("{label}: ECP active {has_ecp}, PySCF {}", case.ecp));
    }
    if mol.nelec() != case.nelectron {
        fails.push(format!(
            "{label}: nelec {} vs PySCF {}",
            mol.nelec(),
            case.nelectron
        ));
    }
    if obs.nbasis() != case.nao {
        fails.push(format!(
            "{label}: nbf {} vs PySCF {} (cart={})",
            obs.nbasis(),
            case.nao,
            case.cart
        ));
        return false;
    }
    let ds = overlap_diag_error(mol, bs, obs);
    if ds > TOL_S {
        fails.push(format!("{label}: overlap diagonal off by {ds:.2e}"));
    }
    true
}

fn check_energy(label: &str, what: &str, got: f64, want: f64, fails: &mut Vec<String>) {
    let d = got - want;
    eprintln!("{label:44} d{what} {d:+.2e}");
    if d.abs() > TOL_E {
        fails.push(format!(
            "{label}: {what} {got:.12} vs PySCF {want:.12} (d {d:.2e})"
        ));
    }
}

/// Run one case; return its failures (empty when it passes).
fn run_case(case: &Case) -> Vec<String> {
    let label = case.label();
    let mut fails = Vec::new();
    let mut mol = molecule(case);
    let bs = basis::bundled(&case.basis).unwrap();
    mol.apply_ecp(&bs);
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    if !check_setup(case, &mol, &bs, &obs, &mut fails) {
        return fails;
    }
    let aux = case.auxbasis.as_deref().map(|a| {
        let aux_bs = basis::bundled(a).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let ds = overlap_diag_error(&mol, &aux_bs, &dfbs);
        if ds > TOL_S {
            fails.push(format!("{label}: aux overlap diagonal off by {ds:.2e}"));
        }
        dfbs
    });
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &scf_config(case),
    )
    .unwrap();
    if !rhf.converged {
        fails.push(format!("{label}: SCF did not converge"));
    }
    let what = if case.kind == "dfhf" {
        "E_dfhf"
    } else {
        "E_rhf"
    };
    check_energy(&label, what, rhf.energy, case.e_rhf, &mut fails);
    if let (Some(dfbs), Some(ref_corr)) = (&aux, case.e_mp2_corr) {
        let mp2 = ri_mp2(&mol, &obs, dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        check_energy(&label, "Ec_rimp2", mp2.mp2_corr, ref_corr, &mut fails);
    }
    fails
}

fn run_kind(kind: &str) {
    let refs = refs();
    // BASIS_COVERAGE_ONLY=<substring of a case label> narrows a run (used for
    // the mutation checks below); unset, every case runs.
    let only = std::env::var("BASIS_COVERAGE_ONLY").unwrap_or_default();
    let cases: Vec<&Case> = refs
        .cases
        .iter()
        .filter(|c| c.kind == kind && c.label().contains(&only))
        .collect();
    assert!(!cases.is_empty(), "no {kind} cases in the reference file");
    let fails: Vec<String> = cases.iter().flat_map(|c| run_case(c)).collect();
    assert!(
        fails.is_empty(),
        "{} of {} {kind} cases failed:\n{}",
        fails.len(),
        cases.len(),
        fails.join("\n")
    );
}

#[test]
fn orbital_sets_match_pyscf_rhf() {
    run_kind("rhf");
}

#[test]
fn auxiliary_sets_match_pyscf_rimp2() {
    run_kind("rimp2");
}

/// J/K fitting sets in their own role: RI-J + RI-K RHF.
#[test]
fn jk_sets_match_pyscf_density_fitted_rhf() {
    run_kind("dfhf");
}

/// cc-pVQZ-PP / aug-cc-pVQZ-PP (and def2-QZVPP on the lanthanides) put h
/// (l = 5) shells on ECP atoms. The ECP integrals stop at l = 4, so that is a
/// clean error at basis setup, naming the shell and the ECP atom, instead of
/// the panic it used to be in the first hcore build.
#[test]
fn ecp_atom_with_h_shells_is_a_clean_setup_error() {
    for name in ["cc-pvqz-pp", "aug-cc-pvqz-pp"] {
        let bs = basis::bundled(name).unwrap();
        let mut mol = Molecule::parse_xyz("1\nPd\nPd 0 0 0\n", 0, 1).unwrap();
        mol.apply_ecp(&bs);
        let msg = match PreparedBasis::new(&mol, &bs) {
            Ok(_) => panic!("{name}/Pd must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("l=5") && msg.contains("ECP") && msg.contains("Pd"),
            "{msg}"
        );
        // The same family at TZ (max l = 4) is accepted.
    }
    let tz = basis::bundled("cc-pvtz-pp").unwrap();
    let mut mol = Molecule::parse_xyz("1\nPd\nPd 0 0 0\n", 0, 1).unwrap();
    mol.apply_ecp(&tz);
    assert!(
        PreparedBasis::new(&mol, &tz).is_ok(),
        "cc-pvtz-pp/Pd (max l = 4) must be accepted"
    );
}

/// Exhaustiveness: every registered set (by primary name) is exercised by at
/// least one reference case, as the orbital or the auxiliary basis.
#[test]
fn every_registered_set_has_a_case() {
    let refs = refs();
    let covered: std::collections::HashSet<&str> = refs
        .cases
        .iter()
        .flat_map(|c| std::iter::once(c.basis.as_str()).chain(c.auxbasis.as_deref()))
        .collect();
    let missing: Vec<&str> = basis::bundled_names()
        .into_iter()
        .filter(|n| !covered.contains(n))
        .collect();
    assert!(
        missing.is_empty(),
        "bundled sets with no PySCF reference case: {missing:?}"
    );
}
