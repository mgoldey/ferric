//! Shared harness for the bundled-basis-set checks against PySCF
//! (`basis_coverage_vs_pyscf.rs`, the per-commit subset and the data-only
//! coverage guards; `validation_basis_coverage.rs`, the full sweep). See
//! `validation_basis_coverage.rs` for what is compared and the mutation record.

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
pub struct Refs {
    pub cases: Vec<Case>,
}

#[derive(Deserialize)]
pub struct Case {
    pub kind: String,
    pub basis: String,
    pub auxbasis: Option<String>,
    pub molecule: String,
    pub atoms_angstrom: String,
    pub charge: i32,
    pub cart: bool,
    pub ecp: bool,
    pub nao: usize,
    pub nelectron: i32,
    pub e_rhf: f64,
    pub e_mp2_corr: Option<f64>,
}

impl Case {
    pub fn label(&self) -> String {
        match &self.auxbasis {
            Some(a) => format!("{}+{}/{}", self.basis, a, self.molecule),
            None => format!("{}/{}", self.basis, self.molecule),
        }
    }
}

pub fn refs() -> Refs {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/basis_coverage_pyscf.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn molecule(case: &Case) -> Molecule {
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

/// Run every `kind` case, or (with `labels`) exactly the named ones, and
/// fail with every mismatch listed. A named label missing from the reference
/// file is itself a failure, so a renamed case cannot shrink a subset to
/// nothing.
pub fn run_cases(kind: &str, labels: Option<&[&str]>) {
    let refs = refs();
    // BASIS_COVERAGE_ONLY=<substring of a case label> narrows a run (used for
    // the mutation checks in validation_basis_coverage.rs); unset, every
    // selected case runs.
    let only = std::env::var("BASIS_COVERAGE_ONLY").unwrap_or_default();
    let cases: Vec<&Case> = refs
        .cases
        .iter()
        .filter(|c| c.kind == kind && c.label().contains(&only))
        .filter(|c| labels.is_none_or(|ls| ls.contains(&c.label().as_str())))
        .collect();
    if let (Some(ls), true) = (labels, only.is_empty()) {
        let found: Vec<String> = cases.iter().map(|c| c.label()).collect();
        let missing: Vec<&&str> = ls
            .iter()
            .filter(|l| !found.iter().any(|f| f == *l))
            .collect();
        assert!(
            missing.is_empty(),
            "{kind} cases not in the reference file: {missing:?}"
        );
    }
    assert!(!cases.is_empty(), "no {kind} cases selected");
    let fails: Vec<String> = cases.iter().flat_map(|c| run_case(c)).collect();
    assert!(
        fails.is_empty(),
        "{} of {} {kind} cases failed:\n{}",
        fails.len(),
        cases.len(),
        fails.join("\n")
    );
}
