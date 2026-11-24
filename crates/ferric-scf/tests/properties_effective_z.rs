//! Atomic properties pair the density with the SCREENED nuclear charge
//! `Atom::effective_z()` (`Z - n_core_ecp`, 0 for a ghost), not the bare `Z`.
//!
//! Found by the book's chapter-27 audit: `esp_at_atoms`, `electric_field_at_atoms`,
//! `esp_at_points`, `becke_charges`, `lowdin_charges`, `mulliken_charges` and
//! `hirshfeld_charges` used `atom.z`; only `dipole_moment` used `effective_z()`.
//! On an ECP atom the density holds valence electrons only, so the bare `Z`
//! double-counts the core; a ghost atom counted as a full nucleus.
//!
//! # Tests
//!
//! 1. ZERO-DENSITY ANALYTIC ANCHORS (no SCF). With `D = 0` every property is
//!    its pure nuclear term, known in closed form from the molecule's
//!    `effective_z()` list: `V(R_A) = sum_{B!=A} Zeff_B/r`, the field
//!    `sum Zeff_B (R_A-R_B)/r^3`, `V(r)` at free points, and the population
//!    charges `q_A = Zeff_A`. Artifact hypothesis: if a function still reads
//!    `atom.z`, the ECP iodine (Z=53 vs 25) and the ghost (Z=53 vs 0) make it
//!    miss by O(1) in every one of these quantities.
//! 2. PYSCF CHAIN on HI and CH3I, def2-SVP (I carries the def2 28-core ECP),
//!    `scripts/validation/gen_ecp_properties.py` ->
//!    `testdata/reference/validation/ecp_properties/`: ferric's own SCF
//!    density -> ESP at the nuclei, Mulliken and Lowdin charges vs PySCF's
//!    SCF + ECP.
//!
//! # Bars
//!
//! * Zero density: arithmetic only (a few ulp of the 1/r sums), 1e-12.
//! * PySCF chain: the SCF energy differs by SCF/ECP-integral noise
//!   (`validation_ecp.rs`: 2.86e-8 Ha max); first-order properties follow.
//!   Measured maxima and the bar are recorded next to [`TOL_CHAIN`].

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::{
    becke_charges, electric_field_at_atoms, esp_at_atoms, esp_at_points, hirshfeld_charges,
    lowdin_charges, mulliken_charges,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const TOL_ZERO_D: f64 = 1e-12;

fn def2_svp() -> BasisSet {
    basis::bundled("def2-svp").unwrap()
}

fn molecule(xyz: &str, bs: &BasisSet) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    mol.apply_ecp(bs);
    mol
}

fn pos(mol: &Molecule) -> Vec<[f64; 3]> {
    mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Closed-form nuclear ESP at nucleus `a`, from `effective_z()`.
fn nuclear_esp_at_atom(mol: &Molecule, a: usize) -> f64 {
    let p = pos(mol);
    (0..p.len())
        .filter(|&b| b != a)
        .map(|b| mol.atoms[b].effective_z() as f64 / dist(p[a], p[b]))
        .sum()
}

/// HI (I carries the 28-electron ECP) with a ghost iodine on the far side.
/// Zeff = [1, 25, 0].
const HI_GHOST_I: &str = "3\nHI + ghost I\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n@I 0.0 0.0 -1.5\n";
const CH3I: &str = "5\nCH3I\nC 0.0 0.0 0.0\nI 0.0 0.0 2.14\nH 1.03 0.0 -0.36\nH -0.51 0.89 -0.36\nH -0.51 -0.89 -0.36\n";

fn check_zero_density_properties(xyz: &str, expect_zeff: &[i32]) {
    let bs = def2_svp();
    let mol = molecule(xyz, &bs);
    let zeff: Vec<i32> = mol.atoms.iter().map(|a| a.effective_z()).collect();
    // TEETH: the fixture must make the bare and the screened charge differ.
    assert_eq!(zeff, expect_zeff, "fixture effective charges");
    assert!(
        mol.atoms.iter().any(|a| a.effective_z() != a.z),
        "fixture must carry an ECP or a ghost"
    );
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    let d0 = Array2::<f64>::zeros((nbf, nbf));
    let p = pos(&mol);
    let natoms = p.len();

    let esp = esp_at_atoms(&mol, &prep, &d0).unwrap();
    for a in 0..natoms {
        let want = nuclear_esp_at_atom(&mol, a);
        assert!(
            (esp[a] - want).abs() < TOL_ZERO_D,
            "esp_at_atoms[{a}] = {} vs Zeff sum {want}",
            esp[a]
        );
    }

    let field = electric_field_at_atoms(&mol, &prep, &d0).unwrap();
    for a in 0..natoms {
        let mut want = [0.0_f64; 3];
        for b in (0..natoms).filter(|&b| b != a) {
            let r = dist(p[a], p[b]);
            for k in 0..3 {
                want[k] += mol.atoms[b].effective_z() as f64 * (p[a][k] - p[b][k]) / r.powi(3);
            }
        }
        for k in 0..3 {
            assert!(
                (field[a][k] - want[k]).abs() < TOL_ZERO_D,
                "electric_field_at_atoms[{a}][{k}] = {} vs {}",
                field[a][k],
                want[k]
            );
        }
    }

    let pts = [[0.7, -0.4, 0.9], [0.1, 1.3, -2.0], [-3.0, 0.5, 4.0]];
    let v = esp_at_points(&mol, &prep, &d0, &pts).unwrap();
    for (i, pt) in pts.iter().enumerate() {
        let want: f64 = (0..natoms)
            .map(|b| mol.atoms[b].effective_z() as f64 / dist(*pt, p[b]))
            .sum();
        assert!(
            (v[i] - want).abs() < TOL_ZERO_D,
            "esp_at_points[{i}] = {} vs {want}",
            v[i]
        );
    }

    // Population charges of an empty density are the screened nuclear charges.
    let charges: [(&str, Vec<f64>); 4] = [
        ("lowdin", lowdin_charges(&mol, &prep, &d0).unwrap()),
        ("mulliken", mulliken_charges(&mol, &prep, &d0).unwrap()),
        ("becke", becke_charges(&mol, &prep, &bs, &d0).unwrap()),
        (
            "hirshfeld",
            hirshfeld_charges(&mol, &bs, &d0, None).unwrap(),
        ),
    ];
    for (name, q) in &charges {
        for a in 0..natoms {
            assert!(
                (q[a] - zeff[a] as f64).abs() < TOL_ZERO_D,
                "{name}_charges[{a}] = {} vs Zeff {}",
                q[a],
                zeff[a]
            );
        }
    }
}

#[test]
fn ecp_atom_and_ghost_use_the_screened_charge_at_zero_density() {
    check_zero_density_properties(HI_GHOST_I, &[1, 25, 0]);
}

#[test]
fn ecp_methyl_iodide_uses_the_screened_charge_at_zero_density() {
    check_zero_density_properties(CH3I, &[6, 25, 1, 1, 1]);
}

// ---------------------------------------------------------------------------
// PySCF chain
// ---------------------------------------------------------------------------

/// Measured max |ferric - PySCF| over ESP at the nuclei, Mulliken and Lowdin
/// charges, ferric's own SCF density (`density_conv` 1e-9, `energy_conv` 1e-11)
/// against PySCF's (2026-11-15): HI 1.3e-10, CH3I 1.8e-9. The bar is ~10x the
/// larger, the SCF/ECP-integral floor (`validation_ecp.rs`: 2.86e-8 Ha energy).
/// The bare-Z mistake misses by tens of a.u., ~10 orders above it.
const TOL_CHAIN: f64 = 2e-8;

fn reference(system: &str) -> Value {
    let path = format!(
        "{}/../../testdata/reference/validation/ecp_properties/{system}_def2-svp.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {path} ({e}); regenerate with \
             scripts/validation/gen_ecp_properties.py"
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn vec1(v: &Value, key: &str) -> Vec<f64> {
    v["properties"][key]
        .as_array()
        .unwrap_or_else(|| panic!("reference key {key} missing"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn pyscf_chain(system: &str) {
    let r = reference(system);
    let xyz = format!(
        "{}/../../testdata/molecules/validation/ecp/{system}.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    let bs = def2_svp();
    let mut mol = Molecule::load_xyz(&xyz).unwrap();
    mol.apply_ecp(&bs);
    let ncore: i32 = mol.atoms.iter().map(|a| a.n_core_ecp).sum();
    assert_eq!(ncore, 28, "I carries the def2 28-electron ECP");
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig::default()
            .with_energy_conv(1e-11)
            .with_density_conv(1e-9)
            .with_max_iter(200),
    )
    .unwrap();
    assert!(rhf.converged);
    let d = rhf.density_total();

    let zref = vec1(&r, "effective_z");
    for (a, z) in zref.iter().enumerate() {
        assert_eq!(mol.atoms[a].effective_z() as f64, *z, "Zeff[{a}]");
    }
    let mut worst = 0.0_f64;
    let mut check = |name: &str, got: Vec<f64>, want: Vec<f64>| {
        for a in 0..want.len() {
            let dlt = (got[a] - want[a]).abs();
            worst = worst.max(dlt);
            println!(
                "{system} {name}[{a}] ferric {:.8} pyscf {:.8} d {dlt:.2e}",
                got[a], want[a]
            );
            assert!(
                dlt < TOL_CHAIN,
                "{system} {name}[{a}]: {} vs {}",
                got[a],
                want[a]
            );
        }
    };
    check(
        "esp_at_atoms",
        esp_at_atoms(&mol, &prep, d).unwrap(),
        vec1(&r, "esp_at_atoms"),
    );
    check(
        "mulliken",
        mulliken_charges(&mol, &prep, d).unwrap(),
        vec1(&r, "mulliken_charges"),
    );
    check(
        "lowdin",
        lowdin_charges(&mol, &prep, d).unwrap(),
        vec1(&r, "lowdin_charges"),
    );
    println!("{system} worst |d| = {worst:.3e}");
}

#[test]
fn hi_ecp_properties_match_pyscf() {
    pyscf_chain("hi");
}

#[test]
fn ch3i_ecp_properties_match_pyscf() {
    pyscf_chain("ch3i");
}
