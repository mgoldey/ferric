//! Validation pass for `S66_FRAGA` in `c9_driver.rs` (C9 benchmark plan, Open
//! issue #2: "not all 66 are independently spot-checked").
//!
//! For each of the 66 S66 dimers, loads the longest-range (×2.00) geometry
//! from `testdata/molecules/c9_systems/s66x8/`, builds a covalent bond graph
//! from interatomic distances (covalent-radius-sum + 1.3x tolerance), finds
//! connected components via union-find, and checks:
//!   (a) exactly 2 connected components (clean dimer split, no 3rd piece, no
//!       accidental complex-spanning bond)
//!   (b) the size of the component containing atom index 0 matches
//!       `S66_FRAGA[idx]` from `c9_driver.rs` (fragment A = mol.atoms[..a_size],
//!       so atom 0 is always in fragment A by the driver's own slicing
//!       convention — see `split_dimer` in c9_driver.rs).
//!
//! This is a pure geometric check — no SCF/integrals/BLAS involved. Cheap:
//! runs in well under a second for all 66 systems.
//!
//! Usage:
//!   cargo run -p ferric-benchmarks --example s66_fraga_validate

use ferric_core::mol::Molecule;
use std::collections::HashMap;
use std::fs;

const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_92;

/// Same table as `benchmarks/harness/examples/c9_driver.rs::S66_FRAGA`.
/// Kept as a literal copy (not shared via a lib) so this validator stays a
/// standalone, easily-diffable check against the production table — if you
/// change one, re-sync the other and note it in the commit message.
const S66_FRAGA: [usize; 67] = [
    0, // padding so index = S66 dimer #
    3, 3, 3, 3, 6, 6, 6, 6, 7, 7,
    7, 7, 12, 12, 12, 12, 12, 3, 6, 8,
    9, 8, 9, 12, 11, 12, 12, 12, 11, 12,
    12, 12, 11, 17, 17, 17, 15, 15, 12, 12,
    12, 12, 12, 6, 4, 12, 12, 11, 12, 12,
    4, 12, 12, 12, 12, 12, 12, 11, 4, 4,
    17, 17, 12, 12, 11, 7,
];

/// Covalent radii (Angstrom), Cordero et al. 2008 single-bond values. Only
/// elements that actually appear in the S66 set are needed; panic loudly if
/// we hit one that isn't tabulated so a silent 0.0 radius never masquerades
/// as "no bond".
fn covalent_radius_angstrom(z: i32) -> f64 {
    match z {
        1 => 0.31,  // H
        6 => 0.76,  // C
        7 => 0.71,  // N
        8 => 0.66,  // O
        9 => 0.57,  // F
        16 => 1.05, // S
        17 => 1.02, // Cl
        _ => panic!(
            "covalent_radius_angstrom: no tabulated radius for Z={z} \
             (extend the table before trusting this validator's output)"
        ),
    }
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Connected components of the covalent bond graph. Returns a list of
/// components, each a sorted Vec of atom indices.
fn connected_components(mol: &Molecule, tol: f64) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let ai = &mol.atoms[i];
            let aj = &mol.atoms[j];
            let dx = ai.x - aj.x;
            let dy = ai.y - aj.y;
            let dz = ai.zpos - aj.zpos;
            let r_bohr = (dx * dx + dy * dy + dz * dz).sqrt();
            let r_ang = r_bohr * BOHR_TO_ANGSTROM;
            let cutoff =
                tol * (covalent_radius_angstrom(ai.z) + covalent_radius_angstrom(aj.z));
            if r_ang < cutoff {
                uf.union(i, j);
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        groups.entry(root).or_default().push(i);
    }
    let mut comps: Vec<Vec<usize>> = groups.into_values().collect();
    for c in comps.iter_mut() {
        c.sort_unstable();
    }
    comps.sort_by_key(|c| c[0]);
    comps
}

fn parse_s66_dimer_index(name: &str) -> Option<usize> {
    name.strip_prefix("s66_")?.split('_').next()?.parse().ok()
}

fn main() {
    let dir = "testdata/molecules/c9_systems/s66x8";
    const TOL: f64 = 1.3;

    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .flatten()
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with("_2.00.xyz"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    println!(
        "{:<45} {:>5} {:>6} {:>8} {:>8} {:>10} {:>10}",
        "name", "idx", "natom", "n_comp", "tbl_A", "found_A", "verdict"
    );

    let mut n_pass = 0;
    let mut n_fail_ncomp = 0;
    let mut n_fail_mismatch = 0;
    let mut mismatches: Vec<(usize, String, usize, usize, usize)> = Vec::new();

    for entry in &entries {
        let stem = entry
            .path()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let idx = match parse_s66_dimer_index(&stem) {
            Some(i) if (1..=66).contains(&i) => i,
            _ => {
                eprintln!("  cannot parse S66 index from {stem}, skipping");
                continue;
            }
        };
        let mol = match Molecule::load_xyz(entry.path().to_str().unwrap()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  load failed for {stem}: {e}");
                continue;
            }
        };
        let n_atoms = mol.atoms.len();
        let comps = connected_components(&mol, TOL);
        let n_comp = comps.len();
        // Fragment A anchors on atom index 0 per c9_driver.rs::split_dimer
        // (mol.atoms[..a_size] is fragment A).
        let comp_a = comps.iter().find(|c| c.contains(&0)).unwrap();
        let found_a = comp_a.len();
        let table_a = S66_FRAGA[idx];

        let verdict = if n_comp != 2 {
            n_fail_ncomp += 1;
            "FAIL(ncomp)"
        } else if found_a != table_a {
            n_fail_mismatch += 1;
            mismatches.push((idx, stem.clone(), table_a, found_a, n_atoms));
            "MISMATCH"
        } else {
            n_pass += 1;
            "pass"
        };

        println!(
            "{:<45} {:>5} {:>6} {:>8} {:>8} {:>10} {:>10}",
            stem, idx, n_atoms, n_comp, table_a, found_a, verdict
        );

        if n_comp != 2 {
            eprintln!(
                "    -> component sizes: {:?}",
                comps.iter().map(|c| c.len()).collect::<Vec<_>>()
            );
        }
    }

    println!();
    println!(
        "=== s66_fraga_validate: {n_pass} pass, {n_fail_mismatch} mismatch, {n_fail_ncomp} bad-ncomp (of {} systems) ===",
        entries.len()
    );

    if !mismatches.is_empty() {
        println!();
        println!("Mismatches (table S66_FRAGA[idx] vs geometrically-found fragment-A size):");
        for (idx, stem, table_a, found_a, n_atoms) in &mismatches {
            println!(
                "  #{idx:>2} {stem:<40} table_A={table_a:>2}  found_A={found_a:>2}  found_B={:>2}  total={n_atoms}",
                n_atoms - found_a
            );
        }
    }
}
