//! C9 benchmark driver — RHF + RI-RPA at cc-pVDZ / cc-pVDZ-RI for the three
//! tiers of the danuglipron-systems benchmark suite:
//!   * `s66x8`        — Rezac/Hobza 66-dimer × 8-distance dispersion benchmark
//!   * `l7`           — Sedlak 2013 large noncovalent complexes
//!   * `danuglipron`  — Pfizer GLP-1R agonist conformer ensemble (single-mol)
//!
//! Outputs one CSV row per system to stdout (and optionally to a file via
//! `FERRIC_C9_OUTPUT_CSV`) with timings, energies, interaction energy and Δ
//! versus the published reference. For dimers in S66x8/L7 the driver runs
//! three SCF+RPA evaluations per row (complex, monA, monB) and reports
//! `E_int = E(complex) - E(monA) - E(monB)`.
//!
//! Usage:
//!   cargo run --release -p ferric-rpa --example c9_driver -- <tier> [--only <names>] [--cp]
//! Examples:
//!   c9_driver -- s66x8 --only s66_01_waterwater_1.00,s66_24_benzenebenzenepipi_1.00
//!   c9_driver -- s66x8 --only s66_01_waterwater_1.00 --cp   (also reports CP-corrected E_int)
//!   c9_driver -- l7
//!   c9_driver -- danuglipron
//!
//! `--cp` (s66x8/l7-dimer rows only; not wired for L7's GGG/PHE 3-body rows)
//! adds a Boys-Bernardi counterpoise-corrected `e_int_cp_kcalmol` CSV column
//! alongside the existing uncorrected `e_int_kcalmol` -- both are reported,
//! neither replaces the other. Each monomer is re-evaluated in the full
//! dimer's AO basis via ghost atoms (`ferric_core::mol::Atom::ghost`,
//! `@`-prefixed in XYZ) standing in for the other fragment's basis functions
//! with zero nuclear charge/electrons.
//!
//! Environment:
//!   FERRIC_C9_ONLY        comma-separated system names (alias of --only)
//!   FERRIC_C9_OUTPUT_CSV  path to also append CSV lines to
//!   FERRIC_C9_CP          set (any value) to enable --cp
//!   OPENBLAS_NUM_THREADS  recommend =1 (see sparse_scaling.rs)
//!
//! THIS IS A SMOKE-TEST DRIVER. Don't run the full 528-system sweep in CI; use
//! it to verify the pipeline end-to-end on a small subset.

use ferric_core::basis;
use ferric_core::error::FerricError;
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Sparsity, PdepRpaConfig};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const HA_TO_KCAL: f64 = 627.509_474_06;
const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_92;

/// S66 fragment-A heavy+H atom count for each S66 dimer index (1..66).
/// Extracted from qcdb's `databases/S66.py` via the `--` fragment separator.
/// (See `scripts/fetch_s66x8.py` for the extraction script.)
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

/// Reference interaction energies in kcal/mol (CCSD(T)/CBS CP from BEGDB).
fn load_refs(path: &str) -> HashMap<String, f64> {
    let text = fs::read_to_string(path).unwrap_or_default();
    serde_json::from_str(&text).unwrap_or_default()
}

#[derive(Debug, Clone)]
struct Csv {
    name: String,
    n_atoms: usize,
    n_ao: usize,
    n_aux: usize,
    e_rhf: f64,
    e_rpa: f64,
    t_rhf_s: f64,
    t_rpa_s: f64,
    e_int_kcalmol: Option<f64>,
    e_int_ref_kcalmol: Option<f64>,
    status: String,
    /// Boys-Bernardi counterpoise-corrected interaction energy, kcal/mol.
    /// `None` unless the driver was run with `--cp` (a NEW, additive column
    /// -- see run_dimer's `cp` argument; the existing uncorrected `e_int_kcalmol` above is
    /// untouched so already-reported/cited numbers elsewhere keep meaning).
    e_int_cp_kcalmol: Option<f64>,
}

impl Csv {
    fn header() -> String {
        "name,n_atoms,n_ao,n_aux,e_rhf,e_rpa,t_rhf_s,t_rpa_s,e_int_kcalmol,e_int_ref_kcalmol,delta_kcalmol,e_int_cp_kcalmol,delta_cp_kcalmol,status".to_string()
    }
    fn line(&self) -> String {
        let e_int = self.e_int_kcalmol.map(|v| format!("{v:.4}")).unwrap_or_default();
        let e_int_ref = self.e_int_ref_kcalmol.map(|v| format!("{v:.4}")).unwrap_or_default();
        let delta = match (self.e_int_kcalmol, self.e_int_ref_kcalmol) {
            (Some(a), Some(b)) => format!("{:.4}", a - b),
            _ => String::new(),
        };
        let e_int_cp = self.e_int_cp_kcalmol.map(|v| format!("{v:.4}")).unwrap_or_default();
        let delta_cp = match (self.e_int_cp_kcalmol, self.e_int_ref_kcalmol) {
            (Some(a), Some(b)) => format!("{:.4}", a - b),
            _ => String::new(),
        };
        format!(
            "{},{},{},{},{:.10},{:.10},{:.3},{:.3},{},{},{},{},{},{}",
            self.name, self.n_atoms, self.n_ao, self.n_aux,
            self.e_rhf, self.e_rpa, self.t_rhf_s, self.t_rpa_s,
            e_int, e_int_ref, delta, e_int_cp, delta_cp, self.status,
        )
    }
}

struct CalcResult {
    e_rhf: f64,
    e_rpa: f64,
    n_ao: usize,
    n_aux: usize,
    t_rhf: f64,
    t_rpa: f64,
}

fn frozen_core_for(mol: &Molecule) -> usize {
    // Frozen-core convention: 1 FC orbital per atom with Z >= 3 (Li and heavier).
    // Ghost atoms (counterpoise basis-only centers) contribute zero occupied
    // orbitals -- they must NOT be counted here, or frozen_core would exceed
    // the fragment's real nocc and active_occ() would underflow/error.
    mol.atoms.iter().filter(|a| !a.ghost && a.z >= 3).count()
}

fn run_rhf_rpa(
    ctx: &ParallelContext,
    mol: &Molecule,
    label: &str,
) -> Result<CalcResult, FerricError> {
    let obs_set = basis::bundled("cc-pvdz")?;
    let dfbs_set = basis::bundled("cc-pvdz-ri")?;
    let obs = PreparedBasis::new(mol, &obs_set)?;
    let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
    let n_ao = obs.nbasis();
    let n_aux = dfbs.nbasis();

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs)?;

    // Production SCF: DF-J + DF-K with def2-universal-jkfit (proper JK-fit aux).
    // Reusing cc-pvdz-ri (MP2-fit aux) for K introduces mHa-scale error and
    // was the source of the ~6e-4 Ha ferric-vs-PySCF E_total gap observed in
    // the C9 prep smoke test. See [[ferric-jk-aux-convention]] for context.
    // RPA correlation still uses cc-pvdz-ri (the aux argument below).
    let rhf_cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-7,
        density_conv: 1e-6,
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };

    let t0 = Instant::now();
    let rhf = solve_rhf(ctx, mol, &obs, op, &bounds, &rhf_cfg)?;
    let t_rhf = t0.elapsed().as_secs_f64();
    eprintln!("  [{label}] RHF: n_AO={n_ao} n_aux={n_aux} E={:.8} t={:.2}s",
              rhf.energy, t_rhf);

    let pdep_cfg = PdepRpaConfig {
        frozen_core: frozen_core_for(mol),
        trunc_thresh: 1e-4,
        eigensolver_conv_thresh: 1e-6,
        chi0_sparsity: Chi0Sparsity::Dense,
        ..Default::default()
    };

    let t1 = Instant::now();
    let rpa = run_pdep_rpa(mol, &obs, &dfbs, op, &rhf, &pdep_cfg)?;
    let t_rpa = t1.elapsed().as_secs_f64();
    eprintln!("  [{label}] RPA: E_corr={:.8} E_tot={:.8} t={:.2}s",
              rpa.e_rpa, rhf.energy + rpa.e_rpa, t_rpa);

    Ok(CalcResult {
        e_rhf: rhf.energy,
        e_rpa: rhf.energy + rpa.e_rpa,
        n_ao,
        n_aux,
        t_rhf,
        t_rpa,
    })
}

/// Split a Molecule into two fragments at index `a_size` (first `a_size`
/// atoms in fragA, remainder in fragB). Coordinates remain in Bohr.
fn split_dimer(mol: &Molecule, a_size: usize) -> (Molecule, Molecule) {
    let atoms_a: Vec<Atom> = mol.atoms[..a_size].to_vec();
    let atoms_b: Vec<Atom> = mol.atoms[a_size..].to_vec();
    let mol_a = Molecule { atoms: atoms_a, charge: 0, multiplicity: 1 };
    let mol_b = Molecule { atoms: atoms_b, charge: 0, multiplicity: 1 };
    (mol_a, mol_b)
}

/// Split a Molecule into three fragments given explicit 0-based atom index
/// lists (one per fragment). Unlike `split_dimer`, fragments need not be
/// contiguous ranges: L7's PHE trimer interleaves its three ~29-atom capped
/// phenylalanine-residue fragments in the source XYZ (verified geometrically
/// via covalent-bond connected components -- see l7_fraga_size doc comment),
/// so a simple `[..n]`/`[n..]` slice cannot separate them. GGG's three
/// guanine fragments ARE contiguous 16-atom blocks and work fine through
/// this same index-list path (indices happen to be a contiguous range).
/// Coordinates remain in Bohr; every index must appear in exactly one of the
/// three lists and every atom of `mol` must be covered (checked by caller).
fn split_trimer(mol: &Molecule, idx_a: &[usize], idx_b: &[usize], idx_c: &[usize]) -> (Molecule, Molecule, Molecule) {
    let pick = |idxs: &[usize]| -> Molecule {
        let atoms: Vec<Atom> = idxs.iter().map(|&i| mol.atoms[i].clone()).collect();
        Molecule { atoms, charge: 0, multiplicity: 1 }
    };
    (pick(idx_a), pick(idx_b), pick(idx_c))
}

/// Build a Boys-Bernardi counterpoise fragment: atoms at `own_indices` stay
/// real, every other atom of `mol` (the full complex) becomes a ghost --
/// same element (so `for_element(z)` picks up its basis functions per the
/// `Atom`/`ghost` doc comment in `ferric_core::mol`), same position, but
/// `ghost: true` so it contributes zero nuclear charge and zero electrons
/// (see `Atom::effective_z`, `Molecule::nelec`/`nuclear_repulsion`, and
/// `basis_bridge.rs`'s `z_eff` handling -- all already ghost-aware, exercised
/// by `crates/ferric-scf/tests/ghost_atoms.rs`). The result is fragment A (or
/// B) evaluated in the FULL complex's AO basis at the FULL complex's
/// geometry, per the standard CP prescription: E(A, basis=AB).
fn make_cp_fragment(mol: &Molecule, own_indices: &[usize]) -> Molecule {
    let own: HashSet<usize> = own_indices.iter().copied().collect();
    let atoms: Vec<Atom> = mol.atoms.iter().enumerate().map(|(i, a)| {
        if own.contains(&i) {
            a.clone()
        } else {
            Atom { ghost: true, ..a.clone() }
        }
    }).collect();
    Molecule { atoms, charge: 0, multiplicity: 1 }
}

fn parse_s66_dimer_index(name: &str) -> Option<usize> {
    // Expected name: s66_NN_<body>_<dist>
    name.strip_prefix("s66_")?.split('_').next()?.parse().ok()
}

fn run_s66x8(ctx: &ParallelContext, only: &Option<HashSet<String>>, cp: bool) -> Vec<Csv> {
    let dir = "testdata/molecules/c9_systems/s66x8";
    let refs = load_refs("testdata/reference/c9_refs/s66x8_ccsdt_cbs.json");
    let mut rows = Vec::new();
    let entries: Vec<_> = match fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<_> = rd.flatten()
                .filter(|e| e.path().extension().map(|x| x == "xyz").unwrap_or(false))
                .collect();
            v.sort_by_key(|e| e.file_name());
            v
        }
        Err(e) => {
            eprintln!("s66x8 dir missing: {e}");
            return rows;
        }
    };
    for entry in entries {
        let stem = entry.path().file_stem().unwrap().to_string_lossy().to_string();
        if let Some(set) = only.as_ref() {
            if !set.contains(&stem) { continue; }
        }
        eprintln!("\n=== {stem} ===");
        let mol = match Molecule::load_xyz(entry.path().to_str().unwrap()) {
            Ok(m) => m,
            Err(e) => { eprintln!("  load failed: {e}"); continue; }
        };
        let idx = match parse_s66_dimer_index(&stem) {
            Some(i) if (1..=66).contains(&i) => i,
            _ => { eprintln!("  cannot parse S66 dimer index from {stem}"); continue; }
        };
        let a_size = S66_FRAGA[idx];
        if a_size == 0 || a_size >= mol.atoms.len() {
            eprintln!("  bad fragment-A size {a_size} for {stem} (total={})",
                      mol.atoms.len());
            continue;
        }
        rows.push(run_dimer(ctx, &stem, &mol, a_size, &refs, cp));
    }
    rows
}

/// L7 fragment-A sizes (atoms in the first molecular fragment as concatenated
/// in the BEGDB L7 XYZ files). Determined by inspecting each XYZ at long
/// separation — for L7 the published "supersystem" geometry is a single
/// arrangement, not a dissociation curve, so the split is fixed per system.
fn l7_fraga_size(name: &str) -> Option<usize> {
    // The L7 paper's monomer atom counts:
    //   C2C2PD: octadecane(C18H38) + octadecane(C18H38) -> 56 + 56 = 112
    //   GGG:    guanine + guanine + guanine -> 16+16+16=48 (3 frags, not 2)
    //   C3A:    circumcoronene(C54H18) + adenine(C5H5N5)  -> 72 + 15 = 87
    //   C3GC:   circumcoronene + GC base pair -> 72 + 28 = 100 (heavy 21)
    //   CBH:    coronene(C24H12) + coronene(C24H12) -> 36 + 36 = 72
    //   GCGC:   GC + GC -> 29 + 29 = 58
    //   PHE:    capped-Phe-residue trimer -> 29+29+29=87 (3 frags, not 2;
    //           see l7_trimer_indices doc comment -- the 23-atom free-amino-
    //           acid count in an earlier version of this comment was WRONG,
    //           verified by covalent-bond connected-component clustering)
    // The 3-fragment systems (GGG, PHE) cannot be split into two monomers
    // for a 2-body interaction energy — they need l7_trimer_indices below.
    match name {
        "C2C2PD" => Some(56),
        "C3A"    => Some(72),  // circumcoronene fragment is first
        "C3GC"   => Some(72),
        "CBH"    => Some(36),
        "GCGC"   => Some(29),
        _ => None, // GGG, PHE: 3-body — handled by l7_trimer_indices
    }
}

/// 0-based atom index lists for L7's two 3-fragment (trimer) systems, one
/// `Vec<usize>` per fragment. Determined by covalent-bond connected-component
/// clustering (Cordero covalent radii, 1.3x sum tolerance, union-find) on the
/// actual downloaded XYZ files in `testdata/molecules/c9_systems/l7/` --- NOT
/// by trusting the L7 paper's summary atom counts blindly:
///
///   GGG (48 atoms total): 3 components of 16 atoms each, EACH an exact
///   contiguous range: [0..16), [16..32), [32..48). Formula per fragment
///   C5H5N5O (guanine) confirms chemical sense. Nearest inter-fragment
///   contact >= 3.2 Angstrom (vs ~1.0-1.5 Angstrom intra-fragment bonds), so
///   the boundary is a clean non-bonded gap, matching split_dimer's existing
///   "long separation" sanity convention.
///
///   PHE (87 atoms total): 3 components of 29 atoms each (NOT 23 -- the
///   c9-benchmark-plan.md comment assuming free phenylalanine, C9H11NO2 x3
///   = 69, was WRONG; each fragment is actually a capped Phe RESIDUE,
///   C11H14N2O2, matching the source file's own comment
///   "phenylalanineresiduestrimer"). Critically, only fragment 1 is a
///   contiguous range ([0..29)); fragments 2 and 3 are INTERLEAVED in the
///   file (their atoms are grouped by element/role across both residues,
///   not by residue) -- confirmed by union-find clustering, not assumed.
///   split_trimer's index-list signature (not split_dimer's `[..n]` slice)
///   is required for PHE for exactly this reason.
fn l7_trimer_indices(name: &str) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    match name {
        "GGG" => Some(((0..16).collect(), (16..32).collect(), (32..48).collect())),
        "PHE" => Some((
            (0..29).collect(),
            vec![
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 51, 52, 55, 56, 62, 63, 64, 76, 77,
                78, 79, 80, 81, 82, 83, 84, 85, 86,
            ],
            vec![
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 57, 58, 59, 60, 61, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75,
            ],
        )),
        _ => None,
    }
}

fn run_l7(ctx: &ParallelContext, only: &Option<HashSet<String>>, cp: bool) -> Vec<Csv> {
    let dir = "testdata/molecules/c9_systems/l7";
    let refs = load_refs("testdata/reference/c9_refs/l7_qcisdt_cbs.json");
    let mut rows = Vec::new();
    let entries: Vec<_> = match fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<_> = rd.flatten()
                .filter(|e| e.path().extension().map(|x| x == "xyz").unwrap_or(false))
                .collect();
            v.sort_by_key(|e| e.file_name());
            v
        }
        Err(e) => { eprintln!("l7 dir missing: {e}"); return rows; }
    };
    for entry in entries {
        let stem = entry.path().file_stem().unwrap().to_string_lossy().to_string();
        if let Some(set) = only.as_ref() {
            if !set.contains(&stem) { continue; }
        }
        eprintln!("\n=== L7 {stem} ===");
        let mol = match Molecule::load_xyz(entry.path().to_str().unwrap()) {
            Ok(m) => m,
            Err(e) => { eprintln!("  load failed: {e}"); continue; }
        };
        if let Some(a_size) = l7_fraga_size(&stem) {
            if a_size < mol.atoms.len() {
                rows.push(run_dimer(ctx, &stem, &mol, a_size, &refs, cp));
                continue;
            }
        }
        if let Some((idx_a, idx_b, idx_c)) = l7_trimer_indices(&stem) {
            // NOTE: trimer CP (3-fragment Boys-Bernardi, ghosting the OTHER
            // TWO fragments per monomer evaluation) is not implemented in
            // this pass -- run_trimer always reports uncorrected E_int. `cp`
            // is intentionally unused here (not silently dropped: this is
            // the documented scope boundary, see driver module doc).
            let _ = cp;
            rows.push(run_trimer(ctx, &stem, &mol, &idx_a, &idx_b, &idx_c, &refs));
            continue;
        }
        eprintln!("  {stem}: no known fragment split; running complex only");
        rows.push(run_complex_only(ctx, &stem, &mol, &refs));
    }
    rows
}

fn run_danuglipron(ctx: &ParallelContext, only: &Option<HashSet<String>>) -> Vec<Csv> {
    let dir = "testdata/molecules/c9_systems/danuglipron";
    let mut rows = Vec::new();
    let entries: Vec<_> = match fs::read_dir(dir) {
        Ok(rd) => {
            let mut v: Vec<_> = rd.flatten()
                .filter(|e| e.path().extension().map(|x| x == "xyz").unwrap_or(false))
                .collect();
            v.sort_by_key(|e| e.file_name());
            v
        }
        Err(e) => { eprintln!("danuglipron dir missing: {e}"); return rows; }
    };
    for entry in entries {
        let stem = entry.path().file_stem().unwrap().to_string_lossy().to_string();
        if let Some(set) = only.as_ref() {
            if !set.contains(&stem) { continue; }
        }
        eprintln!("\n=== danuglipron {stem} ===");
        let mol = match Molecule::load_xyz(entry.path().to_str().unwrap()) {
            Ok(m) => m,
            Err(e) => { eprintln!("  load failed: {e}"); continue; }
        };
        // No interaction energy for a single molecule — record total RHF+RPA.
        rows.push(run_complex_only(ctx, &stem, &mol, &HashMap::new()));
    }
    rows
}

fn run_complex_only(
    ctx: &ParallelContext, name: &str, mol: &Molecule, refs: &HashMap<String, f64>,
) -> Csv {
    let n = mol.atoms.len();
    match run_rhf_rpa(ctx, mol, "complex") {
        Ok(c) => Csv {
            name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
            e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
            e_int_kcalmol: None,
            e_int_ref_kcalmol: refs.get(name).copied(),
            status: "OK".into(),
        e_int_cp_kcalmol: None,
        },
        Err(e) => {
            eprintln!("  FAIL: {e}");
            Csv {
                name: name.into(), n_atoms: n, n_ao: 0, n_aux: 0,
                e_rhf: 0.0, e_rpa: 0.0, t_rhf_s: 0.0, t_rpa_s: 0.0,
                e_int_kcalmol: None,
                e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL:{e}"),
                e_int_cp_kcalmol: None,
            }
        }
    }
}

fn run_dimer(
    ctx: &ParallelContext, name: &str, mol: &Molecule, a_size: usize,
    refs: &HashMap<String, f64>, cp: bool,
) -> Csv {
    let n = mol.atoms.len();
    let (mol_a, mol_b) = split_dimer(mol, a_size);
    eprintln!("  fragments: A={} atoms, B={} atoms", mol_a.atoms.len(), mol_b.atoms.len());

    let c = match run_rhf_rpa(ctx, mol, "complex") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  complex FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: 0, n_aux: 0,
                e_rhf: 0.0, e_rpa: 0.0, t_rhf_s: 0.0, t_rpa_s: 0.0,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                e_int_cp_kcalmol: None,
                status: format!("FAIL_complex:{e}"),
            };
        }
    };
    let a = match run_rhf_rpa(ctx, &mol_a, "monA") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  monA FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
                e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                e_int_cp_kcalmol: None,
                status: format!("FAIL_monA:{e}"),
            };
        }
    };
    let b = match run_rhf_rpa(ctx, &mol_b, "monB") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  monB FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
                e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                e_int_cp_kcalmol: None,
                status: format!("FAIL_monB:{e}"),
            };
        }
    };
    let e_int_ha = c.e_rpa - a.e_rpa - b.e_rpa;
    let e_int_kcal = e_int_ha * HA_TO_KCAL;
    eprintln!("  E_int = {:.4} kcal/mol", e_int_kcal);

    let mut t_rhf_s = c.t_rhf + a.t_rhf + b.t_rhf;
    let mut t_rpa_s = c.t_rpa + a.t_rpa + b.t_rpa;
    let mut e_int_cp_kcalmol = None;
    let mut status = "OK".to_string();

    if cp {
        // Boys-Bernardi CP: monomers re-evaluated in the FULL dimer basis
        // (ghost atoms standing in for the other fragment). Complex energy
        // `c` is unchanged (it already uses the full dimer basis natively).
        let idx_a: Vec<usize> = (0..a_size).collect();
        let idx_b: Vec<usize> = (a_size..n).collect();
        let mol_a_gh = make_cp_fragment(mol, &idx_a);
        let mol_b_gh = make_cp_fragment(mol, &idx_b);
        match (run_rhf_rpa(ctx, &mol_a_gh, "monA+ghostB"), run_rhf_rpa(ctx, &mol_b_gh, "monB+ghostA")) {
            (Ok(a_gh), Ok(b_gh)) => {
                let e_int_cp_ha = c.e_rpa - a_gh.e_rpa - b_gh.e_rpa;
                let e_int_cp_kcal = e_int_cp_ha * HA_TO_KCAL;
                eprintln!("  E_int(CP) = {:.4} kcal/mol", e_int_cp_kcal);
                t_rhf_s += a_gh.t_rhf + b_gh.t_rhf;
                t_rpa_s += a_gh.t_rpa + b_gh.t_rpa;
                e_int_cp_kcalmol = Some(e_int_cp_kcal);
            }
            (Err(e), _) => { eprintln!("  monA+ghostB FAIL: {e}"); status = format!("OK_FAIL_cpA:{e}"); }
            (_, Err(e)) => { eprintln!("  monB+ghostA FAIL: {e}"); status = format!("OK_FAIL_cpB:{e}"); }
        }
    }

    Csv {
        name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
        e_rhf: c.e_rhf, e_rpa: c.e_rpa,
        t_rhf_s, t_rpa_s,
        e_int_kcalmol: Some(e_int_kcal),
        e_int_ref_kcalmol: refs.get(name).copied(),
        e_int_cp_kcalmol,
        status,
    }
}

/// 3-body interaction energy for a trimer: E_int = E(ABC) - E(A) - E(B) - E(C).
/// Runs 4 RHF+RPA evaluations (complex + fragA + fragB + fragC). Implements
/// the GENERAL 3-fragment case (not a symmetric E(ABC)-3*E(A) shortcut) so
/// it stays correct even if the fragments turn out not to be exactly
/// identical (GGG/PHE are chemically identical by construction, but nothing
/// here assumes it).
fn run_trimer(
    ctx: &ParallelContext, name: &str, mol: &Molecule,
    idx_a: &[usize], idx_b: &[usize], idx_c: &[usize],
    refs: &HashMap<String, f64>,
) -> Csv {
    let n = mol.atoms.len();
    let (mol_a, mol_b, mol_c) = split_trimer(mol, idx_a, idx_b, idx_c);
    eprintln!(
        "  fragments: A={} atoms, B={} atoms, C={} atoms",
        mol_a.atoms.len(), mol_b.atoms.len(), mol_c.atoms.len()
    );

    let c = match run_rhf_rpa(ctx, mol, "complex") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  complex FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: 0, n_aux: 0,
                e_rhf: 0.0, e_rpa: 0.0, t_rhf_s: 0.0, t_rpa_s: 0.0,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL_complex:{e}"),
                e_int_cp_kcalmol: None,
            };
        }
    };
    let a = match run_rhf_rpa(ctx, &mol_a, "fragA") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  fragA FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
                e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL_fragA:{e}"),
                e_int_cp_kcalmol: None,
            };
        }
    };
    let b = match run_rhf_rpa(ctx, &mol_b, "fragB") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  fragB FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
                e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL_fragB:{e}"),
                e_int_cp_kcalmol: None,
            };
        }
    };
    let cc = match run_rhf_rpa(ctx, &mol_c, "fragC") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  fragC FAIL: {e}");
            return Csv {
                name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
                e_rhf: c.e_rhf, e_rpa: c.e_rpa, t_rhf_s: c.t_rhf, t_rpa_s: c.t_rpa,
                e_int_kcalmol: None, e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL_fragC:{e}"),
                e_int_cp_kcalmol: None,
            };
        }
    };
    let e_int_ha = c.e_rpa - a.e_rpa - b.e_rpa - cc.e_rpa;
    let e_int_kcal = e_int_ha * HA_TO_KCAL;
    eprintln!("  E_int (3-body) = {:.4} kcal/mol", e_int_kcal);

    Csv {
        name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
        e_rhf: c.e_rhf, e_rpa: c.e_rpa,
        t_rhf_s: c.t_rhf + a.t_rhf + b.t_rhf + cc.t_rhf,
        t_rpa_s: c.t_rpa + a.t_rpa + b.t_rpa + cc.t_rpa,
        e_int_kcalmol: Some(e_int_kcal),
        e_int_ref_kcalmol: refs.get(name).copied(),
        status: "OK".into(),
    e_int_cp_kcalmol: None,
    }
}

fn main() {
    // Args: <tier> [--only a,b,c] [--cp]
    let mut args = std::env::args().skip(1);
    let tier = args.next().unwrap_or_else(|| {
        eprintln!("usage: c9_driver <tier> [--only NAME[,NAME...]] [--cp]");
        eprintln!("       tiers: s66x8 | l7 | danuglipron");
        eprintln!("       --cp: also compute Boys-Bernardi counterpoise-corrected");
        eprintln!("             E_int (dimers only) alongside the uncorrected value.");
        std::process::exit(2);
    });
    let mut only_arg: Option<String> = None;
    let mut cp = false;
    while let Some(a) = args.next() {
        if a == "--only" {
            only_arg = args.next();
        } else if a == "--cp" {
            cp = true;
        }
    }
    let only: Option<HashSet<String>> = only_arg
        .or_else(|| std::env::var("FERRIC_C9_ONLY").ok())
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let cp = cp || std::env::var("FERRIC_C9_CP").is_ok();

    let ctx = ParallelContext::default();

    // CSV writer
    let csv_path = std::env::var("FERRIC_C9_OUTPUT_CSV").ok();
    let mut csv_file = csv_path.as_ref().map(|p| {
        fs::OpenOptions::new().create(true).write(true).truncate(true).open(p)
            .expect("open CSV file")
    });
    let mut emit = |line: String| {
        println!("{line}");
        if let Some(f) = csv_file.as_mut() {
            writeln!(f, "{line}").ok();
            f.flush().ok();
        }
        let _ = std::io::stdout().flush();
    };
    emit(Csv::header());

    let rows = match tier.as_str() {
        "s66x8" => run_s66x8(&ctx, &only, cp),
        "l7" => run_l7(&ctx, &only, cp),
        "danuglipron" => run_danuglipron(&ctx, &only),
        other => {
            eprintln!("unknown tier: {other}");
            std::process::exit(2);
        }
    };
    for row in &rows {
        emit(row.line());
    }
    // Sanity check (silence unused warning)
    let _ = BOHR_TO_ANGSTROM;
    let _ = Path::new("");
    eprintln!("\n=== c9_driver done: {} rows ===", rows.len());
}
