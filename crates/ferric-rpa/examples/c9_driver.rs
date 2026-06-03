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
//!   cargo run --release -p ferric-rpa --example c9_driver -- <tier> [--only <names>]
//! Examples:
//!   c9_driver -- s66x8 --only s66_01_waterwater_1.00,s66_24_benzenebenzenepipi_1.00
//!   c9_driver -- l7
//!   c9_driver -- danuglipron
//!
//! Environment:
//!   FERRIC_C9_ONLY        comma-separated system names (alias of --only)
//!   FERRIC_C9_OUTPUT_CSV  path to also append CSV lines to
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
}

impl Csv {
    fn header() -> String {
        "name,n_atoms,n_ao,n_aux,e_rhf,e_rpa,t_rhf_s,t_rpa_s,e_int_kcalmol,e_int_ref_kcalmol,delta_kcalmol,status".to_string()
    }
    fn line(&self) -> String {
        let e_int = self.e_int_kcalmol.map(|v| format!("{v:.4}")).unwrap_or_default();
        let e_int_ref = self.e_int_ref_kcalmol.map(|v| format!("{v:.4}")).unwrap_or_default();
        let delta = match (self.e_int_kcalmol, self.e_int_ref_kcalmol) {
            (Some(a), Some(b)) => format!("{:.4}", a - b),
            _ => String::new(),
        };
        format!(
            "{},{},{},{},{:.10},{:.10},{:.3},{:.3},{},{},{},{}",
            self.name, self.n_atoms, self.n_ao, self.n_aux,
            self.e_rhf, self.e_rpa, self.t_rhf_s, self.t_rpa_s,
            e_int, e_int_ref, delta, self.status,
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
    mol.atoms.iter().filter(|a| a.z >= 3).count()
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
    let mut rhf_cfg = RhfConfig::default();
    rhf_cfg.max_iter = 200;
    rhf_cfg.energy_conv = 1e-7;
    rhf_cfg.density_conv = 1e-6;
    rhf_cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    rhf_cfg.df_k_aux = Some("def2-universal-jkfit".to_string());

    let t0 = Instant::now();
    let rhf = solve_rhf(ctx, mol, &obs, op, &bounds, &rhf_cfg)?;
    let t_rhf = t0.elapsed().as_secs_f64();
    eprintln!("  [{label}] RHF: n_AO={n_ao} n_aux={n_aux} E={:.8} t={:.2}s",
              rhf.energy, t_rhf);

    let mut pdep_cfg = PdepRpaConfig::default();
    pdep_cfg.frozen_core = frozen_core_for(mol);
    pdep_cfg.trunc_thresh = 1e-4;
    pdep_cfg.davidson_conv_thresh = 1e-6;
    pdep_cfg.chi0_sparsity = Chi0Sparsity::Dense;

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

fn parse_s66_dimer_index(name: &str) -> Option<usize> {
    // Expected name: s66_NN_<body>_<dist>
    name.strip_prefix("s66_")?.split('_').next()?.parse().ok()
}

fn run_s66x8(ctx: &ParallelContext, only: &Option<HashSet<String>>) -> Vec<Csv> {
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
        rows.push(run_dimer(ctx, &stem, &mol, a_size, &refs));
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
    //   PHE:    phe + phe + phe -> 23+23+23=69 (3 frags, not 2)
    // The 3-fragment systems (GGG, PHE) cannot be split into two monomers
    // for a 2-body interaction energy — they need a 3-body decomposition
    // (E_int = E(ABC) - 3*E(A) for symmetric trimers). Driver returns None
    // -> no interaction energy is reported; total energy still goes to CSV.
    match name {
        "C2C2PD" => Some(56),
        "C3A"    => Some(72),  // circumcoronene fragment is first
        "C3GC"   => Some(72),
        "CBH"    => Some(36),
        "GCGC"   => Some(29),
        _ => None, // GGG, PHE: 3-body — handled in run_dimer fallback
    }
}

fn run_l7(ctx: &ParallelContext, only: &Option<HashSet<String>>) -> Vec<Csv> {
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
        match l7_fraga_size(&stem) {
            Some(a_size) if a_size < mol.atoms.len() => {
                rows.push(run_dimer(ctx, &stem, &mol, a_size, &refs));
            }
            _ => {
                eprintln!("  {stem} is a 3-body trimer; running complex only");
                rows.push(run_complex_only(ctx, &stem, &mol, &refs));
            }
        }
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
        },
        Err(e) => {
            eprintln!("  FAIL: {e}");
            Csv {
                name: name.into(), n_atoms: n, n_ao: 0, n_aux: 0,
                e_rhf: 0.0, e_rpa: 0.0, t_rhf_s: 0.0, t_rpa_s: 0.0,
                e_int_kcalmol: None,
                e_int_ref_kcalmol: refs.get(name).copied(),
                status: format!("FAIL:{e}"),
            }
        }
    }
}

fn run_dimer(
    ctx: &ParallelContext, name: &str, mol: &Molecule, a_size: usize,
    refs: &HashMap<String, f64>,
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
                status: format!("FAIL_monB:{e}"),
            };
        }
    };
    let e_int_ha = c.e_rpa - a.e_rpa - b.e_rpa;
    let e_int_kcal = e_int_ha * HA_TO_KCAL;
    eprintln!("  E_int = {:.4} kcal/mol", e_int_kcal);

    Csv {
        name: name.into(), n_atoms: n, n_ao: c.n_ao, n_aux: c.n_aux,
        e_rhf: c.e_rhf, e_rpa: c.e_rpa,
        t_rhf_s: c.t_rhf + a.t_rhf + b.t_rhf,
        t_rpa_s: c.t_rpa + a.t_rpa + b.t_rpa,
        e_int_kcalmol: Some(e_int_kcal),
        e_int_ref_kcalmol: refs.get(name).copied(),
        status: "OK".into(),
    }
}

fn main() {
    // Args: <tier> [--only a,b,c]
    let mut args = std::env::args().skip(1);
    let tier = args.next().unwrap_or_else(|| {
        eprintln!("usage: c9_driver <tier> [--only NAME[,NAME...]]");
        eprintln!("       tiers: s66x8 | l7 | danuglipron");
        std::process::exit(2);
    });
    let mut only_arg: Option<String> = None;
    while let Some(a) = args.next() {
        if a == "--only" {
            only_arg = args.next();
        }
    }
    let only: Option<HashSet<String>> = only_arg
        .or_else(|| std::env::var("FERRIC_C9_ONLY").ok())
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());

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
        "s66x8" => run_s66x8(&ctx, &only),
        "l7" => run_l7(&ctx, &only),
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
