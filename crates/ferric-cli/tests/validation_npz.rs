//! VALIDATION tier — VALIDATION.md row "NPZ export", against PySCF.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-cli --test validation_npz \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What this row pins
//!
//! The CLI's `[rpa] export_npz` path (`ferric_cli::lib.rs`, the block that
//! fills `NpzBundle` and calls `ferric_export::ml::export_npz`) writes arrays
//! that a FOREIGN reader (`numpy.load`) reads back equal to INDEPENDENTLY
//! computed values, and no requested field is missing. The companion pytest
//! (`crates/ferric-python/tests/test_validation_npz_export.py`) checks the
//! same file against ferric's own Python bindings on the default RI-JK
//! reference; this file checks it against PySCF, adds the C6 block, the
//! surface ESP and the PDEP eigenpotentials, and proves the knobs remove keys.
//!
//! System: water / cc-pVDZ, `method.kind = "pdep-rpa"`, RHF reference with
//! EXACT J/K (`[scf] df_j_aux = df_k_aux = "exact"`, so PySCF's exact-ERI RHF
//! is a like-for-like reference; the CLI's pdep-rpa default is RI-JK, whose
//! fitting error would otherwise sit under every full-chain bar),
//! `[rpa] auxbasis = "cc-pvdz-ri"`, every export knob at its default (on) plus
//! `compute_esp_surface = true`.
//!
//! # The foreign reader
//!
//! The NPZ is read by `numpy.load(..., allow_pickle=False)` in a Python
//! subprocess, which returns each array's dtype string, shape, C-contiguity
//! and raw little-endian bytes (hex), so no decimal round trip can blur an
//! "exact" comparison. The interpreter is `$FERRIC_VALIDATION_PYTHON`, else
//! `python3`. DECISION: a missing interpreter or an unimportable numpy is a
//! test FAILURE, not a skip — this is the validation tier, and a row whose
//! reader silently did not run would read as green. The CI validation job
//! installs numpy before the Rust step for this reason.
//!
//! # How each field is checked
//!
//! Reference: `scripts/validation/gen_npz.py` →
//! `testdata/reference/validation/npz/h2o_cc-pvdz.json`. AO order is ferric's;
//! the reference overlap (orbital basis) and Coulomb metric (aux basis) are
//! asserted equal to ferric's own FIRST, so a wrong permutation fails there.
//! "Integral level" = PySCF AO integrals contracted with the density / MOs the
//! NPZ itself carries (isolates the export path from SCF convergence);
//! "chain" = PySCF's own converged RHF.
//!
//! | key | check |
//! |---|---|
//! | coords, atomic_numbers | bit-exact vs ferric's own xyz parse; coords vs PySCF-side Bohr (1e-12) |
//! | density_matrix | chain vs PySCF D; Tr(DS) = N; symmetric |
//! | mo_coeffs | CᵀSC = I (PySCF S); 2 C_occ C_occᵀ = exported D; \|C_refᵀ S C\| = I (sign-invariant; water has no degenerate MOs) |
//! | orbital_energies | chain vs PySCF ε; ascending |
//! | esp_atoms, electric_field | integral level (int1e_rinv, int1e_iprinv); chain |
//! | dipole, density_second_moment | integral level (int1e_r, int1e_rr); chain |
//! | orbital_centers, orbital_spreads | integral level from the exported C (int1e_r, int1e_r2); chain |
//! | lowdin_charges, mulliken_charges | integral level (PySCF S^½, S, AO→atom map); chain |
//! | chelpg_charges, resp_charges | Σq = 0; H1 = H2 (C2v mirror); O negative; CHELPG point charges reproduce `esp_surface` |
//! | hirshfeld_charges | Σq = 0 to the grid floor; H1 = H2; O negative (no independent proatom reference) |
//! | esp_points, esp_surface | point SET vs ferric's documented construction rebuilt on PySCF's Lebedev-110 table; ESP chain vs PySCF int1e_rinv |
//! | alpha_tensor | chain vs numpy dRPA-RI α (same definition as the Static α row) |
//! | alpha_atomic | C2v mirror (H1 ↔ H2 with y → −y); Σ_A α^A vs α printed (NOT additive by construction) |
//! | pdep_eigenvectors | shape (naux, n_keep) with n_keep from the reference spectrum and the CLI's stdout; EᵀVE = I with PySCF's (P\|Q); Eᵀ(V+Π)E diagonal; projector EEᵀ vs PySCF's generalized eigenvectors |
//! | c6_partition, c6_source | decode to "becke", "ts" (the documented TS default) |
//! | c6_freqs, c6_weights | length n_quad = 20; positive |
//! | alpha_atomic_dynamic | TS shape = exported alpha_atomic / its iso average; single London pole whose ω_A = (4/3)C6_free/α_free² from the TS PRL 2009 free-atom table |
//! | c6_iso, c6_aniso, c6_molecular_iso | recomputed by numpy-side Casimir–Polder from the exported α(iω), freqs, weights; London C6_AA = (3/4)a²ω_A (quadrature) |
//!
//! Key set: EXACTLY the documented set for these knobs (29 keys; `boys_coeffs`
//! is a `NpzBundle` field the CLI never fills). Second test: with nine knobs
//! off, exactly their keys disappear (`compute_density_matrix = false` also
//! removes `density_second_moment`, which the CLI derives from the density),
//! and every remaining value still matches the reference.
//!
//! # Physics vs artifact hypotheses
//!
//! * Export correct: integral-level agreement ~1e-12, chain agreement at the
//!   SCF convergence floor (~1e-9).
//! * A field written from the wrong array or density (e.g. Mulliken passed as
//!   Löwdin, or the core guess): misses the integral-level bar by O(0.1).
//! * An (N,3) array transposed: water's field/coords are 3×3 and NOT
//!   symmetric, so the transposed array misses — asserted below as a control.
//! * A requested field dropped: the key-set assertion (the CLI exits 0 for a
//!   property whose arm only warns — see MUTATION B).
//!
//! # TOLERANCES — each bar sits ~10× above the floor measured on this box (the
//! floor is written beside each constant); `TOL_CHELPG_SURF_RRMS` is a fit-quality
//! bound, not a floor.
//!
//! # NEGATIVE CONTROLS (always on)
//!
//! * wrong basis: the water/def2-SVP reference density (same nao = 24) misses
//!   the chain bar by > 1000×;
//! * transposed `electric_field` misses the integral-level bar by > 1000×;
//!   transposed `mo_coeffs` fails CᵀSC = I by > 1000×;
//! * Löwdin and Mulliken charges differ by > 1000× the integral bar (so the
//!   swap in MUTATION A is resolvable);
//! * α with the wrong aux basis (def2-universal-jkfit) misses by > 1000×;
//! * CHELPG charges with flipped sign do not reproduce the surface ESP;
//! * a knob turned off removes its key (second test).
//!
//! # MUTATIONS (documented; each must turn this file red)
//!
//! * A — `crates/ferric-cli/src/lib.rs`, in the `NpzBundle` literal:
//!   `lowdin: lq_vec.as_deref(),` → `lowdin: mq_vec.as_deref(),`
//!   (Mulliken exported as Löwdin). Fails "lowdin_charges integral" by ~0.17 e.
//! * B — same literal: `hirshfeld: hq_vec.as_deref(),` → `hirshfeld: None,`.
//!   The CLI still exits 0 (that arm never records a gap); the key-set check
//!   fails naming `hirshfeld_charges`.
//! * C — `crates/ferric-cli/src/lib.rs`, the `coords_arr` block:
//!   `a[(i, 1)] = atom.y;` → `a[(i, 1)] = atom.zpos;`. Fails the bit-exact
//!   coords check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{s, Array2};
use serde_json::Value;

const REF: &str = "testdata/reference/validation/npz/h2o_cc-pvdz.json";
const XYZ: &str = "testdata/molecules/validation/h2o.xyz";
const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";
/// CLI default `[rpa] n_quad`.
const N_QUAD: usize = 20;

// --- Bars: measured floor on this box, bar ~10x above it.
const TOL_COORDS_REF: f64 = 1e-12; // measured 0.0 measured
const TOL_S: f64 = 1e-14; // measured 8.9e-16
const TOL_V_AUX: f64 = 5e-12; // measured 3.9e-13
const TOL_INT: f64 = 5e-13; // measured 2.8e-14 worst (esp_atoms)
const TOL_ORTHO: f64 = 1e-13; // measured 8.3e-15
const TOL_D_FROM_C: f64 = 1e-9; // measured 9.9e-11
const TOL_NELEC: f64 = 1e-13; // measured 5.3e-15
const TOL_CHAIN_D: f64 = 1e-8; // measured 5.9e-10
const TOL_CHAIN_EPS: f64 = 3e-9; // measured 2.5e-10
const TOL_CHAIN_PROP: f64 = 3e-9; // measured 3.5e-10 worst (second moment)
const TOL_MO_OVERLAP: f64 = 1e-9; // measured 7.6e-11
const TOL_ALPHA_CHAIN_REL: f64 = 1e-9; // measured 9.2e-11
const TOL_PDEP_METRIC: f64 = 5e-11; // measured 4.3e-12
const TOL_PDEP_OFFDIAG: f64 = 2e-10; // measured 1.2e-11 off-diagonal, 4.3e-12 Rayleigh
const TOL_PDEP_PROJ: f64 = 1e-6; // measured 2.6e-7, set by the near-degenerate tail modes
const TOL_SURF_PTS: f64 = 1e-13; // measured 1.4e-15
const TOL_SURF_ESP: f64 = 1e-10; // measured 6.0e-12
const TOL_SUM_EXACT: f64 = 1e-12; // measured 0.0
const TOL_HIRSH_SUM: f64 = 1e-12; // measured 6.7e-16
const TOL_MIRROR: f64 = 5e-13; // measured 1.8e-14
const TOL_C6_RECOMP_REL: f64 = 1e-12; // measured 0.0
const TOL_TS_SHAPE: f64 = 1e-14; // measured 4.4e-16
const TOL_TS_POLE_REL: f64 = 1e-14; // measured 3.4e-16
const TOL_LONDON_REL: f64 = 1e-12; // measured 3.0e-14
const TOL_CHELPG_SURF_RRMS: f64 = 0.25; // measured 0.12: a 3-charge fit, a physics bound not a floor
const TOL_SYM: f64 = 1e-14; // measured 0.0
const TOL_ENUC: f64 = 1e-12; // measured 0.0
const MUST_MISS: f64 = 1000.0;

/// The documented key set for the full run's knobs (all export defaults on,
/// plus `compute_esp_surface = true`).
const FULL_KEYS: [&str; 29] = [
    "mo_coeffs",
    "orbital_energies",
    "pdep_eigenvectors",
    "orbital_centers",
    "orbital_spreads",
    "density_second_moment",
    "coords",
    "atomic_numbers",
    "density_matrix",
    "dipole",
    "esp_atoms",
    "esp_surface",
    "esp_points",
    "alpha_tensor",
    "electric_field",
    "alpha_atomic",
    "hirshfeld_charges",
    "lowdin_charges",
    "mulliken_charges",
    "chelpg_charges",
    "resp_charges",
    "c6_partition",
    "c6_source",
    "c6_freqs",
    "c6_weights",
    "alpha_atomic_dynamic",
    "c6_iso",
    "c6_aniso",
    "c6_molecular_iso",
];

/// Knobs turned OFF in the second test, and the keys each one owns.
const OFF_KNOBS: [(&str, &[&str]); 9] = [
    ("compute_esp", &["esp_atoms"]),
    ("compute_polarizability", &["alpha_tensor"]),
    ("compute_alpha_atomic", &["alpha_atomic"]),
    // density_second_moment is derived from the exported density in the CLI
    // (`density_m2_opt = dm_ref.and_then(..)`), so it goes with it.
    (
        "compute_density_matrix",
        &["density_matrix", "density_second_moment"],
    ),
    ("compute_hirshfeld_charges", &["hirshfeld_charges"]),
    ("compute_lowdin_charges", &["lowdin_charges"]),
    ("compute_chelpg_charges", &["chelpg_charges"]),
    ("compute_resp_charges", &["resp_charges"]),
    (
        "compute_c6",
        &[
            "c6_partition",
            "c6_source",
            "c6_freqs",
            "c6_weights",
            "alpha_atomic_dynamic",
            "c6_iso",
            "c6_aniso",
            "c6_molecular_iso",
        ],
    ),
];

// ---------------------------------------------------------------------------
// Paths, CLI, foreign reader
// ---------------------------------------------------------------------------

/// Workspace root at RUN time (see `mwe_npz_partial_export_is_not_silent.rs`
/// for why a compile-time path breaks under `cargo nextest archive`).
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
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
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// `ferric-cli` binary at RUN time (sibling of the test binary under nextest).
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// Run the CLI on the water/cc-pVDZ pdep-rpa export with `rpa_extra` appended
/// to `[rpa]`. Returns (stdout, stderr, npz path). Panics on a non-zero exit.
fn run_cli(tag: &str, rpa_extra: &str) -> (String, String, PathBuf) {
    let root = workspace_root();
    let dir = root.join("target").join("validation_npz");
    // Absent under a custom CARGO_TARGET_DIR or a nextest archive.
    std::fs::create_dir_all(&dir).expect("create target/validation_npz");
    let npz = dir.join(format!("{tag}.npz"));
    let toml = dir.join(format!("{tag}.toml"));
    let _ = std::fs::remove_file(&npz);
    let body = format!(
        "[molecule]\nxyz = \"{XYZ}\"\n\n\
         [basis]\nname = \"{BASIS}\"\n\n\
         [method]\nkind = \"pdep-rpa\"\n\n\
         [scf]\nmax_iter = 200\nenergy_conv = 1e-9\ndensity_conv = 1e-10\n\
         df_j_aux = \"exact\"\ndf_k_aux = \"exact\"\n\n\
         [rpa]\nauxbasis = \"{AUX}\"\nexport_npz = \"{}\"\n{rpa_extra}\n",
        npz.display()
    );
    std::fs::write(&toml, body).expect("write validation TOML");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to spawn the ferric-cli binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "{tag}: ferric-cli exited {:?} (an incomplete NPZ exits non-zero)\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        out.status.code()
    );
    assert!(
        npz.is_file(),
        "{tag}: CLI exited 0 but wrote no {}",
        npz.display()
    );
    for l in stderr.lines().filter(|l| l.contains("warning")) {
        eprintln!("{tag}: CLI stderr: {l}");
    }
    (stdout, stderr, npz)
}

/// One array as `numpy.load` returned it.
struct Npy {
    dtype: String,
    shape: Vec<usize>,
    c_contiguous: bool,
    bytes: Vec<u8>,
}

impl Npy {
    fn f64s(&self, key: &str) -> Vec<f64> {
        assert_eq!(self.dtype, "<f8", "{key}: dtype {}, want <f8", self.dtype);
        let (chunks, rest) = self.bytes.as_chunks::<8>();
        assert!(rest.is_empty(), "{key}: byte length not a multiple of 8");
        chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
    }
    fn i64s(&self, key: &str) -> Vec<i64> {
        assert_eq!(self.dtype, "<i8", "{key}: dtype {}, want <i8", self.dtype);
        let (chunks, rest) = self.bytes.as_chunks::<8>();
        assert!(rest.is_empty(), "{key}: byte length not a multiple of 8");
        chunks.iter().map(|c| i64::from_le_bytes(*c)).collect()
    }
}

struct Bundle(BTreeMap<String, Npy>);

impl Bundle {
    fn get(&self, key: &str) -> &Npy {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("NPZ has no key {key:?}"))
    }
    fn vec(&self, key: &str) -> Vec<f64> {
        self.get(key).f64s(key)
    }
    fn mat(&self, key: &str) -> Array2<f64> {
        let a = self.get(key);
        assert_eq!(a.shape.len(), 2, "{key}: shape {:?} is not 2-D", a.shape);
        Array2::from_shape_vec((a.shape[0], a.shape[1]), a.f64s(key)).unwrap()
    }
    fn text(&self, key: &str) -> String {
        let a = self.get(key);
        assert_eq!(a.dtype, "|u1", "{key}: dtype {}, want |u1", a.dtype);
        String::from_utf8(a.bytes.clone()).unwrap_or_else(|e| panic!("{key}: not UTF-8: {e}"))
    }
    fn keys(&self) -> BTreeSet<String> {
        self.0.keys().cloned().collect()
    }
}

fn hex_decode(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("bad hex from numpy"))
        .collect()
}

/// The FOREIGN reader. Missing Python or numpy FAILS (validation tier).
fn numpy_load(npz: &Path) -> Bundle {
    const SCRIPT: &str = r#"
import sys, json
try:
    import numpy as np
except Exception as e:
    sys.stderr.write("NUMPY_IMPORT_FAILED: %r\n" % (e,))
    sys.exit(3)
out = {}
with np.load(sys.argv[1], allow_pickle=False) as z:
    for k in z.files:
        a = z[k]
        out[k] = {
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "c_contiguous": bool(a.flags.c_contiguous),
            "hex": np.ascontiguousarray(a).tobytes().hex(),
        }
json.dump({"numpy": np.__version__, "arrays": out}, sys.stdout)
"#;
    let py = std::env::var("FERRIC_VALIDATION_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&py)
        .arg("-c")
        .arg(SCRIPT)
        .arg(npz)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "cannot run the foreign NPZ reader {py:?} ({e}). Set FERRIC_VALIDATION_PYTHON \
                 to an interpreter with numpy; in the validation tier a missing reader is a \
                 FAILURE, never a skip"
            )
        });
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "foreign NPZ reader {py:?} failed (exit {:?}): {stderr}\nSet FERRIC_VALIDATION_PYTHON \
         to an interpreter with numpy.",
        out.status.code()
    );
    let v: Value = serde_json::from_slice(&out.stdout).expect("numpy reader emitted bad JSON");
    eprintln!(
        "foreign reader: {py} numpy {}",
        v["numpy"].as_str().unwrap_or("?")
    );
    let mut map = BTreeMap::new();
    for (k, a) in v["arrays"].as_object().expect("arrays") {
        map.insert(
            k.clone(),
            Npy {
                dtype: a["dtype"].as_str().unwrap().to_string(),
                shape: a["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
                c_contiguous: a["c_contiguous"].as_bool().unwrap(),
                bytes: hex_decode(a["hex"].as_str().unwrap()),
            },
        );
    }
    Bundle(map)
}

// ---------------------------------------------------------------------------
// Reference JSON
// ---------------------------------------------------------------------------

fn reference() -> Value {
    let path = workspace_root().join(REF);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             `scripts/validation/gen_npz.py` — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn at<'a>(v: &'a Value, ptr: &str) -> &'a Value {
    v.pointer(ptr)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
}

fn rvec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("reference array")
        .iter()
        .map(|x| x.as_f64().expect("reference number"))
        .collect()
}

/// Standard-alphabet base64 with `=` padding (what Python's b64encode writes).
fn base64_decode(text: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("base64: invalid byte {c}"),
        }
    }
    let raw = text.as_bytes();
    assert_eq!(
        raw.len() % 4,
        0,
        "base64: length {} is not a multiple of 4",
        raw.len()
    );
    let mut out = Vec::with_capacity(raw.len() / 4 * 3);
    for quad in raw.as_chunks::<4>().0 {
        let pad = quad.iter().rev().take_while(|&&c| c == b'=').count();
        let mut n = 0u32;
        for &c in &quad[..4 - pad] {
            n = (n << 6) | val(c);
        }
        n <<= 6 * pad as u32;
        let b = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&b[..3 - pad]);
    }
    out
}

/// A reference matrix: nested rows, or `{shape, f64le_b64}` (row-major
/// little-endian float64, `gen_npz.py::b64mat`) for the large ones.
fn rmat(v: &Value) -> Array2<f64> {
    if let Some(b64) = v.get("f64le_b64").and_then(Value::as_str) {
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|x| x.as_u64().expect("dim") as usize)
            .collect();
        assert_eq!(shape.len(), 2, "b64 matrix must be 2-D");
        let bytes = base64_decode(b64);
        let vals: Vec<f64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        assert_eq!(vals.len(), shape[0] * shape[1], "b64 matrix length");
        return Array2::from_shape_vec((shape[0], shape[1]), vals).expect("shape");
    }
    let rows = v.as_array().expect("reference matrix");
    let n = rows.len();
    let m = rows.first().and_then(Value::as_array).map_or(0, Vec::len);
    Array2::from_shape_fn((n, m), |(i, j)| rows[i][j].as_f64().expect("number"))
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

fn maxabs<'a>(a: impl IntoIterator<Item = &'a f64>, b: impl IntoIterator<Item = &'a f64>) -> f64 {
    a.into_iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn contract(d: &Array2<f64>, m: &Array2<f64>) -> f64 {
    assert_eq!(d.dim(), m.dim(), "contract: shape mismatch");
    (d * m).sum()
}

/// Collects every failure so one run reports them all (a measurement pass
/// then sees every number, not only the first red one).
#[derive(Default)]
struct Checks {
    fails: Vec<String>,
}

impl Checks {
    fn le(&mut self, what: &str, d: f64, tol: f64) {
        eprintln!("  {what:<52} {d:.2e} (tol {tol:.0e})");
        if d.is_nan() || d > tol {
            self.fails.push(format!("{what}: {d:.3e} > {tol:.0e}"));
        }
    }
    /// Negative control: `d` must exceed `floor`.
    fn gt(&mut self, what: &str, d: f64, floor: f64) {
        eprintln!("  [control] {what:<42} {d:.2e} (must exceed {floor:.0e})");
        if d.is_nan() || d <= floor {
            self.fails.push(format!(
                "control {what}: {d:.3e} does not exceed {floor:.0e}"
            ));
        }
    }
    fn truth(&mut self, what: &str, ok: bool) {
        eprintln!("  {what:<52} {}", if ok { "ok" } else { "FAILED" });
        if !ok {
            self.fails.push(what.to_string());
        }
    }
    fn finish(self, ctx: &str) {
        assert!(
            self.fails.is_empty(),
            "{ctx}: {} check(s) failed:\n  {}",
            self.fails.len(),
            self.fails.join("\n  ")
        );
    }
}

/// Ferric-side molecule + orbital/aux bases; asserts the reference's AO and
/// aux orders against ferric's own integrals before anything else.
fn ferric_side(r: &Value, c: &mut Checks) -> Molecule {
    let root = workspace_root();
    let mol = Molecule::load_xyz(root.join(XYZ).to_str().unwrap()).expect("load water xyz");
    let enuc = at(r, "/nuclear_repulsion").as_f64().unwrap();
    c.le(
        "nuclear repulsion vs reference",
        (mol.nuclear_repulsion() - enuc).abs(),
        TOL_ENUC,
    );
    let bs = basis::bundled(BASIS).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let s = ferric_integrals::oneelectron::overlap(&obs);
    let s_ref = rmat(at(r, "/overlap_ferric_order"));
    assert_eq!(s.dim(), s_ref.dim(), "AO count differs from the reference");
    c.le("overlap (AO permutation check)", maxabs(&s, &s_ref), TOL_S);
    let aux = basis::bundled(AUX).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
    let v = ferric_integrals::threeindex::coulomb_metric_2c(Operator::coulomb(), &dfbs)
        .expect("aux metric");
    let v_ref = rmat(at(r, "/pdep/metric_ferric_order"));
    assert_eq!(v.dim(), v_ref.dim(), "aux count differs from the reference");
    c.le(
        "aux (P|Q) (aux permutation check)",
        maxabs(&v, &v_ref),
        TOL_V_AUX,
    );
    mol
}

fn check_structure(b: &Bundle, expected: &BTreeSet<String>, c: &mut Checks, ctx: &str) {
    let keys = b.keys();
    let missing: Vec<_> = expected.difference(&keys).collect();
    let extra: Vec<_> = keys.difference(expected).collect();
    eprintln!("{ctx}: keys {keys:?}");
    c.truth(
        &format!("{ctx}: no requested key missing {missing:?}"),
        missing.is_empty(),
    );
    c.truth(
        &format!("{ctx}: no undocumented key {extra:?}"),
        extra.is_empty(),
    );
    for (k, a) in &b.0 {
        let want = match k.as_str() {
            "atomic_numbers" => "<i8",
            "c6_partition" | "c6_source" => "|u1",
            _ => "<f8",
        };
        c.truth(
            &format!("{k}: dtype {} == {want}", a.dtype),
            a.dtype == want,
        );
        c.truth(&format!("{k}: C-contiguous as loaded"), a.c_contiguous);
        if want == "<f8" {
            let v = a.f64s(k);
            c.truth(
                &format!("{k}: all finite, not all zero"),
                v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0),
            );
        }
    }
}

fn check_shapes(b: &Bundle, r: &Value, c: &mut Checks) {
    let nao = at(r, "/nao").as_u64().unwrap() as usize;
    let naux = at(r, "/naux").as_u64().unwrap() as usize;
    let nk = at(r, "/pdep/n_keep").as_u64().unwrap() as usize;
    let npts = at(r, "/esp_surface/esp").as_array().unwrap().len();
    let nat = 3;
    let want: BTreeMap<&str, Vec<usize>> = [
        ("mo_coeffs", vec![nao, nao]),
        ("orbital_energies", vec![nao]),
        ("pdep_eigenvectors", vec![naux, nk]),
        ("orbital_centers", vec![nao, 3]),
        ("orbital_spreads", vec![nao]),
        ("density_second_moment", vec![3, 3]),
        ("coords", vec![nat, 3]),
        ("atomic_numbers", vec![nat]),
        ("density_matrix", vec![nao, nao]),
        ("dipole", vec![3]),
        ("esp_atoms", vec![nat]),
        ("esp_surface", vec![npts]),
        ("esp_points", vec![npts, 3]),
        ("alpha_tensor", vec![3, 3]),
        ("electric_field", vec![nat, 3]),
        ("alpha_atomic", vec![nat, 3, 3]),
        ("hirshfeld_charges", vec![nat]),
        ("lowdin_charges", vec![nat]),
        ("mulliken_charges", vec![nat]),
        ("chelpg_charges", vec![nat]),
        ("resp_charges", vec![nat]),
        ("c6_partition", vec!["becke".len()]),
        ("c6_source", vec!["ts".len()]),
        ("c6_freqs", vec![N_QUAD]),
        ("c6_weights", vec![N_QUAD]),
        ("alpha_atomic_dynamic", vec![nat, N_QUAD, 3, 3]),
        ("c6_iso", vec![nat, nat]),
        ("c6_aniso", vec![nat, nat, 3, 3]),
        ("c6_molecular_iso", vec![1]),
    ]
    .into_iter()
    .collect();
    for (k, a) in &b.0 {
        if let Some(w) = want.get(k.as_str()) {
            c.truth(&format!("{k}: shape {:?} == {w:?}", a.shape), &a.shape == w);
        }
    }
}

/// Geometry: bit-exact against ferric's own parse, 1e-12 against the reference.
fn check_geometry(b: &Bundle, mol: &Molecule, r: &Value, c: &mut Checks) {
    let coords = b.vec("coords");
    let own: Vec<f64> = mol.atoms.iter().flat_map(|a| [a.x, a.y, a.zpos]).collect();
    c.truth("coords bit-exact vs ferric's xyz parse", coords == own);
    let rc: Vec<f64> = at(r, "/coords_bohr")
        .as_array()
        .unwrap()
        .iter()
        .flat_map(rvec)
        .collect();
    c.le(
        "coords vs PySCF-side Bohr",
        maxabs(&coords, &rc),
        TOL_COORDS_REF,
    );
    let z = b.get("atomic_numbers").i64s("atomic_numbers");
    let zr: Vec<i64> = rvec(at(r, "/atomic_numbers"))
        .iter()
        .map(|x| *x as i64)
        .collect();
    c.truth(&format!("atomic_numbers {z:?} == {zr:?}"), z == zr);
}

/// Values that exist in BOTH runs (full and knob-off): chain checks only.
fn check_always_on(b: &Bundle, r: &Value, c: &mut Checks) {
    // Orbital energies, MO coefficients.
    let eps = b.vec("orbital_energies");
    let eps_ref = rvec(at(r, "/mo_energy"));
    c.le(
        "orbital_energies chain",
        maxabs(&eps, &eps_ref),
        TOL_CHAIN_EPS,
    );
    c.truth(
        "orbital_energies ascending",
        eps.windows(2).all(|w| w[0] <= w[1]),
    );
    let eps_wrong = rvec(at(r, "/wrong_basis_negative_control/mo_energy"));
    c.gt(
        "wrong-basis (def2-SVP) orbital energies",
        maxabs(&eps, &eps_wrong),
        MUST_MISS * TOL_CHAIN_EPS,
    );

    let s = rmat(at(r, "/overlap_ferric_order"));
    let cm = b.mat("mo_coeffs");
    let ident = |m: &Array2<f64>| {
        m.indexed_iter().fold(0.0_f64, |acc, ((i, j), v)| {
            acc.max((v - f64::from(u8::from(i == j))).abs())
        })
    };
    c.le(
        "mo_coeffs C^T S C - I",
        ident(&cm.t().dot(&s).dot(&cm)),
        TOL_ORTHO,
    );
    let ct = cm.t().to_owned();
    c.gt(
        "transposed mo_coeffs C^T S C - I",
        ident(&ct.t().dot(&s).dot(&ct)),
        MUST_MISS * TOL_ORTHO,
    );
    let c_ref = rmat(at(r, "/mo_coeff_ferric_order"));
    let ov = c_ref.t().dot(&s).dot(&cm).mapv(f64::abs);
    c.le(
        "mo_coeffs | |C_ref^T S C| - I |",
        ident(&ov),
        TOL_MO_OVERLAP,
    );

    // Orbital centroids / spreads: integral level from the exported C, and chain.
    let rx: Vec<Array2<f64>> = (0..3)
        .map(|x| rmat(at(r, &format!("/dipole_ints_ferric_order/{x}"))))
        .collect();
    let r2 = rmat(at(r, "/r2_ints_ferric_order"));
    let n = cm.ncols();
    let mut cen = Array2::<f64>::zeros((n, 3));
    let mut spr = vec![0.0; n];
    for p in 0..n {
        let col = cm.column(p);
        let mut c2 = 0.0;
        for x in 0..3 {
            let v = col.dot(&rx[x].dot(&col));
            cen[(p, x)] = v;
            c2 += v * v;
        }
        spr[p] = (col.dot(&r2.dot(&col)) - c2).max(0.0).sqrt();
    }
    let cen_npz = b.mat("orbital_centers");
    let spr_npz = b.vec("orbital_spreads");
    c.le("orbital_centers integral", maxabs(&cen_npz, &cen), TOL_INT);
    c.le("orbital_spreads integral", maxabs(&spr_npz, &spr), TOL_INT);
    c.le(
        "orbital_centers chain",
        maxabs(&cen_npz, &rmat(at(r, "/orbital_centers"))),
        TOL_CHAIN_PROP,
    );
    c.le(
        "orbital_spreads chain",
        maxabs(&spr_npz, &rvec(at(r, "/orbital_spreads"))),
        TOL_CHAIN_PROP,
    );

    // Chain-level single-density properties.
    let ef = b.mat("electric_field");
    let ef_ref = rmat(at(r, "/electric_field_at_nuclei"));
    c.le("electric_field chain", maxabs(&ef, &ef_ref), TOL_CHAIN_PROP);
    c.le(
        "dipole chain",
        maxabs(&b.vec("dipole"), &rvec(at(r, "/dipole"))),
        TOL_CHAIN_PROP,
    );
    let mq = b.vec("mulliken_charges");
    c.le(
        "mulliken_charges chain",
        maxabs(&mq, &rvec(at(r, "/mulliken_charges"))),
        TOL_CHAIN_PROP,
    );
    c.le(
        "mulliken_charges sum",
        mq.iter().sum::<f64>().abs(),
        TOL_SUM_EXACT,
    );

    // PDEP eigenpotentials.
    let e = b.mat("pdep_eigenvectors");
    let v = rmat(at(r, "/pdep/metric_ferric_order"));
    let pi = rmat(at(r, "/pdep/pi_ferric_order"));
    let e_ref = rmat(at(r, "/pdep/eigvecs_kept_ferric_order"));
    c.le(
        "pdep E^T V E - I",
        ident(&e.t().dot(&v).dot(&e)),
        TOL_PDEP_METRIC,
    );
    let m = e.t().dot(&(&v + &pi)).dot(&e);
    let off = m
        .indexed_iter()
        .filter(|((i, j), _)| i != j)
        .fold(0.0_f64, |a, (_, x)| a.max(x.abs()));
    c.le("pdep E^T (V+Pi_ref) E off-diagonal", off, TOL_PDEP_OFFDIAG);
    let lam_ref = rvec(at(r, "/pdep/lambda_descending"));
    let lam: Vec<f64> = (0..m.nrows()).map(|i| m[(i, i)]).collect();
    c.le(
        "pdep Rayleigh quotients vs PySCF lambda (desc)",
        maxabs(&lam, &lam_ref[..lam.len()]),
        TOL_PDEP_OFFDIAG,
    );
    if e.dim() == e_ref.dim() {
        c.le(
            "pdep projector E E^T vs PySCF",
            maxabs(&e.dot(&e.t()), &e_ref.dot(&e_ref.t())),
            TOL_PDEP_PROJ,
        );
    } else {
        c.truth(
            &format!("pdep shape {:?} == reference {:?}", e.dim(), e_ref.dim()),
            false,
        );
    }
}

fn run_structure_and_values(
    tag: &str,
    rpa_extra: &str,
    expected: &BTreeSet<String>,
) -> (Bundle, Value, Checks, Molecule, String) {
    let r = reference();
    let mut c = Checks::default();
    let mol = ferric_side(&r, &mut c);
    let (stdout, _stderr, npz) = run_cli(tag, rpa_extra);
    let b = numpy_load(&npz);
    check_structure(&b, expected, &mut c, tag);
    check_shapes(&b, &r, &mut c);
    check_geometry(&b, &mol, &r, &mut c);
    // n_keep printed by the CLI: "Eigenpotentials kept:  K / N".
    let nk_ref = at(&r, "/pdep/n_keep").as_u64().unwrap() as usize;
    let kept_line = stdout
        .lines()
        .find(|l| l.starts_with("Eigenpotentials kept:"))
        .unwrap_or_else(|| panic!("{tag}: no 'Eigenpotentials kept:' line on stdout"));
    let k: usize = kept_line
        .trim_start_matches("Eigenpotentials kept:")
        .split('/')
        .next()
        .unwrap()
        .trim()
        .parse()
        .expect("kept count");
    c.truth(
        &format!("stdout kept {k} == reference n_keep {nk_ref}"),
        k == nk_ref,
    );
    check_always_on(&b, &r, &mut c);
    (b, r, c, mol, stdout)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: NPZ export"]
fn npz_export_water_cc_pvdz_vs_pyscf() {
    let expected: BTreeSet<String> = FULL_KEYS.iter().map(|s| s.to_string()).collect();
    let (b, r, mut c, _mol, _stdout) =
        run_structure_and_values("full", "compute_esp_surface = true", &expected);
    let nat = 3;
    let zs: Vec<f64> = rvec(at(&r, "/atomic_numbers"));
    let xyz = b.mat("coords");

    // --- Density matrix.
    let d = b.mat("density_matrix");
    let d_ref = rmat(at(&r, "/density_total_ferric_order"));
    c.le("density_matrix chain", maxabs(&d, &d_ref), TOL_CHAIN_D);
    c.le("density_matrix symmetric", maxabs(&d, &d.t()), TOL_SYM);
    let s = rmat(at(&r, "/overlap_ferric_order"));
    let nelec = at(&r, "/nelectron").as_f64().unwrap();
    c.le("Tr(D S) - N", (contract(&d, &s) - nelec).abs(), TOL_NELEC);
    let d_wrong = rmat(at(
        &r,
        "/wrong_basis_negative_control/density_total_ferric_order",
    ));
    c.gt(
        "wrong-basis (def2-SVP) density",
        maxabs(&d, &d_wrong),
        MUST_MISS * TOL_CHAIN_D,
    );
    let cm = b.mat("mo_coeffs");
    let nocc = at(&r, "/nocc").as_u64().unwrap() as usize;
    let co = cm.slice(s![.., ..nocc]);
    c.le(
        "density_matrix vs 2 C_occ C_occ^T (exported C)",
        maxabs(&d, &(co.dot(&co.t()) * 2.0)),
        TOL_D_FROM_C,
    );

    // --- ESP and field at the nuclei: integral level with the EXPORTED D.
    let esp = b.vec("esp_atoms");
    let ef = b.mat("electric_field");
    let esp_nuc = rvec(at(&r, "/esp_nuclear_part"));
    let mut esp_int = vec![0.0; nat];
    let mut ef_int = Array2::<f64>::zeros((nat, 3));
    for a in 0..nat {
        let rinv = rmat(at(&r, &format!("/rinv_at_nuclei_ferric_order/{a}")));
        esp_int[a] = esp_nuc[a] - contract(&d, &rinv);
        let fnuc = rvec(at(&r, &format!("/field_nuclear_part/{a}")));
        for x in 0..3 {
            let ip = rmat(at(
                &r,
                &format!("/iprinv_sym_at_nuclei_ferric_order/{a}/{x}"),
            ));
            ef_int[(a, x)] = fnuc[x] + contract(&d, &ip);
        }
    }
    c.le("esp_atoms integral", maxabs(&esp, &esp_int), TOL_INT);
    c.le(
        "esp_atoms chain",
        maxabs(&esp, &rvec(at(&r, "/esp_at_nuclei"))),
        TOL_CHAIN_PROP,
    );
    c.le("electric_field integral", maxabs(&ef, &ef_int), TOL_INT);
    c.gt(
        "transposed electric_field vs integral",
        maxabs(&ef.t(), &ef_int),
        MUST_MISS * TOL_INT,
    );

    // --- Dipole and density second moment: integral level.
    let mut dip = [0.0; 3];
    let mut m2 = Array2::<f64>::zeros((3, 3));
    for x in 0..3 {
        let rx = rmat(at(&r, &format!("/dipole_ints_ferric_order/{x}")));
        dip[x] = -contract(&d, &rx) + (0..nat).map(|a| zs[a] * xyz[(a, x)]).sum::<f64>();
        for y in 0..3 {
            let rr = rmat(at(&r, &format!("/second_moment_ints_ferric_order/{x}/{y}")));
            m2[(x, y)] = contract(&d, &rr);
        }
    }
    c.le("dipole integral", maxabs(&b.vec("dipole"), &dip), TOL_INT);
    let m2_npz = b.mat("density_second_moment");
    c.le(
        "density_second_moment integral",
        maxabs(&m2_npz, &m2),
        TOL_INT,
    );
    c.le(
        "density_second_moment chain",
        maxabs(&m2_npz, &rmat(at(&r, "/density_second_moment"))),
        TOL_CHAIN_PROP,
    );

    // --- Löwdin / Mulliken: integral level.
    let ao_atom: Vec<usize> = rvec(at(&r, "/ao_atom_ferric_order"))
        .iter()
        .map(|x| *x as usize)
        .collect();
    let sh = rmat(at(&r, "/overlap_sqrt_ferric_order"));
    let sds = sh.dot(&d).dot(&sh);
    let ds = d.dot(&s);
    let mut lq = zs.clone();
    let mut mq = zs.clone();
    for (u, &a) in ao_atom.iter().enumerate() {
        lq[a] -= sds[(u, u)];
        mq[a] -= ds[(u, u)];
    }
    let lq_npz = b.vec("lowdin_charges");
    let mq_npz = b.vec("mulliken_charges");
    c.le("lowdin_charges integral", maxabs(&lq_npz, &lq), TOL_INT);
    c.le(
        "lowdin_charges chain",
        maxabs(&lq_npz, &rvec(at(&r, "/lowdin_charges"))),
        TOL_CHAIN_PROP,
    );
    c.le(
        "lowdin_charges sum",
        lq_npz.iter().sum::<f64>().abs(),
        TOL_SUM_EXACT,
    );
    c.le("mulliken_charges integral", maxabs(&mq_npz, &mq), TOL_INT);
    c.gt(
        "Löwdin vs Mulliken (MUTATION A resolvable)",
        maxabs(&lq_npz, &mq_npz),
        MUST_MISS * TOL_INT,
    );

    // --- ESP-fitted and Hirshfeld charges: sum rules, C2v symmetry, sign.
    for (key, tol_sum) in [
        ("chelpg_charges", TOL_SUM_EXACT),
        ("resp_charges", TOL_SUM_EXACT),
        ("hirshfeld_charges", TOL_HIRSH_SUM),
    ] {
        let q = b.vec(key);
        c.le(&format!("{key} sum"), q.iter().sum::<f64>().abs(), tol_sum);
        c.le(
            &format!("{key} H1 == H2 (mirror)"),
            (q[1] - q[2]).abs(),
            TOL_MIRROR,
        );
        c.truth(&format!("{key}: O negative ({:.4})", q[0]), q[0] < 0.0);
    }

    // --- Surface ESP: point set vs the documented construction, ESP chain.
    let pts = b.mat("esp_points");
    let vs = b.vec("esp_surface");
    let pts_ref = rmat(at(&r, "/esp_surface/points_bohr"));
    let vs_ref = rvec(at(&r, "/esp_surface/esp"));
    let mut worst_pt = 0.0_f64;
    let mut worst_v = 0.0_f64;
    let mut used = vec![false; pts_ref.nrows()];
    let mut reused = 0usize;
    for i in 0..pts.nrows() {
        let (j, dist) = (0..pts_ref.nrows())
            .map(|j| {
                let d2: f64 = (0..3)
                    .map(|x| (pts[(i, x)] - pts_ref[(j, x)]).powi(2))
                    .sum();
                (j, d2.sqrt())
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        reused += usize::from(used[j]);
        used[j] = true;
        worst_pt = worst_pt.max(dist);
        worst_v = worst_v.max((vs[i] - vs_ref[j]).abs());
    }
    c.truth(
        &format!("esp_points match the reference one-to-one ({reused} reused)"),
        reused == 0 && pts.nrows() == pts_ref.nrows(),
    );
    c.le(
        "esp_points vs rebuilt surface (nearest)",
        worst_pt,
        TOL_SURF_PTS,
    );
    c.le("esp_surface chain (matched points)", worst_v, TOL_SURF_ESP);

    // CHELPG charges reproduce the surface ESP (two exported fields agree).
    let rrms = |q: &[f64]| {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..pts.nrows() {
            let vq: f64 = (0..nat)
                .map(|a| {
                    let d2: f64 = (0..3).map(|x| (pts[(i, x)] - xyz[(a, x)]).powi(2)).sum();
                    q[a] / d2.sqrt()
                })
                .sum();
            num += (vq - vs[i]).powi(2);
            den += vs[i].powi(2);
        }
        (num / den).sqrt()
    };
    let qc = b.vec("chelpg_charges");
    c.le(
        "CHELPG charges reproduce esp_surface (rel RMS)",
        rrms(&qc),
        TOL_CHELPG_SURF_RRMS,
    );
    let qneg: Vec<f64> = qc.iter().map(|x| -x).collect();
    c.gt(
        "sign-flipped CHELPG vs esp_surface (rel RMS)",
        rrms(&qneg),
        1.0,
    );

    // --- Static α.
    let a = b.mat("alpha_tensor");
    let a_ref = rmat(at(&r, "/alpha_drpa_ri"));
    let scale = a_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    c.le(
        "alpha_tensor chain (rel)",
        maxabs(&a, &a_ref) / scale,
        TOL_ALPHA_CHAIN_REL,
    );
    c.gt(
        "alpha vs wrong-aux (def2-universal-jkfit) reference (rel)",
        maxabs(&a, &rmat(at(&r, "/alpha_drpa_ri_swap_aux"))) / scale,
        MUST_MISS * TOL_ALPHA_CHAIN_REL,
    );

    // --- Per-atom Becke α: C2v mirror (y -> -y maps H1 <-> H2).
    let aa = b.vec("alpha_atomic"); // (3, 3, 3)
    let t = |atom: usize, i: usize, j: usize| aa[atom * 9 + i * 3 + j];
    let sgn = |i: usize| if i == 1 { -1.0 } else { 1.0 };
    let mut mir = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            mir = mir.max((t(2, i, j) - sgn(i) * sgn(j) * t(1, i, j)).abs());
            mir = mir.max((t(0, i, j) - sgn(i) * sgn(j) * t(0, i, j)).abs());
        }
    }
    c.le("alpha_atomic C2v mirror", mir, TOL_MIRROR);
    let sum_iso: f64 = (0..nat)
        .map(|at_| (t(at_, 0, 0) + t(at_, 1, 1) + t(at_, 2, 2)) / 3.0)
        .sum();
    eprintln!(
        "  [measurement] sum_A iso(alpha_atomic) = {sum_iso:.6} vs iso(alpha_tensor) = {:.6} \
         (not additive by construction: atom-centred dipoles exclude charge transfer)",
        (a[(0, 0)] + a[(1, 1)] + a[(2, 2)]) / 3.0
    );

    // --- C6 block.
    c.truth(
        &format!("c6_partition == \"becke\" ({:?})", b.text("c6_partition")),
        b.text("c6_partition") == "becke",
    );
    c.truth(
        &format!("c6_source == \"ts\" ({:?})", b.text("c6_source")),
        b.text("c6_source") == "ts",
    );
    let fr = b.vec("c6_freqs");
    let wt = b.vec("c6_weights");
    c.truth(
        "c6_freqs, c6_weights positive",
        fr.iter().chain(&wt).all(|x| *x > 0.0),
    );
    let nf = fr.len();
    let ad = b.vec("alpha_atomic_dynamic"); // (nat, nf, 3, 3)
    let dyn_t = |atom: usize, k: usize, i: usize, j: usize| ad[((atom * nf + k) * 3 + i) * 3 + j];
    let dyn_iso = |atom: usize, k: usize| (0..3).map(|i| dyn_t(atom, k, i, i)).sum::<f64>() / 3.0;

    // TS shape: α^A(iω_k)/iso == alpha_atomic[A]/iso (the Becke static tensor
    // IS the TS shape source for partition = becke).
    let mut shape_dev = 0.0_f64;
    for atom in 0..nat {
        let st_iso = (t(atom, 0, 0) + t(atom, 1, 1) + t(atom, 2, 2)) / 3.0;
        for k in 0..nf {
            let di = dyn_iso(atom, k);
            for i in 0..3 {
                for j in 0..3 {
                    shape_dev =
                        shape_dev.max((dyn_t(atom, k, i, j) / di - t(atom, i, j) / st_iso).abs());
                }
            }
        }
    }
    c.le(
        "alpha_atomic_dynamic shape == alpha_atomic shape",
        shape_dev,
        TOL_TS_SHAPE,
    );

    // Single London pole: 1/α_iso(iω) = 1/a + ω²/(a Ω²); Ω vs the TS table.
    let c6iso = b.mat("c6_iso");
    for atom in 0..nat {
        let (k1, k2) = (0, nf / 2);
        let (y1, y2) = (1.0 / dyn_iso(atom, k1), 1.0 / dyn_iso(atom, k2));
        let slope = (y2 - y1) / (fr[k2].powi(2) - fr[k1].powi(2));
        let icpt = y1 - slope * fr[k1].powi(2);
        let amp = 1.0 / icpt;
        let omega = (icpt / slope).sqrt();
        let pole_dev = (0..nf)
            .map(|k| (dyn_iso(atom, k) - amp / (1.0 + (fr[k] / omega).powi(2))).abs() / amp)
            .fold(0.0_f64, f64::max);
        c.le(
            &format!("atom {atom}: single-pole residual (rel)"),
            pole_dev,
            TOL_TS_POLE_REL,
        );
        let z = zs[atom] as usize;
        let om_ref = at(&r, &format!("/ts_free_atom/{z}/omega_london"))
            .as_f64()
            .unwrap();
        let a_free = at(&r, &format!("/ts_free_atom/{z}/alpha_free"))
            .as_f64()
            .unwrap();
        c.le(
            &format!("atom {atom} (Z={z}): London ω_A vs TS table (rel)"),
            (omega - om_ref).abs() / om_ref,
            TOL_TS_POLE_REL,
        );
        eprintln!(
            "  [measurement] atom {atom}: implied volume ratio a/alpha_free = {:.4}",
            amp / a_free
        );
        let london = 0.75 * amp * amp * omega;
        c.le(
            &format!("atom {atom}: c6_iso[AA] vs London (3/4)a²ω (rel)"),
            (c6iso[(atom, atom)] - london).abs() / london,
            TOL_LONDON_REL,
        );
    }

    // Casimir–Polder recomputed from the exported α(iω), freqs, weights.
    let pref = 3.0 / std::f64::consts::PI;
    let mut iso_dev = 0.0_f64;
    let mut aniso_dev = 0.0_f64;
    let ca = b.vec("c6_aniso"); // (nat, nat, 3, 3)
    let c6_scale = c6iso.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for p in 0..nat {
        for q in 0..nat {
            let cp: f64 = pref
                * (0..nf)
                    .map(|k| wt[k] * dyn_iso(p, k) * dyn_iso(q, k))
                    .sum::<f64>();
            iso_dev = iso_dev.max((c6iso[(p, q)] - cp).abs() / c6_scale);
            for i in 0..3 {
                for j in 0..3 {
                    let cpa: f64 = pref
                        * (0..nf)
                            .map(|k| wt[k] * dyn_t(p, k, i, j) * dyn_t(q, k, i, j))
                            .sum::<f64>();
                    aniso_dev =
                        aniso_dev.max((ca[((p * nat + q) * 3 + i) * 3 + j] - cpa).abs() / c6_scale);
                }
            }
        }
    }
    c.le("c6_iso recomputed (rel)", iso_dev, TOL_C6_RECOMP_REL);
    c.le("c6_aniso recomputed (rel)", aniso_dev, TOL_C6_RECOMP_REL);
    c.le(
        "c6_iso symmetric (rel)",
        maxabs(&c6iso, &c6iso.t()) / c6_scale,
        TOL_C6_RECOMP_REL,
    );
    // TS: the molecular α(iω) is exactly Σ_A α^A(iω).
    let mol_c6: f64 = pref
        * (0..nf)
            .map(|k| {
                let m: f64 = (0..nat).map(|atom| dyn_iso(atom, k)).sum();
                wt[k] * m * m
            })
            .sum::<f64>();
    let c6m = b.vec("c6_molecular_iso")[0];
    c.le(
        "c6_molecular_iso recomputed (rel)",
        (c6m - mol_c6).abs() / mol_c6,
        TOL_C6_RECOMP_REL,
    );

    c.finish("npz_export_water_cc_pvdz_vs_pyscf");
}

#[test]
#[ignore = "validation: NPZ export"]
fn npz_export_knob_off_removes_exactly_its_keys() {
    let mut extra = String::new();
    let mut removed: BTreeSet<String> = BTreeSet::new();
    for (knob, keys) in OFF_KNOBS {
        extra.push_str(&format!("{knob} = false\n"));
        removed.extend(keys.iter().map(|k| k.to_string()));
    }
    // compute_esp_surface defaults OFF, so its two keys are absent too.
    removed.insert("esp_surface".into());
    removed.insert("esp_points".into());
    let expected: BTreeSet<String> = FULL_KEYS
        .iter()
        .map(|s| s.to_string())
        .filter(|k| !removed.contains(k))
        .collect();
    eprintln!("knob-off run: expecting exactly {expected:?}");
    // Resolving power: the knobs must remove something, and leave something
    // whose value can still be checked.
    assert!(removed.len() >= 19 && expected.len() >= 8);
    let (_b, _r, c, _mol, _stdout) = run_structure_and_values("knobs_off", &extra, &expected);
    c.finish("npz_export_knob_off_removes_exactly_its_keys");
}
