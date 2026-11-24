//! `[scf] save_reference` / `read_reference` through the CLI binary.
//!
//! One run solves the reference SCF of a correlated kind and saves it; the
//! next reads it instead of solving. Pinned here: the correlated energy of the
//! read run is bitwise the fresh one's, for another correlated kind as well as
//! the same one, closed shell (RHF) and open shell (UHF); the read run logs no
//! SCF iteration; and a file for another calculation, a corrupted file and a
//! missing one are refused with a message that says why.
//!
//! The reads set `read_reference_other_build = true` because a test build is
//! often a dirty tree, which the checkpoint never counts as provably the same
//! build; `a_read_without_the_override_needs_a_provably_identical_build` pins
//! that gate on its own.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn scratch() -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join("scf_reference_io_test");
    std::fs::create_dir_all(&dir).expect("create the scratch dir");
    dir
}

struct Run {
    ok: bool,
    stderr: String,
    records: Vec<serde_json::Value>,
}

impl Run {
    fn count(&self, record: &str) -> usize {
        self.records
            .iter()
            .filter(|r| r["record"] == record)
            .count()
    }

    /// The correlated `result` record's total, as bits.
    fn total_bits(&self, kind: &str) -> u64 {
        let r = self
            .records
            .iter()
            .find(|r| r["record"] == "result" && r["kind"] == kind)
            .unwrap_or_else(|| panic!("no {kind} result record: {}", self.stderr));
        r["total"].as_f64().expect("numeric total").to_bits()
    }

    fn run_end(&self) -> &serde_json::Value {
        self.records
            .iter()
            .find(|r| r["record"] == "run_end")
            .unwrap_or_else(|| panic!("no run_end record: {}", self.stderr))
    }
}

fn run(tag: &str, body: &str) -> Run {
    let root = workspace_root();
    let dir = scratch();
    let toml = dir.join(format!("{tag}.toml"));
    let log = dir.join(format!("{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml, body).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .arg("--json")
        .arg(&log)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("run ferric-cli");
    let records = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    Run {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        records,
    }
}

fn ok(tag: &str, body: &str) -> Run {
    let r = run(tag, body);
    assert!(r.ok, "{tag} failed: {}", r.stderr);
    r
}

fn refused(tag: &str, body: &str) -> String {
    let r = run(tag, body);
    assert!(!r.ok, "{tag} must fail: {}", r.stderr);
    r.stderr
}

const WATER: &str = "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n";
const OH: &str = "[molecule]\nxyz = \"testdata/molecules/oh.xyz\"\nmultiplicity = 2\n";
const MP2: &str = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";

fn input(mol: &str, basis: &str, kind: &str, scf: &str) -> String {
    format!("{mol}[basis]\nname = \"{basis}\"\n[method]\nkind = \"{kind}\"\n{MP2}[scf]\n{scf}")
}

fn path_key(key: &str, file: &Path) -> String {
    format!("{key} = {:?}\n", file.display().to_string())
}

fn read_key(file: &Path) -> String {
    format!(
        "{}read_reference_other_build = true\n",
        path_key("read_reference", file)
    )
}

/// Save on the first kind, read on it and on the second: each read gives the
/// fresh run's energy bit for bit and runs no SCF iteration. Each kind comes
/// with its own `[scf]` lines (the MP2 family's default RI-JK reference is
/// not CCSD's exact one, so the RHF case asks RI-MP2 for `jk = "exact"`).
fn save_then_read(tag: &str, mol: &str, kinds: [(&str, &str); 2]) {
    let file = scratch().join(format!("{tag}.npz"));
    let _ = std::fs::remove_file(&file);
    let (first, first_scf) = kinds[0];
    let saved = ok(
        &format!("{tag}_save"),
        &input(
            mol,
            "6-31g",
            first,
            &format!("{first_scf}{}", path_key("save_reference", &file)),
        ),
    );
    assert!(saved.count("scf_iter") > 0, "the saving run solves its SCF");
    assert!(
        saved.stderr.contains("SCF reference saved to"),
        "{}",
        saved.stderr
    );
    assert!(file.is_file(), "no file at {}", file.display());
    for (kind, scf) in kinds {
        let fresh = ok(
            &format!("{tag}_{kind}_fresh"),
            &input(mol, "6-31g", kind, scf),
        );
        let read = ok(
            &format!("{tag}_{kind}_read"),
            &input(mol, "6-31g", kind, &format!("{scf}{}", read_key(&file))),
        );
        assert_eq!(
            read.total_bits(kind),
            fresh.total_bits(kind),
            "{tag}: {kind} on the read reference is not the fresh energy"
        );
        assert_eq!(read.count("scf_iter"), 0, "{tag}: {kind} read run iterated");
        assert_eq!(read.count("guess_scf_iter"), 0, "{tag}: {kind}");
        assert!(
            read.stderr.contains("no SCF iterations run"),
            "{}",
            read.stderr
        );
        let end = read.run_end();
        assert_eq!(end["extra"]["scf_iterations"], 0, "{end}");
        assert!(
            end["extra"]["scf_reference"]["file_iterations"]
                .as_u64()
                .unwrap()
                > 0,
            "{end}"
        );
        // The reference's precision record comes back with it.
        assert_eq!(
            end["extra"]["scf_precision"],
            fresh.run_end()["extra"]["scf_precision"],
            "{tag}: {kind}"
        );
        assert_eq!(
            end["energy"].as_f64().unwrap().to_bits(),
            fresh.run_end()["energy"].as_f64().unwrap().to_bits(),
            "{tag}: {kind} reference energy"
        );
    }
}

#[test]
fn an_rhf_reference_saved_by_rimp2_gives_rimp2_and_ccsd_their_fresh_energies() {
    save_then_read("rhf", WATER, [("rimp2", "jk = \"exact\"\n"), ("ccsd", "")]);
}

/// The MP2 family's default reference fits J and K, with the RI-J tensor
/// stored in f32: a mixed-precision record that the file restores too.
#[test]
fn a_mixed_precision_ri_jk_reference_gives_rimp2_and_scs_mp2_their_fresh_energies() {
    save_then_read("rijk", WATER, [("rimp2", ""), ("scs-mp2", "")]);
}

#[test]
fn a_uhf_reference_saved_by_rimp2_gives_rimp2_and_oo_rimp2_their_fresh_energies() {
    save_then_read("uhf", OH, [("rimp2", ""), ("oo-rimp2", "")]);
}

fn saved_water(tag: &str) -> PathBuf {
    let file = scratch().join(format!("{tag}.npz"));
    let _ = std::fs::remove_file(&file);
    ok(
        &format!("{tag}_save"),
        &input(WATER, "sto-3g", "rimp2", &path_key("save_reference", &file)),
    );
    file
}

#[test]
fn a_reference_for_another_basis_geometry_or_route_is_refused() {
    let file = saved_water("mismatch");
    let e = refused(
        "mismatch_basis",
        &input(WATER, "6-31g", "rimp2", &read_key(&file)),
    );
    assert!(
        e.contains("read_reference")
            && e.contains("different calculation")
            && e.contains("basis[Z="),
        "{e}"
    );
    let moved = scratch().join("water_moved.xyz");
    std::fs::write(
        &moved,
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
    )
    .unwrap();
    let moved_mol = format!("[molecule]\nxyz = {:?}\n", moved.display().to_string());
    let e = refused(
        "mismatch_geometry",
        &input(&moved_mol, "sto-3g", "rimp2", &read_key(&file)),
    );
    assert!(e.contains("coordinates"), "{e}");
    // The same molecule and basis solved on another route (exact J/K
    // through the DF-corrected incremental SCF, not RI-JK) is another
    // reference.
    let e = refused(
        "mismatch_route",
        &input(
            WATER,
            "sto-3g",
            "rimp2",
            &format!("df_increments = true\n{}", read_key(&file)),
        ),
    );
    assert!(e.contains("cli_reference_route"), "{e}");
    // A tighter request than the file was converged to.
    let e = refused(
        "mismatch_conv",
        &input(
            WATER,
            "sto-3g",
            "rimp2",
            &format!("density_conv = 1e-9\n{}", read_key(&file)),
        ),
    );
    assert!(e.contains("density_conv"), "{e}");
}

#[test]
fn a_missing_or_corrupted_reference_is_an_error() {
    let missing = scratch().join("no_such_reference.npz");
    let _ = std::fs::remove_file(&missing);
    let e = refused(
        "missing",
        &input(WATER, "sto-3g", "rimp2", &read_key(&missing)),
    );
    assert!(e.contains("does not exist"), "{e}");

    let file = saved_water("corrupt");
    let mut bytes = std::fs::read(&file).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x5a;
    std::fs::write(&file, &bytes).unwrap();
    let e = refused(
        "corrupt",
        &input(WATER, "sto-3g", "rimp2", &read_key(&file)),
    );
    assert!(
        e.contains("read_reference") && (e.contains("checksum") || e.contains("not readable")),
        "{e}"
    );
}

#[test]
fn a_read_without_the_override_needs_a_provably_identical_build() {
    let file = saved_water("build");
    let r = run(
        "build_read",
        &input(WATER, "sto-3g", "rimp2", &path_key("read_reference", &file)),
    );
    let same_build = ferric_build_info::commit_known() && ferric_build_info::DIRTY == Some(false);
    if same_build {
        assert!(r.ok, "{}", r.stderr);
    } else {
        assert!(!r.ok);
        assert!(
            r.stderr.contains("read_reference_other_build"),
            "{}",
            r.stderr
        );
    }
}

#[test]
fn the_keys_are_refused_where_no_shared_reference_is_solved() {
    let file = scratch().join("unused.npz");
    let save = path_key("save_reference", &file);
    for (tag, body, needle) in [
        (
            "scf_kind",
            input(WATER, "sto-3g", "rhf", &save),
            "kind = \"rhf\"",
        ),
        (
            "optimize",
            format!(
                "{WATER}[basis]\nname = \"sto-3g\"\n[method]\nkind = \"rimp2\"\ntask = \"optimize\"\n{MP2}[scf]\n{save}"
            ),
            "task = \"energy\" only",
        ),
        (
            "both",
            input(
                WATER,
                "sto-3g",
                "rimp2",
                &format!("{save}{}", path_key("read_reference", &file)),
            ),
            "cannot both be set",
        ),
        (
            "override_alone",
            input(WATER, "sto-3g", "rimp2", "read_reference_other_build = true\n"),
            "read only with",
        ),
        (
            "stability",
            input(
                WATER,
                "sto-3g",
                "rimp2",
                &format!("check_stability = true\n{}", read_key(&file)),
            ),
            "check_stability",
        ),
    ] {
        let e = refused(&format!("refuse_{tag}"), &body);
        assert!(e.contains(needle), "{tag}: {e}");
    }
    let missing_dir = scratch().join("no_such_dir").join("ref.npz");
    let e = refused(
        "refuse_save_dir",
        &input(
            WATER,
            "sto-3g",
            "rimp2",
            &path_key("save_reference", &missing_dir),
        ),
    );
    assert!(e.contains("does not exist"), "{e}");
}
