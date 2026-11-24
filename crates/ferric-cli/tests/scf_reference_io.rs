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
/// HI: iodine carries the def2 ECP.
const HI: &str = "[molecule]\nxyz = \"target/scf_reference_io_test/hi.xyz\"\n";

/// The correlation aux matching the orbital basis family.
fn aux_for(basis: &str) -> &'static str {
    if basis.starts_with("def2") {
        "def2-svp-rifit"
    } else {
        "cc-pvdz-ri"
    }
}

/// `scf` holds `[scf]` lines, optionally followed by further tables.
fn input(mol: &str, basis: &str, kind: &str, scf: &str) -> String {
    let aux = aux_for(basis);
    format!(
        "{mol}[basis]\nname = \"{basis}\"\n[method]\nkind = \"{kind}\"\n\
         [mp2]\nauxbasis = \"{aux}\"\n[scf]\n{scf}"
    )
}

/// An input with no `[mp2]` table (for kinds that would refuse it).
fn bare(kind: &str, scf: &str) -> String {
    format!("{WATER}[basis]\nname = \"sto-3g\"\n[method]\nkind = \"{kind}\"\n[scf]\n{scf}")
}

/// [`bare`] on the OH radical.
fn bare_oh(kind: &str, scf: &str) -> String {
    format!("{OH}[basis]\nname = \"sto-3g\"\n[method]\nkind = \"{kind}\"\n[scf]\n{scf}")
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
    save_then_read_in(tag, mol, "6-31g", kinds);
}

/// [`save_then_read`] in `basis`. A kind's `[scf]` lines may end with further
/// tables; the `read_reference` lines go before them.
fn save_then_read_in(tag: &str, mol: &str, basis: &str, kinds: [(&str, &str); 2]) {
    let with = |scf: &str, key: &str| {
        if scf.starts_with('[') {
            return format!("{key}{scf}");
        }
        match scf.find("\n[") {
            Some(i) => format!("{}\n{key}{}", &scf[..i], &scf[i + 1..]),
            None => format!("{scf}{key}"),
        }
    };
    let file = scratch().join(format!("{tag}.npz"));
    let _ = std::fs::remove_file(&file);
    let (first, first_scf) = kinds[0];
    let saved = ok(
        &format!("{tag}_save"),
        &input(
            mol,
            basis,
            first,
            &with(first_scf, &path_key("save_reference", &file)),
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
            &input(mol, basis, kind, scf),
        );
        let read = ok(
            &format!("{tag}_{kind}_read"),
            &input(mol, basis, kind, &with(scf, &read_key(&file))),
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
            bare("rimp2\"\ntask = \"optimize", &save),
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
            input(
                WATER,
                "sto-3g",
                "rimp2",
                "read_reference_other_build = true\n",
            ),
            "read only with",
        ),
        ("ksdft", bare("ksdft", &save), "kind = \"ksdft\""),
        ("uhf", bare_oh("uhf", &read_key(&file)), "kind = \"uhf\""),
        ("rohf", bare_oh("rohf", &save), "kind = \"rohf\""),
        ("double_hybrid", bare("b2plyp", &save), "save_reference"),
        (
            "casscf",
            bare(
                "casscf",
                &format!(
                    "{}[cas]\nn_active = 2\nn_elec_active = [1, 1]\n",
                    read_key(&file)
                ),
            ),
            "read_reference",
        ),
        (
            "cell",
            format!(
                "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n[basis]\nname = \"sto-3g\"\n\
                 [method]\nkind = \"rhf\"\n[cell]\nunit = \"bohr\"\n\
                 lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]\n\
                 exxdiv = \"ewald\"\njk = \"dense\"\n[scf]\n{save}"
            ),
            "save_reference",
        ),
        (
            "stability_descent",
            input(
                WATER,
                "sto-3g",
                "rimp2",
                &format!("stability_descent = true\n{}", read_key(&file)),
            ),
            "stability_descent",
        ),
        (
            "open_shell_pdep_rpa",
            input(OH, "sto-3g", "pdep-rpa", &read_key(&file)),
            "open-shell pdep-rpa",
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
    let dir = scratch().join("a_directory.npz");
    std::fs::create_dir_all(&dir).unwrap();
    let e = refused(
        "refuse_save_onto_dir",
        &input(WATER, "sto-3g", "rimp2", &path_key("save_reference", &dir)),
    );
    assert!(e.contains("is a directory"), "{e}");
    #[cfg(unix)]
    {
        let link = scratch().join("a_link.npz");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(scratch().join("target.npz"), &link).unwrap();
        let e = refused(
            "refuse_save_onto_symlink",
            &input(WATER, "sto-3g", "rimp2", &path_key("save_reference", &link)),
        );
        assert!(e.contains("is a symlink"), "{e}");
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

/// A Kohn-Sham reference: PDEP-RPA on a PBE reference (`[rpa] xc`).
#[test]
fn a_kohn_sham_reference_gives_pdep_rpa_its_fresh_energy() {
    let rpa = "[rpa]\nauxbasis = \"cc-pvdz-ri\"\nxc = \"PBE\"\n";
    save_then_read("ks", WATER, [("pdep-rpa", rpa), ("pdep-rpa", rpa)]);
}

/// An ECP reference: HI / def2-SVP, iodine's core replaced by its ECP.
#[test]
fn an_ecp_reference_gives_rimp2_and_mp3_their_fresh_energies() {
    std::fs::write(scratch().join("hi.xyz"), "2\n\nH 0 0 0\nI 0 0 1.609\n").unwrap();
    save_then_read_in(
        "ecp",
        HI,
        "def2-svp",
        [("rimp2", "jk = \"exact\"\n"), ("mp3", "")],
    );
}

/// A file converged tighter than the run asks is accepted, and the run gets
/// that reference: the saving run's energy, which is not the fresh one at
/// the looser default (8.4e-10 Ha apart on water / rimp2 when measured).
#[test]
fn a_tighter_reference_is_accepted_and_gives_the_saving_runs_energy() {
    let file = scratch().join("tight.npz");
    let _ = std::fs::remove_file(&file);
    let tight = "density_conv = 1e-9\n";
    let saved = ok(
        "tight_save",
        &input(
            WATER,
            "6-31g",
            "rimp2",
            &format!("{tight}{}", path_key("save_reference", &file)),
        ),
    );
    let fresh = ok("tight_fresh", &input(WATER, "6-31g", "rimp2", ""));
    let read = ok(
        "tight_read",
        &input(WATER, "6-31g", "rimp2", &read_key(&file)),
    );
    assert_eq!(read.total_bits("rimp2"), saved.total_bits("rimp2"));
    assert_ne!(
        read.total_bits("rimp2"),
        fresh.total_bits("rimp2"),
        "a tighter reference is not the fresh one at the default thresholds"
    );
    assert!(
        read.stderr.contains("converged tighter than this run asks"),
        "{}",
        read.stderr
    );
    let entry = &read.run_end()["extra"]["scf_reference"];
    assert_eq!(entry["file_density_conv"].as_f64(), Some(1e-9), "{entry}");
    assert_eq!(entry["file_energy_conv"].as_f64(), Some(1e-3), "{entry}");
    assert!(entry["file_threads"].as_u64().is_some(), "{entry}");
}

fn routes(run: &Run, kind: &str) -> Vec<(String, String)> {
    let r = run
        .records
        .iter()
        .find(|r| r["record"] == "result" && r["kind"] == kind)
        .unwrap_or_else(|| panic!("no {kind} result record: {}", run.stderr));
    r["components"]["certificate"]["routes"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|r| {
                    let text = |k: &str| r[k].as_str().unwrap_or_default().to_string();
                    (text("name"), text("detail"))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A read run's certificate keeps the reference's in-core ERI route, which
/// only the solver records, and names the file it read.
#[test]
fn a_read_certificate_keeps_the_eri_route_and_names_the_file() {
    let file = scratch().join("eri.npz");
    let _ = std::fs::remove_file(&file);
    let scf = "eri = \"incore\"\n";
    ok(
        "eri_save",
        &input(
            WATER,
            "sto-3g",
            "ccsd",
            &format!("{scf}{}", path_key("save_reference", &file)),
        ),
    );
    let fresh = ok("eri_fresh", &input(WATER, "sto-3g", "ccsd", scf));
    let read = ok(
        "eri_read",
        &input(
            WATER,
            "sto-3g",
            "ccsd",
            &format!("{scf}{}", read_key(&file)),
        ),
    );
    let (fresh, read) = (routes(&fresh, "ccsd"), routes(&read, "ccsd"));
    let eri = fresh
        .iter()
        .find(|(n, _)| n == "scf eri")
        .unwrap_or_else(|| panic!("the fresh run has no eri route: {fresh:?}"));
    assert!(read.contains(eri), "{read:?}");
    let provenance = read
        .iter()
        .find(|(n, _)| n == "scf reference")
        .unwrap_or_else(|| panic!("no provenance route: {read:?}"));
    assert!(
        provenance
            .1
            .starts_with(&format!("read from {} (checksum ", file.display())),
        "{provenance:?}"
    );
    // Everything else is the fresh run's.
    let rest: Vec<_> = read.iter().filter(|(n, _)| n != "scf reference").collect();
    assert_eq!(rest, fresh.iter().collect::<Vec<_>>());
}

/// The file cannot store a COSX final-grid pass or induced dipoles, so both
/// keys refuse those references before the SCF instead of writing a file
/// every read refuses.
#[test]
fn a_reference_the_file_cannot_store_is_refused_before_the_scf() {
    let file = scratch().join("unstorable.npz");
    let _ = std::fs::remove_file(&file);
    let cosx = "df_guess = false\nk_builder = \"cosx\"\n";
    for (tag, key) in [
        ("save", path_key("save_reference", &file)),
        ("read", read_key(&file)),
    ] {
        let e = refused(
            &format!("cosx_{tag}"),
            &input(WATER, "sto-3g", "ccsd", &format!("{cosx}{key}")),
        );
        assert!(e.contains("COSX final-grid pass"), "{e}");
        assert!(!file.exists(), "no file is written");
    }
    // Polarizable embedding: today the QM/MM validator already refuses it
    // on every kind these keys admit; `refuse_unstorable` holds if it widens.
    let thole = "[qmmm]\npqr = \"testdata/molecules/water_thole_sites.pqr\"\n\
                 qm_indices = [0, 1, 2]\npolarizabilities_angstrom3 = [0.0, 0.0, 0.0, \
                 0.0740923557432, 0.0740923557432, 0.163003182635]\nthole_a = 2.1304\n";
    let e = refused(
        "polarizable_save",
        &input(
            WATER,
            "sto-3g",
            "rimp2",
            &format!("{}{thole}", path_key("save_reference", &file)),
        ),
    );
    assert!(e.contains("polarizable"), "{e}");
    assert!(!file.exists(), "no file is written");
    // Without the final pass, COSX exchange is storable.
    ok(
        "cosx_no_final_save",
        &input(
            WATER,
            "sto-3g",
            "ccsd",
            &format!(
                "{cosx}cosx_final_pass = false\n{}",
                path_key("save_reference", &file)
            ),
        ),
    );
    assert!(file.is_file());
}
