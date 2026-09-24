//! Configs the CLI used to ACCEPT and then quietly answer a different
//! question for must now either fail, with a message that says why, or --
//! where ferric has a validated route -- answer the question actually asked.
//!
//! Each refusal was reproduced against the pre-fix release binary before it
//! was guarded (the measured output is quoted in each test). Three former
//! refusals are now ROUTES, and their tests check the NUMBER, not just the
//! exit status: open-shell `ksdft` and `uhf`/`rohf` + `[dft] functional` run
//! UKS/ROKS, open-shell `rimp2`/`oo-rimp2` run their unrestricted variants
//! on a UHF reference, and the MP2 double hybrids honour `[scf] df_j_aux`.
//! The pure config logic is unit-tested in `config.rs` (`compat_guard_tests`);
//! these tests pin that `run()` actually calls it, and that the
//! configurations the guards must NOT touch still run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME (a sibling of the
/// running test binary under nextest, the compile-time path under plain
/// `cargo test`). Same helper as `dispersion_guards.rs`; see there for why
/// a raw `env!` breaks under `cargo nextest archive`.
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

/// Workspace root at RUN time (same helper as `dispersion_guards.rs`).
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

fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root
        .join("target")
        .join(format!("silent_fallback_{tag}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli binary")
}

/// Molecule + STO-3G + `[method]`, followed by `extra` verbatim.
fn body(xyz: &str, multiplicity: usize, kind: &str, task: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"testdata/molecules/{xyz}\"\nmultiplicity = {multiplicity}\n\n\
         [basis]\nname = \"sto-3g\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n{extra}"
    )
}

fn assert_refused(out: &std::process::Output, needles: &[&str]) -> String {
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "config must be REFUSED, but the run succeeded.\nstdout: {}\nstderr: {err}",
        String::from_utf8_lossy(&out.stdout)
    );
    for n in needles {
        assert!(err.contains(n), "error must mention {n:?}, got: {err}");
    }
    err
}

fn assert_runs(out: &std::process::Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} must still RUN; the guard is too broad.\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// OH at the geometry of the PySCF references used below (O at the origin,
/// H at z = 0.97 Angstrom), written under `target/` and returned as a
/// workspace-relative path for `[molecule] xyz`. One file per test (`tag`):
/// the tests run in parallel, and a shared file rewritten by one test while
/// another's CLI reads it would be a race.
/// Bent NH2 (N-H 1.024 A, H-N-H ~103 deg): a doublet whose singly occupied
/// orbital (b1) is NON-degenerate. Used for ROKS because OH's degenerate pi
/// shell leaves ROKS fragile: 1e-7 A perturbations change its iteration count
/// from 57 to 115, 1e-6 to 1e-3 A ones fail to converge in 200 iterations, and
/// on the CI runner (AMD EPYC 9V74) it converged to an excited configuration
/// 0.58 Ha above the ground state. That is a solver defect tracked on its own
/// (validation campaign F6), not something this routing test should depend on.
/// NH2 ROKS converges in 7-9 iterations at every perturbation from 1e-7 to
/// 1e-3 A with the same energy.
fn nh2_xyz(tag: &str) -> String {
    let rel = format!("target/silent_fallback_nh2_{tag}.xyz");
    std::fs::write(
        workspace_root().join(&rel),
        "3\nNH2 doublet\nN 0.0 0.0 0.0\nH 0.0 0.7983 0.6228\nH 0.0 -0.7983 0.6228\n",
    )
    .expect("write NH2 xyz");
    rel
}

fn oh_097_xyz(tag: &str) -> String {
    let rel = format!("target/silent_fallback_oh_097_{tag}.xyz");
    std::fs::write(
        workspace_root().join(&rel),
        "2\nOH at 0.97 Angstrom\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n",
    )
    .expect("write OH xyz");
    rel
}

/// `[molecule]` from an explicit xyz path + `[basis]` + `[method]`, then
/// `extra` verbatim.
fn body_at(
    xyz_path: &str,
    multiplicity: usize,
    basis: &str,
    kind: &str,
    task: &str,
    extra: &str,
) -> String {
    format!(
        "[molecule]\nxyz = \"{xyz_path}\"\nmultiplicity = {multiplicity}\n\n\
         [basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"{task}\"\n\n{extra}"
    )
}

/// The number on the first stdout line whose trimmed text starts with
/// `label` (e.g. `"energy "`), read as the first token after its `=`.
fn stdout_value(out: &std::process::Output, label: &str) -> f64 {
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(label))
        .and_then(|l| l.split_once('=').map(|(_, rest)| rest))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            panic!(
                "no {label:?} line in stdout:\n{stdout}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        })
}

fn run_ok(tag: &str, body: &str) -> std::process::Output {
    let out = run_toml(tag, body);
    assert_runs(&out, tag);
    out
}

/// Tight SCF thresholds shared by every energy-comparison run below, so the
/// comparisons measure the method, not the SCF stopping point.
const TIGHT_SCF: &str = "[scf]\ndensity_conv = 1e-8\nenergy_conv = 1e-10\nmax_iter = 200\n";

/// PySCF UKS/PBE/STO-3G gas-phase energy of OH at 0.97 Angstrom
/// (testdata/reference/oh_sto-3g_uqmmm_dft_pbe.json, `e_gas_phase`), the
/// number `ferric-scf/tests/qmmm_dft_vs_pyscf.rs` holds `solve_uhf` to within
/// 2e-5 under the same RI-JK aux (def2-universal-jkfit) the CLI defaults to.
const PYSCF_OH_UKS_PBE_STO3G: f64 = -74.57265754425002;

fn read_reference(name: &str) -> serde_json::Value {
    let path = workspace_root().join("testdata/reference").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

// ─── Open-shell Kohn-Sham: ksdft / uhf / rohf + functional ──────────────────

/// Pre-#159: `error: SCF ladder failed: ScfConvergence { iterations: 0,
/// last_energy: 0.0 }`; #159 turned that into a refusal; now it runs UKS.
/// The number is held to PySCF's UKS/PBE energy at 2e-5 (the bar
/// `qmmm_dft_vs_pyscf.rs` uses for the same system). Mutations caught: a
/// restored refusal (no success), dropping the functional (UHF, ~0.2 Ha
/// away), or an RKS-style solve (an odd electron count cannot converge
/// there).
#[test]
fn ksdft_on_a_doublet_runs_uks_and_matches_pyscf() {
    let xyz = oh_097_xyz("ksdft_on_a_doublet_runs_uks_and_matches_pyscf");
    let out = run_ok(
        "ksdft_oh_uks",
        &body_at(
            &xyz,
            2,
            "sto-3g",
            "ksdft",
            "energy",
            &format!("[dft]\nfunctional = \"PBE\"\n\n{TIGHT_SCF}"),
        ),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("UKS[PBE]/sto-3g"), "{stdout}");
    let e = stdout_value(&out, "energy ");
    assert!(
        (e - PYSCF_OH_UKS_PBE_STO3G).abs() < 2e-5,
        "CLI UKS/PBE {e:.10} vs PySCF {PYSCF_OH_UKS_PBE_STO3G:.10}"
    );
    // The J/K line names the fitted Coulomb and says a pure GGA builds no K.
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("[ferric] SCF J/K: RI-J via def2-universal-jkfit; no K (pure functional)"),
        "{err}"
    );
}

/// Pre-#159: `kind = "uhf"` + `[dft] functional = "PBE"` on OH printed
/// `energy = -74.3626375456`, the UHF energy (identical to the run without
/// the key); #159 refused it; now the functional promotes the reference.
///
/// * uhf + PBE must EQUAL the ksdft doublet run (one calculation, two
///   spellings: same UKS solver, same RI-JK default) and differ from plain
///   UHF by the correlation-exchange difference (> 0.1 Ha here). Reverting
///   `Config::ks_functional`'s `"uhf"` arm makes uhf + PBE the UHF number and
///   fails both.
/// * rohf + PBE is ROKS: a spin-CONSTRAINED minimization of the same
///   functional, so it lies at or above UKS, within a few mEh, and far from
///   ROHF.
/// * rhf + PBE on (closed-shell) water is RKS, equal to `ksdft`.
#[test]
fn uhf_rohf_and_rhf_with_a_functional_run_kohn_sham() {
    let xyz = nh2_xyz("uhf_rohf_and_rhf_with_a_functional_run_kohn_sham");
    let pbe = format!("[dft]\nfunctional = \"PBE\"\n\n{TIGHT_SCF}");
    let energy = |tag: &str, kind: &str, extra: &str| {
        stdout_value(
            &run_ok(tag, &body_at(&xyz, 2, "sto-3g", kind, "energy", extra)),
            "energy ",
        )
    };
    let e_ksdft = energy("nh2_ksdft_pbe", "ksdft", &pbe);
    let e_uks = energy("nh2_uhf_pbe", "uhf", &pbe);
    let e_uhf = energy("nh2_uhf_plain", "uhf", TIGHT_SCF);
    // Keep the ROKS run's full output: if the gap assertion below fails it
    // prints what the SCF did (iterations, <S^2>, warnings).
    let roks_out = run_ok(
        "nh2_rohf_pbe",
        &body_at(&xyz, 2, "sto-3g", "rohf", "energy", &pbe),
    );
    let e_roks = stdout_value(&roks_out, "energy ");
    let e_rohf = energy("nh2_rohf_plain", "rohf", TIGHT_SCF);
    assert!(
        (e_uks - e_ksdft).abs() < 1e-8,
        "uhf + PBE {e_uks:.10} must be the ksdft UKS {e_ksdft:.10}"
    );
    assert!(
        (e_uks - e_uhf).abs() > 0.1,
        "uhf + PBE {e_uks:.10} is (still) the UHF energy {e_uhf:.10}"
    );
    // The ROKS - UKS gap on NH2, which cancels the grid/fitting offset between
    // codes: PySCF 2.13.1 (ROKS vs UKS, PBE/STO-3G, grids.level 5) gives
    // 6.6033e-4 Ha; ferric measured 6.6031e-4. A ROKS run that fell back to
    // UKS gives 0, one that dropped the functional gives ~0.2 Ha.
    const PYSCF_ROKS_MINUS_UKS: f64 = 6.6033e-4;
    assert!(
        (e_roks - e_uks - PYSCF_ROKS_MINUS_UKS).abs() < 1e-6,
        "ROKS {e_roks:.10} - UKS {e_uks:.10} = {:.4e}; PySCF gives {PYSCF_ROKS_MINUS_UKS:.4e}\n\
         --- ROKS stdout ---\n{}\n--- ROKS stderr ---\n{}",
        e_roks - e_uks,
        String::from_utf8_lossy(&roks_out.stdout),
        String::from_utf8_lossy(&roks_out.stderr)
    );
    assert!(
        (e_roks - e_rohf).abs() > 0.1,
        "rohf + PBE {e_roks:.10} is (still) the ROHF energy {e_rohf:.10}"
    );
    let roks = run_ok(
        "nh2_rohf_pbe_label",
        &body_at(&xyz, 2, "sto-3g", "rohf", "energy", &pbe),
    );
    assert!(String::from_utf8_lossy(&roks.stdout).contains("ROKS[PBE]/sto-3g"));

    let water = "testdata/molecules/water.xyz";
    let rks = run_ok(
        "water_rhf_pbe",
        &body_at(water, 1, "sto-3g", "rhf", "energy", &pbe),
    );
    assert!(String::from_utf8_lossy(&rks.stdout).contains("KS-DFT[PBE]/sto-3g"));
    let e_rks = stdout_value(&rks, "energy ");
    let e_ksdft_water = stdout_value(
        &run_ok(
            "water_ksdft_pbe",
            &body_at(water, 1, "sto-3g", "ksdft", "energy", &pbe),
        ),
        "energy ",
    );
    assert!(
        (e_rks - e_ksdft_water).abs() < 1e-9,
        "rhf + PBE {e_rks:.10} vs ksdft {e_ksdft_water:.10}"
    );
}

/// `task = "optimize"` on the UKS route uses `optimize_geometry_uhf` with
/// `xc` set, i.e. `ks_gradient_uks` (FD- and PySCF-validated in ferric-scf).
/// What this pins is the CLI threading: the run must converge, say UKS, and
/// end at the UKS/PBE minimum. The bar is the RELAXATION energy from the
/// 0.97 Å start, measured with PySCF (UKS PBE/STO-3G, grids.level 5, 1-D
/// bond scan): E(0.97) = -74.5722496885, E_min = -74.5798253258 at r = 1.0608 Å,
/// so ΔE = 7.5756 mEh. The difference cancels the grid offset between the two
/// codes (~4e-4 Ha in absolute energy). Dropping the functional would give UHF's
/// surface, whose relaxation differs, and an unconverged or wrong-gradient walk
/// would stop short of the minimum.
#[test]
fn ksdft_optimize_on_a_doublet_runs_on_the_uks_surface() {
    let xyz = oh_097_xyz("ksdft_optimize_on_a_doublet_runs_on_the_uks_surface");
    let pbe = format!("[dft]\nfunctional = \"PBE\"\n\n{TIGHT_SCF}");
    let e_start = stdout_value(
        &run_ok(
            "oh_uks_sp",
            &body_at(&xyz, 2, "sto-3g", "ksdft", "energy", &pbe),
        ),
        "energy ",
    );
    let out = run_ok(
        "oh_uks_opt",
        &body_at(&xyz, 2, "sto-3g", "ksdft", "optimize", &pbe),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("UKS[PBE] Optimization Result"), "{stdout}");
    assert!(stdout.contains("converged  = true"), "{stdout}");
    let e_opt = stdout_value(&out, "final E");
    const PYSCF_RELAXATION: f64 = 7.5756e-3;
    let relax = e_start - e_opt;
    assert!(
        (relax - PYSCF_RELAXATION).abs() < 1e-4,
        "relaxation {relax:.6e} (optimized {e_opt:.10} vs start {e_start:.10}); \
         PySCF UKS/PBE gives {PYSCF_RELAXATION:.4e}"
    );
}

/// D3(BJ) on the UKS route: the single point applies it (E = E(KS) +
/// E(D3BJ)), and the geometry optimization -- whose open-shell optimizer
/// has no correction hook -- is refused rather than walking the
/// uncorrected surface. Removing `refuse_open_shell_dispersion_gradient`
/// lets the optimize run succeed and fails the refusal.
#[test]
fn open_shell_ks_applies_d3_to_energies_and_refuses_it_on_optimize() {
    let xyz = oh_097_xyz("open_shell_ks_applies_d3_to_energies_and_refuses_it_on_optimize");
    let d3 = "[dft]\nfunctional = \"PBE\"\ndispersion = \"d3bj(pbe)\"\n";
    let out = run_ok(
        "oh_uks_d3",
        &body_at(&xyz, 2, "sto-3g", "ksdft", "energy", d3),
    );
    let e_ks = stdout_value(&out, "E(KS-DFT)");
    let e_d3 = stdout_value(&out, "E(D3BJ)");
    let e = stdout_value(&out, "energy ");
    assert!(
        e_d3 != 0.0 && (e - (e_ks + e_d3)).abs() < 1e-9,
        "{e} {e_ks} {e_d3}"
    );
    let opt = run_toml(
        "oh_uks_d3_opt",
        &body_at(&xyz, 2, "sto-3g", "ksdft", "optimize", d3),
    );
    assert_refused(&opt, &["[dft] dispersion", "UKS[PBE]", "optimize"]);
}

// ─── Open-shell RI-MP2 ──────────────────────────────────────────────────────

/// Pre-#159: triplet water under `rimp2` printed `Total = -74.9987495795`,
/// the singlet's RI-MP2 energy, and exited 0; #159 refused it. Now it is
/// UHF + unrestricted RI-MP2 (UMP2):
///
/// * OH/cc-pVDZ vs the PySCF harness (testdata/reference/
///   oh_cc-pvdz_u-oomp2-fd.json: `e_hf`, `e_corr` with the same cc-pvdz-ri
///   aux). A ROHF reference fed to the same kernel is ~6e-3 Ha off here, so
///   the 1e-4 bar pins the reference as well as the method.
/// * triplet water: the reference printed IS the `kind = "uhf"` energy, and
///   the total is NOT the singlet number #159 measured.
///
/// Mutations caught: restoring the refusal (no success), feeding the
/// closed-shell kernel (the singlet total), a ROHF or MOM reference (the
/// e_hf / e_corr bars).
#[test]
fn rimp2_on_an_open_shell_molecule_runs_ump2_on_a_uhf_reference() {
    let r = read_reference("oh_cc-pvdz_u-oomp2-fd.json");
    let e_hf = r["e_hf"].as_f64().expect("e_hf");
    let e_corr = r["e_corr"].as_f64().expect("e_corr");
    let xyz = oh_097_xyz("rimp2_on_an_open_shell_molecule_runs_ump2_on_a_uhf_reference");
    let out = run_ok(
        "oh_ump2",
        &body_at(
            &xyz,
            2,
            "cc-pvdz",
            "rimp2",
            "energy",
            &format!("[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n{TIGHT_SCF}"),
        ),
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("U-RI-MP2/cc-pvdz"));
    let uhf = stdout_value(&out, "UHF energy");
    let corr = stdout_value(&out, "MP2 corr");
    let total = stdout_value(&out, "Total");
    assert!(
        (uhf - e_hf).abs() < 1e-6,
        "UHF {uhf:.10} vs PySCF {e_hf:.10}"
    );
    assert!(
        (corr - e_corr).abs() < 1e-4,
        "UMP2 corr {corr:.10} vs PySCF {e_corr:.10}"
    );
    assert!((total - (uhf + corr)).abs() < 1e-9);

    let water = "testdata/molecules/water.xyz";
    let mp2 = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    let trip = run_ok(
        "rimp2_triplet",
        &body_at(water, 3, "sto-3g", "rimp2", "energy", mp2),
    );
    let e_uhf = stdout_value(
        &run_ok(
            "uhf_triplet",
            &body_at(water, 3, "sto-3g", "uhf", "energy", ""),
        ),
        "energy ",
    );
    assert!(
        (stdout_value(&trip, "UHF energy") - e_uhf).abs() < 1e-9,
        "the open-shell rimp2 reference must be the uhf run's energy {e_uhf:.10}"
    );
    let singlet_total = -74.998_749_579_527_17;
    let t = stdout_value(&trip, "Total");
    assert!(
        (t - singlet_total).abs() > 1e-2,
        "triplet total {t:.10} is the singlet"
    );
    assert!(stdout_value(&trip, "MP2 corr") < 0.0);
}

/// oo-rimp2 on a doublet runs unrestricted OO-RI-MP2 from the UHF
/// reference (OH/cc-pVDZ, the system `u_oo_rimp2_lowers_energy_on_oh`
/// checks in ferric-mp2). OO-MP2 minimizes the MP2 functional over orbital
/// rotations starting AT the UHF orbitals, where it equals the UMP2 energy,
/// so its total must lie strictly below the `rimp2` total with the same
/// `[scf]`/`[mp2]` keys, by less than 10 mEh. Routing oo-rimp2 to plain UMP2
/// (no orbital relaxation) fails the strict inequality; restoring the
/// refusal fails `run_ok`.
#[test]
fn oo_rimp2_on_a_doublet_runs_unrestricted_orbital_optimization() {
    let xyz = oh_097_xyz("oo_rimp2_on_a_doublet_runs_unrestricted_orbital_optimization");
    let extra = format!("[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n{TIGHT_SCF}");
    let ump2 = stdout_value(
        &run_ok(
            "oh_ump2_for_oo",
            &body_at(&xyz, 2, "cc-pvdz", "rimp2", "energy", &extra),
        ),
        "Total",
    );
    let out = run_ok(
        "oh_uoomp2",
        &body_at(&xyz, 2, "cc-pvdz", "oo-rimp2", "energy", &extra),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("U-OO-RI-MP2/cc-pvdz"), "{stdout}");
    let oo = stdout_value(&out, "Total");
    // Orbital optimization lowers the UMP2 energy by 4.206e-4 Ha here
    // (measured; no independent OO-UMP2 reference exists in testdata, so this
    // is a regression pin, not a physics bar). A run that skipped the orbital
    // optimization gives 0.
    let lowering = ump2 - oo;
    assert!(
        (1e-4..1e-3).contains(&lowering),
        "U-OO-RI-MP2 {oo:.10} vs UMP2 {ump2:.10}: lowering {lowering:.4e}, measured 4.206e-4"
    );
}

/// What stays refused on the open-shell MP2 route, each for a stated
/// reason: no unrestricted MP2 nuclear gradient (optimize), kappa-MP2 is
/// closed-shell only, and kinds with no open-shell variant (mp3). Dropping
/// any of these guards makes the corresponding run succeed.
#[test]
fn open_shell_mp2_refuses_what_it_cannot_honour() {
    let water = "testdata/molecules/water.xyz";
    let mp2 = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    assert_refused(
        &run_toml(
            "rimp2_triplet_opt",
            &body_at(water, 3, "sto-3g", "rimp2", "optimize", mp2),
        ),
        &["multiplicity = 3", "task = \"energy\" only"],
    );
    assert_refused(
        &run_toml(
            "rimp2_triplet_kappa",
            &body_at(
                water,
                3,
                "sto-3g",
                "rimp2",
                "energy",
                "[mp2]\nauxbasis = \"cc-pvdz-ri\"\nkappa = 1.0\n",
            ),
        ),
        &["kappa", "open-shell"],
    );
    assert_refused(
        &run_toml(
            "mp3_triplet",
            &body_at(water, 3, "sto-3g", "mp3", "energy", mp2),
        ),
        &["multiplicity = 3", "closed-shell"],
    );
}

/// Reachability anchor: the open-shell kinds and closed-shell singlets are
/// untouched.
#[test]
fn uhf_doublet_and_closed_shell_rimp2_still_run() {
    assert_runs(
        &run_toml("uhf_oh", &body("oh.xyz", 2, "uhf", "energy", "")),
        "uhf on the OH doublet",
    );
    assert_runs(
        &run_toml(
            "rimp2_singlet",
            &body(
                "water.xyz",
                1,
                "rimp2",
                "energy",
                "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
            ),
        ),
        "rimp2 on singlet water",
    );
}

// ─── [dft] keys the selected kind never reads (item 1) ──────────────────────

/// A functional on a kind that runs none is still refused: `rimp2` builds an
/// RHF reference and would print RI-MP2 on HF under a config naming PBE.
/// Removing the `validate_dft_section` call from `run()` makes this run
/// succeed. (On rhf/uhf/rohf the functional is now READ -- see
/// `uhf_rohf_and_rhf_with_a_functional_run_kohn_sham`.)
#[test]
fn a_functional_on_a_correlated_kind_is_refused() {
    let out = run_toml(
        "rimp2_pbe",
        &body(
            "water.xyz",
            1,
            "rimp2",
            "energy",
            "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n[dft]\nfunctional = \"PBE\"\n",
        ),
    );
    assert_refused(&out, &["[dft] functional", "\"rimp2\""]);
}

// ─── Keys a task path never reads (items 4 + 5) ─────────────────────────────

/// Pre-fix: H2/STO-3G, pdep-rpa, task = "optimize", `[rpa] xc = "PBE"`
/// converged to `final E = -1.1375270338 Hartree (RHF + RPA)` -- RPA@HF, the
/// functional silently dropped. Removing the `validate_task_compat` call
/// from `run()` makes this run (and succeed) again.
#[test]
fn rpa_optimize_with_a_ks_reference_is_refused() {
    let out = run_toml(
        "rpa_opt_xc",
        &body(
            "h2.xyz",
            1,
            "pdep-rpa",
            "optimize",
            "[rpa]\nauxbasis = \"cc-pvdz-ri\"\nxc = \"PBE\"\n",
        ),
    );
    assert_refused(&out, &["[rpa] xc", "optimize"]);
}

/// Pre-fix: water/STO-3G, rhf, task = "optimize", `k_builder = "cosx"` ran
/// to completion with COSX energies and an exact-exchange gradient (FD
/// mismatch -8.9e-6 Ha/Bohr on one H z at STO-3G, -1.4e-5 at cc-pVDZ).
/// Removing the cosx branch of `validate_task_compat` lets it run again. The
/// anchor is the same molecule with COSX on task = "energy", which must run.
#[test]
fn cosx_with_optimize_is_refused_but_a_cosx_energy_runs() {
    let cosx = "[scf]\nk_builder = \"cosx\"\n";
    let out = run_toml("cosx_opt", &body("h2.xyz", 1, "rhf", "optimize", cosx));
    assert_refused(&out, &["k_builder = \"cosx\"", "optimize"]);
    assert_runs(
        &run_toml("cosx_energy", &body("h2.xyz", 1, "rhf", "energy", cosx)),
        "an rhf COSX energy",
    );
}

// ─── [scf] df_j_aux / df_k_aux spellings (item 6) ───────────────────────────

/// Pre-fix: `[scf] df_j_aux = "exact"` failed with "unknown bundled basis:
/// exact", although Python `run_dft` reads that spelling as conventional J.
/// If `ScfCfg::df_*_aux_resolved` stops going through
/// `ferric_scf::rhf::normalize_df_aux`, this run fails the same way again.
#[test]
fn scf_df_aux_accepts_the_shared_opt_out_spellings() {
    let out = run_toml(
        "dfaux_exact",
        &body(
            "water.xyz",
            1,
            "rhf",
            "energy",
            "[scf]\ndf_j_aux = \"exact\"\ndf_k_aux = \"none\"\n",
        ),
    );
    assert_runs(&out, "rhf with df_j_aux = \"exact\"");
}

// ─── The J/K log line (item 7) ──────────────────────────────────────────────

/// Pre-fix: ksdft/PBE with `df_j_aux = ""` logged `[ferric] SCF J/K: RI-JK
/// via ` -- an empty basis name, on the one run that had turned density
/// fitting off. Reverting to the old `Some(aux) => "RI-JK via {aux}"` match
/// brings that line back and fails both assertions.
#[test]
fn jk_log_line_names_exact_coulomb_when_df_is_off() {
    let out = run_toml(
        "jk_log",
        &body(
            "water.xyz",
            1,
            "ksdft",
            "energy",
            "[dft]\nfunctional = \"PBE\"\n\n[scf]\ndf_j_aux = \"\"\n",
        ),
    );
    assert_runs(&out, "ksdft/PBE with df_j_aux = \"\"");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("[ferric] SCF J/K: exact J (four-centre); no K (pure functional)"),
        "{err}"
    );
    assert!(!err.contains("RI-JK via \n"), "{err}");
}

// ─── Double hybrids honour [scf] df_j_aux ───────────────────────────────────

/// `b2plyp`/`dsd-pbep86` used to overwrite `[scf] df_j_aux`/`df_k_aux` with
/// def2-universal-jkfit unconditionally, so a user's aux (or opt-out) was
/// silently ignored. Now an omitted key keeps that default, `""`/"exact"
/// select conventional J, and a name is honoured -- the `ksdft` rule.
///
/// Mutation caught: restoring the unconditional overwrite makes all four runs
/// identical (same J/K line, same energy), failing every assertion below the
/// default run.
#[test]
fn b2plyp_honours_scf_df_j_aux() {
    let water = "testdata/molecules/water.xyz";
    let run = |tag: &str, scf: &str| {
        run_ok(
            tag,
            &body_at(
                water,
                1,
                "sto-3g",
                "b2plyp",
                "energy",
                &format!("[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\n[scf]\n{scf}"),
            ),
        )
    };
    let jk = |out: &std::process::Output| {
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .find(|l| l.starts_with("[ferric] SCF J/K:"))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("no J/K line: {}", String::from_utf8_lossy(&out.stderr)))
    };
    let default = run("b2plyp_default", "");
    let exact = run("b2plyp_exact_j", "df_j_aux = \"\"\n");
    let exact_word = run("b2plyp_exact_word", "df_j_aux = \"exact\"\n");
    let named = run("b2plyp_named_j", "df_j_aux = \"cc-pvdz-ri\"\n");

    assert_eq!(
        jk(&default),
        "[ferric] SCF J/K: RI-JK via def2-universal-jkfit"
    );
    assert_eq!(
        jk(&exact),
        "[ferric] SCF J/K: exact J (four-centre), RI-K via def2-universal-jkfit"
    );
    assert_eq!(jk(&exact_word), jk(&exact));
    assert_eq!(
        jk(&named),
        "[ferric] SCF J/K: RI-J via cc-pvdz-ri, RI-K via def2-universal-jkfit"
    );

    let e_default = stdout_value(&default, "Total");
    let e_exact = stdout_value(&exact, "Total");
    let e_named = stdout_value(&named, "Total");
    // Two spellings of one Hamiltonian: the same number.
    assert_eq!(e_exact, stdout_value(&exact_word, "Total"));
    // Different Coulomb treatments: different numbers. Measured on water:
    // exact - default = 4.39e-4, named - default = -7.63e-4 (fitting error);
    // the 1e-5 floor sits 40x below both. Ignoring df_j_aux gives 0.
    assert!(
        (e_exact - e_default).abs() > 1e-5,
        "{e_exact} vs {e_default}"
    );
    assert!(
        (e_named - e_default).abs() > 1e-5,
        "{e_named} vs {e_default}"
    );
}

/// The df_guess warnings fire only for an EXPLICIT `[scf] df_guess = true`.
/// df_guess defaults on, so a condition on `df_guess_enabled()` printed the
/// "ignored" warning on every plain `rhf` run (ladder path) and every
/// open-shell run (UHF path) although the user asked for nothing. Restoring
/// `df_guess_enabled()` in either condition fails the "default" asserts; a
/// condition that never fires fails the "explicit" ones.
#[test]
fn df_guess_warnings_fire_only_when_the_user_asked_for_df_guess() {
    let water = "testdata/molecules/water.xyz";
    let stderr = |tag: &str, mult: usize, kind: &str, extra: &str| {
        let out = run_ok(tag, &body_at(water, mult, "sto-3g", kind, "energy", extra));
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    let aux = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    let explicit = "[scf]\ndf_guess = true\n";

    let rhf_default = stderr("dfg_rhf_default", 1, "rhf", "");
    assert!(!rhf_default.contains("df_guess"), "{rhf_default}");
    let rhf_explicit = stderr("dfg_rhf_explicit", 1, "rhf", explicit);
    assert!(
        rhf_explicit.contains("df_guess is not yet composed with the rhf convergence ladder"),
        "{rhf_explicit}"
    );

    let uhf_default = stderr("dfg_rimp2_triplet_default", 3, "rimp2", aux);
    assert!(!uhf_default.contains("df_guess"), "{uhf_default}");
    let uhf_explicit = stderr(
        "dfg_rimp2_triplet_explicit",
        3,
        "rimp2",
        &format!("{aux}{explicit}"),
    );
    assert!(
        uhf_explicit.contains("df_guess / df_increments are closed-shell only"),
        "{uhf_explicit}"
    );
}
