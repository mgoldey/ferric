//! TOML keys that wire library capabilities which used to be reachable only
//! from Python into the CLI. Each test runs a tiny input through the real
//! `ferric-cli` binary and holds the printed number to the SAME library call
//! the Python binding makes (or checks that a refusal fires with the right
//! message). Every numerical test also carries a negative control: the key
//! must CHANGE the answer, or "the CLI matches the library" would be true of
//! a key that is silently ignored (both sides would equal the default run).
//!
//! Tolerance: the CLI prints 10 decimals, so agreement is asserted at 1e-8
//! Ha, with SCF thresholds tight enough (density 1e-9) that the converged
//! energies of two slightly different SCF paths (the CLI's ladder rung 0 vs a
//! bare `solve_rhf`, a DF-guess pre-stage vs none) agree far below that.
//!
//! The terf/terfc tests need the interpolation tables and skip cleanly when
//! `FERRIC_TERF_TABLE_DIR` does not point at them (same policy as
//! `r0_sweep.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, SmearedCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::{SchwarzBounds, ScreeningKind};
use ferric_scf::ScfResult;

/// Path to the `ferric-cli` binary, resolved at RUN TIME (see `r0_sweep.rs`
/// for why a raw `env!` breaks under `cargo nextest archive`).
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

/// Workspace root at RUN time (same helper as `r0_sweep.rs`).
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

fn terf_dir() -> Option<String> {
    let d = std::env::var("FERRIC_TERF_TABLE_DIR").ok()?;
    if !d.is_empty() && PathBuf::from(&d).join("16_4_2.bin").exists() {
        Some(d)
    } else {
        None
    }
}

/// Run the CLI on `body` (written under `target/`, one file per `tag` so
/// parallel tests never share a file).
fn run_toml(tag: &str, body: &str) -> std::process::Output {
    let root = workspace_root();
    let dir = root.join("target");
    // Absent under a custom CARGO_TARGET_DIR or a nextest archive.
    std::fs::create_dir_all(&dir).expect("create target/ for the temp toml");
    let path = dir.join(format!("cli_wired_keys_{tag}.toml"));
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

fn run_ok(tag: &str, body: &str) -> String {
    let out = run_toml(tag, body);
    assert!(
        out.status.success(),
        "{tag} must run.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn assert_refused(tag: &str, body: &str, needles: &[&str]) {
    let out = run_toml(tag, body);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{tag} must be REFUSED, but the run succeeded.\nstdout: {}\nstderr: {err}",
        String::from_utf8_lossy(&out.stdout)
    );
    for n in needles {
        assert!(
            err.contains(n),
            "{tag}: error must mention {n:?}, got: {err}"
        );
    }
}

/// The number after the first `=` or `:` on the first stdout line whose
/// trimmed text starts with `label`.
fn value(stdout: &str, label: &str) -> f64 {
    stdout
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(label))
        .and_then(|l| {
            let rest = &l[label.len()..];
            let rest = rest.trim_start_matches([' ', '=', ':']);
            rest.split_whitespace().next()
        })
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no {label:?} line in stdout:\n{stdout}"))
}

/// Molecule + basis + method header, then `extra` verbatim.
fn body(xyz: &str, mult: usize, basis: &str, kind: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"testdata/molecules/{xyz}\"\nmultiplicity = {mult}\n\n\
         [basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\ntask = \"energy\"\n\n{extra}"
    )
}

/// Tight SCF thresholds shared by every energy comparison below.
const TIGHT: &str = "[scf]\ndensity_conv = 1e-9\nenergy_conv = 1e-10\nmax_iter = 200\n";

/// The library side: the molecule, basis and bounds exactly as `run()` builds
/// them (ECP applied, CSB/Schwarz resolved from the default screening kind).
struct Lib {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn lib(xyz: &str, mult: usize, basis_name: &str) -> Lib {
    let path = workspace_root().join("testdata/molecules").join(xyz);
    let mut mol = Molecule::load_xyz_with_charge(path.to_str().unwrap(), 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds =
        SchwarzBounds::compute_for_screening(Operator::coulomb(), &prep, ScreeningKind::default())
            .unwrap();
    Lib {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// `RhfConfig::default()` with the thresholds of [`TIGHT`].
fn tight() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        density_conv: 1e-9,
        energy_conv: 1e-10,
        ..Default::default()
    }
}

fn rhf(l: &Lib, cfg: &RhfConfig) -> ScfResult {
    let r = solve_rhf(&l.ctx, &l.mol, &l.prep, Operator::coulomb(), &l.bounds, cfg).unwrap();
    assert!(r.converged, "library reference SCF did not converge");
    r
}

fn assert_close(what: &str, cli: f64, lib: f64, tol: f64) {
    assert!(
        (cli - lib).abs() < tol,
        "{what}: CLI {cli:.10} vs library {lib:.10} (|d| = {:.2e} > {tol:.0e})",
        (cli - lib).abs()
    );
}

// ─── [pcm] ──────────────────────────────────────────────────────────────────

/// `[pcm] solvent = "water"` on `rhf` is `solve_rhf` with
/// `PcmConfig::for_solvent("water")` (what the Python `run_rhf(solvent=
/// "water")` builds), and `epsilon = 78.4` is the same run. Negative control:
/// the vacuum energy differs by the solvation energy (~6 mHa here).
#[test]
fn pcm_rhf_matches_the_library_and_solvates() {
    let l = lib("water.xyz", 1, "sto-3g");
    let e_lib = rhf(
        &l,
        &RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig::for_solvent("water").unwrap()),
            ..tight()
        },
    )
    .energy;
    let by_name = run_ok(
        "pcm_name",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rhf",
            &format!("{TIGHT}[pcm]\nsolvent = \"water\"\n"),
        ),
    );
    let by_eps = run_ok(
        "pcm_eps",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rhf",
            &format!("{TIGHT}[pcm]\nepsilon = 78.4\n"),
        ),
    );
    let vac = run_ok("pcm_vacuum", &body("water.xyz", 1, "sto-3g", "rhf", TIGHT));
    let (e_name, e_eps, e_vac) = (
        value(&by_name, "energy "),
        value(&by_eps, "energy "),
        value(&vac, "energy "),
    );
    assert_close("[pcm] solvent = water", e_name, e_lib, 1e-8);
    assert_close("[pcm] epsilon = 78.4", e_eps, e_lib, 1e-8);
    assert!(
        e_vac - e_name > 1e-3,
        "PCM must stabilise water: vacuum {e_vac:.10}, solvated {e_name:.10}"
    );
}

/// `uhf` honours `[pcm]` too (the reaction field is folded in by the shared
/// SCF driver): on the closed-shell water singlet UHF must reproduce the
/// solvated RHF energy of the library.
#[test]
fn pcm_uhf_on_a_singlet_matches_solvated_rhf() {
    let l = lib("water.xyz", 1, "sto-3g");
    let e_lib = rhf(
        &l,
        &RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig::for_solvent("water").unwrap()),
            ..tight()
        },
    )
    .energy;
    let out = run_ok(
        "pcm_uhf",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "uhf",
            &format!("{TIGHT}[pcm]\nsolvent = \"water\"\n"),
        ),
    );
    assert_close("uhf + [pcm]", value(&out, "energy "), e_lib, 1e-8);
}

#[test]
fn pcm_is_refused_where_it_is_not_honoured() {
    let pcm = "[pcm]\nsolvent = \"water\"\n";
    assert_refused(
        "pcm_rimp2",
        &body("water.xyz", 1, "sto-3g", "rimp2", pcm),
        &["[pcm]", "rimp2"],
    );
    assert_refused(
        "pcm_opt",
        &body("water.xyz", 1, "sto-3g", "rhf", pcm)
            .replace("task = \"energy\"", "task = \"optimize\""),
        &["[pcm]", "no gradient"],
    );
    assert_refused(
        "pcm_bad_name",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rhf",
            "[pcm]\nsolvent = \"watr\"\n",
        ),
        &["not recognised"],
    );
}

// ─── [scf] stability_descent ────────────────────────────────────────────────

/// O2 triplet / STO-3G: the default UHF guess lands on a SADDLE (-147.63397);
/// the descent reaches the UHF minimum (-147.63530, see
/// `ferric-scf/tests/uhf_o2_sto3g.rs`). The CLI key must give the library's
/// `check_stability + scf_stability_descent` result (the Python
/// `run_uhf(stability_descent=True)` config), and must differ from the run
/// without it by the ~1.3 mHa saddle-to-minimum gap.
#[test]
fn stability_descent_reaches_the_library_uhf_minimum() {
    let l = lib("o2.xyz", 3, "sto-3g");
    let r = ferric_scf::uhf::solve_uhf(
        &l.ctx,
        &l.mol,
        &l.prep,
        &l.bounds,
        &RhfConfig {
            check_stability: true,
            scf_stability_descent: true,
            ..tight()
        },
    )
    .unwrap();
    assert!(r.converged);
    let with = run_ok(
        "sd_on",
        &body(
            "o2.xyz",
            3,
            "sto-3g",
            "uhf",
            &format!("{TIGHT}stability_descent = true\n"),
        ),
    );
    let without = run_ok("sd_off", &body("o2.xyz", 3, "sto-3g", "uhf", TIGHT));
    let (e_on, e_off) = (value(&with, "energy "), value(&without, "energy "));
    assert_close("uhf + stability_descent", e_on, r.energy, 1e-8);
    assert!(
        e_off - e_on > 1e-3,
        "the descent must lower the saddle: without {e_off:.10}, with {e_on:.10}"
    );
}

#[test]
fn stability_descent_is_refused_off_the_uhf_route() {
    let sd = "[scf]\nstability_descent = true\n";
    assert_refused(
        "sd_rohf",
        &body("o2.xyz", 3, "sto-3g", "rohf", sd),
        &["stability_descent", "ROHF"],
    );
    assert_refused(
        "sd_rhf",
        &body("water.xyz", 1, "sto-3g", "rhf", sd),
        &["stability_descent", "UHF/UKS route only"],
    );
}

// ─── [dft] grid_radial / grid_angular ───────────────────────────────────────

/// `ksdft` PBE with a 99x302 grid is the library KS solve on
/// `AtomicGridConfig { n_radial: 99, n_angular: 302, prune: None }` (the
/// Python `grid_radial=99, grid_angular=302` grid), under the CLI's default
/// RI-JK aux for KS. Negative control: the default 75x110 grid differs.
#[test]
fn dft_grid_keys_match_the_library_grid() {
    let l = lib("water.xyz", 1, "sto-3g");
    let jk = Some("def2-universal-jkfit".to_string());
    let e_lib = rhf(
        &l,
        &RhfConfig {
            xc: Some("PBE".into()),
            df_j_aux: jk.clone(),
            df_k_aux: jk,
            dft_grid: Some(ferric_dft::grid::AtomicGridConfig {
                n_radial: 99,
                n_angular: 302,
                prune: None,
            }),
            ..tight()
        },
    )
    .energy;
    let custom = run_ok(
        "grid_custom",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "ksdft",
            &format!("{TIGHT}[dft]\nfunctional = \"PBE\"\ngrid_radial = 99\ngrid_angular = 302\n"),
        ),
    );
    let default = run_ok(
        "grid_default",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "ksdft",
            &format!("{TIGHT}[dft]\nfunctional = \"PBE\"\n"),
        ),
    );
    let (e_c, e_d) = (value(&custom, "energy "), value(&default, "energy "));
    assert_close("ksdft 99x302", e_c, e_lib, 1e-8);
    assert!(
        (e_c - e_d).abs() > 1e-8,
        "a 99x302 grid must change the energy vs 75x110: {e_c:.10} vs {e_d:.10}"
    );
}

#[test]
fn dft_grid_keys_are_refused_where_they_cannot_apply() {
    assert_refused(
        "grid_bad_angular",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "ksdft",
            "[dft]\ngrid_angular = 194\n",
        ),
        &["grid_angular", "Lebedev"],
    );
    assert_refused(
        "grid_opt",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "ksdft",
            "[dft]\ngrid_radial = 99\n",
        )
        .replace("task = \"energy\"", "task = \"optimize\""),
        &["grid_radial", "energy"],
    );
}

// ─── smeared [external_potential] charges ───────────────────────────────────

/// A `width` turns an `[[external_potential.point_charges]]` entry into a
/// Gaussian-smeared charge (Bohr): the CLI must equal `solve_rhf` with that
/// `SmearedCharge` (the Python `smeared_charges=[(q, x, y, z, width)]`).
/// Negative control: the same charge as a POINT gives a different energy.
#[test]
fn a_smeared_charge_matches_the_library() {
    let (q, x, y, z, w) = (-0.8, 0.0, 0.0, 3.5, 1.5);
    let l = lib("water.xyz", 1, "sto-3g");
    let e_lib = rhf(
        &l,
        &RhfConfig {
            external_potential: Some(ExternalPotential {
                point_charges: Vec::new(),
                smeared_charges: vec![SmearedCharge {
                    q,
                    x,
                    y,
                    z,
                    width: w,
                }],
                field: None,
            }),
            ..tight()
        },
    )
    .energy;
    let charge = |width: &str| {
        format!(
            "{TIGHT}[[external_potential.point_charges]]\nq = {q:?}\nx = {x:?}\ny = {y:?}\nz = {z:?}\n{width}"
        )
    };
    let smeared = run_ok(
        "smeared",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rhf",
            &charge(&format!("width = {w:?}\n")),
        ),
    );
    let point = run_ok(
        "smeared_point",
        &body("water.xyz", 1, "sto-3g", "rhf", &charge("")),
    );
    let (e_s, e_p) = (value(&smeared, "energy "), value(&point, "energy "));
    assert_close("smeared charge", e_s, e_lib, 1e-8);
    assert!(
        (e_s - e_p).abs() > 1e-6,
        "smearing must change the energy vs a point charge: {e_s:.10} vs {e_p:.10}"
    );
    assert_refused(
        "smeared_zero_width",
        &body("water.xyz", 1, "sto-3g", "rhf", &charge("width = 0.0\n")),
        &["point_charges[0]", "width"],
    );
}

// ─── [mp2] oo_* ─────────────────────────────────────────────────────────────

/// Closed shell: the CLI's `oo_*` keys build the same `OoRiMp2Config` as the
/// Python `run_oo_rimp2(max_iter=, grad_conv=, level_shift=, diis_size=)`.
#[test]
fn oo_keys_match_the_library_closed_shell() {
    let l = lib("water.xyz", 1, "sto-3g");
    let r = rhf(&l, &tight());
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&l.mol, &aux).unwrap();
    let oo_cfg = ferric_mp2::oo_rimp2::OoRiMp2Config {
        max_iter: 80,
        grad_conv: 1e-7,
        level_shift: 0.2,
        diis_size: 4,
        ..Default::default()
    };
    let o = ferric_mp2::oo_rimp2::oo_ri_mp2(
        &l.mol,
        &l.prep,
        &dfbs,
        Operator::coulomb(),
        &l.bounds,
        &r,
        &oo_cfg,
        None,
    )
    .unwrap();
    assert!(o.converged);
    let out = run_ok(
        "oo_closed",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "oo-rimp2",
            &format!(
                "{TIGHT}[mp2]\noo_max_iter = 80\noo_grad_conv = 1e-7\noo_level_shift = 0.2\n\
                 oo_diis_size = 4\n"
            ),
        ),
    );
    assert_close("oo-rimp2 Total", value(&out, "Total"), o.total_energy, 1e-8);
}

/// Both the closed-shell and the open-shell (UHF-reference, `u_oo_ri_mp2`)
/// paths must read `oo_max_iter`: capped at 2, each prints `iterations = 2`
/// and does not converge, where the default converges.
#[test]
fn oo_max_iter_reaches_both_the_closed_and_open_shell_paths() {
    for (tag, xyz, mult) in [
        ("oo_cap_closed", "water.xyz", 1),
        ("oo_cap_open", "oh.xyz", 2),
    ] {
        let out = run_ok(
            tag,
            &body(
                xyz,
                mult,
                "sto-3g",
                "oo-rimp2",
                &format!("{TIGHT}[mp2]\noo_max_iter = 2\noo_grad_conv = 1e-10\n"),
            ),
        );
        assert_eq!(value(&out, "iterations"), 2.0, "{tag}:\n{out}");
        assert!(out.contains("converged  = false"), "{tag}:\n{out}");
    }
}

// ─── [gw] reference = "rohf" ────────────────────────────────────────────────

/// Open-shell `gw` with `reference = "rohf"` runs `run_u_gw` on a
/// `solve_rohf` reference (the Python `run_u_gw(reference="rohf")`): the
/// printed ROHF energy is the library's, and the alpha-HOMO QP IP matches the
/// library's U-G0W0@ROHF to the printed precision. Negative control: the
/// default (UHF) reference prints a different reference energy.
#[test]
fn gw_rohf_reference_matches_the_library() {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
    let l = lib("oh.xyz", 2, "sto-3g");
    let jk = Some("def2-universal-jkfit".to_string());
    let scf_cfg = RhfConfig {
        df_j_aux: jk.clone(),
        df_k_aux: jk,
        mom_after_iter: 5,
        ..tight()
    };
    let scf = ferric_scf::rohf::solve_rohf(
        &l.ctx,
        &l.mol,
        &l.prep,
        Operator::coulomb(),
        &l.bounds,
        &scf_cfg,
    )
    .unwrap();
    assert!(scf.converged);
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&l.mol, &aux).unwrap();
    let pdep = ferric_rpa::PdepRpaConfig {
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    };
    let gw = ferric_gw::run_u_gw(
        &l.mol,
        &l.prep,
        &dfbs,
        Operator::coulomb(),
        &scf,
        &pdep,
        &ferric_gw::GwConfig::default(),
    )
    .unwrap();
    let nocc_a = (l.mol.nelec() as usize).div_ceil(2);
    let loc = gw
        .mo_indices
        .iter()
        .position(|&m| m == nocc_a - 1)
        .expect("alpha HOMO in the default QP window");
    let ip_lib = -gw.eps_qp_a[loc] * 27.211_386_245_988_f64;

    let gw_toml = |reference: &str| {
        format!(
            "{TIGHT}[rpa]\nauxbasis = \"cc-pvdz-ri\"\nn_quad = 16\nquadrature = \"gauss-legendre\"\n\
             trunc_thresh = 0.0\n\n[gw]\nmethod = \"g0w0\"\n{reference}"
        )
    };
    let rohf_out = run_ok(
        "gw_rohf",
        &body(
            "oh.xyz",
            2,
            "sto-3g",
            "gw",
            &gw_toml("reference = \"rohf\"\n"),
        ),
    );
    let uhf_out = run_ok("gw_uhf", &body("oh.xyz", 2, "sto-3g", "gw", &gw_toml("")));
    assert!(rohf_out.contains("ref: ROHF"), "{rohf_out}");
    assert_close(
        "ROHF reference energy",
        value(&rohf_out, "ROHF energy"),
        scf.energy,
        1e-8,
    );
    let ip_cli = value(&rohf_out, "alpha-HOMO IP");
    assert!(
        (ip_cli - ip_lib).abs() < 2e-4,
        "U-G0W0@ROHF alpha-HOMO IP: CLI {ip_cli:.4} eV vs library {ip_lib:.4} eV"
    );
    let e_uhf = value(&uhf_out, "UHF energy");
    assert!(
        (e_uhf - scf.energy).abs() > 1e-5,
        "UHF and ROHF references must differ on OH: {e_uhf:.10} vs {:.10}",
        scf.energy
    );
}

#[test]
fn gw_reference_is_refused_on_a_closed_shell() {
    assert_refused(
        "gw_ref_closed",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "gw",
            "[gw]\nreference = \"rohf\"\n",
        ),
        &["[gw] reference", "open-shell"],
    );
}

// ─── [mp2] att_operator = "terfc" on att-rimp2, [mp2] terf_omega ────────────

/// `att-rimp2` with `att_operator = "terfc"`, `att_r0 = 1.05` Å is the Python
/// `run_terfc_rimp2`: `ri_mp2` with `Operator::terfc(1.05 Å in Bohr)` on the
/// Coulomb RHF. Negative control: the default erfc operator differs.
#[test]
fn att_rimp2_terfc_matches_the_library() {
    let Some(tdir) = terf_dir() else {
        eprintln!("skipping: terf interpolation tables not found");
        return;
    };
    let _ = tdir; // the CLI child and the library both read the same env var
    let l = lib("water.xyz", 1, "sto-3g");
    let r = rhf(&l, &tight());
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&l.mol, &aux).unwrap();
    let mp2 = ferric_mp2::rimp2::ri_mp2(
        &l.mol,
        &l.prep,
        &dfbs,
        Operator::terfc(1.05 * 1.8897259886),
        &r,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap();
    let terfc = run_ok(
        "att_terfc",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "att-rimp2",
            &format!("{TIGHT}[mp2]\natt_operator = \"terfc\"\natt_r0 = 1.05\n"),
        ),
    );
    let erfc = run_ok(
        "att_erfc",
        &body("water.xyz", 1, "sto-3g", "att-rimp2", TIGHT),
    );
    let (e_t, e_e) = (value(&terfc, "Total"), value(&erfc, "Total"));
    assert!(terfc.contains("(terfc)"), "{terfc}");
    assert_close("att-rimp2 terfc", e_t, mp2.total_energy, 1e-8);
    assert!(
        (e_t - e_e).abs() > 1e-6,
        "terfc and erfc must differ: {e_t:.10} vs {e_e:.10}"
    );
}

/// `rs-mp2-rpa` terf split with `terf_omega` (Å⁻¹) is the library
/// `rs_mp2_lr_rpa` with `RsMp2RpaConfig::terf_omega = Some(ω in Bohr⁻¹)` (the
/// Python `run_rs_mp2_rpa(terf_omega=)`). Negative control: without it the
/// curvature-linked ω gives a different total.
#[test]
fn rs_mp2_rpa_terf_omega_matches_the_library() {
    let Some(tdir) = terf_dir() else {
        eprintln!("skipping: terf interpolation tables not found");
        return;
    };
    let _ = tdir; // the CLI child and the library both read the same env var
    const ANG2BOHR: f64 = 1.8897259886;
    let (r0_ang, w_ang) = (1.2, 2.5);
    let l = lib("water.xyz", 1, "sto-3g");
    let jk = Some("def2-universal-jkfit".to_string());
    let r = rhf(
        &l,
        &RhfConfig {
            df_j_aux: jk.clone(),
            df_k_aux: jk,
            ..tight()
        },
    );
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&l.mol, &aux).unwrap();
    let rs = ferric_rpa::rs_mp2_lr_rpa(
        &l.mol,
        &l.prep,
        &dfbs,
        &r,
        &ferric_rpa::RsMp2RpaConfig {
            attenuator: ferric_rpa::rs_mp2_rpa::Attenuator::Terf,
            r0: r0_ang * ANG2BOHR,
            terf_omega: Some(w_ang * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV),
            ..Default::default()
        },
    )
    .unwrap();
    let toml = |extra: &str| format!("{TIGHT}[mp2]\nattenuator = \"terf\"\nr0 = {r0_ang}\n{extra}");
    let with = run_ok(
        "rs_terf_omega",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rs-mp2-rpa",
            &toml(&format!("terf_omega = {w_ang}\n")),
        ),
    );
    let without = run_ok(
        "rs_terf_linked",
        &body("water.xyz", 1, "sto-3g", "rs-mp2-rpa", &toml("")),
    );
    let (e_w, e_l) = (
        value(&with, "Total energy"),
        value(&without, "Total energy"),
    );
    assert!(with.contains("from [mp2] terf_omega"), "{with}");
    assert_close("rs-mp2-rpa terf_omega", e_w, rs.total_energy, 1e-8);
    assert!(
        (e_w - e_l).abs() > 1e-7,
        "terf_omega must change the answer: {e_w:.10} vs {e_l:.10}"
    );
}

#[test]
fn terf_keys_are_refused_off_their_kind() {
    // Pure config refusals: no tables needed, the run stops before any integral.
    assert_refused(
        "terf_omega_erf",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rs-mp2-rpa",
            "[mp2]\nterf_omega = 0.5\n",
        ),
        &["terf_omega", "attenuator = \"terf\""],
    );
    // att-rimp2 refuses the rs-mp2-rpa split keys, pointing at its own...
    assert_refused(
        "att_attenuator",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "att-rimp2",
            "[mp2]\nattenuator = \"terfc\"\n",
        ),
        &["attenuator", "use att_operator"],
    );
    assert_refused(
        "att_rs_r0",
        &body("water.xyz", 1, "sto-3g", "att-rimp2", "[mp2]\nr0 = 1.0\n"),
        &["use att_operator"],
    );
    // ...its own keys are strict...
    assert_refused(
        "att_erfc_att_r0",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "att-rimp2",
            "[mp2]\natt_r0 = 1.0\n",
        ),
        &["att_r0", "erfc"],
    );
    // ...and rs-mp2-rpa refuses them.
    assert_refused(
        "rs_att_operator",
        &body(
            "water.xyz",
            1,
            "sto-3g",
            "rs-mp2-rpa",
            "[mp2]\natt_operator = \"terfc\"\n",
        ),
        &["att_operator", "\"att-rimp2\" only"],
    );
}

/// `[frequencies] hessian`: water/STO-3G frequencies with the analytic Hessian
/// and with finite differences, each reporting which ran, agreeing to the FD
/// step's truncation; an unknown value is refused. Negative control: the key
/// changes the construction (the printed `Hessian` line and the gradient-count
/// line differ), so a key the CLI ignored could not pass.
#[test]
fn frequencies_hessian_key_selects_the_construction() {
    let toml = |hessian: &str| {
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
             [basis]\nname = \"sto-3g\"\n\n\
             [method]\nkind = \"rhf\"\ntask = \"frequencies\"\n\n\
             [frequencies]\nhessian = \"{hessian}\"\n\n{TIGHT}"
        )
    };
    let modes = |stdout: &str| -> Vec<f64> {
        stdout
            .lines()
            .skip_while(|l| !l.contains("frequency (cm^-1)"))
            .skip(1)
            .map_while(|l| {
                let mut it = l.split_whitespace();
                it.next()?.parse::<usize>().ok()?;
                it.next()?.parse::<f64>().ok()
            })
            .collect()
    };
    let an = run_ok("freq_analytic", &toml("analytic"));
    let fd = run_ok("freq_fd", &toml("fd"));
    assert!(an.contains("Hessian           = analytic"), "{an}");
    assert!(fd.contains("Hessian           = finite-difference"), "{fd}");
    assert_eq!(value(&an, "gradient evals"), 0.0);
    assert_eq!(value(&fd, "gradient evals"), 18.0);
    let (wa, wf) = (modes(&an), modes(&fd));
    assert_eq!(wa.len(), 3, "{an}");
    assert_eq!(wf.len(), 3, "{fd}");
    for (a, f) in wa.iter().zip(&wf) {
        assert!((a - f).abs() < 1.0, "analytic {a} vs FD {f} cm^-1");
    }
    assert_refused("freq_bad", &toml("numerical"), &["hessian", "numerical"]);
}
