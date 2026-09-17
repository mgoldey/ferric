//! **The ROHF half of the open-shell guess defect.** Is ferric's ROHF state
//! multi-valued across orbital guesses, and how broad is the damage?
//!
//! Hypotheses were pre-registered in
//! `tests/HYPOTHESES-rohf-guess-selection.md` BEFORE any number in this file was
//! measured (commit `629e5db1`). Read that first.
//!
//! # Why this exists, separately from `scf_state_selection.rs`
//!
//! `rohf.rs` carried the identical `let _ = hcore_guess(...)` pattern that
//! `uhf.rs` did: the guess density was computed and immediately DROPPED, then
//! bare `h` was diagonalized, and neither `RhfConfig::init_guess_density` nor
//! `use_sad_guess` was referenced anywhere in the file. The UHF lane measured
//! ROHF's damage (`rohf_still_uses_hcore_measure_whether_it_matters`) and
//! deliberately did NOT fix it, because ROHF's Fock construction and coupling
//! coefficients are its own and re-validating them is separate work. This file
//! is that work.
//!
//! # The four hypotheses and the statistic that separates them
//!
//! * **H-GUESS** — ROHF's states are separate SCF basins and the guess picks
//!   one. Observable: E is MULTI-VALUED across guesses; some guess reaches the
//!   PySCF ROHF reference.
//! * **H-DEGEN** — an ordering defect in the Roothaan effective-Fock
//!   diagonalization fixes the occupation at iteration 1. Observable: E is
//!   SINGLE-VALUED at the wrong state across ALL guesses, *including a guess
//!   built from a density that is already correct*.
//! * **H-ARTIFACT** — the guess rows are not actually different inputs.
//!   Observable: BIT-identical energies, `|ΔE|` exactly `0.0`.
//! * **H-NOTAPPLICABLE** — ROHF is materially less affected than UHF.
//!   Observable: the guess changes few or no systems.
//!
//! # A note on ROHF stability analysis: it does NOT exist
//!
//! `stability.rs` implements exactly two operators (`uhf_internal_stability`,
//! `rhf_internal_stability`), and `StabilitySkip::Rohf` exists specifically to
//! record that the Roothaan open-shell Hessian is a THIRD one. So the UHF
//! fix's second half — `scf_stability_descent` — is UNAVAILABLE here, and this
//! file does not fabricate a verdict on the wrong operator. Every ROHF row
//! below reports `stability: None`, which is documented as "not checked" and
//! does NOT mean stable.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;

const HARTREE_TO_EV: f64 = 27.211_386_245_988;

// ===========================================================================
// External references — PySCF 2.13.0 ROHF, produced by a DIFFERENT code than
// ferric at the same geometries and bases. Generated with conv_tol = 1e-12,
// conv_tol_grad = 1e-9, and cross-checked across FIVE PySCF init_guess settings
// (hcore / 1e / atom / huckel / minao) so a guess-dependent reference cannot be
// mistaken for a converged one. Where PySCF itself is guess-dependent, the
// LOWEST converged value is the reference and the fact is recorded on the
// constant.
// ===========================================================================

/// OH doublet/6-31G, R = 0.97 Å. PySCF ROHF, and notably SINGLE-VALUED across
/// all five PySCF guesses including hcore — PySCF reaches this from hcore.
const OH_631G: f64 = -75.361_846_292_5;
/// OH doublet/cc-pVDZ, R = 0.97 Å. Second basis. Also PySCF-single-valued.
const OH_CCPVDZ: f64 = -75.390_002_841_2;
/// HeNe⁺ doublet/def2-SVP, R = 2.0 Å. PySCF ROHF is GUESS-DEPENDENT here:
/// hcore/1e/huckel give −130.4965141210, atom/minao give this lower value. So
/// this system is a genuine two-basin case in the reference implementation too.
const HENE_SVP: f64 = -130.501_303_395_8;
/// HeNe⁺ doublet/6-31G, R = 2.0 Å. Second basis for the same system.
/// PySCF: four guesses give this, huckel alone gives −130.5989400379.
const HENE_631G: f64 = -130.603_322_907_5;
/// N₂⁺ doublet/6-31G, R = 1.1160 Å.
const N2P_631G: f64 = -108.280_584_256_9;
/// N₂⁺ doublet/cc-pVDZ, R = 1.1160 Å. Second basis.
const N2P_CCPVDZ: f64 = -108.368_999_918_9;
/// O₂ triplet/6-31G, R = 1.2075 Å.
const O2_631G: f64 = -149.527_996_633_9;
/// O₂ triplet/cc-pVDZ, R = 1.2075 Å. Second basis.
const O2_CCPVDZ: f64 = -149.608_084_466_2;
/// NO doublet/6-31G, R = 1.1508 Å.
const NO_631G: f64 = -129.168_586_352_5;
/// NO doublet/cc-pVDZ, R = 1.1508 Å. Second basis.
const NO_CCPVDZ: f64 = -129.253_641_192_3;
/// CH₃ doublet/6-31G, planar D3h, R(CH) = 1.079 Å. A POLYATOMIC, so the sweep
/// is not all diatomics.
const CH3_631G: f64 = -39.543_358_941_4;
/// CH₃ doublet/cc-pVDZ. Second basis.
const CH3_CCPVDZ: f64 = -39.559_637_670_7;
/// CO⁺ doublet/6-31G, R = 1.1150 Å.
const COP_631G: f64 = -112.173_400_544_7;
/// CN doublet/6-31G, R = 1.1718 Å. A hard open shell.
const CN_631G: f64 = -92.139_765_531_4;
/// B₂ triplet/6-31G, R = 1.590 Å.
const B2_631G: f64 = -49.058_036_785_0;

// ===========================================================================
// Scaffolding
// ===========================================================================

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn sys_from_xyz(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn diatomic(a: &str, b: &str, r_ang: f64, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let xyz = format!("2\n{a}{b}\n{a} 0.0 0.0 0.0\n{b} 0.0 0.0 {r_ang}\n");
    sys_from_xyz(&xyz, charge, mult, basis_name)
}

/// Planar D3h CH₃, R(CH) = 1.079 Å — the same geometry the PySCF reference used.
fn ch3(basis_name: &str) -> Sys {
    let xyz = "4\nCH3\n\
               C 0.0 0.0 0.0\n\
               H 1.079 0.0 0.0\n\
               H -0.5395 0.934441 0.0\n\
               H -0.5395 -0.934441 0.0\n";
    sys_from_xyz(xyz, 0, 2, basis_name)
}

/// A tight ROHF config with no level shift and no XC. `check_stability` is NOT
/// set: ROHF has no stability operator (see the module doc), so setting it would
/// only print a skip on every row.
fn tight_cfg() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        density_conv: 1e-10,
        energy_conv: 1e-11,
        ..Default::default()
    }
}

/// The PRE-FIX config: the bare hcore guess, which is what `solve_rohf` did
/// unconditionally until this lane. `use_sad_guess = false` with no explicit
/// density is the documented way to ask for it.
fn hcore_cfg() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        ..tight_cfg()
    }
}

/// The MINAO-projection guess, which is what `use_sad_guess = true` resolves to
/// and what `rhf.rs` has defaulted to for a long time.
fn minao_cfg() -> RhfConfig {
    RhfConfig {
        use_sad_guess: true,
        ..tight_cfg()
    }
}

/// Every system in the sweep, with its PySCF ROHF reference.
///
/// TWO BASES for seven of the systems. This is the structural defence against
/// the single-input blind spot pre-registered in the hypotheses doc: a
/// comparison at one input is unfalsifiable regardless of tolerance (measured in
/// this lane's sibling — an anchor matched PySCF to 3e-13 Ha with ferric's
/// solver DELETED), whereas no constant satisfies −75.3618462925 and
/// −130.6033229075 and −149.5279966339 and −39.5433589414 at once.
fn sweep() -> Vec<(&'static str, Sys, f64)> {
    vec![
        ("OH/6-31G", diatomic("O", "H", 0.97, 0, 2, "6-31g"), OH_631G),
        (
            "OH/cc-pVDZ",
            diatomic("O", "H", 0.97, 0, 2, "cc-pvdz"),
            OH_CCPVDZ,
        ),
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
            HENE_SVP,
        ),
        (
            "HeNe+/6-31G",
            diatomic("He", "Ne", 2.0, 1, 2, "6-31g"),
            HENE_631G,
        ),
        (
            "N2+/6-31G",
            diatomic("N", "N", 1.1160, 1, 2, "6-31g"),
            N2P_631G,
        ),
        (
            "N2+/cc-pVDZ",
            diatomic("N", "N", 1.1160, 1, 2, "cc-pvdz"),
            N2P_CCPVDZ,
        ),
        (
            "O2/6-31G",
            diatomic("O", "O", 1.2075, 0, 3, "6-31g"),
            O2_631G,
        ),
        (
            "O2/cc-pVDZ",
            diatomic("O", "O", 1.2075, 0, 3, "cc-pvdz"),
            O2_CCPVDZ,
        ),
        (
            "NO/6-31G",
            diatomic("N", "O", 1.1508, 0, 2, "6-31g"),
            NO_631G,
        ),
        (
            "NO/cc-pVDZ",
            diatomic("N", "O", 1.1508, 0, 2, "cc-pvdz"),
            NO_CCPVDZ,
        ),
        ("CH3/6-31G", ch3("6-31g"), CH3_631G),
        ("CH3/cc-pVDZ", ch3("cc-pvdz"), CH3_CCPVDZ),
        (
            "CO+/6-31G",
            diatomic("C", "O", 1.1150, 1, 2, "6-31g"),
            COP_631G,
        ),
        (
            "CN/6-31G",
            diatomic("C", "N", 1.1718, 0, 2, "6-31g"),
            CN_631G,
        ),
        (
            "B2/6-31G",
            diatomic("B", "B", 1.590, 0, 3, "6-31g"),
            B2_631G,
        ),
    ]
}

fn run(sys: &Sys, cfg: &RhfConfig) -> Result<f64, String> {
    solve_rohf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        cfg,
    )
    .map(|r| r.energy)
    .map_err(|e| format!("{e:?}"))
}

// ===========================================================================
// PART 1 — the discriminating experiment
// ===========================================================================

/// **The guess × system table.** Runs every system under the hcore guess (what
/// ROHF did unconditionally before this lane), the MINAO guess, and an explicit
/// `init_guess_density` taken from the CONVERGED MINAO density, then reports
/// each against its PySCF ROHF reference.
///
/// The third row is the near-decisive one for H-DEGEN: a guess density that is
/// already AT a correct answer cannot be "the wrong basin". If ROHF still walked
/// away from it, the defect would not be basin selection.
///
/// This test asserts CONVERGENCE and the H-ARTIFACT exclusion only. It does NOT
/// assert the deltas, because the deltas are the measurement; the accuracy
/// assertions live in `the_fixed_rohf_path_reaches_every_reference`.
#[test]
fn rohf_guess_by_system_table() {
    println!("\n=== ROHF guess x system, vs PySCF 2.13.0 ROHF ===");
    println!(
        "{:16}  {:>17}  {:>17}  {:>17}  {:>10}  {:>10}",
        "system", "hcore", "minao", "ref(pyscf)", "dE_hcore", "dE_minao"
    );
    let mut n_wrong_hcore = 0usize;
    let mut n_wrong_minao = 0usize;
    let mut worst_hcore: (f64, String) = (0.0, String::new());
    let mut worst_minao: (f64, String) = (0.0, String::new());
    let mut identical_rows: Vec<String> = Vec::new();

    for (name, sys, e_ref) in sweep() {
        let e_h = run(&sys, &hcore_cfg());
        let e_m = run(&sys, &minao_cfg());
        match (&e_h, &e_m) {
            (Ok(eh), Ok(em)) => {
                let d_h = (eh - e_ref) * HARTREE_TO_EV;
                let d_m = (em - e_ref) * HARTREE_TO_EV;
                println!(
                    "{name:16}  {eh:17.10}  {em:17.10}  {e_ref:17.10}  {d_h:+10.4}  {d_m:+10.4}"
                );
                if d_h > 1e-3 {
                    n_wrong_hcore += 1;
                    if d_h > worst_hcore.0 {
                        worst_hcore = (d_h, name.to_string());
                    }
                }
                if d_m > 1e-3 {
                    n_wrong_minao += 1;
                    if d_m > worst_minao.0 {
                        worst_minao = (d_m, name.to_string());
                    }
                }
                // H-ARTIFACT check: on a system where the two guesses disagree
                // at all, they must not be BIT-identical. Exact 0.0 across
                // every row would mean the config field never reached the
                // solver, i.e. one guess measured twice.
                if eh == em {
                    identical_rows.push(name.to_string());
                }
            }
            _ => println!("{name:16}  hcore={e_h:?}  minao={e_m:?}"),
        }
    }
    println!(
        "\nABOVE the reference by >1e-3 eV:  hcore {n_wrong_hcore}/{n}, minao {n_wrong_minao}/{n}",
        n = sweep().len()
    );
    println!(
        "worst hcore: {:+.4} eV ({}); worst minao: {:+.4} eV ({})",
        worst_hcore.0, worst_hcore.1, worst_minao.0, worst_minao.1
    );
    println!(
        "bit-identical guess rows ({} of {}): {identical_rows:?}",
        identical_rows.len(),
        sweep().len()
    );

    // H-ARTIFACT is excluded by this: if EVERY row were bit-identical the two
    // configs would not be two inputs at all and no physics claim could be made
    // from this table. (Individual systems being identical is expected and fine
    // — it means that system has one basin.)
    assert!(
        identical_rows.len() < sweep().len(),
        "H-ARTIFACT: every guess row is BIT-identical, so use_sad_guess never \
         reached the ROHF solver and this table measures one guess twice"
    );
}
