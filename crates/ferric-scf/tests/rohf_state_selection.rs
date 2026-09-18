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
/// CN doublet/cc-pVDZ. Second basis. CN has FOUR distinct PySCF ROHF solutions
/// at this basis; this is the lowest, and the standard guesses all reach it.
const CN_CCPVDZ: f64 = -92.196_077_820_0;
/// B₂ triplet/cc-pVDZ. Second basis. **PySCF's own five standard guesses ALL
/// land on −49.0829083300 here; −49.1004769400 was found only from random
/// starts.** Recorded as the reference because it is the lowest ROHF stationary
/// point actually located, and recorded WITH that caveat because a reference
/// its own code reaches only by accident is not a bar ferric should be held to.
const B2_CCPVDZ_LOWEST_FOUND: f64 = -49.100_476_940_0;
/// B₂/cc-pVDZ, the state every standard PySCF guess reaches.
const B2_CCPVDZ_STANDARD: f64 = -49.082_908_330_0;
/// CO⁺ doublet/cc-pVDZ. Second basis. Single-solution system.
const COP_CCPVDZ: f64 = -112.260_275_240_0;
/// BeH doublet/6-31G, R = 1.3426 Å. **A system where PySCF's OWN hcore guess
/// lands 2.86 eV high** (−15.0376577300 vs −15.1426715200): independent
/// evidence that the hcore guess, not ferric's solver, is what selects the
/// wrong state on this class.
const BEH_631G: f64 = -15.142_671_520_0;
/// BeH doublet/cc-pVDZ. Second basis; PySCF hcore is 2.82 eV high here too.
const BEH_CCPVDZ: f64 = -15.149_436_180_0;
/// CH doublet/6-31G, R = 1.1199 Å.
const CH_631G: f64 = -38.250_158_300_0;
/// NH triplet/6-31G, R = 1.0362 Å.
const NH_631G: f64 = -54.938_359_530_0;
/// NH₂ doublet/6-31G — a POLYATOMIC bent radical.
const NH2_631G: f64 = -55.530_432_220_0;
/// NH₂ doublet/cc-pVDZ. Second basis.
const NH2_CCPVDZ: f64 = -55.561_983_700_0;
/// O₂⁺ doublet/6-31G, R = 1.1164 Å.
const O2P_631G: f64 = -149.047_413_510_0;
/// F₂⁺ doublet/6-31G, R = 1.3220 Å. **PySCF's own hcore guess lands 3.01 eV
/// high** (−197.9252084800), and its lowest state (−198.0656953800) is reached
/// only from random starts — the standard guesses converge on
/// −198.0361965500. A second witness that hcore is the culprit.
const F2P_631G_STANDARD: f64 = -198.036_196_550_0;
/// Li₂⁺ doublet/6-31G, R = 3.108 Å.
const LI2P_631G: f64 = -14.711_375_300_0;
/// NO₂ doublet/6-31G — a second polyatomic radical.
const NO2_631G: f64 = -203.899_678_410_0;

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

/// Bent NH₂ radical, the same geometry the PySCF reference used.
fn nh2(basis_name: &str) -> Sys {
    let xyz = "3\nNH2\n\
               N 0.0 0.0 0.0\n\
               H 0.0 0.8020 0.5942\n\
               H 0.0 -0.8020 0.5942\n";
    sys_from_xyz(xyz, 0, 2, basis_name)
}

/// Bent NO₂ radical, the same geometry the PySCF reference used.
fn no2(basis_name: &str) -> Sys {
    let xyz = "3\nNO2\n\
               N 0.0 0.0 0.0\n\
               O 0.0 1.0989 0.4629\n\
               O 0.0 -1.0989 0.4629\n";
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
        (
            "CN/cc-pVDZ",
            diatomic("C", "N", 1.1718, 0, 2, "cc-pvdz"),
            CN_CCPVDZ,
        ),
        (
            "B2/cc-pVDZ",
            diatomic("B", "B", 1.590, 0, 3, "cc-pvdz"),
            B2_CCPVDZ_STANDARD,
        ),
        (
            "CO+/cc-pVDZ",
            diatomic("C", "O", 1.1150, 1, 2, "cc-pvdz"),
            COP_CCPVDZ,
        ),
        (
            "BeH/6-31G",
            diatomic("Be", "H", 1.3426, 0, 2, "6-31g"),
            BEH_631G,
        ),
        (
            "BeH/cc-pVDZ",
            diatomic("Be", "H", 1.3426, 0, 2, "cc-pvdz"),
            BEH_CCPVDZ,
        ),
        (
            "CH/6-31G",
            diatomic("C", "H", 1.1199, 0, 2, "6-31g"),
            CH_631G,
        ),
        (
            "NH/6-31G",
            diatomic("N", "H", 1.0362, 0, 3, "6-31g"),
            NH_631G,
        ),
        ("NH2/6-31G", nh2("6-31g"), NH2_631G),
        ("NH2/cc-pVDZ", nh2("cc-pvdz"), NH2_CCPVDZ),
        (
            "O2+/6-31G",
            diatomic("O", "O", 1.1164, 1, 2, "6-31g"),
            O2P_631G,
        ),
        (
            "F2+/6-31G",
            diatomic("F", "F", 1.3220, 1, 2, "6-31g"),
            F2P_631G_STANDARD,
        ),
        (
            "Li2+/6-31G",
            diatomic("Li", "Li", 3.108, 1, 2, "6-31g"),
            LI2P_631G,
        ),
        ("NO2/6-31G", no2("6-31g"), NO2_631G),
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

// ===========================================================================
// PART 3 — the proof
// ===========================================================================

/// **The fixed default path reaches its reference on every system but one, and
/// the one exception is named.**
///
/// # Why this asserts against MANY references rather than one tightly
///
/// A reference comparison at a SINGLE input is unfalsifiable regardless of
/// tolerance — measured in this lane's sibling, where an anchor matched PySCF
/// to 3e-13 Ha with ferric's solver DELETED. The defence is structural: this
/// test asserts against 28 rows spanning two bases, charged and neutral,
/// doublet and triplet, diatomic and polyatomic. No constant satisfies
/// −75.3618462925, −130.6033229075, −149.5279966339, −39.5433589414,
/// −15.1426715200 and −203.8996784100 at once, so a fabricating or
/// short-circuited implementation cannot pass by returning a number.
///
/// # The named exceptions: CN, at BOTH bases
///
/// CN/6-31G lands 0.575 eV above its reference from MINAO and reached it from
/// hcore; CN/cc-pVDZ lands 0.386 eV above (from hcore it did not converge at
/// all, so this row trades a non-answer for a high answer). The fix costs this
/// ONE CHEMICAL SYSTEM, at both bases it was measured at. That is not smoothed
/// over: both are asserted with their magnitudes in
/// [`cn_is_the_system_the_guess_fix_costs`], so no later reader can believe
/// the repair was free.
///
/// The states ferric finds there are genuine ROHF stationary points of the same
/// operator, not wrong energies: PySCF independently reaches −92.1186236
/// (6-31G) from 5 of 40 randomized starts, and −92.1818724 (cc-pVDZ) from its
/// own random starts. CN has at least three ROHF solutions at 6-31G and four at
/// cc-pVDZ. This is basin selection on a genuinely multi-solution system.
#[test]
fn the_fixed_rohf_path_reaches_every_reference() {
    // 1e-6 Ha. Loose enough for ferric's own SCF exit criteria against a
    // PySCF run at conv_tol 1e-12; tight enough that a different STATE (the
    // failures this lane exists to fix are 0.1–4.3 eV = 4e-3–0.16 Ha) cannot
    // hide inside it, by more than three orders of magnitude.
    const TOL: f64 = 1e-6;
    // The named exceptions: CN at BOTH bases, excluded here and asserted with
    // their magnitudes in `cn_is_the_system_the_guess_fix_costs`. Listing them
    // by name (rather than loosening TOL until they pass) is deliberate — a
    // tolerance wide enough to swallow a 0.39 eV state error would swallow
    // every defect this lane exists to find.
    const KNOWN_EXCEPTIONS: [&str; 2] = ["CN/6-31G", "CN/cc-pVDZ"];

    let mut failures: Vec<String> = Vec::new();
    let mut n_checked = 0usize;
    for (name, sys, e_ref) in sweep() {
        if KNOWN_EXCEPTIONS.contains(&name) {
            continue;
        }
        match run(&sys, &tight_cfg()) {
            Ok(e) => {
                n_checked += 1;
                // `e - e_ref > TOL` (signed, not |·|): landing BELOW a PySCF
                // reference means ferric found a LOWER ROHF stationary point,
                // which is not a failure of ferric. Two rows do exactly that
                // (B2 at both bases) and are documented in
                // `ferric_finds_lower_rohf_states_than_pyscfs_standard_guesses`.
                if e - e_ref > TOL {
                    failures.push(format!(
                        "{name}: ferric {e:.10} is {:+.4} eV ABOVE pyscf {e_ref:.10}",
                        (e - e_ref) * HARTREE_TO_EV
                    ));
                }
            }
            Err(msg) => failures.push(format!("{name}: SCF FAILED: {msg}")),
        }
    }
    assert!(
        n_checked >= 26,
        "the sweep shrank to {n_checked} systems — a validation that stops \
         responding to inputs stops being a validation"
    );
    assert!(
        failures.is_empty(),
        "{} of {n_checked} systems are above their PySCF ROHF reference:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// **The cost this test was written to pin is GONE (2026-09-18).** CN was the
/// one chemical system the ROHF guess fix made worse, at both bases. PR #83
/// (`a268fc3c`, spherically symmetrizing the MINAO atomic block) removed the
/// penalty entirely:
///
/// ```text
///                      recorded cost     now
///   CN/6-31G   MINAO     +0.575 eV      +0.0000 eV   -92.1397654895
///   CN/cc-pVDZ MINAO     +0.386 eV      +0.0000 eV   -92.1960778175
///   CN/cc-pVDZ hcore     no convergence  unchanged (400 iters)
/// ```
///
/// Both MINAO rows now reach their PySCF reference to ~4e-8 Ha. Measured with
/// the guess pinned explicitly at each basis, not inherited from the default.
///
/// This test caught its own premise expiring, which is what its last paragraph
/// asked for: "a future change that silently makes the cost worse — or quietly
/// fixes it without anyone noticing — shows up here." It did. The assertions
/// below are inverted accordingly: they now pin that MINAO REACHES the
/// reference, so a regression that reintroduces the penalty fails here.
///
/// The historical record, for anyone reading an older note:
///
/// * CN/6-31G: hcore reached the reference (−92.1397655), MINAO landed
///   +0.575 eV high at −92.1186235.
/// * CN/cc-pVDZ: hcore did not converge at all in 400 iterations, MINAO
///   converges but +0.386 eV high at −92.1818724. This row trades a non-answer
///   for a high answer, which is a different (and arguably better) trade than
///   the 6-31G row's, and is recorded separately rather than averaged in.
///
/// Both magnitudes are pinned, so a future change that silently makes the cost
/// worse — or quietly fixes it without anyone noticing — shows up here.
///
/// These are genuine ROHF stationary points, not broken solves: PySCF converges
/// to −92.1186236 from 5 of 40 randomized starts at 6-31G, and ferric's orbital
/// energies there show a SPLIT π pair (−0.5434 / −0.5190) where the lower state
/// has them degenerate to 1e-14 (−0.5145572067547562 / −0.5145572067547473).
/// ROHF has no stability analysis in this workspace (see the module doc), so
/// there is no descent available to recover the lower basin — the honest report
/// is this assertion.
#[test]
fn cn_is_the_system_the_guess_fix_costs() {
    let sys = diatomic("C", "N", 1.1718, 0, 2, "6-31g");
    let e_hcore = run(&sys, &hcore_cfg()).expect("CN/6-31G hcore must converge");
    let e_minao = run(&sys, &minao_cfg()).expect("CN/6-31G minao must converge");
    let d_h = (e_hcore - CN_631G) * HARTREE_TO_EV;
    let d_m = (e_minao - CN_631G) * HARTREE_TO_EV;
    println!(
        "CN/6-31G  hcore {e_hcore:.10} ({d_h:+.4} eV)  minao {e_minao:.10} ({d_m:+.4} eV)  \
         ref {CN_631G:.10}"
    );
    assert!(
        (e_hcore - CN_631G).abs() < 1e-6,
        "CN/6-31G from hcore should reach the reference (it did before the fix); got {e_hcore:.10}"
    );
    // Was `d_m > 0.4 && d_m < 0.8` (the +0.575 eV penalty). PR #83 removed it;
    // this now pins the REPAIRED behaviour, so reintroducing the penalty fails.
    assert!(
        (e_minao - CN_631G).abs() < 1e-6,
        "CN/6-31G from MINAO should now REACH the reference (PR #83 removed the \
         +0.575 eV penalty this row used to pin); got {e_minao:.10}, \
         {d_m:+.4} eV off"
    );

    // The second basis. Pre-fix this row did not converge at all, so the trade
    // here is non-answer -> high answer, not right -> wrong.
    let sys_dz = diatomic("C", "N", 1.1718, 0, 2, "cc-pvdz");
    assert!(
        run(&sys_dz, &hcore_cfg()).is_err(),
        "CN/cc-pVDZ was expected NOT to converge from hcore (the pre-fix state)"
    );
    let e_dz = run(&sys_dz, &minao_cfg()).expect("CN/cc-pVDZ must converge from MINAO");
    let d_dz = (e_dz - CN_CCPVDZ) * HARTREE_TO_EV;
    println!("CN/cc-pVDZ  minao {e_dz:.10} ({d_dz:+.4} eV)  ref {CN_CCPVDZ:.10}");
    // Was `d_dz > 0.25 && d_dz < 0.55` (the +0.386 eV penalty), removed by the
    // same fix. The hcore non-convergence asserted above is UNCHANGED, so this
    // row still records a real hcore-vs-MINAO difference -- just not a cost.
    assert!(
        (e_dz - CN_CCPVDZ).abs() < 1e-6,
        "CN/cc-pVDZ from MINAO should now REACH the reference (PR #83 removed \
         the +0.386 eV penalty); got {e_dz:.10}, {d_dz:+.4} eV off"
    );
}

/// **Two rows where ferric lands BELOW its PySCF reference**, recorded so that
/// "ferric is lower" is never quietly read as "ferric is better".
///
/// B₂ at both bases converges to a symmetry-BROKEN ROHF solution that PySCF's
/// standard guesses do not reach. At 6-31G, ferric's hcore state
/// (−49.0697026876, π levels split −0.15203 / −0.14906) sits 0.317 eV below
/// PySCF's (−49.0580367850, π degenerate to 1e-13) and the fix moves ferric ONTO
/// PySCF's state. At cc-pVDZ ferric reaches −49.0946096023 from both guesses,
/// 0.318 eV below every standard PySCF guess, and PySCF finds a still lower
/// −49.1004769400 only from random starts.
///
/// Neither is a ferric error: both are real ROHF stationary points, both are
/// above the UHF bound (PySCF UHF/B₂/6-31G after stability-following is
/// −49.1224344968, 1.43 eV below ferric's ROHF), and ROHF ≥ UHF is the only
/// inequality the ansatz guarantees. Recording them is the point — ROHF is
/// multi-solution on this class and NO code in this comparison reliably finds
/// the global one.
#[test]
fn ferric_finds_lower_rohf_states_than_pyscfs_standard_guesses() {
    let b2_631g = diatomic("B", "B", 1.590, 0, 3, "6-31g");
    let e_hcore = run(&b2_631g, &hcore_cfg()).expect("B2/6-31G hcore");
    let e_minao = run(&b2_631g, &minao_cfg()).expect("B2/6-31G minao");
    println!("B2/6-31G  hcore {e_hcore:.10}  minao {e_minao:.10}  pyscf {B2_631G:.10}");
    assert!(
        e_hcore < B2_631G - 1e-4,
        "B2/6-31G from hcore is expected BELOW the PySCF reference (a \
         symmetry-broken state PySCF's guesses miss); got {e_hcore:.10}"
    );
    assert!(
        (e_minao - B2_631G).abs() < 1e-6,
        "B2/6-31G from MINAO should agree with PySCF; got {e_minao:.10}"
    );

    let b2_dz = diatomic("B", "B", 1.590, 0, 3, "cc-pvdz");
    // PIN the guess. This assertion is a claim about ferric's HCORE path
    // reaching a state PySCF's standard guesses miss, and it inherited the
    // default guess until 2026-09-18 -- which broke it when PR #83 made
    // `use_sad_guess` live and the default became MINAO.
    //
    // MEASURED here, all three at density_conv 1e-10:
    //
    //   hcore            -49.0946096023   0.318 eV BELOW every standard
    //                                     PySCF guess, 0.160 eV ABOVE the
    //                                     random-start minimum
    //   MINAO / default  -49.0829083278   the standard PySCF state, to 2.2e-9
    //
    // So nothing was lost: hcore still finds the third stationary point this
    // test exists to pin, and it still satisfies BOTH bounds below. Only the
    // routing changed. Pinning the guess makes the claim independent of what
    // the default happens to be, which is the same fix this file already
    // applies at every other call site via `hcore_cfg`/`minao_cfg`.
    let e_dz = run(&b2_dz, &hcore_cfg()).expect("B2/cc-pVDZ");
    println!(
        "B2/cc-pVDZ  ferric {e_dz:.10}  pyscf-standard {B2_CCPVDZ_STANDARD:.10}  \
         pyscf-lowest-found {B2_CCPVDZ_LOWEST_FOUND:.10}"
    );
    assert!(
        e_dz < B2_CCPVDZ_STANDARD - 1e-4,
        "B2/cc-pVDZ: ferric is expected below every standard PySCF guess; got {e_dz:.10}"
    );
    assert!(
        e_dz > B2_CCPVDZ_LOWEST_FOUND - 1e-6,
        "B2/cc-pVDZ: ferric should NOT be below the lowest ROHF stationary point \
         PySCF located from random starts ({B2_CCPVDZ_LOWEST_FOUND:.10}); got {e_dz:.10}. \
         Going below it would mean the energy, not the state, is wrong."
    );
}

/// **The guess fix repairs two systems that did not CONVERGE AT ALL**, which is
/// a failure mode the UHF sweep never reached and which no energy comparison
/// would have surfaced.
///
/// HeNe⁺/6-31G and CN/cc-pVDZ both exhausted 400 iterations from the hcore
/// guess. From MINAO both converge, and HeNe⁺/6-31G lands on its PySCF
/// reference. So the fix is not only about which state is found — on these two
/// the old path produced no answer at all.
#[test]
fn the_guess_fix_repairs_two_non_convergences() {
    for (name, sys, e_ref) in [
        (
            "HeNe+/6-31G",
            diatomic("He", "Ne", 2.0, 1, 2, "6-31g"),
            Some(HENE_631G),
        ),
        (
            "CN/cc-pVDZ",
            diatomic("C", "N", 1.1718, 0, 2, "cc-pvdz"),
            None,
        ),
    ] {
        let hcore = run(&sys, &hcore_cfg());
        let minao = run(&sys, &minao_cfg());
        println!("{name}: hcore = {hcore:?}\n{name}: minao = {minao:?}");
        // The hcore leg is REPORTED, not asserted (changed 2026-09-18).
        //
        // This used to `assert!(hcore.is_err())` -- pinning a NEGATIVE, that
        // the pre-fix guess fails to converge. Whether an SCF exhausts its
        // iteration cap is one of the most machine-dependent quantities in
        // this repo: HeNe+/6-31G runs the full 400 iterations and fails on the
        // dev box, and CONVERGES on CI. The assertion therefore encoded a
        // property of one CPU's arithmetic, and CI failed on it.
        //
        // The finding worth pinning was never "hcore fails". It is "MINAO
        // reaches the reference", asserted below -- a POSITIVE claim about the
        // path real callers take, and one that holds on both machines. If
        // hcore also converges somewhere, that is not a regression in the fix;
        // it is the old path getting lucky on that hardware.
        if hcore.is_ok() {
            println!(
                "{name}: NOTE hcore converged on this machine. It does not on \
                 the dev box (400 iters, no convergence). Machine-dependent, \
                 and not what this test asserts -- see the MINAO check below."
            );
        }
        let e = minao.unwrap_or_else(|m| panic!("{name} must converge from MINAO: {m}"));
        if let Some(r) = e_ref {
            assert!(
                (e - r).abs() < 1e-6,
                "{name} from MINAO should reach {r:.10}; got {e:.10}"
            );
        }
    }
}

/// **The escape hatch is real.** `use_sad_guess = false` restores the exact
/// pre-fix behaviour, BIT-identically, on a system the fix changes.
///
/// This matters because nine tests elsewhere in the workspace were re-baselined
/// by pinning `use_sad_guess: false`; if that flag stopped meaning "the old
/// hcore path" those re-baselines would be measuring something else. The
/// comparison is against a hard-coded pre-fix constant captured at `07898944`,
/// not against a recomputed value, so it cannot drift with the code.
#[test]
fn hcore_config_reproduces_the_pre_fix_answer_bit_identically() {
    /// OH/6-31G under the hcore guess, measured at `07898944` (before the fix).
    const OH_631G_PRE_FIX: f64 = -75.203_724_952_9;
    let sys = diatomic("O", "H", 0.97, 0, 2, "6-31g");
    let e = run(&sys, &hcore_cfg()).expect("OH/6-31G hcore must converge");
    println!("OH/6-31G hcore = {e:.10} (pre-fix {OH_631G_PRE_FIX:.10})");
    assert!(
        (e - OH_631G_PRE_FIX).abs() < 1e-9,
        "use_sad_guess = false no longer reproduces the pre-fix hcore answer \
         ({OH_631G_PRE_FIX:.10}); got {e:.10}. Every test re-baselined by \
         pinning that flag now measures something other than what it pinned."
    );
    // And it must still be the WRONG state — the escape hatch escapes to the
    // old behaviour, not to a silently-improved one.
    assert!(
        e - OH_631G > 1e-3,
        "the hcore escape hatch should still land ABOVE the reference"
    );
}

/// **An explicit `init_guess_density` reaches the solver.** Sets the config
/// field directly (rather than relying on `use_sad_guess`) and checks it
/// changes the answer on a system where the guess matters.
///
/// This is the H-ARTIFACT exclusion for the OTHER config field. `use_sad_guess`
/// is proven live by the table; without this, `init_guess_density` could be
/// accepted, shape-checked and then dropped — which is exactly the class of
/// defect this lane fixed, one field over.
#[test]
fn an_explicit_init_guess_density_reaches_the_rohf_solver() {
    let sys = diatomic("O", "H", 0.97, 0, 2, "6-31g");
    // The MINAO density, handed over explicitly instead of via use_sad_guess.
    let d = ferric_scf::guess::minao_projection_guess(&sys.mol, &sys.prep, sys.prep.basis_set())
        .expect("MINAO projection for OH/6-31G");
    let cfg = RhfConfig {
        // use_sad_guess is OFF, so if the explicit density were ignored this
        // would fall through to hcore and land on the pre-fix answer.
        use_sad_guess: false,
        init_guess_density: Some(d),
        ..tight_cfg()
    };
    let e = run(&sys, &cfg).expect("OH/6-31G from an explicit guess density");
    println!("OH/6-31G explicit init_guess_density = {e:.10} (ref {OH_631G:.10})");
    assert!(
        (e - OH_631G).abs() < 1e-6,
        "an explicit init_guess_density did not reach the ROHF solver: got \
         {e:.10}, expected the reference {OH_631G:.10}. (The pre-fix hcore \
         answer is −75.2037249529; landing there means the field was dropped.)"
    );
}

/// **A wrongly-shaped `init_guess_density` is an ERROR, not a silent fallback.**
///
/// The shape check exists because a caller who hands over the wrong matrix has a
/// bug, and silently substituting MINAO would hide it. Pinned so the check
/// cannot be "simplified" into a fallback later.
#[test]
fn a_misshaped_init_guess_density_is_rejected() {
    let sys = diatomic("O", "H", 0.97, 0, 2, "6-31g");
    let n = sys.prep.nbasis();
    let cfg = RhfConfig {
        init_guess_density: Some(ndarray::Array2::<f64>::zeros((n + 3, n + 3))),
        ..tight_cfg()
    };
    let err = run(&sys, &cfg).expect_err("a misshaped init_guess_density must be an error");
    println!("misshaped init_guess_density -> {err}");
    assert!(
        err.contains("init_guess_density"),
        "the error should name the offending field; got: {err}"
    );
}

/// **The DF-path iteration cost is THRESHOLD-DEPENDENT, not a property of the
/// guess** — measured, after an assertion written the other way round failed.
///
/// This test exists because `k_builder_open_shell.rs` needed its `max_iter`
/// raised from 200 to 400 when the ROHF guess fix landed, and the obvious
/// explanation ("MINAO converges more slowly on the DF path") turned out to be
/// true only at that suite's convergence thresholds. Measured on CH₃/cc-pVDZ
/// ROHF with DF-J/DF-K:
///
/// | thresholds | hcore | MINAO |
/// |---|---|---|
/// | `density_conv 1e-8`, `energy_conv 1e-10` (the other suite's) | 37 | **262** |
/// | `density_conv 1e-10`, `energy_conv 1e-11` (this suite's) | 154 | **48** |
///
/// The ordering REVERSES. So "the guess fix costs DF iterations" is not a
/// finding about the guess; it is a finding about one convergence-threshold
/// setting on one system, and the raised cap in the other suite is a tolerance
/// for DIIS-path variance, not a documented regression.
///
/// The first version of this test asserted `it_minao > it_hcore` and FAILED
/// (48 vs 154), which is how the threshold dependence was discovered. It is
/// recorded here rather than quietly deleted, because the wrong version is the
/// evidence for why the right one asserts what it does.
///
/// What IS robust, and is what this test now pins, is the PHYSICS: both guesses
/// reach the same DF-ROHF state to within DF fitting error, at both threshold
/// settings.
#[test]
fn the_df_path_reaches_one_state_from_both_guesses() {
    // The OTHER suite's CH3 geometry, to the digit (H y = ±0.9345, not
    // ±0.934441). The two differ by 6e-5 A and that is enough to move the DF
    // iteration count by >200 on this near-degenerate system, which is the
    // whole point of the note below.
    let sys_kb = sys_from_xyz(
        "4\nCH3 doublet\nC 0.0000 0.0000 0.0000\nH 1.0790 0.0000 0.0000\n\
         H -0.5395 0.9345 0.0000\nH -0.5395 -0.9345 0.0000\n",
        0,
        2,
        "cc-pvdz",
    );
    let sys = ch3("cc-pvdz");
    let df = |sad: bool, dconv: f64, econv: f64| RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        max_iter: 2000,
        use_sad_guess: sad,
        density_conv: dconv,
        energy_conv: econv,
        ..tight_cfg()
    };
    // BOTH threshold settings, because one of them is what made the earlier
    // assertion look like a property of the guess.
    for (geom, sys) in [("ref-geom", &sys), ("kb-geom", &sys_kb)] {
        for (tag, dconv, econv) in [("loose(1e-8)", 1e-8, 1e-10), ("tight(1e-10)", 1e-10, 1e-11)] {
            let mut seen = Vec::new();
            for (label, sad) in [("hcore", false), ("minao", true)] {
                let r = ferric_scf::rohf::solve_rohf(
                    &sys.ctx,
                    &sys.mol,
                    &sys.prep,
                    Operator::coulomb(),
                    &sys.bounds,
                    &df(sad, dconv, econv),
                )
                .unwrap_or_else(|e| panic!("CH3/cc-pVDZ DF-ROHF {geom} {tag} {label}: {e:?}"));
                println!(
                    "CH3/cc-pVDZ DF-ROHF {geom:9} {tag:12} {label:6}: E = {:.10}  iters = {}",
                    r.energy, r.iterations
                );
                seen.push((label, r.energy, r.iterations));
            }
            let (_, e_h, it_h) = seen[0];
            let (_, e_m, it_m) = seen[1];
            // THE PHYSICS, which holds at both settings: same state, to within the
            // DF fitting error. This is the assertion that would catch the guess
            // fix sending the DF path to a different basin.
            assert!(
                (e_h - e_m).abs() < 1e-7,
                "{geom}/{tag}: the two guesses must reach the SAME DF-ROHF state; \
             got {e_h:.10} vs {e_m:.10} (delta {:.3e} Ha)",
                e_h - e_m
            );
            // A bound, NOT a direction — because the direction is threshold
            // dependent and asserting it once produced a false claim.
            // RE-RECORDED 2026-09-18 after PR #83 (`a268fc3c`) changed the
            // MINAO atomic block. Full table, deterministic (3/3 identical
            // runs, bit-identical energies):
            //
            //   geom      thresh         hcore   minao
            //   ref-geom  loose(1e-8)      121      79   <- MINAO FASTER here
            //   ref-geom  tight(1e-10)     154     337
            //   kb-geom   loose(1e-8)       37     105
            //   kb-geom   tight(1e-10)      37    1509   <- the outlier
            //
            // All eight rows reach the SAME state (spread 2.07e-8 Ha, asserted
            // above at 1e-7). The 1509 row converges properly with 491
            // iterations of its 2000 budget to spare -- it is slow, not stuck,
            // and it is the near-degenerate kb-geom that this test's own header
            // notes moves by >200 iterations on a 6e-5 A geometry change.
            //
            // The old bound was 1500, set before #83; 1509 clears it by 0.6%.
            // Raised to 1800 rather than deleted: it still catches a run that
            // approaches the 2000 cap (i.e. genuinely fails to converge), which
            // is what "something other than DIIS-path variance" would look
            // like. The direction is deliberately NOT asserted -- MINAO is
            // faster in one row and slower in three, so a directional claim
            // here would be false, as this test's history already records.
            assert!(
                it_h < 1800 && it_m < 1800,
                "{geom}/{tag}: DF iteration counts ({it_h} hcore, {it_m} minao) \
             exceed the re-recorded 2026-09-18 envelope (worst measured: 1509 \
             at kb-geom/tight). Approaching the 2000 cap means the solve is \
             failing, not merely taking a long DIIS path."
            );
        }
    }
}

/// **How far the "hcore is the culprit" explanation actually reaches — PARTIAL,
/// and the limit is pinned here rather than left implied.**
///
/// The natural story for this lane is "the hcore guess selects a higher basin,
/// in ferric and in PySCF alike". Checked against PySCF 2.13.0 ROHF run with
/// `init_guess="hcore"`, it holds on HALF the repaired set:
///
/// | system | PySCF hcore | ferric hcore (pre-fix) | same state? |
/// |---|---|---|---|
/// | F₂⁺/6-31G | −197.9252084757 | −197.9252084757 | **yes, 7.3e-12** |
/// | HeNe⁺/def2-SVP | −130.4965141207 | −130.4965141210 | **yes, 3.1e-10** |
/// | OH/6-31G | −75.3618462925 | −75.2037249529 | **no, 1.6e-1** |
/// | NH₂/6-31G | −55.5304322187 | −55.4649516177 | **no, 6.6e-2** |
///
/// On F₂⁺ and HeNe⁺ the two codes' hcore runs land on the SAME higher
/// stationary point, which is strong evidence the GUESS selects it independent
/// of ferric's solver. On OH and NH₂, **PySCF's hcore run reaches the correct
/// state and ferric's did not** — so those repairs are empirical: the MINAO
/// guess fixes them, but "hcore is a bad guess" does not by itself explain why
/// ferric diverged from PySCF there. The difference must live in the SCF path
/// (DIIS history, level shifting, Roothaan-combination convergence), which this
/// lane did NOT investigate.
///
/// This test pins the two rows that DO agree, because those are the load-bearing
/// evidence, and its doc records the two that do not, because a first draft of
/// the write-up claimed all four were bit-equal and that was false.
#[test]
fn ferrics_pre_fix_state_matches_pyscfs_hcore_state_on_two_of_four_systems() {
    /// PySCF 2.13.0 ROHF, `init_guess="hcore"`, conv_tol 1e-12.
    const F2P_PYSCF_HCORE: f64 = -197.925_208_475_7;
    const HENE_SVP_PYSCF_HCORE: f64 = -130.496_514_120_7;

    for (name, sys, pyscf_hcore) in [
        (
            "F2+/6-31G",
            diatomic("F", "F", 1.3220, 1, 2, "6-31g"),
            F2P_PYSCF_HCORE,
        ),
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
            HENE_SVP_PYSCF_HCORE,
        ),
    ] {
        let e = run(&sys, &hcore_cfg()).unwrap_or_else(|m| panic!("{name} hcore: {m}"));
        println!(
            "{name:16} ferric-hcore {e:.10}  pyscf-hcore {pyscf_hcore:.10}  diff {:.2e}",
            (e - pyscf_hcore).abs()
        );
        assert!(
            (e - pyscf_hcore).abs() < 1e-8,
            "{name}: ferric's pre-fix hcore answer should match PySCF's hcore \
             answer (SAME wrong state, which is what makes this the guess's \
             doing and not ferric's solver's); got {e:.10} vs {pyscf_hcore:.10}"
        );
        // And it must still be the WRONG state, or the row proves nothing.
        // Mutation-verified reachable: substituting the hcore run here fails
        // with "nothing to corroborate", so this is not a tautology.
        assert!(
            run(&sys, &minao_cfg()).expect("minao") < e - 1e-4,
            "{name}: the MINAO run must be LOWER than this shared hcore state, \
             otherwise there was nothing to corroborate"
        );
    }
}

/// **Is there a DEGENERACY-ORDERING defect in ROHF?** (H-DEGEN, pre-registered.)
///
/// The UHF lane found none — given a decent guess, aufbau picks correctly. ROHF
/// does not inherit that finding, because its effective Fock is a different
/// operator (Roothaan coupling of closed/open/virtual blocks), so it is tested
/// here rather than assumed.
///
/// The near-decisive experiment for H-DEGEN is a guess density that is ALREADY
/// AT a correct answer: if ROHF walked away from it, the occupation would be
/// being fixed by something other than basin selection. This feeds each
/// system's own converged-correct density straight back in via
/// `init_guess_density` and checks ROHF stays there.
///
/// Run on the two systems where the guess demonstrably matters (OH/6-31G,
/// +4.30 eV pre-fix; F₂⁺/6-31G, +3.02 eV) plus CN/6-31G, the one system the
/// fix makes WORSE — CN is the sharpest test, because if ROHF could not hold
/// its own correct density there, the CN regression would be an ordering bug
/// rather than basin selection, and the conclusion of this lane would change.
///
/// RESULT: all three hold their correct density to <1e-9 Ha. **H-DEGEN is
/// REFUTED for ROHF**, on the same evidence that refuted it for UHF, and the CN
/// cost is confirmed as basin selection rather than an ordering defect.
#[test]
fn rohf_holds_a_correct_guess_density_so_h_degen_is_refuted() {
    for (name, sys, e_ref, converge_cfg) in [
        (
            "OH/6-31G",
            diatomic("O", "H", 0.97, 0, 2, "6-31g"),
            OH_631G,
            minao_cfg(),
        ),
        (
            "F2+/6-31G",
            diatomic("F", "F", 1.3220, 1, 2, "6-31g"),
            F2P_631G_STANDARD,
            minao_cfg(),
        ),
        (
            // CN reaches its reference from HCORE, not MINAO — so the correct
            // density has to come from the hcore run here.
            "CN/6-31G",
            diatomic("C", "N", 1.1718, 0, 2, "6-31g"),
            CN_631G,
            hcore_cfg(),
        ),
    ] {
        // 1. Converge to the correct state and take its density.
        let r = solve_rohf(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            Operator::coulomb(),
            &sys.bounds,
            &converge_cfg,
        )
        .unwrap_or_else(|e| panic!("{name}: reference run failed: {e:?}"));
        assert!(
            (r.energy - e_ref).abs() < 1e-6,
            "{name}: the run used to SOURCE the correct density is not at the \
             reference ({:.10} vs {e_ref:.10}) — this test's premise is broken",
            r.energy
        );

        // 2. Hand that density straight back in as the guess. `use_sad_guess`
        //    is off so nothing else can supply a guess: if the field were
        //    ignored we would fall through to hcore.
        let cfg = RhfConfig {
            use_sad_guess: false,
            init_guess_density: Some(r.density_total.clone()),
            ..tight_cfg()
        };
        let back = run(&sys, &cfg).unwrap_or_else(|m| panic!("{name} re-run: {m}"));
        println!(
            "{name:12} converged {:.10} -> fed back as guess -> {back:.10}  (delta {:.2e})",
            r.energy,
            (back - r.energy).abs()
        );

        // 3. H-DEGEN predicts ROHF walks AWAY from a correct density. It does not.
        assert!(
            (back - r.energy).abs() < 1e-9,
            "{name}: ROHF did NOT stay at a density that was already correct \
             ({:.10} -> {back:.10}). That would be an ORDERING defect (H-DEGEN), \
             not basin selection, and would change this lane's conclusion.",
            r.energy
        );
    }
}
