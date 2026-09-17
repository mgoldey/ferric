//! Atomic ionization-potential anchor for the HeNe⁺ cDFT-ET lane.
//!
//! # Why this test exists
//!
//! Every assertion currently made about the HeNe⁺ charge-transfer diabats is
//! *internal*: self-consistency conditions, identities, and comparisons of
//! ferric against ferric. None of them is referenced to anything outside the
//! code, so none of them can catch a systematic error.
//!
//! The one quantity in that lane which is both (a) load-bearing and (b)
//! trivially checkable against an independent program is the **asymptote** of
//! the diabatic gap:
//!
//! ```text
//!   ΔE(R → ∞)  =  ΔIP  =  [E(He⁺) + E(Ne)] − [E(He) + E(Ne⁺)]
//! ```
//!
//! At infinite separation the two diabats are just non-interacting atoms, the
//! constraint does no work, and the gap is a pure difference of atomic
//! ionization potentials. That number needs **four isolated-atom UHF runs and
//! no cDFT at all**, and PySCF computes exactly the same thing.
//!
//! So this file pins ferric's four atomic energies — and their ΔIP combination
//! — against PySCF UHF/def2-SVP. If ferric and PySCF disagree on an isolated
//! two-electron atom, nothing downstream in the HeNe⁺ lane is interpretable,
//! and this test says so loudly before any cDFT number is trusted.
//!
//! # The stub problem, and how this file answers it
//!
//! A reference-comparison test has a characteristic blind spot: it cannot tell
//! "ferric reproduced PySCF" from "ferric echoed the constant that PySCF's
//! value was pasted into". An adversarial review of the first version of this
//! file replaced `atom_uhf`'s body with `return E_HE_PYSCF` (and the analogous
//! constants for the other three atoms) — deleting the SCF solver outright —
//! and **all five tests still passed**, in 0.26 s instead of 1.95 s. Runtime
//! was the only tell.
//!
//! Every test here is therefore built so that a constant-returning `atom_uhf`
//! *fails*. The mechanisms, in rough order of strength:
//!
//! 1. **Input perturbation with an externally-known response**
//!    (`atomic_energies_respond_to_the_basis_set`). The same four atoms are
//!    solved a second time in def2-TZVP. A stub keyed only on `(symbol,
//!    charge, multiplicity)` cannot see the basis argument and returns the
//!    def2-SVP number, so the variational ordering `E(TZVP) < E(SVP)` and the
//!    PySCF-referenced TZVP values both break. This is the strongest form and
//!    is the one that the defining mutation trips first.
//! 2. **Solver metadata a stub cannot fabricate**
//!    (`atomic_runs_report_real_solver_work`). Iteration counts, ERI quartet
//!    counts and orbital energies come back from the SCF loop; a function that
//!    returns a `f64` constant has none of them, and one that fabricates them
//!    has to fabricate a *consistent* set.
//! 3. **Internal identities on solver output**. He⁺ is a one-electron system,
//!    so its total energy must equal its occupied orbital energy exactly. That
//!    is a fact about the Fock matrix ferric built, not about any constant.
//! 4. **The ΔIP bar referenced to a stored PySCF ΔIP**, not to the same
//!    formula re-evaluated on both sides. The original wrote
//!    `dip_ref = (E_HE_CATION_PYSCF + E_NE_PYSCF) - (E_HE_PYSCF + E_NE_CATION_PYSCF)`
//!    — the identical expression applied to the reference constants — so a
//!    wrong *formula* (say, the sign flipped, or the wrong pair grouped) would
//!    have been applied to both sides and cancelled. `DIP_PYSCF` below is now
//!    the value PySCF itself printed, so the formula is under test too.
//!
//! ## Measured: which of those four actually hold the line
//!
//! The stub was re-run in the foreground against this file and progressively
//! hardened, because "the suite fails" is worth much less than "and here is
//! exactly how far a determined stub gets".
//!
//! | stub | result |
//! |---|---|
//! | constants + arbitrary `eps_alpha` | 3 of 7 fail |
//! | + ascending, occupied-negative `eps_alpha` | 3 of 7 fail |
//! | + `eps[0] == energy` for He⁺ only (self-consistent everywhere) | **2 of 7 fail** |
//!
//! So the honest floor is **two** tests, and both are mechanism (1):
//! `atomic_energies_respond_to_the_basis_set` and
//! `atomic_energy_tolerance_is_violable_by_a_real_input_change`. Mechanisms
//! (2) and (3) — the solver metadata and the He⁺ identity — DO catch a naive
//! stub, and they are worth keeping as a second line and as genuine solver
//! postconditions, but a sufficiently careful fabricator satisfies all of
//! them: `iterations`, `computed_quartets`, `nbasis` and an ascending negative
//! spectrum are all just numbers, and the one-electron identity is one more
//! number to line up.
//!
//! What a stub provably cannot do is make the energy RESPOND to an input it
//! does not read. That is why the basis perturbation is mechanism (1) and not
//! a nice-to-have: **if this file is ever reduced back to a single-basis
//! reference comparison, it returns to being unfalsifiable**, no matter how
//! much metadata it asserts.
//!
//! # Reference data
//!
//! PySCF 2.13.0, `scf.UHF`, bases `def2-svp` and `def2-tzvp`,
//! `conv_tol = 1e-12`, `conv_tol_grad = 1e-9`. Generated by
//! `scripts/gen_pyscf_atomic_ip_refs.py`. Multiplicities: He and Ne are
//! singlets (`spin = 0`), He⁺ and Ne⁺ are doublets (`spin = 1`).
//!
//! # Scope
//!
//! This is the *method-consistent* ΔIP, not the experimental one. The
//! experimental atomic difference is IP(He) − IP(Ne) = 24.587 − 21.565 =
//! 3.022 eV; UHF/def2-SVP lands near 3.63 eV. That ~0.6 eV offset is the
//! expected signature of a correlation-free method in a basis with no diffuse
//! functions, and it is the number a cDFT gap computed with the *same* method
//! must asymptote to — the experimental value is not the target here.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Hartree → electron-volt (CODATA 2018, as used elsewhere in ferric).
const HA2EV: f64 = 27.211_386_245_988;

/// PySCF 2.13.0 UHF/def2-SVP total energies for the isolated atoms (Hartree).
/// See the module docs for the exact PySCF settings.
const E_HE_PYSCF: f64 = -2.855_160_479_347;
const E_HE_CATION_PYSCF: f64 = -1.993_623_160_423;
const E_NE_PYSCF: f64 = -128.376_406_810_031;
const E_NE_CATION_PYSCF: f64 = -127.648_287_776_274;

/// PySCF's own printed ΔIP at def2-SVP, in Hartree. Stored as an INDEPENDENT
/// number rather than recomputed from the four constants above: if the ΔIP
/// formula in the test were wrong, recomputing it on both sides would apply
/// the same error twice and cancel. See `scripts/gen_pyscf_atomic_ip_refs.py`,
/// which prints this line directly.
const DIP_PYSCF: f64 = 0.133_418_285_166;

/// PySCF 2.13.0 UHF/**def2-TZVP** total energies for the same four atoms. The
/// second basis is what makes the anchor perturbable: see
/// `atomic_energies_respond_to_the_basis_set`.
const E_HE_PYSCF_TZ: f64 = -2.859_895_425_684;
const E_HE_CATION_PYSCF_TZ: f64 = -1.998_139_728_836;
const E_NE_PYSCF_TZ: f64 = -128.541_492_758_577;
const E_NE_CATION_PYSCF_TZ: f64 = -127.818_505_264_683;
/// PySCF's printed ΔIP at def2-TZVP (Hartree) = 3.776075 eV.
const DIP_PYSCF_TZ: f64 = 0.138_768_202_955;

/// Agreement bar against PySCF. Both programs converge the same variational
/// problem in the same basis, so the only residual is SCF convergence and
/// arithmetic ordering — this bar is met with room to spare (measured 3e-13 Ha,
/// seven orders of magnitude inside it).
const TOL_HA: f64 = 1e-6;

/// Everything the SCF returned for one isolated atom. The extra fields beyond
/// `energy` exist so that tests can assert ferric *ran a solver*, not merely
/// that a number matched: a stub returning a hardcoded `f64` has no iteration
/// count, no ERI quartet count and no orbital spectrum.
struct AtomRun {
    energy: f64,
    iterations: usize,
    computed_quartets: usize,
    /// Occupied α orbital energies, ascending.
    eps_alpha: Vec<f64>,
    /// Number of AO basis functions the run actually used.
    nbasis: usize,
    /// Number of α electrons (= occupied α orbitals).
    n_alpha: usize,
}

/// Run isolated-atom UHF in `basis_name`.
///
/// `charge` and `multiplicity` are passed straight through to `Molecule`, so
/// the caller is responsible for a physical pairing — He⁺ is a doublet, not a
/// singlet. A mis-paired (charge, multiplicity) still converges smoothly to
/// the *wrong* energy, which is precisely the failure mode the PySCF
/// cross-check exists to catch.
///
/// NOTE FOR MUTATION TESTING: this is the function an adversary stubs. Every
/// test in this file is designed to fail if its body is replaced by a constant
/// — see the module docs.
fn atom_run(symbol: &str, charge: i32, multiplicity: usize, basis_name: &str) -> AtomRun {
    let xyz = format!("1\n{symbol} atom\n{symbol} 0.0 0.0 0.0\n");
    let mol = Molecule::parse_xyz(&xyz, charge, multiplicity).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &config).unwrap();
    assert!(
        res.converged,
        "UHF/{basis_name} did not converge for {symbol} (charge {charge}, mult {multiplicity}): {:?}",
        res.exit
    );
    // n_alpha from the (charge, multiplicity) pairing: n_elec = Z - charge,
    // n_alpha - n_beta = mult - 1.
    let nelec = usize::try_from(mol.nelec()).expect("electron count must be non-negative");
    let n_alpha = (nelec + multiplicity - 1) / 2;
    AtomRun {
        energy: res.energy,
        iterations: res.iterations,
        computed_quartets: res.computed_quartets,
        eps_alpha: res.eps_alpha.clone(),
        nbasis: prep.nbasis(),
        n_alpha,
    }
}

/// Convenience wrapper for the many places that only want the energy.
fn atom_uhf(symbol: &str, charge: i32, multiplicity: usize) -> f64 {
    atom_run(symbol, charge, multiplicity, "def2-svp").energy
}

/// The four (symbol, charge, multiplicity) triples, in the order
/// (He, He⁺, Ne, Ne⁺).
const ATOMS: [(&str, i32, usize); 4] = [("He", 0, 1), ("He", 1, 2), ("Ne", 0, 1), ("Ne", 1, 2)];
const ATOM_LABELS: [&str; 4] = ["He ", "He+", "Ne ", "Ne+"];

/// The four atomic energies in `basis_name`, in the order (He, He⁺, Ne, Ne⁺).
fn atomic_energies_in(basis_name: &str) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (i, &(s, q, m)) in ATOMS.iter().enumerate() {
        out[i] = atom_run(s, q, m, basis_name).energy;
    }
    out
}

/// The four def2-SVP atomic energies, in the order (He, He⁺, Ne, Ne⁺).
fn atomic_energies() -> [f64; 4] {
    atomic_energies_in("def2-svp")
}

/// ΔIP = [E(He⁺) + E(Ne)] − [E(He) + E(Ne⁺)], the R → ∞ diabatic gap.
fn delta_ip([e_he, e_he_cat, e_ne, e_ne_cat]: [f64; 4]) -> f64 {
    (e_he_cat + e_ne) - (e_he + e_ne_cat)
}

/// **The anchor.** ferric's four isolated-atom UHF/def2-SVP energies match
/// PySCF to 1e-6 Ha, and the resulting ΔIP — the R → ∞ limit of the HeNe⁺
/// diabatic gap — is positive and matches PySCF's *separately stored* ΔIP to
/// the same bar.
///
/// ΔIP > 0 is the physically required sign: helium is harder to ionize than
/// neon (IP 24.587 vs 21.565 eV experimentally), so the state carrying the
/// hole on He lies *above* the state carrying it on Ne.
#[test]
fn hene_plus_atomic_ip_difference_anchor() {
    let energies = atomic_energies();
    let refs = [E_HE_PYSCF, E_HE_CATION_PYSCF, E_NE_PYSCF, E_NE_CATION_PYSCF];
    for i in 0..4 {
        let (label, ferric, pyscf) = (ATOM_LABELS[i], energies[i], refs[i]);
        let diff = ferric - pyscf;
        eprintln!("{label}  ferric = {ferric:.12}  pyscf = {pyscf:.12}  diff = {diff:+.3e} Ha");
        assert!(
            diff.abs() < TOL_HA,
            "{label} UHF/def2-SVP: ferric {ferric:.12} vs PySCF {pyscf:.12}, \
             diff {diff:+.3e} Ha exceeds {TOL_HA:.1e}. \
             A disagreement on an ISOLATED ATOM invalidates every downstream \
             HeNe+ diabat number — fix this before interpreting any cDFT result."
        );
    }

    // ΔIP against PySCF's OWN printed ΔIP (DIP_PYSCF), not against the same
    // formula re-applied to the four reference constants. Re-applying the
    // formula would make a wrong formula cancel against itself; this way the
    // combination is under test as well as the four energies.
    let dip = delta_ip(energies);
    eprintln!(
        "dIP  ferric = {dip:.12} Ha ({:.6} eV)   pyscf = {DIP_PYSCF:.12} Ha ({:.6} eV)   diff = {:+.3e} Ha",
        dip * HA2EV,
        DIP_PYSCF * HA2EV,
        dip - DIP_PYSCF
    );
    eprintln!(
        "     experimental IP(He)-IP(Ne) = 3.022 eV; method offset = {:+.3} eV",
        dip * HA2EV - 3.022
    );

    assert!(
        dip > 0.0,
        "dIP must be positive (He is harder to ionize than Ne), got {dip:.12} Ha"
    );
    assert!(
        (dip - DIP_PYSCF).abs() < TOL_HA,
        "dIP: ferric {dip:.12} vs PySCF {DIP_PYSCF:.12} Ha, diff {:+.3e} exceeds {TOL_HA:.1e}",
        dip - DIP_PYSCF
    );

    // Magnitude sanity: O(3-4 eV).
    //
    // HONESTY NOTE (found while auditing this file; NOT flagged by the review
    // that produced the other fixes). This bracket and the `dip > 0.0` check
    // above are both **dominated** by the PySCF comparison between them, and
    // are therefore UNREACHABLE as independent failures. `|dip - DIP_PYSCF| <
    // 1e-6` with DIP_PYSCF = 0.133418 already confines dip to
    // [0.133417, 0.133419] Ha = 3.63049..3.63050 eV, which is positive and
    // inside (3, 4) by construction. There is no value of `dip` that passes
    // the PySCF bar and fails either of these.
    //
    // They are kept deliberately, for two reasons, and a reader should not
    // mistake them for evidence:
    //   * they are ORDERING devices. A sign-flipped or grossly wrong dIP hits
    //     `dip > 0.0` first and reports "He must be harder to ionize than Ne"
    //     rather than a bare 12-digit numeric mismatch, which is the more
    //     useful failure for someone who has just changed the lane's physics.
    //   * they state the physical content of the anchor in the source, where
    //     a maintainer relaxing TOL_HA will see them. If TOL_HA is ever
    //     loosened past ~0.02 Ha they stop being dominated and start doing
    //     real work.
    // What they are NOT is independent corroboration of the anchor: the whole
    // of this test's external force comes from the DIP_PYSCF comparison and
    // from `atomic_energies_respond_to_the_basis_set`.
    let dip_ev = dip * HA2EV;
    assert!(
        (3.0..4.0).contains(&dip_ev),
        "dIP = {dip_ev:.4} eV is outside the expected 3-4 eV window for UHF/def2-SVP"
    );
}

/// **The defining anti-stub test.** The atomic energies must RESPOND to a
/// change in the input.
///
/// The same four (symbol, charge, multiplicity) triples are solved a second
/// time in def2-TZVP. Three independent things then have to hold, none of
/// which a hardcoded constant can produce:
///
/// 1. **Variational ordering**, with no reference data at all: def2-TZVP
///    strictly contains more variational freedom than def2-SVP for these
///    atoms, so `E(TZVP) < E(SVP)` for every one of them. A stub keyed on
///    `(symbol, charge, multiplicity)` returns the same number for both bases
///    and this comparison becomes `E < E`, which is false.
/// 2. **The size of the response is physically graded**: He (2 e, 5 → 6 AOs)
///    gains ~4.7 mHa, Ne (10 e, 14 → 31 AOs) gains ~165 mHa. Correlation-free
///    basis-set improvement scales with the number of electrons and the added
///    AO count, so the Ne lowering must exceed the He one by a wide margin.
/// 3. **The TZVP energies match PySCF's TZVP energies** to the same 1e-6 Ha
///    bar. This is a second, independent external anchor: it is possible in
///    principle to satisfy (1) and (2) with a second set of fabricated
///    constants, but not to do so while matching a program nobody told the
///    stub about.
///
/// Measured: ferric reproduces PySCF/def2-TZVP to the same ~1e-13 Ha as at
/// def2-SVP, and ΔIP moves 3.630496 → 3.776075 eV.
#[test]
fn atomic_energies_respond_to_the_basis_set() {
    let svp = atomic_energies_in("def2-svp");
    let tzvp = atomic_energies_in("def2-tzvp");
    let tz_refs = [
        E_HE_PYSCF_TZ,
        E_HE_CATION_PYSCF_TZ,
        E_NE_PYSCF_TZ,
        E_NE_CATION_PYSCF_TZ,
    ];

    let mut lowering = [0.0f64; 4];
    for i in 0..4 {
        lowering[i] = svp[i] - tzvp[i];
        eprintln!(
            "{}  E(SVP) = {:.12}  E(TZVP) = {:.12}  lowering = {:.9} Ha   \
             (pyscf TZVP {:.12}, diff {:+.3e})",
            ATOM_LABELS[i],
            svp[i],
            tzvp[i],
            lowering[i],
            tz_refs[i],
            tzvp[i] - tz_refs[i]
        );

        // (1) Variational ordering, reference-free. A constant-returning
        //     atom_run makes this `0.0 > 0.0`, which fails.
        assert!(
            lowering[i] > 0.0,
            "{}: E(def2-TZVP) = {:.12} is NOT below E(def2-SVP) = {:.12}. \
             def2-TZVP strictly enlarges the variational space for this atom, so the \
             energy must drop. If the lowering is exactly 0.0, the two runs returned \
             the SAME number and the solver is not seeing the basis argument at all \
             — i.e. these energies are not being computed.",
            ATOM_LABELS[i],
            tzvp[i],
            svp[i]
        );

        // (3) Second external anchor.
        assert!(
            (tzvp[i] - tz_refs[i]).abs() < TOL_HA,
            "{} UHF/def2-TZVP: ferric {:.12} vs PySCF {:.12}, diff {:+.3e} exceeds {TOL_HA:.1e}",
            ATOM_LABELS[i],
            tzvp[i],
            tz_refs[i],
            tzvp[i] - tz_refs[i]
        );
    }

    // (2) The response is graded by electron count / added AO count, not flat.
    //     Measured: He 4.7e-3 / He+ 4.5e-3 / Ne 1.65e-1 / Ne+ 1.70e-1 Ha.
    //     Bar set at 10x, comfortably inside the measured ~35x.
    let he_max = lowering[0].max(lowering[1]);
    let ne_min = lowering[2].min(lowering[3]);
    eprintln!("largest He-family lowering = {he_max:.6} Ha   smallest Ne-family = {ne_min:.6} Ha   ratio = {:.1}x", ne_min / he_max);
    assert!(
        ne_min > 10.0 * he_max,
        "the SVP -> TZVP lowering is not graded by system size: Ne family gains at most \
         {ne_min:.6} Ha while the He family gains up to {he_max:.6} Ha. A 10-electron atom \
         going 14 -> 31 AOs must gain far more than a 2-electron atom going 5 -> 6."
    );

    // ΔIP itself must move, and must land on PySCF's TZVP value.
    let dip_svp = delta_ip(svp);
    let dip_tz = delta_ip(tzvp);
    eprintln!(
        "dIP: SVP = {:.6} eV   TZVP = {:.6} eV   shift = {:+.6} eV   (pyscf TZVP {:.6} eV)",
        dip_svp * HA2EV,
        dip_tz * HA2EV,
        (dip_tz - dip_svp) * HA2EV,
        DIP_PYSCF_TZ * HA2EV
    );
    assert!(
        (dip_tz - DIP_PYSCF_TZ).abs() < TOL_HA,
        "dIP at def2-TZVP: ferric {dip_tz:.12} vs PySCF {DIP_PYSCF_TZ:.12} Ha, \
         diff {:+.3e} exceeds {TOL_HA:.1e}",
        dip_tz - DIP_PYSCF_TZ
    );
    // The anchor is basis-dependent, and that dependence is itself a fact
    // about the lane: a cDFT gap must asymptote to the ΔIP of ITS OWN basis.
    // Measured shift 0.1456 eV; bar at 0.05 eV so the two bases cannot be
    // confused for each other, and a stub returning one value for both fails.
    assert!(
        (dip_tz - dip_svp).abs() * HA2EV > 0.05,
        "dIP did not move between def2-SVP ({:.6} eV) and def2-TZVP ({:.6} eV). \
         These are different variational problems and must give different gaps.",
        dip_svp * HA2EV,
        dip_tz * HA2EV
    );
}

/// **Anti-stub test, solver-metadata form.** The atomic runs must report the
/// work an SCF actually does.
///
/// `energy` alone can be faked by a constant. An iteration count, an ERI
/// quartet count, an AO count and a full orbital spectrum cannot be faked
/// *consistently* by anything short of reimplementing the solver — and the
/// He⁺ identity below is a statement about the Fock matrix ferric built, with
/// no reference value involved at all.
///
/// Measured at def2-SVP: He 7 iters / 126 quartets / 5 AOs; He⁺ 2 / 36 / 5;
/// Ne 13 / 2975 / 14; Ne⁺ 11 / 2523 / 14.
#[test]
fn atomic_runs_report_real_solver_work() {
    for (i, &(sym, q, m)) in ATOMS.iter().enumerate() {
        let r = atom_run(sym, q, m, "def2-svp");
        eprintln!(
            "{}  iters = {:2}  quartets = {:5}  nbf = {:2}  n_alpha = {}  eps[0] = {:.9}",
            ATOM_LABELS[i], r.iterations, r.computed_quartets, r.nbasis, r.n_alpha, r.eps_alpha[0]
        );

        // An SCF that produced this energy took at least one iteration and
        // evaluated at least one ERI quartet.
        assert!(
            r.iterations >= 1,
            "{}: SCF reported {} iterations — no solver ran",
            ATOM_LABELS[i],
            r.iterations
        );
        assert!(
            r.computed_quartets > 0,
            "{}: SCF reported {} two-electron quartets — the Fock build never happened",
            ATOM_LABELS[i],
            r.computed_quartets
        );
        // def2-SVP: He/He+ = 5 AOs, Ne/Ne+ = 14 AOs. The spectrum has one
        // orbital energy per AO.
        let expect_nbf = if sym == "He" { 5 } else { 14 };
        assert_eq!(
            r.nbasis, expect_nbf,
            "{}: def2-SVP should give {expect_nbf} AOs, got {}",
            ATOM_LABELS[i], r.nbasis
        );
        assert_eq!(
            r.eps_alpha.len(),
            r.nbasis,
            "{}: orbital spectrum has {} entries for {} AOs",
            ATOM_LABELS[i],
            r.eps_alpha.len(),
            r.nbasis
        );
        // Occupied orbitals are bound; these are neutral or cationic atoms so
        // every occupied level sits well below zero.
        for (k, &e) in r.eps_alpha.iter().take(r.n_alpha).enumerate() {
            assert!(
                e < 0.0,
                "{}: occupied alpha orbital {k} has energy {e:.9} >= 0",
                ATOM_LABELS[i]
            );
        }
        // Ascending order — an eigensolver postcondition.
        for w in r.eps_alpha.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-12,
                "{}: orbital energies are not ascending: {:.9} then {:.9}",
                ATOM_LABELS[i],
                w[0],
                w[1]
            );
        }
    }

    // He+ is a ONE-ELECTRON system: the Fock operator is exactly the core
    // Hamiltonian, so E_total == eps_HOMO to machine precision. This is an
    // identity on ferric's own output with no external reference at all, and
    // it is exactly the kind of statement a returned constant cannot satisfy
    // (the stub has no orbital spectrum to be consistent with).
    let he_cat = atom_run("He", 1, 2, "def2-svp");
    assert_eq!(
        he_cat.n_alpha, 1,
        "He+ must have exactly one alpha electron"
    );
    let resid = he_cat.energy - he_cat.eps_alpha[0];
    eprintln!(
        "He+ one-electron identity: E = {:.12}, eps_HOMO = {:.12}, |E - eps| = {:.3e}",
        he_cat.energy,
        he_cat.eps_alpha[0],
        resid.abs()
    );
    assert!(
        resid.abs() < 1e-10,
        "He+ is a one-electron system so E_total must equal eps_HOMO exactly; \
         got E = {:.12}, eps = {:.12}, residual {resid:.3e}",
        he_cat.energy,
        he_cat.eps_alpha[0]
    );
}

/// Reachability check for the 1e-6 Ha bar, stated on a quantity **ferric
/// computes** rather than on the reference constant.
///
/// The original version of this test asserted
/// `|E(He) − (E_HE_PYSCF + 1e-5)| > 1e-6`, i.e. the arithmetic `1e-5 > 1e-6`
/// wearing a solver call: the measured output was exactly `1.000e-5`
/// regardless of what ferric returned, so it proved nothing about ferric.
///
/// Here the perturbation is applied to the *physical input* instead. He⁺ in
/// def2-SVP is a different variational problem from He⁺ in def2-TZVP, and the
/// two energies differ by ~4.5 mHa — three orders of magnitude outside
/// `TOL_HA`. So a real, physically meaningful change in what ferric solves
/// does break the bar, which is what "the tolerance is violable" has to mean.
#[test]
fn atomic_energy_tolerance_is_violable_by_a_real_input_change() {
    let e_svp = atom_uhf("He", 1, 2);
    let e_tzvp = atom_run("He", 1, 2, "def2-tzvp").energy;
    let diff = (e_svp - e_tzvp).abs();
    eprintln!(
        "violability probe: |E(He+/def2-SVP) - E(He+/def2-TZVP)| = {:.12} - {:.12} = {diff:.3e} Ha vs bar {TOL_HA:.1e}",
        e_svp, e_tzvp
    );
    assert!(
        diff > TOL_HA,
        "changing the basis from def2-SVP to def2-TZVP moved E(He+) by only {diff:.3e} Ha, \
         which does not break the {TOL_HA:.1e} bar. Either the bar is unfalsifiable by any \
         realistic error, or (far more likely) the two runs are returning the same number \
         and the basis argument is being ignored."
    );
    // And the direction is the variational one, not just "different".
    assert!(
        e_tzvp < e_svp,
        "E(He+/def2-TZVP) = {e_tzvp:.12} must lie below E(He+/def2-SVP) = {e_svp:.12}"
    );
}

/// The multiplicity assignment is load-bearing, and a wrong one converges
/// *smoothly to the wrong answer*. He⁺ as a doublet (one electron) must not
/// equal He⁺ forced through the neutral-He singlet path, and the neutral
/// energies must lie below the cations: ionization costs energy.
#[test]
fn atomic_charge_and_multiplicity_ordering() {
    let [e_he, e_he_cat, e_ne, e_ne_cat] = atomic_energies();
    eprintln!(
        "IP(He) = {:.6} eV   IP(Ne) = {:.6} eV",
        (e_he_cat - e_he) * HA2EV,
        (e_ne_cat - e_ne) * HA2EV
    );
    assert!(
        e_he_cat > e_he,
        "E(He+) = {e_he_cat:.9} must exceed E(He) = {e_he:.9}: ionization costs energy"
    );
    assert!(
        e_ne_cat > e_ne,
        "E(Ne+) = {e_ne_cat:.9} must exceed E(Ne) = {e_ne:.9}: ionization costs energy"
    );
    // The ordering that makes dIP positive, stated independently of dIP itself.
    assert!(
        e_he_cat - e_he > e_ne_cat - e_ne,
        "IP(He) = {:.9} must exceed IP(Ne) = {:.9} Ha",
        e_he_cat - e_he,
        e_ne_cat - e_ne
    );
    // Both IPs must be in the physical range for a closed-shell rare gas at
    // this level of theory (Koopmans-free, ΔSCF). UHF underestimates by a few
    // eV for want of correlation, but not by tens. Measured: 23.44 / 19.81 eV.
    for (label, ip) in [
        ("He", (e_he_cat - e_he) * HA2EV),
        ("Ne", (e_ne_cat - e_ne) * HA2EV),
    ] {
        assert!(
            (15.0..30.0).contains(&ip),
            "IP({label}) = {ip:.4} eV is outside the 15-30 eV window expected for a \
             rare-gas dSCF ionization potential"
        );
    }
}

// ---------------------------------------------------------------------------
// Promolecule reference populations
// ---------------------------------------------------------------------------

/// Becke-cell population of He in promolecule diabat B, measured at the six
/// decimal places it is quoted to.
///
/// Diabat B is "neutral He beside Ne⁺". Its promolecule density is built from
/// two fragments each converged in the *full dimer AO basis* (ghost partner, so
/// the basis is identical to the supermolecule's and the comparison is
/// counterpoise-clean): D_promol^B = D(He, ghost Ne) + D(Ne⁺, ghost He).
///
/// Integrating that density against the driver's own (99,302) Becke weight
/// operator for the He cell gives **N_He^B(2.0 Å) = 1.954484**, not 2.000.
/// Roughly 0.046 e of the neutral helium's density sits outside its own fuzzy
/// cell.
///
/// This matters because the lane's diabats were built with **integer** targets
/// (N_He = 2.000 for B, 1.000 for A). A target of 2.000 is therefore ~0.046 e
/// past the natural diabat, and the constraint has to do real work — and pay
/// real energy — to drag the density there. That mis-targeting, not a physical
/// charge-transfer effect, is the leading candidate for the large Lagrange
/// multiplier and the energy penalty that were previously attributed to physics.
///
/// # Where the tolerance comes from
///
/// The earlier version of this test asserted only the open interval
/// (1.90, 1.99) — 0.09 wide, about 1900× looser than the six digits the number
/// was quoted to, so any value from 1.901 to 1.989 passed identically. The bar
/// below is instead set from a measured grid-convergence study of this exact
/// quantity (fragments fixed, only the Becke quadrature varied):
///
/// ```text
///   (n_radial, n_angular)    N_He^B        N_He + N_Ne
///   ( 49, 110)               1.954569933   11.000071697
///   ( 75, 110)               1.954569582   11.000071393
///   ( 75, 302)               1.954484325   11.000001956
///   ( 99, 302)               1.954484329   11.000001961   <- the driver's grid
///   (125, 302)               1.954484330   11.000001961
///   (150, 302)               1.954484330   11.000001961
///   (200, 302)               1.954484330   11.000001961
/// ```
///
/// The radial axis is converged to 1e-9 from n_radial = 75 up. The angular
/// axis is the limiting one: 110 → 302 moves the population by 8.5e-5. So
/// 8.5e-5 is the honest residual quadrature uncertainty on this number, and
/// `TOL_POP = 2e-4` is set just above it — tight enough that the six quoted
/// digits are actually defended (the bar is ~450× tighter than the old
/// interval), loose enough that it is a quadrature bar and not a bit-identity
/// bar that any grid or BLAS reordering would trip.
#[test]
fn promolecule_diabat_b_population_matches_the_quoted_value() {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    const R: f64 = 2.0; // Å, the separation the lane's diabats were built at
    /// The measured value, to the precision it is quoted at in the lane notes.
    const N_HE_B_REF: f64 = 1.954_484;
    /// Residual (99,302)-vs-(75,110) quadrature spread is 8.5e-5; see doc above.
    const TOL_POP: f64 = 2e-4;

    let bs = basis::bundled("def2-svp").unwrap();
    let ctx = ParallelContext::default();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };

    // Atom order is fixed to (He, Ne) in every geometry so that the AO ordering
    // — and hence the density matrices — are directly summable.
    let dimer = format!("2\nHeNe\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {R}\n");
    let he_g = format!("2\nHe + ghost Ne\nHe 0.0 0.0 0.0\n@Ne 0.0 0.0 {R}\n");
    let ne_g = format!("2\nghost He + Ne\n@He 0.0 0.0 0.0\nNe 0.0 0.0 {R}\n");

    let densities = |xyz: &str, charge: i32, mult: usize| {
        let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &config).unwrap();
        assert!(
            res.converged,
            "fragment {xyz} (q={charge}) did not converge"
        );
        let da = res.density_alpha.clone();
        let db = res.density_beta.clone().unwrap_or_else(|| da.clone());
        (da, db)
    };

    // W^He on the driver's own grid, over the real dimer geometry.
    let mol_dimer = Molecule::parse_xyz(&dimer, 0, 1).unwrap();
    let grid = build_atomic_grid(
        &mol_dimer,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol_dimer, &bs, &pts).unwrap();
    let w_he = build_weight_matrix(&mol_dimer, &grid, &chi, &[0]);
    let w_ne = build_weight_matrix(&mol_dimer, &grid, &chi, &[1]);

    let (he0_a, he0_b) = densities(&he_g, 0, 1); // neutral He, ghost Ne
    let (nep_a, nep_b) = densities(&ne_g, 1, 2); // Ne cation, ghost He

    // Promolecule B = He + Ne⁺.
    let d_b_a = &he0_a + &nep_a;
    let d_b_b = &he0_b + &nep_b;

    let n_he_b = population(&w_he, &d_b_a, &d_b_b, &SpinChannel::Total);
    let n_ne_b = population(&w_ne, &d_b_a, &d_b_b, &SpinChannel::Total);
    eprintln!(
        "N_He^B({R} A) = {n_he_b:.6}   N_Ne^B = {n_ne_b:.6}   sum = {:.6}",
        n_he_b + n_ne_b
    );
    eprintln!(
        "deviation from the integer target 2.000 = {:+.6} e   |N_He^B - quoted {N_HE_B_REF:.6}| = {:.3e}",
        n_he_b - 2.0,
        (n_he_b - N_HE_B_REF).abs()
    );

    // Grid-completeness: the Becke cells must partition all 11 electrons of
    // promolecule B (He 2e + Ne⁺ 9e). This is a check on the quadrature, not an
    // independent physics result — but if it fails, N_He^B means nothing.
    //
    // NOTE ON MUTATION TESTING: this sum check is deliberately listed FIRST so
    // that a reader knows it is the assertion a naive "force N_He^B = 2.000"
    // mutation trips (2.000 + 9.045518 = 11.045518). A mutation that isolates
    // the physics claim below must be SUM-CONSISTENT — see the dedicated
    // sum-consistent mutation recorded in the test body's trailing comment.
    assert!(
        (n_he_b + n_ne_b - 11.0).abs() < 1e-4,
        "Becke cells do not partition 11 e: N_He + N_Ne = {:.6}",
        n_he_b + n_ne_b
    );

    // THE FINDING, at the precision it is quoted to. This replaces the old
    // open interval (1.90, 1.99), which was 0.09 wide and defended none of the
    // six digits.
    assert!(
        (n_he_b - N_HE_B_REF).abs() < TOL_POP,
        "N_He^B = {n_he_b:.9} does not match the quoted {N_HE_B_REF:.6} to {TOL_POP:.0e} \
         (diff {:+.3e}). That bar is the measured (110 -> 302) angular grid spread of 8.5e-5, \
         so a failure here is a real change in the promolecule density or the Becke \
         partition, not quadrature noise.",
        n_he_b - N_HE_B_REF
    );

    // Stated separately from the numeric match, because it is the physical
    // claim rather than the regression: the natural population is measurably
    // BELOW the integer target the lane actually constrained to. The margin is
    // 0.0455 e against a 2e-4 bar, i.e. 227 sigma.
    assert!(
        n_he_b < 2.0 - 50.0 * TOL_POP,
        "N_He^B = {n_he_b:.6} is not clearly below the integer target 2.000. \
         If this is ~2.00, Becke-cell spill cannot explain the constraint penalty and \
         the lane must return to the state-identity question."
    );

    // SUM-CONSISTENT MUTATION (run in the foreground, 2026-09-16): scaling the
    // He fragment density by 1.0233 so that N_He^B -> 1.99977 while the
    // complementary N_Ne^B is rescaled to keep the sum at 11.000002 passes the
    // grid-completeness check above and FAILS both assertions in this block.
    // That isolates the physics claim from the quadrature claim, which the
    // earlier "force N_He^B to 2.000" mutation did not (it died at the sum).
}

/// At R = 2.0 Å the unconstrained HeNe⁺ SCF sits at the natural diabat-B
/// population, and both sit far from the integer target 2.000.
///
/// This is the sharp form of the previous test's claim. If the unconstrained
/// state *is* diabat B, then constraining N_He to the integer 2.000 is not
/// preparing a diabat — it is pushing an already-correct state 0.046 e away
/// from where it wants to be, and every Hartree of constraint work that follows
/// is an artifact of the target, not a charge-transfer energy.
///
/// # SCOPE: this is a statement about R = 2.0 Å ONLY
///
/// The claim is **separation-specific and false at large R**. The
/// `hene_promolecule_probe` example sweeps R and finds (def2-SVP, (99,302)):
///
/// ```text
///   R/Å    N_He^promol_B   N_He^unconstrained
///   2.0    1.954484        1.953538        <- this test; unconstrained IS diabat B
///   2.5    1.973812        1.973294
///   3.0    1.985704        STALLED (unconstrained UHF hit max_iter = 300)
///   4.0    1.995681        0.999323        <- unconstrained is diabat A, NOT B
///   6.0    1.999481        0.999903        <- likewise
/// ```
///
/// So N_He^unconstrained is **non-monotone in R**: it tracks diabat B at
/// bonding separation, fails to converge at 3.0 Å, and has switched to diabat
/// A (hole on He) by 4.0 Å. The blanket statement "the unconstrained state
/// already IS diabat B" is therefore NOT true of the HeNe⁺ system generally —
/// only of the bonding region this lane's diabats were built in. Any use of
/// this finding to argue about the R → ∞ limit is out of scope, and the
/// asymptotic statement belongs to the ΔIP anchor above instead.
///
/// # Assertion structure
///
/// The earlier version compared the two natural estimates via
/// `(n_unconstrained - 2.0).abs() > 10.0 * gap`, with `gap` — the quantity
/// under test — on the RIGHT of a `>`. That rewards the two estimates being
/// identical: driving `gap` to zero makes the assertion pass trivially, so
/// setting `n_promol_b = n_unconstrained` survived as a mutation. The claim is
/// restated below as **two absolute comparisons against fixed bars**, so that
/// shrinking `gap` can never help anything pass.
#[test]
fn unconstrained_hene_cation_matches_natural_diabat_b_at_bonding_separation() {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

    const R: f64 = 2.0;
    /// "The two independent estimates of the natural diabat agree": measured
    /// spread is 9.46e-4 e, bar set at 3e-3 (about 3x headroom). This is an
    /// ABSOLUTE bar — `gap` is compared against a constant, never against
    /// another measured quantity that could shrink to make it pass.
    const TOL_AGREE: f64 = 3e-3;
    /// "Both are far from the integer target": measured distances are 0.0465
    /// and 0.0455 e. Bar at 0.02, comfortably above TOL_AGREE so that the two
    /// statements cannot be satisfied by the same degenerate configuration.
    const MIN_DIST_TO_INTEGER: f64 = 0.02;

    let bs = basis::bundled("def2-svp").unwrap();
    let ctx = ParallelContext::default();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };

    let dimer = format!("2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {R}\n");
    let he_g = format!("2\nHe + ghost Ne\nHe 0.0 0.0 0.0\n@Ne 0.0 0.0 {R}\n");
    let ne_g = format!("2\nghost He + Ne\n@He 0.0 0.0 0.0\nNe 0.0 0.0 {R}\n");

    let densities = |xyz: &str, charge: i32, mult: usize| {
        let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &config).unwrap();
        assert!(
            res.converged,
            "{xyz} (q={charge}) did not converge: {:?}",
            res.exit
        );
        let da = res.density_alpha.clone();
        let db = res.density_beta.clone().unwrap_or_else(|| da.clone());
        (da, db)
    };

    let mol_dimer = Molecule::parse_xyz(&dimer, 0, 1).unwrap();
    let grid = build_atomic_grid(
        &mol_dimer,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol_dimer, &bs, &pts).unwrap();
    let w_he = build_weight_matrix(&mol_dimer, &grid, &chi, &[0]);

    // Promolecule B.
    let (he0_a, he0_b) = densities(&he_g, 0, 1);
    let (nep_a, nep_b) = densities(&ne_g, 1, 2);
    let n_promol_b = population(
        &w_he,
        &(&he0_a + &nep_a),
        &(&he0_b + &nep_b),
        &SpinChannel::Total,
    );

    // The real, unconstrained HeNe⁺ cation.
    let (dim_a, dim_b) = densities(&dimer, 1, 2);
    let n_unconstrained = population(&w_he, &dim_a, &dim_b, &SpinChannel::Total);

    let gap = (n_unconstrained - n_promol_b).abs();
    let d_unc = (n_unconstrained - 2.0).abs();
    let d_promol = (n_promol_b - 2.0).abs();
    eprintln!(
        "N_He: unconstrained = {n_unconstrained:.6}   promolecule B = {n_promol_b:.6}   |diff| = {gap:.3e}"
    );
    eprintln!(
        "distance of each from the integer target 2.000: unconstrained {:+.6}, promolecule B {:+.6}",
        n_unconstrained - 2.0,
        n_promol_b - 2.0
    );

    // (i) The two independent estimates of the natural diabat agree. Bar is a
    //     CONSTANT, so a smaller gap is only ever neutral-to-good here and
    //     cannot rescue anything else.
    assert!(
        gap < TOL_AGREE,
        "unconstrained N_He = {n_unconstrained:.6} differs from the natural diabat-B \
         population {n_promol_b:.6} by {gap:.3e} e, above the {TOL_AGREE:.0e} bar — the claim \
         that the unconstrained state already is diabat B does NOT hold at R = {R} A"
    );

    // (ii) and (iii) BOTH estimates are far from the integer that was actually
    //     targeted. Each is compared against its own FIXED bar. Note that
    //     driving `gap` to zero (the mutation that survived the previous
    //     formulation) does nothing for these two: they are unchanged by how
    //     close the estimates are to each other, and a configuration with
    //     gap = 0 at N = 2.000 fails both.
    assert!(
        d_unc > MIN_DIST_TO_INTEGER,
        "the unconstrained N_He = {n_unconstrained:.6} sits only {d_unc:.6} e from the integer \
         target 2.000, below the {MIN_DIST_TO_INTEGER} e bar — the constraint would then be \
         doing almost no work and the mis-targeting explanation collapses"
    );
    assert!(
        d_promol > MIN_DIST_TO_INTEGER,
        "the promolecule N_He = {n_promol_b:.6} sits only {d_promol:.6} e from the integer \
         target 2.000, below the {MIN_DIST_TO_INTEGER} e bar"
    );

    // (iv) The separation of scales as a MEASURED statement, with `gap` on the
    //     SMALL side of the inequality rather than multiplying the right-hand
    //     side. Written as `gap < d_min / 10` (not `d_min > 10 * gap`) the two
    //     forms are algebraically identical, but this one reads as a bound ON
    //     `gap`: shrinking `gap` is what the claim asserts, and the thing that
    //     has to stay large — the distance to the integer target — is already
    //     pinned independently by (ii) and (iii) against a fixed bar. So a
    //     mutation that sets n_promol_b = n_unconstrained (gap -> 0), which
    //     SURVIVED the previous `> 10.0 * gap` formulation, still has to get
    //     past (ii) and (iii) on its own merits and no longer buys anything.
    let d_min = d_unc.min(d_promol);
    eprintln!(
        "scale separation: gap = {gap:.3e} e vs the smaller distance-to-integer {d_min:.6} e (ratio {:.1}x)",
        d_min / gap
    );
    assert!(
        gap < d_min / 10.0,
        "the unconstrained-vs-promolecule spread ({gap:.3e} e) is not at least 10x smaller \
         than the distance to the integer target ({d_min:.6} e); the two natural estimates \
         cannot be distinguished from the target they were supposed to differ from"
    );
}
