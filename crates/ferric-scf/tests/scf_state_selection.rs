//! **Part 1 — the discriminating experiment.** Is ferric's UNCONSTRAINED
//! open-shell state multi-valued across orbital guesses, and is HeNe⁺ special?
//!
//! Hypotheses were pre-registered in
//! `tests/HYPOTHESES-scf-unconstrained-state-selection.md` BEFORE any number in
//! this file was measured. Read that first.
//!
//! # Why this exists
//!
//! NWChem 7.2.2, ORCA 6.1.1 and PySCF 2.13.0 all put the HeNe⁺/def2-SVP UHF
//! ground state at R = 2.0 Å at E = −130.5053405386 with a σ hole. ferric
//! converges to the ²Π state at −130.50034664, 0.136 eV higher, and ORCA's
//! `STABPerform` independently calls ferric's state UNSTABLE
//! (λ_min = −0.00486901 Eh). ferric's own stability analysis agrees
//! (λ_min = −4.87e-3 Ha, matching PySCF's `gen_g_hop_uhf` to 3.8e-10). So the
//! energy ferric computes is right *for the state it finds*; it finds the
//! wrong state.
//!
//! # The hypotheses and the statistic that separates them
//!
//! * **H-GUESS** — σ and ²Π are separate SCF basins and the guess picks one.
//!   Observable: E is MULTI-VALUED across guesses; some guess reaches σ.
//! * **H-DEGEN** — a degeneracy-ordering defect fixes the hole at iteration 1
//!   and nothing re-examines it. Observable: E is SINGLE-VALUED at ²Π across
//!   ALL guesses, *including a guess built from the converged σ density* —
//!   which is near-decisive, since that guess is already AT the σ answer.
//! * **H-ARTIFACT** — the rows are not actually different inputs. Observable:
//!   BIT-identical energies (|ΔE| exactly 0.0), not merely close.
//!
//! H-GUESS and H-DEGEN predict opposite spreads; H-ARTIFACT is separated from
//! H-DEGEN by bit-identity versus near-identity.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ndarray::Array2;

// ===========================================================================
// External references. Each was produced by a DIFFERENT code than ferric, and
// the def2-SVP row by three independent ones that agree to ~5e-10 Ha.
// ===========================================================================

/// HeNe⁺/def2-SVP, R = 2.0 Å: the σ-hole UHF ground state.
/// NWChem −130.505340538816 / ORCA −130.505340538350 / PySCF −130.5053405386.
const HENE_SVP_SIGMA: f64 = -130.505_340_538_6;
/// ferric's pre-fix ²Π state at the same geometry/basis.
const HENE_SVP_PI: f64 = -130.500_346_64;
/// HeNe⁺/6-31G, R = 2.0 Å, PySCF 2.13.0 UHF stable solution.
/// A SECOND BASIS, so a constant-returning stub cannot satisfy both.
const HENE_631G_SIGMA: f64 = -130.604_326_612_7;
/// N₂⁺/6-31G, R = 1.1160 Å, PySCF 2.13.0 UHF *after* following its own
/// instability: σ hole. PySCF's own default guess lands on the ²Π state at
/// −108.2895400917, 0.79 eV higher — so this system is guess-dependent in
/// PySCF too, and is the sharpest available test of "is ferric special".
const N2P_631G_SIGMA: f64 = -108.318_684_322_8;
/// N₂⁺/6-31G ²Π state (PySCF's default-guess answer).
const N2P_631G_PI: f64 = -108.289_540_091_7;
/// O₂ triplet/6-31G, PySCF 2.13.0 UHF (stable at its default guess).
const O2_631G: f64 = -149.545_574_533_4;
/// NO doublet/6-31G, PySCF 2.13.0 UHF (stable at its default guess).
const NO_631G: f64 = -129.174_067_033_5;
/// CO⁺ doublet/6-31G, PySCF 2.13.0 UHF (stable at its default guess).
///
/// NOTE, recorded rather than smoothed over: ferric and PySCF agree on this
/// ENERGY to 2.3e-8 Ha but the two σ/π LABELS disagree. CO⁺'s 5σ and 1π levels
/// are near-degenerate at this geometry/basis, so the β LUMO is a mixed
/// σ/π orbital whose dominant component flips with an arbitrarily small
/// rotation. The label is therefore not informative for CO⁺ and the ENERGY is
/// the quantity compared. This is stated so no reader concludes ferric found a
/// different state on CO⁺ — it did not.
const COP_631G: f64 = -112.191_039_699_9;

// ===========================================================================
// Scaffolding
// ===========================================================================

pub struct Sys {
    pub mol: Molecule,
    pub bs: basis::BasisSet,
    pub prep: PreparedBasis,
    pub bounds: SchwarzBounds,
    pub ctx: ParallelContext,
}

/// Build a diatomic open-shell system. `r_ang` in Ångström along z.
pub fn diatomic(a: &str, b: &str, r_ang: f64, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let xyz = format!("2\n{a}{b}\n{a} 0.0 0.0 0.0\n{b} 0.0 0.0 {r_ang}\n");
    let mol = Molecule::parse_xyz(&xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// A tight, plain UHF config: no level shift, no XC, no stability machinery.
/// `check_stability` is set so every row reports its own verdict.
pub fn tight_cfg() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        density_conv: 1e-10,
        energy_conv: 1e-11,
        check_stability: true,
        ..Default::default()
    }
}

/// The PRE-FIX config: the bare hcore guess, which is what `solve_uhf` did
/// unconditionally until the guess fix. `use_sad_guess = false` with no
/// explicit density is the documented way to ask for it.
pub fn hcore_cfg() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        ..tight_cfg()
    }
}

/// Number of occupied α / β orbitals for a molecule.
pub fn nocc_ab(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec + two_s) / 2, (nelec - two_s) / 2)
}

/// Classify the HOLE of an open-shell doublet/triplet as σ or π.
///
/// # The hole is in the β VIRTUAL space, not the α occupied space
///
/// This distinction was measured, not assumed. For HeNe⁺ the α SOMO is
/// He-1s/2s dominated in BOTH states (PySCF: `SOMO 5 top AOs = He 1s 0.599,
/// He 2s 0.506`), so classifying the α SOMO calls the ²Π state "sigma" and the
/// label carries no information. The orbital the removed electron came OUT of
/// is the β LUMO, i.e. column `nocc_b` of `c_b`, and that one separates the
/// states cleanly — PySCF gives σ_w = 0.671 / π_w = 0.000 (top AOs Ne 2p_z,
/// Ne 3p_z) for the σ state and σ_w = 0.000 / π_w = 0.677 (Ne 2p_x, 2p_y) for
/// the ²Π state.
///
/// The molecule is along z, so π character lives in the p_x / p_y AOs and σ
/// character in s / p_z. For a pure-spherical l = 1 shell libint2 orders
/// (m = −1, 0, +1) = (y, z, x), so the MIDDLE component of every p shell is σ
/// and the outer two are π; a cartesian l = 1 shell is (x, y, z).
///
/// Returns (sigma_weight, pi_weight) for the β LUMO.
pub fn hole_sigma_pi(prep: &PreparedBasis, c_b: &Array2<f64>, nocc_b: usize) -> (f64, f64) {
    let offs = prep.shell_offsets();
    let shells = prep.located_shells();
    let mut is_pi = vec![false; prep.nbasis()];
    for (sh, ls) in shells.iter().enumerate() {
        if ls.l != 1 {
            continue;
        }
        let o = offs[sh];
        if ls.pure {
            // Pure l = 1 is ordered m = (−1, 0, +1) = (y, z, x): the middle
            // component is σ about z, the outer two are π.
            is_pi[o] = true;
            is_pi[o + 2] = true;
        } else {
            // Cartesian l = 1 is ordered (x, y, z): the first two are π.
            is_pi[o] = true;
            is_pi[o + 1] = true;
        }
    }
    let (mut sig, mut pi) = (0.0, 0.0);
    let col = nocc_b; // the β LUMO — the orbital the electron was removed from
    for mu in 0..c_b.nrows() {
        let w = c_b[(mu, col)] * c_b[(mu, col)];
        if is_pi[mu] {
            pi += w;
        } else {
            sig += w;
        }
    }
    (sig, pi)
}

/// σ/π label for the β-LUMO hole. See [`hole_sigma_pi`] for why it is the β
/// LUMO and not the α SOMO.
pub fn hole_label(prep: &PreparedBasis, c_b: &Array2<f64>, nocc_b: usize) -> &'static str {
    let (s, p) = hole_sigma_pi(prep, c_b, nocc_b);
    if s > p {
        "sigma"
    } else {
        "pi"
    }
}

/// Canonical orthogonalizer `X = U s^{-1/2}` from the overlap matrix.
///
/// `ferric_scf::rhf::canonical_orthogonalizer` is private, so this is rebuilt
/// here (the same way `cdft_state_selection.rs` does). It is NOT
/// lindep-filtered, which is safe ONLY because every basis used in this file
/// has its smallest overlap eigenvalue far above the library's 1e-6 threshold
/// — asserted rather than assumed, so the helper cannot be silently reused on
/// an ill-conditioned basis.
fn orthogonalizer(s: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let (vals, vecs) = s.eigh(UPLO::Lower).unwrap();
    assert!(
        vals[0] > 1e-5,
        "unfiltered orthogonalizer used on a near-linearly-dependent basis \
         (s_min = {:.3e}); this helper is only valid where lindep filtering is a no-op",
        vals[0]
    );
    let mut x = vecs.clone();
    for (j, &v) in vals.iter().enumerate() {
        let inv = 1.0 / v.sqrt();
        for i in 0..x.nrows() {
            x[(i, j)] *= inv;
        }
    }
    x
}

/// Convert a guess DENSITY into guess MOs the UHF path accepts, by
/// diagonalizing the Fock built AT that density. This is what "starting from a
/// density" means for an MO-driven SCF, and it is the same operation `rhf.rs`
/// performs implicitly on its first iteration.
pub fn mos_from_density(
    sys: &Sys,
    cfg: &RhfConfig,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    use ferric_integrals::oneelectron;
    use ndarray_linalg::{Eigh, UPLO};

    let s = oneelectron::overlap(&sys.prep);
    let t = oneelectron::kinetic(&sys.prep);
    let v = oneelectron::nuclear(&sys.prep);
    let h = &t + &v;
    let x = orthogonalizer(&s);
    let n = sys.prep.nbasis();

    // One J/K build per argument density. `build_jk` writes BOTH J[d] and K[d]
    // from the single density it is handed, so the α/β Fock needs three calls:
    // J from the total density, K from each spin's own.
    let jk = |d: &Array2<f64>| -> (Array2<f64>, Array2<f64>) {
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        ferric_scf::rhf::build_jk(
            &sys.ctx,
            &sys.prep,
            &sys.bounds,
            cfg.integral_thresh,
            d,
            &mut j,
            &mut k,
        )
        .unwrap();
        (j, k)
    };
    let (j_tot, _) = jk(&(d_a + d_b));
    let (_, k_a) = jk(d_a);
    let (_, k_b) = jk(d_b);

    // The UHF Fock convention, verbatim from `uhf.rs`'s assembly:
    //   F_σ = h + J[D_α + D_β] − K[D_σ]      (no ½ on J).
    let f_a = &h + &j_tot - &k_a;
    let f_b = &h + &j_tot - &k_b;

    let diag = |f: &Array2<f64>| -> Array2<f64> {
        let fp = x.t().dot(f).dot(&x);
        let (_, cp) = fp.eigh(UPLO::Upper).unwrap();
        let c = x.dot(&cp); // (n, m), m ≤ n
                            // Pad to (n, n): the UHF guess contract requires a square MO matrix,
                            // and a linear-dependence-filtered X has m < n columns.
        let mut out = Array2::<f64>::zeros((n, n));
        for (jc, col) in c.axis_iter(ndarray::Axis(1)).enumerate() {
            out.column_mut(jc).assign(&col);
        }
        out
    };
    (diag(&f_a), diag(&f_b))
}

/// One measured row of the Part-1 table.
pub struct Row {
    pub guess: &'static str,
    pub energy: f64,
    pub hole: &'static str,
    pub iters: usize,
    pub converged: bool,
    pub lambda_min: Option<f64>,
    pub verdict: Option<StabilityVerdict>,
}

impl Row {
    pub fn print(&self, system: &str) {
        println!(
            "{system:16} {:14} E = {:.10}  hole = {:>5}  iters = {:3}  conv = {:5}  \
             lambda_min = {:>13}  {}",
            self.guess,
            self.energy,
            self.hole,
            self.iters,
            self.converged,
            self.lambda_min
                .map(|l| format!("{l:+.4e}"))
                .unwrap_or_else(|| "n/a".into()),
            self.verdict.map(|v| v.label()).unwrap_or("not checked"),
        );
    }
}

/// Run one system from every guess in the six-guess table and return the rows.
///
/// The guesses, in order:
///   1. `default`  — whatever `solve_uhf` does today (hcore, as shipped)
///   2. `hcore`    — explicit hcore density, injected as MOs
///   3. `minao`    — `guess::minao_projection_guess` (what RHF's default is)
///   4. `sad`      — `guess::sad_guess` (per-element free-atom SCF)
///   5. `self`     — the DEFAULT run's own converged MOs (the exactness anchor)
///   6. `sigma`    — MOs from the default run rotated to swap the σ/π hole
///
/// Guess 5 is the anchor; guess 6 is the directed probe that asks whether the σ
/// state is reachable at all.
pub fn six_guess_table(sys: &Sys, cfg: &RhfConfig) -> Vec<Row> {
    use ferric_integrals::oneelectron;
    let (na, nb) = nocc_ab(&sys.mol);
    let n = sys.prep.nbasis();
    let mut rows = Vec::new();

    let measure = |guess: &'static str, r: &ferric_scf::result::ScfResult| Row {
        guess,
        energy: r.energy,
        hole: r
            .mos_beta
            .as_ref()
            .map(|cb| hole_label(&sys.prep, cb, nb))
            .unwrap_or("n/a"),
        iters: r.iterations,
        converged: r.converged,
        lambda_min: r.stability.as_ref().map(|s| s.lowest_eigenvalue),
        verdict: r.stability.as_ref().map(|s| s.verdict()),
    };

    // 1. default
    let def = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, cfg).unwrap();
    rows.push(measure("default", &def));

    // 2. explicit hcore
    let s = oneelectron::overlap(&sys.prep);
    let t = oneelectron::kinetic(&sys.prep);
    let v = oneelectron::nuclear(&sys.prep);
    let h = &t + &v;
    let d_h = ferric_scf::guess::hcore_guess(&s, &h, na.max(1)).unwrap();
    let (ca, cb) = mos_from_density(sys, cfg, &(0.5 * &d_h), &(0.5 * &d_h));
    match solve_uhf_with_guess(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        cfg,
        Some((&ca, &cb)),
    ) {
        Ok(r) => rows.push(measure("hcore", &r)),
        Err(e) => println!("hcore guess failed: {e:?}"),
    }

    // 3. MINAO projection (RHF's shipped default)
    if let Ok(d) = ferric_scf::guess::minao_projection_guess(&sys.mol, &sys.prep, &sys.bs) {
        let (ca, cb) = mos_from_density(sys, cfg, &(0.5 * &d), &(0.5 * &d));
        match solve_uhf_with_guess(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bounds,
            cfg,
            Some((&ca, &cb)),
        ) {
            Ok(r) => rows.push(measure("minao", &r)),
            Err(e) => println!("minao guess failed: {e:?}"),
        }
    }

    // 4. SAD
    if let Ok(d) = ferric_scf::guess::sad_guess(&sys.mol, &sys.prep, &sys.bs) {
        let (ca, cb) = mos_from_density(sys, cfg, &(0.5 * &d), &(0.5 * &d));
        match solve_uhf_with_guess(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bounds,
            cfg,
            Some((&ca, &cb)),
        ) {
            Ok(r) => rows.push(measure("sad", &r)),
            Err(e) => println!("sad guess failed: {e:?}"),
        }
    }

    // 5. self (the exactness anchor): feed the default run its own answer
    let cb0 = def.mos_beta.clone().unwrap();
    match solve_uhf_with_guess(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        cfg,
        Some((&def.mos_alpha, &cb0)),
    ) {
        Ok(r) => rows.push(measure("self", &r)),
        Err(e) => println!("self guess failed: {e:?}"),
    }

    // 6. hole-swapped: move the α hole from its current orbital to the next
    //    virtual, i.e. ask directly whether the other state is reachable.
    if na < n {
        let mut ca_sw = def.mos_alpha.clone();
        for mu in 0..n {
            let tmp = ca_sw[(mu, na - 1)];
            ca_sw[(mu, na - 1)] = ca_sw[(mu, na)];
            ca_sw[(mu, na)] = tmp;
        }
        match solve_uhf_with_guess(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bounds,
            cfg,
            Some((&ca_sw, &cb0)),
        ) {
            Ok(r) => rows.push(measure("swap", &r)),
            Err(e) => println!("swap guess failed: {e:?}"),
        }
    }

    rows
}

// ===========================================================================
// PART 1 — the measurements
// ===========================================================================

/// **The σ/π classifier must actually discriminate.** A label that returns the
/// same string for both states is decoration, and every "hole =" column in this
/// file would be meaningless.
///
/// Pinned against the TWO states ferric itself reaches on HeNe⁺/def2-SVP: the
/// hcore guess (²Π, E = −130.5003466) and the MINAO guess (σ, E = −130.5053405).
/// PySCF independently assigns the same characters to the same two energies
/// (σ: Ne 2p_z, π: Ne 2p_x/2p_y — see `hole_sigma_pi`'s doc).
///
/// This test is the reason the classifier was CHANGED: the first version summed
/// over the α SOMO and called BOTH states "sigma", which this assertion catches.
#[test]
fn the_hole_classifier_separates_the_two_hene_states() {
    let sys = diatomic("He", "Ne", 2.0, 1, 2, "def2-svp");
    let cfg = tight_cfg();
    let (_, nb) = nocc_ab(&sys.mol);

    // The ²Π state is what the BARE HCORE guess reaches. Before the fix that
    // was also what `solve_uhf` did by default; it no longer is, so this test
    // asks for hcore explicitly rather than relying on the default.
    let pi_state = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &hcore_cfg()).unwrap();
    let d = ferric_scf::guess::minao_projection_guess(&sys.mol, &sys.prep, &sys.bs).unwrap();
    let (ca, cb) = mos_from_density(&sys, &cfg, &(0.5 * &d), &(0.5 * &d));
    let sigma_state = solve_uhf_with_guess(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &cfg,
        Some((&ca, &cb)),
    )
    .unwrap();

    let pi_cb = pi_state.mos_beta.as_ref().unwrap();
    let sig_cb = sigma_state.mos_beta.as_ref().unwrap();
    let (ps, pp) = hole_sigma_pi(&sys.prep, pi_cb, nb);
    let (ss, sp) = hole_sigma_pi(&sys.prep, sig_cb, nb);
    println!(
        "hcore state E = {:.10}: sigma_w = {ps:.4}, pi_w = {pp:.4} -> {}",
        pi_state.energy,
        hole_label(&sys.prep, pi_cb, nb)
    );
    println!(
        "minao state E = {:.10}: sigma_w = {ss:.4}, pi_w = {sp:.4} -> {}",
        sigma_state.energy,
        hole_label(&sys.prep, sig_cb, nb)
    );

    assert!(
        (pi_state.energy - HENE_SVP_PI).abs() < 1e-6,
        "the hcore run is no longer the pi state (E = {:.10}); this test's premise is gone",
        pi_state.energy
    );
    assert!(
        (sigma_state.energy - HENE_SVP_SIGMA).abs() < 1e-8,
        "the minao run is no longer the sigma state (E = {:.10}); this test's premise is gone",
        sigma_state.energy
    );
    assert_eq!(
        hole_label(&sys.prep, pi_cb, nb),
        "pi",
        "the classifier calls the 0.136 eV-HIGHER state (E = {:.10}) sigma; it does not \
         discriminate and every 'hole =' column in this file is decoration",
        pi_state.energy
    );
    assert_eq!(
        hole_label(&sys.prep, sig_cb, nb),
        "sigma",
        "the classifier calls the external-reference state (E = {:.10}) pi",
        sigma_state.energy
    );
}

/// **THE EXACTNESS ANCHOR.** Feeding a converged solution back as its own guess
/// must return that same solution. If injecting the answer does not return the
/// answer, the guess-injection harness is broken and no other row in this file
/// means anything.
///
/// This anchor's KNOWN BLIND SPOT (pre-registered): it does ~zero SCF work, so
/// it cannot see a defect in how a FAR guess is processed. That is why every
/// row below reports its iteration count — a far guess that "converges" in one
/// iteration is visibly suspect.
#[test]
fn injecting_a_converged_solution_returns_it() {
    let sys = diatomic("He", "Ne", 2.0, 1, 2, "def2-svp");
    let cfg = tight_cfg();
    let def = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    let cb = def.mos_beta.clone().unwrap();
    let again = solve_uhf_with_guess(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &cfg,
        Some((&def.mos_alpha, &cb)),
    )
    .unwrap();
    let d = (again.energy - def.energy).abs();
    println!(
        "self-guess anchor: E = {:.12} vs {:.12}, |dE| = {d:.3e}",
        again.energy, def.energy
    );
    assert!(
        d <= 1e-9,
        "injecting a converged solution as its own guess changed the energy by {d:.3e} Ha \
         (> 1e-9); the guess-injection harness is broken and every other row here is void"
    );
}

/// **The six-guess table on HeNe⁺/def2-SVP** — the known-defective case.
///
/// Prints the table and asserts only the H-ARTIFACT stop condition: the rows
/// must not be bit-identical. The physics verdict is drawn in the report, not
/// asserted here, because this test's job is to MEASURE the pre-fix landscape.
#[test]
fn hene_svp_six_guess_table() {
    let sys = diatomic("He", "Ne", 2.0, 1, 2, "def2-svp");
    let cfg = tight_cfg();
    let rows = six_guess_table(&sys, &cfg);
    println!("\n=== HeNe+/def2-SVP, R = 2.0 A ===");
    println!(
        "external refs: sigma = {HENE_SVP_SIGMA:.10} (NWChem/ORCA/PySCF), \
         ferric pre-fix pi = {HENE_SVP_PI:.8}"
    );
    for r in &rows {
        r.print("HeNe+/def2-SVP");
    }
    let es: Vec<f64> = rows.iter().map(|r| r.energy).collect();
    let spread = es.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - es.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("spread across guesses = {spread:.6e} Ha");

    // H-ARTIFACT stop condition, pre-registered: bit-identity across genuinely
    // different guesses means the inputs were swallowed and the table measures
    // nothing. Near-identity (~1e-10) is a legitimate H-DEGEN observable; EXACT
    // identity is not.
    let all_bit_identical = es.windows(2).all(|w| w[0].to_bits() == w[1].to_bits());
    assert!(
        !all_bit_identical,
        "every guess returned a BIT-IDENTICAL energy ({:.12}). That is the pre-registered \
         H-ARTIFACT signature -- the guesses are not reaching the solver -- not a physics \
         result. STOP and audit the harness.",
        es[0]
    );
}

/// **The load-bearing question: is this HeNe⁺-specific or systemic?**
///
/// Runs the same default-guess solve on five open-shell systems with
/// near-degenerate frontier orbitals and compares each against PySCF 2.13.0 in
/// the SAME basis and geometry. Pre-committed split: wrong only on HeNe⁺ ⇒
/// narrow bug; wrong on ≥ 2 ⇒ systemic guess defect.
///
/// This test PRINTS and does not assert a pass/fail on the physics, because it
/// is the measurement that decides which fix is correct. Its assertion is only
/// that every system actually converged — an unconverged row would make the
/// comparison meaningless.
#[test]
fn systemic_sweep_default_guess_vs_pyscf() {
    let cases: Vec<(&str, Sys, f64, &str)> = vec![
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
            HENE_SVP_SIGMA,
            "sigma",
        ),
        (
            "HeNe+/6-31G",
            diatomic("He", "Ne", 2.0, 1, 2, "6-31g"),
            HENE_631G_SIGMA,
            "sigma",
        ),
        (
            "O2/6-31G",
            diatomic("O", "O", 1.2075, 0, 3, "6-31g"),
            O2_631G,
            "pi",
        ),
        (
            "NO/6-31G",
            diatomic("N", "O", 1.1508, 0, 2, "6-31g"),
            NO_631G,
            "pi",
        ),
        (
            "N2+/6-31G",
            diatomic("N", "N", 1.1160, 1, 2, "6-31g"),
            N2P_631G_SIGMA,
            "sigma",
        ),
        (
            "CO+/6-31G",
            diatomic("C", "O", 1.1150, 1, 2, "6-31g"),
            COP_631G,
            "pi",
        ),
    ];
    let cfg = tight_cfg();
    println!("\n=== SYSTEMIC SWEEP: ferric default guess vs PySCF 2.13.0 stable UHF ===");
    let mut n_wrong = 0;
    for (name, sys, e_ref, hole_ref) in &cases {
        let (_na, nb) = nocc_ab(&sys.mol);
        let r = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
        assert!(
            r.converged,
            "{name}: ferric did not converge; the comparison is meaningless"
        );
        let hole = r
            .mos_beta
            .as_ref()
            .map(|cb| hole_label(&sys.prep, cb, nb))
            .unwrap_or("n/a");
        let d_ev = (r.energy - e_ref) * 27.211_386_245_988;
        let verdict = r
            .stability
            .as_ref()
            .map(|s| s.verdict().label())
            .unwrap_or("not checked");
        let lmin = r
            .stability
            .as_ref()
            .map(|s| format!("{:+.4e}", s.lowest_eigenvalue))
            .unwrap_or_else(|| "n/a".into());
        let flag = if d_ev > 1e-4 { "  <== ABOVE PySCF" } else { "" };
        if d_ev > 1e-4 {
            n_wrong += 1;
        }
        println!(
            "{name:16} ferric = {:.10} ({hole:>5})  pyscf = {e_ref:.10} ({hole_ref:>5})  \
             dE = {d_ev:+.4} eV  lambda_min = {lmin}  {verdict}{flag}",
            r.energy
        );
    }
    println!(
        "\nsystems above the PySCF reference: {n_wrong} of {}",
        cases.len()
    );
}

/// **N₂⁺ is guess-dependent in PySCF TOO.** PySCF's own default guess lands on
/// the ²Π state (−108.2895400917) and its `stability()` follows an instability
/// down to the σ state (−108.3186843228), 0.79 eV lower. So a code landing on
/// the higher state from one guess is not, by itself, a ferric-specific defect
/// — it is a property of this class of problem.
///
/// This test exists so the report cannot claim "only ferric does this". It
/// asserts nothing about ferric; it records the reference pair so both numbers
/// are in the repo.
#[test]
fn n2_cation_is_multi_valued_in_pyscf_as_well() {
    let gap = (N2P_631G_PI - N2P_631G_SIGMA) * 27.211_386_245_988;
    println!(
        "N2+/6-31G PySCF: default-guess pi = {N2P_631G_PI:.10}, \
         stability-followed sigma = {N2P_631G_SIGMA:.10}, gap = {gap:.4} eV"
    );
    assert!(
        gap > 0.5,
        "the recorded N2+ pi/sigma gap collapsed to {gap:.4} eV; the reference pair is wrong"
    );
}

/// **Does a better GUESS alone fix all three failing systems?** This decides
/// whether the fix is "fix the guess" (cheap, helps everything) or "fix the
/// guess AND keep a stability net" (the guess is necessary but not sufficient).
///
/// Runs MINAO on every system in the sweep and reports whether it reaches the
/// PySCF reference AND whether its own stability check calls it stable.
/// Printing only — this is the measurement the Part-2 decision rests on.
#[test]
fn does_minao_alone_reach_the_reference_everywhere() {
    let cases: Vec<(&str, Sys, f64)> = vec![
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
            HENE_SVP_SIGMA,
        ),
        (
            "HeNe+/6-31G",
            diatomic("He", "Ne", 2.0, 1, 2, "6-31g"),
            HENE_631G_SIGMA,
        ),
        (
            "O2/6-31G",
            diatomic("O", "O", 1.2075, 0, 3, "6-31g"),
            O2_631G,
        ),
        (
            "NO/6-31G",
            diatomic("N", "O", 1.1508, 0, 2, "6-31g"),
            NO_631G,
        ),
        (
            "N2+/6-31G",
            diatomic("N", "N", 1.1160, 1, 2, "6-31g"),
            N2P_631G_SIGMA,
        ),
        (
            "CO+/6-31G",
            diatomic("C", "O", 1.1150, 1, 2, "6-31g"),
            COP_631G,
        ),
    ];
    let cfg = tight_cfg();
    println!("\n=== MINAO guess vs PySCF reference ===");
    for (name, sys, e_ref) in &cases {
        let d = ferric_scf::guess::minao_projection_guess(&sys.mol, &sys.prep, &sys.bs).unwrap();
        let (ca, cb) = mos_from_density(sys, &cfg, &(0.5 * &d), &(0.5 * &d));
        match solve_uhf_with_guess(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bounds,
            &cfg,
            Some((&ca, &cb)),
        ) {
            Ok(r) => {
                let d_ev = (r.energy - e_ref) * 27.211_386_245_988;
                let v = r
                    .stability
                    .as_ref()
                    .map(|s| s.verdict().label())
                    .unwrap_or("not checked");
                let lmin = r
                    .stability
                    .as_ref()
                    .map(|s| format!("{:+.4e}", s.lowest_eigenvalue))
                    .unwrap_or_else(|| "n/a".into());
                println!(
                    "{name:16} minao = {:.10}  ref = {e_ref:.10}  dE = {d_ev:+.4} eV  \
                     iters = {:3}  lambda_min = {lmin}  {v}{}",
                    r.energy,
                    r.iterations,
                    if d_ev > 1e-4 { "  <== STILL ABOVE" } else { "" }
                );
            }
            Err(e) => println!("{name:16} minao FAILED: {e:?}"),
        }
    }
}

/// Config with BOTH the stability check and the descent on. The two knobs are
/// meant to be set together: the descent reads the verdict the check produces.
pub fn descent_cfg() -> RhfConfig {
    RhfConfig {
        scf_stability_descent: true,
        ..tight_cfg()
    }
}

/// **Part 3 — proof the fix reaches the reference.** Every system in the sweep,
/// run through the FIXED default path (MINAO guess) plus the opt-in descent,
/// against its PySCF 2.13.0 reference.
///
/// # Why this is asserted at TWO bases and SIX systems
///
/// Pre-registered blind spot: *a reference comparison at a SINGLE input is
/// unfalsifiable regardless of tolerance* — measured in this lane, where a ΔIP
/// anchor matched PySCF to 3e-13 Ha with ferric's solver deleted. A constant
/// cannot satisfy −130.5053405386 and −130.6043266127 and −149.5455745334 and
/// −108.3186843228 simultaneously, so this assertion responds to inputs a
/// fabricator does not read.
#[test]
fn the_fixed_path_reaches_every_reference() {
    let cases: Vec<(&str, Sys, f64, f64)> = vec![
        // (name, system, reference, tolerance in Ha)
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
            HENE_SVP_SIGMA,
            1e-8,
        ),
        (
            "HeNe+/6-31G",
            diatomic("He", "Ne", 2.0, 1, 2, "6-31g"),
            HENE_631G_SIGMA,
            1e-8,
        ),
        (
            "O2/6-31G",
            diatomic("O", "O", 1.2075, 0, 3, "6-31g"),
            O2_631G,
            1e-7,
        ),
        (
            "NO/6-31G",
            diatomic("N", "O", 1.1508, 0, 2, "6-31g"),
            NO_631G,
            1e-7,
        ),
        (
            "N2+/6-31G",
            diatomic("N", "N", 1.1160, 1, 2, "6-31g"),
            N2P_631G_SIGMA,
            1e-6,
        ),
        (
            "CO+/6-31G",
            diatomic("C", "O", 1.1150, 1, 2, "6-31g"),
            COP_631G,
            1e-7,
        ),
    ];
    let cfg = descent_cfg();
    println!("\n=== FIXED PATH (MINAO guess + descent) vs PySCF 2.13.0 ===");
    let mut failures = Vec::new();
    for (name, sys, e_ref, tol) in &cases {
        let (_na, nb) = nocc_ab(&sys.mol);
        let r = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
        let hole = r
            .mos_beta
            .as_ref()
            .map(|cb| hole_label(&sys.prep, cb, nb))
            .unwrap_or("n/a");
        let d = r.energy - e_ref;
        let v = r
            .stability
            .as_ref()
            .map(|s| s.verdict())
            .unwrap_or(StabilityVerdict::Indeterminate);
        println!(
            "{name:16} ferric = {:.10} ({hole:>5})  ref = {e_ref:.10}  dE = {:+.2e} Ha \
             ({:+.4} eV)  {}",
            r.energy,
            d,
            d * 27.211_386_245_988,
            v.label()
        );
        if d.abs() > *tol {
            failures.push(format!("{name}: |dE| = {:.2e} > {tol:.0e} Ha", d.abs()));
        }
        // A solution ABOVE the reference that its own check calls UNSTABLE is
        // the exact defect this lane exists to remove.
        if d > *tol && v == StabilityVerdict::Unstable {
            failures.push(format!("{name}: UNSTABLE and above the reference"));
        }
    }
    assert!(
        failures.is_empty(),
        "the fixed path did not reach every reference:\n  {}",
        failures.join("\n  ")
    );
}

/// **HeNe⁺ specifically reaches the σ state and reports STABLE**, at BOTH
/// bases, from the DEFAULT config (no descent) — i.e. the guess fix alone
/// suffices here, which is what makes the descent an opt-in net rather than a
/// requirement.
#[test]
fn hene_reaches_sigma_and_is_stable_at_the_default() {
    for (basis, e_ref) in [("def2-svp", HENE_SVP_SIGMA), ("6-31g", HENE_631G_SIGMA)] {
        let sys = diatomic("He", "Ne", 2.0, 1, 2, basis);
        let cfg = tight_cfg(); // check_stability on, descent OFF
        let (_na, nb) = nocc_ab(&sys.mol);
        let r = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
        let cb = r.mos_beta.as_ref().unwrap();
        let st = r.stability.as_ref().expect("check_stability was set");
        println!(
            "HeNe+/{basis}: E = {:.10} (ref {e_ref:.10}), hole = {}, lambda_min = {:+.4e}, {}",
            r.energy,
            hole_label(&sys.prep, cb, nb),
            st.lowest_eigenvalue,
            st.verdict().label()
        );
        assert!(
            (r.energy - e_ref).abs() < 1e-8,
            "HeNe+/{basis}: E = {:.10} != reference {e_ref:.10} (|dE| = {:.2e})",
            r.energy,
            (r.energy - e_ref).abs()
        );
        assert_eq!(
            st.verdict(),
            StabilityVerdict::Stable,
            "HeNe+/{basis}: reached the reference energy but the verdict is {} \
             (lambda_min = {:+.4e})",
            st.verdict().label(),
            st.lowest_eigenvalue
        );
        assert_eq!(
            hole_label(&sys.prep, cb, nb),
            "sigma",
            "HeNe+/{basis}: reached the reference energy with a pi hole"
        );
    }
}

/// **The descent's acceptance guard, tested DIRECTLY.**
///
/// Inline in the descent loop this predicate is unreachable by the suite: on
/// every system measured here each descended candidate is lower, so replacing
/// it with `true` — i.e. accepting a strictly HIGHER state — leaves everything
/// green. It is the entire reason the descent cannot make an answer worse than
/// not having tried, so it is exercised as a function.
#[test]
fn descent_never_accepts_a_higher_state() {
    use ferric_scf::uhf::accepts_candidate_for_test as accepts;
    // strictly lower than the incumbent, nothing better yet -> accept
    assert!(accepts(-1.5, -1.0, None));
    // HIGHER than the incumbent -> reject, whatever else happened
    assert!(!accepts(-0.5, -1.0, None));
    assert!(!accepts(-0.5, -1.0, Some(-1.2)));
    // exactly equal -> reject (strict <, so the answer does not churn)
    assert!(!accepts(-1.0, -1.0, None));
    // lower than the incumbent but NOT better than an earlier candidate -> reject
    assert!(!accepts(-1.1, -1.0, Some(-1.3)));
    // lower than both -> accept
    assert!(accepts(-1.4, -1.0, Some(-1.3)));
}

/// **`scf_stability_descent = false` must reproduce the pre-descent answer
/// EXACTLY**, not approximately: the descent block is skipped entirely, so the
/// two runs are the same arithmetic. Bit-identity is asserted, because anything
/// looser would hide a descent that ran and returned something "close".
#[test]
fn descent_off_is_bit_identical_to_no_descent() {
    for (name, sys) in [
        (
            "HeNe+/def2-SVP",
            diatomic("He", "Ne", 2.0, 1, 2, "def2-svp"),
        ),
        ("O2/6-31G", diatomic("O", "O", 1.2075, 0, 3, "6-31g")),
    ] {
        let off = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &tight_cfg()).unwrap();
        let on = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &descent_cfg()).unwrap();
        println!(
            "{name}: descent off E = {:.12}, on E = {:.12}, delta = {:.3e}",
            off.energy,
            on.energy,
            (on.energy - off.energy).abs()
        );
        // These two systems are STABLE after the guess fix, so the descent is
        // not taken and the energies must agree to the LAST BIT.
        assert_eq!(
            off.energy.to_bits(),
            on.energy.to_bits(),
            "{name}: the descent changed a STABLE solution's energy ({:.12} -> {:.12}); \
             on a stable point it must do nothing at all",
            off.energy,
            on.energy
        );
    }
}

/// **`use_sad_guess = false` restores the pre-fix behaviour EXACTLY.** The
/// escape hatch is a real escape hatch: it reproduces the ²Π saddle the hcore
/// guess always found, bit-for-bit against an explicitly-injected hcore
/// density. Without this, "you can turn the new guess off" would be a claim
/// with no evidence.
#[test]
fn hcore_config_reproduces_the_old_pi_saddle() {
    let sys = diatomic("He", "Ne", 2.0, 1, 2, "def2-svp");
    let (_na, nb) = nocc_ab(&sys.mol);
    let cfg = hcore_cfg();
    let r = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    let st = r.stability.as_ref().expect("check_stability was set");
    println!(
        "use_sad_guess = false: E = {:.12}, hole = {}, lambda_min = {:+.4e}, {}",
        r.energy,
        hole_label(&sys.prep, r.mos_beta.as_ref().unwrap(), nb),
        st.lowest_eigenvalue,
        st.verdict().label()
    );
    assert!(
        (r.energy - HENE_SVP_PI).abs() < 1e-7,
        "use_sad_guess = false gave E = {:.10}, not the pre-fix pi state {HENE_SVP_PI:.8}; \
         the escape hatch does not restore the old behaviour",
        r.energy
    );
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "the pre-fix state should still be reported UNSTABLE (lambda_min = {:+.4e})",
        st.lowest_eigenvalue
    );
}

/// **The descent, not the guess, is what fixes N₂⁺.** Pins the split so a later
/// reader cannot attribute the whole repair to either half alone:
///   * guess alone (descent off): 0.7931 eV ABOVE the reference, UNSTABLE;
///   * guess + descent:            at the reference, STABLE.
#[test]
fn n2_cation_needs_the_descent_not_just_the_guess() {
    let sys = diatomic("N", "N", 1.1160, 1, 2, "6-31g");
    let no_descent = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &tight_cfg()).unwrap();
    let with_descent =
        solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &descent_cfg()).unwrap();
    let gap = (no_descent.energy - with_descent.energy) * 27.211_386_245_988;
    println!(
        "N2+/6-31G: guess only = {:.10} ({}), guess + descent = {:.10} ({}), \
         descent recovered {gap:.4} eV",
        no_descent.energy,
        no_descent
            .stability
            .as_ref()
            .map(|s| s.verdict().label())
            .unwrap_or("?"),
        with_descent.energy,
        with_descent
            .stability
            .as_ref()
            .map(|s| s.verdict().label())
            .unwrap_or("?"),
    );
    assert!(
        no_descent.energy - N2P_631G_SIGMA > 1e-3,
        "N2+ no longer needs the descent (guess-only E = {:.10} is already at the \
         reference); this test's premise is gone and the descent's justification with it",
        no_descent.energy
    );
    assert_eq!(
        no_descent.stability.as_ref().map(|s| s.verdict()).unwrap(),
        StabilityVerdict::Unstable,
        "the guess-only N2+ solution is no longer UNSTABLE"
    );
    assert!(
        (with_descent.energy - N2P_631G_SIGMA).abs() < 1e-6,
        "the descent did not reach the N2+ reference: E = {:.10} vs {N2P_631G_SIGMA:.10}",
        with_descent.energy
    );
    assert_eq!(
        with_descent
            .stability
            .as_ref()
            .map(|s| s.verdict())
            .unwrap(),
        StabilityVerdict::Stable,
        "the descended N2+ solution is not STABLE"
    );
}

/// **OH/6-31G: a SEVENTH system, found by a test in another crate.**
///
/// `ferric-dft`'s `fxc::tests::gga_fxc_matches_finite_difference_of_vxc` builds
/// its finite-difference reference density by calling `solve_uhf` on OH/6-31G.
/// After the guess fix that test began failing — which traced not to the f_xc
/// kernel but to the DENSITY it was differentiating:
///
/// ```text
///   hcore guess : E = -75.207996974998   UNSTABLE
///   MINAO guess : E = -75.363168246116   MARGINAL
///   PySCF 2.13.0: E = -75.363168249577   (stable at its own default guess)
/// ```
///
/// The pre-fix answer was **0.155 Ha = 4.22 eV above** the reference and its own
/// stability check called it a saddle. The fixed path agrees with PySCF to
/// 3.5e-9 Ha. So OH joins HeNe⁺ (two bases) and N₂⁺: **four of the seven
/// open-shell systems now measured were landing on the wrong state**, and this
/// one was found by an unrelated crate's test rather than by looking.
///
/// It is also why that f_xc test's residual improved by ~6 orders of magnitude
/// (6.1e-4 → 1.1e-9 at ε = 1e-2): a finite-difference check of an analytic
/// kernel agrees far better when both sides are evaluated at a density that is
/// actually a minimum. See that test for the guard that had to be relaxed as a
/// consequence.
#[test]
fn oh_reaches_the_pyscf_reference_after_the_fix() {
    /// PySCF 2.13.0 UHF, OH/6-31G at r(OH) = 0.97 A, stable at its own default
    /// guess (0 stability-following rounds).
    const OH_631G_PYSCF: f64 = -75.363_168_249_577;
    /// What ferric returned from the bare hcore guess before the fix.
    const OH_631G_HCORE: f64 = -75.207_996_974_998;

    let sys = diatomic("O", "H", 0.97, 0, 2, "6-31g");
    let fixed = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &tight_cfg()).unwrap();
    let old = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &hcore_cfg()).unwrap();
    println!(
        "OH/6-31G: hcore = {:.12} ({}), fixed = {:.12} ({}), pyscf = {OH_631G_PYSCF:.12}, \
         recovered {:.4} eV",
        old.energy,
        old.stability
            .as_ref()
            .map(|s| s.verdict().label())
            .unwrap_or("?"),
        fixed.energy,
        fixed
            .stability
            .as_ref()
            .map(|s| s.verdict().label())
            .unwrap_or("?"),
        (old.energy - fixed.energy) * 27.211_386_245_988
    );
    assert!(
        (fixed.energy - OH_631G_PYSCF).abs() < 1e-7,
        "OH/6-31G: the fixed path gives {:.12}, not the PySCF reference \
         {OH_631G_PYSCF:.12} (|dE| = {:.2e})",
        fixed.energy,
        (fixed.energy - OH_631G_PYSCF).abs()
    );
    assert!(
        (old.energy - OH_631G_HCORE).abs() < 1e-7,
        "the hcore guess no longer reproduces the pre-fix OH state {OH_631G_HCORE:.12} \
         (got {:.12}); this test's before/after premise is gone",
        old.energy
    );
    assert_eq!(
        old.stability.as_ref().map(|s| s.verdict()),
        Some(StabilityVerdict::Unstable),
        "the pre-fix OH state should still be reported UNSTABLE"
    );
}

/// **The descent must SKIP, not guess, on a reference it cannot analyse.**
///
/// `stability_uhf` already sets `ScfResult::stability` to `None` for a KS
/// reference whose f_xc response kernel cannot be built (range-separated,
/// meta-GGA) — analysing the HF Hessian at a KS density instead would be a
/// wrong-operator verdict. The descent reads that same field, so it inherits
/// the gate; this test pins that it actually does.
///
/// # Why it asserts the ENERGY and not just `stability.is_none()`
///
/// The first version of this test asserted only that the verdict field is
/// `None`, and a mutation that made the descent FABRICATE an `Unstable`
/// verdict whenever the field was `None` left it GREEN — the field really is
/// `None` either way, so checking it says nothing about what the descent then
/// does with it. The behavioural assertion is that turning the descent ON
/// changes NOTHING on such a reference: same energy, to the last bit. A
/// fabricating descent would rotate and re-converge, and could not be
/// bit-identical.
///
/// A range-separated functional is used because `ks_reference_is_analysable`
/// rejects it outright, so the skip is reached deterministically rather than
/// depending on a kernel build happening to fail.
#[test]
fn the_descent_skips_a_reference_it_cannot_analyse() {
    let sys = diatomic("O", "H", 0.97, 0, 2, "6-31g");
    let base = RhfConfig {
        xc: Some("wB97X-V".into()),
        ..tight_cfg()
    };
    let with_descent = RhfConfig {
        scf_stability_descent: true,
        ..base.clone()
    };
    let off = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &base);
    let on = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &with_descent);
    match (off, on) {
        (Ok(a), Ok(b)) => {
            println!(
                "RSH UKS OH/6-31G: descent off E = {:.12} (stability = {:?}), \
                 on E = {:.12} (stability = {:?})",
                a.energy,
                a.stability.as_ref().map(|s| s.verdict()),
                b.energy,
                b.stability.as_ref().map(|s| s.verdict())
            );
            assert!(
                a.stability.is_none(),
                "the RSH reference produced a stability verdict ({:?}); it should be \
                 None (not checked), because analysing the HF Hessian at a KS density \
                 would be a wrong-operator verdict",
                a.stability.as_ref().map(|s| s.verdict())
            );
            // THE BEHAVIOURAL ASSERTION: with no verdict to act on, the descent
            // must do nothing at all -- bit-identically, not approximately.
            assert_eq!(
                a.energy.to_bits(),
                b.energy.to_bits(),
                "turning on scf_stability_descent changed the energy of a reference \
                 whose stability was NEVER COMPUTED ({:.12} -> {:.12}). The descent is \
                 acting on a verdict it does not have.",
                a.energy,
                b.energy
            );
        }
        // Not converging is acceptable -- the point is that the descent does
        // not fabricate a verdict, and an Err never reaches it at all.
        (off, on) => println!(
            "RSH UKS OH/6-31G did not converge (off: {:?}, on: {:?}); descent unreachable",
            off.err(),
            on.err()
        ),
    }
}
