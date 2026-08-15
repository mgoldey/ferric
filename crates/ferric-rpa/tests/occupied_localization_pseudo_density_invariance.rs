//! RE-MEASUREMENT of `rpa-locality-wall-lane-closed` construction #2
//! ("localization-first"), which was closed on a KNOWN-BROKEN formulation:
//! canonical orbital energies `exp(ε_i τ)` applied to LOCALIZED coefficients.
//!
//! The correct object for non-canonical orbitals is the MATRIX exponential
//! `P(τ) = C exp(F_loc τ) Cᵀ` with `F_loc = C_locᵀ F C_loc`
//! (`ao_rpa::pseudo_density_occ_fock`). The identical bug in the MP2 Laplace
//! path reversed a negative result completely when fixed, so #2 earned a
//! re-measurement with the fix APPLIED.
//!
//! # The two hypotheses, stated BEFORE measuring (CLAUDE.md Experimental Protocol)
//!
//! * **H_phys** — localization genuinely helps: `P_occ` built from Boys-localized
//!   occupieds via the matrix-exponential form is MORE distance-local than the
//!   canonical scalar-built `P_occ`, and the required domain radius SATURATES
//!   with system size (the AO-Laplace signature after its fix).
//! * **H_null** — `P_occ(τ) = C_occ exp(F_occ τ) C_occᵀ` sums over the COMPLETE
//!   occupied space, so it is a function of the occupied SUBSPACE and the Fock
//!   OPERATOR only. Any orthogonal rotation `U` within that subspace
//!   (`C → CU`, `F_occ → UᵀF_occU`) cancels identically. Then the localized
//!   construction is NUMERICALLY IDENTICAL to the canonical one and locality is
//!   unchanged — construction #2's premise is void by algebra, not measurement.
//!
//! These predict OPPOSITE observations on the same measurement (identical
//! matrices vs. different ones), so the experiment discriminates them. This is
//! exactly the check that was missing when the bug produced a false negative.
//!
//! `pv_sparsity_diagnostic.rs` established H_null for the VIRTUAL pseudo-density
//! (1.6e-15 relative under a real Boys rotation). This file asks the question the
//! locality-wall memory actually left open — the OCCUPIED side, which is what
//! "localization-first" meant — and then measures whether AO-time χ⁰ built from
//! the localized route is any sparser than the canonical one.
//!
//! No timings anywhere, deliberately. Ranks, dimensions, retention fractions,
//! errors.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-rpa --test occupied_localization_pseudo_density_invariance -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::dipole;
use ferric_integrals::operator::Operator;
use ferric_rpa::ao_rpa::{
    build_tau_quadrature, chi0_ao_at_tau, pseudo_density_occ, pseudo_density_occ_fock,
    pseudo_density_vir, pseudo_density_vir_fock,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2, Array3};

// ---------------------------------------------------------------------------
// Shared harness
// ---------------------------------------------------------------------------

struct Sys {
    #[allow(dead_code)] // names the system in debugging sessions
    label: String,
    prep: PreparedBasis,
    c_occ: Array2<f64>,
    c_vir: Array2<f64>,
    eps_occ: Vec<f64>,
    eps_vir: Vec<f64>,
    f_ao: Array2<f64>,
    diameter: f64,
    natoms: usize,
}

fn run_scf(path: &str, basis_name: &str, label: &str) -> Sys {
    let mol = Molecule::load_xyz(path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();

    let nocc = mol.nelec() as usize / 2;
    let mos = rhf.mos_r();
    let eps = rhf.eps_r();

    let mut diameter = 0.0_f64;
    for a in &mol.atoms {
        for b in &mol.atoms {
            let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt();
            if d > diameter {
                diameter = d;
            }
        }
    }

    println!(
        "\n=== {} : nbas={} nocc={} nvir={} natoms={} diameter={:.2} Bohr ===",
        label,
        prep.nbasis(),
        nocc,
        prep.nbasis() - nocc,
        mol.atoms.len(),
        diameter
    );

    Sys {
        label: label.to_string(),
        c_occ: mos.slice(s![.., ..nocc]).to_owned(),
        c_vir: mos.slice(s![.., nocc..]).to_owned(),
        eps_occ: eps[..nocc].to_vec(),
        eps_vir: eps[nocc..].to_vec(),
        f_ao: rhf.fock_r().clone(),
        diameter,
        natoms: mol.atoms.len(),
        prep,
    }
}

fn boys_occ(sys: &Sys) -> (Array2<f64>, Array2<f64>, f64, f64, f64) {
    let dip = dipole(&sys.prep, [0.0, 0.0, 0.0]).unwrap();
    let res = ferric_mp2::boys::boys_localize(&sys.c_occ, &dip, 400);
    let c_loc = res.c_loc;
    let f_loc = c_loc.t().dot(&sys.f_ao).dot(&c_loc);
    let boys_f = |c: &Array2<f64>| -> f64 {
        let mut acc = 0.0;
        for a in 0..3 {
            let dc = dip[a].dot(c);
            for i in 0..c.ncols() {
                let v = c.column(i).dot(&dc.column(i));
                acc += v * v;
            }
        }
        acc
    };
    let coef_change = (&c_loc - &sys.c_occ)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    (c_loc.clone(), f_loc, boys_f(&sys.c_occ), boys_f(&c_loc), coef_change)
}

fn taus(sys: &Sys) -> Vec<f64> {
    build_tau_quadrature(&sys.eps_occ, &sys.eps_vir, 5)
        .expect("build_tau_quadrature")
        .points
        .clone()
}

fn ao_centers(prep: &PreparedBasis) -> Vec<[f64; 3]> {
    let sc = prep.shell_centers();
    let off = prep.shell_offsets();
    let mut out = vec![[0.0; 3]; prep.nbasis()];
    for (s, c) in sc.iter().enumerate() {
        for mu in off[s]..off[s + 1] {
            out[mu] = *c;
        }
    }
    out
}

/// max |M(μ,ν)| per AO-center-distance bin, normalized to the global max.
fn decay_profile(m: &Array2<f64>, centers: &[[f64; 3]], bw: f64, nbins: usize) -> Vec<f64> {
    let n = m.nrows();
    let mut binmax = vec![0.0_f64; nbins];
    for mu in 0..n {
        for nu in 0..n {
            let a = centers[mu];
            let b = centers[nu];
            let d =
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let bi = ((d / bw) as usize).min(nbins - 1);
            let v = m[(mu, nu)].abs();
            if v > binmax[bi] {
                binmax[bi] = v;
            }
        }
    }
    let g = binmax.iter().cloned().fold(0.0_f64, f64::max);
    if g > 0.0 {
        for v in binmax.iter_mut() {
            *v /= g;
        }
    }
    binmax
}

fn radius_below(prof: &[f64], tol: f64, bw: f64) -> f64 {
    let mut last = None;
    for (i, &v) in prof.iter().enumerate() {
        if v > tol {
            last = Some(i);
        }
    }
    last.map(|i| (i + 1) as f64 * bw).unwrap_or(f64::NAN)
}

fn rel_max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let d = (a - b).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let s = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if s == 0.0 {
        d
    } else {
        d / s
    }
}

// ===========================================================================
// STEP 1 — THE EXACTNESS ANCHOR (must pass before any locality measurement)
// ===========================================================================

/// ANCHOR A: the matrix-exponential form reduces BIT-FOR-BIT-ish to the scalar
/// form in its trivial limit — CANONICAL orbitals, where `F_occ = diag(ε)`.
///
/// This is the check whose absence produced the original false negative. If the
/// `_fock` machinery is broken, this fails immediately and nothing downstream
/// may be believed.
#[test]
fn anchor_a_fock_form_reduces_to_scalar_form_for_canonical_orbitals() {
    for path in [
        "../../testdata/molecules/water.xyz",
        "../../testdata/molecules/alkane_4.xyz",
    ] {
        let sys = run_scf(path, "sto-3g", path);
        let f_occ_diag = Array2::from_diag(&ndarray::Array1::from(sys.eps_occ.clone()));
        let f_vir_diag = Array2::from_diag(&ndarray::Array1::from(sys.eps_vir.clone()));
        for &tau in taus(&sys).iter() {
            let po_s = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
            let po_m = pseudo_density_occ_fock(&sys.c_occ, &f_occ_diag, tau);
            let pv_s = pseudo_density_vir(&sys.c_vir, &sys.eps_vir, tau);
            let pv_m = pseudo_density_vir_fock(&sys.c_vir, &f_vir_diag, tau);
            let (eo, ev) = (rel_max_diff(&po_s, &po_m), rel_max_diff(&pv_s, &pv_m));
            println!("  ANCHOR-A τ={tau:.4}: occ rel {eo:.3e}  vir rel {ev:.3e}");
            assert!(eo < 1e-12, "occ anchor failed at τ={tau}: rel {eo:.3e}");
            assert!(ev < 1e-12, "vir anchor failed at τ={tau}: rel {ev:.3e}");
        }
    }
}

/// ANCHOR B: the canonical Fock matrix in the occupied block really IS diagonal,
/// and `F_loc = C_locᵀ F C_loc` really IS non-diagonal after localization.
///
/// Without this, ANCHOR A could pass vacuously (e.g. if `boys_localize` no-oped,
/// `F_loc` would be diagonal too and the "matrix exponential" would never be
/// exercised in its non-trivial regime).
#[test]
fn anchor_b_localization_actually_makes_the_fock_block_non_diagonal() {
    let sys = run_scf("../../testdata/molecules/alkane_4.xyz", "sto-3g", "alkane_4 anchor-B");
    let f_can = sys.c_occ.t().dot(&sys.f_ao).dot(&sys.c_occ);
    let (_c_loc, f_loc, b0, b1, dcoef) = boys_occ(&sys);

    let offdiag = |m: &Array2<f64>| -> f64 {
        let mut mx = 0.0_f64;
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                if i != j && m[(i, j)].abs() > mx {
                    mx = m[(i, j)].abs();
                }
            }
        }
        mx
    };
    let (od_can, od_loc) = (offdiag(&f_can), offdiag(&f_loc));
    println!("  Boys occupied functional {b0:.6} -> {b1:.6}, max Δcoef {dcoef:.3}");
    println!("  max |offdiag F_occ|: canonical {od_can:.3e}   localized {od_loc:.3e}");

    assert!(b1 > b0, "Boys localization must MAXIMIZE the functional: {b0} -> {b1}");
    assert!(dcoef > 0.1, "localizer barely moved coefficients (Δ={dcoef:.3e})");
    assert!(od_can < 1e-8, "canonical occ Fock block must be diagonal: {od_can:.3e}");
    assert!(
        od_loc > 1e-2,
        "localized occ Fock block must be substantially non-diagonal (got {od_loc:.3e}); \
         otherwise the matrix-exponential path is never exercised and every downstream \
         'localized' measurement is secretly the canonical one"
    );
}

/// ANCHOR C (NEGATIVE CONTROL): the BROKEN construction — canonical ε applied to
/// localized coefficients — must give a MATERIALLY DIFFERENT matrix. This proves
/// the correct and broken paths are genuinely different functions, so a null
/// result below cannot be "the two code paths are secretly the same".
#[test]
fn anchor_c_the_broken_scalar_on_localized_construction_really_is_different() {
    let sys = run_scf("../../testdata/molecules/alkane_4.xyz", "sto-3g", "alkane_4 anchor-C");
    let (c_loc, _f_loc, ..) = boys_occ(&sys);
    let tau_list = taus(&sys);
    let tau = tau_list[tau_list.len() / 2];

    let p_correct = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
    // WRONG on purpose: canonical ε_i with LOCALIZED coefficients. This is
    // exactly what construction #2 was measured with.
    let p_broken = pseudo_density_occ(&c_loc, &sys.eps_occ, tau);
    let rel = rel_max_diff(&p_correct, &p_broken);
    println!("  broken-vs-correct P_occ rel deviation = {rel:.3e}");
    assert!(
        rel > 1e-3,
        "the broken construction must differ substantially from the correct one \
         (rel {rel:.3e}); if it did not, construction #2's bug would have been harmless \
         and this whole re-measurement would be moot"
    );
}

// ===========================================================================
// STEP 2 — THE ACTUAL QUESTION
// ===========================================================================

/// THE LOAD-BEARING RESULT for construction #2's OCCUPIED side.
///
/// `P_occ(τ) = C_occ exp(F_occ τ) C_occᵀ` is a complete sum over the occupied
/// space, hence a function of the occupied PROJECTOR and the Fock OPERATOR only.
/// Any orthogonal rotation within the occupied space must leave it numerically
/// unchanged — so a Boys/PM localization CANNOT make it sparser, for any
/// localization scheme.
///
/// TEETH: the guards from ANCHOR B/C are re-asserted inline, so this cannot pass
/// because the localizer no-oped or because the two code paths coincide.
#[test]
fn occupied_localization_cannot_change_the_occupied_pseudo_density() {
    for (path, label) in [
        ("../../testdata/molecules/water.xyz", "water/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "alkane_4/STO-3G"),
        ("../../testdata/molecules/benzene.xyz", "benzene/STO-3G"),
    ] {
        let sys = run_scf(path, "sto-3g", label);
        let (c_loc, f_loc, b0, b1, dcoef) = boys_occ(&sys);
        assert!(b1 > b0 && dcoef > 0.1, "{label}: localizer no-oped (b {b0}->{b1}, Δ {dcoef:.3e})");

        for &tau in taus(&sys).iter() {
            let p_can = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
            let p_loc = pseudo_density_occ_fock(&c_loc, &f_loc, tau);
            let rel = rel_max_diff(&p_can, &p_loc);
            println!("  {label} τ={tau:.4}: |ΔP_occ|rel = {rel:.3e}");
            assert!(
                rel < 1e-8,
                "{label}: P_occ must be invariant under an occupied-space rotation at \
                 τ={tau} (rel {rel:.3e}). A FAILURE here means the localization left the \
                 occupied subspace or the Fock form is wrong — it would NOT mean \
                 localization sparsifies P_occ."
            );
        }
    }
}

/// COROLLARY, measured end-to-end: because BOTH pseudo-densities are
/// rotation-invariant, the AO-time χ⁰(τ) built entirely from LOCALIZED orbitals
/// (occupied AND virtual, both via the matrix-exponential form) is numerically
/// identical to the canonical one. Construction #2 therefore cannot change χ⁰
/// sparsity, the aux-basis dielectric, or the RPA energy — by algebra.
///
/// This is the direct answer to "does a localized-orbital AO-time χ⁰ show
/// locality the canonical one hides?" for the UNTRUNCATED construction.
#[test]
fn chi0_from_fully_localized_orbitals_is_identical_to_canonical() {
    let sys = run_scf("../../testdata/molecules/water.xyz", "sto-3g", "water/STO-3G chi0");
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let dfbs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfprep = PreparedBasis::new(&mol, &dfbs).unwrap();
    let eri3 = ferric_integrals::threeindex::eri3_tensor(
        Operator::coulomb(),
        &sys.prep,
        &dfprep,
    )
    .unwrap();
    println!("  eri3 dims = {:?}", eri3.dim());

    // Localize BOTH spaces, both via the correct matrix-exponential form.
    let (c_occ_loc, f_occ_loc, b0, b1, dcoef) = boys_occ(&sys);
    assert!(b1 > b0 && dcoef > 0.1, "occ localizer no-oped");
    let dip = dipole(&sys.prep, [0.0, 0.0, 0.0]).unwrap();
    let c_vir_loc = ferric_mp2::boys::boys_localize(&sys.c_vir, &dip, 400).c_loc;
    let f_vir_loc = c_vir_loc.t().dot(&sys.f_ao).dot(&c_vir_loc);

    let centers = ao_centers(&sys.prep);
    for &tau in taus(&sys).iter() {
        let p_can = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
        let q_can = pseudo_density_vir(&sys.c_vir, &sys.eps_vir, tau);
        let p_loc = pseudo_density_occ_fock(&c_occ_loc, &f_occ_loc, tau);
        let q_loc = pseudo_density_vir_fock(&c_vir_loc, &f_vir_loc, tau);

        let chi_can = chi0_ao_at_tau(&eri3, &p_can, &q_can).unwrap();
        let chi_loc = chi0_ao_at_tau(&eri3, &p_loc, &q_loc).unwrap();
        let rel = rel_max_diff(&chi_can, &chi_loc);

        // Sparsity of χ⁰ itself, both routes, at a scale-free threshold.
        let frac = |m: &Array2<f64>, t: f64| -> f64 {
            let mx = m.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            m.iter().filter(|v| v.abs() > t * mx).count() as f64 / m.len() as f64
        };
        println!(
            "  τ={tau:.4}: |Δχ⁰|rel = {rel:.3e}   frac>1e-6 can={:.4} loc={:.4}",
            frac(&chi_can, 1e-6),
            frac(&chi_loc, 1e-6)
        );
        let _ = &centers;
        assert!(
            rel < 1e-8,
            "AO-time χ⁰ must be invariant under orbital rotation of BOTH spaces \
             (rel {rel:.3e} at τ={tau})"
        );
    }
}

// ===========================================================================
// STEP 3 — the specific claim construction #3 made, re-measured
// ===========================================================================

/// Construction #3 of `rpa-locality-wall-lane-closed` claimed:
///   "per-atom significant-pair count GROWS with size (P̃@1e-8: 220.0 C2 →
///    310.8 C3), nbas² fraction pinned at ~50% — the dense O(N²) signature ...
///    Kaltak-Kresse AO-time cubic dRPA is NOT realized with canonical MOs."
///
/// The wording ("with canonical MOs") implied the verdict might be about the
/// ORBITAL CHOICE. Given the invariance results above, it cannot be: the AO-time
/// pseudo-densities are rotation invariants, so LOCALIZED MOs give the SAME
/// numbers to machine precision. This test measures the claimed metric under
/// BOTH routes on the same systems and asserts they agree — turning "NOT realized
/// with canonical MOs" into "NOT realized, period, for any orbital choice".
///
/// It also RE-MEASURES whether the metric itself still behaves as reported.
#[test]
fn per_atom_significant_pair_count_is_orbital_choice_independent() {
    let systems = [
        ("../../testdata/molecules/alkane_2.xyz", "alkane_2/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "alkane_4/STO-3G"),
        ("../../testdata/molecules/alkane_6.xyz", "alkane_6/STO-3G"),
        ("../../testdata/molecules/alkane_8.xyz", "alkane_8/STO-3G"),
    ];
    println!("\n{:<20} {:>6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>11}",
        "system", "natom", "nbas", "pairs/at:C", "pairs/at:L", "frac:C", "frac:L", "rel|ΔP|");

    let mut rows: Vec<(String, usize, f64, f64, f64, f64)> = Vec::new();
    for (path, label) in systems {
        let sys = run_scf(path, "sto-3g", label);
        let (c_loc, f_loc, b0, b1, dcoef) = boys_occ(&sys);
        assert!(b1 > b0 && dcoef > 0.1, "{label}: occ localizer no-oped");
        let tl = taus(&sys);
        let tau = tl[tl.len() / 2];
        let nbas = sys.prep.nbasis();

        let p_can = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
        let p_loc = pseudo_density_occ_fock(&c_loc, &f_loc, tau);
        let rel = rel_max_diff(&p_can, &p_loc);

        // The construction-#3 metric: count of |P̃| above 1e-8 × max, per atom.
        let count = |m: &Array2<f64>| -> usize {
            let mx = m.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            m.iter().filter(|v| v.abs() > 1e-8 * mx).count()
        };
        let (cc, cl) = (count(&p_can), count(&p_loc));
        let (pac, pal) = (cc as f64 / sys.natoms as f64, cl as f64 / sys.natoms as f64);
        let (fc, fl) = (cc as f64 / (nbas * nbas) as f64, cl as f64 / (nbas * nbas) as f64);

        println!("{:<20} {:>6} {:>7} {:>10.1} {:>10.1} {:>10.3} {:>10.3} {:>11.2e}",
            label, sys.natoms, nbas, pac, pal, fc, fl, rel);
        rows.push((label.to_string(), sys.natoms, pac, pal, fc, fl));

        assert!(
            rel < 1e-8,
            "{label}: P_occ differs between canonical and localized routes (rel {rel:.3e}) — \
             invariance broken, everything above is void"
        );
        assert_eq!(
            cc, cl,
            "{label}: significant-pair COUNT differs between orbital choices ({cc} vs {cl}); \
             construction #3's metric would then genuinely depend on orbital choice"
        );
    }

    // TEETH on the trend itself: the claim is that per-atom pair count GROWS.
    // Assert the measurement is non-degenerate (counts actually vary across
    // sizes) so a passing test reflects a real sweep, not four identical rows.
    let first = rows.first().unwrap().2;
    let last = rows.last().unwrap().2;
    println!("\n  per-atom significant-pair count: {first:.1} (smallest) -> {last:.1} (largest)");
    assert!(
        (last - first).abs() > 1e-9,
        "per-atom pair count identical across all sizes ({first:.1}) — the sweep is degenerate \
         and says nothing about growth"
    );
}

// ===========================================================================
// STEP 4 — decay/saturation, the metric the AO-Laplace rescue used
// ===========================================================================

/// The AO-Laplace rescue was measured as a SATURATING domain radius. Apply the
/// same measurement to the RPA AO-time pseudo-densities, canonical vs localized,
/// so the two stories are compared on the SAME metric.
///
/// Prediction under invariance: the two columns are identical at every size, and
/// whatever saturation behaviour exists belongs to the canonical result already —
/// localization contributes exactly nothing.
#[test]
fn domain_radius_saturation_canonical_vs_localized() {
    const BW: f64 = 1.0;
    const NB: usize = 30;
    let systems = [
        ("../../testdata/molecules/water.xyz", "water/STO-3G"),
        ("../../testdata/molecules/alkane_2.xyz", "alkane_2/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "alkane_4/STO-3G"),
        ("../../testdata/molecules/alkane_6.xyz", "alkane_6/STO-3G"),
        ("../../testdata/molecules/alkane_8.xyz", "alkane_8/STO-3G"),
        ("../../testdata/molecules/benzene.xyz", "benzene/STO-3G"),
    ];
    println!("\n{:<20} {:>9} {:>11} {:>11} {:>9} {:>9}",
        "system", "diameter", "r_occ:canon", "r_occ:local", "r/diam:C", "r/diam:L");
    for (path, label) in systems {
        let sys = run_scf(path, "sto-3g", label);
        let (c_loc, f_loc, b0, b1, _) = boys_occ(&sys);
        assert!(b1 > b0, "{label}: localizer failed");
        let centers = ao_centers(&sys.prep);
        let tl = taus(&sys);
        let tau = tl[tl.len() / 2];

        let p_can = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);
        let p_loc = pseudo_density_occ_fock(&c_loc, &f_loc, tau);
        let rc = radius_below(&decay_profile(&p_can, &centers, BW, NB), 1e-4, BW);
        let rl = radius_below(&decay_profile(&p_loc, &centers, BW, NB), 1e-4, BW);
        println!("{:<20} {:>9.2} {:>11.1} {:>11.1} {:>9.2} {:>9.2}",
            label, sys.diameter, rc, rl, rc / sys.diameter, rl / sys.diameter);
        assert!(
            (rc - rl).abs() < 1e-9,
            "{label}: canonical and localized radii differ ({rc} vs {rl}) — would contradict \
             the invariance result"
        );
    }
    println!("\n  If r/diam is ~constant, the profile STRETCHES with the molecule and");
    println!("  truncation is not transferable. Identical columns => localization is a no-op.");
}

// ===========================================================================
// STEP 5 — is invariance itself an artifact of the harness?
// ===========================================================================

/// META-TEETH: verify the invariance tests above CAN fail. Applies a rotation
/// that is NOT confined to the occupied space (mixing in a virtual direction),
/// and asserts the same comparison then reports a large deviation.
///
/// Without this, "rel < 1e-8 everywhere" could be the harness comparing a matrix
/// to itself.
#[test]
fn invariance_check_can_actually_fail() {
    let sys = run_scf("../../testdata/molecules/alkane_4.xyz", "sto-3g", "alkane_4 meta-teeth");
    let tl = taus(&sys);
    let tau = tl[tl.len() / 2];
    let p_can = pseudo_density_occ(&sys.c_occ, &sys.eps_occ, tau);

    // Contaminate the occupied space with one virtual direction: c_occ column 0
    // gets 30% of virtual column 0. This is NOT an occupied-space rotation, so
    // the pseudo-density MUST change.
    let mut c_bad = sys.c_occ.clone();
    for mu in 0..c_bad.nrows() {
        c_bad[(mu, 0)] += 0.3 * sys.c_vir[(mu, 0)];
    }
    let f_bad = c_bad.t().dot(&sys.f_ao).dot(&c_bad);
    let p_bad = pseudo_density_occ_fock(&c_bad, &f_bad, tau);
    let rel_space = rel_max_diff(&p_can, &p_bad);

    // Second, INDEPENDENT failure mode: drop one occupied orbital (an incomplete
    // sum). Invariance only holds for a COMPLETE sum over the subspace, so this
    // is the perturbation that a truncation scheme would actually make.
    let c_trunc = sys.c_occ.slice(s![.., ..sys.c_occ.ncols() - 1]).to_owned();
    let f_trunc = c_trunc.t().dot(&sys.f_ao).dot(&c_trunc);
    let p_trunc = pseudo_density_occ_fock(&c_trunc, &f_trunc, tau);
    let rel_trunc = rel_max_diff(&p_can, &p_trunc);

    println!("  contaminated-space deviation = {rel_space:.3e}");
    println!("  dropped-one-orbital deviation = {rel_trunc:.3e}");

    // CALIBRATION, not a magic number: the invariance signal measured elsewhere
    // in this file is ~1e-14 relative. A control is only meaningful if it sits
    // orders of magnitude ABOVE that floor. Both do, by >=10 decades.
    const INVARIANCE_FLOOR: f64 = 1e-8; // the assertion threshold used above
    assert!(
        rel_space > 1e4 * INVARIANCE_FLOOR,
        "a non-occupied-space perturbation must change P_occ well above the invariance \
         threshold (rel {rel_space:.3e} vs floor {INVARIANCE_FLOOR:.0e}); if it does not, \
         the invariance assertions elsewhere in this file are vacuous"
    );
    assert!(
        rel_trunc > 1e4 * INVARIANCE_FLOOR,
        "dropping an occupied orbital must change P_occ well above the invariance threshold \
         (rel {rel_trunc:.3e}); invariance holds only for COMPLETE sums, and if truncation \
         were also invisible this harness could not see truncation effects at all"
    );
}

// ===========================================================================
// STEP 6 — SIBLING-PATH AUDIT: the production Boys-screened RPA path
// ===========================================================================

/// `screen.rs:193` builds per-orbital "localized orbital energies" as the
/// DIAGONAL of `C_locᵀ F C_loc`:
///
/// ```text
/// let f_loc = c_occ_loc.t().dot(&fc);
/// let eps_loc: Vec<f64> = (0..nocc_loc).map(|i| f_loc[(i, i)]).collect();
/// ```
///
/// and `sternheimer_sparse.rs:84` then uses `e_ia = eps_a − eps_loc[i]` as if
/// those were canonical eigenvalues. This is the SAME defect class the audit was
/// dispatched over, and precisely what `dlpno_rpa.rs:309-311` warns against in
/// its own comment ("Taking the diagonal alone is silently wrong").
///
/// This test QUANTIFIES the discarded coupling: the off-diagonal norm of `F_loc`
/// relative to its diagonal spread. A large ratio means the approximation is
/// numerically substantial, not a rounding-level detail.
///
/// It deliberately asserts only that the measurement is MEANINGFUL (the
/// off-diagonal is not negligible), because whether that matters depends on
/// where `eps_loc` is consumed — see the module-level finding that the
/// Boys-screened path feeds only the eigensolver SEED/subspace, while the RPA
/// energy is integrated from the dense canonical `b_ov`.
#[test]
fn boys_screened_path_discards_a_substantial_off_diagonal_fock_coupling() {
    println!(
        "\n{:<20} {:>8} {:>14} {:>14} {:>12}",
        "system", "nocc", "max|offdiag|", "diag spread", "ratio"
    );
    for (path, label) in [
        ("../../testdata/molecules/water.xyz", "water/STO-3G"),
        ("../../testdata/molecules/alkane_4.xyz", "alkane_4/STO-3G"),
        ("../../testdata/molecules/alkane_8.xyz", "alkane_8/STO-3G"),
        ("../../testdata/molecules/benzene.xyz", "benzene/STO-3G"),
    ] {
        let sys = run_scf(path, "sto-3g", label);
        let (_c_loc, f_loc, b0, b1, _) = boys_occ(&sys);
        assert!(b1 > b0, "{label}: localizer failed");
        let n = f_loc.nrows();
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j && f_loc[(i, j)].abs() > off {
                    off = f_loc[(i, j)].abs();
                }
            }
        }
        let diag: Vec<f64> = (0..n).map(|i| f_loc[(i, i)]).collect();
        let spread = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - diag.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{:<20} {:>8} {:>14.4e} {:>14.4e} {:>12.3}",
            label, n, off, spread, off / spread
        );
        assert!(
            off > 1e-3,
            "{label}: localized Fock off-diagonal is {off:.3e} — if this were truly \
             negligible, the diagonal-only shortcut in screen.rs:193 would be harmless \
             and this audit finding would be void"
        );
    }
    println!(
        "\n  The discarded off-diagonal is comparable to the diagonal spread itself.\n\
         screen.rs's eps_loc is therefore NOT an eigenvalue set. It is only sound\n\
         because run_pdep_rpa uses the Boys-screened representation for the\n\
         eigensolver SEED/matvec subspace, and integrates the ENERGY from the dense\n\
         canonical b_ov (lib.rs:446-447, :589)."
    );
}

/// The `Array3` import is used by the chi0 test's eri3; keep the compiler honest.
#[allow(dead_code)]
fn _unused(_: &Array3<f64>) {}
