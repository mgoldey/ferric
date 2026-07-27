//! Distance vs estimated-pair-energy screening, head to head.
//!
//! # The question
//!
//! `build_pair_domains` screens on Boys-center distance in **Bohr**;
//! `build_pair_domains_by_energy` screens on the estimated MP2 **pair energy**
//! (ORCA's `T_CutPairs` criterion). Distance is system-size-dependent — a fixed
//! Bohr value does nothing until it drops below the size of the molecule, then
//! removes everything at once. Energy is comparable across systems by
//! construction. This measures whether that translates into a better
//! accuracy/retention curve in practice.
//!
//! # What "better" means here
//!
//! Not "retains more" — that is trivially achievable by screening less. The
//! useful criterion is **error at matched retention**: at the same fraction of
//! pairs kept, which rule keeps the pairs that matter? A criterion that drops
//! low-energy pairs first should win, because pair energies are exactly what the
//! correlation energy is a sum of.
//!
//! Counts and energies only — no wall clocks. The box these were taken on was
//! contested, and a timing from a loaded machine is untrustworthy in both
//! directions.

use ferric_cc::dlpno_ccsd::pair_mask_retention;
use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_mp2::boys::boys_localize;
use ferric_mp2::pair_domains::build_pair_domains;
use ferric_mp2::pair_energy_screen::{build_pair_domains_by_energy, estimate_pair_energies};
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use ndarray::Array2;

struct System {
    label: String,
    centers: Array2<f64>,
    g: Array2<f64>,
    eps: Vec<f64>,
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
}

/// Converge RHF, Boys-localize, and build the `(ia|jb)` matrix once. Both
/// criteria are then measured against the SAME orbitals and integrals, so any
/// difference is the screening rule and nothing else.
fn prepare(label: &str, path: &str, bas: &str) -> System {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(path).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(bas).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    assert!(rhf.converged, "{label}: SCF must converge");

    let inter = compute_rpa_intermediates(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    let (nocc, nvir, naux) = (inter.nocc, inter.nvir, inter.naux);
    let b = &inter.b_ov;
    let n = nocc * nvir;
    let g = Array2::from_shape_fn((n, n), |(p, q)| {
        (0..naux).map(|k| b[(k, p)] * b[(k, q)]).sum()
    });

    let c_occ = rhf.mos_r().slice(ndarray::s![.., ..nocc + inter.first_occ]).to_owned();
    let dip = ferric_integrals::oneelectron::dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
    let all_centers = boys_localize(&c_occ, &dip, 200).centers;
    // Boys ran over all occupied; the domains are built over the ACTIVE window.
    let centers = all_centers
        .slice(ndarray::s![inter.first_occ..inter.first_occ + nocc, ..])
        .to_owned();

    System {
        label: label.to_string(),
        centers,
        g,
        eps: rhf.eps_r().to_vec(),
        nocc,
        nvir,
        first_occ: inter.first_occ,
        nocc_total: inter.nocc_total,
    }
}

/// Correlation energy from only the retained pairs — the quantity screening
/// actually costs you.
fn energy_from_retained(sys: &System, domains: &ferric_mp2::pair_domains::PairDomains) -> f64 {
    let pe = estimate_pair_energies(
        sys.g.view(),
        &sys.eps,
        sys.nocc,
        sys.nvir,
        sys.first_occ,
        sys.nocc_total,
    )
    .unwrap();
    // Sum over UNIQUE i <= j pairs: PairEnergies already folds the off-diagonal
    // mirror factor into e_ij, so iterating the full grid would double-count.
    // `domains.pairs` is itself stored as i <= j, which is exactly what we want.
    domains.pairs.iter().map(|&(i, j)| pe.e_ij(i, j)).sum()
}

fn total_energy(sys: &System) -> f64 {
    estimate_pair_energies(
        sys.g.view(),
        &sys.eps,
        sys.nocc,
        sys.nvir,
        sys.first_occ,
        sys.nocc_total,
    )
    .unwrap()
    .total()
}

/// THE HEAD-TO-HEAD. Reports both criteria on the same systems, so the curves
/// can be compared at matched retention.
#[test]
fn distance_vs_pair_energy_screening() {
    let systems = [
        prepare("water/6-31G", "../../testdata/molecules/water.xyz", "6-31g"),
        prepare("benzene/STO-3G", "../../testdata/molecules/benzene.xyz", "sto-3g"),
    ];

    for sys in &systems {
        let e_full = total_energy(sys);
        eprintln!(
            "\n=== {} (nocc={}, nvir={})  full E_corr(MP2) = {:.10}",
            sys.label, sys.nocc, sys.nvir, e_full
        );

        eprintln!("  DISTANCE criterion (Bohr):");
        eprintln!("    {:>8}  {:>10}  {:>12}", "cutoff", "retention", "|dE| (Ha)");
        let mut dist_pts: Vec<(f64, f64)> = Vec::new();
        for cut in [f64::INFINITY, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5] {
            let d = build_pair_domains(&sys.centers, cut, f64::INFINITY).unwrap();
            let ret = pair_mask_retention(&d);
            let err = (energy_from_retained(sys, &d) - e_full).abs();
            eprintln!("    {cut:>8.1}  {ret:>10.4}  {err:>12.3e}");
            dist_pts.push((ret, err));
        }

        eprintln!("  PAIR-ENERGY criterion (Eh):");
        eprintln!("    {:>8}  {:>10}  {:>12}", "t_cut", "retention", "|dE| (Ha)");
        let pe = estimate_pair_energies(
            sys.g.view(),
            &sys.eps,
            sys.nocc,
            sys.nvir,
            sys.first_occ,
            sys.nocc_total,
        )
        .unwrap();
        let mut en_pts: Vec<(f64, f64)> = Vec::new();
        for t in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
            let d = build_pair_domains_by_energy(&sys.centers, &pe, t, f64::INFINITY).unwrap();
            let ret = pair_mask_retention(&d);
            let err = (energy_from_retained(sys, &d) - e_full).abs();
            eprintln!("    {t:>8.0e}  {ret:>10.4}  {err:>12.3e}");
            en_pts.push((ret, err));
        }

        // --- The comparison that matters: error at MATCHED retention. ---
        //
        // "Retains more" is trivially achievable by screening less, so it proves
        // nothing. For each distance point that actually screened something, find
        // the energy-criterion point with the closest retention and compare their
        // errors.
        eprintln!("  MATCHED-RETENTION COMPARISON (lower error at equal retention wins):");
        eprintln!("    {:>10}  {:>12}  {:>12}  {:>8}", "retention", "dist |dE|", "energy |dE|", "winner");
        let mut energy_wins = 0;
        let mut compared = 0;
        for &(dret, derr) in &dist_pts {
            if dret >= 0.999 {
                continue; // screened nothing; nothing to compare
            }
            let (eret, eerr) = en_pts
                .iter()
                .min_by(|a, b| {
                    (a.0 - dret).abs().partial_cmp(&(b.0 - dret).abs()).unwrap()
                })
                .copied()
                .unwrap();
            if (eret - dret).abs() > 0.15 {
                continue; // no comparable point; skip rather than mislead
            }
            compared += 1;
            let winner = if eerr < derr { energy_wins += 1; "energy" } else { "distance" };
            eprintln!("    {dret:>10.4}  {derr:>12.3e}  {eerr:>12.3e}  {winner:>8}");
        }
        eprintln!("  -> energy criterion won {energy_wins}/{compared} matched comparisons");
    }

    // Structural invariants that must hold regardless of which curve is better.
    for sys in &systems {
        let e_full = total_energy(sys);
        let pe = estimate_pair_energies(
            sys.g.view(),
            &sys.eps,
            sys.nocc,
            sys.nvir,
            sys.first_occ,
            sys.nocc_total,
        )
        .unwrap();

        // Zero threshold is exact.
        let d0 = build_pair_domains_by_energy(&sys.centers, &pe, 0.0, f64::INFINITY).unwrap();
        assert!(d0.is_complete(), "{}: t_cut=0 must retain every pair", sys.label);
        assert!(
            (energy_from_retained(sys, &d0) - e_full).abs() < 1e-12,
            "{}: t_cut=0 must reproduce the full correlation energy",
            sys.label
        );

        // Retention is monotone in the threshold.
        let mut last = 2.0_f64;
        for t in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
            let d = build_pair_domains_by_energy(&sys.centers, &pe, t, f64::INFINITY).unwrap();
            let r = pair_mask_retention(&d);
            assert!(r <= last + 1e-12, "{}: retention rose with the threshold", sys.label);
            last = r;
        }
    }
}

/// VALIDATE THE DEFAULT across a chemically diverse set.
///
/// The `t_cut_pairs = 1e-5` default was derived from two molecules. This checks
/// it on nine spanning polar/nonpolar, single/double/triple bonds, first- and
/// second-row heteroatoms, and a saturated chain — asking one question: at the
/// default, is the error small enough to be a safe default?
///
/// "Safe" is taken as < 1 kcal/mol (1.594e-3 Ha) of the total correlation
/// energy, an order of magnitude inside chemical accuracy. The per-molecule
/// numbers are printed so a reader can judge for themselves rather than trust
/// the threshold I picked.
#[test]
fn default_threshold_validated_across_molecules() {
    const KCAL: f64 = 1.593_601e-3; // Ha per kcal/mol
    let mols: [(&str, &str); 9] = [
        ("water", "water.xyz"),
        ("methanol", "ch3oh.xyz"),
        ("formaldehyde", "h2co.xyz"),
        ("CO2", "co2.xyz"),
        ("ethylene", "c2h4.xyz"),
        ("ethane", "c2h6.xyz"),
        ("H2S", "h2s.xyz"),
        ("dimethyl ether", "ch3och3.xyz"),
        ("benzene", "benzene.xyz"),
    ];

    eprintln!("\n=== t_cut_pairs default validation (STO-3G), threshold = 1e-5 Eh");
    eprintln!(
        "{:16} {:>5} {:>14} {:>10} {:>12} {:>10}",
        "molecule", "nocc", "E_corr", "retention", "|dE| (Ha)", "kcal/mol"
    );

    let mut worst = 0.0_f64;
    let mut worst_name = String::new();
    for (label, file) in mols {
        let sys = prepare(label, &format!("../../testdata/molecules/{file}"), "sto-3g");
        let e_full = total_energy(&sys);
        let pe = estimate_pair_energies(
            sys.g.view(), &sys.eps, sys.nocc, sys.nvir, sys.first_occ, sys.nocc_total,
        )
        .unwrap();
        let d = build_pair_domains_by_energy(&sys.centers, &pe, 1e-5, f64::INFINITY).unwrap();
        let ret = pair_mask_retention(&d);
        let err = (energy_from_retained(&sys, &d) - e_full).abs();
        eprintln!(
            "{label:16} {:>5} {e_full:>14.8} {ret:>10.4} {err:>12.3e} {:>10.4}",
            sys.nocc,
            err / KCAL
        );
        if err > worst {
            worst = err;
            worst_name = label.to_string();
        }
    }

    eprintln!(
        "worst case: {worst_name} at {worst:.3e} Ha = {:.4} kcal/mol",
        worst / KCAL
    );
    assert!(
        worst < KCAL,
        "the default t_cut_pairs = 1e-5 costs {:.4} kcal/mol on {worst_name}, \
         which is too much for a DEFAULT -- either loosen the claim or tighten \
         the threshold",
        worst / KCAL
    );
}
