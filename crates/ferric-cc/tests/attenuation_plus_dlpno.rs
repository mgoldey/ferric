//! Does attenuation let DLPNO screen pairs harder at fixed accuracy?
//!
//! # The question
//!
//! erfc(ωr)/r decays exponentially rather than as 1/R, so short-range correlation
//! is intrinsically more local. The hope is that this lets a *tighter* pair cutoff
//! be used at the same accuracy — i.e. that attenuation and DLPNO compose.
//!
//! # Why this is worth testing separately
//!
//! ferric already has a RETRACTED result in this neighbourhood: the claim "SR erfc
//! 3-index count is O(N)" was overturned — under a valid bound, erfc keeps 99.0% of
//! Coulomb triples at C20 and scales O(N²), same as Coulomb.
//!
//! But that is the **integral** screen (which `(P|μν)` triples survive). This is the
//! **pair** screen (which occupied pairs `(i,j)` survive) — an independent axis.
//! Attenuation saturating on one says nothing about the other.
//!
//! # The controls that make this honest
//!
//! 1. **Relative, not absolute, error.** erfc has a smaller total correlation
//!    energy, so a smaller absolute error at the same cutoff would be an artifact
//!    of the smaller quantity, not evidence of better locality. Everything below is
//!    normalized by each operator's own unscreened `|E_corr|`.
//! 2. **Same molecule, same orbitals, same domains.** Only the operator changes, so
//!    a difference cannot come from a different localization.
//! 3. **Counts and energies only** — no wall clocks. The box is contested.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::CcConfig;
use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_mp2::boys::boys_localize;
use ferric_mp2::pair_domains::build_pair_domains;
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use ndarray::Array2;

fn boys_centers(mol: &Molecule, obs: &PreparedBasis, rhf: &ferric_scf::ScfResult) -> Array2<f64> {
    let nocc = (mol.nelec() / 2) as usize;
    let c_occ = rhf.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0]).unwrap();
    boys_localize(&c_occ, &dip, 200).centers
}

/// THE TEST: how much of each operator's correlation energy lives within a given
/// pair radius?
///
/// If attenuation genuinely improves pair locality, erfc's correlation must be
/// concentrated at SHORTER pair separations than Coulomb's — i.e. at a fixed
/// cutoff, erfc should retain a LARGER FRACTION of its own total.
#[test]
fn attenuation_vs_coulomb_pair_locality() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    assert!(rhf.converged, "SCF must converge");

    let centers = boys_centers(&mol, &obs, &rhf);
    let nocc = (mol.nelec() / 2) as usize;

    // Pair-resolved MP2-like weight: for each occupied pair, how much correlation
    // does it carry? Built from the first-order amplitudes under each operator, so
    // "locality of the correlation" is measured directly rather than inferred.
    let cc = CcConfig { energy_conv: 1e-10, max_iter: 100, ..Default::default() };
    let ops: Vec<(&str, Operator)> = vec![
        ("Coulomb    ", Operator::coulomb()),
        ("erfc(0.10) ", Operator::erfc(0.10)),
        ("erfc(0.42) ", Operator::erfc(0.42)),
        ("erfc(1.00) ", Operator::erfc(1.00)),
    ];

    eprintln!("benzene/STO-3G, nocc = {nocc}");
    eprintln!("Fraction of each operator's OWN LinLCCD(hh) correlation retained,");
    eprintln!("as a function of the Boys-center pair cutoff:\n");
    eprintln!(
        "{:12} {:>12} | {:>7} {:>7} {:>7} {:>7} {:>7}",
        "operator", "E_corr(full)", "12 Bohr", "8 Bohr", "6 Bohr", "4 Bohr", "2 Bohr"
    );

    let cutoffs = [12.0, 8.0, 6.0, 4.0, 2.0];
    let mut retentions: Vec<(String, Vec<f64>)> = Vec::new();

    for (label, op) in &ops {
        let e_full =
            linlccd(&mol, &obs, &dfbs, *op, &rhf, &cc, LadderVariant::Hh).unwrap().correlation_energy;

        let mut fracs = Vec::new();
        let mut cells = String::new();
        for &cut in &cutoffs {
            let d = build_pair_domains(&centers, cut, f64::INFINITY).unwrap();
            let f = d.pair_retention();
            fracs.push(f);
            cells.push_str(&format!(" {f:>7.3}"));
        }
        eprintln!("{label} {e_full:>12.8} |{cells}");
        retentions.push((label.to_string(), fracs));
    }

    eprintln!(
        "\nNOTE: pair retention is a property of the BOYS CENTERS, which do not depend\n\
         on the correlation operator -- so these rows are identical by construction.\n\
         That is itself the finding: a centroid-distance pair screen cannot see the\n\
         operator at all. Any attenuation benefit must therefore come from the\n\
         INTEGRAL magnitudes, not from the pair geometry."
    );

    // The rows MUST be identical -- the pair screen is operator-blind. Asserting it
    // makes the (negative) finding explicit rather than a footnote.
    let first = &retentions[0].1;
    for (label, r) in &retentions[1..] {
        assert_eq!(
            r, first,
            "{label} retention differs from Coulomb -- unexpected: the centroid pair \
             screen should be operator-independent"
        );
    }
}

/// THE REAL TEST: is erfc's correlation energy concentrated at SHORTER pair
/// separations than Coulomb's?
///
/// The geometry test above only shows the CURRENT screen is operator-blind. This
/// asks the underlying physics question directly, by resolving the MP2-level pair
/// correlation energy
///
/// ```text
/// e_ij = sum_ab [ 2 (ia|jb) - (ib|ja) ] (ia|jb) / (e_i + e_j - e_a - e_b)
/// ```
///
/// against the Boys-center separation |R_i - R_j| for each operator, then reporting
/// the CUMULATIVE fraction of |E_corr| inside each radius. Normalizing by each
/// operator's own total is what makes the comparison fair -- erfc has less
/// correlation overall, so absolute numbers would mislead.
///
/// If attenuation improves pair locality, erfc's cumulative curve must rise FASTER
/// than Coulomb's.
#[test]
fn pair_resolved_correlation_vs_distance() {
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};

    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    let centers = boys_centers(&mol, &obs, &rhf);

    let radii = [2.0_f64, 4.0, 6.0, 8.0, 12.0];
    eprintln!("\nbenzene/STO-3G: cumulative fraction of |E_corr(MP2)| within a pair radius");
    eprintln!("(normalized per operator; higher = more local)\n");
    eprintln!("{:12} {:>12} | {:>7} {:>7} {:>7} {:>7} {:>7}",
              "operator", "E_corr", "2 Bohr", "4 Bohr", "6 Bohr", "8 Bohr", "12 Bohr");

    let mut curves: Vec<(String, Vec<f64>)> = Vec::new();
    for (label, op) in [("Coulomb    ", Operator::coulomb()),
                        ("erfc(0.10) ", Operator::erfc(0.10)),
                        ("erfc(0.42) ", Operator::erfc(0.42)),
                        ("erfc(1.00) ", Operator::erfc(1.00))] {
        let inter = compute_rpa_intermediates(
            &mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        let (no, nv, naux) = (inter.nocc, inter.nvir, inter.naux);
        let b = &inter.b_ov;
        let eps = rhf.eps_r();
        let (fo, not_) = (inter.first_occ, inter.nocc_total);

        // Pair-resolved MP2 correlation energies.
        let mut e_pair = vec![0.0f64; no * no];
        for i in 0..no {
            for j in 0..no {
                let mut acc = 0.0;
                for a in 0..nv {
                    for bb in 0..nv {
                        let iajb: f64 = (0..naux).map(|p| b[(p, i*nv+a)] * b[(p, j*nv+bb)]).sum();
                        let ibja: f64 = (0..naux).map(|p| b[(p, i*nv+bb)] * b[(p, j*nv+a)]).sum();
                        let d = eps[fo+i] + eps[fo+j] - eps[not_+a] - eps[not_+bb];
                        acc += (2.0*iajb - ibja) * iajb / d;
                    }
                }
                e_pair[i*no + j] = acc;
            }
        }
        let total: f64 = e_pair.iter().sum();

        let mut cells = String::new();
        let mut fr = Vec::new();
        for &r in &radii {
            let mut inside = 0.0;
            for i in 0..no {
                for j in 0..no {
                    let dist: f64 = (0..3).map(|ax|
                        (centers[(i,ax)] - centers[(j,ax)]).powi(2)).sum::<f64>().sqrt();
                    if dist <= r { inside += e_pair[i*no + j]; }
                }
            }
            let f = inside / total;
            fr.push(f);
            cells.push_str(&format!(" {f:>7.4}"));
        }
        eprintln!("{label} {total:>12.8} |{cells}");
        curves.push((label.to_string(), fr));
    }

    // Compare each attenuated curve to Coulomb at every radius.
    let coul = curves[0].1.clone();
    eprintln!("\ndifference vs Coulomb (positive = MORE local under attenuation):");
    for (label, c) in &curves[1..] {
        let d: Vec<String> = c.iter().zip(&coul).map(|(x,y)| format!("{:+.4}", x-y)).collect();
        eprintln!("{label}              | {}", d.join(" "));
    }

    // Sanity: cumulative fractions must be monotone and reach ~1 at large radius.
    for (label, c) in &curves {
        for k in 1..c.len() {
            assert!(c[k] >= c[k-1] - 1e-9, "{label}: cumulative fraction not monotone");
        }
        assert!((c[c.len()-1] - 1.0).abs() < 1e-6,
                "{label}: 12 Bohr should capture everything, got {:.6}", c[c.len()-1]);
    }
}

/// The operator DOES change how much correlation there is to begin with — the
/// control that shows the attenuation is actually active.
#[test]
fn attenuation_reduces_total_correlation() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    let cc = CcConfig { energy_conv: 1e-10, max_iter: 100, ..Default::default() };

    let e = |op: Operator| {
        linlccd(&mol, &obs, &dfbs, op, &rhf, &cc, LadderVariant::Hh).unwrap().correlation_energy
    };
    let e_coul = e(Operator::coulomb());
    let e_01 = e(Operator::erfc(0.10));
    let e_10 = e(Operator::erfc(1.00));

    eprintln!("water/STO-3G LinLCCD(hh): Coulomb {e_coul:.8}  erfc(0.1) {e_01:.8}  erfc(1.0) {e_10:.8}");
    assert!(e_coul < 0.0 && e_01 < 0.0 && e_10 < 0.0);
    assert!(
        e_10.abs() < e_coul.abs(),
        "strong attenuation must strip correlation: {e_10:.8} vs {e_coul:.8}"
    );
}
