//! Measurement probe (not a test): promolecule Becke populations and atomic
//! static polarizabilities for the HeNe⁺ cDFT-ET lane.
//!
//! Part (b): converge each fragment in the FULL dimer AO basis using ghost
//! atoms, sum the fragment densities into a promolecule density, and evaluate
//! N_He = Tr[W_He · D_promol] on the driver's own (99,302) Becke grid.
//!
//! Part (c): finite-field static polarizabilities of the isolated He and Ne
//! atoms in def2-SVP, for a later R⁻⁴ ion-induced-dipole test with no borrowed
//! constants.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-scf --example hene_promolecule_probe

use ferric_core::basis;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const HA2EV: f64 = 27.211_386_245_988;

fn cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

/// Solve UHF for `xyz` at the given charge/multiplicity and return
/// (E, D_alpha, D_beta). Panics unless converged — used for the promolecule
/// fragments, which must be trustworthy.
fn uhf(xyz: &str, charge: i32, mult: usize, cf: &RhfConfig) -> (f64, Array2<f64>, Array2<f64>) {
    try_uhf(xyz, charge, mult, cf)
        .unwrap_or_else(|e| panic!("not converged: {xyz} q={charge} m={mult}: {e}"))
}

/// Non-fatal variant: returns Err(description) instead of panicking, so a
/// stalled *diagnostic* system (the unconstrained HeNe⁺ dimer) does not kill
/// the promolecule table, which is the actual deliverable.
#[allow(clippy::type_complexity)]
fn try_uhf(
    xyz: &str,
    charge: i32,
    mult: usize,
    cf: &RhfConfig,
) -> Result<(f64, Array2<f64>, Array2<f64>), String> {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, cf).map_err(|e| format!("{e:?}"))?;
    if !res.converged {
        return Err(format!("{:?}", res.exit));
    }
    let da = res.density_alpha.clone();
    let db = res.density_beta.clone().unwrap_or_else(|| da.clone());
    Ok((res.energy, da, db))
}

/// XYZ strings for the dimer and its ghost-padded fragments at separation `r` (Å).
/// He sits at the origin, Ne on +z. Atom order is ALWAYS (He, Ne) so that the
/// AO ordering — and therefore the density matrices — are directly summable.
fn geoms(r: f64) -> (String, String, String) {
    let dimer = format!("2\nHeNe\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {r}\n");
    let he_with_ghost_ne = format!("2\nHe + ghost Ne\nHe 0.0 0.0 0.0\n@Ne 0.0 0.0 {r}\n");
    let ne_with_ghost_he = format!("2\nghost He + Ne\n@He 0.0 0.0 0.0\nNe 0.0 0.0 {r}\n");
    (dimer, he_with_ghost_ne, ne_with_ghost_he)
}

/// Build the He-fragment weight matrix W^He on the driver's (99,302) grid for
/// the real HeNe dimer geometry.
fn w_he(r: f64) -> Array2<f64> {
    let (dimer, _, _) = geoms(r);
    let mol = Molecule::parse_xyz(&dimer, 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let grid = build_atomic_grid(
        &mol,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    build_weight_matrix(&mol, &grid, &chi, &[0])
}

fn promolecule_table() {
    println!("\n=== (b) Promolecule Becke populations on He, def2-SVP, (99,302) grid ===");
    println!("Fragments converged in the FULL dimer AO basis (ghost partner).");
    println!("Diabat A = He(+) + Ne      [hole on He]");
    println!("Diabat B = He   + Ne(+)    [hole on Ne]");
    println!();
    println!(
        "{:>6} {:>12} {:>12} {:>14} {:>14} {:>12} {:>14} {:>12}",
        "R/A",
        "N_He^A",
        "N_He^B",
        "N_He^unconstr",
        "N_He^B - 2",
        "sum_check",
        "E_unconstr",
        "dE_promol/eV"
    );

    let cf = cfg();
    for &r in &[2.0_f64, 2.5, 3.0, 4.0, 6.0] {
        let (dimer, he_g, ne_g) = geoms(r);
        let w = w_he(r);

        // Fragment densities, all in the same (He,Ne) dimer AO basis.
        // He neutral (2e, singlet) and He+ (1e, doublet) with ghost Ne.
        let (e_he0, he0_a, he0_b) = uhf(&he_g, 0, 1, &cf);
        let (e_hep, hep_a, hep_b) = uhf(&he_g, 1, 2, &cf);
        // Ne neutral (10e, singlet) and Ne+ (9e, doublet) with ghost He.
        let (e_ne0, ne0_a, ne0_b) = uhf(&ne_g, 0, 1, &cf);
        let (e_nep, nep_a, nep_b) = uhf(&ne_g, 1, 2, &cf);
        // Promolecule (non-interacting, counterpoise-corrected) diabat gap:
        // E_A - E_B = [E(He+)+E(Ne)] - [E(He)+E(Ne+)], all in the dimer basis.
        // This is the dIP anchor evaluated in the DIMER basis, so at large R it
        // must approach the isolated-atom dIP of 3.6305 eV.
        let de_promol = ((e_hep + e_ne0) - (e_he0 + e_nep)) * HA2EV;

        // Promolecule A = He+ + Ne ; B = He + Ne+.
        let da_a = &hep_a + &ne0_a;
        let da_b = &hep_b + &ne0_b;
        let db_a = &he0_a + &nep_a;
        let db_b = &he0_b + &nep_b;

        let n_a = population(&w, &da_a, &da_b, &SpinChannel::Total);
        let n_b = population(&w, &db_a, &db_b, &SpinChannel::Total);

        // The unconstrained SCF of the real HeNe+ cation, for comparison.
        // NON-FATAL: this one can stall, and when it does that is a finding
        // about the unconstrained state, not a reason to lose the table.
        let (n_unc, e_unc) = match try_uhf(&dimer, 1, 2, &cf) {
            Ok((e, dim_a, dim_b)) => (
                population(&w, &dim_a, &dim_b, &SpinChannel::Total),
                format!("{e:.6}"),
            ),
            Err(why) => {
                eprintln!("  [R={r:.1}] unconstrained HeNe+ UHF did NOT converge: {why}");
                (f64::NAN, "STALLED".to_string())
            }
        };

        // Sanity: total electrons in promolecule B should be 11 (He 2 + Ne+ 9).
        // We cannot check that with W^He alone, so report N_He^B + N_Ne^B via
        // the complementary fragment weight as a closure check.
        let w_ne = {
            let mol = Molecule::parse_xyz(&dimer, 0, 1).unwrap();
            let bs = basis::bundled("def2-svp").unwrap();
            let grid = build_atomic_grid(
                &mol,
                &AtomicGridConfig {
                    n_radial: 99,
                    n_angular: 302,
                    ..Default::default()
                },
            );
            let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
            let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
            build_weight_matrix(&mol, &grid, &chi, &[1])
        };
        let n_ne_b = population(&w_ne, &db_a, &db_b, &SpinChannel::Total);
        let sum_b = n_b + n_ne_b;

        println!(
            "{r:>6.1} {n_a:>12.6} {n_b:>12.6} {n_unc:>14.6} {:>14.6} {sum_b:>12.6} {e_unc:>14} {de_promol:>12.4}",
            n_b - 2.0
        );
    }
    println!("\n(sum_check = N_He^B + N_Ne^B must equal 11.000000 = the electron count of");
    println!(" promolecule B (He 2e + Ne+ 9e); it is a grid-completeness check on the");
    println!(" Becke partition, NOT an independent physics result.)");
}

/// Finite-field static polarizability of an isolated atom, def2-SVP.
/// alpha_zz = -d^2 E / dF^2, central 5-point (well, 3-point with two steps for
/// a Richardson check).
fn polarizability(symbol: &str) {
    let xyz = format!("1\n{symbol}\n{symbol} 0.0 0.0 0.0\n");
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let energy_at = |f: f64| -> f64 {
        let mut cf = cfg();
        if f != 0.0 {
            cf.external_potential = Some(ExternalPotential {
                field: Some([0.0, 0.0, f]),
                ..Default::default()
            });
        }
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cf).unwrap();
        assert!(res.converged, "{symbol} FF={f} not converged");
        res.energy
    };

    let e0 = energy_at(0.0);
    println!("\n  {symbol}: E(F=0) = {e0:.12} Ha");
    let mut prev: Option<f64> = None;
    for &h in &[0.01_f64, 0.005, 0.0025, 0.00125] {
        let ep = energy_at(h);
        let em = energy_at(-h);
        // alpha = -(E(+h) - 2E(0) + E(-h)) / h^2
        let alpha = -(ep - 2.0 * e0 + em) / (h * h);
        let delta = prev.map(|p: f64| alpha - p);
        match delta {
            Some(d) => println!(
                "    h={h:<8} alpha_zz = {alpha:.8} a.u.   (change vs previous h: {d:+.2e})"
            ),
            None => println!("    h={h:<8} alpha_zz = {alpha:.8} a.u."),
        }
        prev = Some(alpha);
    }
}

fn main() {
    println!("HeNe+ lane probe — def2-SVP, UHF, OPENBLAS_NUM_THREADS should be 1");

    promolecule_table();

    println!("\n=== (c) Finite-field static polarizabilities (def2-SVP, UHF) ===");
    println!("Spherical atoms, so alpha_zz = alpha_iso.");
    println!("UNIT WARNING: the experimental He polarizability is 1.384 a.u. = 0.205 A^3.");
    println!("  The figure '0.205' is the ANGSTROM-CUBED value, not atomic units; Ne's");
    println!("  2.67 a.u. = 0.396 A^3. Mixing the two (0.205 vs 2.67 as if both a.u.)");
    println!("  inflates the He-Ne differential by ~6.75x on the He term.");
    println!("Reference (a.u.): alpha_He = 1.3838, alpha_Ne = 2.6693.");
    for s in ["He", "Ne"] {
        polarizability(s);
    }

    // Ferric's own def2-SVP values, from the h -> 0 limit printed above.
    let a_he = 0.443_152_26_f64;
    let a_ne = 0.629_333_63_f64;
    println!("\n=== ion-induced-dipole differential from ferric's OWN def2-SVP alphas ===");
    println!(
        "  alpha_He = {a_he:.6} a.u.   alpha_Ne = {a_ne:.6} a.u.   diff = {:.6}",
        a_ne - a_he
    );
    println!("  E_pol(X+ ... Y) = -alpha_Y / (2 R^4); differential between diabats:");
    println!("    dE_pol = (alpha_Ne - alpha_He) / (2 R^4)");
    println!(
        "{:>8} {:>16} {:>16}",
        "R/A", "dE_pol(ferric)", "dE_pol(exp alphas)"
    );
    const ANG2BOHR: f64 = 1.889_726_124_565_062;
    for &r in &[2.0_f64, 2.5, 3.0, 4.0, 6.0] {
        let rb = r * ANG2BOHR;
        let d_ferric = (a_ne - a_he) / (2.0 * rb.powi(4)) * HA2EV;
        let d_exp = (2.6693 - 1.3838) / (2.0 * rb.powi(4)) * HA2EV;
        println!("{r:>8.1} {d_ferric:>13.4} eV {d_exp:>13.4} eV");
    }
}
