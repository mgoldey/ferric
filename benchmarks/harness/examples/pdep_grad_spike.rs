//! C-grad magnitude spike: does PDEP truncation perturb nuclear gradients?
//!
//! For H2O/cc-pVDZ + cc-pVDZ-RI we compute central-difference forces
//! (h = 5e-4 Bohr) at trunc_thresh ∈ {0, 1e-6, 1e-4, 1e-3, 1e-2}, then
//! compare each truncated force vector to the dense (trunc=0) baseline.
//!
//! Run: cargo run --release -p ferric-rpa --example pdep_grad_spike

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA_TO_KCAL: f64 = 627.509;

fn rpa_energy(mol: &Molecule, trunc_thresh: f64) -> (f64, usize) {
    let ctx = ParallelContext::default();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 40,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh,
        davidson_conv_thresh: 1e-10,
        ..Default::default()
    };

    let r = run_pdep_rpa(mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    // Total energy = E_HF + E_RPA. For finite differences only the geometry-dependent
    // E_RPA differences matter when comparing across thresholds at the same geometry,
    // but we include E_HF because the gradient of interest is dE_total/dR.
    (rhf.energy + r.e_rpa, r.n_eigenpotentials)
}

fn perturb(mol: &Molecule, atom: usize, dim: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match dim {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        2 => m.atoms[atom].zpos += h,
        _ => unreachable!(),
    }
    m
}

fn perturb_vec(mol: &Molecule, dirs: &[(usize, [f64; 3])], h: f64) -> Molecule {
    let mut m = mol.clone();
    for (atom, d) in dirs {
        m.atoms[*atom].x += h * d[0];
        m.atoms[*atom].y += h * d[1];
        m.atoms[*atom].zpos += h * d[2];
    }
    m
}

fn force_vector(mol: &Molecule, thresh: f64, h: f64) -> (Vec<f64>, usize) {
    let natoms = mol.atoms.len();
    let mut forces = vec![0.0; 3 * natoms];
    let mut n_modes = 0;
    for a in 0..natoms {
        for d in 0..3 {
            let mp = perturb(mol, a, d, h);
            let mm = perturb(mol, a, d, -h);
            let (ep, nm) = rpa_energy(&mp, thresh);
            let (em, _) = rpa_energy(&mm, thresh);
            forces[3 * a + d] = -(ep - em) / (2.0 * h);
            n_modes = nm;
            eprintln!(
                "  thresh={:.0e}  atom={} dim={}  E+={:.10} E-={:.10}  F={:+.6e}",
                thresh, a, d, ep, em, forces[3 * a + d]
            );
        }
    }
    (forces, n_modes)
}

fn main() {
    let mol = Molecule::load_xyz("testdata/molecules/water.xyz")
        .or_else(|_| Molecule::load_xyz("../../testdata/molecules/water.xyz"))
        .expect("load water.xyz");

    println!("# PDEP truncation force-magnitude spike");
    println!("Geometry: H2O (water.xyz), basis cc-pVDZ / RI cc-pVDZ-RI");
    println!("Displacement h = 5e-4 Bohr; central differences");
    println!("Forces in Ha/Bohr; reported max errors vs trunc=0.0 baseline.\n");

    let h = 5e-4;
    let thresholds = [0.0f64, 1e-6, 1e-4, 1e-3, 1e-2];

    let mut results: Vec<(f64, Vec<f64>, usize)> = Vec::new();
    for &t in &thresholds {
        eprintln!("\n== trunc_thresh = {:.0e} ==", t);
        let (f, nm) = force_vector(&mol, t, h);
        results.push((t, f, nm));
    }

    let baseline = results[0].1.clone();

    println!("\n## Full force table (Ha/Bohr)");
    println!("Components ordered (atom0_x, atom0_y, atom0_z, atom1_x, ..., atom2_z):");
    for (t, f, _nm) in &results {
        print!("  trunc={:.0e}: [", t);
        for (i, v) in f.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.6e}", v);
        }
        println!("]");
    }

    println!("\n## Per-component force errors vs baseline (trunc=0)");
    println!(
        "{:<12} | {:>7} | {:>20} | {:>26} | {:>14}",
        "trunc_thresh", "n_modes", "max |ΔF| (Ha/Bohr)", "max |ΔF| (kcal/mol/Bohr)", "max rel err"
    );
    println!("{}", "-".repeat(95));
    let mut prod_err_kcal: Option<f64> = None;
    for (t, f, nm) in &results {
        if *t == 0.0 {
            println!(
                "{:<12.0e} | {:>7} | {:>20} | {:>26} | {:>14}",
                t, nm, "0.0 (baseline)", "0.0", "0%"
            );
            continue;
        }
        let max_abs = f
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_rel = f
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| if b.abs() > 1e-10 { (a - b).abs() / b.abs() } else { 0.0 })
            .fold(0.0f64, f64::max);
        let max_kcal = max_abs * HA_TO_KCAL;
        if (*t - 1e-4).abs() < 1e-12 {
            prod_err_kcal = Some(max_kcal);
        }
        println!(
            "{:<12.0e} | {:>7} | {:>20.6e} | {:>26.6e} | {:>13.4}%",
            t,
            nm,
            max_abs,
            max_kcal,
            100.0 * max_rel
        );
    }

    // O-H stretch direction probe (chemically relevant).
    // OH bond is along (atom1 - atom0). Use unit vector. Displace atom1 along that direction.
    let a0 = &mol.atoms[0];
    let a1 = &mol.atoms[1];
    let dx = a1.x - a0.x;
    let dy = a1.y - a0.y;
    let dz = a1.zpos - a0.zpos;
    let nrm = (dx * dx + dy * dy + dz * dz).sqrt();
    let u = [dx / nrm, dy / nrm, dz / nrm];

    println!("\n## O-H stretch probe");
    println!("Unit vector (atom1 along OH bond): [{:+.4}, {:+.4}, {:+.4}]", u[0], u[1], u[2]);
    println!("Projecting -dE/ds along that direction (s = displacement of atom1).");

    let mut stretch_forces: Vec<(f64, f64, usize)> = Vec::new();
    for &t in &thresholds {
        let mp = perturb_vec(&mol, &[(1usize, u)], h);
        let mm = perturb_vec(&mol, &[(1usize, u)], -h);
        let (ep, nm) = rpa_energy(&mp, t);
        let (em, _) = rpa_energy(&mm, t);
        let fstretch = -(ep - em) / (2.0 * h);
        eprintln!(
            "  stretch thresh={:.0e}  E+={:.10} E-={:.10}  F_s={:+.6e}",
            t, ep, em, fstretch
        );
        stretch_forces.push((t, fstretch, nm));
    }

    let f0_stretch = stretch_forces[0].1;
    println!(
        "{:<12} | {:>7} | {:>18} | {:>24} | {:>22}",
        "trunc_thresh", "n_modes", "F_stretch (Ha/Bohr)", "ΔF_stretch (Ha/Bohr)", "ΔF (kcal/mol/Bohr)"
    );
    println!("{}", "-".repeat(95));
    for (t, f, nm) in &stretch_forces {
        let df = f - f0_stretch;
        println!(
            "{:<12.0e} | {:>7} | {:>+18.6e} | {:>+24.6e} | {:>+22.6e}",
            t,
            nm,
            f,
            df,
            df.abs() * HA_TO_KCAL
        );
    }

    // Recommendation
    println!("\n## Recommendation");
    let prod_err = prod_err_kcal.unwrap_or(f64::NAN);
    println!(
        "Based on H2O/cc-pVDZ data at production trunc_thresh = 1e-4,",
    );
    println!("max |ΔF| = {:.4e} kcal/mol/Bohr.", prod_err);
    println!("- If X < 0.01: projection-fixed gradient is sufficient.");
    println!("- If 0.01 < X < 0.05: borderline; choose based on scaling preference.");
    println!("- If X > 0.05: need full Hellmann-Feynman through the projection.");
    let verdict = if prod_err < 0.01 {
        "projection-fixed"
    } else if prod_err < 0.05 {
        "borderline"
    } else {
        "full-response"
    };
    println!("\nVERDICT: {}", verdict);
}
