//! VV10-on-HF-density cost / screen-error harness (MP2-V post-HF half).
//!
//! usage: vv10_scaling <xyz> <basis> <n_radial> <n_angular> <spec>...
//!   spec = dense | prod | cut:R:k | tail:R:k | fast_cut:R:k | fast_tail:R:k
//!   (R in Bohr, k = cells per R; fast_* also tabulate the damping factor)
//!     dense  exact O(N^2) pair sum (no cutoff)
//!     prod   the production path (40 Bohr cell list, `vv10_energy_on_density`)
//!     cut    Vv10Screen truncation only;  tail  truncation + monopole far field
//! Prints, per spec and per damping (terfc r0=1 A linked, and undamped):
//! E_nl, active points, near/far pair evaluations, wall seconds.
//! MP2-V published VV10 parameters (b = 11.0, C = 0.0089).

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::libxc::Vv10Params;
use ferric_dft::vv10::{Vv10Damping, Vv10Screen};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{vv10_energy_on_density, vv10_energy_on_density_screened, BOHR_PER_ANG};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let xyz = std::fs::read_to_string(&a[1]).unwrap();
    let basis_name = &a[2];
    let n_radial: usize = a[3].parse().unwrap();
    let n_angular: usize = a[4].parse().unwrap();
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let t0 = Instant::now();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-7,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "# {} atoms, nbf basis {}, E_HF={:.8}, SCF {:.1}s",
        mol.atoms.len(),
        basis_name,
        rhf.energy,
        t0.elapsed().as_secs_f64()
    );
    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let r0 = 1.0 * BOHR_PER_ANG;
    let dampings = [
        (
            "damped",
            Vv10Damping::Terfc {
                r0_bohr: r0,
                omega_bohr_inv: None,
            },
        ),
        ("undamped", Vv10Damping::None),
    ];
    let grid = AtomicGridConfig {
        n_radial,
        n_angular,
        prune: None,
    };
    let d = rhf.density_total();
    for spec in &a[5..] {
        for (dn, damp) in &dampings {
            let t = Instant::now();
            let (e, n_grid, n_act, near, far) = if spec == "prod" {
                let (e, n) = vv10_energy_on_density(&mol, &bs, d, &params, *damp, &grid).unwrap();
                (e, n, 0, 0, 0)
            } else {
                let sc = if spec == "dense" {
                    None
                } else {
                    let p: Vec<&str> = spec.split(':').collect();
                    Some(Vv10Screen {
                        r_cut_bohr: p[1].parse().unwrap(),
                        near_shells: p[2].parse().unwrap(),
                        far_field: p[0].ends_with("tail"),
                        fast_damping: p[0].starts_with("fast_"),
                    })
                };
                let r = vv10_energy_on_density_screened(&mol, &bs, d, &params, *damp, &grid, sc)
                    .unwrap();
                (r.e_nl, r.n_grid, r.n_active, r.pairs_near, r.pairs_far)
            };
            println!(
                "{} natoms={} grid={}x{} spec={} {} n_grid={} n_active={} near={} far={} E_nl={:.10} t={:.2}s",
                a[1].rsplit('/').next().unwrap(),
                mol.atoms.len(),
                n_radial,
                n_angular,
                spec,
                dn,
                n_grid,
                n_act,
                near,
                far,
                e,
                t.elapsed().as_secs_f64()
            );
        }
    }
}
