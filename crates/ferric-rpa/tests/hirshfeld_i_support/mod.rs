//! Hirshfeld-I driver of `validation_properties_hirshfeld.rs` (weekly only;
//! no per-commit test uses this module): water's Hirshfeld-I charges with
//! ad-hoc same-basis NEUTRAL + ION proatoms (atomic SCF per element and
//! integer charge, spherically averaged on a shared radial grid). The
//! per-commit `properties_hirshfeld.rs::h2o_adhoc_neutral_hirshfeld_charges`
//! calls the same library `hirshfeld_i_charges` with its own NEUTRAL-only
//! proatom closure, so the ion-proatom charge iteration runs weekly only.
//! A cc-pVDZ per-commit canary of this driver was measured and not added
//! (2026-10-30, loaded box): q = [-0.9052, +0.4526, +0.4526] in 48 s at
//! `n_radii` = 200 and 94 s at 600 (identical charges), over the ~30 s
//! per-commit budget.

use ferric_core::basis;
use ferric_core::elements::z_to_symbol;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{hirshfeld_i_charges, spherically_averaged_proatom, RadialProatom};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Hirshfeld-I charges `[q_O, q_H, q_H]` of `water.xyz` in `basis_name`, with
/// proatoms on the radial grid r_k = 0.05·k Bohr, k = 1..=`n_radii`.
pub fn water_hirshfeld_i_charges(basis_name: &str, n_radii: usize) -> Vec<f64> {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs_bs = basis::bundled(basis_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let ctx0 = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx0, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let ctx = ParallelContext::default();
    let radii: Vec<f64> = (1..=n_radii).map(|k| k as f64 * 0.05).collect();
    // Ground-state multiplicity (2S+1) of an atom/ion with N electrons, from the
    // aufbau filling + Hund's rule on the open subshell (good for N ≤ 18).
    let mult_for_n = |n: i32| -> usize {
        if n <= 0 {
            return 1;
        }
        // (subshell capacity) in aufbau order through 3p.
        let shells = [2, 2, 6, 2, 6]; // 1s,2s,2p,3s,3p
        let mut rem = n;
        let mut unpaired = 0i32;
        for &cap in &shells {
            let inshell = rem.min(cap);
            rem -= inshell;
            // unpaired in this subshell by Hund: cap/2 orbitals, fill singly first.
            let norb = cap / 2;
            unpaired = if inshell <= norb {
                inshell
            } else {
                cap - inshell
            };
            if rem == 0 {
                break;
            }
        }
        (unpaired.unsigned_abs() as usize) + 1
    };

    // Proatom closure: run an atomic SCF for element z at integer charge qi
    // (N = z − qi electrons) in the molecule's basis, spherically average to a
    // radial proatom. Neutral + ions for full Hirshfeld-I charge interpolation.
    let bs = obs_bs.clone();
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        let n_elec = z - qi;
        if n_elec <= 0 {
            // Bare nucleus (e.g. H+): zero electron density.
            return RadialProatom::new(radii.clone(), vec![0.0; radii.len()]).ok();
        }
        let mult = mult_for_n(n_elec);
        let sym = z_to_symbol(z).unwrap_or("X");
        let xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&xyz, qi, mult).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut cfg = RhfConfig::default();
        // Closed-shell singlet → RHF; open-shell → UHF with MOM. Anions in a
        // non-augmented basis may not bind the extra electron; if SCF fails the
        // `?`/.ok() returns None and hirshfeld_i falls back to the nearest state.
        let dens = if mult == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &cfg)
                .ok()?
                .density_r()
                .to_owned()
        } else {
            cfg.mom_after_iter = 5;
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &cfg)
                .ok()?
                .density_total()
                .to_owned()
        };
        spherically_averaged_proatom(z, &bs, &dens, &radii).ok()
    };
    hirshfeld_i_charges(&mol, &obs_bs, rhf.density_r(), &proatom).unwrap()
}
