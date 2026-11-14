//! Thole polarization energy of a `[qmmm]` run with `polarizabilities_angstrom3`.
//!
//! The induced dipoles are solved inside every SCF iteration and their energy
//! is part of the SCF energy; the SCF result carries the dipoles but not the
//! energy term. [`report`] recomputes it with one `induce` call at the
//! converged density, the same call Python `run_qmmm` makes for
//! `QmmmResult.e_pol`, and prints and logs it.

use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::RhfConfig;

/// The polarization energy and induced dipoles of a converged polarizable SCF.
pub struct Polarization {
    scf_energy: f64,
    e_pol: f64,
    n_sites: usize,
    thole_a: Option<f64>,
    dipoles: Vec<Vec<f64>>,
}

/// Evaluate the polarization energy of `result`; `None` unless
/// `config.polarizable` is set (a failed evaluation warns and is `None`).
pub fn evaluate(
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    result: &ScfResult,
) -> Option<Polarization> {
    let pol = config.polarizable.as_ref()?;
    let site_xyz: Vec<[f64; 4]> = pol
        .sites
        .iter()
        .map(|s| [s.x, s.y, s.z, pol.dipole_zeta])
        .collect();
    let outcome = ferric_integrals::site_basis::SiteBasis::new(&site_xyz, 1).and_then(|sb| {
        ferric_scf::polarizable::induce(
            mol,
            prep,
            config.external_potential.as_ref(),
            pol,
            &sb,
            result.density_total(),
        )
    });
    match outcome {
        Ok(ir) => Some(Polarization {
            scf_energy: result.energy,
            e_pol: ir.e_pol,
            n_sites: pol.sites.len(),
            thole_a: pol.thole_a,
            dipoles: ir.dipoles.outer_iter().map(|d| d.to_vec()).collect(),
        }),
        Err(e) => {
            eprintln!("warning: could not evaluate the polarization energy: {e}");
            None
        }
    }
}

/// Print and log `polarization` (no-op for `None`, the non-polarizable run).
pub fn report(polarization: Option<Polarization>) {
    polarization.iter().for_each(Polarization::print_and_log);
}

impl Polarization {
    /// Print the energy and write a `polarization` run-log note (the SCF
    /// energy it belongs to, which includes `e_pol`, the energy term, the
    /// Thole parameter and the induced dipoles).
    fn print_and_log(&self) {
        println!(
            "  E_pol      = {:.10} Hartree (Thole polarization, {} sites; included in the \
             energy above)",
            self.e_pol, self.n_sites
        );
        if let Some(rl) = ferric_scf::runlog::log() {
            rl.note(
                "polarization",
                serde_json::json!({
                    "scf_energy": self.scf_energy,
                    "e_pol": self.e_pol,
                    "n_sites": self.n_sites,
                    "thole_a": self.thole_a,
                    "induced_dipoles": self.dipoles,
                }),
            );
        }
    }
}
