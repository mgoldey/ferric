//! Emit the Hirshfeld proatom-fallback warning so a test can READ it.
//!
//! The warning goes to stderr via `eprintln!`, which an in-process test cannot
//! capture. `tests/hirshfeld_fallback_warning.rs` runs this as a subprocess and
//! asserts the text. Argument: `all` or `mixed`.
//!
//! Exists because the two cases need DIFFERENT wording and nothing was
//! checking the wording -- merging them back into one message passed the
//! charge-value assertions unchanged (mutation-verified 2026-09-20).

use ferric_core::{basis, mol::Molecule};
use ferric_rpa::properties::{hirshfeld_charges, spherically_averaged_proatom, ProatomProvider};

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    let xyz = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse");
    let bs = basis::bundled("sto-3g").expect("basis");
    let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &bs).expect("prep");
    let dens = ndarray::Array2::<f64>::eye(prep.nbasis()) * 0.2;

    match which.as_str() {
        // No provider at all: EVERY atom falls back.
        "all" => {
            hirshfeld_charges(&mol, &bs, &dens, None).expect("charges");
        }
        // A provider that answers for oxygen only: the hydrogens fall back.
        "mixed" => {
            let radii = [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0];
            let o_mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).expect("o mol");
            let o_prep =
                ferric_integrals::basis_bridge::PreparedBasis::new(&o_mol, &bs).expect("o prep");
            let o_dens = ndarray::Array2::<f64>::eye(o_prep.nbasis()) * 0.5;
            let partial = |z: i32, _q: i32| {
                if z == 1 {
                    None
                } else {
                    spherically_averaged_proatom(z, &bs, &o_dens, &radii).ok()
                }
            };
            let pp: &ProatomProvider = &partial;
            hirshfeld_charges(&mol, &bs, &dens, Some(pp)).expect("charges");
        }
        other => {
            eprintln!("usage: hirshfeld_warning_probe all|mixed (got {other:?})");
            std::process::exit(2);
        }
    }
}
