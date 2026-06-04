//! Many-Body Dispersion (MBD@TS): coupled-dipole screening of the TS per-atom
//! polarizabilities. See docs/superpowers/specs/2026-06-04-mbd-screened-c6-design.md.

use crate::dispersion::free_atom_ref::ts_free_atom;

/// Per-atom TS parameters: (α_eff, ω_A) for each atom.
///
/// α_eff = (volume ratio) · α_free; C6_eff = ratio²·C6_free; ω_A = (4/3)C6_eff/α_eff².
/// For Z outside the table, falls back to the static isotropic α with the H
/// London frequency (matches `ts_dynamic_polarizability`'s fallback).
pub fn ts_atom_params(
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
) -> Vec<(f64, f64)> {
    z.iter()
        .enumerate()
        .map(|(a, &za)| {
            let st = alpha_static[a];
            let st_iso = (st[0][0] + st[1][1] + st[2][2]) / 3.0;
            let (alpha_eff, c6_eff) = match ts_free_atom(za) {
                Some((alpha_free, c6_free, _)) => {
                    let r = vol_ratio[a];
                    (r * alpha_free, r * r * c6_free)
                }
                None => {
                    let (af_h, c6_h, _) = ts_free_atom(1).unwrap();
                    let omega_h = (4.0 / 3.0) * c6_h / (af_h * af_h);
                    let a_iso = st_iso.max(1e-6);
                    (a_iso, 0.75 * a_iso * a_iso * omega_h)
                }
            };
            let alpha_eff = alpha_eff.max(1e-8);
            let omega_a = (4.0 / 3.0) * c6_eff / (alpha_eff * alpha_eff);
            (alpha_eff, omega_a)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_atom_params_free_atom_reproduces_table() {
        // Carbon at ratio=1: α_eff = α_free = 12.0, ω_A = (4/3)·46.6/12² = 0.4315.
        let st = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
        let p = ts_atom_params(&[6], &[1.0], &[st]);
        assert!((p[0].0 - 12.0).abs() < 1e-9, "α_eff = {}", p[0].0);
        let omega_expected = (4.0 / 3.0) * 46.6 / (12.0 * 12.0);
        assert!((p[0].1 - omega_expected).abs() < 1e-9, "ω_A = {}", p[0].1);
    }
}
