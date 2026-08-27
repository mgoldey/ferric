//! `SiteBasis`: one Gaussian per MM site, packaged as a [`PreparedBasis`] so
//! the existing 3-centre engines (`Engine::new_3center`/`compute_eri3`,
//! `Engine::new_3center_deriv`/`compute_eri3_deriv`) can be reused to compute
//! the Gaussian-smeared-charge potential and its analytic gradient without any
//! new C++.
//!
//! # The trick
//!
//! `⟨μ|erf(√ζ|r−R_i|)/|r−R_i||ν⟩ = (μν|g_i) / ∫g_i` — a 3-centre Coulomb
//! integral against a single normalised s-Gaussian `g_i` centred at the site,
//! divided by that Gaussian's integral (because libint2 normalises every
//! shell to unit self-overlap, not unit integral). Each MM site becomes one
//! shell of a synthetic auxiliary basis on a synthetic `Molecule` of ghost
//! pseudo-atoms — `PreparedBasis::new` already supports this: it looks up
//! shells by the raw `atom.z` (so a pseudo-element `z = 1000 + k` finds the
//! shell we registered for it) but computes the nuclear-attraction point
//! charge from `effective_z()`, which is forced to zero by `ghost: true`. So
//! the pseudo-atoms carry basis functions but contribute no nuclear charge —
//! exactly what a "fake auxiliary basis" needs.
//!
//! # Artifact hypothesis
//!
//! If `norm_int` (the assumed closed form `∫g = 2^{3/4} π^{3/4} ζ^{-3/4}`) is
//! wrong, EVERY use of `SiteBasis` is wrong by that same multiplicative
//! constant, uniformly across sites, geometries, and molecules — a
//! systematic error that would NOT show up as noise or as a
//! geometry-dependent artifact. It would show up as: the tight-ζ site
//! potential disagreeing with the point-charge nuclear-attraction block by a
//! constant ratio (not by a geometry-dependent residual). That is exactly
//! what `tight_site_gaussian_reproduces_point_charge_attraction` checks —
//! it does not trust the formula, it MEASURES the ratio at ζ=1e6 (where a
//! Gaussian charge distribution is numerically indistinguishable from a point
//! charge) against the independent point-charge nuclear-attraction path
//! (`oneelectron::hcore_with_external` / `PointCharge`), which shares no code
//! with `SiteBasis`. If instead the shell-grouping / pseudo-element wiring
//! were wrong (e.g. `site_shell` pointed at the wrong shell, or two distinct
//! widths collapsed into one group), the symptom would be different: either
//! a hard PreparedBasis error, or a WRONG BUT CONSISTENT mapping that
//! `distinct_widths_form_distinct_groups` catches by checking group identity
//! and count directly, independent of the integral values.

use crate::basis_bridge::PreparedBasis;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::FerricError;
use std::collections::HashMap;

/// Base pseudo-atomic number for site-basis pseudo-elements. Chosen well
/// above any real or ECP-extended element (`z=1000+k` cannot collide with a
/// real basis-set entry, which tops out in the hundreds at most).
const SITE_Z_BASE: i32 = 1000;

/// A synthetic auxiliary basis: one s- (or p-, etc.) Gaussian per MM site,
/// grouped by distinct exponent `ζ` into pseudo-elements so libint2 can treat
/// them as an ordinary (if fictitious) basis set.
pub struct SiteBasis {
    /// The prepared basis: one ghost pseudo-atom per site, `nshells() ==
    /// sites.len()` (each site gets its own shell, even when two sites share
    /// a `ζ` and therefore a pseudo-element).
    pub prep: PreparedBasis,
    /// `site_shell[i]` is the shell index in `prep` for site `i`.
    pub site_shell: Vec<usize>,
    /// `zeta[i]` is the Gaussian exponent of site `i` (Bohr⁻²), copied
    /// straight from the input.
    pub zeta: Vec<f64>,
    /// `norm_int[i]` is `∫ g_i(r) dr` for the libint2-normalised (unit
    /// self-overlap) shell at site `i`: `(2^{3/4} π^{3/4}) ζ_i^{-3/4}` for an
    /// s-shell (`l=0`). Dividing a raw `(μν|g_i)` 3-centre integral by this
    /// turns it into the erf-Coulomb potential integral `⟨μ|erf(√ζ r)/r|ν⟩`.
    pub norm_int: Vec<f64>,
}

impl SiteBasis {
    /// Build a site basis from `(x, y, z, zeta)` tuples (Bohr / Bohr⁻²).
    /// `l` is the shared angular momentum of every site shell (`0` for
    /// monopole/charge potentials, `1` for the dipole-potential sites Lane B
    /// will add).
    ///
    /// Errors if `sites` is empty, or if any `zeta` is non-finite or `<= 0`
    /// (a non-positive exponent is not a valid Gaussian and would either
    /// diverge or silently degenerate).
    pub fn new(sites: &[[f64; 4]], l: usize) -> Result<Self, FerricError> {
        if sites.is_empty() {
            return Err(FerricError::General(
                "SiteBasis::new: sites must be non-empty".to_string(),
            ));
        }
        for (i, s) in sites.iter().enumerate() {
            let zeta = s[3];
            if !zeta.is_finite() || zeta <= 0.0 {
                return Err(FerricError::General(format!(
                    "SiteBasis::new: site {i} has non-positive or non-finite zeta = {zeta}"
                )));
            }
        }

        // Group sites by distinct zeta (bitwise equality — sites constructed
        // from the same source value, e.g. the same `width`, compare equal
        // exactly; this is a grouping optimization only, never load-bearing
        // for correctness, since every site gets its own shell regardless).
        let mut group_of_bits: HashMap<u64, i32> = HashMap::new();
        let mut next_group: i32 = 0;
        let mut pseudo_z: Vec<i32> = Vec::with_capacity(sites.len());
        for s in sites {
            let bits = s[3].to_bits();
            let g = *group_of_bits.entry(bits).or_insert_with(|| {
                let g = next_group;
                next_group += 1;
                g
            });
            pseudo_z.push(SITE_Z_BASE + g);
        }

        // One BasisSet shell per DISTINCT zeta group (not per site — the
        // PreparedBasis shell-lookup is per pseudo-element, so all sites
        // sharing a zeta share a shell TEMPLATE; each becomes its own SHELL
        // INSTANCE once attached to its own pseudo-atom, since `for_element`
        // is looked up once per atom in `PreparedBasis::new`).
        let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
        for (i, s) in sites.iter().enumerate() {
            let z = pseudo_z[i];
            shells.entry(z).or_insert_with(|| {
                vec![Shell {
                    l: l as i32,
                    pure: true,
                    exponents: vec![s[3]],
                    coefficients: vec![1.0],
                }]
            });
        }

        let bs = BasisSet {
            name: "site-basis".to_string(),
            shells,
            ecps: HashMap::new(),
        };

        let atoms: Vec<Atom> = sites
            .iter()
            .zip(pseudo_z.iter())
            .map(|(s, &z)| Atom {
                symbol: "X".to_string(),
                z,
                x: s[0],
                y: s[1],
                zpos: s[2],
                ghost: true,
                n_core_ecp: 0,
            })
            .collect();

        let mol = Molecule {
            atoms,
            charge: 0,
            multiplicity: 1,
        };

        let prep = PreparedBasis::new(&mol, &bs)?;

        // Each pseudo-atom has exactly one shell (the template above), so
        // atom i's shell is shell i, in order — but derive it from
        // shell_to_atom rather than assuming, so a future change to
        // PreparedBasis's shell ordering cannot silently desync this.
        let shell_to_atom = prep.shell_to_atom();
        let mut site_shell = vec![usize::MAX; sites.len()];
        for (sh, &atom_idx) in shell_to_atom.iter().enumerate() {
            debug_assert_eq!(site_shell[atom_idx], usize::MAX, "site {atom_idx} has more than one shell");
            site_shell[atom_idx] = sh;
        }
        assert!(
            site_shell.iter().all(|&s| s != usize::MAX),
            "SiteBasis::new: internal error, some site has no shell"
        );

        let zeta: Vec<f64> = sites.iter().map(|s| s[3]).collect();
        let norm_int: Vec<f64> = zeta.iter().map(|&z| norm_int_s_shell(z)).collect();

        Ok(SiteBasis { prep, site_shell, zeta, norm_int })
    }
}

/// `∫ g(r) dr` for a libint2-normalised (unit self-overlap) primitive
/// s-Gaussian of exponent `ζ`: `g(r) = N exp(-ζ r²)` with `N` fixed by
/// `∫ g² dr = 1`, i.e. `N = (2ζ/π)^{3/4}`. Then
/// `∫ g dr = N (π/ζ)^{3/2} = (2ζ/π)^{3/4} (π/ζ)^{3/2} = 2^{3/4} π^{3/4}
/// ζ^{-3/4}`.
///
/// This closed form is not trusted blindly — see the module doc's artifact
/// hypothesis and `tight_site_gaussian_reproduces_point_charge_attraction`,
/// which measures it against an independent code path.
fn norm_int_s_shell(zeta: f64) -> f64 {
    2f64.powf(0.75) * std::f64::consts::PI.powf(0.75) * zeta.powf(-0.75)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};

    fn water_prep() -> PreparedBasis {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    /// EXACTNESS ANCHOR (pins `norm_int`): at a very tight exponent (ζ=1e6,
    /// width ≈ 1e-3 Bohr), the Gaussian-smeared potential is numerically a
    /// point charge. Build `V_site[μν] = -q (μν|g)/norm_int` by hand via the
    /// SiteBasis machinery and compare against `oneelectron::hcore_with_external`
    /// minus the plain `hcore` — an INDEPENDENT code path (point-charge
    /// nuclear attraction via libint2's native `OP_NUCLEAR`, not the 3-centre
    /// ERI engine at all) that shares no arithmetic with SiteBasis. If
    /// `norm_int`'s closed form were off by any constant factor, this test
    /// would fail by that same factor — not by noise.
    #[test]
    fn tight_site_gaussian_reproduces_point_charge_attraction() {
        let obs = water_prep();
        let q = 1.0;
        let site_xyz = [0.0, 0.0, -6.0];

        // Independent path: point-charge nuclear attraction.
        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q, x: site_xyz[0], y: site_xyz[1], z: site_xyz[2] }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let h_ext = crate::oneelectron::hcore_with_external(&obs, Some(&ext)).unwrap();
        let h_plain = crate::oneelectron::hcore(&obs);
        let v_point = &h_ext - &h_plain;

        // SiteBasis path: tight Gaussian at the same site.
        let zeta = 1e6;
        let site = SiteBasis::new(&[[site_xyz[0], site_xyz[1], site_xyz[2], zeta]], 0).unwrap();
        let mut eng = Engine::new_3center(Operator::coulomb(), &obs, &site.prep, 1e-14).unwrap();

        let nbasis = obs.nbasis();
        let mut v_site = ndarray::Array2::<f64>::zeros((nbasis, nbasis));
        let offs = obs.shell_offsets();
        let dims = obs.shell_dims();
        let sh_p = site.site_shell[0];
        let norm = site.norm_int[0];
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                if let Some(block) = eng.compute_eri3(&obs, &site.prep, sh_p, s1, s2) {
                    let n1 = dims[s1];
                    let n2 = dims[s2];
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let mu = offs[s1] + i;
                            let nu = offs[s2] + j;
                            // (P|mu nu) is stored P-major: index = (p*n1 + i)*n2 + j
                            // with a single P function (s-shell), p=0.
                            let val = block[i * n2 + j];
                            v_site[(mu, nu)] += -q * val / norm;
                        }
                    }
                }
            }
        }

        let max_diff = (&v_site - &v_point).iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            max_diff < 1e-9,
            "tight-zeta SiteBasis attraction disagrees with point-charge attraction by {max_diff:.3e} \
             (this pins the ∫g normalisation constant — see module doc's artifact hypothesis)"
        );
    }

    /// Two sites with distinct ζ must land in two distinct pseudo-elements
    /// (two shells with different exponents), and `site_shell` must map each
    /// site to ITS OWN shell — a construction bug here (e.g. collapsing both
    /// sites onto one shell, or aliasing the exponent) would not show up as a
    /// small numeric residual; it would show up as a wrong shell count or a
    /// wrong exponent readback, which is what this test checks directly
    /// rather than via any integral.
    #[test]
    fn distinct_widths_form_distinct_groups() {
        let sites = [[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 4.0]];
        let sb = SiteBasis::new(&sites, 0).unwrap();
        assert_eq!(sb.prep.nshells(), 2, "two distinct zetas must produce two shells");
        assert_eq!(sb.site_shell.len(), 2);
        assert_ne!(sb.site_shell[0], sb.site_shell[1], "distinct sites must map to distinct shells");
        assert_eq!(sb.zeta, vec![1.0, 4.0]);
        // norm_int must track each site's OWN zeta, not get shared/aliased.
        assert!((sb.norm_int[0] - norm_int_s_shell(1.0)).abs() < 1e-14);
        assert!((sb.norm_int[1] - norm_int_s_shell(4.0)).abs() < 1e-14);
        assert!((sb.norm_int[0] - sb.norm_int[1]).abs() > 1e-3, "distinct zetas must give distinct norm_int");
    }

    /// Two sites sharing the SAME zeta land on the same pseudo-element (basis
    /// grouping optimisation) but still get their OWN shell instance (one per
    /// pseudo-atom), so they remain independently addressable sites.
    #[test]
    fn shared_zeta_sites_still_have_independent_shells() {
        let sites = [[0.0, 0.0, 0.0, 2.0], [3.0, 0.0, 0.0, 2.0]];
        let sb = SiteBasis::new(&sites, 0).unwrap();
        assert_eq!(sb.prep.nshells(), 2);
        assert_ne!(sb.site_shell[0], sb.site_shell[1]);
        assert_eq!(sb.zeta, vec![2.0, 2.0]);
    }

    #[test]
    fn rejects_empty_sites() {
        assert!(SiteBasis::new(&[], 0).is_err());
    }

    #[test]
    fn rejects_nonpositive_zeta() {
        assert!(SiteBasis::new(&[[0.0, 0.0, 0.0, 0.0]], 0).is_err());
        assert!(SiteBasis::new(&[[0.0, 0.0, 0.0, -1.0]], 0).is_err());
        assert!(SiteBasis::new(&[[0.0, 0.0, 0.0, f64::NAN]], 0).is_err());
    }
}
