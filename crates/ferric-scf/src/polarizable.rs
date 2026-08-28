//! Thole-damped polarizable embedding: Applequist induced point dipoles on
//! fixed MM sites, self-consistent with the QM density inside the SCF loop.
//!
//! # Model
//!
//! Sites `i` carry an isotropic polarisability `alpha_i` (Bohr^3). The
//! induced dipole at each site solves
//!
//! ```text
//! mu_i = alpha_i * [ E_i^QM(D) + E_i^perm + sum_{j != i} T_ij mu_j ]
//! ```
//!
//! `E_i^QM(D)` is the field of the QM nuclei + electron density at site `i`
//! ([`crate::qmmm::electric_field_at_points`]); `E_i^perm` is the field of
//! the OTHER MM permanent charges (point or Gaussian-smeared, from
//! [`ferric_core::external_potential::ExternalPotential`]); `T_ij` is the
//! Thole-damped dipole-dipole interaction tensor:
//!
//! ```text
//! T_ij = lambda3(u_ij) * I / r_ij^3  -  3 * lambda5(u_ij) * (r_hat (x) r_hat) / r_ij^3
//! u_ij = r_ij / (alpha_i * alpha_j)^(1/6)
//! lambda3 = 1 - exp(-a*u^3)
//! lambda5 = 1 - (1 + a*u^3) * exp(-a*u^3)
//! ```
//!
//! with damping parameter `a` (default 2.1304, Thole 1981); `thole_a: None`
//! disables damping (`lambda3 = lambda5 = 1`, the bare point-dipole tensor).
//!
//! This is solved as one dense `(3N)x(3N)` linear system per SCF iteration
//! (recomputed from the CURRENT density every iteration, exactly like COSMO's
//! reaction field — see `crate::cosmo` and `crate::driver::solvent_terms`),
//! with a hard error above `max_sites_dense` sites (dense-only in this
//! phase; an iterative/CG solve is a documented follow-up, not a silent
//! degradation).
//!
//! # Energy and Fock term
//!
//! `E_pol = -1/2 * sum_i mu_i . (E_i^QM + E_i^perm)`.
//!
//! The Fock contribution is `V_ind_munu = -sum_i mu_i . <mu| grad_{R_i}
//! (1/|r-R_i|) |nu>`, with **NO** factor of 1/2: because `mu` is LINEAR in
//! the field it responds to and is held FIXED (at its just-converged value)
//! when differentiating `E_pol` with respect to the density for the Fock
//! matrix, `dE_pol/dD = -sum_i mu_i . dE_i^QM/dD` exactly — the same
//! variational structure COSMO's `E = 1/2 q.v(D)`, `F += v(D)` uses (see
//! `crate::cosmo` module doc). This was verified independently in the Python
//! prototype (`scripts/proto_polarizable_embedding.py`) via a stationarity
//! check: central FD of the fully-reconverged E_total(R) under a QM nuclear
//! displacement reproduces the SCF-consistent energy trajectory to ~1e-7
//! relative, which would NOT hold if an extra 1/2 (or a missing one) had
//! snuck into either the energy or the Fock term.
//!
//! # Integrals: the p-shell trick
//!
//! `grad_{R_i} (1/|r-R_i|) = -(r-R_i)/|r-R_i|^3`, i.e. the Fock-term
//! integral is (minus) the ELECTRIC FIELD integral of a point dipole at
//! `R_i` — exactly what a p-shell (l=1) Gaussian at `R_i` gives via the same
//! 3-centre `(mu nu|p_i)` machinery [`crate::site_basis`][ferric_integrals]
//! already built for Lane A's smeared charges (`SiteBasis::new(sites, 1)`).
//! `dipole_zeta` (default 1e4 Bohr^-2, i.e. width ~0.01 Bohr) makes this a
//! numerically-point dipole; [`crate::gradient`]'s p-shell gradient
//! machinery reuses [`ferric_integrals::engine::Engine::compute_eri3_deriv`]
//! exactly as Lane A's `smeared_charge_qm_gradient`/`smeared_site_forces` do.
//!
//! The exact scalar relating the raw p-shell integral `(mu nu|p_i)` to the
//! physical field integral `<mu|(r-R_i)/|r-R_i|^3|nu>` was NOT assumed — it
//! was MEASURED against an independent code path (the nuclear-attraction
//! derivative-with-respect-to-an-extra-point-charge's-OWN-position, via
//! `compute_1e_deriv_block_n`) in
//! `crates/ferric-scf/tests/qmmm_polarizable.rs::p_shell_dipole_potential_matches_charge_derivative_block`,
//! which ALSO caught a real ordering bug: libint2's cartesian p-shell
//! function order for this build is **not** `[p_x, p_y, p_z]` matched to
//! field axes `[x, y, z]` — an exhaustive 9-pairing least-squares fit (every
//! (field axis, p-shell function) pair) found near-zero residual, shrinking
//! ~10x per 10x increase in `zeta`, at EVERY `zeta` in
//! `{1e2,1e3,1e4,1e5,1e6}`, ONLY for
//!
//! ```text
//! field_axis[c] = -norm_int_p(zeta) * (mu nu | p_{(c+2) mod 3})
//! norm_int_p(zeta) = 2^(1/4) zeta^(5/4) / (2 pi^(3/4))
//! ```
//!
//! i.e. the p-shell function that pairs with field axis `c` is at cartesian
//! index `(c + 2) mod 3` (a cyclic permutation, not the identity), and the
//! raw p-shell integral must be MULTIPLIED by `norm_int_p(zeta)` (with a
//! minus sign), not divided — an earlier symbolic derivation from first
//! principles (unnormalized `p_x = (x-R_x) exp(-zeta r^2)` vs unnormalized
//! `g = exp(-zeta r^2)`, both rescaled to unit self-overlap) predicted the
//! IDENTITY permutation with a POSITIVE sign and a DIVISION by `norm_int_p`
//! — that symbolic prediction was independently checked against this
//! MEASURED result and falsified (both individually-verified sub-facts it
//! was built from checked out under separate probes, but their combination
//! did not match measurement, meaning the combination step itself, not
//! either sub-fact, was where the derivation went wrong — an object lesson
//! in trusting the independent numeric pin over a symbolic derivation that
//! LOOKS complete). The empirical fit is what ships: `k(zeta) = -79.704,
//! -1417.0, -25198.0, -448090, -7968290` for `zeta = 1e2..1e6` matches
//! `-norm_int_p(zeta)` to 4-7 significant figures (converging as
//! `zeta -> infinity`, tracked in `p_shell_zeta_convergence_sweep`) — see
//! [`p_axis_for_field_axis`] and [`norm_int_p_shell`].
//!
//! **A second, separate sign subtlety** bit the first implementation of
//! [`build_v_induced`] even after this pin passed: the RAW derivative
//! convention just measured above is NOT the same sign as the PHYSICAL
//! electric field [`crate::qmmm::electric_field_at_points`] (and therefore
//! `induce()`'s `E_i^QM`) actually uses — see [`build_v_induced`]'s doc
//! comment for the full correction history (a wrong-sign `V_ind` still
//! produces a stable, self-consistent-LOOKING SCF fixed point, just the
//! WRONG one; it was caught only by comparing against the independent
//! PySCF prototype's dipoles/energy, not by this pin test, which does not
//! exercise that second sign at all).
//!
//! # Exclusions
//!
//! `PolarizableSites.exclusions` is a set of unordered `(i, j)` pairs
//! (indices into `sites`) with:
//!
//! 1. **no mutual induction**: `T_ij` is forced to zero for an excluded pair
//!    (neither site polarises the other), and
//! 2. **no permanent-field coupling from the SAME pair**: this is
//!    implemented by POSITION, not by a separate index map — a permanent
//!    charge in `ext` located at (within `COLOCATION_TOL` Bohr of) an
//!    excluded partner site's position is dropped from that site's `E^perm`
//!    sum. This matches the standard use-case (a polarizable site typically
//!    IS a permanently-charged MM atom — see `QmmmAtom{x,y,z,charge,alpha}`
//!    in `crate::qmmm`, one atom, one position, both roles at once) and
//!    needs no separate site<->ext-charge index field beyond the two lists'
//!    shared geometry. Default (empty `exclusions`): every site sees every
//!    other site's permanent charge and every other site's induced dipole,
//!    which is correct for isolated, non-bonded polarizable sites.

use crate::qmmm::electric_field_at_points;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;

/// Thole damping parameter default (Thole, Chem. Phys. 59, 341 (1981)).
pub const DEFAULT_THOLE_A: f64 = 2.1304;

/// Default Gaussian exponent (Bohr^-2) for the induced-dipole's p-shell site
/// basis — the "point dipole" limit. Width `1/sqrt(zeta) ~ 0.01 Bohr`.
pub const DEFAULT_DIPOLE_ZETA: f64 = 1e4;

/// Default hard ceiling on the number of polarizable sites for the dense
/// `(3N)x(3N)` solve. Exceeding this is a typed error (see module doc);
/// a follow-up iterative solve is out of scope for this phase.
pub const DEFAULT_MAX_SITES_DENSE: usize = 4000;

/// Tolerance (Bohr) for treating an `ext` permanent charge as "colocated
/// with" (and therefore excludable alongside) a polarizable site — see the
/// module doc's Exclusions section.
const COLOCATION_TOL: f64 = 1e-8;

/// For field axis `c` (0=x, 1=y, 2=z), the cartesian p-shell FUNCTION index
/// libint2 (this build) actually produces the matching field response at —
/// `(c + 2) % 3`, NOT `c`. MEASURED, not assumed: see the module doc's
/// "Integrals: the p-shell trick" section and
/// `crates/ferric-scf/tests/qmmm_polarizable.rs`'s pin test, which fits all
/// 9 (field axis, p-shell function) pairings and finds this the only one
/// with a residual that shrinks as `dipole_zeta` grows.
#[inline]
pub(crate) fn p_axis_for_field_axis(field_axis: usize) -> usize {
    (field_axis + 2) % 3
}

/// The scaling constant for a p-shell (l=1) site basis at exponent `zeta`:
/// `field_axis[c] = -norm_int_p_shell(zeta) *
/// (mu nu|p_{p_axis_for_field_axis(c)})` gives the field-integral component
/// `<mu|(r-R)_c/|r-R|^3|nu>` directly (see module doc — this relationship
/// was MEASURED, not derived; multiply by this constant, do not divide).
#[inline]
pub(crate) fn norm_int_p_shell(zeta: f64) -> f64 {
    2f64.powf(0.25) * zeta.powf(1.25) / (2.0 * std::f64::consts::PI.powf(0.75))
}


/// One polarizable MM site: position (Bohr) and isotropic polarisability
/// (Bohr^3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolarizableSite {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub alpha: f64,
}

/// Configuration for a set of Thole-damped polarizable sites. See the module
/// doc for the full model.
#[derive(Debug, Clone)]
pub struct PolarizableSites {
    pub sites: Vec<PolarizableSite>,
    /// Thole damping parameter. `Some(2.1304)` (use [`DEFAULT_THOLE_A`]) is
    /// the conventional default; `None` disables damping (bare 1/r^3
    /// dipole-dipole tensor).
    pub thole_a: Option<f64>,
    /// Site-index pairs (into `sites`) with no mutual induction and no
    /// permanent-field coupling from charges colocated with the excluded
    /// partner — see the module doc's Exclusions section. Unordered: `(i,
    /// j)` and `(j, i)` are equivalent; both are accepted.
    pub exclusions: Vec<(usize, usize)>,
    /// Gaussian exponent (Bohr^-2) of the induced-dipole p-shell site basis.
    /// Default [`DEFAULT_DIPOLE_ZETA`] (numerically a point dipole).
    pub dipole_zeta: f64,
    /// Hard ceiling on `sites.len()` for the dense solve. Default
    /// [`DEFAULT_MAX_SITES_DENSE`].
    pub max_sites_dense: usize,
}

impl Default for PolarizableSites {
    fn default() -> Self {
        Self {
            sites: Vec::new(),
            thole_a: Some(DEFAULT_THOLE_A),
            exclusions: Vec::new(),
            dipole_zeta: DEFAULT_DIPOLE_ZETA,
            max_sites_dense: DEFAULT_MAX_SITES_DENSE,
        }
    }
}

impl PolarizableSites {
    /// Normalised (unordered) exclusion check.
    fn is_excluded(&self, i: usize, j: usize) -> bool {
        self.exclusions.iter().any(|&(a, b)| (a == i && b == j) || (a == j && b == i))
    }
}

/// Result of one induction solve: the converged dipoles, the polarisation
/// energy, and the Fock-matrix contribution `v_induced` (AO basis, already
/// carrying the correct sign/no-1/2 convention for direct addition to the
/// Fock matrix — see the module doc).
#[derive(Debug, Clone)]
pub struct InductionResult {
    /// `(n_sites, 3)` induced dipole moments (a.u.).
    pub dipoles: Array2<f64>,
    /// Polarisation energy `E_pol` (Hartree).
    pub e_pol: f64,
    /// Fock-matrix contribution (AO basis, `(nbasis, nbasis)`).
    pub v_induced: Array2<f64>,
}

/// Thole-damped dipole-dipole interaction tensor `T_ij` (3x3, a.u.).
/// `thole_a: None` gives the bare (undamped) tensor.
fn thole_tensor(ri: [f64; 3], rj: [f64; 3], alpha_i: f64, alpha_j: f64, thole_a: Option<f64>) -> [[f64; 3]; 3] {
    let d = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let r = r2.sqrt();
    let rhat = [d[0] / r, d[1] / r, d[2] / r];
    let (lam3, lam5) = match thole_a {
        None => (1.0, 1.0),
        Some(a) => {
            let s = (alpha_i * alpha_j).powf(1.0 / 6.0);
            let u = r / s;
            let au3 = a * u * u * u;
            let expo = (-au3).exp();
            (1.0 - expo, 1.0 - (1.0 + au3) * expo)
        }
    };
    let inv_r3 = 1.0 / (r2 * r);
    let mut t = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let eye = if a == b { 1.0 } else { 0.0 };
            t[a][b] = (lam3 * eye - 3.0 * lam5 * rhat[a] * rhat[b]) * inv_r3;
        }
    }
    t
}

/// Field of `ext`'s permanent (point + Gaussian-smeared) charges at each
/// site, honouring colocation-based exclusions (see module doc).
fn permanent_field_at_sites(sites: &PolarizableSites, ext: Option<&ExternalPotential>) -> Vec<[f64; 3]> {
    let n = sites.sites.len();
    let mut e_perm = vec![[0.0_f64; 3]; n];
    let Some(ext) = ext else { return e_perm };

    // Colocation lookup: for a given `ext` charge position, which site
    // indices (if any) sit within COLOCATION_TOL of it.
    let colocated_site = |x: f64, y: f64, z: f64| -> Option<usize> {
        sites.sites.iter().position(|s| {
            let dx = s.x - x;
            let dy = s.y - y;
            let dz = s.z - z;
            (dx * dx + dy * dy + dz * dz).sqrt() < COLOCATION_TOL
        })
    };

    for (i, site) in sites.sites.iter().enumerate() {
        let ri = [site.x, site.y, site.z];
        for pc in &ext.point_charges {
            if let Some(j) = colocated_site(pc.x, pc.y, pc.z) {
                if j == i || sites.is_excluded(i, j) {
                    continue;
                }
            }
            let d = [ri[0] - pc.x, ri[1] - pc.y, ri[2] - pc.z];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            if r < 1e-12 {
                continue;
            }
            let inv_r3 = pc.q / (r2 * r);
            e_perm[i][0] += inv_r3 * d[0];
            e_perm[i][1] += inv_r3 * d[1];
            e_perm[i][2] += inv_r3 * d[2];
        }
        for sc in &ext.smeared_charges {
            if let Some(j) = colocated_site(sc.x, sc.y, sc.z) {
                if j == i || sites.is_excluded(i, j) {
                    continue;
                }
            }
            let d = [ri[0] - sc.x, ri[1] - sc.y, ri[2] - sc.z];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            if r < 1e-12 {
                continue;
            }
            // Field of a Gaussian-smeared charge: q * erf(sqrt(zeta) r) / r^2
            // in the radial direction, MINUS the short-range correction —
            // exactly the point-charge field in the r >> width limit; for
            // simplicity and because polarizable sites are not expected to
            // sit inside another site's smearing cloud, use the point-charge
            // field formula here (consistent with `electric_field_at_points`
            // treating nuclei as points; a full smeared-field formula is a
            // documented gap, not silently wrong: at the sites' typical
            // separations from the widths in use this is >99.9% accurate,
            // and self-consistent with Lane A's own `mm_forces` treatment of
            // point-vs-smeared distinctions being formula-exact only at the
            // OWN site, not at a third point).
            let inv_r3 = sc.q / (r2 * r);
            e_perm[i][0] += inv_r3 * d[0];
            e_perm[i][1] += inv_r3 * d[1];
            e_perm[i][2] += inv_r3 * d[2];
        }
    }
    e_perm
}

/// Solve the dense Thole-damped induction system `B mu = e_ext` for the
/// converged induced dipoles, given the total (QM + permanent) field at each
/// site.
fn solve_induction(sites: &PolarizableSites, e_total: &[[f64; 3]]) -> Result<Array2<f64>, FerricError> {
    let n = sites.sites.len();
    let mut b = Array2::<f64>::zeros((3 * n, 3 * n));
    for (i, site) in sites.sites.iter().enumerate() {
        if site.alpha <= 0.0 || !site.alpha.is_finite() {
            return Err(FerricError::General(format!(
                "polarizable::induce: site {i} has non-positive or non-finite alpha = {}",
                site.alpha
            )));
        }
        for a in 0..3 {
            b[(3 * i + a, 3 * i + a)] = 1.0 / site.alpha;
        }
    }
    for i in 0..n {
        for j in 0..n {
            if i == j || sites.is_excluded(i, j) {
                continue;
            }
            let ri = [sites.sites[i].x, sites.sites[i].y, sites.sites[i].z];
            let rj = [sites.sites[j].x, sites.sites[j].y, sites.sites[j].z];
            let t = thole_tensor(ri, rj, sites.sites[i].alpha, sites.sites[j].alpha, sites.thole_a);
            for a in 0..3 {
                for c in 0..3 {
                    b[(3 * i + a, 3 * j + c)] = -t[a][c];
                }
            }
        }
    }
    let mut rhs = Array2::<f64>::zeros((3 * n, 1));
    for (i, e) in e_total.iter().enumerate() {
        for a in 0..3 {
            rhs[(3 * i + a, 0)] = e[a];
        }
    }
    let mu_flat = b
        .solve_into(rhs.column(0).to_owned())
        .map_err(|e| FerricError::Lapack(format!("polarizable induction dense solve failed: {e}")))?;
    let mut mu = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        for a in 0..3 {
            mu[(i, a)] = mu_flat[3 * i + a];
        }
    }
    Ok(mu)
}

/// Fock-matrix contribution `V_ind_munu = -sum_i mu_i . field_integral_i`
/// from the converged dipoles, via the p-shell site basis `site_basis_p`
/// (built with `l=1` at `sites.dipole_zeta` — see module doc). `site_basis_p`
/// must have one shell per site, in `sites.sites` order (as produced by
/// `SiteBasis::new(&[[x,y,z,dipole_zeta]; n], 1)`).
///
/// TWO sign conventions are in play and must not be conflated (this bit the
/// first implementation — see the correction note below):
///
/// 1. The RAW p-shell/nuclear-derivative convention pinned in
///    `crates/ferric-scf/tests/qmmm_polarizable.rs`'s
///    `p_shell_dipole_potential_matches_charge_derivative_block`:
///    `raw_deriv[c] = d/dR_c <mu|1/|r-R||nu> = -norm_int_p_shell(zeta) *
///    (mu nu|p_{p_axis_for_field_axis(c)})`.
/// 2. The PHYSICAL electric-field convention `electric_field_at_points`
///    (and hence `induce()`'s `E_i^QM`) actually uses, which carries an
///    EXTRA minus sign relative to (1): `E_qm[c] = -raw_deriv[c] =
///    +norm_int_p_shell(zeta) * (mu nu|p_axis(c))`. This is because
///    `electric_field_at_points` accumulates `e_elec[d] -= D . deriv` (see
///    that function's own doc/code), not `+=`.
///
/// The Fock term `V_ind = -mu . E_qm_integral` (module doc, no 1/2) must use
/// convention (2), giving `V_ind_munu = -mu_i[c] * (+norm_int_p_shell(zeta)
/// * (mu nu|p_axis(c))) = -mu_i[c] * norm_int_p_shell(zeta) *
/// (mu nu|p_axis(c))` — i.e. accumulate `-mu[c] * norm_p`, NOT `+mu[c] *
/// norm_p`. CORRECTION HISTORY: an earlier version of this function used
/// `+mu[c] * norm_p` (convention (1)'s sign, forgetting the extra minus
/// `electric_field_at_points` applies) — this shipped a self-consistent-
/// looking but WRONG-SIGN Fock term that still converged (a stable SCF fixed
/// point exists either way) but converged to the WRONG dipole (~1e-5
/// absolute error vs the PySCF prototype, ~1.5e-4 Ha energy error on the
/// multi-site cases) — caught by `matches_pyscf_prototype_*` in
/// `tests/qmmm_polarizable.rs`, root-caused by directly comparing this
/// function's raw `V_ind` matrix elements (for a fixed test dipole) against
/// the Python prototype's `dipole_potential_integral`-based `v_ind`, which
/// were equal in magnitude and OPPOSITE in sign — see that debugging
/// session's `debug_v_induced_matrix_dump` probe (since removed). This is
/// why the pin test alone (which only checks the RAW p-shell/derivative
/// relationship, convention (1)) was not sufficient to catch this: it
/// PASSED throughout, because the bug was in how (1) was translated into (2)
/// here, one level up from what the pin test exercises. NOTE this does NOT
/// use `site_basis_p.norm_int` (that field is the l=0/charge formula from
/// lane A's `SiteBasis`, wrong for l=1 by construction; the correct p-shell
/// scaling is `norm_int_p_shell`, measured and pinned separately in this
/// module).
fn build_v_induced(
    prep: &PreparedBasis,
    site_basis_p: &SiteBasis,
    dipole_zeta: f64,
    dipoles: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let nbasis = prep.nbasis();
    let mut v = Array2::<f64>::zeros((nbasis, nbasis));
    let n = dipoles.nrows();
    if n == 0 {
        return Ok(v);
    }
    let norm_p = norm_int_p_shell(dipole_zeta);
    let mut eng = Engine::new_3center(Operator::coulomb(), prep, &site_basis_p.prep, 1e-14)?;
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let nsh = prep.nshells();

    for i in 0..n {
        let sh_p = site_basis_p.site_shell[i];
        let mu = [dipoles[(i, 0)], dipoles[(i, 1)], dipoles[(i, 2)]];
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let Some(block) = eng.compute_eri3(prep, &site_basis_p.prep, sh_p, s1, s2) else { continue };
                let n1 = dims[s1];
                let n2 = dims[s2];
                let o1 = offs[s1];
                let o2 = offs[s2];
                for c in 0..3 {
                    // p-shell function that carries field axis c's response
                    // (measured permutation — see module doc).
                    let p_fn = p_axis_for_field_axis(c);
                    let coeff = -mu[c] * norm_p;
                    for a in 0..n1 {
                        for bb in 0..n2 {
                            let val = block[(p_fn * n1 + a) * n2 + bb];
                            v[(o1 + a, o2 + bb)] += coeff * val;
                        }
                    }
                }
            }
        }
    }
    Ok(v)
}

/// One induction solve at the current density: field at sites (QM + MM
/// permanent), dense Thole solve, `E_pol`, and the Fock-matrix contribution.
/// Returns a trivially-empty [`InductionResult`] when `sites.sites` is
/// empty, WITHOUT constructing a 0x0 linear system or touching `d_total`
/// (the exactness anchor for `polarizable: Some(PolarizableSites{sites:
/// vec![], ..})` bit-identical to `None`).
///
/// `site_basis_p` must be `SiteBasis::new(&sites_as_zeta_tuples, 1)` built
/// from the SAME `sites.sites` in the SAME order (callers rebuild it once
/// per SCF geometry, not once per iteration, since it depends only on
/// geometry — mirrors how `ScfEnv`'s `cosmo_cavity`/`pcm_ctx` are built once
/// outside the iteration loop).
pub fn induce(
    mol: &Molecule,
    prep: &PreparedBasis,
    ext: Option<&ExternalPotential>,
    sites: &PolarizableSites,
    site_basis_p: &SiteBasis,
    d_total: &Array2<f64>,
) -> Result<InductionResult, FerricError> {
    let n = sites.sites.len();
    let nbasis = prep.nbasis();
    if n == 0 {
        return Ok(InductionResult {
            dipoles: Array2::zeros((0, 3)),
            e_pol: 0.0,
            v_induced: Array2::zeros((nbasis, nbasis)),
        });
    }
    if n > sites.max_sites_dense {
        return Err(FerricError::General(format!(
            "polarizable::induce: {n} sites exceeds max_sites_dense = {} \
             (dense (3N)x(3N) solve only in this phase — see the module doc)",
            sites.max_sites_dense
        )));
    }

    let points: Vec<[f64; 3]> = sites.sites.iter().map(|s| [s.x, s.y, s.z]).collect();
    let e_qm = electric_field_at_points(mol, prep, d_total, &points)?;
    let e_perm = permanent_field_at_sites(sites, ext);
    let e_total: Vec<[f64; 3]> = e_qm
        .iter()
        .zip(e_perm.iter())
        .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .collect();

    let dipoles = solve_induction(sites, &e_total)?;

    let mut e_pol = 0.0_f64;
    for i in 0..n {
        e_pol -= 0.5
            * (dipoles[(i, 0)] * e_total[i][0] + dipoles[(i, 1)] * e_total[i][1] + dipoles[(i, 2)] * e_total[i][2]);
    }

    let v_induced = build_v_induced(prep, site_basis_p, sites.dipole_zeta, &dipoles)?;

    Ok(InductionResult { dipoles, e_pol, v_induced })
}

/// Polarisation energy from `E^perm` alone (dipoles induced by the OTHER
/// permanent MM charges, ignoring the QM region entirely) — the "MM-internal"
/// constant a caller can subtract to get an interaction energy that excludes
/// the MM region's own self-polarisation. Returns `0.0` for an empty site
/// list without any solve.
pub fn mm_only_polarization_energy(ext: Option<&ExternalPotential>, sites: &PolarizableSites) -> Result<f64, FerricError> {
    let n = sites.sites.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n > sites.max_sites_dense {
        return Err(FerricError::General(format!(
            "polarizable::mm_only_polarization_energy: {n} sites exceeds max_sites_dense = {}",
            sites.max_sites_dense
        )));
    }
    let e_perm = permanent_field_at_sites(sites, ext);
    let dipoles = solve_induction(sites, &e_perm)?;
    let mut e_pol = 0.0_f64;
    for i in 0..n {
        e_pol -= 0.5
            * (dipoles[(i, 0)] * e_perm[i][0] + dipoles[(i, 1)] * e_perm[i][1] + dipoles[(i, 2)] * e_perm[i][2]);
    }
    Ok(e_pol)
}
