//! Periodic effective core potentials (stage 2 "periodic ECP"; the Rust port
//! of `reference/pbc/pbc_ecp.py`, FINDINGS "Iteration 14").
//!
//! A periodic ECP is two things and only two:
//!
//! 1. **Z_eff wherever a nucleus is a Coulomb source.** [`Cell::nuclear_charges`]
//!    already returns `Atom::effective_z` (`Z − n_core_ecp`), so the Ewald
//!    `E_nn`, the SR/LR nuclear attraction and the structure factors follow
//!    `Molecule::apply_ecp` — PROVIDED the cell's molecule went through it.
//!    That call is the single point of failure and a bare Z is INVISIBLE to
//!    the k-mesh ≡ supercell anchor (both sides carry the same wrong charge;
//!    measured in the prototype), so [`check_ecp_applied`] is a hard,
//!    typed guard that `periodic_hcore` / `periodic_hcore_kpts` run first.
//! 2. **A short-range Bloch sum of ordinary molecular ECP integrals:**
//!
//! ```text
//! V_L[m, n] = Σ_{C, M} ⟨φ_m(r) | U_C(r − R_C − M) | φ_n(r − L)⟩
//! V_ECP(k)  = Σ_L e^{ik·L} V_L          (Gamma: Σ_L V_L)
//! ```
//!
//! over orbital images `L` and ECP-centre images `M`. Every radial term of
//! `U_C` is a Gaussian `r^{n−2} e^{−ζ r²}`, so nothing touches G = 0 or
//! the Madelung constant.
//!
//! # Kernel
//!
//! One `ferric_ecp_block` call per orbital image `L` (the libecpint
//! per-shell-pair kernel `ECPIntegral::compute_shell_pair` behind the
//! exception-safe C shim, `crates/ferric-integrals/shim/ecp_shim.cc`): bra =
//! the home shells, ket = the home shells translated by `L`, ECP centres =
//! the images `R_C + M` that survive the screen for at least one shell pair,
//! and a per-(bra shell, ket shell, centre) mask carrying the screen. No
//! 4× "square supermolecule" waste and no C++-side screening: the
//! truncation below is the only one, and it is reported.
//!
//! # Screening (all derived from `precision`)
//!
//! For a triple (shell a at A, shell b at B + L, centre u at C + M) with most
//! diffuse exponents `α_a`, `α_b`, `ζ_u`, the integrand is a product of three
//! Gaussians, bounded pointwise by each PAIR product: `e^{−α|r−A|²}
//! e^{−ζ|r−C|²} ≤ e^{−μ(α,ζ)|A−C|²}`, `μ(x, y) = xy/(x+y)`. So
//!
//! ```text
//! E = max( μ(α_a,ζ_u)|A−C−M|², μ(α_b,ζ_u)|B+L−C−M|², μ(α_a,α_b)|A−B−L|² )
//! keep the triple iff E <= ln(D_u / precision) + ECP_LOG_MARGIN
//! ```
//!
//! with `D_u = Σ_t |d_t|` (the ECP's coefficient sum) and a margin `e^3` for
//! the polynomial prefactors of `l > 0` functions (as `SR_MARGIN_BOHR` in
//! `hcore`). `μ` is increasing in both exponents, so the most diffuse
//! primitive gives the slowest decay (conservative). The screen is a
//! SYMMETRIC function of the three distances, so the transposed triple
//! (b at B, a at A − L, centre at C + M − L) is kept exactly when the triple
//! is: `V_{−L} = V_Lᵀ` holds by construction, and the residual
//! ([`PeriodicEcpImages::asymmetry`]) is reported. The global radii
//! `r_ecp = √(X_max/μ_min(α, ζ))` and `r_pair = min(√(X_max/μ_min(α, α)),
//! 2 r_ecp)` only enumerate candidates.
//!
//! The optional caps ([`PeriodicEcpConfig::ecp_radius_cap`],
//! [`PeriodicEcpConfig::pair_radius_cap`]) truncate the SAME symmetric
//! distances (`|A − C − M|`, `|B + L − C − M|` resp. `|A − B − L|`), so a
//! capped sum is a partial sum of the same numbers — the convergence study
//! of `tests/pbc_ecp.rs` (prototype: Gaussian decay, 8.8e-11 at 16 Bohr on
//! HI/LANL2DZ, not a plateau).
//!
//! # Memory
//!
//! The per-image blocks (`n_L × n² × 8` bytes), the translation lists and the
//! centre list are reserved on the [`crate::budget`] ledger BEFORE they are
//! allocated.
//!
//! # Scope
//!
//! Scalar (spin-free) ECPs: energies (Gamma and k) and the Gamma force term
//! ([`periodic_ecp_gradient`], FINDINGS "Iteration 22"; added to every Gamma
//! force by `crate::grad`). Stress and k-point forces with an ECP are
//! refused by their callers. Ghost atoms carry no ECP (no nucleus); the
//! molecular `ecp_potential` includes a ghost's ECP, which is a
//! molecular-path quirk not reproduced here.
//!
//! # Forces
//!
//! Each kept triple `⟨μ_0|U_C(r − R_C − M)|ν_L⟩` depends on three centres
//! that all move with their atoms, in every image: the bra `R_A`, the ket
//! image `R_B + L` and the ECP image `R_C + M`. The force differentiates
//! exactly the energy's triples (one shared `EcpPlan` screen, the
//! "frozen image set" of the prototype), with the centre derivative
//! `−(∂_bra + ∂_ket)` per triple (translation invariance). The kernel is
//! `ferric_ecp_block_deriv` (libecpint's raised/lowered VALUE-integral
//! shell derivative, true partials also for a shell on its own ECP centre,
//! where libecpint's `compute_shell_pair_derivative` would report
//! `A = −B, C = 0`). Do not pin to PySCF 2.13's `ECPscalar_ipnuc`
//! (1.1e-7 off on off-centre elements) or `pbc.gto.ecp.ecp_int` (FINDINGS
//! Iterations 14 and 22).

use crate::budget::{bytes_of, Ledger};
use crate::kpts::{lattice_coords, KPointMesh};
use crate::lattice::Cell;
use crate::ordered::{ordered_units, window_budget, Stored};
use ferric_core::basis::BasisSet;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::ecp::{
    ecp_block_deriv_spherical, ecp_block_spherical, gto_norm, EcpCenter, EcpGaussianShell,
};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;

/// Default truncation target of the ECP lattice sum (Hartree per neglected
/// triple, before the prefactor margin) — `periodic_hcore`'s default.
pub const DEFAULT_ECP_PRECISION: f64 = 1e-14;

/// Extra log-magnitude on every screen (`e^3 ≈ 20`): polynomial prefactors of
/// `l > 0` shells and the `r^{n−2}` radial powers are not in the s-type
/// Gaussian-product bound.
pub const ECP_LOG_MARGIN: f64 = 3.0;

/// Typed failure of the periodic-ECP consistency guard ([`check_ecp_applied`]).
#[derive(Debug, Clone, PartialEq)]
pub enum PeriodicEcpError {
    /// The basis carries an ECP for this atom's element but the atom's
    /// `n_core_ecp` does not match it: the molecule was not passed through
    /// `Molecule::apply_ecp(bs)` (bare Z in every Coulomb source — a charged
    /// cell for the electrons, and invisible to the k-mesh anchor).
    EcpNotApplied {
        /// Atom index in the cell's molecule.
        atom: usize,
        /// Atomic number.
        z: i32,
        /// `n_core` of the basis's ECP for this element.
        expected_n_core: i32,
        /// The atom's `n_core_ecp`.
        found_n_core: i32,
    },
    /// The atom carries `n_core_ecp != 0` but the basis has no ECP for its
    /// element (`apply_ecp` run with a DIFFERENT basis): Z_eff without the
    /// potential that justifies it.
    StaleEcpCore {
        /// Atom index in the cell's molecule.
        atom: usize,
        /// Atomic number.
        z: i32,
        /// The atom's `n_core_ecp`.
        found_n_core: i32,
    },
    /// A shell whose spherical dimension the ECP kernel cannot match to the
    /// `PreparedBasis` (Cartesian `l >= 2`: the kernel returns `2l+1`
    /// functions per shell, libint2 `(l+1)(l+2)/2`).
    UnsupportedShell {
        /// Shell index (PreparedBasis order).
        shell: usize,
        /// Angular momentum.
        l: i32,
        /// Functions the PreparedBasis has for it.
        prep_dim: usize,
    },
}

impl fmt::Display for PeriodicEcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EcpNotApplied {
                atom,
                z,
                expected_n_core,
                found_n_core,
            } => write!(
                f,
                "periodic ECP: atom {atom} (Z = {z}) has n_core_ecp = {found_n_core} but the basis \
                 carries an ECP with n_core = {expected_n_core}; call Molecule::apply_ecp(&bs) \
                 before building the Cell (a bare Z charges the cell and the k-mesh anchor cannot \
                 see it)"
            ),
            Self::StaleEcpCore {
                atom,
                z,
                found_n_core,
            } => write!(
                f,
                "periodic ECP: atom {atom} (Z = {z}) has n_core_ecp = {found_n_core} but the basis \
                 carries no ECP for Z = {z}; apply_ecp was run with a different basis"
            ),
            Self::UnsupportedShell { shell, l, prep_dim } => write!(
                f,
                "periodic ECP: shell {shell} (l = {l}) has {prep_dim} functions in the \
                 PreparedBasis but the ECP kernel returns {} (Cartesian l >= 2 is unsupported)",
                2 * l + 1
            ),
        }
    }
}

impl std::error::Error for PeriodicEcpError {}

impl From<PeriodicEcpError> for FerricError {
    fn from(e: PeriodicEcpError) -> Self {
        FerricError::Basis(e.to_string())
    }
}

/// The guard: every non-ghost atom whose element has an ECP in `bs` must carry
/// exactly that ECP's `n_core` in `n_core_ecp`, and every other non-ghost atom
/// must carry 0. Ghost atoms are skipped (their effective charge is 0
/// whatever `n_core_ecp` says). Returns the FIRST offending atom.
pub fn check_ecp_applied(cell: &Cell, bs: &BasisSet) -> Result<(), PeriodicEcpError> {
    for (atom, a) in cell.mol().atoms.iter().enumerate() {
        if a.ghost {
            continue;
        }
        match bs.ecp_for_element(a.z) {
            Some(def) if a.n_core_ecp != def.n_core => {
                return Err(PeriodicEcpError::EcpNotApplied {
                    atom,
                    z: a.z,
                    expected_n_core: def.n_core,
                    found_n_core: a.n_core_ecp,
                })
            }
            None if a.n_core_ecp != 0 => {
                return Err(PeriodicEcpError::StaleEcpCore {
                    atom,
                    z: a.z,
                    found_n_core: a.n_core_ecp,
                })
            }
            _ => {}
        }
    }
    Ok(())
}

/// Test/diagnostic image-set mutations (FINDINGS Iteration 14 `_MUTANT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcpMutation {
    /// Only `M = 0` and `L = 0`: "called the molecular ECP routine on the
    /// cell basis". Hermitian and k-independent (so an SCF still runs); the
    /// k-mesh ≡ supercell anchor sees it, the box limit does not.
    MolecularOnly,
}

/// Settings for [`periodic_ecp_images`].
#[derive(Debug, Clone, Copy)]
pub struct PeriodicEcpConfig {
    /// Truncation target in `(0, 1)` (module doc).
    pub precision: f64,
    /// Optional cap (Bohr) on `|A − C − M|` and `|B + L − C − M|` (ECP-image
    /// range), applied on top of the derived screen. `None` in production.
    pub ecp_radius_cap: Option<f64>,
    /// Optional cap (Bohr) on `|A − B − L|` (orbital-image range). `None` in
    /// production.
    pub pair_radius_cap: Option<f64>,
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    #[doc(hidden)]
    pub mutation: Option<EcpMutation>,
}

impl PeriodicEcpConfig {
    /// Production settings at the given precision (no caps, no mutation).
    pub fn with_precision(precision: f64) -> Self {
        Self {
            precision,
            ecp_radius_cap: None,
            pair_radius_cap: None,
            budget_bytes: None,
            mutation: None,
        }
    }

    fn validate(&self) -> Result<(), FerricError> {
        if !(f64::MIN_POSITIVE..1.0).contains(&self.precision) {
            return Err(FerricError::General(format!(
                "periodic ECP: precision must lie in (0, 1), got {}",
                self.precision
            )));
        }
        for (name, cap) in [
            ("ecp_radius_cap", self.ecp_radius_cap),
            ("pair_radius_cap", self.pair_radius_cap),
        ] {
            if let Some(c) = cap {
                if !(c >= 0.0) || !c.is_finite() {
                    return Err(FerricError::General(format!(
                        "periodic ECP: {name} must be finite and >= 0, got {c}"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl Default for PeriodicEcpConfig {
    fn default() -> Self {
        Self::with_precision(DEFAULT_ECP_PRECISION)
    }
}

/// Per-orbital-image ECP blocks `V_L` (module doc) and the screen's counts.
#[derive(Debug, Clone)]
pub struct PeriodicEcpImages {
    /// AO dimension `n` of every block.
    pub nbasis: usize,
    /// Orbital images `L` (Bohr) with at least one kept triple.
    pub images: Vec<[f64; 3]>,
    /// `V_L`, `(nbasis, nbasis)` in PreparedBasis AO order, same order as
    /// `images`.
    pub blocks: Vec<Array2<f64>>,
    /// Kept (shell, shell, centre-image) triples.
    pub n_triples: usize,
    /// Distinct ECP-centre images used.
    pub n_ecp_images: usize,
    /// `ferric_ecp_block` calls (one per kept `L`).
    pub n_calls: usize,
    /// Derived ECP-image radius (before caps), Bohr.
    pub r_ecp: f64,
    /// Derived orbital-image radius (before caps), Bohr.
    pub r_pair: f64,
    /// `max_L max |V_L − V_{−L}ᵀ|` (a missing `−L` counts its whole block):
    /// the lattice sum's exact Hermiticity, before any symmetrisation.
    pub asymmetry: f64,
}

impl PeriodicEcpImages {
    /// Gamma-point `V_ECP = Σ_L V_L`, symmetrised (`½(V + Vᵀ)`; the residual
    /// is [`Self::asymmetry`]-sized).
    pub fn gamma(&self) -> Array2<f64> {
        let n = self.nbasis;
        let mut v = Array2::<f64>::zeros((n, n));
        for b in &self.blocks {
            v += b;
        }
        let vt = v.t().to_owned();
        (&v + &vt) * 0.5
    }

    /// `V_ECP(k) = Σ_L e^{ik·L} V_L` on every mesh point (mesh order),
    /// Hermitised (`½(X + X^H)`). The phases are `mesh.phase` (exact, as in
    /// `hcore::kpoint`), so `V(−k) = V(k)*` exactly.
    pub fn at_kpts(
        &self,
        cell: &Cell,
        mesh: &KPointMesh,
    ) -> Result<Vec<Array2<Complex64>>, FerricError> {
        let n = self.nbasis;
        let nk = mesh.nk();
        let b = cell.reciprocal();
        let mut out: Vec<Array2<Complex64>> = (0..nk)
            .map(|_| Array2::<Complex64>::zeros((n, n)))
            .collect();
        for (l, blk) in self.images.iter().zip(&self.blocks) {
            let nl = lattice_coords(&b, l);
            for (k, vk) in out.iter_mut().enumerate() {
                let ph = mesh.phase(k, nl);
                vk.zip_mut_with(blk, |z, &x| *z += ph * x);
            }
        }
        Ok(out
            .iter()
            .map(|m| Array2::from_shape_fn((n, n), |(i, j)| 0.5 * (m[(i, j)] + m[(j, i)].conj())))
            .collect())
    }
}

/// One ECP definition flattened for the shim, with its screen constants.
struct EcpTemplate {
    ams: Vec<i32>,
    ns: Vec<i32>,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    zmin: f64,
    /// `ln(D_u / precision) + ECP_LOG_MARGIN`, clamped at 0.
    x: f64,
}

#[inline]
fn mu(x: f64, y: f64) -> f64 {
    x * y / (x + y)
}

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// One ECP-centre image `R_C + M` that survives the candidate filter.
#[derive(Debug, Clone, Copy)]
struct Site {
    /// Index into the plan's templates.
    tmpl: usize,
    /// `R_C + M` (Bohr).
    pos: [f64; 3],
    /// The cell atom `C` (whose motion moves this image).
    atom: usize,
    /// `M = 0`.
    home: bool,
}

/// The screen shared by the energy ([`periodic_ecp_images`]) and the force
/// ([`periodic_ecp_gradient`]): home shells, ECP templates, candidate
/// centre images and orbital images, and the per-(shell, shell, centre)
/// test. Both evaluators walk `l_list` and call [`EcpPlan::kept`] with the
/// same arguments, so the force differentiates EXACTLY the energy's partial
/// sum at this geometry (the "frozen image set" of FINDINGS Iteration 22).
struct EcpPlan {
    templates: Vec<EcpTemplate>,
    /// Home shells in PreparedBasis order, `gto_norm` folded.
    shells: Vec<EcpGaussianShell>,
    /// Cell atom of each home shell.
    shell_atom: Vec<usize>,
    amin: Vec<f64>,
    log_pref: Vec<f64>,
    x_max: f64,
    sites: Vec<Site>,
    l_list: Vec<[f64; 3]>,
    cap_ecp2: Option<f64>,
    cap_pair2: Option<f64>,
    r_ecp: f64,
    r_pair: f64,
    nbasis: usize,
}

impl EcpPlan {
    /// Validates, builds the templates / shells / candidate lists (reserving
    /// them and one image's screen mask on `ledger`). `Ok(None)` when no
    /// non-ghost atom has an ECP in the basis.
    fn build(
        cell: &Cell,
        prep: &PreparedBasis,
        cfg: &PeriodicEcpConfig,
        ledger: &mut Ledger,
    ) -> Result<Option<Self>, FerricError> {
        cfg.validate()?;
        let bs = prep.basis_set();
        check_ecp_applied(cell, bs)?;
        if bs.ecps.is_empty() {
            return Ok(None);
        }
        let pos = cell.positions();
        if prep.natoms() != pos.len() {
            return Err(FerricError::General(format!(
                "periodic ECP: PreparedBasis has {} atoms but the cell has {}",
                prep.natoms(),
                pos.len()
            )));
        }
        for (k, (a, p)) in prep.atoms().iter().zip(&pos).enumerate() {
            let d = dist2([a.x, a.y, a.z], *p).sqrt();
            if d > 1e-10 {
                return Err(FerricError::General(format!(
                    "periodic ECP: PreparedBasis atom {k} is {d:.3e} Bohr from the cell's \
                     atom {k}; build the PreparedBasis from cell.mol()"
                )));
            }
        }

        // --- ECP templates and home centres (non-ghost atoms only).
        let mut templates: Vec<EcpTemplate> = Vec::new();
        let mut tmpl_of_z: HashMap<i32, usize> = HashMap::new();
        let mut home_centres: Vec<(usize, usize, [f64; 3])> = Vec::new();
        for (atom, (a, p)) in cell.mol().atoms.iter().zip(&pos).enumerate() {
            if a.ghost {
                continue;
            }
            let Some(def) = bs.ecp_for_element(a.z) else {
                continue;
            };
            let it = match tmpl_of_z.get(&a.z) {
                Some(&i) => i,
                None => {
                    let (mut ams, mut ns, mut exponents, mut coefficients) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for ch in &def.shells {
                        for t in &ch.terms {
                            ams.push(ch.angular_momentum);
                            ns.push(t.r_exp);
                            exponents.push(t.gexp);
                            coefficients.push(t.coef);
                        }
                    }
                    if exponents.is_empty() || exponents.iter().any(|&z| !(z > 0.0)) {
                        return Err(FerricError::Basis(format!(
                            "periodic ECP: ECP for Z = {} is empty or has a non-positive exponent",
                            a.z
                        )));
                    }
                    let zmin = exponents.iter().copied().fold(f64::INFINITY, f64::min);
                    let dsum: f64 = coefficients.iter().map(|c: &f64| c.abs()).sum();
                    let x = ((dsum.max(f64::MIN_POSITIVE) / cfg.precision).ln() + ECP_LOG_MARGIN)
                        .max(0.0);
                    templates.push(EcpTemplate {
                        ams,
                        ns,
                        exponents,
                        coefficients,
                        zmin,
                        x,
                    });
                    tmpl_of_z.insert(a.z, templates.len() - 1);
                    templates.len() - 1
                }
            };
            home_centres.push((it, atom, *p));
        }
        if home_centres.is_empty() {
            return Ok(None);
        }

        // --- Home shells in PreparedBasis order, gto_norm folded (the
        // molecular `oneelectron::ecp_potential` convention).
        let located = prep.located_shells();
        let dims = prep.shell_dims();
        let sh2at = prep.shell_to_atom();
        let mut shells: Vec<EcpGaussianShell> = Vec::with_capacity(located.len());
        let mut amin: Vec<f64> = Vec::with_capacity(located.len());
        for (s, sh) in located.iter().enumerate() {
            if sh.l < 0 || (2 * sh.l + 1) as usize != dims[s] {
                return Err(PeriodicEcpError::UnsupportedShell {
                    shell: s,
                    l: sh.l,
                    prep_dim: dims[s],
                }
                .into());
            }
            if sh.exponents.is_empty() || sh.exponents.iter().any(|&a| !(a > 0.0)) {
                return Err(FerricError::Basis(format!(
                    "periodic ECP: shell {s} is empty or has a non-positive exponent"
                )));
            }
            amin.push(sh.exponents.iter().copied().fold(f64::INFINITY, f64::min));
            shells.push(EcpGaussianShell {
                l: sh.l,
                center: sh.center,
                exponents: sh.exponents.to_vec(),
                coefficients: sh
                    .exponents
                    .iter()
                    .zip(sh.coefficients)
                    .map(|(&a, &c)| c * gto_norm(sh.l, a))
                    .collect(),
            });
        }
        let shell_atom: Vec<usize> = (0..shells.len()).map(|s| sh2at[s]).collect();
        let nsh = shells.len();

        // Per-shell log prefactor. Each pair bound e^{-mu|X-Y|^2} drops the
        // other two Gaussian factors pointwise; integrating what is left
        // leaves the third function's volume factor (pi/alpha)^{3/2} and the
        // contraction magnitudes. Omitting them under-estimated diffuse
        // triples by >1e4 (precision sweep plateaued at 1.3e-6 for
        // p = 1e-6..1e-10). This is the per-shell half, ln(sum|c|) + (3/4)
        // ln(pi/alpha_min); a triple adds both shells' halves to its
        // threshold.
        let log_pref: Vec<f64> = shells
            .iter()
            .zip(&amin)
            .map(|(sh, &am)| {
                let csum: f64 = sh.coefficients.iter().map(|c| c.abs()).sum();
                csum.max(f64::MIN_POSITIVE).ln() + 0.75 * (std::f64::consts::PI / am).ln()
            })
            .collect();
        // The candidate radii must admit every triple the prefactor-widened
        // screen keeps: widen by the largest pair prefactor.
        let pref_max = 2.0 * log_pref.iter().copied().fold(0.0_f64, f64::max);
        // --- Global candidate radii.
        let x_max = templates.iter().map(|t| t.x).fold(0.0_f64, f64::max) + pref_max;
        let a_lo = amin.iter().copied().fold(f64::INFINITY, f64::min);
        let z_lo = templates
            .iter()
            .map(|t| t.zmin)
            .fold(f64::INFINITY, f64::min);
        let r_ecp = (x_max / mu(a_lo, z_lo)).sqrt();
        let r_pair = (x_max / mu(a_lo, a_lo)).sqrt().min(2.0 * r_ecp);
        let molecular = cfg.mutation == Some(EcpMutation::MolecularOnly);
        let r_ecp_eff = cfg.ecp_radius_cap.map_or(r_ecp, |c| c.min(r_ecp));
        let r_pair_eff = cfg
            .pair_radius_cap
            .map_or(r_pair, |c| c.min(r_pair))
            .min(2.0 * r_ecp_eff);

        // --- Candidate ECP-centre images: C + M within r_ecp_eff of a home
        // atom.
        let (m_list, l_list) = if molecular {
            (vec![[0.0; 3]], vec![[0.0; 3]])
        } else {
            ledger.reserve(
                &format!("periodic ECP centre-image translations (r_ecp = {r_ecp_eff:.2} Bohr)"),
                bytes_of(cell.translation_count_bound(r_ecp_eff)?, 24),
            )?;
            let m_list = cell.translations(r_ecp_eff)?;
            ledger.reserve(
                &format!("periodic ECP orbital-image translations (r_pair = {r_pair_eff:.2} Bohr)"),
                bytes_of(cell.translation_count_bound(r_pair_eff)?, 24),
            )?;
            let l_list = cell.translations(r_pair_eff)?;
            (m_list, l_list)
        };
        ledger.reserve(
            "periodic ECP centre-image list",
            bytes_of((m_list.len() * home_centres.len()) as u64, 48),
        )?;
        let r2_ecp = r_ecp_eff * r_ecp_eff;
        let mut sites: Vec<Site> = Vec::new();
        for m in &m_list {
            let home = m.iter().all(|v| v.abs() < 1e-9);
            for &(it, atom, c) in &home_centres {
                let cm = [c[0] + m[0], c[1] + m[1], c[2] + m[2]];
                if molecular || pos.iter().any(|p| dist2(*p, cm) <= r2_ecp) {
                    sites.push(Site {
                        tmpl: it,
                        pos: cm,
                        atom,
                        home,
                    });
                }
            }
        }
        ledger.reserve(
            "periodic ECP screen mask (one image)",
            bytes_of((nsh * nsh * sites.len().max(1)) as u64, 1 + 8),
        )?;

        Ok(Some(Self {
            templates,
            shells,
            shell_atom,
            amin,
            log_pref,
            x_max,
            sites,
            l_list,
            cap_ecp2: cfg.ecp_radius_cap.map(|c| c * c),
            cap_pair2: cfg.pair_radius_cap.map(|c| c * c),
            r_ecp,
            r_pair,
            nbasis: prep.nbasis(),
        }))
    }

    /// The home shells translated by `l` (the ket of image `L`).
    fn ket_shells(&self, l: &[f64; 3]) -> Vec<EcpGaussianShell> {
        self.shells
            .iter()
            .map(|s| EcpGaussianShell {
                l: s.l,
                center: [s.center[0] + l[0], s.center[1] + l[1], s.center[2] + l[2]],
                exponents: s.exponents.clone(),
                coefficients: s.coefficients.clone(),
            })
            .collect()
    }

    /// Kept `(bra shell, ket shell, site)` triples of orbital image `ket`
    /// (module doc, "Screening"). Deterministic in its inputs: the energy and
    /// the force call it identically.
    fn kept(&self, ket: &[EcpGaussianShell]) -> Vec<(usize, usize, usize)> {
        let nsh = self.shells.len();
        let mut kept = Vec::new();
        for a in 0..nsh {
            let ra = self.shells[a].center;
            for b in 0..nsh {
                let rb = ket[b].center;
                let dab2 = dist2(ra, rb);
                if self.cap_pair2.is_some_and(|c| dab2 > c) {
                    continue;
                }
                let e_ab = mu(self.amin[a], self.amin[b]) * dab2;
                if e_ab > self.x_max {
                    continue;
                }
                for (u, site) in self.sites.iter().enumerate() {
                    let t = &self.templates[site.tmpl];
                    let dac2 = dist2(ra, site.pos);
                    let dbc2 = dist2(rb, site.pos);
                    if self.cap_ecp2.is_some_and(|c| dac2 > c || dbc2 > c) {
                        continue;
                    }
                    let e = e_ab
                        .max(mu(self.amin[a], t.zmin) * dac2)
                        .max(mu(self.amin[b], t.zmin) * dbc2);
                    if e <= t.x + (self.log_pref[a] + self.log_pref[b]).max(0.0) {
                        kept.push((a, b, u));
                    }
                }
            }
        }
        kept
    }

    /// Compact `kept` to the sites it uses: the shim's centre list, the
    /// site index of each local centre, and the `(a, b, local)` mask.
    fn compact(&self, kept: &[(usize, usize, usize)]) -> (Vec<EcpCenter>, Vec<usize>, Vec<u8>) {
        let nsh = self.shells.len();
        let mut local_of: HashMap<usize, usize> = HashMap::new();
        let mut centres: Vec<EcpCenter> = Vec::new();
        let mut site_of: Vec<usize> = Vec::new();
        for &(_, _, u) in kept {
            if let std::collections::hash_map::Entry::Vacant(e) = local_of.entry(u) {
                let site = self.sites[u];
                let t = &self.templates[site.tmpl];
                e.insert(centres.len());
                site_of.push(u);
                centres.push(EcpCenter {
                    center: site.pos,
                    ams: t.ams.clone(),
                    ns: t.ns.clone(),
                    exponents: t.exponents.clone(),
                    coefficients: t.coefficients.clone(),
                });
            }
        }
        let nu = centres.len();
        let mut mask = vec![0u8; nsh * nsh * nu];
        for &(a, b, u) in kept {
            mask[(a * nsh + b) * nu + local_of[&u]] = 1;
        }
        (centres, site_of, mask)
    }
}

/// Build the per-image blocks `V_L` for `cell` in the AO basis `prep` (built
/// from `cell.mol()`, carrying the ECP in `prep.basis_set()`).
///
/// Returns `Ok(None)` (and does no work) when no non-ghost atom of the cell
/// has an ECP in the basis. Runs [`check_ecp_applied`] first.
pub fn periodic_ecp_images(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicEcpConfig,
) -> Result<Option<PeriodicEcpImages>, FerricError> {
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    periodic_ecp_images_on(cell, prep, cfg, &mut ledger)
}

/// [`periodic_ecp_images`] reserving on the caller's ledger (the `hcore`
/// builders' own budget).
pub(crate) fn periodic_ecp_images_on(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicEcpConfig,
    ledger: &mut Ledger,
) -> Result<Option<PeriodicEcpImages>, FerricError> {
    let Some(plan) = EcpPlan::build(cell, prep, cfg, ledger)? else {
        return Ok(None);
    };
    let n = plan.nbasis;

    // Worst case every candidate L keeps a block.
    ledger.reserve(
        &format!(
            "periodic ECP per-image blocks ({} candidate L × n² with n = {n})",
            plan.l_list.len()
        ),
        bytes_of((plan.l_list.len() * n * n) as u64, 8),
    )?;

    let mut images = Vec::new();
    let mut blocks = Vec::new();
    let mut n_triples = 0usize;
    let mut used_site = vec![false; plan.sites.len()];
    for l in &plan.l_list {
        let ket = plan.ket_shells(l);
        let kept = plan.kept(&ket);
        if kept.is_empty() {
            continue;
        }
        let (centres, site_of, mask) = plan.compact(&kept);
        for &u in &site_of {
            used_site[u] = true;
        }
        n_triples += kept.len();
        let flat = ecp_block_spherical(&plan.shells, &ket, &centres, Some(mask.as_slice()))?;
        if flat.len() != n * n {
            return Err(FerricError::Libint(format!(
                "periodic ECP: block has {} entries, expected {n}²",
                flat.len()
            )));
        }
        images.push(*l);
        blocks.push(
            Array2::from_shape_vec((n, n), flat)
                .map_err(|e| FerricError::General(format!("periodic ECP block shape: {e}")))?,
        );
    }

    // --- Exact Hermiticity of the lattice sum: V_{-L} = V_Lᵀ.
    let b = cell.reciprocal();
    let idx: HashMap<[i64; 3], usize> = images
        .iter()
        .enumerate()
        .map(|(i, l)| (lattice_coords(&b, l), i))
        .collect();
    let mut asymmetry = 0.0_f64;
    for (i, l) in images.iter().enumerate() {
        let nl = lattice_coords(&b, l);
        let d = match idx.get(&[-nl[0], -nl[1], -nl[2]]) {
            Some(&j) => blocks[i]
                .iter()
                .zip(blocks[j].t().iter())
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs())),
            None => blocks[i].iter().fold(0.0_f64, |m, x| m.max(x.abs())),
        };
        asymmetry = asymmetry.max(d);
    }

    Ok(Some(PeriodicEcpImages {
        nbasis: n,
        n_calls: images.len(),
        images,
        blocks,
        n_triples,
        n_ecp_images: used_site.iter().filter(|&&u| u).count(),
        r_ecp: plan.r_ecp,
        r_pair: plan.r_pair,
        asymmetry,
    }))
}

// ============================================================ forces

/// Deliberate defects of the ECP force term (test only; FINDINGS
/// Iteration 22 `assemble_ecp_grad(mutant=...)`). Prototype magnitudes on
/// HI/LANL2DZ (6×6×7, full frozen set, SCF-FD anchor): `NoCentre` 3.1e-1,
/// `CentreSign` 6.3e-1 (ΣF sees both); `L0Only` 1.6e-2, `M0Only` 4.2e-3
/// (ΣF is BLIND to both: every kept triple is still translation invariant,
/// so only the FD anchor catches them).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcpGradMutation {
    /// Drop the ECP-centre derivative.
    NoCentre,
    /// Centre `= +(bra + ket)` instead of `−(bra + ket)`.
    CentreSign,
    /// Orbital image `L = 0` only in the derivative (the ket images treated
    /// as if they did not exist).
    L0Only,
    /// Home ECP image `M = 0` only in the derivative.
    M0Only,
    /// NOT a defect: the FROZEN pre-parallel serial image loop instead of
    /// the ordered-parallel one ([`crate::ordered`]); the force must be
    /// BIT-IDENTICAL either way. Never list it as a must-fail mutant.
    SerialImages,
}

/// The periodic-ECP force term ([`periodic_ecp_gradient`]).
#[derive(Debug, Clone)]
pub struct PeriodicEcpGradient {
    /// `dE_ECP/dR_A`, `natoms × 3` (Hartree/Bohr, per cell); `= bra + ket +
    /// centre`.
    pub grad: Array2<f64>,
    /// Bra-shell motion, folded by the bra shell's atom.
    pub bra: Array2<f64>,
    /// Ket-image motion (every `L`), folded by the ket shell's atom.
    pub ket: Array2<f64>,
    /// ECP-image motion (every `M`), folded by the centre's cell atom.
    pub centre: Array2<f64>,
    /// Kept triples of the screen — the energy's count
    /// (`PeriodicHcore::n_ecp_triples` / [`PeriodicEcpImages::n_triples`]
    /// at the same geometry and config).
    pub n_triples: usize,
    /// Triples actually differentiated (`n_triples` unless a mutation drops
    /// some).
    pub n_triples_evaluated: usize,
    /// `ferric_ecp_block_deriv` calls (one per kept `L`).
    pub n_calls: usize,
}

/// The Gamma-point periodic-ECP force term (FINDINGS Iteration 22):
///
/// ```text
/// dE_ECP/dR_A = Σ_L Σ_{(C, M)} Σ_μν D_μν [ δ_{A,atom μ} ∂_{A_μ} + δ_{A,atom ν} ∂_{B_ν}
///                                         + δ_{A,C} ∂_C ] ⟨μ_0 | U_C(r − R_C − M) | ν_L⟩
/// ∂_C = −(∂_{A_μ} + ∂_{B_ν})     per (bra, centre image, ket image) triple
/// ```
///
/// The ket image `R_B + L` and the ECP image `R_C + M` move with their atoms
/// for EVERY `L`, `M`. The triples are EXACTLY the energy's
/// ([`periodic_ecp_images`] at the same `cfg`: both walk one `EcpPlan`),
/// so this is the exact derivative of the energy's partial sum at this
/// geometry; a finite difference of the energy additionally sees screen
/// flips (triples entering or leaving the set under the displacement), each
/// bounded by `cfg.precision` in the energy.
///
/// `d` is the TOTAL density (`D_α + D_β`; `E_ECP = Σ D V_ECP`, `V_ECP`
/// symmetrised, `D` symmetric). The per-triple derivatives come from
/// [`ecp_block_deriv_spherical`] — true partials also for shells on their
/// own ECP centre (see there). PySCF 2.13's `ECPscalar_ipnuc` is NOT a
/// reference for this (FINDINGS Iteration 22: 1.1e-7 off on an off-centre
/// H1s–H1s(L) element).
///
/// `Ok(None)` when no non-ghost atom has an ECP in the basis.
pub fn periodic_ecp_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicEcpConfig,
    d: &Array2<f64>,
) -> Result<Option<PeriodicEcpGradient>, FerricError> {
    periodic_ecp_gradient_with(cell, prep, cfg, d, None)
}

/// [`periodic_ecp_gradient`] with a test-only [`EcpGradMutation`].
#[doc(hidden)]
pub fn periodic_ecp_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicEcpConfig,
    d: &Array2<f64>,
    mutation: Option<EcpGradMutation>,
) -> Result<Option<PeriodicEcpGradient>, FerricError> {
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    periodic_ecp_gradient_on(cell, prep, cfg, d, mutation, &mut ledger)
}

/// [`periodic_ecp_gradient_with`] on the caller's ledger (the Gamma force
/// assembly's).
pub(crate) fn periodic_ecp_gradient_on(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicEcpConfig,
    d: &Array2<f64>,
    mutation: Option<EcpGradMutation>,
    ledger: &mut Ledger,
) -> Result<Option<PeriodicEcpGradient>, FerricError> {
    let Some(plan) = EcpPlan::build(cell, prep, cfg, ledger)? else {
        return Ok(None);
    };
    let n = plan.nbasis;
    if d.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "periodic ECP gradient: density is {:?}, expected ({n}, {n})",
            d.dim()
        )));
    }
    let natoms = cell.positions().len();
    // AO -> cell atom (PreparedBasis order; the block rows/columns follow
    // the same shells with 2l+1 functions each, checked in the plan).
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let mut aoat = vec![0usize; n];
    for (s, &at) in plan.shell_atom.iter().enumerate() {
        for k in 0..dims[s] {
            aoat[offs[s] + k] = at;
        }
    }
    let nc: usize = plan
        .shells
        .iter()
        .map(|s| ((s.l + 1) * (s.l + 2) / 2) as usize)
        .sum();

    let mut acc = EcpGradAcc {
        g_bra: Array2::<f64>::zeros((natoms, 3)),
        g_ket: Array2::<f64>::zeros((natoms, 3)),
        g_cen: Array2::<f64>::zeros((natoms, 3)),
        n_triples: 0,
        n_eval: 0,
        n_calls: 0,
    };
    if mutation == Some(EcpGradMutation::SerialImages) {
        acc = ecp_gradient_serial(&plan, d, &aoat, nc, natoms, None, ledger)?;
    } else {
        let ledger_ro: &Ledger = ledger;
        ordered_units(
            plan.l_list.len(),
            window_budget(ledger_ro.remaining()),
            24 * n * n,
            |li| ecp_image_part(&plan, li, d, nc, mutation, ledger_ro),
            |_, part: EcpImagePart| {
                acc.add(&part, &aoat);
                Ok(())
            },
        )?;
    }
    let EcpGradAcc {
        g_bra,
        g_ket,
        g_cen,
        n_triples,
        n_eval,
        n_calls,
    } = acc;
    let grad = &(&g_bra + &g_ket) + &g_cen;
    Ok(Some(PeriodicEcpGradient {
        grad,
        bra: g_bra,
        ket: g_ket,
        centre: g_cen,
        n_triples,
        n_triples_evaluated: n_eval,
        n_calls,
    }))
}

/// Running sums of the periodic-ECP force term.
struct EcpGradAcc {
    g_bra: Array2<f64>,
    g_ket: Array2<f64>,
    g_cen: Array2<f64>,
    n_triples: usize,
    n_eval: usize,
    n_calls: usize,
}

/// One orbital image `L`'s force pieces, formed in parallel by
/// [`ecp_image_part`]: every scalar the serial loop adds into a shared row,
/// each computed exactly as the serial body computes it (products
/// `D_μν ∂_B V`, the from-zero bra row sums, the signed centre traces), in
/// the serial order. `ket` is `(3, n, n)`, `bra` `(3, n)`, `centre`
/// `(3, ng)` against `centre_atom` (empty for `NoCentre`).
struct EcpImagePart {
    n_triples: usize,
    n_eval: usize,
    called: bool,
    ket: Vec<f64>,
    bra: Vec<f64>,
    centre: Vec<f64>,
    centre_atom: Vec<usize>,
}

impl Stored for EcpImagePart {
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + 8 * (self.ket.len() + self.bra.len() + self.centre.len() + self.centre_atom.len())
    }
}

impl EcpGradAcc {
    /// The serial loop's accumulation of one image, in its exact per-row
    /// order: for each `x`, `μ`, `ν`: `g_ket[atom ν] += D ∂_B`; then
    /// `g_bra[atom μ] += Σ_ν D ∂_A`; then the centre traces per group.
    fn add(&mut self, part: &EcpImagePart, aoat: &[usize]) {
        self.n_triples += part.n_triples;
        self.n_eval += part.n_eval;
        if !part.called {
            return;
        }
        self.n_calls += 1;
        let n = aoat.len();
        let ng = part.centre_atom.len();
        for x in 0..3 {
            for mu in 0..n {
                let row = x * n * n + mu * n;
                for nu in 0..n {
                    self.g_ket[(aoat[nu], x)] += part.ket[row + nu];
                }
                self.g_bra[(aoat[mu], x)] += part.bra[x * n + mu];
            }
            for (g, &at) in part.centre_atom.iter().enumerate() {
                self.g_cen[(at, x)] += part.centre[x * ng + g];
            }
        }
    }
}

/// The PURE part of orbital image `plan.l_list[li]` (the serial loop body
/// up to its `+=`): screen, mutation filter, compaction, the derivative
/// block, and every addend. The transient derivative blocks are checked
/// once per possible concurrent image (threads + 1), as the parallel SR
/// walks check their scratch.
fn ecp_image_part(
    plan: &EcpPlan,
    li: usize,
    d: &Array2<f64>,
    nc: usize,
    mutation: Option<EcpGradMutation>,
    ledger: &Ledger,
) -> Result<EcpImagePart, FerricError> {
    let n = plan.nbasis;
    let l = &plan.l_list[li];
    let ket = plan.ket_shells(l);
    let kept = plan.kept(&ket);
    let mut part = EcpImagePart {
        n_triples: kept.len(),
        n_eval: 0,
        called: false,
        ket: Vec::new(),
        bra: Vec::new(),
        centre: Vec::new(),
        centre_atom: Vec::new(),
    };
    let l0 = l.iter().all(|v| v.abs() < 1e-9);
    let kept: Vec<(usize, usize, usize)> = match mutation {
        Some(EcpGradMutation::L0Only) if !l0 => Vec::new(),
        Some(EcpGradMutation::M0Only) => kept
            .into_iter()
            .filter(|&(_, _, u)| plan.sites[u].home)
            .collect(),
        _ => kept,
    };
    if kept.is_empty() {
        return Ok(part);
    }
    part.n_eval = kept.len();
    let (centres, site_of, mask) = plan.compact(&kept);
    let (groups, group_atom) = centre_groups(plan, &site_of);
    let ng = group_atom.len();
    let concurrent = rayon::current_num_threads().saturating_add(1);
    ledger.check(
        &format!(
            "periodic ECP derivative blocks for one image ({ng} centre atoms) × {concurrent} \
             concurrent images"
        ),
        bytes_of(((6 + 3 * ng) * (nc * nc + n * n)) as u64, 8).saturating_mul(concurrent),
    )?;
    let blk = ecp_block_deriv_spherical(
        &plan.shells,
        &ket,
        &centres,
        Some(mask.as_slice()),
        &groups,
        ng,
    )?;
    if blk.nrow != n || blk.ncol != n {
        return Err(FerricError::Libint(format!(
            "periodic ECP gradient: derivative block is {}×{}, expected {n}×{n}",
            blk.nrow, blk.ncol
        )));
    }
    part.called = true;
    part.ket = vec![0.0; 3 * n * n];
    part.bra = vec![0.0; 3 * n];
    let no_centre = matches!(mutation, Some(EcpGradMutation::NoCentre));
    let sign = if matches!(mutation, Some(EcpGradMutation::CentreSign)) {
        -1.0
    } else {
        1.0
    };
    for x in 0..3 {
        let (bx, kx) = (&blk.bra[x], &blk.ket[x]);
        for mu in 0..n {
            let row = mu * n;
            let mut acc_b = 0.0;
            for nu in 0..n {
                let dmn = d[(mu, nu)];
                acc_b += dmn * bx[row + nu];
                part.ket[x * n * n + row + nu] = dmn * kx[row + nu];
            }
            part.bra[x * n + mu] = acc_b;
        }
        if no_centre {
            continue;
        }
        for cxg in blk.centre.iter().take(ng) {
            let cx = &cxg[x];
            let s: f64 = d.iter().zip(cx.iter()).map(|(a, b)| a * b).sum();
            part.centre.push(sign * s);
        }
    }
    if !no_centre {
        // `centre` was pushed x-major, group-minor: `[x · ng + g]`.
        part.centre_atom = group_atom;
    }
    Ok(part)
}

/// Local centre groups of one image: the distinct cell atoms among its
/// sites, in first-appearance order, and each local centre's group.
fn centre_groups(plan: &EcpPlan, site_of: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut group_of_atom: HashMap<usize, usize> = HashMap::new();
    let mut group_atom: Vec<usize> = Vec::new();
    let groups: Vec<usize> = site_of
        .iter()
        .map(|&u| {
            let at = plan.sites[u].atom;
            *group_of_atom.entry(at).or_insert_with(|| {
                group_atom.push(at);
                group_atom.len() - 1
            })
        })
        .collect();
    (groups, group_atom)
}

/// FROZEN pre-parallel serial image loop of [`periodic_ecp_gradient_on`]
/// (the bit-identity oracle of the ordered-parallel loop; do not
/// "improve").
#[allow(clippy::too_many_arguments)]
fn ecp_gradient_serial(
    plan: &EcpPlan,
    d: &Array2<f64>,
    aoat: &[usize],
    nc: usize,
    natoms: usize,
    mutation: Option<EcpGradMutation>,
    ledger: &Ledger,
) -> Result<EcpGradAcc, FerricError> {
    let n = plan.nbasis;
    let mut g_bra = Array2::<f64>::zeros((natoms, 3));
    let mut g_ket = Array2::<f64>::zeros((natoms, 3));
    let mut g_cen = Array2::<f64>::zeros((natoms, 3));
    let (mut n_triples, mut n_eval, mut n_calls) = (0usize, 0usize, 0usize);
    for l in &plan.l_list {
        let ket = plan.ket_shells(l);
        let kept = plan.kept(&ket);
        n_triples += kept.len();
        let l0 = l.iter().all(|v| v.abs() < 1e-9);
        let kept: Vec<(usize, usize, usize)> = match mutation {
            Some(EcpGradMutation::L0Only) if !l0 => Vec::new(),
            Some(EcpGradMutation::M0Only) => kept
                .into_iter()
                .filter(|&(_, _, u)| plan.sites[u].home)
                .collect(),
            _ => kept,
        };
        if kept.is_empty() {
            continue;
        }
        n_eval += kept.len();
        let (centres, site_of, mask) = plan.compact(&kept);
        // Local centre groups: the distinct cell atoms among this L's sites.
        let mut group_of_atom: HashMap<usize, usize> = HashMap::new();
        let mut group_atom: Vec<usize> = Vec::new();
        let groups: Vec<usize> = site_of
            .iter()
            .map(|&u| {
                let at = plan.sites[u].atom;
                *group_of_atom.entry(at).or_insert_with(|| {
                    group_atom.push(at);
                    group_atom.len() - 1
                })
            })
            .collect();
        let ng = group_atom.len();
        // Transient: Cartesian (shim) + spherical (wrapper) blocks, 3 bra +
        // 3 ket + 3 per group.
        ledger.check(
            &format!("periodic ECP derivative blocks for one image ({ng} centre atoms)"),
            bytes_of(((6 + 3 * ng) * (nc * nc + n * n)) as u64, 8),
        )?;
        let blk = ecp_block_deriv_spherical(
            &plan.shells,
            &ket,
            &centres,
            Some(mask.as_slice()),
            &groups,
            ng,
        )?;
        if blk.nrow != n || blk.ncol != n {
            return Err(FerricError::Libint(format!(
                "periodic ECP gradient: derivative block is {}×{}, expected {n}×{n}",
                blk.nrow, blk.ncol
            )));
        }
        n_calls += 1;
        for x in 0..3 {
            let (bx, kx) = (&blk.bra[x], &blk.ket[x]);
            for mu in 0..n {
                let am = aoat[mu];
                let row = mu * n;
                let mut acc_b = 0.0;
                for nu in 0..n {
                    let dmn = d[(mu, nu)];
                    acc_b += dmn * bx[row + nu];
                    g_ket[(aoat[nu], x)] += dmn * kx[row + nu];
                }
                g_bra[(am, x)] += acc_b;
            }
            if matches!(mutation, Some(EcpGradMutation::NoCentre)) {
                continue;
            }
            let sign = if matches!(mutation, Some(EcpGradMutation::CentreSign)) {
                -1.0
            } else {
                1.0
            };
            for (g, &at) in group_atom.iter().enumerate() {
                let cx = &blk.centre[g][x];
                let s: f64 = d.iter().zip(cx.iter()).map(|(a, b)| a * b).sum();
                g_cen[(at, x)] += sign * s;
            }
        }
    }
    Ok(EcpGradAcc {
        g_bra,
        g_ket,
        g_cen,
        n_triples,
        n_eval,
        n_calls,
    })
}
